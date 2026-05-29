#!/usr/bin/env python3
"""Backtest the high-profit BTC-USDT-SWAP top/bottom strategy."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from backtest_okx_btc import BAR_MS, Candle, compute_rsi, format_ts, load_candles_csv
from realtime_recommendation import calculate_entry_zone, wick_ratio
from strategy_research import compute_atr, latest_cache


@dataclass(frozen=True)
class ProfitConfig:
    name: str
    initial_equity: float = 1000.0
    leverage: float = 20.0
    margin_pct: float = 0.10
    fee_rate: float = 0.0005
    slippage_bps: float = 1.0
    take_profit_r: float = 1.5
    stop_liq_fraction: float = 0.8
    entry_mode: str = "immediate"
    max_entry_delay_minutes: int = 30
    max_hold_hours: float | None = None


@dataclass(frozen=True)
class TopBottomSignal:
    h4_idx: int
    entry_idx: int
    side: str
    signal_close: float
    structural_stop: float
    atr14: float
    rsi14: float
    wick_pct: float
    note: str


@dataclass(frozen=True)
class ProfitTrade:
    strategy: str
    side: str
    entry_ts: int
    exit_ts: int
    signal_price: float
    entry_price: float
    exit_price: float
    structural_stop: float
    stop_price: float
    take_profit_price: float
    liquidation_price: float
    notional: float
    margin_used: float
    net_pnl: float
    return_pct: float
    r_multiple: float
    exit_reason: str
    equity_before: float
    equity_after: float
    hold_hours: float
    note: str


def rolling_channel(candles: list[Candle], index: int, window: int) -> tuple[float, float]:
    channel = candles[index - window : index]
    return min(c.low for c in channel), max(c.high for c in channel)


def first_15m_at_or_after(candles_15m: list[Candle], ts: int) -> int | None:
    timestamps = [c.ts for c in candles_15m]
    idx = bisect.bisect_left(timestamps, ts)
    return idx if idx < len(candles_15m) else None


def generate_top_bottom_signals(
    candles_15m: list[Candle],
    candles_4h: list[Candle],
    channel_window: int = 60,
    proximity_pct: float = 0.005,
    min_wick_ratio: float = 0.50,
) -> list[TopBottomSignal]:
    closes = [c.close for c in candles_4h]
    rsi = compute_rsi(closes, 14)
    atr = compute_atr(candles_4h, 14)
    signals: list[TopBottomSignal] = []

    for i in range(channel_window + 1, len(candles_4h)):
        if rsi[i] is None or atr[i] is None:
            continue
        candle = candles_4h[i]
        previous = candles_4h[i - 1]
        lower, upper = rolling_channel(candles_4h, i, channel_window)
        atr_pct = atr[i] / candle.close if candle.close else 0.0
        if not (0.006 <= atr_pct <= 0.06):
            continue

        entry_idx = first_15m_at_or_after(candles_15m, candle.ts + BAR_MS["4H"])
        if entry_idx is None:
            continue

        lower_wick = wick_ratio(candle.open, candle.high, candle.low, candle.close, "long")
        upper_wick = wick_ratio(candle.open, candle.high, candle.low, candle.close, "short")

        long_signal = (
            candle.low <= lower * (1 + proximity_pct)
            and rsi[i] <= 35
            and candle.close > candle.open
            and candle.close > previous.close
            and lower_wick >= min_wick_ratio
        )
        short_signal = (
            candle.high >= upper * (1 - proximity_pct)
            and rsi[i] >= 65
            and candle.close < candle.open
            and candle.close < previous.close
            and upper_wick >= min_wick_ratio
        )
        if long_signal:
            signals.append(
                TopBottomSignal(
                    h4_idx=i,
                    entry_idx=entry_idx,
                    side="long",
                    signal_close=candle.close,
                    structural_stop=candle.low,
                    atr14=float(atr[i]),
                    rsi14=float(rsi[i]),
                    wick_pct=lower_wick * 100,
                    note="4H触及唐奇安60下轨后收阳反转",
                )
            )
        elif short_signal:
            signals.append(
                TopBottomSignal(
                    h4_idx=i,
                    entry_idx=entry_idx,
                    side="short",
                    signal_close=candle.close,
                    structural_stop=candle.high,
                    atr14=float(atr[i]),
                    rsi14=float(rsi[i]),
                    wick_pct=upper_wick * 100,
                    note="4H触及唐奇安60上轨后收阴反转",
                )
            )
    return signals


def liquidation_price(entry_price: float, leverage: float, side: str) -> float:
    liquidation_gap = 0.90 / leverage
    if side == "long":
        return entry_price * (1 - liquidation_gap)
    if side == "short":
        return entry_price * (1 + liquidation_gap)
    raise ValueError(f"Unsupported side: {side}")


def stop_price(entry_price: float, structural_stop: float, config: ProfitConfig, side: str) -> float:
    hard_gap = config.stop_liq_fraction / config.leverage
    if side == "long":
        hard_stop = entry_price * (1 - hard_gap)
        return max(structural_stop, hard_stop)
    if side == "short":
        hard_stop = entry_price * (1 + hard_gap)
        return min(structural_stop, hard_stop)
    raise ValueError(f"Unsupported side: {side}")


def choose_entry(
    candles_15m: list[Candle],
    signal: TopBottomSignal,
    config: ProfitConfig,
) -> tuple[int, float] | None:
    slippage = config.slippage_bps / 10_000
    if config.entry_mode == "immediate":
        candle = candles_15m[signal.entry_idx]
        if signal.side == "long":
            return signal.entry_idx, candle.open * (1 + slippage)
        return signal.entry_idx, candle.open * (1 - slippage)

    if config.entry_mode != "optimal_30m":
        raise ValueError(f"Unsupported entry mode: {config.entry_mode}")

    zone = calculate_entry_zone(signal.side, signal.signal_close, signal.atr14)
    max_delay_ms = config.max_entry_delay_minutes * 60 * 1000
    last_ts = candles_15m[signal.entry_idx].ts + max_delay_ms
    for idx in range(signal.entry_idx, len(candles_15m)):
        candle = candles_15m[idx]
        if candle.ts >= last_ts:
            break
        if signal.side == "long" and candle.low <= zone.ideal:
            return idx, zone.ideal
        if signal.side == "short" and candle.high >= zone.ideal:
            return idx, zone.ideal
    return None


def simulate(
    candles_15m: list[Candle],
    signals: list[TopBottomSignal],
    config: ProfitConfig,
    start_ts: int,
    end_ts: int,
) -> list[ProfitTrade]:
    trades: list[ProfitTrade] = []
    equity = config.initial_equity
    next_allowed_idx = 0
    slippage = config.slippage_bps / 10_000

    for signal in signals:
        if signal.entry_idx < next_allowed_idx:
            continue
        entry_choice = choose_entry(candles_15m, signal, config)
        if entry_choice is None:
            continue
        entry_idx, entry = entry_choice
        if entry_idx < next_allowed_idx:
            continue
        entry_ts = candles_15m[entry_idx].ts
        if not (start_ts <= entry_ts < end_ts):
            continue

        stop = stop_price(entry, signal.structural_stop, config, signal.side)
        if signal.side == "long" and stop >= entry:
            continue
        if signal.side == "short" and stop <= entry:
            continue

        risk = abs(entry - stop)
        tp = entry + risk * config.take_profit_r if signal.side == "long" else entry - risk * config.take_profit_r
        liq = liquidation_price(entry, config.leverage, signal.side)
        notional = equity * config.margin_pct * config.leverage
        margin_used = equity * config.margin_pct
        entry_fee = notional * config.fee_rate
        max_hold_ms = None if config.max_hold_hours is None else int(config.max_hold_hours * 60 * 60 * 1000)
        equity_before = equity

        for idx in range(entry_idx, len(candles_15m)):
            candle = candles_15m[idx]
            if candle.ts >= end_ts:
                exit_price = candle.open * (1 - slippage) if signal.side == "long" else candle.open * (1 + slippage)
                reason = "区间结束"
            elif max_hold_ms is not None and candle.ts >= entry_ts + max_hold_ms:
                exit_price = candle.open * (1 - slippage) if signal.side == "long" else candle.open * (1 + slippage)
                reason = "时间止盈止损"
            elif signal.side == "long" and candle.low <= liq:
                exit_price = liq
                reason = "强平"
            elif signal.side == "short" and candle.high >= liq:
                exit_price = liq
                reason = "强平"
            elif signal.side == "long" and candle.low <= stop:
                exit_price = stop * (1 - slippage)
                reason = "止损"
            elif signal.side == "short" and candle.high >= stop:
                exit_price = stop * (1 + slippage)
                reason = "止损"
            elif signal.side == "long" and candle.high >= tp:
                exit_price = tp * (1 - slippage)
                reason = "止盈"
            elif signal.side == "short" and candle.low <= tp:
                exit_price = tp * (1 + slippage)
                reason = "止盈"
            elif idx == len(candles_15m) - 1:
                exit_price = candle.close * (1 - slippage) if signal.side == "long" else candle.close * (1 + slippage)
                reason = "数据结束"
            else:
                continue

            if reason == "强平":
                net_pnl = -margin_used - entry_fee
                r_multiple = -math.inf
            else:
                move = (exit_price - entry) / entry if signal.side == "long" else (entry - exit_price) / entry
                gross_pnl = notional * move
                exit_notional = notional * (exit_price / entry)
                exit_fee = exit_notional * config.fee_rate
                net_pnl = gross_pnl - entry_fee - exit_fee
                r_multiple = (exit_price - entry) / risk if signal.side == "long" else (entry - exit_price) / risk

            equity += net_pnl
            trades.append(
                ProfitTrade(
                    strategy=config.name,
                    side=signal.side,
                    entry_ts=entry_ts,
                    exit_ts=candle.ts,
                    signal_price=signal.signal_close,
                    entry_price=entry,
                    exit_price=exit_price,
                    structural_stop=signal.structural_stop,
                    stop_price=stop,
                    take_profit_price=tp,
                    liquidation_price=liq,
                    notional=notional,
                    margin_used=margin_used,
                    net_pnl=net_pnl,
                    return_pct=net_pnl / equity_before * 100 if equity_before else 0.0,
                    r_multiple=r_multiple,
                    exit_reason=reason,
                    equity_before=equity_before,
                    equity_after=equity,
                    hold_hours=(candle.ts - entry_ts) / 3_600_000,
                    note=signal.note,
                )
            )
            next_allowed_idx = idx + 1
            break

        if equity <= 0:
            break

    return trades


def summarize(trades: list[ProfitTrade], initial_equity: float) -> dict[str, float | int]:
    equity = initial_equity
    peak = initial_equity
    max_dd = 0.0
    wins: list[float] = []
    losses: list[float] = []
    max_consecutive_losses = 0
    current_losses = 0
    exit_counts: dict[str, int] = {}
    for trade in trades:
        equity += trade.net_pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100 if peak else 0.0)
        exit_counts[trade.exit_reason] = exit_counts.get(trade.exit_reason, 0) + 1
        if trade.net_pnl > 0:
            wins.append(trade.net_pnl)
            current_losses = 0
        else:
            losses.append(trade.net_pnl)
            current_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, current_losses)

    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": len(wins) / len(trades) * 100 if trades else 0.0,
        "final_equity": equity,
        "return_pct": (equity / initial_equity - 1) * 100 if initial_equity else 0.0,
        "max_drawdown_pct": max_dd,
        "profit_factor": gross_profit / gross_loss if gross_loss else math.inf if gross_profit else 0.0,
        "avg_trade": statistics.mean([t.net_pnl for t in trades]) if trades else 0.0,
        "avg_win": statistics.mean(wins) if wins else 0.0,
        "avg_loss": statistics.mean(losses) if losses else 0.0,
        "best_trade": max((t.net_pnl for t in trades), default=0.0),
        "worst_trade": min((t.net_pnl for t in trades), default=0.0),
        "avg_hold_hours": statistics.mean([t.hold_hours for t in trades]) if trades else 0.0,
        "max_consecutive_losses": max_consecutive_losses,
        "take_profit": exit_counts.get("止盈", 0),
        "stop_loss": exit_counts.get("止损", 0),
        "time_exit": exit_counts.get("时间止盈止损", 0),
        "liquidations": exit_counts.get("强平", 0),
    }


def export_trades(path: Path, trades: list[ProfitTrade]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "策略",
                "方向",
                "入场时间",
                "出场时间",
                "信号价",
                "入场价",
                "出场价",
                "结构止损",
                "实际止损",
                "止盈",
                "强平价",
                "名义仓位",
                "保证金",
                "净盈亏U",
                "本次收益率%",
                "R倍数",
                "持仓小时",
                "出场原因",
                "入场前权益",
                "出场后权益",
                "备注",
            ]
        )
        for trade in trades:
            writer.writerow(
                [
                    trade.strategy,
                    "做多" if trade.side == "long" else "做空",
                    format_ts(trade.entry_ts),
                    format_ts(trade.exit_ts),
                    f"{trade.signal_price:.2f}",
                    f"{trade.entry_price:.2f}",
                    f"{trade.exit_price:.2f}",
                    f"{trade.structural_stop:.2f}",
                    f"{trade.stop_price:.2f}",
                    f"{trade.take_profit_price:.2f}",
                    f"{trade.liquidation_price:.2f}",
                    f"{trade.notional:.2f}",
                    f"{trade.margin_used:.2f}",
                    f"{trade.net_pnl:.2f}",
                    f"{trade.return_pct:.2f}",
                    f"{trade.r_multiple:.2f}" if math.isfinite(trade.r_multiple) else "-inf",
                    f"{trade.hold_hours:.2f}",
                    trade.exit_reason,
                    f"{trade.equity_before:.2f}",
                    f"{trade.equity_after:.2f}",
                    trade.note,
                ]
            )


def load_latest_pair(cache_dir: Path, inst_id: str) -> tuple[list[Candle], list[Candle]]:
    candles_15m = load_candles_csv(latest_cache(cache_dir, f"okx_{inst_id}_15m_*.csv"))
    candles_4h = load_candles_csv(latest_cache(cache_dir, f"okx_{inst_id}_4H_*.csv"))
    return candles_15m, candles_4h


def print_summary(
    inst_id: str,
    start_ts: int,
    end_ts: int,
    signal_count: int,
    rows: list[tuple[ProfitConfig, dict[str, float | int]]],
) -> None:
    print("BTC-USDT-SWAP 高收益摸顶摸底策略回测")
    print(f"交易品种：{inst_id}")
    print(f"回测区间：{format_ts(start_ts)} -> {format_ts(end_ts)}")
    print(f"信号数量：{signal_count}")
    print("规则：4H收盘确认，下一根15m执行，唐奇安60 + RSI14 + 长影线反转")
    print("费用/滑点：单边手续费0.05%，滑点1bps；同一时间只持有一笔")
    print()
    print("策略 | 交易 | 胜率 | 收益 | 最大回撤 | 盈亏比 | 平均每笔 | 平均盈利 | 平均亏损 | 最大连亏 | 止盈/止损/时间/强平")
    print("-" * 128)
    for config, metrics in rows:
        print(
            f"{config.name} | {metrics['trades']} | {metrics['win_rate_pct']:.2f}% | "
            f"{metrics['return_pct']:.2f}% | {metrics['max_drawdown_pct']:.2f}% | "
            f"{metrics['profit_factor']:.2f} | {metrics['avg_trade']:.2f}U | "
            f"{metrics['avg_win']:.2f}U | {metrics['avg_loss']:.2f}U | "
            f"{metrics['max_consecutive_losses']} | "
            f"{metrics['take_profit']}/{metrics['stop_loss']}/{metrics['time_exit']}/{metrics['liquidations']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BTC-USDT-SWAP 高收益摸顶摸底策略回测")
    parser.add_argument("--inst-id", default="BTC-USDT-SWAP")
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--export-dir", type=Path, default=Path("data"))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--initial-equity", type=float, default=1000.0)
    parser.add_argument("--leverage", type=float, default=20.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    args = build_parser().parse_args(argv)
    try:
        candles_15m, candles_4h = load_latest_pair(args.cache_dir, args.inst_id)
        end_ts = candles_15m[-1].ts + BAR_MS["15m"]
        start_ts = end_ts - args.days * 24 * 60 * 60 * 1000
        signals = generate_top_bottom_signals(candles_15m, candles_4h)

        configs = [
            ProfitConfig(
                name="高收益_立即入场_1.5R",
                initial_equity=args.initial_equity,
                leverage=args.leverage,
                margin_pct=0.10,
                fee_rate=args.fee_rate,
                slippage_bps=args.slippage_bps,
                take_profit_r=1.5,
                entry_mode="immediate",
                max_hold_hours=None,
            ),
            ProfitConfig(
                name="高收益_最多持有4H",
                initial_equity=args.initial_equity,
                leverage=args.leverage,
                margin_pct=0.10,
                fee_rate=args.fee_rate,
                slippage_bps=args.slippage_bps,
                take_profit_r=1.5,
                entry_mode="immediate",
                max_hold_hours=4.0,
            ),
            ProfitConfig(
                name="高收益_最优价30m_1.5R",
                initial_equity=args.initial_equity,
                leverage=args.leverage,
                margin_pct=0.10,
                fee_rate=args.fee_rate,
                slippage_bps=args.slippage_bps,
                take_profit_r=1.5,
                entry_mode="optimal_30m",
                max_hold_hours=None,
            ),
            ProfitConfig(
                name="原高胜率_立即入场_0.25R",
                initial_equity=args.initial_equity,
                leverage=args.leverage,
                margin_pct=0.05,
                fee_rate=args.fee_rate,
                slippage_bps=args.slippage_bps,
                take_profit_r=0.25,
                entry_mode="immediate",
                max_hold_hours=None,
            ),
        ]

        rows = []
        for config in configs:
            trades = simulate(candles_15m, signals, config, start_ts, end_ts)
            export_trades(args.export_dir / f"{config.name}.csv", trades)
            rows.append((config, summarize(trades, args.initial_equity)))

        print_summary(args.inst_id, start_ts, end_ts, len(signals), rows)
        print()
        print(f"逐笔交易CSV目录：{args.export_dir}")
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
