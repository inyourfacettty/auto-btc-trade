#!/usr/bin/env python3
"""Backtest half-margin high-leverage lottery strategies on BTC-USDT-SWAP."""

from __future__ import annotations

import argparse
import bisect
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from backtest_okx_btc import BAR_MS, Candle, compute_ema, compute_rsi, format_ts, load_candles_csv
from realtime_recommendation import wick_ratio
from strategy_research import compute_atr, latest_cache, rolling_high, rolling_low


@dataclass(frozen=True)
class LotteryConfig:
    name: str
    signal_family: str
    leverage: float
    stop_pct: float
    target_pct: float
    margin_pct: float = 0.50
    max_hold_hours: float = 720.0
    fee_rate: float = 0.0005
    slippage_bps: float = 1.0


@dataclass(frozen=True)
class Signal:
    family: str
    side: str
    entry_idx: int
    note: str


@dataclass(frozen=True)
class LotteryTrade:
    strategy: str
    family: str
    side: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    liquidation_price: float
    margin_used: float
    notional: float
    pnl: float
    equity_before: float
    equity_after: float
    exit_reason: str
    hold_hours: float
    note: str


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def first_15m_at_or_after(candles_15m: list[Candle], ts: int) -> int | None:
    timestamps = [c.ts for c in candles_15m]
    idx = bisect.bisect_left(timestamps, ts)
    return idx if idx < len(candles_15m) else None


def generate_breakout_signals(candles_15m: list[Candle], candles_4h: list[Candle]) -> list[Signal]:
    closes = [c.close for c in candles_4h]
    ema20 = compute_ema(closes, 20)
    ema60 = compute_ema(closes, 60)
    ema120 = compute_ema(closes, 120)
    rsi = compute_rsi(closes, 14)
    atr = compute_atr(candles_4h, 14)
    high40 = rolling_high(candles_4h, 40)
    low40 = rolling_low(candles_4h, 40)
    signals: list[Signal] = []

    for i in range(130, len(candles_4h)):
        required = (ema20[i], ema60[i], ema120[i], rsi[i], atr[i], high40[i], low40[i])
        if any(value is None for value in required):
            continue
        candle = candles_4h[i]
        atr_pct = atr[i] / candle.close if candle.close else 0.0
        entry_idx = first_15m_at_or_after(candles_15m, candle.ts + BAR_MS["4H"])
        if entry_idx is None or not (0.004 <= atr_pct <= 0.05):
            continue

        if candle.close > high40[i] and candle.close > ema120[i] and ema20[i] > ema60[i] and rsi[i] >= 52:
            signals.append(Signal("breakout", "long", entry_idx, "4H突破40根高点"))
        elif candle.close < low40[i] and candle.close < ema120[i] and ema20[i] < ema60[i] and rsi[i] <= 48:
            signals.append(Signal("breakout", "short", entry_idx, "4H跌破40根低点"))
    return signals


def generate_reversal_signals(candles_15m: list[Candle], candles_4h: list[Candle]) -> list[Signal]:
    closes = [c.close for c in candles_4h]
    rsi = compute_rsi(closes, 14)
    atr = compute_atr(candles_4h, 14)
    window = 60
    signals: list[Signal] = []

    for i in range(window + 1, len(candles_4h)):
        if rsi[i] is None or atr[i] is None:
            continue
        candle = candles_4h[i]
        previous = candles_4h[i - 1]
        lower = min(c.low for c in candles_4h[i - window : i])
        upper = max(c.high for c in candles_4h[i - window : i])
        atr_pct = atr[i] / candle.close if candle.close else 0.0
        entry_idx = first_15m_at_or_after(candles_15m, candle.ts + BAR_MS["4H"])
        if entry_idx is None or not (0.006 <= atr_pct <= 0.06):
            continue

        lower_wick = wick_ratio(candle.open, candle.high, candle.low, candle.close, "long")
        upper_wick = wick_ratio(candle.open, candle.high, candle.low, candle.close, "short")
        if (
            candle.low <= lower * 1.005
            and rsi[i] <= 35
            and candle.close > candle.open
            and candle.close > previous.close
            and lower_wick >= 0.50
        ):
            signals.append(Signal("reversal", "long", entry_idx, "唐奇安60下轨反转"))
        elif (
            candle.high >= upper * 0.995
            and rsi[i] >= 65
            and candle.close < candle.open
            and candle.close < previous.close
            and upper_wick >= 0.50
        ):
            signals.append(Signal("reversal", "short", entry_idx, "唐奇安60上轨反转"))
    return signals


def liquidation_price(entry_price: float, leverage: float, side: str) -> float:
    gap = 0.90 / leverage
    if side == "long":
        return entry_price * (1 - gap)
    if side == "short":
        return entry_price * (1 + gap)
    raise ValueError(f"Unsupported side: {side}")


def simulate_trade(
    candles_15m: list[Candle],
    signal: Signal,
    config: LotteryConfig,
    equity: float,
) -> LotteryTrade | None:
    if signal.entry_idx >= len(candles_15m):
        return None

    entry_candle = candles_15m[signal.entry_idx]
    slippage = config.slippage_bps / 10_000
    entry = entry_candle.open * (1 + slippage) if signal.side == "long" else entry_candle.open * (1 - slippage)
    margin = equity * config.margin_pct
    notional = margin * config.leverage
    entry_fee = notional * config.fee_rate

    stop_u = equity * config.stop_pct
    target_u = equity * config.target_pct
    round_trip_fee_budget = notional * config.fee_rate * 2
    stop_move = (stop_u + round_trip_fee_budget) / notional
    target_move = (target_u + round_trip_fee_budget) / notional

    stop = entry * (1 - stop_move) if signal.side == "long" else entry * (1 + stop_move)
    target = entry * (1 + target_move) if signal.side == "long" else entry * (1 - target_move)
    liq = liquidation_price(entry, config.leverage, signal.side)

    unsafe_stop = (signal.side == "long" and stop <= liq) or (signal.side == "short" and stop >= liq)
    if unsafe_stop:
        return None

    max_exit_ts = entry_candle.ts + int(config.max_hold_hours * 60 * 60 * 1000)
    for idx in range(signal.entry_idx, len(candles_15m)):
        candle = candles_15m[idx]
        if candle.ts >= max_exit_ts:
            exit_price = candle.open * (1 - slippage) if signal.side == "long" else candle.open * (1 + slippage)
            reason = "超时"
        elif signal.side == "long" and candle.low <= stop:
            exit_price = stop * (1 - slippage)
            reason = "止损"
        elif signal.side == "short" and candle.high >= stop:
            exit_price = stop * (1 + slippage)
            reason = "止损"
        elif signal.side == "long" and candle.low <= liq:
            exit_price = liq
            reason = "强平"
        elif signal.side == "short" and candle.high >= liq:
            exit_price = liq
            reason = "强平"
        elif signal.side == "long" and candle.high >= target:
            exit_price = target * (1 - slippage)
            reason = "目标止盈"
        elif signal.side == "short" and candle.low <= target:
            exit_price = target * (1 + slippage)
            reason = "目标止盈"
        elif idx == len(candles_15m) - 1:
            exit_price = candle.close * (1 - slippage) if signal.side == "long" else candle.close * (1 + slippage)
            reason = "数据结束"
        else:
            continue

        if reason == "强平":
            pnl = -margin - entry_fee
        else:
            move = (exit_price - entry) / entry if signal.side == "long" else (entry - exit_price) / entry
            gross = notional * move
            exit_fee = notional * (exit_price / entry) * config.fee_rate
            pnl = gross - entry_fee - exit_fee
        return LotteryTrade(
            strategy=config.name,
            family=signal.family,
            side=signal.side,
            entry_ts=entry_candle.ts,
            exit_ts=candle.ts,
            entry_price=entry,
            exit_price=exit_price,
            stop_price=stop,
            target_price=target,
            liquidation_price=liq,
            margin_used=margin,
            notional=notional,
            pnl=pnl,
            equity_before=equity,
            equity_after=equity + pnl,
            exit_reason=reason,
            hold_hours=(candle.ts - entry_candle.ts) / 3_600_000,
            note=signal.note,
        )
    return None


def simulate_independent(
    candles_15m: list[Candle],
    signals: list[Signal],
    config: LotteryConfig,
    start_ts: int,
    end_ts: int,
    initial_equity: float,
) -> list[LotteryTrade]:
    trades: list[LotteryTrade] = []
    for signal in sorted(signals, key=lambda s: s.entry_idx):
        entry_ts = candles_15m[signal.entry_idx].ts
        if not (start_ts <= entry_ts < end_ts):
            continue
        trade = simulate_trade(candles_15m, signal, config, initial_equity)
        if trade is not None:
            trades.append(trade)
    return trades


def simulate_sequential(
    candles_15m: list[Candle],
    signals: list[Signal],
    config: LotteryConfig,
    start_ts: int,
    end_ts: int,
    initial_equity: float,
    stop_after_target: bool = False,
) -> list[LotteryTrade]:
    trades: list[LotteryTrade] = []
    equity = initial_equity
    next_allowed = 0
    for signal in sorted(signals, key=lambda s: s.entry_idx):
        if signal.entry_idx < next_allowed:
            continue
        entry_ts = candles_15m[signal.entry_idx].ts
        if not (start_ts <= entry_ts < end_ts):
            continue
        if equity <= 0:
            break
        trade = simulate_trade(candles_15m, signal, config, equity)
        if trade is None:
            continue
        trades.append(trade)
        equity = trade.equity_after
        next_allowed = bisect.bisect_right([c.ts for c in candles_15m], trade.exit_ts)
        if stop_after_target and trade.exit_reason == "目标止盈":
            break
    return trades


def summarize(trades: list[LotteryTrade], initial_equity: float, sequential: bool = False) -> dict[str, float | int]:
    if sequential:
        equity = trades[-1].equity_after if trades else initial_equity
    else:
        equity = initial_equity + sum(t.pnl for t in trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    peak = initial_equity
    max_dd = 0.0
    if sequential:
        for trade in trades:
            peak = max(peak, trade.equity_before, trade.equity_after)
            max_dd = max(max_dd, (peak - trade.equity_after) / peak * 100 if peak else 0.0)
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
        "final": equity,
        "return_pct": (equity / initial_equity - 1) * 100 if initial_equity else 0.0,
        "max_dd": max_dd,
        "avg_win": statistics.mean([t.pnl for t in wins]) if wins else 0.0,
        "avg_loss": statistics.mean([t.pnl for t in losses]) if losses else 0.0,
        "target": sum(1 for t in trades if t.exit_reason == "目标止盈"),
        "stop": sum(1 for t in trades if t.exit_reason == "止损"),
        "timeout": sum(1 for t in trades if t.exit_reason == "超时"),
        "liq": sum(1 for t in trades if t.exit_reason == "强平"),
        "avg_hold": statistics.mean([t.hold_hours for t in trades]) if trades else 0.0,
        "max_loss_streak": max_loss_streak(trades),
    }


def max_loss_streak(trades: list[LotteryTrade]) -> int:
    current = 0
    best = 0
    for trade in trades:
        if trade.pnl <= 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def export_trades(path: Path, trades: list[LotteryTrade]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["策略", "信号", "方向", "入场时间", "出场时间", "入场", "出场", "止损", "目标", "强平", "保证金", "名义仓位", "盈亏U", "入场前权益", "出场后权益", "出场原因", "持仓小时", "备注"])
        for t in trades:
            writer.writerow([
                t.strategy,
                "摸底摸顶" if t.family == "reversal" else "趋势突破",
                "做多" if t.side == "long" else "做空",
                format_ts(t.entry_ts),
                format_ts(t.exit_ts),
                f"{t.entry_price:.2f}",
                f"{t.exit_price:.2f}",
                f"{t.stop_price:.2f}",
                f"{t.target_price:.2f}",
                f"{t.liquidation_price:.2f}",
                f"{t.margin_used:.2f}",
                f"{t.notional:.2f}",
                f"{t.pnl:.2f}",
                f"{t.equity_before:.2f}",
                f"{t.equity_after:.2f}",
                t.exit_reason,
                f"{t.hold_hours:.2f}",
                t.note,
            ])


def print_rows(rows: list[tuple[LotteryConfig, dict, dict, dict]]) -> None:
    print("策略 | 独立样本 | 独立命中 | 独立EV | 连续交易 | 连续收益 | 最大回撤 | 目标/止损/超时 | 止盈即撤权益")
    print("-" * 128)
    for config, one, seq, stop_once in rows:
        ev = (one["final"] - 1000.0) / one["trades"] if one["trades"] else 0.0
        print(
            f"{config.name} | {one['trades']} | {one['win_rate']:.1f}% | {ev:.1f}U | "
            f"{seq['trades']} | {seq['return_pct']:.1f}% | {seq['max_dd']:.1f}% | "
            f"{seq['target']}/{seq['stop']}/{seq['timeout']} | {stop_once['final']:.2f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="半仓高杠杆彩票策略回测")
    parser.add_argument("--inst-id", default="BTC-USDT-SWAP")
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--export-dir", type=Path, default=Path("data"))
    parser.add_argument("--initial-equity", type=float, default=1000.0)
    parser.add_argument("--days", type=int, default=365)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    args = build_parser().parse_args(argv)
    try:
        candles_15m = load_candles_csv(latest_cache(args.cache_dir, f"okx_{args.inst_id}_15m_*.csv"))
        candles_4h = load_candles_csv(latest_cache(args.cache_dir, f"okx_{args.inst_id}_4H_*.csv"))
        end_ts = candles_15m[-1].ts + BAR_MS["15m"]
        start_ts = end_ts - args.days * 24 * 60 * 60 * 1000
        signals = generate_breakout_signals(candles_15m, candles_4h) + generate_reversal_signals(candles_15m, candles_4h)
        configs = [
            LotteryConfig("A_摸底摸顶_30x_止损200_目标500", "reversal", 30, 0.20, 0.50),
            LotteryConfig("B_摸底摸顶_30x_止损150_目标500", "reversal", 30, 0.15, 0.50),
            LotteryConfig("C_摸底摸顶_50x_止损150_目标800", "reversal", 50, 0.15, 0.80),
            LotteryConfig("D_摸底摸顶_50x_止损150_目标1000", "reversal", 50, 0.15, 1.00),
            LotteryConfig("E_摸底摸顶_20x_止损150_目标300", "reversal", 20, 0.15, 0.30),
            LotteryConfig("F_趋势突破_30x_止损150_目标500", "breakout", 30, 0.15, 0.50),
            LotteryConfig("G_趋势突破_50x_止损150_目标500", "breakout", 50, 0.15, 0.50),
        ]

        rows = []
        for config in configs:
            selected = [s for s in signals if s.family == config.signal_family]
            independent = simulate_independent(candles_15m, selected, config, start_ts, end_ts, args.initial_equity)
            sequential = simulate_sequential(candles_15m, selected, config, start_ts, end_ts, args.initial_equity)
            stop_once = simulate_sequential(candles_15m, selected, config, start_ts, end_ts, args.initial_equity, stop_after_target=True)
            rows.append((config, summarize(independent, args.initial_equity), summarize(sequential, args.initial_equity, True), summarize(stop_once, args.initial_equity, True)))
            export_trades(args.export_dir / f"lottery_{config.name}_sequential.csv", sequential)

        print(f"回测区间：{format_ts(start_ts)} -> {format_ts(end_ts)}")
        print("设定：1000U本金，每次50%权益作保证金；止损/目标按当前权益百分比复利；同一时间只持有一笔。")
        print_rows(rows)
        export_summary(args.export_dir / "lottery_strategy_summary.csv", rows)
        print()
        print(f"汇总CSV：{args.export_dir / 'lottery_strategy_summary.csv'}")
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


def export_summary(path: Path, rows: list[tuple[LotteryConfig, dict, dict, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["策略", "独立样本", "独立命中率", "独立单次EV", "连续交易", "连续收益", "连续最终权益", "最大回撤", "目标止盈", "止损", "超时", "强平", "平均盈利", "平均亏损", "最大连亏", "止盈即撤权益"])
        for config, one, seq, stop_once in rows:
            ev = (one["final"] - 1000.0) / one["trades"] if one["trades"] else 0.0
            writer.writerow([
                config.name,
                one["trades"],
                f"{one['win_rate']:.2f}",
                f"{ev:.2f}",
                seq["trades"],
                f"{seq['return_pct']:.2f}",
                f"{seq['final']:.2f}",
                f"{seq['max_dd']:.2f}",
                seq["target"],
                seq["stop"],
                seq["timeout"],
                seq["liq"],
                f"{seq['avg_win']:.2f}",
                f"{seq['avg_loss']:.2f}",
                seq["max_loss_streak"],
                f"{stop_once['final']:.2f}",
            ])


if __name__ == "__main__":
    raise SystemExit(main())
