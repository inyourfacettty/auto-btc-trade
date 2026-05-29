#!/usr/bin/env python3
"""High-leverage BTC-USDT reversal strategy research with liquidation checks."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from backtest_okx_btc import BAR_MS, Candle, compute_ema, compute_rsi, format_ts, load_candles_csv
from strategy_research import compute_atr, latest_cache, rolling_high, rolling_low


@dataclass(frozen=True)
class HighLeverageConfig:
    initial_equity: float = 1000.0
    leverage: float = 20.0
    margin_pct: float = 0.05
    fee_rate: float = 0.0005
    slippage_bps: float = 1.0
    stop_liq_fraction: float = 0.62
    take_profit_r: float = 1.45


@dataclass(frozen=True)
class LeveragedSignal:
    entry_idx: int
    side: str
    note: str


@dataclass(frozen=True)
class LeveragedTrade:
    side: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    liquidation_price: float
    stop_price: float
    take_profit_price: float
    net_pnl: float
    return_pct: float
    exit_reason: str
    equity_after: float
    note: str


def liquidation_price(entry_price: float, leverage: float, side: str) -> float:
    distance = 0.90 / leverage
    if side == "long":
        return entry_price * (1 - distance)
    if side == "short":
        return entry_price * (1 + distance)
    raise ValueError(f"Unsupported side: {side}")


def position_notional(equity: float, config: HighLeverageConfig) -> float:
    return equity * config.margin_pct * config.leverage


def generate_extreme_reversal_signals(candles_15m: list[Candle], candles_4h: list[Candle]) -> list[LeveragedSignal]:
    closes = [c.close for c in candles_4h]
    ema120 = compute_ema(closes, 120)
    rsi = compute_rsi(closes, 14)
    atr = compute_atr(candles_4h, 14)
    high80 = rolling_high(candles_4h, 80)
    low80 = rolling_low(candles_4h, 80)
    timestamps_15m = [c.ts for c in candles_15m]
    signals: list[LeveragedSignal] = []

    for i in range(122, len(candles_4h)):
        required = (ema120[i], rsi[i], atr[i], high80[i], low80[i])
        if any(v is None for v in required):
            continue
        atr_pct = atr[i] / candles_4h[i].close
        if not (0.006 <= atr_pct <= 0.055):
            continue

        entry_ts = candles_4h[i].ts + BAR_MS["4H"]
        entry_idx = lower_bound_ts(timestamps_15m, entry_ts)
        if entry_idx is None:
            continue

        candle = candles_4h[i]
        prev = candles_4h[i - 1]
        near_low = candle.low <= low80[i] * 1.012
        near_high = candle.high >= high80[i] * 0.988
        bullish_reversal = candle.close > candle.open and candle.close > prev.close
        bearish_reversal = candle.close < candle.open and candle.close < prev.close

        if near_low and bullish_reversal and rsi[i] <= 38 and candle.close < ema120[i] * 1.02:
            signals.append(LeveragedSignal(entry_idx, "long", "80根4H低位反转做多"))
        elif near_high and bearish_reversal and rsi[i] >= 62 and candle.close > ema120[i] * 0.98:
            signals.append(LeveragedSignal(entry_idx, "short", "80根4H高位反转做空"))
    return signals


def lower_bound_ts(timestamps: list[int], ts: int) -> int | None:
    lo = 0
    hi = len(timestamps)
    while lo < hi:
        mid = (lo + hi) // 2
        if timestamps[mid] < ts:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(timestamps) else None


def simulate_high_leverage(
    candles_15m: list[Candle],
    signals: list[LeveragedSignal],
    config: HighLeverageConfig,
    start_ts: int,
    end_ts: int,
) -> list[LeveragedTrade]:
    trades: list[LeveragedTrade] = []
    equity = config.initial_equity
    next_allowed = 0
    slippage = config.slippage_bps / 10_000

    for signal in sorted(signals, key=lambda s: s.entry_idx):
        if signal.entry_idx < next_allowed or signal.entry_idx >= len(candles_15m):
            continue
        entry_candle = candles_15m[signal.entry_idx]
        if not (start_ts <= entry_candle.ts < end_ts):
            continue

        entry = entry_candle.open * (1 + slippage) if signal.side == "long" else entry_candle.open * (1 - slippage)
        liq = liquidation_price(entry, config.leverage, signal.side)
        liq_distance = abs(entry - liq)
        stop_distance = liq_distance * config.stop_liq_fraction
        stop = entry - stop_distance if signal.side == "long" else entry + stop_distance
        tp = entry + stop_distance * config.take_profit_r if signal.side == "long" else entry - stop_distance * config.take_profit_r
        notional = position_notional(equity, config)
        margin_used = equity * config.margin_pct
        entry_fee = notional * config.fee_rate

        for idx in range(signal.entry_idx, len(candles_15m)):
            candle = candles_15m[idx]
            if candle.ts >= end_ts:
                exit_price = candle.open * (1 - slippage) if signal.side == "long" else candle.open * (1 + slippage)
                reason = "区间结束"
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
            else:
                move = (exit_price - entry) / entry if signal.side == "long" else (entry - exit_price) / entry
                gross = notional * move
                exit_fee = notional * (exit_price / entry) * config.fee_rate
                net_pnl = gross - entry_fee - exit_fee
            equity += net_pnl
            trades.append(
                LeveragedTrade(
                    side=signal.side,
                    entry_ts=entry_candle.ts,
                    exit_ts=candle.ts,
                    entry_price=entry,
                    exit_price=exit_price,
                    liquidation_price=liq,
                    stop_price=stop,
                    take_profit_price=tp,
                    net_pnl=net_pnl,
                    return_pct=net_pnl / (equity - net_pnl) * 100 if equity - net_pnl else 0.0,
                    exit_reason=reason,
                    equity_after=equity,
                    note=signal.note,
                )
            )
            next_allowed = idx + 1
            break
        if equity <= 0:
            break
    return trades


def summarize(trades: list[LeveragedTrade], initial_equity: float) -> dict[str, float | int]:
    equity = initial_equity
    peak = initial_equity
    max_dd = 0.0
    wins = 0
    liqs = 0
    gross_profit = 0.0
    gross_loss = 0.0
    for trade in trades:
        pnl = trade.net_pnl
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100 if peak else 0.0)
        if pnl > 0:
            wins += 1
            gross_profit += pnl
        else:
            gross_loss -= pnl
        if trade.exit_reason == "强平":
            liqs += 1
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "liquidations": liqs,
        "win_rate_pct": wins / len(trades) * 100 if trades else 0.0,
        "final_equity": equity,
        "return_pct": (equity / initial_equity - 1) * 100 if initial_equity else 0.0,
        "max_drawdown_pct": max_dd,
        "profit_factor": gross_profit / gross_loss if gross_loss else math.inf if gross_profit else 0.0,
    }


def current_market_snapshot(candles_15m: list[Candle], candles_4h: list[Candle]) -> dict[str, float | str | bool]:
    closes = [c.close for c in candles_4h]
    ema120 = compute_ema(closes, 120)
    rsi = compute_rsi(closes, 14)
    atr = compute_atr(candles_4h, 14)
    low80 = rolling_low(candles_4h, 80)
    high80 = rolling_high(candles_4h, 80)
    last = candles_4h[-1]
    prev = candles_4h[-2]
    price = candles_15m[-1].close
    near_low = last.low <= low80[-1] * 1.012 if low80[-1] is not None else False
    bullish = last.close > last.open and last.close > prev.close
    bottom_signal = bool(near_low and bullish and rsi[-1] is not None and rsi[-1] <= 38 and ema120[-1] is not None and last.close < ema120[-1] * 1.02)
    return {
        "time": format_ts(candles_15m[-1].ts),
        "price": price,
        "h4_close": last.close,
        "ema120": ema120[-1] or 0.0,
        "rsi14": rsi[-1] or 0.0,
        "atr14": atr[-1] or 0.0,
        "atr_pct": (atr[-1] / last.close * 100) if atr[-1] else 0.0,
        "low80": low80[-1] or 0.0,
        "high80": high80[-1] or 0.0,
        "near_80_low": near_low,
        "bullish_reversal": bullish,
        "bottom_signal": bottom_signal,
    }


def export_trades(path: Path, trades: list[LeveragedTrade]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["方向", "入场时间", "出场时间", "入场价", "出场价", "强平价", "止损价", "止盈价", "净盈亏", "收益率%", "出场原因", "权益", "备注"])
        for trade in trades:
            writer.writerow(
                [
                    "做多" if trade.side == "long" else "做空",
                    format_ts(trade.entry_ts),
                    format_ts(trade.exit_ts),
                    f"{trade.entry_price:.2f}",
                    f"{trade.exit_price:.2f}",
                    f"{trade.liquidation_price:.2f}",
                    f"{trade.stop_price:.2f}",
                    f"{trade.take_profit_price:.2f}",
                    f"{trade.net_pnl:.2f}",
                    f"{trade.return_pct:.2f}",
                    trade.exit_reason,
                    f"{trade.equity_after:.2f}",
                    trade.note,
                ]
            )


def print_results(rows: list[tuple[HighLeverageConfig, dict[str, float | int]]], snapshot: dict[str, float | str | bool]) -> None:
    print("高杠杆抄底/摸顶反转策略回测")
    print("策略：80根4H极值 + RSI极端 + 4H反转K，下一根15m开盘进场")
    print()
    print("当前市场快照：")
    print(f"时间：{snapshot['time']}，BTC-USDT：{snapshot['price']:.2f}")
    print(f"4H RSI14：{snapshot['rsi14']:.2f}，EMA120：{snapshot['ema120']:.2f}，ATR14：{snapshot['atr14']:.2f} ({snapshot['atr_pct']:.2f}%)")
    print(f"80根4H低点：{snapshot['low80']:.2f}，80根4H高点：{snapshot['high80']:.2f}")
    print(f"接近80根低点：{snapshot['near_80_low']}，4H反转阳线：{snapshot['bullish_reversal']}，抄底信号：{snapshot['bottom_signal']}")
    print()
    print("杠杆 | 保证金占权益 | 交易 | 胜率 | 强平 | 收益 | 最大回撤 | 盈亏比因子 | 最终权益")
    print("-" * 92)
    for config, metrics in rows:
        print(
            f"{config.leverage:g}x | {config.margin_pct*100:.1f}% | {metrics['trades']} | "
            f"{metrics['win_rate_pct']:.2f}% | {metrics['liquidations']} | {metrics['return_pct']:.2f}% | "
            f"{metrics['max_drawdown_pct']:.2f}% | {metrics['profit_factor']:.2f} | {metrics['final_equity']:.2f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="高杠杆 BTC-USDT 抄底/摸顶策略回测。")
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--initial-equity", type=float, default=1000.0)
    parser.add_argument("--margin-pct", type=float, default=0.05)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--leverages", default="20,50,100")
    parser.add_argument("--export-dir", type=Path, default=Path("data"))
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    args = build_parser().parse_args(argv)
    try:
        candles_15m = load_candles_csv(latest_cache(args.cache_dir, "okx_BTC-USDT_15m_*.csv"))
        candles_4h = load_candles_csv(latest_cache(args.cache_dir, "okx_BTC-USDT_4H_*.csv"))
        start_ts = max(candles_15m[0].ts + 20 * 24 * 60 * 60 * 1000, candles_4h[0].ts + 25 * 24 * 60 * 60 * 1000)
        end_ts = candles_15m[-1].ts + BAR_MS["15m"]
        signals = generate_extreme_reversal_signals(candles_15m, candles_4h)
        rows = []
        for leverage in [float(x.strip()) for x in args.leverages.split(",") if x.strip()]:
            config = HighLeverageConfig(
                initial_equity=args.initial_equity,
                leverage=leverage,
                margin_pct=args.margin_pct,
                fee_rate=args.fee_rate,
                slippage_bps=args.slippage_bps,
            )
            trades = simulate_high_leverage(candles_15m, signals, config, start_ts, end_ts)
            export_trades(args.export_dir / f"high_leverage_reversal_{int(leverage)}x.csv", trades)
            rows.append((config, summarize(trades, args.initial_equity)))
        print_results(rows, current_market_snapshot(candles_15m, candles_4h))
        print()
        print(f"交易明细目录：{args.export_dir}")
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
