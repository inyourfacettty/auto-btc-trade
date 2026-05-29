#!/usr/bin/env python3
"""Unified backtest for GitHub-inspired BTC-USDT-SWAP strategy families."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from backtest_okx_btc import BAR_MS, Candle, compute_ema, compute_rsi, format_ts, load_candles_csv
from realtime_recommendation import wick_ratio
from strategy_research import compute_atr, latest_cache, rolling_high, rolling_low, split_periods


@dataclass(frozen=True)
class Signal:
    family: str
    side: str
    entry_idx: int
    stop_price: float
    target_r: float
    note: str


@dataclass(frozen=True)
class StrategySpec:
    name: str
    families: tuple[str, ...]
    target_r: float | None = None
    max_hold_hours: float = 720.0
    use_atr_lock: bool = True


@dataclass(frozen=True)
class RunConfig:
    strategy_name: str
    families: tuple[str, ...]
    leverage: float
    margin_pct: float
    target_r_override: float | None
    max_hold_hours: float
    use_atr_lock: bool
    initial_equity: float = 1000.0
    fee_rate: float = 0.0005
    slippage_bps: float = 1.0


@dataclass(frozen=True)
class Trade:
    strategy_name: str
    family: str
    side: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    initial_stop_price: float
    final_stop_price: float
    target_price: float
    liquidation_price: float
    leverage: float
    margin_pct: float
    margin_used: float
    notional: float
    net_pnl: float
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


def rolling_mean(values: list[float], window: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if window <= 0:
        return result
    total = 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        if i >= window - 1:
            result[i] = total / window
    return result


def bollinger_bands(values: list[float], period: int, stdev_mult: float) -> tuple[list[float | None], list[float | None], list[float | None]]:
    lower: list[float | None] = [None] * len(values)
    mid: list[float | None] = [None] * len(values)
    upper: list[float | None] = [None] * len(values)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        average = statistics.mean(window)
        deviation = statistics.pstdev(window)
        mid[i] = average
        lower[i] = average - deviation * stdev_mult
        upper[i] = average + deviation * stdev_mult
    return lower, mid, upper


def compute_mfi(candles: list[Candle], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(candles)
    if len(candles) <= period:
        return result

    typical = [(c.high + c.low + c.close) / 3 for c in candles]
    positive = [0.0] * len(candles)
    negative = [0.0] * len(candles)
    for i in range(1, len(candles)):
        money_flow = typical[i] * candles[i].volume
        if typical[i] > typical[i - 1]:
            positive[i] = money_flow
        elif typical[i] < typical[i - 1]:
            negative[i] = money_flow

    for i in range(period, len(candles)):
        pos_sum = sum(positive[i - period + 1 : i + 1])
        neg_sum = sum(negative[i - period + 1 : i + 1])
        if neg_sum == 0:
            result[i] = 100.0 if pos_sum > 0 else 50.0
        else:
            ratio = pos_sum / neg_sum
            result[i] = 100 - 100 / (1 + ratio)
    return result


def compute_adx(candles: list[Candle], period: int) -> list[float | None]:
    adx: list[float | None] = [None] * len(candles)
    if len(candles) <= period * 2:
        return adx

    true_ranges = [0.0] * len(candles)
    plus_dm = [0.0] * len(candles)
    minus_dm = [0.0] * len(candles)
    for i in range(1, len(candles)):
        high_diff = candles[i].high - candles[i - 1].high
        low_diff = candles[i - 1].low - candles[i].low
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0.0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0.0
        prev_close = candles[i - 1].close
        true_ranges[i] = max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - prev_close),
            abs(candles[i].low - prev_close),
        )

    tr_smooth = sum(true_ranges[1 : period + 1])
    plus_smooth = sum(plus_dm[1 : period + 1])
    minus_smooth = sum(minus_dm[1 : period + 1])
    dx: list[float | None] = [None] * len(candles)

    for i in range(period, len(candles)):
        if i > period:
            tr_smooth = tr_smooth - tr_smooth / period + true_ranges[i]
            plus_smooth = plus_smooth - plus_smooth / period + plus_dm[i]
            minus_smooth = minus_smooth - minus_smooth / period + minus_dm[i]
        if tr_smooth <= 0:
            continue
        plus_di = 100 * plus_smooth / tr_smooth
        minus_di = 100 * minus_smooth / tr_smooth
        denom = plus_di + minus_di
        dx[i] = 0.0 if denom == 0 else 100 * abs(plus_di - minus_di) / denom

    seed_values = [value for value in dx[period : period * 2] if value is not None]
    if len(seed_values) < period:
        return adx
    adx[period * 2 - 1] = sum(seed_values[:period]) / period
    for i in range(period * 2, len(candles)):
        if dx[i] is None or adx[i - 1] is None:
            continue
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx


def generate_trend_breakout_signals(candles_15m: list[Candle], candles_4h: list[Candle]) -> list[Signal]:
    closes = [c.close for c in candles_4h]
    volumes = [c.volume for c in candles_4h]
    ema20 = compute_ema(closes, 20)
    ema60 = compute_ema(closes, 60)
    ema120 = compute_ema(closes, 120)
    rsi = compute_rsi(closes, 14)
    atr = compute_atr(candles_4h, 14)
    adx = compute_adx(candles_4h, 14)
    volume_ma20 = rolling_mean(volumes, 20)
    high40 = rolling_high(candles_4h, 40)
    low40 = rolling_low(candles_4h, 40)
    signals: list[Signal] = []

    for i in range(130, len(candles_4h)):
        required = (ema20[i], ema60[i], ema120[i], rsi[i], atr[i], adx[i], volume_ma20[i], high40[i], low40[i])
        if any(value is None for value in required):
            continue
        candle = candles_4h[i]
        atr_pct = atr[i] / candle.close if candle.close else 0.0
        entry_idx = first_15m_at_or_after(candles_15m, candle.ts + BAR_MS["4H"])
        if entry_idx is None or not (0.004 <= atr_pct <= 0.05):
            continue
        if candle.volume < volume_ma20[i]:
            continue

        if candle.close > high40[i] and candle.close > ema120[i] and ema20[i] > ema60[i] and rsi[i] >= 52 and adx[i] >= 20:
            stop = candle.close - atr[i] * 2.2
            signals.append(Signal("趋势突破增强", "long", entry_idx, stop, 3.5, "4H突破40根高点+趋势/成交量确认"))
        elif candle.close < low40[i] and candle.close < ema120[i] and ema20[i] < ema60[i] and rsi[i] <= 48 and adx[i] >= 20:
            stop = candle.close + atr[i] * 2.2
            signals.append(Signal("趋势突破增强", "short", entry_idx, stop, 3.5, "4H跌破40根低点+趋势/成交量确认"))
    return signals


def generate_hlhb_signals(candles_15m: list[Candle], candles_4h: list[Candle]) -> list[Signal]:
    closes = [c.close for c in candles_4h]
    volumes = [c.volume for c in candles_4h]
    ema5 = compute_ema(closes, 5)
    ema10 = compute_ema(closes, 10)
    ema60 = compute_ema(closes, 60)
    rsi10 = compute_rsi(closes, 10)
    atr = compute_atr(candles_4h, 14)
    adx = compute_adx(candles_4h, 14)
    volume_ma20 = rolling_mean(volumes, 20)
    signals: list[Signal] = []

    for i in range(70, len(candles_4h)):
        required = (ema5[i], ema10[i], ema60[i], rsi10[i], rsi10[i - 1], atr[i], adx[i], volume_ma20[i])
        if any(value is None for value in required):
            continue
        candle = candles_4h[i]
        atr_pct = atr[i] / candle.close if candle.close else 0.0
        entry_idx = first_15m_at_or_after(candles_15m, candle.ts + BAR_MS["4H"])
        if entry_idx is None or not (0.004 <= atr_pct <= 0.05):
            continue
        if candle.volume < volume_ma20[i] * 0.8 or adx[i] < 25:
            continue

        crossed_up = rsi10[i - 1] <= 50 < rsi10[i] and ema5[i - 1] <= ema10[i - 1] and ema5[i] > ema10[i]
        crossed_down = rsi10[i - 1] >= 50 > rsi10[i] and ema5[i - 1] >= ema10[i - 1] and ema5[i] < ema10[i]
        if crossed_up and candle.close > ema60[i]:
            stop = candle.close - atr[i] * 1.8
            signals.append(Signal("HLHB趋势确认", "long", entry_idx, stop, 2.8, "RSI10上穿50+EMA5上穿EMA10+ADX确认"))
        elif crossed_down and candle.close < ema60[i]:
            stop = candle.close + atr[i] * 1.8
            signals.append(Signal("HLHB趋势确认", "short", entry_idx, stop, 2.8, "RSI10下穿50+EMA5下穿EMA10+ADX确认"))
    return signals


def generate_bollinger_reversal_signals(candles_15m: list[Candle], candles_4h: list[Candle]) -> list[Signal]:
    closes = [c.close for c in candles_4h]
    ema20 = compute_ema(closes, 20)
    ema60 = compute_ema(closes, 60)
    ema120 = compute_ema(closes, 120)
    rsi = compute_rsi(closes, 14)
    mfi = compute_mfi(candles_4h, 14)
    atr = compute_atr(candles_4h, 14)
    adx = compute_adx(candles_4h, 14)
    lower, _middle, upper = bollinger_bands(closes, 20, 2.0)
    signals: list[Signal] = []

    for i in range(130, len(candles_4h)):
        required = (ema20[i], ema60[i], ema120[i], rsi[i], mfi[i], atr[i], adx[i], lower[i], upper[i])
        if any(value is None for value in required):
            continue
        candle = candles_4h[i]
        atr_pct = atr[i] / candle.close if candle.close else 0.0
        entry_idx = first_15m_at_or_after(candles_15m, candle.ts + BAR_MS["4H"])
        if entry_idx is None or not (0.005 <= atr_pct <= 0.06):
            continue

        lower_wick = wick_ratio(candle.open, candle.high, candle.low, candle.close, "long")
        upper_wick = wick_ratio(candle.open, candle.high, candle.low, candle.close, "short")
        strong_downtrend = candle.close < ema120[i] and ema20[i] < ema60[i] and adx[i] > 35
        strong_uptrend = candle.close > ema120[i] and ema20[i] > ema60[i] and adx[i] > 35

        if (
            candle.low < lower[i]
            and candle.close > lower[i]
            and rsi[i] <= 35
            and mfi[i] <= 40
            and lower_wick >= 0.35
            and not strong_downtrend
        ):
            stop = min(candle.low - atr[i] * 0.2, candle.close - atr[i] * 1.2)
            signals.append(Signal("布林反转", "long", entry_idx, stop, 2.0, "跌破布林下轨后收回+RSI/MFI低位+下影线"))
        elif (
            candle.high > upper[i]
            and candle.close < upper[i]
            and rsi[i] >= 65
            and mfi[i] >= 60
            and upper_wick >= 0.35
            and not strong_uptrend
        ):
            stop = max(candle.high + atr[i] * 0.2, candle.close + atr[i] * 1.2)
            signals.append(Signal("布林反转", "short", entry_idx, stop, 2.0, "突破布林上轨后收回+RSI/MFI高位+上影线"))
    return signals


def liquidation_price(entry_price: float, leverage: float, side: str) -> float:
    gap = 0.90 / leverage
    if side == "long":
        return entry_price * (1 - gap)
    if side == "short":
        return entry_price * (1 + gap)
    raise ValueError(f"Unsupported side: {side}")


def simulate(
    candles_15m: list[Candle],
    signals: list[Signal],
    config: RunConfig,
    start_ts: int,
    end_ts: int,
) -> list[Trade]:
    selected = [signal for signal in signals if signal.family in config.families]
    trades: list[Trade] = []
    equity = config.initial_equity
    next_allowed = 0
    slip = config.slippage_bps / 10_000
    max_hold_ms = int(config.max_hold_hours * 60 * 60 * 1000)

    for signal in sorted(selected, key=lambda s: s.entry_idx):
        if signal.entry_idx < next_allowed or signal.entry_idx >= len(candles_15m):
            continue
        entry_candle = candles_15m[signal.entry_idx]
        if not (start_ts <= entry_candle.ts < end_ts):
            continue
        if equity <= 0:
            break

        entry = entry_candle.open * (1 + slip) if signal.side == "long" else entry_candle.open * (1 - slip)
        initial_stop = signal.stop_price
        if signal.side == "long" and initial_stop >= entry:
            continue
        if signal.side == "short" and initial_stop <= entry:
            continue

        margin_used = equity * config.margin_pct
        notional = margin_used * config.leverage
        if margin_used <= 0 or notional <= 0:
            continue

        target_r = config.target_r_override if config.target_r_override is not None else signal.target_r
        risk = abs(entry - initial_stop)
        target = entry + risk * target_r if signal.side == "long" else entry - risk * target_r
        liq = liquidation_price(entry, config.leverage, signal.side)
        qty = notional / entry
        entry_fee = notional * config.fee_rate
        active_stop = initial_stop
        stop_level = 0
        equity_before = equity
        max_exit_ts = entry_candle.ts + max_hold_ms

        for idx in range(signal.entry_idx, len(candles_15m)):
            candle = candles_15m[idx]
            exit_price: float | None = None
            reason: str | None = None

            if candle.ts >= end_ts:
                exit_price = candle.open * (1 - slip) if signal.side == "long" else candle.open * (1 + slip)
                reason = "区间结束"
            elif candle.ts >= max_exit_ts:
                exit_price = candle.open * (1 - slip) if signal.side == "long" else candle.open * (1 + slip)
                reason = "时间退出"
            elif signal.side == "long" and candle.low <= active_stop:
                exit_price = active_stop * (1 - slip)
                reason = "初始止损" if stop_level == 0 else "保本/锁盈止损"
            elif signal.side == "short" and candle.high >= active_stop:
                exit_price = active_stop * (1 + slip)
                reason = "初始止损" if stop_level == 0 else "保本/锁盈止损"
            elif signal.side == "long" and candle.low <= liq:
                exit_price = liq
                reason = "强平"
            elif signal.side == "short" and candle.high >= liq:
                exit_price = liq
                reason = "强平"
            elif signal.side == "long" and candle.high >= target:
                exit_price = target * (1 - slip)
                reason = "目标止盈"
            elif signal.side == "short" and candle.low <= target:
                exit_price = target * (1 + slip)
                reason = "目标止盈"
            elif idx == len(candles_15m) - 1:
                exit_price = candle.close * (1 - slip) if signal.side == "long" else candle.close * (1 + slip)
                reason = "数据结束"

            if exit_price is not None and reason is not None:
                if reason == "强平":
                    net = -margin_used - entry_fee
                else:
                    move = (exit_price - entry) / entry if signal.side == "long" else (entry - exit_price) / entry
                    gross = notional * move
                    exit_fee = qty * exit_price * config.fee_rate
                    net = gross - entry_fee - exit_fee
                equity_after = equity + net
                trades.append(
                    Trade(
                        strategy_name=config.strategy_name,
                        family=signal.family,
                        side=signal.side,
                        entry_ts=entry_candle.ts,
                        exit_ts=candle.ts,
                        entry_price=entry,
                        exit_price=exit_price,
                        initial_stop_price=initial_stop,
                        final_stop_price=active_stop,
                        target_price=target,
                        liquidation_price=liq,
                        leverage=config.leverage,
                        margin_pct=config.margin_pct,
                        margin_used=margin_used,
                        notional=notional,
                        net_pnl=net,
                        equity_before=equity,
                        equity_after=equity_after,
                        exit_reason=reason,
                        hold_hours=(candle.ts - entry_candle.ts) / 3_600_000,
                        note=signal.note,
                    )
                )
                equity = equity_after
                next_allowed = bisect.bisect_right([c.ts for c in candles_15m], candle.ts)
                break

            if config.use_atr_lock:
                if signal.side == "long":
                    if candle.close >= entry + risk * 2 and active_stop < entry + risk:
                        active_stop = entry + risk
                        stop_level = 2
                    elif candle.close >= entry + risk and active_stop < entry:
                        active_stop = entry
                        stop_level = max(stop_level, 1)
                else:
                    if candle.close <= entry - risk * 2 and active_stop > entry - risk:
                        active_stop = entry - risk
                        stop_level = 2
                    elif candle.close <= entry - risk and active_stop > entry:
                        active_stop = entry
                        stop_level = max(stop_level, 1)
    return trades


def summarize(trades: list[Trade], initial_equity: float) -> dict[str, float | int | str]:
    equity = trades[-1].equity_after if trades else initial_equity
    wins = [trade for trade in trades if trade.net_pnl > 0]
    losses = [trade for trade in trades if trade.net_pnl <= 0]
    peak = initial_equity
    max_dd = 0.0
    for trade in trades:
        peak = max(peak, trade.equity_before, trade.equity_after)
        if peak > 0:
            max_dd = max(max_dd, (peak - trade.equity_after) / peak * 100)
    gross_win = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
        "final": equity,
        "return_pct": (equity / initial_equity - 1) * 100 if initial_equity else 0.0,
        "max_dd": max_dd,
        "avg_win": statistics.mean([t.net_pnl for t in wins]) if wins else 0.0,
        "avg_loss": statistics.mean([t.net_pnl for t in losses]) if losses else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss else (math.inf if gross_win > 0 else 0.0),
        "target": sum(1 for t in trades if t.exit_reason == "目标止盈"),
        "initial_stop": sum(1 for t in trades if t.exit_reason == "初始止损"),
        "locked_stop": sum(1 for t in trades if t.exit_reason == "保本/锁盈止损"),
        "liq": sum(1 for t in trades if t.exit_reason == "强平"),
        "avg_hold": statistics.mean([t.hold_hours for t in trades]) if trades else 0.0,
        "max_loss_streak": max_loss_streak(trades),
    }


def max_loss_streak(trades: list[Trade]) -> int:
    current = 0
    best = 0
    for trade in trades:
        if trade.net_pnl <= 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def export_trades(path: Path, trades: list[Trade]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "策略",
            "信号族",
            "方向",
            "入场时间",
            "出场时间",
            "入场价",
            "出场价",
            "初始止损",
            "最终止损",
            "目标价",
            "强平价",
            "杠杆",
            "保证金比例",
            "保证金U",
            "名义仓位U",
            "盈亏U",
            "入场前权益",
            "出场后权益",
            "出场原因",
            "持仓小时",
            "备注",
        ])
        for trade in trades:
            writer.writerow([
                trade.strategy_name,
                trade.family,
                "做多" if trade.side == "long" else "做空",
                format_ts(trade.entry_ts),
                format_ts(trade.exit_ts),
                f"{trade.entry_price:.2f}",
                f"{trade.exit_price:.2f}",
                f"{trade.initial_stop_price:.2f}",
                f"{trade.final_stop_price:.2f}",
                f"{trade.target_price:.2f}",
                f"{trade.liquidation_price:.2f}",
                f"{trade.leverage:.0f}",
                f"{trade.margin_pct * 100:.0f}%",
                f"{trade.margin_used:.2f}",
                f"{trade.notional:.2f}",
                f"{trade.net_pnl:.2f}",
                f"{trade.equity_before:.2f}",
                f"{trade.equity_after:.2f}",
                trade.exit_reason,
                f"{trade.hold_hours:.2f}",
                trade.note,
            ])


def export_summary(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "策略",
        "杠杆",
        "保证金比例",
        "交易数",
        "胜率",
        "全年收益",
        "最终权益",
        "最大回撤",
        "后1/3交易数",
        "后1/3收益",
        "平均盈利",
        "平均亏损",
        "盈亏因子",
        "目标止盈",
        "初始止损",
        "保本锁盈止损",
        "强平",
        "最大连亏",
        "平均持仓小时",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统一回测 GitHub 思路衍生 BTC 策略")
    parser.add_argument("--inst-id", default="BTC-USDT-SWAP")
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--export-dir", type=Path, default=Path("data"))
    parser.add_argument("--initial-equity", type=float, default=1000.0)
    parser.add_argument("--days", type=int, default=365)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    args = build_parser().parse_args(argv)
    candles_15m = load_candles_csv(latest_cache(args.cache_dir, f"okx_{args.inst_id}_15m_*.csv"))
    candles_4h = load_candles_csv(latest_cache(args.cache_dir, f"okx_{args.inst_id}_4H_*.csv"))
    end_ts = candles_15m[-1].ts + BAR_MS["15m"]
    start_ts = end_ts - args.days * 24 * 60 * 60 * 1000
    _train_start, _train_end, test_start, _test_end = split_periods(start_ts, end_ts)

    signals = (
        generate_trend_breakout_signals(candles_15m, candles_4h)
        + generate_hlhb_signals(candles_15m, candles_4h)
        + generate_bollinger_reversal_signals(candles_15m, candles_4h)
    )
    specs = [
        StrategySpec("趋势突破增强_ATR风控", ("趋势突破增强",)),
        StrategySpec("HLHB趋势确认_ATR风控", ("HLHB趋势确认",)),
        StrategySpec("布林反转_ATR风控", ("布林反转",)),
        StrategySpec("三策略组合_ATR风控", ("趋势突破增强", "HLHB趋势确认", "布林反转")),
    ]
    leverages = [3.0, 5.0, 8.0]
    margin_pcts = [0.10, 0.20, 0.30]

    rows: list[dict[str, float | int | str]] = []
    all_runs: list[tuple[RunConfig, list[Trade], dict[str, float | int | str]]] = []
    for spec in specs:
        for leverage in leverages:
            for margin_pct in margin_pcts:
                config = RunConfig(
                    strategy_name=spec.name,
                    families=spec.families,
                    leverage=leverage,
                    margin_pct=margin_pct,
                    target_r_override=spec.target_r,
                    max_hold_hours=spec.max_hold_hours,
                    use_atr_lock=spec.use_atr_lock,
                    initial_equity=args.initial_equity,
                )
                trades = simulate(candles_15m, signals, config, start_ts, end_ts)
                test_trades = simulate(candles_15m, signals, config, test_start, end_ts)
                summary = summarize(trades, args.initial_equity)
                test_summary = summarize(test_trades, args.initial_equity)
                all_runs.append((config, trades, summary))
                rows.append({
                    "策略": spec.name,
                    "杠杆": f"{leverage:.0f}x",
                    "保证金比例": f"{margin_pct * 100:.0f}%",
                    "交易数": summary["trades"],
                    "胜率": f"{summary['win_rate']:.2f}%",
                    "全年收益": f"{summary['return_pct']:.2f}%",
                    "最终权益": f"{summary['final']:.2f}",
                    "最大回撤": f"{summary['max_dd']:.2f}%",
                    "后1/3交易数": test_summary["trades"],
                    "后1/3收益": f"{test_summary['return_pct']:.2f}%",
                    "平均盈利": f"{summary['avg_win']:.2f}",
                    "平均亏损": f"{summary['avg_loss']:.2f}",
                    "盈亏因子": "inf" if math.isinf(float(summary["profit_factor"])) else f"{summary['profit_factor']:.2f}",
                    "目标止盈": summary["target"],
                    "初始止损": summary["initial_stop"],
                    "保本锁盈止损": summary["locked_stop"],
                    "强平": summary["liq"],
                    "最大连亏": summary["max_loss_streak"],
                    "平均持仓小时": f"{summary['avg_hold']:.2f}",
                })

    summary_path = args.export_dir / "unified_strategy_summary.csv"
    export_summary(summary_path, rows)

    best_by_strategy: list[tuple[RunConfig, list[Trade], dict[str, float | int | str]]] = []
    for spec in specs:
        candidates = [run for run in all_runs if run[0].strategy_name == spec.name and run[2]["trades"]]
        if not candidates:
            continue
        best = sorted(
            candidates,
            key=lambda run: (float(run[2]["return_pct"]), -float(run[2]["max_dd"]), float(run[2]["win_rate"])),
            reverse=True,
        )[0]
        best_by_strategy.append(best)
        config, trades, _summary = best
        file_name = f"unified_best_{config.strategy_name}_{config.leverage:.0f}x_{config.margin_pct * 100:.0f}pct.csv"
        export_trades(args.export_dir / file_name, trades)

    print(f"回测区间：{format_ts(start_ts)} -> {format_ts(end_ts)}")
    print("设定：BTC-USDT-SWAP，1000U本金，手续费0.05%，滑点1bp；4H收盘确认，下一根15m开盘入场。")
    print("统一扫描：杠杆 3x/5x/8x；保证金比例 10%/20%/30%。")
    print()
    print("各策略最优组合：")
    print("策略 | 最优杠杆/保证金 | 交易 | 胜率 | 收益 | 最大回撤 | 后1/3收益 | 平均盈利/亏损 | 止盈/初始止损/锁盈止损")
    print("-" * 150)
    for config, _trades, summary in best_by_strategy:
        test_trades = simulate(candles_15m, signals, config, test_start, end_ts)
        test_summary = summarize(test_trades, args.initial_equity)
        print(
            f"{config.strategy_name} | {config.leverage:.0f}x/{config.margin_pct * 100:.0f}% | "
            f"{summary['trades']} | {summary['win_rate']:.1f}% | {summary['return_pct']:.1f}% | "
            f"{summary['max_dd']:.1f}% | {test_summary['return_pct']:.1f}% | "
            f"{summary['avg_win']:.2f}/{summary['avg_loss']:.2f}U | "
            f"{summary['target']}/{summary['initial_stop']}/{summary['locked_stop']}"
        )
    print()
    print(f"汇总CSV：{summary_path}")
    print("每套策略最优组合明细：data\\unified_best_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
