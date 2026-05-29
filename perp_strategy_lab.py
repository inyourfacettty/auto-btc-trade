#!/usr/bin/env python3
"""Search BTC-USDT-SWAP long/short strategies with out-of-sample checks."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from backtest_okx_btc import BAR_MS, Candle, compute_ema, compute_rsi, format_ts, load_candles_csv
from strategy_research import compute_atr, latest_cache, rolling_high, rolling_low, split_periods
from unified_strategy_backtest import compute_adx


MS_PER_HOUR = 60 * 60 * 1000
LIQUIDATION_BUFFER = 0.90


@dataclass(frozen=True)
class Signal:
    family: str
    side: str
    entry_idx: int
    stop_price: float
    atr_price: float
    note: str


@dataclass(frozen=True)
class Config:
    family: str
    name: str
    leverage: float
    margin_pct: float
    stop_mult: float
    trail_mult: float
    max_hold_hours: float
    take_profit_r: float | None
    breakeven_r: float
    initial_equity: float = 1000.0
    fee_rate: float = 0.0005
    slippage_bps: float = 1.0


@dataclass(frozen=True)
class Trade:
    strategy: str
    family: str
    side: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    initial_stop: float
    final_stop: float
    target_price: float | None
    liquidation_price: float
    leverage: float
    margin_pct: float
    margin_used: float
    notional: float
    net_pnl: float
    return_pct: float
    equity_before: float
    equity_after: float
    exit_reason: str
    hold_hours: float
    note: str


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def resample_candles(candles: list[Candle], interval_ms: int) -> list[Candle]:
    buckets: dict[int, list[Candle]] = {}
    for candle in candles:
        bucket_ts = candle.ts - candle.ts % interval_ms
        buckets.setdefault(bucket_ts, []).append(candle)

    result: list[Candle] = []
    expected = interval_ms // BAR_MS["15m"]
    for bucket_ts in sorted(buckets):
        chunk = buckets[bucket_ts]
        if len(chunk) != expected:
            continue
        result.append(
            Candle(
                ts=bucket_ts,
                open=chunk[0].open,
                high=max(c.high for c in chunk),
                low=min(c.low for c in chunk),
                close=chunk[-1].close,
                volume=sum(c.volume for c in chunk),
            )
        )
    return result


def first_15m_at_or_after(candles_15m: list[Candle], timestamps: list[int], ts: int) -> int | None:
    idx = bisect.bisect_left(timestamps, ts)
    return idx if idx < len(candles_15m) else None


def sma(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    total = 0.0
    for idx, value in enumerate(values):
        total += value
        if idx >= period:
            total -= values[idx - period]
        if idx >= period - 1:
            result[idx] = total / period
    return result


def rolling_std(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    for idx in range(period - 1, len(values)):
        result[idx] = statistics.pstdev(values[idx - period + 1 : idx + 1])
    return result


def percentile_rank(values: list[float | None], lookback: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    for idx in range(lookback, len(values)):
        current = values[idx]
        if current is None:
            continue
        window = [v for v in values[idx - lookback : idx] if v is not None]
        if not window:
            continue
        result[idx] = sum(1 for v in window if v <= current) / len(window)
    return result


def liquidation_price(entry: float, leverage: float, side: str) -> float:
    gap = LIQUIDATION_BUFFER / leverage
    if side == "long":
        return entry * (1 - gap)
    if side == "short":
        return entry * (1 + gap)
    raise ValueError(f"Unsupported side: {side}")


def build_donchian_trend_signals(
    family: str,
    candles_15m: list[Candle],
    signal_candles: list[Candle],
    entry_timestamps: list[int],
    interval_ms: int,
    lookback: int,
    ema_period: int,
    stop_mult: float,
    adx_floor: float,
    atr_pct_floor: float,
    atr_pct_ceiling: float,
) -> list[Signal]:
    closes = [c.close for c in signal_candles]
    ema = compute_ema(closes, ema_period)
    atr = compute_atr(signal_candles, 14)
    adx = compute_adx(signal_candles, 14)
    highest = rolling_high(signal_candles, lookback)
    lowest = rolling_low(signal_candles, lookback)
    signals: list[Signal] = []

    start = max(ema_period + 1, lookback + 1, 32)
    for idx in range(start, len(signal_candles)):
        required = (ema[idx], atr[idx], adx[idx], highest[idx], lowest[idx])
        if any(v is None for v in required):
            continue
        candle = signal_candles[idx]
        atr_value = float(atr[idx])
        atr_pct = atr_value / candle.close if candle.close else 0.0
        if not (atr_pct_floor <= atr_pct <= atr_pct_ceiling) or float(adx[idx]) < adx_floor:
            continue
        entry_idx = first_15m_at_or_after(candles_15m, entry_timestamps, candle.ts + interval_ms)
        if entry_idx is None:
            continue
        if candle.close > float(highest[idx]) and candle.close > float(ema[idx]):
            stop = candle.close - atr_value * stop_mult
            signals.append(Signal(family, "long", entry_idx, stop, atr_value, f"{family} 向上突破{lookback}根高点"))
        elif candle.close < float(lowest[idx]) and candle.close < float(ema[idx]):
            stop = candle.close + atr_value * stop_mult
            signals.append(Signal(family, "short", entry_idx, stop, atr_value, f"{family} 向下跌破{lookback}根低点"))
    return signals


def build_squeeze_breakout_signals(
    family: str,
    candles_15m: list[Candle],
    signal_candles: list[Candle],
    entry_timestamps: list[int],
    interval_ms: int,
    range_lookback: int,
    squeeze_lookback: int,
    squeeze_rank_max: float,
    stop_mult: float,
    ema_period: int,
) -> list[Signal]:
    closes = [c.close for c in signal_candles]
    middle = sma(closes, 20)
    std = rolling_std(closes, 20)
    width: list[float | None] = [None] * len(signal_candles)
    for idx in range(len(signal_candles)):
        if middle[idx] is None or std[idx] is None or middle[idx] == 0:
            continue
        width[idx] = float(std[idx]) * 4 / float(middle[idx])
    width_rank = percentile_rank(width, squeeze_lookback)
    ema = compute_ema(closes, ema_period)
    atr = compute_atr(signal_candles, 14)
    highest = rolling_high(signal_candles, range_lookback)
    lowest = rolling_low(signal_candles, range_lookback)
    signals: list[Signal] = []

    start = max(squeeze_lookback + 1, range_lookback + 1, ema_period + 1)
    for idx in range(start, len(signal_candles)):
        required = (width_rank[idx - 1], ema[idx], atr[idx], highest[idx], lowest[idx])
        if any(v is None for v in required):
            continue
        if float(width_rank[idx - 1]) > squeeze_rank_max:
            continue
        candle = signal_candles[idx]
        atr_value = float(atr[idx])
        atr_pct = atr_value / candle.close if candle.close else 0.0
        if not (0.004 <= atr_pct <= 0.06):
            continue
        entry_idx = first_15m_at_or_after(candles_15m, entry_timestamps, candle.ts + interval_ms)
        if entry_idx is None:
            continue
        if candle.close > float(highest[idx]) and candle.close > float(ema[idx]):
            stop = candle.close - atr_value * stop_mult
            signals.append(Signal(family, "long", entry_idx, stop, atr_value, f"{family} 压缩后向上突破"))
        elif candle.close < float(lowest[idx]) and candle.close < float(ema[idx]):
            stop = candle.close + atr_value * stop_mult
            signals.append(Signal(family, "short", entry_idx, stop, atr_value, f"{family} 压缩后向下跌破"))
    return signals


def build_pullback_signals(
    family: str,
    candles_15m: list[Candle],
    candles_1h: list[Candle],
    entry_timestamps: list[int],
    stop_mult: float,
    ema_fast_period: int,
    ema_slow_period: int,
    rsi_period: int,
) -> list[Signal]:
    closes = [c.close for c in candles_1h]
    ema_fast = compute_ema(closes, ema_fast_period)
    ema_slow = compute_ema(closes, ema_slow_period)
    rsi = compute_rsi(closes, rsi_period)
    atr = compute_atr(candles_1h, 14)
    signals: list[Signal] = []

    start = max(ema_slow_period + 10, 80)
    for idx in range(start, len(candles_1h)):
        required = (ema_fast[idx], ema_slow[idx], ema_slow[idx - 12], rsi[idx], atr[idx])
        if any(v is None for v in required):
            continue
        candle = candles_1h[idx]
        previous = candles_1h[idx - 1]
        fast = float(ema_fast[idx])
        slow = float(ema_slow[idx])
        atr_value = float(atr[idx])
        atr_pct = atr_value / candle.close if candle.close else 0.0
        if not (0.004 <= atr_pct <= 0.05):
            continue
        entry_idx = first_15m_at_or_after(candles_15m, entry_timestamps, candle.ts + MS_PER_HOUR)
        if entry_idx is None:
            continue

        long_trend = candle.close > slow and fast > slow and slow > float(ema_slow[idx - 12])
        short_trend = candle.close < slow and fast < slow and slow < float(ema_slow[idx - 12])
        reclaimed_long = previous.close < fast and candle.close > fast and candle.close > candle.open
        reclaimed_short = previous.close > fast and candle.close < fast and candle.close < candle.open
        if long_trend and reclaimed_long and 42 <= float(rsi[idx]) <= 62:
            swing_stop = min(c.low for c in candles_1h[max(0, idx - 6) : idx + 1])
            stop = min(swing_stop, candle.close - atr_value * stop_mult)
            signals.append(Signal(family, "long", entry_idx, stop, atr_value, f"{family} 多头回踩后重新站上EMA{ema_fast_period}"))
        elif short_trend and reclaimed_short and 38 <= float(rsi[idx]) <= 58:
            swing_stop = max(c.high for c in candles_1h[max(0, idx - 6) : idx + 1])
            stop = max(swing_stop, candle.close + atr_value * stop_mult)
            signals.append(Signal(family, "short", entry_idx, stop, atr_value, f"{family} 空头反弹后重新跌破EMA{ema_fast_period}"))
    return signals


def simulate(candles_15m: list[Candle], signals: list[Signal], config: Config, start_ts: int, end_ts: int) -> list[Trade]:
    trades: list[Trade] = []
    selected = [s for s in signals if s.family == config.family]
    equity = config.initial_equity
    next_allowed_idx = 0
    slippage = config.slippage_bps / 10_000
    max_hold_ms = int(config.max_hold_hours * MS_PER_HOUR)

    for signal in sorted(selected, key=lambda s: s.entry_idx):
        if signal.entry_idx < next_allowed_idx or signal.entry_idx >= len(candles_15m):
            continue
        entry_candle = candles_15m[signal.entry_idx]
        if not (start_ts <= entry_candle.ts < end_ts) or equity <= 0:
            continue
        entry = entry_candle.open * (1 + slippage) if signal.side == "long" else entry_candle.open * (1 - slippage)
        initial_stop = signal.stop_price
        if signal.side == "long" and initial_stop >= entry:
            continue
        if signal.side == "short" and initial_stop <= entry:
            continue

        risk = abs(entry - initial_stop)
        if risk <= 0:
            continue
        target = None
        if config.take_profit_r is not None:
            target = entry + risk * config.take_profit_r if signal.side == "long" else entry - risk * config.take_profit_r
        liq = liquidation_price(entry, config.leverage, signal.side)
        margin_used = equity * config.margin_pct
        notional = margin_used * config.leverage
        qty = notional / entry
        entry_fee = notional * config.fee_rate
        equity_before = equity
        active_stop = initial_stop
        high_water = entry
        low_water = entry
        stop_stage = "初始止损"
        last_idx = signal.entry_idx

        for idx in range(signal.entry_idx, len(candles_15m)):
            candle = candles_15m[idx]
            last_idx = idx
            exit_price: float | None = None
            reason: str | None = None

            if candle.ts >= end_ts:
                exit_price = candle.open * (1 - slippage) if signal.side == "long" else candle.open * (1 + slippage)
                reason = "区间结束"
            elif candle.ts >= entry_candle.ts + max_hold_ms:
                exit_price = candle.open * (1 - slippage) if signal.side == "long" else candle.open * (1 + slippage)
                reason = "时间退出"
            elif signal.side == "long" and candle.low <= active_stop and active_stop > liq:
                exit_price = active_stop * (1 - slippage)
                reason = stop_stage
            elif signal.side == "short" and candle.high >= active_stop and active_stop < liq:
                exit_price = active_stop * (1 + slippage)
                reason = stop_stage
            elif signal.side == "long" and candle.low <= liq:
                exit_price = liq
                reason = "强平"
            elif signal.side == "short" and candle.high >= liq:
                exit_price = liq
                reason = "强平"
            elif signal.side == "long" and candle.low <= active_stop:
                exit_price = active_stop * (1 - slippage)
                reason = stop_stage
            elif signal.side == "short" and candle.high >= active_stop:
                exit_price = active_stop * (1 + slippage)
                reason = stop_stage
            elif target is not None and signal.side == "long" and candle.high >= target:
                exit_price = target * (1 - slippage)
                reason = "目标止盈"
            elif target is not None and signal.side == "short" and candle.low <= target:
                exit_price = target * (1 + slippage)
                reason = "目标止盈"
            elif idx == len(candles_15m) - 1:
                exit_price = candle.close * (1 - slippage) if signal.side == "long" else candle.close * (1 + slippage)
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
                        strategy=config.name,
                        family=config.family,
                        side=signal.side,
                        entry_ts=entry_candle.ts,
                        exit_ts=candle.ts,
                        entry_price=entry,
                        exit_price=exit_price,
                        initial_stop=initial_stop,
                        final_stop=active_stop,
                        target_price=target,
                        liquidation_price=liq,
                        leverage=config.leverage,
                        margin_pct=config.margin_pct,
                        margin_used=margin_used,
                        notional=notional,
                        net_pnl=net,
                        return_pct=net / equity_before * 100 if equity_before else 0.0,
                        equity_before=equity_before,
                        equity_after=equity_after,
                        exit_reason=reason,
                        hold_hours=(candle.ts - entry_candle.ts) / MS_PER_HOUR,
                        note=signal.note,
                    )
                )
                equity = equity_after
                next_allowed_idx = last_idx + 1
                break

            if signal.side == "long":
                high_water = max(high_water, candle.high)
                if candle.high >= entry + risk * config.breakeven_r and active_stop < entry:
                    active_stop = entry
                    stop_stage = "保本止损"
                trailing_stop = high_water - signal.atr_price * config.trail_mult
                if candle.high >= entry + signal.atr_price and trailing_stop > active_stop:
                    active_stop = trailing_stop
                    stop_stage = "跟踪止损"
            else:
                low_water = min(low_water, candle.low)
                if candle.low <= entry - risk * config.breakeven_r and active_stop > entry:
                    active_stop = entry
                    stop_stage = "保本止损"
                trailing_stop = low_water + signal.atr_price * config.trail_mult
                if candle.low <= entry - signal.atr_price and trailing_stop < active_stop:
                    active_stop = trailing_stop
                    stop_stage = "跟踪止损"
    return trades


def summarize(trades: list[Trade], initial_equity: float) -> dict[str, float | int]:
    equity = initial_equity
    peak = initial_equity
    wins: list[float] = []
    losses: list[float] = []
    max_dd = 0.0
    max_loss_streak = 0
    loss_streak = 0
    for trade in trades:
        equity = trade.equity_after
        peak = max(peak, trade.equity_before, trade.equity_after)
        max_dd = max(max_dd, (peak - equity) / peak * 100 if peak else 0.0)
        if trade.net_pnl > 0:
            wins.append(trade.net_pnl)
            loss_streak = 0
        else:
            losses.append(trade.net_pnl)
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
        "return_pct": (equity / initial_equity - 1) * 100 if initial_equity else 0.0,
        "final_equity": equity,
        "max_dd": max_dd,
        "pf": gross_profit / gross_loss if gross_loss else math.inf if gross_profit else 0.0,
        "avg": statistics.mean([t.net_pnl for t in trades]) if trades else 0.0,
        "avg_win": statistics.mean(wins) if wins else 0.0,
        "avg_loss": statistics.mean(losses) if losses else 0.0,
        "best": max((t.net_pnl for t in trades), default=0.0),
        "worst": min((t.net_pnl for t in trades), default=0.0),
        "avg_hold": statistics.mean([t.hold_hours for t in trades]) if trades else 0.0,
        "max_loss_streak": max_loss_streak,
        "liq": sum(1 for t in trades if t.exit_reason == "强平"),
        "longs": sum(1 for t in trades if t.side == "long"),
        "shorts": sum(1 for t in trades if t.side == "short"),
    }


def max_monthly_drawdown_proxy(trades: list[Trade], initial_equity: float) -> float:
    if not trades:
        return 0.0
    month_start_equity = initial_equity
    month_peak = initial_equity
    current_month = format_ts(trades[0].entry_ts)[:7]
    worst = 0.0
    for trade in trades:
        month = format_ts(trade.entry_ts)[:7]
        if month != current_month:
            current_month = month
            month_start_equity = trade.equity_before
            month_peak = trade.equity_before
        month_peak = max(month_peak, trade.equity_before, trade.equity_after, month_start_equity)
        if month_peak > 0:
            worst = max(worst, (month_peak - trade.equity_after) / month_peak * 100)
    return worst


def score_row(row: dict) -> float:
    full = row["full"]
    test = row["test"]
    walk = row["walk"]
    return (
        full["return_pct"]
        + test["return_pct"] * 2.0
        + min(full["pf"], 3.0) * 8
        + min(test["pf"], 3.0) * 8
        - full["max_dd"] * 1.1
        - test["max_dd"] * 1.3
        - full["max_loss_streak"] * 1.5
        - full["liq"] * 100
        + min(walk, 6) * 2
    )


def build_all_rows(candles_15m: list[Candle], start_ts: int, end_ts: int, initial_equity: float) -> list[dict]:
    timestamps_15m = [c.ts for c in candles_15m]
    candles_1h = resample_candles(candles_15m, MS_PER_HOUR)
    candles_4h = resample_candles(candles_15m, BAR_MS["4H"])
    train_start, train_end, test_start, test_end = split_periods(start_ts, end_ts)
    rows: list[dict] = []

    signal_sets: dict[str, list[Signal]] = {}
    for lookback in [48, 72, 96]:
        for ema_period in [150, 200]:
            for stop_mult in [2.4]:
                for adx_floor in [20, 24]:
                    family = f"1H唐奇安趋势_L{lookback}_EMA{ema_period}_ADX{adx_floor:g}_S{stop_mult:g}"
                    signal_sets[family] = build_donchian_trend_signals(
                        family,
                        candles_15m,
                        candles_1h,
                        timestamps_15m,
                        MS_PER_HOUR,
                        lookback,
                        ema_period,
                        stop_mult,
                        adx_floor,
                        0.003,
                        0.04,
                    )

    for lookback in [20, 40, 60]:
        for ema_period in [120, 180]:
            for stop_mult in [2.4]:
                for adx_floor in [18, 22]:
                    family = f"4H唐奇安趋势_L{lookback}_EMA{ema_period}_ADX{adx_floor:g}_S{stop_mult:g}"
                    signal_sets[family] = build_donchian_trend_signals(
                        family,
                        candles_15m,
                        candles_4h,
                        timestamps_15m,
                        BAR_MS["4H"],
                        lookback,
                        ema_period,
                        stop_mult,
                        adx_floor,
                        0.004,
                        0.06,
                    )

    for range_lookback in [48, 72]:
        for squeeze_rank in [0.2]:
            for stop_mult in [2.2]:
                family = f"1H波动压缩突破_R{range_lookback}_Q{squeeze_rank:g}_S{stop_mult:g}"
                signal_sets[family] = build_squeeze_breakout_signals(
                    family,
                    candles_15m,
                    candles_1h,
                    timestamps_15m,
                    MS_PER_HOUR,
                    range_lookback,
                    240,
                    squeeze_rank,
                    stop_mult,
                    150,
                )

    for stop_mult in [1.6]:
        for fast in [34]:
            for slow in [150, 200]:
                family = f"1H趋势回踩_EMA{fast}_{slow}_S{stop_mult:g}"
                signal_sets[family] = build_pullback_signals(
                    family,
                    candles_15m,
                    candles_1h,
                    timestamps_15m,
                    stop_mult,
                    fast,
                    slow,
                    14,
                )

    for family, signals in signal_sets.items():
        if len(signals) < 8:
            continue
        for leverage in [5, 8, 10]:
            for margin_pct in [0.10, 0.15, 0.20]:
                for trail_mult in [2.4, 3.0]:
                    for take_profit_r in [None, 3.0]:
                        for max_hold in [96.0, 168.0, 336.0]:
                            if take_profit_r is None and max_hold < 96.0:
                                continue
                            name = (
                                f"{family}_{leverage:g}x_{margin_pct*100:.0f}%_"
                                f"T{trail_mult:g}_{'无固定止盈' if take_profit_r is None else str(take_profit_r)+'R'}_"
                                f"{int(max_hold)}H"
                            )
                            config = Config(
                                family=family,
                                name=name,
                                leverage=leverage,
                                margin_pct=margin_pct,
                                stop_mult=0.0,
                                trail_mult=trail_mult,
                                max_hold_hours=max_hold,
                                take_profit_r=take_profit_r,
                                breakeven_r=1.0,
                                initial_equity=initial_equity,
                            )
                            full_trades = simulate(candles_15m, signals, config, start_ts, end_ts)
                            if len(full_trades) < 8:
                                continue
                            train_trades = simulate(candles_15m, signals, config, train_start, train_end)
                            test_trades = simulate(candles_15m, signals, config, test_start, test_end)
                            full = summarize(full_trades, initial_equity)
                            train = summarize(train_trades, initial_equity)
                            test = summarize(test_trades, initial_equity)
                            if test["trades"] == 0:
                                continue
                            monthly_dd = max_monthly_drawdown_proxy(full_trades, initial_equity)
                            rows.append(
                                {
                                    "config": config,
                                    "signals": signals,
                                    "full_trades": full_trades,
                                    "full": full,
                                    "train": train,
                                    "test": test,
                                    "walk": sum(
                                        1
                                        for part in [train, test]
                                        if part["return_pct"] > 0 and part["pf"] >= 1.05 and part["trades"] >= 3
                                    ),
                                    "monthly_dd": monthly_dd,
                                }
                            )
    rows.sort(key=score_row, reverse=True)
    return rows


def export_scan(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "策略",
                "杠杆",
                "保证金比例",
                "交易数",
                "多单",
                "空单",
                "胜率",
                "全年收益",
                "最终权益",
                "最大回撤",
                "月内最大回撤",
                "PF",
                "平均单笔",
                "平均盈利",
                "平均亏损",
                "最大单笔盈利",
                "最大单笔亏损",
                "最大连亏",
                "强平",
                "样本外交易",
                "样本外收益",
                "样本外回撤",
                "样本外PF",
                "平均持仓小时",
            ]
        )
        for row in rows:
            c = row["config"]
            f = row["full"]
            t = row["test"]
            writer.writerow(
                [
                    c.name,
                    c.leverage,
                    f"{c.margin_pct:.2f}",
                    f["trades"],
                    f["longs"],
                    f["shorts"],
                    f"{f['win_rate']:.2f}%",
                    f"{f['return_pct']:.2f}%",
                    f"{f['final_equity']:.2f}",
                    f"{f['max_dd']:.2f}%",
                    f"{row['monthly_dd']:.2f}%",
                    f"{f['pf']:.2f}",
                    f"{f['avg']:.2f}",
                    f"{f['avg_win']:.2f}",
                    f"{f['avg_loss']:.2f}",
                    f"{f['best']:.2f}",
                    f"{f['worst']:.2f}",
                    f["max_loss_streak"],
                    f["liq"],
                    t["trades"],
                    f"{t['return_pct']:.2f}%",
                    f"{t['max_dd']:.2f}%",
                    f"{t['pf']:.2f}",
                    f"{f['avg_hold']:.2f}",
                ]
            )


def export_trades(path: Path, trades: list[Trade]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "策略",
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
                "收益率",
                "入场前权益",
                "出场后权益",
                "出场原因",
                "持仓小时",
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
                    f"{trade.entry_price:.2f}",
                    f"{trade.exit_price:.2f}",
                    f"{trade.initial_stop:.2f}",
                    f"{trade.final_stop:.2f}",
                    "" if trade.target_price is None else f"{trade.target_price:.2f}",
                    f"{trade.liquidation_price:.2f}",
                    f"{trade.leverage:.0f}",
                    f"{trade.margin_pct * 100:.0f}%",
                    f"{trade.margin_used:.2f}",
                    f"{trade.notional:.2f}",
                    f"{trade.net_pnl:.2f}",
                    f"{trade.return_pct:.2f}%",
                    f"{trade.equity_before:.2f}",
                    f"{trade.equity_after:.2f}",
                    trade.exit_reason,
                    f"{trade.hold_hours:.2f}",
                    trade.note,
                ]
            )


def print_top(rows: list[dict], limit: int) -> None:
    print("BTC-USDT-SWAP 永续策略扫描结果")
    print("筛选优先级：全年为正、样本外为正、无强平、交易数足够、回撤不能靠爆仓换收益。")
    print()
    print("策略 | 交易 | 胜率 | 收益 | 回撤 | PF | 平均盈利 | 平均亏损 | 样本外收益 | 样本外交易")
    print("-" * 150)
    for row in rows[:limit]:
        f = row["full"]
        t = row["test"]
        print(
            f"{row['config'].name} | {f['trades']} | {f['win_rate']:.1f}% | {f['return_pct']:.1f}% | "
            f"{f['max_dd']:.1f}% | {f['pf']:.2f} | {f['avg_win']:.1f}U | {f['avg_loss']:.1f}U | "
            f"{t['return_pct']:.1f}% | {t['trades']}"
        )


def select_executable(rows: list[dict]) -> list[dict]:
    candidates = [
        row
        for row in rows
        if row["full"]["return_pct"] > 0
        and row["test"]["return_pct"] > 0
        and row["full"]["pf"] >= 1.15
        and row["test"]["pf"] >= 1.05
        and row["full"]["max_dd"] <= 35
        and row["full"]["liq"] == 0
        and row["full"]["trades"] >= 12
        and row["test"]["trades"] >= 4
        and row["full"]["max_loss_streak"] <= 6
    ]
    candidates.sort(key=score_row, reverse=True)
    return candidates


def select_recommended(candidates: list[dict]) -> dict | None:
    balanced = [
        row
        for row in candidates
        if 50 <= row["full"]["avg_win"] <= 100
        and row["full"]["max_dd"] <= 20
        and row["test"]["return_pct"] > 0
    ]
    if balanced:
        return balanced[0]
    return candidates[0] if candidates else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="扫描 BTC-USDT-SWAP 长短双向永续策略")
    parser.add_argument("--inst-id", default="BTC-USDT-SWAP")
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--export-dir", type=Path, default=Path("data"))
    parser.add_argument("--initial-equity", type=float, default=1000.0)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--top", type=int, default=15)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    args = build_parser().parse_args(argv)
    try:
        candles_15m = load_candles_csv(latest_cache(args.cache_dir, f"okx_{args.inst_id}_15m_*.csv"))
        end_ts = candles_15m[-1].ts + BAR_MS["15m"]
        start_ts = end_ts - args.days * 24 * MS_PER_HOUR
        rows = build_all_rows(candles_15m, start_ts, end_ts, args.initial_equity)
        export_scan(args.export_dir / "perp_strategy_scan.csv", rows)
        candidates = select_executable(rows)
        print(f"数据：{args.inst_id}，区间 {format_ts(start_ts)} -> {format_ts(end_ts)}")
        print(f"扫描组合：{len(rows)}，候选：{len(candidates)}")
        if candidates:
            print_top(candidates, args.top)
            best = candidates[0]
            recommended = select_recommended(candidates)
            export_trades(args.export_dir / "perp_strategy_best_trades.csv", best["full_trades"])
            if recommended is not None:
                export_trades(args.export_dir / "perp_strategy_recommended_trades.csv", recommended["full_trades"])
            print()
            print(f"最佳逐笔交易：{args.export_dir / 'perp_strategy_best_trades.csv'}")
            if recommended is not None:
                print(f"推荐实盘观察版逐笔交易：{args.export_dir / 'perp_strategy_recommended_trades.csv'}")
            print(f"完整扫描表：{args.export_dir / 'perp_strategy_scan.csv'}")
        else:
            print_top(rows, args.top)
            print()
            print("没有找到通过实盘候选门槛的策略；上面只展示原始评分最高组合，不能直接实盘。")
            print(f"完整扫描表：{args.export_dir / 'perp_strategy_scan.csv'}")
        return 0
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
