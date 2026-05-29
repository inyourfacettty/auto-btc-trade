#!/usr/bin/env python3
"""Research several BTC-USDT long-only strategies on cached OKX candles."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from backtest_okx_btc import (
    BAR_MS,
    Candle,
    compute_ema,
    compute_rsi,
    format_ts,
    load_candles_csv,
)


@dataclass(frozen=True)
class ResearchConfig:
    initial_equity: float = 1000.0
    max_leverage: float = 3.0
    risk_pct: float = 0.01
    fee_rate: float = 0.0005
    slippage_bps: float = 1.0


@dataclass(frozen=True)
class Signal:
    strategy: str
    entry_idx: int
    stop_price: float
    take_profit_r: float
    note: str
    side: str = "long"


@dataclass(frozen=True)
class ResearchTrade:
    strategy: str
    side: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    stop_price: float
    take_profit_price: float
    notional: float
    net_pnl: float
    fees: float
    return_pct: float
    r_multiple: float
    exit_reason: str
    equity_before: float
    equity_after: float
    note: str


def calculate_position_notional(
    equity: float,
    entry_price: float,
    stop_price: float,
    config: ResearchConfig,
    side: str = "long",
) -> float:
    if side == "long":
        risk_per_unit = entry_price - stop_price
    elif side == "short":
        risk_per_unit = stop_price - entry_price
    else:
        raise ValueError(f"Unsupported side: {side}")

    if equity <= 0 or entry_price <= 0 or risk_per_unit <= 0:
        return 0.0

    risk_budget = equity * config.risk_pct
    risk_sized_notional = risk_budget / (risk_per_unit / entry_price)
    max_notional = equity * config.max_leverage
    return min(max_notional, risk_sized_notional)


def split_periods(start_ts: int, end_ts: int) -> tuple[int, int, int, int]:
    train_end = start_ts + (end_ts - start_ts) * 2 // 3
    return start_ts, train_end, train_end, end_ts


def compute_atr(candles: list[Candle], period: int) -> list[float | None]:
    atr: list[float | None] = [None] * len(candles)
    if len(candles) <= period:
        return atr

    true_ranges = [0.0]
    for i in range(1, len(candles)):
        prev_close = candles[i - 1].close
        tr = max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - prev_close),
            abs(candles[i].low - prev_close),
        )
        true_ranges.append(tr)

    seed = sum(true_ranges[1 : period + 1]) / period
    atr[period] = seed
    for i in range(period + 1, len(candles)):
        prev = atr[i - 1]
        if prev is None:
            raise RuntimeError("ATR seed missing")
        atr[i] = (prev * (period - 1) + true_ranges[i]) / period
    return atr


def rolling_high(candles: list[Candle], window: int) -> list[float | None]:
    values: list[float | None] = [None] * len(candles)
    for i in range(window, len(candles)):
        values[i] = max(c.high for c in candles[i - window : i])
    return values


def rolling_low(candles: list[Candle], window: int) -> list[float | None]:
    values: list[float | None] = [None] * len(candles)
    for i in range(window, len(candles)):
        values[i] = min(c.low for c in candles[i - window : i])
    return values


def bollinger_lower(values: list[float], period: int, stdev_mult: float) -> list[float | None]:
    lower: list[float | None] = [None] * len(values)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        lower[i] = statistics.mean(window) - statistics.pstdev(window) * stdev_mult
    return lower


def latest_cache(cache_dir: Path, prefix: str) -> Path:
    matches = sorted(cache_dir.glob(prefix), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"没有找到缓存文件：{cache_dir / prefix}")
    return matches[0]


def build_h4_index(candles_15m: list[Candle], candles_4h: list[Candle]) -> list[int]:
    context: list[int] = []
    h4_idx = -1
    for candle in candles_15m:
        signal_close_ts = candle.ts + BAR_MS["15m"]
        while h4_idx + 1 < len(candles_4h) and candles_4h[h4_idx + 1].ts + BAR_MS["4H"] <= signal_close_ts:
            h4_idx += 1
        context.append(h4_idx)
    return context


def first_15m_at_or_after(candles_15m: list[Candle], ts: int) -> int | None:
    timestamps = [c.ts for c in candles_15m]
    idx = bisect.bisect_left(timestamps, ts)
    if idx >= len(candles_15m):
        return None
    return idx


def simulate_signals(
    candles_15m: list[Candle],
    signals: list[Signal],
    config: ResearchConfig,
    start_ts: int,
    end_ts: int,
) -> list[ResearchTrade]:
    trades: list[ResearchTrade] = []
    equity = config.initial_equity
    next_allowed_idx = 0
    slippage = config.slippage_bps / 10_000

    for signal in sorted(signals, key=lambda s: s.entry_idx):
        if signal.entry_idx < next_allowed_idx:
            continue
        if signal.entry_idx >= len(candles_15m):
            continue

        entry_candle = candles_15m[signal.entry_idx]
        if not (start_ts <= entry_candle.ts < end_ts):
            continue

        if signal.side == "long":
            entry_price = entry_candle.open * (1 + slippage)
        elif signal.side == "short":
            entry_price = entry_candle.open * (1 - slippage)
        else:
            continue

        if signal.stop_price <= 0:
            continue
        if signal.side == "long" and signal.stop_price >= entry_price:
            continue
        if signal.side == "short" and signal.stop_price <= entry_price:
            continue

        notional = calculate_position_notional(equity, entry_price, signal.stop_price, config, signal.side)
        if notional <= 0:
            continue

        qty = notional / entry_price
        if signal.side == "long":
            risk_per_unit = entry_price - signal.stop_price
            take_profit_price = entry_price + risk_per_unit * signal.take_profit_r
        else:
            risk_per_unit = signal.stop_price - entry_price
            take_profit_price = entry_price - risk_per_unit * signal.take_profit_r
        entry_fee = notional * config.fee_rate

        for idx in range(signal.entry_idx, len(candles_15m)):
            candle = candles_15m[idx]
            if candle.ts >= end_ts:
                exit_price = candle.open * (1 - slippage) if signal.side == "long" else candle.open * (1 + slippage)
                exit_reason = "区间结束"
            elif signal.side == "long" and candle.low <= signal.stop_price:
                exit_price = signal.stop_price * (1 - slippage)
                exit_reason = "止损"
            elif signal.side == "short" and candle.high >= signal.stop_price:
                exit_price = signal.stop_price * (1 + slippage)
                exit_reason = "止损"
            elif signal.side == "long" and candle.high >= take_profit_price:
                exit_price = take_profit_price * (1 - slippage)
                exit_reason = "止盈"
            elif signal.side == "short" and candle.low <= take_profit_price:
                exit_price = take_profit_price * (1 + slippage)
                exit_reason = "止盈"
            elif idx == len(candles_15m) - 1:
                exit_price = candle.close * (1 - slippage) if signal.side == "long" else candle.close * (1 + slippage)
                exit_reason = "数据结束"
            else:
                continue

            exit_notional = qty * exit_price
            exit_fee = exit_notional * config.fee_rate
            if signal.side == "long":
                gross_pnl = qty * (exit_price - entry_price)
                r_multiple = (exit_price - entry_price) / risk_per_unit
            else:
                gross_pnl = qty * (entry_price - exit_price)
                r_multiple = (entry_price - exit_price) / risk_per_unit
            fees = entry_fee + exit_fee
            net_pnl = gross_pnl - fees
            equity_after = equity + net_pnl
            return_pct = net_pnl / equity * 100 if equity else 0.0
            trades.append(
                ResearchTrade(
                    strategy=signal.strategy,
                    side=signal.side,
                    entry_ts=entry_candle.ts,
                    exit_ts=candle.ts,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_price=signal.stop_price,
                    take_profit_price=take_profit_price,
                    notional=notional,
                    net_pnl=net_pnl,
                    fees=fees,
                    return_pct=return_pct,
                    r_multiple=r_multiple,
                    exit_reason=exit_reason,
                    equity_before=equity,
                    equity_after=equity_after,
                    note=signal.note,
                )
            )
            equity = equity_after
            next_allowed_idx = idx + 1
            break

    return trades


def metrics_from_trades(trades: list[ResearchTrade], initial_equity: float) -> dict[str, float | int]:
    equity = initial_equity
    peak = initial_equity
    max_dd = 0.0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    total_fees = 0.0
    r_values = []

    for trade in trades:
        pnl = equity * (trade.return_pct / 100)
        equity += pnl
        peak = max(peak, equity)
        if peak:
            max_dd = max(max_dd, (peak - equity) / peak * 100)
        if pnl > 0:
            wins += 1
            gross_profit += pnl
        else:
            gross_loss -= pnl
        total_fees += equity * 0 + trade.fees
        r_values.append(trade.r_multiple)

    trade_count = len(trades)
    return {
        "trades": trade_count,
        "wins": wins,
        "losses": trade_count - wins,
        "win_rate_pct": wins / trade_count * 100 if trade_count else 0.0,
        "final_equity": equity,
        "return_pct": (equity / initial_equity - 1) * 100 if initial_equity else 0.0,
        "max_drawdown_pct": max_dd,
        "profit_factor": gross_profit / gross_loss if gross_loss else math.inf if gross_profit else 0.0,
        "avg_r": statistics.mean(r_values) if r_values else 0.0,
        "fees": total_fees,
    }


def baseline_reclaim_signals(
    candles_15m: list[Candle],
    candles_4h: list[Candle],
    h4_by_15m: list[int],
) -> list[Signal]:
    closes_15m = [c.close for c in candles_15m]
    closes_4h = [c.close for c in candles_4h]
    ema15 = compute_ema(closes_15m, 20)
    rsi15 = compute_rsi(closes_15m, 12)
    ema4_fast = compute_ema(closes_4h, 20)
    ema4_slow = compute_ema(closes_4h, 60)

    signals = []
    for i in range(1, len(candles_15m) - 1):
        h = h4_by_15m[i]
        if h < 0 or ema4_fast[h] is None or ema4_slow[h] is None:
            continue
        if not (ema4_fast[h] > ema4_slow[h] and candles_4h[h].close > ema4_slow[h]):
            continue
        if None in (ema15[i - 1], ema15[i], rsi15[i - 1], rsi15[i]):
            continue
        pulled = any(
            ema15[j] is not None and candles_15m[j].low <= ema15[j] * 1.001
            for j in range(max(0, i - 7), i + 1)
        )
        if not pulled:
            continue
        if not (candles_15m[i - 1].close <= ema15[i - 1] and candles_15m[i].close > ema15[i]):
            continue
        if not (40 <= rsi15[i - 1] <= 55 and rsi15[i] > rsi15[i - 1]):
            continue
        stop = min(c.low for c in candles_15m[max(0, i - 5) : i + 1])
        signals.append(Signal("S1 原始15m回踩_风险控仓", i + 1, stop, 2.0, "原始规则，风险控仓"))
    return signals


def strong_pullback_signals(
    candles_15m: list[Candle],
    candles_4h: list[Candle],
    h4_by_15m: list[int],
) -> list[Signal]:
    closes_15m = [c.close for c in candles_15m]
    closes_4h = [c.close for c in candles_4h]
    ema15 = compute_ema(closes_15m, 20)
    rsi15 = compute_rsi(closes_15m, 12)
    atr15 = compute_atr(candles_15m, 14)
    ema4_fast = compute_ema(closes_4h, 20)
    ema4_slow = compute_ema(closes_4h, 60)
    rsi4 = compute_rsi(closes_4h, 14)

    signals = []
    for i in range(1, len(candles_15m) - 1):
        h = h4_by_15m[i]
        if h < 66:
            continue
        required = (ema4_fast[h], ema4_slow[h], ema4_fast[h - 3], ema4_slow[h - 6], rsi4[h])
        if any(v is None for v in required):
            continue
        if not (
            ema4_fast[h] > ema4_slow[h]
            and candles_4h[h].close > ema4_fast[h]
            and ema4_fast[h] > ema4_fast[h - 3]
            and ema4_slow[h] > ema4_slow[h - 6]
            and rsi4[h] >= 54
        ):
            continue
        if None in (ema15[i - 1], ema15[i], rsi15[i - 1], rsi15[i], atr15[i]):
            continue
        if candles_15m[i].close > ema15[i] * 1.006:
            continue
        pulled = any(
            ema15[j] is not None and candles_15m[j].low <= ema15[j] * 1.0015
            for j in range(max(0, i - 9), i + 1)
        )
        if not pulled:
            continue
        if not (candles_15m[i - 1].close <= ema15[i - 1] and candles_15m[i].close > ema15[i]):
            continue
        if not (45 <= rsi15[i - 1] <= 58 and rsi15[i] > rsi15[i - 1]):
            continue
        swing_stop = min(c.low for c in candles_15m[max(0, i - 7) : i + 1])
        atr_stop = candles_15m[i].close - atr15[i] * 1.2
        stop = min(swing_stop, atr_stop)
        signals.append(Signal("S2 强趋势15m回踩", i + 1, stop, 2.5, "4H强趋势过滤"))
    return signals


def h4_pullback_signals(candles_15m: list[Candle], candles_4h: list[Candle]) -> list[Signal]:
    closes_4h = [c.close for c in candles_4h]
    ema20 = compute_ema(closes_4h, 20)
    ema60 = compute_ema(closes_4h, 60)
    rsi = compute_rsi(closes_4h, 14)
    atr = compute_atr(candles_4h, 14)

    signals = []
    for i in range(66, len(candles_4h)):
        required = (ema20[i], ema60[i], ema60[i - 6], rsi[i], atr[i])
        if any(v is None for v in required):
            continue
        if not (ema20[i] > ema60[i] and ema60[i] > ema60[i - 6] and 50 <= rsi[i] <= 66):
            continue
        if not (candles_4h[i].low <= ema20[i] * 1.004 and candles_4h[i].close > ema20[i]):
            continue
        if i > 0 and candles_4h[i - 1].close > ema20[i - 1] * 1.012:
            continue
        entry_idx = first_15m_at_or_after(candles_15m, candles_4h[i].ts + BAR_MS["4H"])
        if entry_idx is None:
            continue
        swing_stop = min(c.low for c in candles_4h[max(0, i - 3) : i + 1])
        atr_stop = candles_4h[i].close - atr[i] * 1.5
        stop = max(swing_stop, atr_stop)
        signals.append(Signal("S3 4H趋势回踩", entry_idx, stop, 3.0, "4H回踩EMA20后收回"))
    return signals


def h4_breakout_signals(candles_15m: list[Candle], candles_4h: list[Candle]) -> list[Signal]:
    closes_4h = [c.close for c in candles_4h]
    ema20 = compute_ema(closes_4h, 20)
    ema60 = compute_ema(closes_4h, 60)
    rsi = compute_rsi(closes_4h, 14)
    atr = compute_atr(candles_4h, 14)
    prev_high20 = rolling_high(candles_4h, 20)

    signals = []
    for i in range(66, len(candles_4h)):
        required = (ema20[i], ema60[i], rsi[i], atr[i], prev_high20[i])
        if any(v is None for v in required):
            continue
        atr_pct = atr[i] / candles_4h[i].close
        if not (
            ema20[i] > ema60[i]
            and candles_4h[i].close > prev_high20[i]
            and candles_4h[i].close > ema20[i]
            and rsi[i] >= 55
            and 0.004 <= atr_pct <= 0.04
        ):
            continue
        entry_idx = first_15m_at_or_after(candles_15m, candles_4h[i].ts + BAR_MS["4H"])
        if entry_idx is None:
            continue
        stop = candles_4h[i].close - atr[i] * 2.0
        signals.append(Signal("S4 4H唐奇安突破", entry_idx, stop, 3.5, "4H突破20根高点"))
    return signals


def h4_breakout_long_short_signals(candles_15m: list[Candle], candles_4h: list[Candle]) -> list[Signal]:
    closes_4h = [c.close for c in candles_4h]
    ema20 = compute_ema(closes_4h, 20)
    ema60 = compute_ema(closes_4h, 60)
    rsi = compute_rsi(closes_4h, 14)
    atr = compute_atr(candles_4h, 14)
    prev_high30 = rolling_high(candles_4h, 30)
    prev_low30 = rolling_low(candles_4h, 30)

    signals = []
    for i in range(80, len(candles_4h)):
        required = (ema20[i], ema60[i], ema60[i - 8], rsi[i], atr[i], prev_high30[i], prev_low30[i])
        if any(v is None for v in required):
            continue
        atr_pct = atr[i] / candles_4h[i].close
        if not (0.006 <= atr_pct <= 0.05):
            continue
        entry_idx = first_15m_at_or_after(candles_15m, candles_4h[i].ts + BAR_MS["4H"])
        if entry_idx is None:
            continue

        if (
            ema20[i] > ema60[i]
            and ema60[i] > ema60[i - 8]
            and candles_4h[i].close > prev_high30[i]
            and rsi[i] >= 56
        ):
            stop = candles_4h[i].close - atr[i] * 2.2
            signals.append(Signal("S6 4H突破多空", entry_idx, stop, 3.0, "4H向上突破30根高点", "long"))
        elif (
            ema20[i] < ema60[i]
            and ema60[i] < ema60[i - 8]
            and candles_4h[i].close < prev_low30[i]
            and rsi[i] <= 44
        ):
            stop = candles_4h[i].close + atr[i] * 2.2
            signals.append(Signal("S6 4H突破多空", entry_idx, stop, 3.0, "4H向下跌破30根低点", "short"))
    return signals


def daily_regime_h4_pullback_signals(candles_15m: list[Candle], candles_4h: list[Candle]) -> list[Signal]:
    closes_4h = [c.close for c in candles_4h]
    ema20 = compute_ema(closes_4h, 20)
    ema60 = compute_ema(closes_4h, 60)
    ema120 = compute_ema(closes_4h, 120)
    rsi = compute_rsi(closes_4h, 14)
    atr = compute_atr(candles_4h, 14)

    signals = []
    for i in range(132, len(candles_4h)):
        required = (ema20[i], ema60[i], ema120[i], ema60[i - 12], rsi[i], atr[i])
        if any(v is None for v in required):
            continue
        entry_idx = first_15m_at_or_after(candles_15m, candles_4h[i].ts + BAR_MS["4H"])
        if entry_idx is None:
            continue
        atr_pct = atr[i] / candles_4h[i].close
        if not (0.004 <= atr_pct <= 0.045):
            continue

        if (
            candles_4h[i].close > ema120[i]
            and ema20[i] > ema60[i] > ema120[i]
            and ema60[i] > ema60[i - 12]
            and 48 <= rsi[i] <= 64
            and candles_4h[i].low <= ema20[i] * 1.006
            and candles_4h[i].close > ema20[i]
        ):
            stop = min(c.low for c in candles_4h[max(0, i - 4) : i + 1])
            signals.append(Signal("S7 4H顺势回踩多空", entry_idx, stop, 2.4, "多头排列回踩EMA20", "long"))
        elif (
            candles_4h[i].close < ema120[i]
            and ema20[i] < ema60[i] < ema120[i]
            and ema60[i] < ema60[i - 12]
            and 36 <= rsi[i] <= 52
            and candles_4h[i].high >= ema20[i] * 0.994
            and candles_4h[i].close < ema20[i]
        ):
            stop = max(c.high for c in candles_4h[max(0, i - 4) : i + 1])
            signals.append(Signal("S7 4H顺势回踩多空", entry_idx, stop, 2.4, "空头排列反弹EMA20", "short"))
    return signals


def selected_h4_breakout_signals(candles_15m: list[Candle], candles_4h: list[Candle]) -> list[Signal]:
    closes_4h = [c.close for c in candles_4h]
    ema20 = compute_ema(closes_4h, 20)
    ema60 = compute_ema(closes_4h, 60)
    ema120 = compute_ema(closes_4h, 120)
    rsi = compute_rsi(closes_4h, 14)
    atr = compute_atr(candles_4h, 14)
    prev_high40 = rolling_high(candles_4h, 40)
    prev_low40 = rolling_low(candles_4h, 40)

    signals = []
    for i in range(132, len(candles_4h)):
        required = (ema20[i], ema60[i], ema120[i], rsi[i], atr[i], prev_high40[i], prev_low40[i])
        if any(v is None for v in required):
            continue
        atr_pct = atr[i] / candles_4h[i].close
        if not (0.006 <= atr_pct <= 0.05):
            continue
        entry_idx = first_15m_at_or_after(candles_15m, candles_4h[i].ts + BAR_MS["4H"])
        if entry_idx is None:
            continue

        if (
            candles_4h[i].close > ema120[i]
            and ema20[i] > ema60[i]
            and candles_4h[i].close > prev_high40[i]
            and rsi[i] >= 52
        ):
            stop = candles_4h[i].close - atr[i] * 3.0
            signals.append(Signal("S8 可执行_4H多空突破", entry_idx, stop, 2.0, "EMA120上方突破40根高点", "long"))
        elif (
            candles_4h[i].close < ema120[i]
            and ema20[i] < ema60[i]
            and candles_4h[i].close < prev_low40[i]
            and rsi[i] <= 48
        ):
            stop = candles_4h[i].close + atr[i] * 3.0
            signals.append(Signal("S8 可执行_4H多空突破", entry_idx, stop, 2.0, "EMA120下方跌破40根低点", "short"))
    return signals


def mean_reversion_signals(
    candles_15m: list[Candle],
    candles_4h: list[Candle],
    h4_by_15m: list[int],
) -> list[Signal]:
    closes_15m = [c.close for c in candles_15m]
    closes_4h = [c.close for c in candles_4h]
    lower = bollinger_lower(closes_15m, 20, 2.0)
    rsi15 = compute_rsi(closes_15m, 12)
    atr15 = compute_atr(candles_15m, 14)
    ema4_fast = compute_ema(closes_4h, 20)
    ema4_slow = compute_ema(closes_4h, 60)

    signals = []
    for i in range(1, len(candles_15m) - 1):
        h = h4_by_15m[i]
        if h < 0 or ema4_fast[h] is None or ema4_slow[h] is None:
            continue
        if not (ema4_fast[h] > ema4_slow[h] and candles_4h[h].close > ema4_slow[h]):
            continue
        if None in (lower[i - 1], lower[i], rsi15[i - 1], rsi15[i], atr15[i]):
            continue
        if not (candles_15m[i - 1].close < lower[i - 1] and candles_15m[i].close > lower[i]):
            continue
        if not (rsi15[i - 1] <= 38 and rsi15[i] > rsi15[i - 1]):
            continue
        swing_stop = min(c.low for c in candles_15m[max(0, i - 7) : i + 1])
        atr_stop = candles_15m[i].close - atr15[i] * 1.0
        stop = min(swing_stop, atr_stop)
        signals.append(Signal("S5 布林下轨反弹", i + 1, stop, 1.5, "4H多头内短线超跌收回"))
    return signals


def run_research(candles_15m: list[Candle], candles_4h: list[Candle], config: ResearchConfig):
    start_ts = max(candles_15m[0].ts + 2 * 24 * 60 * 60 * 1000, candles_4h[0].ts + 20 * 24 * 60 * 60 * 1000)
    end_ts = candles_15m[-1].ts + BAR_MS["15m"]
    train_start, train_end, test_start, test_end = split_periods(start_ts, end_ts)

    h4_by_15m = build_h4_index(candles_15m, candles_4h)
    strategy_signals = [
        baseline_reclaim_signals(candles_15m, candles_4h, h4_by_15m),
        strong_pullback_signals(candles_15m, candles_4h, h4_by_15m),
        h4_pullback_signals(candles_15m, candles_4h),
        h4_breakout_signals(candles_15m, candles_4h),
        h4_breakout_long_short_signals(candles_15m, candles_4h),
        daily_regime_h4_pullback_signals(candles_15m, candles_4h),
        selected_h4_breakout_signals(candles_15m, candles_4h),
        mean_reversion_signals(candles_15m, candles_4h, h4_by_15m),
    ]

    rows = []
    all_trades: list[ResearchTrade] = []
    for signals in strategy_signals:
        if not signals:
            continue
        name = signals[0].strategy
        trades = simulate_signals(candles_15m, signals, config, start_ts, end_ts)
        all_trades.extend(trades)
        train_trades = [t for t in trades if train_start <= t.entry_ts < train_end]
        test_trades = [t for t in trades if test_start <= t.entry_ts < test_end]
        full = metrics_from_trades(trades, config.initial_equity)
        train = metrics_from_trades(train_trades, config.initial_equity)
        test = metrics_from_trades(test_trades, config.initial_equity)
        rows.append(
            {
                "strategy": name,
                "signals": len(signals),
                "full": full,
                "train": train,
                "test": test,
            }
        )

    return rows, all_trades, (start_ts, end_ts, train_start, train_end, test_start, test_end)


def export_research_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "策略",
                "信号数",
                "全年交易",
                "全年收益率",
                "全年最大回撤",
                "全年胜率",
                "全年盈亏比因子",
                "训练收益率",
                "测试收益率",
                "测试最大回撤",
                "手续费",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["strategy"],
                    row["signals"],
                    row["full"]["trades"],
                    f"{row['full']['return_pct']:.2f}",
                    f"{row['full']['max_drawdown_pct']:.2f}",
                    f"{row['full']['win_rate_pct']:.2f}",
                    f"{row['full']['profit_factor']:.2f}",
                    f"{row['train']['return_pct']:.2f}",
                    f"{row['test']['return_pct']:.2f}",
                    f"{row['test']['max_drawdown_pct']:.2f}",
                    f"{row['full']['fees']:.2f}",
                ]
            )


def print_research(rows: list[dict], periods: tuple[int, int, int, int, int, int]) -> None:
    start_ts, end_ts, train_start, train_end, test_start, test_end = periods
    print("多策略研究回测")
    print(f"总区间：{format_ts(start_ts)} -> {format_ts(end_ts)}")
    print(f"训练段：{format_ts(train_start)} -> {format_ts(train_end)}")
    print(f"测试段：{format_ts(test_start)} -> {format_ts(test_end)}")
    print()
    print("策略 | 交易 | 全年收益 | 最大回撤 | 胜率 | 盈亏比因子 | 训练收益 | 测试收益")
    print("-" * 92)
    for row in sorted(rows, key=lambda r: r["full"]["return_pct"], reverse=True):
        full = row["full"]
        train = row["train"]
        test = row["test"]
        print(
            f"{row['strategy']} | {full['trades']} | {full['return_pct']:.2f}% | "
            f"{full['max_drawdown_pct']:.2f}% | {full['win_rate_pct']:.2f}% | "
            f"{full['profit_factor']:.2f} | {train['return_pct']:.2f}% | {test['return_pct']:.2f}%"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="研究多个 BTC-USDT 量化策略。")
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--initial-equity", type=float, default=1000.0)
    parser.add_argument("--max-leverage", type=float, default=3.0)
    parser.add_argument("--risk-pct", type=float, default=0.01)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--export", type=Path, default=Path("data/strategy_research_summary.csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    args = build_parser().parse_args(argv)
    try:
        candles_15m = load_candles_csv(latest_cache(args.cache_dir, "okx_BTC-USDT_15m_*.csv"))
        candles_4h = load_candles_csv(latest_cache(args.cache_dir, "okx_BTC-USDT_4H_*.csv"))
        config = ResearchConfig(
            initial_equity=args.initial_equity,
            max_leverage=args.max_leverage,
            risk_pct=args.risk_pct,
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage_bps,
        )
        rows, _, periods = run_research(candles_15m, candles_4h, config)
        export_research_csv(args.export, rows)
        print_research(rows, periods)
        print()
        print(f"研究汇总CSV：{args.export}")
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
