#!/usr/bin/env python3
"""Research GitHub-inspired strategies on OKX BTC-USDT-SWAP candles."""

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
from strategy_research import compute_atr, latest_cache, split_periods


_TIMESTAMP_CACHE: dict[int, list[int]] = {}


@dataclass(frozen=True)
class Signal:
    family: str
    side: str
    entry_idx: int
    note: str


@dataclass(frozen=True)
class ExitProfile:
    name: str
    hard_stop_pct: float
    target_pct: float | None
    min_profit_for_signal_exit: float = 0.0
    use_signal_exit: bool = True
    use_custom_trailing: bool = False
    max_hold_hours: float = 720.0


@dataclass(frozen=True)
class RunConfig:
    family: str
    leverage: float
    margin_pct: float
    exit_profile: ExitProfile
    initial_equity: float = 1000.0
    fee_rate: float = 0.0005
    slippage_bps: float = 1.0


@dataclass(frozen=True)
class Trade:
    family: str
    side: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float | None
    liquidation_price: float
    leverage: float
    margin_pct: float
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
    cache_key = id(candles_15m)
    timestamps = _TIMESTAMP_CACHE.get(cache_key)
    if timestamps is None:
        timestamps = [c.ts for c in candles_15m]
        _TIMESTAMP_CACHE[cache_key] = timestamps
    idx = bisect.bisect_left(timestamps, ts)
    return idx if idx < len(candles_15m) else None


def resample_candles(candles: list[Candle], timeframe: str) -> tuple[list[Candle], int]:
    if timeframe == "15m":
        return candles, BAR_MS["15m"]
    ratio_by_timeframe = {"30m": 2, "1h": 4}
    if timeframe not in ratio_by_timeframe:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    ratio = ratio_by_timeframe[timeframe]
    result: list[Candle] = []
    for i in range(0, len(candles) - ratio + 1, ratio):
        chunk = candles[i : i + ratio]
        result.append(
            Candle(
                ts=chunk[0].ts,
                open=chunk[0].open,
                high=max(c.high for c in chunk),
                low=min(c.low for c in chunk),
                close=chunk[-1].close,
                volume=sum(c.volume for c in chunk),
            )
        )
    return result, BAR_MS["15m"] * ratio


def simple_moving_average(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if period <= 0:
        return result
    rolling = 0.0
    for i, value in enumerate(values):
        rolling += value
        if i >= period:
            rolling -= values[i - period]
        if i >= period - 1:
            result[i] = rolling / period
    return result


def rolling_std(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    for i in range(period - 1, len(values)):
        result[i] = statistics.pstdev(values[i - period + 1 : i + 1])
    return result


def wma(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    weights = list(range(1, period + 1))
    weight_sum = sum(weights)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        result[i] = sum(v * w for v, w in zip(window, weights)) / weight_sum
    return result


def hma(values: list[float], period: int) -> list[float | None]:
    half = max(1, period // 2)
    sqrt_period = max(1, int(math.sqrt(period)))
    wma_half = wma(values, half)
    wma_full = wma(values, period)
    raw: list[float] = []
    for a, b in zip(wma_half, wma_full):
        raw.append(0.0 if a is None or b is None else 2 * a - b)
    raw_hma = wma(raw, sqrt_period)
    return [value if i >= period - 1 else None for i, value in enumerate(raw_hma)]


def bollinger(candles: list[Candle], period: int, stdev_mult: float) -> tuple[list[float | None], list[float | None], list[float | None]]:
    typical = [(c.high + c.low + c.close) / 3 for c in candles]
    sma = simple_moving_average(typical, period)
    std = rolling_std(typical, period)
    lower: list[float | None] = [None] * len(candles)
    upper: list[float | None] = [None] * len(candles)
    for i in range(len(candles)):
        if sma[i] is None or std[i] is None:
            continue
        lower[i] = sma[i] - std[i] * stdev_mult
        upper[i] = sma[i] + std[i] * stdev_mult
    return lower, sma, upper


def mean_deviation(values: list[float]) -> float:
    mean = statistics.mean(values)
    return statistics.mean(abs(value - mean) for value in values)


def compute_cci(candles: list[Candle], period: int = 20) -> list[float | None]:
    typical = [(c.high + c.low + c.close) / 3 for c in candles]
    result: list[float | None] = [None] * len(candles)
    for i in range(period - 1, len(candles)):
        window = typical[i - period + 1 : i + 1]
        dev = mean_deviation(window)
        if dev == 0:
            result[i] = 0.0
        else:
            result[i] = (typical[i] - statistics.mean(window)) / (0.015 * dev)
    return result


def compute_cmf(candles: list[Candle], period: int = 20) -> list[float | None]:
    result: list[float | None] = [None] * len(candles)
    mfv: list[float] = []
    volumes: list[float] = []
    for candle in candles:
        span = candle.high - candle.low
        multiplier = 0.0 if span == 0 else ((candle.close - candle.low) - (candle.high - candle.close)) / span
        mfv.append(multiplier * candle.volume)
        volumes.append(candle.volume)
    for i in range(period - 1, len(candles)):
        volume_sum = sum(volumes[i - period + 1 : i + 1])
        result[i] = 0.0 if volume_sum == 0 else sum(mfv[i - period + 1 : i + 1]) / volume_sum
    return result


def compute_mfi(candles: list[Candle], period: int = 14) -> list[float | None]:
    result: list[float | None] = [None] * len(candles)
    typical = [(c.high + c.low + c.close) / 3 for c in candles]
    positive = [0.0] * len(candles)
    negative = [0.0] * len(candles)
    for i in range(1, len(candles)):
        flow = typical[i] * candles[i].volume
        if typical[i] > typical[i - 1]:
            positive[i] = flow
        elif typical[i] < typical[i - 1]:
            negative[i] = flow
    for i in range(period, len(candles)):
        pos = sum(positive[i - period + 1 : i + 1])
        neg = sum(negative[i - period + 1 : i + 1])
        result[i] = 100.0 if neg == 0 and pos > 0 else 50.0 if neg == 0 else 100 - 100 / (1 + pos / neg)
    return result


def crossed_above(a: list[float | None], b: list[float | None], i: int) -> bool:
    return i > 0 and a[i - 1] is not None and b[i - 1] is not None and a[i] is not None and b[i] is not None and a[i - 1] <= b[i - 1] and a[i] > b[i]


def compute_ewo(candles: list[Candle], fast: int = 50, slow: int = 200) -> list[float | None]:
    closes = [c.close for c in candles]
    lows = [c.low for c in candles]
    ema_fast = compute_ema(closes, fast)
    ema_slow = compute_ema(closes, slow)
    result: list[float | None] = [None] * len(candles)
    for i in range(len(candles)):
        if ema_fast[i] is not None and ema_slow[i] is not None and lows[i] != 0:
            result[i] = (ema_fast[i] - ema_slow[i]) / lows[i] * 100
    return result


def williams_r(candles: list[Candle], period: int = 14) -> list[float | None]:
    result: list[float | None] = [None] * len(candles)
    for i in range(period - 1, len(candles)):
        highest = max(c.high for c in candles[i - period + 1 : i + 1])
        lowest = min(c.low for c in candles[i - period + 1 : i + 1])
        result[i] = -50.0 if highest == lowest else -100 * (highest - candles[i].close) / (highest - lowest)
    return result


def fisher_from_rsi(rsi: list[float | None]) -> list[float | None]:
    result: list[float | None] = [None] * len(rsi)
    for i, value in enumerate(rsi):
        if value is None:
            continue
        scaled = 0.1 * (value - 50)
        exp_value = math.exp(2 * scaled)
        result[i] = (exp_value - 1) / (exp_value + 1)
    return result


def map_entry(candles_15m: list[Candle], candle: Candle, interval_ms: int) -> int | None:
    return first_15m_at_or_after(candles_15m, candle.ts + interval_ms)


def build_smart_money(candles_15m: list[Candle]) -> tuple[list[Signal], dict[tuple[str, str], set[int]]]:
    candles, interval_ms = resample_candles(candles_15m, "30m")
    closes = [c.close for c in candles]
    ema200 = compute_ema(closes, 200)
    mfi = compute_mfi(candles, 14)
    cmf = compute_cmf(candles, 20)
    entries: list[Signal] = []
    exits: set[int] = set()
    for i, candle in enumerate(candles):
        if ema200[i] is None or mfi[i] is None or cmf[i] is None:
            continue
        idx = map_entry(candles_15m, candle, interval_ms)
        if idx is None:
            continue
        if candle.close < ema200[i] and mfi[i] < 35 and cmf[i] < -0.07:
            entries.append(Signal("SmartMoney_30m", "long", idx, "close<EMA200 + MFI<35 + CMF<-0.07"))
        if candle.close > ema200[i] and mfi[i] > 70 and cmf[i] > 0.20:
            exits.add(idx)
    return entries, {("SmartMoney_30m", "long"): exits}


def build_fisher_hull(candles_15m: list[Candle]) -> tuple[list[Signal], dict[tuple[str, str], set[int]]]:
    candles, interval_ms = resample_candles(candles_15m, "15m")
    closes = [c.close for c in candles]
    hma14 = hma(closes, 14)
    cci14 = compute_cci(candles, 14)
    fisher = fisher_from_rsi(compute_rsi(closes, 14))
    entries: list[Signal] = []
    exits: set[int] = set()
    for i, candle in enumerate(candles):
        if i == 0 or hma14[i] is None or hma14[i - 1] is None or cci14[i] is None or fisher[i] is None:
            continue
        idx = map_entry(candles_15m, candle, interval_ms)
        if idx is None:
            continue
        if hma14[i] < hma14[i - 1] and cci14[i] <= -50 and fisher[i] < -0.5:
            entries.append(Signal("FisherHull_15m", "long", idx, "HMA下行 + CCI<=-50 + FisherRSI<-0.5"))
        if hma14[i] > hma14[i - 1] and cci14[i] >= 100 and fisher[i] > 0.5:
            exits.add(idx)
    return entries, {("FisherHull_15m", "long"): exits}


def build_cci_bb(candles_15m: list[Candle]) -> tuple[list[Signal], dict[tuple[str, str], set[int]]]:
    candles, interval_ms = resample_candles(candles_15m, "15m")
    cci20 = compute_cci(candles, 20)
    lower, mid, upper = bollinger(candles, 20, 2.0)
    entries: list[Signal] = []
    exits: set[int] = set()
    for i, candle in enumerate(candles):
        if cci20[i] is None or lower[i] is None or mid[i] is None or upper[i] is None:
            continue
        idx = map_entry(candles_15m, candle, interval_ms)
        if idx is None:
            continue
        if cci20[i] <= -134 and candle.close < lower[i]:
            entries.append(Signal("CCI_BB_15m", "long", idx, "CCI<=-134 + close<BB下轨"))
        if candle.close > mid[i] or cci20[i] >= 100:
            exits.add(idx)
    return entries, {("CCI_BB_15m", "long"): exits}


def build_nasos_ewo(candles_15m: list[Candle]) -> tuple[list[Signal], dict[tuple[str, str], set[int]]]:
    candles, interval_ms = resample_candles(candles_15m, "15m")
    closes = [c.close for c in candles]
    ema8 = compute_ema(closes, 8)
    ema12 = compute_ema(closes, 12)
    ema13 = compute_ema(closes, 13)
    ema16 = compute_ema(closes, 16)
    ema26 = compute_ema(closes, 26)
    ema100 = compute_ema(closes, 100)
    sma9 = simple_moving_average(closes, 9)
    sma15 = simple_moving_average(closes, 15)
    ma_sell16 = compute_ema(closes, 16)
    hma50 = hma(closes, 50)
    lower2, _mid2, upper2 = bollinger(candles, 20, 2.0)
    lower3, _mid3, _upper3 = bollinger(candles, 20, 3.0)
    cci25 = compute_cci(candles, 25)
    rsi = compute_rsi(closes, 14)
    rsi_fast = compute_rsi(closes, 4)
    rsi_slow = compute_rsi(closes, 20)
    ewo = compute_ewo(candles, 50, 200)
    wr14 = williams_r(candles, 14)
    volume_mean4 = simple_moving_average([c.volume for c in candles], 4)
    entries: list[Signal] = []
    exits: set[int] = set()
    for i, candle in enumerate(candles):
        required = (
            ema8[i],
            ema12[i],
            ema13[i],
            ema16[i],
            ema26[i],
            ema100[i],
            sma9[i],
            sma15[i],
            ma_sell16[i],
            hma50[i],
            lower2[i],
            lower3[i],
            upper2[i],
            cci25[i],
            rsi[i],
            rsi_fast[i],
            rsi_slow[i],
            ewo[i],
            wr14[i],
            volume_mean4[i],
        )
        if any(value is None for value in required):
            continue
        idx = map_entry(candles_15m, candle, interval_ms)
        if idx is None:
            continue
        closedelta = abs(candle.close - candles[i - 1].close) if i > 0 else 0.0
        is_local_uptrend = (
            ema26[i] > ema12[i]
            and (ema26[i] - ema12[i]) > candle.open * 0.022
            and i > 0
            and (ema26[i - 1] or 0) - (ema12[i - 1] or 0) > candle.open / 100
            and candle.close < lower2[i] * 0.995
            and closedelta > candle.close * 15.0 / 1000
        )
        is_ewo = rsi_fast[i] < 37 and candle.close < ema8[i] * 0.981 and ewo[i] > -14.378 and candle.close < ema16[i] * 1.097 and rsi[i] < 78
        is_ewo2 = rsi_fast[i] < 37 and candle.close < ema8[i] * 0.942 and ewo[i] > 3.553 and candle.close < ema16[i] * 1.472 and rsi[i] < 78
        is_nfi32 = rsi_slow[i] < (rsi_slow[i - 1] or 999) and rsi_fast[i] < 46 and rsi[i] > 19 and candle.close < sma15[i] * 0.942
        is_nfi33 = candle.close < ema13[i] * 0.978 and ewo[i] > 8 and rsi[i] < 32 and wr14[i] < -98 and candle.volume < volume_mean4[i] * 2.5
        is_bb = cci25[i] <= -116 and candle.close < lower3[i] * 0.999
        if is_local_uptrend or is_ewo or is_ewo2 or is_nfi32 or is_nfi33 or is_bb:
            entries.append(Signal("NASOS_EWO_15m", "long", idx, "EWO/NFI/BB_RPB 多条件低位信号"))
        exit_signal = (
            candle.close > sma9[i]
            and candle.close > ma_sell16[i] * 0.997
            and rsi[i] > 50
            and rsi_fast[i] > rsi_slow[i]
        ) or (
            i > 0
            and sma9[i] > (sma9[i - 1] or sma9[i]) * 1.005
            and candle.close < hma50[i]
            and candle.close > ma_sell16[i] * 0.991
            and rsi_fast[i] > rsi_slow[i]
        )
        if exit_signal:
            exits.add(idx)
    return entries, {("NASOS_EWO_15m", "long"): exits}


def build_apollo11(candles_15m: list[Candle]) -> tuple[list[Signal], dict[tuple[str, str], set[int]]]:
    candles, interval_ms = resample_candles(candles_15m, "15m")
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]
    ema3 = compute_ema(closes, 3)
    ema5 = compute_ema(closes, 5)
    ema10 = compute_ema(closes, 10)
    ema20 = compute_ema(closes, 20)
    ema50 = compute_ema(closes, 50)
    ema200 = compute_ema(closes, 200)
    sma49 = simple_moving_average(closes, 49)
    std64 = rolling_std(closes, 64)
    atr14 = compute_atr(candles, 14)
    sma50 = simple_moving_average(closes, 50)
    lower3, _mid3, _upper3 = bollinger(candles, 20, 3.0)
    fast_vwma = [None if v is None else v for v in compute_ema([volumes[i] * closes[i] for i in range(len(candles))], 10)]
    slow_vwma = [None if v is None else v for v in compute_ema([volumes[i] * closes[i] for i in range(len(candles))], 20)]
    fast_vol = compute_ema(volumes, 10)
    slow_vol = compute_ema(volumes, 20)
    s3_fast = [None if fast_vwma[i] is None or fast_vol[i] in (None, 0) else fast_vwma[i] / fast_vol[i] for i in range(len(candles))]
    s3_slow = [None if slow_vwma[i] is None or slow_vol[i] in (None, 0) else slow_vwma[i] / slow_vol[i] for i in range(len(candles))]
    vw_fast_num = compute_ema([volumes[i] * closes[i] for i in range(len(candles))], 12)
    vw_slow_num = compute_ema([volumes[i] * closes[i] for i in range(len(candles))], 26)
    vw_fast_den = compute_ema(volumes, 12)
    vw_slow_den = compute_ema(volumes, 26)
    vwmacd = [
        None
        if vw_fast_num[i] is None or vw_slow_num[i] is None or vw_fast_den[i] in (None, 0) or vw_slow_den[i] in (None, 0)
        else vw_fast_num[i] / vw_fast_den[i] - vw_slow_num[i] / vw_slow_den[i]
        for i in range(len(candles))
    ]
    vwmacd_for_ema = [0.0 if value is None else value for value in vwmacd]
    signal = compute_ema(vwmacd_for_ema, 9)
    entries: list[Signal] = []
    for i, candle in enumerate(candles):
        required = (
            ema3[i],
            ema5[i],
            ema10[i],
            ema20[i],
            ema50[i],
            ema200[i],
            sma49[i],
            std64[i],
            atr14[i],
            sma50[i],
            lower3[i],
            s3_slow[i],
            signal[i],
            vwmacd[i],
        )
        if any(value is None for value in required):
            continue
        idx = map_entry(candles_15m, candle, interval_ms)
        if idx is None:
            continue
        bb_lower = sma49[i] - std64[i] * 3
        fib_lower = sma50[i] - atr14[i] * 4.236
        signal1 = (
            vwmacd[i] < signal[i]
            and candle.low < ema200[i]
            and candle.close > ema200[i]
            and crossed_above(ema5, ema10, i)
            and ema3[i] < ema50[i]
        )
        signal2 = i > 0 and fib_lower > bb_lower and (sma50[i - 1] or sma50[i]) - (atr14[i - 1] or atr14[i]) * 4.236 <= (sma49[i - 1] or sma49[i]) - (std64[i - 1] or std64[i]) * 3 and candle.close < ema50[i] * 2
        signal3 = candle.low < lower3[i] and candle.high > s3_slow[i] and candle.high < ema50[i]
        if signal1 or signal2 or signal3:
            tag = "Apollo signal1" if signal1 else "Apollo signal2" if signal2 else "Apollo signal3"
            entries.append(Signal("Apollo11_15m", "long", idx, tag))
    return entries, {("Apollo11_15m", "long"): set()}


def build_all_signals(candles_15m: list[Candle]) -> tuple[list[Signal], dict[tuple[str, str], set[int]]]:
    all_signals: list[Signal] = []
    all_exits: dict[tuple[str, str], set[int]] = {}
    for builder in (build_smart_money, build_fisher_hull, build_cci_bb, build_nasos_ewo, build_apollo11):
        signals, exits = builder(candles_15m)
        all_signals.extend(signals)
        for key, values in exits.items():
            all_exits.setdefault(key, set()).update(values)
    return all_signals, all_exits


def liquidation_price(entry_price: float, leverage: float, side: str) -> float:
    gap = 0.90 / leverage
    return entry_price * (1 - gap) if side == "long" else entry_price * (1 + gap)


def initial_stop(entry: float, profile: ExitProfile, side: str) -> float:
    return entry * (1 - profile.hard_stop_pct) if side == "long" else entry * (1 + profile.hard_stop_pct)


def target_price(entry: float, profile: ExitProfile, side: str) -> float | None:
    if profile.target_pct is None:
        return None
    return entry * (1 + profile.target_pct) if side == "long" else entry * (1 - profile.target_pct)


def update_custom_stop(entry: float, close: float, active_stop: float, side: str) -> float:
    profit = (close - entry) / entry if side == "long" else (entry - close) / entry
    if profit > 0.20:
        lock = 0.04
    elif profit > 0.10:
        lock = 0.03
    elif profit > 0.065:
        lock = 0.04
    elif profit > 0.04:
        lock = 0.015
    elif profit > 0.019:
        lock = 0.011
    else:
        return active_stop
    locked_stop = entry * (1 + lock) if side == "long" else entry * (1 - lock)
    return max(active_stop, locked_stop) if side == "long" else min(active_stop, locked_stop)


def simulate(
    candles_15m: list[Candle],
    signals: list[Signal],
    exit_signals: dict[tuple[str, str], set[int]],
    config: RunConfig,
    start_ts: int,
    end_ts: int,
) -> list[Trade]:
    trades: list[Trade] = []
    equity = config.initial_equity
    next_allowed = 0
    slip = config.slippage_bps / 10_000
    timestamps = [c.ts for c in candles_15m]
    selected = [
        s
        for s in signals
        if s.family == config.family and s.entry_idx < len(candles_15m) and start_ts <= candles_15m[s.entry_idx].ts < end_ts
    ]
    exit_set = exit_signals.get((config.family, "long"), set())

    for signal in sorted(selected, key=lambda s: s.entry_idx):
        if signal.entry_idx < next_allowed or signal.entry_idx >= len(candles_15m):
            continue
        entry_candle = candles_15m[signal.entry_idx]
        if not (start_ts <= entry_candle.ts < end_ts) or equity <= 0:
            continue
        entry = entry_candle.open * (1 + slip) if signal.side == "long" else entry_candle.open * (1 - slip)
        stop = initial_stop(entry, config.exit_profile, signal.side)
        target = target_price(entry, config.exit_profile, signal.side)
        liq = liquidation_price(entry, config.leverage, signal.side)
        margin_used = equity * config.margin_pct
        notional = margin_used * config.leverage
        qty = notional / entry
        entry_fee = notional * config.fee_rate
        equity_before = equity
        max_exit_ts = entry_candle.ts + int(config.exit_profile.max_hold_hours * 3_600_000)

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
            elif signal.side == "long" and candle.low <= liq:
                exit_price = liq
                reason = "强平"
            elif signal.side == "short" and candle.high >= liq:
                exit_price = liq
                reason = "强平"
            elif signal.side == "long" and candle.low <= stop:
                exit_price = stop * (1 - slip)
                reason = "止损/移动止损"
            elif signal.side == "short" and candle.high >= stop:
                exit_price = stop * (1 + slip)
                reason = "止损/移动止损"
            elif target is not None and signal.side == "long" and candle.high >= target:
                exit_price = target * (1 - slip)
                reason = "固定止盈"
            elif target is not None and signal.side == "short" and candle.low <= target:
                exit_price = target * (1 + slip)
                reason = "固定止盈"
            elif (
                config.exit_profile.use_signal_exit
                and idx in exit_set
                and ((candle.open - entry) / entry if signal.side == "long" else (entry - candle.open) / entry)
                >= config.exit_profile.min_profit_for_signal_exit
            ):
                exit_price = candle.open * (1 - slip) if signal.side == "long" else candle.open * (1 + slip)
                reason = "信号退出"
            elif idx == len(candles_15m) - 1:
                exit_price = candle.close * (1 - slip) if signal.side == "long" else candle.close * (1 + slip)
                reason = "数据结束"

            if exit_price is not None and reason is not None:
                if reason == "强平":
                    pnl = -margin_used - entry_fee
                else:
                    move = (exit_price - entry) / entry if signal.side == "long" else (entry - exit_price) / entry
                    gross = notional * move
                    exit_fee = qty * exit_price * config.fee_rate
                    pnl = gross - entry_fee - exit_fee
                equity_after = equity + pnl
                trades.append(
                    Trade(
                        family=config.family,
                        side=signal.side,
                        entry_ts=entry_candle.ts,
                        exit_ts=candle.ts,
                        entry_price=entry,
                        exit_price=exit_price,
                        stop_price=stop,
                        target_price=target,
                        liquidation_price=liq,
                        leverage=config.leverage,
                        margin_pct=config.margin_pct,
                        margin_used=margin_used,
                        notional=notional,
                        pnl=pnl,
                        equity_before=equity,
                        equity_after=equity_after,
                        exit_reason=reason,
                        hold_hours=(candle.ts - entry_candle.ts) / 3_600_000,
                        note=signal.note,
                    )
                )
                equity = equity_after
                next_allowed = bisect.bisect_right(timestamps, candle.ts)
                break

            if config.exit_profile.use_custom_trailing:
                stop = update_custom_stop(entry, candle.close, stop, signal.side)
    return trades


def summarize(trades: list[Trade], initial_equity: float) -> dict[str, float | int]:
    final = trades[-1].equity_after if trades else initial_equity
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    peak = initial_equity
    max_dd = 0.0
    for trade in trades:
        peak = max(peak, trade.equity_before, trade.equity_after)
        max_dd = max(max_dd, (peak - trade.equity_after) / peak * 100 if peak else 0.0)
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
        "final": final,
        "return_pct": (final / initial_equity - 1) * 100 if initial_equity else 0.0,
        "max_dd": max_dd,
        "avg_win": statistics.mean([t.pnl for t in wins]) if wins else 0.0,
        "avg_loss": statistics.mean([t.pnl for t in losses]) if losses else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss else (math.inf if gross_win > 0 else 0.0),
        "max_loss_streak": max_loss_streak(trades),
        "target": sum(1 for t in trades if t.exit_reason == "固定止盈"),
        "signal_exit": sum(1 for t in trades if t.exit_reason == "信号退出"),
        "stop": sum(1 for t in trades if t.exit_reason == "止损/移动止损"),
        "time_exit": sum(1 for t in trades if t.exit_reason == "时间退出"),
        "liq": sum(1 for t in trades if t.exit_reason == "强平"),
        "avg_hold": statistics.mean([t.hold_hours for t in trades]) if trades else 0.0,
    }


def max_loss_streak(trades: list[Trade]) -> int:
    current = 0
    best = 0
    for trade in trades:
        if trade.pnl <= 0:
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
            "方向",
            "入场时间",
            "出场时间",
            "入场价",
            "出场价",
            "止损价",
            "止盈价",
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
                trade.family,
                "做多" if trade.side == "long" else "做空",
                format_ts(trade.entry_ts),
                format_ts(trade.exit_ts),
                f"{trade.entry_price:.2f}",
                f"{trade.exit_price:.2f}",
                f"{trade.stop_price:.2f}",
                "" if trade.target_price is None else f"{trade.target_price:.2f}",
                f"{trade.liquidation_price:.2f}",
                f"{trade.leverage:.0f}x",
                f"{trade.margin_pct * 100:.0f}%",
                f"{trade.margin_used:.2f}",
                f"{trade.notional:.2f}",
                f"{trade.pnl:.2f}",
                f"{trade.equity_before:.2f}",
                f"{trade.equity_after:.2f}",
                trade.exit_reason,
                f"{trade.hold_hours:.2f}",
                trade.note,
            ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GitHub策略移植回测")
    parser.add_argument("--inst-id", default="BTC-USDT-SWAP")
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--export-dir", type=Path, default=Path("data"))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--initial-equity", type=float, default=1000.0)
    parser.add_argument("--deep", action="store_true", help="扫描8x和30%保证金等更激进组合")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    args = build_parser().parse_args(argv)
    candles_15m = load_candles_csv(latest_cache(args.cache_dir, f"okx_{args.inst_id}_15m_*.csv"))
    end_ts = candles_15m[-1].ts + BAR_MS["15m"]
    start_ts = end_ts - args.days * 24 * 60 * 60 * 1000
    _train_start, _train_end, test_start, _test_end = split_periods(start_ts, end_ts)
    signals, exit_signals = build_all_signals(candles_15m)
    signal_counts = {family: sum(1 for signal in signals if signal.family == family) for family in sorted({s.family for s in signals})}
    print(f"信号数量：{signal_counts}", flush=True)

    profiles = [
        ExitProfile("固定1%止盈_1%止损", 0.01, 0.01, 0.0, False, False, 72),
        ExitProfile("固定1.5%止盈_1%止损", 0.01, 0.015, 0.0, False, False, 96),
        ExitProfile("固定2%止盈_1.5%止损", 0.015, 0.02, 0.0, False, False, 120),
        ExitProfile("固定3%止盈_2%止损", 0.02, 0.03, 0.0, False, False, 168),
        ExitProfile("信号退出_2%止损", 0.02, None, 0.003, True, False, 240),
        ExitProfile("自定义锁盈_4%止损", 0.04, None, 0.003, True, True, 360),
        ExitProfile("信号退出_8%止损", 0.08, None, 0.005, True, False, 720),
        ExitProfile("固定2%止盈_4%止损", 0.04, 0.02, 0.0, False, False, 240),
        ExitProfile("自定义锁盈_8%止损", 0.08, None, 0.005, True, True, 720),
        ExitProfile("自定义锁盈_12%止损", 0.12, None, 0.005, True, True, 720),
    ]
    if args.deep:
        profiles.extend([
            ExitProfile("信号退出_12%止损", 0.12, None, 0.005, True, False, 720),
            ExitProfile("固定4%止盈_6%止损", 0.06, 0.04, 0.0, False, False, 360),
        ])
    leverages = [3.0, 5.0, 8.0] if args.deep else [3.0, 5.0]
    margins = [0.10, 0.20, 0.30] if args.deep else [0.10, 0.20]
    families = sorted({signal.family for signal in signals})

    rows: list[dict[str, str]] = []
    best_runs: list[tuple[RunConfig, list[Trade], dict[str, float | int], dict[str, float | int]]] = []
    for family in families:
        family_runs = []
        for profile in profiles:
            for leverage in leverages:
                for margin_pct in margins:
                    config = RunConfig(family, leverage, margin_pct, profile, args.initial_equity)
                    trades = simulate(candles_15m, signals, exit_signals, config, start_ts, end_ts)
                    test_trades = simulate(candles_15m, signals, exit_signals, config, test_start, end_ts)
                    summary = summarize(trades, args.initial_equity)
                    test_summary = summarize(test_trades, args.initial_equity)
                    if summary["trades"]:
                        family_runs.append((config, trades, summary, test_summary))
                    rows.append({
                        "策略": family,
                        "退出": profile.name,
                        "杠杆": f"{leverage:.0f}x",
                        "保证金": f"{margin_pct * 100:.0f}%",
                        "交易数": str(summary["trades"]),
                        "胜率": f"{summary['win_rate']:.2f}%",
                        "全年收益": f"{summary['return_pct']:.2f}%",
                        "最终权益": f"{summary['final']:.2f}",
                        "最大回撤": f"{summary['max_dd']:.2f}%",
                        "后1/3交易数": str(test_summary["trades"]),
                        "后1/3收益": f"{test_summary['return_pct']:.2f}%",
                        "盈亏因子": "inf" if math.isinf(float(summary["profit_factor"])) else f"{summary['profit_factor']:.2f}",
                        "平均盈利": f"{summary['avg_win']:.2f}",
                        "平均亏损": f"{summary['avg_loss']:.2f}",
                        "最大连亏": str(summary["max_loss_streak"]),
                        "止盈": str(summary["target"]),
                        "信号退出": str(summary["signal_exit"]),
                        "止损": str(summary["stop"]),
                        "时间退出": str(summary["time_exit"]),
                        "强平": str(summary["liq"]),
                        "平均持仓小时": f"{summary['avg_hold']:.2f}",
                    })
        if family_runs:
            best = sorted(
                family_runs,
                key=lambda item: (
                    item[3]["return_pct"] > 0,
                    item[2]["return_pct"],
                    -item[2]["max_dd"],
                    item[2]["profit_factor"],
                ),
                reverse=True,
            )[0]
            best_runs.append(best)

    summary_path = args.export_dir / "github_strategy_scan.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    for config, trades, _summary, _test_summary in best_runs:
        safe_profile = config.exit_profile.name.replace("%", "pct").replace("/", "_")
        export_trades(
            args.export_dir / f"github_best_{config.family}_{safe_profile}_{config.leverage:.0f}x_{config.margin_pct * 100:.0f}pct.csv",
            trades,
        )

    print(f"回测区间：{format_ts(start_ts)} -> {format_ts(end_ts)}")
    print("数据：OKX BTC-USDT-SWAP 15m，GitHub策略规则移植版；手续费0.05%，滑点1bp。")
    print("扫描：3x/5x/8x，保证金10%/20%/30%，多种止损止盈/信号退出。")
    print()
    print("每类策略最优组合：")
    print("策略 | 退出 | 杠杆/保证金 | 交易 | 胜率 | 全年收益 | 最大回撤 | 后1/3收益 | 平均盈亏 | 出场分布")
    print("-" * 150)
    for config, _trades, summary, test_summary in best_runs:
        print(
            f"{config.family} | {config.exit_profile.name} | {config.leverage:.0f}x/{config.margin_pct * 100:.0f}% | "
            f"{summary['trades']} | {summary['win_rate']:.1f}% | {summary['return_pct']:.1f}% | "
            f"{summary['max_dd']:.1f}% | {test_summary['return_pct']:.1f}% | "
            f"{summary['avg_win']:.2f}/{summary['avg_loss']:.2f}U | "
            f"止盈{summary['target']} 信号{summary['signal_exit']} 止损{summary['stop']} 时间{summary['time_exit']}"
        )
    print()
    print(f"汇总CSV：{summary_path}")
    print("最优组合明细：data\\github_best_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
