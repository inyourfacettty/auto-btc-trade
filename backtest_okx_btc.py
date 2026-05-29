#!/usr/bin/env python3
"""Backtest the BTC-USDT EMA/RSI long-only strategy on OKX candles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo


MS_PER_MINUTE = 60_000
BAR_MS = {
    "15m": 15 * MS_PER_MINUTE,
    "4H": 4 * 60 * MS_PER_MINUTE,
}
DIRECT_REQUEST_FAILED = False


@dataclass(frozen=True)
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def replace(self, **changes: float | int) -> "Candle":
        return replace(self, **changes)


@dataclass(frozen=True)
class NetworkConfig:
    proxy_url: str | None = "http://127.0.0.1:7897"
    proxy_mode: str = "fallback"


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 1000.0
    leverage: float = 3.0
    fee_rate: float = 0.0005
    slippage_bps: float = 0.0
    ema_fast: int = 20
    ema_slow: int = 60
    rsi_period: int = 12
    min_rsi: float = 40.0
    max_rsi: float = 55.0
    take_profit_r: float = 2.0
    pullback_lookback: int = 8
    pullback_tolerance_pct: float = 0.001
    stop_lookback: int = 6
    signal_start_ts: int | None = None


@dataclass(frozen=True)
class Trade:
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    stop_price: float
    take_profit_price: float
    qty: float
    gross_pnl: float
    fees: float
    net_pnl: float
    return_pct: float
    r_multiple: float
    exit_reason: str
    equity_before: float
    equity_after: float


@dataclass(frozen=True)
class BacktestResult:
    trades: list[Trade]
    equity_curve: list[tuple[int, float]]
    metrics: dict[str, float | int | str]


def compute_ema(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("EMA period must be positive")

    ema: list[float | None] = [None] * len(values)
    if len(values) < period:
        return ema

    seed = sum(values[:period]) / period
    ema[period - 1] = seed
    alpha = 2 / (period + 1)

    for i in range(period, len(values)):
        previous = ema[i - 1]
        if previous is None:
            raise RuntimeError("EMA seed missing")
        ema[i] = values[i] * alpha + previous * (1 - alpha)

    return ema


def compute_rsi(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("RSI period must be positive")

    rsi: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return rsi

    gains = []
    losses = []
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    rsi[period] = _rsi_from_averages(avg_gain, avg_loss)

    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rsi[i] = _rsi_from_averages(avg_gain, avg_loss)

    return rsi


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    if avg_gain == 0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def run_backtest(
    candles_15m: list[Candle],
    candles_4h: list[Candle],
    config: BacktestConfig,
) -> BacktestResult:
    if len(candles_15m) < config.ema_fast + 2:
        raise ValueError("Not enough 15m candles")
    if len(candles_4h) < config.ema_slow:
        raise ValueError("Not enough 4H candles")

    closes_15m = [c.close for c in candles_15m]
    ema_15m = compute_ema(closes_15m, config.ema_fast)
    rsi_15m = compute_rsi(closes_15m, config.rsi_period)

    closes_4h = [c.close for c in candles_4h]
    ema_4h_fast = compute_ema(closes_4h, config.ema_fast)
    ema_4h_slow = compute_ema(closes_4h, config.ema_slow)
    h4_context = [
        {
            "close_ts": candle.ts + BAR_MS["4H"],
            "close": candle.close,
            "ema_fast": ema_4h_fast[i],
            "ema_slow": ema_4h_slow[i],
        }
        for i, candle in enumerate(candles_4h)
    ]

    trades: list[Trade] = []
    equity = config.initial_equity
    equity_curve: list[tuple[int, float]] = []
    h4_idx = -1
    i = 1

    while i < len(candles_15m) - 1:
        signal_close_ts = candles_15m[i].ts + BAR_MS["15m"]
        while h4_idx + 1 < len(h4_context) and h4_context[h4_idx + 1]["close_ts"] <= signal_close_ts:
            h4_idx += 1

        if config.signal_start_ts is not None and signal_close_ts < config.signal_start_ts:
            i += 1
            continue

        if not _has_signal(i, candles_15m, ema_15m, rsi_15m, h4_context, h4_idx, config):
            i += 1
            continue

        trade, exit_idx = _simulate_trade(i, candles_15m, equity, config)
        if trade is None:
            i += 1
            continue

        trades.append(trade)
        equity = trade.equity_after
        equity_curve.append((trade.exit_ts, equity))

        if equity <= 0:
            break
        i = exit_idx + 1

    metrics = _build_metrics(trades, equity_curve, config.initial_equity, equity)
    return BacktestResult(trades=trades, equity_curve=equity_curve, metrics=metrics)


def _has_signal(
    i: int,
    candles: list[Candle],
    ema_15m: list[float | None],
    rsi_15m: list[float | None],
    h4_context: list[dict[str, float | None]],
    h4_idx: int,
    config: BacktestConfig,
) -> bool:
    if h4_idx < 0:
        return False

    h4 = h4_context[h4_idx]
    h4_fast = h4["ema_fast"]
    h4_slow = h4["ema_slow"]
    if h4_fast is None or h4_slow is None:
        return False
    if not (h4_fast > h4_slow and h4["close"] > h4_slow):
        return False

    current_ema = ema_15m[i]
    previous_ema = ema_15m[i - 1]
    current_rsi = rsi_15m[i]
    previous_rsi = rsi_15m[i - 1]
    if None in (current_ema, previous_ema, current_rsi, previous_rsi):
        return False

    assert current_ema is not None
    assert previous_ema is not None
    assert current_rsi is not None
    assert previous_rsi is not None

    reclaimed_ema = candles[i - 1].close <= previous_ema and candles[i].close > current_ema
    rsi_restrengthened = config.min_rsi <= previous_rsi <= config.max_rsi and current_rsi > previous_rsi
    return reclaimed_ema and rsi_restrengthened and _had_pullback(i, candles, ema_15m, config)


def _had_pullback(
    i: int,
    candles: list[Candle],
    ema_15m: list[float | None],
    config: BacktestConfig,
) -> bool:
    start = max(0, i - config.pullback_lookback + 1)
    for idx in range(start, i + 1):
        ema_value = ema_15m[idx]
        if ema_value is None:
            continue
        if candles[idx].low <= ema_value * (1 + config.pullback_tolerance_pct):
            return True
    return False


def _simulate_trade(
    signal_idx: int,
    candles: list[Candle],
    equity: float,
    config: BacktestConfig,
) -> tuple[Trade | None, int]:
    entry_idx = signal_idx + 1
    entry_candle = candles[entry_idx]
    slippage = config.slippage_bps / 10_000
    entry_price = entry_candle.open * (1 + slippage)
    stop_start = max(0, signal_idx - config.stop_lookback + 1)
    stop_price = min(c.low for c in candles[stop_start : signal_idx + 1])

    if stop_price <= 0 or entry_price <= stop_price:
        return None, signal_idx

    risk_per_btc = entry_price - stop_price
    take_profit_price = entry_price + risk_per_btc * config.take_profit_r
    entry_notional = equity * config.leverage
    qty = entry_notional / entry_price
    entry_fee = entry_notional * config.fee_rate

    for idx in range(entry_idx, len(candles)):
        candle = candles[idx]
        stop_hit = candle.low <= stop_price
        take_profit_hit = candle.high >= take_profit_price

        if stop_hit:
            exit_price = stop_price * (1 - slippage)
            exit_reason = "stop_loss"
        elif take_profit_hit:
            exit_price = take_profit_price * (1 - slippage)
            exit_reason = "take_profit"
        elif idx == len(candles) - 1:
            exit_price = candle.close * (1 - slippage)
            exit_reason = "end_of_data"
        else:
            continue

        exit_notional = qty * exit_price
        exit_fee = exit_notional * config.fee_rate
        gross_pnl = qty * (exit_price - entry_price)
        fees = entry_fee + exit_fee
        net_pnl = gross_pnl - fees
        equity_after = equity + net_pnl
        return_pct = (net_pnl / equity) * 100 if equity else 0
        r_multiple = (exit_price - entry_price) / risk_per_btc

        return (
            Trade(
                entry_ts=entry_candle.ts,
                exit_ts=candle.ts,
                entry_price=entry_price,
                exit_price=exit_price,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                qty=qty,
                gross_pnl=gross_pnl,
                fees=fees,
                net_pnl=net_pnl,
                return_pct=return_pct,
                r_multiple=r_multiple,
                exit_reason=exit_reason,
                equity_before=equity,
                equity_after=equity_after,
            ),
            idx,
        )

    return None, signal_idx


def _build_metrics(
    trades: list[Trade],
    equity_curve: list[tuple[int, float]],
    initial_equity: float,
    final_equity: float,
) -> dict[str, float | int | str]:
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = -sum(t.net_pnl for t in losses)
    returns = [t.return_pct for t in trades]

    peak = initial_equity
    max_drawdown_pct = 0.0
    for _, equity in equity_curve:
        peak = max(peak, equity)
        if peak:
            max_drawdown_pct = max(max_drawdown_pct, ((peak - equity) / peak) * 100)

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": (len(wins) / len(trades) * 100) if trades else 0.0,
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "net_pnl": final_equity - initial_equity,
        "total_return_pct": ((final_equity / initial_equity - 1) * 100) if initial_equity else 0.0,
        "max_drawdown_pct": max_drawdown_pct,
        "profit_factor": (gross_profit / gross_loss) if gross_loss else math.inf if gross_profit else 0.0,
        "avg_trade_return_pct": statistics.mean(returns) if returns else 0.0,
        "avg_r_multiple": statistics.mean([t.r_multiple for t in trades]) if trades else 0.0,
        "fees_paid": sum(t.fees for t in trades),
        "best_trade": max((t.net_pnl for t in trades), default=0.0),
        "worst_trade": min((t.net_pnl for t in trades), default=0.0),
    }


def fetch_candles(
    source: str,
    inst_id: str,
    bar: str,
    start_ms: int,
    end_ms: int,
    cache_dir: Path | None = None,
    refresh: bool = False,
    network: NetworkConfig | None = None,
) -> list[Candle]:
    if source == "okx":
        return fetch_okx_candles(inst_id, bar, start_ms, end_ms, cache_dir, refresh, network)
    if source == "cryptocompare":
        return fetch_cryptocompare_candles(inst_id, bar, start_ms, end_ms, cache_dir, refresh, network)
    raise ValueError(f"Unsupported source: {source}")


def fetch_okx_candles(
    inst_id: str,
    bar: str,
    start_ms: int,
    end_ms: int,
    cache_dir: Path | None = None,
    refresh: bool = False,
    network: NetworkConfig | None = None,
) -> list[Candle]:
    if bar not in BAR_MS:
        raise ValueError(f"Unsupported bar: {bar}")

    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"okx_{inst_id}_{bar}_{start_ms}_{end_ms}.csv"
        if cache_path.exists() and not refresh:
            return load_candles_csv(cache_path)

    candles_by_ts: dict[int, Candle] = {}
    cursor = end_ms
    empty_pages = 0

    while True:
        params = {
            "instId": inst_id,
            "bar": bar,
            "limit": "300",
            "after": str(cursor),
        }
        payload = _okx_get_json("/api/v5/market/history-candles", params, network)
        rows = payload.get("data", [])
        if not rows:
            empty_pages += 1
            if empty_pages >= 2:
                break
            cursor -= BAR_MS[bar] * 300
            continue

        timestamps = []
        for row in rows:
            candle = _parse_okx_candle(row)
            timestamps.append(candle.ts)
            if start_ms <= candle.ts <= end_ms:
                candles_by_ts[candle.ts] = candle

        oldest = min(timestamps)
        if oldest <= start_ms:
            break
        if oldest >= cursor:
            break
        cursor = oldest
        time.sleep(0.12)

    candles = sorted(candles_by_ts.values(), key=lambda c: c.ts)
    if not candles:
        raise RuntimeError(
            "No OKX candles were downloaded. Check network access or rerun with an existing cache."
        )

    expected_first = start_ms + BAR_MS[bar]
    if candles[0].ts > expected_first:
        raise RuntimeError(
            f"Downloaded {bar} data starts too late: first candle {format_ts(candles[0].ts)}"
        )

    if cache_path is not None:
        save_candles_csv(cache_path, candles)
    return candles


def fetch_cryptocompare_candles(
    inst_id: str,
    bar: str,
    start_ms: int,
    end_ms: int,
    cache_dir: Path | None = None,
    refresh: bool = False,
    network: NetworkConfig | None = None,
) -> list[Candle]:
    if bar not in BAR_MS:
        raise ValueError(f"Unsupported bar: {bar}")

    base, quote = split_inst_id(inst_id)
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"cryptocompare_{inst_id}_{bar}_{start_ms}_{end_ms}.csv"
        if cache_path.exists() and not refresh:
            return load_candles_csv(cache_path)

    if bar == "15m":
        endpoint = "/data/v2/histominute"
        aggregate = "15"
    elif bar == "4H":
        endpoint = "/data/v2/histohour"
        aggregate = "4"
    else:
        raise ValueError(f"Unsupported CryptoCompare bar: {bar}")

    candles_by_ts: dict[int, Candle] = {}
    to_ts = end_ms // 1000

    while True:
        payload = _cryptocompare_get_json(
            endpoint,
            {
                "fsym": base,
                "tsym": quote,
                "limit": "2000",
                "aggregate": aggregate,
                "toTs": str(to_ts),
            },
            network,
        )
        rows = payload.get("Data", {}).get("Data", [])
        if not rows:
            break

        times = []
        for row in rows:
            candle = Candle(
                ts=int(row["time"]) * 1000,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volumefrom", 0.0)),
            )
            times.append(candle.ts)
            if start_ms <= candle.ts <= end_ms:
                candles_by_ts[candle.ts] = candle

        oldest = min(times)
        if oldest <= start_ms:
            break
        to_ts = oldest // 1000 - 1
        time.sleep(0.12)

    candles = sorted(candles_by_ts.values(), key=lambda c: c.ts)
    if not candles:
        raise RuntimeError(
            "No CryptoCompare candles were downloaded. Check network access or rerun with an existing cache."
        )

    if cache_path is not None:
        save_candles_csv(cache_path, candles)
    return candles


def _okx_get_json(path: str, params: dict[str, str], network: NetworkConfig | None = None) -> dict:
    query = urllib.parse.urlencode(params)
    url = f"https://www.okx.com{path}?{query}"
    data = request_json(url, "OKX", network)

    if data.get("code") != "0":
        raise RuntimeError(f"OKX API error: {data}")
    return data


def _cryptocompare_get_json(path: str, params: dict[str, str], network: NetworkConfig | None = None) -> dict:
    query = urllib.parse.urlencode(params)
    url = f"https://min-api.cryptocompare.com{path}?{query}"
    data = request_json(url, "CryptoCompare", network)

    if data.get("Response") != "Success":
        raise RuntimeError(f"CryptoCompare API error: {data}")
    return data


def request_json(url: str, service_name: str, network: NetworkConfig | None = None) -> dict:
    network = network or NetworkConfig()
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "btc-ema-rsi-backtester/1.0",
        },
    )

    try:
        with _open_request(request, network) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"连接 {service_name} 失败：{exc}") from exc


def _open_request(request: urllib.request.Request, network: NetworkConfig):
    global DIRECT_REQUEST_FAILED

    if network.proxy_mode not in {"off", "on", "fallback"}:
        raise ValueError(f"Unsupported proxy mode: {network.proxy_mode}")

    if network.proxy_mode == "off" or not network.proxy_url:
        return urllib.request.urlopen(request, timeout=20)

    if network.proxy_mode == "on":
        return _open_with_proxy(request, network.proxy_url)

    if DIRECT_REQUEST_FAILED:
        return _open_with_proxy(request, network.proxy_url)

    try:
        return urllib.request.urlopen(request, timeout=20)
    except (urllib.error.URLError, TimeoutError, OSError):
        DIRECT_REQUEST_FAILED = True
        return _open_with_proxy(request, network.proxy_url)


def _open_with_proxy(request: urllib.request.Request, proxy_url: str):
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(
            {
                "http": proxy_url,
                "https": proxy_url,
            }
        )
    )
    return opener.open(request, timeout=20)


def _parse_okx_candle(row: list[str]) -> Candle:
    return Candle(
        ts=int(row[0]),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
    )


def save_candles_csv(path: Path, candles: Iterable[Candle]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ts", "open", "high", "low", "close", "volume"])
        for candle in candles:
            writer.writerow([candle.ts, candle.open, candle.high, candle.low, candle.close, candle.volume])


def load_candles_csv(path: Path) -> list[Candle]:
    candles: list[Candle] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candles.append(
                Candle(
                    ts=int(row["ts"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return sorted(candles, key=lambda c: c.ts)


def export_trades_csv(path: Path, trades: list[Trade]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "入场时间",
                "出场时间",
                "入场价",
                "出场价",
                "止损价",
                "止盈价",
                "净盈亏",
                "收益率百分比",
                "R倍数",
                "手续费",
                "出场原因",
                "出场后权益",
            ]
        )
        for trade in trades:
            writer.writerow(
                [
                    format_ts(trade.entry_ts),
                    format_ts(trade.exit_ts),
                    f"{trade.entry_price:.2f}",
                    f"{trade.exit_price:.2f}",
                    f"{trade.stop_price:.2f}",
                    f"{trade.take_profit_price:.2f}",
                    f"{trade.net_pnl:.4f}",
                    f"{trade.return_pct:.4f}",
                    f"{trade.r_multiple:.4f}",
                    f"{trade.fees:.4f}",
                    translate_exit_reason(trade.exit_reason),
                    f"{trade.equity_after:.4f}",
                ]
            )


def normalize_inst_id(value: str) -> str:
    return value.upper().replace("_", "-")


def split_inst_id(inst_id: str) -> tuple[str, str]:
    parts = normalize_inst_id(inst_id).split("-")
    if len(parts) != 2:
        raise ValueError(f"Instrument must look like BTC-USDT, got {inst_id}")
    return parts[0], parts[1]


def floor_to_bar(ts: datetime, bar_ms: int) -> datetime:
    seconds = int(ts.timestamp())
    bar_seconds = bar_ms // 1000
    return datetime.fromtimestamp(seconds - (seconds % bar_seconds), timezone.utc)


def parse_utc_datetime(value: str) -> datetime:
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use ISO datetime, e.g. 2026-05-27T00:00:00Z") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def to_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def format_ts(ts_ms: int, tz_name: str = "Asia/Shanghai") -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000, timezone.utc).astimezone(ZoneInfo(tz_name))
    return dt.strftime("%Y-%m-%d %H:%M")


def print_report(
    result: BacktestResult,
    config: BacktestConfig,
    inst_id: str,
    source: str,
    start_ms: int,
    end_ms: int,
    candles_15m: list[Candle],
    candles_4h: list[Candle],
    max_trades: int,
) -> None:
    metrics = result.metrics
    print(f"回测：{inst_id} {config.leverage:g}倍 做多 EMA/RSI")
    print(f"数据源：{source}")
    print(f"区间：{format_ts(start_ms)} -> {format_ts(end_ms)}")
    print(f"数据：15分钟K线={len(candles_15m)}，4小时K线={len(candles_4h)}")
    print(
        "规则：4小时 EMA{fast}>EMA{slow}，15分钟重新站上 EMA{fast}，RSI{rsi} 从 {lo:g}-{hi:g} 区间向上，止盈={tp:g}R".format(
            fast=config.ema_fast,
            slow=config.ema_slow,
            rsi=config.rsi_period,
            lo=config.min_rsi,
            hi=config.max_rsi,
            tp=config.take_profit_r,
        )
    )
    print()
    print(f"交易次数：{metrics['trades']} | 盈利：{metrics['wins']} | 亏损：{metrics['losses']}")
    print(f"胜率：{metrics['win_rate_pct']:.2f}%")
    print(f"初始权益：{metrics['initial_equity']:.2f} USDT")
    print(f"最终权益：{metrics['final_equity']:.2f} USDT")
    print(f"净盈亏：{metrics['net_pnl']:.2f} USDT ({metrics['total_return_pct']:.2f}%)")
    print(f"最大回撤：{metrics['max_drawdown_pct']:.2f}%")
    print(f"盈亏比因子：{_format_float(metrics['profit_factor'])}")
    print(f"平均单笔收益率：{metrics['avg_trade_return_pct']:.2f}%")
    print(f"平均R倍数：{metrics['avg_r_multiple']:.2f}")
    print(f"手续费合计：{metrics['fees_paid']:.2f} USDT")
    print()

    if not result.trades:
        print("没有交易符合策略条件。")
        return

    print("交易明细：")
    for trade in result.trades[-max_trades:]:
        print(
            f"- {format_ts(trade.entry_ts)} -> {format_ts(trade.exit_ts)} "
            f"{translate_exit_reason(trade.exit_reason)} 入场={trade.entry_price:.2f} 出场={trade.exit_price:.2f} "
            f"盈亏={trade.net_pnl:.2f} ({trade.return_pct:.2f}%) 权益={trade.equity_after:.2f}"
        )


def translate_exit_reason(reason: str) -> str:
    return {
        "stop_loss": "止损",
        "take_profit": "止盈",
        "end_of_data": "数据结束平仓",
    }.get(reason, reason)


def _format_float(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and math.isinf(value):
        return "inf"
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="回测 BTC-USDT 3倍做多 EMA/RSI 策略。")
    parser.add_argument("--source", choices=["okx", "cryptocompare"], default="okx")
    parser.add_argument("--proxy-mode", choices=["fallback", "on", "off"], default="fallback", help="代理模式：fallback=直连失败后走代理，on=强制代理，off=不用代理")
    parser.add_argument("--proxy-url", default="http://127.0.0.1:7897", help="代理地址，默认 http://127.0.0.1:7897")
    parser.add_argument("--inst-id", default="BTC-USDT", help="交易对，例如 BTC-USDT 或 BTC_USDT")
    parser.add_argument("--days", type=int, default=30, help="信号回测天数")
    parser.add_argument("--end", type=parse_utc_datetime, default=None, help="UTC ISO结束时间，默认当前时间")
    parser.add_argument("--initial-equity", type=float, default=1000.0)
    parser.add_argument("--leverage", type=float, default=3.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005, help="单边手续费率，默认 taker 0.05%%")
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--ema-fast", type=int, default=20)
    parser.add_argument("--ema-slow", type=int, default=60)
    parser.add_argument("--rsi-period", type=int, default=12)
    parser.add_argument("--min-rsi", type=float, default=40.0)
    parser.add_argument("--max-rsi", type=float, default=55.0)
    parser.add_argument("--take-profit-r", type=float, default=2.0)
    parser.add_argument("--pullback-lookback", type=int, default=8)
    parser.add_argument("--pullback-tolerance-pct", type=float, default=0.001)
    parser.add_argument("--stop-lookback", type=int, default=6)
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--refresh", action="store_true", help="忽略缓存，重新下载K线")
    parser.add_argument("--export", type=Path, default=Path("data/trades_btc_usdt_3x.csv"))
    parser.add_argument("--show-trades", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    args = build_parser().parse_args(argv)
    inst_id = normalize_inst_id(args.inst_id)

    end_dt = args.end or datetime.now(timezone.utc)
    end_dt = floor_to_bar(end_dt, BAR_MS["15m"])
    start_dt = end_dt - timedelta(days=args.days)
    warmup_15m_days = 2
    warmup_4h_days = max(18, math.ceil(args.ema_slow * 4 / 24) + 5)
    fetch_start_15m_dt = start_dt - timedelta(days=warmup_15m_days)
    fetch_start_4h_dt = start_dt - timedelta(days=warmup_4h_days)

    start_ms = to_ms(start_dt)
    end_ms = to_ms(end_dt)
    fetch_start_15m_ms = to_ms(fetch_start_15m_dt)
    fetch_start_4h_ms = to_ms(fetch_start_4h_dt)

    config = BacktestConfig(
        initial_equity=args.initial_equity,
        leverage=args.leverage,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
        ema_fast=args.ema_fast,
        ema_slow=args.ema_slow,
        rsi_period=args.rsi_period,
        min_rsi=args.min_rsi,
        max_rsi=args.max_rsi,
        take_profit_r=args.take_profit_r,
        pullback_lookback=args.pullback_lookback,
        pullback_tolerance_pct=args.pullback_tolerance_pct,
        stop_lookback=args.stop_lookback,
        signal_start_ts=start_ms,
    )
    network = NetworkConfig(proxy_url=args.proxy_url, proxy_mode=args.proxy_mode)

    try:
        candles_15m = fetch_candles(args.source, inst_id, "15m", fetch_start_15m_ms, end_ms, args.cache_dir, args.refresh, network)
        candles_4h = fetch_candles(args.source, inst_id, "4H", fetch_start_4h_ms, end_ms, args.cache_dir, args.refresh, network)
        result = run_backtest(candles_15m, candles_4h, config)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    export_trades_csv(args.export, result.trades)
    print_report(result, config, inst_id, args.source, start_ms, end_ms, candles_15m, candles_4h, args.show_trades)
    print()
    print(f"交易CSV：{args.export}")
    return 0


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
