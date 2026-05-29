#!/usr/bin/env python3
"""Realtime signal checker for the BTC-USDT-SWAP 4H Donchian breakout strategy."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backtest_okx_btc import (
    BAR_MS,
    NetworkConfig,
    compute_ema,
    fetch_candles,
    floor_to_bar,
    format_ts,
    to_ms,
)
from strategy_research import compute_atr, rolling_high, rolling_low
from unified_strategy_backtest import compute_adx


@dataclass(frozen=True)
class StrategyParams:
    donchian: int = 40
    ema: int = 180
    adx_floor: float = 22.0
    atr_period: int = 14
    stop_atr_mult: float = 2.4
    take_profit_r: float = 3.0
    leverage: float = 8.0
    margin_pct: float = 0.15
    initial_equity: float = 1000.0


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查 BTC-USDT-SWAP 4H唐奇安趋势突破实时信号")
    parser.add_argument("--inst-id", default="BTC-USDT-SWAP")
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--proxy-mode", choices=["fallback", "on", "off"], default="fallback")
    parser.add_argument("--proxy-url", default="http://127.0.0.1:7897")
    parser.add_argument("--initial-equity", type=float, default=1000.0)
    parser.add_argument("--leverage", type=float, default=8.0)
    parser.add_argument("--margin-pct", type=float, default=0.15)
    return parser


def closed_4h_index(candles, now_ms: int) -> int:
    for idx in range(len(candles) - 1, -1, -1):
        if candles[idx].ts + BAR_MS["4H"] <= now_ms:
            return idx
    raise RuntimeError("没有找到已收盘4H K线")


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    args = build_parser().parse_args(argv)
    params = StrategyParams(
        leverage=args.leverage,
        margin_pct=args.margin_pct,
        initial_equity=args.initial_equity,
    )
    network = NetworkConfig(proxy_url=args.proxy_url, proxy_mode=args.proxy_mode)
    now = datetime.now(timezone.utc)
    end_dt = floor_to_bar(now, BAR_MS["15m"])
    start_4h = end_dt - timedelta(days=140)
    start_15m = end_dt - timedelta(days=3)

    candles_4h = fetch_candles(
        "okx",
        args.inst_id,
        "4H",
        to_ms(start_4h),
        to_ms(end_dt),
        args.cache_dir,
        refresh=True,
        network=network,
    )
    candles_15m = fetch_candles(
        "okx",
        args.inst_id,
        "15m",
        to_ms(start_15m),
        to_ms(end_dt),
        args.cache_dir,
        refresh=True,
        network=network,
    )

    closes = [c.close for c in candles_4h]
    ema = compute_ema(closes, params.ema)
    atr = compute_atr(candles_4h, params.atr_period)
    adx = compute_adx(candles_4h, 14)
    upper = rolling_high(candles_4h, params.donchian)
    lower = rolling_low(candles_4h, params.donchian)

    idx = closed_4h_index(candles_4h, to_ms(now))
    candle = candles_4h[idx]
    required = (ema[idx], atr[idx], adx[idx], upper[idx], lower[idx])
    if any(value is None for value in required):
        raise RuntimeError("指标数据不足，请扩大拉取区间")

    ema_value = float(ema[idx])
    atr_value = float(atr[idx])
    adx_value = float(adx[idx])
    upper_value = float(upper[idx])
    lower_value = float(lower[idx])
    atr_pct = atr_value / candle.close * 100 if candle.close else 0.0
    long_signal = candle.close > upper_value and candle.close > ema_value and adx_value >= params.adx_floor and 0.4 <= atr_pct <= 6.0
    short_signal = candle.close < lower_value and candle.close < ema_value and adx_value >= params.adx_floor and 0.4 <= atr_pct <= 6.0

    print(f"{args.inst_id} 4H唐奇安趋势突破实时检查")
    print(f"更新时间：{format_ts(to_ms(now))}")
    print()
    print(f"最新已收4H：{format_ts(candle.ts)}  收盘价：{candle.close:.2f}")
    print(f"唐奇安40上轨：{upper_value:.2f}  下轨：{lower_value:.2f}")
    print(f"EMA180：{ema_value:.2f}  ADX14：{adx_value:.2f}  ATR14：{atr_value:.2f} ({atr_pct:.2f}%)")
    print()

    if not long_signal and not short_signal:
        print("信号：无开仓信号")
        print(f"距做多突破上轨：{(upper_value / candle.close - 1) * 100:.2f}%")
        print(f"距做空跌破下轨：{(candle.close / lower_value - 1) * 100:.2f}%")
        print("纪律：未收盘突破不交易，不能用感觉提前进。")
        return 0

    side = "做多" if long_signal else "做空"
    signal_close_ts = candle.ts + BAR_MS["4H"]
    entry_candle = next((c for c in candles_15m if c.ts >= signal_close_ts), None)
    entry_price = entry_candle.open if entry_candle is not None else candle.close
    if long_signal:
        stop = candle.close - atr_value * params.stop_atr_mult
        risk = entry_price - stop
        target = entry_price + risk * params.take_profit_r
    else:
        stop = candle.close + atr_value * params.stop_atr_mult
        risk = stop - entry_price
        target = entry_price - risk * params.take_profit_r

    margin = params.initial_equity * params.margin_pct
    notional = margin * params.leverage
    qty = notional / entry_price if entry_price else 0.0
    window_minutes = (to_ms(now) - signal_close_ts) / 60_000
    status = "可执行窗口内" if 0 <= window_minutes <= 30 else "信号已过30分钟，谨慎追单或放弃"

    print(f"信号：{side}")
    print(f"执行状态：{status}")
    print(f"建议入场参考：{entry_price:.2f}")
    print(f"初始止损：{stop:.2f}")
    print(f"3R止盈：{target:.2f}")
    print(f"单笔保证金：{margin:.2f}U  杠杆：{params.leverage:g}x  名义仓位：{notional:.2f}U  数量约：{qty:.6f} BTC")
    print("开仓后：立刻挂止损和止盈；最多持仓96小时；不加仓。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
