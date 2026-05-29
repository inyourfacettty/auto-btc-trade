#!/usr/bin/env python3
"""Watch OKX BTC-USDT-SWAP for the executable 4H Donchian breakout strategy."""

from __future__ import annotations

import argparse
import bisect
import os
import sys
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone

from backtest_okx_btc import BAR_MS, Candle, NetworkConfig, compute_ema, format_ts, request_json
from strategy_research import compute_atr, rolling_high, rolling_low
from unified_strategy_backtest import compute_adx


@dataclass(frozen=True)
class StrategyParams:
    donchian: int = 40
    ema_period: int = 180
    adx_floor: float = 22.0
    atr_period: int = 14
    stop_atr_mult: float = 2.4
    take_profit_r: float = 3.0
    min_atr_pct: float = 0.4
    max_atr_pct: float = 6.0
    signal_window_minutes: int = 30
    max_hold_hours: int = 96


@dataclass(frozen=True)
class Opportunity:
    candle: Candle
    is_closed: bool
    upper: float
    lower: float
    ema: float
    adx: float
    atr: float
    atr_pct: float
    long_score: int
    short_score: int
    long_signal: bool
    short_signal: bool
    long_reasons: list[str]
    short_reasons: list[str]


@dataclass(frozen=True)
class EntryPlan:
    side: str | None
    status: str
    entry_price: float | None
    stop_price: float | None
    take_profit_price: float | None
    risk: float | None
    margin: float
    notional: float
    qty: float | None
    signal_age_minutes: float | None


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_okx_candle(row: list[str]) -> Candle:
    return Candle(
        ts=int(row[0]),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
    )


def fetch_latest_okx_candles(inst_id: str, bar: str, limit: int, network: NetworkConfig) -> list[Candle]:
    query = urllib.parse.urlencode({"instId": inst_id, "bar": bar, "limit": str(limit)})
    payload = request_json(f"https://www.okx.com/api/v5/market/candles?{query}", "OKX", network)
    if payload.get("code") != "0":
        raise RuntimeError(f"OKX API错误：{payload}")
    return sorted((parse_okx_candle(row) for row in payload.get("data", [])), key=lambda c: c.ts)


def latest_confirmed_index(candles: list[Candle], now_ms: int) -> int:
    for idx in range(len(candles) - 1, -1, -1):
        if candles[idx].ts + BAR_MS["4H"] <= now_ms:
            return idx
    raise RuntimeError("没有找到已收盘4H K线")


def score_distance_to_breakout(close: float, breakout_price: float, side: str) -> tuple[float, str]:
    if breakout_price <= 0:
        return 0.0, "突破价异常"
    if side == "long":
        distance_pct = (breakout_price / close - 1) * 100
        if close > breakout_price:
            return 45.0, f"已高于上轨 {abs(distance_pct):.2f}%"
        if distance_pct <= 1.0:
            return 35.0 + (1.0 - distance_pct) * 10.0, f"距上轨 {distance_pct:.2f}%"
        if distance_pct <= 3.0:
            return 20.0 + (3.0 - distance_pct) / 2.0 * 15.0, f"距上轨 {distance_pct:.2f}%"
        return 0.0, f"距上轨 {distance_pct:.2f}%"

    distance_pct = (close / breakout_price - 1) * 100
    if close < breakout_price:
        return 45.0, f"已低于下轨 {abs(distance_pct):.2f}%"
    if distance_pct <= 1.0:
        return 35.0 + (1.0 - distance_pct) * 10.0, f"距下轨 {distance_pct:.2f}%"
    if distance_pct <= 3.0:
        return 20.0 + (3.0 - distance_pct) / 2.0 * 15.0, f"距下轨 {distance_pct:.2f}%"
    return 0.0, f"距下轨 {distance_pct:.2f}%"


def build_opportunity(candles_4h: list[Candle], index: int, now_ms: int, params: StrategyParams) -> Opportunity:
    if index < params.ema_period:
        raise RuntimeError("4H K线数量不足，无法计算 EMA180")

    closes = [c.close for c in candles_4h]
    ema = compute_ema(closes, params.ema_period)
    atr = compute_atr(candles_4h, params.atr_period)
    adx = compute_adx(candles_4h, 14)
    upper = rolling_high(candles_4h, params.donchian)
    lower = rolling_low(candles_4h, params.donchian)
    required = (ema[index], atr[index], adx[index], upper[index], lower[index])
    if any(value is None for value in required):
        raise RuntimeError("指标数据不足，无法计算完整机会")

    candle = candles_4h[index]
    ema_value = float(ema[index])
    atr_value = float(atr[index])
    adx_value = float(adx[index])
    upper_value = float(upper[index])
    lower_value = float(lower[index])
    atr_pct = atr_value / candle.close * 100 if candle.close else 0.0
    atr_ok = params.min_atr_pct <= atr_pct <= params.max_atr_pct
    adx_ok = adx_value >= params.adx_floor
    is_closed = candle.ts + BAR_MS["4H"] <= now_ms

    long_position_points, long_position_text = score_distance_to_breakout(candle.close, upper_value, "long")
    short_position_points, short_position_text = score_distance_to_breakout(candle.close, lower_value, "short")

    long_reasons = [long_position_text]
    short_reasons = [short_position_text]
    long_score = long_position_points
    short_score = short_position_points

    if candle.close > ema_value:
        long_score += 20
        long_reasons.append("收盘价在EMA180上方")
    else:
        long_reasons.append("收盘价未站上EMA180")
    if candle.close < ema_value:
        short_score += 20
        short_reasons.append("收盘价在EMA180下方")
    else:
        short_reasons.append("收盘价未跌破EMA180")

    if adx_ok:
        long_score += 20
        short_score += 20
        adx_text = f"ADX14={adx_value:.2f}，趋势强度达标"
    else:
        adx_text = f"ADX14={adx_value:.2f}，趋势强度不足"
    long_reasons.append(adx_text)
    short_reasons.append(adx_text)

    if atr_ok:
        long_score += 15
        short_score += 15
        atr_text = f"ATR%={atr_pct:.2f}%，波动环境达标"
    else:
        atr_text = f"ATR%={atr_pct:.2f}%，波动环境不达标"
    long_reasons.append(atr_text)
    short_reasons.append(atr_text)

    long_signal = candle.close > upper_value and candle.close > ema_value and adx_ok and atr_ok
    short_signal = candle.close < lower_value and candle.close < ema_value and adx_ok and atr_ok
    return Opportunity(
        candle=candle,
        is_closed=is_closed,
        upper=upper_value,
        lower=lower_value,
        ema=ema_value,
        adx=adx_value,
        atr=atr_value,
        atr_pct=atr_pct,
        long_score=round(clamp(long_score, 0, 100)),
        short_score=round(clamp(short_score, 0, 100)),
        long_signal=long_signal,
        short_signal=short_signal,
        long_reasons=long_reasons,
        short_reasons=short_reasons,
    )


def build_realtime_and_confirmed(candles_4h: list[Candle], now_ms: int, params: StrategyParams) -> tuple[Opportunity, Opportunity]:
    realtime = build_opportunity(candles_4h, len(candles_4h) - 1, now_ms, params)
    confirmed_idx = latest_confirmed_index(candles_4h, now_ms)
    confirmed = build_opportunity(candles_4h, confirmed_idx, now_ms, params)
    return realtime, confirmed


def build_entry_plan(
    confirmed: Opportunity,
    candles_15m: list[Candle],
    now_ms: int,
    params: StrategyParams,
    initial_equity: float,
    leverage: float,
    margin_pct: float,
) -> EntryPlan:
    margin = initial_equity * margin_pct
    notional = margin * leverage
    side: str | None
    if confirmed.long_signal:
        side = "long"
    elif confirmed.short_signal:
        side = "short"
    else:
        return EntryPlan(
            side=None,
            status="收盘确认没有交易信号，继续等待",
            entry_price=None,
            stop_price=None,
            take_profit_price=None,
            risk=None,
            margin=margin,
            notional=notional,
            qty=None,
            signal_age_minutes=None,
        )

    signal_close_ts = confirmed.candle.ts + BAR_MS["4H"]
    timestamps_15m = [c.ts for c in candles_15m]
    entry_idx = bisect.bisect_left(timestamps_15m, signal_close_ts)
    entry_price = candles_15m[entry_idx].open if entry_idx < len(candles_15m) else confirmed.candle.close
    if side == "long":
        stop = confirmed.candle.close - confirmed.atr * params.stop_atr_mult
        risk = entry_price - stop
        take_profit = entry_price + risk * params.take_profit_r
    else:
        stop = confirmed.candle.close + confirmed.atr * params.stop_atr_mult
        risk = stop - entry_price
        take_profit = entry_price - risk * params.take_profit_r

    qty = notional / entry_price if entry_price > 0 else None
    age_minutes = (now_ms - signal_close_ts) / 60_000
    if 0 <= age_minutes <= params.signal_window_minutes:
        status = f"可执行窗口内，距离4H收盘约 {age_minutes:.0f} 分钟"
    elif age_minutes > params.signal_window_minutes:
        status = f"信号已超过 {params.signal_window_minutes} 分钟，谨慎追单或放弃"
    else:
        status = "信号K线尚未收盘，只能预警不能交易"

    return EntryPlan(
        side=side,
        status=status,
        entry_price=entry_price,
        stop_price=stop,
        take_profit_price=take_profit,
        risk=risk,
        margin=margin,
        notional=notional,
        qty=qty,
        signal_age_minutes=age_minutes,
    )


def signal_label(opportunity: Opportunity) -> str:
    if opportunity.long_signal:
        return "做多机会"
    if opportunity.short_signal:
        return "做空机会"
    if opportunity.long_score >= 80:
        return "接近做多触发"
    if opportunity.short_score >= 80:
        return "接近做空触发"
    return "无开仓信号"


def print_opportunity_block(title: str, opportunity: Opportunity) -> None:
    status = "已收盘" if opportunity.is_closed else "未收盘"
    print(f"[{title}] {signal_label(opportunity)} ({status})")
    print(f"4H时间：{format_ts(opportunity.candle.ts)}  当前/收盘价：{opportunity.candle.close:.2f}")
    print(f"唐奇安40上轨：{opportunity.upper:.2f}  下轨：{opportunity.lower:.2f}")
    print(f"EMA180：{opportunity.ema:.2f}  ADX14：{opportunity.adx:.2f}  ATR14：{opportunity.atr:.2f} ({opportunity.atr_pct:.2f}%)")
    print(f"做多机会分：{opportunity.long_score}/100  做空机会分：{opportunity.short_score}/100")
    print("- 做多条件：" + "；".join(opportunity.long_reasons))
    print("- 做空条件：" + "；".join(opportunity.short_reasons))


def print_entry_plan(plan: EntryPlan, params: StrategyParams) -> None:
    print("[执行计划]")
    if plan.side is None:
        print(plan.status)
        print(f"默认仓位模板：保证金 {plan.margin:.2f}U，名义仓位 {plan.notional:.2f}U")
        return

    side_label = "做多" if plan.side == "long" else "做空"
    print(f"方向：{side_label}")
    print(f"状态：{plan.status}")
    print(f"建议入场参考：{plan.entry_price:.2f}")
    print(f"初始止损：{plan.stop_price:.2f}")
    print(f"3R止盈：{plan.take_profit_price:.2f}")
    print(f"单笔风险距离：{plan.risk:.2f}")
    print(f"保证金：{plan.margin:.2f}U  名义仓位：{plan.notional:.2f}U  数量约：{plan.qty:.6f} BTC")
    print(f"最长持仓：{params.max_hold_hours}小时；开仓后立刻挂止损和止盈；不加仓。")


def print_report(
    inst_id: str,
    realtime: Opportunity,
    confirmed: Opportunity,
    plan: EntryPlan,
    params: StrategyParams,
) -> None:
    os.system("cls" if os.name == "nt" else "clear")
    print(f"数据源：OKX 公共行情 /api/v5/market/candles，instId={inst_id}")
    print(f"{inst_id} 4H唐奇安趋势突破机会扫描")
    print(f"更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print_opportunity_block("实时预警", realtime)
    print()
    print_opportunity_block("收盘确认", confirmed)
    print()
    print_entry_plan(plan, params)
    print()
    print("解释：机会分只用于盯盘；真正交易只看“收盘确认”。")
    print("执行纪律：4H收盘后30分钟内执行；未收盘突破不交易；没有信号就空仓。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="扫描 BTC-USDT-SWAP 4H唐奇安趋势突破机会")
    parser.add_argument("--inst-id", default="BTC-USDT-SWAP")
    parser.add_argument("--watch", type=int, default=0, help="循环刷新间隔秒数；0表示只显示一次")
    parser.add_argument("--proxy-mode", choices=["fallback", "on", "off"], default="fallback")
    parser.add_argument("--proxy-url", default="http://127.0.0.1:7897")
    parser.add_argument("--initial-equity", type=float, default=1000.0)
    parser.add_argument("--leverage", type=float, default=8.0)
    parser.add_argument("--margin-pct", type=float, default=0.15, help="单笔保证金占账户比例，例如0.15表示15%")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    args = build_parser().parse_args(argv)
    params = StrategyParams()
    network = NetworkConfig(proxy_url=args.proxy_url, proxy_mode=args.proxy_mode)
    while True:
        try:
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            candles_4h = fetch_latest_okx_candles(args.inst_id, "4H", 300, network)
            candles_15m = fetch_latest_okx_candles(args.inst_id, "15m", 80, network)
            realtime, confirmed = build_realtime_and_confirmed(candles_4h, now_ms, params)
            plan = build_entry_plan(
                confirmed,
                candles_15m,
                now_ms,
                params,
                args.initial_equity,
                args.leverage,
                args.margin_pct,
            )
            print_report(args.inst_id, realtime, confirmed, plan, params)
        except Exception as exc:
            print(f"错误：{exc}", file=sys.stderr)
            if args.watch <= 0:
                return 1

        if args.watch <= 0:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
