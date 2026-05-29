#!/usr/bin/env python3
"""Realtime BTC-USDT long/short recommendation score for the top/bottom strategy."""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backtest_okx_btc import (
    BAR_MS,
    Candle,
    NetworkConfig,
    compute_rsi,
    format_ts,
    load_candles_csv,
    request_json,
)
from strategy_research import compute_atr, latest_cache


@dataclass(frozen=True)
class ScoreDetail:
    label: str
    long_points: float
    short_points: float
    text: str


@dataclass(frozen=True)
class Recommendation:
    score: int
    label: str
    long_score: float
    short_score: float
    details: list[ScoreDetail]
    candle: Candle
    previous: Candle
    lower_channel: float
    upper_channel: float
    chart_lower_channel: float
    chart_upper_channel: float
    rsi14: float
    atr14: float
    atr_pct: float
    is_closed: bool


@dataclass(frozen=True)
class EntryZone:
    ideal: float
    acceptable_low: float
    acceptable_high: float


@dataclass(frozen=True)
class EntryPlan:
    side: str | None
    signal_price: float
    current_price: float
    ideal_entry: float | None
    acceptable_low: float | None
    acceptable_high: float | None
    structural_stop: float | None
    effective_stop: float | None
    take_profit: float | None
    status: str


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def combine_scores(long_score: float, short_score: float) -> int:
    return round(clamp(50 + (long_score - short_score) / 2, 0, 100))


def recommendation_label(score: int) -> str:
    if score >= 80:
        return "强烈偏多"
    if score >= 60:
        return "偏多预警"
    if score <= 20:
        return "强烈偏空"
    if score <= 40:
        return "偏空预警"
    return "中性观望"


def wick_ratio(open_price: float, high: float, low: float, close: float, side: str) -> float:
    full_range = high - low
    if full_range <= 0:
        return 0.0
    if side == "long":
        wick = min(open_price, close) - low
    elif side == "short":
        wick = high - max(open_price, close)
    else:
        raise ValueError(f"Unsupported side: {side}")
    return clamp(wick / full_range, 0, 1)


def proximity_points(distance_pct: float, full_threshold: float, zero_threshold: float) -> float:
    if distance_pct <= full_threshold:
        return 25.0
    if distance_pct >= zero_threshold:
        return 0.0
    return 25.0 * (zero_threshold - distance_pct) / (zero_threshold - full_threshold)


def rsi_long_points(rsi: float) -> float:
    if rsi <= 35:
        return 25.0
    if rsi >= 50:
        return 0.0
    return 25.0 * (50 - rsi) / 15


def rsi_short_points(rsi: float) -> float:
    if rsi >= 65:
        return 25.0
    if rsi <= 50:
        return 0.0
    return 25.0 * (rsi - 50) / 15


def atr_points(atr_pct: float) -> float:
    if 0.6 <= atr_pct <= 6.0:
        return 10.0
    if atr_pct < 0.6:
        return 10.0 * clamp(atr_pct / 0.6, 0, 1)
    if atr_pct <= 9.0:
        return 10.0 * (9.0 - atr_pct) / 3.0
    return 0.0


def entry_side(score: int) -> str | None:
    if score >= 80:
        return "long"
    if score <= 20:
        return "short"
    return None


def calculate_entry_zone(side: str, signal_price: float, atr: float) -> EntryZone:
    pullback = 0.10 * atr
    outer_band = 0.30 * atr
    chase_band = 0.10 * atr
    if side == "long":
        return EntryZone(
            ideal=signal_price - pullback,
            acceptable_low=signal_price - outer_band,
            acceptable_high=signal_price + chase_band,
        )
    if side == "short":
        return EntryZone(
            ideal=signal_price + pullback,
            acceptable_low=signal_price - chase_band,
            acceptable_high=signal_price + outer_band,
        )
    raise ValueError(f"Unsupported side: {side}")


def build_entry_plan(
    realtime: Recommendation,
    confirmed: Recommendation,
    leverage: float = 20.0,
    liquidation_stop_fraction: float = 0.8,
) -> EntryPlan:
    signal_price = confirmed.candle.close
    current_price = realtime.candle.close
    side = entry_side(confirmed.score)
    if side is None:
        return EntryPlan(
            side=None,
            signal_price=signal_price,
            current_price=current_price,
            ideal_entry=None,
            acceptable_low=None,
            acceptable_high=None,
            structural_stop=None,
            effective_stop=None,
            take_profit=None,
            status="收盘确认不是强信号，先观望",
        )

    zone = calculate_entry_zone(side, signal_price, confirmed.atr14)
    hard_stop_gap = liquidation_stop_fraction / leverage
    if side == "long":
        structural_stop = confirmed.candle.low
        hard_stop = zone.ideal * (1 - hard_stop_gap)
        effective_stop = max(structural_stop, hard_stop)
        risk = zone.ideal - effective_stop
        take_profit = zone.ideal + 0.25 * risk if risk > 0 else None
        if current_price <= structural_stop:
            status = "当前价跌破结构低点，信号失效"
        elif zone.acceptable_low <= current_price <= zone.acceptable_high:
            status = "当前价在可接受入场区，接近理想入场价"
        elif current_price > zone.acceptable_high:
            status = "当前价高于追价上限，等待回落"
        else:
            status = "当前价低于入场区，价格更便宜但动能偏弱，等15m重新转强"
    else:
        structural_stop = confirmed.candle.high
        hard_stop = zone.ideal * (1 + hard_stop_gap)
        effective_stop = min(structural_stop, hard_stop)
        risk = effective_stop - zone.ideal
        take_profit = zone.ideal - 0.25 * risk if risk > 0 else None
        if current_price >= structural_stop:
            status = "当前价突破结构高点，信号失效"
        elif zone.acceptable_low <= current_price <= zone.acceptable_high:
            status = "当前价在可接受入场区，接近理想入场价"
        elif current_price < zone.acceptable_low:
            status = "当前价低于追空下限，等待反抽"
        else:
            status = "当前价高于入场区，价格更高但风险偏大，等15m重新转弱"

    return EntryPlan(
        side=side,
        signal_price=signal_price,
        current_price=current_price,
        ideal_entry=zone.ideal,
        acceptable_low=zone.acceptable_low,
        acceptable_high=zone.acceptable_high,
        structural_stop=structural_stop,
        effective_stop=effective_stop,
        take_profit=take_profit,
        status=status,
    )


def score_candle(candles_4h: list[Candle], index: int, now_ms: int | None = None) -> Recommendation:
    if index < 61:
        raise ValueError("需要至少61根4H K线才能计算唐奇安60")
    closes = [c.close for c in candles_4h]
    rsi = compute_rsi(closes, 14)
    atr = compute_atr(candles_4h, 14)
    if rsi[index] is None or atr[index] is None:
        raise ValueError("K线数量不足，无法计算 RSI14 / ATR14")

    candle = candles_4h[index]
    previous = candles_4h[index - 1]
    channel = candles_4h[index - 60 : index]
    upper = max(c.high for c in channel)
    lower = min(c.low for c in channel)
    chart_channel = candles_4h[index - 59 : index + 1]
    chart_upper = max(c.high for c in chart_channel)
    chart_lower = min(c.low for c in chart_channel)

    low_distance = max(0.0, (candle.low - lower) / lower * 100)
    high_distance = max(0.0, (upper - candle.high) / upper * 100)
    long_wick = wick_ratio(candle.open, candle.high, candle.low, candle.close, "long")
    short_wick = wick_ratio(candle.open, candle.high, candle.low, candle.close, "short")
    rsi_value = float(rsi[index])
    atr_value = float(atr[index])
    atr_pct_value = atr_value / candle.close * 100 if candle.close else 0.0

    details: list[ScoreDetail] = []

    long_position = proximity_points(low_distance, 0.5, 2.0)
    short_position = proximity_points(high_distance, 0.5, 2.0)
    details.append(ScoreDetail("位置", long_position, short_position, f"距下轨 {low_distance:.2f}%，距上轨 {high_distance:.2f}%"))

    long_rsi = rsi_long_points(rsi_value)
    short_rsi = rsi_short_points(rsi_value)
    details.append(ScoreDetail("RSI14", long_rsi, short_rsi, f"RSI14={rsi_value:.2f}"))

    long_shape = 0.0
    short_shape = 0.0
    shape_text = []
    if candle.close > candle.open:
        long_shape += 10
        shape_text.append("阳线")
    if candle.close < candle.open:
        short_shape += 10
        shape_text.append("阴线")
    if candle.close > previous.close:
        long_shape += 10
        shape_text.append("收盘高于前4H")
    if candle.close < previous.close:
        short_shape += 10
        shape_text.append("收盘低于前4H")
    details.append(ScoreDetail("K线方向", long_shape, short_shape, "，".join(shape_text) or "方向不明显"))

    long_wick_points = 20.0 * clamp(long_wick / 0.5, 0, 1)
    short_wick_points = 20.0 * clamp(short_wick / 0.5, 0, 1)
    details.append(ScoreDetail("影线", long_wick_points, short_wick_points, f"下影 {long_wick*100:.1f}%，上影 {short_wick*100:.1f}%"))

    volatility = atr_points(atr_pct_value)
    details.append(ScoreDetail("ATR环境", volatility, volatility, f"ATR14={atr_value:.2f}，ATR%={atr_pct_value:.2f}%"))

    long_score = sum(d.long_points for d in details)
    short_score = sum(d.short_points for d in details)
    score = combine_scores(long_score, short_score)
    now_ms = now_ms if now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
    is_closed = candle.ts + BAR_MS["4H"] <= now_ms

    return Recommendation(
        score=score,
        label=recommendation_label(score),
        long_score=long_score,
        short_score=short_score,
        details=details,
        candle=candle,
        previous=previous,
        lower_channel=lower,
        upper_channel=upper,
        chart_lower_channel=chart_lower,
        chart_upper_channel=chart_upper,
        rsi14=rsi_value,
        atr14=atr_value,
        atr_pct=atr_pct_value,
        is_closed=is_closed,
    )


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


def load_cached_4h(cache_dir: Path, inst_id: str = "BTC-USDT-SWAP") -> list[Candle]:
    return load_candles_csv(latest_cache(cache_dir, f"okx_{inst_id}_4H_*.csv"))


def build_recommendations(candles_4h: list[Candle]) -> tuple[Recommendation, Recommendation]:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    realtime = score_candle(candles_4h, len(candles_4h) - 1, now_ms)
    confirmed_index = len(candles_4h) - 1
    if not realtime.is_closed:
        confirmed_index -= 1
    confirmed = score_candle(candles_4h, confirmed_index, now_ms)
    return realtime, confirmed


def print_recommendation_legacy(realtime: Recommendation, confirmed: Recommendation, inst_id: str) -> None:
    os.system("cls" if os.name == "nt" else "clear")
    print(f"数据源：OKX 公共行情 /api/v5/market/candles，instId={inst_id}，bar=4H")
    print(f"{inst_id} 摸顶摸底实时推荐值")
    print(f"更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print_score_block("实时预警", realtime)
    print()
    print_score_block("收盘确认", confirmed)
    print()
    print("解释：0=强烈偏空，50=中性，100=强烈偏多。")
    print("执行纪律：只按“收盘确认”交易；实时预警只用来盯盘。")


def print_recommendation(realtime: Recommendation, confirmed: Recommendation, inst_id: str) -> None:
    os.system("cls" if os.name == "nt" else "clear")
    print(f"数据源：OKX 公共行情 /api/v5/market/candles，instId={inst_id}，bar=4H")
    print(f"{inst_id} 摸顶摸底实时推荐值")
    print(f"更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print_score_block("实时预警", realtime)
    print()
    print_score_block("收盘确认", confirmed)
    print()
    print_entry_plan(build_entry_plan(realtime, confirmed))
    print()
    print("解释：0=强烈偏空，50=中性，100=强烈偏多。")
    print("执行纪律：只按“收盘确认”交易；实时预警只用来盯盘。")


def print_score_block_legacy(title: str, recommendation: Recommendation) -> None:
    status = "已收盘" if recommendation.is_closed else "未收盘"
    print(f"[{title}] {recommendation.score}/100 {recommendation.label} ({status})")
    print(f"4H时间：{format_ts(recommendation.candle.ts)}  收盘价：{recommendation.candle.close:.2f}")
    print(f"唐奇安60下轨：{recommendation.lower_channel:.2f}  上轨：{recommendation.upper_channel:.2f}")
    print(f"RSI14：{recommendation.rsi14:.2f}  ATR14：{recommendation.atr14:.2f} ({recommendation.atr_pct:.2f}%)")
    print(f"多头证据：{recommendation.long_score:.1f}  空头证据：{recommendation.short_score:.1f}")
    for detail in recommendation.details:
        print(f"- {detail.label}：多 {detail.long_points:.1f} / 空 {detail.short_points:.1f}，{detail.text}")


def print_score_block_legacy_2(title: str, recommendation: Recommendation) -> None:
    status = "已收盘" if recommendation.is_closed else "未收盘"
    print(f"[{title}] {recommendation.score}/100 {recommendation.label} ({status})")
    print(f"4H时间：{format_ts(recommendation.candle.ts)}  收盘价：{recommendation.candle.close:.2f}")
    print(f"唐奇安60策略下轨：{recommendation.lower_channel:.2f}  策略上轨：{recommendation.upper_channel:.2f}")
    print(f"唐奇安60图表下轨：{recommendation.chart_lower_channel:.2f}  图表上轨：{recommendation.chart_upper_channel:.2f}")
    print(f"RSI14：{recommendation.rsi14:.2f}  ATR14：{recommendation.atr14:.2f} ({recommendation.atr_pct:.2f}%)")
    print(f"多头证据：{recommendation.long_score:.1f}  空头证据：{recommendation.short_score:.1f}")
    for detail in recommendation.details:
        print(f"- {detail.label}：多 {detail.long_points:.1f} / 空 {detail.short_points:.1f}，{detail.text}")


def print_score_block(title: str, recommendation: Recommendation) -> None:
    status = "已收盘" if recommendation.is_closed else "未收盘"
    print(f"[{title}] {recommendation.score}/100 {recommendation.label} ({status})")
    print(f"4H时间：{format_ts(recommendation.candle.ts)}  收盘价：{recommendation.candle.close:.2f}")
    print(f"唐奇安60策略下轨：{recommendation.lower_channel:.2f}  策略上轨：{recommendation.upper_channel:.2f}")
    print(f"唐奇安60图表下轨：{recommendation.chart_lower_channel:.2f}  图表上轨：{recommendation.chart_upper_channel:.2f}")
    print(f"RSI14：{recommendation.rsi14:.2f}  ATR14：{recommendation.atr14:.2f} ({recommendation.atr_pct:.2f}%)")
    print(f"多头证据：{recommendation.long_score:.1f}  空头证据：{recommendation.short_score:.1f}")
    for detail in recommendation.details:
        print(f"- {detail.label}：多 {detail.long_points:.1f} / 空 {detail.short_points:.1f}，{detail.text}")


def format_price(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}"


def print_entry_plan(plan: EntryPlan) -> None:
    if plan.side is None:
        print("[最优入场] 暂无交易入场价")
        print(f"信号价：{plan.signal_price:.2f}  当前价：{plan.current_price:.2f}")
        print(f"状态：{plan.status}")
        return

    side_label = "做多" if plan.side == "long" else "做空"
    print(f"[最优入场] {side_label}")
    print(f"信号价：{plan.signal_price:.2f}  当前价：{plan.current_price:.2f}")
    print(f"理想入场价：{format_price(plan.ideal_entry)}")
    print(f"可接受入场区：{format_price(plan.acceptable_low)} - {format_price(plan.acceptable_high)}")
    print(f"结构失效价：{format_price(plan.structural_stop)}")
    print(f"止损参考：{format_price(plan.effective_stop)}  0.25R止盈参考：{format_price(plan.take_profit)}")
    print(f"状态：{plan.status}")


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BTC-USDT 摸顶摸底实时推荐值工具。")
    parser.add_argument("--inst-id", default="BTC-USDT-SWAP")
    parser.add_argument("--proxy-mode", choices=["fallback", "on", "off"], default="fallback")
    parser.add_argument("--proxy-url", default="http://127.0.0.1:7897")
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-only", action="store_true", help="只使用本地缓存，不访问OKX")
    parser.add_argument("--watch", type=int, default=0, help="循环刷新间隔秒数；0表示只显示一次")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    args = build_parser().parse_args(argv)
    network = NetworkConfig(proxy_url=args.proxy_url, proxy_mode=args.proxy_mode)

    while True:
        try:
            candles = load_cached_4h(args.cache_dir, args.inst_id) if args.cache_only else fetch_latest_okx_candles(args.inst_id, "4H", 220, network)
            realtime, confirmed = build_recommendations(candles)
            print_recommendation(realtime, confirmed, args.inst_id)
        except Exception as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 1
        if args.watch <= 0:
            break
        time.sleep(args.watch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
