#!/usr/bin/env python3
"""Five-year backtest for the fixed BTC-USDT-SWAP 4H Donchian breakout strategy."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backtest_okx_btc import (
    BAR_MS,
    Candle,
    NetworkConfig,
    fetch_candles,
    floor_to_bar,
    format_ts,
    load_candles_csv,
    parse_utc_datetime,
    save_candles_csv,
    to_ms,
)
from perp_strategy_lab import (
    Config,
    Trade,
    build_donchian_trend_signals,
    export_trades,
    resample_candles,
    simulate,
    summarize,
)


MS_PER_HOUR = 60 * 60 * 1000


@dataclass(frozen=True)
class FixedStrategy:
    family: str = "4H唐奇安趋势_L40_EMA180_ADX22_S2.4"
    leverage: float = 8.0
    margin_pct: float = 0.15
    trail_mult: float = 3.0
    take_profit_r: float = 3.0
    max_hold_hours: float = 96.0
    initial_equity: float = 1000.0


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def monthly_returns(trades: list[Trade], initial_equity: float) -> list[dict[str, float | int | str]]:
    months: dict[str, list[Trade]] = defaultdict(list)
    for trade in trades:
        months[format_ts(trade.exit_ts)[:7]].append(trade)

    rows: list[dict[str, float | int | str]] = []
    equity_at_month_start = initial_equity
    for month in sorted(months):
        month_trades = months[month]
        pnl = sum(t.net_pnl for t in month_trades)
        wins = sum(1 for t in month_trades if t.net_pnl > 0)
        final_equity = month_trades[-1].equity_after
        rows.append(
            {
                "月份": month,
                "交易数": len(month_trades),
                "盈利数": wins,
                "亏损数": len(month_trades) - wins,
                "净盈亏U": pnl,
                "月收益": pnl / equity_at_month_start * 100 if equity_at_month_start else 0.0,
                "月末权益": final_equity,
            }
        )
        equity_at_month_start = final_equity
    return rows


def yearly_returns(trades: list[Trade], initial_equity: float) -> list[dict[str, float | int | str]]:
    years: dict[str, list[Trade]] = defaultdict(list)
    for trade in trades:
        years[format_ts(trade.exit_ts)[:4]].append(trade)

    rows: list[dict[str, float | int | str]] = []
    equity_at_year_start = initial_equity
    for year in sorted(years):
        year_trades = years[year]
        pnl = sum(t.net_pnl for t in year_trades)
        wins = sum(1 for t in year_trades if t.net_pnl > 0)
        final_equity = year_trades[-1].equity_after
        rows.append(
            {
                "年份": year,
                "交易数": len(year_trades),
                "盈利数": wins,
                "亏损数": len(year_trades) - wins,
                "净盈亏U": pnl,
                "年度收益": pnl / equity_at_year_start * 100 if equity_at_year_start else 0.0,
                "年末权益": final_equity,
            }
        )
        equity_at_year_start = final_equity
    return rows


def export_period_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key, value in row.items():
                formatted[key] = f"{value:.2f}" if isinstance(value, float) else value
            writer.writerow(formatted)


def export_summary(path: Path, summary: dict[str, float | int], extra: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["指标", "数值"])
        for key, value in extra.items():
            writer.writerow([key, value])
        for key, value in summary.items():
            writer.writerow([key, f"{value:.2f}" if isinstance(value, float) and not math.isinf(value) else value])


def fetch_15m_chunked(
    inst_id: str,
    start_dt: datetime,
    end_dt: datetime,
    cache_dir: Path,
    refresh: bool,
    network: NetworkConfig,
    chunk_days: int,
) -> list[Candle]:
    start_ms = to_ms(start_dt)
    end_ms = to_ms(end_dt)
    combined_cache = cache_dir / f"okx_{inst_id}_15m_5y_{start_ms}_{end_ms}.csv"
    if combined_cache.exists() and not refresh:
        print(f"使用已有5年合并缓存：{combined_cache}", flush=True)
        return load_candles_csv(combined_cache)

    ranges: list[tuple[datetime, datetime]] = []
    cursor = start_dt
    while cursor < end_dt:
        next_cursor = min(cursor + timedelta(days=chunk_days), end_dt)
        ranges.append((cursor, next_cursor))
        cursor = next_cursor

    candles_by_ts: dict[int, Candle] = {}
    for idx, (chunk_start, chunk_end) in enumerate(ranges, start=1):
        print(
            f"[{idx}/{len(ranges)}] 拉取15m：{format_ts(to_ms(chunk_start))} -> {format_ts(to_ms(chunk_end))}",
            flush=True,
        )
        chunk = fetch_candles(
            "okx",
            inst_id,
            "15m",
            to_ms(chunk_start),
            to_ms(chunk_end),
            cache_dir,
            refresh=refresh,
            network=network,
        )
        for candle in chunk:
            candles_by_ts[candle.ts] = candle
        print(
            f"    完成：{len(chunk)} 根，本次首尾 {format_ts(chunk[0].ts)} -> {format_ts(chunk[-1].ts)}，累计 {len(candles_by_ts)} 根",
            flush=True,
        )

    candles = sorted(candles_by_ts.values(), key=lambda c: c.ts)
    save_candles_csv(combined_cache, candles)
    print(f"已写入5年合并缓存：{combined_cache}", flush=True)
    return candles


def print_summary(
    inst_id: str,
    start_ms: int,
    end_ms: int,
    candle_count: int,
    signal_count: int,
    strategy: FixedStrategy,
    trades: list[Trade],
    summary: dict[str, float | int],
    years: list[dict[str, float | int | str]],
) -> None:
    print(f"5年固定策略回测：{inst_id}")
    print(f"区间：{format_ts(start_ms)} -> {format_ts(end_ms)}")
    print(f"15m K线数量：{candle_count}  信号数量：{signal_count}")
    print("策略：4H唐奇安40突破 + EMA180 + ADX22 + 2.4ATR止损 + 3R止盈 + 96小时退出")
    print(
        f"仓位：{strategy.initial_equity:.0f}U初始权益，{strategy.leverage:g}x，"
        f"单笔保证金{strategy.margin_pct * 100:.0f}%，手续费0.05%，滑点1bp"
    )
    print()
    print(
        f"交易：{summary['trades']} | 胜率：{summary['win_rate']:.2f}% | "
        f"总收益：{summary['return_pct']:.2f}% | 最终权益：{summary['final_equity']:.2f}U | "
        f"最大回撤：{summary['max_dd']:.2f}% | PF：{summary['pf']:.2f}"
    )
    print(
        f"平均盈利：{summary['avg_win']:.2f}U | 平均亏损：{summary['avg_loss']:.2f}U | "
        f"最大盈利：{summary['best']:.2f}U | 最大亏损：{summary['worst']:.2f}U | "
        f"最大连亏：{summary['max_loss_streak']} | 强平：{summary['liq']}"
    )
    print()
    print("年度结果：")
    for row in years:
        print(
            f"- {row['年份']}：交易 {row['交易数']}，收益 {row['年度收益']:.2f}%，"
            f"净盈亏 {row['净盈亏U']:.2f}U，年末权益 {row['年末权益']:.2f}U"
        )
    if trades:
        print()
        print("最近10笔：")
        for trade in trades[-10:]:
            side = "做多" if trade.side == "long" else "做空"
            print(
                f"- {format_ts(trade.entry_ts)} {side} -> {format_ts(trade.exit_ts)} "
                f"{trade.exit_reason} 盈亏 {trade.net_pnl:.2f}U 权益 {trade.equity_after:.2f}U"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="拉取并回测过去5年 BTC-USDT-SWAP 固定趋势突破策略")
    parser.add_argument("--inst-id", default="BTC-USDT-SWAP")
    parser.add_argument("--days", type=int, default=365 * 5)
    parser.add_argument("--end", type=parse_utc_datetime, default=None, help="UTC ISO结束时间，例如 2026-05-29T07:00:00Z")
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--export-dir", type=Path, default=Path("data"))
    parser.add_argument("--proxy-mode", choices=["fallback", "on", "off"], default="fallback")
    parser.add_argument("--proxy-url", default="http://127.0.0.1:7897")
    parser.add_argument("--chunk-days", type=int, default=90, help="分段拉取天数，默认90天")
    parser.add_argument("--initial-equity", type=float, default=1000.0)
    parser.add_argument("--leverage", type=float, default=8.0)
    parser.add_argument("--margin-pct", type=float, default=0.15)
    parser.add_argument("--refresh", action="store_true", help="强制重新拉取，不使用已有缓存")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    args = build_parser().parse_args(argv)
    strategy = FixedStrategy(
        leverage=args.leverage,
        margin_pct=args.margin_pct,
        initial_equity=args.initial_equity,
    )
    network = NetworkConfig(proxy_url=args.proxy_url, proxy_mode=args.proxy_mode)
    end_dt = args.end if args.end is not None else datetime.now(timezone.utc)
    end_dt = floor_to_bar(end_dt, BAR_MS["15m"])
    start_dt = end_dt - timedelta(days=args.days)
    start_ms = to_ms(start_dt)
    end_ms = to_ms(end_dt)

    try:
        candles_15m = fetch_15m_chunked(
            args.inst_id,
            start_dt,
            end_dt,
            args.cache_dir,
            args.refresh,
            network,
            args.chunk_days,
        )
        candles_4h = resample_candles(candles_15m, BAR_MS["4H"])
        timestamps_15m = [c.ts for c in candles_15m]
        signals = build_donchian_trend_signals(
            strategy.family,
            candles_15m,
            candles_4h,
            timestamps_15m,
            BAR_MS["4H"],
            lookback=40,
            ema_period=180,
            stop_mult=2.4,
            adx_floor=22,
            atr_pct_floor=0.004,
            atr_pct_ceiling=0.06,
        )
        config = Config(
            family=strategy.family,
            name=f"{strategy.family}_5年固定版_{strategy.leverage:g}x_{strategy.margin_pct*100:.0f}%",
            leverage=strategy.leverage,
            margin_pct=strategy.margin_pct,
            stop_mult=0.0,
            trail_mult=strategy.trail_mult,
            max_hold_hours=strategy.max_hold_hours,
            take_profit_r=strategy.take_profit_r,
            breakeven_r=1.0,
            initial_equity=strategy.initial_equity,
        )
        trades = simulate(candles_15m, signals, config, start_ms, end_ms)
        summary = summarize(trades, strategy.initial_equity)
        months = monthly_returns(trades, strategy.initial_equity)
        years = yearly_returns(trades, strategy.initial_equity)

        file_prefix = f"fixed_strategy_5y_{strategy.leverage:g}x_{strategy.margin_pct * 100:.0f}pct"
        trades_path = args.export_dir / f"{file_prefix}_trades.csv"
        monthly_path = args.export_dir / f"{file_prefix}_monthly.csv"
        yearly_path = args.export_dir / f"{file_prefix}_yearly.csv"
        summary_path = args.export_dir / f"{file_prefix}_summary.csv"
        export_trades(trades_path, trades)
        export_period_rows(monthly_path, months)
        export_period_rows(yearly_path, years)
        export_summary(
            summary_path,
            summary,
            {
                "交易品种": args.inst_id,
                "开始时间": format_ts(start_ms),
                "结束时间": format_ts(end_ms),
                "15mK线数量": len(candles_15m),
                "4HK线数量": len(candles_4h),
                "信号数量": len(signals),
                "策略": config.name,
            },
        )
        print_summary(args.inst_id, start_ms, end_ms, len(candles_15m), len(signals), strategy, trades, summary, years)
        print()
        print(f"逐笔交易：{trades_path}")
        print(f"月度结果：{monthly_path}")
        print(f"年度结果：{yearly_path}")
        print(f"汇总结果：{summary_path}")
        return 0
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
