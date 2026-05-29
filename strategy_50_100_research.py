#!/usr/bin/env python3
"""Research BTC-USDT-SWAP strategies targeting 50-100 USDT average winning trades."""

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
from strategy_research import compute_atr, latest_cache, rolling_high, rolling_low


@dataclass(frozen=True)
class Signal:
    family: str
    entry_idx: int
    side: str
    stop_price: float
    take_profit_r: float
    note: str


@dataclass(frozen=True)
class Config:
    family: str
    name: str
    initial_equity: float = 1000.0
    leverage: float = 20.0
    margin_pct: float = 0.10
    fee_rate: float = 0.0005
    slippage_bps: float = 1.0
    take_profit_r: float = 2.0
    max_hold_hours: float | None = None


@dataclass(frozen=True)
class Trade:
    family: str
    name: str
    side: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    stop_price: float
    take_profit_price: float
    liquidation_price: float
    notional: float
    margin_used: float
    net_pnl: float
    return_pct: float
    exit_reason: str
    equity_before: float
    equity_after: float
    hold_hours: float
    note: str


def first_15m_at_or_after(candles_15m: list[Candle], ts: int) -> int | None:
    timestamps = [c.ts for c in candles_15m]
    idx = bisect.bisect_left(timestamps, ts)
    return idx if idx < len(candles_15m) else None


def liquidation_price(entry_price: float, leverage: float, side: str) -> float:
    gap = 0.90 / leverage
    if side == "long":
        return entry_price * (1 - gap)
    if side == "short":
        return entry_price * (1 + gap)
    raise ValueError(f"Unsupported side: {side}")


def top_bottom_signals(candles_15m: list[Candle], candles_4h: list[Candle]) -> list[Signal]:
    closes = [c.close for c in candles_4h]
    rsi = compute_rsi(closes, 14)
    atr = compute_atr(candles_4h, 14)
    signals: list[Signal] = []
    window = 60
    for i in range(window + 1, len(candles_4h)):
        if rsi[i] is None or atr[i] is None:
            continue
        candle = candles_4h[i]
        previous = candles_4h[i - 1]
        lower = min(c.low for c in candles_4h[i - window : i])
        upper = max(c.high for c in candles_4h[i - window : i])
        atr_pct = atr[i] / candle.close if candle.close else 0.0
        if not (0.006 <= atr_pct <= 0.06):
            continue
        entry_idx = first_15m_at_or_after(candles_15m, candle.ts + BAR_MS["4H"])
        if entry_idx is None:
            continue
        lower_wick = wick_ratio(candle.open, candle.high, candle.low, candle.close, "long")
        upper_wick = wick_ratio(candle.open, candle.high, candle.low, candle.close, "short")
        if (
            candle.low <= lower * 1.005
            and rsi[i] <= 35
            and candle.close > candle.open
            and candle.close > previous.close
            and lower_wick >= 0.50
        ):
            signals.append(Signal("摸底摸顶", entry_idx, "long", candle.low, 1.0, "唐奇安60下轨反转"))
        elif (
            candle.high >= upper * 0.995
            and rsi[i] >= 65
            and candle.close < candle.open
            and candle.close < previous.close
            and upper_wick >= 0.50
        ):
            signals.append(Signal("摸底摸顶", entry_idx, "short", candle.high, 1.0, "唐奇安60上轨反转"))
    return signals


def trend_breakout_signals(candles_15m: list[Candle], candles_4h: list[Candle]) -> list[Signal]:
    closes = [c.close for c in candles_4h]
    ema20 = compute_ema(closes, 20)
    ema60 = compute_ema(closes, 60)
    ema120 = compute_ema(closes, 120)
    rsi = compute_rsi(closes, 14)
    atr = compute_atr(candles_4h, 14)
    high40 = rolling_high(candles_4h, 40)
    low40 = rolling_low(candles_4h, 40)
    signals: list[Signal] = []
    for i in range(130, len(candles_4h)):
        required = (ema20[i], ema60[i], ema120[i], rsi[i], atr[i], high40[i], low40[i])
        if any(v is None for v in required):
            continue
        candle = candles_4h[i]
        atr_pct = atr[i] / candle.close if candle.close else 0.0
        if not (0.004 <= atr_pct <= 0.05):
            continue
        entry_idx = first_15m_at_or_after(candles_15m, candle.ts + BAR_MS["4H"])
        if entry_idx is None:
            continue
        if candle.close > high40[i] and candle.close > ema120[i] and ema20[i] > ema60[i] and rsi[i] >= 54:
            stop = candle.close - atr[i] * 2.4
            signals.append(Signal("趋势突破", entry_idx, "long", stop, 1.0, "4H突破40根高点"))
        elif candle.close < low40[i] and candle.close < ema120[i] and ema20[i] < ema60[i] and rsi[i] <= 46:
            stop = candle.close + atr[i] * 2.4
            signals.append(Signal("趋势突破", entry_idx, "short", stop, 1.0, "4H跌破40根低点"))
    return signals


def trend_pullback_signals(candles_15m: list[Candle], candles_4h: list[Candle]) -> list[Signal]:
    closes = [c.close for c in candles_4h]
    ema20 = compute_ema(closes, 20)
    ema60 = compute_ema(closes, 60)
    ema120 = compute_ema(closes, 120)
    rsi = compute_rsi(closes, 14)
    atr = compute_atr(candles_4h, 14)
    signals: list[Signal] = []
    for i in range(132, len(candles_4h)):
        required = (ema20[i], ema60[i], ema120[i], rsi[i], atr[i])
        if any(v is None for v in required):
            continue
        candle = candles_4h[i]
        atr_pct = atr[i] / candle.close if candle.close else 0.0
        if not (0.004 <= atr_pct <= 0.045):
            continue
        entry_idx = first_15m_at_or_after(candles_15m, candle.ts + BAR_MS["4H"])
        if entry_idx is None:
            continue
        long_trend = candle.close > ema120[i] and ema20[i] > ema60[i] > ema120[i] and ema60[i] > ema60[i - 12]
        short_trend = candle.close < ema120[i] and ema20[i] < ema60[i] < ema120[i] and ema60[i] < ema60[i - 12]
        if long_trend and 45 <= rsi[i] <= 62 and candle.low <= ema20[i] * 1.006 and candle.close > ema20[i]:
            swing_stop = min(c.low for c in candles_4h[max(0, i - 5) : i + 1])
            stop = min(swing_stop, candle.close - atr[i] * 1.6)
            signals.append(Signal("趋势回踩", entry_idx, "long", stop, 1.0, "多头趋势回踩EMA20后收回"))
        elif short_trend and 38 <= rsi[i] <= 55 and candle.high >= ema20[i] * 0.994 and candle.close < ema20[i]:
            swing_stop = max(c.high for c in candles_4h[max(0, i - 5) : i + 1])
            stop = max(swing_stop, candle.close + atr[i] * 1.6)
            signals.append(Signal("趋势回踩", entry_idx, "short", stop, 1.0, "空头趋势反弹EMA20后压回"))
    return signals


def h4_momentum_retest_signals(candles_15m: list[Candle], candles_4h: list[Candle]) -> list[Signal]:
    closes = [c.close for c in candles_4h]
    ema60 = compute_ema(closes, 60)
    rsi = compute_rsi(closes, 14)
    atr = compute_atr(candles_4h, 14)
    high20 = rolling_high(candles_4h, 20)
    low20 = rolling_low(candles_4h, 20)
    signals: list[Signal] = []
    for i in range(70, len(candles_4h)):
        required = (ema60[i], rsi[i], atr[i], high20[i], low20[i])
        if any(v is None for v in required):
            continue
        candle = candles_4h[i]
        prev = candles_4h[i - 1]
        atr_pct = atr[i] / candle.close if candle.close else 0.0
        if not (0.005 <= atr_pct <= 0.055):
            continue
        entry_idx = first_15m_at_or_after(candles_15m, candle.ts + BAR_MS["4H"])
        if entry_idx is None:
            continue
        if candle.close > high20[i] and candle.close > ema60[i] and rsi[i] >= 58 and prev.close <= high20[i]:
            stop = candle.close - atr[i] * 1.8
            signals.append(Signal("动量突破", entry_idx, "long", stop, 1.0, "4H放量式向上动量突破"))
        elif candle.close < low20[i] and candle.close < ema60[i] and rsi[i] <= 42 and prev.close >= low20[i]:
            stop = candle.close + atr[i] * 1.8
            signals.append(Signal("动量突破", entry_idx, "short", stop, 1.0, "4H放量式向下动量突破"))
    return signals


def simulate(
    candles_15m: list[Candle],
    signals: list[Signal],
    config: Config,
    start_ts: int,
    end_ts: int,
) -> list[Trade]:
    trades: list[Trade] = []
    equity = config.initial_equity
    next_allowed = 0
    slip = config.slippage_bps / 10_000
    max_hold_ms = None if config.max_hold_hours is None else int(config.max_hold_hours * 60 * 60 * 1000)

    for signal in sorted(signals, key=lambda s: s.entry_idx):
        if signal.family != config.family or signal.entry_idx < next_allowed or signal.entry_idx >= len(candles_15m):
            continue
        entry_candle = candles_15m[signal.entry_idx]
        if not (start_ts <= entry_candle.ts < end_ts):
            continue
        entry = entry_candle.open * (1 + slip) if signal.side == "long" else entry_candle.open * (1 - slip)
        if signal.side == "long" and signal.stop_price >= entry:
            continue
        if signal.side == "short" and signal.stop_price <= entry:
            continue
        risk = abs(entry - signal.stop_price)
        tp = entry + risk * config.take_profit_r if signal.side == "long" else entry - risk * config.take_profit_r
        liq = liquidation_price(entry, config.leverage, signal.side)
        notional = equity * config.margin_pct * config.leverage
        margin_used = equity * config.margin_pct
        entry_fee = notional * config.fee_rate
        equity_before = equity

        for idx in range(signal.entry_idx, len(candles_15m)):
            candle = candles_15m[idx]
            if candle.ts >= end_ts:
                exit_price = candle.open * (1 - slip) if signal.side == "long" else candle.open * (1 + slip)
                reason = "区间结束"
            elif max_hold_ms is not None and candle.ts >= entry_candle.ts + max_hold_ms:
                exit_price = candle.open * (1 - slip) if signal.side == "long" else candle.open * (1 + slip)
                reason = "时间退出"
            elif signal.side == "long" and candle.low <= liq:
                exit_price = liq
                reason = "强平"
            elif signal.side == "short" and candle.high >= liq:
                exit_price = liq
                reason = "强平"
            elif signal.side == "long" and candle.low <= signal.stop_price:
                exit_price = signal.stop_price * (1 - slip)
                reason = "止损"
            elif signal.side == "short" and candle.high >= signal.stop_price:
                exit_price = signal.stop_price * (1 + slip)
                reason = "止损"
            elif signal.side == "long" and candle.high >= tp:
                exit_price = tp * (1 - slip)
                reason = "止盈"
            elif signal.side == "short" and candle.low <= tp:
                exit_price = tp * (1 + slip)
                reason = "止盈"
            elif idx == len(candles_15m) - 1:
                exit_price = candle.close * (1 - slip) if signal.side == "long" else candle.close * (1 + slip)
                reason = "数据结束"
            else:
                continue

            if reason == "强平":
                net = -margin_used - entry_fee
            else:
                move = (exit_price - entry) / entry if signal.side == "long" else (entry - exit_price) / entry
                gross = notional * move
                exit_fee = notional * (exit_price / entry) * config.fee_rate
                net = gross - entry_fee - exit_fee
            equity += net
            trades.append(
                Trade(
                    family=signal.family,
                    name=config.name,
                    side=signal.side,
                    entry_ts=entry_candle.ts,
                    exit_ts=candle.ts,
                    entry_price=entry,
                    exit_price=exit_price,
                    stop_price=signal.stop_price,
                    take_profit_price=tp,
                    liquidation_price=liq,
                    notional=notional,
                    margin_used=margin_used,
                    net_pnl=net,
                    return_pct=net / equity_before * 100 if equity_before else 0.0,
                    exit_reason=reason,
                    equity_before=equity_before,
                    equity_after=equity,
                    hold_hours=(candle.ts - entry_candle.ts) / 3_600_000,
                    note=signal.note,
                )
            )
            next_allowed = idx + 1
            break
        if equity <= 0:
            break
    return trades


def summarize(trades: list[Trade], initial_equity: float) -> dict[str, float | int]:
    equity = initial_equity
    peak = initial_equity
    wins: list[float] = []
    losses: list[float] = []
    max_dd = 0.0
    loss_streak = 0
    max_loss_streak = 0
    exit_counts: dict[str, int] = {}
    for trade in trades:
        equity += trade.net_pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100 if peak else 0.0)
        exit_counts[trade.exit_reason] = exit_counts.get(trade.exit_reason, 0) + 1
        if trade.net_pnl > 0:
            wins.append(trade.net_pnl)
            loss_streak = 0
        else:
            losses.append(trade.net_pnl)
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
        "return_pct": (equity / initial_equity - 1) * 100 if initial_equity else 0.0,
        "final_equity": equity,
        "max_dd": max_dd,
        "pf": gross_profit / gross_loss if gross_loss else math.inf if gross_profit else 0.0,
        "avg_trade": statistics.mean([t.net_pnl for t in trades]) if trades else 0.0,
        "avg_win": statistics.mean(wins) if wins else 0.0,
        "avg_loss": statistics.mean(losses) if losses else 0.0,
        "best": max((t.net_pnl for t in trades), default=0.0),
        "worst": min((t.net_pnl for t in trades), default=0.0),
        "avg_hold": statistics.mean([t.hold_hours for t in trades]) if trades else 0.0,
        "max_loss_streak": max_loss_streak,
        "liq": exit_counts.get("强平", 0),
        "tp": exit_counts.get("止盈", 0),
        "sl": exit_counts.get("止损", 0),
        "time": exit_counts.get("时间退出", 0),
    }


def split_periods(start_ts: int, end_ts: int) -> tuple[int, int, int, int]:
    split = start_ts + (end_ts - start_ts) * 2 // 3
    return start_ts, split, split, end_ts


def scan_configs(candles_15m: list[Candle], signals: list[Signal], start_ts: int, end_ts: int, initial_equity: float) -> list[dict]:
    train_start, train_end, test_start, test_end = split_periods(start_ts, end_ts)
    rows = []
    families = sorted({s.family for s in signals})
    for family in families:
        for leverage in [10, 15, 20, 30, 50, 75, 100]:
            for margin_pct in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
                for tp_r in [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
                    for max_hold in [None, 4.0, 8.0, 12.0, 24.0]:
                        name = f"{family}_{leverage:g}x_{margin_pct*100:.0f}%_{tp_r:g}R_{'不限时' if max_hold is None else str(int(max_hold))+'H'}"
                        config = Config(family=family, name=name, initial_equity=initial_equity, leverage=leverage, margin_pct=margin_pct, take_profit_r=tp_r, max_hold_hours=max_hold)
                        full_trades = simulate(candles_15m, signals, config, start_ts, end_ts)
                        if len(full_trades) < 8:
                            continue
                        train_trades = simulate(candles_15m, signals, config, train_start, train_end)
                        test_trades = simulate(candles_15m, signals, config, test_start, test_end)
                        full = summarize(full_trades, initial_equity)
                        train = summarize(train_trades, initial_equity)
                        test = summarize(test_trades, initial_equity)
                        rows.append({"config": config, "full": full, "train": train, "test": test})
    return rows


def score_row(row: dict) -> float:
    full = row["full"]
    test = row["test"]
    avg_win_penalty = abs(full["avg_win"] - 75) / 75
    return (
        full["return_pct"]
        + test["return_pct"] * 1.5
        + min(full["pf"], 5) * 5
        - full["max_dd"] * 1.2
        - test["max_dd"] * 1.0
        - avg_win_penalty * 20
        - full["liq"] * 50
    )


def export_scan(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "策略",
                "杠杆",
                "保证金比例",
                "止盈R",
                "最长持仓小时",
                "交易",
                "胜率",
                "收益",
                "最大回撤",
                "盈亏比",
                "平均盈利U",
                "平均亏损U",
                "最大连亏",
                "强平",
                "测试段交易",
                "测试段收益",
                "测试段回撤",
                "测试段平均盈利U",
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
                    f"{c.margin_pct:.4f}",
                    c.take_profit_r,
                    "" if c.max_hold_hours is None else c.max_hold_hours,
                    f["trades"],
                    f"{f['win_rate']:.2f}",
                    f"{f['return_pct']:.2f}",
                    f"{f['max_dd']:.2f}",
                    f"{f['pf']:.2f}",
                    f"{f['avg_win']:.2f}",
                    f"{f['avg_loss']:.2f}",
                    f["max_loss_streak"],
                    f["liq"],
                    t["trades"],
                    f"{t['return_pct']:.2f}",
                    f"{t['max_dd']:.2f}",
                    f"{t['avg_win']:.2f}",
                ]
            )


def export_trades(path: Path, trades: list[Trade]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["策略", "方向", "入场时间", "出场时间", "入场", "出场", "止损", "止盈", "强平", "名义仓位", "保证金", "净盈亏U", "出场原因", "权益", "持仓小时", "备注"])
        for t in trades:
            writer.writerow(
                [
                    t.name,
                    "做多" if t.side == "long" else "做空",
                    format_ts(t.entry_ts),
                    format_ts(t.exit_ts),
                    f"{t.entry_price:.2f}",
                    f"{t.exit_price:.2f}",
                    f"{t.stop_price:.2f}",
                    f"{t.take_profit_price:.2f}",
                    f"{t.liquidation_price:.2f}",
                    f"{t.notional:.2f}",
                    f"{t.margin_used:.2f}",
                    f"{t.net_pnl:.2f}",
                    t.exit_reason,
                    f"{t.equity_after:.2f}",
                    f"{t.hold_hours:.2f}",
                    t.note,
                ]
            )


def print_top(rows: list[dict], limit: int = 12) -> None:
    print("满足单笔盈利目标的候选策略")
    print("筛选：平均盈利 50-100U，全年收益为正，测试段收益为正，强平=0")
    print()
    print("策略 | 交易 | 胜率 | 收益 | 回撤 | PF | 平均盈利 | 平均亏损 | 测试收益 | 测试交易")
    print("-" * 120)
    for row in rows[:limit]:
        f = row["full"]
        t = row["test"]
        print(
            f"{row['config'].name} | {f['trades']} | {f['win_rate']:.1f}% | {f['return_pct']:.1f}% | "
            f"{f['max_dd']:.1f}% | {f['pf']:.2f} | {f['avg_win']:.1f}U | {f['avg_loss']:.1f}U | "
            f"{t['return_pct']:.1f}% | {t['trades']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="扫描单笔盈利 50-100U 的 BTC-USDT-SWAP 策略")
    parser.add_argument("--inst-id", default="BTC-USDT-SWAP")
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--export-dir", type=Path, default=Path("data"))
    parser.add_argument("--initial-equity", type=float, default=1000.0)
    parser.add_argument("--days", type=int, default=365)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    args = build_parser().parse_args(argv)
    try:
        candles_15m = load_candles_csv(latest_cache(args.cache_dir, f"okx_{args.inst_id}_15m_*.csv"))
        candles_4h = load_candles_csv(latest_cache(args.cache_dir, f"okx_{args.inst_id}_4H_*.csv"))
        end_ts = candles_15m[-1].ts + BAR_MS["15m"]
        start_ts = end_ts - args.days * 24 * 60 * 60 * 1000
        signals = top_bottom_signals(candles_15m, candles_4h)
        signals += trend_breakout_signals(candles_15m, candles_4h)
        signals += trend_pullback_signals(candles_15m, candles_4h)
        signals += h4_momentum_retest_signals(candles_15m, candles_4h)

        rows = scan_configs(candles_15m, signals, start_ts, end_ts, args.initial_equity)
        rows.sort(key=score_row, reverse=True)
        export_scan(args.export_dir / "strategy_50_100_scan.csv", rows)

        candidates = [
            row
            for row in rows
            if 50 <= row["full"]["avg_win"] <= 100
            and row["full"]["return_pct"] > 0
            and row["test"]["return_pct"] > 0
            and row["full"]["liq"] == 0
            and row["test"]["trades"] >= 3
            and row["full"]["trades"] >= 10
        ]
        candidates.sort(key=score_row, reverse=True)
        print(f"数据：{args.inst_id}，区间 {format_ts(start_ts)} -> {format_ts(end_ts)}")
        print(f"信号数量：{len(signals)}，扫描组合：{len(rows)}")
        print_top(candidates)
        if candidates:
            best = candidates[0]
            best_trades = simulate(candles_15m, signals, best["config"], start_ts, end_ts)
            export_trades(args.export_dir / "strategy_50_100_best_trades.csv", best_trades)
            print()
            print(f"最佳策略逐笔交易：{args.export_dir / 'strategy_50_100_best_trades.csv'}")
        else:
            print()
            print("没有找到同时满足 50-100U平均盈利、全年为正、测试段为正、无强平的候选。")
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
