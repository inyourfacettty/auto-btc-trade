#!/usr/bin/env python3
"""Scheduled GitHub Actions email report for the BTC 4H strategy."""

from __future__ import annotations

import argparse
import os
import smtplib
import ssl
import sys
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any

from backtest_okx_btc import BAR_MS, NetworkConfig, format_ts
from trend_breakout_opportunity_scanner import (
    StrategyParams,
    build_entry_plan,
    build_realtime_and_confirmed,
    fetch_latest_okx_candles,
    signal_label,
)


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def local_now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def direction_text(side: str | None) -> str:
    if side == "long":
        return "做多"
    if side == "short":
        return "做空"
    return "无信号"


def fmt(value: float | int | None, digits: int = 2) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def candle_payload(candle: Any) -> dict[str, Any]:
    return {
        "ts": candle.ts,
        "time": format_ts(candle.ts),
        "close_time": format_ts(candle.ts + BAR_MS["4H"]),
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
    }


def opportunity_payload(opportunity: Any) -> dict[str, Any]:
    return {
        "label": signal_label(opportunity),
        "is_closed": opportunity.is_closed,
        "candle": candle_payload(opportunity.candle),
        "upper": opportunity.upper,
        "lower": opportunity.lower,
        "ema": opportunity.ema,
        "adx": opportunity.adx,
        "atr": opportunity.atr,
        "atr_pct": opportunity.atr_pct,
        "long_score": opportunity.long_score,
        "short_score": opportunity.short_score,
        "long_signal": opportunity.long_signal,
        "short_signal": opportunity.short_signal,
        "long_reasons": opportunity.long_reasons,
        "short_reasons": opportunity.short_reasons,
    }


def plan_payload(plan: Any) -> dict[str, Any]:
    return {
        "side": plan.side,
        "side_text": direction_text(plan.side),
        "status": plan.status,
        "entry_price": plan.entry_price,
        "stop_price": plan.stop_price,
        "take_profit_price": plan.take_profit_price,
        "risk": plan.risk,
        "margin": plan.margin,
        "notional": plan.notional,
        "qty": plan.qty,
        "signal_age_minutes": plan.signal_age_minutes,
    }


def build_status_payload(
    inst_id: str,
    initial_equity: float,
    leverage: float,
    margin_pct: float,
    proxy_url: str,
    proxy_mode: str,
) -> dict[str, Any]:
    params = StrategyParams()
    network = NetworkConfig(proxy_url=proxy_url, proxy_mode=proxy_mode)
    current_ms = now_ms()
    candles_4h = fetch_latest_okx_candles(inst_id, "4H", 300, network)
    candles_15m = fetch_latest_okx_candles(inst_id, "15m", 80, network)
    realtime, confirmed = build_realtime_and_confirmed(candles_4h, current_ms, params)
    plan = build_entry_plan(
        confirmed,
        candles_15m,
        current_ms,
        params,
        initial_equity,
        leverage,
        margin_pct,
    )
    return {
        "ok": True,
        "updated_at": local_now_text(),
        "inst_id": inst_id,
        "settings": {
            "initial_equity": initial_equity,
            "leverage": leverage,
            "margin_pct": margin_pct,
            "max_hold_hours": params.max_hold_hours,
        },
        "realtime": opportunity_payload(realtime),
        "confirmed": opportunity_payload(confirmed),
        "plan": plan_payload(plan),
    }


def build_operation_summary(plan: dict[str, Any]) -> str:
    side = plan.get("side")
    lines: list[str]
    if side is None:
        lines = [
            "操作建议：不操作，继续等待收盘确认信号。",
            f"当前状态：{plan.get('status', '--')}",
        ]
    else:
        lines = [
            f"操作建议：{direction_text(side)}",
            f"当前状态：{plan.get('status', '--')}",
            f"入场参考：{fmt(plan.get('entry_price'))}",
            f"初始止损：{fmt(plan.get('stop_price'))}",
            f"3R止盈：{fmt(plan.get('take_profit_price'))}",
            f"单笔风险距离：{fmt(plan.get('risk'))}",
            f"计划数量：{fmt(plan.get('qty'), 6)} BTC",
        ]
    lines.extend(
        [
            f"计划保证金：{fmt(plan.get('margin'))}U",
            f"名义仓位：{fmt(plan.get('notional'))}U",
        ]
    )
    return "\n".join(lines)


def render_reasons(title: str, reasons: list[str]) -> str:
    if not reasons:
        return f"{title}：--"
    return "\n".join([f"{title}："] + [f"- {item}" for item in reasons])


def build_email_subject(payload: dict[str, Any]) -> str:
    confirmed = payload.get("confirmed", {})
    plan = payload.get("plan", {})
    return (
        f"{payload.get('inst_id', 'BTC-USDT-SWAP')} 策略状态："
        f"{confirmed.get('label', '--')} / {direction_text(plan.get('side'))} / {payload.get('updated_at', '--')}"
    )


def build_email_body(payload: dict[str, Any]) -> str:
    if not payload.get("ok", True):
        return (
            "BTC-USDT-SWAP 4H策略状态\n\n"
            f"更新时间：{payload.get('updated_at', '--')}\n"
            f"扫描失败：{payload.get('error', '--')}\n\n"
            "本邮件只做策略提醒，不会自动下单。"
        )
    confirmed = payload["confirmed"]
    realtime = payload["realtime"]
    plan = payload["plan"]
    candle = confirmed["candle"]
    settings = payload.get("settings", {})
    return "\n".join(
        [
            f"{payload.get('inst_id', 'BTC-USDT-SWAP')} 4H策略状态",
            f"更新时间：{payload.get('updated_at', '--')}",
            "",
            f"实时预警：{realtime.get('label', '--')}",
            f"收盘确认：{confirmed.get('label', '--')}",
            f"4H时间：{candle.get('time', '--')}  收盘价：{fmt(candle.get('close'))}",
            f"唐奇安40：下轨 {fmt(confirmed.get('lower'))} / 上轨 {fmt(confirmed.get('upper'))}",
            f"EMA180：{fmt(confirmed.get('ema'))}",
            f"ADX14：{fmt(confirmed.get('adx'))}",
            f"ATR14：{fmt(confirmed.get('atr'))} ({fmt(confirmed.get('atr_pct'))}%)",
            f"多头机会分：{confirmed.get('long_score', '--')}",
            f"空头机会分：{confirmed.get('short_score', '--')}",
            "",
            build_operation_summary(plan),
            "",
            render_reasons("多头证据", confirmed.get("long_reasons", [])),
            "",
            render_reasons("空头证据", confirmed.get("short_reasons", [])),
            "",
            f"参数：本金 {fmt(settings.get('initial_equity'))}U，杠杆 {fmt(settings.get('leverage'), 0)}x，"
            f"保证金比例 {fmt((settings.get('margin_pct') or 0) * 100, 0)}%，最长持仓 {settings.get('max_hold_hours', '--')}小时",
            "",
            "本邮件只做策略提醒，不会自动下单。",
        ]
    )


def env_required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"缺少 GitHub Secret 或环境变量：{name}")
    return value


def send_email(subject: str, body: str) -> None:
    smtp_host = env_required("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "465"))
    smtp_username = env_required("SMTP_USERNAME")
    smtp_password = env_required("SMTP_PASSWORD")
    mail_to = env_required("MAIL_TO")
    mail_from = os.environ.get("MAIL_FROM", smtp_username).strip() or smtp_username

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = mail_from
    message["To"] = mail_to
    message.set_content(body)

    if smtp_port == 465:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context, timeout=30) as server:
            server.login(smtp_username, smtp_password)
            server.send_message(message)
    else:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(smtp_username, smtp_password)
            server.send_message(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="发送 BTC 4H 策略状态邮件")
    parser.add_argument("--inst-id", default=os.environ.get("INST_ID", "BTC-USDT-SWAP"))
    parser.add_argument("--initial-equity", type=float, default=float(os.environ.get("INITIAL_EQUITY", "1000")))
    parser.add_argument("--leverage", type=float, default=float(os.environ.get("LEVERAGE", "8")))
    parser.add_argument("--margin-pct", type=float, default=float(os.environ.get("MARGIN_PCT", "0.10")))
    parser.add_argument("--proxy-mode", choices=["fallback", "on", "off"], default=os.environ.get("PROXY_MODE", "off"))
    parser.add_argument("--proxy-url", default=os.environ.get("PROXY_URL", ""))
    parser.add_argument("--send", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    args = build_parser().parse_args(argv)
    try:
        payload = build_status_payload(
            inst_id=args.inst_id,
            initial_equity=args.initial_equity,
            leverage=args.leverage,
            margin_pct=args.margin_pct,
            proxy_url=args.proxy_url,
            proxy_mode=args.proxy_mode,
        )
    except Exception as exc:
        payload = {
            "ok": False,
            "updated_at": local_now_text(),
            "inst_id": args.inst_id,
            "error": str(exc),
        }
    subject = build_email_subject(payload)
    body = build_email_body(payload)
    print(subject)
    print()
    print(body)
    if args.send:
        send_email(subject, body)
        print("\n邮件已发送。")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
