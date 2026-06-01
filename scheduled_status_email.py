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
from html import escape
from typing import Any
from zoneinfo import ZoneInfo

from backtest_okx_btc import BAR_MS, NetworkConfig, format_ts
from trend_breakout_opportunity_scanner import (
    StrategyParams,
    build_entry_plan,
    build_realtime_and_confirmed,
    fetch_latest_okx_candles,
    signal_label,
)

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def format_time_in_shanghai(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(SHANGHAI_TZ).strftime("%Y-%m-%d %H:%M:%S")


def local_now_text() -> str:
    return format_time_in_shanghai(datetime.now(timezone.utc))


def direction_text(side: str | None) -> str:
    if side == "long":
        return "做多"
    if side == "short":
        return "做空"
    return "无信号"


def fmt(value: float | int | None, digits: int = 2) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def fmt_pct(value: float | int | None) -> str:
    return "--" if value is None else f"{float(value):.2f}%"


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


def html_td(value: Any, *, strong: bool = False, color: str | None = None) -> str:
    text = escape("--" if value is None else str(value))
    style = f' style="color:{color};font-weight:700;"' if color else ""
    if strong:
        text = f"<strong>{text}</strong>"
    return f"<td{style}>{text}</td>"


def html_th(value: str) -> str:
    return f"<th>{escape(value)}</th>"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    header_html = "".join(html_th(header) for header in headers)
    row_html = "\n".join("<tr>" + "".join(html_td(cell) for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{row_html}</tbody></table>"


def opportunity_column(opportunity: dict[str, Any], fallback_label: str) -> dict[str, str]:
    candle = opportunity.get("candle", {})
    status = "已收盘" if opportunity.get("is_closed") else "未收盘"
    return {
        "title": fallback_label,
        "status": status,
        "label": str(opportunity.get("label", "--")),
        "time": str(candle.get("time", "--")),
        "open": fmt(candle.get("open")),
        "high": fmt(candle.get("high")),
        "low": fmt(candle.get("low")),
        "close": fmt(candle.get("close")),
        "donchian": f"{fmt(opportunity.get('lower'))} / {fmt(opportunity.get('upper'))}",
        "ema": fmt(opportunity.get("ema")),
        "adx": fmt(opportunity.get("adx")),
        "atr": f"{fmt(opportunity.get('atr'))} ({fmt_pct(opportunity.get('atr_pct'))})",
        "long_score": str(opportunity.get("long_score", "--")),
        "short_score": str(opportunity.get("short_score", "--")),
    }


def build_kline_table(confirmed: dict[str, Any], realtime: dict[str, Any]) -> str:
    previous = opportunity_column(confirmed, "上个已收盘4H")
    current = opportunity_column(realtime, "当前未收盘4H")
    rows = [
        ["K线时间", previous["time"], current["time"]],
        ["状态", previous["status"], current["status"]],
        ["信号标签", previous["label"], current["label"]],
        ["开盘价", previous["open"], current["open"]],
        ["最高价", previous["high"], current["high"]],
        ["最低价", previous["low"], current["low"]],
        ["收盘/当前价", previous["close"], current["close"]],
        ["唐奇安40 下轨/上轨", previous["donchian"], current["donchian"]],
        ["EMA180", previous["ema"], current["ema"]],
        ["ADX14", previous["adx"], current["adx"]],
        ["ATR14", previous["atr"], current["atr"]],
        ["多头机会分", previous["long_score"], current["long_score"]],
        ["空头机会分", previous["short_score"], current["short_score"]],
    ]
    return table(["项目", previous["title"], current["title"]], rows)


def build_plan_table(plan: dict[str, Any]) -> str:
    rows = [
        ["操作建议", "不操作，继续等待收盘确认信号" if plan.get("side") is None else direction_text(plan.get("side"))],
        ["当前状态", plan.get("status", "--")],
        ["入场参考", fmt(plan.get("entry_price"))],
        ["初始止损", fmt(plan.get("stop_price"))],
        ["3R止盈", fmt(plan.get("take_profit_price"))],
        ["单笔风险距离", fmt(plan.get("risk"))],
        ["计划保证金", f"{fmt(plan.get('margin'))}U"],
        ["名义仓位", f"{fmt(plan.get('notional'))}U"],
        ["计划数量", f"{fmt(plan.get('qty'), 6)} BTC"],
    ]
    return table(["项目", "值"], rows)


def reasons_html(title: str, reasons: list[str]) -> str:
    if not reasons:
        return f"<h3>{escape(title)}</h3><p>--</p>"
    items = "".join(f"<li>{escape(item)}</li>" for item in reasons)
    return f"<h3>{escape(title)}</h3><ul>{items}</ul>"


def build_email_html(payload: dict[str, Any]) -> str:
    if not payload.get("ok", True):
        return f"""<!doctype html>
<html><body>
  <h2>BTC-USDT-SWAP 4H策略状态</h2>
  <p><strong>更新时间（北京时间）：</strong>{escape(str(payload.get('updated_at', '--')))}</p>
  <p><strong>扫描失败：</strong>{escape(str(payload.get('error', '--')))}</p>
  <p>本邮件只做策略提醒，不会自动下单。</p>
</body></html>"""

    confirmed = payload["confirmed"]
    realtime = payload["realtime"]
    plan = payload["plan"]
    settings = payload.get("settings", {})
    summary_rows = [
        ["品种", payload.get("inst_id", "BTC-USDT-SWAP")],
        ["更新时间（北京时间）", payload.get("updated_at", "--")],
        ["实时预警", realtime.get("label", "--")],
        ["收盘确认", confirmed.get("label", "--")],
        ["操作方向", direction_text(plan.get("side"))],
        [
            "参数",
            f"本金 {fmt(settings.get('initial_equity'))}U / 杠杆 {fmt(settings.get('leverage'), 0)}x / "
            f"保证金 {fmt((settings.get('margin_pct') or 0) * 100, 0)}% / 最长持仓 {settings.get('max_hold_hours', '--')}小时",
        ],
    ]
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ margin:0; padding:24px; background:#f4f6f5; color:#1f2925; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif; }}
    .wrap {{ max-width:920px; margin:0 auto; background:#ffffff; border:1px solid #d8ddd8; border-radius:8px; padding:20px; }}
    h2 {{ margin:0 0 12px; font-size:20px; }}
    h3 {{ margin:22px 0 8px; font-size:15px; }}
    table {{ width:100%; border-collapse:collapse; margin:10px 0 18px; font-size:13px; }}
    th {{ background:#eef1ed; color:#33423b; text-align:left; padding:9px 10px; border:1px solid #d8ddd8; }}
    td {{ padding:9px 10px; border:1px solid #e5e9e5; vertical-align:top; }}
    tr:nth-child(even) td {{ background:#fafbf9; }}
    .notice {{ margin-top:18px; padding:12px; background:#fff7e8; border:1px solid #efd3a3; color:#6f4612; border-radius:6px; }}
    ul {{ margin-top:8px; padding-left:20px; line-height:1.7; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2>{escape(str(payload.get('inst_id', 'BTC-USDT-SWAP')))} 4H策略状态</h2>
    {table(["摘要", "值"], summary_rows)}
    <h3>操作计划</h3>
    {build_plan_table(plan)}
    <h3>4H K线对比</h3>
    {build_kline_table(confirmed, realtime)}
    {reasons_html("上个已收盘4H：多头证据", confirmed.get("long_reasons", []))}
    {reasons_html("上个已收盘4H：空头证据", confirmed.get("short_reasons", []))}
    <div class="notice">本邮件只做策略提醒，不会自动下单。交易以“收盘确认”为准，当前未收盘4H只用于盯盘。</div>
  </div>
</body>
</html>"""


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
            f"更新时间（北京时间）：{payload.get('updated_at', '--')}\n"
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
            f"更新时间（北京时间）：{payload.get('updated_at', '--')}",
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


def send_email(subject: str, body: str, html_body: str | None = None) -> None:
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
    if html_body:
        message.add_alternative(html_body, subtype="html")

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
    html_body = build_email_html(payload)
    print(subject)
    print()
    print(body)
    if args.send:
        send_email(subject, body, html_body)
        print("\n邮件已发送。")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
