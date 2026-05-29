#!/usr/bin/env python3
"""Local dry-run web UI for the BTC-USDT-SWAP 4H breakout strategy."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from backtest_okx_btc import BAR_MS, NetworkConfig, format_ts
from trend_breakout_opportunity_scanner import (
    EntryPlan,
    Opportunity,
    StrategyParams,
    build_entry_plan,
    build_realtime_and_confirmed,
    fetch_latest_okx_candles,
    signal_label,
)
from okx_client import OkxRestClient, build_order_preview, load_okx_settings


STATE_VERSION = 2


@dataclass(frozen=True)
class AppConfig:
    inst_id: str
    host: str
    port: int
    refresh_seconds: int
    proxy_url: str
    proxy_mode: str
    initial_equity: float
    leverage: float
    margin_pct: float
    state_path: Path
    env_path: Path


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def side_text(side: str | None) -> str:
    if side == "long":
        return "做多"
    if side == "short":
        return "做空"
    return "无信号"


def fmt(value: float | None, digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


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


def opportunity_payload(opportunity: Opportunity) -> dict[str, Any]:
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


def plan_payload(plan: EntryPlan) -> dict[str, Any]:
    return {
        "side": plan.side,
        "side_text": side_text(plan.side),
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


class LocalAutoTraderApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.params = StrategyParams()
        self.network = NetworkConfig(proxy_url=config.proxy_url, proxy_mode=config.proxy_mode)
        self.lock = threading.Lock()
        self.cached_status: dict[str, Any] | None = None
        self.cached_at = 0.0
        self.okx_settings = load_okx_settings(config.env_path)
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        state: dict[str, Any]
        if self.config.state_path.exists():
            try:
                state = json.loads(self.config.state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                state = {}
        else:
            state = {}
        state.setdefault("version", STATE_VERSION)
        state.setdefault("dry_runs", [])
        state.setdefault("demo_orders", [])
        state.setdefault("logs", [])
        settings = state.setdefault("settings", {})
        settings.setdefault("mode", "dry-run")
        settings.setdefault("real_orders_enabled", False)
        settings.setdefault("simulated_okx_enabled", False)
        return state

    def _save_state(self) -> None:
        self.config.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.state_path.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def append_log(self, level: str, message: str) -> None:
        logs = self.state.setdefault("logs", [])
        logs.insert(0, {"time": now_text(), "level": level, "message": message})
        del logs[200:]
        self._save_state()

    def reload_okx_settings(self) -> Any:
        self.okx_settings = load_okx_settings(self.config.env_path)
        return self.okx_settings

    def okx_status_payload(self) -> dict[str, Any]:
        settings = self.reload_okx_settings()
        return {
            "enabled": settings.enabled,
            "demo": settings.demo,
            "allow_demo_orders": settings.allow_demo_orders,
            "pos_side": settings.pos_side,
            "message": settings.message,
            "env_path": str(self.config.env_path),
        }

    def okx_client(self) -> OkxRestClient:
        settings = self.reload_okx_settings()
        if not settings.enabled or settings.credentials is None:
            raise RuntimeError(settings.message)
        return OkxRestClient(settings.credentials, proxy_url=self.config.proxy_url, proxy_mode=self.config.proxy_mode)

    def test_okx_connection(self) -> tuple[int, dict[str, Any]]:
        settings = self.reload_okx_settings()
        if not settings.enabled:
            self.append_log("提示", f"OKX 模拟盘未就绪：{settings.message}")
            return HTTPStatus.CONFLICT, {"ok": False, "message": settings.message, "okx": self.okx_status_payload()}
        try:
            client = self.okx_client()
            account_config = client.account_config()
            balance = client.account_balance()
            positions = client.positions(self.config.inst_id)
            rules = client.instrument_rules(self.config.inst_id)
            for name, payload in (
                ("账户配置", account_config),
                ("账户余额", balance),
                ("持仓", positions),
            ):
                if payload.get("code") != "0":
                    raise RuntimeError(f"{name}读取失败：{payload.get('msg') or payload}")
            payload = {
                "ok": True,
                "message": "OKX 模拟盘连接正常",
                "account_config_rows": len(account_config.get("data", [])),
                "balance_rows": len(balance.get("data", [])),
                "positions": len(positions.get("data", [])),
                "instrument_rules": {
                    "inst_id": rules.inst_id,
                    "ct_val": rules.ct_val,
                    "lot_sz": rules.lot_sz,
                    "min_sz": rules.min_sz,
                    "tick_sz": rules.tick_sz,
                },
            }
            self.append_log("OKX", "模拟盘连接测试通过")
            return HTTPStatus.OK, payload
        except Exception as exc:
            self.append_log("错误", f"OKX 模拟盘连接失败：{exc}")
            return HTTPStatus.BAD_GATEWAY, {"ok": False, "message": str(exc), "okx": self.okx_status_payload()}

    def status(self, force: bool = False) -> dict[str, Any]:
        with self.lock:
            if not force and self.cached_status is not None and time.time() - self.cached_at < self.config.refresh_seconds:
                return self.cached_status

        try:
            current_ms = now_ms()
            candles_4h = fetch_latest_okx_candles(self.config.inst_id, "4H", 300, self.network)
            candles_15m = fetch_latest_okx_candles(self.config.inst_id, "15m", 80, self.network)
            realtime, confirmed = build_realtime_and_confirmed(candles_4h, current_ms, self.params)
            plan = build_entry_plan(
                confirmed,
                candles_15m,
                current_ms,
                self.params,
                self.config.initial_equity,
                self.config.leverage,
                self.config.margin_pct,
            )
            payload = {
                "ok": True,
                "updated_at": now_text(),
                "mode": "dry-run",
                "real_orders_enabled": False,
                "inst_id": self.config.inst_id,
                "okx": self.okx_status_payload(),
                "settings": {
                    "initial_equity": self.config.initial_equity,
                    "leverage": self.config.leverage,
                    "margin_pct": self.config.margin_pct,
                    "refresh_seconds": self.config.refresh_seconds,
                    "proxy_mode": self.config.proxy_mode,
                    "proxy_url": self.config.proxy_url,
                    "env_path": str(self.config.env_path),
                    "max_hold_hours": self.params.max_hold_hours,
                },
                "realtime": opportunity_payload(realtime),
                "confirmed": opportunity_payload(confirmed),
                "plan": plan_payload(plan),
                "records": self.state.get("dry_runs", [])[:50],
                "demo_orders": self.state.get("demo_orders", [])[:50],
                "logs": self.state.get("logs", [])[:80],
                "risk": {
                    "one_position_only": True,
                    "requires_confirmed_signal": True,
                    "server_side_orders": self.okx_settings.allow_demo_orders,
                    "dry_run_only": not self.okx_settings.allow_demo_orders,
                    "live_orders_enabled": False,
                },
            }
            with self.lock:
                self.cached_status = payload
                self.cached_at = time.time()
            return payload
        except Exception as exc:
            self.append_log("错误", f"行情扫描失败：{exc}")
            return {
                "ok": False,
                "updated_at": now_text(),
                "mode": "dry-run",
                "real_orders_enabled": False,
                "inst_id": self.config.inst_id,
                "okx": self.okx_status_payload(),
                "error": str(exc),
                "records": self.state.get("dry_runs", [])[:50],
                "demo_orders": self.state.get("demo_orders", [])[:50],
                "logs": self.state.get("logs", [])[:80],
            }

    def record_dry_run(self) -> tuple[int, dict[str, Any]]:
        status = self.status(force=True)
        if not status.get("ok"):
            return HTTPStatus.BAD_GATEWAY, {"ok": False, "message": status.get("error", "扫描失败")}
        plan = status["plan"]
        confirmed = status["confirmed"]
        if plan["side"] is None:
            self.append_log("提示", "收盘确认没有交易信号，未生成 dry-run 计划")
            return HTTPStatus.CONFLICT, {"ok": False, "message": "收盘确认没有交易信号，不能记录拟下单"}

        signal_key = f"{confirmed['candle']['ts']}:{plan['side']}"
        records = self.state.setdefault("dry_runs", [])
        if any(record.get("signal_key") == signal_key for record in records):
            self.append_log("提示", f"同一4H信号已记录过 dry-run：{signal_key}")
            return HTTPStatus.CONFLICT, {"ok": False, "message": "同一4H信号已记录过，避免重复开仓"}

        record = {
            "id": f"DRY-{int(time.time())}",
            "signal_key": signal_key,
            "created_at": now_text(),
            "inst_id": self.config.inst_id,
            "mode": "dry-run",
            "side": plan["side"],
            "side_text": plan["side_text"],
            "entry_price": plan["entry_price"],
            "stop_price": plan["stop_price"],
            "take_profit_price": plan["take_profit_price"],
            "risk": plan["risk"],
            "margin": plan["margin"],
            "notional": plan["notional"],
            "qty": plan["qty"],
            "status": "已记录拟下单",
            "signal_time": confirmed["candle"]["time"],
        }
        records.insert(0, record)
        del records[100:]
        self.append_log("dry-run", f"记录拟下单：{record['side_text']} 入场 {fmt(record['entry_price'])}")
        self.cached_status = None
        return HTTPStatus.OK, {"ok": True, "record": record}

    def place_demo_order(self) -> tuple[int, dict[str, Any]]:
        settings = self.reload_okx_settings()
        if not settings.enabled:
            self.append_log("提示", f"OKX 模拟盘未就绪：{settings.message}")
            return HTTPStatus.CONFLICT, {"ok": False, "message": settings.message, "okx": self.okx_status_payload()}
        if not settings.allow_demo_orders:
            message = "模拟盘下单安全开关未打开：请在 .env 设置 OKX_ENABLE_DEMO_ORDER=true"
            self.append_log("提示", message)
            return HTTPStatus.CONFLICT, {"ok": False, "message": message, "okx": self.okx_status_payload()}

        status = self.status(force=True)
        if not status.get("ok"):
            return HTTPStatus.BAD_GATEWAY, {"ok": False, "message": status.get("error", "扫描失败")}
        plan = status["plan"]
        confirmed = status["confirmed"]
        if plan["side"] is None:
            self.append_log("提示", "收盘确认没有交易信号，未提交 OKX 模拟盘订单")
            return HTTPStatus.CONFLICT, {"ok": False, "message": "收盘确认没有交易信号，不能提交 OKX 模拟盘订单"}

        signal_key = f"{confirmed['candle']['ts']}:{plan['side']}"
        records = self.state.setdefault("demo_orders", [])
        if any(record.get("signal_key") == signal_key for record in records):
            self.append_log("提示", f"同一4H信号已提交过 OKX 模拟盘：{signal_key}")
            return HTTPStatus.CONFLICT, {"ok": False, "message": "同一4H信号已提交过 OKX 模拟盘，避免重复开仓"}

        try:
            client = self.okx_client()
            rules = client.instrument_rules(self.config.inst_id)
            preview = build_order_preview(
                side=plan["side"],
                entry_price=float(plan["entry_price"]),
                stop_price=float(plan["stop_price"]),
                take_profit_price=float(plan["take_profit_price"]),
                notional=float(plan["notional"]),
                rules=rules,
                pos_side=settings.pos_side,
            )
            client_order_id = f"SIM{int(time.time())}{plan['side'][0].upper()}"
            leverage_response = client.set_leverage(
                self.config.inst_id,
                self.config.leverage,
                mgn_mode=preview["tdMode"],
                pos_side=preview["posSide"],
            )
            if leverage_response.get("code") != "0":
                raise RuntimeError(f"设置杠杆失败：{leverage_response.get('msg') or leverage_response}")
            order_response = client.place_market_order_with_tpsl(self.config.inst_id, preview, client_order_id)
            if order_response.get("code") != "0":
                raise RuntimeError(f"提交订单失败：{order_response.get('msg') or order_response}")

            record = {
                "id": client_order_id,
                "signal_key": signal_key,
                "created_at": now_text(),
                "inst_id": self.config.inst_id,
                "mode": "okx-demo",
                "side": plan["side"],
                "side_text": plan["side_text"],
                "entry_price": plan["entry_price"],
                "stop_price": plan["stop_price"],
                "take_profit_price": plan["take_profit_price"],
                "margin": plan["margin"],
                "notional": plan["notional"],
                "status": "已提交 OKX 模拟盘",
                "signal_time": confirmed["candle"]["time"],
                "order_preview": preview,
                "leverage_response": leverage_response,
                "okx_order_response": order_response,
            }
            records.insert(0, record)
            del records[100:]
            self._save_state()
            self.append_log("OKX", f"提交模拟盘订单：{record['side_text']} 合约张数 {preview['sz']}")
            self.cached_status = None
            return HTTPStatus.OK, {"ok": True, "record": record}
        except Exception as exc:
            self.append_log("错误", f"OKX 模拟盘下单失败：{exc}")
            return HTTPStatus.BAD_GATEWAY, {"ok": False, "message": str(exc), "okx": self.okx_status_payload()}


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BTC 本机自动交易测试台</title>
  <style>
    :root {
      --paper: #f3f4f1;
      --panel: #ffffff;
      --ink: #1f2925;
      --muted: #66726c;
      --line: #d8ddd8;
      --green: #0c7a61;
      --red: #b64234;
      --amber: #bd7b1f;
      --black: #111714;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    .shell { max-width: 1440px; margin: 0 auto; padding: 18px; }
    .topbar {
      display: grid;
      grid-template-columns: 1.2fr repeat(5, minmax(120px, .5fr));
      gap: 10px;
      align-items: stretch;
      margin-bottom: 12px;
    }
    .brand, .metric, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      box-shadow: 0 1px 0 rgba(17, 23, 20, .04);
    }
    .brand { padding: 14px 16px; }
    .brand h1 { font-size: 20px; line-height: 1.2; margin: 0 0 6px; font-weight: 760; }
    .brand p, .small { margin: 0; color: var(--muted); font-size: 12px; }
    .metric { padding: 12px; min-height: 72px; }
    .label { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
    .value { font-size: 20px; font-weight: 760; line-height: 1.2; overflow-wrap: anywhere; }
    .value.small-value { font-size: 15px; }
    .grid {
      display: grid;
      grid-template-columns: 1.05fr 1.05fr .9fr .9fr;
      gap: 12px;
      align-items: start;
    }
    .panel { padding: 14px; }
    .panel h2 {
      font-size: 15px;
      margin: 0 0 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--line);
    }
    .signal-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      height: 24px;
      padding: 0 9px;
      border-radius: 999px;
      border: 1px solid var(--line);
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .badge.green { color: var(--green); border-color: rgba(12,122,97,.35); background: rgba(12,122,97,.08); }
    .badge.red { color: var(--red); border-color: rgba(182,66,52,.35); background: rgba(182,66,52,.08); }
    .badge.amber { color: var(--amber); border-color: rgba(189,123,31,.35); background: rgba(189,123,31,.10); }
    .badge.dark { color: var(--black); border-color: #bfc7bf; background: #eef1ed; }
    .scores {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 10px 0;
    }
    .scorebox {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      min-height: 70px;
      background: #fafbf9;
    }
    .scorebox strong { display: block; font-size: 24px; line-height: 1; margin-top: 4px; }
    .kv {
      display: grid;
      grid-template-columns: 128px 1fr;
      gap: 8px;
      padding: 7px 0;
      border-bottom: 1px solid #edf0ed;
      font-size: 13px;
    }
    .kv:last-child { border-bottom: 0; }
    .kv span:first-child { color: var(--muted); }
    .reasons {
      margin: 10px 0 0;
      padding-left: 16px;
      color: #33423b;
      font-size: 12px;
      line-height: 1.65;
    }
    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }
    button {
      border: 1px solid var(--black);
      background: var(--black);
      color: white;
      height: 36px;
      padding: 0 14px;
      border-radius: 6px;
      font-weight: 760;
      cursor: pointer;
    }
    button.secondary { background: white; color: var(--black); border-color: var(--line); }
    button.danger { background: var(--red); border-color: var(--red); }
    button:disabled { opacity: .45; cursor: not-allowed; }
    .wide { margin-top: 12px; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { text-align: left; padding: 9px 8px; border-bottom: 1px solid #edf0ed; vertical-align: top; }
    th { color: var(--muted); font-weight: 700; }
    .logline { display: grid; grid-template-columns: 144px 70px 1fr; gap: 8px; padding: 8px 0; border-bottom: 1px solid #edf0ed; font-size: 12px; }
    .empty { color: var(--muted); font-size: 13px; padding: 12px 0; }
    .status-ok { color: var(--green); }
    .status-bad { color: var(--red); }
    @media (max-width: 1100px) {
      .topbar, .grid, .wide { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="topbar">
      <div class="brand">
        <h1>BTC 本机自动交易测试台</h1>
        <p id="subtitle">加载中</p>
      </div>
      <div class="metric"><div class="label">运行模式</div><div class="value" id="mode">--</div></div>
      <div class="metric"><div class="label">真实下单</div><div class="value" id="realOrders">--</div></div>
      <div class="metric"><div class="label">模拟盘</div><div class="value small-value" id="demoOrders">--</div></div>
      <div class="metric"><div class="label">更新时间</div><div class="value small-value" id="updatedAt">--</div></div>
      <div class="metric"><div class="label">连接状态</div><div class="value small-value" id="health">--</div></div>
    </section>

    <section class="grid">
      <div class="panel" id="realtimePanel"></div>
      <div class="panel" id="confirmedPanel"></div>
      <div class="panel">
        <h2>执行计划</h2>
        <div id="plan"></div>
        <div class="actions">
          <button id="refreshBtn" class="secondary">刷新行情</button>
          <button id="dryRunBtn">记录 Dry-run</button>
        </div>
      </div>
      <div class="panel">
        <h2>OKX 模拟盘</h2>
        <div id="okxPanel"></div>
        <div class="actions">
          <button id="testOkxBtn" class="secondary">测试连接</button>
          <button id="demoOrderBtn" class="danger">提交模拟盘订单</button>
        </div>
      </div>
    </section>

    <section class="wide">
      <div class="panel">
        <h2>Dry-run 记录</h2>
        <div id="records"></div>
      </div>
      <div class="panel">
        <h2>模拟盘订单</h2>
        <div id="demoOrderRecords"></div>
      </div>
      <div class="panel">
        <h2>运行日志</h2>
        <div id="logs"></div>
      </div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    let latest = null;

    function money(v, d = 2) {
      return v === null || v === undefined ? '--' : Number(v).toFixed(d);
    }

    function pct(v) {
      return v === null || v === undefined ? '--' : (Number(v) * 100).toFixed(0) + '%';
    }

    function badgeClass(label) {
      if (label.includes('做多')) return 'green';
      if (label.includes('做空')) return 'red';
      if (label.includes('接近')) return 'amber';
      return 'dark';
    }

    function kv(label, value) {
      return `<div class="kv"><span>${label}</span><span>${value}</span></div>`;
    }

    function renderOpportunity(title, data) {
      if (!data) return '<h2>' + title + '</h2><div class="empty">暂无数据</div>';
      const status = data.is_closed ? '已收盘' : '未收盘';
      const reasons = [...data.long_reasons, ...data.short_reasons].slice(0, 8)
        .map(x => `<li>${x}</li>`).join('');
      return `
        <div class="signal-title">
          <h2>${title}</h2>
          <span class="badge ${badgeClass(data.label)}">${data.label} · ${status}</span>
        </div>
        <div class="scores">
          <div class="scorebox"><div class="label">做多机会分</div><strong class="status-ok">${data.long_score}</strong></div>
          <div class="scorebox"><div class="label">做空机会分</div><strong class="status-bad">${data.short_score}</strong></div>
        </div>
        ${kv('4H时间', data.candle.time)}
        ${kv('价格', money(data.candle.close))}
        ${kv('唐奇安40', money(data.lower) + ' / ' + money(data.upper))}
        ${kv('EMA180', money(data.ema))}
        ${kv('ADX14', money(data.adx))}
        ${kv('ATR14', money(data.atr) + ' (' + money(data.atr_pct) + '%)')}
        <ul class="reasons">${reasons}</ul>
      `;
    }

    function renderPlan(plan, settings) {
      if (!plan) return '<div class="empty">暂无执行计划</div>';
      const enabled = plan.side !== null;
      $('dryRunBtn').disabled = !enabled;
      return `
        <span class="badge ${enabled ? badgeClass(plan.side_text) : 'dark'}">${plan.side_text}</span>
        ${kv('状态', plan.status)}
        ${kv('入场参考', money(plan.entry_price))}
        ${kv('初始止损', money(plan.stop_price))}
        ${kv('3R止盈', money(plan.take_profit_price))}
        ${kv('单笔风险距离', money(plan.risk))}
        ${kv('保证金', money(plan.margin) + 'U')}
        ${kv('名义仓位', money(plan.notional) + 'U')}
        ${kv('数量', money(plan.qty, 6) + ' BTC')}
        ${kv('最长持仓', settings.max_hold_hours + '小时')}
      `;
    }

    function renderOkx(okx, plan) {
      if (!okx) {
        $('testOkxBtn').disabled = false;
        $('demoOrderBtn').disabled = true;
        return '<div class="empty">未读取到 OKX 配置状态</div>';
      }
      const hasSignal = plan && plan.side !== null;
      $('testOkxBtn').disabled = false;
      $('demoOrderBtn').disabled = !(okx.enabled && okx.allow_demo_orders && hasSignal);
      const orderSwitch = okx.allow_demo_orders ? '<span class="status-ok">已打开</span>' : '<span class="status-bad">关闭</span>';
      return `
        <span class="badge ${okx.enabled ? 'green' : 'amber'}">${okx.enabled ? '配置完整' : '未就绪'}</span>
        ${kv('账户类型', okx.demo ? '模拟盘' : '实盘禁用')}
        ${kv('下单开关', orderSwitch)}
        ${kv('持仓模式', okx.pos_side === 'net' ? '单向持仓' : '多空双向')}
        ${kv('配置文件', okx.env_path || '.env')}
        ${kv('状态', okx.message || '--')}
      `;
    }

    function renderRecords(records) {
      if (!records || records.length === 0) return '<div class="empty">暂无 dry-run 记录</div>';
      const rows = records.slice(0, 20).map(r => `
        <tr>
          <td>${r.created_at}</td>
          <td>${r.side_text}</td>
          <td>${money(r.entry_price)}</td>
          <td>${money(r.stop_price)}</td>
          <td>${money(r.take_profit_price)}</td>
          <td>${money(r.notional)}U</td>
          <td>${r.status}</td>
        </tr>`).join('');
      return `<table><thead><tr><th>时间</th><th>方向</th><th>入场</th><th>止损</th><th>止盈</th><th>仓位</th><th>状态</th></tr></thead><tbody>${rows}</tbody></table>`;
    }

    function renderDemoOrders(records) {
      if (!records || records.length === 0) return '<div class="empty">暂无模拟盘订单</div>';
      const rows = records.slice(0, 20).map(r => `
        <tr>
          <td>${r.created_at}</td>
          <td>${r.side_text}</td>
          <td>${r.order_preview ? r.order_preview.sz : '--'}</td>
          <td>${money(r.entry_price)}</td>
          <td>${money(r.stop_price)}</td>
          <td>${money(r.take_profit_price)}</td>
          <td>${r.status}</td>
        </tr>`).join('');
      return `<table><thead><tr><th>时间</th><th>方向</th><th>张数</th><th>入场</th><th>止损</th><th>止盈</th><th>状态</th></tr></thead><tbody>${rows}</tbody></table>`;
    }

    function renderLogs(logs) {
      if (!logs || logs.length === 0) return '<div class="empty">暂无日志</div>';
      return logs.slice(0, 60).map(l => `<div class="logline"><span>${l.time}</span><strong>${l.level}</strong><span>${l.message}</span></div>`).join('');
    }

    async function refresh(force = false) {
      $('health').innerHTML = '<span>刷新中</span>';
      const res = await fetch('/api/status' + (force ? '?force=1' : ''));
      const data = await res.json();
      latest = data;
      $('subtitle').textContent = `${data.inst_id || 'BTC-USDT-SWAP'} · 4H唐奇安趋势突破 · 本机 dry-run / OKX模拟盘`;
      $('mode').textContent = data.mode || '--';
      $('realOrders').innerHTML = data.real_orders_enabled ? '<span class="status-bad">开启</span>' : '<span class="status-ok">关闭</span>';
      $('demoOrders').innerHTML = data.okx && data.okx.allow_demo_orders ? '<span class="status-ok">可提交</span>' : '<span class="status-bad">关闭</span>';
      $('updatedAt').textContent = data.updated_at || '--';
      $('health').innerHTML = data.ok ? '<span class="status-ok">正常</span>' : '<span class="status-bad">异常</span>';
      if (!data.ok) {
        $('plan').innerHTML = `<div class="empty">${data.error || '扫描失败'}</div>`;
        $('dryRunBtn').disabled = true;
        $('demoOrderBtn').disabled = true;
      } else {
        $('realtimePanel').innerHTML = renderOpportunity('实时预警', data.realtime);
        $('confirmedPanel').innerHTML = renderOpportunity('收盘确认', data.confirmed);
        $('plan').innerHTML = renderPlan(data.plan, data.settings);
      }
      $('okxPanel').innerHTML = renderOkx(data.okx, data.plan);
      $('records').innerHTML = renderRecords(data.records);
      $('demoOrderRecords').innerHTML = renderDemoOrders(data.demo_orders);
      $('logs').innerHTML = renderLogs(data.logs);
    }

    async function recordDryRun() {
      $('dryRunBtn').disabled = true;
      const res = await fetch('/api/dry-run', { method: 'POST' });
      await res.json();
      await refresh(true);
    }

    async function postAndRefresh(path, buttonId) {
      const btn = $(buttonId);
      btn.disabled = true;
      const res = await fetch(path, { method: 'POST' });
      const payload = await res.json();
      if (!payload.ok && payload.message) {
        window.alert(payload.message);
      }
      await refresh(true);
    }

    $('refreshBtn').addEventListener('click', () => refresh(true));
    $('dryRunBtn').addEventListener('click', recordDryRun);
    $('testOkxBtn').addEventListener('click', () => postAndRefresh('/api/okx/test', 'testOkxBtn'));
    $('demoOrderBtn').addEventListener('click', () => postAndRefresh('/api/okx/demo-order', 'demoOrderBtn'));
    refresh(true);
    setInterval(() => refresh(false), 15000);
  </script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    app: LocalAutoTraderApp

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        body = HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html()
            return
        if parsed.path == "/api/status":
            self._send_json(HTTPStatus.OK, self.app.status(force="force=1" in parsed.query))
            return
        if parsed.path == "/api/health":
            self._send_json(HTTPStatus.OK, {"ok": True, "time": now_text(), "mode": "dry-run"})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "message": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/dry-run":
            status, payload = self.app.record_dry_run()
            self._send_json(status, payload)
            return
        if parsed.path == "/api/okx/test":
            status, payload = self.app.test_okx_connection()
            self._send_json(status, payload)
            return
        if parsed.path == "/api/okx/demo-order":
            status, payload = self.app.place_demo_order()
            self._send_json(status, payload)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "message": "not found"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 BTC 本机自动交易 dry-run UI")
    parser.add_argument("--inst-id", default="BTC-USDT-SWAP")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--refresh-seconds", type=int, default=30)
    parser.add_argument("--proxy-mode", choices=["fallback", "on", "off"], default="fallback")
    parser.add_argument("--proxy-url", default="http://127.0.0.1:7897")
    parser.add_argument("--initial-equity", type=float, default=1000.0)
    parser.add_argument("--leverage", type=float, default=8.0)
    parser.add_argument("--margin-pct", type=float, default=0.10)
    parser.add_argument("--state-path", type=Path, default=Path("data/local_auto_trader_state.json"))
    parser.add_argument("--env-path", type=Path, default=Path(".env"))
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    args = build_parser().parse_args(argv)
    config = AppConfig(
        inst_id=args.inst_id,
        host=args.host,
        port=args.port,
        refresh_seconds=args.refresh_seconds,
        proxy_url=args.proxy_url,
        proxy_mode=args.proxy_mode,
        initial_equity=args.initial_equity,
        leverage=args.leverage,
        margin_pct=args.margin_pct,
        state_path=args.state_path,
        env_path=args.env_path,
    )
    app = LocalAutoTraderApp(config)
    RequestHandler.app = app
    server = ThreadingHTTPServer((config.host, config.port), RequestHandler)
    url = f"http://{config.host}:{config.port}"
    print(f"BTC 本机自动交易 dry-run UI 已启动：{url}")
    print("真实下单：关闭")
    print(f"OKX 配置文件：{config.env_path}")
    print("按 Ctrl+C 停止。")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
        print("程序已停止。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
