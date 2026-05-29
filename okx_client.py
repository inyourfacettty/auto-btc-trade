#!/usr/bin/env python3
"""Small OKX v5 REST client for demo-trading integration."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any


OKX_BASE_URL = "https://www.okx.com"


@dataclass(frozen=True)
class OkxCredentials:
    api_key: str
    secret_key: str
    passphrase: str
    demo: bool = True


@dataclass(frozen=True)
class InstrumentRules:
    inst_id: str
    ct_val: float
    lot_sz: float
    min_sz: float
    tick_sz: float


@dataclass(frozen=True)
class OkxApiSettings:
    enabled: bool
    demo: bool
    allow_demo_orders: bool
    pos_side: str
    credentials: OkxCredentials | None
    message: str


def parse_env_text(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return parse_env_text(path.read_text(encoding="utf-8"))


def load_okx_settings(path: Path = Path(".env")) -> OkxApiSettings:
    env = load_env_file(path)
    trading_mode = env.get("OKX_TRADING_MODE", "demo").strip().lower()
    demo = trading_mode != "live"
    allow_demo_orders = env.get("OKX_ENABLE_DEMO_ORDER", "false").strip().lower() in {"1", "true", "yes", "on"}
    pos_side = env.get("OKX_POS_SIDE", "net").strip() or "net"
    api_key = env.get("OKX_API_KEY", "").strip()
    secret_key = env.get("OKX_SECRET_KEY", "").strip()
    passphrase = env.get("OKX_PASSPHRASE", "").strip()
    missing = [
        name
        for name, value in (
            ("OKX_API_KEY", api_key),
            ("OKX_SECRET_KEY", secret_key),
            ("OKX_PASSPHRASE", passphrase),
        )
        if not value
    ]
    if missing:
        return OkxApiSettings(
            enabled=False,
            demo=demo,
            allow_demo_orders=False,
            pos_side=pos_side,
            credentials=None,
            message=f"缺少配置：{', '.join(missing)}",
        )
    if not demo:
        return OkxApiSettings(
            enabled=False,
            demo=False,
            allow_demo_orders=False,
            pos_side=pos_side,
            credentials=None,
            message="实盘模式被本版本禁用，请先使用 OKX_TRADING_MODE=demo",
        )
    return OkxApiSettings(
        enabled=True,
        demo=True,
        allow_demo_orders=allow_demo_orders,
        pos_side=pos_side,
        credentials=OkxCredentials(api_key=api_key, secret_key=secret_key, passphrase=passphrase, demo=True),
        message="OKX模拟盘配置已加载",
    )


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def build_okx_signature(timestamp: str, method: str, request_path: str, body: str, secret_key: str) -> str:
    prehash = f"{timestamp}{method.upper()}{request_path}{body}"
    digest = hmac.new(secret_key.encode("utf-8"), prehash.encode("utf-8"), hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def build_request_headers(
    api_key: str,
    passphrase: str,
    signature: str,
    timestamp: str,
    demo: bool,
) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 btc-local-auto-trader/1.0",
        "OK-ACCESS-KEY": api_key,
        "OK-ACCESS-SIGN": signature,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": passphrase,
    }
    if demo:
        headers["x-simulated-trading"] = "1"
    return headers


def _decimal_floor(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def _step_decimals(step: float) -> int:
    step_text = str(step)
    if "e" in step_text.lower():
        step_text = format(Decimal(step_text), "f")
    if "." not in step_text:
        return 0
    return len(step_text.split(".", 1)[1].rstrip("0"))


def _format_decimal(value: Decimal, decimals: int) -> str:
    if decimals <= 0:
        return format(value.quantize(Decimal("1"), rounding=ROUND_DOWN), "f")
    quant = Decimal("1").scaleb(-decimals)
    return format(value.quantize(quant, rounding=ROUND_DOWN), "f")


def round_down_to_step(value: float, step: float) -> str:
    rounded = _decimal_floor(Decimal(str(value)), Decimal(str(step)))
    return _format_decimal(rounded, _step_decimals(step))


def round_price(value: float, tick_size: float) -> str:
    rounded = _decimal_floor(Decimal(str(value)), Decimal(str(tick_size)))
    return _format_decimal(rounded, _step_decimals(tick_size))


def build_order_preview(
    side: str,
    entry_price: float,
    stop_price: float,
    take_profit_price: float,
    notional: float,
    rules: InstrumentRules,
    pos_side: str = "net",
) -> dict[str, str]:
    if entry_price <= 0 or rules.ct_val <= 0:
        raise ValueError("entry_price and ct_val must be positive")
    raw_contracts = notional / (entry_price * rules.ct_val)
    contracts = max(float(rules.min_sz), raw_contracts)
    size = round_down_to_step(contracts, rules.lot_sz)
    if float(size) < rules.min_sz:
        raise ValueError(f"order size {size} is smaller than min size {rules.min_sz}")
    if side == "long":
        order_side = "buy"
        pos = "long" if pos_side != "net" else "net"
    elif side == "short":
        order_side = "sell"
        pos = "short" if pos_side != "net" else "net"
    else:
        raise ValueError(f"unsupported side: {side}")
    return {
        "side": order_side,
        "posSide": pos,
        "sz": size,
        "tdMode": "isolated",
        "ordType": "market",
        "tpTriggerPx": round_price(take_profit_price, rules.tick_sz),
        "tpOrdPx": "-1",
        "slTriggerPx": round_price(stop_price, rules.tick_sz),
        "slOrdPx": "-1",
    }


class OkxRestClient:
    def __init__(self, credentials: OkxCredentials, proxy_url: str | None = None, proxy_mode: str = "fallback") -> None:
        self.credentials = credentials
        self.proxy_url = proxy_url
        self.proxy_mode = proxy_mode

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        method = method.upper()
        query_or_body = ""
        request_path = path
        data: bytes | None = None
        if method == "GET" and payload:
            query = urllib.parse.urlencode({k: str(v) for k, v in payload.items() if v is not None})
            request_path = f"{path}?{query}" if query else path
        elif payload is not None:
            query_or_body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
            data = query_or_body.encode("utf-8")
        timestamp = utc_timestamp()
        signature = build_okx_signature(timestamp, method, request_path, query_or_body, self.credentials.secret_key)
        headers = build_request_headers(
            self.credentials.api_key,
            self.credentials.passphrase,
            signature,
            timestamp,
            self.credentials.demo,
        )
        url = f"{OKX_BASE_URL}{request_path}"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        return self._open_json(request)

    def _open_json(self, request: urllib.request.Request) -> dict[str, Any]:
        errors: list[str] = []
        if self.proxy_mode != "on":
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {body}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                errors.append(f"direct: {exc}")
                if self.proxy_mode == "off" or not self.proxy_url:
                    raise
        if not self.proxy_url:
            raise RuntimeError("; ".join(errors) or "proxy_url not configured")
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": self.proxy_url, "https": self.proxy_url})
        )
        try:
            with opener.open(request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {body}") from exc

    def account_balance(self) -> dict[str, Any]:
        return self.request("GET", "/api/v5/account/balance", {})

    def account_config(self) -> dict[str, Any]:
        return self.request("GET", "/api/v5/account/config", {})

    def positions(self, inst_id: str) -> dict[str, Any]:
        return self.request("GET", "/api/v5/account/positions", {"instId": inst_id})

    def instrument_rules(self, inst_id: str) -> InstrumentRules:
        payload = self.request("GET", "/api/v5/public/instruments", {"instType": "SWAP", "instId": inst_id})
        rows = payload.get("data", [])
        if payload.get("code") != "0" or not rows:
            raise RuntimeError(f"无法读取合约信息：{payload}")
        row = rows[0]
        return InstrumentRules(
            inst_id=inst_id,
            ct_val=float(row["ctVal"]),
            lot_sz=float(row["lotSz"]),
            min_sz=float(row["minSz"]),
            tick_sz=float(row["tickSz"]),
        )

    def set_leverage(self, inst_id: str, leverage: float, mgn_mode: str = "isolated", pos_side: str = "net") -> dict[str, Any]:
        payload: dict[str, Any] = {
            "instId": inst_id,
            "lever": str(int(leverage) if float(leverage).is_integer() else leverage),
            "mgnMode": mgn_mode,
        }
        if pos_side != "net":
            payload["posSide"] = pos_side
        return self.request("POST", "/api/v5/account/set-leverage", payload)

    def place_market_order_with_tpsl(
        self,
        inst_id: str,
        preview: dict[str, str],
        client_order_id: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "instId": inst_id,
            "tdMode": preview["tdMode"],
            "side": preview["side"],
            "ordType": preview["ordType"],
            "sz": preview["sz"],
            "clOrdId": client_order_id,
            "attachAlgoOrds": [
                {
                    "tpTriggerPx": preview["tpTriggerPx"],
                    "tpOrdPx": preview["tpOrdPx"],
                    "slTriggerPx": preview["slTriggerPx"],
                    "slOrdPx": preview["slOrdPx"],
                }
            ],
        }
        if preview["posSide"] != "net":
            payload["posSide"] = preview["posSide"]
        return self.request("POST", "/api/v5/trade/order", payload)
