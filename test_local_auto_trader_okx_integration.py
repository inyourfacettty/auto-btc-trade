import tempfile
import unittest
from http import HTTPStatus
from pathlib import Path

from local_auto_trader_ui import AppConfig, LocalAutoTraderApp
from okx_client import InstrumentRules


def write_env(path: Path, *, allow_demo_order: bool = False) -> None:
    path.write_text(
        "\n".join(
            [
                "OKX_TRADING_MODE=demo",
                "OKX_API_KEY=dummy-key",
                "OKX_SECRET_KEY=dummy-secret",
                "OKX_PASSPHRASE=dummy-passphrase",
                "OKX_POS_SIDE=net",
                f"OKX_ENABLE_DEMO_ORDER={'true' if allow_demo_order else 'false'}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def make_config(tmp: Path, env_path: Path) -> AppConfig:
    return AppConfig(
        inst_id="BTC-USDT-SWAP",
        host="127.0.0.1",
        port=8765,
        refresh_seconds=30,
        proxy_url="http://127.0.0.1:7897",
        proxy_mode="off",
        initial_equity=1000.0,
        leverage=8.0,
        margin_pct=0.10,
        state_path=tmp / "state.json",
        env_path=env_path,
    )


def signal_status() -> dict:
    return {
        "ok": True,
        "plan": {
            "side": "long",
            "side_text": "做多",
            "status": "确认突破，可按计划执行",
            "entry_price": 73500.0,
            "stop_price": 72000.0,
            "take_profit_price": 78000.0,
            "risk": 1500.0,
            "margin": 100.0,
            "notional": 800.0,
            "qty": 0.010884,
            "signal_age_minutes": 7.0,
        },
        "confirmed": {
            "candle": {
                "ts": 1_780_000_000_000,
                "time": "2026-05-29 16:00",
            }
        },
    }


class FakeOkxClient:
    def __init__(self) -> None:
        self.leverage_calls = []
        self.order_calls = []

    def instrument_rules(self, inst_id: str) -> InstrumentRules:
        return InstrumentRules(inst_id=inst_id, ct_val=0.01, lot_sz=0.01, min_sz=0.01, tick_sz=0.1)

    def set_leverage(self, inst_id: str, leverage: float, mgn_mode: str = "isolated", pos_side: str = "net") -> dict:
        self.leverage_calls.append((inst_id, leverage, mgn_mode, pos_side))
        return {"code": "0", "data": [{"lever": str(leverage)}]}

    def place_market_order_with_tpsl(self, inst_id: str, preview: dict[str, str], client_order_id: str) -> dict:
        self.order_calls.append((inst_id, preview, client_order_id))
        return {"code": "0", "data": [{"ordId": "1234567890", "clOrdId": client_order_id}]}


class FakeTradingApp(LocalAutoTraderApp):
    def __init__(self, config: AppConfig, status_payload: dict, client: FakeOkxClient) -> None:
        super().__init__(config)
        self._status_payload = status_payload
        self._client = client

    def status(self, force: bool = False) -> dict:
        return self._status_payload

    def okx_client(self) -> FakeOkxClient:
        return self._client


class LocalAutoTraderOkxIntegrationTests(unittest.TestCase):
    def test_okx_status_reports_missing_env_without_secrets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            app = LocalAutoTraderApp(make_config(tmp, tmp / ".env"))

            payload = app.okx_status_payload()

            self.assertFalse(payload["enabled"])
            self.assertFalse(payload["allow_demo_orders"])
            self.assertIn("OKX_API_KEY", payload["message"])
            self.assertNotIn("dummy-secret", str(payload))

    def test_place_demo_order_requires_safety_switch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            env_path = tmp / ".env"
            write_env(env_path, allow_demo_order=False)
            client = FakeOkxClient()
            app = FakeTradingApp(make_config(tmp, env_path), signal_status(), client)

            status, payload = app.place_demo_order()

            self.assertEqual(status, HTTPStatus.CONFLICT)
            self.assertFalse(payload["ok"])
            self.assertIn("OKX_ENABLE_DEMO_ORDER", payload["message"])
            self.assertEqual(client.order_calls, [])

    def test_place_demo_order_builds_preview_sets_leverage_and_saves_record(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            env_path = tmp / ".env"
            write_env(env_path, allow_demo_order=True)
            client = FakeOkxClient()
            app = FakeTradingApp(make_config(tmp, env_path), signal_status(), client)

            status, payload = app.place_demo_order()

            self.assertEqual(status, HTTPStatus.OK)
            self.assertTrue(payload["ok"])
            self.assertEqual(client.leverage_calls, [("BTC-USDT-SWAP", 8.0, "isolated", "net")])
            self.assertEqual(len(client.order_calls), 1)
            _inst_id, preview, client_order_id = client.order_calls[0]
            self.assertEqual(preview["side"], "buy")
            self.assertEqual(preview["sz"], "1.08")
            self.assertTrue(client_order_id.startswith("SIM"))
            records = app.state["demo_orders"]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["okx_order_response"]["code"], "0")


if __name__ == "__main__":
    unittest.main()
