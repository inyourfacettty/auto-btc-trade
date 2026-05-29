import base64
import hashlib
import hmac
import unittest

from okx_client import (
    InstrumentRules,
    build_okx_signature,
    build_order_preview,
    build_request_headers,
    parse_env_text,
)


class OkxAuthTests(unittest.TestCase):
    def test_build_okx_signature_uses_timestamp_method_path_body(self):
        timestamp = "2026-05-29T08:00:00.000Z"
        method = "POST"
        path = "/api/v5/trade/order"
        body = '{"instId":"BTC-USDT-SWAP"}'
        secret = "secret"

        signature = build_okx_signature(timestamp, method, path, body, secret)

        expected = base64.b64encode(
            hmac.new(secret.encode(), f"{timestamp}{method}{path}{body}".encode(), hashlib.sha256).digest()
        ).decode()
        self.assertEqual(signature, expected)

    def test_build_request_headers_include_demo_header_when_enabled(self):
        headers = build_request_headers(
            api_key="key",
            passphrase="pass",
            signature="sig",
            timestamp="2026-05-29T08:00:00.000Z",
            demo=True,
        )

        self.assertEqual(headers["OK-ACCESS-KEY"], "key")
        self.assertEqual(headers["OK-ACCESS-PASSPHRASE"], "pass")
        self.assertEqual(headers["OK-ACCESS-SIGN"], "sig")
        self.assertEqual(headers["x-simulated-trading"], "1")
        self.assertEqual(headers["Accept"], "application/json")
        self.assertIn("btc-local-auto-trader", headers["User-Agent"])


class OkxEnvTests(unittest.TestCase):
    def test_parse_env_text_supports_quotes_and_comments(self):
        parsed = parse_env_text(
            """
            # demo
            OKX_API_KEY="abc"
            OKX_SECRET_KEY=def
            EMPTY=
            """
        )

        self.assertEqual(parsed["OKX_API_KEY"], "abc")
        self.assertEqual(parsed["OKX_SECRET_KEY"], "def")
        self.assertEqual(parsed["EMPTY"], "")


class OkxOrderPreviewTests(unittest.TestCase):
    def test_build_order_preview_converts_usdt_notional_to_swap_contracts(self):
        rules = InstrumentRules(inst_id="BTC-USDT-SWAP", ct_val=0.01, lot_sz=0.01, min_sz=0.01, tick_sz=0.1)

        preview = build_order_preview(
            side="long",
            entry_price=73_500,
            stop_price=72_000,
            take_profit_price=78_000,
            notional=1_200,
            rules=rules,
        )

        self.assertEqual(preview["side"], "buy")
        self.assertEqual(preview["posSide"], "net")
        self.assertEqual(preview["sz"], "1.63")
        self.assertEqual(preview["slTriggerPx"], "72000.0")
        self.assertEqual(preview["tpTriggerPx"], "78000.0")


if __name__ == "__main__":
    unittest.main()
