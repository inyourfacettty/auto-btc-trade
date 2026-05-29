import unittest
from contextlib import redirect_stdout
from io import StringIO
from urllib.error import URLError
from unittest.mock import Mock, patch

import backtest_okx_btc
from backtest_okx_btc import (
    Candle,
    BacktestConfig,
    BacktestResult,
    NetworkConfig,
    compute_ema,
    compute_rsi,
    request_json,
    print_report,
    run_backtest,
)


class IndicatorTests(unittest.TestCase):
    def test_ema_uses_exponential_weighting(self):
        values = [10, 11, 12, 13, 14]

        ema = compute_ema(values, 3)

        self.assertEqual(ema[:2], [None, None])
        self.assertAlmostEqual(ema[2], 11.0)
        self.assertAlmostEqual(ema[3], 12.0)
        self.assertAlmostEqual(ema[4], 13.0)

    def test_rsi_reports_full_strength_after_only_gains(self):
        values = [10, 11, 12, 13, 14, 15]

        rsi = compute_rsi(values, 3)

        self.assertEqual(rsi[:3], [None, None, None])
        self.assertEqual(rsi[3:], [100.0, 100.0, 100.0])


class BacktestExecutionTests(unittest.TestCase):
    def test_backtest_enters_next_open_and_uses_stop_first_when_ambiguous(self):
        candles_15m = []
        start = 1_700_000_000_000
        for i in range(35):
            close = 100 + i * 0.1
            candles_15m.append(
                Candle(
                    ts=start + i * 900_000,
                    open=close - 0.05,
                    high=close + 0.2,
                    low=close - 0.2,
                    close=close,
                    volume=1,
                )
            )

        # Build a pullback then a reclaim signal at index 30, entry on index 31.
        candles_15m[28] = candles_15m[28].replace(open=103, high=103.2, low=99.8, close=100.0)
        candles_15m[29] = candles_15m[29].replace(open=100, high=100.5, low=99.7, close=99.9)
        candles_15m[30] = candles_15m[30].replace(open=99.9, high=102.0, low=99.8, close=101.8)
        candles_15m[31] = candles_15m[31].replace(open=102.0, high=106.0, low=98.0, close=103.0)

        candles_4h = [
            Candle(
                ts=start - 80 * 14_400_000 + i * 14_400_000,
                open=90 + i,
                high=91 + i,
                low=89 + i,
                close=90 + i,
                volume=1,
            )
            for i in range(90)
        ]

        result = run_backtest(
            candles_15m,
            candles_4h,
            BacktestConfig(
                initial_equity=1000,
                leverage=3,
                fee_rate=0,
                slippage_bps=0,
                take_profit_r=2,
                pullback_tolerance_pct=0.02,
                stop_lookback=3,
                min_rsi=0,
                max_rsi=100,
                signal_start_ts=start,
            ),
        )

        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertEqual(trade.entry_ts, candles_15m[31].ts)
        self.assertEqual(trade.exit_reason, "stop_loss")
        self.assertLess(trade.net_pnl, 0)


class ReportTests(unittest.TestCase):
    def test_report_outputs_chinese_labels(self):
        output = StringIO()

        with redirect_stdout(output):
            print_report(
                BacktestResult(
                    trades=[],
                    equity_curve=[],
                    metrics={
                        "trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "win_rate_pct": 0.0,
                        "initial_equity": 1000.0,
                        "final_equity": 1000.0,
                        "net_pnl": 0.0,
                        "total_return_pct": 0.0,
                        "max_drawdown_pct": 0.0,
                        "profit_factor": 0.0,
                        "avg_trade_return_pct": 0.0,
                        "avg_r_multiple": 0.0,
                        "fees_paid": 0.0,
                        "best_trade": 0.0,
                        "worst_trade": 0.0,
                    },
                ),
                BacktestConfig(),
                "BTC-USDT",
                "okx",
                1_700_000_000_000,
                1_700_086_400_000,
                [],
                [],
                10,
            )

        text = output.getvalue()
        self.assertIn("回测", text)
        self.assertIn("数据源", text)
        self.assertIn("没有交易符合策略条件", text)


class NetworkTests(unittest.TestCase):
    def test_request_json_falls_back_to_configured_proxy(self):
        backtest_okx_btc.DIRECT_REQUEST_FAILED = False

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"code":"0","data":[]}'

        opener = Mock()
        opener.open.return_value = FakeResponse()

        with patch(
            "backtest_okx_btc.urllib.request.urlopen",
            side_effect=URLError("direct failed"),
        ) as direct_open:
            with patch("backtest_okx_btc.urllib.request.build_opener", return_value=opener) as build_opener:
                payload = request_json(
                    "https://www.okx.com/api/v5/market/history-candles",
                    "OKX",
                    NetworkConfig(proxy_url="http://127.0.0.1:7897", proxy_mode="fallback"),
                )
                second_payload = request_json(
                    "https://www.okx.com/api/v5/market/history-candles",
                    "OKX",
                    NetworkConfig(proxy_url="http://127.0.0.1:7897", proxy_mode="fallback"),
                )

        self.assertEqual(payload, {"code": "0", "data": []})
        self.assertEqual(second_payload, {"code": "0", "data": []})
        self.assertEqual(direct_open.call_count, 1)
        self.assertEqual(build_opener.call_count, 2)
        self.assertEqual(opener.open.call_count, 2)


if __name__ == "__main__":
    unittest.main()
