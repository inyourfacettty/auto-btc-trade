import unittest

from scheduled_status_email import (
    build_email_body,
    build_email_html,
    build_email_subject,
    build_operation_summary,
    direction_text,
)


class ScheduledStatusEmailFormattingTests(unittest.TestCase):
    def test_direction_text_uses_chinese_labels(self):
        self.assertEqual(direction_text("long"), "做多")
        self.assertEqual(direction_text("short"), "做空")
        self.assertEqual(direction_text(None), "无信号")

    def test_operation_summary_waits_when_no_signal(self):
        plan = {
            "side": None,
            "status": "收盘确认没有交易信号，继续等待",
            "margin": 100.0,
            "notional": 800.0,
        }

        summary = build_operation_summary(plan)

        self.assertIn("操作建议：不操作", summary)
        self.assertIn("继续等待", summary)
        self.assertIn("计划保证金：100.00U", summary)

    def test_operation_summary_includes_trade_levels_when_signal_exists(self):
        plan = {
            "side": "short",
            "status": "可执行窗口内",
            "entry_price": 73500.0,
            "stop_price": 75200.0,
            "take_profit_price": 68400.0,
            "risk": 1700.0,
            "margin": 100.0,
            "notional": 800.0,
            "qty": 0.010884,
        }

        summary = build_operation_summary(plan)

        self.assertIn("操作建议：做空", summary)
        self.assertIn("入场参考：73500.00", summary)
        self.assertIn("初始止损：75200.00", summary)
        self.assertIn("3R止盈：68400.00", summary)

    def test_email_subject_contains_signal_label_and_direction(self):
        subject = build_email_subject(
            {
                "inst_id": "BTC-USDT-SWAP",
                "updated_at": "2026-05-29 20:15:00",
                "confirmed": {"label": "触发做多"},
                "plan": {"side": "long"},
            }
        )

        self.assertEqual(subject, "BTC-USDT-SWAP 策略状态：触发做多 / 做多 / 2026-05-29 20:15:00")

    def test_email_body_contains_status_and_risk_notice(self):
        body = build_email_body(
            {
                "ok": True,
                "updated_at": "2026-05-29 20:15:00",
                "inst_id": "BTC-USDT-SWAP",
                "confirmed": {
                    "label": "接近做空触发",
                    "candle": {"time": "2026-05-29 16:00", "close": 73823.0},
                    "upper": 78066.7,
                    "lower": 72554.0,
                    "ema": 76978.74,
                    "adx": 29.58,
                    "atr": 792.33,
                    "atr_pct": 1.07,
                    "long_score": 35,
                    "short_score": 84,
                    "long_reasons": ["距上轨 5.75%"],
                    "short_reasons": ["距下轨 1.75%"],
                },
                "realtime": {
                    "label": "接近做空触发",
                    "is_closed": False,
                    "candle": {"time": "2026-05-29 20:00", "open": 73823.0, "high": 73900.0, "low": 73300.0, "close": 73420.0},
                    "upper": 78066.7,
                    "lower": 72554.0,
                    "ema": 76950.0,
                    "adx": 30.0,
                    "atr": 800.0,
                    "atr_pct": 1.09,
                    "long_score": 33,
                    "short_score": 86,
                    "long_reasons": ["距上轨 6.33%"],
                    "short_reasons": ["距下轨 1.19%"],
                },
                "plan": {"side": None, "status": "继续等待", "margin": 100.0, "notional": 800.0},
            }
        )

        self.assertIn("BTC-USDT-SWAP 4H策略状态", body)
        self.assertIn("收盘确认：接近做空触发", body)
        self.assertIn("操作建议：不操作", body)
        self.assertIn("本邮件只做策略提醒，不会自动下单。", body)

    def test_email_html_uses_tables_for_confirmed_and_realtime_candles(self):
        html = build_email_html(
            {
                "ok": True,
                "updated_at": "2026-05-29 20:15:00",
                "inst_id": "BTC-USDT-SWAP",
                "confirmed": {
                    "label": "接近做空触发",
                    "is_closed": True,
                    "candle": {"time": "2026-05-29 16:00", "open": 73200.0, "high": 73900.0, "low": 73100.0, "close": 73823.0},
                    "upper": 78066.7,
                    "lower": 72554.0,
                    "ema": 76978.74,
                    "adx": 29.58,
                    "atr": 792.33,
                    "atr_pct": 1.07,
                    "long_score": 35,
                    "short_score": 84,
                    "long_reasons": ["距上轨 5.75%"],
                    "short_reasons": ["距下轨 1.75%"],
                },
                "realtime": {
                    "label": "接近做空触发",
                    "is_closed": False,
                    "candle": {"time": "2026-05-29 20:00", "open": 73823.0, "high": 73950.0, "low": 73400.0, "close": 73520.0},
                    "upper": 78066.7,
                    "lower": 72554.0,
                    "ema": 76920.0,
                    "adx": 30.0,
                    "atr": 810.0,
                    "atr_pct": 1.10,
                    "long_score": 34,
                    "short_score": 87,
                    "long_reasons": ["距上轨 6.18%"],
                    "short_reasons": ["距下轨 1.33%"],
                },
                "plan": {"side": None, "status": "继续等待", "margin": 100.0, "notional": 800.0},
                "settings": {"initial_equity": 1000.0, "leverage": 8.0, "margin_pct": 0.10, "max_hold_hours": 96},
            }
        )

        self.assertIn("<table", html)
        self.assertIn("上个已收盘4H", html)
        self.assertIn("当前未收盘4H", html)
        self.assertIn("2026-05-29 16:00", html)
        self.assertIn("2026-05-29 20:00", html)
        self.assertIn("操作建议", html)


if __name__ == "__main__":
    unittest.main()
