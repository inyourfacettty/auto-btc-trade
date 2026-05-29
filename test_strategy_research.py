import unittest

from strategy_research import ResearchConfig, calculate_position_notional, split_periods


class PositionSizingTests(unittest.TestCase):
    def test_position_notional_caps_at_max_leverage(self):
        config = ResearchConfig(initial_equity=1000, max_leverage=3, risk_pct=0.01)

        notional = calculate_position_notional(equity=1000, entry_price=100, stop_price=99.9, config=config)

        self.assertEqual(notional, 3000)

    def test_position_notional_respects_risk_when_stop_is_wide(self):
        config = ResearchConfig(initial_equity=1000, max_leverage=3, risk_pct=0.01)

        notional = calculate_position_notional(equity=1000, entry_price=100, stop_price=95, config=config)

        self.assertEqual(notional, 200)

    def test_position_notional_supports_short_risk(self):
        config = ResearchConfig(initial_equity=1000, max_leverage=3, risk_pct=0.01)

        notional = calculate_position_notional(
            equity=1000,
            entry_price=100,
            stop_price=105,
            config=config,
            side="short",
        )

        self.assertEqual(notional, 200)


class PeriodSplitTests(unittest.TestCase):
    def test_split_periods_uses_first_two_thirds_as_training(self):
        start = 0
        end = 90

        train_start, train_end, test_start, test_end = split_periods(start, end)

        self.assertEqual((train_start, train_end), (0, 60))
        self.assertEqual((test_start, test_end), (60, 90))


if __name__ == "__main__":
    unittest.main()
