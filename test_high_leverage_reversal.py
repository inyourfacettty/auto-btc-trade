import unittest

from high_leverage_reversal import HighLeverageConfig, liquidation_price, position_notional


class HighLeverageRiskTests(unittest.TestCase):
    def test_long_liquidation_price_gets_closer_at_higher_leverage(self):
        liq_20 = liquidation_price(100, 20, "long")
        liq_100 = liquidation_price(100, 100, "long")

        self.assertAlmostEqual(liq_20, 95.5)
        self.assertAlmostEqual(liq_100, 99.1)

    def test_short_liquidation_price_gets_closer_at_higher_leverage(self):
        liq_20 = liquidation_price(100, 20, "short")
        liq_100 = liquidation_price(100, 100, "short")

        self.assertAlmostEqual(liq_20, 104.5)
        self.assertAlmostEqual(liq_100, 100.9)

    def test_position_notional_uses_margin_fraction_times_leverage(self):
        config = HighLeverageConfig(initial_equity=1000, leverage=50, margin_pct=0.05)

        self.assertEqual(position_notional(1000, config), 2500)


if __name__ == "__main__":
    unittest.main()
