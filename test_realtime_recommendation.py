import unittest

from realtime_recommendation import calculate_entry_zone, combine_scores, entry_side, recommendation_label, wick_ratio


class RecommendationScoreTests(unittest.TestCase):
    def test_combine_scores_maps_strong_long_to_high_value(self):
        self.assertEqual(combine_scores(long_score=80, short_score=0), 90)

    def test_combine_scores_maps_strong_short_to_low_value(self):
        self.assertEqual(combine_scores(long_score=0, short_score=80), 10)

    def test_combine_scores_is_neutral_when_no_edge(self):
        self.assertEqual(combine_scores(long_score=0, short_score=0), 50)

    def test_recommendation_label_uses_clear_zones(self):
        self.assertEqual(recommendation_label(90), "强烈偏多")
        self.assertEqual(recommendation_label(10), "强烈偏空")
        self.assertEqual(recommendation_label(50), "中性观望")


class WickRatioTests(unittest.TestCase):
    def test_lower_wick_ratio(self):
        ratio = wick_ratio(open_price=100, high=110, low=90, close=105, side="long")

        self.assertAlmostEqual(ratio, 0.5)

    def test_upper_wick_ratio(self):
        ratio = wick_ratio(open_price=105, high=110, low=90, close=100, side="short")

        self.assertAlmostEqual(ratio, 0.25)


class EntryPlanTests(unittest.TestCase):
    def test_entry_side_only_trades_strong_zones(self):
        self.assertEqual(entry_side(80), "long")
        self.assertEqual(entry_side(20), "short")
        self.assertIsNone(entry_side(60))

    def test_long_entry_zone_waits_for_small_pullback(self):
        zone = calculate_entry_zone("long", signal_price=100, atr=10)

        self.assertEqual(zone.ideal, 99)
        self.assertEqual(zone.acceptable_low, 97)
        self.assertEqual(zone.acceptable_high, 101)

    def test_short_entry_zone_waits_for_small_rebound(self):
        zone = calculate_entry_zone("short", signal_price=100, atr=10)

        self.assertEqual(zone.ideal, 101)
        self.assertEqual(zone.acceptable_low, 99)
        self.assertEqual(zone.acceptable_high, 103)


if __name__ == "__main__":
    unittest.main()
