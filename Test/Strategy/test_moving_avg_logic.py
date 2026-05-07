import unittest

import pandas as pd

from project.strategy.Moving_Avg_Logic import (
    StrategyConfig,
    _calculate_scores,
    _decide_position_action,
    moveingAvgPerdiction,
    moving_avg_prediction,
)


class TestMovingAvgLogic(unittest.TestCase):
    def setUp(self):
        self.base_df: tuple[pd.DataFrame, str, int] = (
            pd.DataFrame(
                {
                    "close": [100.0, 101.0],
                    "EMA_21": [100.5, 100.8],
                    "SMA_200": [99.0, 100.0],
                    "EMA_50": [100.0, 100.7],
                    "VWMA_200": [99.0, 100.0],
                }
            ),
            "XAUUSD",
            60,
        )

    def test_calculate_scores_returns_expected_keys(self):
        scores = _calculate_scores(self.base_df[0], StrategyConfig())
        self.assertEqual(set(scores.keys()), {"sma", "ema21", "ema50", "vwma"})

    def test_open_long_when_score_reaches_threshold(self):
        result = moving_avg_prediction(self.base_df, current_position=0, verbose=False)
        self.assertEqual(result["position"], 1)
        self.assertEqual(result["action"], "open_long")
        self.assertGreaterEqual(result["total_score"], 3)

    def test_close_long_when_score_non_positive(self):
        position, action = _decide_position_action(
            total_score=0, current_position=1, open_position_threshold=3
        )
        self.assertEqual(position, 0)
        self.assertEqual(action, "close_long")

    def test_legacy_wrapper_returns_int_score(self):
        total_score = moveingAvgPerdiction(self.base_df[1])
        self.assertIsInstance(total_score, int)

    def test_validate_min_rows(self):
        one_row_df = self.base_df[0].head(1)
        with self.assertRaises(ValueError):
            moving_avg_prediction(one_row_df, verbose=False)


if __name__ == "__main__":
    unittest.main()
