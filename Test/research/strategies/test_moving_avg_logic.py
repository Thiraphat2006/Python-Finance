import unittest

import numpy as np
import pandas as pd

from research.indicators.moving_avg import MovingAvg
from research.strategies.moving_avg_logic import MovingAvgLogic, StrategyConfig


def make_raw(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """ข้อมูลจำลอง: random walk + trend เปลี่ยนเป็นช่วง ๆ"""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.normal(0, 0.4, n // 300 + 1), 300)[:n]
    close = 1500 + np.cumsum(drift + rng.normal(0, 2, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, 1.5, n)
    low = np.minimum(open_, close) - rng.uniform(0, 1.5, n)
    idx = pd.date_range("2022-01-03", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": rng.integers(500, 4000, n),
            "spread": 14,
        },
        index=idx,
    )


class TestMovingAvgLogic(unittest.TestCase):
    def setUp(self):
        self.raw = make_raw()
        self.ind = MovingAvg(self.raw).compute_indicators()

    def test_indicator_does_not_mutate_input(self):
        before = self.raw.copy()
        MovingAvg(self.raw).compute_indicators()
        pd.testing.assert_frame_equal(self.raw, before)

    def test_indicator_has_no_lookahead(self):
        """ตัดข้อมูลอนาคตทิ้ง ค่า indicator ในอดีตต้องเท่าเดิม"""
        full = MovingAvg(self.raw).compute_indicators()
        part = MovingAvg(self.raw.iloc[:2000]).compute_indicators()
        cols = ["SMA_200", "EMA_21", "EMA_50", "VWMA_200", "atr", "adx"]
        pd.testing.assert_frame_equal(
            full.loc[part.index, cols], part[cols], check_exact=False, rtol=1e-9
        )

    def test_short_entries_exist(self):
        """บั๊กเดิม: short_entry เป็น False ตลอดเพราะ filter close > ma200"""
        out = MovingAvgLogic(
            self.ind, StrategyConfig(adx_min=15, use_atr_filter=False)
        )()
        self.assertGreater(out["long_entry"].sum(), 0)
        self.assertGreater(out["short_entry"].sum(), 0)

    def test_entry_only_when_trend_aligned(self):
        out = MovingAvgLogic(self.ind, StrategyConfig(adx_min=15))()
        longs, shorts = out[out.long_entry], out[out.short_entry]
        self.assertTrue((longs.close > longs.SMA_200).all())
        self.assertTrue((longs.EMA_21 > longs.EMA_50).all())
        self.assertTrue((shorts.close < shorts.SMA_200).all())
        self.assertTrue((shorts.EMA_21 < shorts.EMA_50).all())

    def test_long_only_config(self):
        out = MovingAvgLogic(self.ind, StrategyConfig(adx_min=15, allow_short=False))()
        self.assertEqual(out["short_entry"].sum(), 0)

    def test_does_not_mutate_indicator_frame(self):
        cols_before = list(self.ind.columns)
        MovingAvgLogic(self.ind)()
        self.assertEqual(list(self.ind.columns), cols_before)

    def test_ema_cross_exit_is_not_later_than_opposite_trend(self):
        slow = MovingAvgLogic(self.ind, StrategyConfig(exit_mode="opposite_trend"))()
        fast = MovingAvgLogic(self.ind, StrategyConfig(exit_mode="ema_cross"))()
        # ema_cross เป็นเงื่อนไขที่หลวมกว่า จึงต้องเกิดบ่อยกว่าหรือเท่ากัน
        self.assertGreaterEqual(fast["exit_long"].sum(), slow["exit_long"].sum())

    def test_missing_column_raises(self):
        with self.assertRaises(ValueError):
            MovingAvgLogic(self.ind.drop(columns=["EMA_50"]))()

    def test_min_rows(self):
        with self.assertRaises(ValueError):
            MovingAvg(self.raw.head(100)).compute_indicators()


if __name__ == "__main__":
    unittest.main()
