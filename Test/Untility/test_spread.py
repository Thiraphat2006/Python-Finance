import pandas as pd
import pytest

from project.Untility.spread import fix_zero_spread


def frame(spreads, start="2019-12-30", freq="D"):
    idx = pd.date_range(start, periods=len(spreads), freq=freq)
    return pd.DataFrame({"spread": spreads, "open": 100.0}, index=idx)


def test_replaces_zero_with_same_year_median():
    # 2019: 2 แท่ง (20, 0)  2020: 4 แท่ง (40, 0, 60, 0)
    df = frame([20, 0, 40, 0, 60, 0])
    out, info = fix_zero_spread(df)
    assert out["spread"].tolist() == [20, 20, 40, 50, 60, 50]
    assert info["replaced_bars"] == 3 and info["fixed"] is True
    assert info["zero_before_%"] == 50.0


def test_year_with_no_valid_value_falls_back_to_overall_median():
    df = frame([0, 0, 30, 50])  # ปี 2019 เป็น 0 หมด -> ใช้ median ทั้งช่วง (=40)
    out, _ = fix_zero_spread(df)
    assert out["spread"].tolist()[:2] == [40, 40]


def test_never_lowers_cost_and_keeps_original_untouched():
    df = frame([10, 0, 30, 0, 25])
    before = df.copy()
    out, _ = fix_zero_spread(df)
    assert (out["spread"] >= df["spread"]).all()
    pd.testing.assert_frame_equal(df, before)


def test_no_zero_or_no_column_is_noop():
    df = frame([10, 20, 30])
    out, info = fix_zero_spread(df)
    assert out is df and info["fixed"] is False
    out2, info2 = fix_zero_spread(df.drop(columns=["spread"]))
    assert "spread" not in out2.columns and info2["fixed"] is False


def test_all_zero_raises():
    with pytest.raises(ValueError):
        fix_zero_spread(frame([0, 0, 0]))
