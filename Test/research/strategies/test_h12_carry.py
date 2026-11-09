import numpy as np
import pandas as pd
import pytest

from research.strategies.h12_carry import (
    PAIRS,
    carry_unit,
    placebo_sharpes,
    rank_signal,
    run_h12,
)

CCYS = ["EUR", "JPY", "GBP", "AUD", "CAD", "NZD", "CHF", "NOK", "SEK"]
POINTS = {s: 0.00001 for s in PAIRS}


def _make_data(end: str = "2013-12-31", seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-01", end)
    out = {}
    for sym in PAIRS:
        close = 1.2 * np.exp(np.cumsum(rng.normal(0, 0.006, len(idx))))
        open_ = np.r_[close[0], close[:-1]]
        out[sym] = pd.DataFrame(
            {"open": open_, "close": close, "spread": 20.0}, index=idx
        )
    return out


def _random_rates(end: str = "2013-12", seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.period_range("2009-01", end, freq="M")
    return pd.DataFrame(
        rng.uniform(0, 6, (len(months), 10)), index=months, columns=[*CCYS, "USD"]
    )


def _fixed_rates(end: str = "2013-12") -> pd.DataFrame:
    months = pd.period_range("2009-01", end, freq="M")
    level = dict(
        zip([*CCYS, "USD"], [5.0, 1.0, 4.5, 4.0, 3.0, 3.5, 1.5, 2.5, 2.0, 0.5])
    )
    return pd.DataFrame({c: [v] * len(months) for c, v in level.items()}, index=months)


def test_rank_signal_top_bottom_and_tie_alphabetical() -> None:
    row = pd.Series({"A": 5.0, "B": 5.0, "C": 3.0, "D": 2.0, "E": 1.0, "F": 0.0})
    sig = rank_signal(row, 2)
    assert sig["A"] == 1.0 and sig["B"] == 1.0
    assert sig["E"] == -1.0 and sig["F"] == -1.0
    assert sig["C"] == 0.0 and sig["D"] == 0.0
    tied = rank_signal(pd.Series({"B": 1.0, "A": 1.0, "C": 1.0, "D": 1.0}), 1)
    assert tied["A"] == 1.0 and tied["B"] == -1.0
    assert (tied == 1.0).sum() == 1 and (tied == -1.0).sum() == 1


def test_rank_signal_nan_gives_zero() -> None:
    row = pd.Series({"A": 1.0, "B": np.nan, "C": 2.0, "D": 3.0})
    assert (rank_signal(row, 1) == 0.0).all()


def test_carry_unit_uses_previous_month_and_sums_to_monthly_carry() -> None:
    idx = pd.bdate_range("2012-01-02", "2012-03-30")
    months = pd.period_range("2011-12", "2012-03", freq="M")
    rates = pd.DataFrame(0.0, index=months, columns=["EUR", "USD"])
    rates.loc[pd.Period("2012-01", "M"), "EUR"] = 4.0
    rates.loc[pd.Period("2012-01", "M"), "USD"] = 1.0
    unit = carry_unit(rates, ["EUR"], idx)
    assert unit.loc["2012-01", "EUR"].sum() == 0.0  # ใช้ดอกเบี้ยเดือน 2011-12 = 0
    assert unit.loc["2012-02", "EUR"].sum() == pytest.approx(0.03 / 12)
    assert unit.loc["2012-03", "EUR"].sum() == 0.0  # ใช้ดอกเบี้ยเดือน 2012-02 = 0


def test_pair_direction_usd_base_is_opposite() -> None:
    res = run_h12(_make_data(), _fixed_rates(), POINTS)
    w, pw = res["weights"], res["pair_weights"]
    assert (w["EUR"] > 0).all() and (w["JPY"] < 0).all()
    assert (w["NOK"] == 0).all()
    assert (pw["EUR"] > 0).all()  # long EUR = long EURUSD
    assert (pw["JPY"] > 0).all()  # short JPY = long USDJPY


def test_no_lookahead_truncation() -> None:
    data, rates = _make_data(), _random_rates()
    full = run_h12(data, rates, POINTS)
    cut = pd.Timestamp("2012-06-29")  # วันทำการสุดท้ายของ 2012-06
    data_cut = {
        s: df.loc[df.index <= cut + pd.Timedelta(days=5)] for s, df in data.items()
    }
    rates_cut = rates.loc[rates.index <= pd.Period("2012-06", "M")]
    part = run_h12(data_cut, rates_cut, POINTS)
    upto = pd.Timestamp("2012-07-02")  # วันแรกที่ใช้น้ำหนักที่ตัดสินใจ ณ cut
    pd.testing.assert_frame_equal(
        full["weights"].loc[:upto], part["weights"].loc[:upto], check_freq=False
    )


def test_holdout_hard_cap() -> None:
    data = _make_data(end="2025-12-31")
    res = run_h12(data, _random_rates(end="2025-12"), POINTS)
    assert res["returns"].index.max() <= pd.Timestamp("2023-12-31")


def test_placebo_is_deterministic() -> None:
    data, rates = _make_data(), _random_rates()
    a = placebo_sharpes(data, rates, POINTS, n=3)
    b = placebo_sharpes(data, rates, POINTS, n=3)
    np.testing.assert_array_equal(a, b)
