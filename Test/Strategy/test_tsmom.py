import numpy as np
import pandas as pd
import pytest

from project.Strategy.tsmom import (
    TsmomConfig,
    evaluate_pass,
    metrics,
    monthly_rebalanced,
    run_tsmom,
    signals,
    target_weights,
)


def make(n=900, drift=0.0005, vol=0.01, seed=0, spread=20.0, start="2018-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    c = 100 * np.cumprod(1 + rng.normal(drift, vol, n))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame(
        {
            "open": o,
            "high": np.maximum(o, c),
            "low": np.minimum(o, c),
            "close": c,
            "spread": spread,
        },
        index=idx,
    )


CFG = TsmomConfig(lookback=100, vol_window=30)


def test_signal_sign_follows_trailing_return():
    df = make(drift=0.003, vol=0.002, n=300)
    s = signals(df[["close"]], CFG)["close"]
    assert s.iloc[:100].eq(0).all()  # ช่วงอุ่นเครื่องไม่มีสัญญาณ
    assert s.iloc[150] == 1.0
    down = make(drift=-0.003, vol=0.002, n=300)
    assert signals(down[["close"]], CFG)["close"].iloc[150] == -1.0


def test_weights_scale_inversely_with_volatility_and_are_capped():
    calm, wild = (
        make(vol=0.005, drift=0.001, seed=1),
        make(vol=0.02, drift=0.001, seed=2),
    )
    close = pd.DataFrame({"a": calm["close"], "b": wild["close"]})
    w = target_weights(
        close, TsmomConfig(lookback=100, vol_window=30, target_vol=0.10, max_weight=5)
    )
    assert w["a"].abs().iloc[-1] > w["b"].abs().iloc[-1]
    capped = target_weights(
        close, TsmomConfig(lookback=100, vol_window=30, target_vol=5.0, max_weight=0.3)
    )
    assert capped.abs().max().max() <= 0.3 + 1e-12


def test_monthly_rebalance_holds_weights_within_month():
    idx = pd.bdate_range("2020-01-01", periods=60)
    w = pd.DataFrame({"x": np.arange(60, dtype=float)}, index=idx)
    m = monthly_rebalanced(w)
    feb = m.loc[
        "2020-02-01":"2020-02-27", "x"
    ]  # ถือค่าสิ้นเดือน ม.ค. ตลอดเดือน ก.พ. (จนก่อนวันทำการสุดท้าย)
    assert feb.nunique() == 1
    last_jan = w.loc[:"2020-01-31", "x"].iloc[-1]
    assert feb.iloc[0] == last_jan
    # วันทำการสุดท้ายของ ก.พ. = วันที่รู้น้ำหนักชุดใหม่ (run_tsmom จะ shift 1 วันก่อนใช้จริง)
    assert m.loc["2020-02-28", "x"] == w.loc["2020-02-28", "x"]


def test_no_lookahead_changing_future_does_not_change_past():
    d = {"a": make(seed=3), "b": make(seed=4)}
    pts = {"a": 0.01, "b": 0.01}
    r1 = run_tsmom(d, pts, CFG)["returns"]
    d2 = {k: v.copy() for k, v in d.items()}
    cut = d2["a"].index[600]
    for frame in d2.values():
        frame.loc[frame.index > cut, ["open", "close"]] *= 3.0  # ทำลายอนาคต
    r2 = run_tsmom(d2, pts, CFG)["returns"]
    common = r1.index[r1.index < cut - pd.Timedelta(days=5)]
    pd.testing.assert_series_equal(r1.loc[common], r2.loc[common])


def test_costs_reduce_returns_and_stress_costs_more():
    d = {"a": make(seed=5, spread=30.0), "b": make(seed=6, spread=30.0)}
    pts = {"a": 0.01, "b": 0.01}
    res = run_tsmom(d, pts, CFG)
    assert res["cost"].sum() > 0
    assert res["returns"].sum() < res["gross_returns"].sum()
    stress = run_tsmom(
        d, pts, TsmomConfig(lookback=100, vol_window=30, spread_mult=2.0)
    )
    assert stress["cost"].sum() == pytest.approx(2 * res["cost"].sum(), rel=1e-6)


def test_fix_spread_removes_zero_cost_loophole():
    d = {"a": make(seed=7, spread=0.0)}
    d["a"].loc[d["a"].index[::2], "spread"] = 40.0  # ครึ่งหนึ่งเป็น 0 ครึ่งหนึ่งเป็น 40
    fixed = run_tsmom(d, {"a": 0.01}, CFG, fix_spread=True)["cost"].sum()
    raw = run_tsmom(d, {"a": 0.01}, CFG, fix_spread=False)["cost"].sum()
    assert fixed >= raw


def test_trending_universe_is_profitable_and_pass_report_shape():
    d = {s: make(seed=i, drift=0.0008, vol=0.008, n=1500) for i, s in enumerate("abcd")}
    pts = {s: 0.01 for s in d}
    out = evaluate_pass(d, pts, TsmomConfig(lookback=100, vol_window=30))
    assert out["base"]["return_%"] > 0
    assert set(out["checks"]) and isinstance(out["pass"], bool)
    assert set(out["lookback_stability_sharpe"]) == {126, 189, 252}


def test_late_starting_asset_does_not_change_earlier_weights_or_cut_sample():
    a = make(seed=8)
    b = make(seed=9).iloc[300:]  # b มีประวัติเริ่มทีหลัง
    both = run_tsmom({"a": a, "b": b}, {"a": 0.01, "b": 0.01}, CFG)
    solo = run_tsmom({"a": a}, {"a": 0.01}, CFG)
    before = both["weights"].index < b.index[0]
    assert before.sum() > 50  # ไม่ได้ตัดช่วงของ a ทิ้งตามตัวที่สั้นกว่า
    assert both["weights"].loc[before, "b"].eq(0).all()
    days = both["weights"].index[before]
    pd.testing.assert_series_equal(
        both["weights"].loc[days, "a"], solo["weights"].loc[days, "a"]
    )
    # หลัง b active (มีข้อมูลครบ lookback) น้ำหนักต้องไม่เป็น 0 ตลอด
    late = both["weights"].loc[
        both["weights"].index > b.index[CFG.lookback + CFG.vol_window + 5], "b"
    ]
    assert late.abs().sum() > 0


def test_validates_input():
    a = make(seed=8)
    with pytest.raises(ValueError):
        run_tsmom({}, {}, CFG)
    with pytest.raises(ValueError):
        run_tsmom({"a": a.drop(columns=["open"])}, {"a": 0.01}, CFG)


def test_holiday_gap_in_one_market_means_zero_return_not_nan():
    a, b = make(seed=10), make(seed=11)
    b = b.drop(b.index[400:405])  # b ปิด 5 วัน
    res = run_tsmom({"a": a, "b": b}, {"a": 0.01, "b": 0.01}, CFG)
    assert res["returns"].notna().all()


def test_metrics_handles_flat_series():
    r = pd.Series(0.0, index=pd.bdate_range("2020-01-01", periods=10))
    assert np.isnan(metrics(r)["sharpe"])
