import logging

import pandas as pd
import pytest

from project.Execution.capital_data import CapitalData, build_d1_bars
from project.Risk.position_sizing import SymbolSpec

RAW = [  # time, O, H, L, C (bid) จาก probe 2026-11-04; ask ไม่ใช้ (ใส่ bid+0.5)
    ("2026-10-22", 4125.22, 4125.95, 4020.10, 4023.61),
    ("2026-10-23", 4023.48, 4033.99, 3975.24, 3995.27),
    ("2026-10-25", 3998.19, 4006.74, 3985.84, 3997.37),
    ("2026-10-26", 3997.27, 4029.72, 3977.55, 4005.40),
    ("2026-10-27", 4005.38, 4065.52, 3988.95, 4062.62),
    ("2026-10-28", 4062.53, 4082.59, 4024.44, 4054.18),
    ("2026-10-29", 4054.23, 4068.69, 4005.06, 4007.74),
    ("2026-10-30", 4007.75, 4018.71, 3869.79, 3875.20),
    ("2026-11-01", 3882.24, 3901.95, 3880.69, 3898.09),
    ("2026-11-02", 3898.10, 3970.04, 3890.95, 3947.91),
    ("2026-11-03", 3947.90, 3970.98, 3895.29, 3938.06),
    ("2026-11-04", 3938.17, 4049.41, 3919.69, 4040.30),
]


def _prices(rows=RAW):
    def px(v):
        return {"bid": v, "ask": round(v + 0.5, 2)}

    return [
        {
            "snapshotTimeUTC": f"{t}T00:00:00",
            "openPrice": px(o),
            "highPrice": px(h),
            "lowPrice": px(lo),
            "closePrice": px(c),
        }
        for t, o, h, lo, c in rows
    ]


def test_merge_sunday_and_cut_today():
    df = build_d1_bars(_prices(), pd.Timestamp("2026-11-04 15:25"))
    assert [d.strftime("%m-%d") for d in df.index] == [
        "10-22", "10-23", "10-26", "10-27", "10-28", "10-29", "10-30", "11-02", "11-03",
    ]  # fmt: skip
    mon = df.loc["2026-11-02"]
    assert (mon.open, mon.high, mon.low, mon.close) == (
        3882.24,
        3970.04,
        3880.69,
        3947.91,
    )
    assert "atr" in df.columns


def test_monday_run_last_bar_is_friday():
    df = build_d1_bars(_prices(), pd.Timestamp("2026-11-02 00:30"))
    assert df.index[-1] == pd.Timestamp("2026-10-30")


def test_count_limits_bars():
    df = build_d1_bars(_prices(), pd.Timestamp("2026-11-04 15:25"), count=3)
    assert len(df) == 3 and df.index[-1] == pd.Timestamp("2026-11-03")


def test_orphan_sunday_dropped_with_warning(caplog):
    rows = [RAW[8], RAW[10]]  # 11-01 (อาทิตย์) ไม่มี 11-02 + 11-03
    with caplog.at_level(logging.WARNING, logger="autotrader"):
        df = build_d1_bars(_prices(rows), pd.Timestamp("2026-11-04 15:25"))
    assert list(df.index) == [pd.Timestamp("2026-11-03")]
    assert "อาทิตย์" in caplog.text


class FakeClient:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, path, params=None):
        self.calls.append((path, params))
        return self.routes[path]


SPEC = SymbolSpec("GOLD", point=0.01, contract_size=1.0, volume_min=0.01,
                  volume_step=0.01, volume_max=50000.0)  # fmt: skip
NOW_MS = 1793806174185  # 2026-11-04 15:29:34.185Z


def _data(snapshot, now_ms=NOW_MS):
    routes = {"/time": {"serverTime": now_ms}, "/markets/GOLD": {"snapshot": snapshot}}
    return CapitalData(FakeClient(routes), "GOLD", SPEC)


SNAP = {"marketStatus": "TRADEABLE", "updateTime": "2026-11-04T22:29:34.359",
        "bid": 4034.02, "offer": 4034.52}  # fmt: skip


def test_server_now_and_quote():
    d = _data(SNAP)
    assert d.get_server_now() == pd.Timestamp("2026-11-04 15:29:34.185")
    assert d.get_quote() == (4034.02, 4034.52)


def test_market_open_fresh_tick():
    assert _data(SNAP).is_market_open() is True


@pytest.mark.parametrize(
    "patch",
    [
        {"marketStatus": "CLOSED"},
        {"updateTime": "2026-11-04T22:20:00.000"},  # tick เก่า ~574 วินาที
        {"bid": 4000.0, "offer": 4006.0},  # spread 600 points
        {"updateTime": None},
    ],
)
def test_market_not_open(patch):
    assert _data({**SNAP, **patch}).is_market_open() is False


def test_fetch_d1_bars_requests_day_prices():
    routes = {"/time": {"serverTime": NOW_MS}, "/prices/GOLD": {"prices": _prices()}}
    client = FakeClient(routes)
    df = CapitalData(client, "GOLD", SPEC).fetch_d1_bars(count=120)
    assert client.calls[-1] == ("/prices/GOLD", {"resolution": "DAY", "max": 157})
    assert len(df) == 9


def test_log_recent_deals_noop():
    d = _data(SNAP)
    assert d.log_recent_deals(db=None) == 0
    assert d.client.calls == []


def test_rejects_zero_point():
    bad = SymbolSpec("GOLD", point=0.0, contract_size=1.0, volume_min=0.01,
                     volume_step=0.01, volume_max=50000.0)  # fmt: skip
    with pytest.raises(ValueError):
        CapitalData(FakeClient({}), "GOLD", bad)
