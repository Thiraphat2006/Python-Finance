import pandas as pd
import pytest

from project.Execution.paper_broker import PaperBroker
from project.Execution.replay import run_paper
from project.Risk.position_sizing import SymbolSpec, atr_stop, position_size
from project.Risk.risk_engine import RiskConfig, RiskEngine

GOLD = SymbolSpec(
    "XAUUSD", point=0.01, contract_size=100, volume_min=0.01, volume_step=0.01
)


# ---------- position sizing ----------
def test_atr_stop_long_short():
    assert atr_stop(100, 5, "long", 2) == 90
    assert atr_stop(100, 5, "short", 2) == 110
    with pytest.raises(ValueError):
        atr_stop(100, 0, "long")


def test_position_size_rounds_down():
    # เสี่ยง 50 ดอลลาร์, ระยะ stop 20 -> 50/(20*100)=0.025 -> ปัดลง 0.02
    assert position_size(10_000, 0.005, 100, 80, GOLD) == 0.02


def test_position_size_zero_when_below_min_lot():
    # equity เล็กมาก ไม้เล็กสุดยังเสี่ยงเกิน -> ไม่เข้า (ไม่ปัดขึ้น)
    assert position_size(100, 0.005, 100, 80, GOLD) == 0.0


def test_position_size_invalid_inputs():
    assert position_size(0, 0.005, 100, 90, GOLD) == 0.0
    assert position_size(10_000, 0.005, 100, 100, GOLD) == 0.0


# ---------- risk engine ----------
def ts(day, hour=0):
    return pd.Timestamp(f"2024-01-{day:02d} {hour:02d}:00")


def test_daily_limit_blocks_then_resets_next_day():
    e = RiskEngine(RiskConfig(daily_loss_limit=0.02), 10_000)
    e.update(ts(1), 10_000)
    assert e.check_entry(10_000, 0).allowed
    e.update(ts(1, 5), 9_790)  # -2.1%
    d = e.check_entry(9_790, 0)
    assert not d.allowed and d.reason == "daily_loss_limit"
    assert e.should_flatten(9_790)
    e.update(ts(2), 9_790)  # วันใหม่ ฐานใหม่
    assert e.check_entry(9_790, 0).allowed


def test_kill_switch_latches_until_manual_reset():
    e = RiskEngine(RiskConfig(max_drawdown_kill=0.10, daily_loss_limit=0.5), 10_000)
    e.update(ts(1), 10_000)
    e.update(ts(2), 8_900)  # DD 11%
    assert e.killed
    e.update(ts(3), 10_500)  # equity ฟื้นก็ไม่ปลด
    assert e.killed and not e.check_entry(10_500, 0).allowed
    e.reset_kill()
    assert e.check_entry(10_500, 0).allowed


def test_max_positions_and_spread_filter():
    e = RiskEngine(RiskConfig(max_open_positions=1, max_spread_points=50), 10_000)
    e.update(ts(1), 10_000)
    assert e.check_entry(10_000, 1).reason == "max_open_positions"
    assert e.check_entry(10_000, 0, spread_points=80).reason == "spread_too_wide"


# ---------- paper broker ----------
def test_broker_pnl_with_spread_cost():
    b = PaperBroker(GOLD, 10_000)
    b.open("long", 0.1, 100.0, 0.5, 90.0, ts(1))  # เข้า 100.5
    t = b.close(110.0, 0.5, ts(2), "signal")  # ออก 109.5
    assert t.pnl == pytest.approx(9.0 * 0.1 * 100)
    assert b.cash == pytest.approx(10_090)


def test_broker_stop_gap_fills_at_open():
    b = PaperBroker(GOLD, 10_000)
    b.open("long", 0.1, 100.0, 0.0, 90.0, ts(1))
    t = b.check_stop(o=85.0, h=86.0, l=80.0, half_spread=0.0, ts=ts(2))
    assert t is not None and t.exit == 85.0  # gap ผ่าน stop เสียหนักกว่า 90


# ---------- replay ----------
def synthetic(n=40):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "atr": 5.0},
        index=idx,
    )
    for c in ("long_entry", "short_entry", "exit_long", "exit_short"):
        df[c] = False
    return df


def test_replay_stop_loss_equals_risk_fraction():
    df = synthetic()
    df.loc[df.index[10], "long_entry"] = True  # เข้าที่ open แท่ง 11
    df.loc[df.index[12], "low"] = 80.0  # ชน stop ที่ 90
    r = run_paper(df, GOLD, RiskConfig(risk_per_trade=0.005), 10_000)
    assert len(r.trades) == 1
    t = r.trades.iloc[0]
    assert t["reason"] == "stop"
    assert t["pnl"] == pytest.approx(-50.0)  # 0.5% ของ 10,000 พอดี (spread = 0)
    assert r.equity.iloc[-1] == pytest.approx(9_950)


def test_replay_rejects_after_daily_limit():
    df = synthetic(60)
    # วันแรก: เทรดแรกขาดทุน 0.5% x ... ใช้ risk 1.5% + daily limit 1% -> ชน limit ทันทีหลัง stop
    df.loc[df.index[2], "long_entry"] = True
    df.loc[df.index[4], "low"] = 80.0
    df.loc[df.index[8], "long_entry"] = True  # ควรถูกปฏิเสธ (วันเดียวกัน)
    r = run_paper(
        df, GOLD, RiskConfig(risk_per_trade=0.015, daily_loss_limit=0.01), 10_000
    )
    assert len(r.trades) == 1
    assert r.rejections.get("daily_loss_limit") == 1


def test_replay_requires_columns():
    with pytest.raises(ValueError):
        run_paper(synthetic().drop(columns=["atr"]), GOLD)
