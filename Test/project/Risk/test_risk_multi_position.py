import math
from types import SimpleNamespace

import pandas as pd
import pytest

from project.Risk.position_sizing import SymbolSpec
from project.Risk.risk_engine import (
    RiskConfig,
    RiskEngine,
    load_engine,
    open_risk_fraction,
    position_risk_money,
    save_engine,
)
from project.Storage.trader_db import TraderDB

GOLD = SymbolSpec("XAUUSD", point=0.01, contract_size=100)


def ts(day, hour=0):
    return pd.Timestamp(f"2026-10-{day:02d} {hour:02d}:00")


def pos(side, lots, entry, stop):
    return SimpleNamespace(side=side, lots=lots, entry=entry, stop=stop)


# ---------- หลายสถานะ + เพดานรวม ----------
def test_total_risk_cap_blocks_when_exceeded():
    e = RiskEngine(RiskConfig(max_open_positions=5, max_total_risk=0.02), 10_000)
    e.update(ts(1), 10_000)
    # เปิดอยู่ 1.75% + ใหม่ 0.5% = 2.25% > 2% -> ปฏิเสธ
    assert e.check_entry(10_000, 3, open_risk=0.0175).reason == "max_total_risk"
    # 1.5% + 0.5% = 2.0% พอดี -> ผ่าน
    assert e.check_entry(10_000, 3, open_risk=0.015).allowed


def test_new_risk_override_used_for_cap():
    e = RiskEngine(RiskConfig(max_open_positions=5, max_total_risk=0.02), 10_000)
    e.update(ts(1), 10_000)
    assert not e.check_entry(10_000, 1, open_risk=0.01, new_risk=0.015).allowed
    assert e.check_entry(10_000, 1, open_risk=0.01, new_risk=0.01).allowed


def test_multiple_positions_when_limit_raised():
    e = RiskEngine(RiskConfig(max_open_positions=3), 10_000)
    e.update(ts(1), 10_000)
    assert e.check_entry(10_000, 2, open_risk=0.01).allowed
    assert e.check_entry(10_000, 3, open_risk=0.01).reason == "max_open_positions"


def test_one_position_per_symbol():
    e = RiskEngine(RiskConfig(max_open_positions=5), 10_000)
    e.update(ts(1), 10_000)
    d = e.check_entry(10_000, 1, symbol="XAUUSD", open_symbols={"XAUUSD"})
    assert d.reason == "symbol_already_open"
    assert e.check_entry(10_000, 1, symbol="EURUSD", open_symbols={"XAUUSD"}).allowed


def test_defaults_unchanged_single_position():
    e = RiskEngine(RiskConfig(), 10_000)
    e.update(ts(1), 10_000)
    assert e.cfg.max_open_positions == 1
    assert e.check_entry(10_000, 1).reason == "max_open_positions"


# ---------- คำนวณความเสี่ยงของสถานะที่ถืออยู่ ----------
def test_position_risk_long_short_and_locked_profit():
    assert position_risk_money(pos("long", 0.1, 2000, 1980), GOLD) == pytest.approx(
        200.0
    )
    assert position_risk_money(pos("short", 0.1, 2000, 2020), GOLD) == pytest.approx(
        200.0
    )
    # stop เลื่อนไปอยู่ฝั่งกำไรแล้ว ไม่มีความเสี่ยงเหลือ (ไม่เป็นค่าลบ)
    assert position_risk_money(pos("long", 0.1, 2000, 2010), GOLD) == 0.0


def test_position_without_stop_has_unlimited_risk():
    assert position_risk_money(pos("long", 0.1, 2000, None), GOLD) == math.inf


def test_open_risk_fraction_sums_positions():
    items = [
        (pos("long", 0.1, 2000, 1990), GOLD),
        (pos("short", 0.05, 2000, 2020), GOLD),
    ]
    # 100 + 100 = 200 จาก equity 10,000 = 2%
    assert open_risk_fraction(items, 10_000) == pytest.approx(0.02)
    assert open_risk_fraction([], 10_000) == 0.0
    assert open_risk_fraction(items, 0) == 0.0


def test_open_risk_without_stop_blocks_new_entry():
    items = [(pos("long", 0.1, 2000, None), GOLD)]
    assert open_risk_fraction(items, 10_000) == math.inf
    engine = RiskEngine(RiskConfig(max_total_risk=0.02), 10_000)
    assert not engine.check_entry(
        10_000,
        1,
        open_risk=open_risk_fraction(items, 10_000),
    ).allowed


# ---------- บันทึก/โหลดสถานะข้ามการรัน ----------
def test_export_restore_roundtrip():
    e = RiskEngine(RiskConfig(), 10_000)
    e.update(ts(1), 10_500)
    f = RiskEngine(RiskConfig(), 1)  # ค่าตั้งต้นต่างกันโดยตั้งใจ
    f.restore_state(e.export_state())
    assert f.peak_equity == 10_500 and f.day_start_equity == 10_500
    assert f.day == e.day and not f.killed


def test_kill_switch_survives_restart_via_db():
    cfg = RiskConfig(max_drawdown_kill=0.10, daily_loss_limit=0.5)
    with TraderDB(":memory:") as db:
        e = load_engine(db, cfg, 10_000)  # ครั้งแรก: สร้างใหม่
        e.update(ts(1), 10_000)
        e.update(ts(2), 8_900)  # DD 11% -> kill
        assert e.killed
        save_engine(db, e)

        # "รีสตาร์ท" วันถัดมา: equity ฟื้นแล้ว แต่ kill ต้องยังอยู่
        e2 = load_engine(db, cfg, 10_500)
        assert e2.peak_equity == 10_000  # peak ไม่ถูกรีเซ็ตเป็น equity ใหม่ตอนโหลด
        e2.update(ts(3), 10_500)
        assert e2.killed and "max drawdown" in e2.kill_reason
        assert not e2.check_entry(10_500, 0).allowed


def test_reset_kill_then_save_clears_latch():
    cfg = RiskConfig(max_drawdown_kill=0.10, daily_loss_limit=0.5)
    with TraderDB(":memory:") as db:
        e = load_engine(db, cfg, 10_000)
        e.update(ts(1), 10_000)
        e.update(ts(2), 8_900)
        save_engine(db, e)
        e.reset_kill()
        save_engine(db, e)
        assert not load_engine(db, cfg, 8_900).killed


def test_peak_and_daily_base_persist_same_day_and_reset_next_day():
    cfg = RiskConfig(daily_loss_limit=0.02)
    with TraderDB(":memory:") as db:
        e = load_engine(db, cfg, 10_000)
        e.update(ts(1, 0), 10_000)
        save_engine(db, e)

        # รันอีกรอบในวันเดียวกัน: ฐาน daily ต้องเป็น 10,000 เดิม ไม่ใช่ equity ปัจจุบัน
        e2 = load_engine(db, cfg, 9_790)
        e2.update(ts(1, 9), 9_790)
        assert e2.daily_limit_hit(9_790)

        # วันใหม่: ฐานถูกรีเซ็ต
        e2.update(ts(2, 0), 9_790)
        assert not e2.daily_limit_hit(9_790)


def test_load_engine_fresh_when_nothing_saved():
    with TraderDB(":memory:") as db:
        e = load_engine(db, RiskConfig(), 12_345)
        assert e.peak_equity == 12_345 and e.day is None and not e.killed
