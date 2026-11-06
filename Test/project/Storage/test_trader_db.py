import logging
import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest

from project.Storage.log_setup import setup_logging
from project.Storage.trader_db import SCHEMA_VERSION, TraderDB, iso


@pytest.fixture
def db():
    d = TraderDB(":memory:")
    yield d
    d.close()


def test_schema_created_with_version(db):
    tables = {
        r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"runs", "orders", "deals", "quotes", "account_snapshots", "state"} <= tables
    assert db.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_file_db_is_reopenable_and_persists(tmp_path):
    path = tmp_path / "sub" / "t.db"  # สร้างโฟลเดอร์ให้เอง
    with TraderDB(path) as d:
        d.set_state("k", "v")
    with TraderDB(path) as d:
        assert d.get_state("k") == "v"


def test_newer_schema_is_refused(tmp_path):
    path = tmp_path / "t.db"
    TraderDB(path).close()
    raw = sqlite3.connect(path)
    raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    raw.commit()
    raw.close()
    with pytest.raises(RuntimeError):
        TraderDB(path)


def test_last_processed_bar_ignores_errors(db):
    assert db.last_processed_bar("tsmom", "XAUUSD") is None
    db.log_run("tsmom", "XAUUSD", "2026-10-22", "hold")
    db.log_run("tsmom", "XAUUSD", "2026-10-23", "error", status="error")
    assert db.last_processed_bar("tsmom", "XAUUSD").startswith("2026-10-22")
    assert db.already_processed("tsmom", "XAUUSD", "2026-10-22")
    assert not db.already_processed(
        "tsmom", "XAUUSD", "2026-10-23"
    )  # รอบที่ error ต้องรันใหม่ได้
    assert not db.already_processed("tsmom", "EURUSD", "2026-10-22")


def test_order_requires_existing_run(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.log_order("XAUUSD", "open", "long", 0.01, ok=True, run_id=999)
    run_id = db.log_run("test", "XAUUSD", "2026-10-26", "enter")
    oid = db.log_order(
        "XAUUSD",
        "open",
        "long",
        0.01,
        ok=True,
        run_id=run_id,
        requested_price=2000.0,
        fill_price=2000.3,
        sl=1980.0,
        retcode=10009,
        ticket=123,
    )
    row = db.query("SELECT * FROM orders WHERE order_id=?", (oid,))[0]
    assert row["fill_price"] == 2000.3 and row["ok"] == 1


def test_order_accepts_uuid_string_ticket(db):
    run_id = db.log_run("test", "XAUUSD", "2026-10-26", "enter")
    oid = db.log_order(
        "XAUUSD", "open", "long", 0.01, ok=True, run_id=run_id, ticket="abc-uuid"
    )
    assert db.query("SELECT ticket FROM orders WHERE order_id=?", (oid,))[0][
        "ticket"
    ] == ("abc-uuid")


def test_log_deal_idempotent_and_accepts_namedtuple_like(db):
    deal = SimpleNamespace(
        ticket=555,
        position_id=123,
        symbol="XAUUSD",
        time=1_761_000_000,
        type=1,
        entry=1,
        volume=0.01,
        price=2010.5,
        profit=10.0,
        commission=-0.07,
        swap=-0.2,
        fee=0.0,
        reason=4,
    )
    assert db.log_deal(deal) is True
    assert db.log_deal(deal) is False  # log ซ้ำ ไม่เพิ่มแถว
    rows = db.query("SELECT * FROM deals")
    assert len(rows) == 1 and rows[0]["commission"] == -0.07
    assert rows[0]["time"].startswith("2025-")  # epoch วินาที -> ISO


def test_quote_spread_points_and_dedupe(db):
    db.log_quote("XAUUSD", 2000.00, 2000.25, point=0.01, ts="2026-10-26 04:05:00")
    db.log_quote(
        "XAUUSD", 2000.00, 2000.30, point=0.01, ts="2026-10-26 04:05:00"
    )  # เวลาเดิม ข้าม
    rows = db.query("SELECT * FROM quotes")
    assert len(rows) == 1 and rows[0]["spread_points"] == pytest.approx(25.0)


def test_snapshot_upserts_per_day(db):
    db.log_snapshot(10_000, 10_000, 0, ts="2026-10-26 09:00:00")
    db.log_snapshot(10_000, 9_950, 1, ts="2026-10-26 21:00:00")
    db.log_snapshot(9_950, 9_950, 0, ts="2026-10-27 09:00:00")
    rows = db.query("SELECT * FROM account_snapshots ORDER BY snap_date")
    assert len(rows) == 2 and rows[0]["equity"] == 9_950


def test_state_roundtrip_and_overwrite(db):
    assert db.get_state("killed", "0") == "0"
    db.set_state("peak_equity", 10_500.5)
    db.set_state("peak_equity", 10_800)
    assert float(db.get_state("peak_equity")) == 10_800


def test_iso_handles_types():
    assert iso(pd.Timestamp("2026-10-26 04:00")).startswith("2026-10-26T04:00")
    assert iso(0).startswith("1970-01-01")
    assert len(iso(None)) >= 19


def test_setup_logging_writes_file_and_is_idempotent(tmp_path):
    name = "autotrader_test"
    log = setup_logging(tmp_path, name=name)
    again = setup_logging(tmp_path, name=name)
    assert log is again and len(log.handlers) == 2
    log.info("สวัสดี")
    for h in log.handlers:
        h.flush()
    text = (tmp_path / f"{name}.log").read_text(encoding="utf-8")
    assert "สวัสดี" in text
    for h in list(log.handlers):  # ปิดไฟล์ (Windows ลบ tmp ไม่ได้ถ้ายังเปิดอยู่)
        h.close()
        log.removeHandler(h)
    logging.getLogger(name)._autotrader_configured = False
