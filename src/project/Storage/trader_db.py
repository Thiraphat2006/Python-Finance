"""SQLite journal: เก็บเฉพาะสิ่งที่ดึงย้อนหลังจาก MT5 ไม่ได้

- แท่งราคา OHLC ไม่เก็บ (ดึงซ้ำจาก MT5 ได้เสมอ)
- ทุกเมธอด log_* commit ทันที (รันวันละครั้ง ไม่ต้องกังวลเรื่องความเร็ว)
- เวลาเก็บเป็นข้อความ ISO 8601 (ตัวเลข = epoch วินาที ตามที่ MT5 ให้มา)
- ไม่มี dependency เพิ่ม: sqlite3 มากับ Python
"""

import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import pandas as pd

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at     TEXT NOT NULL,
    strategy       TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    bar_time       TEXT NOT NULL,      -- เวลาเปิดของแท่งล่าสุดที่ปิดแล้วและถูกใช้ตัดสินใจ
    signal         TEXT,               -- เช่น long / short / flat
    target_weight  REAL,
    lots_requested REAL,
    decision       TEXT NOT NULL,      -- enter / exit / hold / blocked / error
    reason         TEXT NOT NULL DEFAULT 'ok',   -- เหตุผลจาก risk engine ถ้าถูกบล็อก
    status         TEXT NOT NULL DEFAULT 'done'  -- done / error
);
CREATE INDEX IF NOT EXISTS idx_runs_key ON runs(strategy, symbol, bar_time);

CREATE TABLE IF NOT EXISTS orders (
    order_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES runs(run_id),
    logged_at       TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    kind            TEXT NOT NULL,     -- open / close
    side            TEXT NOT NULL,     -- long / short
    lots            REAL NOT NULL,
    requested_price REAL,
    fill_price      REAL,
    sl              REAL,
    retcode         INTEGER,
    ticket          INTEGER,
    comment         TEXT,
    ok              INTEGER NOT NULL   -- 1 = broker รับ, 0 = ถูกปฏิเสธ/ผิดพลาด
);
CREATE INDEX IF NOT EXISTS idx_orders_run ON orders(run_id);

CREATE TABLE IF NOT EXISTS deals (
    deal_ticket     INTEGER PRIMARY KEY,   -- ticket ของ broker: ซ้ำไม่ได้ จึง log ซ้ำได้ปลอดภัย
    position_ticket INTEGER,
    symbol          TEXT NOT NULL,
    time            TEXT NOT NULL,
    type            INTEGER,               -- รหัสดิบจาก MT5 (DEAL_TYPE_*)
    entry           INTEGER,               -- รหัสดิบ (DEAL_ENTRY_IN / OUT)
    volume          REAL,
    price           REAL,
    profit          REAL,
    commission      REAL,
    swap            REAL,
    fee             REAL,
    reason          INTEGER,               -- รหัสดิบ (DEAL_REASON_*)
    logged_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_deals_pos ON deals(position_ticket);

CREATE TABLE IF NOT EXISTS quotes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    bid           REAL NOT NULL,
    ask           REAL NOT NULL,
    spread_points REAL NOT NULL,
    UNIQUE(symbol, ts)
);

CREATE TABLE IF NOT EXISTS account_snapshots (
    snap_date      TEXT PRIMARY KEY,       -- 1 แถวต่อวัน (รันซ้ำวันเดียวกันจะเขียนทับ)
    ts             TEXT NOT NULL,
    balance        REAL NOT NULL,
    equity         REAL NOT NULL,
    open_positions INTEGER NOT NULL
);

-- สถานะที่ต้องอยู่ข้ามการรัน เช่น peak equity, kill switch (RiskEngine อยู่ในหน่วยความจำ รีสตาร์ททุกวัน)
CREATE TABLE IF NOT EXISTS state (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

Timeish = None | str | int | float | datetime | pd.Timestamp


def iso(ts: Timeish = None) -> str:
    """แปลงเวลาเป็น ISO 8601; None = ตอนนี้ (UTC); ตัวเลข = epoch วินาที"""
    if ts is None:
        return datetime.now(UTC).isoformat(timespec="seconds")
    if isinstance(ts, (int, float)):
        return pd.Timestamp(ts, unit="s").isoformat()
    return pd.Timestamp(ts).isoformat()


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """อ่านฟิลด์จาก dict หรือ object (MT5 TradeDeal เป็น namedtuple)"""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


class TraderDB:
    def __init__(self, path: str | Path = "data/trader.db"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        current = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if current > SCHEMA_VERSION:
            raise RuntimeError(
                f"ไฟล์ฐานข้อมูลเป็น schema v{current} ใหม่กว่าโค้ด (v{SCHEMA_VERSION})"
            )
        with self.conn:
            self.conn.executescript(_SCHEMA)
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------- runs ----------
    def log_run(
        self,
        strategy: str,
        symbol: str,
        bar_time: Timeish,
        decision: str,
        signal: str | None = None,
        target_weight: float | None = None,
        lots_requested: float | None = None,
        reason: str = "ok",
        status: str = "done",
        started_at: Timeish = None,
    ) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO runs (started_at, strategy, symbol, bar_time, signal,"
                " target_weight, lots_requested, decision, reason, status)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    iso(started_at),
                    strategy,
                    symbol,
                    iso(bar_time),
                    signal,
                    target_weight,
                    lots_requested,
                    decision,
                    reason,
                    status,
                ),
            )
        if cur.lastrowid is None:
            raise sqlite3.DatabaseError("insert did not return a run id")
        return int(cur.lastrowid)

    def last_processed_bar(self, strategy: str, symbol: str) -> str | None:
        """แท่งล่าสุดที่ประมวลผลสำเร็จแล้ว (status='done') ใช้กันออเดอร์ซ้ำเมื่อรันรอบเดิม"""
        row = self.conn.execute(
            "SELECT MAX(bar_time) FROM runs"
            " WHERE strategy=? AND symbol=? AND status='done'",
            (strategy, symbol),
        ).fetchone()
        return row[0]

    def already_processed(self, strategy: str, symbol: str, bar_time: Timeish) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM runs WHERE strategy=? AND symbol=? AND bar_time=?"
            " AND status='done' LIMIT 1",
            (strategy, symbol, iso(bar_time)),
        ).fetchone()
        return row is not None

    # ---------- orders ----------
    def log_order(
        self,
        symbol: str,
        kind: str,
        side: str,
        lots: float,
        ok: bool,
        run_id: int | None = None,
        requested_price: float | None = None,
        fill_price: float | None = None,
        sl: float | None = None,
        retcode: int | None = None,
        ticket: int | None = None,
        comment: str | None = None,
    ) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO orders (run_id, logged_at, symbol, kind, side, lots,"
                " requested_price, fill_price, sl, retcode, ticket, comment, ok)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    iso(),
                    symbol,
                    kind,
                    side,
                    lots,
                    requested_price,
                    fill_price,
                    sl,
                    retcode,
                    ticket,
                    comment,
                    int(ok),
                ),
            )
        if cur.lastrowid is None:
            raise sqlite3.DatabaseError("insert did not return an order id")
        return int(cur.lastrowid)

    # ---------- deals ----------
    def log_deal(self, deal: Any) -> bool:
        """บันทึก deal จาก MT5 (dict หรือ TradeDeal) คืน True ถ้าเป็น deal ใหม่ ซ้ำจะถูกข้าม"""
        with self.conn:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO deals (deal_ticket, position_ticket, symbol, time,"
                " type, entry, volume, price, profit, commission, swap, fee, reason, logged_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    _get(deal, "ticket"),
                    _get(deal, "position_id"),
                    _get(deal, "symbol"),
                    iso(_get(deal, "time")),
                    _get(deal, "type"),
                    _get(deal, "entry"),
                    _get(deal, "volume"),
                    _get(deal, "price"),
                    _get(deal, "profit"),
                    _get(deal, "commission"),
                    _get(deal, "swap"),
                    _get(deal, "fee"),
                    _get(deal, "reason"),
                    iso(),
                ),
            )
        return cur.rowcount == 1

    # ---------- quotes / snapshots ----------
    def log_quote(
        self, symbol: str, bid: float, ask: float, point: float, ts: Timeish = None
    ) -> None:
        """spread_points = (ask - bid) / point (ข้อมูลของข้อ 1.6: ประวัติ spread จริงจาก broker)"""
        spread_points = round((ask - bid) / point, 4)
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO quotes (ts, symbol, bid, ask, spread_points)"
                " VALUES (?,?,?,?,?)",
                (iso(ts), symbol, bid, ask, spread_points),
            )

    def log_snapshot(
        self, balance: float, equity: float, open_positions: int, ts: Timeish = None
    ) -> None:
        stamp = iso(ts)
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO account_snapshots"
                " (snap_date, ts, balance, equity, open_positions) VALUES (?,?,?,?,?)",
                (stamp[:10], stamp, balance, equity, open_positions),
            )

    # ---------- persistent state ----------
    def set_state(self, key: str, value: Any) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO state (key, value, updated_at) VALUES (?,?,?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
                " updated_at=excluded.updated_at",
                (key, str(value), iso()),
            )

    def get_state(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM state WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else default

    # ---------- generic read ----------
    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]
