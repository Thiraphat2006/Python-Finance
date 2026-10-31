from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from project.Execution.daily_runner import run_once
from project.Execution.mt5_broker import MT5Broker
from project.Execution.mt5_data import (
    add_atr,
    fetch_d1_bars,
    is_market_open,
    log_recent_deals,
)
from project.Execution.paper_broker import PaperBroker
from project.Execution.signals import ScheduledTestStrategy
from project.Risk.position_sizing import SymbolSpec
from project.Risk.risk_engine import RiskConfig, RiskEngine, load_engine, save_engine
from project.Storage.trader_db import TraderDB

GOLD = SymbolSpec(
    "XAUUSD", point=0.01, contract_size=100, volume_min=0.01, volume_step=0.01
)
QUOTE = (2000.0, 2000.3)  # ask 2000.3, ATR 20 -> stop 1960.3, ไม้ 0.01 เสี่ยง $40 = 0.4%


def T(s):  # 2026-10-26 = จันทร์
    return pd.Timestamp(s)


MON, TUE, WED, THU, FRI = (T(f"2026-10-{d} 02:00") for d in (26, 27, 28, 29, 30))


def make_bars(atr=20.0, n=30):
    idx = pd.date_range(end="2026-10-23", periods=n, freq="D")
    return pd.DataFrame(
        {"open": 2000.0, "high": 2010.0, "low": 1990.0, "close": 2000.0, "atr": atr},
        index=idx,
    )


@pytest.fixture
def db():
    d = TraderDB(":memory:")
    yield d
    d.close()


def run(db, broker, now, quote=QUOTE, cfg=None, **kw):
    return run_once(
        broker=broker,
        db=db,
        cfg=cfg or RiskConfig(),
        strategy=ScheduledTestStrategy(GOLD),
        spec=GOLD,
        bars=make_bars(),
        quote=quote,
        now=now,
        **kw,
    )


# ---------- ตารางสัญญาณทดสอบ ----------
def test_monday_enters_min_lot_with_atr_stop_and_journals(db):
    b = PaperBroker(GOLD, 10_000)
    r = run(db, b, MON)
    assert (r.status, r.decision) == ("done", "enter")
    assert b.position.side == "long" and b.position.lots == 0.01
    assert b.position.stop == pytest.approx(2000.3 - 2 * 20)
    runs = db.query("SELECT * FROM runs")
    assert (
        len(runs) == 1
        and runs[0]["decision"] == "enter"
        and runs[0]["strategy"] == "test_schedule"
    )
    orders = db.query("SELECT * FROM orders")
    assert len(orders) == 1 and orders[0]["kind"] == "open" and orders[0]["ok"] == 1
    assert orders[0]["run_id"] == runs[0]["run_id"] and orders[0][
        "sl"
    ] == pytest.approx(1960.3)
    q = db.query("SELECT * FROM quotes")[0]
    assert q["spread_points"] == pytest.approx(30.0)
    snap = db.query("SELECT * FROM account_snapshots")[0]
    assert snap["open_positions"] == 1


def test_same_day_rerun_is_skipped_and_never_doubles_orders(db):
    b = PaperBroker(GOLD, 10_000)
    run(db, b, MON)
    r = run(db, b, MON.replace(hour=9))
    assert (r.status, r.decision) == ("skipped", "skip")
    assert (
        len(db.query("SELECT * FROM orders")) == 1
        and len(db.query("SELECT * FROM runs")) == 1
    )


def test_full_week_cycle(db):
    b = PaperBroker(GOLD, 10_000)
    assert run(db, b, MON).decision == "enter"
    t = run(db, b, TUE, quote=(2010.0, 2010.3))
    assert t.decision == "exit" and b.position is None
    assert b.trades[-1].reason == "schedule" and b.trades[-1].pnl == pytest.approx(9.7)
    assert run(db, b, WED).decision == "enter"
    assert run(db, b, THU).decision == "exit"
    assert run(db, b, FRI).decision == "hold"
    kinds = [o["kind"] for o in db.query("SELECT kind FROM orders ORDER BY order_id")]
    assert kinds == ["open", "close", "open", "close"]


def test_missed_exit_day_closes_as_stale(db):
    b = PaperBroker(GOLD, 10_000)
    run(db, b, MON)
    r = run(db, b, WED)  # ข้ามวันอังคาร (เครื่องดับ): ถือครบ 2 วัน -> ปิดเอง
    assert r.decision == "exit" and r.reason == "stale"
    assert b.trades[-1].reason == "stale"


# ---------- risk engine ในลูป ----------
def test_kill_switch_blocks_entry(db):
    cfg = RiskConfig()
    e = RiskEngine(cfg, 10_000)
    e.update(MON, 10_000)
    e.killed, e.kill_reason = True, "ทดสอบ"
    save_engine(db, e)
    b = PaperBroker(GOLD, 10_000)
    r = run(db, b, MON)
    assert r.decision == "blocked" and r.reason.startswith("kill_switch")
    assert b.position is None and db.query("SELECT * FROM orders") == []


def test_kill_switch_flattens_open_position_before_strategy(db):
    b = PaperBroker(GOLD, 10_000)
    run(db, b, MON)
    e = load_engine(db, RiskConfig(), 10_000)
    e.killed, e.kill_reason = True, "ทดสอบ"
    save_engine(db, e)
    r = run(db, b, WED)
    assert r.decision == "exit" and r.reason == "kill_switch"
    assert b.position is None


def test_total_risk_cap_blocks_and_logs_reason(db):
    b = PaperBroker(GOLD, 10_000)
    r = run(db, b, MON, cfg=RiskConfig(max_total_risk=0.002))  # ไม้เล็กสุดเสี่ยง 0.4% > 0.2%
    assert (r.status, r.decision, r.reason) == ("done", "blocked", "max_total_risk")
    assert b.position is None
    assert db.query("SELECT reason FROM runs")[0]["reason"] == "max_total_risk"


def test_risk_state_persists_between_runs(db):
    b = PaperBroker(GOLD, 10_000)
    run(db, b, MON)
    e = load_engine(db, RiskConfig(), 1)
    assert e.peak_equity >= 10_000 - 1 and e.day == MON.date()


# ---------- ผิดพลาด ----------
class Boom(PaperBroker):
    def open(self, *a, **k):
        raise RuntimeError("broker down")


def test_error_is_logged_and_rerun_allowed(db):
    r = run(db, Boom(GOLD, 10_000), MON)
    assert r.status == "error" and r.decision == "error" and "broker down" in r.reason
    assert db.query("SELECT status FROM runs")[0]["status"] == "error"
    o = db.query("SELECT * FROM orders")[0]
    assert o["ok"] == 0 and "broker down" in o["comment"]
    # รอบเดิมวันเดียวกันรันใหม่ได้ (ไม่ถูกนับว่าประมวลผลแล้ว)
    b = PaperBroker(GOLD, 10_000)
    assert run(db, b, MON.replace(hour=5)).decision == "enter"


def test_sync_note_is_saved_in_run_result(db):
    class SyncNoteBroker(PaperBroker):
        def sync(self):
            return "พบสถานะ ticket 7 ที่โปรแกรมไม่ได้เปิดรอบนี้"

    r = run(db, SyncNoteBroker(GOLD, 10_000), MON)
    assert r.sync_note == "พบสถานะ ticket 7 ที่โปรแกรมไม่ได้เปิดรอบนี้"


# ---------- mt5_data ----------
def _rates(n, start="2026-10-05"):
    t0 = int(pd.Timestamp(start).timestamp())
    dt = np.dtype(
        [
            ("time", "i8"),
            ("open", "f8"),
            ("high", "f8"),
            ("low", "f8"),
            ("close", "f8"),
            ("tick_volume", "i8"),
            ("spread", "i4"),
        ]
    )
    arr = np.zeros(n, dtype=dt)
    for i in range(n):
        arr[i] = (t0 + i * 86400, 2000 + i, 2010 + i, 1990 + i, 2005 + i, 100, 25)
    return arr


def test_fetch_d1_bars_drops_forming_bar_and_adds_atr():
    mt5 = SimpleNamespace(
        TIMEFRAME_D1=16408,
        last_error=lambda: (0, ""),
        copy_rates_from_pos=lambda s, tf, pos, n: _rates(20),
    )
    forming_day = pd.Timestamp("2026-10-05") + pd.Timedelta(days=19)
    bars = fetch_d1_bars(mt5, "XAUUSD", server_now=forming_day + pd.Timedelta(hours=5))
    assert len(bars) == 19 and bars.index[-1] == forming_day - pd.Timedelta(days=1)
    assert bars["atr"].iloc[-1] > 0 and np.isnan(bars["atr"].iloc[0])


def _tick_mt5(tick_time, trade_mode=4):
    return SimpleNamespace(
        SYMBOL_TRADE_MODE_FULL=4,
        symbol_info_tick=lambda s: SimpleNamespace(
            time=tick_time, bid=2000.0, ask=2000.4
        ),
        symbol_info=lambda s: SimpleNamespace(trade_mode=trade_mode, point=0.01),
    )


class FakeClock:
    def __init__(self):
        self.elapsed = 0.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.elapsed += seconds


def test_market_open_when_tick_changes_after_35_seconds():
    clock = FakeClock()
    calls = 0

    def tick(_symbol):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            time=1 if clock.elapsed < 35 else 2, bid=2000.0, ask=2000.4
        )

    mt5 = SimpleNamespace(
        SYMBOL_TRADE_MODE_FULL=4,
        symbol_info_tick=tick,
        symbol_info=lambda s: SimpleNamespace(trade_mode=4, point=0.01),
    )
    assert is_market_open(mt5, "XAUUSD", sleep=clock.sleep) is True
    assert clock.elapsed == pytest.approx(35.0)
    assert calls > 1


def test_market_closed_when_tick_never_changes_after_default_wait():
    clock = FakeClock()
    mt5 = _tick_mt5(1)
    assert is_market_open(mt5, "XAUUSD", sleep=clock.sleep) is False
    assert clock.elapsed == pytest.approx(60.0)


def test_market_closed_when_symbol_is_not_fully_tradable():
    clock = FakeClock()
    mt5 = _tick_mt5(1, trade_mode=0)
    assert is_market_open(mt5, "XAUUSD", sleep=clock.sleep) is False
    assert clock.sleeps == []


def test_market_closed_when_there_is_no_tick():
    clock = FakeClock()
    mt5 = SimpleNamespace(
        SYMBOL_TRADE_MODE_FULL=4,
        symbol_info_tick=lambda s: None,
        symbol_info=lambda s: SimpleNamespace(trade_mode=4),
    )
    assert is_market_open(mt5, "XAUUSD", sleep=clock.sleep) is False
    assert clock.sleeps == []


def test_market_open_when_tick_changes_with_normal_spread():
    clock = FakeClock()
    calls = 0

    def tick(_symbol):
        nonlocal calls
        calls += 1
        return SimpleNamespace(time=1 if calls == 1 else 2, bid=2000.0, ask=2000.4)

    mt5 = SimpleNamespace(
        SYMBOL_TRADE_MODE_FULL=4,
        symbol_info_tick=tick,
        symbol_info=lambda s: SimpleNamespace(trade_mode=4, point=0.01),
    )
    assert is_market_open(mt5, "XAUUSD", wait_s=1.0, poll_s=0.5, sleep=clock.sleep)


def test_market_closed_when_changed_tick_has_abnormal_spread():
    clock = FakeClock()
    calls = 0

    def tick(_symbol):
        nonlocal calls
        calls += 1
        return SimpleNamespace(time=1 if calls == 1 else 2, bid=2000.0, ask=2058.12)

    mt5 = SimpleNamespace(
        SYMBOL_TRADE_MODE_FULL=4,
        symbol_info_tick=tick,
        symbol_info=lambda s: SimpleNamespace(trade_mode=4, point=0.01),
    )
    assert not is_market_open(mt5, "XAUUSD", wait_s=1.0, poll_s=0.5, sleep=clock.sleep)


def test_add_atr_known_value():
    df = pd.DataFrame(
        {"open": 1.0, "high": [11] * 20, "low": [1] * 20, "close": [6] * 20}
    )
    assert add_atr(df, 14)["atr"].iloc[-1] == pytest.approx(10.0)  # TR คงที่ 10


def test_log_recent_deals_filters_and_dedupes(db):
    def deal(ticket, symbol="XAUUSD", magic=7):
        return SimpleNamespace(
            ticket=ticket,
            position_id=1,
            symbol=symbol,
            time=1_761_000_000,
            type=0,
            entry=0,
            volume=0.01,
            price=2000.0,
            profit=0.0,
            commission=-0.07,
            swap=0.0,
            fee=0.0,
            reason=0,
            magic=magic,
        )

    mt5 = SimpleNamespace(
        last_error=lambda: (0, ""),
        history_deals_get=lambda a, b: [
            deal(1),
            deal(2, magic=8),
            deal(3, symbol="EURUSD"),
            deal(4),
        ],
    )
    assert log_recent_deals(db, mt5, "XAUUSD", 7) == 2
    assert log_recent_deals(db, mt5, "XAUUSD", 7) == 0
    assert {r["deal_ticket"] for r in db.query("SELECT deal_ticket FROM deals")} == {
        1,
        4,
    }


# ---------- ต่อกับ MT5Broker จริง (ด้วย MT5 จำลอง) ----------
class FakeMT5:
    POSITION_TYPE_BUY, POSITION_TYPE_SELL = 0, 1
    ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
    TRADE_ACTION_DEAL, ORDER_TIME_GTC = 1, 0
    ORDER_FILLING_IOC, ORDER_FILLING_FOK, ORDER_FILLING_RETURN = 1, 0, 2
    DEAL_ENTRY_IN, DEAL_ENTRY_OUT, DEAL_REASON_SL = 0, 1, 4

    def __init__(self):
        self.positions, self.deals = [], []
        self.balance, self.bid, self.ask, self.clock, self._n = (
            10_000.0,
            2000.0,
            2000.3,
            1_761_000_000,
            1000,
        )

    def initialize(self):
        return True

    def shutdown(self):
        pass

    def last_error(self):
        return (0, "")

    def symbol_select(self, s, v):
        return True

    def account_info(self):
        unreal = sum((self.bid - p.price_open) * p.volume * 100 for p in self.positions)
        return SimpleNamespace(
            trade_mode=0, balance=self.balance, equity=self.balance + unreal
        )

    def symbol_info(self, s):
        return SimpleNamespace(
            visible=True,
            trade_contract_size=100.0,
            filling_mode=2,
            volume_step=0.01,
            volume_min=0.01,
            volume_max=100.0,
            digits=2,
            point=0.01,
        )

    def symbol_info_tick(self, s):
        return SimpleNamespace(bid=self.bid, ask=self.ask, time=self.clock)

    def positions_get(self, symbol=None):
        return list(self.positions)

    def _deal(self, pos, entry, price, profit=0.0, comm=-0.07):
        self._n += 1
        self.deals.append(
            SimpleNamespace(
                ticket=self._n,
                position_id=pos.ticket,
                symbol="XAUUSD",
                time=self.clock,
                type=0,
                entry=entry,
                volume=pos.volume,
                price=price,
                profit=profit,
                commission=comm,
                swap=0.0,
                fee=0.0,
                reason=0,
                magic=pos.magic,
            )
        )

    def order_send(self, req):
        if "position" in req:
            pos = next(p for p in self.positions if p.ticket == req["position"])
            profit = (self.bid - pos.price_open) * pos.volume * 100
            self._deal(pos, self.DEAL_ENTRY_OUT, self.bid, profit)
            self.balance += profit - 0.07
            self.positions.remove(pos)
            return SimpleNamespace(retcode=10009, order=pos.ticket, comment="")
        self._n += 1
        pos = SimpleNamespace(
            ticket=self._n,
            type=self.POSITION_TYPE_BUY,
            volume=req["volume"],
            price_open=self.ask,
            sl=req["sl"],
            time=self.clock,
            magic=req["magic"],
            symbol=req["symbol"],
        )
        self.positions.append(pos)
        self._deal(pos, self.DEAL_ENTRY_IN, self.ask)
        return SimpleNamespace(retcode=10009, order=pos.ticket, comment="")

    def history_deals_get(self, *a, **kw):
        if "position" in kw:
            return [d for d in self.deals if d.position_id == kw["position"]]
        return list(self.deals)


def test_end_to_end_with_real_mt5broker_across_two_processes(db):
    fake = FakeMT5()

    def new_process_run(
        now,
    ):  # จำลองรีสตาร์ทโปรแกรมทุกวัน: broker ใหม่ ต้อง sync จาก terminal
        broker = MT5Broker(
            GOLD, "XAUUSD", mt5_module=fake, initialize=False, retry_sleep=0
        )
        return broker, run(
            db, broker, now, quote=(fake.bid, fake.ask), mt5=fake, magic=broker.magic
        )

    broker, r = new_process_run(MON)
    assert r.decision == "enter" and len(fake.positions) == 1
    assert (
        fake.positions[0].sl == pytest.approx(1960.3)
        and fake.positions[0].volume == 0.01
    )
    assert db.query("SELECT fill_price FROM orders")[0]["fill_price"] == pytest.approx(
        2000.3
    )

    fake.bid, fake.ask, fake.clock = 2010.0, 2010.3, fake.clock + 86400
    broker, r = new_process_run(TUE)
    assert r.decision == "exit" and fake.positions == []
    assert broker.trades[-1].reason == "schedule"
    # กำไร (2010.0 - 2000.3) * 0.01 * 100 = 9.70 หัก commission ขาเข้า+ขาออก 0.07 * 2
    assert broker.trades[-1].pnl == pytest.approx(9.56)

    deals = db.query("SELECT entry FROM deals ORDER BY deal_ticket")
    assert [d["entry"] for d in deals] == [0, 1]  # IN แล้ว OUT ถูกบันทึกครบ
    assert [
        o["kind"] for o in db.query("SELECT kind FROM orders ORDER BY order_id")
    ] == ["open", "close"]
    close_order = db.query("SELECT * FROM orders WHERE kind='close'")[0]
    assert close_order["requested_price"] == pytest.approx(fake.bid)
    # รันซ้ำวันเดียวกัน ต้องไม่เปิดออเดอร์เพิ่มแม้ broker ใหม่
    _, r = new_process_run(TUE.replace(hour=8))
    assert r.decision == "skip" and fake.positions == []
