"""ทดสอบ MT5Broker ด้วย MT5 ปลอม (ไม่ต้องมี terminal) วางที่ Test/Execution/"""

from types import SimpleNamespace as NS

import pandas as pd
import pytest

from project.Execution.mt5_broker import BrokerError, MT5Broker


class FakeMT5:
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    DEAL_ENTRY_IN = 0
    DEAL_ENTRY_OUT = 1
    DEAL_REASON_SL = 4
    DEAL_REASON_CLIENT = 0

    def __init__(self, demo=True, contract_size=100.0, retcodes=None):
        self.demo = demo
        self.contract_size = contract_size
        self.retcodes = list(retcodes or [])  # คิว retcode ที่จะตอบก่อนสำเร็จ
        self.bid, self.ask = 2000.0, 2000.4
        self.positions = []
        self.deals = []
        self.balance = 10_000.0
        self.sent = []
        self._next = 1

    def initialize(self):
        return True

    def last_error(self):
        return (0, "ok")

    def account_info(self):
        return NS(trade_mode=0 if self.demo else 2, equity=self.balance, balance=self.balance)

    def symbol_info(self, s):
        return NS(visible=True, trade_contract_size=self.contract_size, filling_mode=2,
                  volume_step=0.01, volume_min=0.01, volume_max=50.0, digits=2)

    def symbol_select(self, s, v):
        return True

    def symbol_info_tick(self, s):
        return NS(bid=self.bid, ask=self.ask)

    def positions_get(self, symbol=None):
        return list(self.positions)

    def history_deals_get(self, position=None):
        return [d for d in self.deals if d.position_id == position]

    def order_send(self, req):
        self.sent.append(dict(req))
        if self.retcodes:
            return NS(retcode=self.retcodes.pop(0), comment="x", order=0)
        if "position" in req:  # ปิด
            pos = next(p for p in self.positions if p.ticket == req["position"])
            px = req["price"]
            sign = 1 if pos.type == 0 else -1
            self.positions.remove(pos)
            self._deal(pos.ticket, self.DEAL_ENTRY_OUT, px, self.DEAL_REASON_CLIENT,
                       profit=sign * (px - pos.price_open) * pos.volume * self.contract_size)
            return NS(retcode=10009, comment="", order=99)
        t = self._next
        self._next += 1
        buy = req["type"] == 0
        self.positions.append(NS(ticket=t, magic=req["magic"], type=0 if buy else 1,
                                 volume=req["volume"], price_open=req["price"], sl=req["sl"], time=1_700_000_000))
        self._deal(t, self.DEAL_ENTRY_IN, req["price"], self.DEAL_REASON_CLIENT)
        return NS(retcode=10009, comment="", order=t)

    def _deal(self, pid, entry, price, reason, profit=0.0, commission=-0.5, swap=0.0):
        self.deals.append(NS(position_id=pid, entry=entry, price=price, reason=reason, profit=profit,
                             commission=commission, swap=swap, fee=0.0, time=1_700_000_100))

    def hit_stop(self):
        """จำลอง broker ตัด SL"""
        pos = self.positions[0]
        sign = 1 if pos.type == 0 else -1
        self.positions.clear()
        self._deal(pos.ticket, self.DEAL_ENTRY_OUT, pos.sl, self.DEAL_REASON_SL,
                   profit=sign * (pos.sl - pos.price_open) * pos.volume * self.contract_size)


SPEC = NS(contract_size=100.0)
TS = pd.Timestamp("2026-10-24")


def make(**kw):
    fake = FakeMT5(**kw)
    return fake, MT5Broker(SPEC, "XAUUSD", mt5_module=fake, retry_sleep=0)


def test_refuses_non_demo():
    fake = FakeMT5(demo=False)
    with pytest.raises(BrokerError):
        MT5Broker(SPEC, "XAUUSD", mt5_module=fake)
    MT5Broker(SPEC, "XAUUSD", mt5_module=fake, allow_live=True)


def test_contract_size_mismatch():
    with pytest.raises(BrokerError):
        MT5Broker(NS(contract_size=1.0), "XAUUSD", mt5_module=FakeMT5())


def test_open_sends_stop_and_uses_real_fill():
    fake, b = make()
    b.open("long", 0.057, 0, 0, stop=1990.0, ts=TS)
    req = fake.sent[0]
    assert req["sl"] == 1990.0 and req["volume"] == 0.05  # ปัดลง
    assert b.position.entry == 2000.4 and b.open_positions == 1


def test_stop_on_wrong_side_rejected():
    _, b = make()
    with pytest.raises(ValueError):
        b.open("long", 0.1, 0, 0, stop=2010.0, ts=TS)
    with pytest.raises(ValueError):
        b.open("short", 0.1, 0, 0, stop=1990.0, ts=TS)


def test_lots_below_min_rejected():
    _, b = make()
    with pytest.raises(ValueError):
        b.open("long", 0.004, 0, 0, stop=1990.0, ts=TS)


def test_close_returns_trade_with_costs():
    fake, b = make()
    b.open("long", 0.1, 0, 0, stop=1990.0, ts=TS)
    fake.bid = 2010.0
    t = b.close(0, 0, TS, "exit")
    assert t.reason == "exit" and b.open_positions == 0
    # commission -0.5 ทั้งตอนเข้าและตอนออก (deal ละ 1 ครั้ง) รวม -1.0
    assert t.pnl == pytest.approx((2010.0 - 2000.4) * 0.1 * 100 - 1.0)


def test_check_stop_detects_server_side_sl():
    fake, b = make()
    b.open("long", 0.1, 0, 0, stop=1990.0, ts=TS)
    assert b.check_stop() is None
    fake.hit_stop()
    t = b.check_stop()
    assert t.reason == "stop" and t.exit == 1990.0 and t.pnl < 0
    assert b.open_positions == 0 and len(b.trades) == 1


def test_retry_on_requote_then_success():
    fake, b = make(retcodes=[10004, 10020])
    b.open("short", 0.1, 0, 0, stop=2010.0, ts=TS)
    assert len(fake.sent) == 3 and b.position is not None


def test_ambiguous_failure_not_retried():
    fake, b = make(retcodes=[10012])  # timeout
    with pytest.raises(BrokerError):
        b.open("long", 0.1, 0, 0, stop=1990.0, ts=TS)
    assert len(fake.sent) == 1


def test_sync_adopts_existing_position():
    fake, b = make()
    b.open("short", 0.1, 0, 0, stop=2010.0, ts=TS)
    b2 = MT5Broker(SPEC, "XAUUSD", mt5_module=fake)
    assert b2.position.side == "short" and b2.open_positions == 1
    with pytest.raises(RuntimeError):
        b2.open("long", 0.1, 0, 0, stop=1990.0, ts=TS)


def test_sync_reports_missing_position():
    fake, b = make()
    b.open("long", 0.1, 0, 0, stop=1990.0, ts=TS)
    ticket = b._ticket
    fake.positions.clear()
    assert b.sync() == f"สถานะ ticket {ticket} หายจาก broker (โดน SL/ปิดนอกโปรแกรม)"


def test_sync_reports_unexpected_position():
    fake, b = make()
    fake.positions.append(NS(ticket=7, magic=b.magic, type=0, volume=0.1,
                              price_open=2000.0, sl=1990.0, time=1_700_000_000))
    assert b.sync() == "พบสถานะ ticket 7 ที่โปรแกรมไม่ได้เปิดรอบนี้"


def test_sync_reports_ticket_change():
    fake, b = make()
    b.open("long", 0.1, 0, 0, stop=1990.0, ts=TS)
    old_ticket = b._ticket
    fake.positions[0].ticket = 8
    assert b.sync() == f"สถานะ ticket เก่า {old_ticket} เปลี่ยนเป็น ticket ใหม่ 8"


def test_sync_matching_position_returns_none():
    fake, b = make()
    b.open("long", 0.1, 0, 0, stop=1990.0, ts=TS)
    assert b.sync() is None


def test_sync_with_no_positions_returns_none():
    _, b = make()
    assert b.sync() is None


def test_sync_uses_persisted_expected_ticket_across_broker_processes():
    fake, first = make()
    first.open("long", 0.1, 0, 0, stop=1990.0, ts=TS)
    ticket = first._ticket

    second = MT5Broker(SPEC, "XAUUSD", mt5_module=fake)
    assert second.sync(expected_ticket=ticket) is None


def test_sync_reports_persisted_ticket_removed_before_new_process():
    fake, first = make()
    first.open("long", 0.1, 0, 0, stop=1990.0, ts=TS)
    ticket = first._ticket
    fake.positions.clear()

    second = MT5Broker(SPEC, "XAUUSD", mt5_module=fake)
    assert second.sync(expected_ticket=ticket) == (
        f"สถานะ ticket {ticket} หายจาก broker (โดน SL/ปิดนอกโปรแกรม)"
    )