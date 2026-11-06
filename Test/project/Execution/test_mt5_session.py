from types import SimpleNamespace as NS

import pandas as pd
import pytest

from project.Execution.interfaces import Broker, Session
from project.Execution.mt5_session import (
    MT5Data,
    MT5InitError,
    MT5Session,
    SymbolNotFoundError,
)
from project.Risk.position_sizing import SymbolSpec


class FakeMT5:
    def __init__(self, initialize_result=True, symbol_info=None):
        self.initialize_result = initialize_result
        self.info = symbol_info or NS(
            visible=True,
            point=0.01,
            trade_contract_size=100.0,
            volume_min=0.01,
            volume_step=0.01,
            volume_max=50.0,
        )
        self.initialize_calls = 0
        self.select_calls = []
        self.shutdown_calls = 0

    def initialize(self):
        self.initialize_calls += 1
        return self.initialize_result

    def last_error(self):
        return (1, "failed")

    def symbol_info(self, symbol):
        return self.info

    def symbol_select(self, symbol, selected):
        self.select_calls.append((symbol, selected))
        return True

    def account_info(self):
        return NS(trade_mode=0)

    def positions_get(self, symbol=None):
        return []

    def history_deals_get(self, *args):
        return []

    def shutdown(self):
        self.shutdown_calls += 1


def test_connect_creates_broker_and_data():
    mt5 = FakeMT5()
    session = MT5Session("XAUUSD", SymbolSpec("XAUUSD"), mt5_module=mt5)

    assert session.connect() is None
    assert session.broker.symbol == "XAUUSD"
    assert isinstance(session.data, MT5Data)
    b: Broker = session.broker
    s: Session = session
    assert b is session.broker
    assert s is session
    assert mt5.initialize_calls == 1


def test_connect_initialize_failure():
    mt5 = FakeMT5(initialize_result=False)
    session = MT5Session("XAUUSD", SymbolSpec("XAUUSD"), mt5_module=mt5)

    with pytest.raises(MT5InitError, match="initialize ไม่สำเร็จ"):
        session.connect()


def test_connect_missing_symbol():
    mt5 = FakeMT5(symbol_info=None)
    mt5.info = None
    session = MT5Session("XAUUSD", SymbolSpec("XAUUSD"), mt5_module=mt5)

    with pytest.raises(SymbolNotFoundError, match="ไม่พบ symbol XAUUSD"):
        session.connect()


def test_connect_second_invisible_symbol_select_comes_from_mt5_broker():
    mt5 = FakeMT5(
        symbol_info=NS(
            visible=False,
            point=0.01,
            trade_contract_size=100.0,
            volume_min=0.01,
            volume_step=0.01,
            volume_max=50.0,
        )
    )
    session = MT5Session("XAUUSD", SymbolSpec("XAUUSD"), mt5_module=mt5)

    session.connect()

    assert mt5.select_calls == [("XAUUSD", True), ("XAUUSD", True)]


def test_connect_builds_spec_from_symbol_info():
    mt5 = FakeMT5(
        symbol_info=NS(
            visible=True,
            point=0.1,
            trade_contract_size=10.0,
            volume_min=0.1,
            volume_step=0.1,
            volume_max=25.0,
        )
    )
    session = MT5Session("XAUUSD", mt5_module=mt5)

    session.connect()

    assert session.spec == SymbolSpec(
        "XAUUSD",
        point=0.1,
        contract_size=10.0,
        volume_min=0.1,
        volume_step=0.1,
        volume_max=25.0,
    )


def test_close_shuts_down():
    mt5 = FakeMT5()
    session = MT5Session("XAUUSD", SymbolSpec("XAUUSD"), mt5_module=mt5)

    session.close()

    assert mt5.shutdown_calls == 1


def test_data_delegates_to_mt5_data_functions(monkeypatch):
    mt5 = object()
    data = MT5Data(mt5, "XAUUSD", 42)
    now = pd.Timestamp("2026-10-31")
    bars = pd.DataFrame({"close": [1.0]})
    calls = []

    monkeypatch.setattr(
        "project.Execution.mt5_session.get_server_now",
        lambda module, symbol: calls.append(("now", module, symbol)) or now,
    )
    monkeypatch.setattr(
        "project.Execution.mt5_session.get_quote",
        lambda module, symbol: calls.append(("quote", module, symbol)) or (1.0, 2.0),
    )
    monkeypatch.setattr(
        "project.Execution.mt5_session.is_market_open",
        lambda module, symbol, wait_s, poll_s: (
            calls.append(("open", module, symbol, wait_s, poll_s)) or True
        ),
    )
    monkeypatch.setattr(
        "project.Execution.mt5_session.fetch_d1_bars",
        lambda module, symbol, count, atr_period, server_now: (
            calls.append(("bars", module, symbol, count, atr_period, server_now))
            or bars
        ),
    )
    monkeypatch.setattr(
        "project.Execution.mt5_session.log_recent_deals",
        lambda db, module, symbol, magic, days: (
            calls.append(("deals", db, module, symbol, magic, days)) or 3
        ),
    )

    assert data.get_server_now() == now
    assert data.get_quote() == (1.0, 2.0)
    assert data.is_market_open(10.0, 0.25)
    assert data.fetch_d1_bars(5, 7, now) is bars
    assert data.log_recent_deals("db", 2) == 3
    assert calls == [
        ("now", mt5, "XAUUSD"),
        ("quote", mt5, "XAUUSD"),
        ("open", mt5, "XAUUSD", 10.0, 0.25),
        ("bars", mt5, "XAUUSD", 5, 7, now),
        ("deals", "db", mt5, "XAUUSD", 42, 2),
    ]
