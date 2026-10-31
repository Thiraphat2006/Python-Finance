import logging
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from project.Execution import run_daily
from project.Execution.mt5_broker import MT5Broker
from project.Execution.mt5_session import MT5InitError, SymbolNotFoundError
from project.Risk.position_sizing import SymbolSpec


def test_alert_swallows_value_error(monkeypatch):
    def raising_send(*args, **kwargs):
        raise ValueError("invalid notification")

    monkeypatch.setattr(run_daily, "send", raising_send)
    run_daily._alert("XAUUSD", "error", "failed")


class FakeClock:
    def __init__(self):
        self.current = datetime(2026, 1, 1, tzinfo=UTC)

    def now(self):
        return self.current

    def sleep(self, seconds):
        self.current += timedelta(seconds=seconds)


def _run_main(monkeypatch, outcomes, *, retry_until=10, retry_every=300, clock=None):
    calls = []
    alerts = []
    backups = []

    def fake_run(args, **kwargs):
        calls.append(kwargs)
        return outcomes.pop(0)

    monkeypatch.setattr(run_daily, "load_env", lambda: None)
    monkeypatch.setattr(run_daily, "_run_trading", fake_run)
    monkeypatch.setattr(
        run_daily, "_alert", lambda symbol, level, text: alerts.append((level, text))
    )
    monkeypatch.setattr(
        run_daily, "_backup_and_alert", lambda args: backups.append(args)
    )
    clock = clock or FakeClock()
    code = run_daily.main(
        [
            "--no-backup",
            "--retry-until-minutes",
            str(retry_until),
            "--retry-every-seconds",
            str(retry_every),
        ],
        clock=clock.now,
        sleep=clock.sleep,
    )
    return code, calls, alerts, backups, clock


def test_retries_until_success_without_intermediate_alerts(monkeypatch):
    code, calls, alerts, backups, _ = _run_main(
        monkeypatch,
        [
            run_daily.TradingResult(0, "market_closed"),
            run_daily.TradingResult(0, "market_closed"),
            run_daily.TradingResult(0, None),
        ],
    )

    assert code == 0
    assert len(calls) == 3
    assert len(calls[0]) == 1 and calls[0]["quiet_retryable"]
    assert alerts == []
    assert len(backups) == 1


def test_gives_up_with_one_market_closed_alert(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="autotrader")
    code, calls, alerts, backups, clock = _run_main(
        monkeypatch,
        [run_daily.TradingResult(0, "market_closed")] * 10,
        retry_until=10,
        retry_every=300,
    )

    assert code == 0
    assert len(calls) == 3
    assert clock.current == datetime(2026, 1, 1, 0, 10, tzinfo=UTC)
    assert alerts == [("warn", "ตลาดปิด/ไม่มี tick ตลอด 10 นาที ไม่ได้ประมวลผลวันนี้")]
    assert caplog.messages == ["give up after 3 attempts: market closed / no tick"]
    assert len(backups) == 1


def test_wall_clock_jump_gives_up_without_extra_sleep(monkeypatch):
    class WakingClock(FakeClock):
        def sleep(self, seconds):
            self.current += timedelta(minutes=20)

    code, calls, alerts, backups, _ = _run_main(
        monkeypatch,
        [run_daily.TradingResult(0, "market_closed")] * 2,
        retry_until=10,
        clock=WakingClock(),
    )

    assert code == 0
    assert len(calls) == 2
    assert alerts == [("warn", "ตลาดปิด/ไม่มี tick ตลอด 10 นาที ไม่ได้ประมวลผลวันนี้")]
    assert len(backups) == 1


def test_init_failure_retries_then_succeeds(monkeypatch):
    code, calls, alerts, backups, _ = _run_main(
        monkeypatch,
        [
            run_daily.TradingResult(1, "init_failed"),
            run_daily.TradingResult(0, None),
        ],
    )

    assert code == 0
    assert len(calls) == 2
    assert alerts == []
    assert len(backups) == 1


def test_init_failure_until_deadline_sends_one_error_and_backs_up_once(
    monkeypatch, tmp_path
):
    class FailedMT5:
        def __init__(self):
            self.initialize_calls = 0

        def initialize(self):
            self.initialize_calls += 1
            return False

        def last_error(self):
            return "terminal unavailable"

    mt5 = FailedMT5()
    monkeypatch.setitem(sys.modules, "MetaTrader5", mt5)
    monkeypatch.setattr(run_daily, "load_env", lambda: None)
    alerts = []
    backups = []
    monkeypatch.setattr(
        run_daily,
        "_alert",
        lambda symbol, level, text: alerts.append((symbol, level, text)),
    )
    monkeypatch.setattr(
        run_daily, "_backup_and_alert", lambda args: backups.append(args)
    )
    clock = FakeClock()

    code = run_daily.main(
        [
            "--db",
            str(tmp_path / "trader.db"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--no-backup",
            "--retry-until-minutes",
            "10",
            "--retry-every-seconds",
            "300",
        ],
        clock=clock.now,
        sleep=clock.sleep,
    )

    assert code == 1
    assert mt5.initialize_calls == 3
    assert alerts == [("XAUUSD", "error", "initialize ล้มเหลวตลอด 10 นาที")]
    assert len(backups) == 1


def test_non_retryable_result_stops_immediately(monkeypatch):
    code, calls, alerts, backups, _ = _run_main(
        monkeypatch, [run_daily.TradingResult(1, None)]
    )

    assert code == 1
    assert len(calls) == 1
    assert alerts == []
    assert len(backups) == 1


def test_retry_disabled_makes_one_original_style_call(monkeypatch):
    calls = []
    backups = []
    result = run_daily.TradingResult(1, "init_failed")

    monkeypatch.setattr(run_daily, "load_env", lambda: None)
    monkeypatch.setattr(
        run_daily, "_run_trading", lambda args: calls.append(args) or result
    )
    monkeypatch.setattr(
        run_daily, "_backup_and_alert", lambda args: backups.append(args)
    )

    assert run_daily.main(["--no-backup"]) == 1
    assert len(calls) == 1
    assert len(backups) == 1


class DailyFakeMT5:
    def __init__(self, *, demo=True, symbol=True):
        self.demo = demo
        self.symbol = symbol
        self.shutdown_calls = 0

    def initialize(self):
        return True

    def last_error(self):
        return "unavailable"

    def symbol_info(self, symbol):
        if not self.symbol:
            return None
        return SimpleNamespace(
            visible=True,
            point=0.01,
            trade_contract_size=100.0,
            volume_min=0.01,
            volume_step=0.01,
            volume_max=50.0,
        )

    def account_info(self):
        return SimpleNamespace(
            trade_mode=0 if self.demo else 2, equity=10_000.0, balance=10_000.0
        )

    def symbol_select(self, symbol, visible):
        return True

    def positions_get(self, symbol=None):
        return []

    def shutdown(self):
        self.shutdown_calls += 1


def daily_args(tmp_path):
    return SimpleNamespace(
        symbol="XAUUSD",
        db=str(tmp_path / "trader.db"),
        log_dir=str(tmp_path / "logs"),
        broker="mt5",
    )


class DailyFakeData:
    def __init__(self, *, market_open=True):
        self.market_open = market_open

    def is_market_open(self):
        return self.market_open

    def get_server_now(self):
        return pd.Timestamp("2026-01-01")

    def fetch_d1_bars(self, **kwargs):
        return pd.DataFrame()

    def get_quote(self):
        return (2000.0, 2000.3)

    def log_recent_deals(self, db):
        return 0


class DailyFakeSession:
    def __init__(self, mt5, *, market_open=True):
        self.mt5 = mt5
        self.market_open = market_open
        self.spec = None
        self.broker = None
        self.data = None
        self.close_calls = 0

    def connect(self):
        if not self.mt5.initialize():
            raise MT5InitError(self.mt5.last_error())
        info = self.mt5.symbol_info("XAUUSD")
        if info is None:
            raise SymbolNotFoundError("missing")
        self.spec = SymbolSpec(
            "XAUUSD",
            point=info.point,
            contract_size=info.trade_contract_size,
            volume_min=info.volume_min,
            volume_step=info.volume_step,
            volume_max=info.volume_max,
        )
        self.broker = MT5Broker(
            self.spec, "XAUUSD", mt5_module=self.mt5, initialize=False
        )
        self.data = DailyFakeData(market_open=self.market_open)

    def close(self):
        self.close_calls += 1
        self.mt5.shutdown()


def install_daily_dependencies(monkeypatch, mt5, *, market_open=True):
    monkeypatch.setattr(
        run_daily, "setup_logging", lambda path: logging.getLogger("test")
    )
    monkeypatch.setattr(
        run_daily,
        "MT5Session",
        lambda symbol: DailyFakeSession(mt5, market_open=market_open),
    )


def test_run_trading_initialize_failure_alerts(monkeypatch, tmp_path, caplog):
    class FailedMT5(DailyFakeMT5):
        def initialize(self):
            return False

    mt5 = FailedMT5()
    install_daily_dependencies(monkeypatch, mt5)
    alerts = []
    monkeypatch.setattr(run_daily, "_alert", lambda *args: alerts.append(args))

    result = run_daily._run_trading(daily_args(tmp_path))

    assert result == run_daily.TradingResult(1, "init_failed")
    assert alerts == [("XAUUSD", "error", "initialize ไม่สำเร็จ")]
    assert "initialize ไม่สำเร็จ: unavailable" in caplog.text


def test_run_trading_missing_symbol_alerts(monkeypatch, tmp_path):
    mt5 = DailyFakeMT5(symbol=False)
    sessions = []

    def make_session(symbol):
        session = DailyFakeSession(mt5)
        sessions.append(session)
        return session

    monkeypatch.setattr(run_daily, "MT5Session", make_session)
    alerts = []
    monkeypatch.setattr(run_daily, "_alert", lambda *args: alerts.append(args))

    result = run_daily._run_trading(daily_args(tmp_path))

    assert result == run_daily.TradingResult(1, None)
    assert alerts == [("XAUUSD", "error", "ไม่พบ symbol")]
    assert sessions[0].close_calls == 1


def test_run_trading_market_closed_sends_ok_alert(monkeypatch, tmp_path):
    mt5 = DailyFakeMT5()
    install_daily_dependencies(monkeypatch, mt5, market_open=False)
    alerts = []
    monkeypatch.setattr(run_daily, "_alert", lambda *args: alerts.append(args))

    result = run_daily._run_trading(daily_args(tmp_path))

    assert result == run_daily.TradingResult(0, "market_closed")
    assert alerts == [("XAUUSD", "ok", "ตลาดปิด ข้ามรอบ")]


def test_run_trading_refuses_non_demo_account(monkeypatch, tmp_path):
    mt5 = DailyFakeMT5(demo=False)
    install_daily_dependencies(monkeypatch, mt5)
    alerts = []
    monkeypatch.setattr(run_daily, "_alert", lambda *args: alerts.append(args))

    result = run_daily._run_trading(daily_args(tmp_path))

    assert result == run_daily.TradingResult(1, None)
    assert alerts == [("XAUUSD", "error", "ลูปล้มเหลว BrokerError")]


@pytest.mark.parametrize(
    ("decision", "expected_alert"),
    [
        ("enter", ("XAUUSD", "trade", "เปิดสถานะ: signal")),
        ("exit", ("XAUUSD", "trade", "ปิดสถานะ: signal")),
    ],
)
def test_run_trading_normal_enter_exit_alerts(
    monkeypatch, tmp_path, decision, expected_alert
):
    mt5 = DailyFakeMT5()
    install_daily_dependencies(monkeypatch, mt5)
    alerts = []
    monkeypatch.setattr(run_daily, "_alert", lambda *args: alerts.append(args))
    monkeypatch.setattr(
        run_daily,
        "run_once",
        lambda **kwargs: SimpleNamespace(
            status="done", decision=decision, reason="signal", sync_note=None
        ),
    )

    result = run_daily._run_trading(daily_args(tmp_path))

    assert result == run_daily.TradingResult(0, None)
    assert alerts == [expected_alert]


def test_run_trading_sync_note_and_error_alerts(monkeypatch, tmp_path):
    mt5 = DailyFakeMT5()
    install_daily_dependencies(monkeypatch, mt5)
    alerts = []
    monkeypatch.setattr(run_daily, "_alert", lambda *args: alerts.append(args))
    monkeypatch.setattr(
        run_daily,
        "run_once",
        lambda **kwargs: SimpleNamespace(
            status="error",
            decision="hold",
            reason="database failure",
            sync_note="ticket mismatch",
        ),
    )

    result = run_daily._run_trading(daily_args(tmp_path))

    assert result == run_daily.TradingResult(1, None)
    assert alerts == [
        ("XAUUSD", "warn", "สถานะไม่ตรง broker: ticket mismatch"),
        ("XAUUSD", "error", "database failure"),
    ]


@pytest.mark.parametrize("broker", ["capital", "paper"])
def test_run_trading_unimplemented_broker_does_not_touch_mt5(
    monkeypatch, tmp_path, broker
):
    touched = []
    alerts = []

    class UnexpectedSession:
        def __init__(self, symbol):
            touched.append(symbol)

    monkeypatch.setattr(run_daily, "MT5Session", UnexpectedSession)
    monkeypatch.setattr(
        run_daily, "setup_logging", lambda path: logging.getLogger("test")
    )
    monkeypatch.setattr(run_daily, "_alert", lambda *args: alerts.append(args))
    args = daily_args(tmp_path)
    args.broker = broker

    assert run_daily._run_trading(args) == run_daily.TradingResult(1, None)
    assert touched == []
    assert alerts == [("XAUUSD", "error", "broker not implemented yet")]


def test_run_trading_session_constructor_failure_alerts(monkeypatch, tmp_path):
    class BrokenSession:
        def __init__(self, symbol):
            raise ImportError("MetaTrader5 unavailable")

    monkeypatch.setattr(run_daily, "MT5Session", BrokenSession)
    monkeypatch.setattr(
        run_daily, "setup_logging", lambda path: logging.getLogger("test")
    )
    alerts = []
    monkeypatch.setattr(run_daily, "_alert", lambda *args: alerts.append(args))

    result = run_daily._run_trading(daily_args(tmp_path))

    assert result == run_daily.TradingResult(1, None)
    assert alerts == [("XAUUSD", "error", "ลูปล้มเหลว ImportError")]
