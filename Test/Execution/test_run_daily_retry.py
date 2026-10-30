from datetime import datetime, timedelta, timezone

from project.Execution import run_daily


class FakeClock:
    def __init__(self):
        self.current = datetime(2026, 1, 1, tzinfo=timezone.utc)

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


def test_gives_up_with_one_market_closed_alert(monkeypatch):
    code, calls, alerts, backups, clock = _run_main(
        monkeypatch,
        [run_daily.TradingResult(0, "market_closed")] * 10,
        retry_until=10,
        retry_every=300,
    )

    assert code == 0
    assert len(calls) == 3
    assert clock.current == datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)
    assert alerts == [
        ("warn", "ตลาดปิด/ไม่มี tick ตลอด 10 นาที ไม่ได้ประมวลผลวันนี้")
    ]
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
    assert alerts == [
        ("warn", "ตลาดปิด/ไม่มี tick ตลอด 10 นาที ไม่ได้ประมวลผลวันนี้")
    ]
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
    monkeypatch.setattr(run_daily, "_run_trading", lambda args: calls.append(args) or result)
    monkeypatch.setattr(run_daily, "_backup_and_alert", lambda args: backups.append(args))

    assert run_daily.main(["--no-backup"]) == 1
    assert len(calls) == 1
    assert len(backups) == 1
