import sqlite3
import logging
import zipfile
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from project.Execution import run_daily
from project.Storage import backup as backup_module
from project.Storage.backup import backup_database, backup_logs, prune, run_backup
from project.Storage.trader_db import TraderDB


NOW = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)


def test_database_backup_is_wal_safe_and_integrity_checked(tmp_path):
    source = tmp_path / "t.db"
    with TraderDB(source) as db:
        db.set_state("one", "1")
        db.set_state("two", "2")
        result, changed = backup_database(source, tmp_path / "backups", now=NOW)
        with sqlite3.connect(result) as copy:
            assert copy.execute("SELECT value FROM state ORDER BY key").fetchall() == [
                ("1",),
                ("2",),
            ]
            assert copy.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert changed is True
    assert not list((tmp_path / "backups").glob("*-wal"))
    assert not list((tmp_path / "backups").glob("*-shm"))


def test_database_backup_reruns_and_dates_are_separate(tmp_path):
    source = tmp_path / "t.db"
    with TraderDB(source) as db:
        db.set_state("value", "one")
    dest = tmp_path / "backups"
    first, first_changed = backup_database(source, dest, now=NOW)
    second, second_changed = backup_database(source, dest, now=NOW.replace(hour=13))
    with TraderDB(source) as db:
        db.set_state("value", "two")
    third, third_changed = backup_database(source, dest, now=NOW.replace(day=29))
    assert first == second and third != first
    assert first_changed is True and second_changed is False and third_changed is True
    assert len(list(dest.glob("trader-*.db"))) == 2


def test_database_backup_unchanged_preserves_existing_file(tmp_path):
    source = tmp_path / "t.db"
    TraderDB(source).close()
    dest = tmp_path / "backups"
    first, _ = backup_database(source, dest, now=NOW)
    content = first.read_bytes()
    mtime = first.stat().st_mtime_ns

    second, changed = backup_database(source, dest, now=NOW.replace(hour=13))

    assert second == first
    assert changed is False
    assert first.read_bytes() == content
    assert first.stat().st_mtime_ns == mtime


def test_database_backup_same_day_changed_replaces_existing_file(tmp_path):
    source = tmp_path / "t.db"
    with TraderDB(source) as db:
        db.set_state("value", "one")
    dest = tmp_path / "backups"

    first, first_changed = backup_database(source, dest, now=NOW)
    with TraderDB(source) as db:
        db.set_state("value", "two")
    second, second_changed = backup_database(source, dest, now=NOW)

    assert first == second
    assert first_changed is True and second_changed is True
    with sqlite3.connect(second) as connection:
        assert connection.execute(
            "SELECT value FROM state WHERE key = 'value'"
        ).fetchone() == ("two",)


def test_database_backup_corrupt_newest_backup_is_replaced(tmp_path):
    source = tmp_path / "t.db"
    TraderDB(source).close()
    dest = tmp_path / "backups"
    dest.mkdir()
    newest = dest / "trader-2026-10-28.db"
    newest.write_bytes(b"not sqlite")

    result, changed = backup_database(source, dest, now=NOW)

    assert result.exists()
    assert changed is True
    with sqlite3.connect(result) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_database_backup_survives_prune_failure(tmp_path, monkeypatch):
    source = tmp_path / "t.db"
    TraderDB(source).close()
    monkeypatch.setattr(
        backup_module,
        "prune",
        lambda *args: (_ for _ in ()).throw(PermissionError("locked")),
    )

    result, _ = backup_database(source, tmp_path / "backups", now=NOW)

    assert result.exists()


def test_logs_backup_survives_prune_failure(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "a.log").write_text("a")
    monkeypatch.setattr(
        backup_module,
        "prune",
        lambda *args: (_ for _ in ()).throw(PermissionError("locked")),
    )

    result = backup_logs(logs, tmp_path / "out", now=NOW)

    assert result is not None and result.exists()


def test_run_backup_is_ok_when_prune_fails(tmp_path, monkeypatch):
    source = tmp_path / "t.db"
    TraderDB(source).close()
    monkeypatch.setattr(
        backup_module,
        "prune",
        lambda *args: (_ for _ in ()).throw(PermissionError("locked")),
    )

    result = run_backup(
        source, tmp_path / "missing-logs", tmp_path / "backups", now=NOW
    )

    assert result.ok is True


def test_prune_does_not_delete_unrelated_files(tmp_path):
    for day in range(1, 36):
        (tmp_path / f"trader-2026-01-{day:02d}.db").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("keep")
    (tmp_path / "other.db").write_bytes(b"keep")
    prune(tmp_path, "trader-", ".db", 30)
    assert len(list(tmp_path.glob("trader-*.db"))) == 30
    assert (tmp_path / "notes.txt").exists() and (tmp_path / "other.db").exists()


def test_logs_backup_skips_unreadable_files_and_empty_returns_none(
    tmp_path, monkeypatch
):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "a.log").write_text("a")
    (logs / "locked.log.1").write_text("b")
    original = zipfile.ZipFile.write

    def write(self, filename, *args, **kwargs):
        if str(filename).endswith("locked.log.1"):
            raise PermissionError("locked")
        return original(self, filename, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "write", write)
    result = backup_logs(logs, tmp_path / "out", now=NOW)
    assert result is not None
    with zipfile.ZipFile(result) as archive:
        assert archive.namelist() == ["a.log"]
    assert backup_logs(tmp_path / "empty", tmp_path / "out", now=NOW) is None


def test_run_backup_never_raises_and_db_failure_does_not_block_logs(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "a.log").write_text("a")
    result = run_backup(tmp_path / "missing.db", logs, tmp_path / "out", now=NOW)
    assert result.ok is False
    assert "database backup failed" in result.summary
    assert (tmp_path / "out" / "logs-2026-10-28.zip").exists()
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    result = run_backup(tmp_path / "missing.db", logs, blocked, now=NOW)
    assert result.ok is False


def test_run_backup_only_writes_logs_when_database_changes(tmp_path):
    source = tmp_path / "t.db"
    TraderDB(source).close()
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "a.log").write_text("a")
    dest = tmp_path / "out"

    first = run_backup(source, logs, dest, now=NOW)
    assert first.ok and first.changed
    log_zip = dest / "logs-2026-10-28.zip"
    assert log_zip.exists()
    log_mtime = log_zip.stat().st_mtime_ns

    second = run_backup(source, logs, dest, now=NOW.replace(hour=13))
    assert second.ok and not second.changed
    assert log_zip.stat().st_mtime_ns == log_mtime
    assert len(list(dest.glob("logs-*.zip"))) == 1


def _args(**changes):
    values = dict(
        symbol="XAUUSD", db="db", log_dir="logs", backup_dir="backups", no_backup=False
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_backup_wrapper_alerts_and_preserves_exit_code(monkeypatch, caplog):
    alerts = []
    backup_calls = []
    caplog.set_level(logging.INFO, logger="autotrader")
    monkeypatch.delenv("AUTOTRADER_BACKUP_DIR", raising=False)
    monkeypatch.setattr(
        run_daily, "_alert", lambda symbol, level, text: alerts.append((level, text))
    )
    monkeypatch.setattr(
        run_daily,
        "run_backup",
        lambda *args: (
            backup_calls.append(args)
            or SimpleNamespace(ok=True, changed=True, summary="backup ok")
        ),
    )
    monkeypatch.setattr(
        run_daily,
        "_run_trading",
        lambda args: run_daily.TradingResult(1, None),
    )
    assert run_daily.main(["--no-backup"]) == 1
    assert alerts == []
    assert backup_calls == []
    assert run_daily._backup_and_alert(_args()) is None
    assert alerts == []

    monkeypatch.setattr(
        run_daily,
        "run_backup",
        lambda *args: SimpleNamespace(
            ok=True, changed=False, summary="backup skipped: db unchanged"
        ),
    )
    run_daily._backup_and_alert(_args())
    assert alerts == []
    assert caplog.messages[-1] == "backup skipped: db unchanged"

    def fail(*args):
        raise RuntimeError("boom")

    monkeypatch.setattr(run_daily, "run_backup", fail)
    assert run_daily.main(["--backup-dir", "backups"]) == 1
    run_daily._backup_and_alert(_args())
    assert alerts[-1][0] == "warn"
    assert caplog.records[-1].levelno == logging.ERROR

    monkeypatch.setattr(
        run_daily,
        "run_backup",
        lambda *args: SimpleNamespace(
            ok=True,
            changed=True,
            summary="backup ok: trader-2026-10-28.db, logs-2026-10-28.zip",
        ),
    )
    run_daily._backup_and_alert(_args())
    assert caplog.messages[-1] == (
        "backup ok: trader-2026-10-28.db, logs-2026-10-28.zip"
    )

    monkeypatch.setattr(
        run_daily,
        "run_backup",
        lambda *args: SimpleNamespace(
            ok=False, changed=True, summary="backup FAILED: database unavailable"
        ),
    )
    run_daily._backup_and_alert(_args())
    assert alerts[-1] == ("warn", "backup FAILED: database unavailable")
    assert caplog.messages[-1] == "backup failed: database unavailable"
    assert caplog.records[-1].levelno == logging.WARNING

    monkeypatch.setenv("AUTOTRADER_BACKUP_DIR", "")
    backup_calls.clear()
    run_daily._backup_and_alert(_args(backup_dir=None))
    assert alerts[-1] == ("warn", "backup: AUTOTRADER_BACKUP_DIR not set")
    assert backup_calls == []
