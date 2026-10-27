import sqlite3
import zipfile
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from project.Execution import run_daily
from project.Storage.backup import backup_database, backup_logs, prune, run_backup
from project.Storage.trader_db import TraderDB


NOW = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)


def test_database_backup_is_wal_safe_and_integrity_checked(tmp_path):
    source = tmp_path / "t.db"
    with TraderDB(source) as db:
        db.set_state("one", "1")
        db.set_state("two", "2")
        result = backup_database(source, tmp_path / "backups", now=NOW)
        with sqlite3.connect(result) as copy:
            assert copy.execute("SELECT value FROM state ORDER BY key").fetchall() == [("1",), ("2",)]
            assert copy.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert not list((tmp_path / "backups").glob("*-wal"))
    assert not list((tmp_path / "backups").glob("*-shm"))


def test_database_backup_reruns_and_dates_are_separate(tmp_path):
    source = tmp_path / "t.db"
    TraderDB(source).close()
    dest = tmp_path / "backups"
    first = backup_database(source, dest, now=NOW)
    second = backup_database(source, dest, now=NOW.replace(hour=13))
    third = backup_database(source, dest, now=NOW.replace(day=29))
    assert first == second and third != first
    assert len(list(dest.glob("trader-*.db"))) == 2


def test_prune_does_not_delete_unrelated_files(tmp_path):
    for day in range(1, 36):
        (tmp_path / f"trader-2026-01-{day:02d}.db").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("keep")
    (tmp_path / "other.db").write_bytes(b"keep")
    prune(tmp_path, "trader-", ".db", 30)
    assert len(list(tmp_path.glob("trader-*.db"))) == 30
    assert (tmp_path / "notes.txt").exists() and (tmp_path / "other.db").exists()


def test_logs_backup_skips_unreadable_files_and_empty_returns_none(tmp_path, monkeypatch):
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
    assert result.ok and "no db yet" in result.summary
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    result = run_backup(tmp_path / "missing.db", logs, blocked, now=NOW)
    assert result.ok is False


def _args(**changes):
    values = dict(symbol="XAUUSD", db="db", log_dir="logs", backup_dir="backups", no_backup=False)
    values.update(changes)
    return SimpleNamespace(**values)


def test_backup_wrapper_alerts_and_preserves_exit_code(monkeypatch):
    alerts = []
    monkeypatch.delenv("AUTOTRADER_BACKUP_DIR", raising=False)
    monkeypatch.setattr(run_daily, "_alert", lambda symbol, level, text: alerts.append((level, text)))
    monkeypatch.setattr(run_daily, "run_backup", lambda *args: SimpleNamespace(ok=True, summary="backup ok"))
    monkeypatch.setattr(run_daily, "_run_trading", lambda args: 1)
    assert run_daily.main(["--no-backup"]) == 1
    assert alerts == []
    assert run_daily._backup_and_alert(_args()) is None
    assert alerts[-1] == ("ok", "backup ok")

    def fail(*args):
        raise RuntimeError("boom")

    monkeypatch.setattr(run_daily, "run_backup", fail)
    run_daily._backup_and_alert(_args())
    assert alerts[-1][0] == "warn"
    monkeypatch.setenv("AUTOTRADER_BACKUP_DIR", "")
    run_daily._backup_and_alert(_args(backup_dir=None))
    assert alerts[-1] == ("warn", "backup: AUTOTRADER_BACKUP_DIR not set")
