"""Daily backups of the trading database and log files."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import zipfile
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class BackupResult:
    """Outcome and short message suitable for a Telegram notification."""

    ok: bool
    summary: str
    changed: bool = True


def _date(now: datetime) -> str:
    return now.astimezone(timezone.utc).date().isoformat()


def _size(path: Path) -> str:
    return f"{max(1, round(path.stat().st_size / 1024))} KB"


def _content_digest(connection: sqlite3.Connection) -> str:
    return sha256("\n".join(connection.iterdump()).encode()).hexdigest()


def _newest_backup(destination: Path) -> Path | None:
    pattern = re.compile(r"^trader-\d{4}-\d{2}-\d{2}\.db$")
    candidates = [
        path for path in destination.iterdir()
        if path.is_file() and pattern.fullmatch(path.name)
    ]
    return max(candidates, key=lambda path: path.name, default=None)


def backup_database(
    db_path: str | os.PathLike[str],
    dest_dir: str | os.PathLike[str],
    *,
    now: datetime,
    keep: int = 30,
) -> tuple[Path, bool]:
    """Create an integrity-checked, WAL-safe SQLite snapshot."""
    destination = Path(dest_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for stale in destination.glob("trader-*.db.tmp"):
        if stale.is_file():
            stale.unlink()

    name = f"trader-{_date(now)}.db"
    target = destination / name
    temporary = destination / f"{name}.tmp"
    source_conn: sqlite3.Connection | None = None
    copy_conn: sqlite3.Connection | None = None
    log = logging.getLogger(__name__)
    try:
        source_conn = sqlite3.connect(db_path)
        copy_conn = sqlite3.connect(temporary)
        source_conn.backup(copy_conn)
        copy_conn.execute("PRAGMA journal_mode=DELETE").fetchone()
        result = copy_conn.execute("PRAGMA integrity_check").fetchone()
        if result is None or result[0] != "ok":
            raise sqlite3.DatabaseError(f"integrity_check returned {result!r}")
        copy_conn.commit()
        new_digest = _content_digest(copy_conn)
        newest = _newest_backup(destination)
        if newest is not None:
            unchanged = False
            try:
                uri = f"file:{newest.resolve().as_posix()}?mode=ro"
                with closing(sqlite3.connect(uri, uri=True)) as old_conn:
                    old_integrity = old_conn.execute("PRAGMA integrity_check").fetchone()
                    if old_integrity is not None and old_integrity[0] == "ok":
                        if _content_digest(old_conn) == new_digest:
                            unchanged = True
            except Exception:
                pass
            if unchanged:
                copy_conn.close()
                copy_conn = None
                source_conn.close()
                source_conn = None
                temporary.unlink()
                return newest, False
        copy_conn.close()
        copy_conn = None
        source_conn.close()
        source_conn = None
        os.replace(temporary, target)
        try:
            prune(destination, "trader-", ".db", keep)
        except Exception as exc:
            log.warning("Prune failed: %s", type(exc).__name__)
        return target, True
    finally:
        if copy_conn is not None:
            copy_conn.close()
        if source_conn is not None:
            source_conn.close()
        for path in (temporary, Path(f"{temporary}-wal"), Path(f"{temporary}-shm")):
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass


def backup_logs(
    log_dir: str | os.PathLike[str],
    dest_dir: str | os.PathLike[str],
    *,
    now: datetime,
    keep: int = 30,
) -> Path | None:
    """Archive log files, skipping files that cannot be read."""
    destination = Path(dest_dir)
    destination.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in Path(log_dir).glob("*.log*") if path.is_file())
    if not files:
        return None

    name = f"logs-{_date(now)}.zip"
    target = destination / name
    temporary = destination / f"{name}.tmp"
    log = logging.getLogger(__name__)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                try:
                    archive.write(path, arcname=path.name)
                except Exception as exc:
                    log.warning("Skipping log %s: %s", path.name, type(exc).__name__)
        os.replace(temporary, target)
        try:
            prune(destination, "logs-", ".zip", keep)
        except Exception as exc:
            log.warning("Prune failed: %s", type(exc).__name__)
        return target
    finally:
        if temporary.exists():
            temporary.unlink()


def prune(
    dest_dir: str | os.PathLike[str],
    pattern_prefix: str,
    suffix: str,
    keep: int,
) -> None:
    """Delete old daily backup files without touching unrelated files."""
    if keep < 0:
        raise ValueError("keep must be non-negative")
    patterns = {
        ("trader-", ".db"): re.compile(r"^trader-\d{4}-\d{2}-\d{2}\.db$"),
        ("logs-", ".zip"): re.compile(r"^logs-\d{4}-\d{2}-\d{2}\.zip$"),
    }
    regex = patterns.get((pattern_prefix, suffix))
    if regex is None:
        return
    destination = Path(dest_dir)
    candidates = sorted(
        (path for path in destination.iterdir() if path.is_file() and regex.fullmatch(path.name)),
        key=lambda path: path.name,
        reverse=True,
    )
    for path in candidates[keep:]:
        path.unlink()


def run_backup(
    db_path: str | os.PathLike[str],
    log_dir: str | os.PathLike[str],
    dest_dir: str | os.PathLike[str],
    *,
    now: datetime | None = None,
    keep: int = 30,
) -> BackupResult:
    """Run database and log backups independently and never raise."""
    current = now or datetime.now(timezone.utc)
    log = logging.getLogger(__name__)
    db_path_obj = Path(db_path)
    db_result: Path | None = None
    db_changed = False
    log_result: Path | None = None
    db_error: Exception | None = None
    log_error: Exception | None = None

    if not db_path_obj.exists():
        db_error = FileNotFoundError(db_path_obj)
        log.info("Database backup skipped: source does not exist")
    else:
        try:
            db_result, db_changed = backup_database(db_path_obj, dest_dir, now=current, keep=keep)
            log.info("Database backup complete: %s", db_result.name)
        except Exception as exc:
            db_error = exc
            log.exception("Database backup failed")

    if db_changed or db_error is not None:
        try:
            log_result = backup_logs(log_dir, dest_dir, now=current, keep=keep)
            if log_result is None:
                log.info("Log backup skipped: no log files")
            else:
                log.info("Log backup complete: %s", log_result.name)
        except Exception as exc:
            log_error = exc
            log.exception("Log backup failed")

    parts: list[str] = []
    if db_result is not None:
        parts.append(f"{db_result.name} ({_size(db_result)})")
    elif db_error is not None:
        parts.append(f"database backup failed: {type(db_error).__name__}")
    else:
        parts.append("no db yet")
    if log_result is not None:
        parts.append(f"{log_result.name} ({_size(log_result)})")
    elif log_error is not None:
        parts.append(f"logs backup failed: {type(log_error).__name__}")
    if db_result is not None and not db_changed:
        return BackupResult(True, "backup skipped: db unchanged", changed=False)
    summary = "backup ok: " + ", ".join(parts)
    if log_error is not None or db_error is not None:
        summary = "backup FAILED: " + ", ".join(parts)
    return BackupResult(db_error is None and log_error is None, summary[:300], changed=db_changed)
