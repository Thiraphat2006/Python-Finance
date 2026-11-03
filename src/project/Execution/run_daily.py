"""จุดเริ่มรันจริง (Task Scheduler เรียกไฟล์นี้วันละครั้ง)

    uv run python -m project.Execution.run_daily --symbol XAUUSD

Exit code: 0 = สำเร็จหรือข้ามเพราะรันแล้ว, 1 = ผิดพลาด (ไว้ให้ alert ข้อ 1.3 จับ)
ปฏิเสธบัญชีที่ไม่ใช่ demo เสมอ (ไม่มีตัวเลือก allow_live ที่นี่โดยตั้งใจ)
"""

import argparse
import logging
import os
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Literal, NamedTuple

import pandas as pd

from project.Execution.daily_runner import run_once
from project.Execution.mt5_session import (
    MT5InitError,
    MT5Session,
    SymbolNotFoundError,
)
from project.Execution.signals import ScheduledTestStrategy
from project.Risk.risk_engine import RiskConfig
from project.Storage.backup import run_backup
from project.Storage.log_setup import setup_logging
from project.Storage.notify import load_env, send
from project.Storage.trader_db import TraderDB


class TradingResult(NamedTuple):
    code: int
    retry_reason: Literal["market_closed", "init_failed"] | None


def _alert(
    symbol: str,
    level: Literal["ok", "trade", "warn", "error", "risk"],
    text: str,
) -> None:
    emoji = {
        "ok": "⚪",
        "trade": "🟢",
        "warn": "🟡",
        "error": "🔴",
        "risk": "🚨",
    }[level]
    try:
        send(f"{emoji} {symbol} · {text}", silent=level == "ok")
    except Exception as exc:  # noqa: BLE001
        # Notifications must never raise or affect trading.
        logging.getLogger(__name__).warning("แจ้งเตือนไม่สำเร็จ: %s", type(exc).__name__)


def _run_trading(
    args: argparse.Namespace, *, quiet_retryable: bool = False
) -> TradingResult:
    log = setup_logging(args.log_dir)
    if args.broker != "mt5":
        log.error("broker not implemented yet: %s", args.broker)
        _alert(args.symbol, "error", "broker not implemented yet")
        return TradingResult(1, None)

    try:
        session = MT5Session(args.symbol)
    except Exception as exc:
        log.exception("ลูปรายวันล้มเหลว")
        _alert(args.symbol, "error", f"ลูปล้มเหลว {type(exc).__name__}")
        return TradingResult(1, None)

    try:
        session.connect()
    except MT5InitError as exc:
        log.error("initialize ไม่สำเร็จ: %s", exc.last_error)
        if quiet_retryable:
            log.warning("initialize ไม่สำเร็จ จะลองใหม่")
        else:
            _alert(args.symbol, "error", "initialize ไม่สำเร็จ")
        return TradingResult(1, "init_failed")
    except SymbolNotFoundError:
        session.close()
        log.error("ไม่พบ symbol %s", args.symbol)
        _alert(args.symbol, "error", "ไม่พบ symbol")
        return TradingResult(1, None)
    except Exception as exc:
        log.exception("ลูปรายวันล้มเหลว")
        _alert(args.symbol, "error", f"ลูปล้มเหลว {type(exc).__name__}")
        session.close()
        return TradingResult(1, None)

    try:
        if not session.data.is_market_open():
            # ไม่บันทึกอะไรเลย (ไม่ใช้กุญแจของวัน, ไม่เก็บ quote เก่า) รันใหม่ตอนตลาดเปิดได้
            log.warning("ตลาด %s ปิดอยู่ (ไม่มี tick ใหม่) ข้ามรอบนี้", args.symbol)
            if not quiet_retryable:
                _alert(args.symbol, "ok", "ตลาดปิด ข้ามรอบ")
            return TradingResult(0, "market_closed")
        spec = session.spec
        if spec is None:
            raise RuntimeError("session symbol specification unavailable")
        broker = session.broker
        strategy = ScheduledTestStrategy(spec)
        with TraderDB(args.db) as db:
            server_now = session.data.get_server_now()
            bars = session.data.fetch_d1_bars(server_now=server_now)
            result = run_once(
                broker=broker,
                db=db,
                cfg=RiskConfig(),
                strategy=strategy,
                spec=spec,
                bars=bars,
                quote=session.data.get_quote(),
                now=pd.Timestamp.now(tz="UTC").tz_localize(None),
                data=session.data,
                logger=log,
                broker_name=args.broker,
            )
        if result.sync_note:
            _alert(args.symbol, "warn", f"สถานะไม่ตรง broker: {result.sync_note}")
        if result.status == "error":
            _alert(args.symbol, "error", result.reason)
        elif result.status == "skipped":
            _alert(args.symbol, "ok", "ข้าม (ประมวลผลแล้ว)")
        elif result.decision == "exit" and result.reason in (
            "kill_switch",
            "daily_limit",
        ):
            _alert(args.symbol, "risk", f"ปิดสถานะ: {result.reason}")
        elif result.decision in ("enter", "exit"):
            action = "เปิดสถานะ" if result.decision == "enter" else "ปิดสถานะ"
            _alert(args.symbol, "trade", f"{action}: {result.reason}")
        elif result.decision == "blocked":
            _alert(args.symbol, "warn", f"blocked: {result.reason}")
        elif result.decision == "hold":
            _alert(args.symbol, "ok", "hold")
        return TradingResult(1 if result.status == "error" else 0, None)
    except Exception as exc:
        log.exception("ลูปรายวันล้มเหลว")
        _alert(args.symbol, "error", f"ลูปล้มเหลว {type(exc).__name__}")
        return TradingResult(1, None)
    finally:
        session.close()


def _backup_and_alert(args: argparse.Namespace) -> None:
    """Run the daily backup without affecting trading's exit status."""
    log = logging.getLogger("autotrader")
    if args.no_backup:
        return
    backup_dir = args.backup_dir or os.environ.get("AUTOTRADER_BACKUP_DIR")
    if not backup_dir:
        log.warning("AUTOTRADER_BACKUP_DIR not set")
        _alert(args.symbol, "warn", "backup: AUTOTRADER_BACKUP_DIR not set")
        return
    try:
        result = run_backup(args.db, args.log_dir, backup_dir)
        if result.ok:
            log.info(result.summary)
        else:
            reason = result.summary.removeprefix("backup FAILED: ")
            log.warning("backup failed: %s", reason)
            _alert(args.symbol, "warn", "backup FAILED: " + reason)
    except Exception as exc:
        log.exception("Backup wrapper failed")
        _alert(args.symbol, "warn", f"backup FAILED: {type(exc).__name__}")


def main(
    argv: list[str] | None = None,
    *,
    clock: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> int:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--db", default="data/trader.db")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--backup-dir", default=None)
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--retry-until-minutes", type=int, default=0)
    ap.add_argument("--retry-every-seconds", type=int, default=300)
    ap.add_argument(
        "--broker",
        choices=("mt5", "capital", "paper"),
        default=os.environ.get("AUTOTRADER_BROKER", "mt5"),
    )
    args = ap.parse_args(argv)
    exit_code: int
    try:
        if args.retry_until_minutes <= 0:
            result = _run_trading(args)
            exit_code = result.code
        else:
            now = clock or (lambda: datetime.now(UTC))
            wait = sleep or time.sleep
            start = now()
            deadline = start + timedelta(minutes=args.retry_until_minutes)
            attempts = 0
            while True:
                attempts += 1
                result = _run_trading(args, quiet_retryable=True)
                if result.retry_reason is None:
                    exit_code = result.code
                    break
                if now() + timedelta(seconds=args.retry_every_seconds) > deadline:
                    if result.retry_reason == "market_closed":
                        logging.getLogger("autotrader").warning(
                            "give up after %d attempts: market closed / no tick",
                            attempts,
                        )
                        _alert(
                            args.symbol,
                            "warn",
                            f"ตลาดปิด/ไม่มี tick ตลอด {args.retry_until_minutes} นาที ไม่ได้ประมวลผลวันนี้",
                        )
                    else:
                        _alert(
                            args.symbol,
                            "error",
                            f"initialize ล้มเหลวตลอด {args.retry_until_minutes} นาที",
                        )
                    exit_code = result.code
                    break
                wait(args.retry_every_seconds)
    finally:
        _backup_and_alert(args)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
