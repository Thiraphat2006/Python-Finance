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
from datetime import datetime, timedelta, timezone
from typing import Callable, Literal, NamedTuple, Optional

import pandas as pd

from project.Execution.daily_runner import run_once
from project.Execution.mt5_broker import MT5Broker
from project.Execution.mt5_data import (
    fetch_d1_bars,
    get_quote,
    get_server_now,
    is_market_open,
)
from project.Execution.signals import ScheduledTestStrategy
from project.Risk.position_sizing import SymbolSpec
from project.Risk.risk_engine import RiskConfig
from project.Storage.log_setup import setup_logging
from project.Storage.notify import load_env, send
from project.Storage.backup import run_backup
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
    except Exception as exc:
        # Notifications must never change the trading flow.
        logging.getLogger(__name__).warning("แจ้งเตือนไม่สำเร็จ: %s", type(exc).__name__)


def _run_trading(
    args: argparse.Namespace, *, quiet_retryable: bool = False
) -> TradingResult:
    log = setup_logging(args.log_dir)
    import MetaTrader5 as mt5  # Windows เท่านั้น

    if not mt5.initialize():
        log.error("initialize ไม่สำเร็จ: %s", mt5.last_error())
        if quiet_retryable:
            log.warning("initialize ไม่สำเร็จ จะลองใหม่")
        else:
            _alert(args.symbol, "error", "initialize ไม่สำเร็จ")
        return TradingResult(1, "init_failed")
    try:
        info = mt5.symbol_info(args.symbol)
        if info is None:
            log.error("ไม่พบ symbol %s", args.symbol)
            _alert(args.symbol, "error", "ไม่พบ symbol")
            return 1
        if not info.visible:
            mt5.symbol_select(args.symbol, True)
        if not is_market_open(mt5, args.symbol):
            # ไม่บันทึกอะไรเลย (ไม่ใช้กุญแจของวัน, ไม่เก็บ quote เก่า) รันใหม่ตอนตลาดเปิดได้
            log.warning("ตลาด %s ปิดอยู่ (ไม่มี tick ใหม่) ข้ามรอบนี้", args.symbol)
            if not quiet_retryable:
                _alert(args.symbol, "ok", "ตลาดปิด ข้ามรอบ")
            return TradingResult(0, "market_closed")
        spec = SymbolSpec(
            args.symbol, point=info.point, contract_size=info.trade_contract_size,
            volume_min=info.volume_min, volume_step=info.volume_step,
            volume_max=info.volume_max,
        )
        broker = MT5Broker(spec, args.symbol, mt5_module=mt5, initialize=False)
        strategy = ScheduledTestStrategy(spec)
        with TraderDB(args.db) as db:
            bars = fetch_d1_bars(mt5, args.symbol, server_now=get_server_now(mt5, args.symbol))
            result = run_once(
                broker=broker, db=db, cfg=RiskConfig(), strategy=strategy, spec=spec,
                bars=bars, quote=get_quote(mt5, args.symbol),
                now=pd.Timestamp.now(tz="UTC").tz_localize(None),
                mt5=mt5, magic=broker.magic, logger=log,
            )
        if result.sync_note:
            _alert(args.symbol, "warn", f"สถานะไม่ตรง broker: {result.sync_note}")
        if result.status == "error":
            _alert(args.symbol, "error", result.reason)
        elif result.status == "skipped":
            _alert(args.symbol, "ok", "ข้าม (ประมวลผลแล้ว)")
        elif result.decision == "exit" and result.reason in ("kill_switch", "daily_limit"):
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
        mt5.shutdown()


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
    argv: Optional[list[str]] = None,
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
    args = ap.parse_args(argv)
    exit_code: int
    try:
        if args.retry_until_minutes <= 0:
            result = _run_trading(args)
            exit_code = result.code
        else:
            now = clock or (lambda: datetime.now(timezone.utc))
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