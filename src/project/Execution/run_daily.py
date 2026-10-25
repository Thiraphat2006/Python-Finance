"""จุดเริ่มรันจริง (Task Scheduler เรียกไฟล์นี้วันละครั้ง)

    uv run python -m project.Execution.run_daily --symbol XAUUSD

Exit code: 0 = สำเร็จหรือข้ามเพราะรันแล้ว, 1 = ผิดพลาด (ไว้ให้ alert ข้อ 1.3 จับ)
ปฏิเสธบัญชีที่ไม่ใช่ demo เสมอ (ไม่มีตัวเลือก allow_live ที่นี่โดยตั้งใจ)
"""

import argparse
import sys
from typing import Optional

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
from project.Storage.trader_db import TraderDB


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--db", default="data/trader.db")
    ap.add_argument("--log-dir", default="logs")
    args = ap.parse_args(argv)

    log = setup_logging(args.log_dir)
    import MetaTrader5 as mt5  # Windows เท่านั้น

    if not mt5.initialize():
        log.error("initialize ไม่สำเร็จ: %s", mt5.last_error())
        return 1
    try:
        info = mt5.symbol_info(args.symbol)
        if info is None:
            log.error("ไม่พบ symbol %s", args.symbol)
            return 1
        if not info.visible:
            mt5.symbol_select(args.symbol, True)
        if not is_market_open(mt5, args.symbol):
            # ไม่บันทึกอะไรเลย (ไม่ใช้กุญแจของวัน, ไม่เก็บ quote เก่า) รันใหม่ตอนตลาดเปิดได้
            log.warning("ตลาด %s ปิดอยู่ (ไม่มี tick ใหม่) ข้ามรอบนี้", args.symbol)
            return 0
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
        return 1 if result.status == "error" else 0
    except Exception:
        log.exception("ลูปรายวันล้มเหลว")
        return 1
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())