"""ตั้งค่า logging: ไฟล์หมุนเวียน (อ่านง่ายเวลาโปรแกรมพัง) + แสดงบนหน้าจอด้วย"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(
    log_dir: str | Path = "logs",
    name: str = "autotrader",
    level: int = logging.INFO,
) -> logging.Logger:
    """เรียกซ้ำได้ ไม่เพิ่ม handler ซ้ำ"""
    logger = logging.getLogger(name)
    if getattr(logger, "_autotrader_configured", False):
        return logger

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(_FORMAT)

    file_handler = RotatingFileHandler(
        Path(log_dir) / f"{name}.log",
        maxBytes=1_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)

    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.propagate = False
    logger._autotrader_configured = True  # type: ignore[attr-defined]
    return logger
