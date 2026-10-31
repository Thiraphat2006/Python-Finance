"""ตัวช่วยดึงข้อมูลจาก MT5 สำหรับลูปรายวัน (รับโมดูล mt5 เข้ามา จึงทดสอบด้วยของจำลองได้)

ข้อควรรู้: ฟังก์ชันในไฟล์นี้ยังไม่เคยรันกับ terminal จริง ให้ยืนยันตอน smoke test (1.1)
โดยเฉพาะ history_deals_get ที่เรื่องโซนเวลาของ MT5 ไม่ชัดเจน (จึงขยายช่วงค้นให้กว้างไว้ก่อน)
"""

import logging
import time
from datetime import datetime, timedelta

import pandas as pd

log = logging.getLogger("autotrader")

MAX_SPREAD_POINTS = 500


def get_server_now(mt5, symbol: str) -> pd.Timestamp:
    """เวลา server ประมาณจาก tick ล่าสุด (ตลาดปิด = เวลาของ tick สุดท้ายก่อนปิด)"""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"อ่าน tick ไม่ได้: {mt5.last_error()}")
    return pd.Timestamp(tick.time, unit="s")


def get_quote(mt5, symbol: str) -> tuple[float, float]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"อ่านราคาไม่ได้: {mt5.last_error()}")
    return float(tick.bid), float(tick.ask)


def is_market_open(
    mt5, symbol: str, wait_s: float = 60.0, poll_s: float = 0.5, sleep=time.sleep
) -> bool:
    """ตลาดเปิดจริงไหม: symbol เทรดได้เต็ม และมี tick ใหม่เข้ามาภายใน wait_s วินาที

    ใช้ "tick เปลี่ยน" แทนการเทียบเวลา เพราะ tick.time เป็นเวลา server (ไม่ใช่ UTC) เทียบตรง ๆ ไม่ได้
    ตลาดเปิด: คืน True ทันทีที่ tick เปลี่ยน (อาจใช้เวลาหลายสิบวินาที); ตลาดปิด: รอครบ wait_s แล้วคืน False
    """
    info = mt5.symbol_info(symbol)
    if info is None or info.trade_mode != getattr(mt5, "SYMBOL_TRADE_MODE_FULL", 4):
        return False

    def stamp(t):
        return getattr(t, "time_msc", None) or t.time

    def acceptable(tick):
        return (
            tick is not None
            and (float(tick.ask) - float(tick.bid)) / float(info.point)
            <= MAX_SPREAD_POINTS
        )

    first = mt5.symbol_info_tick(symbol)
    if first is None:
        return False
    base = stamp(first)
    waited = 0.0
    while waited < wait_s:
        sleep(poll_s)
        waited += poll_s
        t = mt5.symbol_info_tick(symbol)
        if acceptable(t) and (base is None or stamp(t) != base):
            return True
    return False


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ATR แบบ Wilder (ewm alpha=1/period); ถ้าสูตรกลยุทธ์ของคุณต่างจากนี้ ให้ใช้ของคุณแทน"""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out = df.copy()
    out["atr"] = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return out


def fetch_d1_bars(
    mt5,
    symbol: str,
    count: int = 120,
    atr_period: int = 14,
    server_now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """แท่ง D1 ที่ปิดแล้วเท่านั้น (ตัดแท่งที่วันที่ตรงกับวันปัจจุบันของ server ซึ่งกำลังก่อตัวอยู่)

    ข้อจำกัด: ช่วงเสาร์-อาทิตย์ tick สุดท้ายคือวันศุกร์ แท่งศุกร์จึงถูกมองว่ายังไม่ปิดจนถึงวันจันทร์
    (ไม่กระทบกลยุทธ์ที่รันวันธรรมดา)
    """
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, count + 1)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"ดึงแท่ง D1 ไม่ได้: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    times = pd.DatetimeIndex(pd.to_datetime(df["time"], unit="s"))
    df.index = times
    df.index.name = "time"
    df = df.drop(columns=["time"])
    now = server_now if server_now is not None else get_server_now(mt5, symbol)
    df = df[times.normalize() < now.normalize()]
    return add_atr(df, atr_period)


def log_recent_deals(db, mt5, symbol: str, magic: int | None, days: int = 7) -> int:
    """บันทึก deal ของเราจากประวัติ broker ลง DB (ซ้ำไม่เป็นไร) คืนจำนวนที่เพิ่มใหม่

    จับกรณี stop ถูกชนตอนโปรแกรมไม่ได้รันด้วย: deal ปิดจะถูกบันทึกในรอบถัดไป
    """
    now = datetime.now()
    deals = mt5.history_deals_get(now - timedelta(days=days), now + timedelta(days=2))
    if deals is None:
        log.warning("history_deals_get ล้มเหลว: %s", mt5.last_error())
        return 0
    added = 0
    for d in deals:
        if d.symbol == symbol and (magic is None or d.magic == magic):
            added += int(db.log_deal(d))
    return added
