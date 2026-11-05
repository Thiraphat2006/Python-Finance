"""MarketData ของ Capital (อ่านอย่างเดียว ไม่ส่งออเดอร์)

แท่ง D1: ใช้ bid; ยุบแท่งอาทิตย์เข้าจันทร์ก่อน แล้วตัดแท่งวันนี้ (ยังไม่ปิด) ทิ้ง
เวลาทั้งหมดเป็น UTC แบบ naive (ตรงกับ index แท่ง) ข้อมูลที่วัดแล้วอยู่ CAPITAL_API.md
"""

import logging
import math
from typing import Any

import pandas as pd

from project.Execution.mt5_data import MAX_SPREAD_POINTS, add_atr
from project.Risk.position_sizing import SymbolSpec

log = logging.getLogger("autotrader")

MAX_QUOTE_AGE_S = 300.0  # UNVERIFIED: ยังไม่วัดความถี่ tick ทอง
TZ_OFFSET = pd.Timedelta(hours=7)  # ฟิลด์ที่ไม่มี UTC ท้ายชื่อ = UTC+7


def _naive_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _bid_frame(prices: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "time": _naive_utc(p["snapshotTimeUTC"]).normalize(),
            "open": float(p["openPrice"]["bid"]),
            "high": float(p["highPrice"]["bid"]),
            "low": float(p["lowPrice"]["bid"]),
            "close": float(p["closePrice"]["bid"]),
        }
        for p in prices
    ]
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
    df = df.set_index("time").sort_index()
    return df[~df.index.duplicated(keep="last")]


_COLS = ["open", "high", "low", "close"]


def _merge_sundays(df: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """ยุบแท่งอาทิตย์เข้าจันทร์ (open อาทิตย์, high/low = max/min, close จันทร์)
    อาทิตย์ที่ไม่มีจันทร์: ทิ้ง (warning เฉพาะเมื่อจันทร์ควรมีแล้ว)"""
    rows: dict[pd.Timestamp, dict[str, float]] = {
        pd.Timestamp(ts): {c: float(df[c].iloc[i]) for c in _COLS}
        for i, ts in enumerate(df.index)
    }
    for ts in sorted(rows):
        if ts.dayofweek != 6:
            continue
        monday = ts + pd.Timedelta(days=1)
        sun = rows.pop(ts)
        if monday in rows:
            mon = rows[monday]
            mon["open"] = sun["open"]
            mon["high"] = max(sun["high"], mon["high"])
            mon["low"] = min(sun["low"], mon["low"])
        elif ts < today:
            log.warning("แท่งอาทิตย์ %s ไม่มีแท่งจันทร์ให้ยุบ ทิ้งแท่งนี้", ts.date())
    keys = sorted(rows)
    return pd.DataFrame(
        [rows[k] for k in keys], index=pd.DatetimeIndex(keys), columns=_COLS
    )


def build_d1_bars(
    prices: list[dict[str, Any]],
    server_now: pd.Timestamp,
    atr_period: int = 14,
    count: int = 120,
) -> pd.DataFrame:
    """ฟังก์ชันล้วน ไม่ใช้ network: ยุบอาทิตย์ -> ตัดแท่งวันนี้ -> tail(count) -> ATR"""
    today = server_now.normalize()
    df = _merge_sundays(_bid_frame(prices), today)
    df = df[df.index < today].tail(count)
    df.index.name = "time"
    return add_atr(df, atr_period)


class CapitalData:
    def __init__(self, client: Any, epic: str, spec: SymbolSpec):
        if not spec.point > 0:
            raise ValueError("SymbolSpec.point ต้อง > 0 (Capital GOLD = 0.01)")
        self.client = client
        self.epic = epic
        self.spec = spec

    def get_server_now(self) -> pd.Timestamp:
        return pd.Timestamp(int(self.client.get("/time")["serverTime"]), unit="ms")

    def _snapshot(self) -> dict[str, Any]:
        snap = self.client.get(f"/markets/{self.epic}").get("snapshot")
        if not snap or snap.get("bid") is None or snap.get("offer") is None:
            raise RuntimeError(f"อ่านราคา {self.epic} ไม่ได้")
        return snap

    def get_quote(self) -> tuple[float, float]:
        snap = self._snapshot()
        return float(snap["bid"]), float(snap["offer"])

    def is_market_open(self, wait_s: float = 60.0, poll_s: float = 0.5) -> bool:
        """เรียก API ครั้งเดียว ไม่รอ (wait_s/poll_s รับไว้ให้ตรง interface);
        ไม่เปิด = คืน False แล้ว retry loop ของ run_daily ลองใหม่เอง"""
        snap = self._snapshot()
        if snap.get("marketStatus") != "TRADEABLE":
            return False
        spread_points = (float(snap["offer"]) - float(snap["bid"])) / self.spec.point
        if spread_points > MAX_SPREAD_POINTS:
            return False
        stamp = snap.get("updateTime")
        if not stamp:
            return False
        last_tick = pd.Timestamp(stamp) - TZ_OFFSET  # updateTime = UTC+7
        age_s = (self.get_server_now() - last_tick).total_seconds()
        return age_s <= MAX_QUOTE_AGE_S

    def fetch_d1_bars(
        self,
        count: int = 120,
        atr_period: int = 14,
        server_now: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        now = server_now if server_now is not None else self.get_server_now()
        n = math.ceil((count + 1) * 1.25) + 5  # เผื่อแท่งอาทิตย์ที่จะยุบ
        resp = self.client.get(
            f"/prices/{self.epic}", params={"resolution": "DAY", "max": n}
        )
        prices = resp.get("prices") or []
        if not prices:
            raise RuntimeError(f"ดึงแท่ง D1 ของ {self.epic} ไม่ได้")
        return build_d1_bars(prices, now, atr_period=atr_period, count=count)

    def log_recent_deals(self, db: Any, days: int = 7) -> int:
        """no-op จนรู้ที่มา pnl/swap (ROADMAP 1.8 (5))"""
        return 0
