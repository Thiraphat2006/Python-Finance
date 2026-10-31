"""Paper broker: จำลองการส่งออเดอร์ด้วยต้นทุนแบบเดียวกับ backtest (ครึ่ง spread ต่อการเข้า/ออก)

รองรับ 1 สถานะต่อ symbol (พอสำหรับตอนนี้) interface ตั้งใจให้ live broker (MT5) ใช้ชื่อเมธอดเดียวกันได้ภายหลัง
"""

from dataclasses import dataclass

import pandas as pd

from project.Risk.position_sizing import SymbolSpec


@dataclass
class Position:
    side: str  # "long" | "short"
    lots: float
    entry: float
    stop: float
    entry_time: pd.Timestamp


@dataclass(frozen=True)
class Trade:
    side: str
    lots: float
    entry: float
    exit: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    pnl: float
    reason: str


class PaperBroker:
    def __init__(self, spec: SymbolSpec, cash: float = 10_000.0):
        self.spec = spec
        self.cash = cash
        self.position: Position | None = None
        self.trades: list[Trade] = []

    # ---------- state ----------
    def equity(self, price: float) -> float:
        return self.cash + self._unrealized(price)

    def _unrealized(self, price: float) -> float:
        p = self.position
        if p is None:
            return 0.0
        sign = 1 if p.side == "long" else -1
        return sign * (price - p.entry) * p.lots * self.spec.contract_size

    @property
    def open_positions(self) -> int:
        return 0 if self.position is None else 1

    # ---------- orders ----------
    def open(
        self,
        side: str,
        lots: float,
        price: float,
        half_spread: float,
        stop: float,
        ts: pd.Timestamp,
    ) -> None:
        if self.position is not None:
            raise RuntimeError("มีสถานะเปิดอยู่แล้ว")
        if lots <= 0:
            raise ValueError("lots ต้องมากกว่า 0")
        # long ซื้อที่ ask (สูงกว่า) / short ขายที่ bid (ต่ำกว่า)
        fill = price + half_spread if side == "long" else price - half_spread
        self.position = Position(side, lots, fill, stop, ts)

    def close(
        self, price: float, half_spread: float, ts: pd.Timestamp, reason: str
    ) -> Trade:
        p = self.position
        if p is None:
            raise RuntimeError("ไม่มีสถานะให้ปิด")
        fill = price - half_spread if p.side == "long" else price + half_spread
        sign = 1 if p.side == "long" else -1
        pnl = sign * (fill - p.entry) * p.lots * self.spec.contract_size
        self.cash += pnl
        trade = Trade(p.side, p.lots, p.entry, fill, p.entry_time, ts, pnl, reason)
        self.trades.append(trade)
        self.position = None
        return trade

    def check_stop(
        self, o: float, h: float, l: float, half_spread: float, ts: pd.Timestamp
    ) -> Trade | None:
        """ถ้าแท่งนี้แตะ stop -> ปิดที่ stop (ถ้าเปิดแท่งกระโดดข้าม stop ปิดที่ open ซึ่งแย่กว่า)"""
        p = self.position
        if p is None:
            return None
        if p.side == "long" and l <= p.stop:
            return self.close(min(p.stop, o), half_spread, ts, "stop")
        if p.side == "short" and h >= p.stop:
            return self.close(max(p.stop, o), half_spread, ts, "stop")
        return None
