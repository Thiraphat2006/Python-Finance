"""Risk engine: กฎตายตัวที่ไม่ขึ้นกับกลยุทธ์ (ไม่มี AI/ML ในนี้)

กฎ:
1. daily loss limit: ขาดทุนวันนี้ (นับ unrealized ด้วย) เกิน X% ของ equity ต้นวัน -> ห้ามเปิดใหม่ + ปิดสถานะทั้งหมด จนถึงวันใหม่ (UTC)
2. kill switch: drawdown จาก peak เกิน Y% -> หยุดถาวร (latched) จนกว่าคนจะเรียก reset_kill() เอง
3. จำกัดจำนวนสถานะ, spread สูงผิดปกติ -> ไม่เข้า
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.005  # 0.5% ของ equity ต่อเทรด
    daily_loss_limit: float = 0.02  # 2% ของ equity ต้นวัน
    max_drawdown_kill: float = 0.10  # 10% จาก peak
    atr_stop_mult: float = 2.0
    max_open_positions: int = 1
    max_spread_points: Optional[float] = None  # None = ไม่จำกัด


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = "ok"


class RiskEngine:
    def __init__(self, cfg: RiskConfig, equity: float):
        self.cfg = cfg
        self.peak_equity = equity
        self.day: Optional[date] = None
        self.day_start_equity = equity
        self.killed = False
        self.kill_reason: Optional[str] = None

    # ---------- state ----------
    def update(self, ts: pd.Timestamp, equity: float) -> None:
        """เรียกทุกครั้งที่รู้ equity ใหม่ (ต้นแท่งและปิดแท่ง)"""
        d = ts.date()
        if d != self.day:  # วันใหม่ (UTC) -> รีเซ็ตฐานของ daily limit
            self.day = d
            self.day_start_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        dd = self.drawdown(equity)
        if not self.killed and dd >= self.cfg.max_drawdown_kill:
            self.killed = True
            self.kill_reason = f"max drawdown {dd:.1%} >= {self.cfg.max_drawdown_kill:.1%}"

    def reset_kill(self) -> None:
        """ให้คนเรียกเองหลังตรวจสอบแล้วเท่านั้น (ห้ามเรียกจากโค้ดเทรดอัตโนมัติ)"""
        self.killed = False
        self.kill_reason = None
        self.peak_equity = self.day_start_equity

    # ---------- metrics ----------
    def drawdown(self, equity: float) -> float:
        return (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0

    def daily_loss(self, equity: float) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return (self.day_start_equity - equity) / self.day_start_equity

    def daily_limit_hit(self, equity: float) -> bool:
        return self.daily_loss(equity) >= self.cfg.daily_loss_limit

    # ---------- decisions ----------
    def should_flatten(self, equity: float) -> bool:
        return self.killed or self.daily_limit_hit(equity)

    def check_entry(
        self, equity: float, open_positions: int, spread_points: float = 0.0
    ) -> Decision:
        if self.killed:
            return Decision(False, f"kill_switch: {self.kill_reason}")
        if self.daily_limit_hit(equity):
            return Decision(False, "daily_loss_limit")
        if open_positions >= self.cfg.max_open_positions:
            return Decision(False, "max_open_positions")
        if (
            self.cfg.max_spread_points is not None
            and spread_points > self.cfg.max_spread_points
        ):
            return Decision(False, "spread_too_wide")
        return Decision(True)