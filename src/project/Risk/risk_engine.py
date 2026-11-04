"""Risk engine: กฎตายตัวที่ไม่ขึ้นกับกลยุทธ์ (ไม่มี AI/ML ในนี้)

กฎ:
1. daily loss limit: ขาดทุนวันนี้ (นับ unrealized ด้วย) เกิน X% ของ equity ต้นวัน -> ห้ามเปิดใหม่ + ปิดสถานะทั้งหมด จนถึงวันใหม่ (UTC)
2. kill switch: drawdown จาก peak เกิน Y% -> หยุดถาวร (latched) จนกว่าคนจะเรียก reset_kill() เอง
3. จำกัดจำนวนสถานะรวม, 1 สถานะต่อ symbol, spread สูงผิดปกติ -> ไม่เข้า
4. เพดาน open risk รวม (max_total_risk): ผลรวมความเสี่ยงของสถานะที่ถืออยู่ + เทรดใหม่ ต้องไม่เกินเพดาน

ถือหลายสถานะได้โดยตั้ง max_open_positions > 1 (ค่าเริ่มต้นยัง 1 เหมือนเดิม replay/test เดิมไม่เปลี่ยน)
daily limit / kill switch เป็นระดับบัญชีเสมอ

สถานะข้ามการรัน: ลูปรายวันเริ่มโปรแกรมใหม่ทุกครั้ง ต้องบันทึก/โหลดผ่าน save_engine / load_engine
(ไม่เช่นนั้น peak equity และ kill switch จะหายทุกเช้า)
"""

import math
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.005  # 0.5% ของ equity ต่อเทรด
    daily_loss_limit: float = 0.02  # 2% ของ equity ต้นวัน
    max_drawdown_kill: float = 0.10  # 10% จาก peak
    atr_stop_mult: float = 2.0
    max_open_positions: int = 1
    max_spread_points: float | None = None  # None = ไม่จำกัด
    max_total_risk: float = 0.02  # เพดานความเสี่ยงที่เปิดอยู่รวมกัน (สัดส่วนของ equity)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = "ok"


def position_risk_money(pos: Any, spec: Any) -> float:
    """เงินที่จะเสียถ้าสถานะนี้โดน stop (inf ถ้าไม่มี stop ที่ยืนยันได้)

    pos ต้องมี side/lots/entry/stop (Position จาก paper_broker หรือ mt5_broker)
    spec ต้องมี contract_size (SymbolSpec)
    """
    if pos.stop is None:
        return math.inf
    if pos.side == "long":
        distance = pos.entry - pos.stop
    else:
        distance = pos.stop - pos.entry
    return max(0.0, distance) * pos.lots * spec.contract_size


def open_risk_fraction(items: Iterable[tuple[Any, Any]], equity: float) -> float:
    """ความเสี่ยงรวมของสถานะที่ถืออยู่ เป็นสัดส่วนของ equity; items = [(position, spec), ...]"""
    if equity <= 0:
        return 0.0
    return sum(position_risk_money(p, s) for p, s in items) / equity


class RiskEngine:
    def __init__(self, cfg: RiskConfig, equity: float):
        self.cfg = cfg
        self.peak_equity = equity
        self.day: date | None = None
        self.day_start_equity = equity
        self.killed = False
        self.kill_reason: str | None = None

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
            self.kill_reason = (
                f"max drawdown {dd:.1%} >= {self.cfg.max_drawdown_kill:.1%}"
            )

    def reset_kill(self) -> None:
        """ให้คนเรียกเองหลังตรวจสอบแล้วเท่านั้น (ห้ามเรียกจากโค้ดเทรดอัตโนมัติ)
        เรียกแล้วต้อง save_engine() อีกครั้ง ไม่เช่นนั้นรอบถัดไปจะโหลดสถานะ killed เดิมกลับมา"""
        self.killed = False
        self.kill_reason = None
        self.peak_equity = self.day_start_equity

    # ---------- persistence ----------
    def export_state(self) -> dict[str, str]:
        return {
            "peak_equity": repr(self.peak_equity),
            "day": self.day.isoformat() if self.day else "",
            "day_start_equity": repr(self.day_start_equity),
            "killed": "1" if self.killed else "0",
            "kill_reason": self.kill_reason or "",
        }

    def restore_state(self, state: dict[str, str]) -> None:
        self.peak_equity = float(state["peak_equity"])
        self.day = date.fromisoformat(state["day"]) if state.get("day") else None
        self.day_start_equity = float(state["day_start_equity"])
        self.killed = state.get("killed", "0") == "1"
        self.kill_reason = state.get("kill_reason") or None

    # ---------- metrics ----------
    def drawdown(self, equity: float) -> float:
        return (
            (self.peak_equity - equity) / self.peak_equity
            if self.peak_equity > 0
            else 0.0
        )

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
        self,
        equity: float,
        open_positions: int,
        spread_points: float = 0.0,
        *,
        symbol: str | None = None,
        open_symbols: Collection[str] = (),
        open_risk: float = 0.0,
        new_risk: float | None = None,
    ) -> Decision:
        """open_risk = ความเสี่ยงรวมที่เปิดอยู่ (สัดส่วนของ equity, ดู open_risk_fraction)
        new_risk = ความเสี่ยงของเทรดใหม่ (ไม่ใส่ = risk_per_trade)"""
        if self.killed:
            return Decision(False, f"kill_switch: {self.kill_reason}")
        if self.daily_limit_hit(equity):
            return Decision(False, "daily_loss_limit")
        if symbol is not None and symbol in open_symbols:
            return Decision(False, "symbol_already_open")
        if open_positions >= self.cfg.max_open_positions:
            return Decision(False, "max_open_positions")
        add = self.cfg.risk_per_trade if new_risk is None else new_risk
        if open_risk + add > self.cfg.max_total_risk + 1e-12:
            return Decision(False, "max_total_risk")
        if (
            self.cfg.max_spread_points is not None
            and spread_points > self.cfg.max_spread_points
        ):
            return Decision(False, "spread_too_wide")
        return Decision(True)


# ---------- บันทึก/โหลดผ่าน TraderDB (duck-typed: ต้องมี set_state/get_state) ----------
def save_engine(db: Any, engine: RiskEngine, prefix: str = "risk.") -> None:
    for k, v in engine.export_state().items():
        db.set_state(prefix + k, v)


def load_engine(
    db: Any, cfg: RiskConfig, equity: float, prefix: str = "risk."
) -> RiskEngine:
    """โหลดสถานะที่บันทึกไว้ ถ้ายังไม่เคยบันทึก (ครั้งแรก) สร้างใหม่จาก equity ปัจจุบัน"""
    engine = RiskEngine(cfg, equity)
    keys = ["peak_equity", "day", "day_start_equity", "killed", "kill_reason"]
    saved = {k: db.get_state(prefix + k) for k in keys}
    if saved["peak_equity"] is None:
        return engine
    engine.restore_state({k: ("" if v is None else v) for k, v in saved.items()})
    return engine
