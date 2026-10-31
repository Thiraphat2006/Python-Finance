"""Signal + กลยุทธ์ที่ลูปรายวันเรียกใช้

Strategy interface (duck-typed):
    name: str
    key(bars, now) -> Timestamp   กุญแจกันรันซ้ำ (กลยุทธ์รายแท่งใช้เวลาแท่งล่าสุด)
    decide(bars, position, now) -> Signal

ScheduledTestStrategy ใช้ทดสอบ "ระบบ" ไม่ใช่หากำไร: เปิด/ปิดตามตารางวันตายตัวด้วยไม้เล็กสุด
เพื่อให้ได้ข้อมูล fill/swap/commission จริงจำนวนมากพอใน forward test (TSMOM ปรับพอร์ตแค่เดือนละครั้ง)
"""

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from project.Risk.position_sizing import SymbolSpec


@dataclass(frozen=True)
class Signal:
    action: str = "hold"  # enter_long | enter_short | exit | hold
    reason: str = "ok"
    lots: Optional[float] = (
        None  # None = ให้ fixed fractional คำนวณ; มีค่า = เพดานจากกลยุทธ์ (min กับ fixed fractional)
    )
    target_weight: Optional[float] = None  # ไว้บันทึกลง runs (TSMOM)
    ignore_risk_sizing: bool = (
        False  # True = ใช้ lots ตรง ๆ (โหมดทดสอบ) แต่ยังผ่าน risk engine ทุกกฎ
    )


class ScheduledTestStrategy:
    """เปิดวันจันทร์/พุธ ปิดวันอังคาร/พฤหัส (ค่าเริ่มต้น) ตามวันของ `now` (UTC)

    ถ้าพลาดวันปิด (เครื่องดับ) สถานะที่ถือครบ max_hold_days จะถูกปิดเองเป็น reason "stale"
    """

    name = "test_schedule"

    def __init__(
        self,
        spec: SymbolSpec,
        side: str = "long",
        entry_weekdays: tuple[int, ...] = (0, 2),
        exit_weekdays: tuple[int, ...] = (1, 3),
        max_hold_days: int = 2,
    ):
        if side not in ("long", "short"):
            raise ValueError("side ต้องเป็น long หรือ short")
        self.spec = spec
        self.side = side
        self.entry_weekdays = entry_weekdays
        self.exit_weekdays = exit_weekdays
        self.max_hold_days = max_hold_days

    def key(self, bars: pd.DataFrame, now: pd.Timestamp) -> pd.Timestamp:
        # ใช้ "วันที่รัน" ไม่ใช่เวลาแท่ง: วันจันทร์อาจยังไม่มีแท่งใหม่ (ตลาดปิดเสาร์-อาทิตย์) แต่ต้องเทรดตามตาราง
        return now.normalize()

    def decide(self, bars: pd.DataFrame, position: Any, now: pd.Timestamp) -> Signal:
        wd = now.weekday()
        if position is not None:
            if wd in self.exit_weekdays:
                return Signal("exit", "schedule")
            if (now - position.entry_time).days >= self.max_hold_days:
                return Signal("exit", "stale")
            return Signal("hold")
        if wd in self.entry_weekdays:
            return Signal(
                f"enter_{self.side}",
                "schedule",
                lots=self.spec.volume_min,
                ignore_risk_sizing=True,
            )
        return Signal("hold")
