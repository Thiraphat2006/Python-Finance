"""คำนวณขนาดไม้จากความเสี่ยงต่อเทรด (fixed fractional) + ATR stop

สมมติฐาน: บัญชีเป็น USD และ quote currency ของ symbol เป็น USD
(XAUUSD, EURUSD, US500 ใช้ได้) ถ้าใช้คู่อื่นต้องแปลงมูลค่าเอง
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolSpec:
    """ข้อมูลสัญญาของ symbol ดึงจาก mt5.symbol_info ได้ (trade_contract_size, volume_min ฯลฯ)"""

    symbol: str
    point: float = 0.01
    contract_size: float = 100.0  # XAUUSD = 100 oz ต่อ 1 lot, EURUSD = 100000
    volume_min: float = 0.01
    volume_step: float = 0.01
    volume_max: float = 100.0


def atr_stop(entry: float, atr: float, side: str, mult: float = 2.0) -> float:
    """ราคา stop-loss = entry -/+ mult * ATR"""
    if atr <= 0 or math.isnan(atr):
        raise ValueError("ATR ต้องมากกว่า 0")
    if side == "long":
        return entry - mult * atr
    if side == "short":
        return entry + mult * atr
    raise ValueError("side ต้องเป็น 'long' หรือ 'short'")


def round_down_to_step(volume: float, step: float) -> float:
    steps = math.floor(volume / step + 1e-9)
    return round(steps * step, 8)


def position_size(
    equity: float,
    risk_fraction: float,
    entry: float,
    stop: float,
    spec: SymbolSpec,
) -> float:
    """จำนวน lot ที่ถ้าโดน stop แล้วขาดทุน ~ equity * risk_fraction

    ปัดลงเสมอ (ไม่ปัดขึ้น) และคืน 0.0 ถ้าไม้เล็กสุดของโบรกเกอร์ยังเสี่ยงเกินที่ตั้งไว้
    เพื่อไม่ให้ความเสี่ยงจริงสูงกว่ากฎ
    """
    if equity <= 0 or risk_fraction <= 0:
        return 0.0
    distance = abs(entry - stop)
    if distance <= 0:
        return 0.0
    risk_money = equity * risk_fraction
    raw = risk_money / (distance * spec.contract_size)
    lots = round_down_to_step(min(raw, spec.volume_max), spec.volume_step)
    if lots < spec.volume_min:
        return 0.0
    return lots