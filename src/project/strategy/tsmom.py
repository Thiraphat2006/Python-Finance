"""Time-series momentum (TSMOM) หลายสินทรัพย์ บน D1 + volatility targeting

สมมติฐาน (ตั้งไว้ล่วงหน้า): สินทรัพย์ที่ขึ้น/ลงในรอบ ~12 เดือนมีแนวโน้มไปทางเดิมต่อ
edge ที่มีหลักฐานมาจากการกระจายข้ามสินทรัพย์จำนวนมาก ถ้ามีแค่ 3 ตัวโอกาสผ่านต่ำ นั่นคือผลที่ต้องยอมรับถ้าไม่ผ่าน

กฎ (ไม่มีพารามิเตอร์ที่จูนจากข้อมูล):
- สัญญาณ ณ ปิดวัน t = sign(close_t / close_{t-lookback} - 1)  (lookback=252)
- น้ำหนักของสินทรัพย์ i = สัญญาณ * (target_vol / vol_i) / N  โดย vol_i = std ผลตอบแทนรายวัน 60 วัน x sqrt(252)
  จำกัดน้ำหนักต่อสินทรัพย์ไม่เกิน max_weight
- ปรับพอร์ตเดือนละครั้ง (วันทำการสุดท้ายของเดือน) ที่เหลือถือน้ำหนักเดิม
- ตัดสินใจที่ปิดวัน t เข้า/ออกที่ open วัน t+1 (ไม่มี lookahead)
- ต้นทุน = |เปลี่ยนน้ำหนัก| x ครึ่ง spread (ดึงจากคอลัมน์ spread ที่ผ่าน fix_zero_spread แล้ว)
- ไม่รวม swap/ค่าถือข้ามคืน (ทองและ FX มี swap จริง ผลจริงจึงแย่กว่านี้เล็กน้อย)

เกณฑ์ผ่าน (ล็อกก่อนเห็นผล ห้ามแก้หลังรัน):
- net Sharpe > 0.4 ทั้งช่วง
- สัดส่วนปีปฏิทินที่กำไร >= 60%
- max drawdown < 25%
- stress (spread x2) แล้ว Sharpe ยัง > 0.3
lookback 126/189 ใช้ดูความเสถียรเท่านั้น ห้ามเลือกตัวที่ดีที่สุด

Usage:
    from project.strategy.tsmom import run_tsmom, evaluate_pass
    res = run_tsmom(data, points)          # data = {symbol: DataFrame D1}, points = {symbol: point}
    print(evaluate_pass(data, points))
"""

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from project.Untility.spread import fix_zero_spread

# เกณฑ์ผ่าน (ล็อกไว้ล่วงหน้า)
PASS_SHARPE = 0.4
PASS_POSITIVE_YEARS = 0.60
PASS_MAX_DD = 0.25
PASS_STRESS_SHARPE = 0.3
STRESS_SPREAD_MULT = 2.0


@dataclass(frozen=True)
class TsmomConfig:
    lookback: int = 252  # วันทำการ ~12 เดือน
    vol_window: int = 60
    target_vol: float = 0.10  # เป้า volatility ของพอร์ตรวม (ต่อปี)
    max_weight: float = 1.0  # เพดานน้ำหนักต่อสินทรัพย์ (1.0 = ไม่เกินทุนเต็ม ไม่ใช้ leverage เกิน)
    periods_per_year: int = 252
    spread_mult: float = 1.0  # 2.0 = stress test


def _aligned(data: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """รวมปฏิทินทุกสินทรัพย์ (union) ตลาดที่ปิด/วันหยุดใช้ราคาเดิม = ผลตอบแทน 0 ไม่เทรดจริงวันนั้น

    สินทรัพย์ที่ประวัติเริ่มทีหลังจะมี NaN ช่วงต้น ไม่ถูกนับเข้าพอร์ตจนกว่าจะมีข้อมูลของตัวเองพอ
    (lookback + vol_window) จึงไม่ต้องตัดช่วงทดสอบของตัวอื่นตามตัวที่สั้นที่สุด
    """
    if not data:
        raise ValueError("ต้องมีข้อมูลอย่างน้อย 1 สินทรัพย์")
    idx = None
    for sym, df in data.items():
        for col in ("open", "close"):
            if col not in df.columns:
                raise ValueError(f"{sym} ไม่มีคอลัมน์ {col}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{sym}: index ต้องเป็น DatetimeIndex")
        idx = df.index if idx is None else idx.union(df.index)
    idx = idx.sort_values()
    # ffill เติมเฉพาะช่องว่างหลังข้อมูลตัวเองเริ่มแล้ว ช่วงก่อนเริ่มยังเป็น NaN
    return {sym: df.reindex(idx).ffill() for sym, df in data.items()}


def signals(close: pd.DataFrame, cfg: TsmomConfig) -> pd.DataFrame:
    """sign ของผลตอบแทน lookback วัน: +1 / -1 / 0 (NaN ช่วงอุ่นเครื่อง = 0)"""
    ret = close / close.shift(cfg.lookback) - 1
    return np.sign(ret).fillna(0.0)


def target_weights(close: pd.DataFrame, cfg: TsmomConfig) -> pd.DataFrame:
    """น้ำหนักเป้าหมาย 'รู้ ณ ปิดวัน t' (ยังไม่ปรับรายเดือน, ยังไม่ shift)"""
    daily_ret = close / close.shift(1) - 1
    vol = daily_ret.rolling(cfg.vol_window).std() * np.sqrt(cfg.periods_per_year)
    # สินทรัพย์ "active" = มีข้อมูลของตัวเองครบทั้ง lookback และ vol_window แล้ว
    active = close.shift(cfg.lookback).notna() & vol.notna() & (vol > 0)
    n_active = active.sum(axis=1).clip(lower=1)  # หารด้วยจำนวนที่ active ณ วันนั้น
    raw = (signals(close, cfg) * (cfg.target_vol / vol)).div(n_active, axis=0)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    w = raw.clip(-cfg.max_weight, cfg.max_weight)
    return w.where(active, 0.0).fillna(0.0)


def monthly_rebalanced(weights: pd.DataFrame) -> pd.DataFrame:
    """ถือน้ำหนักของวันทำการสุดท้ายของแต่ละเดือน จนถึงสิ้นเดือนถัดไป (วัน t รู้ที่ปิดวัน t)"""
    idx = weights.index
    month = idx.to_period("M")
    is_last = pd.Series(month, index=idx) != pd.Series(month, index=idx).shift(-1)  # วันสุดท้ายของเดือนในข้อมูล
    snap = weights.copy()
    snap.loc[~is_last.to_numpy()] = np.nan  # เหลือเฉพาะแถววันสุดท้ายของเดือน
    return snap.ffill().fillna(0.0)


def run_tsmom(
    data: Mapping[str, pd.DataFrame],
    points: Mapping[str, float],
    cfg: Optional[TsmomConfig] = None,
    fix_spread: bool = True,
) -> dict:
    """คืน dict: returns (รายวันสุทธิ), gross_returns, weights (ที่ถือจริงรายวัน), cost, equity"""
    cfg = cfg or TsmomConfig()
    prepared = {}
    for sym, df in data.items():
        if fix_spread and "spread" in df.columns:
            df, _ = fix_zero_spread(df)
        prepared[sym] = df
    data = _aligned(prepared)

    close = pd.DataFrame({s: d["close"].astype(float) for s, d in data.items()})
    opens = pd.DataFrame({s: d["open"].astype(float) for s, d in data.items()})

    # น้ำหนักที่ตัดสินใจที่ปิด t ถือจริงตั้งแต่ open t+1
    decided = monthly_rebalanced(target_weights(close, cfg))
    held = decided.shift(1).fillna(0.0)  # held[d] = น้ำหนักที่ถือระหว่าง open d -> open d+1

    open_to_open = opens.shift(-1) / opens - 1  # ผลตอบแทนถือจาก open d ถึง open d+1
    gross = (held * open_to_open).sum(axis=1, min_count=1)

    # ต้นทุน: เปลี่ยนน้ำหนักที่ open d x ครึ่ง spread เป็นสัดส่วนของราคา
    turnover = held.diff().abs()
    turnover.iloc[0] = held.iloc[0].abs()
    half_spread = pd.DataFrame(
        {
            s: (
                data[s]["spread"].astype(float) * cfg.spread_mult * points[s] / 2 / opens[s]
                if "spread" in data[s].columns
                else 0.0
            )
            for s in data
        },
        index=opens.index,
    ).fillna(0.0)  # ช่วงก่อนสินทรัพย์เริ่มมีข้อมูล น้ำหนัก = 0 อยู่แล้ว
    cost = (turnover * half_spread).sum(axis=1)

    net = (gross - cost).dropna()
    # ตัดช่วงอุ่นเครื่อง (ยังไม่มีสัญญาณ) ออกจากการวัดผล
    first_active = held.abs().sum(axis=1).gt(0).idxmax() if held.abs().sum().sum() > 0 else net.index[0]
    net = net.loc[net.index >= first_active]
    gross = gross.dropna().loc[net.index]
    return {
        "returns": net,
        "gross_returns": gross,
        "cost": cost.loc[net.index],
        "weights": held.loc[net.index],
        "equity": (1 + net).cumprod(),
        "cfg": cfg,
    }


def metrics(returns: pd.Series, periods_per_year: Optional[float] = None) -> dict:
    """periods_per_year=None -> นับจำนวนแท่งต่อปีจริงจากข้อมูล (ไม่ hardcode 252)"""
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return {"sharpe": np.nan, "return_%": 0.0, "maxdd_%": 0.0, "positive_years_%": np.nan}
    if periods_per_year is None:
        years = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
        periods_per_year = len(r) / years
    eq = (1 + r).cumprod()
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    return {
        "sharpe": round(float(r.mean() / r.std() * np.sqrt(periods_per_year)), 2),
        "return_%": round(float(eq.iloc[-1] - 1) * 100, 1),
        "maxdd_%": round(abs(float((eq / eq.cummax() - 1).min())) * 100, 1),
        "positive_years_%": round(float((yearly > 0).mean()) * 100, 0),
        "years": int(len(yearly)),
    }


def evaluate_pass(
    data: Mapping[str, pd.DataFrame],
    points: Mapping[str, float],
    cfg: Optional[TsmomConfig] = None,
) -> dict:
    """รันตามเกณฑ์ที่ล็อกไว้: ผลหลัก + stress (spread x2) + ความเสถียรของ lookback (ดูเฉย ๆ)"""
    cfg = cfg or TsmomConfig()
    base = metrics(run_tsmom(data, points, cfg)["returns"])
    stress_cfg = TsmomConfig(**{**cfg.__dict__, "spread_mult": STRESS_SPREAD_MULT})
    stress = metrics(run_tsmom(data, points, stress_cfg)["returns"])

    stability = {}
    for lb in (126, 189, 252):  # ดูความเสถียรเท่านั้น ไม่ใช้เลือกตัวดีสุด
        c = TsmomConfig(**{**cfg.__dict__, "lookback": lb})
        stability[lb] = metrics(run_tsmom(data, points, c)["returns"])["sharpe"]

    checks = {
        f"sharpe > {PASS_SHARPE}": bool(base["sharpe"] > PASS_SHARPE),
        f"positive years >= {PASS_POSITIVE_YEARS:.0%}": bool(
            base["positive_years_%"] >= PASS_POSITIVE_YEARS * 100
        ),
        f"max DD < {PASS_MAX_DD:.0%}": bool(base["maxdd_%"] < PASS_MAX_DD * 100),
        f"stress sharpe > {PASS_STRESS_SHARPE}": bool(stress["sharpe"] > PASS_STRESS_SHARPE),
    }
    return {
        "base": base,
        "stress_spread_x2": stress,
        "lookback_stability_sharpe": stability,
        "checks": checks,
        "pass": all(checks.values()),
        "n_assets": len(data),
    }