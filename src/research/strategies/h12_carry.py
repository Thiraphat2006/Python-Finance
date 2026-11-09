"""H12: FX carry cross-sectional (รายเดือน) ล็อกสเปกใน EXPERIMENTS.md ก่อนรัน

กฎ (ห้ามแก้หลังเห็นผล):
- จัดอันดับสกุลต่างประเทศ 9 สกุลตามดอกเบี้ย 3 เดือนของเดือน M-1: top n = +1, bottom n = -1, ที่เหลือ 0
  เสมอกัน = เรียงตามตัวอักษร
- น้ำหนักสกุล = สัญญาณ x (target_vol / vol_60d) / N_active เพดาน max_weight (เหมือน TSMOM)
  สกุลที่ USD เป็น base (USDJPY ฯลฯ) = ตำแหน่งคู่ตรงข้ามน้ำหนักสกุล
- ตัดสินใจที่ปิดวันทำการสุดท้ายของเดือน M-1 เข้า open วันทำการแรกของเดือน M (ใช้ monthly_rebalanced + shift(1))
- ผลตอบแทน = ราคา (open->open) + carry (น้ำหนัก x (r_สกุล - r_USD) เดือน M-1 / 12 กระจายรายวัน) - ต้นทุนครึ่ง spread
- ข้อมูลถูกตัดที่ LAST_DATE (holdout 2024+ ห้ามแตะ) ทั้งราคาและดอกเบี้ย

ผู้เรียกต้องแก้ spread (fix_zero_spread) มาก่อนส่งเข้า data
data = {"EURUSD": DataFrame D1 [open, close, spread]}, points = {"EURUSD": point}
rates = DataFrame ดอกเบี้ย % ต่อปี index รายเดือน (DatetimeIndex หรือ PeriodIndex) คอลัมน์ EUR JPY ... USD
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from research.strategies.tsmom import metrics, monthly_rebalanced

LAST_DATE = pd.Timestamp("2023-12-31")  # hard cap: holdout 2024-01 เป็นต้นไป ห้ามแตะ
START_DATE = pd.Timestamp("2010-01-01")

# pair -> (สกุลต่างประเทศ, +1 ถ้าสกุลนั้นเป็น base / -1 ถ้า USD เป็น base)
PAIRS: dict[str, tuple[str, int]] = {
    "EURUSD": ("EUR", 1),
    "USDJPY": ("JPY", -1),
    "GBPUSD": ("GBP", 1),
    "AUDUSD": ("AUD", 1),
    "USDCAD": ("CAD", -1),
    "NZDUSD": ("NZD", 1),
    "USDCHF": ("CHF", -1),
    "USDNOK": ("NOK", -1),
    "USDSEK": ("SEK", -1),
}

# เกณฑ์ผ่าน (ล็อกไว้ล่วงหน้า)
PASS_SHARPE = 0.5
PASS_POSITIVE_YEARS = 0.60
PASS_MAX_DD = 0.25
PASS_STRESS_SHARPE = 0.3
STRESS_SPREAD_MULT = 2.0
STRESS_CARRY_MULT = 0.5
PLACEBO_SEED = 12


@dataclass(frozen=True)
class H12Config:
    n_legs: int = 3  # 2 และ 4 ใช้ดูความเสถียรเท่านั้น
    vol_window: int = 60
    target_vol: float = 0.10
    max_weight: float = 1.0
    periods_per_year: int = 252
    spread_mult: float = 1.0
    carry_mult: float = 1.0


@dataclass(frozen=True)
class _Prep:
    idx: pd.DatetimeIndex
    ccys: list[str]
    sign: pd.Series
    vol: pd.DataFrame
    active: pd.DataFrame
    n_active: pd.Series
    o2o: pd.DataFrame
    half_spread: pd.DataFrame  # สัดส่วนราคา ที่ spread_mult = 1
    carry_unit: pd.DataFrame  # carry ต่อวันต่อน้ำหนัก 1.0
    rates_m: pd.DataFrame


def _cap(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("index ต้องเป็น DatetimeIndex")
    return df.loc[df.index <= LAST_DATE]


def _month_rates(rates: pd.DataFrame) -> pd.DataFrame:
    raw = rates.index
    if isinstance(raw, pd.DatetimeIndex):
        periods = raw.to_period("M")
    elif isinstance(raw, pd.PeriodIndex):
        periods = raw.asfreq("M")
    else:
        raise TypeError("rates index ต้องเป็น DatetimeIndex หรือ PeriodIndex")
    out = rates.astype(float).copy()
    out.index = periods
    return out.loc[out.index <= LAST_DATE.to_period("M")]


def rank_signal(rates_row: pd.Series, n_legs: int) -> pd.Series:
    """+1 top n, -1 bottom n ตามดอกเบี้ย; เสมอ = ตัวอักษร; มี NaN หรือสกุลไม่พอ = 0 ทั้งแถว"""
    out = pd.Series(0.0, index=rates_row.index)
    if rates_row.isna().any() or len(rates_row) < 2 * n_legs:
        return out
    hi = sorted(rates_row.index, key=lambda c: (-float(rates_row[c]), str(c)))[:n_legs]
    lo_all = sorted(rates_row.index, key=lambda c: (float(rates_row[c]), str(c)))
    lo = [c for c in lo_all if c not in hi][:n_legs]
    out[hi] = 1.0
    out[lo] = -1.0
    return out


def monthly_signals(
    rates_m: pd.DataFrame, ccys: list[str], n_legs: int
) -> pd.DataFrame:
    """สัญญาณรายเดือน index = เดือนของดอกเบี้ย (M-1 ของเดือนที่ถือ)"""
    sub = rates_m[ccys]
    rows = [rank_signal(row, n_legs).to_numpy() for _, row in sub.iterrows()]
    vals = np.vstack(rows) if rows else np.empty((0, len(ccys)))
    return pd.DataFrame(vals, index=sub.index, columns=ccys)


def _daily(sig_m: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """ขยายสัญญาณรายเดือนเป็นรายวัน: วัน t ใช้ดอกเบี้ยของเดือนที่ t อยู่ (ใช้ที่ปิดวันสุดท้ายของเดือน)"""
    sig = sig_m.reindex(idx.to_period("M")).fillna(0.0)
    sig.index = idx
    return sig


def carry_unit(
    rates_m: pd.DataFrame, ccys: list[str], idx: pd.DatetimeIndex
) -> pd.DataFrame:
    """carry รายวันต่อน้ำหนัก 1.0: (r_สกุล - r_USD) ของเดือน M-1 / 100 / 12 หารจำนวนวันในเดือน M"""
    periods = idx.to_period("M")
    diff = rates_m[ccys].sub(rates_m["USD"], axis=0) / 100.0 / 12.0
    unit = diff.reindex(periods - 1).fillna(0.0)
    unit.index = idx
    n_days = pd.Series(1.0, index=idx).groupby(periods).transform("sum")
    return unit.div(n_days, axis=0)


def _prepare(
    data: Mapping[str, pd.DataFrame],
    rates: pd.DataFrame,
    points: Mapping[str, float],
) -> _Prep:
    if not data:
        raise ValueError("ต้องมีข้อมูลอย่างน้อย 1 คู่")
    unknown = set(data) - set(PAIRS)
    if unknown:
        raise ValueError(f"คู่ที่ไม่อยู่ใน universe: {sorted(unknown)}")
    capped = {s: _cap(df) for s, df in data.items()}
    idx: pd.DatetimeIndex | None = None
    for sym, df in capped.items():
        for col in ("open", "close"):
            if col not in df.columns:
                raise ValueError(f"{sym} ไม่มีคอลัมน์ {col}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"{sym}: index ต้องเป็น DatetimeIndex")
        idx = df.index if idx is None else idx.union(df.index)
    if idx is None:
        raise ValueError("ต้องมีข้อมูลอย่างน้อย 1 คู่")
    idx = idx.sort_values()

    syms = [s for s in PAIRS if s in capped]
    ccys = [PAIRS[s][0] for s in syms]
    rates_m = _month_rates(rates)
    missing = [c for c in [*ccys, "USD"] if c not in rates_m.columns]
    if missing:
        raise ValueError(f"rates ขาดคอลัมน์ {missing}")

    full = {s: capped[s].reindex(idx).ffill() for s in syms}
    close = pd.DataFrame({PAIRS[s][0]: full[s]["close"].astype(float) for s in syms})
    opens = pd.DataFrame({PAIRS[s][0]: full[s]["open"].astype(float) for s in syms})
    half = pd.DataFrame(
        {
            PAIRS[s][0]: (
                full[s]["spread"].astype(float) * points[s] / 2 / opens[PAIRS[s][0]]
                if "spread" in full[s].columns
                else 0.0
            )
            for s in syms
        },
        index=idx,
    ).fillna(0.0)

    daily_ret = close / close.shift(1) - 1
    vol = daily_ret.rolling(60).std() * np.sqrt(252)
    active = vol.notna() & (vol > 0)
    return _Prep(
        idx=idx,
        ccys=ccys,
        sign=pd.Series({PAIRS[s][0]: float(PAIRS[s][1]) for s in syms}),
        vol=vol,
        active=active,
        n_active=active.sum(axis=1).clip(lower=1),
        o2o=opens.shift(-1) / opens - 1,
        half_spread=half,
        carry_unit=carry_unit(rates_m, ccys, idx),
        rates_m=rates_m,
    )


def _simulate(prep: _Prep, sig_daily: pd.DataFrame, cfg: H12Config) -> dict[str, Any]:
    w = (sig_daily * (cfg.target_vol / prep.vol)).div(prep.n_active, axis=0)
    w = w.replace([np.inf, -np.inf], np.nan).clip(-cfg.max_weight, cfg.max_weight)
    w = w.where(prep.active, 0.0).fillna(0.0)

    decided = monthly_rebalanced(w)  # รู้ที่ปิดวันสุดท้ายของเดือน
    held = decided.shift(1).fillna(0.0)  # held[d] = น้ำหนักสกุลระหว่าง open d -> open d+1
    pair_w = held * prep.sign  # ตำแหน่งคู่จริง (USDxxx = ตรงข้าม)

    gross = (pair_w * prep.o2o).sum(axis=1, min_count=1)
    turnover = pair_w.diff().abs()
    turnover.iloc[0] = pair_w.iloc[0].abs()
    cost = (turnover * prep.half_spread * cfg.spread_mult).sum(axis=1)
    carry = (held * prep.carry_unit).sum(axis=1) * cfg.carry_mult

    net = (gross + carry - cost).dropna()
    live = held.abs().sum(axis=1).gt(0)
    if not live.any():
        raise ValueError("ไม่มีวันที่มีน้ำหนัก (ดอกเบี้ยไม่ครบ?)")
    live_days = live.index[live.to_numpy()]
    start = max(pd.Timestamp(live_days[0]), START_DATE)
    net = net.loc[net.index >= start]
    return {
        "returns": net,
        "price_returns": (gross - cost).loc[net.index],
        "carry_returns": carry.loc[net.index],
        "cost": cost.loc[net.index],
        "weights": held.loc[net.index],
        "pair_weights": pair_w.loc[net.index],
        "equity": (1 + net).cumprod(),
        "cfg": cfg,
    }


def run_h12(
    data: Mapping[str, pd.DataFrame],
    rates: pd.DataFrame,
    points: Mapping[str, float],
    cfg: H12Config | None = None,
) -> dict[str, Any]:
    cfg = cfg or H12Config()
    prep = _prepare(data, rates, points)
    sig_m = monthly_signals(prep.rates_m, prep.ccys, cfg.n_legs)
    return _simulate(prep, _daily(sig_m, prep.idx), cfg)


def placebo_sharpes(
    data: Mapping[str, pd.DataFrame],
    rates: pd.DataFrame,
    points: Mapping[str, float],
    cfg: H12Config | None = None,
    n: int = 1000,
    seed: int = PLACEBO_SEED,
) -> np.ndarray:
    """สลับสกุลที่ได้ขา +1/-1/0 แบบสุ่มอิสระรายเดือน (carry ที่ได้ยังตามดอกเบี้ยจริงของสกุลที่ถือ)"""
    cfg = cfg or H12Config()
    prep = _prepare(data, rates, points)
    vals = monthly_signals(prep.rates_m, prep.ccys, cfg.n_legs).to_numpy()
    cols = prep.ccys
    month_idx = monthly_signals(prep.rates_m, prep.ccys, cfg.n_legs).index
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        order = np.argsort(rng.random(vals.shape), axis=1)
        shuffled = pd.DataFrame(
            np.take_along_axis(vals, order, axis=1), index=month_idx, columns=cols
        )
        res = _simulate(prep, _daily(shuffled, prep.idx), cfg)
        out[i] = metrics(res["returns"])["sharpe"]
    return out


def evaluate_pass(
    data: Mapping[str, pd.DataFrame],
    rates: pd.DataFrame,
    points: Mapping[str, float],
    cfg: H12Config | None = None,
    placebo_n: int = 1000,
) -> dict[str, Any]:
    """ผลหลัก + stress (carry x0.5 และ spread x2 พร้อมกัน) + ขา 2/3/4 + แยกราคา/carry + placebo"""
    cfg = cfg or H12Config()
    res = run_h12(data, rates, points, cfg)
    base = metrics(res["returns"])
    stress_cfg = replace(
        cfg, spread_mult=STRESS_SPREAD_MULT, carry_mult=STRESS_CARRY_MULT
    )
    stress = metrics(run_h12(data, rates, points, stress_cfg)["returns"])
    legs = {
        k: metrics(run_h12(data, rates, points, replace(cfg, n_legs=k))["returns"])[
            "sharpe"
        ]
        for k in (2, 3, 4)  # ดูความเสถียรเท่านั้น ห้ามเลือกตัวดีสุด
    }
    price_m = metrics(res["price_returns"])
    carry_ann_pct = round(
        float(res["carry_returns"].mean()) * cfg.periods_per_year * 100, 2
    )

    placebo: dict[str, Any] = {}
    if placebo_n > 0:
        dist = placebo_sharpes(data, rates, points, cfg, n=placebo_n)
        placebo = {
            "n": placebo_n,
            "real_sharpe": base["sharpe"],
            "fraction_below_real": round(float((dist < base["sharpe"]).mean()), 3),
            "placebo_mean": round(float(np.nanmean(dist)), 2),
            "placebo_p95": round(float(np.nanpercentile(dist, 95)), 2),
        }

    checks = {
        f"sharpe > {PASS_SHARPE}": bool(base["sharpe"] > PASS_SHARPE),
        f"positive years >= {PASS_POSITIVE_YEARS:.0%}": bool(
            base["positive_years_%"] >= PASS_POSITIVE_YEARS * 100
        ),
        f"max DD < {PASS_MAX_DD:.0%}": bool(base["maxdd_%"] < PASS_MAX_DD * 100),
        f"stress sharpe > {PASS_STRESS_SHARPE}": bool(
            stress["sharpe"] > PASS_STRESS_SHARPE
        ),
    }
    return {
        "base": base,
        "stress": stress,
        "legs_sharpe": legs,
        "price_part": price_m,
        "carry_part_ann_%": carry_ann_pct,
        "placebo": placebo,
        "checks": checks,
        "pass": all(checks.values()),
        "n_pairs": len(data),
    }
