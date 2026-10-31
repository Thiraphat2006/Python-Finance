"""แก้ต้นทุน spread ในประวัติ MT5 ที่เป็น 0 (ทำให้ backtest ถูกเกินจริง)

หลักการ: spread == 0 ในประวัติเก่าคือ "ไม่มีข้อมูล" ไม่ใช่ spread ศูนย์จริง
จึงแทนด้วย median ของ spread ที่ไม่ใช่ 0 ใน "ปีเดียวกัน" ของ symbol นั้น
(ถ้าทั้งปีไม่มีค่าเลยใช้ median ทั้งช่วง) ไม่ต้องเดาตัวเลขเอง และแก้ได้ทางเดียว: ต้นทุนเพิ่มขึ้นหรือเท่าเดิม ไม่ลดลง

Usage:
    from project.Untility.spread import fix_zero_spread
    raw_fixed, info = fix_zero_spread(raw)
"""

import pandas as pd


def fix_zero_spread(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """คืน (DataFrame ใหม่ที่ spread ไม่มี 0, รายงาน) ไม่แก้ DataFrame ต้นฉบับ"""
    info = {"zero_before_%": 0.0, "replaced_bars": 0, "fixed": False}
    if "spread" not in df.columns or len(df) == 0:
        return df, info

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index ต้องเป็น DatetimeIndex")

    s = df["spread"].astype(float)
    zero = s <= 0
    info["zero_before_%"] = round(float(zero.mean()) * 100, 1)
    info["replaced_bars"] = int(zero.sum())
    if not zero.any():
        return df, info

    valid = s.where(~zero)  # ค่าที่ใช้ได้ (ไม่ใช่ 0)
    overall = valid.median()
    if pd.isna(overall):
        # ทั้งชุดเป็น 0 หมด ไม่มีอะไรให้อ้างอิง ให้คนตัดสินใจเอง
        raise ValueError("spread เป็น 0 ทุกแท่ง แก้อัตโนมัติไม่ได้ ใส่ extra_spread_points เอง")

    yearly = valid.groupby(df.index.year).transform("median")
    fill = yearly.fillna(overall)

    out = df.copy()
    out["spread"] = s.where(~zero, fill)
    info["fixed"] = True
    return out, info
