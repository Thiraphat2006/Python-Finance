"""รัน walk-forward ทุกคู่ symbol x timeframe ในครั้งเดียว แล้วสรุปเป็นตารางเดียว

Usage (ใน notebook):
    from project.Untility.run_matrix import run_matrix
    summary, folds = run_matrix(["XAUUSD", "EURUSD", "US500"], ["H1", "H4", "D1"])

ผลจะถูกบันทึกที่ results/summary.csv และ results/folds.csv ส่งสองไฟล์นี้ (หรือก๊อปตารางสรุปที่พิมพ์ออกมา) มาดูได้เลย
ชื่อ symbol ต้องตรงกับ Market Watch ใน MT5 ของ broker (เช่น US500 / SPX500 / USTEC)
"""

import contextlib
import datetime
import io
import time
from pathlib import Path
from typing import Optional, Sequence

import MetaTrader5 as mt5
import pandas as pd

from project.Untility.data_loader import DataLoader
from project.Untility.evaluate import walk_forward
from project.Untility.spread import fix_zero_spread

TIMEFRAMES = {
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

# เกณฑ์ผ่าน กำหนดไว้ล่วงหน้า ห้ามแก้หลังเห็นผล (ไม่งั้นก็คือ overfit ระดับเกณฑ์)
PASS_OOS_SHARPE = 0.5
PASS_POSITIVE_FRAC = 2 / 3  # สัดส่วนรอบทดสอบที่ test_sharpe > 0
MIN_FOLDS = 6  # น้อยกว่านี้ถือว่าข้อมูลสั้นเกินจะสรุป


def _run_pair(
    symbol: str,
    tf_name: str,
    date_from: datetime.datetime,
    date_to: datetime.datetime,
    terminal_path: Optional[str],
    train_years: int,
    test_years: int,
    fix_spread: bool = True,
    extra_spread_points: float = 0.0,
):
    loader = DataLoader(symbol, TIMEFRAMES[tf_name], terminal_path=terminal_path)
    raw = loader.load_with_date(date_from, date_to)
    point = loader.point()

    row = {
        "symbol": symbol,
        "tf": tf_name,
        "bars": len(raw),
        "start": str(raw.index.min().date()),
        "end": str(raw.index.max().date()),
    }
    if "spread" in raw.columns:
        # spread ในประวัติเก่าของ MT5 บางที่เป็น 0 ทำให้ต้นทุนต่ำเกินจริง (วัดจากข้อมูลดิบก่อนแก้)
        row["median_spread_pts"] = float(raw["spread"].median())
        row["zero_spread_%"] = round(float((raw["spread"] == 0).mean()) * 100, 1)

    # แทน spread=0 ด้วย median ที่ไม่ใช่ 0 ของปีเดียวกัน (ต้นทุนไม่มีทางลดลง) ดู spread.py
    if fix_spread and "spread" in raw.columns:
        raw, _ = fix_zero_spread(raw)
    row["spread_fixed"] = bool(fix_spread and "spread" in raw.columns)
    row["extra_spread_pts"] = extra_spread_points

    # walk_forward พิมพ์รายละเอียดเยอะ เก็บไว้ไม่ให้รกตอนรันหลายคู่
    with contextlib.redirect_stdout(io.StringIO()):
        res = walk_forward(
            raw,
            train_years=train_years,
            test_years=test_years,
            point=point,
            extra_spread_points=extra_spread_points,
        )

    folds = res.get("folds", pd.DataFrame())
    if "oos_summary" not in res or len(folds) == 0:
        row["note"] = "ข้อมูลสั้นเกินไปสำหรับ walk-forward"
        row["pass"] = False
        return row, pd.DataFrame()

    s = res["oos_summary"]
    row.update(
        {
            "folds": s["folds"],
            "oos_sharpe": s["oos_sharpe"],
            "oos_return_%": s["oos_return_%"],
            "oos_maxdd_%": s["oos_maxdd_%"],
            "folds_positive": s["folds_positive"],
            "folds_beat_median": s["folds_beat_median"],
            "avg_train_sharpe": s["avg_train_sharpe"],
            "avg_test_sharpe": s["avg_test_sharpe"],
        }
    )
    row["pass"] = bool(
        s["folds"] >= MIN_FOLDS
        and s["oos_sharpe"] > PASS_OOS_SHARPE
        and s["folds_positive"] >= PASS_POSITIVE_FRAC * s["folds"]
    )
    row["note"] = "" if s["folds"] >= MIN_FOLDS else f"folds < {MIN_FOLDS}"

    folds = folds.copy()
    folds.insert(0, "tf", tf_name)
    folds.insert(0, "symbol", symbol)
    return row, folds


def run_matrix(
    symbols: Sequence[str] = ("XAUUSD", "EURUSD", "US500"),
    timeframes: Sequence[str] = ("H1", "H4", "D1"),
    date_from: datetime.datetime = datetime.datetime(2010, 1, 1),
    date_to: datetime.datetime = datetime.datetime(2024, 1, 1),
    train_years: int = 2,
    test_years: int = 1,
    terminal_path: Optional[str] = None,
    out_dir: str = "results",
    fix_spread: bool = True,
    extra_spread_points: float = 0.0,
):
    """walk-forward ทุกคู่ symbol x timeframe; คู่ไหนโหลดไม่ได้หรือพังจะข้ามพร้อมบอกเหตุผล ไม่ล้มทั้งชุด

    fix_spread: แทน spread=0 ในประวัติด้วย median ที่ไม่ใช่ 0 ของปีเดียวกัน (ค่าเริ่มต้นเปิด)
    extra_spread_points: บวก spread เพิ่มทุกแท่ง (หน่วย point) ไว้ทดสอบว่าผลทนต้นทุนที่แย่กว่าจริงได้ไหม
    ระวัง: out_dir ค่าเดิม "results" จะทับไฟล์ผลชุดก่อนหน้า ถ้าอยากเก็บไว้เทียบให้ส่ง out_dir="results_spreadfix"
    """
    unknown = [t for t in timeframes if t not in TIMEFRAMES]
    if unknown:
        raise ValueError(f"timeframe ไม่รองรับ: {unknown} (ใช้ได้: {list(TIMEFRAMES)})")

    pairs = [(s, t) for s in symbols for t in timeframes]
    rows, all_folds = [], []
    for i, (symbol, tf_name) in enumerate(pairs, 1):
        t0 = time.time()
        print(f"[{i}/{len(pairs)}] {symbol} {tf_name} ...", end=" ", flush=True)
        try:
            row, folds = _run_pair(
                symbol, tf_name, date_from, date_to, terminal_path, train_years, test_years,
                fix_spread, extra_spread_points,
            )
            rows.append(row)
            if len(folds):
                all_folds.append(folds)
            print(f"เสร็จ ({time.time() - t0:.0f} วินาที)")
        except Exception as e:  # noqa: BLE001 - อยากให้ทั้งชุดรันต่อ
            rows.append(
                {"symbol": symbol, "tf": tf_name, "pass": False,
                 "note": f"ERROR {type(e).__name__}: {e}"}
            )
            print(f"ข้าม ({type(e).__name__}: {e})")

    summary = pd.DataFrame(rows)
    if "oos_sharpe" in summary.columns:
        summary = summary.sort_values("oos_sharpe", ascending=False, na_position="last")
    folds_df = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    folds_df.to_csv(out / "folds.csv", index=False)

    print("\n=== สรุป out-of-sample ทุกคู่ (เรียงตาม oos_sharpe) ===")
    print(summary.to_string(index=False))
    n_pass = int(summary["pass"].sum()) if "pass" in summary else 0
    print(
        f"\nเกณฑ์ผ่าน (ตั้งไว้ล่วงหน้า): oos_sharpe > {PASS_OOS_SHARPE}, "
        f"folds_positive >= {PASS_POSITIVE_FRAC:.0%} ของรอบ, folds >= {MIN_FOLDS}"
    )
    print(f"ลองทั้งหมด {len(pairs)} คู่ ผ่าน {n_pass} คู่")
    if n_pass:
        print("คู่ที่ผ่านต้องยืนยันซ้ำด้วยข้อมูลช่วงอื่น/สินทรัพย์ใกล้เคียง "
              "เพราะลองหลายคู่ย่อมมีบางคู่ดูดีโดยบังเอิญ")
    print(f"บันทึกแล้วที่ {out / 'summary.csv'} และ {out / 'folds.csv'}")
    return summary, folds_df


if __name__ == "__main__":
    run_matrix()