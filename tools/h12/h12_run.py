"""H12 FX carry: one-shot runner (spec locked in EXPERIMENTS.md).

Default = dry check: structure only, NO returns/Sharpe computed or printed.
`--run` = the single real run on 2010-2023; refuses if result.json exists.
Run from anywhere: uv run python tools/h12/h12_run.py [--run]
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from research.data.data_loader import DataLoader
from research.data.spread import fix_zero_spread
from research.strategies.h12_carry import (
    LAST_DATE,
    PAIRS,
    START_DATE,
    _prepare,
    evaluate_pass,
)

ROOT = Path(__file__).resolve().parents[2]
RATES_DIR = ROOT / "data" / "h12" / "rates"
RESULT = ROOT / "data" / "h12" / "result.json"

LEAD_IN = datetime(2009, 7, 1)  # ราคาสำหรับ vol เท่านั้น (ไม่มีดอกเบี้ย = สัญญาณ 0)
DATE_TO = datetime(2023, 12, 31, 23, 59)  # holdout 2024+ ไม่โหลด

# สกุล -> (ชุด FRED, sha256 16 ตัวแรกที่บันทึกไว้ใน EXPERIMENTS.md)
SERIES = {
    "EUR": ("IR3TIB01EZM156N", "bda18d6a61bc4905"),
    "JPY": ("IR3TIB01JPM156N", "7d7336c147e74b18"),
    "GBP": ("IR3TIB01GBM156N", "40c23cc654a1380e"),
    "AUD": ("IR3TIB01AUM156N", "cc5a480e77a97ad2"),
    "CAD": ("IR3TIB01CAM156N", "74d6cbbb2cf93db7"),
    "NZD": ("IR3TIB01NZM156N", "3303a69593f6972d"),
    "CHF": ("IR3TIB01CHM156N", "5edd57c52a5e513c"),
    "NOK": ("IR3TIB01NOM156N", "93ed548691739a1b"),
    "SEK": ("IR3TIB01SEM156N", "0c9aadaef12653aa"),
    "USD": ("TB3MS", "33fae406051a658c"),
}


def load_rates() -> tuple[pd.DataFrame, dict[str, bool]]:
    cols: dict[str, pd.Series] = {}
    hash_ok: dict[str, bool] = {}
    for ccy, (sid, expected) in SERIES.items():
        path = RATES_DIR / f"{sid}.csv"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        hash_ok[ccy] = digest == expected
        raw = pd.read_csv(path, index_col="observation_date", parse_dates=True)
        cols[ccy] = pd.to_numeric(raw[sid], errors="coerce")
    rates = pd.DataFrame(cols).loc["2010-01-01":"2023-12-01"]
    if len(rates) != 168 or rates.isna().any().any():
        raise SystemExit(f"rates ไม่ครบ 168 เดือนหรือมี NaN: {rates.shape}")
    return rates, hash_ok


def load_data() -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    data: dict[str, pd.DataFrame] = {}
    points: dict[str, float] = {}
    for sym in PAIRS:
        loader = DataLoader(sym, mt5.TIMEFRAME_D1)
        raw = loader.load_with_date(LEAD_IN, DATE_TO)
        df, info = fix_zero_spread(raw)
        data[sym] = df.loc[df.index <= LAST_DATE]
        points[sym] = loader.point()
        print(
            sym,
            len(data[sym]),
            data[sym].index.min().date(),
            data[sym].index.max().date(),
            "point",
            points[sym],
            "fix_zero_spread:",
            info,
        )
    return data, points


def describe(
    data: dict[str, pd.DataFrame],
    rates: pd.DataFrame,
    points: dict[str, float],
    hash_ok: dict[str, bool],
) -> None:
    print("rates:", rates.shape, rates.index.min().date(), rates.index.max().date())
    print("rates sha256 match:", hash_ok)
    prep = _prepare(data, rates, points)
    na = prep.n_active
    active = na[na > 0]
    print("START_DATE:", START_DATE.date())
    print("prep.idx:", prep.idx.min().date(), prep.idx.max().date(), len(prep.idx))
    print("ccys:", prep.ccys)
    first = active.index.min().date() if len(active) else None
    print("first day n_active>0:", first, "| n_active last day:", int(na.iloc[-1]))


def git_head() -> str:
    r = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return r.stdout.strip() or "unknown"


def _jsonable(o: object) -> object:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="real one-shot run")
    args = ap.parse_args()

    if args.run and RESULT.exists():
        print(f"REFUSED: {RESULT} exists. H12 รันได้ครั้งเดียว")
        return 1
    rates, hash_ok = load_rates()
    if args.run and not all(hash_ok.values()):
        print(f"REFUSED: rates hash ไม่ตรง {hash_ok}")
        return 1
    if not mt5.initialize():
        print(f"mt5.initialize failed: {mt5.last_error()}")
        return 1
    try:
        data, points = load_data()
    finally:
        mt5.shutdown()

    if not args.run:
        describe(data, rates, points, hash_ok)
        print("DRY CHECK เสร็จ (ไม่ได้คำนวณผลตอบแทน)")
        return 0

    out = evaluate_pass(data, rates, points)
    payload = {
        "hypothesis": "H12",
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "rates_hash_ok": hash_ok,
        "result": out,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("x", encoding="utf-8") as f:  # "x" = ห้ามเขียนทับ
        f.write(text)
    for k, v in out.items():
        print(k, ":", v)
    print("saved:", RESULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
