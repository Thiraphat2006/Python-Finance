"""ประเมินกลยุทธ์แบบไม่หลอกตัวเอง: เลือกพารามิเตอร์จากช่วงฝึก แล้วดูผลบนช่วงทดสอบที่ไม่เคยแตะ

Usage (ใน notebook):
    from project.Untility.evaluate import evaluate
    res = evaluate(raw, split="2023-01-01", point=loader.point())
    res["grid"]; res["by_direction"]; res["yearly"]
"""

import itertools
from typing import Any

import numpy as np
import pandas as pd

from project.Indicator.moving_avg import MovingAvg
from project.Strategy.moving_avg_logic import MovingAvgLogic, StrategyConfig
from project.Untility.back_testing import back_test, sharpe

MIN_TRAIN_TRADES = 30  # น้อยกว่านี้ Sharpe ไม่มีความหมาย


def _score(sig: pd.DataFrame, **bt_kwargs) -> dict:
    pf = back_test(sig, **bt_kwargs)
    n = int(pf.trades.count())
    sh = sharpe(pf) if n > 0 else np.nan
    return {
        "sharpe": round(sh, 2),
        "return_%": round(float(pf.total_return()) * 100, 1),
        "maxdd_%": round(abs(float(pf.max_drawdown())) * 100, 1),
        "trades": n,
    }


def evaluate(
    raw: pd.DataFrame,
    split: str = "2023-01-01",
    adx_grid=(20, 25, 30),
    exit_modes=("opposite_trend", "ema_cross"),
    shorts=(True, False),
    **bt_kwargs,
) -> dict:
    ind = MovingAvg(
        raw
    ).compute_indicators()  # indicator ใช้เฉพาะอดีต จึงคำนวณรวมแล้วค่อยตัดได้
    cut = pd.Timestamp(split)

    rows: list[dict[str, Any]] = []
    for adx_min, mode, allow_short in itertools.product(adx_grid, exit_modes, shorts):
        cfg = StrategyConfig(adx_min=adx_min, exit_mode=mode, allow_short=allow_short)
        sig = MovingAvgLogic(ind, cfg)()
        tr = _score(sig[sig.index < cut], **bt_kwargs)
        te = _score(sig[sig.index >= cut], **bt_kwargs)
        rows.append(
            {
                "adx_min": adx_min,
                "exit_mode": mode,
                "short": allow_short,
                **{f"train_{k}": v for k, v in tr.items()},
                **{f"test_{k}": v for k, v in te.items()},
            }
        )
    grid = pd.DataFrame(rows).sort_values("train_sharpe", ascending=False)

    eligible = grid[grid["train_trades"] >= MIN_TRAIN_TRADES]
    best = eligible.iloc[0] if len(eligible) else grid.iloc[0]
    best_cfg = StrategyConfig(
        adx_min=float(best["adx_min"]),
        exit_mode=str(best["exit_mode"]),
        allow_short=bool(best["short"]),
    )

    # รายละเอียดของ config ที่ชนะช่วงฝึก บนข้อมูลทั้งช่วง
    sig_all = MovingAvgLogic(ind, best_cfg)()
    pf = back_test(sig_all, **bt_kwargs)
    trades = pf.trades.records_readable
    by_dir = (
        trades.groupby("Direction")["PnL"].agg(["count", "sum", "mean"]).round(2)
        if len(trades)
        else pd.DataFrame()
    )
    r = pf.returns()
    yearly = (
        ((1 + r).groupby(r.index.year).prod() - 1).mul(100).round(1).rename("return_%")
    )

    print(f"split = {split}  (ฝึกก่อนวันนี้ / ทดสอบหลังวันนี้)")
    print("\nทุก config เรียงตาม train_sharpe:")
    print(grid.to_string(index=False))
    print(f"\nเลือกจากช่วงฝึก: {best_cfg}")
    print(
        f"  -> train sharpe {best['train_sharpe']} | TEST sharpe {best['test_sharpe']} "
        f"({int(best['test_trades'])} trades)"
    )
    print("\nแยก long/short (ทั้งช่วง):\n", by_dir)
    print("\nผลรายปี %:\n", yearly)
    print(
        "\nวิธีอ่าน: ถ้า test_sharpe ต่ำกว่า train_sharpe มาก หรือติดลบ = ผลช่วงฝึกเป็นแค่โชค/overfit"
    )
    return {
        "grid": grid,
        "best_cfg": best_cfg,
        "by_direction": by_dir,
        "yearly": yearly,
        "pf": pf,
    }


def _ann_sharpe(r: pd.Series) -> float:
    """Sharpe แบบ annualize ตามจำนวนแท่งจริงต่อปี (ใช้กับ return ที่ต่อกันจากหลายช่วงทดสอบ)"""
    r = r.dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    years = (r.index[-1] - r.index[0]).days / 365.25
    bars_per_year = len(r) / years if years > 0 else len(r)
    return float(r.mean() / r.std() * np.sqrt(bars_per_year))


def walk_forward(
    raw: pd.DataFrame,
    train_years: int = 2,
    test_years: int = 1,
    adx_grid=(20, 25, 30),
    exit_modes=("opposite_trend", "ema_cross"),
    shorts=(True, False),
    **bt_kwargs,
) -> dict:
    """Walk-forward: เลือก config จากช่วงฝึกเท่านั้น แล้วใช้ทดสอบช่วงถัดไป เลื่อนหน้าต่างทีละ test_years

    ผลรวม (stitched) คือผล out-of-sample ล้วน ๆ ใช้เทียบกับ "median ของทุก config"
    ถ้ากลยุทธ์ที่เลือกจากช่วงฝึกไม่ชนะ median ในช่วงทดสอบ = การเลือกพารามิเตอร์ไม่มีคุณค่า

    Usage:
        res = walk_forward(raw, point=loader.point())
        res["folds"]; res["oos_summary"]; res["oos_yearly"]
    """
    ind = MovingAvg(raw).compute_indicators()
    start = ind.index.min().normalize()
    end = ind.index.max()

    # สร้างสัญญาณของทุก config ครั้งเดียว (indicator ใช้เฉพาะอดีต จึงตัดช่วงทีหลังได้)
    combos = list(itertools.product(adx_grid, exit_modes, shorts))
    sigs = {
        c: MovingAvgLogic(
            ind, StrategyConfig(adx_min=c[0], exit_mode=c[1], allow_short=c[2])
        )()
        for c in combos
    }

    folds, oos_returns = [], []
    t0 = start
    while True:
        t1 = t0 + pd.DateOffset(years=train_years)
        t2 = t1 + pd.DateOffset(years=test_years)
        if t1 >= end:
            break
        t2 = min(t2, end + pd.Timedelta(seconds=1))

        rows: list[dict[str, Any]] = []
        for c, sig in sigs.items():
            tr = _score(sig[(sig.index >= t0) & (sig.index < t1)], **bt_kwargs)
            te_sig = sig[(sig.index >= t1) & (sig.index < t2)]
            if len(te_sig) < 2:
                continue
            te = _score(te_sig, **bt_kwargs)
            rows.append({"cfg": c, "train": tr, "test": te})
        if not rows:
            break

        eligible = [
            x
            for x in rows
            if x["train"]["trades"] >= MIN_TRAIN_TRADES
            and not np.isnan(x["train"]["sharpe"])
        ]
        pool = (
            eligible
            if eligible
            else [x for x in rows if not np.isnan(x["train"]["sharpe"])]
        )
        if not pool:
            t0 = t0 + pd.DateOffset(years=test_years)
            continue
        best = max(pool, key=lambda x: x["train"]["sharpe"])

        test_sharpes = [
            x["test"]["sharpe"] for x in rows if not np.isnan(x["test"]["sharpe"])
        ]
        pf_te = back_test(
            sigs[best["cfg"]][
                (sigs[best["cfg"]].index >= t1) & (sigs[best["cfg"]].index < t2)
            ],
            **bt_kwargs,
        )
        oos_returns.append(pf_te.returns())

        folds.append(
            {
                "train": f"{t0.date()}→{t1.date()}",
                "test": f"{t1.date()}→{min(t2, end).date()}",
                "adx_min": best["cfg"][0],
                "exit_mode": best["cfg"][1],
                "short": best["cfg"][2],
                "train_sharpe": best["train"]["sharpe"],
                "test_sharpe": best["test"]["sharpe"],
                "test_return_%": best["test"]["return_%"],
                "test_maxdd_%": best["test"]["maxdd_%"],
                "test_trades": best["test"]["trades"],
                "median_test_sharpe_all_cfg": round(float(np.median(test_sharpes)), 2)
                if test_sharpes
                else np.nan,
            }
        )
        t0 = t0 + pd.DateOffset(years=test_years)

    folds_df = pd.DataFrame(folds)
    if not len(folds_df):
        print("ข้อมูลสั้นเกินไปสำหรับ walk-forward (ต้องมากกว่า train_years)")
        return {"folds": folds_df}

    oos = pd.concat(oos_returns)
    oos = oos[~oos.index.duplicated()]
    equity = (1 + oos).cumprod()
    oos_summary = {
        "folds": len(folds_df),
        "oos_sharpe": round(_ann_sharpe(oos), 2),
        "oos_return_%": round(float(equity.iloc[-1] - 1) * 100, 1),
        "oos_maxdd_%": round(abs(float((equity / equity.cummax() - 1).min())) * 100, 1),
        "folds_positive": int((folds_df["test_sharpe"] > 0).sum()),
        "folds_beat_median": int(
            (folds_df["test_sharpe"] > folds_df["median_test_sharpe_all_cfg"]).sum()
        ),
        "avg_train_sharpe": round(float(folds_df["train_sharpe"].mean()), 2),
        "avg_test_sharpe": round(float(folds_df["test_sharpe"].mean()), 2),
    }
    oos_yearly = (
        ((1 + oos).groupby(oos.index.year).prod() - 1)
        .mul(100)
        .round(1)
        .rename("return_%")
    )

    print(f"walk-forward: ฝึก {train_years} ปี / ทดสอบ {test_years} ปี")
    print(folds_df.to_string(index=False))
    print("\nสรุปผล out-of-sample (ต่อกันทุกช่วงทดสอบ):")
    for k, v in oos_summary.items():
        print(f"  {k}: {v}")
    print("\nผล OOS รายปี %:\n", oos_yearly)
    print(
        "\nวิธีอ่าน: avg_test_sharpe ต่ำกว่า avg_train_sharpe มาก = overfit; "
        "folds_beat_median ~ครึ่งหนึ่ง = การเลือก config ไม่ได้เพิ่มคุณค่า; "
        "oos_sharpe < ~0.5 และ folds_positive ไม่ถึงส่วนใหญ่ = ยังไม่มี edge"
    )
    return {
        "folds": folds_df,
        "oos_summary": oos_summary,
        "oos_yearly": oos_yearly,
        "oos_returns": oos,
    }
