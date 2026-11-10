"""DIAG-1: per-trade diagnostics for MA+ADX XAUUSD H4. Read-only, no tuning.

Questions and definitions are locked in claude/DIAG_trades.md (written before running).
Usage in the notebook, after the first cell (needs `r` and `raw`):

    import sys
    sys.path.insert(0, r"<repo>\\experiments")
    import diag_trades
    diag_trades.diagnose(r.trades, raw, ret_pct=r.summary()["return_%"])
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOLDOUT_START = pd.Timestamp("2024-01-01")
ATR_N = 14
STOP_ATR_MULT = 2.0
R_SANITY_RANGE = (0.7, 1.3)


def _prep_bars(raw: pd.DataFrame) -> pd.DataFrame:
    bars = raw.copy()
    bars.columns = [str(c).lower() for c in bars.columns]
    if "time" in bars.columns:
        bars = bars.set_index("time")
    bars.index = pd.to_datetime(bars.index)
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    missing = {"open", "high", "low", "close"} - set(bars.columns)
    if missing:
        raise ValueError(
            f"raw missing {sorted(missing)}; columns = {list(raw.columns)}"
        )
    bars = bars.sort_index()
    if bars.index.max() >= HOLDOUT_START:
        raise ValueError("bars reach holdout (>= 2024-01-01); refusing to run")
    return bars[["open", "high", "low", "close"]]


def _atr_wilder(bars: pd.DataFrame) -> pd.Series:
    prev_close = bars["close"].shift(1)
    tr = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / ATR_N, adjust=False, min_periods=ATR_N).mean()


def _trade_table(trades: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    atr_prev = _atr_wilder(bars).shift(1)  # ATR of the bar before entry (no lookahead)
    rows = []
    for t in trades.itertuples(index=False):
        sign = 1.0 if t.side == "long" else -1.0
        seg = bars.loc[t.entry_time : t.exit_time]
        if t.reason != "stop" and len(seg) > 1 and seg.index[-1] == t.exit_time:
            seg = seg.iloc[:-1]  # signal exit happens at the open of the exit bar
        if sign > 0:
            mfe = seg["high"].max() - t.entry
            mae = t.entry - seg["low"].min()
        else:
            mfe = t.entry - seg["low"].min()
            mae = seg["high"].max() - t.entry
        mfe, mae = max(mfe, 0.0), max(mae, 0.0)
        move = sign * (t.exit - t.entry)
        r_unit = STOP_ATR_MULT * atr_prev.get(t.entry_time, np.nan)
        rows.append(
            {
                "side": t.side,
                "reason": t.reason,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "pnl": t.pnl,
                "move": move,
                "mfe": mfe,
                "mae": mae,
                "r_unit": r_unit,
                "bars_held": len(seg),
                "hold_days": (t.exit_time - t.entry_time).total_seconds() / 86400,
            }
        )
    df = pd.DataFrame(rows)
    for col in ("move", "mfe", "mae"):
        df[col + "_r"] = df[col] / df["r_unit"]
    df["giveback"] = np.where(
        df["move"] > 0, (df["mfe"] - df["move"]) / df["mfe"], np.nan
    )
    return df


def _q1(df: pd.DataFrame) -> None:
    print("\n=== Q1 โครงสร้างกำไร/ขาดทุน ===")
    win, lose = df[df["pnl"] > 0], df[df["pnl"] <= 0]
    gross_win, gross_loss = win["pnl"].sum(), -lose["pnl"].sum()
    print(
        f"n={len(df)} win_rate={len(win) / len(df) * 100:.1f}% "
        f"avg_win={win['pnl'].mean():.2f} avg_loss={lose['pnl'].mean():.2f} "
        f"PF={gross_win / gross_loss if gross_loss else float('inf'):.2f} "
        f"expectancy={df['pnl'].mean():.2f} total={df['pnl'].sum():.2f}"
    )
    for key in ("side", "reason"):
        g = df.groupby(key)["pnl"].agg(
            n="count", total="sum", wins=lambda s: (s > 0).sum()
        )
        print(f"\nby {key}:\n{g.round(2).to_string()}")
    top = win["pnl"].sort_values(ascending=False)  # winners only
    if gross_win > 0:
        print(
            f"\nbest 1 = {top.iloc[0] / gross_win * 100:.0f}% of gross win, "
            f"best {min(3, len(top))} = {top.iloc[:3].sum() / gross_win * 100:.0f}%; "
            f"total without best trade = {df['pnl'].sum() - top.iloc[0]:.2f}"
        )


def _q2(df: pd.DataFrame) -> None:
    print("\n=== Q2 MFE/MAE ===")
    stops = df[(df["reason"] == "stop") & df["r_unit"].notna()]
    if len(stops):
        ratio = (stops["move"].abs() / stops["r_unit"]).median()
        ok = R_SANITY_RANGE[0] <= ratio <= R_SANITY_RANGE[1]
        print(
            f"R sanity: median(|move|/R) of stop trades = {ratio:.2f} -> {'OK' if ok else 'R not usable -> use USD columns'}"
        )
    else:
        ok = False
        print("R sanity: no stop trades with ATR -> R not verifiable; use USD columns")
    losers, winners = df[df["pnl"] <= 0], df[df["pnl"] > 0]
    unit = "_r" if ok else ""
    thr = 1.0 if ok else float("nan")
    if ok:
        print(
            f"losers with MFE>=1R: {(losers['mfe_r'] >= thr).sum()}/{len(losers)} | "
            f"stop trades with MFE>=1R: {(stops['mfe_r'] >= thr).sum()}/{len(stops)}"
        )
    print(
        f"winners: median giveback={winners['giveback'].median():.2f} "
        f"median MAE{unit}={winners['mae' + unit].median():.2f}"
    )
    cols = ["side", "reason", "entry_time", "pnl", "mfe", "mae"] + (
        ["mfe_r", "mae_r"] if ok else []
    )
    print("\nper trade (mfe/mae in USD/oz):")
    print(
        df[cols]
        .round({"pnl": 2, "mfe": 2, "mae": 2, "mfe_r": 2, "mae_r": 2})
        .to_string()
    )


def _q3(df: pd.DataFrame, bars: pd.DataFrame) -> None:
    print("\n=== Q3 รายปี (ตามปีที่ปิด) เทียบ buy&hold ===")
    by_year = df.groupby(df["exit_time"].dt.year)["pnl"].agg(n="count", pnl="sum")
    yr_close = bars["close"].groupby(bars.index.year).last()
    prev = yr_close.shift(1)
    prev.iloc[0] = bars["open"].iloc[0]
    by_year["buyhold_%"] = (yr_close / prev - 1) * 100
    print(by_year.round(2).to_string())


def _q4(df: pd.DataFrame, bars: pd.DataFrame, ret_pct: float | None) -> None:
    print("\n=== Q4 เทียบ buy&hold ===")
    bh = (bars["close"].iloc[-1] / bars["open"].iloc[0] - 1) * 100
    exposure = df["bars_held"].sum() / len(bars) * 100
    strat = f"{ret_pct:.2f}%" if ret_pct is not None else "n/a"
    print(f"buy&hold={bh:.2f}% strategy={strat} exposure~{exposure:.1f}% of bars")
    print(
        f"pnl long={df.loc[df['side'] == 'long', 'pnl'].sum():.2f} short={df.loc[df['side'] == 'short', 'pnl'].sum():.2f}"
    )


def _q5(df: pd.DataFrame) -> None:
    print("\n=== Q5 ระยะเวลาถือ (median วัน) ===")
    df = df.assign(outcome=np.where(df["pnl"] > 0, "win", "loss"))
    print(
        df.groupby("outcome")["hold_days"].agg(["count", "median"]).round(2).to_string()
    )
    print(
        df.groupby("reason")["hold_days"].agg(["count", "median"]).round(2).to_string()
    )


def diagnose(
    trades: pd.DataFrame, raw: pd.DataFrame, ret_pct: float | None = None
) -> pd.DataFrame:
    """Print Q1-Q5 and return the per-trade table. Changes nothing, writes nothing."""
    bars = _prep_bars(raw)
    df = _trade_table(trades, bars)
    _q1(df)
    _q2(df)
    _q3(df, bars)
    _q4(df, bars, ret_pct)
    _q5(df)
    return df
