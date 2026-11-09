"""H12 step 1a (read-only): D1 bar coverage of the 9 candidates on MT5.

Entry rule (EXPERIMENTS.md, H12): D1 bars continuous from <= 2010-12 to 2023-12,
no gap > 5 business days. Computes NO returns. Run with the MT5 terminal open.
"""

from datetime import UTC, datetime

import MetaTrader5 as mt5
import numpy as np

CANDIDATES = [
    "EURUSD",
    "USDJPY",
    "GBPUSD",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
    "USDCHF",
    "USDNOK",
    "USDSEK",
]
FIRST_BY = np.datetime64("2010-12-31")
END = np.datetime64("2023-12-31")
MAX_GAP_BDAYS = 5


def check_days(days: np.ndarray) -> tuple[bool, str]:
    """days: sorted unique datetime64[D] of bars. Returns (passed, detail)."""
    idx = int(np.searchsorted(days, FIRST_BY, side="right")) - 1
    if idx < 0:
        return False, f"first bar {days[0]} is after {FIRST_BY}"
    seg = np.append(days[idx : np.searchsorted(days, END, side="right")], END)
    gaps = np.busday_count(seg[:-1], seg[1:])
    worst = int(np.argmax(gaps))
    max_gap = int(gaps[worst])
    detail = f"max_gap={max_gap}bd ({seg[worst]} -> {seg[worst + 1]})"
    return max_gap <= MAX_GAP_BDAYS, detail


def check(symbol: str) -> str:
    if not mt5.symbol_select(symbol, True):
        return f"{symbol:8s} NOT_FOUND ({mt5.last_error()})"
    rates = mt5.copy_rates_range(
        symbol,
        mt5.TIMEFRAME_D1,
        datetime(2000, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 15, tzinfo=UTC),
    )
    if rates is None or len(rates) == 0:
        return f"{symbol:8s} NO_DATA ({mt5.last_error()})"
    days = np.unique(rates["time"].astype("datetime64[s]").astype("datetime64[D]"))
    ok, detail = check_days(days)
    return (
        f"{symbol:8s} {'PASS' if ok else 'FAIL'}  bars={len(days)} "
        f"first={days[0]} last={days[-1]}  {detail}"
    )


def main() -> int:
    if not mt5.initialize():
        print(f"mt5.initialize failed: {mt5.last_error()}")
        return 1
    try:
        for sym in CANDIDATES:
            print(check(sym))
    finally:
        mt5.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
