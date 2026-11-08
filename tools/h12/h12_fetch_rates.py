"""H12 step 1b/2: download FRED 3M interbank series, check monthly continuity, hash.

Checks 2010-01..2023-12 (168 months) have a numeric value every month. Saves raw
CSV + prints sha256 so the exact dataset version is recorded. Computes NO returns.
"""

import hashlib
import io
from pathlib import Path

import httpx
import pandas as pd

# currency -> OECD country code used in IR3TIB01{code}M156N
SERIES = {
    "EUR": "EZ",
    "JPY": "JP",
    "GBP": "GB",
    "AUD": "AU",
    "CAD": "CA",
    "NZD": "NZ",
    "CHF": "CH",
    "NOK": "NO",
    "SEK": "SE",
    "USD": "US",
    "USD*": "TB3MS",
}
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
OUT = Path("data/h12/rates")
EXPECTED = pd.date_range("2010-01-01", "2023-12-01", freq="MS")

def sid_of(cc: str) -> str:
    return cc if cc == "TB3MS" else f"IR3TIB01{cc}M156N"

def check(ccy: str, cc: str) -> bool:
    sid = sid_of(cc)
    resp = httpx.get(URL.format(sid=sid), timeout=30, follow_redirects=True)
    resp.raise_for_status()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{sid}.csv").write_bytes(resp.content)
    digest = hashlib.sha256(resp.content).hexdigest()[:16]
    df = pd.read_csv(io.BytesIO(resp.content), na_values=[".", ""])
    dates = pd.to_datetime(df.iloc[:, 0])
    vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    have = pd.DatetimeIndex(dates[vals.notna()])
    missing = EXPECTED.difference(have)
    ok = len(missing) == 0
    print(
        f"{ccy} {cc} {'PASS' if ok else 'FAIL'} rows={len(df)} "
        f"first={dates.iloc[0].date()} last={dates.iloc[-1].date()} "
        f"missing_in_window={len(missing)} sha256={digest}"
    )
    if not ok:
        print("   missing:", [d.strftime("%Y-%m") for d in missing][:12])
    return ok


def main() -> int:
    results = []
    for ccy, cc in SERIES.items():
        try:
            results.append(check(ccy, cc))
        except Exception as exc:  # noqa: BLE001
            print(f"{ccy} {cc} ERROR {exc!r}")
            results.append(False)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
