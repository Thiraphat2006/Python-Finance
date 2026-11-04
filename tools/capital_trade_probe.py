"""One-off DEMO trade probe: open 0.01 GOLD with a stop, then close it, and record
the raw API responses (accountId/accountName masked). Needs --yes to place orders.
"""

import argparse
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from project.Execution.capital_client import CapitalClient, CapitalClientError

EPIC = "GOLD"
STOP_DISTANCE = 80.0  # far from price: the stop must not trigger during the probe
_HIDDEN = {"accountId", "accountName", "clientId"}


def mask(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: "<hidden>" if k in _HIDDEN else mask(v) for k, v in value.items()}
    if isinstance(value, list):
        return [mask(v) for v in value]
    return value


def step(results: dict[str, Any], name: str, call: Any) -> dict[str, Any]:
    try:
        result = mask(call())
    except (CapitalClientError, httpx.HTTPError) as exc:
        result = {"error": type(exc).__name__, "detail": str(exc)}
    results[name] = result
    print(f"\n=== {name} ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _first_deal_id(payload: dict[str, Any]) -> str | None:
    deals = payload.get("affectedDeals")
    if isinstance(deals, list) and deals and isinstance(deals[0], dict):
        deal_id = deals[0].get("dealId")
        return None if deal_id in (None, "") else str(deal_id)
    return None


def _save(results: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved: {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="place the demo order")
    parser.add_argument("--out", type=Path, default=Path("data/probe/trade_probe.json"))
    parser.add_argument("--size", type=float, default=0.01)
    args = parser.parse_args(argv)

    if not 0.01 <= args.size <= 1.0:
        print("ABORT: --size must be between 0.01 and 1.0 (demo probe)")
        return 2
    if not args.yes:
        print("dry run: add --yes to open and close a 0.01 GOLD DEMO position")
        return 2

    results: dict[str, Any] = {}
    started = datetime.now(UTC)
    with CapitalClient.from_env() as client:
        before = step(results, "positions_before", lambda: client.get("/positions"))
        if "error" in before or before.get("positions"):
            print("ABORT: cannot read positions or a position is already open")
            _save(results, args.out)
            return 1
        step(results, "accounts_before", lambda: client.get("/accounts"))
        market = step(results, "market", lambda: client.get(f"/markets/{EPIC}"))
        snapshot = market.get("snapshot")
        offer = snapshot.get("offer") if isinstance(snapshot, dict) else None
        if not isinstance(offer, int | float) or snapshot.get("marketStatus") != (
            "TRADEABLE"
        ):
            print("ABORT: market not tradeable or offer missing")
            _save(results, args.out)
            return 1

        body = {
            "epic": EPIC,
            "direction": "BUY",
            "size": args.size,
            "stopLevel": round(offer - STOP_DISTANCE, 2),
        }
        opened = step(
            results, "open_post", lambda: client.post("/positions", json=body)
        )
        reference = opened.get("dealReference")
        if not isinstance(reference, str):
            print("ORDER OUTCOME UNKNOWN: check positions in the Capital.com app")
            _save(results, args.out)
            return 1
        time.sleep(1.0)
        confirm = step(
            results, "open_confirm", lambda: client.get(f"/confirms/{reference}")
        )
        time.sleep(3.0)
        positions = step(results, "positions_open", lambda: client.get("/positions"))
        step(results, "accounts_open", lambda: client.get("/accounts"))

        deal_id = _first_deal_id(confirm)
        if deal_id is None:
            listed = positions.get("positions")
            if isinstance(listed, list) and listed and isinstance(listed[0], dict):
                position = listed[0].get("position")
                if isinstance(position, dict) and position.get("dealId"):
                    deal_id = str(position["dealId"])
        if deal_id is None:
            print("NO dealId FOUND: close the position manually in the Capital.com app")
            _save(results, args.out)
            return 1

        closed = step(
            results, "close_delete", lambda: client.delete(f"/positions/{deal_id}")
        )
        close_ref = closed.get("dealReference")
        time.sleep(1.0)
        if isinstance(close_ref, str):
            step(
                results,
                "close_confirm",
                lambda: client.get(f"/confirms/{close_ref}"),
            )
        time.sleep(2.0)
        after = step(results, "positions_after", lambda: client.get("/positions"))
        step(results, "accounts_after", lambda: client.get("/accounts"))

        stamp = "%Y-%m-%dT%H:%M:%S"
        window = {
            "from": (started - timedelta(minutes=1)).strftime(stamp),
            "to": (datetime.now(UTC) + timedelta(minutes=1)).strftime(stamp),
        }
        step(
            results,
            "history_activity",
            lambda: client.get(
                "/history/activity", params={**window, "detailed": True}
            ),
        )
        step(
            results,
            "history_transactions",
            lambda: client.get("/history/transactions", params=window),
        )
        step(
            results,
            "history_transactions_period",
            lambda: client.get("/history/transactions", params={"lastPeriod": 600}),
        )

        _save(results, args.out)
        if after.get("positions"):
            print(f"WARNING: position still open, close dealId {deal_id} manually")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
