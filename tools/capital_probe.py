import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from project.Storage.notify import load_env

BASE_URL = "https://demo-api-capital.backend-capital.com/api/v1"
EXPECTED_HOST = "demo-api-capital.backend-capital.com"
PROBE_DIR = Path("data/probe")
REDACTED = "<redacted>"
REDACT_KEYS = {"accountId", "clientId", "currentAccountId"}
REQUEST_DELAY = 1.1


class ProbeStopped(Exception):
    def __init__(self, status: int) -> None:
        self.status = status


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: REDACTED if key in REDACT_KEYS else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def save_response(name: str, response: httpx.Response) -> Any:
    payload = redact(response.json())
    (PROBE_DIR / name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def request_json(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
) -> httpx.Response:
    time.sleep(REQUEST_DELAY)
    response = client.request(method, path, headers=headers, json=json_body)
    if response.status_code in (401, 429):
        raise ProbeStopped(response.status_code)
    response.raise_for_status()
    return response


def value(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current


def format_rule(rule: Any) -> str:
    if not isinstance(rule, dict):
        return str(rule)
    return f"{rule.get('value', rule)} {rule.get('unit', '')}".strip()


def format_opening_hours(hours: Any) -> str:
    if not isinstance(hours, dict):
        return str(hours)
    days = ", ".join(
        f"{day}={hours.get(day)}"
        for day in (
            "mon",
            "tue",
            "wed",
            "thu",
            "fri",
            "sat",
            "sun",
        )
    )
    return f"{days}; zone={hours.get('zone')}"


def print_summary(
    epic: str,
    markets: Any,
    market: dict[str, Any],
    positions: Any,
    prices: Any,
    server_time: Any,
    preferences: Any,
    session: Any,
) -> None:
    missing = "MISSING"

    def field(data: Any, key: str) -> Any:
        return data.get(key, missing) if isinstance(data, dict) else missing

    def rule(data: Any, key: str) -> str:
        item = field(data, key)
        if item == missing:
            return missing
        return format_rule(item)

    market_items = markets.get("markets", []) if isinstance(markets, dict) else []
    epic_found = any(
        isinstance(item, dict) and item.get("epic") == epic for item in market_items
    )
    snapshot = market.get("snapshot", {})
    instrument = market.get("instrument", {})
    rules = market.get("dealingRules", {})
    hours = field(instrument, "openingHours")
    prices_items = prices.get("prices", []) if isinstance(prices, dict) else []
    position_items = (
        positions.get("positions", []) if isinstance(positions, dict) else []
    )
    spread = (
        snapshot["offer"] - snapshot["bid"]
        if isinstance(snapshot.get("offer"), (int, float))
        and isinstance(snapshot.get("bid"), (int, float))
        else missing
    )
    last_bars = prices_items[-3:]
    if hours == missing:
        opening_hours = missing
    elif isinstance(hours, dict):
        opening_hours = "; ".join(
            f"{day}={hours.get(day, missing)}"
            for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
        )
        opening_hours += f"; zone={hours.get('zone', missing)}"
    else:
        opening_hours = str(hours)
    print(f"epic found: {epic_found}")
    print(
        "marketStatus: "
        f"{field(snapshot, 'marketStatus')}; "
        f"bid/offer/spread: {field(snapshot, 'bid')}/{field(snapshot, 'offer')}/"
        f"{spread}; updateTime={field(snapshot, 'updateTime')}; "
        f"marketModes={field(snapshot, 'marketModes')}"
    )
    print(
        "instrument: "
        f"lotSize={field(instrument, 'lotSize')}; "
        f"type/currency/marginFactor={field(instrument, 'type')}/"
        f"{field(instrument, 'currency')}/{field(instrument, 'marginFactor')}; "
        f"marginFactorUnit={field(instrument, 'marginFactorUnit')}; "
        f"guaranteedStopAllowed={field(instrument, 'guaranteedStopAllowed')}"
    )
    print(
        "dealingRules: "
        f"minDealSize={rule(rules, 'minDealSize')}, "
        f"maxDealSize={rule(rules, 'maxDealSize')}, "
        f"minSizeIncrement={rule(rules, 'minSizeIncrement')}, "
        "minStopOrProfitDistance="
        f"{rule(rules, 'minStopOrProfitDistance')}"
    )
    print(f"openingHours: {opening_hours}")
    print(f"overnightFee: {field(instrument, 'overnightFee')}")
    print(f"hedgingMode: {field(preferences, 'hedgingMode')}")
    print(f"leverages: {field(preferences, 'leverages')}")
    print(f"open positions: {len(position_items)}")
    print(
        "DAY bars: "
        f"count={len(prices_items)}; last3="
        f"{[(field(bar, 'snapshotTime'), field(bar, 'snapshotTimeUTC')) for bar in last_bars]}"
    )
    print(f"timezoneOffset: {field(session, 'timezoneOffset')}")
    print(f"serverTime: {field(server_time, 'serverTime')}")


def run(epic: str) -> None:
    if urlsplit(BASE_URL).hostname != EXPECTED_HOST:
        raise RuntimeError("Probe base URL host is not the Capital.com demo host")

    load_env()
    credentials = {
        "CAP_API_KEY": os.environ.get("CAP_API_KEY"),
        "CAP_IDENTIFIER": os.environ.get("CAP_IDENTIFIER"),
        "CAP_API_PASSWORD": os.environ.get("CAP_API_PASSWORD"),
    }
    if not all(credentials.values()):
        raise RuntimeError(
            "CAP_API_KEY, CAP_IDENTIFIER, and CAP_API_PASSWORD must be set in .env"
        )

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    token_headers: dict[str, str] = {}
    session_started = False
    with httpx.Client(timeout=10.0, base_url=BASE_URL) as client:
        try:
            response = request_json(
                client,
                "POST",
                "/session",
                headers={"X-CAP-API-KEY": credentials["CAP_API_KEY"] or ""},
                json_body={
                    "identifier": credentials["CAP_IDENTIFIER"],
                    "password": credentials["CAP_API_PASSWORD"],
                    "encryptedPassword": False,
                },
            )
            session_started = True
            token_headers = {
                "CST": response.headers["CST"],
                "X-SECURITY-TOKEN": response.headers["X-SECURITY-TOKEN"],
            }
            save_response("session.json", response)

            get_calls = [
                ("/session", "session.json", token_headers),
                ("/accounts", "accounts.json", token_headers),
                ("/accounts/preferences", "preferences.json", token_headers),
                ("/markets?searchTerm=gold", "markets_search_gold.json", token_headers),
                (f"/markets/{epic}", f"market_{epic}.json", token_headers),
                (
                    f"/prices/{epic}?resolution=DAY&max=30",
                    f"prices_day_{epic}.json",
                    token_headers,
                ),
                ("/positions", "positions.json", token_headers),
                ("/workingorders", "workingorders.json", token_headers),
                ("/time", "time.json", None),
            ]
            payloads: dict[str, Any] = {}
            for path, filename, headers in get_calls:
                response = request_json(client, "GET", path, headers=headers)
                payloads[filename] = save_response(filename, response)
            print_summary(
                epic,
                payloads["markets_search_gold.json"],
                payloads[f"market_{epic}.json"],
                payloads["positions.json"],
                payloads[f"prices_day_{epic}.json"],
                payloads["time.json"],
                payloads["preferences.json"],
                payloads["session.json"],
            )
        finally:
            if session_started:
                request_json(client, "DELETE", "/session", headers=token_headers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Capital.com demo REST API")
    parser.add_argument("--epic", default="GOLD")
    args = parser.parse_args()
    try:
        run(args.epic)
    except ProbeStopped as exc:
        print(f"HTTP status: {exc.status}")


if __name__ == "__main__":
    main()
