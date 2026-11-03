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
) -> None:
    market_items = markets.get("markets", []) if isinstance(markets, dict) else []
    epic_found = any(
        isinstance(item, dict) and item.get("epic") == epic for item in market_items
    )
    instrument = market.get("instrument", {})
    rules = market.get("dealingRules", {})
    hours = market.get("openingHours", {})
    prices_items = prices.get("prices", []) if isinstance(prices, dict) else []
    position_items = (
        positions.get("positions", []) if isinstance(positions, dict) else []
    )
    first = prices_items[0] if prices_items else {}
    last = prices_items[-1] if prices_items else {}
    currency = value(
        instrument, "currency", "code", default=value(instrument, "currency")
    )
    print(f"epic found: {epic_found}")
    print(
        "marketStatus: "
        f"{market.get('marketStatus')}; "
        f"bid/offer/spread: {market.get('bid')}/{market.get('offer')}/"
        f"{market.get('spread')}"
    )
    print(
        "instrument: "
        f"lotSize={instrument.get('lotSize')}; "
        f"type/currency/marginFactor={instrument.get('type')}/{currency}/"
        f"{instrument.get('marginFactor')}"
    )
    print(
        "dealingRules: "
        f"minDealSize={format_rule(rules.get('minDealSize'))}, "
        f"maxDealSize={format_rule(rules.get('maxDealSize'))}, "
        f"minSizeIncrement={format_rule(rules.get('minSizeIncrement'))}, "
        "minStopOrProfitDistance="
        f"{format_rule(rules.get('minStopOrProfitDistance'))}"
    )
    print(f"openingHours: {format_opening_hours(hours)}")
    print(f"overnightFee: {market.get('overnightFee')}")
    print(f"hedgingMode: {market.get('hedgingMode')}")
    print(f"open positions: {len(position_items)}")
    print(
        "DAY bars: "
        f"first={first.get('snapshotTimeUTC')}/{first.get('snapshotTime')}; "
        f"last={last.get('snapshotTimeUTC')}/{last.get('snapshotTime')}"
    )
    timezone_offset = (
        server_time.get("timezoneOffset") if isinstance(server_time, dict) else None
    )
    print(f"timezoneOffset: {market.get('timezoneOffset', timezone_offset)}")


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
