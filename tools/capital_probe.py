import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from project.Execution.capital_client import (
    CapitalApiError,
    CapitalClient,
    CapitalClientError,
)
from project.Storage.notify import load_env

BASE_URL = "https://demo-api-capital.backend-capital.com/api/v1"
EXPECTED_HOST = "demo-api-capital.backend-capital.com"
PROBE_DIR = Path("data/probe")
REDACTED = "<redacted>"
REDACT_KEYS = {"accountId", "accountName", "clientId", "currentAccountId"}


class ProbeStopped(Exception):
    def __init__(self, status: int, error_keys: tuple[str, ...] = ()) -> None:
        self.status = status
        self.error_keys = error_keys


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: REDACTED if key in REDACT_KEYS else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def save_response(name: str, payload: Any) -> Any:
    (PROBE_DIR / name).write_text(
        json.dumps(redact(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def get_payload(client: CapitalClient, path: str) -> dict[str, Any]:
    try:
        return client.get(path)
    except CapitalApiError as exc:
        raise ProbeStopped(exc.status_code, exc.error_keys) from None


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


def mask_identifier(value: Any) -> str:
    text = str(value)
    return f"{text[:3]}***"


def print_accounts_summary(accounts: Any, preferences: Any) -> bool:
    if not isinstance(accounts, dict) or "accounts" not in accounts:
        print("accounts: MISSING")
        return False
    account_items = accounts["accounts"]
    if not isinstance(account_items, list):
        print("accounts: MISSING (accounts is not a list)")
        return False
    valid = True
    print(f"accounts: {len(account_items)}")
    for index, account in enumerate(account_items, start=1):
        if not isinstance(account, dict):
            print(f"account[{index}] is not an object")
            valid = False
            continue
        keys = sorted(str(key) for key in account)
        print(f"account[{index}] keys: {keys}")
        identifiers = []
        for key in ("accountId", "accountName"):
            if key in account:
                identifiers.append(f"{key}={mask_identifier(account[key])}")
        if identifiers:
            print(f"account[{index}] identifiers: {', '.join(identifiers)}")
        balance = account.get("balance")
        if isinstance(balance, dict):
            balance_parts = [
                f"{key}({type(value).__name__})={value}"
                for key, value in sorted(balance.items())
            ]
            print(f"account[{index}] balance: {', '.join(balance_parts)}")
        else:
            print(f"account[{index}] balance: MISSING")
            valid = False
        preference_parts = [
            f"{key}={account[key]}"
            for key in ("currency", "hedgingMode", "accountType")
            if key in account
        ]
        if isinstance(preferences, dict):
            preference_parts.extend(
                f"{key}={preferences[key]}"
                for key in ("currency", "hedgingMode", "accountType")
                if key in preferences and key not in account
            )
        if preference_parts:
            print(f"account[{index}] preferences: {', '.join(preference_parts)}")
    return valid


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
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, Any] = {}
    with CapitalClient.from_env() as client:
        get_calls = [
            ("/session", "session.json"),
            ("/accounts", "accounts.json"),
            ("/accounts/preferences", "preferences.json"),
            ("/markets?searchTerm=gold", "markets_search_gold.json"),
            (f"/markets/{epic}", f"market_{epic}.json"),
            (f"/prices/{epic}?resolution=DAY&max=30", f"prices_day_{epic}.json"),
            ("/positions", "positions.json"),
            ("/workingorders", "workingorders.json"),
            ("/time", "time.json"),
        ]
        for path, filename in get_calls:
            payloads[filename] = save_response(filename, get_payload(client, path))
        if not print_accounts_summary(
            payloads["accounts.json"], payloads["preferences.json"]
        ):
            account_payload = payloads["accounts.json"]
            keys = (
                sorted(str(key) for key in account_payload)
                if isinstance(account_payload, dict)
                else []
            )
            raise RuntimeError(f"Unexpected /accounts response shape; keys: {keys}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Capital.com demo REST API")
    parser.add_argument("--epic", default="GOLD")
    args = parser.parse_args()
    try:
        run(args.epic)
    except ProbeStopped as exc:
        print(f"HTTP status: {exc.status}")
        print(f"error body keys: {list(exc.error_keys)}")
        raise SystemExit(1) from None
    except (CapitalClientError, RuntimeError, ValueError) as exc:
        print(f"Probe failed: {exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
