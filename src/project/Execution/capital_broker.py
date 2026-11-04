import logging
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from project.Execution.capital_client import CapitalClient
from project.Execution.capital_symbols import CAPITAL_GOLD_SPEC, EPIC
from project.Execution.interfaces import BrokerError
from project.Execution.paper_broker import Position
from project.Risk.position_sizing import SymbolSpec


class CapitalBrokerError(BrokerError):
    """A Capital.com broker state cannot be used safely."""


@dataclass(frozen=True, slots=True)
class CapitalPosition:
    ticket: str
    direction: str
    size: float
    open_price: float
    sl: float | None
    pnl: float | None
    entry_time: pd.Timestamp


class CapitalBroker:
    """Read-only Capital.com broker adapter; not wired into the broker factory.

    Cash/equity semantics are UNVERIFIED assumptions and must be measured with
    an open position in step (3c).
    """

    def __init__(
        self,
        client: CapitalClient,
        epic: str = EPIC,
        spec: SymbolSpec = CAPITAL_GOLD_SPEC,
    ) -> None:
        self.client = client
        self.epic = epic
        self.spec = spec
        self.position: Position | None = None
        self._ticket: str | None = None
        self._log = logging.getLogger("autotrader")
        self.sync()

    def fetch_positions(self) -> list[CapitalPosition]:
        """Fetch open positions; the exact position timestamp format is UNVERIFIED."""
        payload = self.client.get("/positions")
        raw_positions = payload.get("positions", [])
        if not isinstance(raw_positions, list):
            raise CapitalBrokerError(
                "Capital.com positions response has an unexpected shape"
            )

        positions: list[CapitalPosition] = []
        for raw in raw_positions:
            parsed = self._parse_position(raw)
            if parsed is not None:
                positions.append(parsed)
        return positions

    @property
    def open_positions(self) -> int:
        return len(self.fetch_positions())

    @property
    def cash(self) -> float:
        """Return account balance; whether it includes unrealized P/L is UNVERIFIED."""
        balance, _ = self._account_values()
        return balance

    def equity(self, price: float = 0.0) -> float:
        """Return balance plus profit/loss; whether balance includes unrealized P/L is UNVERIFIED."""
        balance, profit_loss = self._account_values()
        return balance + profit_loss

    def sync(self, expected_ticket: str | None = None) -> str | None:
        positions = self.fetch_positions()
        if len(positions) > 1:
            raise CapitalBrokerError(
                f"พบ {len(positions)} สถานะของ {self.epic} (รองรับ 1)"
            )

        reference_ticket = (
            expected_ticket if expected_ticket is not None else self._ticket
        )
        if not positions:
            note = (
                f"สถานะ ticket {reference_ticket} หายจาก broker (โดน SL/ปิดนอกโปรแกรม)"
                if reference_ticket is not None
                else None
            )
            self.position, self._ticket = None, None
            return note

        capital_position = positions[0]
        note = None
        if reference_ticket is None:
            note = f"พบสถานะ ticket {capital_position.ticket} ที่โปรแกรมไม่ได้เปิดรอบนี้"
        elif reference_ticket != capital_position.ticket:
            note = (
                f"สถานะ ticket เก่า {reference_ticket} "
                f"เปลี่ยนเป็น ticket ใหม่ {capital_position.ticket}"
            )
        self._ticket = capital_position.ticket
        self.position = Position(
            "long" if capital_position.direction == "BUY" else "short",
            capital_position.size,
            capital_position.open_price,
            capital_position.sl,
            capital_position.entry_time,
        )
        return note

    def _account_values(self) -> tuple[float, float]:
        payload = self.client.get("/accounts")
        accounts = payload.get("accounts")
        if not isinstance(accounts, list) or not accounts:
            raise CapitalBrokerError(
                "Capital.com accounts response has an unexpected shape"
            )
        account = accounts[0]
        if not isinstance(account, dict):
            raise CapitalBrokerError(
                "Capital.com accounts response has an unexpected shape"
            )
        balance = account.get("balance")
        if not isinstance(balance, dict):
            raise CapitalBrokerError(
                "Capital.com account balance has an unexpected shape"
            )
        total = _number(balance.get("balance"))
        profit_loss = _number(balance.get("profitLoss"))
        if total is None or profit_loss is None:
            raise CapitalBrokerError(
                "Capital.com account balance has an unexpected shape"
            )
        return total, profit_loss

    def _parse_position(self, raw: Any) -> CapitalPosition | None:
        if not isinstance(raw, dict):
            self._log.warning("Skipping malformed Capital.com position: %r", raw)
            return None
        position = raw.get("position")
        market = raw.get("market")
        if not isinstance(position, dict) or not isinstance(market, dict):
            self._log.warning("Skipping malformed Capital.com position: %r", raw)
            return None
        if market.get("epic") != self.epic:
            return None

        ticket = position.get("dealId")
        direction = position.get("direction")
        size = _number(position.get("size"))
        open_price = _number(position.get("level"))
        if (
            ticket is None
            or str(ticket) == ""
            or direction not in ("BUY", "SELL")
            or size is None
            or open_price is None
        ):
            self._log.warning("Skipping malformed Capital.com position: %r", raw)
            return None
        entry_time = _parse_entry_time(position.get("createdDateUTC"))
        return CapitalPosition(
            str(ticket),
            direction,
            size,
            open_price,
            _number(position.get("stopLevel")),
            _number(position.get("upl")),
            entry_time,
        )


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _parse_entry_time(value: Any) -> pd.Timestamp:
    if not isinstance(value, str):
        raise CapitalBrokerError(
            "Capital.com position has a non-string createdDateUTC (UNVERIFIED format)"
        )
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CapitalBrokerError(
            f"Capital.com position has an invalid createdDateUTC (UNVERIFIED format): {value!r}"
        ) from exc
    if pd.isna(timestamp):
        raise CapitalBrokerError(
            "Capital.com position has an invalid createdDateUTC (UNVERIFIED format)"
        )
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp
