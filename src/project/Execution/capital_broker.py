import logging
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from project.Execution.capital_client import CapitalClient
from project.Execution.capital_symbols import CAPITAL_GOLD_SPEC, EPIC
from project.Execution.paper_broker import Position
from project.Risk.position_sizing import SymbolSpec


class CapitalBrokerError(RuntimeError):
    """A Capital.com broker state cannot be used safely."""


@dataclass(frozen=True, slots=True)
class CapitalPosition:
    ticket: str
    direction: str
    size: float
    open_price: float
    sl: float | None
    pnl: float | None


class CapitalBroker:
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

    def open_positions(self) -> list[CapitalPosition]:
        payload = self.client.get("/positions")
        raw_positions = payload.get("positions", [])
        if not isinstance(raw_positions, list):
            self._log.warning("Capital.com positions response is not a list")
            return []

        positions: list[CapitalPosition] = []
        for raw in raw_positions:
            parsed = self._parse_position(raw)
            if parsed is not None:
                positions.append(parsed)
        return positions

    def sync(self, expected_ticket: str | None = None) -> str | None:
        positions = self.open_positions()
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
            capital_position.sl if capital_position.sl is not None else 0.0,
            _created_at(),
        )
        return note

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
        return CapitalPosition(
            str(ticket),
            direction,
            size,
            open_price,
            _number(position.get("stopLevel")),
            _number(position.get("upl")),
        )


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _created_at() -> pd.Timestamp:
    return pd.Timestamp(0, unit="s")
