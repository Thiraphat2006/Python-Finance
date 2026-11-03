from typing import Any, Protocol

import pandas as pd

from project.Execution.paper_broker import Trade


class Broker(Protocol):
    @property
    def cash(self) -> float: ...

    @property
    def open_positions(self) -> int: ...

    def equity(self, price: float = 0.0) -> float: ...

    def sync(self, expected_ticket: str | None = None) -> str | None: ...

    def open(
        self,
        side: str,
        lots: float,
        price: float,
        half_spread: float,
        stop: float,
        ts: pd.Timestamp,
    ) -> None: ...

    def close(
        self, price: float, half_spread: float, ts: pd.Timestamp, reason: str
    ) -> Trade: ...

    def check_stop(
        self,
        o: float = 0.0,
        h: float = 0.0,
        l: float = 0.0,
        half_spread: float = 0.0,
        ts: pd.Timestamp | None = None,
    ) -> Trade | None: ...


class MarketData(Protocol):
    def get_server_now(self) -> pd.Timestamp: ...

    def get_quote(self) -> tuple[float, float]: ...

    def is_market_open(self, wait_s: float = 60.0, poll_s: float = 0.5) -> bool: ...

    def fetch_d1_bars(
        self,
        count: int = 120,
        atr_period: int = 14,
        server_now: pd.Timestamp | None = None,
    ) -> pd.DataFrame: ...

    def log_recent_deals(self, db: Any, days: int = 7) -> int: ...


class Session(Protocol):
    broker: Broker
    data: MarketData

    def connect(self) -> None: ...

    def close(self) -> None: ...
