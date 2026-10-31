from typing import Any

import pandas as pd

from project.Execution.interfaces import MarketData
from project.Execution.mt5_broker import MT5Broker
from project.Execution.mt5_data import (
    fetch_d1_bars,
    get_quote,
    get_server_now,
    is_market_open,
    log_recent_deals,
)
from project.Risk.position_sizing import SymbolSpec


class MT5Data:
    def __init__(self, mt5: Any, symbol: str, magic: int | None):
        self.mt5 = mt5
        self.symbol = symbol
        self.magic = magic

    def get_server_now(self) -> pd.Timestamp:
        return get_server_now(self.mt5, self.symbol)

    def get_quote(self) -> tuple[float, float]:
        return get_quote(self.mt5, self.symbol)

    def is_market_open(self, wait_s: float = 60.0, poll_s: float = 0.5) -> bool:
        return is_market_open(self.mt5, self.symbol, wait_s=wait_s, poll_s=poll_s)

    def fetch_d1_bars(
        self,
        count: int = 120,
        atr_period: int = 14,
        server_now: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        return fetch_d1_bars(
            self.mt5,
            self.symbol,
            count=count,
            atr_period=atr_period,
            server_now=server_now,
        )

    def log_recent_deals(self, db: Any, days: int = 7) -> int:
        return log_recent_deals(db, self.mt5, self.symbol, self.magic, days=days)


class MT5Session:
    broker: MT5Broker
    data: MarketData

    def __init__(
        self,
        symbol: str,
        spec: SymbolSpec,
        magic: int = 26102024,
        mt5_module: Any = None,
    ):
        if mt5_module is None:
            import MetaTrader5 as imported_mt5

            mt5_module = imported_mt5
        self.mt5 = mt5_module
        self.symbol = symbol
        self.spec = spec
        self.magic = magic

    def connect(self) -> None:
        if not self.mt5.initialize():
            raise RuntimeError(f"initialize ไม่สำเร็จ: {self.mt5.last_error()}")
        info = self.mt5.symbol_info(self.symbol)
        if info is None:
            raise RuntimeError(f"ไม่พบ symbol {self.symbol}")
        if not info.visible:
            self.mt5.symbol_select(self.symbol, True)
        self.broker = MT5Broker(
            self.spec,
            self.symbol,
            mt5_module=self.mt5,
            magic=self.magic,
            initialize=False,
        )
        self.data = MT5Data(self.mt5, self.symbol, self.magic)

    def close(self) -> None:
        self.mt5.shutdown()
