import datetime
import MetaTrader5 as mt5
import pandas as pd
from typing import Optional


class DataLoader:
    def __init__(self, symbol: str, time_frame=mt5.TIMEFRAME_H1):
        self.symbol = symbol
        self.time_frame = time_frame
        self.df: Optional[pd.DataFrame] = None

    def _initialize(self) -> None:
        if not mt5.initialize():  # type: ignore
            raise RuntimeError("MetaTrader5 initialize failed")

    def load_with_date(
        self, utc_from: datetime.datetime, utc_to: datetime.datetime
    ) -> "DataLoader":
        """Load data for the specified date range.

        Usage:
            time_start = datetime(2023, 1, 1)
            time_end = datetime(2023, 6, 1)
            loader = DataLoader("XAUUSD").load_with_date(time_start, time_end)
        """

        self._initialize()
        rates = mt5.copy_rates_range(self.symbol, self.time_frame, utc_from, utc_to)  # type: ignore

        if rates is None:
            raise ValueError(
                "No data returned from MetaTrader5 for the specified date range"
            )
        self.df = pd.DataFrame(rates)

        return self

    def load_data(self, count: int = 1000) -> "DataLoader":
        """Load a specified number of data points.

        Usage:
            loader = DataLoader("XAUUSD").load_data(count=500)
        """

        self._initialize()

        rates = mt5.copy_rates_from_pos(self.symbol, self.time_frame, 0, count)  # type: ignore

        if rates is None:
            raise ValueError("No data returned from MetaTrader5")

        self.df = pd.DataFrame(rates)
        return self
