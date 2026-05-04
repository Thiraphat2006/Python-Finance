import datetime
import logging
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as pdt

logger = logging.getLogger(__name__)


class MovingAvg:
    """Helper class to load price data and compute moving average indicators.

    Usage:
        ma = MovingAvg("XAUUSD")
        df = ma.load_data().compute_indicators().to_dataframe()
    """

    def __init__(self, symbol: str, time_frame=mt5.TIMEFRAME_H1):
        self.symbol = symbol
        self.time_frame = time_frame
        self.df: Optional[pd.DataFrame] = None

    def initialize(self) -> None:
        if not mt5.initialize():  # type: ignore
            raise RuntimeError("MetaTrader5 initialize failed")

    def load_with_date(
        self, utc_from: datetime.datetime, utc_to: datetime.datetime
    ) -> "MovingAvg":
        self.initialize()
        rates = mt5.copy_rates_range(self.symbol, self.time_frame, utc_from, utc_to)  # type: ignore
        if rates is None:
            raise ValueError(
                "No data returned from MetaTrader5 for the specified date range"
            )
        self.df = pd.DataFrame(rates)
        return self

    def load_data(self, count: int = 1000) -> "MovingAvg":
        self.initialize()
        rates = mt5.copy_rates_from_pos(self.symbol, self.time_frame, 0, count)  # type: ignore
        if rates is None:
            raise ValueError("No data returned from MetaTrader5")
        self.df = pd.DataFrame(rates)
        return self

    def compute_indicators(self) -> "MovingAvg":
        if self.df is None:
            raise ValueError(
                "Data frame is not loaded. Call load_data() or load_with_date() first."
            )

        df = self.df
        df["SMA_200"] = df["close"].rolling(window=200).mean()
        df["EMA_21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["VWMA_200"] = pdt.vwma(df["close"], df["tick_volume"], length=200)  # type: ignore

        self.df = df.dropna()
        return self

    def to_dataframe(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError(
                "Indicators have not been computed yet. Call compute_indicators() first."
            )
        return self.df


def movingAvg(symbol: str, timeFrame=mt5.TIMEFRAME_H1) -> pd.DataFrame:
    """Legacy function wrapper: quickly return computed DataFrame for a symbol/timeframe."""
    ma = MovingAvg(symbol, timeFrame)
    return ma.load_data().compute_indicators().to_dataframe()


# Backwards-compatible alias for code which imports the old class name
moving_avg = MovingAvg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    ma = MovingAvg("XAUUSD")
    df = ma.load_data().compute_indicators().to_dataframe()
    print(df)
