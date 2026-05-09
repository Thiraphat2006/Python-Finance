import pandas as pd
import pandas_ta as pdt

from project.Entity.moving_avg_entity import MovingAvgEntity


class MovingAvg:
    """Class to compute moving averages (SMA, EMA, VWMA) for a given symbol and timeframe.

    Usage:
        ma = MovingAvg(dataFrame)
        df = ma.compute_indicators().to_dataframe()


    :param data: tuple[pd.DataFrame, str, int] - A tuple containing the DataFrame with price data, the symbol, and the timeframe.
    :return: MovingAvg - An instance of the MovingAvg class with computed indicators.
    """

    def __init__(self, data: tuple[pd.DataFrame, str, int]):
        self.df = data[0]
        self.symbol = data[1]
        self.time_frame = data[2]

    def __repr__(self) -> str:
        return f"MovingAvg(symbol={self.symbol}, timeframe={self.time_frame})"

    def compute_indicators(self) -> "MovingAvg":
        if self.df is None or len(self.df) < 1000:
            raise ValueError(
                "DataFrame is not loaded or has insufficient data. Please load data before computing indicators."
            )

        df = self.df
        df["SMA_200"] = df["close"].rolling(window=200).mean()
        df["EMA_21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["VWMA_200"] = pdt.vwma(df["close"], df["tick_volume"], length=200)  # type: ignore

        self.df = df.dropna()
        return self

    def get_series(self) -> MovingAvgEntity:
        if self.df is None:
            raise ValueError(
                "Indicators have not been computed yet. Call compute_indicators() first."
            )
        return {
            "close": self.df["close"],
            "SMA_200": self.df["SMA_200"],
            "EMA_21": self.df["EMA_21"],
            "EMA_50": self.df["EMA_50"],
            "VWMA_200": self.df["VWMA_200"],
        }
