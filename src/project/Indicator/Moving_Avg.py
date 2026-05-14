import pandas as pd
import pandas_ta as pdt


class MovingAvg:
    """Class to compute moving averages (SMA, EMA, VWMA) for a given symbol and timeframe.

    Usage:
        ma = MovingAvg(dataFrame)
        df = ma.compute_indicators()


    :param data: pd.DataFrame - The DataFrame with price data.
    :return: pd.DataFrame - The DataFrame with computed indicators.
    """

    def __init__(self, data: pd.DataFrame):
        self.df = data

    def compute_indicators(self) -> pd.DataFrame:
        if self.df is None or len(self.df) < 1000:
            raise ValueError(
                "DataFrame is not loaded or has insufficient data. Please load data before computing indicators."
            )

        df = self.df
        df["SMA_200"] = df["close"].rolling(window=200).mean()
        df["EMA_21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["VWMA_200"] = pdt.vwma(df["close"], df["tick_volume"], length=200)  # type: ignore

        self.df = df.dropna().copy()
        return self.df
