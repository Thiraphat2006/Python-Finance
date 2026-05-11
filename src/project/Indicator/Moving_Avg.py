import pandas as pd
import pandas_ta as pdt


class MovingAvg:
    """Class to compute moving averages (SMA, EMA, VWMA) for a given symbol and timeframe.

    Usage:
        ma = MovingAvg(dataFrame)
        df = ma.compute_indicators()


    :param data: tuple[pd.DataFrame, str, int] - A tuple containing the DataFrame with price data, the symbol, and the timeframe.
    :return: list - A list containing the DataFrame with computed indicators, the symbol, and the timeframe.
    """

    def __init__(self, data: tuple[pd.DataFrame, str, int]):
        self.df = data[0]
        self.symbol = data[1]
        self.time_frame = data[2]

    def __repr__(self) -> str:
        return f"MovingAvg(symbol={self.symbol}, timeframe={self.time_frame})"

    def compute_indicators(self) -> list:
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
        return [self.df, self.symbol, self.time_frame]
