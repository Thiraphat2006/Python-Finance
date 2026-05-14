import logging
import pandas as pd

logger = logging.getLogger(__name__)


class MovingAvgLogic:
    def __init__(self, data: pd.DataFrame) -> None:
        self.df = data

    def _validate_data(self) -> None:
        required_columns = ["close", "SMA_200", "EMA_21", "EMA_50", "VWMA_200"]
        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"DataFrame is missing required columns: {', '.join(missing_columns)}"
            )

    def _sma_trend(self) -> pd.DataFrame:

        close = self.df["close"]
        sma_200 = self.df["SMA_200"]

        up_trend = (
            (close > sma_200)
            & (close.shift(1) > sma_200.shift(1))
            & (close.shift(1).notna())
        )

        down_trend = (
            (close < sma_200)
            & (close.shift(1) < sma_200.shift(1))
            & (close.shift(1).notna())
        )

        self.df["Trend_Signal"] = 0
        self.df.loc[up_trend, "Trend_Signal"] = 1
        self.df.loc[down_trend, "Trend_Signal"] = -1

        return self.df

    def _ema_signal(self) -> pd.DataFrame:

        ema_fast = self.df["EMA_21"]
        ema_slow = self.df["EMA_50"]
        buy_condition = (ema_fast > ema_slow) & (self.df["close"].shift(1).notna())

        sell_condition = (ema_fast < ema_slow) & (self.df["close"].shift(1).notna())

        self.df["Trading_Signal"] = 0
        self.df.loc[buy_condition, "Trading_Signal"] = 1
        self.df.loc[sell_condition, "Trading_Signal"] = -1

        return self.df

    def _vwma_signal(self) -> pd.DataFrame:

        vwma = self.df["VWMA_200"]
        buy_condition = (self.df["close"] > vwma) & (self.df["close"].shift(1).notna())

        sell_condition = (self.df["close"] < vwma) & (self.df["close"].shift(1).notna())

        self.df["Volume_Signal"] = 0
        self.df.loc[buy_condition, "Volume_Signal"] = 1
        self.df.loc[sell_condition, "Volume_Signal"] = -1

        return self.df

    def _decision_logic(self) -> pd.DataFrame:

        sma = self.df["Trend_Signal"]
        ema = self.df["Trading_Signal"]
        vwma = self.df["Volume_Signal"]

        trend_up = (sma == 1) & (ema == 1) & (vwma == 1)
        trend_down = (sma == -1) & (ema == -1) & (vwma == -1)

        self.df["long_entry"] = trend_up & (~trend_up.shift(1).fillna(False))
        self.df["short_entry"] = trend_down & (~trend_down.shift(1).fillna(False))
        self.df["exit_long"] = trend_down
        self.df["exit_short"] = trend_up

        return self.df

    def __call__(self) -> pd.DataFrame:
        self._validate_data()

        self._sma_trend()
        self._ema_signal()
        self._vwma_signal()
        self._decision_logic()

        drop_columns = [
            "open",
            "high",
            "low",
            "tick_volume",
            "spread",
            "real_volume",
            "SMA_200",
            "EMA_21",
            "EMA_50",
            "VWMA_200",
        ]
        self.df = self.df.drop(
            drop_columns,
            axis=1,
        )

        return self.df
