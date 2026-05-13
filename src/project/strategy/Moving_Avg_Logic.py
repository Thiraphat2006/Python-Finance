import logging
import pandas as pd
from project.Indicator.Moving_Avg import MovingAvg

logger = logging.getLogger(__name__)


def sma_trend(df: pd.DataFrame) -> pd.DataFrame:

    close = df["close"]
    sma_200 = df["SMA_200"]

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

    df["Trend_Signal"] = 0
    df.loc[up_trend, "Trend_Signal"] = 1
    df.loc[down_trend, "Trend_Signal"] = -1

    logger.info(f"Trend signals computed based on SMA_200 for {len(df)} records")

    return df


def ema_signal(df: pd.DataFrame) -> pd.DataFrame:

    ema_fast = df["EMA_21"]
    ema_slow = df["EMA_50"]
    buy_condition = (ema_fast > ema_slow) & (df["close"].shift(1).notna())

    sell_condition = (ema_fast < ema_slow) & (df["close"].shift(1).notna())

    df["Trading_Signal"] = 0
    df.loc[buy_condition, "Trading_Signal"] = 1
    df.loc[sell_condition, "Trading_Signal"] = -1

    logger.info(f"EMA crossover signals computed for {len(df)} records")
    return df


def vwma_signal(df: pd.DataFrame) -> pd.DataFrame:

    vwma = df["VWMA_200"]
    buy_condition = (df["close"] > vwma) & (df["close"].shift(1).notna())

    sell_condition = (df["close"] < vwma) & (df["close"].shift(1).notna())

    df["Volume_Signal"] = 0
    df.loc[buy_condition, "Volume_Signal"] = 1
    df.loc[sell_condition, "Volume_Signal"] = -1

    logger.info(f"VWMA crossover signals computed for {len(df)} records")
    return df


def decision_logic(df: pd.DataFrame) -> pd.DataFrame:

    sma = df["Trend_Signal"]
    ema = df["Trading_Signal"]
    vwma = df["Volume_Signal"]

    long_entry = (sma == 1) & (vwma == 1) & (ema == 1)

    short_entry = (sma == -1) & (vwma == -1) & (ema == -1)

    exit_long = (sma == -1) & (vwma == -1) & (ema == -1)

    exit_short = (sma == 1) & (vwma == 1) & (ema == 1)

    df["long_entry"] = long_entry
    df["short_entry"] = short_entry
    df["exit_long"] = exit_long
    df["exit_short"] = exit_short

    return df


def moving_avg_prediction(df: pd.DataFrame) -> pd.DataFrame:

    ma_indicators = df

    df = sma_trend(ma_indicators)
    df = ema_signal(df)
    df = vwma_signal(df)

    df = decision_logic(df)

    return df
