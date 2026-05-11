import logging
import pandas as pd
from project.Indicator.Moving_Avg import MovingAvg

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame, columns_to_drop: list[str]) -> pd.DataFrame:
    df = df.dropna()
    df = df.drop(
        columns=[
            "open",
            "high",
            "low",
            "tick_volume",
            "spread",
            "real_volume",
            *columns_to_drop,
        ],
        errors="ignore",
    )
    return df


def sma_trend(df: pd.DataFrame) -> pd.DataFrame:

    columns_to_drop = ["EMA_21", "EMA_50", "VWMA_200"]
    df = clean_data(df, columns_to_drop)

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

    df = df.drop(columns=["SMA_200", "close"], errors="ignore")
    logger.info(f"Trend signals computed based on SMA_200 for {len(df)} records")

    return df


def ema_signal(df: pd.DataFrame) -> pd.DataFrame:

    columns_to_drop = ["SMA_200", "VWMA_200"]
    df = clean_data(df, columns_to_drop)

    ema_fast = df["EMA_21"]
    ema_slow = df["EMA_50"]
    buy_condition = (
        (ema_fast > ema_slow)
        & (ema_fast.shift(1) <= ema_slow.shift(1))
        & (df["close"].shift(1).notna())
    )

    sell_condition = (
        (ema_fast < ema_slow)
        & (ema_fast.shift(1) >= ema_slow.shift(1))
        & (df["close"].shift(1).notna())
    )
    df["Trading_Signal"] = 0
    df.loc[buy_condition, "Trading_Signal"] = 1
    df.loc[sell_condition, "Trading_Signal"] = -1

    df = df.drop(columns=["EMA_21", "EMA_50", "close"], errors="ignore")

    logger.info(f"EMA crossover signals computed for {len(df)} records")
    return df


def vwma_signal(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = ["SMA_200", "EMA_21", "EMA_50"]
    df = clean_data(df, columns_to_drop)

    vwma = df["VWMA_200"]
    buy_condition = (
        (df["close"] > vwma)
        & (df["close"].shift(1) <= vwma.shift(1))
        & (df["close"].shift(1).notna())
    )

    sell_condition = (
        (df["close"] < vwma)
        & (df["close"].shift(1) >= vwma.shift(1))
        & (df["close"].shift(1).notna())
    )
    df["Volume_Signal"] = 0
    df.loc[buy_condition, "Volume_Signal"] = 1
    df.loc[sell_condition, "Volume_Signal"] = -1

    df = df.drop(columns=["VWMA_200", "close"], errors="ignore")
    logger.info(f"VWMA crossover signals computed for {len(df)} records")
    return df


def moving_avg_prediction(list: list) -> pd.DataFrame:
    df, symbol, time_frame = list
    ma_indicators = MovingAvg((df, symbol, time_frame)).compute_indicators()[0]

    sma_score = sma_trend(ma_indicators)
    ema_score = ema_signal(ma_indicators)
    vwma_score = vwma_signal(ma_indicators)

    return pd.concat(
        [sma_score, ema_score["Trading_Signal"], vwma_score["Volume_Signal"]], axis=1
    )
