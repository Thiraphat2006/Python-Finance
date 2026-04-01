import sys
import os
import MetaTrader5 as mt5
import pandas as pd
from typing import TypedDict, Optional
from enum import Enum

sys.path.append(os.path.abspath("."))
from src.Indicator.Moving_Avg import movingAvg

movingAvg("XAUUSD", timeFrame=mt5.TIMEFRAME_W1)


def moveingAvgPerdiction(dataFrame: pd.DataFrame):

    class trendEnum(Enum):
        Up = "Up"
        Down = "Down"


    class resuitType(TypedDict, total=False):
        Trend: trendEnum
        Signal: str | None  # TODO Signal Enum
        Pullback: str | None  # TODO Pullback Enum

    EMA_21 = dataFrame["EMA_21"]
    SMA_200 = dataFrame["SMA_200"]
    EMA_50 = dataFrame["EMA_50"]
    VWMA_200 = dataFrame["VWMA_200"]

    resuit: resuitType = {}

    # Trend
    if EMA_21.iloc[-1] > EMA_21.iloc[-1] and EMA_21.iloc[-2] > EMA_21.iloc[-2]:
        resuit = {"Trend": trendEnum.Up}
    elif EMA_21.iloc[-1] < EMA_21.iloc[-1] and EMA_21.iloc[-2] < EMA_21.iloc[-2]:
        resuit = {"Trend": trendEnum.Down}

    # TODO Logical
