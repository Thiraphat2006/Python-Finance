import pandas as pd
from typing import TypedDict


class MovingAvgEntity(TypedDict):
    close: pd.Series
    SMA_200: pd.Series
    EMA_21: pd.Series
    EMA_50: pd.Series
    VWMA_200: pd.Series
