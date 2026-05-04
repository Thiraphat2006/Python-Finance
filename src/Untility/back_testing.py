import sys
import os
import pandas as pd

sys.path.append(os.path.abspath("."))
from src.strategy.Moving_Avg_Logic import moveingAvgPerdiction as mpd


def back_test(data_frame: pd.DataFrame):

    pass


if __name__ == "__main__":
    print(mpd("XAUUSD"))


# TODO ทำ backtesting
