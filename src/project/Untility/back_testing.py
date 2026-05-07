import sys
import os
import pandas as pd

sys.path.append(os.path.abspath("."))
from project.strategy.Moving_Avg_Logic import moving_avg_prediction as mpd


def back_test(data_frame: pd.DataFrame):

    pass


if __name__ == "__main__":
    # Load your data into a DataFrame (replace with actual data loading)
    df = pd.DataFrame()  # Placeholder for your data loading logic

    # Run backtesting
    back_test(df)


# TODO ทำ backtesting
