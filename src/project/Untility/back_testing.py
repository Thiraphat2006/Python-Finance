import pandas as pd

from project.Indicator.Moving_Avg import MovingAvg as MA
from project.strategy.Moving_Avg_Logic import moving_avg_prediction as MAL
from project.Untility.data_loader import DataLoader as DL


def back_test(data_frame: pd.DataFrame):

    pass


if __name__ == "__main__":
    # Load your data into a DataFrame (replace with actual data loading)
    df = pd.DataFrame()  # Placeholder for your data loading logic

    # Run backtesting
    back_test(df)
