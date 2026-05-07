from project.Indicator.Moving_Avg import MovingAvg as MA
from project.strategy.Moving_Avg_Logic import moving_avg_prediction as MAL
from project.Untility.data_loader import DataLoader as DL


def main():

    ROWS = 1000
    df = DL("XAUUSD").load_data(count=ROWS).get_dataframe()

    if df is None:
        raise ValueError("Data loading failed. DataFrame is None.")

    ma = MA(df).compute_indicators().get_series()

    print(MAL(ma))


if __name__ == "__main__":
    main()
