import logging

from project.strategy.Moving_Avg_Logic import moving_avg_prediction as MAL
from project.Untility.data_loader import DataLoader as DL
from pprint import pprint as pprint

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(module)s: %(message)s"
)


def main():
    data = DL("XAUUSD").load_data()
    result = MAL(data)
    # result = result.to_frame()
    pprint(result.groupby("Trend_Signal").size())
    pprint(result.groupby("Trading_Signal").size())
    pprint(result.groupby("Volume_Signal").size())


if __name__ == "__main__":
    main()
