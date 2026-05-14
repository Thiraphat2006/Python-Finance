import logging

from project.strategy.Moving_Avg_Logic import MovingAvgLogic as MAL
from project.Untility.data_loader import DataLoader as DL
from project.Indicator.Moving_Avg import MovingAvg as MA
from pprint import pprint as pprint

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(module)s: %(message)s"
)


def main():
    data = DL("XAUUSD").load_data(5000)
    data = MA(data).compute_indicators()
    result = MAL(data)()

    pprint(result.head())


if __name__ == "__main__":
    main()
