import MetaTrader5 as mt5
import pandas as pd
import time


def movingAvg(symbol: str, timeFrame=mt5.TIMEFRAME_M1) -> None:

    if not mt5.initialize():
        print("Initialize failed")
        quit()

    rates = mt5.copy_rates_from_pos(symbol, timeFrame, 0, 400)

    df = pd.DataFrame(rates)
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["EMA_100"] = df["close"].ewm(span=100, adjust=False).mean()
    df["EMA_200"] = df["close"].ewm(span=200, adjust=False).mean()
    print(df)
    print("EMA_20  :", df["EMA_20"].iloc[-1])
    print("EMA_50  :", df["EMA_50"].iloc[-1])
    print("EMA_100 :", df["EMA_100"].iloc[-1])
    print("EMA_200 :", df["EMA_200"].iloc[-1])


movingAvg("XAUUSD", timeFrame=mt5.TIMEFRAME_W1)

