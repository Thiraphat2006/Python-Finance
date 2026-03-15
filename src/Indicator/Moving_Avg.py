import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as pdt

def movingAvg(symbol: str, timeFrame=mt5.TIMEFRAME_H1):

    if not mt5.initialize():# type: ignore
        print("Initialize failed")
        quit()

    rates = mt5.copy_rates_from_pos(symbol, timeFrame, 0,1000) # type: ignore
    if rates is None:
        print("No data")
        return

    df = pd.DataFrame(rates)

    df["SMA_200"] = df["close"].rolling(window=200).mean()
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["VWMA_200"] = pdt.vwma(df['close'],df['tick_volume'],length=200) # type: ignore

    print(df)
    print("SMA_200  :", df["SMA_200"].iloc[-1])
    print("EMA_20:", df["EMA_20"] .iloc[-1])
    print("EMA_50 :", df["EMA_50"].iloc[-1])
    print("VWMA_200 :",  df["VWMA_200"].iloc[-1])
    return df


movingAvg("XAUUSD")

