import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as pdt


def movingAvg(symbol: str, timeFrame=mt5.TIMEFRAME_H1) -> pd.DataFrame | str:

    if not mt5.initialize():  # type: ignore
        print("Initialize failed")
        quit()

    rates = mt5.copy_rates_from_pos(symbol, timeFrame, 0, 1000)  # type: ignore
    if rates is None:
        return "No data returned from MetaTrader5"

    df = pd.DataFrame(rates)

    df["SMA_200"] = df["close"].rolling(window=200).mean()
    df["EMA_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["VWMA_200"] = pdt.vwma(df["close"], df["tick_volume"], length=200)  # type: ignore

    return df.dropna()
