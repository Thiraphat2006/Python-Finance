import sys
import os
import MetaTrader5 as mt5

sys.path.append(os.path.abspath("."))
from src.Indicator.Moving_Avg import movingAvg

movingAvg("XAUUSD", timeFrame=mt5.TIMEFRAME_W1)

