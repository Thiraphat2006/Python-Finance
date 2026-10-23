import numpy as np
import pandas as pd


def _rma(s: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing (RMA) ใช้กับ ATR / ADX"""
    return s.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(high, low, close, length: int = 14) -> pd.Series:
    return _rma(true_range(high, low, close), length)


def adx(high, low, close, length: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=high.index
    )
    atr_ = atr(high, low, close, length)
    plus_di = 100 * _rma(plus_dm, length) / atr_
    minus_di = 100 * _rma(minus_dm, length) / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _rma(dx.fillna(0.0), length)


def vwma(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    pv = (close * volume).rolling(length).sum()
    return pv / volume.rolling(length).sum()


class MovingAvg:
    """คำนวณ indicator ทั้งหมดที่กลยุทธ์ใช้ (ไม่แก้ DataFrame ที่ส่งเข้ามา)

    Usage:
        df = MovingAvg(data).compute_indicators()

    คอลัมน์ที่ต้องมี: open, high, low, close, tick_volume
    คอลัมน์ที่เพิ่ม: SMA_200, EMA_21, EMA_50, VWMA_200, atr, atr_ma, adx
    """

    def __init__(self, data: pd.DataFrame, min_rows: int = 1000):
        self.df = data
        self.min_rows = min_rows

    def compute_indicators(self) -> pd.DataFrame:
        if self.df is None or len(self.df) < self.min_rows:
            raise ValueError(
                f"DataFrame is not loaded or has fewer than {self.min_rows} rows."
            )

        df = self.df.copy()
        close, high, low = df["close"], df["high"], df["low"]

        df["SMA_200"] = close.rolling(200).mean()
        df["EMA_21"] = close.ewm(span=21, adjust=False).mean()
        df["EMA_50"] = close.ewm(span=50, adjust=False).mean()
        df["VWMA_200"] = vwma(close, df["tick_volume"], 200)
        df["atr"] = atr(high, low, close, 14)
        df["atr_ma"] = df["atr"].rolling(50).mean()
        df["adx"] = adx(high, low, close, 14)

        # ตัดช่วงอุ่นเครื่องของ indicator ออก (ไม่ตัดคอลัมน์ spread ฯลฯ ทิ้ง)
        return df.dropna(subset=["SMA_200", "VWMA_200", "atr_ma", "adx"]).copy()