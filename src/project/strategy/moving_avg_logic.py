import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyConfig:
    adx_min: float = 25.0  # trend ต้องแรงพอ
    use_atr_filter: bool = True  # ATR > ค่าเฉลี่ย ATR (ตลาดมี volatility)
    # "opposite_trend": ออกเมื่อ 3 เงื่อนไขกลับทิศครบ (ช้า)
    # "ema_cross":      ออกเมื่อ EMA21 ตัดกลับผ่าน EMA50 (เร็วกว่า)
    exit_mode: str = "opposite_trend"
    allow_long: bool = True
    allow_short: bool = True


REQUIRED_COLUMNS = ["close", "SMA_200", "EMA_21", "EMA_50", "VWMA_200", "adx"]


class MovingAvgLogic:
    """สร้างสัญญาณจาก SMA200 + EMA21/50 + VWMA200 โดยมี ADX/ATR เป็นตัวกรอง

    สัญญาณที่ได้ "รู้ ณ ปิดแท่งนั้น" ตอน backtest ต้อง shift(1) แล้วเข้าที่ open ถัดไป
    (back_testing.py ทำให้แล้ว)
    """

    def __init__(self, data: pd.DataFrame, config: StrategyConfig | None = None):
        self.df = data
        self.cfg = config or StrategyConfig()

    def _validate_data(self) -> None:
        cols = list(REQUIRED_COLUMNS)
        if self.cfg.use_atr_filter:
            cols += ["atr", "atr_ma"]
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {', '.join(missing)}"
            )
        if self.cfg.exit_mode not in ("opposite_trend", "ema_cross"):
            raise ValueError("exit_mode must be 'opposite_trend' or 'ema_cross'")

    def __call__(self) -> pd.DataFrame:
        self._validate_data()
        df = self.df.copy()  # ไม่แก้ข้อมูลต้นฉบับ
        c = df["close"]

        # --- สามเงื่อนไขทิศทาง (ยืนยัน 2 แท่งสำหรับ SMA) ---
        sma_up = (c > df["SMA_200"]) & (c.shift(1) > df["SMA_200"].shift(1))
        sma_dn = (c < df["SMA_200"]) & (c.shift(1) < df["SMA_200"].shift(1))
        ema_up = df["EMA_21"] > df["EMA_50"]
        ema_dn = df["EMA_21"] < df["EMA_50"]
        vw_up = c > df["VWMA_200"]
        vw_dn = c < df["VWMA_200"]

        trend_up = sma_up & ema_up & vw_up
        trend_dn = sma_dn & ema_dn & vw_dn

        # --- ตัวกรองความแรง "ไม่เกี่ยวกับทิศทาง" ใช้ร่วมกันได้ทั้ง long/short ---
        strong = df["adx"] > self.cfg.adx_min
        if self.cfg.use_atr_filter:
            strong &= df["atr"] > df["atr_ma"]

        new_up = trend_up & ~trend_up.shift(1, fill_value=False)
        new_dn = trend_dn & ~trend_dn.shift(1, fill_value=False)

        df["long_entry"] = new_up & strong if self.cfg.allow_long else False
        df["short_entry"] = new_dn & strong if self.cfg.allow_short else False

        if self.cfg.exit_mode == "ema_cross":
            df["exit_long"] = ema_dn
            df["exit_short"] = ema_up
        else:
            df["exit_long"] = trend_dn
            df["exit_short"] = trend_up

        for col in ("long_entry", "short_entry", "exit_long", "exit_short"):
            df[col] = df[col].astype(bool)
        return df
