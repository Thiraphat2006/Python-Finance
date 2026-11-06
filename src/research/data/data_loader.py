import datetime
import logging
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from research.paths import DATA_CACHE_DIR

logger = logging.getLogger(__name__)


def _to_utc(dt: datetime.datetime) -> datetime.datetime:
    """MT5 ต้องการเวลา UTC แบบมี tzinfo ถ้าไม่ใส่จะเพี้ยนตาม timezone เครื่อง"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.UTC)
    return dt.astimezone(datetime.UTC)


class DataLoader:
    def __init__(
        self,
        symbol: str,
        time_frame=mt5.TIMEFRAME_H4,
        cache_dir: str | Path | None = DATA_CACHE_DIR,
        terminal_path: str | None = None,
    ):
        # terminal_path เช่น r"C:\Program Files\MetaTrader 5\terminal64.exe"
        # ใส่เมื่อเครื่องมี MT5 หลายตัว หรือ initialize หาโปรแกรมไม่เจอ
        self.terminal_path = terminal_path
        self.symbol = symbol
        self.time_frame = time_frame
        self.cache_dir = Path(cache_dir) if cache_dir else None

    # ---------- helpers ----------
    def _initialize(self) -> None:
        ok = (
            mt5.initialize(self.terminal_path)  # type: ignore
            if self.terminal_path
            else mt5.initialize()  # type: ignore
        )
        if not ok:
            raise RuntimeError(f"MetaTrader5 initialize failed: {mt5.last_error()}")  # type: ignore

    @staticmethod
    def _to_frame(rates) -> pd.DataFrame:
        df = pd.DataFrame(rates)
        # ใช้เวลาจริงเป็น index เพื่อให้ vectorbt รู้ความถี่ของข้อมูล
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time").sort_index()
        return df[~df.index.duplicated(keep="last")]

    def point(self) -> float:
        """ขนาด 1 point ของ symbol (XAUUSD ส่วนใหญ่ = 0.01) ใช้แปลงคอลัมน์ spread เป็นราคา"""
        self._initialize()
        try:
            info = mt5.symbol_info(self.symbol)  # type: ignore
            if info is None:
                raise ValueError(f"symbol not found: {self.symbol}")
            return float(info.point)
        finally:
            mt5.shutdown()  # type: ignore

    # ---------- public ----------
    def load_with_date(
        self,
        utc_from: datetime.datetime,
        utc_to: datetime.datetime,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """โหลดแท่งเทียนตามช่วงวันที่ (ครั้งแรกดึงจาก MT5 ครั้งต่อไปอ่านจากไฟล์แคช)

        Usage:
            loader = DataLoader("XAUUSD")
            df = loader.load_with_date(datetime(2020, 1, 1), datetime(2024, 1, 1))
        """
        utc_from, utc_to = _to_utc(utc_from), _to_utc(utc_to)

        cache_file = None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            name = (
                f"{self.symbol}_{self.time_frame}_"
                f"{utc_from:%Y%m%d}_{utc_to:%Y%m%d}.parquet"
            )
            cache_file = self.cache_dir / name
            if use_cache and cache_file.exists():
                logger.info(f"Loaded from cache: {cache_file}")
                return pd.read_parquet(cache_file)

        self._initialize()
        try:
            rates = mt5.copy_rates_range(  # type: ignore
                self.symbol, self.time_frame, utc_from, utc_to
            )
        finally:
            mt5.shutdown()  # type: ignore

        if rates is None or len(rates) == 0:
            raise ValueError("No data returned from MetaTrader5 for the date range")

        df = self._to_frame(rates)
        logger.info(f"Loaded {self.symbol} {utc_from} -> {utc_to}: {len(df)} rows")

        if cache_file is not None:
            df.to_parquet(cache_file)
        return df

    def load_data(self, count: int = 1000) -> pd.DataFrame:
        """โหลดแท่งล่าสุดตามจำนวน (ไม่ผ่านแคช เพราะข้อมูลเปลี่ยนตลอด)"""
        self._initialize()
        try:
            rates = mt5.copy_rates_from_pos(  # type: ignore
                self.symbol, self.time_frame, 0, count
            )
        finally:
            mt5.shutdown()  # type: ignore

        if rates is None or len(rates) == 0:
            raise ValueError("No data returned from MetaTrader5")

        df = self._to_frame(rates)
        logger.info(f"Loaded {self.symbol}: {len(df)} rows")
        return df
