import pandas as pd
import vectorbt as vbt

from project.Indicator.moving_avg import MovingAvg
from project.strategy.moving_avg_logic import MovingAvgLogic, StrategyConfig


def back_test(
    df: pd.DataFrame,
    init_cash: float = 10_000,
    point: float = 0.01,
    extra_spread_points: float = 0.0,
    fees: float = 0.0,
):
    """รัน backtest จาก DataFrame ที่มีสัญญาณแล้ว (ผลลัพธ์ของ MovingAvgLogic)

    - สัญญาณ shift(1): รู้ตอนปิดแท่ง t แล้วเข้าที่ราคา open ของแท่ง t+1
    - ต้นทุน = ครึ่ง spread ต่อการเข้า/ออกหนึ่งครั้ง (ใช้คอลัมน์ spread จาก MT5 ของแต่ละแท่ง)
      เพิ่ม slippage เองได้ด้วย extra_spread_points
    - freq อ่านจากข้อมูลจริง; ดูผลด้วย stats(pf) / sharpe(pf) เพื่อให้ annualize ถูก
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index ต้องเป็นเวลา (DatetimeIndex) ดู DataLoader")

    sig = df[["long_entry", "short_entry", "exit_long", "exit_short"]].shift(1)
    sig = sig.fillna(False).astype(bool)

    # ครึ่ง spread (หน่วยราคา) หารด้วยราคา open = สัดส่วน slippage ต่อแท่ง
    spread_points = df["spread"] if "spread" in df.columns else 0.0
    half_spread = (spread_points + extra_spread_points) * point / 2
    slippage = (half_spread / df["open"]).astype(float)

    # ความถี่ของแท่งอ่านจากข้อมูลจริง ไม่ hardcode
    freq = df.index.to_series().diff().mode().iloc[0]

    return vbt.Portfolio.from_signals(
        close=df["close"],
        open=df["open"],
        price=df["open"],
        entries=sig["long_entry"],
        exits=sig["exit_long"],
        short_entries=sig["short_entry"],
        short_exits=sig["exit_short"],
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        freq=freq,
    )


def year_freq_of(index: pd.DatetimeIndex) -> pd.Timedelta:
    """ความยาว 'หนึ่งปีเทรด' = จำนวนแท่งต่อปีจริง x ความถี่แท่ง

    vectorbt ค่าเริ่มต้นนับ 365 วัน x 24 ชม. แต่ตลาดทองปิดเสาร์-อาทิตย์
    แท่งจริงต่อปีน้อยกว่านั้น ทำให้ Sharpe เพี้ยนถ้าไม่ปรับ
    """
    freq = index.to_series().diff().mode().iloc[0]
    years = (index[-1] - index[0]) / pd.Timedelta(days=365.25)
    return freq * (len(index) / years)


def stats(pf) -> pd.Series:
    """pf.stats() ที่ annualize ถูกต้อง (ใช้แทน pf.stats() ตรง ๆ)"""
    return pf.stats(settings={"year_freq": year_freq_of(pf.wrapper.index)})


def sharpe(pf) -> float:
    return float(pf.sharpe_ratio(year_freq=year_freq_of(pf.wrapper.index)))


def run(raw: pd.DataFrame, config: StrategyConfig | None = None, **kwargs):
    """raw (จาก DataLoader) -> indicator -> สัญญาณ -> backtest"""
    ind = MovingAvg(raw).compute_indicators()
    sig = MovingAvgLogic(ind, config)()
    return back_test(sig, **kwargs)


if __name__ == "__main__":
    from datetime import datetime

    from project.Untility.data_loader import DataLoader

    loader = DataLoader("XAUUSD")
    raw = loader.load_with_date(datetime(2020, 1, 1), datetime(2024, 1, 1))
    pf = run(raw, point=loader.point())
    print(stats(pf))

    from project.Execution.replay import run_paper
    from project.Risk.position_sizing import SymbolSpec
    from project.Risk.risk_engine import RiskConfig

    ind = MovingAvg(raw).compute_indicators()
    sig = MovingAvgLogic(ind, StrategyConfig(exit_mode="ema_cross"))()
    r = run_paper(sig, SymbolSpec("XAUUSD"), RiskConfig())
    print(r.summary())

    print(r.trades["pnl"].min())  # ควรอยู่ราว -50 ถึง -60 (0.5% ของ 10,000 + spread/gap)
    print(r.trades["reason"].value_counts())  # ดูสัดส่วน stop / signal
