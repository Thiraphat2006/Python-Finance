import pandas as pd
import pytest

from project.Risk.position_sizing import SymbolSpec
from project.Risk.risk_engine import RiskConfig
from research.backtest.replay import run_paper

GOLD = SymbolSpec(
    "XAUUSD", point=0.01, contract_size=100, volume_min=0.01, volume_step=0.01
)


# ---------- replay ----------
def synthetic(n=40):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "atr": 5.0},
        index=idx,
    )
    for c in ("long_entry", "short_entry", "exit_long", "exit_short"):
        df[c] = False
    return df


def test_replay_stop_loss_equals_risk_fraction():
    df = synthetic()
    df.loc[df.index[10], "long_entry"] = True  # เข้าที่ open แท่ง 11
    df.loc[df.index[12], "low"] = 80.0  # ชน stop ที่ 90
    r = run_paper(df, GOLD, RiskConfig(risk_per_trade=0.005), 10_000)
    assert len(r.trades) == 1
    t = r.trades.iloc[0]
    assert t["reason"] == "stop"
    assert t["pnl"] == pytest.approx(-50.0)  # 0.5% ของ 10,000 พอดี (spread = 0)
    assert r.equity.iloc[-1] == pytest.approx(9_950)


def test_replay_rejects_after_daily_limit():
    df = synthetic(60)
    # วันแรก: เทรดแรกขาดทุน 0.5% x ... ใช้ risk 1.5% + daily limit 1% -> ชน limit ทันทีหลัง stop
    df.loc[df.index[2], "long_entry"] = True
    df.loc[df.index[4], "low"] = 80.0
    df.loc[df.index[8], "long_entry"] = True  # ควรถูกปฏิเสธ (วันเดียวกัน)
    r = run_paper(
        df, GOLD, RiskConfig(risk_per_trade=0.015, daily_loss_limit=0.01), 10_000
    )
    assert len(r.trades) == 1
    assert r.rejections.get("daily_loss_limit") == 1


def test_replay_requires_columns():
    with pytest.raises(ValueError):
        run_paper(synthetic().drop(columns=["atr"]), GOLD)
