"""Replay: เอา DataFrame สัญญาณ (ผลของ MovingAvgLogic หรือกลยุทธ์ไหนก็ได้ที่มีคอลัมน์เดียวกัน)
วิ่งผ่าน risk engine + paper broker ทีละแท่ง

Interface ของกลยุทธ์ที่ต้องมี: open, high, low, close, atr, long_entry, short_entry, exit_long, exit_short
(คอลัมน์ spread ไม่บังคับ) สัญญาณรู้ ณ ปิดแท่ง t -> เข้า/ออกที่ open แท่ง t+1 เหมือน back_testing.py

ลำดับในแต่ละแท่ง: อัปเดต risk -> ออกตามสัญญาณที่ open -> เข้าที่ open (ผ่าน risk) -> เช็ค stop ด้วย high/low
-> mark equity ที่ close -> ถ้าชน daily limit/kill switch ปิดสถานะที่ close
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from project.Execution.paper_broker import PaperBroker
from project.Risk.position_sizing import SymbolSpec, atr_stop, position_size
from project.Risk.risk_engine import RiskConfig, RiskEngine

SIGNAL_COLS = ["long_entry", "short_entry", "exit_long", "exit_short"]
REQUIRED = ["open", "high", "low", "close", "atr", *SIGNAL_COLS]


@dataclass
class ReplayResult:
    equity: pd.Series
    trades: pd.DataFrame
    rejections: dict = field(default_factory=dict)
    killed: bool = False
    kill_reason: Optional[str] = None

    def summary(self) -> dict:
        eq = self.equity
        peak = eq.cummax()
        pnl = self.trades["pnl"] if len(self.trades) else pd.Series(dtype=float)
        return {
            "return_%": round(float((eq.iloc[-1] / eq.iloc[0] - 1) * 100), 2),
            "max_dd_%": round(float(((peak - eq) / peak).max() * 100), 2),
            "trades": int(len(self.trades)),
            "win_rate_%": round(float((pnl > 0).mean() * 100), 1) if len(pnl) else 0.0,
            "rejections": dict(self.rejections),
            "killed": self.killed,
        }


def _f(sig: pd.DataFrame, col: str) -> np.ndarray:
    """คอลัมน์ -> numpy float array (ให้ type เป็น float จริง ไม่ใช่ pandas Scalar)"""
    return sig[col].to_numpy(dtype=float)


def _b(sig: pd.DataFrame, col: str) -> np.ndarray:
    """คอลัมน์สัญญาณ -> bool array เลื่อน 1 แท่ง (รู้ตอนปิดแท่ง t เข้าที่ open t+1)"""
    return sig[col].shift(1).fillna(False).to_numpy(dtype=bool)


def run_paper(
    sig: pd.DataFrame,
    spec: SymbolSpec,
    risk_cfg: Optional[RiskConfig] = None,
    init_cash: float = 10_000.0,
    extra_spread_points: float = 0.0,
) -> ReplayResult:
    missing = [c for c in REQUIRED if c not in sig.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    if not isinstance(sig.index, pd.DatetimeIndex):
        raise ValueError("index ต้องเป็น DatetimeIndex")

    cfg = risk_cfg or RiskConfig()
    broker = PaperBroker(spec, init_cash)
    engine = RiskEngine(cfg, init_cash)

    opens, highs = _f(sig, "open"), _f(sig, "high")
    lows, closes = _f(sig, "low"), _f(sig, "close")
    prev_atr = sig["atr"].shift(1).to_numpy(dtype=float)
    spreads = (
        _f(sig, "spread")
        if "spread" in sig.columns
        else np.zeros(len(sig), dtype=float)
    )
    p_long, p_short = _b(sig, "long_entry"), _b(sig, "short_entry")
    p_exit_long, p_exit_short = _b(sig, "exit_long"), _b(sig, "exit_short")
    times = list(sig.index)

    rejections: dict[str, int] = {}
    equity_curve: list[float] = []

    for i, ts in enumerate(times):
        o, h, l, c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
        a = float(prev_atr[i])
        spread_pts = float(spreads[i]) + extra_spread_points
        hs = spread_pts * spec.point / 2

        engine.update(ts, broker.equity(o))

        # 1) ออกตามสัญญาณที่ open
        pos = broker.position
        if pos is not None and (
            (pos.side == "long" and p_exit_long[i])
            or (pos.side == "short" and p_exit_short[i])
        ):
            broker.close(o, hs, ts, "signal")

        # 2) เข้าที่ open ถ้าสัญญาณมี และ risk engine อนุญาต
        side = "long" if p_long[i] else "short" if p_short[i] else None
        if side and broker.position is None:
            decision = engine.check_entry(
                broker.equity(o), broker.open_positions, spread_pts
            )
            if not decision.allowed:
                key = decision.reason.split(":")[0]
                rejections[key] = rejections.get(key, 0) + 1
            elif np.isnan(a) or a <= 0:
                rejections["no_atr"] = rejections.get("no_atr", 0) + 1
            else:
                est_entry = o + hs if side == "long" else o - hs
                stop = atr_stop(est_entry, a, side, cfg.atr_stop_mult)
                lots = position_size(
                    broker.equity(o), cfg.risk_per_trade, est_entry, stop, spec
                )
                if lots <= 0:
                    rejections["lot_below_min"] = rejections.get("lot_below_min", 0) + 1
                else:
                    broker.open(side, lots, o, hs, stop, ts)

        # 3) เช็ค stop ภายในแท่ง
        broker.check_stop(o, h, l, hs, ts)

        # 4) mark equity ที่ close + บังคับปิดถ้าชนกฎ
        eq = broker.equity(c)
        engine.update(ts, eq)
        if broker.position is not None and engine.should_flatten(eq):
            reason = "kill_switch" if engine.killed else "daily_limit"
            broker.close(c, hs, ts, reason)
            eq = broker.equity(c)
        equity_curve.append(eq)

    trades = pd.DataFrame([t.__dict__ for t in broker.trades])
    return ReplayResult(
        equity=pd.Series(equity_curve, index=sig.index, name="equity"),
        trades=trades,
        rejections=rejections,
        killed=engine.killed,
        kill_reason=engine.kill_reason,
    )
