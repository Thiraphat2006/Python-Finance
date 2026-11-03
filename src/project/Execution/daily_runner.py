"""ลูปรายวัน: ประกอบ TraderDB + RiskEngine + broker (PaperBroker หรือ MT5Broker) + กลยุทธ์

run_once() รับข้อมูลที่ดึงมาแล้ว (bars, quote, now) จึงทดสอบได้โดยไม่ต้องมี MT5

ลำดับ: เช็คว่าแท่ง/วันนี้ประมวลผลแล้วหรือยัง -> sync broker -> โหลดสถานะ risk -> ตัดสินใจ
-> สั่งออเดอร์ -> บันทึก runs/orders/deals/snapshot -> เซฟสถานะ risk

ความปลอดภัย:
- รอบที่ status='error' ไม่นับว่าประมวลผลแล้ว (รันซ้ำได้) แต่ broker.sync() ตอนต้น + กฎ symbol_already_open
  กันการเปิดออเดอร์ซ้ำ
- daily limit / kill switch ปิดสถานะก่อนเสมอ ไม่ว่ากลยุทธ์จะสั่งอะไร
- เวลา `now` ใช้ UTC ตลอด (ฐานของ daily loss limit ต้องสม่ำเสมอระหว่างรอบ)
"""

import inspect
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from project.Execution.interfaces import MarketData
from project.Execution.signals import Signal
from project.Risk.position_sizing import SymbolSpec, atr_stop, position_size
from project.Risk.risk_engine import (
    RiskConfig,
    load_engine,
    open_risk_fraction,
    save_engine,
)


@dataclass(frozen=True)
class RunResult:
    status: str  # done | skipped | error
    decision: str  # enter | exit | hold | blocked | error | skip
    reason: str = "ok"
    run_id: int | None = None
    sync_note: str | None = None


def _open_ticket_key(broker: str, strategy: str, symbol: str) -> str:
    return f"open_ticket:{broker}:{strategy}:{symbol}"


def _persist_open_ticket_state(
    db: Any, broker: Any, state_key: str, legacy_state_key: str, broker_name: str
) -> None:
    if getattr(broker, "position", None) is None:
        db.delete_state(state_key)
        if broker_name == "mt5":
            db.delete_state(legacy_state_key)
        return
    db.set_state(
        state_key,
        "" if getattr(broker, "_ticket", None) is None else broker._ticket,
    )


def _order(symbol: str, kind: str, side: str, lots: float, ok: bool, **kw: Any) -> dict:
    return {"symbol": symbol, "kind": kind, "side": side, "lots": lots, "ok": ok, **kw}


def _act(
    *,
    broker,
    engine,
    cfg,
    strategy,
    spec,
    bars,
    bid,
    ask,
    mid,
    hs,
    spread_pts,
    equity,
    now,
    other_positions,
    orders,
    log,
):
    """คืน (decision, reason, lots_requested, signal)"""
    symbol = spec.symbol
    pos = broker.position
    sig: Signal = strategy.decide(bars, pos, now)
    flatten = engine.should_flatten(equity)

    # ---- ออก ----
    if pos is not None and (flatten or sig.action == "exit"):
        if flatten:
            reason = "kill_switch" if engine.killed else "daily_limit"
        else:
            reason = sig.reason
        ticket = getattr(broker, "_ticket", None)
        requested = bid if pos.side == "long" else ask
        try:
            trade = broker.close(mid, hs, now, reason)
        except Exception as exc:
            orders.append(
                _order(
                    symbol,
                    "close",
                    pos.side,
                    pos.lots,
                    False,
                    requested_price=requested,
                    ticket=ticket,
                    comment=str(exc)[:500],
                )
            )
            raise
        orders.append(
            _order(
                symbol,
                "close",
                pos.side,
                pos.lots,
                True,
                requested_price=requested,
                fill_price=trade.exit,
                ticket=ticket,
                comment=reason,
            )
        )
        log.info(
            "ปิดสถานะ %s %s lots=%s ที่ %.2f pnl=%.2f (%s)",
            symbol,
            pos.side,
            pos.lots,
            trade.exit,
            trade.pnl,
            reason,
        )
        return "exit", reason, pos.lots, sig

    # ---- เข้า ----
    if sig.action in ("enter_long", "enter_short"):
        side = "long" if sig.action == "enter_long" else "short"
        atr = float(bars["atr"].iloc[-1])
        if not atr > 0:  # รวม NaN
            return "blocked", "no_atr", None, sig
        est = ask if side == "long" else bid
        stop = atr_stop(est, atr, side, cfg.atr_stop_mult)
        sized = position_size(equity, cfg.risk_per_trade, est, stop, spec)
        if sig.ignore_risk_sizing:
            lots = sig.lots or spec.volume_min
        elif sig.lots is None:
            lots = sized
        else:
            lots = min(sig.lots, sized)
        if lots <= 0:
            return "blocked", "lot_below_min", None, sig

        new_risk = abs(est - stop) * lots * spec.contract_size / equity
        held = ([(pos, spec)] if pos is not None else []) + list(other_positions)
        decision = engine.check_entry(
            equity,
            len(held),
            spread_pts,
            symbol=symbol,
            open_symbols={symbol} if pos is not None else set(),
            open_risk=open_risk_fraction(held, equity),
            new_risk=new_risk,
        )
        if not decision.allowed:
            return "blocked", decision.reason, lots, sig

        try:
            broker.open(side, lots, mid, hs, stop, now)
        except Exception as exc:
            orders.append(
                _order(
                    symbol,
                    "open",
                    side,
                    lots,
                    False,
                    requested_price=est,
                    sl=stop,
                    comment=str(exc)[:500],
                )
            )
            raise
        p = broker.position
        orders.append(
            _order(
                symbol,
                "open",
                side,
                lots,
                True,
                requested_price=est,
                fill_price=p.entry if p else None,
                sl=stop,
                ticket=getattr(broker, "_ticket", None),
                comment=sig.reason,
            )
        )
        log.info(
            "เปิดสถานะ %s %s lots=%s ขอ %.2f เติม %s SL %.2f",
            symbol,
            side,
            lots,
            est,
            p.entry if p else "?",
            stop,
        )
        return "enter", "ok", lots, sig

    return "hold", "ok", None, sig


def run_once(
    *,
    broker,
    db,
    cfg: RiskConfig,
    strategy,
    spec: SymbolSpec,
    bars: pd.DataFrame,
    quote: tuple[float, float],
    now: pd.Timestamp,
    data: MarketData | None = None,
    other_positions: Sequence = (),
    logger: logging.Logger | None = None,
    broker_name: str = "mt5",
) -> RunResult:
    """bars = แท่ง D1 ที่ปิดแล้ว (คอลัมน์ open/high/low/close/atr), quote = (bid, ask), now = UTC แบบ naive
    other_positions = [(position, spec)] ของ symbol อื่น (ไว้นับ open risk รวมเมื่อขยายเป็นหลาย symbol)"""
    log = logger or logging.getLogger("autotrader")
    symbol = spec.symbol
    state_key = _open_ticket_key(broker_name, strategy.name, symbol)
    legacy_state_key = f"open_ticket:{strategy.name}:{symbol}"
    key = strategy.key(bars, now)
    if db.already_processed(strategy.name, symbol, key):
        log.info("ข้าม: %s %s ประมวลผลแล้ว (%s)", strategy.name, symbol, key)
        return RunResult("skipped", "skip", "already_processed")

    bid, ask = quote
    mid, hs = (bid + ask) / 2, (ask - bid) / 2
    spread_pts = (ask - bid) / spec.point

    raw_ticket = db.get_state(state_key)
    if raw_ticket is None and broker_name == "mt5":
        raw_ticket = db.get_state(legacy_state_key)
        if raw_ticket is not None:
            db.set_state(state_key, raw_ticket)
            db.delete_state(legacy_state_key)

    sync_note = None
    if hasattr(broker, "sync"):  # live broker: ดึงสถานะจริงก่อนตัดสินใจทุกครั้ง
        expected_ticket = raw_ticket if raw_ticket not in (None, "") else None
        sync_method = broker.sync
        if "expected_ticket" in inspect.signature(sync_method).parameters:
            sync_note = sync_method(expected_ticket=expected_ticket)
        else:
            sync_note = sync_method()
        if sync_note is not None:
            log.warning("สถานะ broker ไม่ตรงกับโปรแกรม: %s", sync_note)
    db.log_quote(symbol, bid, ask, spec.point, ts=now)
    equity = broker.equity(mid)
    engine = load_engine(db, cfg, equity)
    engine.update(now, equity)
    save_engine(db, engine)  # เซฟทันที: kill switch ต้องไม่หายถ้ารอบนี้พังกลางทาง

    orders: list[dict] = []
    sig, lots_req = Signal(), None
    try:
        decision, reason, lots_req, sig = _act(
            broker=broker,
            engine=engine,
            cfg=cfg,
            strategy=strategy,
            spec=spec,
            bars=bars,
            bid=bid,
            ask=ask,
            mid=mid,
            hs=hs,
            spread_pts=spread_pts,
            equity=equity,
            now=now,
            other_positions=other_positions,
            orders=orders,
            log=log,
        )
        status = "done"
    except Exception as exc:
        log.exception("รอบนี้ผิดพลาด")
        decision, reason, status = (
            "error",
            f"{type(exc).__name__}: {exc}"[:500],
            "error",
        )

    run_id = db.log_run(
        strategy.name,
        symbol,
        key,
        decision,
        signal=sig.action,
        target_weight=sig.target_weight,
        lots_requested=lots_req,
        reason=reason,
        status=status,
        started_at=now,
    )
    for o in orders:
        db.log_order(run_id=run_id, **o)

    for what, fn in (
        ("เซฟสถานะ risk", lambda: save_engine(db, engine)),
        (
            "บันทึก snapshot",
            lambda: db.log_snapshot(
                broker.cash, broker.equity(mid), broker.open_positions, ts=now
            ),
        ),
        ("บันทึก deals", lambda: data.log_recent_deals(db) if data is not None else 0),
        (
            "บันทึก ticket สถานะ",
            lambda: _persist_open_ticket_state(
                db, broker, state_key, legacy_state_key, broker_name
            ),
        ),
    ):
        try:
            fn()
        except Exception:
            log.exception("%s ไม่สำเร็จ (ไม่กระทบการเทรด)", what)

    log.info(
        "จบรอบ %s %s: %s (%s) status=%s",
        strategy.name,
        symbol,
        decision,
        reason,
        status,
    )
    return RunResult(status, decision, reason, run_id, sync_note)
