"""MT5 broker: ส่งออเดอร์จริงไปที่ MetaTrader 5 ด้วย interface เดียวกับ PaperBroker

กติกาความปลอดภัย:
- ปฏิเสธบัญชีที่ไม่ใช่ demo เว้นแต่ส่ง allow_live=True ชัดเจน (ค่าเริ่มต้น False)
- stop loss ถูกส่งไปพร้อมออเดอร์เสมอ (ฝั่ง server เป็นคนตัด ไม่พึ่งโปรแกรมนี้ออนไลน์)
- ตรวจ contract_size ของ spec ให้ตรงกับ broker ก่อนเริ่ม
- ถ้า order_send ผิดพลาดแบบกำกวม (timeout ฯลฯ) จะไม่ retry เอง ให้เรียก sync() เพื่อตรวจสถานะจริง

ข้อต่างจาก PaperBroker:
- ราคาเติมจริงมาจาก broker (พารามิเตอร์ price / half_spread ถูกละเว้นตอน open/close)
- pnl รวม commission + swap + fee (paper ไม่มี swap)
- check_stop ไม่ได้คำนวณจากแท่งราคา แต่ถามว่าสถานะยังอยู่ที่ broker ไหม
"""

import time
from typing import Any

import pandas as pd

from project.Execution.interfaces import BrokerError
from project.Execution.paper_broker import Position, Trade
from project.Risk.position_sizing import SymbolSpec

# retcode ที่ retry ได้อย่างปลอดภัย (ออเดอร์ยังไม่ถูกเติมแน่นอน)
_RETRYABLE = {10004, 10020, 10021}  # REQUOTE, PRICE_CHANGED, PRICE_OFF
_RETCODE_DONE = 10009
_ACCOUNT_TRADE_MODE_DEMO = 0


class MT5Broker:
    def __init__(
        self,
        spec: SymbolSpec,
        symbol: str,
        mt5_module=None,
        magic: int = 26102024,
        deviation: int = 20,
        allow_live: bool = False,
        max_retries: int = 3,
        retry_sleep: float = 0.5,
        initialize: bool = True,
    ):
        if mt5_module is None:
            import MetaTrader5 as imported_mt5  # import ตอนใช้จริงเท่านั้น (Windows)

            mt5_module = imported_mt5
        self.mt5 = mt5_module
        self.spec = spec
        self.symbol = symbol
        self.magic = magic
        self.deviation = deviation
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.position: Position | None = None
        self.trades: list[Trade] = []
        self._ticket: int | None = None

        if initialize and not self.mt5.initialize():
            raise BrokerError(f"initialize ไม่สำเร็จ: {self.mt5.last_error()}")
        self._check_account(allow_live)
        self._check_symbol()
        self.sync()

    # ---------- startup checks ----------
    def _check_account(self, allow_live: bool) -> None:
        info = self.mt5.account_info()
        if info is None:
            raise BrokerError(f"อ่าน account_info ไม่ได้: {self.mt5.last_error()}")
        if info.trade_mode != _ACCOUNT_TRADE_MODE_DEMO and not allow_live:
            raise BrokerError(
                "บัญชีนี้ไม่ใช่ demo ปฏิเสธการทำงาน (ตั้ง allow_live=True ถ้าตั้งใจจริง)"
            )

    def _check_symbol(self) -> None:
        info = self.mt5.symbol_info(self.symbol)
        if info is None:
            raise BrokerError(f"ไม่พบ symbol {self.symbol}")
        if not info.visible and not self.mt5.symbol_select(self.symbol, True):
            raise BrokerError(f"เปิด symbol {self.symbol} ใน Market Watch ไม่ได้")
        if abs(info.trade_contract_size - self.spec.contract_size) > 1e-9:
            raise BrokerError(
                f"contract_size ไม่ตรง: spec={self.spec.contract_size} "
                f"broker={info.trade_contract_size}"
            )

    # ---------- state ----------
    def equity(self, price: float = 0.0) -> float:
        info = self.mt5.account_info()
        if info is None:
            raise BrokerError(f"อ่าน account_info ไม่ได้: {self.mt5.last_error()}")
        return float(info.equity)

    @property
    def cash(self) -> float:
        info = self.mt5.account_info()
        if info is None:
            raise BrokerError(f"อ่าน account_info ไม่ได้: {self.mt5.last_error()}")
        return float(info.balance)

    @property
    def open_positions(self) -> int:
        return 0 if self.position is None else 1

    def _own_positions(self):
        res = self.mt5.positions_get(symbol=self.symbol)
        if res is None:
            raise BrokerError(f"positions_get ล้มเหลว: {self.mt5.last_error()}")
        return [p for p in res if p.magic == self.magic]

    def sync(self, expected_ticket: int | str | None = None) -> str | None:
        """ดึงสถานะจริงจาก broker (ใช้ตอนเริ่ม / หลัง reconnect / หลังผิดพลาดแบบกำกวม)"""
        if expected_ticket is not None:
            try:
                expected_ticket = int(expected_ticket)
            except ValueError as exc:
                raise BrokerError(
                    f"invalid expected ticket {expected_ticket!r}"
                ) from exc
        own = self._own_positions()
        if len(own) > 1:
            raise BrokerError(
                f"พบ {len(own)} สถานะของ magic นี้ (รองรับ 1) ตรวจใน terminal ก่อน"
            )
        reference_ticket = (
            expected_ticket if expected_ticket is not None else self._ticket
        )
        if not own:
            note = (
                f"สถานะ ticket {reference_ticket} หายจาก broker (โดน SL/ปิดนอกโปรแกรม)"
                if reference_ticket is not None
                else None
            )
            self.position, self._ticket = None, None
            return note
        p = own[0]
        note = None
        if reference_ticket is None:
            note = f"พบสถานะ ticket {p.ticket} ที่โปรแกรมไม่ได้เปิดรอบนี้"
        elif reference_ticket != p.ticket:
            note = f"สถานะ ticket เก่า {reference_ticket} เปลี่ยนเป็น ticket ใหม่ {p.ticket}"
        side = "long" if p.type == self.mt5.POSITION_TYPE_BUY else "short"
        self._ticket = p.ticket
        self.position = Position(
            side,
            float(p.volume),
            float(p.price_open),
            float(p.sl),
            pd.Timestamp(p.time, unit="s"),
        )
        return note

    # ---------- helpers ----------
    def _filling(self):
        mode = self.mt5.symbol_info(self.symbol).filling_mode
        if mode & 2:
            return self.mt5.ORDER_FILLING_IOC
        if mode & 1:
            return self.mt5.ORDER_FILLING_FOK
        return self.mt5.ORDER_FILLING_RETURN

    def _normalize_lots(self, lots: float) -> float:
        info = self.mt5.symbol_info(self.symbol)
        step = info.volume_step
        v = int(lots / step + 1e-9) * step  # ปัดลงเท่านั้น ไม่เสี่ยงเกินที่ risk engine อนุญาต
        v = round(v, 8)
        if v < info.volume_min:
            raise ValueError(f"lots {lots} ต่ำกว่าขั้นต่ำของ broker ({info.volume_min})")
        if v > info.volume_max:
            raise ValueError(f"lots {lots} เกินสูงสุดของ broker ({info.volume_max})")
        return v

    def _send(self, request: dict):
        last: Any = None
        for attempt in range(self.max_retries):
            result = self.mt5.order_send(request)
            if result is None:
                raise BrokerError(
                    f"order_send คืน None: {self.mt5.last_error()} (เรียก sync())"
                )
            if result.retcode == _RETCODE_DONE:
                return result
            if result.retcode in _RETRYABLE:
                last = result
                tick = self.mt5.symbol_info_tick(self.symbol)
                if tick is not None:  # ราคาเปลี่ยน ต้องอัปเดตก่อนลองใหม่
                    is_buy = request["type"] == self.mt5.ORDER_TYPE_BUY
                    request["price"] = tick.ask if is_buy else tick.bid
                time.sleep(self.retry_sleep)
                continue
            raise BrokerError(
                f"order ถูกปฏิเสธ retcode={result.retcode} {result.comment}"
            )
        if last is None:
            raise BrokerError(f"retry ครบ {self.max_retries} ครั้งแล้วยังไม่สำเร็จ")
        raise BrokerError(
            f"retry ครบ {self.max_retries} ครั้งแล้วยังไม่สำเร็จ retcode={last.retcode}"
        )

    def _deals_for(self, ticket: int):
        deals = self.mt5.history_deals_get(position=ticket)
        if deals is None:
            raise BrokerError(f"history_deals_get ล้มเหลว: {self.mt5.last_error()}")
        return list(deals)

    def _build_trade(self, reason_override: str | None = None) -> Trade:
        if self._ticket is None:
            raise BrokerError("ไม่พบ ticket ของสถานะที่ปิด")
        deals = self._deals_for(self._ticket)
        out = [d for d in deals if d.entry == self.mt5.DEAL_ENTRY_OUT]
        if not out:
            raise BrokerError("ไม่พบ deal ปิดสถานะในประวัติ (ลองอีกครั้งหรือตรวจใน terminal)")
        exit_deal = out[-1]
        pnl = float(sum(d.profit + d.commission + d.swap + d.fee for d in deals))
        if reason_override is not None:
            reason = reason_override
        elif exit_deal.reason == self.mt5.DEAL_REASON_SL:
            reason = "stop"
        else:
            reason = "external"
        p = self.position
        if p is None:
            raise BrokerError("ไม่พบสถานะที่ปิด")
        trade = Trade(
            p.side,
            p.lots,
            p.entry,
            float(exit_deal.price),
            p.entry_time,
            pd.Timestamp(exit_deal.time, unit="s"),
            pnl,
            reason,
        )
        self.trades.append(trade)
        self.position, self._ticket = None, None
        return trade

    # ---------- orders (interface เดียวกับ PaperBroker) ----------
    def open(
        self,
        side: str,
        lots: float,
        price: float,
        half_spread: float,
        stop: float,
        ts: pd.Timestamp,
    ) -> None:
        if self.position is not None:
            raise RuntimeError("มีสถานะเปิดอยู่แล้ว")
        if side not in ("long", "short"):
            raise ValueError("side ต้องเป็น long หรือ short")
        if lots <= 0:
            raise ValueError("lots ต้องมากกว่า 0")
        if self._own_positions():
            raise RuntimeError("พบสถานะค้างที่ broker เรียก sync() ก่อน")

        lots = self._normalize_lots(lots)
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise BrokerError(f"อ่านราคาไม่ได้: {self.mt5.last_error()}")
        is_buy = side == "long"
        mkt = tick.ask if is_buy else tick.bid
        if (is_buy and stop >= mkt) or (not is_buy and stop <= mkt):
            raise ValueError(f"stop {stop} อยู่ผิดด้านของราคาตลาด {mkt}")
        digits = self.mt5.symbol_info(self.symbol).digits

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lots,
            "type": self.mt5.ORDER_TYPE_BUY if is_buy else self.mt5.ORDER_TYPE_SELL,
            "price": mkt,
            "sl": round(stop, digits),
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": "autotrader",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(),
        }
        result = self._send(request)
        self._ticket = result.order if getattr(result, "order", 0) else None
        self.sync()  # ใช้ข้อมูลจริงที่ broker เป็นแหล่งความจริง (ราคาเติม, ticket)
        if self.position is None:
            raise BrokerError("ส่งออเดอร์สำเร็จแต่ไม่พบสถานะ เรียกตรวจใน terminal")

    def close(
        self, price: float, half_spread: float, ts: pd.Timestamp, reason: str
    ) -> Trade:
        if self.position is None:
            raise RuntimeError("ไม่มีสถานะให้ปิด")
        self.sync()
        if self.position is None:
            # ถูกปิดไปแล้วที่ broker (เช่นโดน stop ระหว่างทาง)
            raise RuntimeError("สถานะถูกปิดที่ broker ไปแล้ว ใช้ check_stop()")
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise BrokerError(f"อ่านราคาไม่ได้: {self.mt5.last_error()}")
        closing_long = self.position.side == "long"
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.position.lots,
            "type": self.mt5.ORDER_TYPE_SELL
            if closing_long
            else self.mt5.ORDER_TYPE_BUY,
            "position": self._ticket,
            "price": tick.bid if closing_long else tick.ask,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": "autotrader close",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(),
        }
        self._send(request)
        return self._build_trade(reason_override=reason)

    def check_stop(
        self,
        o: float = 0.0,
        h: float = 0.0,
        l: float = 0.0,
        half_spread: float = 0.0,
        ts: pd.Timestamp | None = None,
    ) -> Trade | None:
        """ถ้าสถานะหายจาก broker (โดน SL หรือปิดมือ) -> คืน Trade จากประวัติจริง"""
        if self.position is None:
            return None
        if any(p.ticket == self._ticket for p in self._own_positions()):
            return None
        return self._build_trade()
