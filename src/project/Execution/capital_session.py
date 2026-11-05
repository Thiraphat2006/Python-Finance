"""Session ของ Capital.com: ประกอบ CapitalClient + CapitalBroker + CapitalData (demo เท่านั้น)"""

import logging

import httpx

from project.Execution.capital_broker import CapitalBroker
from project.Execution.capital_client import CapitalApiError, CapitalClient
from project.Execution.capital_data import CapitalData
from project.Execution.capital_symbols import CAPITAL_GOLD_SPEC, EPIC
from project.Risk.position_sizing import SymbolSpec

log = logging.getLogger("autotrader")
_EPICS = {"XAUUSD": EPIC}


class CapitalInitError(RuntimeError):
    """เชื่อมต่อ/login ไม่สำเร็จแบบลองใหม่ได้ (network, 5xx)"""

    def __init__(self, last_error: object):
        self.last_error = str(last_error)
        super().__init__(f"เชื่อม Capital.com ไม่สำเร็จ: {self.last_error}")


class CapitalSymbolError(RuntimeError):
    pass


class CapitalSession:
    broker: CapitalBroker
    data: CapitalData

    def __init__(self, symbol: str, client: CapitalClient | None = None) -> None:
        self.symbol = symbol
        self.spec: SymbolSpec | None = None
        self._client = client
        self._connected = False
        self._closed = False

    def connect(self) -> None:
        epic = _EPICS.get(self.symbol)
        if epic is None:
            raise CapitalSymbolError(f"ไม่รองรับ symbol {self.symbol} บน Capital")
        if self._client is None:
            self._client = CapitalClient.from_env()  # ขาด CAP_* = CapitalClientError
        try:
            self.broker = CapitalBroker(self._client, epic, CAPITAL_GOLD_SPEC)
        except httpx.TransportError as exc:
            self.close()
            raise CapitalInitError(type(exc).__name__) from exc
        except CapitalApiError as exc:
            self.close()
            if exc.status_code >= 500:
                raise CapitalInitError(f"status {exc.status_code}") from exc
            raise
        except Exception:
            self.close()
            raise
        self.spec = CAPITAL_GOLD_SPEC
        self.data = CapitalData(self._client, epic, CAPITAL_GOLD_SPEC)
        self._connected = True

    def close(self) -> None:
        """idempotent; logout (DELETE /session) แบบ best-effort ไม่ raise (UNVERIFIED: ยังไม่เคย probe)"""
        if self._closed:
            return
        self._closed = True
        client = self._client
        if client is None:
            return
        if self._connected:
            try:
                client.delete("/session")
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "logout Capital ไม่สำเร็จ (ปล่อยหมดอายุเอง): %s", type(exc).__name__
                )
        client.close()
