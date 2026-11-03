import logging

import httpx
import pytest

from project.Execution.capital_broker import CapitalBroker, CapitalPosition
from project.Execution.capital_client import CapitalClient


def make_client(
    payload: dict,
) -> CapitalClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/session"):
            return httpx.Response(
                200, headers={"CST": "cst", "X-SECURITY-TOKEN": "token"}
            )
        return httpx.Response(200, json=payload)

    return CapitalClient(
        "key",
        "identifier",
        "password",
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
        clock=lambda: 0.0,
    )


def raw_position(
    *,
    ticket: str = "ticket-1",
    direction: str = "BUY",
    epic: str = "GOLD",
    **position: object,
) -> dict:
    values = {
        "dealId": ticket,
        "size": 1.0,
        "direction": direction,
        "level": 3936.94,
        "upl": 0.23,
        "createdDateUTC": "2026-11-03T10:00:00.000",
    }
    values.update(position)
    return {"position": values, "market": {"epic": epic}}


def broker(payload: dict) -> CapitalBroker:
    return CapitalBroker(make_client(payload))


def test_open_positions_empty_list() -> None:
    assert broker({"positions": []}).open_positions() == []


def test_open_positions_parses_buy_without_stop_level() -> None:
    positions = broker({"positions": [raw_position()]}).open_positions()
    assert positions == [CapitalPosition("ticket-1", "BUY", 1.0, 3936.94, None, 0.23)]


def test_open_positions_parses_stop_sell_and_filters_other_epic() -> None:
    positions = broker(
        {
            "positions": [
                raw_position(direction="SELL", ticket="sell-1", stopLevel=3945.0),
                raw_position(ticket="other-1", epic="SILVER"),
            ]
        }
    ).open_positions()
    assert positions == [CapitalPosition("sell-1", "SELL", 1.0, 3936.94, 3945.0, 0.23)]


def test_open_positions_keeps_ticket_as_string_and_missing_upl_as_none() -> None:
    positions = broker(
        {"positions": [raw_position(ticket=123, upl=None)]}
    ).open_positions()
    assert positions[0].ticket == "123"
    assert positions[0].pnl is None


def test_malformed_entry_is_skipped_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="autotrader"):
        positions = broker(
            {"positions": [{"position": {}, "market": {"epic": "GOLD"}}]}
        ).open_positions()
    assert positions == []
    assert "Skipping malformed Capital.com position" in caplog.text


def test_sync_matching_position_returns_none() -> None:
    b = broker({"positions": [raw_position()]})
    assert b.sync(expected_ticket="ticket-1") is None
    assert b._ticket == "ticket-1"
    assert b.position is not None
    assert b.position.side == "long"


def test_sync_reports_missing_position() -> None:
    b = broker({"positions": []})
    assert b.sync(expected_ticket="ticket-1") == (
        "สถานะ ticket ticket-1 หายจาก broker (โดน SL/ปิดนอกโปรแกรม)"
    )
    assert b.position is None


def test_sync_reports_unexpected_and_changed_ticket() -> None:
    payload = {"positions": []}
    b = broker(payload)
    payload["positions"] = [raw_position()]
    assert b.sync() == "พบสถานะ ticket ticket-1 ที่โปรแกรมไม่ได้เปิดรอบนี้"
    assert b.sync(expected_ticket="old") == (
        "สถานะ ticket เก่า old เปลี่ยนเป็น ticket ใหม่ ticket-1"
    )


def test_sync_rejects_multiple_positions() -> None:
    with pytest.raises(RuntimeError, match="รองรับ 1"):
        broker({"positions": [raw_position(), raw_position(ticket="ticket-2")]})
