import logging

import httpx
import pandas as pd
import pytest

from project.Execution.capital_broker import (
    CapitalBroker,
    CapitalBrokerError,
    CapitalPosition,
)
from project.Execution.capital_client import CapitalClient


def make_client(
    payload: dict,
    session_account_id: str | None = "acc-1",
) -> CapitalClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/session"):
            if request.method == "POST":
                return httpx.Response(
                    200, headers={"CST": "cst", "X-SECURITY-TOKEN": "token"}
                )
            body = (
                {} if session_account_id is None else {"accountId": session_account_id}
            )
            return httpx.Response(200, json=body)
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
        "createdDateUTC": "2026-11-03T00:30:12.345",
    }
    values.update(position)
    return {"position": values, "market": {"epic": epic}}


def broker(payload: dict, session_account_id: str | None = "acc-1") -> CapitalBroker:
    return CapitalBroker(make_client(payload, session_account_id))


def test_open_positions_empty_list() -> None:
    b = broker({"positions": []})
    assert b.fetch_positions() == []
    assert b.open_positions == 0


def test_open_positions_parses_buy_without_stop_level() -> None:
    positions = broker({"positions": [raw_position()]}).fetch_positions()
    assert positions[0].sl is None
    assert positions[0].entry_time == pd.Timestamp("2026-11-03T00:30:12.345")


def test_open_positions_parses_stop_sell_and_filters_other_epic() -> None:
    positions = broker(
        {
            "positions": [
                raw_position(direction="SELL", ticket="sell-1", stopLevel=3945.0),
                raw_position(ticket="other-1", epic="SILVER"),
            ]
        }
    ).fetch_positions()
    assert positions == [
        CapitalPosition(
            "sell-1",
            "SELL",
            1.0,
            3936.94,
            3945.0,
            0.23,
            pd.Timestamp("2026-11-03T00:30:12.345"),
        )
    ]


def test_open_positions_keeps_ticket_as_string_and_missing_upl_as_none() -> None:
    positions = broker(
        {"positions": [raw_position(ticket=123, upl=None)]}
    ).fetch_positions()
    assert positions[0].ticket == "123"
    assert positions[0].pnl is None


def test_malformed_entry_is_skipped_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="autotrader"):
        positions = broker(
            {"positions": [{"position": {}, "market": {"epic": "GOLD"}}]}
        ).fetch_positions()
    assert positions == []
    assert "Skipping malformed Capital.com position" in caplog.text


def test_sync_matching_position_returns_none() -> None:
    b = broker({"positions": [raw_position()]})
    assert b.sync(expected_ticket="ticket-1") is None
    assert b._ticket == "ticket-1"
    assert b.position is not None
    assert b.position.side == "long"
    assert b.position.stop is None


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


def test_missing_created_date_raises() -> None:
    with pytest.raises(CapitalBrokerError, match="createdDateUTC"):
        broker({"positions": [raw_position(createdDateUTC=None)]})


def test_malformed_created_date_raises() -> None:
    with pytest.raises(CapitalBrokerError, match="createdDateUTC"):
        broker({"positions": [raw_position(createdDateUTC="not-a-date")]})


def test_non_string_created_date_raises() -> None:
    with pytest.raises(CapitalBrokerError, match="createdDateUTC"):
        broker({"positions": [raw_position(createdDateUTC=1762129812345)]})


def test_account_values_ignore_account_type() -> None:
    b = broker(
        {
            "positions": [],
            "accounts": [
                {
                    "accountId": "acc-1",
                    "accountType": "CFD",
                    "balance": {
                        "available": 959.32,
                        "balance": 999.74,
                        "deposit": 1000.25,
                        "profitLoss": -0.51,
                    },
                }
            ],
        }
    )
    assert b.cash == 1000.25
    assert b.equity() == 999.74


@pytest.mark.parametrize(
    "accounts",
    [
        None,
        [],
        [{"accountId": "acc-1"}],
        [{"accountId": "acc-1", "balance": {"available": 1.0, "balance": 2.0}}],
    ],
)
def test_unexpected_account_shape_raises(accounts: object) -> None:
    b = broker({"positions": [], "accounts": accounts})
    with pytest.raises(CapitalBrokerError, match="accounts|balance"):
        _ = b.cash


def _two_accounts() -> dict:
    def acc(account_id: str, value: float) -> dict:
        return {
            "accountId": account_id,
            "balance": {
                "balance": value,
                "deposit": value,
                "profitLoss": 0.0,
                "available": value,
            },
        }

    return {"positions": [], "accounts": [acc("acc-1", 999.8), acc("acc-2", 1000.06)]}


def test_account_values_use_session_account_not_first() -> None:
    second = broker(_two_accounts(), session_account_id="acc-2")
    assert second.cash == 1000.06
    assert second.equity() == 1000.06
    first = broker(_two_accounts(), session_account_id="acc-1")
    assert first.cash == 999.8


def test_session_account_missing_from_accounts_raises() -> None:
    b = broker(_two_accounts(), session_account_id="acc-9")
    with pytest.raises(CapitalBrokerError, match="not found"):
        _ = b.cash


def test_session_without_account_id_raises() -> None:
    b = broker(_two_accounts(), session_account_id=None)
    with pytest.raises(CapitalBrokerError, match="accountId"):
        _ = b.cash
