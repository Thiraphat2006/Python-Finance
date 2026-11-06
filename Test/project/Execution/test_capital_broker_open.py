import json

import httpx
import pandas as pd
import pytest

from project.Execution.capital_broker import CapitalBroker, CapitalBrokerError
from project.Execution.capital_client import CapitalApiError, CapitalClient

TS = pd.Timestamp("2026-11-04 00:30:00")
CREATED = "2026-11-04T00:30:00.000"


def raw_position(direction="BUY", size=0.01, level=3936.5, ticket="deal-1"):
    return {
        "position": {
            "dealId": ticket,
            "direction": direction,
            "size": size,
            "level": level,
            "createdDateUTC": CREATED,
        },
        "market": {"epic": "GOLD"},
    }


def accepted():
    return httpx.Response(
        200,
        json={
            "dealStatus": "ACCEPTED",
            "level": 3936.5,
            "affectedDeals": [{"dealId": "deal-1", "status": "OPENED"}],
        },
    )


def not_found():
    return httpx.Response(404, json={"errorCode": "error.not-found"})


class Fake:
    def __init__(self):
        self.positions = []
        self.confirms = [accepted]
        self.bodies = []
        self.calls = []
        self.on_post = lambda request: httpx.Response(
            200, json={"dealReference": "o_1"}
        )
        self.on_delete = self._default_delete

    def __call__(self, request):
        path = request.url.path.removeprefix("/api/v1")
        self.calls.append((request.method, path))
        if path == "/session":
            return httpx.Response(
                200, headers={"CST": "c", "X-SECURITY-TOKEN": "t"}, json={}
            )
        if request.method == "GET" and path == "/positions":
            return httpx.Response(200, json={"positions": self.positions})
        if request.method == "POST" and path == "/positions":
            self.bodies.append(json.loads(request.content))
            return self.on_post(request)
        if request.method == "GET" and path.startswith("/confirms/"):
            return self.confirms.pop(0)() if self.confirms else not_found()
        if request.method == "DELETE" and path.startswith("/positions/"):
            return self.on_delete(request)
        return httpx.Response(500)

    def count(self, method, path):
        return sum(1 for call in self.calls if call == (method, path))

    def post_then(self, make_response):
        def on_post(request):
            self.positions.append(raw_position())
            return make_response(request)

        self.on_post = on_post

    def _default_delete(self, request):
        self.positions.clear()
        return httpx.Response(200, json={"dealReference": "p_deal-1"})


def timeout(request):
    raise httpx.ReadTimeout("timeout", request=request)


def make(fake):
    client = CapitalClient(
        "k", "i", "p", transport=httpx.MockTransport(fake), sleep=lambda s: None
    )
    return CapitalBroker(client, sleep=lambda s: None)


def test_open_long_success_sets_position_and_sends_once():
    fake = Fake()
    b = make(fake)
    b.open("long", 0.01, 3936.0, 0.25, 3900.123, TS)
    assert fake.bodies == [
        {"epic": "GOLD", "direction": "BUY", "size": 0.01, "stopLevel": 3900.12}
    ]
    assert b.position is not None
    assert (b.position.side, b.position.lots) == ("long", 0.01)
    assert b.position.entry == 3936.5
    assert b.position.stop == 3900.12
    assert b.position.entry_time == TS
    assert b._ticket == "deal-1"
    assert fake.count("POST", "/positions") == 1


def test_open_short_sends_sell():
    fake = Fake()
    b = make(fake)
    b.open("short", 0.01, 3936.0, 0.25, 3970.0, TS)
    assert fake.bodies[0]["direction"] == "SELL"
    assert b.position is not None and b.position.side == "short"


def test_rejected_raises_and_leaves_no_position():
    fake = Fake()
    fake.confirms = [
        lambda: httpx.Response(
            200, json={"dealStatus": "REJECTED", "reason": "INSUFFICIENT_FUNDS"}
        )
    ]
    b = make(fake)
    with pytest.raises(CapitalBrokerError, match="rejected"):
        b.open("long", 0.01, 3936.0, 0.25, 3900.0, TS)
    assert b.position is None


def test_existing_position_blocks_before_any_post():
    fake = Fake()
    fake.positions = [raw_position()]
    b = make(fake)
    with pytest.raises(CapitalBrokerError, match="อยู่แล้ว"):
        b.open("long", 0.01, 3936.0, 0.25, 3900.0, TS)
    assert fake.count("POST", "/positions") == 0


@pytest.mark.parametrize("lots", [0.0, 0.005, 0.015, float("nan"), 60000.0])
def test_invalid_size_never_posts(lots):
    fake = Fake()
    b = make(fake)
    with pytest.raises(CapitalBrokerError, match="size"):
        b.open("long", lots, 3936.0, 0.25, 3900.0, TS)
    assert fake.count("POST", "/positions") == 0


@pytest.mark.parametrize("stop", [3936.5, 0.0, -1.0, float("nan")])
def test_invalid_long_stop_never_posts(stop):
    fake = Fake()
    b = make(fake)
    with pytest.raises(CapitalBrokerError, match="stop"):
        b.open("long", 0.01, 3936.0, 0.25, stop, TS)
    assert fake.count("POST", "/positions") == 0


def test_post_timeout_then_position_found_is_recovered_without_resend():
    fake = Fake()
    fake.post_then(timeout)
    b = make(fake)
    b.open("long", 0.01, 3936.0, 0.25, 3900.0, TS)
    assert b.position is not None
    assert b._ticket == "deal-1"
    assert b.position.entry == 3936.5
    assert b.position.entry_time == pd.Timestamp(CREATED)
    assert fake.count("POST", "/positions") == 1


def test_post_timeout_and_nothing_found_raises_ambiguous_without_resend():
    fake = Fake()
    fake.on_post = timeout
    b = make(fake)
    with pytest.raises(CapitalBrokerError, match="ambiguous"):
        b.open("long", 0.01, 3936.0, 0.25, 3900.0, TS)
    assert b.position is None
    assert fake.count("POST", "/positions") == 1


def test_post_500_is_treated_as_ambiguous():
    fake = Fake()
    fake.on_post = lambda request: httpx.Response(500, json={"errorCode": "boom"})
    b = make(fake)
    with pytest.raises(CapitalBrokerError, match="ambiguous"):
        b.open("long", 0.01, 3936.0, 0.25, 3900.0, TS)
    assert fake.count("POST", "/positions") == 1


def test_post_400_raises_api_error_without_recovery():
    fake = Fake()
    fake.on_post = lambda request: httpx.Response(
        400, json={"errorCode": "error.invalid.size"}
    )
    b = make(fake)
    gets_before = fake.count("GET", "/positions")
    with pytest.raises(CapitalApiError):
        b.open("long", 0.01, 3936.0, 0.25, 3900.0, TS)
    assert b.position is None
    assert fake.count("GET", "/positions") == gets_before + 1  # only the pre-check


def test_confirm_not_found_then_accepted_retries_confirm_only():
    fake = Fake()
    fake.confirms = [not_found, accepted]
    b = make(fake)
    b.open("long", 0.01, 3936.0, 0.25, 3900.0, TS)
    assert b._ticket == "deal-1"
    assert fake.count("POST", "/positions") == 1


def test_confirm_never_available_falls_back_to_positions():
    fake = Fake()
    fake.confirms = []
    fake.post_then(lambda request: httpx.Response(200, json={"dealReference": "o_1"}))
    b = make(fake)
    b.open("long", 0.01, 3936.0, 0.25, 3900.0, TS)
    assert b._ticket == "deal-1"
    assert fake.count("POST", "/positions") == 1


def test_missing_deal_reference_falls_back_to_positions():
    fake = Fake()
    fake.post_then(lambda request: httpx.Response(200, json={}))
    b = make(fake)
    b.open("long", 0.01, 3936.0, 0.25, 3900.0, TS)
    assert b._ticket == "deal-1"
    assert fake.count("POST", "/positions") == 1


def test_recovery_with_mismatched_position_raises_ambiguous():
    fake = Fake()

    def on_post(request):
        fake.positions.append(raw_position(direction="SELL"))
        raise httpx.ReadTimeout("timeout", request=request)

    fake.on_post = on_post
    b = make(fake)
    with pytest.raises(CapitalBrokerError, match="ambiguous"):
        b.open("long", 0.01, 3936.0, 0.25, 3900.0, TS)
    assert b.position is None


def close_ok(profit=0.005, level=3937.0):
    body = {
        "status": "CLOSED",
        "dealStatus": "ACCEPTED",
        "dealReference": "p_deal-1",
        "dealId": "deal-1",
        "affectedDeals": [{"dealId": "deal-1", "status": "FULLY_CLOSED"}],
        "level": level,
        "size": 0.01,
        "direction": "SELL",
        "profitCurrency": "USD",
    }
    if profit is not None:
        body["profit"] = profit
    return lambda: httpx.Response(200, json=body)


def held():
    fake = Fake()
    fake.positions = [raw_position()]
    return fake, make(fake)


def test_close_returns_trade_from_confirm():
    fake, b = held()
    fake.confirms = [close_ok()]
    trade = b.close(3936.8, 0.25, TS, "schedule")
    assert (trade.side, trade.lots, trade.entry) == ("long", 0.01, 3936.5)
    assert trade.exit == 3937.0
    assert trade.pnl == 0.005
    assert trade.exit_time == TS
    assert trade.entry_time == pd.Timestamp(CREATED)
    assert trade.reason == "schedule"
    assert b.position is None and b._ticket is None
    assert b.trades == [trade]
    assert fake.count("DELETE", "/positions/deal-1") == 1


def test_close_without_position_raises():
    b = make(Fake())
    with pytest.raises(RuntimeError, match="ไม่มีสถานะ"):
        b.close(3936.8, 0.25, TS, "schedule")


def test_close_when_position_gone_does_not_delete():
    fake, b = held()
    fake.positions = []
    with pytest.raises(CapitalBrokerError, match="ไม่อยู่ที่ broker"):
        b.close(3936.8, 0.25, TS, "schedule")
    assert fake.count("DELETE", "/positions/deal-1") == 0
    assert b.position is not None


def test_close_delete_timeout_is_ambiguous_without_resend():
    fake, b = held()
    fake.on_delete = timeout
    with pytest.raises(CapitalBrokerError, match="ambiguous"):
        b.close(3936.8, 0.25, TS, "schedule")
    assert fake.count("DELETE", "/positions/deal-1") == 1
    assert b.position is not None


def test_close_delete_500_is_ambiguous():
    fake, b = held()
    fake.on_delete = lambda request: httpx.Response(500, json={"errorCode": "boom"})
    with pytest.raises(CapitalBrokerError, match="ambiguous"):
        b.close(3936.8, 0.25, TS, "schedule")
    assert fake.count("DELETE", "/positions/deal-1") == 1
    assert b.position is not None


def test_close_delete_400_raises_api_error_and_keeps_position():
    fake, b = held()
    fake.on_delete = lambda request: httpx.Response(
        400, json={"errorCode": "error.invalid"}
    )
    with pytest.raises(CapitalApiError):
        b.close(3936.8, 0.25, TS, "schedule")
    assert b.position is not None


def test_close_rejected_keeps_position():
    fake, b = held()
    fake.confirms = [
        lambda: httpx.Response(
            200, json={"dealStatus": "REJECTED", "reason": "MARKET_CLOSED"}
        )
    ]
    with pytest.raises(CapitalBrokerError, match="rejected"):
        b.close(3936.8, 0.25, TS, "schedule")
    assert b.position is not None


def test_close_confirm_unavailable_is_ambiguous():
    fake, b = held()
    fake.confirms = []
    with pytest.raises(CapitalBrokerError, match="ambiguous"):
        b.close(3936.8, 0.25, TS, "schedule")
    assert fake.count("DELETE", "/positions/deal-1") == 1
    assert b.position is not None


def test_close_without_profit_computes_gross_pnl():
    fake, b = held()
    fake.confirms = [close_ok(profit=None)]
    trade = b.close(3936.8, 0.25, TS, "schedule")
    assert trade.pnl == pytest.approx(0.005)


def test_check_stop_none_without_position():
    assert make(Fake()).check_stop() is None


def test_check_stop_none_while_position_is_at_broker():
    _, b = held()
    assert b.check_stop() is None


def test_check_stop_raises_when_position_vanished():
    fake, b = held()
    fake.positions = []
    with pytest.raises(CapitalBrokerError, match="ไม่อยู่ที่ broker"):
        b.check_stop()
