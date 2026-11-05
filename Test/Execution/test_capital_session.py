import httpx
import pytest

from project.Execution.capital_client import CapitalClient
from project.Execution.capital_session import (
    CapitalInitError,
    CapitalSession,
    CapitalSymbolError,
)


def _client(handler):
    return CapitalClient(
        "key",
        "id",
        "pw",
        transport=httpx.MockTransport(handler),
        sleep=lambda s: None,
    )


def _ok(calls):
    def handler(request):
        calls.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path.endswith("/session"):
            return httpx.Response(
                200, headers={"CST": "c", "X-SECURITY-TOKEN": "t"}, json={}
            )
        if request.url.path.endswith("/positions"):
            return httpx.Response(200, json={"positions": []})
        return httpx.Response(200, json={})

    return handler


def test_connect_wires_components_and_close_logs_out_once():
    calls = []
    session = CapitalSession("XAUUSD", client=_client(_ok(calls)))
    session.connect()
    assert session.spec is not None and session.spec.point == 0.01
    assert session.broker.position is None
    session.close()
    session.close()
    assert calls.count(("DELETE", "/api/v1/session")) == 1


def test_unsupported_symbol_makes_no_request():
    def handler(request):
        raise AssertionError("no request expected")

    session = CapitalSession("EURUSD", client=_client(handler))
    with pytest.raises(CapitalSymbolError):
        session.connect()
    session.close()


def test_logout_failure_is_swallowed():
    def handler(request):
        if request.method == "DELETE":
            return httpx.Response(500, json={})
        return _ok([])(request)

    session = CapitalSession("XAUUSD", client=_client(handler))
    session.connect()
    session.close()


def test_transport_error_is_retryable_init_error_without_logout():
    calls = []

    def handler(request):
        calls.append(request.method)
        raise httpx.ConnectError("boom", request=request)

    session = CapitalSession("XAUUSD", client=_client(handler))
    with pytest.raises(CapitalInitError):
        session.connect()
    assert "DELETE" not in calls


def test_server_error_is_init_error():
    def handler(request):
        if request.method == "POST":
            return httpx.Response(
                200, headers={"CST": "c", "X-SECURITY-TOKEN": "t"}, json={}
            )
        return httpx.Response(503, json={})

    session = CapitalSession("XAUUSD", client=_client(handler))
    with pytest.raises(CapitalInitError):
        session.connect()
