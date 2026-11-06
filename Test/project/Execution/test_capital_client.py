from collections.abc import Callable

import httpx
import pytest

from project.Execution.capital_client import (
    CapitalApiError,
    CapitalAuthError,
    CapitalClient,
    CapitalClientError,
)


def make_transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_lazy_login_and_authenticated_request() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path.endswith("/session"):
            return httpx.Response(
                200, headers={"CST": "cst", "X-SECURITY-TOKEN": "security"}
            )
        return httpx.Response(200, json={"ok": True})

    client = CapitalClient(
        "key", "identifier", "password", transport=make_transport(handler)
    )
    assert calls == []
    assert client.get("/accounts") == {"ok": True}
    assert calls[0].content == (
        b'{"identifier":"identifier","password":"password","encryptedPassword":false}'
    )
    assert calls[0].headers["X-CAP-API-KEY"] == "key"
    assert calls[1].headers["CST"] == "cst"
    assert calls[1].headers["X-SECURITY-TOKEN"] == "security"
    assert calls[1].headers["X-CAP-API-KEY"] == "key"


def test_tokens_are_reused() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if request.url.path.endswith("/session"):
            return httpx.Response(
                200, headers={"CST": "cst", "X-SECURITY-TOKEN": "token"}
            )
        return httpx.Response(200, json={})

    client = CapitalClient("key", "id", "password", transport=make_transport(handler))
    client.get("/one")
    client.get("/two")
    assert calls == 3


def test_post_sends_json_and_auth_headers() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/session"):
            return httpx.Response(
                200,
                headers={"CST": "cst", "X-SECURITY-TOKEN": "security"},
            )
        return httpx.Response(200, json={"accepted": True})

    client = CapitalClient("key", "id", "password", transport=make_transport(handler))
    assert client.post("/positions", json={"epic": "GOLD"}) == {"accepted": True}
    request = requests[1]
    assert request.method == "POST"
    assert request.url.path == "/api/v1/positions"
    assert request.content == b'{"epic":"GOLD"}'
    assert request.headers["CST"] == "cst"
    assert request.headers["X-SECURITY-TOKEN"] == "security"
    assert request.headers["X-CAP-API-KEY"] == "key"


def test_delete_sends_authenticated_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/session"):
            return httpx.Response(
                200,
                headers={"CST": "cst", "X-SECURITY-TOKEN": "security"},
            )
        return httpx.Response(200, json={"deleted": True})

    client = CapitalClient("key", "id", "password", transport=make_transport(handler))
    assert client.delete("/positions/abc") == {"deleted": True}
    request = requests[1]
    assert request.method == "DELETE"
    assert request.url.path == "/api/v1/positions/abc"
    assert request.headers["CST"] == "cst"
    assert request.headers["X-SECURITY-TOKEN"] == "security"
    assert request.headers["X-CAP-API-KEY"] == "key"


def test_plain_text_api_error_has_no_error_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/session"):
            return httpx.Response(
                200,
                headers={"CST": "cst", "X-SECURITY-TOKEN": "security"},
            )
        return httpx.Response(
            502, headers={"content-type": "text/plain"}, content=b"bad gateway"
        )

    client = CapitalClient("key", "id", "password", transport=make_transport(handler))
    with pytest.raises(CapitalApiError) as exc_info:
        client.get("/accounts")
    assert exc_info.value.status_code == 502
    assert exc_info.value.error_code is None


def test_triple_slash_path_is_rejected_before_transport() -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json={})

    client = CapitalClient("key", "id", "password", transport=make_transport(handler))
    with pytest.raises(CapitalClientError):
        client.get("///x")
    assert requests == 0


def test_401_relogs_in_once() -> None:
    session_count = 0
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal session_count, request_count
        if request.url.path.endswith("/session"):
            session_count += 1
            return httpx.Response(
                200,
                headers={"CST": f"cst-{session_count}", "X-SECURITY-TOKEN": "token"},
            )
        request_count += 1
        return httpx.Response(401 if request_count == 1 else 200, json={"ok": True})

    client = CapitalClient("key", "id", "password", transport=make_transport(handler))
    assert client.get("/accounts") == {"ok": True}
    assert session_count == 2
    assert request_count == 2


def test_relogin_does_not_send_stale_tokens() -> None:
    sessions: list[httpx.Request] = []
    normal_requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal normal_requests
        if request.url.path.endswith("/session"):
            sessions.append(request)
            return httpx.Response(
                200,
                headers={"CST": f"cst-{len(sessions)}", "X-SECURITY-TOKEN": "token"},
            )
        normal_requests += 1
        return httpx.Response(401 if normal_requests == 1 else 200, json={})

    client = CapitalClient("key", "id", "password", transport=make_transport(handler))
    assert client.get("/accounts") == {}
    assert len(sessions) == 2
    for session in sessions:
        assert "CST" not in session.headers
        assert "X-SECURITY-TOKEN" not in session.headers


@pytest.mark.parametrize("status_code", [401, 403])
def test_login_rejection_raises_auth_error(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"errorCode": "PRIVATE"})

    client = CapitalClient("key", "id", "password", transport=make_transport(handler))
    with pytest.raises(CapitalAuthError, match="login rejected"):
        client.get("/accounts")


def test_login_server_error_raises_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"errorCode": "SERVER"})

    client = CapitalClient("key", "id", "password", transport=make_transport(handler))
    with pytest.raises(CapitalApiError) as exc_info:
        client.get("/accounts")
    assert exc_info.value.status_code == 500
    assert exc_info.value.error_code == "SERVER"


def test_absolute_and_relative_urls_reach_expected_transport_urls() -> None:
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        if request.url.path.endswith("/session"):
            return httpx.Response(
                200, headers={"CST": "cst", "X-SECURITY-TOKEN": "token"}
            )
        return httpx.Response(200, json={})

    client = CapitalClient("key", "id", "password", transport=make_transport(handler))
    client.get("https://demo-api-capital.backend-capital.com/api/v1/accounts?x=1")
    client.get("accounts")
    assert urls[1] == "https://demo-api-capital.backend-capital.com/api/v1/accounts?x=1"
    assert urls[2] == "https://demo-api-capital.backend-capital.com/api/v1/accounts"


def test_from_env_success_uses_environment_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAP_API_KEY", "env-key")
    monkeypatch.setenv("CAP_IDENTIFIER", "env-id")
    monkeypatch.setenv("CAP_API_PASSWORD", "env-password")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/session"):
            return httpx.Response(
                200, headers={"CST": "cst", "X-SECURITY-TOKEN": "token"}
            )
        return httpx.Response(200, json={})

    client = CapitalClient.from_env(transport=make_transport(handler))
    client.get("/accounts")
    assert requests[0].headers["X-CAP-API-KEY"] == "env-key"
    assert requests[0].content == (
        b'{"identifier":"env-id","password":"env-password","encryptedPassword":false}'
    )


@pytest.mark.parametrize(
    "content_type,content",
    [("text/plain", b"not-json"), ("application/json", b"[1, 2]")],
)
def test_non_mapping_success_body_returns_empty(
    content_type: str, content: bytes
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/session"):
            return httpx.Response(
                200, headers={"CST": "cst", "X-SECURITY-TOKEN": "token"}
            )
        return httpx.Response(
            200, headers={"content-type": content_type}, content=content
        )

    client = CapitalClient("key", "id", "password", transport=make_transport(handler))
    assert client.get("/accounts") == {}


def test_auth_error_messages_and_logs_do_not_contain_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    values = ("secret-key", "secret-password", "secret-cst", "secret-token")
    normal_requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal normal_requests
        if request.url.path.endswith("/session"):
            return httpx.Response(
                200,
                headers={"CST": values[2], "X-SECURITY-TOKEN": values[3]},
            )
        normal_requests += 1
        return httpx.Response(401)

    client = CapitalClient(
        values[0],
        "id",
        values[1],
        transport=make_transport(handler),
    )
    with caplog.at_level("DEBUG"), pytest.raises(CapitalAuthError) as exc_info:
        client.get("/accounts")
    text = f"{exc_info.value} {caplog.text} {client!r}"
    assert normal_requests == 2
    assert all(value not in text for value in values)


def test_two_401s_raise_auth_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/session"):
            return httpx.Response(
                200, headers={"CST": "cst", "X-SECURITY-TOKEN": "token"}
            )
        return httpx.Response(401)

    client = CapitalClient("key", "id", "password", transport=make_transport(handler))
    with pytest.raises(CapitalAuthError):
        client.get("/accounts")


@pytest.mark.parametrize("missing", ["CST", "X-SECURITY-TOKEN"])
def test_missing_login_header_raises(missing: str) -> None:
    headers = {"CST": "cst", "X-SECURITY-TOKEN": "token"}
    del headers[missing]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=headers)

    client = CapitalClient("key", "id", "password", transport=make_transport(handler))
    with pytest.raises(CapitalAuthError):
        client.get("/accounts")


def test_login_spacing_uses_injected_clock_and_sleep() -> None:
    now = 0.0
    sleeps: list[float] = []
    session_count = 0

    def clock() -> float:
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal session_count
        if request.url.path.endswith("/session"):
            session_count += 1
            return httpx.Response(
                200, headers={"CST": "cst", "X-SECURITY-TOKEN": "token"}
            )
        return httpx.Response(401 if session_count == 1 else 200, json={})

    client = CapitalClient(
        "key",
        "id",
        "password",
        transport=make_transport(handler),
        clock=clock,
        sleep=sleep,
    )
    client.get("/accounts")
    assert sleeps == [pytest.approx(1.1)]


@pytest.mark.parametrize(
    "path",
    [
        "https://evil.example/x",
        "http://demo-api-capital.backend-capital.com/x",
        "//evil.example/x",
    ],
)
def test_foreign_urls_are_rejected_before_transport(path: str) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json={})

    client = CapitalClient("key", "id", "password", transport=make_transport(handler))
    with pytest.raises(CapitalClientError):
        client.get(path)
    assert requests == 0


def test_from_env_reports_only_missing_names_and_repr_hides_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAP_API_KEY", raising=False)
    monkeypatch.setenv("CAP_IDENTIFIER", "identifier")
    monkeypatch.setenv("CAP_API_PASSWORD", "password")
    with pytest.raises(CapitalClientError) as exc_info:
        CapitalClient.from_env()
    assert "CAP_API_KEY" in str(exc_info.value)
    assert "password" not in str(exc_info.value)

    client = CapitalClient(
        "secret-key",
        "id",
        "secret-password",
        transport=make_transport(lambda _: httpx.Response(200)),
    )
    assert "secret" not in repr(client)


def test_api_error_and_empty_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/session"):
            return httpx.Response(
                200, headers={"CST": "cst", "X-SECURITY-TOKEN": "token"}
            )
        if request.url.path.endswith("/bad"):
            return httpx.Response(429, json={"errorCode": "RATE_LIMIT"})
        return httpx.Response(204)

    client = CapitalClient("key", "id", "password", transport=make_transport(handler))
    with pytest.raises(CapitalApiError) as exc_info:
        client.get("/bad")
    assert exc_info.value.status_code == 429
    assert exc_info.value.error_code == "RATE_LIMIT"
    assert client.get("/empty") == {}


def test_context_manager_closes_client() -> None:
    with CapitalClient(
        "key", "id", "password", transport=make_transport(lambda _: httpx.Response(200))
    ) as client:
        client.close()
