import os
import time
from collections.abc import Callable
from typing import Any, Self
from urllib.parse import urlsplit, urlunsplit

import httpx

DEMO_BASE_URL = "https://demo-api-capital.backend-capital.com/api/v1"
_EXPECTED_HOST = "demo-api-capital.backend-capital.com"
_SESSION_DELAY = 1.1
_CREDENTIAL_ENV_VARS = ("CAP_API_KEY", "CAP_IDENTIFIER", "CAP_API_PASSWORD")


class CapitalClientError(Exception):
    """Base error for the Capital.com client."""


class CapitalAuthError(CapitalClientError):
    """Authentication failed or a session could not be refreshed."""


class CapitalApiError(CapitalClientError):
    """The Capital.com API returned a non-success status."""

    def __init__(
        self,
        status_code: int,
        error_code: str | None = None,
        error_keys: tuple[str, ...] = (),
    ) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.error_keys = error_keys
        detail = f", errorCode={error_code}" if error_code is not None else ""
        super().__init__(f"Capital.com API error: status={status_code}{detail}")


class CapitalClient:
    def __init__(
        self,
        api_key: str,
        identifier: str,
        password: str,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._identifier = identifier
        self._password = password
        self._sleep = sleep
        self._clock = clock
        self._client = httpx.Client(
            base_url=DEMO_BASE_URL,
            transport=transport,
            timeout=timeout,
        )
        self._cst: str | None = None
        self._security_token: str | None = None
        self._last_session_at: float | None = None

    @classmethod
    def from_env(
        cls,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        timeout: float = 10.0,
    ) -> "CapitalClient":
        values = {name: os.environ.get(name) for name in _CREDENTIAL_ENV_VARS}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise CapitalClientError(
                f"Missing environment variables: {', '.join(missing)}"
            )
        return cls(
            values["CAP_API_KEY"] or "",
            values["CAP_IDENTIFIER"] or "",
            values["CAP_API_PASSWORD"] or "",
            transport=transport,
            sleep=sleep,
            clock=clock,
            timeout=timeout,
        )

    def __repr__(self) -> str:
        return "CapitalClient(<credentials hidden>)"

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
    ) -> dict[str, Any]:
        url = self._safe_url(path)
        if self._cst is None or self._security_token is None:
            self._login()

        response = self._send(method, url, params=params, json=json)
        if response.status_code == 401:
            self._login()
            response = self._send(method, url, params=params, json=json)
            if response.status_code == 401:
                raise CapitalAuthError("Capital.com authentication failed")
        return self._parse_response(response)

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("GET", path, params=params)

    def post(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
    ) -> dict[str, Any]:
        return self.request("POST", path, params=params, json=json)

    def delete(
        self, path: str, *, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.request("DELETE", path, params=params)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _safe_url(self, path: str) -> str:
        parsed = urlsplit(path)
        if parsed.scheme or parsed.netloc:
            if (
                parsed.scheme != "https"
                or parsed.hostname != _EXPECTED_HOST
                or parsed.port not in (None, 443)
                or parsed.username is not None
                or parsed.password is not None
            ):
                raise CapitalClientError(
                    "Request URL is outside the Capital.com demo API"
                )
            return urlunsplit(
                ("https", parsed.netloc, parsed.path, parsed.query, parsed.fragment)
            )
        elif path.startswith("//"):
            raise CapitalClientError("Request URL is outside the Capital.com demo API")

        if not path.startswith("/"):
            path = f"/{path}"
        return f"{DEMO_BASE_URL}{path}"

    def _send(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None,
        json: Any,
        authenticated: bool = True,
    ) -> httpx.Response:
        headers = {"X-CAP-API-KEY": self._api_key}
        if authenticated and self._cst is not None and self._security_token is not None:
            headers.update({"CST": self._cst, "X-SECURITY-TOKEN": self._security_token})
        return self._client.request(
            method, url, headers=headers, params=params, json=json
        )

    def _login(self) -> None:
        now = self._clock()
        if self._last_session_at is not None:
            remaining = _SESSION_DELAY - (now - self._last_session_at)
            if remaining > 0:
                self._sleep(remaining)
        self._last_session_at = self._clock()
        self._cst = None
        self._security_token = None
        response = self._send(
            "POST",
            f"{DEMO_BASE_URL}/session",
            params=None,
            json={
                "identifier": self._identifier,
                "password": self._password,
                "encryptedPassword": False,
            },
            authenticated=False,
        )
        if response.status_code in (401, 403):
            raise CapitalAuthError("Capital.com login rejected")
        if not 200 <= response.status_code < 300:
            raise self._api_error(response)
        cst = response.headers.get("CST")
        security_token = response.headers.get("X-SECURITY-TOKEN")
        if not cst or not security_token:
            raise CapitalAuthError(
                "Capital.com session response missing authentication headers"
            )
        self._cst = cst
        self._security_token = security_token

    def _parse_response(self, response: httpx.Response) -> dict[str, Any]:
        if not 200 <= response.status_code < 300:
            raise self._api_error(response)
        if not response.content:
            return {}
        try:
            payload = response.json()
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _api_error(response: httpx.Response) -> CapitalApiError:
        error_code: str | None = None
        error_keys: tuple[str, ...] = ()
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            error_keys = tuple(sorted(str(key) for key in payload))
            if payload.get("errorCode") is not None:
                error_code = str(payload["errorCode"])
        return CapitalApiError(response.status_code, error_code, error_keys)
