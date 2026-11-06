from collections.abc import Callable
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import httpx
import pytest

from project.Execution.capital_client import CapitalClient

_PROBE_SPEC = spec_from_file_location(
    "capital_probe", Path(__file__).parents[2] / "tools" / "capital_probe.py"
)
assert _PROBE_SPEC is not None and _PROBE_SPEC.loader is not None
capital_probe = module_from_spec(_PROBE_SPEC)
_PROBE_SPEC.loader.exec_module(capital_probe)
ProbeStopped = capital_probe.ProbeStopped
get_payload = capital_probe.get_payload
print_accounts_summary = capital_probe.print_accounts_summary
print_summary = capital_probe.print_summary
save_response = capital_probe.save_response


def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> CapitalClient:
    return CapitalClient(
        "api-key",
        "identifier",
        "password",
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )


def test_accounts_probe_uses_authenticated_get_and_masks_sensitive_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/session"):
            return httpx.Response(
                200,
                headers={"CST": "cst-secret", "X-SECURITY-TOKEN": "token-secret"},
            )
        return httpx.Response(
            200,
            json={
                "accounts": [
                    {
                        "accountId": "ABC123",
                        "accountName": "DemoAccount",
                        "balance": {
                            "balance": 1000.5,
                            "deposit": 1000.0,
                            "profitLoss": 0.5,
                            "available": 999.5,
                        },
                        "currency": "USD",
                        "accountType": "DEMO",
                    }
                ]
            },
        )

    with make_client(handler) as client:
        accounts = get_payload(client, "/accounts")

    assert print_accounts_summary(accounts, {}) is True
    output = capsys.readouterr().out
    assert "accountId=ABC***" in output
    assert "accountName=Dem***" in output
    assert "balance(float)=1000.5" in output
    assert "accountType=DEMO" in output
    assert "cst-secret" not in output
    assert "token-secret" not in output
    assert requests[1].method == "GET"
    assert requests[1].headers["CST"] == "cst-secret"
    assert requests[1].headers["X-SECURITY-TOKEN"] == "token-secret"


def test_saved_accounts_response_redacts_identifiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capital_probe, "PROBE_DIR", tmp_path)

    save_response(
        "accounts.json",
        {
            "accounts": [
                {
                    "accountId": "ABC123",
                    "accountName": "DemoAccount",
                }
            ]
        },
    )

    saved = (tmp_path / "accounts.json").read_text(encoding="utf-8")
    assert '"accountId": "<redacted>"' in saved
    assert '"accountName": "<redacted>"' in saved
    assert "ABC123" not in saved
    assert "DemoAccount" not in saved


def test_print_summary_does_not_emit_identifier_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_summary(
        "GOLD",
        {"markets": []},
        {"snapshot": {}, "instrument": {}, "dealingRules": {}},
        {
            "positions": [
                {
                    "accountId": "ABC123",
                    "clientId": "CLIENT123",
                    "currentAccountId": "CURRENT123",
                }
            ]
        },
        {"prices": []},
        {},
        {},
        {},
    )

    output = capsys.readouterr().out
    assert "ABC123" not in output
    assert "CLIENT123" not in output
    assert "CURRENT123" not in output


def test_accounts_probe_reports_error_status_and_body_keys() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/session"):
            return httpx.Response(
                200,
                headers={"CST": "cst", "X-SECURITY-TOKEN": "token"},
            )
        return httpx.Response(403, json={"errorCode": "DENIED", "message": "no"})

    with make_client(handler) as client, pytest.raises(ProbeStopped) as exc_info:
        get_payload(client, "/accounts")

    assert exc_info.value.status == 403
    assert exc_info.value.error_keys == ("errorCode", "message")


def test_accounts_probe_rejects_missing_account_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert print_accounts_summary({"unexpected": []}, {}) is False
    assert "accounts: MISSING" in capsys.readouterr().out
