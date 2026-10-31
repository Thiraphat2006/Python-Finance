import logging
import os
from unittest.mock import Mock, patch

from project.Storage.notify import load_env, send


def test_load_env_reads_values_and_preserves_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "\n# comment\n NEW_VALUE = 'from-file' \nEXISTING=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EXISTING", "from-environment")

    load_env(env)

    assert os.environ["NEW_VALUE"] == "from-file"
    assert os.environ["EXISTING"] == "from-environment"


def test_load_env_missing_file_does_not_error(tmp_path):
    load_env(tmp_path / "missing.env")


def test_send_without_configuration_does_not_call_urlopen(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with patch("project.Storage.notify.request.urlopen") as urlopen:
        assert send("hello") is False
    urlopen.assert_not_called()


def test_send_success_posts_chat_id_and_text(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    response = Mock(status=200)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    with patch(
        "project.Storage.notify.request.urlopen", return_value=response
    ) as urlopen:
        assert send("hello") is True
    body = urlopen.call_args.args[0].data.decode("utf-8")
    assert body == "chat_id=chat&text=hello"


def test_send_silent_adds_disable_notification(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    response = Mock(status=200)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    with patch(
        "project.Storage.notify.request.urlopen", return_value=response
    ) as urlopen:
        assert send("hello", silent=True) is True
    body = urlopen.call_args.args[0].data.decode("utf-8")
    assert body == "chat_id=chat&text=hello&disable_notification=true"


def test_send_not_silent_does_not_add_disable_notification(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    response = Mock(status=200)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    with patch(
        "project.Storage.notify.request.urlopen", return_value=response
    ) as urlopen:
        assert send("hello", silent=False) is True
    body = urlopen.call_args.args[0].data.decode("utf-8")
    assert "disable_notification" not in body


def test_send_urlopen_error_returns_false_without_raising(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    with patch(
        "project.Storage.notify.request.urlopen",
        side_effect=RuntimeError("secret-token"),
    ):
        assert send("hello") is False


def test_send_value_error_returns_false_without_raising(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    with patch(
        "project.Storage.notify.request.urlopen",
        side_effect=ValueError("invalid response"),
    ):
        assert send("hello") is False


def test_send_truncates_long_text(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    response = Mock(status=200)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    with patch(
        "project.Storage.notify.request.urlopen", return_value=response
    ) as urlopen:
        assert send("x" * 5000) is True
    body = urlopen.call_args.args[0].data.decode("utf-8")
    assert len(body.split("text=", 1)[1]) == 4000


def test_send_error_log_does_not_contain_token(monkeypatch, caplog):
    token = "secret-token"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    with (
        caplog.at_level(logging.WARNING),
        patch(
            "project.Storage.notify.request.urlopen", side_effect=RuntimeError(token)
        ),
    ):
        assert send("hello") is False
    assert token not in caplog.text
