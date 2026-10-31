import logging
import os
from pathlib import Path
from urllib import parse, request


log = logging.getLogger(__name__)


def load_env(path=None):
    env_path = (
        Path(path) if path is not None else Path(__file__).resolve().parents[3] / ".env"
    )
    if not env_path.is_file():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def send(text, *, timeout=5.0, silent=False) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.warning("Telegram configuration missing")
        return False

    try:
        body = {
            "chat_id": chat_id,
            "text": str(text)[:4000],
        }
        if silent:
            body["disable_notification"] = "true"
        payload = parse.urlencode(body).encode("utf-8")
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        req = request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with request.urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except Exception as exc:
        log.warning("Telegram send failed: %s", type(exc).__name__)
        return False
