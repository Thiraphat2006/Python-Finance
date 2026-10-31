---
name: autotrader-guardrails
description: Hard rules and known traps for Auto Trader (time handling, Windows file handles, demo-only, secrets, scope). Read before any edit to trading, scheduling, backup or alert code.
---
# Hard rules
- Demo accounts only. `run_daily` must keep refusing non-demo accounts; never add `allow_live` to it.
- Never put API keys, tokens or secrets in code, tests or logs. They live in `.env` (git-ignored) only.
- Do not change trading logic (`run_once`, order code, risk rules, signals) unless the task explicitly says so. Stay inside the files named in the task.
- No production code that exists only to satisfy tests (e.g. isinstance shims). Fix the tests instead.
- Do not retry after errors that may have already sent an order. Only "market closed / no tick" and "initialize failed" are retryable.
- Alerts: success paths stay silent (log only); do not add new Telegram messages without being asked.

# Known traps
- MT5 `tick.time`, deal time and position time are broker SERVER time (UTC+2 now), not UTC. NEVER compare them to `time.time()`. `is_market_open` waits for a tick change (`wait_s=60`); XAUUSD demo ticks arrive every ~25-35 s.
- App day boundary is UTC midnight (07:00 Thailand). Loop logic uses UTC everywhere.
- Windows: an open sqlite3 connection blocks `os.replace`/unlink of that file. `with sqlite3.connect()` only commits, it does NOT close; use `contextlib.closing` and close before replacing.
- Retry deadlines use wall-clock UTC, not `time.monotonic` (the PC may sleep).
- `MetaTrader5` imports only on Windows; keep the import lazy.
- Prune/cleanup failures must never fail a backup.
- `orders.requested_price` on close = bid for long / ask for short (never mid).
- Do not commit or leave behind: `.env`, `*.db`, `logs/`, `diff.txt`, scratch `*.txt`/`*.xml` diffs and task exports.

# Workflow
- Show the real diff for review; small focused changes; run `uv run pytest` and report the summary line.
- Use PowerShell syntax in any shell commands you suggest.
