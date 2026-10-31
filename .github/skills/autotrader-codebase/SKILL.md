---
name: autotrader-codebase
description: Layout, module APIs, DB tables and test conventions of the Auto Trader repo. Use when adding or changing code in src/project or Test.
---
# Layout (src/project)
- Untility/: data_loader, back_testing, evaluate (walk_forward), run_matrix, spread (`fix_zero_spread`)
- Indicator/moving_avg.py, strategy/moving_avg_logic.py (closed), strategy/tsmom.py (not wired into the daily loop)
- risk/: position_sizing (`SymbolSpec`, `atr_stop`, `position_size`), risk_engine (`RiskConfig`, `RiskEngine`, `save_engine(db, engine)`, `load_engine(db, cfg, equity)`)
- Execution/: paper_broker, replay (`run_paper`), mt5_broker (`MT5Broker`), signals (`Signal`, `ScheduledTestStrategy`), daily_runner (`run_once`), mt5_data, run_daily
- Storage/: trader_db (`TraderDB`), log_setup (`setup_logging`), notify (`load_env`, `send`), backup
- Folder casing matters on Windows: copy the exact casing used by existing imports (`project.Execution...`, `project.Storage...`, `project.Risk...`/`risk`); never guess.

# Key APIs
- Broker interface (paper and MT5): open / close / check_stop / equity / open_positions; `MT5Broker.sync(expected_ticket=None) -> str | None` (None = consistent).
- `run_once(broker, db, cfg, strategy, spec, bars, quote, now, mt5=None, magic=None, other_positions=()) -> RunResult(status, decision, reason, run_id, sync_note)`; status `error` rows do not count as processed.
- Strategy interface: `name`, `key(bars, now)`, `decide(bars, position, now)`.
- `mt5_data`: `get_server_now`, `get_quote`, `is_market_open(mt5, symbol, wait_s=60.0, poll_s=0.5, sleep=time.sleep)`, `add_atr` (Wilder 14, single place to change), `fetch_d1_bars` (drops the forming bar), `log_recent_deals`.
- `run_daily`: `TradingResult(code, retry_reason)` with retry_reason in {"market_closed","init_failed",None}; `_run_trading(args, *, quiet_retryable=False)`; `main(argv, *, clock=None, sleep=None)`; CLI: --symbol --db --log-dir --backup-dir --no-backup --retry-until-minutes (default 0 = single attempt) --retry-every-seconds (default 300). `_alert(symbol, level, text)` levels: ok/trade/warn/error/risk.
- `backup_database(...) -> (path, changed)`; `run_backup(...) -> BackupResult(ok, summary, changed=True)`, never raises.
- DB (SQLite, WAL): runs, orders, deals (PK=ticket), quotes, account_snapshots, state (`get_state`/`set_state`); schema version via `PRAGMA user_version`.

# Test conventions
- FakeMT5 for broker/runner tests (no terminal needed); real files via `tmp_path`; in-memory DB fixture `TraderDB(":memory:")` then close.
- Logging tests must flush/close handlers before temp dirs are removed (Windows).
- Time-dependent code: inject clock/sleep; a test using the same clock on both sides will NOT catch server-time offset bugs.
- Cover same-day/same-filename cases for file replacement logic, not only different dates.
- Commands: `uv run pytest` (all), `uv run pytest Test/Storage/test_backup.py`.
