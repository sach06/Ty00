"""
state_writer.py – Persists bot state and trade history to JSON files.

The Flask API server (server.py) reads these; the React dashboard polls the API.
"""
import json
import os
import time
from collections import deque
from datetime import datetime, timezone

STATE_FILE  = "bot_state.json"
TRADES_FILE = "trades.json"

# Keep up to 288 equity snapshots (one per minute for 4 h 48 min of trading)
_equity_history: deque = deque(maxlen=288)


def write_state(
    risk_manager,
    tracked_positions: dict,
    dry_run: bool,
    sentiments: dict | None = None,
    rl_summary: dict | None = None,
):
    """Write full bot state to bot_state.json on every tick."""
    import config as _cfg
    now_iso = datetime.now(timezone.utc).isoformat()

    # Accumulate equity history
    _equity_history.append({"t": now_iso, "v": round(risk_manager.equity, 2)})

    positions_out = []
    for epic, pos in tracked_positions.items():
        positions_out.append({
            "epic":      epic,
            "direction": pos["direction"],
            "entry":     round(pos["entry"],  5),
            "stop":      round(pos["stop"],   5),
            "tp":        round(pos["tp"],     5),
            "size":      pos["size"],
            "exposure_units": pos.get("exposure_units"),
            "opened_at": datetime.fromtimestamp(pos["opened_at"], tz=timezone.utc).isoformat(),
            "age_hours": round((time.time() - pos["opened_at"]) / 3600, 2),
            "dry_run":   pos.get("dry_run", False),
        })

    if not positions_out and os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                existing_state = json.load(f)
            existing_positions = existing_state.get("open_positions", [])
            if isinstance(existing_positions, list):
                positions_out = [
                    pos for pos in existing_positions
                    if isinstance(pos, dict) and pos.get("source") == "capital-account"
                ]
        except (json.JSONDecodeError, OSError):
            pass

    # ── Budget utilisation ────────────────────────────────────────────────────
    budget = float(getattr(risk_manager, "budget", _cfg.TRADING_BUDGET))
    # Approximate margin used from underlying exposure units; notional / 30 leverage.
    budget_used = sum(
        float(p.get("exposure_units") or p["size"] or 0.0) * p["entry"] / 30
        for p in positions_out
    )
    budget_used = round(min(budget_used, budget), 2)
    budget_available = round(max(0.0, budget - budget_used), 2)

    state = {
        "last_updated":       now_iso,
        "dry_run":            dry_run,
        "halted":             risk_manager.halted,
        "equity":             round(risk_manager.equity, 2),
        "peak_equity":        round(risk_manager.peak_equity, 2),
        "drawdown_pct":       round(risk_manager.drawdown_pct * 100, 2),
        "daily_pnl":          round(risk_manager._daily_pnl, 2),
        "win_rate":           round(risk_manager.win_rate * 100, 1),
        "total_trades":       risk_manager.total_trades,
        "winning_trades":     risk_manager.winning_trades,
        "consecutive_losses": risk_manager.consecutive_losses,
        "open_positions":     positions_out,
        "equity_history":     list(_equity_history),
        # Budget fields
        "budget":             round(budget, 2),
        "budget_used":        budget_used,
        "budget_available":   budget_available,
        # Sentiment: {epic: score}
        "sentiment":          sentiments or {},
        # RL agent performance summary
        "rl":                 rl_summary or {},
    }

    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except OSError:
        pass


def append_trade(
    epic:        str,
    direction:   str,
    entry:       float,
    exit_price:  float,
    size:        float,
    pnl_eur:     float,
    reason:      str = "",
):
    """Append a closed-trade record to trades.json."""
    trades = []
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, encoding="utf-8") as f:
                trades = json.load(f)
        except (json.JSONDecodeError, OSError):
            trades = []

    trades.append({
        "time":      datetime.now(timezone.utc).isoformat(),
        "epic":      epic,
        "direction": direction,
        "entry":     round(entry,      5),
        "exit":      round(exit_price, 5),
        "size":      size,
        "pnl_eur":   round(pnl_eur, 2),
        "reason":    reason,
    })

    # Cap at 200 records
    trades = trades[-200:]
    try:
        with open(TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump(trades, f, indent=2)
    except OSError:
        pass
