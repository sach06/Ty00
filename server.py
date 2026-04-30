"""
server.py – Lightweight Flask API consumed by the React dashboard.

Endpoints
---------
GET  /api/state      – current bot state (equity, positions, risk metrics)
GET  /api/trades     – last 50 closed trades (newest first)
GET  /api/logs       – last 100 lines from logs/bot.log
PUT  /api/budget     – update TRADING_BUDGET at runtime and in .env

Run alongside the bot:
    python server.py
Then open: http://localhost:5000  (or start the React dev server on :5173)
"""
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import config
import runtime_settings
from capital_client import CapitalClient

app = Flask(__name__, static_folder="dashboard/dist", static_url_path="")
CORS(app)  # allow Vite dev server on :5173 to call :5000

logger = logging.getLogger(__name__)
_dashboard_client: CapitalClient | None = None


def _to_float(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("€", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0


def _load_json_file(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return default


def _capital_client() -> CapitalClient | None:
    global _dashboard_client

    if _dashboard_client is None:
        client = CapitalClient()
        if not client.authenticate():
            logger.warning("Dashboard API could not authenticate with Capital.com")
            return None
        _dashboard_client = client
        return _dashboard_client

    _dashboard_client._ensure_session()
    return _dashboard_client


def _iso_age_hours(created_at: str | None) -> float:
    if not created_at:
        return 0.0
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - created).total_seconds() / 3600, 2)
    except ValueError:
        return 0.0


def _fetch_live_positions() -> list[dict]:
    client = _capital_client()
    if not client:
        logger.warning("Live positions fetch skipped: no authenticated Capital client")
        return _fetch_live_positions_subprocess()

    data = client.get_positions() or {}
    count = len(data.get("positions", []))
    logger.info("Live positions raw count from Capital: %d", count)
    if count == 0:
        return _fetch_live_positions_subprocess()
    output = []
    for entry in data.get("positions", []):
        position = entry.get("position", {})
        market = entry.get("market", {})
        created_at = position.get("createdDateUTC")
        output.append({
            "deal_id": position.get("dealId"),
            "epic": market.get("epic") or market.get("symbol") or "UNKNOWN",
            "direction": position.get("direction", "—"),
            "entry": float(position.get("level") or 0.0),
            "stop": None,
            "tp": None,
            "size": float(position.get("size") or 0.0),
            "opened_at": created_at,
            "age_hours": _iso_age_hours(created_at),
            "dry_run": False,
            "source": "capital-account",
            "upl": float(position.get("upl") or 0.0),
        })
    return output


def _fetch_live_positions_subprocess() -> list[dict]:
    code = (
        "import json; "
        "from capital_client import CapitalClient; "
        "c=CapitalClient(); "
        "ok=c.authenticate(); "
        "data=(c.get_positions() or {}) if ok else {}; "
        "print(json.dumps(data))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        payload = json.loads((result.stdout or "{}").strip() or "{}")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []

    output = []
    for entry in payload.get("positions", []):
        position = entry.get("position", {})
        market = entry.get("market", {})
        created_at = position.get("createdDateUTC")
        output.append({
            "deal_id": position.get("dealId"),
            "epic": market.get("epic") or market.get("symbol") or "UNKNOWN",
            "direction": position.get("direction", "—"),
            "entry": float(position.get("level") or 0.0),
            "stop": None,
            "tp": None,
            "size": float(position.get("size") or 0.0),
            "opened_at": created_at,
            "age_hours": _iso_age_hours(created_at),
            "dry_run": False,
            "source": "capital-account",
            "upl": float(position.get("upl") or 0.0),
        })
    return output


def _fetch_live_transactions() -> list[dict]:
    client = _capital_client()
    if not client:
        return []

    data = client._get("/history/transactions") or {}
    transactions = data.get("transactions", [])
    output = []
    for item in transactions[-50:]:
        output.append({
            "time": item.get("dateUtc") or item.get("date"),
            "epic": item.get("instrumentName") or item.get("reference") or "UNKNOWN",
            "direction": item.get("transactionType") or "—",
            "entry": None,
            "exit": None,
            "size": item.get("size") or item.get("quantity") or 0,
            "pnl_eur": _to_float(item.get("profitAndLoss")),
            "reason": item.get("transactionType") or item.get("cashTransactionType") or "history",
        })
    return list(reversed(output))


def _compute_realized_pnl_from_transactions() -> dict:
    client = _capital_client()
    if not client:
        return {"realized_pnl_today": 0.0, "realized_pnl_all": 0.0, "transaction_count": 0}

    data = client._get("/history/transactions") or {}
    txs = data.get("transactions", [])

    today = datetime.now(timezone.utc).date()
    realized_today = 0.0
    realized_all = 0.0
    counted = 0

    for tx in txs:
        tx_type = str(tx.get("transactionType") or tx.get("cashTransactionType") or "").upper()
        # Keep only broker trade transaction rows.
        if tx_type != "TRADE":
            continue

        pnl_raw = tx.get("profitAndLoss")
        pnl = _to_float(pnl_raw)
        has_pnl = pnl_raw is not None and str(pnl_raw).strip() != ""
        realized_all += pnl
        if has_pnl:
            counted += 1

        raw_date = tx.get("dateUtc") or tx.get("date")
        if not raw_date:
            continue
        try:
            dt = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt.date() == today and has_pnl:
                realized_today += pnl
        except ValueError:
            continue

    return {
        "realized_pnl_today": round(realized_today, 2),
        "realized_pnl_all": round(realized_all, 2),
        "transaction_count": counted,
    }


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/state")
def api_state():
    state = _load_json_file("bot_state.json", None)
    if state is None:
        return jsonify({"error": "Bot not running or no state written yet."}), 503

    live_positions = _fetch_live_positions()
    logger.info(
        "API /state merge | bot_positions=%d | live_positions=%d",
        len(state.get("open_positions", [])),
        len(live_positions),
    )
    if not state.get("open_positions") and live_positions:
        state["open_positions"] = live_positions
        state["account_open_positions"] = live_positions
    else:
        state["account_open_positions"] = live_positions

    return jsonify(state)


@app.route("/api/trades")
def api_trades():
    live = _fetch_live_transactions()
    if live:
        return jsonify(live)
    return jsonify([])


@app.route("/api/realized-pnl")
def api_realized_pnl():
    return jsonify(_compute_realized_pnl_from_transactions())


@app.route("/api/logs")
def api_logs():
    log_path = os.path.join("logs", "bot.log")
    if os.path.exists(log_path):
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()[-100:]
            return jsonify([l.rstrip() for l in lines])
        except OSError:
            pass
    return jsonify(["No log file found yet. Start the bot with: python main.py"])


@app.route("/api/budget", methods=["PUT"])
def api_set_budget():
    """Update TRADING_BUDGET at runtime and persist to .env."""
    data = request.get_json(silent=True) or {}
    try:
        new_budget = float(data.get("budget", ""))
    except (TypeError, ValueError):
        return jsonify({"error": "budget must be a number"}), 400

    if new_budget < 1:
        return jsonify({"error": "budget must be ≥ €1"}), 400
    if new_budget > config.INITIAL_BALANCE * 2:
        return jsonify({"error": f"budget cannot exceed €{config.INITIAL_BALANCE * 2:.2f}"}), 400

    # Update runtime value immediately (picked up by bot on next tick)
    config.TRADING_BUDGET = new_budget
    runtime_settings.set_budget(new_budget)

    # Persist to .env so it survives restarts
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            content = f.read()
        # Replace the existing TRADING_BUDGET line, or append it
        pattern = r"^TRADING_BUDGET=.*$"
        replacement = f"TRADING_BUDGET={new_budget:.2f}"
        if re.search(pattern, content, flags=re.MULTILINE):
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        else:
            content += f"\n{replacement}\n"
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(content)

    return jsonify({"budget": new_budget, "status": "ok"})


# ── Serve the production React build (after: cd dashboard && npm run build) ───

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    dist = os.path.join("dashboard", "dist")
    target = os.path.join(dist, path)
    if path and os.path.exists(target):
        return send_from_directory(dist, path)
    index = os.path.join(dist, "index.html")
    if os.path.exists(index):
        return send_from_directory(dist, "index.html")
    return (
        "<h2>Dashboard not built yet.</h2>"
        "<p>Run: <code>cd dashboard &amp;&amp; npm install &amp;&amp; npm run build</code></p>"
        "<p>Or for live development: <code>cd dashboard &amp;&amp; npm run dev</code></p>",
        200,
    )


if __name__ == "__main__":
    print("Dashboard API: http://localhost:5000/api/state")
    print("React dev server (run separately): http://localhost:5173")
    app.run(host="0.0.0.0", port=5000, debug=False)
