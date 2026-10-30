"""
main.py – Intraday mean-reversion trading bot for Capital.com

Usage
-----
    python main.py            # DRY_RUN=true in .env → signals only, no orders
    python main.py --live     # override DRY_RUN to False (requires confirmed setup)

Safety checklist before first live run
---------------------------------------
  1. Fill in CAPITAL_IDENTIFIER (your login email) in .env
  2. Confirm DRY_RUN=false only after reviewing 20+ dry-run sessions in logs/
  3. Keep MAX_DRAWDOWN_PCT=0.07 and RISK_PER_TRADE_PCT=0.01 for the first week
  4. Regenerate your Capital.com API key after sharing it anywhere

Architecture
------------
  main loop (every POLL_INTERVAL seconds)
  ├─ auth health check
  ├─ equity sync → risk manager update
  ├─ kill-switch evaluation
  ├─ session-end force-flat (>= SESSION_END_HOUR)
  └─ per-instrument
       ├─ sync live positions with our state
       ├─ exit check on open positions (time stop)
       └─ entry check on flat instruments (spread → signal → size → order)
"""

import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import config
import bot_logger
import state_writer
import news_sentiment
import runtime_settings
from capital_client import CapitalClient
from risk_manager import RiskManager
from strategy import generate_signal, update_trailing_stop
from rl_agent import RLAgent
from fx_agent_committee import run_fx_agent_committee
from ny_session_agent import run_us_tech_agent

# ── Initialise logging before anything else ───────────────────────────────────
bot_logger.setup()
logger = logging.getLogger("main")

CRYPTO_CANDIDATES = ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "DOGEUSD", "BNBUSD", "LTCUSD", "ADAUSD"]
FX_REJECT_SWITCH_THRESHOLD = 5
SPRINT_LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprint.lock")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mid(market: dict) -> float:
    try:
        snap = market["snapshot"]
        return (snap["bid"] + snap["offer"]) / 2
    except (KeyError, TypeError):
        return 0.0


def _spread(market: dict) -> float:
    try:
        snap = market["snapshot"]
        return snap["offer"] - snap["bid"]
    except (KeyError, TypeError):
        return 9999.0


def _execution_price(market: dict, direction: str, fallback: float) -> float:
    try:
        snap = market["snapshot"]
        if direction == "BUY":
            return float(snap.get("offer", fallback))
        if direction == "SELL":
            return float(snap.get("bid", fallback))
    except (KeyError, TypeError, ValueError):
        pass
    return fallback


def _in_session() -> bool:
    hour = datetime.now(timezone.utc).hour
    return config.SESSION_START_HOUR <= hour < config.SESSION_END_HOUR


def _fetch_live_balance(client: CapitalClient, fallback: float) -> float:
    data = client.get_accounts()
    if not data:
        return fallback
    try:
        accounts = data.get("accounts", [data])
        for acct in accounts:
            bal = acct.get("balance", {})
            if "balance" in bal:
                return float(bal["balance"])
    except Exception:
        pass
    return fallback


def _is_fx_epic(epic: str) -> bool:
    if len(epic) != 6:
        return False
    majors = {"EUR", "GBP", "USD", "JPY", "AUD", "NZD", "CAD", "CHF"}
    base = epic[:3]
    quote = epic[3:]
    return base in majors and quote in majors


def _discover_top_cryptos(client: CapitalClient, limit: int = 5) -> list[str]:
    discovered: list[str] = []
    for epic in CRYPTO_CANDIDATES:
        if client.get_market_info(epic):
            discovered.append(epic)
        if len(discovered) >= limit:
            break
    return discovered


def _sync_positions(client: CapitalClient, tracked: dict) -> tuple[dict, list, set[str], bool]:
    """
    Compare our tracked positions with live Capital.com positions.
    If a position we track has disappeared, it was closed externally.
    Returns updated tracking, externally closed positions, occupied instruments,
    and whether the broker snapshot was valid.
    """
    live_data = client.get_positions()
    if not isinstance(live_data, dict) or not isinstance(live_data.get("positions"), list):
        logger.warning("Position reconciliation unavailable; retaining state and blocking new entries")
        return tracked, [], set(), False

    live_ids: set[str] = set()
    live_epics: set[str] = set()
    for entry in live_data["positions"]:
        position = entry.get("position", {})
        market = entry.get("market", {})
        deal_id = position.get("dealId")
        epic = market.get("epic") or market.get("symbol")
        if deal_id:
            live_ids.add(str(deal_id))
        if epic:
            live_epics.add(str(epic))

    closed = []
    for epic in list(tracked.keys()):
        pos = tracked[epic]
        if pos.get("dry_run"):
            continue  # dry-run positions are managed locally
        if str(pos.get("deal_id")) not in live_ids:
            logger.info(
                "%s: position %s no longer live (closed externally)",
                epic, pos.get("deal_id"),
            )
            closed.append((epic, pos))
            tracked.pop(epic)

    return tracked, closed, live_epics, True


# ── Main trading loop ─────────────────────────────────────────────────────────

def run():
    # ── Startup validation ────────────────────────────────────────────────────
    if config.IDENTIFIER == "YOUR_CAPITAL_COM_EMAIL_HERE" or not config.IDENTIFIER:
        logger.critical(
            "CAPITAL_IDENTIFIER is not set. Open .env and replace "
            "'YOUR_CAPITAL_COM_EMAIL_HERE' with your Capital.com login email."
        )
        sys.exit(1)

    dry_run = config.DRY_RUN
    if "--live" in sys.argv:
        dry_run = False
    if dry_run:
        logger.info("Running in DRY-RUN mode (no orders will be submitted)")
    else:
        logger.warning("LIVE mode active – real orders WILL be submitted to Capital.com")

    # ── Connect ───────────────────────────────────────────────────────────────
    client = CapitalClient()
    if not client.authenticate():
        logger.critical("Cannot authenticate with Capital.com. Check credentials in .env.")
        sys.exit(1)

    balance = _fetch_live_balance(client, config.INITIAL_BALANCE)
    risk    = RiskManager(initial_balance=balance)
    rl_agent = RLAgent()
    logger.info(
        "Bot started | balance=%.2f EUR | budget=%.2f EUR | dry_run=%s | demo=%s | instruments=%s",
        balance, config.TRADING_BUDGET, dry_run, config.USE_DEMO, config.INSTRUMENTS,
    )

    # Keyed by epic; each value: {deal_id, direction, entry, stop, tp, size, opened_at, dry_run}
    tracked: dict[str, dict] = {}
    active_instruments = list(config.INSTRUMENTS)
    fx_size_rejects = 0
    # Sentiment polled every 10 minutes (Capital.com /clientsentiment rate limit)
    sentiments: dict[str, float] = {}
    _sentiment_last_fetch: float = 0.0
    _SENTIMENT_INTERVAL   = 600  # seconds between sentiment refreshes

    # ── Loop ──────────────────────────────────────────────────────────────────
    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            logger.info("── tick %s UTC ──────────────────────────", now_utc.strftime("%H:%M:%S"))

            runtime_budget = runtime_settings.get_budget(config.TRADING_BUDGET)
            if abs(runtime_budget - risk.budget) > 1e-9:
                risk.budget = runtime_budget
                config.TRADING_BUDGET = runtime_budget
                logger.info("Runtime budget updated to €%.2f", runtime_budget)

            # ── Fetch news/sentiment (every 10 min, staggered from tick start) ────
            if time.time() - _sentiment_last_fetch >= _SENTIMENT_INTERVAL:
                try:
                    sentiments = news_sentiment.get_all_sentiments(active_instruments, client)
                    _sentiment_last_fetch = time.time()
                except Exception as _exc:
                    logger.debug("Sentiment fetch error: %s", _exc)

            # ── Sync live equity ──────────────────────────────────────────────
            live_balance = _fetch_live_balance(client, risk.equity)
            risk.update_equity(live_balance)
            logger.info("Risk: %s", risk.status_line())

            # ── Write state for dashboard (always, before any continue) ──────
            state_writer.write_state(risk, tracked, dry_run, sentiments,
                                     rl_summary=rl_agent.performance_summary())

            # ── Sync positions (detect SL/TP hits) ───────────────────────────
            broker_open_epics: set[str] = set()
            positions_synced = True
            if not dry_run:
                tracked, extern_closed, broker_open_epics, positions_synced = _sync_positions(client, tracked)
                # RL update for positions closed externally by SL or TP
                for epic, pos in extern_closed:
                    if "rl_state" not in pos:
                        continue
                    market = client.get_market_info(epic)
                    current_mid = _mid(market) if market else pos["entry"]
                    pnl_est = client.estimate_pnl(epic, pos["direction"], pos["entry"], current_mid, pos["size"], market=market)
                    close_reason = "extern_profit_close" if pnl_est >= 0 else "sl_hit"
                    logger.info(
                        "%s: RL update (extern close %s) | pnl_est=%.2f EUR",
                        epic, close_reason, pnl_est,
                    )
                    rl_agent.update(
                        pos["rl_state"], pos["rl_action"],
                        pnl_est, pos.get("rl_risk", 1.0),
                    )
                    risk.record_trade_result(pnl_est)
                    state_writer.append_trade(
                        epic, pos["direction"], pos["entry"],
                        current_mid, pos["size"], pnl_est, close_reason,
                    )

            # ── Kill switch ───────────────────────────────────────────────────
            # Skip halt if sprint.py is actively running – let it manage its own
            # positions and risk. The sprint lock is removed when sprint ends.
            halt, reason = risk.check_kill_switch()
            if halt and not os.path.exists(SPRINT_LOCK_FILE):
                logger.warning("HALT: %s – closing all positions", reason)
                if not dry_run:
                    client.close_all_positions()
                else:
                    tracked.clear()
                time.sleep(config.POLL_INTERVAL * 5)
                continue
            elif halt:
                logger.warning("HALT condition met (%s) but sprint is active – deferring to sprint", reason)

            # ── Force-flat at session end ─────────────────────────────────────
            if now_utc.hour >= config.SESSION_END_HOUR:
                if tracked:
                    logger.info("Session end – force-closing all positions")
                    for epic, pos in list(tracked.items()):
                        if not dry_run and not pos.get("dry_run"):
                            client.close_position(pos["deal_id"])
                        market = client.get_market_info(epic) if not dry_run else None
                        current_mid = _mid(market) if market else pos["entry"]
                        pnl_est = client.estimate_pnl(epic, pos["direction"], pos["entry"], current_mid, pos["size"], market=market)
                        if "rl_state" in pos:
                            rl_agent.update(
                                pos["rl_state"], pos["rl_action"],
                                pnl_est, pos.get("rl_risk", 1.0),
                            )
                        risk.record_trade_result(pnl_est)
                        state_writer.append_trade(
                            epic, pos["direction"], pos["entry"],
                            current_mid, pos["size"], pnl_est, "session_end",
                        )
                        tracked.pop(epic)
                        logger.info("Closed %s at session end | pnl_est=%.2f EUR", epic, pnl_est)
                time.sleep(60)
                continue

            # ── Outside session window ────────────────────────────────────────
            if not _in_session():
                logger.debug("Outside session window, waiting…")
                time.sleep(60)
                continue

            # ── FX Agent Committee ────────────────────────────────────────────
            # Build market_info_map from a single batch of get_market_info calls
            # (we need these anyway for spread + mid; committee reuses them for free).
            _committee_mkt_map: dict = {}
            for _ce in active_instruments:
                _cm = client.get_market_info(_ce)
                if _cm:
                    _committee_mkt_map[_ce] = _cm
            committee_decisions = run_fx_agent_committee(
                active_instruments, _committee_mkt_map, sentiments
            )

            # ── Per-instrument logic ───────────────────────────────────────────
            for epic in active_instruments:

                # ── Check time stop on open position ─────────────────────────
                if epic in tracked:
                    pos = tracked[epic]
                    market = client.get_market_info(epic)
                    current_mid = _mid(market) if market else pos["entry"]
                    previous_stop = pos["stop"]
                    pos["stop"], pos["best_price"] = update_trailing_stop(
                        pos["direction"],
                        pos["entry"],
                        pos.get("initial_stop", pos["stop"]),
                        pos["stop"],
                        pos.get("best_price", pos["entry"]),
                        current_mid,
                    )
                    if abs(pos["stop"] - previous_stop) > 1e-9:
                        logger.info(
                            "%s: trailing stop moved %.5f -> %.5f | price=%.5f",
                            epic, previous_stop, pos["stop"], current_mid,
                        )

                    stop_hit = (
                        (pos["direction"] == "BUY" and current_mid <= pos["stop"])
                        or (pos["direction"] == "SELL" and current_mid >= pos["stop"])
                    )
                    if stop_hit:
                        pnl_est = client.estimate_pnl(epic, pos["direction"], pos["entry"], current_mid, pos["size"], market=market)
                        close_reason = "trailing_stop_hit" if abs(pos["stop"] - pos.get("initial_stop", pos["stop"])) > 1e-9 else "sl_hit"
                        logger.info(
                            "%s: %s → closing | est P&L=%.2f EUR",
                            epic, close_reason, pnl_est,
                        )
                        if not dry_run and not pos.get("dry_run"):
                            client.close_position(pos["deal_id"])
                        risk.record_trade_result(pnl_est)
                        if "rl_state" in pos:
                            rl_agent.update(
                                pos["rl_state"], pos["rl_action"],
                                pnl_est, pos.get("rl_risk", 1.0),
                            )
                        state_writer.append_trade(
                            epic, pos["direction"], pos["entry"],
                            current_mid, pos["size"], pnl_est, close_reason,
                        )
                        tracked.pop(epic)
                        continue

                    age_hours = (time.time() - pos["opened_at"]) / 3600

                    if age_hours >= config.TIME_STOP_HOURS:
                        pnl_est = client.estimate_pnl(epic, pos["direction"], pos["entry"], current_mid, pos["size"], market=market)
                        logger.info(
                            "%s: time stop (%.1f h) → closing | est P&L=%.2f EUR",
                            epic, age_hours, pnl_est,
                        )
                        if not dry_run and not pos.get("dry_run"):
                            client.close_position(pos["deal_id"])
                        risk.record_trade_result(pnl_est)
                        if "rl_state" in pos:
                            rl_agent.update(
                                pos["rl_state"], pos["rl_action"],
                                pnl_est, pos.get("rl_risk", 1.0),
                            )
                        state_writer.append_trade(
                            epic, pos["direction"], pos["entry"],
                            current_mid, pos["size"], pnl_est, "time_stop",
                        )
                        tracked.pop(epic)
                    continue   # don't try to open a new position this tick

                # ── Attempt new entry ─────────────────────────────────────────                # Sprint guard: don't open new positions while sprint.py is active.
                if os.path.exists(SPRINT_LOCK_FILE):
                    logger.debug("%s: sprint lock active – skipping new entry", epic)
                    continue

                if not positions_synced:
                    logger.warning("%s: broker positions unknown – skipping new entry", epic)
                    continue
                if epic in broker_open_epics:
                    logger.warning("%s: broker already has an untracked position – skipping duplicate entry", epic)
                    continue

                # ── FX Agent Committee gate: Flow (60%) + Macro (40%) ≥ 0.35 ──
                committee_result = committee_decisions.get(epic, {})
                if committee_result.get("action") != "execute":
                    logger.debug(
                        "%s: committee filter – %s",
                        epic, committee_result.get("reason", "no decision"),
                    )
                    continue

                market = _committee_mkt_map.get(epic) or client.get_market_info(epic)
                if not market:
                    logger.warning("%s: could not fetch market info, skipping", epic)
                    continue

                mid   = _mid(market)
                sprd  = _spread(market)

                if not risk.spread_ok(sprd, epic):
                    continue  # spread filter logged inside RiskManager

                prices = client.get_prices(epic, resolution="MINUTE_15", count=70)
                if not prices:
                    logger.warning("%s: could not fetch price history, skipping", epic)
                    continue

                sentiment_score = sentiments.get(epic, 0.0)

                # ── Pass 1: default params to get ATR for RL state encoding ──
                signal0 = generate_signal(prices, mid, sentiment_score=sentiment_score)
                atr_pct = (signal0.atr / mid) if mid > 0 and signal0.atr > 0 else 0.0

                # ── RL: select best parameter variant for this context ────────
                rl_params, rl_action, rl_state = rl_agent.select(
                    epic, now_utc.hour, sentiment_score, atr_pct,
                    consecutive_losses=risk.consecutive_losses,
                )

                # ── Pass 2: generate final signal with RL-chosen params ───────
                signal = generate_signal(
                    prices, mid,
                    sentiment_score   = sentiment_score,
                    rsi_oversold      = rl_params[0],
                    rsi_overbought    = rl_params[1],
                    bb_std_override   = rl_params[2],
                    stop_atr_mult     = rl_params[3],
                    tp_rr_ratio       = rl_params[4],
                )

                if signal.direction == "NONE":
                    logger.debug("%s: no signal | %s", epic, signal.reason)
                    continue

                entry_price = _execution_price(market, signal.direction, signal.entry_price)
                stop_distance = abs(signal.entry_price - signal.stop_loss)
                tp_distance = abs(signal.take_profit - signal.entry_price)
                if stop_distance <= 0 or tp_distance <= 0:
                    logger.debug("%s: invalid distance values, skipping", epic)
                    continue

                signal.entry_price = entry_price
                if signal.direction == "BUY":
                    signal.stop_loss = entry_price - stop_distance
                    signal.take_profit = entry_price + tp_distance
                else:
                    signal.stop_loss = entry_price + stop_distance
                    signal.take_profit = entry_price - tp_distance

                lot_size = risk.calculate_lot_size(signal.entry_price, signal.stop_loss, epic)
                if lot_size == 0:
                    logger.debug("%s: lot size 0 – skipping", epic)
                    continue

                if _is_fx_epic(epic):
                    pip_size = 0.0001
                    min_stop_distance = max(_spread(market) * 2, pip_size)
                    if abs(signal.entry_price - signal.stop_loss) < min_stop_distance:
                        logger.info(
                            "%s: stop distance %.5f too tight for spread %.5f, skipping",
                            epic, abs(signal.entry_price - signal.stop_loss), _spread(market),
                        )
                        continue
                    if abs(signal.take_profit - signal.entry_price) <= 2 * _spread(market):
                        logger.info(
                            "%s: TP %.5f not sufficiently larger than spread %.5f, skipping",
                            epic, abs(signal.take_profit - signal.entry_price), _spread(market),
                        )
                        continue

                risk_eur = min(risk.equity, risk.budget) * config.RISK_PER_TRADE_PCT
                order_size, min_size, size_inc = client.calculate_order_size_for_risk(
                    epic, risk_eur, abs(signal.entry_price - signal.stop_loss), market=market,
                )
                if order_size <= 0:
                    continue
                if client.size_uses_units(epic, market=market):
                    max_size = 2_000.0
                else:
                    max_size = 10.0
                if order_size > max_size:
                    capped_size = max_size
                    order_size, min_size, size_inc = client.normalize_order_size(epic, capped_size, market=market)
                if abs(order_size - lot_size) > 1e-9:
                    logger.warning(
                        "%s: size adjusted %.5f -> %.5f (min=%.5f, inc=%.5f)",
                        epic, lot_size, order_size, min_size, size_inc,
                    )

                logger.info(
                    "%s SIGNAL ► %s | entry~%.5f | SL=%.5f | TSL=armed@+0.5R gap=0.5R | size=%.5f | %s",
                    epic, signal.direction, signal.entry_price,
                    signal.stop_loss, order_size, signal.reason,
                )

                # Approximate EUR risk for this trade (used by RL reward normalisation)
                rl_risk = abs(signal.entry_price - signal.stop_loss) * client.size_to_exposure_units(epic, order_size, market=market)
                rl_risk = max(rl_risk, 0.01)   # guard against division by zero
                exposure_units = client.size_to_exposure_units(epic, order_size, market=market)

                if dry_run:
                    logger.info("%s: DRY-RUN – order not submitted", epic)
                    tracked[epic] = {
                        "deal_id":   f"DRYRUN_{epic}_{int(time.time())}",
                        "direction": signal.direction,
                        "entry":     signal.entry_price,
                        "stop":      signal.stop_loss,
                        "initial_stop": signal.stop_loss,
                        "tp":        signal.take_profit,
                        "best_price": signal.entry_price,
                        "size":      order_size,
                        "exposure_units": exposure_units,
                        "opened_at": time.time(),
                        "dry_run":   True,
                        "rl_state":  rl_state,
                        "rl_action": rl_action,
                        "rl_risk":   rl_risk,
                    }
                else:
                    result, status = client.open_position(
                        epic, signal.direction, order_size,
                        signal.stop_loss,
                        profit_level=signal.take_profit,
                    )
                    if status in (200, 201) and result:
                        deal_ref = result.get("dealReference", "unknown")
                        logger.info("%s: order submitted | dealReference=%s", epic, deal_ref)
                        # Resolve dealId from positions (needed for close)
                        time.sleep(1)
                        pos_data = client.get_positions()
                        deal_id = deal_ref  # fallback
                        if pos_data:
                            for p in pos_data.get("positions", []):
                                if p.get("position", {}).get("dealReference") == deal_ref:
                                    deal_id = p["position"]["dealId"]
                                    break
                        tracked[epic] = {
                            "deal_id":   deal_id,
                            "direction": signal.direction,
                            "entry":     signal.entry_price,
                            "stop":      signal.stop_loss,
                            "initial_stop": signal.stop_loss,
                            "tp":        signal.take_profit,
                            "best_price": signal.entry_price,
                            "size":      order_size,
                            "exposure_units": exposure_units,
                            "opened_at": time.time(),
                            "dry_run":   False,
                            "rl_state":  rl_state,
                            "rl_action": rl_action,
                            "rl_risk":   rl_risk,
                        }
                    else:
                        code = (result or {}).get("errorCode", "") if isinstance(result, dict) else ""
                        if code == "error.invalid.size.minvalue":
                            if _is_fx_epic(epic):
                                fx_size_rejects += 1
                            bump = size_inc if size_inc > 0 else max(order_size * 0.1, 0.01)
                            retry_size = round(order_size + bump, 8)
                            retry_risk = abs(signal.entry_price - signal.stop_loss) * client.size_to_exposure_units(
                                epic, retry_size, market=market,
                            )
                            if retry_risk <= risk_eur + 1e-9:
                                logger.warning(
                                    "%s: min-size reject at %.5f, retrying with risk-checked size %.5f",
                                    epic, order_size, retry_size,
                                )
                                retry_result, retry_status = client.open_position(
                                    epic, signal.direction, retry_size,
                                    signal.stop_loss,
                                    profit_level=signal.take_profit,
                                )
                                if retry_status in (200, 201) and retry_result:
                                    deal_ref = retry_result.get("dealReference", "unknown")
                                    logger.info("%s: order submitted on retry | dealReference=%s", epic, deal_ref)
                                    tracked[epic] = {
                                        "deal_id":   deal_ref,
                                        "direction": signal.direction,
                                        "entry":     signal.entry_price,
                                        "stop":      signal.stop_loss,
                                        "initial_stop": signal.stop_loss,
                                        "tp":        signal.take_profit,
                                        "best_price": signal.entry_price,
                                        "size":      retry_size,
                                        "exposure_units": client.size_to_exposure_units(epic, retry_size, market=market),
                                        "opened_at": time.time(),
                                        "dry_run":   False,
                                        "rl_state":  rl_state,
                                        "rl_action": rl_action,
                                        "rl_risk":   max(retry_risk, 0.01),
                                    }
                                    continue
                            else:
                                logger.warning(
                                    "%s: broker minimum retry blocked; risk %.2f EUR exceeds %.2f EUR cap",
                                    epic, retry_risk, risk_eur,
                                )
                            if fx_size_rejects >= FX_REJECT_SWITCH_THRESHOLD and all(_is_fx_epic(e) for e in active_instruments):
                                cryptos = _discover_top_cryptos(client, limit=5)
                                if cryptos:
                                    active_instruments = cryptos
                                    fx_size_rejects = 0
                                    logger.warning(
                                        "Switched to top-volume crypto universe after repeated FX size rejects: %s",
                                        active_instruments,
                                    )
                        logger.error(
                            "%s: order REJECTED | status=%s | response=%s",
                            epic, status, result,
                        )

            # ── US100 NY Session Agent ────────────────────────────────────────
            # Handled separately from FX: own session gate, spread check, and
            # trend-breakout signal logic.  Uses config.US_TECH_CONFIG for risk.
            _us100_epic = config.US_TECH_CONFIG["epic"]
            if not os.path.exists(SPRINT_LOCK_FILE):
                if _us100_epic in tracked:
                    # ── Manage open US100 position ────────────────────────────
                    pos = tracked[_us100_epic]
                    mkt = client.get_market_info(_us100_epic)
                    cur = _mid(mkt) if mkt else pos["entry"]
                    previous_stop = pos["stop"]
                    pos["stop"], pos["best_price"] = update_trailing_stop(
                        pos["direction"],
                        pos["entry"],
                        pos.get("initial_stop", pos["stop"]),
                        pos["stop"],
                        pos.get("best_price", pos["entry"]),
                        cur,
                    )
                    if abs(pos["stop"] - previous_stop) > 1e-9:
                        logger.info(
                            "%s: trailing stop moved %.2f -> %.2f | price=%.2f",
                            _us100_epic, previous_stop, pos["stop"], cur,
                        )

                    stop_hit = (
                        (pos["direction"] == "BUY"  and cur <= pos["stop"])
                        or (pos["direction"] == "SELL" and cur >= pos["stop"])
                    )
                    if stop_hit:
                        pnl_est = client.estimate_pnl(
                            _us100_epic, pos["direction"], pos["entry"],
                            cur, pos["size"], market=mkt,
                        )
                        close_reason = (
                            "trailing_stop_hit"
                            if abs(pos["stop"] - pos.get("initial_stop", pos["stop"])) > 1e-9
                            else "sl_hit"
                        )
                        logger.info("%s: %s | pnl≈%+.2f EUR", _us100_epic, close_reason, pnl_est)
                        if not dry_run and not pos.get("dry_run"):
                            client.close_position(pos["deal_id"])
                        risk.record_trade_result(pnl_est)
                        if "rl_state" in pos:
                            rl_agent.update(pos["rl_state"], pos["rl_action"], pnl_est, pos.get("rl_risk", 1.0))
                        state_writer.append_trade(
                            _us100_epic, pos["direction"], pos["entry"],
                            cur, pos["size"], pnl_est, close_reason,
                        )
                        tracked.pop(_us100_epic)
                    else:
                        age_hours = (time.time() - pos["opened_at"]) / 3600
                        if age_hours >= config.US_TECH_CONFIG["time_stop_hours"]:
                            pnl_est = client.estimate_pnl(
                                _us100_epic, pos["direction"], pos["entry"],
                                cur, pos["size"], market=mkt,
                            )
                            logger.info(
                                "%s: time stop (%.1f h) | pnl≈%+.2f EUR",
                                _us100_epic, age_hours, pnl_est,
                            )
                            if not dry_run and not pos.get("dry_run"):
                                client.close_position(pos["deal_id"])
                            risk.record_trade_result(pnl_est)
                            if "rl_state" in pos:
                                rl_agent.update(pos["rl_state"], pos["rl_action"], pnl_est, pos.get("rl_risk", 1.0))
                            state_writer.append_trade(
                                _us100_epic, pos["direction"], pos["entry"],
                                cur, pos["size"], pnl_est, "time_stop",
                            )
                            tracked.pop(_us100_epic)
                else:
                    # ── Attempt new US100 entry ───────────────────────────────
                    us_mkt = client.get_market_info(_us100_epic)
                    us_decision = run_us_tech_agent(client, us_mkt)
                    logger.info(
                        "[US100] tick decision: %s | reason=%s",
                        us_decision["action"],
                        us_decision.get("reason") or us_decision.get("detail", "—"),
                    )
                    if us_decision["action"] == "execute":
                        sig        = us_decision["signal"]
                        us_mid     = us_decision["mid"]
                        base_risk  = config.US_TECH_CONFIG["base_risk_eur"]
                        stop_dist  = abs(sig.entry_price - sig.stop_loss)
                        us_order_size, us_min_size, us_size_inc = (
                            client.calculate_order_size_for_risk(
                                _us100_epic, base_risk, stop_dist, market=us_mkt,
                            )
                        )
                        if us_order_size > 0:
                            rl_risk = max(
                                stop_dist * client.size_to_exposure_units(
                                    _us100_epic, us_order_size, market=us_mkt,
                                ),
                                0.01,
                            )
                            logger.info(
                                "%s SIGNAL ► %s | entry~%.2f | SL=%.2f | TP=%.2f | "
                                "size=%.5f | risk=€%.2f | %s",
                                _us100_epic, sig.direction, sig.entry_price,
                                sig.stop_loss, sig.take_profit,
                                us_order_size, base_risk, sig.reason,
                            )
                            if dry_run:
                                logger.info("%s: [DRY-RUN] position recorded", _us100_epic)
                                tracked[_us100_epic] = {
                                    "deal_id":    f"DRY_{_us100_epic}_{int(time.time())}",
                                    "direction":  sig.direction,
                                    "entry":      us_mid,
                                    "stop":       sig.stop_loss,
                                    "initial_stop": sig.stop_loss,
                                    "tp":         sig.take_profit,
                                    "best_price": us_mid,
                                    "size":       us_order_size,
                                    "exposure_units": client.size_to_exposure_units(
                                        _us100_epic, us_order_size, market=us_mkt,
                                    ),
                                    "opened_at":  time.time(),
                                    "dry_run":    True,
                                    "rl_risk":    rl_risk,
                                }
                            else:
                                us_result, us_status = client.open_position(
                                    _us100_epic, sig.direction,
                                    us_order_size, sig.stop_loss,
                                    profit_level=sig.take_profit,
                                )
                                if us_status in (200, 201) and us_result:
                                    deal_ref = us_result.get("dealReference", "unknown")
                                    logger.info(
                                        "%s: order submitted | dealRef=%s",
                                        _us100_epic, deal_ref,
                                    )
                                    time.sleep(1)
                                    pos_data = client.get_positions()
                                    deal_id  = deal_ref
                                    if pos_data:
                                        for p in pos_data.get("positions", []):
                                            if p.get("position", {}).get("dealReference") == deal_ref:
                                                deal_id = p["position"]["dealId"]
                                                break
                                    tracked[_us100_epic] = {
                                        "deal_id":    deal_id,
                                        "direction":  sig.direction,
                                        "entry":      us_mid,
                                        "stop":       sig.stop_loss,
                                        "initial_stop": sig.stop_loss,
                                        "tp":         sig.take_profit,
                                        "best_price": us_mid,
                                        "size":       us_order_size,
                                        "exposure_units": client.size_to_exposure_units(
                                            _us100_epic, us_order_size, market=us_mkt,
                                        ),
                                        "opened_at":  time.time(),
                                        "dry_run":    False,
                                        "rl_risk":    rl_risk,
                                    }
                                else:
                                    logger.error(
                                        "%s: order REJECTED | status=%s | response=%s",
                                        _us100_epic, us_status, us_result,
                                    )

            # (state already written at top of tick)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received – shutting down cleanly")
            logger.info("Closing all tracked positions…")
            if not dry_run:
                for epic, pos in tracked.items():
                    if not pos.get("dry_run"):
                        client.close_position(pos["deal_id"])
                        logger.info("Closed %s", epic)
            logger.info("Bot stopped.")
            break

        except Exception as exc:
            logger.error("Unexpected error in main loop: %s", exc, exc_info=True)

        time.sleep(config.POLL_INTERVAL)


if __name__ == "__main__":
    run()
