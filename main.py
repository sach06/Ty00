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

# ── Initialise logging before anything else ───────────────────────────────────
bot_logger.setup()
logger = logging.getLogger("main")

CRYPTO_CANDIDATES = ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "DOGEUSD", "BNBUSD", "LTCUSD", "ADAUSD"]
FX_REJECT_SWITCH_THRESHOLD = 5


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


def _sync_positions(client: CapitalClient, tracked: dict) -> tuple[dict, list]:
    """
    Compare our tracked positions with live Capital.com positions.
    If a position we track has disappeared, it was closed externally.
    Returns (updated_tracked, closed_positions) where closed_positions is a
    list of (epic, pos_dict) tuples for externally closed positions.
    """
    live_data = client.get_positions()
    live_ids: set[str] = set()
    if live_data:
        for entry in live_data.get("positions", []):
            did = entry.get("position", {}).get("dealId")
            if did:
                live_ids.add(did)

    closed = []
    for epic in list(tracked.keys()):
        pos = tracked[epic]
        if pos.get("dry_run"):
            continue  # dry-run positions are managed locally
        if pos["deal_id"] not in live_ids:
            logger.info(
                "%s: position %s no longer live (closed externally)",
                epic, pos["deal_id"],
            )
            closed.append((epic, pos))
            tracked.pop(epic)

    return tracked, closed


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
    # Sentiment cache refreshed every tick (internally cached for 5 min)
    sentiments: dict[str, float] = {}

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

            # ── Fetch news/sentiment (cached 5 min, fast after first call) ────
            try:
                sentiments = news_sentiment.get_all_sentiments(active_instruments, client)
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
            if not dry_run:
                tracked, extern_closed = _sync_positions(client, tracked)
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
            halt, reason = risk.check_kill_switch()
            if halt:
                logger.warning("HALT: %s – closing all positions", reason)
                if not dry_run:
                    client.close_all_positions()
                else:
                    tracked.clear()
                time.sleep(config.POLL_INTERVAL * 5)
                continue

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

                # ── Attempt new entry ─────────────────────────────────────────
                market = client.get_market_info(epic)
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

                lot_size = risk.calculate_lot_size(signal.entry_price, signal.stop_loss, epic)
                if lot_size == 0:
                    logger.debug("%s: lot size 0 – skipping", epic)
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
                        "entry":     mid,
                        "stop":      signal.stop_loss,
                        "initial_stop": signal.stop_loss,
                        "tp":        signal.take_profit,
                        "best_price": mid,
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
                            "entry":     mid,
                            "stop":      signal.stop_loss,
                            "initial_stop": signal.stop_loss,
                            "tp":        signal.take_profit,
                            "best_price": mid,
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
                            logger.warning(
                                "%s: min-size reject at %.5f, retrying with %.5f",
                                epic, order_size, retry_size,
                            )
                            retry_result, retry_status = client.open_position(
                                epic, signal.direction, retry_size,
                                signal.stop_loss,
                            )
                            if retry_status in (200, 201) and retry_result:
                                deal_ref = retry_result.get("dealReference", "unknown")
                                logger.info("%s: order submitted on retry | dealReference=%s", epic, deal_ref)
                                tracked[epic] = {
                                    "deal_id":   deal_ref,
                                    "direction": signal.direction,
                                    "entry":     mid,
                                    "stop":      signal.stop_loss,
                                    "initial_stop": signal.stop_loss,
                                    "tp":        signal.take_profit,
                                    "best_price": mid,
                                    "size":      retry_size,
                                    "exposure_units": client.size_to_exposure_units(epic, retry_size, market=market),
                                    "opened_at": time.time(),
                                    "dry_run":   False,
                                    "rl_state":  rl_state,
                                    "rl_action": rl_action,
                                    "rl_risk":   max(abs(signal.entry_price - signal.stop_loss) * client.size_to_exposure_units(epic, retry_size, market=market), 0.01),
                                }
                                continue
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
