"""
sprint.py – High-frequency 1-hour live trading sprint.

Trades EURUSD and GBPUSD simultaneously with:
  • 15-second poll interval  (vs 60 s in main bot)
  • 1-minute candles         (vs 15-min)
  • Relaxed RSI 35/65        (vs 30/70 – more signal opportunities)
  • Shorter time-stop 30 min (vs 4 h)
  • Re-entry allowed on same epic after a trade closes
  • Total session budget €50 enforced hard
  • Auto-terminates after SPRINT_MINUTES (default 60)

Safety:
  • MAX_DAILY_LOSS_PCT 3% of budget → halt
  • MAX_DRAWDOWN_PCT   5% of equity → halt
  • Max 2 open positions at once (one per epic)
  • All positions force-closed at sprint end

Usage
-----
    python sprint.py               # 60-min sprint
    python sprint.py --minutes 30  # 30-min sprint
    python sprint.py --dry         # paper mode (no real orders)
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone

import config
import bot_logger
import state_writer
import news_sentiment
import runtime_settings
from capital_client import CapitalClient
from risk_manager import RiskManager
from rl_agent import RLAgent
from strategy import _rsi, _bollinger, _atr, Signal

import numpy as np
import pandas as pd

bot_logger.setup()
logger = logging.getLogger("sprint")

# ── Sprint-specific constants ────────────────────────────────────────────────
POLL_SECS         = 15        # tick every 15 seconds
CANDLE_RESOLUTION = "MINUTE"   # Capital.com code for 1-minute bars
CANDLE_COUNT      = 120       # 2 hours of 1-min bars
RSI_OS            = 42        # more permissive for live opportunity capture
RSI_OB            = 58        # more permissive for live opportunity capture
BB_STD            = 2.0
STOP_MULT         = 1.5
TP_RR             = 2.0
TIME_STOP_MINS    = 30        # force-close after 30 min
SPRINT_BUDGET     = 50.0      # EUR
MAX_LOSS_PCT      = 0.03      # 3% of budget = €1.50 daily halt
MAX_DD_PCT        = 0.05      # 5% equity drawdown = kill
INSTRUMENTS       = ["EURUSD", "GBPUSD"]
CRYPTO_CANDIDATES = ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "DOGEUSD", "BNBUSD", "LTCUSD", "ADAUSD"]
FX_REJECT_SWITCH_THRESHOLD = 5


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


def _fetch_balance(client: CapitalClient, fallback: float) -> float:
    data = client.get_accounts()
    if not data:
        return fallback
    try:
        for acct in data.get("accounts", [data]):
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
    found: list[str] = []
    for epic in CRYPTO_CANDIDATES:
        if client.get_market_info(epic):
            found.append(epic)
        if len(found) >= limit:
            break
    return found


def build_df(candles: list) -> pd.DataFrame:
    rows = []
    for c in candles:
        try:
            rows.append({
                "open":  (c["openPrice"]["bid"]  + c["openPrice"]["ask"])  / 2,
                "high":  (c["highPrice"]["bid"]   + c["highPrice"]["ask"])  / 2,
                "low":   (c["lowPrice"]["bid"]    + c["lowPrice"]["ask"])   / 2,
                "close": (c["closePrice"]["bid"]  + c["closePrice"]["ask"]) / 2,
            })
        except (KeyError, TypeError):
            continue
    return pd.DataFrame(rows)


def get_signal(df: pd.DataFrame, mid: float, sentiment: float) -> Signal:
    """Fast signal generation on 1-min bars with relaxed thresholds."""
    if len(df) < 30:
        return Signal("NONE", mid, 0, 0, 0, 0, f"need ≥30 bars, have {len(df)}")

    closes = df["close"]
    rsi_s = _rsi(closes, 14)
    _, _, lower_bb = _bollinger(closes, 20, BB_STD)
    upper_bb, _, _ = _bollinger(closes, 20, BB_STD)
    atr_s = _atr(df["high"], df["low"], closes, 14)

    curr_rsi   = rsi_s.iloc[-1]
    curr_close = closes.iloc[-1]
    curr_lower = lower_bb.iloc[-1]
    curr_upper = upper_bb.iloc[-1]
    curr_atr   = atr_s.iloc[-1]

    if any(pd.isna(v) for v in [curr_rsi, curr_lower, curr_upper, curr_atr]) or curr_atr == 0:
        return Signal("NONE", mid, 0, 0, 0, curr_rsi or 0, "indicators not ready")

    stop_d = curr_atr * STOP_MULT
    tp_d   = stop_d * TP_RR

    near_lower = curr_close <= curr_lower * 1.0006
    near_upper = curr_close >= curr_upper * 0.9994

    if curr_rsi < RSI_OS or near_lower:
        if sentiment <= -0.35:
            return Signal("NONE", mid, 0, 0, 0, curr_rsi,
                          f"BUY suppressed: bearish sentiment={sentiment:.2f}")
        return Signal("BUY", mid, mid - stop_d, mid + tp_d, curr_atr, curr_rsi,
                      f"BUY trigger RSI={curr_rsi:.1f}<{RSI_OS} or nearLowerBB | sent={sentiment:+.2f}")

    if curr_rsi > RSI_OB or near_upper:
        if sentiment >= 0.35:
            return Signal("NONE", mid, 0, 0, 0, curr_rsi,
                          f"SELL suppressed: bullish sentiment={sentiment:.2f}")
        return Signal("SELL", mid, mid + stop_d, mid - tp_d, curr_atr, curr_rsi,
                      f"SELL trigger RSI={curr_rsi:.1f}>{RSI_OB} or nearUpperBB | sent={sentiment:+.2f}")

    return Signal("NONE", mid, 0, 0, 0, curr_rsi, f"RSI={curr_rsi:.1f} – no extreme")


def sprint(minutes: int, dry: bool):
    end_ts  = time.time() + minutes * 60
    start_budget = runtime_settings.get_budget(config.TRADING_BUDGET if config.TRADING_BUDGET > 0 else SPRINT_BUDGET)
    logger.warning(
        "═══ SPRINT START ═══ %d minutes | budget=€%.0f | dry=%s | instruments=%s",
        minutes, start_budget, dry, INSTRUMENTS,
    )

    client = CapitalClient()
    if not client.authenticate():
        logger.critical("Authentication failed – check .env")
        sys.exit(1)

    balance   = _fetch_balance(client, config.INITIAL_BALANCE)
    risk      = RiskManager(initial_balance=balance)
    risk.budget = start_budget
    rl        = RLAgent()
    tracked   : dict[str, dict] = {}
    active_instruments = list(INSTRUMENTS)
    fx_size_rejects = 0
    sentiments: dict[str, float] = {}
    sprint_pnl = 0.0
    tick_n     = 0

    logger.info("Balance: €%.2f | RL states loaded: %d", balance, len(rl.Q))

    while time.time() < end_ts:
        tick_n += 1
        remaining = int(end_ts - time.time())
        now_utc   = datetime.now(timezone.utc)
        logger.info("── sprint tick %d | ⏱ %02d:%02d remaining ──",
                    tick_n, remaining // 60, remaining % 60)

        # ── Re-auth if needed ──────────────────────────────────────────────
        client._ensure_session()

        runtime_budget = runtime_settings.get_budget(risk.budget)
        if abs(runtime_budget - risk.budget) > 1e-9:
            risk.budget = runtime_budget
            logger.info("Runtime budget updated to €%.2f", runtime_budget)

        # ── Equity sync ────────────────────────────────────────────────────
        live_bal = _fetch_balance(client, risk.equity)
        risk.update_equity(live_bal)

        # ── Sentiment (cached 5 min) ───────────────────────────────────────
        try:
            sentiments = news_sentiment.get_all_sentiments(active_instruments, client)
        except Exception as exc:
            logger.debug("Sentiment error: %s", exc)

        # ── Write dashboard state ──────────────────────────────────────────
        state_writer.write_state(risk, tracked, dry, sentiments,
                                 rl_summary=rl.performance_summary())

        # ── Sync live positions (detect SL/TP hits) ────────────────────────
        if not dry:
            live_data = client.get_positions()
            live_ids  = set()
            if live_data:
                for entry in live_data.get("positions", []):
                    did = entry.get("position", {}).get("dealId")
                    if did:
                        live_ids.add(did)

            for epic in list(tracked.keys()):
                pos = tracked[epic]
                if pos.get("dry_run"):
                    continue
                if pos["deal_id"] not in live_ids:
                    # Determine most likely exit (SL vs TP)
                    mkt = client.get_market_info(epic)
                    cur = _mid(mkt) if mkt else pos["entry"]
                    direction_sign = 1 if pos["direction"] == "BUY" else -1
                    pnl_pct = direction_sign * (cur - pos["entry"])
                    if pnl_pct >= 0:
                        pnl = client.estimate_pnl(epic, pos["direction"], pos["entry"], pos["tp"], pos["size"], market=mkt)
                        why = "tp_hit"
                    else:
                        pnl = client.estimate_pnl(epic, pos["direction"], pos["entry"], pos["stop"], pos["size"], market=mkt)
                        why = "sl_hit"
                    sprint_pnl += pnl
                    logger.info("%s: closed externally (%s) | pnl≈%+.2f EUR", epic, why, pnl)
                    if "rl_state" in pos:
                        rl.update(pos["rl_state"], pos["rl_action"], pnl, pos.get("rl_risk", 1.0))
                    risk.record_trade_result(pnl)
                    state_writer.append_trade(
                        epic, pos["direction"], pos["entry"], cur, pos["size"], pnl, why)
                    tracked.pop(epic)

        # ── Kill switch (sprint budget guard) ─────────────────────────────
        if sprint_pnl < -(risk.budget * MAX_LOSS_PCT):
            logger.warning("SPRINT HALT: daily loss limit hit | pnl=%.2f", sprint_pnl)
            break

        equity_dd = (risk.peak_equity - risk.equity) / risk.peak_equity if risk.peak_equity > 0 else 0
        if equity_dd >= MAX_DD_PCT:
            logger.warning("SPRINT HALT: drawdown %.1f%% ≥ %.0f%%", equity_dd * 100, MAX_DD_PCT * 100)
            break

        # ── Per-instrument logic ───────────────────────────────────────────
        for epic in active_instruments:

            # ── Time stop ─────────────────────────────────────────────────
            if epic in tracked:
                pos      = tracked[epic]
                age_mins = (time.time() - pos["opened_at"]) / 60
                if age_mins >= TIME_STOP_MINS:
                    mkt = client.get_market_info(epic)
                    cur = _mid(mkt) if mkt else pos["entry"]
                    pnl = client.estimate_pnl(epic, pos["direction"], pos["entry"], cur, pos["size"], market=mkt)
                    logger.info("%s: time stop (%.0f min) | pnl≈%+.2f EUR", epic, age_mins, pnl)
                    if not dry and not pos.get("dry_run"):
                        client.close_position(pos["deal_id"])
                    sprint_pnl += pnl
                    if "rl_state" in pos:
                        rl.update(pos["rl_state"], pos["rl_action"], pnl, pos.get("rl_risk", 1.0))
                    risk.record_trade_result(pnl)
                    state_writer.append_trade(
                        epic, pos["direction"], pos["entry"], cur, pos["size"], pnl, "time_stop")
                    tracked.pop(epic)
                else:
                    logger.debug("%s: position open %.1f min | SL=%.5f TP=%.5f",
                                 epic, age_mins, pos["stop"], pos["tp"])
                    continue   # still in trade, no new entry

            # ── Attempt new entry ──────────────────────────────────────────
            mkt = client.get_market_info(epic)
            if not mkt:
                logger.warning("%s: no market info", epic)
                continue

            mid  = _mid(mkt)
            sprd = _spread(mkt)

            # Spread check (use pip-equivalent: 1 pip = 0.0001 for FX)
            sprd_pips = sprd / 0.0001
            if sprd_pips > config.MAX_SPREAD_PIPS:
                logger.debug("%s: spread %.1f pips > %.1f max, skipping", epic, sprd_pips, config.MAX_SPREAD_PIPS)
                continue

            prices = client.get_prices(epic, resolution=CANDLE_RESOLUTION, count=CANDLE_COUNT)
            if not prices:
                logger.warning("%s: no price data", epic)
                continue

            df   = build_df(prices.get("prices", []))
            sent = sentiments.get(epic, 0.0)
            sig  = get_signal(df, mid, sent)

            if sig.direction == "NONE":
                logger.info("%s: no signal | %s", epic, sig.reason)
                continue

            # ── RL param selection ─────────────────────────────────────────
            atr_pct = (sig.atr / mid) if mid > 0 and sig.atr > 0 else 0.003
            rl_params, rl_action, rl_state = rl.select(
                epic, now_utc.hour, sent, atr_pct,
                consecutive_losses=risk.consecutive_losses,
            )

            # Size: risk 2% of sprint budget per trade (slightly higher for volume)
            risk_eur  = risk.budget * 0.02
            stop_dist = abs(sig.entry_price - sig.stop_loss)
            if stop_dist == 0:
                continue
            requested_size, min_size, size_inc = client.calculate_order_size_for_risk(
                epic, risk_eur, stop_dist, market=mkt,
            )
            order_size = requested_size
            if order_size <= 0:
                continue
            if client.size_uses_units(epic, market=mkt):
                max_size = 50_000.0
            else:
                max_size = 1.0
            if order_size > max_size:
                order_size = max_size
                order_size, min_size, size_inc = client.normalize_order_size(epic, order_size, market=mkt)
            if abs(order_size - requested_size) > 1e-9:
                logger.warning(
                    "%s: size capped %.5f -> %.5f (min=%.5f, inc=%.5f)",
                    epic, requested_size, order_size, min_size, size_inc,
                )

            rl_risk = stop_dist * client.size_to_exposure_units(epic, order_size, market=mkt)
            rl_risk = max(rl_risk, 0.01)
            exposure_units = client.size_to_exposure_units(epic, order_size, market=mkt)

            logger.info(
                "▶ %s %s | entry=%.5f | SL=%.5f | TP=%.5f | size=%.5f | %s",
                epic, sig.direction, sig.entry_price,
                sig.stop_loss, sig.take_profit, order_size, sig.reason,
            )

            if dry:
                tracked[epic] = {
                    "deal_id":   f"DRY_{epic}_{int(time.time())}",
                    "direction": sig.direction,
                    "entry":     mid,
                    "stop":      sig.stop_loss,
                    "tp":        sig.take_profit,
                    "size":      order_size,
                    "exposure_units": exposure_units,
                    "opened_at": time.time(),
                    "dry_run":   True,
                    "rl_state":  rl_state,
                    "rl_action": rl_action,
                    "rl_risk":   rl_risk,
                }
                logger.info("%s: [DRY-RUN] position recorded", epic)
            else:
                result, status = client.open_position(
                    epic, sig.direction, order_size, sig.stop_loss, sig.take_profit,
                )
                if status in (200, 201) and result:
                    deal_ref = result.get("dealReference", "unknown")
                    logger.info("%s: order submitted | dealRef=%s", epic, deal_ref)
                    time.sleep(1)
                    pos_data = client.get_positions()
                    deal_id  = deal_ref
                    if pos_data:
                        for p in pos_data.get("positions", []):
                            if p.get("position", {}).get("dealReference") == deal_ref:
                                deal_id = p["position"]["dealId"]
                                break
                    tracked[epic] = {
                        "deal_id":   deal_id,
                        "direction": sig.direction,
                        "entry":     mid,
                        "stop":      sig.stop_loss,
                        "tp":        sig.take_profit,
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
                            epic, sig.direction, retry_size, sig.stop_loss, sig.take_profit,
                        )
                        if retry_status in (200, 201) and retry_result:
                            deal_ref = retry_result.get("dealReference", "unknown")
                            logger.info("%s: order submitted on retry | dealRef=%s", epic, deal_ref)
                            tracked[epic] = {
                                "deal_id":   deal_ref,
                                "direction": sig.direction,
                                "entry":     mid,
                                "stop":      sig.stop_loss,
                                "tp":        sig.take_profit,
                                "size":      retry_size,
                                "exposure_units": client.size_to_exposure_units(epic, retry_size, market=mkt),
                                "opened_at": time.time(),
                                "dry_run":   False,
                                "rl_state":  rl_state,
                                "rl_action": rl_action,
                                "rl_risk":   max(stop_dist * client.size_to_exposure_units(epic, retry_size, market=mkt), 0.01),
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
                    logger.error("%s: order REJECTED | status=%s | %s", epic, status, result)

        logger.info("Sprint P&L so far: %+.2f EUR | equity: %.2f EUR",
                    sprint_pnl, risk.equity)
        time.sleep(POLL_SECS)

    # ── Sprint end: force-close all open positions ─────────────────────────
    logger.warning("═══ SPRINT END ═══ Closing all open positions…")
    for epic, pos in list(tracked.items()):
        try:
            if not dry and not pos.get("dry_run"):
                client.close_position(pos["deal_id"])
            mkt = client.get_market_info(epic) if not dry else None
            cur = _mid(mkt) if mkt else pos["entry"]
            pnl = client.estimate_pnl(epic, pos["direction"], pos["entry"], cur, pos["size"], market=mkt)
            sprint_pnl += pnl
            if "rl_state" in pos:
                rl.update(pos["rl_state"], pos["rl_action"], pnl, pos.get("rl_risk", 1.0))
            risk.record_trade_result(pnl)
            state_writer.append_trade(
                epic, pos["direction"], pos["entry"], cur, pos["size"], pnl, "sprint_end")
            logger.info("%s: closed at sprint end | pnl≈%+.2f EUR", epic, pnl)
        except Exception as exc:
            logger.error("Error closing %s: %s", epic, exc)

    final_bal = _fetch_balance(client, risk.equity)
    logger.warning(
        "═══ SPRINT SUMMARY ═══\n"
        "  Duration  : %d min\n"
        "  Trades    : %d (W:%d L:%d)\n"
        "  Sprint P&L: %+.2f EUR\n"
        "  Win rate  : %.0f%%\n"
        "  Final bal : %.2f EUR\n"
        "  RL updates: %d",
        minutes,
        risk.total_trades, risk.winning_trades,
        risk.total_trades - risk.winning_trades,
        sprint_pnl,
        risk.win_rate * 100,
        final_bal,
        rl.total_updates,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-frequency 1-hour trading sprint")
    parser.add_argument("--minutes", type=int, default=60, help="Sprint duration in minutes (default: 60)")
    parser.add_argument("--dry",     action="store_true",  help="Paper mode – log signals but submit NO real orders")
    args = parser.parse_args()

    sprint(args.minutes, args.dry)
