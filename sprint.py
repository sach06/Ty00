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
from strategy import _rsi, _bollinger, _atr, Signal, update_trailing_stop

import numpy as np
import pandas as pd

bot_logger.setup()
logger = logging.getLogger("sprint")

# ── Sprint-specific constants ────────────────────────────────────────────────
POLL_SECS         = 15        # tick every 15 seconds
CANDLE_RESOLUTION = "MINUTE_5" # 5-minute candles – less noise / spread impact than 1-min
CANDLE_COUNT      = 80        # ~6.5 hours of 5-min bars (sufficient for all indicators)
RSI_OS            = 35        # tighter entry – only act on genuine oversold
RSI_OB            = 65        # tighter entry – only act on genuine overbought
BB_STD            = 2.0       # fixed – not tuned by RL to avoid overfitting
STOP_MULT         = 1.5       # fixed – 1.5×ATR initial stop
TP_RR             = 2.0
TIME_STOP_MINS    = 30        # force-close after 30 min
SPRINT_BUDGET     = 50.0      # EUR
MAX_LOSS_PCT      = 0.03      # 3% of budget = €1.50 daily halt
MAX_DD_PCT        = 0.05      # 5% equity drawdown = kill
TRAIL_ARM_R       = 1.0       # arm trailing stop after +1R (1.5×ATR) profit
TRAIL_GAP_R       = 1.0       # trail 1R (≈1.5×ATR) behind best price
CORR_THRESHOLD    = 0.75      # max allowed EURUSD/GBPUSD correlation before halving risk
CORR_BARS         = 60        # bars used for rolling correlation check
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


def _correlation(prices_a: list, prices_b: list, n_bars: int = CORR_BARS) -> float:
    """Pearson correlation of close mid-prices between two instruments over the last n_bars."""
    def _closes(prices: list) -> np.ndarray:
        out = []
        for c in prices[-n_bars:]:
            try:
                out.append((c["closePrice"]["bid"] + c["closePrice"]["ask"]) / 2)
            except (KeyError, TypeError):
                continue
        return np.array(out, dtype=float)

    ca = _closes(prices_a)
    cb = _closes(prices_b)
    n  = min(len(ca), len(cb))
    if n < 10:
        return 0.0
    ca, cb = ca[-n:], cb[-n:]
    if ca.std() == 0 or cb.std() == 0:
        return 0.0
    return float(np.corrcoef(ca, cb)[0, 1])


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


def get_signal(
    df: pd.DataFrame,
    mid: float,
    sentiment: float,
    rsi_os: float = RSI_OS,
    rsi_ob: float = RSI_OB,
) -> Signal:
    """Signal generation on 5-min bars.  rsi_os/rsi_ob may be overridden by RL.
    BB_STD and STOP_MULT are intentionally fixed to prevent RL overfitting."""
    if len(df) < 30:
        return Signal("NONE", mid, 0, 0, 0, 0, f"need ≥30 bars, have {len(df)}")

    closes = df["close"]
    rsi_s               = _rsi(closes, 14)
    upper_bb, _, lower_bb = _bollinger(closes, 20, BB_STD)
    atr_s               = _atr(df["high"], df["low"], closes, 14)

    curr_rsi   = rsi_s.iloc[-1]
    curr_close = closes.iloc[-1]
    curr_lower = lower_bb.iloc[-1]
    curr_upper = upper_bb.iloc[-1]
    curr_atr   = atr_s.iloc[-1]

    if any(pd.isna(v) for v in [curr_rsi, curr_lower, curr_upper, curr_atr]) or curr_atr == 0:
        return Signal("NONE", mid, 0, 0, 0, curr_rsi or 0, "indicators not ready")

    stop_d = curr_atr * STOP_MULT
    tp_d   = stop_d * TP_RR

    # Require both RSI extreme AND price at/near the band (no "or" — stricter entry)
    near_lower = curr_close <= curr_lower * 1.0003
    near_upper = curr_close >= curr_upper * 0.9997

    if curr_rsi <= rsi_os and near_lower:
        if sentiment <= -0.65:
            return Signal("NONE", mid, 0, 0, 0, curr_rsi,
                          f"BUY suppressed: bearish sentiment={sentiment:.2f}")
        return Signal("BUY", mid, mid - stop_d, mid + tp_d, curr_atr, curr_rsi,
                      f"BUY RSI={curr_rsi:.1f}≤{rsi_os} & nearLowerBB | sent={sentiment:+.2f}")

    if curr_rsi >= rsi_ob and near_upper:
        if sentiment >= 0.65:
            return Signal("NONE", mid, 0, 0, 0, curr_rsi,
                          f"SELL suppressed: bullish sentiment={sentiment:.2f}")
        return Signal("SELL", mid, mid + stop_d, mid - tp_d, curr_atr, curr_rsi,
                      f"SELL RSI={curr_rsi:.1f}≥{rsi_ob} & nearUpperBB | sent={sentiment:+.2f}")

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
                    mkt = client.get_market_info(epic)
                    cur = _mid(mkt) if mkt else pos["entry"]
                    pnl = client.estimate_pnl(epic, pos["direction"], pos["entry"], cur, pos["size"], market=mkt)
                    why = "extern_profit_close" if pnl >= 0 else "sl_hit"
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
        # Pre-fetch price data for correlation check (avoids duplicate API calls)
        price_cache: dict[str, list] = {}
        for _epic in active_instruments:
            _p = client.get_prices(_epic, resolution=CANDLE_RESOLUTION, count=CANDLE_COUNT)
            if _p:
                price_cache[_epic] = _p.get("prices", [])

        # Correlation check: if EURUSD & GBPUSD are highly correlated (> CORR_THRESHOLD)
        # halve the risk on any second simultaneous position to avoid double exposure.
        corr_risk_factor: dict[str, float] = {e: 1.0 for e in active_instruments}
        if len(active_instruments) == 2 and all(e in price_cache for e in active_instruments):
            rho = _correlation(price_cache[active_instruments[0]], price_cache[active_instruments[1]])
            if abs(rho) > CORR_THRESHOLD:
                logger.info(
                    "Correlation ρ=%.2f > %.2f – halving risk on second instrument",
                    rho, CORR_THRESHOLD,
                )
                # First instrument in list takes full risk; second takes 50%
                corr_risk_factor[active_instruments[1]] = 0.5

        for epic in active_instruments:

            # ── Time stop ─────────────────────────────────────────────────
            if epic in tracked:
                pos = tracked[epic]
                mkt = client.get_market_info(epic)
                cur = _mid(mkt) if mkt else pos["entry"]
                previous_stop = pos["stop"]
                pos["stop"], pos["best_price"] = update_trailing_stop(
                    pos["direction"],
                    pos["entry"],
                    pos.get("initial_stop", pos["stop"]),
                    pos["stop"],
                    pos.get("best_price", pos["entry"]),
                    cur,
                    arm_r=TRAIL_ARM_R,
                    trail_gap_r=TRAIL_GAP_R,
                )
                if abs(pos["stop"] - previous_stop) > 1e-9:
                    logger.info(
                        "%s: trailing stop moved %.5f -> %.5f | price=%.5f",
                        epic, previous_stop, pos["stop"], cur,
                    )

                stop_hit = (
                    (pos["direction"] == "BUY" and cur <= pos["stop"])
                    or (pos["direction"] == "SELL" and cur >= pos["stop"])
                )
                if stop_hit:
                    close_reason = "trailing_stop_hit" if abs(pos["stop"] - pos.get("initial_stop", pos["stop"])) > 1e-9 else "sl_hit"
                    pnl = client.estimate_pnl(epic, pos["direction"], pos["entry"], cur, pos["size"], market=mkt)
                    logger.info("%s: %s | pnl≈%+.2f EUR", epic, close_reason, pnl)
                    if not dry and not pos.get("dry_run"):
                        client.close_position(pos["deal_id"])
                    sprint_pnl += pnl
                    if "rl_state" in pos:
                        rl.update(pos["rl_state"], pos["rl_action"], pnl, pos.get("rl_risk", 1.0))
                    risk.record_trade_result(pnl)
                    state_writer.append_trade(
                        epic, pos["direction"], pos["entry"], cur, pos["size"], pnl, close_reason)
                    tracked.pop(epic)
                    continue

                age_mins = (time.time() - pos["opened_at"]) / 60
                if age_mins >= TIME_STOP_MINS:
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
                    logger.debug("%s: position open %.1f min | stop=%.5f best=%.5f",
                                 epic, age_mins, pos["stop"], pos.get("best_price", pos["entry"]))
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
                logger.info("%s: spread %.1f pips > %.1f max, skipping", epic, sprd_pips, config.MAX_SPREAD_PIPS)
                continue

            # Use cached prices (already fetched for correlation check above)
            raw_prices = price_cache.get(epic)
            if not raw_prices:
                logger.warning("%s: no price data", epic)
                continue

            df   = build_df(raw_prices)
            sent = sentiments.get(epic, 0.0)

            # ── RL: select RSI thresholds only (BB/stop fixed to prevent overfitting) ─
            rl_params, rl_action, rl_state = rl.select(
                epic, now_utc.hour, sent, 0.003,
                consecutive_losses=risk.consecutive_losses,
            )
            sig = get_signal(df, mid, sent, rsi_os=float(rl_params[0]), rsi_ob=float(rl_params[1]))

            if sig.direction == "NONE":
                logger.info("%s: no signal | %s", epic, sig.reason)
                continue

            # Size: base €1.50 risk, halved when EURUSD/GBPUSD correlation is high.
            base_risk_eur = min(risk.budget * 0.005, 1.50)
            risk_eur = base_risk_eur * corr_risk_factor.get(epic, 1.0)
            if corr_risk_factor.get(epic, 1.0) < 1.0:
                logger.info("%s: risk halved to €%.2f (correlation cap)", epic, risk_eur)
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
                # Cap FX unit size so margin per position stays within ~€90
                # (account_balance / n_instruments / leverage * safety)
                # e.g. 537 / 2 / 30 * 0.7 / 1.35 ≈ 4,640; use 2,000 hard cap
                max_size = 2_000.0
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
                "▶ %s %s | entry=%.5f | SL=%.5f | TSL=armed@+%.1fR gap=%.1fR | size=%.5f | %s",
                epic, sig.direction, sig.entry_price,
                sig.stop_loss, TRAIL_ARM_R, TRAIL_GAP_R, order_size, sig.reason,
            )

            if dry:
                tracked[epic] = {
                    "deal_id":   f"DRY_{epic}_{int(time.time())}",
                    "direction": sig.direction,
                    "entry":     mid,
                    "stop":      sig.stop_loss,
                    "initial_stop": sig.stop_loss,
                    "tp":        sig.take_profit,
                    "best_price": mid,
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
                    epic, sig.direction, order_size, sig.stop_loss,
                )
                if status in (200, 201) and result:
                    deal_ref = result.get("dealReference", "unknown")
                    logger.info("%s: order received by broker | dealRef=%s", epic, deal_ref)
                    time.sleep(2)
                    # ── Confirm the deal was actually executed ─────────────────
                    confirm = client.confirm_deal(deal_ref)
                    if confirm:
                        deal_status = confirm.get("dealStatus", "UNKNOWN")
                        reason      = confirm.get("reason", "")
                        deal_id_confirmed = confirm.get("dealId", deal_ref)
                        if deal_status == "ACCEPTED":
                            logger.warning(
                                "%s: deal ACCEPTED ✓ | dealId=%s | dealRef=%s",
                                epic, deal_id_confirmed, deal_ref,
                            )
                        else:
                            logger.error(
                                "%s: deal REJECTED by broker | status=%s | reason=%s | dealRef=%s",
                                epic, deal_status, reason, deal_ref,
                            )
                            continue  # skip tracking – order not live
                    else:
                        logger.warning("%s: could not confirm deal, assuming accepted | dealRef=%s", epic, deal_ref)
                        deal_id_confirmed = deal_ref
                    # Fall back to scanning positions list for dealId
                    pos_data = client.get_positions()
                    deal_id  = deal_id_confirmed
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
                        "initial_stop": sig.stop_loss,
                        "tp":        sig.take_profit,
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
                            epic, sig.direction, retry_size, sig.stop_loss,
                        )
                        if retry_status in (200, 201) and retry_result:
                            deal_ref = retry_result.get("dealReference", "unknown")
                            logger.info("%s: order submitted on retry | dealRef=%s", epic, deal_ref)
                            tracked[epic] = {
                                "deal_id":   deal_ref,
                                "direction": sig.direction,
                                "entry":     mid,
                                "stop":      sig.stop_loss,
                                "initial_stop": sig.stop_loss,
                                "tp":        sig.take_profit,
                                "best_price": mid,
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
