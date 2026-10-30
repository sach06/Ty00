"""
crypto_sprint.py – 24/7 mean-reversion sprint for crypto CFDs on Capital.com.

Instruments : BTCUSD, ETHUSD, SOLUSD (Capital.com crypto CFDs)
Timeframe   : 5-minute candles, 100-bar lookback
Strategy    : EMA(20) ± 2.5×ATR(14) envelope + RSI(14) confirmation
              (see crypto_strategy.py for full indicator/signal logic)

Uses the same CapitalClient and credentials as sprint.py / main.py —
no additional API keys or third-party exchange required.

Safety rails
------------
  • RISK_EUR_PER_TRADE  €1.00 per trade (hard cap, configurable below)
  • TIME_STOP           45 minutes max hold per position
  • PENALTY_BOX         5 min per epic after any broker rejection
  • FEE_FILTER          skip trade if spread cost > 15% of expected profit
  • VOLATILITY_FILTER   skip if ATR > 3× 48-bar rolling average (in crypto_strategy)
  • SPREAD_GUARD        skip if spread > MAX_SPREAD_PCT of mid-price
  • MAX_DRAWDOWN        halt if session P&L drops below MAX_LOSS_EUR

Usage
-----
    python crypto_sprint.py                    # default epics
    python crypto_sprint.py --epics BTCUSD ETHUSD
    python crypto_sprint.py --minutes 120      # run for 2 hours then exit
    python crypto_sprint.py --dry              # paper mode (no real orders)
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

import config
import bot_logger
import state_writer
import runtime_settings
from capital_client import CapitalClient
from crypto_strategy import (
    calculate_indicators,
    determine_signal,
    update_trailing_stop,
)

bot_logger.setup()
logger = logging.getLogger("crypto_sprint")

# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_EPICS    = ["BTCUSD", "ETHUSD", "SOLUSD"]
CANDLE_RESOLUTION = "MINUTE_5"
CANDLE_COUNT      = 120       # ~10 hours of 5-min bars
POLL_SECS         = 60        # one poll per candle – halves API request load

RISK_EUR_PER_TRADE = 1.00     # EUR risked per trade (stop-distance based)
MAX_LOSS_EUR       = -10.00   # halt session if P&L drops below this
TIME_STOP_MINS     = 45       # force-close after 45 minutes
TRAIL_ARM_R        = 1.0      # arm trailing stop after +1R profit
TRAIL_GAP_R        = 1.0      # trail 1R behind best price
MAX_SPREAD_PCT     = 0.60     # skip if spread > 0.60% of mid (raised to allow SOLUSD)
REJECT_COOLDOWN    = 300      # 5-minute penalty box after broker REJECTED


# ── Candle → DataFrame ─────────────────────────────────────────────────────────

def _build_df(candles: list) -> pd.DataFrame:
    """Convert Capital.com price list to OHLC DataFrame."""
    rows = []
    for c in candles:
        try:
            rows.append({
                "open":   (c["openPrice"]["bid"]   + c["openPrice"]["ask"])   / 2,
                "high":   (c["highPrice"]["bid"]    + c["highPrice"]["ask"])   / 2,
                "low":    (c["lowPrice"]["bid"]     + c["lowPrice"]["ask"])    / 2,
                "close":  (c["closePrice"]["bid"]   + c["closePrice"]["ask"]) / 2,
                "volume": c.get("lastTradedVolume", 0),
            })
        except (KeyError, TypeError):
            continue
    return pd.DataFrame(rows)


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


# ── Main loop ──────────────────────────────────────────────────────────────────

def sprint(epics: list[str], minutes: int, dry: bool):
    end_ts = time.time() + minutes * 60 if minutes > 0 else float("inf")
    logger.warning(
        "═══ CRYPTO SPRINT START ═══ epics=%s | minutes=%s | dry=%s",
        epics, minutes if minutes > 0 else "∞", dry,
    )

    client = CapitalClient()
    if not client.authenticate():
        logger.critical("Capital.com authentication failed – check .env")
        sys.exit(1)

    balance     = _fetch_balance(client, config.INITIAL_BALANCE)
    tracked:     dict[str, dict]  = {}   # epic -> open position
    penalty_box: dict[str, float] = {}   # epic -> unlock timestamp
    session_pnl = 0.0
    tick_n      = 0

    logger.info("Balance: €%.2f | risk=€%.2f/trade | time_stop=%dmin",
                balance, RISK_EUR_PER_TRADE, TIME_STOP_MINS)

    try:
        while time.time() < end_ts:
            tick_n += 1
            now_utc   = datetime.now(timezone.utc)
            remaining = int(end_ts - time.time()) if minutes > 0 else -1
            rem_str   = f"{remaining//60:02d}:{remaining%60:02d}" if remaining >= 0 else "∞"
            logger.info("── crypto tick %d | ⏱ %s remaining | P&L %+.2f EUR ──",
                        tick_n, rem_str, session_pnl)

            client._ensure_session()
            balance = _fetch_balance(client, balance)

            # ── Session halt guard ─────────────────────────────────────────
            if session_pnl <= MAX_LOSS_EUR:
                logger.warning("CRYPTO SPRINT HALT: session P&L %.2f ≤ %.2f limit",
                               session_pnl, MAX_LOSS_EUR)
                break

            # ── Sync live positions (detect external SL/TP hits) ──────────
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
                        pnl = client.estimate_pnl(epic, pos["direction"],
                                                  pos["entry"], cur, pos["size"], market=mkt)
                        session_pnl += pnl
                        logger.info("%s: closed externally | pnl≈%+.2f EUR", epic, pnl)
                        state_writer.append_trade(epic, pos["direction"],
                                                  pos["entry"], cur, pos["size"], pnl,
                                                  "extern_close")
                        tracked.pop(epic)

            # ── Per-epic logic ─────────────────────────────────────────────
            for epic in epics:
                # 2s stagger between epics – prevents burst traffic on the API
                time.sleep(2)

                # Penalty box
                if epic in penalty_box:
                    if time.time() < penalty_box[epic]:
                        rem_p = int(penalty_box[epic] - time.time())
                        logger.info("%s: penalty box – skipping %ds", epic, rem_p)
                        continue
                    del penalty_box[epic]
                    logger.info("%s: penalty box expired – re-enabled", epic)

                # ── Manage existing position ───────────────────────────────
                if epic in tracked:
                    pos = tracked[epic]
                    mkt = client.get_market_info(epic)
                    cur = _mid(mkt) if mkt else pos["entry"]

                    # Trailing stop
                    prev_stop = pos["stop"]
                    pos["stop"], pos["best_price"] = update_trailing_stop(
                        pos["direction"],
                        pos["entry"],
                        pos["initial_stop"],
                        pos["stop"],
                        pos["best_price"],
                        cur,
                        arm_atr=TRAIL_ARM_R,
                        atr=pos["atr"],
                    )
                    if abs(pos["stop"] - prev_stop) > 1e-8:
                        logger.info("%s: trailing stop %.4f → %.4f | price=%.4f",
                                    epic, prev_stop, pos["stop"], cur)

                    stop_hit = (
                        (pos["direction"] == "BUY"  and cur <= pos["stop"]) or
                        (pos["direction"] == "SELL" and cur >= pos["stop"])
                    )
                    age_mins = (time.time() - pos["opened_at"]) / 60
                    time_hit = age_mins >= TIME_STOP_MINS

                    if stop_hit or time_hit:
                        why = "sl_hit" if stop_hit else f"time_stop ({age_mins:.0f}min)"
                        pnl = client.estimate_pnl(epic, pos["direction"],
                                                  pos["entry"], cur, pos["size"], market=mkt)
                        logger.info("%s: %s | pnl≈%+.2f EUR", epic, why, pnl)
                        if not dry and not pos.get("dry_run"):
                            client.close_position(pos["deal_id"])
                        session_pnl += pnl
                        state_writer.append_trade(epic, pos["direction"],
                                                  pos["entry"], cur, pos["size"], pnl, why)
                        tracked.pop(epic)
                    else:
                        logger.debug("%s: holding %.1fmin | stop=%.4f best=%.4f mid=%.4f",
                                     epic, age_mins, pos["stop"], pos["best_price"], cur)
                    continue   # no new entry while in a position

                # ── Fetch candles & signals ────────────────────────────────
                mkt = client.get_market_info(epic)
                if not mkt:
                    logger.warning("%s: no market info", epic)
                    continue

                mid  = _mid(mkt)
                sprd = _spread(mkt)
                if mid <= 0:
                    continue

                sprd_pct = sprd / mid * 100
                if sprd_pct > MAX_SPREAD_PCT:
                    logger.info("%s: spread %.3f%% > %.2f%% – skipping",
                                epic, sprd_pct, MAX_SPREAD_PCT)
                    continue

                raw = client.get_prices(epic, resolution=CANDLE_RESOLUTION, count=CANDLE_COUNT)
                if not raw:
                    logger.warning("%s: no price data", epic)
                    continue

                df = _build_df(raw.get("prices", []))
                if df.empty or len(df) < 30:
                    logger.warning("%s: too few candles (%d)", epic, len(df))
                    continue

                df  = calculate_indicators(df)
                sig = determine_signal(df, mid, taker_rate=0.0)  # CFD: no taker fee gate

                if sig.direction == "NONE":
                    logger.info("%s: no signal | %s", epic, sig.reason)
                    continue

                # ── Size (risk-based via Capital.com constraints) ──────────
                stop_dist = abs(sig.entry_price - sig.stop_loss)
                if stop_dist <= 0:
                    continue

                order_size, min_size, size_inc = client.calculate_order_size_for_risk(
                    epic, RISK_EUR_PER_TRADE, stop_dist, market=mkt,
                )

                if order_size <= 0:
                    continue

                if min_size > 0 and order_size < min_size:
                    logger.info("%s: size %.6f < min %.6f – skipping",
                                epic, order_size, min_size)
                    continue

                logger.warning(
                    "▶ %s %s | entry=%.4f | SL=%.4f | TP=%.4f | size=%.6f | %s",
                    epic, sig.direction,
                    sig.entry_price, sig.stop_loss, sig.take_profit,
                    order_size, sig.reason,
                )

                if dry:
                    tracked[epic] = {
                        "deal_id":     f"DRY_{epic}_{int(time.time())}",
                        "direction":   sig.direction,
                        "entry":       mid,
                        "initial_stop": sig.stop_loss,
                        "stop":        sig.stop_loss,
                        "tp":          sig.take_profit,
                        "best_price":  mid,
                        "atr":         sig.atr,
                        "size":        order_size,
                        "opened_at":   time.time(),
                        "dry_run":     True,
                    }
                    logger.info("%s: [DRY] position recorded", epic)
                    continue

                # ── Live order ─────────────────────────────────────────────
                result, status = client.open_position(
                    epic, sig.direction, order_size, sig.stop_loss,
                )

                if status in (200, 201) and result:
                    deal_ref = result.get("dealReference", "unknown")
                    logger.info("%s: order received | dealRef=%s", epic, deal_ref)
                    time.sleep(2)

                    confirm = client.confirm_deal(deal_ref)
                    if confirm:
                        deal_status = confirm.get("dealStatus", "UNKNOWN")
                        reason      = confirm.get("reason", "") or confirm.get("errorCode", "")
                        error_code  = confirm.get("errorCode", "")
                        deal_id     = confirm.get("dealId", deal_ref)
                        if deal_status != "ACCEPTED":
                            logger.error(
                                "%s: REJECTED | status=%s reason=%s errorCode=%s",
                                epic, deal_status, reason, error_code,
                            )
                            penalty_box[epic] = time.time() + REJECT_COOLDOWN
                            logger.warning("%s: penalty box %ds", epic, REJECT_COOLDOWN)
                            continue
                        logger.warning("%s: deal ACCEPTED ✓ | dealId=%s", epic, deal_id)
                    else:
                        logger.warning("%s: confirm unavailable – assuming accepted", epic)
                        deal_id = deal_ref

                    tracked[epic] = {
                        "deal_id":      deal_id,
                        "direction":    sig.direction,
                        "entry":        mid,
                        "initial_stop": sig.stop_loss,
                        "stop":         sig.stop_loss,
                        "tp":           sig.take_profit,
                        "best_price":   mid,
                        "atr":          sig.atr,
                        "size":         order_size,
                        "opened_at":    time.time(),
                        "dry_run":      False,
                    }
                else:
                    code = (result or {}).get("errorCode", "") if isinstance(result, dict) else ""
                    logger.error("%s: order failed status=%s errorCode=%s",
                                 epic, status, code)
                    penalty_box[epic] = time.time() + REJECT_COOLDOWN

            time.sleep(POLL_SECS)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt – closing all positions…")

    finally:
        # Graceful shutdown
        for epic, pos in list(tracked.items()):
            if not dry and not pos.get("dry_run"):
                try:
                    client.close_position(pos["deal_id"])
                    logger.info("%s: force-closed at shutdown", epic)
                except Exception as exc:
                    logger.error("%s: shutdown close failed: %s", epic, exc)

        logger.warning(
            "═══ CRYPTO SPRINT END ═══ session P&L: %+.2f EUR",
            session_pnl,
        )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crypto CFD mean-reversion sprint on Capital.com"
    )
    parser.add_argument(
        "--epics", nargs="+", default=DEFAULT_EPICS,
        help=f"Capital.com epics to trade (default: {' '.join(DEFAULT_EPICS)})",
    )
    parser.add_argument(
        "--minutes", type=int, default=0,
        help="Run for N minutes then exit (default: 0 = run forever)",
    )
    parser.add_argument(
        "--dry", action="store_true",
        help="Paper mode – no real orders submitted",
    )
    args = parser.parse_args()
    sprint(args.epics, args.minutes, args.dry)
