"""
seed_rl.py – Bootstrap the RL agent with 10 simulated historical trades.

Fetches recent 15-min candles from Capital.com, walks through the price history
to find valid entry signals, then simulates each trade's outcome (SL hit vs TP
hit) using subsequent candles.  Every closed trade is fed into the RL agent so
it starts with real learned Q-values instead of exploring blind.

Usage
-----
    python seed_rl.py              # default: 10 trades
    python seed_rl.py --trades 20  # seed with 20 trades
    python seed_rl.py --dry        # just print signals, don't save RL state

The script is safe to re-run; it accumulates on top of any existing rl_state.json.
"""

import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd

import bot_logger
import config
from capital_client import CapitalClient
from rl_agent import RLAgent, PARAM_VARIANTS
from strategy import _rsi, _bollinger, _atr, Signal

bot_logger.setup()
logger = logging.getLogger("seed_rl")

# ── Candle fetch ──────────────────────────────────────────────────────────────
CANDLE_COUNT   = 500   # how many 15-min bars to fetch per instrument (≈5 days)
MIN_HOLD_BARS  = 1     # minimum bars to hold before checking SL/TP
MAX_HOLD_BARS  = 16    # 4 h max hold = TIME_STOP_HOURS * 4 bars


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


def simulate_trade_outcome(df: pd.DataFrame, entry_idx: int,
                           direction: str, entry_price: float,
                           stop_loss: float, take_profit: float) -> tuple:
    """
    Walk forward from entry_idx through subsequent candles.
    Returns (exit_price, exit_reason) where exit_reason ∈ {tp_hit, sl_hit, time_stop}.
    """
    for i in range(entry_idx + 1, min(entry_idx + 1 + MAX_HOLD_BARS, len(df))):
        high = df["high"].iloc[i]
        low  = df["low"].iloc[i]

        if direction == "BUY":
            if low <= stop_loss:
                return stop_loss, "sl_hit"
            if high >= take_profit:
                return take_profit, "tp_hit"
        else:  # SELL
            if high >= stop_loss:
                return stop_loss, "sl_hit"
            if low <= take_profit:
                return take_profit, "tp_hit"

    # Time stop: use close of last bar
    exit_price = df["close"].iloc[min(entry_idx + MAX_HOLD_BARS, len(df) - 1)]
    return exit_price, "time_stop"


def scan_signals(df: pd.DataFrame, start: int,
                 rsi_os: float, rsi_ob: float, bb_std: float,
                 stop_m: float, tp_rr: float) -> list:
    """
    Scan df from start index and return list of signal dicts.
    Each dict: {idx, direction, entry, stop, tp, atr, rsi}
    """
    closes = df["close"]
    rsi_s  = _rsi(closes, config.RSI_PERIOD)
    _, _, lower_bb = _bollinger(closes, config.BB_PERIOD, bb_std)
    upper_bb, _, _ = _bollinger(closes, config.BB_PERIOD, bb_std)
    atr_s  = _atr(df["high"], df["low"], closes, config.ATR_PERIOD)

    signals = []
    for i in range(max(config.BB_PERIOD + 5, start), len(df) - MAX_HOLD_BARS - 1):
        rsi    = rsi_s.iloc[i]
        low_bb = lower_bb.iloc[i]
        up_bb  = upper_bb.iloc[i]
        atr    = atr_s.iloc[i]
        close  = closes.iloc[i]

        if any(pd.isna(v) for v in [rsi, low_bb, up_bb, atr]) or atr == 0:
            continue

        stop_d = atr * stop_m
        tp_d   = stop_d * tp_rr

        if rsi < rsi_os and close <= low_bb:
            entry = close
            signals.append({
                "idx":       i,
                "direction": "BUY",
                "entry":     entry,
                "stop":      entry - stop_d,
                "tp":        entry + tp_d,
                "atr":       atr,
                "rsi":       rsi,
            })
        elif rsi > rsi_ob and close >= up_bb:
            entry = close
            signals.append({
                "idx":       i,
                "direction": "SELL",
                "entry":     entry,
                "stop":      entry + stop_d,
                "tp":        entry - tp_d,
                "atr":       atr,
                "rsi":       rsi,
            })

    return signals


def seed(n_trades: int, dry_run: bool):
    logger.info("RL seeding starting | target=%d trades | dry=%s", n_trades, dry_run)

    client = CapitalClient()
    if not client.authenticate():
        logger.critical("Authentication failed – check .env credentials.")
        sys.exit(1)

    rl = RLAgent()
    trades_done = 0

    # Alternate between instruments and RL action variants
    instruments = config.INSTRUMENTS
    instrument_idx = 0

    while trades_done < n_trades:
        epic = instruments[instrument_idx % len(instruments)]
        instrument_idx += 1

        # Fetch extended history
        prices_data = client.get_prices(epic, resolution="MINUTE_15", count=CANDLE_COUNT)
        if not prices_data or not prices_data.get("prices"):
            logger.warning("%s: no price data", epic)
            time.sleep(2)
            continue

        df = build_df(prices_data["prices"])
        if len(df) < config.BB_PERIOD + MAX_HOLD_BARS + 10:
            logger.warning("%s: not enough candles (%d)", epic, len(df))
            continue

        mid_price = df["close"].iloc[-1]

        # Determine RL state for this context (use last bar's indicators)
        atr_last = _atr(df["high"], df["low"], df["close"], config.ATR_PERIOD).iloc[-1]
        atr_pct  = atr_last / mid_price if mid_price > 0 else 0.003

        # Cycle through variants so we seed all of them
        variant_idx = trades_done % len(PARAM_VARIANTS)
        params   = PARAM_VARIANTS[variant_idx]
        rl_state = rl.encode_state(epic, 14, 0.0, atr_pct)  # midday, neutral as baseline
        rl_action = variant_idx

        rsi_os, rsi_ob, bb_std, stop_m, tp_rr = params

        # Find signals in the historical data
        signals = scan_signals(df, start=30,
                               rsi_os=rsi_os, rsi_ob=rsi_ob, bb_std=bb_std,
                               stop_m=stop_m, tp_rr=tp_rr)

        if not signals:
            logger.info("%s: no signals found in %d candles for variant %d – trying next",
                        epic, len(df), variant_idx)
            # Try with relaxed params to find at least one signal
            signals = scan_signals(df, start=30,
                                   rsi_os=35, rsi_ob=65, bb_std=1.5,
                                   stop_m=1.5, tp_rr=2.0)
            if not signals:
                continue

        # Pick the first (oldest) signal so we use well-separated bar data
        sig = signals[0]

        exit_price, exit_reason = simulate_trade_outcome(
            df, sig["idx"], sig["direction"],
            sig["entry"], sig["stop"], sig["tp"],
        )

        direction_sign = 1 if sig["direction"] == "BUY" else -1
        pnl_pips = direction_sign * (exit_price - sig["entry"])
        # Approximate EUR P&L: assume 0.10 lot (10,000 units), 1 pip ≈ 0.0001 * notional / price
        notional   = 10_000
        pnl_eur    = pnl_pips * notional / mid_price if mid_price > 0 else pnl_pips * 10_000
        risk_eur   = abs(sig["entry"] - sig["stop"]) * notional / mid_price if mid_price > 0 else 1.0

        result_emoji = "✓ WIN" if pnl_eur >= 0 else "✗ LOSS"
        logger.info(
            "Simulated trade %2d/%d | %s %-6s | %s | "
            "entry=%.5f exit=%.5f | pnl≈%+.2f EUR | reason=%s | variant=%s",
            trades_done + 1, n_trades, epic, sig["direction"], result_emoji,
            sig["entry"], exit_price, pnl_eur, exit_reason,
            f"[{variant_idx}]({PARAM_VARIANTS[variant_idx]})",
        )

        if not dry_run:
            reward = rl.update(rl_state, rl_action, pnl_eur, risk_eur)
            logger.info("  → RL reward=%.3f | state=%s", reward, rl_state)

        trades_done += 1
        time.sleep(0.3)  # be gentle on the API rate limit

    logger.info(
        "Seeding complete | %d trades processed | RL states=%d | total_updates=%d",
        trades_done, len(rl.Q), rl.total_updates,
    )
    summary = rl.performance_summary()
    logger.info("Best global strategy: %s (Q=%.4f)",
                summary["best_global_action"], summary["best_global_q"])
    if summary["per_state"]:
        logger.info("Top learned state: %s → %s (Q=%.4f)",
                    summary["per_state"][0]["state"],
                    summary["per_state"][0]["best_action"],
                    summary["per_state"][0]["best_q"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed RL agent with simulated historical trades")
    parser.add_argument("--trades", type=int, default=10, help="Number of trades to simulate (default: 10)")
    parser.add_argument("--dry",    action="store_true",  help="Print signals only – do NOT update rl_state.json")
    args = parser.parse_args()

    seed(args.trades, args.dry)
