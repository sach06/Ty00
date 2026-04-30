"""
Mean-reversion strategy: RSI(14) + Bollinger Bands(20, 2σ) on 15-minute candles.

Entry rules
-----------
LONG  : RSI crosses below 30  AND  close touches or breaks below lower BB
SHORT : RSI crosses above 70  AND  close touches or breaks above upper BB

Exit rules (handled by the trading loop, not here)
-----------
- Stop loss   : entry ± 1.5 × ATR(14)
- Trailing SL : armed once price reaches +0.5R, then trails by 0.5R
- Time stop   : 4 hours (hard limit, set in config)
- Initial SL is submitted at trade open; the trading loop manages the trailing stop

All indicators computed with pure pandas/numpy – no extra libraries required.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    direction: str       # "BUY" | "SELL" | "NONE"
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    rsi: float
    reason: str


def update_trailing_stop(
    direction: str,
    entry_price: float,
    initial_stop: float,
    current_stop: float,
    best_price: float,
    current_price: float,
    arm_r: float = 0.5,
    trail_gap_r: float = 0.5,
) -> tuple[float, float]:
    """Return the updated stop and best favorable price for a monotonic trailing stop."""
    risk_distance = abs(entry_price - initial_stop)
    if risk_distance <= 0:
        return current_stop, best_price

    if direction == "BUY":
        best_favorable = max(best_price, current_price)
        profit_distance = best_favorable - entry_price
        if profit_distance < arm_r * risk_distance:
            return current_stop, best_favorable
        trailing_candidate = best_favorable - trail_gap_r * risk_distance
        new_stop = max(current_stop, entry_price, trailing_candidate)
        return new_stop, best_favorable

    if direction == "SELL":
        best_favorable = min(best_price, current_price)
        profit_distance = entry_price - best_favorable
        if profit_distance < arm_r * risk_distance:
            return current_stop, best_favorable
        trailing_candidate = best_favorable + trail_gap_r * risk_distance
        new_stop = min(current_stop, entry_price, trailing_candidate)
        return new_stop, best_favorable

    return current_stop, best_price


# ── Indicator helpers ──────────────────────────────────────────────────────────

def _rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain  = delta.clip(lower=0.0)
    loss  = (-delta).clip(lower=0.0)
    # Wilder smoothing (EWM with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _bollinger(closes: pd.Series, period: int = 20, std: float = 2.0):
    """Return (upper, mid, lower) Bollinger Band series."""
    mid   = closes.rolling(period).mean()
    sigma = closes.rolling(period).std(ddof=0)
    return mid + std * sigma, mid, mid - std * sigma


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


# ── Signal generation ─────────────────────────────────────────────────────────

def generate_signal(
    prices_data: dict,
    mid_price: float,
    sentiment_score: float = 0.0,
    rsi_oversold: Optional[float] = None,
    rsi_overbought: Optional[float] = None,
    bb_std_override: Optional[float] = None,
    stop_atr_mult: Optional[float] = None,
    tp_rr_ratio: Optional[float] = None,
) -> Signal:
    """
    Derive a trading signal from the raw price history returned by
    ``CapitalClient.get_prices()``.

    Parameters
    ----------
    prices_data      : dict   API response from GET /prices/{epic}
    mid_price        : float  Current live mid-price (bid+ask)/2
    sentiment_score  : float  Aggregated news sentiment in [-1, +1].
                              Signals that contradict strong sentiment
                              (|score| >= 0.35) are suppressed.
    rsi_oversold     : float  Override RSI oversold threshold (default: config)
    rsi_overbought   : float  Override RSI overbought threshold (default: config)
    bb_std_override  : float  Override Bollinger Band std multiplier (default: config)
    stop_atr_mult    : float  Override stop-loss ATR multiplier (default: config)
    tp_rr_ratio      : float  Override take-profit R:R ratio (default: config)

    Returns
    -------
    Signal with direction "BUY", "SELL", or "NONE"
    """
    # ── Apply RL overrides (fall back to config defaults when not provided) ──
    _rsi_os  = rsi_oversold   if rsi_oversold   is not None else config.RSI_OVERSOLD
    _rsi_ob  = rsi_overbought if rsi_overbought  is not None else config.RSI_OVERBOUGHT
    _bb_std  = bb_std_override if bb_std_override is not None else config.BB_STD
    _stop_m  = stop_atr_mult  if stop_atr_mult   is not None else config.STOP_ATR_MULT
    _tp_rr   = tp_rr_ratio    if tp_rr_ratio     is not None else config.TP_RR_RATIO

    candles = prices_data.get("prices", [])
    min_required = max(config.BB_PERIOD, config.RSI_PERIOD, config.ATR_PERIOD) + 5

    if len(candles) < min_required:
        return Signal("NONE", mid_price, 0, 0, 0, 0,
                      f"insufficient candles ({len(candles)} < {min_required})")

    # Build OHLC dataframe using mid (average of bid and ask) prices
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

    if len(rows) < min_required:
        return Signal("NONE", mid_price, 0, 0, 0, 0, "malformed candle data")

    df = pd.DataFrame(rows)
    closes = df["close"]

    rsi_series            = _rsi(closes, config.RSI_PERIOD)
    upper_bb, mid_bb, lower_bb = _bollinger(closes, config.BB_PERIOD, _bb_std)
    atr_series            = _atr(df["high"], df["low"], closes, config.ATR_PERIOD)

    curr_rsi   = rsi_series.iloc[-1]
    prev_rsi   = rsi_series.iloc[-2]
    curr_close = closes.iloc[-1]
    curr_lower = lower_bb.iloc[-1]
    curr_upper = upper_bb.iloc[-1]
    curr_atr   = atr_series.iloc[-1]

    if any(pd.isna(v) for v in [curr_rsi, prev_rsi, curr_lower, curr_upper, curr_atr]):
        return Signal("NONE", mid_price, 0, 0, 0, curr_rsi or 0, "indicators not ready")

    if curr_atr == 0:
        return Signal("NONE", mid_price, 0, 0, 0, curr_rsi, "ATR is zero")

    stop_dist = curr_atr * _stop_m
    tp_dist   = stop_dist * _tp_rr

    # ── LONG: RSI oversold + price at/below lower Bollinger Band ──────────────
    long_rsi_trigger  = curr_rsi < _rsi_os
    long_bb_trigger   = curr_close <= curr_lower

    if long_rsi_trigger and long_bb_trigger:
        # Suppress if strong bearish sentiment contradicts the BUY signal
        if sentiment_score <= -0.35:
            return Signal(
                direction   = "NONE",
                entry_price = mid_price,
                stop_loss   = 0,
                take_profit = 0,
                atr         = curr_atr,
                rsi         = curr_rsi,
                reason      = f"BUY signal suppressed: bearish sentiment={sentiment_score:.2f}",
            )
        sentiment_tag = f" | sentiment={sentiment_score:+.2f}" if abs(sentiment_score) >= 0.15 else ""
        entry = mid_price
        sl    = entry - stop_dist
        tp    = entry + tp_dist
        return Signal(
            direction   = "BUY",
            entry_price = entry,
            stop_loss   = sl,
            take_profit = tp,
            atr         = curr_atr,
            rsi         = curr_rsi,
            reason      = f"RSI={curr_rsi:.1f} (oversold), close={curr_close:.5f} ≤ LowerBB={curr_lower:.5f}{sentiment_tag}",
        )

    # ── SHORT: RSI overbought + price at/above upper Bollinger Band ───────────
    short_rsi_trigger = curr_rsi > _rsi_ob
    short_bb_trigger  = curr_close >= curr_upper

    if short_rsi_trigger and short_bb_trigger:
        # Suppress if strong bullish sentiment contradicts the SELL signal
        if sentiment_score >= 0.35:
            return Signal(
                direction   = "NONE",
                entry_price = mid_price,
                stop_loss   = 0,
                take_profit = 0,
                atr         = curr_atr,
                rsi         = curr_rsi,
                reason      = f"SELL signal suppressed: bullish sentiment={sentiment_score:.2f}",
            )
        sentiment_tag = f" | sentiment={sentiment_score:+.2f}" if abs(sentiment_score) >= 0.15 else ""
        entry = mid_price
        sl    = entry + stop_dist
        tp    = entry - tp_dist
        return Signal(
            direction   = "SELL",
            entry_price = entry,
            stop_loss   = sl,
            take_profit = tp,
            atr         = curr_atr,
            rsi         = curr_rsi,
            reason      = f"RSI={curr_rsi:.1f} (overbought), close={curr_close:.5f} ≥ UpperBB={curr_upper:.5f}{sentiment_tag}",
        )

    return Signal(
        direction   = "NONE",
        entry_price = mid_price,
        stop_loss   = 0,
        take_profit = 0,
        atr         = curr_atr,
        rsi         = curr_rsi,
        reason      = f"RSI={curr_rsi:.1f}, LowerBB={curr_lower:.5f}, UpperBB={curr_upper:.5f}",
    )
