"""
crypto_strategy.py – Indicator engine for crypto perpetual mean-reversion.

Strategy: price reverts to EMA(20) after touching EMA ± 2.5×ATR(14) bands
while RSI(14) confirms an extreme (< 35 long / > 65 short).

All calculations are vectorised pandas/numpy — no external TA libraries needed.
"""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default parameters (can be overridden per call) ──────────────────────────
EMA_PERIOD   = 20
ATR_PERIOD   = 14
RSI_PERIOD   = 14
ATR_MULT     = 2.5   # envelope width: EMA ± ATR_MULT × ATR
RSI_OS       = 35    # oversold threshold (long entry)
RSI_OB       = 65    # overbought threshold (short entry)
STOP_ATR     = 2.0   # stop-loss distance in ATR units
TP_ATR       = 1.5   # take-profit distance in ATR units (reversion target)
MAX_ATR_MULT = 3.0   # volatility filter: skip if ATR > MAX_ATR_MULT × rolling mean


@dataclass
class CryptoSignal:
    direction:   str    # "BUY" | "SELL" | "NONE"
    entry_price: float
    stop_loss:   float
    take_profit: float
    atr:         float
    rsi:         float
    ema:         float
    reason:      str


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder-smoothed ATR (same as TradingView default)."""
    hi, lo, cl = df["high"], df["low"], df["close"]
    prev_cl = cl.shift(1)
    tr = pd.concat([
        hi - lo,
        (hi - prev_cl).abs(),
        (lo - prev_cl).abs(),
    ], axis=1).max(axis=1)
    # Wilder smoothing = RMA = EMA with alpha = 1/period
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder RSI (same as TradingView)."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_indicators(df: pd.DataFrame,
                         ema_period: int = EMA_PERIOD,
                         atr_period: int = ATR_PERIOD,
                         rsi_period: int = RSI_PERIOD) -> pd.DataFrame:
    """
    Append EMA, ATR, RSI, upper/lower envelope columns to df in-place.
    Returns the same DataFrame so callers can chain.
    Minimum required columns: open, high, low, close, volume.
    """
    df = df.copy()
    df["ema"]   = _ema(df["close"], ema_period)
    df["atr"]   = _atr(df, atr_period)
    df["rsi"]   = _rsi(df["close"], rsi_period)
    df["upper"] = df["ema"] + ATR_MULT * df["atr"]
    df["lower"] = df["ema"] - ATR_MULT * df["atr"]
    return df


def _volatility_ok(df: pd.DataFrame, lookback: int = 48) -> tuple[bool, str]:
    """
    Returns (True, '') when current ATR is within normal bounds.
    Rejects if ATR > MAX_ATR_MULT × rolling mean of ATR over `lookback` bars.
    Typical crypto cascade / liquidation spike protection.
    """
    if len(df) < lookback + 1:
        return True, ""
    rolling_avg = df["atr"].iloc[-lookback:-1].mean()
    curr_atr    = df["atr"].iloc[-1]
    if rolling_avg > 0 and curr_atr > MAX_ATR_MULT * rolling_avg:
        return False, (
            f"volatility spike: ATR={curr_atr:.4f} > "
            f"{MAX_ATR_MULT}×avg({rolling_avg:.4f})"
        )
    return True, ""


def fee_filter(entry_price: float,
               atr: float,
               taker_rate: float,
               max_cost_ratio: float = 0.15) -> tuple[bool, str]:
    """
    Returns (pass, reason).
    Reject if round-trip taker fees > max_cost_ratio × expected gross profit.

    Round-trip cost  = 2 × entry_price × taker_rate
    Expected profit  = TP_ATR × ATR  (reversion to midpoint)
    """
    round_trip_cost   = 2 * entry_price * taker_rate
    expected_profit   = TP_ATR * atr
    if expected_profit <= 0:
        return False, "expected profit is zero"
    ratio = round_trip_cost / expected_profit
    if ratio > max_cost_ratio:
        return False, (
            f"fee ratio {ratio:.1%} > {max_cost_ratio:.0%} limit "
            f"(fees={round_trip_cost:.4f}, expected={expected_profit:.4f})"
        )
    return True, ""


def determine_signal(df: pd.DataFrame,
                     mid: float,
                     taker_rate: float = 0.0005,
                     rsi_os: float = RSI_OS,
                     rsi_ob: float = RSI_OB,
                     stop_atr: float = STOP_ATR,
                     tp_atr: float = TP_ATR) -> CryptoSignal:
    """
    Main signal generator.  Returns a CryptoSignal with direction NONE if no
    tradeable condition is met.

    Entry rules:
        LONG  : mid ≤ lower_envelope AND RSI < rsi_os
        SHORT : mid ≥ upper_envelope AND RSI > rsi_ob

    Filtered by:
        1. Minimum bar requirement (need at least max(EMA_PERIOD, ATR_PERIOD, RSI_PERIOD) bars)
        2. Volatility spike filter
        3. Fee efficiency filter
    """
    min_bars = max(EMA_PERIOD, ATR_PERIOD, RSI_PERIOD) + 5
    if len(df) < min_bars or "ema" not in df.columns:
        return CryptoSignal("NONE", mid, 0, 0, 0, 0, 0,
                            f"insufficient bars: {len(df)} < {min_bars}")

    row = df.iloc[-1]
    ema   = float(row["ema"])
    atr   = float(row["atr"])
    rsi   = float(row["rsi"])
    upper = float(row["upper"])
    lower = float(row["lower"])

    if atr <= 0:
        return CryptoSignal("NONE", mid, 0, 0, 0, rsi, ema, "ATR is zero")

    # ── Volatility filter ─────────────────────────────────────────────────
    vol_ok, vol_reason = _volatility_ok(df)
    if not vol_ok:
        return CryptoSignal("NONE", mid, 0, 0, atr, rsi, ema, vol_reason)

    # ── Directional signal ────────────────────────────────────────────────
    direction = "NONE"
    reason    = f"RSI={rsi:.1f} price={mid:.4f} ema={ema:.4f} – no extreme"

    if mid <= lower and rsi < rsi_os:
        direction = "BUY"
        stop_loss   = mid - stop_atr * atr
        take_profit = mid + tp_atr * atr    # partial target: midpoint / EMA
        reason = (
            f"BUY: price={mid:.4f} ≤ lower={lower:.4f} "
            f"& RSI={rsi:.1f} < {rsi_os}"
        )
    elif mid >= upper and rsi > rsi_ob:
        direction = "SELL"
        stop_loss   = mid + stop_atr * atr
        take_profit = mid - tp_atr * atr
        reason = (
            f"SELL: price={mid:.4f} ≥ upper={upper:.4f} "
            f"& RSI={rsi:.1f} > {rsi_ob}"
        )
    else:
        return CryptoSignal("NONE", mid, 0, 0, atr, rsi, ema, reason)

    # ── Fee filter ────────────────────────────────────────────────────────
    fee_ok, fee_reason = fee_filter(mid, atr, taker_rate)
    if not fee_ok:
        return CryptoSignal("NONE", mid, 0, 0, atr, rsi, ema,
                            f"fee filter: {fee_reason}")

    return CryptoSignal(direction, mid, stop_loss, take_profit, atr, rsi, ema, reason)


def calculate_position_size(equity: float,
                             risk_pct: float,
                             entry: float,
                             stop_loss: float,
                             step_size: float,
                             min_qty: float,
                             leverage: float = 10.0) -> float:
    """
    Returns quantity in base currency (e.g. BTC) or 0 if below minimum.

    risk_pct  : fraction of equity to risk, e.g. 0.005 = 0.5%
    step_size : exchange minimum increment (e.g. 0.001 for BTCUSDT)
    min_qty   : exchange minimum order quantity
    leverage  : applied leverage (used to verify margin requirement)

    Size = risk_amount / stop_distance
    Strict: if size < min_qty → return 0 (never round up).
    """
    stop_dist = abs(entry - stop_loss)
    if stop_dist <= 0 or entry <= 0:
        return 0.0

    risk_amount = equity * risk_pct
    raw_size    = risk_amount / stop_dist   # in base units

    # Truncate to step_size (never round up)
    if step_size > 0:
        raw_size = float(int(raw_size / step_size)) * step_size

    if raw_size < min_qty:
        logger.info(
            "Size %.6f < min_qty %.6f – skipping (never round up)",
            raw_size, min_qty,
        )
        return 0.0

    # Margin sanity: notional = size × entry / leverage must be ≤ 20% of equity
    notional = raw_size * entry / leverage
    if notional > equity * 0.20:
        capped = float(int((equity * 0.20 * leverage / entry) / step_size)) * step_size
        logger.info(
            "Size capped %.6f → %.6f (notional guard 20%% equity)",
            raw_size, capped,
        )
        raw_size = capped

    return max(raw_size, 0.0)


def update_trailing_stop(direction: str,
                         entry: float,
                         initial_stop: float,
                         current_stop: float,
                         best_price: float,
                         current_price: float,
                         arm_atr: float,
                         atr: float) -> tuple[float, float]:
    """
    Monotonically trails stop loss once price has moved arm_atr × ATR in our favour.
    Returns (new_stop, new_best_price).
    """
    risk_dist = abs(entry - initial_stop)
    if risk_dist <= 0 or atr <= 0:
        return current_stop, best_price

    trail_gap = arm_atr * atr   # trail by 0.5 ATR gap once armed

    if direction == "BUY":
        best = max(best_price, current_price)
        if (best - entry) >= arm_atr * atr:
            new_stop = best - trail_gap
            return max(new_stop, current_stop), best
        return current_stop, best

    else:  # SELL
        best = min(best_price, current_price)
        if (entry - best) >= arm_atr * atr:
            new_stop = best + trail_gap
            return min(new_stop, current_stop), best
        return current_stop, best
