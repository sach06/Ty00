"""
ny_session_agent.py – US Tech 100 (Nasdaq 100) trend-following agent.

Operates ONLY during NYSE regular session hours (09:30–16:00 ET, Mon–Fri).
Uses an EMA crossover + RSI momentum + N-bar breakout confirmation strategy,
distinct from the mean-reversion logic used for FX instruments.

Signal logic
------------
  BUY  : EMA(20) > EMA(50)  AND  RSI(14) > 55  AND  close breaks above 20-bar high
  SELL : EMA(20) < EMA(50)  AND  RSI(14) < 45  AND  close breaks below 20-bar low
  Stop : entry ± 1.5 × ATR(14)
  TP   : 2.5 × stop distance  (2.5R)

All indicator helpers are re-used from strategy.py; no duplicate code.
"""

import logging
from datetime import datetime, time as _time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config
from strategy import Signal, _rsi, _atr

logger = logging.getLogger(__name__)

# ── Constants (overridden at import time by config.US_TECH_CONFIG) ─────────────
_CFG            = config.US_TECH_CONFIG
_NY_TZ          = ZoneInfo("America/New_York")
_SESSION_START  = _time(*_CFG["ny_session_start"])   # 09:30 ET
_SESSION_END    = _time(*_CFG["ny_session_end"])      # 16:00 ET

EMA_FAST        = 20
EMA_SLOW        = 50
BREAKOUT_BARS   = 20        # price must clear the N-bar high / low of preceding bars
MIN_CANDLES     = EMA_SLOW + BREAKOUT_BARS + 5
STOP_ATR_MULT   = _CFG["stop_atr_mult"]   # 1.5
TP_RATIO        = _CFG["tp_ratio"]        # 2.5
MAX_SPREAD_PTS  = _CFG["max_spread_pts"]  # 2.5 index points


# ── Session gate ───────────────────────────────────────────────────────────────

def is_ny_regular_session() -> bool:
    """Return True during NYSE regular session Mon–Fri 09:30–16:00 ET."""
    now_ny = datetime.now(_NY_TZ)
    t = now_ny.time()
    in_session = (
        now_ny.weekday() < 5
        and _SESSION_START <= t < _SESSION_END
    )
    logger.info(
        "[US-Tech] session gate | NY=%s %s | active=%s",
        now_ny.strftime("%a %H:%M"),
        now_ny.tzname(),   # shows EDT or EST so DST is visible
        in_session,
    )
    return in_session


# ── Indicator helper ───────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ── Signal generation ──────────────────────────────────────────────────────────

def get_us100_signal(prices_data: dict, mid_price: float) -> Signal:
    """
    Derive a trend-breakout Signal for US100 from OHLC candle data.

    Parameters
    ----------
    prices_data : dict   API response from CapitalClient.get_prices()
    mid_price   : float  Current live mid-price

    Returns
    -------
    Signal with direction "BUY", "SELL", or "NONE"
    """
    candles = prices_data.get("prices", [])
    if len(candles) < MIN_CANDLES:
        return Signal("NONE", mid_price, 0, 0, 0, 0,
                      f"US100: need ≥{MIN_CANDLES} candles, have {len(candles)}")

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

    if len(rows) < MIN_CANDLES:
        return Signal("NONE", mid_price, 0, 0, 0, 0, "US100: malformed candle data")

    df     = pd.DataFrame(rows)
    closes = df["close"]

    ema_fast_s = _ema(closes, EMA_FAST)
    ema_slow_s = _ema(closes, EMA_SLOW)
    rsi_s      = _rsi(closes, 14)
    atr_s      = _atr(df["high"], df["low"], closes, 14)

    ema_fast   = ema_fast_s.iloc[-1]
    ema_slow   = ema_slow_s.iloc[-1]
    curr_rsi   = rsi_s.iloc[-1]
    curr_atr   = atr_s.iloc[-1]
    curr_close = closes.iloc[-1]

    if any(pd.isna(v) for v in [ema_fast, ema_slow, curr_rsi, curr_atr]) \
            or not np.isfinite(curr_atr) or curr_atr <= 0:
        return Signal("NONE", mid_price, 0, 0, 0, curr_rsi or 0,
                      "US100: indicators not ready")

    # N-bar high/low excludes the current (forming) bar
    lookback = df["high"].iloc[-(BREAKOUT_BARS + 1):-1]
    lookback_low = df["low"].iloc[-(BREAKOUT_BARS + 1):-1]
    recent_high = float(lookback.max())
    recent_low  = float(lookback_low.min())

    stop_dist = curr_atr * STOP_ATR_MULT
    tp_dist   = stop_dist * TP_RATIO

    # ── Bullish breakout ───────────────────────────────────────────────────────
    if ema_fast > ema_slow and curr_rsi > 55 and curr_close > recent_high:
        return Signal(
            "BUY",
            mid_price,
            mid_price - stop_dist,
            mid_price + tp_dist,
            curr_atr,
            curr_rsi,
            (
                f"US100 BUY breakout | "
                f"EMA{EMA_FAST}={ema_fast:.1f} > EMA{EMA_SLOW}={ema_slow:.1f} | "
                f"RSI={curr_rsi:.1f} | close={curr_close:.1f} > {BREAKOUT_BARS}barHigh={recent_high:.1f}"
            ),
        )

    # ── Bearish breakout ───────────────────────────────────────────────────────
    if ema_fast < ema_slow and curr_rsi < 45 and curr_close < recent_low:
        return Signal(
            "SELL",
            mid_price,
            mid_price + stop_dist,
            mid_price - tp_dist,
            curr_atr,
            curr_rsi,
            (
                f"US100 SELL breakout | "
                f"EMA{EMA_FAST}={ema_fast:.1f} < EMA{EMA_SLOW}={ema_slow:.1f} | "
                f"RSI={curr_rsi:.1f} | close={curr_close:.1f} < {BREAKOUT_BARS}barLow={recent_low:.1f}"
            ),
        )

    trend_tag = "up" if ema_fast > ema_slow else "down"
    return Signal(
        "NONE", mid_price, 0, 0, 0, curr_rsi,
        (
            f"US100 no breakout | trend={trend_tag} "
            f"RSI={curr_rsi:.1f} | {BREAKOUT_BARS}barH={recent_high:.1f} "
            f"{BREAKOUT_BARS}barL={recent_low:.1f}"
        ),
    )


# ── Public entry point ─────────────────────────────────────────────────────────

def run_us_tech_agent(client, market_info: dict) -> dict:
    """
    Evaluate a potential US100 entry through session, spread, and signal gates.

    Parameters
    ----------
    client      : CapitalClient instance
    market_info : dict returned by client.get_market_info("US100")

    Returns
    -------
    dict with keys:
        action      : 'execute' | 'skip'
        reason      : why (on skip)
        signal      : Signal instance (on execute)
        mid         : mid price (on execute)
        spread_pts  : observed spread in index points
    """
    logger.info("[US-Tech] evaluating US100")

    if not is_ny_regular_session():
        logger.info(
            "[US-Tech] outside NYSE regular session (%s–%s ET) – skip",
            _SESSION_START.strftime("%H:%M"), _SESSION_END.strftime("%H:%M"),
        )
        return {"action": "skip", "reason": "session_closed"}

    if not market_info:
        logger.warning("[US-Tech] no market info – skip")
        return {"action": "skip", "reason": "no_market_info"}

    snap  = market_info.get("snapshot", {})
    bid   = float(snap.get("bid")   or snap.get("offer") or 0)
    offer = float(snap.get("offer") or snap.get("ask")   or bid)
    mid   = (bid + offer) / 2.0 if bid and offer else 0.0
    spread_pts = offer - bid

    if spread_pts > MAX_SPREAD_PTS:
        logger.info(
            "[US-Tech] spread %.2f pts > %.2f max – skip",
            spread_pts, MAX_SPREAD_PTS,
        )
        return {"action": "skip", "reason": "high_spread", "spread_pts": spread_pts}

    prices = client.get_prices(
        _CFG["epic"],
        resolution=_CFG["candle_resolution"],
        count=_CFG["candle_count"],
    )
    if not prices:
        logger.warning("[US-Tech] could not fetch price history – skip")
        return {"action": "skip", "reason": "no_price_data"}

    signal = get_us100_signal(prices, mid)

    if signal.direction == "NONE":
        logger.info("[US-Tech] no signal: %s", signal.reason)
        return {"action": "skip", "reason": "no_signal", "detail": signal.reason}

    logger.info("[US-Tech] signal approved: %s", signal.reason)
    return {
        "action":     "execute",
        "signal":     signal,
        "mid":        mid,
        "spread_pts": spread_pts,
    }
