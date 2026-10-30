"""
fx_agent_committee.py – Multi-agent FX consensus filter.

Architecture
------------
  Flow Agent   (60 % weight) — evaluates spread/liquidity from market info
  Macro Agent  (40 % weight) — evaluates directional sentiment
  CIO Node                   — combines scores, gates trades at 0.35 threshold

Design constraints
------------------
* Does NOT make extra Capital.com API calls per tick.
  - Spread is read from the market_info dict already fetched by the caller.
  - Sentiment is read from the pre-cached sentiments dict (polled every 10 min).
* No module-level globals / shared session state.
* All logging uses the standard logger; no f-string log calls (PEP 282).
"""
import logging

logger = logging.getLogger(__name__)

# ── Tunable constants ──────────────────────────────────────────────────────────
MAX_SPREAD_PCT     = 0.08   # spread threshold for the Flow Agent (%)
MIN_COMPOSITE      = 0.35   # CIO "bouncer" threshold

WEIGHT_LIQUIDITY   = 0.60
WEIGHT_SENTIMENT   = 0.40


# ── Internal helpers ───────────────────────────────────────────────────────────

def _spread_pct(market_info: dict) -> float:
    """Return bid/offer spread as a percentage of mid price."""
    bid = float(market_info.get("bid") or market_info.get("offer") or 0.0)
    offer = float(market_info.get("offer") or market_info.get("ask") or bid)
    mid = (bid + offer) / 2.0 if bid and offer else 0.0
    if mid <= 0:
        return 0.0
    return (offer - bid) / mid * 100.0


def _liquidity_score(spread_pct: float) -> float:
    """Flow Agent: linearly map spread onto [0, 1]; returns None if spread too wide."""
    if spread_pct > MAX_SPREAD_PCT:
        return None   # signal rejection
    return max(0.0, 1.0 - (spread_pct / MAX_SPREAD_PCT))


def _sentiment_score(raw_sentiment: float) -> float:
    """Macro Agent: map raw sentiment [-1, 1] onto [0, 1].

    We use the absolute value so both strong bullish and strong bearish
    conviction count as high-quality signal; neutral (~0) counts as low quality.
    """
    return min(1.0, max(0.0, abs(float(raw_sentiment))))


# ── Public API ─────────────────────────────────────────────────────────────────

def evaluate(
    epic: str,
    market_info: dict,
    sentiment: float,
) -> dict:
    """Evaluate a single epic through the Flow → Macro → CIO pipeline.

    Parameters
    ----------
    epic        : instrument code, e.g. 'EURUSD'
    market_info : dict returned by ``CapitalClient.get_market_info(epic)``
    sentiment   : pre-cached sentiment score in [-1, 1] (from news_sentiment module)

    Returns
    -------
    dict with keys:
        action   : 'execute' | 'skip'
        reason   : why (on skip)
        score    : composite score (on execute)
        liquidity: liquidity sub-score
        sentiment: sentiment sub-score
    """
    # ── Flow Agent ─────────────────────────────────────────────────────────────
    spread = _spread_pct(market_info)
    liq = _liquidity_score(spread)

    if liq is None:
        logger.info(
            "[FlowAgent] %s: spread %.4f%% > %.4f%% max – rejected",
            epic, spread, MAX_SPREAD_PCT,
        )
        return {"action": "skip", "reason": "high_spread", "spread_pct": spread}

    # ── Macro Agent ────────────────────────────────────────────────────────────
    sent = _sentiment_score(sentiment)

    # ── CIO Node ───────────────────────────────────────────────────────────────
    composite = WEIGHT_LIQUIDITY * liq + WEIGHT_SENTIMENT * sent
    logger.info(
        "[CIO] %s | spread=%.4f%% liq=%.2f sent=%.2f composite=%.2f (%s)",
        epic, spread, liq, sent, composite,
        "PASS" if composite >= MIN_COMPOSITE else "FAIL",
    )

    if composite >= MIN_COMPOSITE:
        return {
            "action":    "execute",
            "score":     round(composite, 4),
            "liquidity": round(liq,       4),
            "sentiment": round(sent,      4),
            "spread_pct": spread,
        }
    return {
        "action": "skip",
        "reason": "consensus_too_low",
        "score":  round(composite, 4),
    }


def run_fx_agent_committee(
    epics: list,
    market_info_map: dict,
    sentiments: dict,
) -> dict[str, dict]:
    """Run the committee for all requested epics.

    Parameters
    ----------
    epics           : list of epic codes to evaluate
    market_info_map : {epic: market_info_dict} – caller supplies pre-fetched data
    sentiments      : {epic: float} – pre-cached sentiment scores

    Returns
    -------
    {epic: decision_dict}  – see ``evaluate()`` for decision structure
    """
    decisions = {}
    for epic in epics:
        mkt = market_info_map.get(epic) or {}
        if not mkt:
            logger.debug("[Committee] %s: no market info – skipping", epic)
            decisions[epic] = {"action": "skip", "reason": "no_market_info"}
            continue
        raw_sent = sentiments.get(epic, 0.0)
        decisions[epic] = evaluate(epic, mkt, raw_sent)
    return decisions
