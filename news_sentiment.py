"""
news_sentiment.py – Aggregates public news & sentiment to fine-tune signals.

Sources (no paid API keys required):
  1. Yahoo Finance RSS      – recent headlines per instrument ticker
  2. Reddit r/Forex JSON    – community posts via public JSON API
  3. Capital.com clientsentiment endpoint – proprietary long/short ratios

Returns {epic: score} where score ∈ [-1.0, +1.0]
  +1.0 = strongly bullish   |   -1.0 = strongly bearish   |   0.0 = neutral
Results are cached for 5 minutes to avoid hammering free APIs.
"""

import logging
import time
import xml.etree.ElementTree as ET
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Sentiment keyword sets ────────────────────────────────────────────────────

BULLISH_WORDS = {
    "bullish", "rally", "surge", "jump", "gain", "gains", "rise", "rising",
    "strong", "buy", "positive", "boom", "breakout", "momentum", "recover",
    "rebound", "hawkish", "rate hike", "growth", "optimism", "optimistic",
    "upside", "outperform", "beat", "beats", "soar", "soars", "advance",
}
BEARISH_WORDS = {
    "bearish", "drop", "fall", "falls", "decline", "weak", "sell", "negative",
    "crash", "plunge", "downside", "risk-off", "recession", "slowdown",
    "dovish", "rate cut", "rate cuts", "worries", "fear", "concern",
    "uncertainty", "dump", "miss", "misses", "below expectations", "contraction",
}

# ── Instrument → search keywords ─────────────────────────────────────────────

EPIC_KEYWORDS: dict[str, list[str]] = {
    "EURUSD": ["EUR", "euro", "EURUSD", "EUR/USD", "eurozone", "ECB",
               "European Central Bank", "euro area"],
    "GBPUSD": ["GBP", "pound", "sterling", "GBPUSD", "GBP/USD",
               "Bank of England", "BOE", "UK economy", "British"],
    "USDJPY": ["JPY", "yen", "USDJPY", "USD/JPY", "BOJ", "Bank of Japan"],
    "BTCUSD": ["bitcoin", "BTC", "crypto", "BTCUSD", "cryptocurrency"],
    "GOLD":   ["gold", "XAU", "bullion", "precious metal"],
    "US500":  ["S&P", "SPX", "S&P 500", "Wall Street", "US stocks", "Nasdaq"],
    "DAX":    ["DAX", "German stocks", "Germany economy"],
    "OIL_CRUDE": ["crude oil", "WTI", "OPEC", "oil price", "brent"],
}

# Yahoo Finance ticker mapping for RSS
_YAHOO_TICKERS: dict[str, str] = {
    "EURUSD":    "EURUSD=X",
    "GBPUSD":    "GBPUSD=X",
    "USDJPY":    "USDJPY=X",
    "BTCUSD":    "BTC-USD",
    "GOLD":      "GC=F",
    "US500":     "%5EGSPC",
    "DAX":       "%5EGDAXI",
    "OIL_CRUDE": "CL=F",
}

# Cache: epic → (timestamp, score)
_cache: dict[str, tuple[float, float]] = {}
CACHE_TTL = 300  # 5 minutes

_http = requests.Session()
_http.headers.update({
    "User-Agent": "Mozilla/5.0 TradingBot/1.0 (+https://github.com/research)",
    "Accept": "application/rss+xml, application/xml, text/xml, application/json, */*",
})


# ── Text scoring helpers ──────────────────────────────────────────────────────

def _score_text(text: str) -> float:
    """Keyword-based sentiment score in [-1, +1]."""
    lower = text.lower()
    bull = sum(1 for w in BULLISH_WORDS if w in lower)
    bear = sum(1 for w in BEARISH_WORDS if w in lower)
    total = bull + bear
    if total == 0:
        return 0.0
    return max(-1.0, min(1.0, (bull - bear) / max(total, 1)))


def _is_relevant(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return any(k.lower() in lower for k in keywords)


# ── Data source fetchers ──────────────────────────────────────────────────────

def _fetch_yahoo_rss(epic: str, keywords: list[str]) -> Optional[float]:
    """Parse Yahoo Finance RSS headlines for the epic."""
    ticker = _YAHOO_TICKERS.get(epic)
    if not ticker:
        return None
    url = (
        f"https://feeds.finance.yahoo.com/rss/2.0/headline"
        f"?s={ticker}&region=US&lang=en-US"
    )
    try:
        resp = _http.get(url, timeout=8)
        if resp.status_code != 200:
            return None
        root = ET.fromstring(resp.text)
        items = root.findall(".//item")
        scores = []
        for item in items[:20]:
            title = item.findtext("title") or ""
            desc  = item.findtext("description") or ""
            text  = f"{title} {desc}"
            if _is_relevant(text, keywords):
                scores.append(_score_text(text))
        if not scores:
            return None
        score = sum(scores) / len(scores)
        logger.debug("%s Yahoo RSS: %d relevant headlines → score=%.2f", epic, len(scores), score)
        return score
    except Exception as exc:
        logger.debug("Yahoo RSS fetch failed for %s: %s", epic, exc)
        return None


def _fetch_reddit(epic: str, keywords: list[str]) -> Optional[float]:
    """Fetch Reddit r/Forex + r/investing posts via public JSON API."""
    query = keywords[0]
    scores = []
    for sub in ("Forex", "investing", "economics", "StockMarket"):
        url = (
            f"https://www.reddit.com/r/{sub}/search.json"
            f"?q={requests.utils.quote(query)}&sort=new&limit=15&restrict_sr=1"
        )
        try:
            resp = _http.get(url, timeout=8)
            if resp.status_code != 200:
                continue
            data = resp.json()
            posts = data.get("data", {}).get("children", [])
            for p in posts:
                pd_data = p.get("data", {})
                text = f"{pd_data.get('title', '')} {pd_data.get('selftext', '')}"
                if _is_relevant(text, keywords):
                    scores.append(_score_text(text))
        except Exception as exc:
            logger.debug("Reddit fetch failed for r/%s %s: %s", sub, epic, exc)
            continue
    if not scores:
        return None
    score = sum(scores) / len(scores)
    logger.debug("%s Reddit: %d relevant posts → score=%.2f", epic, len(scores), score)
    return score


def _fetch_capital_sentiment(epic: str, capital_client) -> Optional[float]:
    """Use Capital.com's own long/short client sentiment endpoint."""
    try:
        resp = capital_client._get(f"/clientsentiment/{epic}")
        if not resp or "longPositionPercentage" not in resp:
            return None
        long_pct  = resp["longPositionPercentage"]
        short_pct = resp["shortPositionPercentage"]
        # Map: 80% long → +0.60,  50/50 → 0.0,  20% long → -0.60
        score = (long_pct - short_pct) / 100.0
        logger.debug(
            "%s Capital.com sentiment: long=%.1f%% short=%.1f%% → score=%.2f",
            epic, long_pct, short_pct, score,
        )
        return score
    except Exception as exc:
        logger.debug("Capital.com sentiment failed for %s: %s", epic, exc)
        return None


# ── Public interface ──────────────────────────────────────────────────────────

def get_sentiment(epic: str, capital_client=None) -> float:
    """
    Return aggregated sentiment score for *epic* in [-1.0, +1.0].
    Results are cached for CACHE_TTL seconds.
    """
    now = time.time()
    cached = _cache.get(epic)
    if cached and (now - cached[0]) < CACHE_TTL:
        return cached[1]

    keywords = EPIC_KEYWORDS.get(epic, [epic])
    source_scores: list[tuple[str, float, float]] = []  # (name, score, weight)

    # 1. Yahoo Finance RSS (weight 1.0)
    y = _fetch_yahoo_rss(epic, keywords)
    if y is not None:
        source_scores.append(("yahoo", y, 1.0))

    # 2. Reddit community posts (weight 0.6 – noisier)
    r = _fetch_reddit(epic, keywords)
    if r is not None:
        source_scores.append(("reddit", r, 0.6))

    # 3. Capital.com proprietary sentiment (weight 1.5 – most direct)
    if capital_client is not None:
        c = _fetch_capital_sentiment(epic, capital_client)
        if c is not None:
            source_scores.append(("capital", c, 1.5))

    if not source_scores:
        score = 0.0
    else:
        total_w = sum(w for _, _, w in source_scores)
        score   = sum(s * w for _, s, w in source_scores) / total_w
        score   = max(-1.0, min(1.0, score))
        sources = ", ".join(f"{n}={s:+.2f}" for n, s, _ in source_scores)
        logger.info("%s aggregated sentiment=%.2f  [%s]", epic, score, sources)

    _cache[epic] = (now, score)
    return score


def get_all_sentiments(epics: list[str], capital_client=None) -> dict[str, float]:
    """Fetch and return sentiment scores for all given epics."""
    return {epic: get_sentiment(epic, capital_client) for epic in epics}
