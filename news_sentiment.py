"""
news_sentiment.py – Aggregates public news & sentiment to fine-tune signals.

Sources:
  1. Yahoo Finance RSS      – recent headlines per instrument ticker
  2. Reddit r/Forex JSON    – community posts via public JSON API
  3. Capital.com clientsentiment endpoint – proprietary long/short ratios
  4. X.com / Twitter        – real-time social signal
  5. NewsAPI                – professional financial headlines
  6. Finnhub economic calendar – macro surprise signal
  7. VIX implied volatility  – risk-on/risk-off market internals

Returns {epic: score} where score ∈ [-1.0, +1.0]
  +1.0 = strongly bullish   |   -1.0 = strongly bearish   |   0.0 = neutral
Results are cached for CACHE_TTL seconds to avoid hammering APIs.
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import Optional

import requests

import config

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
CACHE_TTL = 600  # 10 minutes – reduces Capital.com /clientsentiment request rate

_http = requests.Session()
_http.headers.update({
    "User-Agent": "Mozilla/5.0 TradingBot/1.0 (+https://github.com/research)",
    "Accept": "application/rss+xml, application/xml, text/xml, application/json, */*",
})


# ── Text scoring helpers ──────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    return text.lower()


def _score_text(text: str) -> float:
    lower = _normalize_text(text)
    bull = sum(1 for w in BULLISH_WORDS if w in lower)
    bear = sum(1 for w in BEARISH_WORDS if w in lower)
    total = bull + bear
    if total < 2:
        return 0.0
    return max(-1.0, min(1.0, (bull - bear) / total))


def _is_relevant(text: str, keywords: list[str]) -> bool:
    lower = _normalize_text(text)
    return any(k.lower() in lower for k in keywords)


def _keywords_query(keywords: list[str], max_terms: int = 8) -> str:
    parts = []
    for keyword in keywords[:max_terms]:
        kw = keyword.strip()
        if not kw:
            continue
        if " " in kw:
            parts.append(f'"{kw}"')
        else:
            parts.append(kw)
    return " OR ".join(parts)


def _numeric(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


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
        resp = _http.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        root = ET.fromstring(resp.text)
        items = root.findall(".//item")
        scores = []
        for item in items[:20]:
            title = item.findtext("title") or ""
            desc = item.findtext("description") or ""
            text = f"{title} {desc}"
            if _is_relevant(text, keywords):
                score = _score_text(text)
                if score != 0.0:
                    scores.append(score)
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
            resp = _http.get(url, timeout=10, headers={"User-Agent": _http.headers["User-Agent"]})
            if resp.status_code != 200:
                continue
            data = resp.json()
            posts = data.get("data", {}).get("children", [])
            for p in posts:
                pd_data = p.get("data", {})
                text = f"{pd_data.get('title', '')} {pd_data.get('selftext', '')}"
                if _is_relevant(text, keywords):
                    score = _score_text(text)
                    if score != 0.0:
                        scores.append(score)
        except Exception as exc:
            logger.debug("Reddit fetch failed for r/%s %s: %s", sub, epic, exc)
            continue
    if not scores:
        return None
    score = sum(scores) / len(scores)
    logger.debug("%s Reddit: %d relevant posts → score=%.2f", epic, len(scores), score)
    return score


def _fetch_twitter(epic: str, keywords: list[str]) -> Optional[float]:
    bearer = config.TWITTER_BEARER_TOKEN
    if not bearer:
        return None
    query = _keywords_query(keywords)
    if not query:
        return None
    query = f"({query}) lang:en -is:retweet -is:reply"
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": query,
        "max_results": 15,
        "tweet.fields": "lang,public_metrics,created_at",
    }
    headers = {"Authorization": f"Bearer {bearer}"}
    try:
        resp = _http.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code != 200:
            logger.debug("Twitter fetch failed for %s: %s", epic, resp.status_code)
            return None
        payload = resp.json()
        tweets = payload.get("data", [])
        if not tweets:
            return None
        scores = []
        weights = []
        for tweet in tweets:
            text = tweet.get("text", "")
            if not _is_relevant(text, keywords):
                continue
            score = _score_text(text)
            if abs(score) < 0.08:
                continue
            metrics = tweet.get("public_metrics", {})
            weight = 1.0 + min(int(metrics.get("retweet_count", 0)), 5) * 0.1
            scores.append(score * weight)
            weights.append(weight)
        if not scores:
            return None
        score = sum(scores) / sum(weights)
        logger.debug("%s Twitter: %d relevant tweets → score=%.2f", epic, len(scores), score)
        return score
    except Exception as exc:
        logger.debug("Twitter fetch failed for %s: %s", epic, exc)
        return None


def _fetch_newsapi(epic: str, keywords: list[str]) -> Optional[float]:
    key = config.NEWSAPI_KEY
    if not key:
        return None
    query = _keywords_query(keywords)
    if not query:
        return None
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": config.NEWSAPI_PAGE_SIZE,
        "sortBy": "relevancy",
    }
    headers = {"X-Api-Key": key}
    try:
        resp = _http.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code != 200:
            logger.debug("NewsAPI fetch failed for %s: %s", epic, resp.status_code)
            return None
        data = resp.json()
        articles = data.get("articles", [])
        scores = []
        for art in articles:
            title = art.get("title", "")
            desc = art.get("description", "")
            text = f"{title} {desc}"
            if _is_relevant(text, keywords):
                score = _score_text(text)
                if score != 0.0:
                    scores.append(score)
        if not scores:
            return None
        score = sum(scores) / len(scores)
        logger.debug("%s NewsAPI: %d relevant articles → score=%.2f", epic, len(scores), score)
        return score
    except Exception as exc:
        logger.debug("NewsAPI fetch failed for %s: %s", epic, exc)
        return None


def _map_economic_surprise(epic: str, currency: str, surprise: float) -> float:
    base = epic[:3].upper()
    quote = epic[3:].upper()
    if epic in ("EURUSD", "GBPUSD"):
        if currency == base:
            return surprise
        if currency == quote:
            return -surprise
    if epic == "USDJPY":
        if currency == "USD":
            return surprise
        if currency == "JPY":
            return -surprise
    if epic in {"BTCUSD", "ETHUSD", "US500", "DAX", "OIL_CRUDE"}:
        if currency == "USD":
            return -surprise
    if epic == "GOLD":
        if currency == "USD":
            return -surprise
    if quote == "USD":
        return -surprise
    if base == "USD":
        return surprise
    return 0.0


def _fetch_finnhub_economic(epic: str, keywords: list[str]) -> Optional[float]:
    key = config.FINNHUB_API_KEY
    if not key:
        return None
    today = date.today()
    window = config.SENTIMENT_CALENDAR_DAYS
    from_date = today - timedelta(days=window)
    to_date = today + timedelta(days=window)
    url = "https://finnhub.io/api/v1/calendar/economic"
    params = {
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "token": key,
    }
    try:
        resp = _http.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            logger.debug("Finnhub calendar fetch failed for %s: %s", epic, resp.status_code)
            return None
        payload = resp.json()
        events = payload.get("economicCalendar") or payload.get("data") or []
        if not events:
            return None
        scores = []
        for event in events:
            currency = str(event.get("currency", "")).upper()
            if not currency:
                continue
            actual = _numeric(event.get("actual"))
            forecast = _numeric(event.get("forecast"))
            if actual is None or forecast is None:
                continue
            surprise = actual - forecast
            if surprise == 0:
                continue
            impact = str(event.get("impact", "medium")).lower()
            weight = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(impact, 0.5)
            direction = _map_economic_surprise(epic, currency, 1.0 if surprise > 0 else -1.0)
            if direction == 0.0:
                continue
            scores.append(direction * weight)
        if not scores:
            return None
        score = sum(scores) / len(scores)
        score = max(-1.0, min(1.0, score))
        logger.debug("%s Finnhub econ: %d events → score=%.2f", epic, len(scores), score)
        return score
    except Exception as exc:
        logger.debug("Finnhub econ fetch failed for %s: %s", epic, exc)
        return None


def _fetch_vix(epic: str) -> Optional[float]:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
    params = {"interval": "1d", "range": "7d"}
    try:
        resp = _http.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        payload = resp.json()
        results = payload.get("chart", {}).get("result")
        if not results:
            return None
        indicators = results[0].get("indicators", {}).get("quote", [])
        if not indicators:
            return None
        closes = indicators[0].get("close", [])
        prices = [p for p in closes if isinstance(p, (int, float))]
        if len(prices) < 2:
            return None
        pct_change = (prices[-1] - prices[-2]) / max(prices[-2], 1e-9)
        if abs(pct_change) < 0.03:
            return None
        mapping = {
            "EURUSD": -1.0,
            "GBPUSD": -1.0,
            "USDJPY": 1.0,
            "US500": -1.0,
            "DAX": -1.0,
            "BTCUSD": -1.0,
            "ETHUSD": -1.0,
            "GOLD": 1.0,
        }
        factor = mapping.get(epic, -1.0)
        score = max(-1.0, min(1.0, factor * pct_change * 2.0))
        logger.debug("%s VIX: pct_change=%.3f → score=%.2f", epic, pct_change, score)
        return score
    except Exception as exc:
        logger.debug("VIX fetch failed for %s: %s", epic, exc)
        return None


def _fetch_capital_sentiment(epic: str, capital_client) -> Optional[float]:
    """Use Capital.com's own long/short client sentiment endpoint."""
    try:
        resp = capital_client._get(f"/clientsentiment/{epic}")
        if not resp or "longPositionPercentage" not in resp:
            return None
        long_pct  = resp["longPositionPercentage"]
        short_pct = resp["shortPositionPercentage"]
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

_SOURCE_WEIGHTS: dict[str, float] = {
    "capital": 2.0,
    "newsapi": 1.4,
    "twitter": 1.0,
    "reddit": 0.6,
    "economic": 1.1,
    "vix": 0.9,
}

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
        source_scores.append(("reddit", r, _SOURCE_WEIGHTS["reddit"]))

    # 3. Capital.com proprietary sentiment (weight 2.0)
    if capital_client is not None:
        c = _fetch_capital_sentiment(epic, capital_client)
        if c is not None:
            source_scores.append(("capital", c, _SOURCE_WEIGHTS["capital"]))

    # 4. X.com / Twitter sentiment
    t = _fetch_twitter(epic, keywords)
    if t is not None:
        source_scores.append(("twitter", t, _SOURCE_WEIGHTS["twitter"]))

    # 5. NewsAPI headlines
    n = _fetch_newsapi(epic, keywords)
    if n is not None:
        source_scores.append(("newsapi", n, _SOURCE_WEIGHTS["newsapi"]))

    # 6. Macro surprises from Finnhub economic calendar
    e = _fetch_finnhub_economic(epic, keywords)
    if e is not None:
        source_scores.append(("economic", e, _SOURCE_WEIGHTS["economic"]))

    # 7. VIX implied volatility risk signal
    v = _fetch_vix(epic)
    if v is not None:
        source_scores.append(("vix", v, _SOURCE_WEIGHTS["vix"]))

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
    """Fetch and return sentiment scores for all given epics.

    A 1-second sleep is inserted between each epic to prevent burst traffic
    against the Capital.com /clientsentiment endpoint (limit: 1 req/s).
    """
    results: dict[str, float] = {}
    for i, epic in enumerate(epics):
        if i > 0:
            time.sleep(1)   # stagger – never fire two requests back-to-back
        results[epic] = get_sentiment(epic, capital_client)
    return results
