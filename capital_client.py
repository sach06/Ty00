"""
Capital.com REST API client.

Handles:
  - session creation and automatic token refresh
  - all HTTP verbs with retry-on-401
  - market info, price history, account info, positions
"""
import time
import logging
import math
from typing import Optional

import requests

import config

logger = logging.getLogger(__name__)


class CapitalClient:
    def __init__(self):
        self._http = requests.Session()
        self._http.headers.update({"Content-Type": "application/json"})
        self._cst: Optional[str] = None
        self._security_token: Optional[str] = None
        self._token_ts: float = 0.0           # epoch time of last successful auth
        self._last_auth_attempt: float = 0.0  # epoch time of last auth attempt (incl. failures)

    # ── Authentication ─────────────────────────────────────────────────────────

    def authenticate(self, _retries: int = 4) -> bool:
        """Create a new Capital.com session; populates CST and X-SECURITY-TOKEN.

        Retries up to ``_retries`` times with exponential back-off when a 429
        (Too Many Requests) response is received.
        Capital.com /session limit: 1 req/s – a 2s guard is enforced here.
        """
        if not config.API_KEY or not config.IDENTIFIER or not config.API_PASSWORD:
            logger.error(
                "Missing API credentials. Set CAPITAL_API_KEY, CAPITAL_IDENTIFIER, "
                "and CAPITAL_API_PASSWORD in your .env file."
            )
            return False
        # Enforce minimum 2s between ANY two auth requests (API hard limit is 1 req/s)
        since_last = time.time() - self._last_auth_attempt
        if since_last < 2.0:
            time.sleep(2.0 - since_last)
        self._last_auth_attempt = time.time()
        for attempt in range(1, _retries + 1):
            try:
                resp = self._http.post(
                    f"{config.BASE_URL}/session",
                    json={
                        "identifier": config.IDENTIFIER,
                        "password": config.API_PASSWORD,
                        "encryptedPassword": False,
                    },
                    headers={"X-CAP-API-KEY": config.API_KEY},
                    timeout=15,
                )
                if resp.status_code == 200:
                    self._cst = resp.headers.get("CST")
                    self._security_token = resp.headers.get("X-SECURITY-TOKEN")
                    self._token_ts = time.time()
                    self._last_auth_attempt = self._token_ts
                    logger.info("Session authenticated with Capital.com  [demo=%s]", config.USE_DEMO)
                    return True
                elif resp.status_code == 429:
                    if not self._handle_rate_limit(attempt - 1, max_retries=_retries):
                        break
                    self._last_auth_attempt = time.time()
                else:
                    logger.error("Authentication failed: %s %s", resp.status_code, resp.text[:300])
                    # Bump token_ts so _ensure_session backs off for 60s instead of retrying immediately
                    self._token_ts = time.time() - config.SESSION_REFRESH_SECS + 60
                    return False
            except requests.RequestException as exc:
                logger.error("Authentication request error: %s", exc)
                self._token_ts = time.time() - config.SESSION_REFRESH_SECS + 60
                return False
        logger.error("Authentication failed after %d retries (rate-limited)", _retries)
        self._token_ts = time.time() - config.SESSION_REFRESH_SECS + 60
        return False

    def _ensure_session(self):
        """Re-authenticate only when the session token is near expiry.

        Also enforces a 60s minimum gap between consecutive auth attempts so a
        string of API errors cannot cause a tight re-auth loop.
        """
        token_expired = time.time() - self._token_ts > config.SESSION_REFRESH_SECS
        recently_attempted = time.time() - self._last_auth_attempt < 60
        if (token_expired or not self._cst) and not recently_attempted:
            self.authenticate()

    def _auth_headers(self) -> dict:
        return {
            "X-CAP-API-KEY": config.API_KEY,
            "CST": self._cst or "",
            "X-SECURITY-TOKEN": self._security_token or "",
        }

    # ── HTTP helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _handle_rate_limit(retry_count: int, max_retries: int = 4) -> bool:
        """Progressive backoff on 429 responses: 15s, 30s, 45s, 60s …

        Returns True to continue retrying, False when max_retries is exhausted.
        """
        if retry_count >= max_retries:
            logger.error("Max retries (%d) reached – halting request.", max_retries)
            return False
        wait_time = 15 * (retry_count + 1)
        logger.warning(
            "Rate-limited (429) – waiting %ds before retry %d/%d",
            wait_time, retry_count + 1, max_retries,
        )
        time.sleep(wait_time)
        return True

    def _get(self, path: str, params: Optional[dict] = None, _max_retries: int = 4) -> Optional[dict]:
        self._ensure_session()
        rate_retries = 0
        for attempt in range(_max_retries):
            try:
                resp = self._http.get(
                    f"{config.BASE_URL}{path}",
                    params=params,
                    headers=self._auth_headers(),
                    timeout=15,
                )
                if resp.status_code == 401 and attempt == 0:
                    self.authenticate()
                    continue
                if resp.status_code == 429:
                    if rate_retries >= 3:
                        logger.error("GET %s – too many 429 responses, giving up", path)
                        return None
                    self._handle_rate_limit(rate_retries)
                    rate_retries += 1
                    continue
                if resp.status_code == 200:
                    return resp.json()
                logger.warning("GET %s → %s", path, resp.status_code)
                return None
            except requests.RequestException as exc:
                logger.error("GET %s error: %s", path, exc)
                return None
        return None

    def _post(self, path: str, body: dict, _max_retries: int = 4) -> tuple[Optional[dict], int]:
        self._ensure_session()
        rate_retries = 0
        for attempt in range(_max_retries):
            try:
                resp = self._http.post(
                    f"{config.BASE_URL}{path}",
                    json=body,
                    headers=self._auth_headers(),
                    timeout=15,
                )
                if resp.status_code == 401 and attempt == 0:
                    self.authenticate()
                    continue
                if resp.status_code == 429:
                    if rate_retries >= 3:
                        logger.error("POST %s – too many 429 responses, giving up", path)
                        return None, 429
                    self._handle_rate_limit(rate_retries)
                    rate_retries += 1
                    continue
                try:
                    return resp.json(), resp.status_code
                except ValueError:
                    return None, resp.status_code
            except requests.RequestException as exc:
                logger.error("POST %s error: %s", path, exc)
                return None, 0
        return None, 0

    def _delete(self, path: str, _max_retries: int = 4) -> tuple[Optional[dict], int]:
        self._ensure_session()
        rate_retries = 0
        for attempt in range(_max_retries):
            try:
                resp = self._http.delete(
                    f"{config.BASE_URL}{path}",
                    headers=self._auth_headers(),
                    timeout=15,
                )
                if resp.status_code == 401 and attempt == 0:
                    self.authenticate()
                    continue
                if resp.status_code == 429:
                    if rate_retries >= 3:
                        logger.error("DELETE %s – too many 429 responses, giving up", path)
                        return None, 429
                    self._handle_rate_limit(rate_retries)
                    rate_retries += 1
                    continue
                try:
                    return resp.json(), resp.status_code
                except ValueError:
                    return None, resp.status_code
            except requests.RequestException as exc:
                logger.error("DELETE %s error: %s", path, exc)
                return None, 0
        return None, 0

    # ── Public API calls ───────────────────────────────────────────────────────

    def get_accounts(self) -> Optional[dict]:
        return self._get("/accounts")

    def get_market_info(self, epic: str) -> Optional[dict]:
        return self._get(f"/markets/{epic}")

    def size_uses_units(self, epic: str, market: Optional[dict] = None) -> bool:
        """Return True when broker size is already in underlying units, not standard lots."""
        data = market or self.get_market_info(epic) or {}
        rules = data.get("dealingRules", {})
        min_size = float(rules.get("minDealSize", {}).get("value") or 0.0)
        increment = float(rules.get("minSizeIncrement", {}).get("value") or 0.0)
        return len(epic) == 6 and min_size >= 100 and increment >= 100

    def get_size_constraints(self, epic: str, market: Optional[dict] = None) -> tuple[float, float]:
        """Return (min_size, size_increment) from market dealing rules."""
        data = market or self.get_market_info(epic) or {}
        rules = data.get("dealingRules", {})
        min_size = float(rules.get("minDealSize", {}).get("value") or 0.0)
        increment = float(rules.get("minSizeIncrement", {}).get("value") or 0.0)
        return min_size, increment

    def normalize_order_size(
        self,
        epic: str,
        size: float,
        market: Optional[dict] = None,
    ) -> tuple[float, float, float]:
        """
        Apply Capital dealing-size constraints.
        Returns (normalized_size, min_size, increment).
        """
        min_size, increment = self.get_size_constraints(epic, market=market)
        adjusted = float(size)

        if min_size > 0 and adjusted < min_size:
            adjusted = min_size

        if increment > 0:
            steps = int((adjusted + increment - 1e-12) / increment)
            adjusted = steps * increment

        return round(adjusted, 8), min_size, increment

    def size_to_exposure_units(
        self,
        epic: str,
        size: float,
        market: Optional[dict] = None,
    ) -> float:
        """Convert broker order size into approximate underlying units for risk/PnL math."""
        return float(size) * self._size_exposure_multiplier(epic, market=market)

    def _size_exposure_multiplier(
        self,
        epic: str,
        market: Optional[dict] = None,
    ) -> float:
        """Return exposure multiplier per 1 broker size unit.

        - FX in unit mode: 1 size = 1 unit
        - FX in lot mode: 1 size = 100,000 units
        - Non-FX CFDs (commodities/indices/shares/crypto): treat size as contract count (1×)
        """
        # Known crypto base currencies – must NOT be treated as FX lots
        _CRYPTO_BASES = {
            "BTC", "ETH", "SOL", "XRP", "ADA", "DOT", "LTC",
            "BCH", "BNB", "DOGE", "MATIC", "LINK", "AVAX", "UNI",
        }
        data = market or self.get_market_info(epic) or {}
        if self.size_uses_units(epic, market=data):
            return 1.0

        instrument_type = str(data.get("instrumentType") or "").upper()
        # Explicit broker type takes priority
        if instrument_type in {"CURRENCIES", "FOREX"}:
            return 100_000.0
        if instrument_type in {"CRYPTOCURRENCIES", "CRYPTO"}:
            return 1.0
        # Heuristic: 6-char all-alpha epic – but exclude crypto base currencies
        base = epic[:3].upper()
        looks_like_fx = len(epic) == 6 and epic.isalpha() and base not in _CRYPTO_BASES
        if looks_like_fx:
            return 100_000.0
        return 1.0

    def calculate_order_size_for_risk(
        self,
        epic: str,
        risk_eur: float,
        stop_distance: float,
        market: Optional[dict] = None,
    ) -> tuple[float, float, float]:
        """Return the largest allowed broker size that stays within target risk."""
        if risk_eur <= 0 or stop_distance <= 0:
            return 0.0, 0.0, 0.0

        multiplier = self._size_exposure_multiplier(epic, market=market)
        raw_size = risk_eur / (stop_distance * max(multiplier, 1e-9))
        min_size, increment = self.get_size_constraints(epic, market=market)
        if min_size > 0 and raw_size < min_size - 1e-9:
            return 0.0, min_size, increment

        if increment > 0:
            base_size = min_size if min_size > 0 else 0.0
            steps = math.floor((raw_size - base_size) / increment + 1e-12)
            raw_size = base_size + max(steps, 0) * increment

        if raw_size <= 0 or (min_size > 0 and raw_size < min_size - 1e-9):
            return 0.0, min_size, increment
        return round(raw_size, 8), min_size, increment

    def estimate_pnl(
        self,
        epic: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        size: float,
        market: Optional[dict] = None,
    ) -> float:
        """Approximate trade PnL from broker size using the correct exposure model."""
        direction_sign = 1 if direction == "BUY" else -1
        exposure_units = self.size_to_exposure_units(epic, size, market=market)
        return direction_sign * (exit_price - entry_price) * exposure_units

    def get_prices(
        self,
        epic: str,
        resolution: str = "MINUTE_15",
        count: int = 60,
    ) -> Optional[dict]:
        return self._get(
            f"/prices/{epic}",
            params={"resolution": resolution, "max": count},
        )

    def get_positions(self) -> Optional[dict]:
        return self._get("/positions")

    def confirm_deal(self, deal_reference: str) -> Optional[dict]:
        """GET /confirms/{dealReference} – returns the deal confirmation status from Capital.com."""
        return self._get(f"/confirms/{deal_reference}")

    def open_position(
        self,
        epic: str,
        direction: str,
        size: float,
        stop_level: float,
        profit_level: Optional[float] = None,
    ) -> tuple[Optional[dict], int]:
        body = {
            "direction": direction,        # "BUY" or "SELL"
            "epic": epic,
            "size": size,
            "guaranteedStop": False,
            "stopLevel": round(stop_level, 5),
        }
        if profit_level is not None:
            body["profitLevel"] = round(profit_level, 5)
        logger.info("POST /positions body: %s", body)
        result, status = self._post("/positions", body)
        logger.info("POST /positions response: status=%s body=%s", status, result)

        # Capital.com may reject the stop if it's below their minimum distance.
        # The error code carries the required minimum, e.g.:
        #   "error.invalid.stoploss.minvalue: 84.4639"
        # Retry once with that minimum stop level.
        if status == 400 and isinstance(result, dict):
            err = str(result.get("errorCode", ""))
            if "stoploss.minvalue" in err:
                try:
                    min_stop = float(err.split(":")[-1].strip())
                    logger.warning(
                        "%s: stop %s below minimum %s – retrying with minimum",
                        epic, stop_level, min_stop,
                    )
                    body["stopLevel"] = round(min_stop, 5)
                    result, status = self._post("/positions", body)
                    logger.info("POST /positions retry response: status=%s body=%s", status, result)
                except (ValueError, IndexError):
                    pass

        return result, status

    def close_position(self, deal_id: str) -> tuple[Optional[dict], int]:
        return self._delete(f"/positions/{deal_id}")

    def close_all_positions(self):
        """Best-effort close all open positions (used on shutdown / kill switch)."""
        data = self.get_positions()
        if not data:
            return
        for entry in data.get("positions", []):
            deal_id = entry.get("position", {}).get("dealId")
            if deal_id:
                result, status = self.close_position(deal_id)
                logger.info("Closed position %s → status %s", deal_id, status)
