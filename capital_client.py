"""
Capital.com REST API client.

Handles:
  - session creation and automatic token refresh
  - all HTTP verbs with retry-on-401
  - market info, price history, account info, positions
"""
import time
import logging
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
        self._token_ts: float = 0.0      # epoch time of last authentication

    # ── Authentication ─────────────────────────────────────────────────────────

    def authenticate(self) -> bool:
        """Create a new Capital.com session; populates CST and X-SECURITY-TOKEN."""
        if not config.API_KEY or not config.IDENTIFIER or not config.API_PASSWORD:
            logger.error(
                "Missing API credentials. Set CAPITAL_API_KEY, CAPITAL_IDENTIFIER, "
                "and CAPITAL_API_PASSWORD in your .env file."
            )
            return False
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
                logger.info("Session authenticated with Capital.com  [demo=%s]", config.USE_DEMO)
                return True
            else:
                logger.error("Authentication failed: %s %s", resp.status_code, resp.text[:300])
                return False
        except requests.RequestException as exc:
            logger.error("Authentication request error: %s", exc)
            return False

    def _ensure_session(self):
        """Re-authenticate if the session token is close to expiry."""
        if time.time() - self._token_ts > config.SESSION_REFRESH_SECS or not self._cst:
            self.authenticate()

    def _auth_headers(self) -> dict:
        return {
            "X-CAP-API-KEY": config.API_KEY,
            "CST": self._cst or "",
            "X-SECURITY-TOKEN": self._security_token or "",
        }

    # ── HTTP helpers ───────────────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        self._ensure_session()
        for attempt in range(2):
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
                if resp.status_code == 200:
                    return resp.json()
                logger.warning("GET %s → %s", path, resp.status_code)
                return None
            except requests.RequestException as exc:
                logger.error("GET %s error: %s", path, exc)
                return None
        return None

    def _post(self, path: str, body: dict) -> tuple[Optional[dict], int]:
        self._ensure_session()
        for attempt in range(2):
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
                try:
                    return resp.json(), resp.status_code
                except ValueError:
                    return None, resp.status_code
            except requests.RequestException as exc:
                logger.error("POST %s error: %s", path, exc)
                return None, 0
        return None, 0

    def _delete(self, path: str) -> tuple[Optional[dict], int]:
        self._ensure_session()
        for attempt in range(2):
            try:
                resp = self._http.delete(
                    f"{config.BASE_URL}{path}",
                    headers=self._auth_headers(),
                    timeout=15,
                )
                if resp.status_code == 401 and attempt == 0:
                    self.authenticate()
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
        if self.size_uses_units(epic, market=market):
            return float(size)
        return float(size) * 100_000

    def calculate_order_size_for_risk(
        self,
        epic: str,
        risk_eur: float,
        stop_distance: float,
        market: Optional[dict] = None,
    ) -> tuple[float, float, float]:
        """Return normalized broker size for a target risk amount."""
        if risk_eur <= 0 or stop_distance <= 0:
            return 0.0, 0.0, 0.0

        if self.size_uses_units(epic, market=market):
            raw_size = risk_eur / stop_distance
        else:
            raw_size = risk_eur / (stop_distance * 100_000)

        return self.normalize_order_size(epic, raw_size, market=market)

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
