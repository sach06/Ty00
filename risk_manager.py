"""
Risk manager: position sizing, daily/drawdown kill-switches, spread filtering.

Design rules
------------
- 1 % of live equity at risk per trade (hard cap)
- 2 % daily loss → halt for the rest of the session
- 7 % drawdown from equity peak → permanent halt until manual reset
- 3 consecutive losses → cool-down pause
- All positions sized in lots (1 lot = 100,000 units for FX)
- Minimum lot step: 0.01  (≈ micro-lot / 1 000 units)
"""
import logging
from datetime import date
from typing import Tuple

import config

logger = logging.getLogger(__name__)

# pip sizes for supported instruments (price units per pip)
PIP_SIZES: dict[str, float] = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "DAX":    1.0,
    "US500":  0.1,
}

# approximate EUR-value of one pip per standard lot for FX
# For EURUSD: 1 pip × 100 000 units = $10 ≈ €9.20, approximated as €10 for simplicity
# We use price-difference math instead, so pip_value_eur is only for logging
LOT_SIZE_UNITS = 100_000
MIN_LOT_STEP   = 0.01
MAX_LOT_SIZE   = 10.0    # hard ceiling regardless of calculation


class RiskManager:
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.equity          = initial_balance
        self.peak_equity     = initial_balance
        self._daily_pnl      = 0.0
        self._trading_day    = date.today()
        self.consecutive_losses = 0
        self.halted          = False   # set True by kill switch; requires manual reset
        self.total_trades    = 0
        self.winning_trades  = 0
        # Budget: the bot only risks money from this envelope
        self.budget          = config.TRADING_BUDGET
        self._budget_peak    = config.TRADING_BUDGET  # tracks high-watermark within budget

    # ── Equity / P&L tracking ─────────────────────────────────────────────────

    def update_equity(self, equity: float):
        self.equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        # Budget peak: only grows, never shrinks (tracks gains within the budget envelope)
        effective = min(equity, self.budget)
        if effective > self._budget_peak:
            self._budget_peak = effective
        # Reset daily P&L counter on new calendar day
        today = date.today()
        if today != self._trading_day:
            logger.info("New trading day. Resetting daily P&L counter.")
            self._daily_pnl   = 0.0
            self._trading_day = today
            # Do NOT auto-reset halted flag here – require manual acknowledgement

    def record_trade_result(self, pnl_eur: float):
        self._daily_pnl += pnl_eur
        self.total_trades += 1
        if pnl_eur > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        logger.info(
            "Trade result: pnl=%.2f EUR | daily_pnl=%.2f | consec_losses=%d | "
            "win_rate=%.0f%%",
            pnl_eur,
            self._daily_pnl,
            self.consecutive_losses,
            self.win_rate * 100,
        )

    # ── Kill-switch checks ────────────────────────────────────────────────────

    def check_kill_switch(self) -> Tuple[bool, str]:
        """Return (should_halt, reason). Sets self.halted if triggered."""
        if self.halted:
            return True, "Previously halted (manual reset required)"

        dd = self.drawdown_pct
        if dd >= config.MAX_DRAWDOWN_PCT:
            self.halted = True
            reason = f"Max drawdown {dd:.1%} ≥ {config.MAX_DRAWDOWN_PCT:.0%} limit"
            logger.critical("KILL SWITCH: %s", reason)
            return True, reason

        # Daily-loss check is relative to TRADING_BUDGET for tighter budget control
        dl = abs(self._daily_pnl) / self.budget if self.budget > 0 else 0.0
        if self._daily_pnl < 0 and dl >= config.MAX_DAILY_LOSS_PCT:
            reason = f"Daily loss €{abs(self._daily_pnl):.2f} ≥ {config.MAX_DAILY_LOSS_PCT:.0%} of budget"
            logger.warning("DAILY HALT: %s", reason)
            return True, reason

        if self.consecutive_losses >= config.MAX_CONSEC_LOSSES:
            reason = f"{self.consecutive_losses} consecutive losses – cooling down"
            logger.warning("CONSEC-LOSS PAUSE: %s", reason)
            return True, reason

        return False, ""

    # ── Position sizing ────────────────────────────────────────────────────────

    def calculate_lot_size(
        self,
        entry: float,
        stop_loss: float,
        epic: str = "EURUSD",
    ) -> float:
        """
        Return lot size (in standard lots) so that the trade risks exactly
        RISK_PER_TRADE_PCT of current equity.

        For FX:  risk_eur = lots × 100 000 × |entry - stop|  (approximation;
                 ignores cross-currency conversion for non-EUR base pairs).
        """
        halt, _ = self.check_kill_switch()
        if halt:
            return 0.0

        stop_distance = abs(entry - stop_loss)
        pip_size = PIP_SIZES.get(epic, 0.0001)

        if stop_distance < pip_size:
            logger.warning("%s: stop distance %.6f < 1 pip – skipping", epic, stop_distance)
            return 0.0

        risk_eur = min(self.equity, self.budget) * config.RISK_PER_TRADE_PCT

        # FX: P&L ≈ lots × 100,000 × price_change
        raw_lots = risk_eur / (LOT_SIZE_UNITS * stop_distance)

        # Round down to nearest MIN_LOT_STEP, apply bounds
        lot_size = max(MIN_LOT_STEP, min(MAX_LOT_SIZE, int(raw_lots / MIN_LOT_STEP) * MIN_LOT_STEP))

        logger.debug(
            "%s sizing: risk=%.2f EUR, stop_dist=%.5f, raw_lots=%.4f → %.2f lots",
            epic, risk_eur, stop_distance, raw_lots, lot_size,
        )
        return lot_size

    # ── Spread filter ──────────────────────────────────────────────────────────

    def spread_ok(self, spread: float, epic: str = "EURUSD") -> bool:
        pip_size = PIP_SIZES.get(epic, 0.0001)
        spread_pips = spread / pip_size
        if spread_pips > config.MAX_SPREAD_PIPS:
            logger.debug("%s: spread %.1f pips > %.1f limit", epic, spread_pips, config.MAX_SPREAD_PIPS)
            return False
        return True

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity == 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    @property
    def daily_loss_pct(self) -> float:
        if self._daily_pnl >= 0 or self.equity == 0:
            return 0.0
        return -self._daily_pnl / self.equity

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades else 0.0

    def status_line(self) -> str:
        return (
            f"equity={self.equity:.2f} | "
            f"drawdown={self.drawdown_pct:.2%} | "
            f"daily_pnl={self._daily_pnl:+.2f} | "
            f"consec_losses={self.consecutive_losses} | "
            f"win_rate={self.win_rate:.0%} ({self.winning_trades}/{self.total_trades})"
        )
