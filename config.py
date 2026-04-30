"""
Central configuration – all tuneable parameters in one place.
Values are loaded from .env; sensible defaults are set where safe.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Capital.com API ────────────────────────────────────────────────────────────
API_KEY      = os.getenv("CAPITAL_API_KEY", "")
API_PASSWORD = os.getenv("CAPITAL_API_PASSWORD", "")
IDENTIFIER   = os.getenv("CAPITAL_IDENTIFIER", "")   # your login email

USE_DEMO = os.getenv("USE_DEMO", "false").lower() == "true"
BASE_URL = (
    "https://demo-api-capital.backend-capital.com/api/v1"
    if USE_DEMO
    else "https://api-capital.backend-capital.com/api/v1"
)

# ── Safety flags ───────────────────────────────────────────────────────────────
# DRY_RUN=true → signals are computed and logged but NO orders are submitted
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# ── Account ────────────────────────────────────────────────────────────────────
INITIAL_BALANCE    = float(os.getenv("INITIAL_BALANCE", "300"))
# Only trade with this amount – the rest of the account equity is untouched.
# Risk, lot-sizing and kill-switch calculations are all relative to TRADING_BUDGET.
TRADING_BUDGET     = float(os.getenv("TRADING_BUDGET", "200.0"))

# ── Risk controls (never change these without understanding the maths) ─────────
RISK_PER_TRADE_PCT  = 0.01   # 1 % of live equity per trade
MAX_DAILY_LOSS_PCT  = 0.02   # 2 % daily loss → halt for the day
MAX_DRAWDOWN_PCT    = 0.07   # 7 % drawdown from equity peak → kill switch
MAX_CONSEC_LOSSES   = 3      # pause after N consecutive losing trades
MAX_SPREAD_PIPS     = 3.0    # skip entry if spread > this many pips

# ── Strategy parameters ────────────────────────────────────────────────────────
RSI_PERIOD      = 14
RSI_OVERSOLD    = 30
RSI_OVERBOUGHT  = 70
BB_PERIOD       = 20
BB_STD          = 2.0
ATR_PERIOD      = 14
STOP_ATR_MULT   = 1.5    # stop-loss distance = 1.5 × ATR
TP_RR_RATIO     = 2.0    # take-profit = 2 × stop distance (2 : 1 R/R)
TIME_STOP_HOURS = 4      # force-close position after 4 hours regardless

# ── Instruments (Capital.com epic codes) ──────────────────────────────────────
# Start with just the two most liquid FX majors; expand after validation
INSTRUMENTS = ["EURUSD", "GBPUSD"]

# ── Session window (UTC) ──────────────────────────────────────────────────────
# New York forex session: 13:00 UTC (NY pre-market / London overlap)
#                         to 22:00 UTC (NYSE close + 1 h wind-down)
SESSION_START_HOUR = int(os.getenv("SESSION_START_HOUR", "13"))  # 09:00 ET
SESSION_END_HOUR   = int(os.getenv("SESSION_END_HOUR",   "22"))  # 18:00 ET

# ── Loop timing ───────────────────────────────────────────────────────────────
POLL_INTERVAL = 60        # seconds between main-loop ticks
SESSION_REFRESH_SECS = 540  # re-authenticate every 9 minutes (tokens expire at ~10 min)
