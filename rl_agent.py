"""
rl_agent.py – Active reinforcement learning agent for strategy parameter adaptation.

Algorithm
---------
UCB1 contextual bandit.  For each discrete market context (state) the agent
maintains a Q-estimate for each parameter variant (action).  At entry time it
picks the action with the highest Upper Confidence Bound; after the trade
closes it updates the Q-estimate with the normalised P&L as the reward signal.

Over many trades the agent learns which parameter set performs best for each
combination of instrument × session slot × sentiment × volatility regime, and
exploits that knowledge while continuing to explore under-sampled variants.

State space
-----------
  instrument  : e.g. "EURUSD", "GBPUSD"
  session slot: morning (08-10 UTC) | midday (10-14 UTC) | afternoon (14-16 UTC)
  sentiment   : bearish (<-0.20) | neutral | bullish (>+0.20)
  volatility  : low ATR (<0.30% of price) | mid | high (>0.70%)
  → max 2 × 3 × 3 × 3 = 54 unique states

Action space (12 parameter variants)
--------------------------------------
  Each action = (rsi_oversold, rsi_overbought, bb_std, stop_atr_mult, tp_rr_ratio)

Reward
------
  r = clip(pnl / risk_amount, -3, +4)   (normalised, dimensionless)
  Updated via incremental mean:  Q ← Q + (r - Q) / n
"""

import json
import math
import os
import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

# ── Parameter action space ────────────────────────────────────────────────────
# (rsi_oversold, rsi_overbought, bb_std, stop_atr_mult, tp_rr_ratio)
PARAM_VARIANTS: list[tuple] = [
    (30, 70, 2.0, 1.5, 2.0),   # [0]  baseline
    (28, 72, 2.0, 1.5, 2.0),   # [1]  tighter RSI entry
    (25, 75, 2.0, 1.5, 2.0),   # [2]  strictest RSI entry
    (32, 68, 2.0, 1.5, 2.0),   # [3]  relaxed RSI entry
    (30, 70, 1.8, 1.5, 2.0),   # [4]  tighter Bollinger (1.8σ)
    (30, 70, 2.2, 1.5, 2.0),   # [5]  wider  Bollinger (2.2σ)
    (30, 70, 2.0, 1.2, 2.5),   # [6]  tight stop / wider TP
    (30, 70, 2.0, 2.0, 2.0),   # [7]  wider stop
    (30, 70, 2.0, 1.5, 1.5),   # [8]  lower TP ratio (quick profit)
    (30, 70, 2.0, 1.5, 3.0),   # [9]  higher TP ratio (let winners run)
    (28, 72, 1.8, 1.2, 3.0),   # [10] aggressive  (tight signals, big TP)
    (32, 68, 2.2, 2.0, 1.8),   # [11] conservative (relaxed + wide stop)
]

PARAM_NAMES: list[str] = [
    "baseline",
    "tighter-RSI",
    "strictest-RSI",
    "relaxed-RSI",
    "tighter-BB",
    "wider-BB",
    "tight-stop/wide-TP",
    "wider-stop",
    "quick-TP",
    "let-winners-run",
    "aggressive",
    "conservative",
]

N_ACTIONS     = len(PARAM_VARIANTS)
UCB_C         = math.sqrt(2)        # exploration constant (≈1.414)
REWARD_CLIP   = (-3.0, 4.0)         # clip normalised rewards
RL_STATE_FILE = "rl_state.json"
SAFE_ACTIONS = [0, 1, 4, 8, 9, 11]
AGGRESSIVE_ACTIONS = [6, 10]


class RLAgent:
    """UCB1 contextual bandit for intraday strategy parameter adaptation."""

    def __init__(self):
        # Q[state][action] = [cumulative_mean_reward, visit_count]
        self.Q: dict[str, list[list]] = defaultdict(
            lambda: [[0.0, 0] for _ in range(N_ACTIONS)]
        )
        self.total_updates: int = 0   # closed trades processed
        self._load()

    # ── State encoding ────────────────────────────────────────────────────────

    @staticmethod
    def encode_state(instrument: str, hour_utc: int,
                     sentiment: float, atr_pct: float) -> str:
        """Map continuous market context to a compact discrete state key."""
        if hour_utc < 10:
            sess = "morn"
        elif hour_utc < 14:
            sess = "mid"
        else:
            sess = "aftn"

        if sentiment < -0.2:
            sent = "bear"
        elif sentiment > 0.2:
            sent = "bull"
        else:
            sent = "neut"

        if atr_pct < 0.003:
            vol = "low"
        elif atr_pct < 0.007:
            vol = "mid"
        else:
            vol = "high"

        return f"{instrument}|{sess}|{sent}|{vol}"

    # ── Action selection ──────────────────────────────────────────────────────

    def select(self, instrument: str, hour_utc: int,
               sentiment: float, atr_pct: float,
               consecutive_losses: int = 0) -> tuple:
        """
        UCB1 action selection.

        Unvisited actions are always tried first (in index order) to ensure
        every variant is explored at least once per state before exploitation
        begins.  After that, picks argmax UCB1 score.

        Returns
        -------
        (params_tuple, action_index, state_key)
        """
        state   = self.encode_state(instrument, hour_utc, sentiment, atr_pct)
        q_state = self.Q[state]

        if consecutive_losses >= 2:
            fallback = 11  # conservative
            logger.info(
                "RL fallback | state=%-26s | action=%2d %-22s | consecutive_losses=%d",
                state, fallback, f"({PARAM_NAMES[fallback]})", consecutive_losses,
            )
            return PARAM_VARIANTS[fallback], fallback, state

        safe_visited = [(a, q, c) for a, (q, c) in enumerate(q_state) if c > 0 and a in SAFE_ACTIONS]
        safe_avg_q = (sum(q * c for a, q, c in safe_visited) / sum(c for _, _, c in safe_visited)) if safe_visited else 0.0
        allow_aggressive = safe_avg_q > 0.25 and sum(c for _, _, c in safe_visited) >= 6

        # Explore only the safer subset first. Aggressive actions are unlocked
        # only after the state shows a positive edge with enough observations.
        exploration_order = SAFE_ACTIONS + ([a for a in AGGRESSIVE_ACTIONS if allow_aggressive])
        for a in exploration_order:
            _, cnt = q_state[a]
            if cnt == 0:
                logger.debug(
                    "RL explore  | state=%-26s | action=%2d %-22s [first visit]",
                    state, a, f"({PARAM_NAMES[a]})",
                )
                return PARAM_VARIANTS[a], a, state

        # UCB1 exploitation + exploration
        allowed_actions = SAFE_ACTIONS + ([a for a in AGGRESSIVE_ACTIONS if allow_aggressive])
        total_visits = sum(q_state[a][1] for a in allowed_actions)
        best_a, best_ucb = 0, -float("inf")
        for a in allowed_actions:
            q_val, cnt = q_state[a]
            ucb = q_val + UCB_C * math.sqrt(math.log(total_visits) / cnt)
            if ucb > best_ucb:
                best_ucb, best_a = ucb, a

        logger.debug(
            "RL select   | state=%-26s | action=%2d %-22s | ucb=%.4f",
            state, best_a, f"({PARAM_NAMES[best_a]})", best_ucb,
        )
        return PARAM_VARIANTS[best_a], best_a, state

    # ── Reward update ─────────────────────────────────────────────────────────

    def update(self, state: str, action: int,
               pnl: float, risk_amount: float) -> float:
        """
        Update Q(state, action) after a trade closes.

        reward = clip(pnl / risk_amount, REWARD_CLIP)
        Q updated via online incremental mean (Monte-Carlo style).

        Returns the normalised reward applied.
        """
        if risk_amount <= 0:
            risk_amount = 1.0
        reward = float(max(REWARD_CLIP[0], min(REWARD_CLIP[1], pnl / risk_amount)))

        q_val, cnt = self.Q[state][action]
        new_cnt = cnt + 1
        new_q   = q_val + (reward - q_val) / new_cnt   # incremental mean
        self.Q[state][action] = [new_q, new_cnt]
        self.total_updates += 1

        logger.info(
            "RL ▸ update | state=%-26s | action=%2d %-22s | "
            "pnl=%+.2f | r=%+.3f | Q=%+.4f | n=%d",
            state, action, f"({PARAM_NAMES[action]})",
            pnl, reward, new_q, new_cnt,
        )
        self._save()
        return reward

    # ── Self-evaluation & introspection ───────────────────────────────────────

    def performance_summary(self) -> dict:
        """
        Return a structured summary for the dashboard / state file.

        Includes:
          - global best action (by visit-weighted mean reward)
          - per-state best action with learned Q-value
          - per-action aggregate stats
        """
        per_state_rows = []
        global_counts  = [0]   * N_ACTIONS
        global_rewards = [0.0] * N_ACTIONS

        for state, actions in self.Q.items():
            visited = [(a, q, c) for a, (q, c) in enumerate(actions) if c > 0]
            if not visited:
                continue

            best_a, best_q, best_c = max(visited, key=lambda x: x[1])
            total_v = sum(c for _, _, c in visited)

            per_state_rows.append({
                "state":        state,
                "best_action":  PARAM_NAMES[best_a],
                "best_q":       round(best_q, 4),
                "total_visits": total_v,
                "params": {
                    "rsi_os":    PARAM_VARIANTS[best_a][0],
                    "rsi_ob":    PARAM_VARIANTS[best_a][1],
                    "bb_std":    PARAM_VARIANTS[best_a][2],
                    "stop_mult": PARAM_VARIANTS[best_a][3],
                    "tp_ratio":  PARAM_VARIANTS[best_a][4],
                },
            })

            for a, (q_val, cnt) in enumerate(actions):
                global_counts[a]  += cnt
                global_rewards[a] += q_val * cnt

        global_avg = [
            global_rewards[a] / global_counts[a] if global_counts[a] > 0 else 0.0
            for a in range(N_ACTIONS)
        ]
        best_global = (
            max(range(N_ACTIONS), key=lambda a: global_avg[a])
            if any(global_counts) else 0
        )

        action_stats = [
            {
                "name":   PARAM_NAMES[a],
                "visits": global_counts[a],
                "avg_q":  round(global_avg[a], 4),
                "params": {
                    "rsi_os":    PARAM_VARIANTS[a][0],
                    "rsi_ob":    PARAM_VARIANTS[a][1],
                    "bb_std":    PARAM_VARIANTS[a][2],
                    "stop_mult": PARAM_VARIANTS[a][3],
                    "tp_ratio":  PARAM_VARIANTS[a][4],
                },
            }
            for a in range(N_ACTIONS)
        ]

        return {
            "total_updates":      self.total_updates,
            "states_explored":    len(per_state_rows),
            "best_global_action": PARAM_NAMES[best_global],
            "best_global_q":      round(global_avg[best_global], 4),
            "per_state":          sorted(
                per_state_rows, key=lambda r: r["best_q"], reverse=True
            )[:15],
            "action_stats":       action_stats,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        try:
            with open(RL_STATE_FILE, "w") as fh:
                json.dump(
                    {"Q": {k: v for k, v in self.Q.items()},
                     "total_updates": self.total_updates},
                    fh,
                )
        except OSError as exc:
            logger.warning("RL state save failed: %s", exc)

    def _load(self):
        if not os.path.exists(RL_STATE_FILE):
            logger.info("RL agent initialised (no prior state – starting fresh)")
            return
        try:
            with open(RL_STATE_FILE) as fh:
                data = json.load(fh)
            for k, v in data.get("Q", {}).items():
                self.Q[k] = v
            self.total_updates = data.get("total_updates", 0)
            logger.info(
                "RL agent loaded | states=%d | total_updates=%d",
                len(self.Q), self.total_updates,
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("RL state load failed: %s – starting fresh", exc)
