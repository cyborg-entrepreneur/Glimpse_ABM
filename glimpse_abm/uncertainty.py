"""
Knightian Uncertainty Environment for Glimpse ABM.

This module implements the four-dimensional Knightian uncertainty framework as
theoretically specified in:

    Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).
    Are the futures computable? Knightian uncertainty & artificial intelligence.
    Academy of Management Review, 50(2), 415-440.

    Townsend, D. M., Hunt, R. A., & Rady, J. (2024). Chance, probability, &
    uncertainty at the edge of human reasoning: What is Knightian uncertainty?
    Strategic Entrepreneurship Journal, 18(3), 451-474.

Theoretical Foundation
----------------------
Knight (1921) distinguished between calculable risk and incalculable uncertainty.
While risk involves known probability distributions over known outcomes, true
uncertainty involves situations where either the outcomes, their probabilities,
or both are fundamentally unknowable. Townsend et al. (2025) extend this framework
to identify four distinct sources of Knightian uncertainty that are particularly
relevant to entrepreneurial action under AI augmentation:

1. **Actor Ignorance**: The entrepreneur's incomplete knowledge of the opportunity
   landscape, market conditions, and causal relationships. AI can reduce ignorance
   by expanding information access but may also create false confidence through
   hallucinations or biased outputs.

2. **Practical Indeterminism**: The inherent unpredictability of execution paths,
   timing dependencies, and path-dependent outcomes. Even with perfect information,
   the future unfolds through contingent processes that cannot be fully predicted.

3. **Agentic Novelty**: The creative potential for genuinely new combinations,
   innovations, and possibilities that did not previously exist. AI may facilitate
   novel combinations but can also constrain novelty through pattern-matching to
   historical data.

4. **Competitive Recursion**: The strategic interdependence among actors where
   each agent's actions depend on anticipations of others' actions, creating
   irreducible game-theoretic complexity. AI adoption can intensify herding
   behavior and recursive strategic dynamics.

The Paradox of Future Knowledge
-------------------------------
A central insight from Townsend et al. (2025) is that AI creates a "paradox of
future knowledge": while AI tools reduce actor ignorance and improve information
quality, they may simultaneously:
- Increase practical indeterminism through faster competitive dynamics
- Reduce agentic novelty by anchoring on historical patterns
- Intensify competitive recursion through correlated recommendations

This module operationalizes these theoretical constructs to enable empirical
investigation of how AI augmentation affects entrepreneurial decision-making
under fundamental uncertainty.

References
----------
Knight, F. H. (1921). Risk, uncertainty, and profit. Houghton Mifflin.

Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).
    Are the futures computable? Knightian uncertainty & artificial intelligence.
    Academy of Management Review, 50(2), 415-440.

Townsend, D. M., Hunt, R. A., & Rady, J. (2024). Chance, probability, &
    uncertainty at the edge of human reasoning: What is Knightian uncertainty?
    Strategic Entrepreneurship Journal, 18(3), 451-474.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional, Set

import numpy as np

from .config import EmergentConfig
from .utils import fast_mean, safe_mean, safe_exp, stable_sigmoid, normalize_ai_label


class KnightianUncertaintyEnvironment:
    """
    Environment tracking and measuring the four dimensions of Knightian uncertainty.

    This class implements the theoretical framework from Townsend et al. (2025, AMR)
    which decomposes Knightian uncertainty into four irreducible components that
    affect entrepreneurial action differently and respond differently to AI
    augmentation.

    Attributes
    ----------
    actor_ignorance_state : dict
        Tracks unknown opportunities, knowledge gaps, and emergence potential.
        Corresponds to the entrepreneur's incomplete mental model of the
        opportunity landscape (Townsend et al., 2025, Proposition 1).

    practical_indeterminism_state : dict
        Tracks path volatility, timing criticality, and regime instability.
        Captures the inherent unpredictability of execution paths even with
        complete information (Townsend et al., 2025, Proposition 2).

    agentic_novelty_state : dict
        Tracks creative momentum, disruption potential, and novelty levels.
        Measures the system's capacity for genuinely new combinations that
        transcend historical patterns (Townsend et al., 2025, Proposition 3).

    competitive_recursion_state : dict
        Tracks strategic opacity, herding pressure, and game complexity.
        Captures the irreducible complexity arising from strategic
        interdependence among agents (Townsend et al., 2025, Proposition 4).

    ai_uncertainty_signals : dict
        Tracks AI-specific uncertainty contributions including hallucination
        events, confidence miscalibration, and AI-induced herding patterns.
        These signals enable measurement of the "paradox of future knowledge."

    Notes
    -----
    The four uncertainty dimensions are measured at both the environment level
    (aggregate market conditions) and the agent level (individual perceptions).
    This dual-level measurement enables investigation of how AI affects the
    gap between objective conditions and subjective perceptions.
    """

    def __init__(self, config: EmergentConfig, knowledge_base: Optional["KnowledgeBase"] = None):
        self.config = config
        self.knowledge_base = knowledge_base
        self.actor_ignorance_state = {
            "unknown_opportunities": {},
            "knowledge_gaps": {},
            "emergence_potential": 0.0,
        }
        self.practical_indeterminism_state = {
            "path_volatility": 0.0,
            "timing_criticality": {},
            "regime_instability": 0.0,
        }
        self.agentic_novelty_state = {
            "creative_momentum": 0.0,
            "disruption_potential": {},
            "novelty_level": 0.0,
        }
        self.competitive_recursion_state = {
            "strategic_opacity": 0.0,
            "herding_pressure": {},
            "game_complexity": 0.0,
        }
        self.uncertainty_evolution = []
        self.ai_uncertainty_signals = {
            "hallucination_events": [],
            "confidence_miscalibration": [],
            "ai_herding_patterns": {},
        }
        history_window = max(5, int(getattr(self.config, "NOVELTY_HISTORY_WINDOW", 15)))
        self.innovation_success_tracker = collections.deque(maxlen=history_window)
        self.exploration_outcomes = collections.deque(maxlen=30)
        self.market_regime_history = collections.deque(maxlen=20)
        self.opportunity_discovery_rate = 0.5
        self.short_term_window = int(getattr(self.config, "UNCERTAINTY_SHORT_WINDOW", 5))
        self.short_term_decay_factor = getattr(self.config, "UNCERTAINTY_SHORT_DECAY", 0.85)
        self._global_short_term_buffers = {
            "actor_ignorance": collections.deque(maxlen=self.short_term_window),
            "practical_indeterminism": collections.deque(maxlen=self.short_term_window),
            "agentic_novelty": collections.deque(maxlen=self.short_term_window),
            "competitive_recursion": collections.deque(maxlen=self.short_term_window),
        }
        self._agent_short_term_buffers: Dict[int, Dict[str, collections.deque]] = collections.defaultdict(
            lambda: {
                "actor_ignorance": collections.deque(maxlen=self.short_term_window),
                "practical_indeterminism": collections.deque(maxlen=self.short_term_window),
                "agentic_novelty": collections.deque(maxlen=self.short_term_window),
                "competitive_recursion": collections.deque(maxlen=self.short_term_window),
            }
        )
        volatility_window = max(3, int(getattr(self.config, "UNCERTAINTY_VOLATILITY_WINDOW", 12)))
        self._volatility_state = {
            "action_prev": np.zeros(4, dtype=float),
            "ai_prev": np.zeros(4, dtype=float),
            "market_prev": 0.0,
            "history": collections.deque(maxlen=volatility_window),
            "volatility_metric": 0.0,
            "last_action_shares": np.zeros(4, dtype=float),
            "last_ai_shares": np.zeros(4, dtype=float),
        }
        self._ai_signal_history = max(10, int(getattr(self.config, "AI_SIGNAL_HISTORY", 200)))
        self._action_history = collections.deque(maxlen=history_window)
        self._novelty_diagnostics: Dict[str, Any] = {}
        self._tier_smoothing = {
            "none": 1.0,
            "basic": 1.0,
            "advanced": 1.0,
            "premium": 1.0,
        }

    def _short_buffer(self, metric: str, agent_id: Optional[int]) -> collections.deque:
        if agent_id is None:
            return self._global_short_term_buffers[metric]
        return self._agent_short_term_buffers[agent_id][metric]

    def _update_short_term(self, metric: str, value: float, agent_id: Optional[int] = None) -> None:
        if value is None:
            return
        if isinstance(value, (float, int)) and not np.isnan(value):
            self._short_buffer(metric, agent_id).append(float(value))

    def _decay_short_term(self, metric: str, fallback: float, agent_id: Optional[int] = None) -> None:
        buffer = self._short_buffer(metric, agent_id)
        if buffer:
            baseline = buffer[-1]
        else:
            baseline = fallback if fallback is not None else 0.0
        if baseline is None or np.isnan(baseline):
            baseline = 0.0
        # No decay: retain last signal strength when no new data arrives.
        buffer.append(baseline)

    def _update_volatility_state(self, action_shares: np.ndarray, ai_shares: np.ndarray, market: "MarketEnvironment") -> float:
        cfg = self.config
        action_prev = self._volatility_state.get("action_prev")
        if action_prev is None or len(action_prev) != len(action_shares):
            action_prev = np.zeros_like(action_shares)
        ai_prev = self._volatility_state.get("ai_prev")
        if ai_prev is None or len(ai_prev) != len(ai_shares):
            ai_prev = np.zeros_like(ai_shares)

        action_delta = float(np.mean(np.abs(action_shares - action_prev)))
        ai_delta = float(np.mean(np.abs(ai_shares - ai_prev)))

        market_signal = float(abs(getattr(market, "market_momentum", 0.0)) + abs(getattr(market, "volatility", 0.0)))
        market_prev = float(self._volatility_state.get("market_prev", 0.0))
        market_delta = abs(market_signal - market_prev)

        action_weight = float(getattr(cfg, "UNCERTAINTY_ACTION_VARIANCE_WEIGHT", 0.10))
        ai_weight = float(getattr(cfg, "UNCERTAINTY_AI_SWITCH_WEIGHT", 0.07))
        market_weight = float(getattr(cfg, "UNCERTAINTY_MARKET_RETURN_WEIGHT", 0.12))

        raw = action_weight * action_delta + ai_weight * ai_delta + market_weight * market_delta
        decay = float(getattr(cfg, "UNCERTAINTY_VOLATILITY_DECAY", 0.85))
        prev_vol = float(self._volatility_state.get("volatility_metric", 0.0))
        smoothed = prev_vol * decay + (1.0 - decay) * raw
        scaling = float(getattr(cfg, "UNCERTAINTY_VOLATILITY_SCALING", 0.18))
        volatility = smoothed * scaling

        self._volatility_state.update(
            {
                "action_prev": action_shares.copy(),
                "ai_prev": ai_shares.copy(),
                "market_prev": market_signal,
                "volatility_metric": volatility,
            }
        )
        return volatility

    def _normalize_metric(self, metric: str, value: float) -> float:
        if value is None or np.isnan(value):
            return 0.0
        # Let raw magnitude pass through; only floor at zero.
        return float(max(0.0, value))

    def _get_short_term_average(self, metric: str, fallback: float = 0.0, agent_id: Optional[int] = None) -> float:
        buffer = None
        if agent_id is not None and agent_id in self._agent_short_term_buffers:
            buffer = self._agent_short_term_buffers[agent_id].get(metric)
        if buffer is None:
            buffer = self._global_short_term_buffers.get(metric)
        if buffer and len(buffer) > 0:
            avg = float(np.mean(buffer))
        else:
            avg = fallback if fallback is not None else 0.0
        return self._normalize_metric(metric, avg)

    def record_ai_signals(self, round_num: int, agent_actions: List[Dict[str, Any]]) -> None:
        if not agent_actions:
            return
        hallucinations = self.ai_uncertainty_signals["hallucination_events"]
        confidence_vals = self.ai_uncertainty_signals["confidence_miscalibration"]
        herding_patterns = self.ai_uncertainty_signals["ai_herding_patterns"]
        herding_counts = collections.Counter()

        hallucinations_this_round = 0
        for action in agent_actions:
            ai_level = normalize_ai_label(action.get("ai_level_used", "none"))
            if ai_level == "none":
                continue
            agent_id = action.get("agent_id")
            domain = action.get("ai_analysis_domain")

            if action.get("ai_contains_hallucination"):
                hallucinations_this_round += 1
                hallucinations.append(
                    {
                        "round": round_num,
                        "agent_id": agent_id,
                        "ai_level": ai_level,
                        "domain": domain,
                    }
                )
            confidence = action.get("ai_confidence")
            accuracy = action.get("ai_actual_accuracy")
            try:
                if confidence is not None and accuracy is not None:
                    miscalibration = float(confidence) - float(accuracy)
                    if np.isfinite(miscalibration):
                        confidence_vals.append(miscalibration)
            except (TypeError, ValueError):
                pass

            if action.get("action") == "invest":
                details = action.get("chosen_opportunity_details") or {}
                opp_id = details.get("id") or action.get("opportunity_id")
                if opp_id:
                    herding_counts[opp_id] += 1

        decay = float(getattr(self.config, "AI_HERDING_DECAY", 1.0))
        for key in list(herding_patterns.keys()):
            herding_patterns[key] *= decay
        for opp_id, count in herding_counts.items():
            prior = herding_patterns.get(opp_id, 0.0) * decay
            herding_patterns[opp_id] = prior + count

        max_history = self._ai_signal_history
        if len(hallucinations) > max_history:
            del hallucinations[:-max_history]
        if len(confidence_vals) > max_history:
            del confidence_vals[:-max_history]

        if hallucinations_this_round > 0:
            spike = float(np.clip(0.15 + 0.05 * hallucinations_this_round, 0.0, 1.0))
            self._update_short_term("actor_ignorance", spike)
            self._update_short_term("agentic_novelty", spike * 0.6)

    def _summarize_actions(self, agent_actions: List[Dict[str, Any]], market: Optional["MarketEnvironment"]) -> Dict[str, Any]:
        summary = {
            "new_combos": 0,
            "innovate": 0,
            "new_niches": 0,
            "explore": 0,
            "derivative_adoption": 0,
            "invest": 0,
            "combo_hhi": 0.0,
            "sector_hhi": 0.0,
            "invest_hhi": 0.0,
            "herding_counts": {},
            "invest_by_ai": {},
            "tier_stats": {},
            "ai_action_correlation": 0.0,  # NEW: Track AI-induced action correlation
        }
        invest_counts = collections.Counter()
        invest_by_ai = collections.Counter()
        tiers = ["none", "basic", "advanced", "premium"]

        # NEW: Track actions by tier for correlation calculation
        actions_by_tier = {tier: [] for tier in tiers}
        opportunities_by_tier = {tier: collections.Counter() for tier in tiers}
        tier_template = {
            "total_actions": 0,
            "innovate": 0,
            "new_combos": 0,
            "reuse_hits": 0,
            "explore": 0,
            "new_niches": 0,
            "invest": 0,
            "derivative_adoption": 0,
        }
        tier_stats: Dict[str, Dict[str, float]] = {tier: tier_template.copy() for tier in tiers}
        for action in agent_actions:
            act_type = action.get("action")
            tier = normalize_ai_label(action.get("ai_level_used", "none"))
            if tier not in tier_stats:
                tier_stats[tier] = tier_template.copy()
            tier_stats[tier]["total_actions"] += 1

            # NEW: Track actions for correlation calculation
            actions_by_tier[tier].append(act_type)

            if act_type == "innovate":
                summary["innovate"] += 1
                if action.get("is_new_combination"):
                    summary["new_combos"] += 1
                    tier_stats[tier]["new_combos"] += 1
                elif action.get("combination_signature"):
                    tier_stats[tier]["reuse_hits"] += 1
                tier_stats[tier]["innovate"] += 1
            elif act_type == "explore":
                summary["explore"] += 1
                if action.get("created_niche") or action.get("discovered_niche"):
                    summary["new_niches"] += 1
                    tier_stats[tier]["new_niches"] += 1
                tier_stats[tier]["explore"] += 1
            elif act_type == "invest":
                summary["invest"] += 1
                tier_stats[tier]["invest"] += 1
                if action.get("invested_derivative"):
                    summary["derivative_adoption"] += 1
                    tier_stats[tier]["derivative_adoption"] += 1
                opp_id = None
                details = action.get("chosen_opportunity_details") or {}
                opp_id = details.get("id") or action.get("opportunity_id")
                if opp_id:
                    invest_counts[str(opp_id)] += 1
                    # NEW: Track opportunity choices by tier
                    opportunities_by_tier[tier][str(opp_id)] += 1
                tier = normalize_ai_label(action.get("ai_level_used", "none"))
                invest_by_ai[tier] += 1
            else:
                tier_stats[tier]["total_actions"] += 0  # explicit no-op for clarity
        invest_total = summary["invest"]
        if invest_total > 0 and invest_counts:
            summary["invest_hhi"] = float(sum((count / invest_total) ** 2 for count in invest_counts.values()))
        summary["herding_counts"] = dict(invest_counts)
        summary["invest_by_ai"] = dict(invest_by_ai)
        tier_combo_rate: Dict[str, float] = {}
        tier_reuse_pressure: Dict[str, float] = {}
        tier_new_poss_rate: Dict[str, float] = {}
        tier_adoption_rate: Dict[str, float] = {}
        for tier, stats in tier_stats.items():
            innov = max(1, stats["innovate"])
            invest_count = max(1, stats["invest"])
            action_total = max(1, stats["total_actions"])
            tier_combo_rate[tier] = float(np.clip(stats["new_combos"] / innov, 0.0, 1.0))
            tier_reuse_pressure[tier] = float(np.clip(stats["reuse_hits"] / innov, 0.0, 1.0))
            tier_new_poss_rate[tier] = float(
                np.clip((stats["new_combos"] + stats["new_niches"]) / action_total, 0.0, 1.0)
            )
            tier_adoption_rate[tier] = float(np.clip(stats["derivative_adoption"] / invest_count, 0.0, 1.0))
        summary["tier_stats"] = tier_stats
        summary["tier_combo_rate"] = tier_combo_rate
        summary["tier_reuse_pressure"] = tier_reuse_pressure
        summary["tier_new_possibility_rate"] = tier_new_poss_rate
        summary["tier_adoption_rate"] = tier_adoption_rate
        if market is not None and hasattr(market, "get_combination_diversity_metrics"):
            combo_hhi, sector_hhi = market.get_combination_diversity_metrics()
            summary["combo_hhi"] = combo_hhi
            summary["sector_hhi"] = sector_hhi

        # NEW: Calculate AI action correlation for competitive recursion
        # When agents with AI choose similar actions/opportunities, recursion increases
        ai_tiers = ["basic", "advanced", "premium"]
        ai_action_counts = sum(len(actions_by_tier[tier]) for tier in ai_tiers)

        if ai_action_counts >= 2:
            # Calculate correlation based on opportunity clustering among AI agents
            ai_opportunity_hhi = 0.0
            total_ai_investments = sum(sum(opportunities_by_tier[tier].values()) for tier in ai_tiers)

            if total_ai_investments > 0:
                # HHI of opportunities among AI agents (higher = more correlated)
                all_ai_opps = collections.Counter()
                for tier in ai_tiers:
                    for opp_id, count in opportunities_by_tier[tier].items():
                        all_ai_opps[opp_id] += count

                ai_opportunity_hhi = sum(
                    (count / total_ai_investments) ** 2 for count in all_ai_opps.values()
                )

            # Correlation ranges from baseline 0.3 (no AI) to 0.6+ (high AI adoption with clustering)
            # Scale by AI adoption rate and opportunity clustering
            ai_adoption_rate = ai_action_counts / max(1, len(agent_actions))
            summary["ai_action_correlation"] = float(
                np.clip(
                    0.30 + 0.30 * ai_adoption_rate * ai_opportunity_hhi,
                    0.30,
                    0.70
                )
            )
        else:
            summary["ai_action_correlation"] = 0.30  # Baseline correlation without AI

        return summary

    def _compute_component_scarcity(self) -> float:
        if self.knowledge_base is not None:
            raw_scarcity = float(self.knowledge_base.get_component_scarcity_metric())
        else:
            raw_scarcity = float(self.agentic_novelty_state.get("scarcity_signal", 0.5))
        prev = self.agentic_novelty_state.get("scarcity_signal")
        if prev is None:
            return raw_scarcity
        blend = float(np.clip(0.65 * prev + 0.35 * raw_scarcity, 0.0, 1.0))
        return blend

    def register_innovation_event(
        self,
        opportunity_id: str,
        success: Optional[bool] = None,
        impact: Optional[float] = None,
        combination_signature: Optional[str] = None,
        new_possibility_rate: Optional[float] = None,
        scarcity: Optional[float] = None,
    ) -> None:
        """Track innovation outcomes to update novelty perceptions."""
        record = {
            "opportunity_id": opportunity_id,
            "success": bool(success) if success is not None else None,
            "impact": float(impact) if impact is not None else None,
            "combination_signature": combination_signature,
        }
        self.innovation_success_tracker.append(record)
        if new_possibility_rate is not None:
            self.agentic_novelty_state["recent_new_possibility_rate"] = float(np.clip(new_possibility_rate, 0.0, 1.0))
        if scarcity is not None:
            self.agentic_novelty_state["recent_scarcity_signal"] = float(np.clip(scarcity, 0.0, 1.0))

    def measure_uncertainty_state(
        self,
        market: "MarketEnvironment",
        agent_actions: List[Dict],
        innovations: List[Dict],
        round_num: int,
    ) -> Dict:
        """
        Measure environment-level uncertainty across the four Knightian dimensions.

        This method computes aggregate uncertainty metrics that characterize the
        market environment as a whole. These environment-level measures serve as
        the objective baseline against which individual agent perceptions can be
        compared, enabling investigation of how AI affects the accuracy of
        entrepreneurs' uncertainty assessments.

        Parameters
        ----------
        market : MarketEnvironment
            Current market state including opportunities, regime, and volatility.
        agent_actions : List[Dict]
            Actions taken by all agents in the current round, used to compute
            aggregate behavioral patterns (e.g., herding, exploration rates).
        innovations : List[Dict]
            Innovations created in the current round, used to assess the rate
            of genuinely novel combinations entering the system.
        round_num : int
            Current simulation round for temporal tracking.

        Returns
        -------
        Dict
            Dictionary containing measurements for each uncertainty dimension:

            actor_ignorance : dict
                - level: Aggregate ignorance level [0,1]
                - knowledge_gaps: Map of opportunity IDs to knowledge gap severity
                - gap_pressure: Mean knowledge gap across opportunities
                - volatility: Information volatility measure
                - ai_delta: Difference in ignorance with vs. without AI

            practical_indeterminism : dict
                - level: Aggregate path unpredictability [0,1]
                - timing_pressure: Map of opportunity IDs to timing criticality
                - volatility: Execution path volatility
                - crowding_pressure: Capital crowding intensity
                - ai_herding_intensity: AI-induced herding pressure
                - ai_delta: AI's contribution to indeterminism

            agentic_novelty : dict
                - level: System-wide novelty potential [0,1]
                - new_possibilities: Count of new combinations created
                - new_possibility_rate: Rate of novel combination creation
                - innovation_births: Count of successful innovations
                - combo_rate: Rate of new knowledge combinations
                - reuse_pressure: Pressure toward existing combinations
                - ai_delta: AI's effect on novelty potential

            competitive_recursion : dict
                - level: Strategic interdependence intensity [0,1]
                - herding_pressure: Map of opportunity IDs to herding intensity
                - ai_premium_share: Share of premium AI users
                - ai_herding_intensity: AI-induced strategic correlation
                - ai_delta: AI's contribution to recursive dynamics

        Notes
        -----
        The ai_delta fields capture the "paradox of future knowledge" by
        measuring how AI augmentation shifts each uncertainty dimension
        relative to a counterfactual baseline without AI. Positive values
        indicate AI increases that uncertainty dimension; negative values
        indicate AI reduces it.

        See Also
        --------
        perceive_uncertainty : Agent-level uncertainty perception method.

        References
        ----------
        Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).
            Are the futures computable? Knightian uncertainty & artificial
            intelligence. Academy of Management Review, 50(2), 415-440.
        """
        state: Dict[str, Dict[str, float]] = {}
        ai_effects: Dict[str, float] = {}
        total_actions = len(agent_actions)
        action_counter = collections.Counter(action.get("action", "maintain") for action in agent_actions)
        if total_actions > 0:
            action_shares = np.array(
                [
                    action_counter.get("invest", 0) / total_actions,
                    action_counter.get("innovate", 0) / total_actions,
                    action_counter.get("explore", 0) / total_actions,
                    action_counter.get("maintain", 0) / total_actions,
                ],
                dtype=float,
            )
        else:
            action_shares = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        ai_counter = collections.Counter(
            normalize_ai_label(action.get("ai_level_used", "none")) for action in agent_actions
        )
        if total_actions > 0:
            ai_shares = np.array(
                [
                    ai_counter.get("none", 0) / total_actions,
                    ai_counter.get("basic", 0) / total_actions,
                    ai_counter.get("advanced", 0) / total_actions,
                    ai_counter.get("premium", 0) / total_actions,
                ],
                dtype=float,
            )
        else:
            ai_shares = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        ai_share_none = float(ai_shares[0]) if ai_shares.size else 1.0

        action_summary = self._summarize_actions(agent_actions, market)
        tier_invest_share: Dict[str, float] = {}
        invest_total = action_summary.get("invest", 0)
        if invest_total > 0:
            for tier, count in action_summary.get("invest_by_ai", {}).items():
                tier_invest_share[tier] = float(np.clip(count / invest_total, 0.0, 1.0))
        if market is not None and tier_invest_share:
            market._tier_invest_share = tier_invest_share
        if not tier_invest_share:
            tier_invest_share = {
                tier: float(ai_shares[idx]) if ai_shares.size > idx else 0.0
                for idx, tier in enumerate(["none", "basic", "advanced", "premium"])
            }
        self._action_history.append(action_summary)
        ai_herding_patterns = self.ai_uncertainty_signals.get("ai_herding_patterns", {})
        if ai_herding_patterns:
            herding_total = float(sum(ai_herding_patterns.values()))
            ai_herding_intensity = float(np.clip(herding_total / max(1.0, total_actions), 0.0, 1.0))
        else:
            ai_herding_intensity = 0.0

        new_possibilities = 0
        for action in agent_actions:
            if action.get("discovered_niche") or action.get("created_opportunity") or action.get("new_opportunity_id"):
                new_possibilities += 1
        innovation_births = sum(1 for inn in innovations if inn is not None)
        new_possibility_rate = new_possibilities / max(1, total_actions)

        volatility = self._update_volatility_state(action_shares, ai_shares, market)
        self._volatility_state["last_action_shares"] = action_shares.copy()
        self._volatility_state["last_ai_shares"] = ai_shares.copy()
        share_invest, share_innovate, share_explore, share_maintain = action_shares.tolist()

        knowledge_gaps = {}
        for action in agent_actions:
            perception = action.get("perception_at_decision", {})
            ignorance = perception.get("actor_ignorance", {})
            gap_map = ignorance.get("knowledge_gaps", {})
            if gap_map:
                knowledge_gaps.update(gap_map)

        avg_knowledge = float(self.knowledge_base.get_average_agent_knowledge()) if self.knowledge_base else 0.0
        total_opportunities = len(getattr(market, "opportunities", [])) if market is not None else 0
        knowledge_norm = avg_knowledge / max(1.0, total_opportunities / max(1, getattr(self.config, "N_AGENTS", 1) / 4))
        gap_values = list(knowledge_gaps.values())
        gap_pressure = float(np.clip(safe_mean(gap_values), 0.0, 1.0)) if gap_values else 0.0
        gap_coverage = len(knowledge_gaps) / max(1, total_opportunities) if total_opportunities else 0.0
        hallucination_rate = len(self.ai_uncertainty_signals["hallucination_events"]) / max(1, self._ai_signal_history)
        knowledge_gap_term = 1.0 - np.clip(knowledge_norm, 0.0, 1.0)

        # ACTOR IGNORANCE: Linear additive formula (consistent with other dimensions)
        # Base level + weighted drivers that can accumulate or decline
        # Theoretical rationale: Ignorance can spike during paradigm shifts or collapse
        # with learning breakthroughs - sigmoid would artificially constrain these dynamics

        actor_level = float(np.clip(
            0.25                                # base level
            + 0.28 * knowledge_gap_term         # knowledge gaps increase ignorance
            + 0.18 * gap_pressure               # pressure from gaps
            + 0.12 * gap_coverage               # breadth of gaps
            + 0.15 * hallucination_rate         # AI hallucinations increase ignorance
            - 0.10 * share_explore              # exploration reduces ignorance
            + 0.03 * share_maintain,            # maintaining slightly increases (not learning)
            0.0, 1.0
        ))

        # Baseline (no AI) actor ignorance
        actor_level_no_ai = float(np.clip(
            0.25
            + 0.28 * knowledge_gap_term
            + 0.18 * gap_pressure
            + 0.12 * gap_coverage
            - 0.10 * share_explore
            + 0.03 * share_maintain,
            0.0, 1.0
        ))
        ai_effects["ai_ignorance_delta"] = actor_level - actor_level_no_ai
        self._update_short_term("actor_ignorance", actor_level)
        state["actor_ignorance"] = {
            "level": actor_level,
            "knowledge_gaps": knowledge_gaps,
            "gap_pressure": gap_pressure,
            "volatility": volatility,
            "ai_delta": ai_effects.get("ai_ignorance_delta"),
        }

        timing_levels = []
        timing_pressure = {}
        if market is not None and getattr(market, "opportunities", None):
            for opp in market.opportunities:
                acceleration = getattr(opp, "market_share", 0.0) * getattr(opp, "competition", 0.0)
                timing_pressure[getattr(opp, "id", f"opp_{id(opp)}")] = acceleration
                timing_levels.append(acceleration)
        base_practical = safe_mean(timing_levels) if timing_levels else 0.3
        timing_variability = float(np.std(timing_levels)) if timing_levels else 0.0
        market_volatility = float(getattr(market, "volatility", 0.0)) if market is not None else 0.0
        regime = getattr(market, "market_regime", "normal") if market is not None else "normal"
        regime_uncertainty = {
            "boom": 0.35,
            "growth": 0.22,
            "normal": 0.18,
            "volatile": 0.32,
            "recession": 0.45,
            "crisis": 0.65,
        }.get(regime, 0.2)
        path_component = float(np.clip(timing_variability * 1.5, 0.0, 1.0))
        crowding_metrics = getattr(market, "_last_crowding_metrics", {}) if market is not None else {}
        crowding_index = float(crowding_metrics.get("crowding_index", 0.0))
        ai_usage_pressure = float(crowding_metrics.get("ai_usage_share", 0.0))
        crowding_baseline = 0.25
        crowding_pressure = max(0.0, crowding_index - crowding_baseline)
        ai_pressure_term = max(0.0, ai_usage_pressure - 0.30)
        crowd_weight = float(getattr(self.config, "UNCERTAINTY_CROWDING_WEIGHT", 0.12))
        competitive_weight = float(getattr(self.config, "UNCERTAINTY_COMPETITIVE_WEIGHT", 0.08))
        herding_weight = float(getattr(self.config, "UNCERTAINTY_AI_HERDING_WEIGHT", 0.1))
        practical_level = float(
            np.clip(
                0.25
                + 0.30 * market_volatility
                + 0.25 * regime_uncertainty
                + 0.25 * np.clip(base_practical, 0.0, 1.0)
                + 0.20 * path_component
                + 0.15 * share_invest
                + crowd_weight * crowding_pressure
                + competitive_weight * ai_pressure_term
                + herding_weight * ai_herding_intensity,
                0.0,
                1.0,
            )
        )
        practical_level_no_ai = float(
            np.clip(
                0.25
                + 0.30 * market_volatility
                + 0.25 * regime_uncertainty
                + 0.25 * np.clip(base_practical, 0.0, 1.0)
                + 0.20 * path_component
                + 0.15 * share_invest,
                0.0,
                1.0,
            )
        )
        ai_effects["ai_indeterminism_delta"] = practical_level - practical_level_no_ai
        self._update_short_term("practical_indeterminism", practical_level)
        self.practical_indeterminism_state["path_volatility"] = path_component
        self.practical_indeterminism_state["timing_variability"] = timing_variability
        self.practical_indeterminism_state["regime_instability"] = regime_uncertainty
        self.practical_indeterminism_state["crowding_pressure"] = crowding_pressure
        self.practical_indeterminism_state["ai_pressure"] = ai_usage_pressure
        self.practical_indeterminism_state["ai_herding_intensity"] = ai_herding_intensity
        state["practical_indeterminism"] = {
            "level": practical_level,
            "timing_pressure": timing_pressure,
            "volatility": volatility,
            "crowding_pressure": crowding_pressure,
            "ai_herding_intensity": ai_herding_intensity,
            "ai_delta": ai_effects.get("ai_indeterminism_delta"),
        }

        history = list(self._action_history)
        total_innov = sum(item['innovate'] for item in history[-5:]) if history else 0
        total_explore = sum(item['explore'] for item in history[-5:]) if history else 0
        total_invest = sum(item['invest'] for item in history[-5:]) if history else 0
        combo_rate = (sum(item['new_combos'] for item in history[-5:]) / max(1, total_innov)) if history else 0.0
        niche_rate = (sum(item['new_niches'] for item in history[-5:]) / max(1, total_explore)) if history else 0.0
        adoption_rate = (sum(item['derivative_adoption'] for item in history[-5:]) / max(1, total_invest)) if history else 0.0
        combo_hhi_avg = safe_mean([item['combo_hhi'] for item in history[-5:]]) if history else 0.0
        sector_hhi_avg = safe_mean([item['sector_hhi'] for item in history[-5:]]) if history else 0.0
        diversity_term = float(np.clip(1.0 - combo_hhi_avg, 0.0, 1.0))
        sector_diversity = float(np.clip(1.0 - sector_hhi_avg, 0.0, 1.0))
        scarcity_signal = self._compute_component_scarcity()
        innovation_intensity = innovation_births / max(1, total_actions)
        disruption_state = self.agentic_novelty_state.setdefault("disruption_potential", {})
        if not disruption_state and hasattr(self.config, "SECTORS"):
            for sector in getattr(self.config, "SECTORS", []):
                disruption_state[sector] = 0.3
        disruption_decay = float(getattr(self.config, "DISRUPTION_STATE_DECAY", 0.92))
        for sector in list(disruption_state.keys()):
            disruption_state[sector] = float(np.clip(disruption_state[sector] * disruption_decay, 0.05, 0.95))
        for innovation in innovations:
            sector = getattr(innovation, "sector", None)
            if not sector:
                continue
            success_flag = 1.0 if getattr(innovation, "success", False) else 0.0
            novelty_score = float(getattr(innovation, "novelty", 0.0) or 0.0)
            impact_signal = float(getattr(innovation, "market_impact", 0.0) or 0.0)
            bump = 0.03 + 0.04 * novelty_score + 0.02 * max(impact_signal, 0.0) + 0.04 * success_flag
            disruption_state[sector] = float(np.clip(disruption_state.get(sector, 0.3) + bump, 0.05, 0.95))
        for action in agent_actions:
            if action.get("action") != "explore":
                continue
            sector = action.get("base_sector") or action.get("domain")
            if not sector:
                continue
            bump = 0.02 if (action.get("created_niche") or action.get("discovered_niche")) else 0.01
            disruption_state[sector] = float(np.clip(disruption_state.get(sector, 0.3) + bump, 0.05, 0.95))
        disruption_avg = float(np.clip(fast_mean(list(disruption_state.values())) if disruption_state else 0.3, 0.0, 1.0))
        innovation_momentum = np.clip(innovation_intensity + new_possibility_rate, 0.0, 1.0)
        reuse_pressure = 0.6 * combo_hhi_avg + 0.4 * sector_hhi_avg
        # AGENTIC NOVELTY: Linear additive formula (consistent with other dimensions)
        # Theoretical rationale: Novelty can collapse (combinatorial exhaustion) or
        # explode (paradigm shifts) - sigmoid would artificially prevent these valid extremes

        novelty_diagnostics = {
            "combo_rate": combo_rate,
            "niche_rate": niche_rate,
            "adoption_rate": adoption_rate,
            "diversity_term": diversity_term,
            "sector_diversity": sector_diversity,
            "scarcity": scarcity_signal,
            "innovation_intensity": innovation_intensity,
            "new_possibility_rate": new_possibility_rate,
            "reuse_pressure": reuse_pressure,
            "disruption_avg": disruption_avg,
        }
        disruption_state = self.agentic_novelty_state.setdefault("disruption_potential", {})
        self.agentic_novelty_state["disruption_potential"] = disruption_state

        # AI quality component
        ai_quality = float(np.clip(ai_usage_pressure, 0.0, 1.0))
        ai_novelty_uplift = float(getattr(self.config, "AI_NOVELTY_UPLIFT", 0.08))

        # Linear additive: drivers that increase novelty potential minus drags
        agentic_level = float(np.clip(
            0.25                                # base level
            + 0.18 * combo_rate                 # new combinations increase novelty
            + 0.15 * new_possibility_rate       # rate of new possibilities
            + 0.12 * niche_rate                 # niche discovery
            + 0.10 * adoption_rate              # adoption of innovations
            + 0.10 * innovation_intensity       # innovation activity
            + 0.08 * disruption_avg             # disruption potential
            + 0.10 * scarcity_signal            # component scarcity (novel combos harder)
            + 0.08 * sector_diversity           # diversity enables novelty
            - 0.20 * reuse_pressure             # reuse reduces novelty (but kept positive for balance)
            + ai_novelty_uplift * ai_quality,   # AI effect on novelty
            0.0, 1.0
        ))

        agentic_level_no_ai = float(np.clip(
            0.25
            + 0.18 * combo_rate
            + 0.15 * new_possibility_rate
            + 0.12 * niche_rate
            + 0.10 * adoption_rate
            + 0.10 * innovation_intensity
            + 0.08 * disruption_avg
            + 0.10 * scarcity_signal
            + 0.08 * sector_diversity
            - 0.20 * reuse_pressure,
            0.0, 1.0
        ))
        self._update_short_term("agentic_novelty", agentic_level)
        self.agentic_novelty_state["creative_momentum"] = agentic_level
        self.agentic_novelty_state["new_possibilities"] = new_possibilities
        self.agentic_novelty_state["new_possibility_rate"] = new_possibility_rate
        self.agentic_novelty_state["innovation_births"] = innovation_births
        self.agentic_novelty_state["scarcity_signal"] = scarcity_signal
        self.agentic_novelty_state["recent_scarcity_signal"] = scarcity_signal
        self.agentic_novelty_state["novelty_level"] = agentic_level
        self.agentic_novelty_state["drivers"] = novelty_diagnostics
        self._novelty_diagnostics = novelty_diagnostics
        state["agentic_novelty"] = {
            "level": agentic_level,
            "novelty_potential": agentic_level,
            "new_possibilities": new_possibilities,
            "new_possibility_rate": new_possibility_rate,
            "innovation_births": innovation_births,
            "volatility": volatility,
            "scarcity_signal": scarcity_signal,
            "component_scarcity": scarcity_signal,
            "disruption_potential": disruption_state,
            "combo_rate": combo_rate,
            "reuse_pressure": reuse_pressure,
            "adoption_rate": adoption_rate,
            "drivers": novelty_diagnostics,
            "ai_delta": agentic_level - agentic_level_no_ai,
        }
        tier_combo_rate = action_summary.get("tier_combo_rate", {})
        tier_reuse_pressure = action_summary.get("tier_reuse_pressure", {})
        tier_new_poss_rate = action_summary.get("tier_new_possibility_rate", {})
        tier_adoption_rate = action_summary.get("tier_adoption_rate", {})
        tier_drivers = {
            "combo_rate": tier_combo_rate,
            "reuse_pressure": tier_reuse_pressure,
            "new_possibility_rate": tier_new_poss_rate,
            "adoption_rate": tier_adoption_rate,
        }
        self.agentic_novelty_state["tier_drivers"] = tier_drivers
        state["agentic_novelty"]["tier_combo_rate"] = tier_combo_rate
        state["agentic_novelty"]["tier_reuse_pressure"] = tier_reuse_pressure
        state["agentic_novelty"]["tier_new_possibility_rate"] = tier_new_poss_rate
        state["agentic_novelty"]["tier_adoption_rate"] = tier_adoption_rate

        invest_hhi = action_summary.get("invest_hhi", 0.0)
        premium_share = float(ai_shares[3]) if ai_shares.size else 0.0
        tier_reuse_map = action_summary.get("tier_reuse_pressure", {})
        # FIXED: Use average reuse across all tiers instead of singling out premium
        # Previously: premium_reuse = float(tier_reuse_map.get("premium", 0.0))
        # This created artificial bias against premium tier
        reuse_values = [float(v) for v in tier_reuse_map.values() if v is not None]
        avg_reuse = float(np.mean(reuse_values)) if reuse_values else 0.0
        knowledge_overlap = float(action_summary.get("sector_hhi", 0.0))
        agent_count = max(1, getattr(self.config, "N_AGENTS", 1))
        alive_agents = getattr(self, "_last_alive_agents", agent_count)
        population_factor = float(np.clip(alive_agents / agent_count, 0.15, 1.0))

        # Track AI action correlation for analysis (no direct effect on recursion)
        ai_correlation = float(action_summary.get("ai_action_correlation", 0.30))

        rw = getattr(self.config, "RECURSION_WEIGHTS", {}) or {}
        crowd_w = float(rw.get("crowd_weight", 0.35))
        vol_w = float(rw.get("volatility_weight", 0.30))
        herd_w = float(rw.get("ai_herd_weight", 0.40))
        premium_reuse_w = float(rw.get("premium_reuse_weight", 0.20))

        crowding_component = (
            0.45 * invest_hhi
            + 0.35 * premium_share
            + crowd_w * crowding_pressure
            + 0.12 * knowledge_overlap
        )
        scale = 0.5 + 0.5 * population_factor
        recursion_level = float(
            np.clip(
                crowding_component * scale
                + vol_w * volatility
                + herd_w * ai_herding_intensity
                + premium_reuse_w * avg_reuse,  # FIXED: tier-agnostic reuse
                0.0,
                1.0,
            )
        )
        recursion_level_no_ai = float(
            np.clip(
                crowding_component * scale
                + 0.15 * volatility,
                0.0,
                1.0,
            )
        )
        ai_effects["ai_recursion_delta"] = recursion_level - recursion_level_no_ai
        self._update_short_term("competitive_recursion", recursion_level)
        herding_pressure = {
            opp_id: count / max(1, action_summary.get("invest", 1))
            for opp_id, count in action_summary.get("herding_counts", {}).items()
        }
        invest_by_ai = action_summary.get("invest_by_ai", {})
        state["competitive_recursion"] = {
            "level": recursion_level,
            "herding_pressure": herding_pressure,
            "volatility": volatility,
            "ai_premium_share": premium_share,
            "ai_herding_intensity": ai_herding_intensity,
            "ai_action_correlation": ai_correlation,  # NEW: Track correlation metric
            "ai_delta": ai_effects.get("ai_recursion_delta"),
        }

        self.uncertainty_evolution.append((round_num, state))
        self._last_environment_state = state
        return state

    def perceive_uncertainty(
        self,
        agent_traits: Dict,
        visible_opportunities: List,
        market_conditions: Dict,
        ai_level: str = "none",
        ai_learning_profile: Optional[Any] = None,
        ai_analysis_history: Optional[List] = None,
        agent_resources: Optional[Any] = None,
        agent_knowledge: Optional[Set[str]] = None,
        recent_outcomes: Optional[List[Dict]] = None,
        cached_metrics: Optional[Dict[str, Any]] = None,
        agent_id: Optional[int] = None,
        action_history: Optional[List[str]] = None,
    ) -> Dict:
        """
        Compute an individual agent's subjective perception of uncertainty.

        While measure_uncertainty_state() computes objective environment-level
        uncertainty, this method models how individual agents perceive uncertainty
        based on their traits, knowledge, AI augmentation level, and experience.
        The gap between objective and perceived uncertainty is central to the
        paradox of future knowledge (Townsend et al., 2025).

        Parameters
        ----------
        agent_traits : Dict
            Agent's psychological traits including uncertainty_tolerance,
            competence, ai_trust, exploration_tendency, and innovativeness.
        visible_opportunities : List
            Opportunities currently visible to the agent.
        market_conditions : Dict
            Current market state including regime, volatility, and crowding.
        ai_level : str, default "none"
            Agent's current AI augmentation tier: "none", "basic", "advanced",
            or "premium". Higher tiers provide better information quality but
            may also amplify certain uncertainty dimensions.
        ai_learning_profile : AILearningProfile, optional
            Agent's learned understanding of AI capabilities and trustworthiness
            across different domains.
        ai_analysis_history : List, optional
            History of AI analyses received by the agent.
        agent_resources : AgentResources, optional
            Agent's current resource state including capital and knowledge.
        agent_knowledge : Set[str], optional
            Set of knowledge component IDs known to the agent.
        recent_outcomes : List[Dict], optional
            Recent investment/innovation outcomes for experience-based learning.
        cached_metrics : Dict, optional
            Pre-computed metrics to avoid redundant calculation.
        agent_id : int, optional
            Agent identifier for agent-specific short-term buffering.
        action_history : List[str], optional
            Agent's recent action history for behavioral pattern analysis.

        Returns
        -------
        Dict
            Agent's subjective uncertainty perception across dimensions:

            actor_ignorance : dict
                Agent's perceived knowledge gaps. AI with high info_quality
                reduces perceived ignorance, but hallucinations and
                overconfidence can create false certainty.

            practical_indeterminism : dict
                Agent's perceived execution risk and timing uncertainty.
                AI may reduce perceived indeterminism while objective
                indeterminism increases due to competitive dynamics.

            agentic_novelty : dict
                Agent's perceived creative potential. Tier-specific combo
                rates and reuse pressures affect how agents perceive the
                novelty frontier differently based on their AI usage.

            competitive_recursion : dict
                Agent's awareness of strategic interdependence. Higher AI
                tiers may provide better competitive intelligence but also
                contribute to herding through correlated recommendations.

            decision_confidence : float
                Overall confidence level for decision-making, integrating
                all uncertainty dimensions and AI trust calibration.

        Notes
        -----
        The perception process implements several key theoretical mechanisms:

        1. **Information Quality Effect**: Higher AI tiers reduce perceived
           actor ignorance through better info_quality and info_breadth.

        2. **Trust Calibration**: Agent's learned AI trust (from ai_learning_profile)
           modulates how much AI information reduces perceived uncertainty.

        3. **Tier-Specific Novelty**: Different AI tiers exhibit different
           rates of novel combination creation vs. reuse, affecting perceived
           agentic novelty.

        4. **Herding Awareness**: Agents perceive competitive recursion based
           on observable herding patterns and their own AI tier's contribution.

        References
        ----------
        Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).
            Are the futures computable? Knightian uncertainty & artificial
            intelligence. Academy of Management Review, 50(2), 415-440.
        """
        perception: Dict[str, Dict] = {}
        env_state = getattr(self, "_last_environment_state", {})
        current_round = market_conditions.get("round", 0)
        volatility = float(self._volatility_state.get("volatility_metric", 0.0))
        action_shares = self._volatility_state.get("last_action_shares")
        if action_shares is None or len(action_shares) != 4:
            share_invest = share_innovate = share_explore = share_maintain = 0.0
        else:
            share_invest, share_innovate, share_explore, share_maintain = map(float, action_shares)
        agent_counts = collections.Counter(action_history or [])
        agent_total = sum(agent_counts.values())
        if agent_total > 0:
            agent_share_invest = agent_counts.get("invest", 0) / agent_total
            agent_share_innovate = agent_counts.get("innovate", 0) / agent_total
            agent_share_explore = agent_counts.get("explore", 0) / agent_total
            agent_share_maintain = agent_counts.get("maintain", 0) / agent_total
        else:
            agent_share_invest = share_invest
            agent_share_innovate = share_innovate
            agent_share_explore = share_explore
            agent_share_maintain = share_maintain
        blend = 0.85
        share_invest = blend * agent_share_invest + (1.0 - blend) * share_invest
        share_innovate = blend * agent_share_innovate + (1.0 - blend) * share_innovate
        share_explore = blend * agent_share_explore + (1.0 - blend) * share_explore
        share_maintain = blend * agent_share_maintain + (1.0 - blend) * share_maintain
        ai_config = self.config.AI_LEVELS.get(ai_level, self.config.AI_LEVELS["none"])
        info_quality = ai_config.get("info_quality", 0.0)
        info_breadth = ai_config.get("info_breadth", 0.0)

        last_ai_shares = self._volatility_state.get("last_ai_shares")
        if last_ai_shares is None or len(last_ai_shares) != 4:
            ai_shares = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            ai_shares = np.asarray(last_ai_shares, dtype=float)

        known_count = len(visible_opportunities)
        discovery_window = 10
        recent_discoveries = 0
        if known_count and current_round > discovery_window:
            recent_discoveries = sum(
                1
                for opp in visible_opportunities
                if hasattr(opp, "discovery_round")
                and opp.discovery_round is not None
                and opp.discovery_round >= current_round - discovery_window
            ) / known_count
        discovery_momentum = self.opportunity_discovery_rate
        if known_count:
            self.opportunity_discovery_rate = float(
                np.clip(0.8 * self.opportunity_discovery_rate + 0.2 * recent_discoveries, 0.0, 1.0)
            )
        else:
            self.opportunity_discovery_rate = float(np.clip(self.opportunity_discovery_rate * 0.92, 0.0, 1.0))

        sector_knowledge = getattr(agent_resources, "knowledge", {}) if agent_resources is not None else {}
        cache = cached_metrics if cached_metrics is not None else None
        avg_knowledge_level = None
        if cache is not None:
            avg_knowledge_level = cache.get("avg_knowledge_level")
        if avg_knowledge_level is None:
            if isinstance(sector_knowledge, dict) and sector_knowledge:
                avg_knowledge_level = fast_mean(sector_knowledge.values())
                if not np.isfinite(avg_knowledge_level):
                    avg_knowledge_level = 0.0
            else:
                avg_knowledge_level = 0.0
            avg_knowledge_level = float(avg_knowledge_level)
            if cache is not None:
                cache["avg_knowledge_level"] = avg_knowledge_level
        else:
            avg_knowledge_level = float(avg_knowledge_level) if np.isfinite(avg_knowledge_level) else 0.0
            if cache is not None:
                cache["avg_knowledge_level"] = avg_knowledge_level
        knowledge_deficit = None
        if cache is not None:
            knowledge_deficit = cache.get("knowledge_deficit")
        if knowledge_deficit is None or not np.isfinite(knowledge_deficit):
            knowledge_deficit = max(0.0, 1.0 - avg_knowledge_level)
            if cache is not None:
                cache["knowledge_deficit"] = knowledge_deficit
        else:
            knowledge_deficit = float(knowledge_deficit)
        knowledge_deficit = float(np.clip(knowledge_deficit - 0.2 * info_breadth, 0.0, 1.2))

        base_ai_trust = agent_traits.get("ai_trust", 0.5)
        avg_ai_trust = base_ai_trust
        hallucination_rate = 0.0
        if ai_learning_profile is not None:
            trusts = list(ai_learning_profile.domain_trust.values())
            if trusts:
                avg_ai_trust = float(np.clip(np.mean(trusts), 0.0, 1.0))
            total_hall = sum(ai_learning_profile.hallucination_experiences.values())
            total_usage = max(1, sum(ai_learning_profile.usage_count.values()))
            hallucination_rate = float(np.clip(total_hall / total_usage, 0.0, 1.0))
        avg_ai_trust = float(np.clip(0.5 * base_ai_trust + 0.5 * avg_ai_trust, 0.0, 1.0))

        competence_trait = float(agent_traits.get("competence", 0.5))
        ai_trust_trait = float(agent_traits.get("ai_trust", 0.5))
        exploration_trait = float(agent_traits.get("exploration_tendency", 0.5))
        innovation_trait = float(agent_traits.get("innovativeness", 0.5))
        uncertainty_trait = float(agent_traits.get("uncertainty_tolerance", 0.5))

        personal_knowledge = 0.0
        knowledge_span = 0.0
        agent_knowledge_ids = set(agent_knowledge or [])
        if agent_resources is not None:
            knowledge_vals = list(getattr(agent_resources, "knowledge", {}).values())
            if knowledge_vals:
                personal_knowledge = float(np.clip(np.mean(knowledge_vals), 0.0, 1.0))
            knowledge_span = len(getattr(agent_resources, "knowledge", {})) / max(1, len(self.config.SECTOR_PROFILES))
        personal_knowledge = float(np.clip(personal_knowledge + info_quality * 0.15, 0.0, 1.0))
        knowledge_span = float(np.clip(knowledge_span + info_breadth * 0.2, 0.0, 1.0))
        if self.knowledge_base is not None and agent_knowledge_ids:
            total_pieces = max(1, len(self.knowledge_base.knowledge_pieces))
            knowledge_span = max(knowledge_span, len(agent_knowledge_ids) / total_pieces)

        discovery_shortfall = max(0.0, 1.0 - min(1.0, discovery_momentum + 0.2 * share_explore))
        base_ignorance = (
            0.25 * (1.0 - info_quality)
            + 0.2 * (1.0 - info_breadth)
            + 0.35 * knowledge_deficit
            + 0.2 * discovery_shortfall
        )
        ignorance_reduction = info_quality * (0.12 + 0.25 * avg_ai_trust) + info_breadth * 0.08
        estimated_ignorance = float(
            np.clip(
                base_ignorance
                - ignorance_reduction
                - 0.08 * share_explore
                - 0.05 * volatility
                + 0.05 * share_maintain,
                0.0,
                1.0,
            )
        )
        estimated_ignorance = float(
            np.clip(
                estimated_ignorance
                - 0.4 * personal_knowledge
                - 0.15 * knowledge_span
                - 0.05 * ai_trust_trait
                + 0.15 * (1.0 - competence_trait),
                0.01,
                0.99,
            )
        )
        knowledge_gaps = {}
        analytical_modifier = 1 - agent_traits.get("analytical_ability", 0.5) * 0.4

        for opp in visible_opportunities:
            sector = getattr(opp, "sector", None)
            sector_familiarity = sector_knowledge.get(sector, 0.0) if isinstance(sector_knowledge, dict) else 0.0
            component_ids = []
            if hasattr(opp, "knowledge_components") and opp.knowledge_components:
                component_ids = list(opp.knowledge_components)
            elif hasattr(opp, "combination_signature") and opp.combination_signature:
                component_ids = opp.combination_signature.split("||")
            if component_ids and agent_knowledge_ids:
                familiar_components = len(agent_knowledge_ids.intersection(component_ids))
                component_familiarity = familiar_components / max(1, len(component_ids))
            else:
                component_familiarity = sector_familiarity
            base_gap = 1.0 - max(sector_familiarity, component_familiarity)
            gap = max(0.0, min(1.0, base_gap * analytical_modifier))
            gap *= max(0.0, 1 - info_quality * 0.5)
            if gap > 0.02:
                knowledge_gaps[getattr(opp, "id", f"opp_{id(opp)}")] = gap

        tier_key = normalize_ai_label(ai_level)
        tier_factor = self._tier_smoothing.get(tier_key, 1.0)

        short_actor = self._get_short_term_average("actor_ignorance", estimated_ignorance, agent_id=agent_id)
        quality_term = 0.3 * info_quality + 0.2 * info_breadth
        actor_mix_base = 0.45 + quality_term - 0.25 * knowledge_deficit
        short_weight = float(np.clip(0.32 - 0.28 * info_quality + 0.12 * tier_factor, 0.03, 0.45))
        actor_mix = float(np.clip(actor_mix_base, 0.05, 0.95))
        ignorance_level = (1.0 - short_weight) * actor_mix * estimated_ignorance + short_weight * short_actor
        ignorance_level = float(np.clip(ignorance_level, 0.0, 1.0))
        gap_values = list(knowledge_gaps.values())
        gap_pressure = float(np.clip(fast_mean(gap_values), 0.0, 1.0)) if gap_values else 0.0

        knowledge_signal = {
            "confidence": float(max(0.0, 1.0 - ignorance_level)),
            "ignorance_level": float(max(0.0, ignorance_level)),
            "knowledge_gaps": knowledge_gaps,
            "gap_pressure": gap_pressure,
            "info_quality": info_quality,
            "info_breadth": info_breadth,
            "personal_knowledge": personal_knowledge,
            "knowledge_span": knowledge_span,
            "discovery_momentum": discovery_momentum,
            "volatility": volatility,
        }
        perception["knowledge_signal"] = knowledge_signal
        perception["actor_ignorance"] = {
            "ignorance_level": knowledge_signal["ignorance_level"],
            "knowledge_gaps": knowledge_gaps,
            "info_sufficiency": info_quality,
            "discovery_momentum": discovery_momentum,
            "volatility": volatility,
            "gap_pressure": gap_pressure,
        }
        self._update_short_term("actor_ignorance", knowledge_signal["ignorance_level"], agent_id=agent_id)

        market_volatility = market_conditions.get("volatility", 0.2)
        regime = market_conditions.get("regime", "normal")
        regime_uncertainty = {"boom": 0.3, "growth": 0.2, "normal": 0.1, "recession": 0.4, "crisis": 0.7}.get(regime, 0.2)
        regime_uncertainty = float(np.clip(regime_uncertainty * (1 - 0.3 * avg_ai_trust) + hallucination_rate * 0.2, 0.0, 1.0))
        visible_sectors = set(getattr(opp, "sector", None) or "unknown" for opp in visible_opportunities)
        known_sectors = 0
        if isinstance(sector_knowledge, dict) and sector_knowledge:
            known_sectors = sum(1 for value in sector_knowledge.values() if value >= 0.4)
        path_complexity = max(0.0, len(visible_sectors) - known_sectors) / max(1, len(self.config.SECTORS))

        cached_timing = cache.get("timing_pressure") if cache is not None else None
        timing_pressure: Dict[str, float]
        avg_timing_pressure = cache.get("avg_timing_pressure") if cache is not None else None
        if cached_timing is not None:
            timing_pressure = dict(cached_timing)
        else:
            timing_pressure = {}

        if cached_timing is None:
            if visible_opportunities:
                for opp in visible_opportunities:
                    lifecycle_stage = getattr(opp, "lifecycle_stage", "emerging")
                    competition = getattr(opp, "competition", 0.0)
                    adoption_rate = getattr(opp, "market_share", 0.0)
                    urgency = competition * 0.4 + adoption_rate * 0.3
                    if lifecycle_stage == "declining":
                        urgency *= 0.3
                    elif lifecycle_stage == "mature":
                        urgency *= 0.6
                    elif lifecycle_stage == "growing":
                        urgency *= 1.2
                    timing_pressure[getattr(opp, "id", f"opp_{id(opp)}")] = urgency
            avg_timing_pressure = fast_mean(timing_pressure.values()) if timing_pressure else 0.0
            if not np.isfinite(avg_timing_pressure):
                avg_timing_pressure = 0.0
            if cache is not None:
                cache["timing_pressure"] = dict(timing_pressure)
                cache["avg_timing_pressure"] = avg_timing_pressure
        else:
            if avg_timing_pressure is None or not np.isfinite(avg_timing_pressure):
                avg_timing_pressure = fast_mean(timing_pressure.values()) if timing_pressure else 0.0
            if not np.isfinite(avg_timing_pressure):
                avg_timing_pressure = 0.0
            if cache is not None:
                cache["avg_timing_pressure"] = avg_timing_pressure
        path_component = float(np.clip(path_complexity, 0.0, 1.0))
        crowding_pressure = float(self.practical_indeterminism_state.get("crowding_pressure", 0.0))
        ai_pressure_level = float(self.practical_indeterminism_state.get("ai_pressure", 0.0))
        ai_herding_intensity = float(self.practical_indeterminism_state.get("ai_herding_intensity", 0.0))
        volatility_crowd_weight = float(getattr(self.config, "UNCERTAINTY_CROWDING_WEIGHT", 0.12))
        volatility_comp_weight = float(getattr(self.config, "UNCERTAINTY_COMPETITIVE_WEIGHT", 0.08))
        ai_pressure_term = max(0.0, ai_pressure_level - 0.30)

        agent_path_risk = 0.18 + 0.22 * gap_pressure + 0.12 * share_invest + 0.10 * max(0.0, share_innovate - share_explore)
        agent_volatility = float(np.clip(abs(share_invest - share_maintain) + abs(share_innovate - share_explore), 0.0, 1.0))
        agent_uncertainty = (
            0.12 * (1.0 - competence_trait)
            + 0.08 * (0.6 - personal_knowledge)
            + 0.05 * max(0.0, 0.55 - ai_trust_trait)
            + 0.08 * discovery_shortfall
        )
        system_term = (
            0.08 * market_volatility
            + 0.07 * regime_uncertainty
            + 0.06 * avg_timing_pressure
            + 0.05 * path_component
            + 0.05 * volatility
        )
        crowd_term = (
            volatility_crowd_weight * max(0.0, crowding_pressure) * 0.5
            + volatility_comp_weight * ai_pressure_term * 0.5
            + 0.05 * ai_herding_intensity
        )
        indeterminism_value = agent_path_risk + 0.15 * agent_volatility + agent_uncertainty + system_term + crowd_term
        indeterminism_value = float(np.clip(indeterminism_value, 0.0, 1.0))
        short_practical = self._get_short_term_average("practical_indeterminism", indeterminism_value, agent_id=agent_id)
        short_weight_practical = float(np.clip(0.34 - 0.30 * info_quality + 0.15 * tier_factor, 0.04, 0.5))
        practical_mix = float(np.clip(0.25 + 0.45 * info_quality + 0.25 * info_breadth - 0.2 * tier_factor, 0.1, 0.9))
        indeterminism_level = float(
            np.clip(
                (1.0 - short_weight_practical) * practical_mix * indeterminism_value
                + short_weight_practical * short_practical,
                0.0,
                1.0,
            )
        )
        execution_signal = {
            "risk_level": indeterminism_level,
            "timing_pressure": timing_pressure,
            "market_regime_risk": regime_uncertainty,
            "volatility": volatility,
            "crowding_pressure": crowding_pressure,
            "ai_pressure": ai_pressure_level,
            "ai_herding_intensity": ai_herding_intensity,
        }
        perception["execution_risk"] = execution_signal
        practical_indeterminism_payload = {
            "indeterminism_level": indeterminism_level,
            "timing_criticality": timing_pressure,
            "market_regime_risk": regime_uncertainty,
            "volatility": volatility,
            "crowding_pressure": crowding_pressure,
            "ai_pressure": ai_pressure_level,
            "ai_herding_intensity": ai_herding_intensity,
        }
        # Backward-compatible aliases for existing consumers
        practical_indeterminism_payload["timing_pressure"] = timing_pressure
        practical_indeterminism_payload["regime_stability"] = regime_uncertainty
        perception["practical_indeterminism"] = practical_indeterminism_payload
        self._update_short_term("practical_indeterminism", indeterminism_level, agent_id=agent_id)

        recent_new_possibility_rate = float(self.agentic_novelty_state.get("new_possibility_rate", 0.0))
        recent_innovation_births = int(self.agentic_novelty_state.get("innovation_births", 0))
        recent_new_possibilities = int(self.agentic_novelty_state.get("new_possibilities", 0))
        scarcity_signal = float(self.agentic_novelty_state.get("recent_scarcity_signal", self.agentic_novelty_state.get("scarcity_signal", 0.5)))

        novelty_level = float(self.agentic_novelty_state.get("novelty_level", 0.5))
        driver_snapshot = self.agentic_novelty_state.get("drivers", {}) or {}
        base_combo_rate = float(driver_snapshot.get("combo_rate", 0.0))
        base_reuse_pressure = float(driver_snapshot.get("reuse_pressure", 0.0))
        base_new_poss_rate = float(driver_snapshot.get("new_possibility_rate", recent_new_possibility_rate))
        base_adoption_rate = float(driver_snapshot.get("adoption_rate", 0.0))
        tier_driver_map = self.agentic_novelty_state.get("tier_drivers", {}) or {}
        tier_combo_rate = float(
            tier_driver_map.get("combo_rate", {}).get(ai_level, base_combo_rate)
        )
        tier_reuse_pressure = float(
            tier_driver_map.get("reuse_pressure", {}).get(ai_level, base_reuse_pressure)
        )
        tier_new_poss_rate = float(
            tier_driver_map.get("new_possibility_rate", {}).get(ai_level, base_new_poss_rate)
        )
        tier_adoption_rate = float(
            tier_driver_map.get("adoption_rate", {}).get(ai_level, base_adoption_rate)
        )
        novelty_adjustment = (
            0.25 * (tier_combo_rate - base_combo_rate)
            - 0.2 * (tier_reuse_pressure - base_reuse_pressure)
            + 0.15 * (tier_new_poss_rate - base_new_poss_rate)
            + 0.1 * (tier_adoption_rate - base_adoption_rate)
        )
        novelty_level_agent = float(np.clip(novelty_level + novelty_adjustment, 0.05, 0.95))
        novelty_level_agent = float(
            np.clip(
                novelty_level_agent
                + 0.22 * (innovation_trait - 0.5)
                + 0.15 * (exploration_trait - 0.5)
                + 0.10 * personal_knowledge,
                0.0,
                1.0,
            )
        )
        short_novelty = self._get_short_term_average("agentic_novelty", novelty_level_agent, agent_id=agent_id)
        short_weight_novelty = float(np.clip(0.30 - 0.26 * info_quality + 0.12 * tier_factor, 0.03, 0.45))
        novelty_mix = float(np.clip(0.2 + 0.4 * info_quality + 0.25 * info_breadth - 0.15 * tier_factor, 0.1, 0.95))
        novelty_level_agent = float(
            np.clip(
                (1.0 - short_weight_novelty) * novelty_mix * novelty_level_agent
                + short_weight_novelty * short_novelty,
                0.0,
                1.0,
            )
        )
        creative_momentum = novelty_level_agent
        disruption_potential = dict(self.agentic_novelty_state.get("disruption_potential", {}))
        if not disruption_potential and hasattr(self.config, "SECTORS"):
            disruption_potential = {sector: 0.3 for sector in getattr(self.config, "SECTORS", [])}

        innovation_signal = {
            "novelty_potential": novelty_level_agent,
            "creative_confidence": creative_momentum,
            "component_scarcity": scarcity_signal,
            "new_possibility_rate": recent_new_possibility_rate,
            "innovation_births": recent_innovation_births,
            "new_possibilities": recent_new_possibilities,
            "combo_rate": tier_combo_rate,
            "reuse_pressure": tier_reuse_pressure,
            "adoption_rate": tier_adoption_rate,
        }
        perception["innovation_signal"] = innovation_signal
        perception["agentic_novelty"] = {
            "novelty_potential": novelty_level_agent,
            "creative_confidence": creative_momentum,
            "disruption_potential": disruption_potential,
            "volatility": volatility,
            "new_possibility_rate": recent_new_possibility_rate,
            "innovation_births": recent_innovation_births,
            "new_possibilities": recent_new_possibilities,
            "component_scarcity": scarcity_signal,
            "drivers": driver_snapshot,
            "combo_rate": tier_combo_rate,
            "reuse_pressure": tier_reuse_pressure,
            "adoption_rate": tier_adoption_rate,
            "tier_combo_rate": tier_combo_rate,
            "tier_reuse_pressure": tier_reuse_pressure,
            "tier_new_possibility_rate": tier_new_poss_rate,
            "tier_adoption_rate": tier_adoption_rate,
            "global_combo_rate": base_combo_rate,
            "global_reuse_pressure": base_reuse_pressure,
            "tier_novelty_adjustment": novelty_adjustment,
        }
        self._update_short_term("agentic_novelty", novelty_level_agent, agent_id=agent_id)

        if hasattr(self, "ai_uncertainty_signals"):
            hallucination_events = len(self.ai_uncertainty_signals.get("hallucination_events", []))
            confidence_miscalibration = safe_mean(
                self.ai_uncertainty_signals.get("confidence_miscalibration", [0.0])
            )
        else:
            hallucination_events = 0
            confidence_miscalibration = 0.0

        herding_patterns = self.ai_uncertainty_signals.get("ai_herding_patterns", {})
        herding_pressure = {}
        for opp in visible_opportunities:
            demand = opportunity_demand = getattr(opp, "competition", 0.0)
            herding_pressure[getattr(opp, "id", f"opp_{id(opp)}")] = float(np.clip(demand, 0.0, 1.0))

        pressure_vals = list(herding_pressure.values())
        avg_pressure = fast_mean(pressure_vals) if pressure_vals else 0.0
        if not np.isfinite(avg_pressure):
            avg_pressure = 0.0
        if not np.isfinite(confidence_miscalibration):
            confidence_miscalibration = 0.0
        ai_herding_intensity = float(self.practical_indeterminism_state.get("ai_herding_intensity", 0.0))
        capital_crowding = float(market_conditions.get("crowding_metrics", {}).get("crowding_index", 0.0))
        recursion_raw = (
            0.12
            + avg_pressure * 0.45
            + confidence_miscalibration * 0.18
            + hallucination_rate * 0.12
            + volatility * 0.08
            + share_invest * 0.08
            + ai_herding_intensity * 0.15
            + max(0.0, capital_crowding) * 0.08
        )
        recursion_raw += (
            0.10 * max(0.0, knowledge_span - 0.35)
            - 0.10 * (exploration_trait - 0.5)
            + 0.08 * max(0.0, 0.55 - personal_knowledge)
        )
        recursion_level = float(np.clip(recursion_raw, 0.0, 1.0))
        strategic_opacity = float(
            np.clip(
                self.competitive_recursion_state.get("strategic_opacity", 0.0) * 0.9
                + hallucination_events * 0.03
                + avg_pressure * 0.1
                + volatility * 0.1,
                0.0,
                1.0,
            )
        )
        self.competitive_recursion_state["strategic_opacity"] = strategic_opacity
        short_recursion = self._get_short_term_average("competitive_recursion", recursion_level, agent_id=agent_id)
        recursion_mix = float(np.clip(0.25 + 0.45 * info_quality + 0.25 * info_breadth - 0.2 * tier_factor, 0.15, 0.9))
        recursion_short_weight = float(np.clip(0.32 - 0.28 * info_quality + 0.15 * tier_factor, 0.04, 0.5))
        recursion_level = float(
            np.clip(
                (1.0 - recursion_short_weight) * recursion_mix * recursion_level + recursion_short_weight * short_recursion,
                0.0,
                1.0,
            )
        )
        ai_usage_share = float(1.0 - ai_shares[0]) if ai_shares.size else 0.0
        competition_signal = {
            "pressure_level": recursion_level,
            "herding_awareness": avg_pressure,
            "herding_pressure": herding_pressure,
            "ai_usage_share": ai_usage_share,
            "capital_crowding": capital_crowding,
            "ai_herding_intensity": ai_herding_intensity,
        }
        perception["competition_signal"] = competition_signal
        perception["competitive_recursion"] = {
            "recursion_level": recursion_level,
            "herding_awareness": avg_pressure,
            "herding_pressure": herding_pressure,
            "ai_herding_patterns": dict(herding_patterns),
            "ai_herding_intensity": ai_herding_intensity,
            "strategic_opacity": strategic_opacity,
            "volatility": volatility,
            "ai_usage_share": ai_usage_share,
        }
        self._update_short_term("competitive_recursion", recursion_level, agent_id=agent_id)
        perception["crowding_metrics"] = market_conditions.get("crowding_metrics", {}) or {}

        perception["volatility_metric"] = volatility
        perception["action_profile"] = {
            "invest": share_invest,
            "innovate": share_innovate,
            "explore": share_explore,
            "maintain": share_maintain,
        }

        total_uncertainty = (
            perception["actor_ignorance"]["ignorance_level"] * 0.35
            + perception["practical_indeterminism"]["indeterminism_level"] * 0.25
            + (1 - perception["agentic_novelty"]["novelty_potential"]) * 0.20
            + perception["competitive_recursion"]["recursion_level"] * 0.20
        )

        # FIXED: Increase coefficient to allow info_quality to meaningfully impact decision confidence
        # Previously 0.2 was too weak - Premium AI (info_quality=0.95) only gave 0.19 boost
        # Now Premium gives 0.38 boost vs None (0.45) giving 0.18 - proper differentiation
        ai_confidence_boost = info_quality * 0.4
        recent_success_rate = 0.5
        ai_success_rate = 0.5
        mean_roi = 0.0
        if recent_outcomes:
            invest_outcomes = [o for o in recent_outcomes if o.get("action") == "invest"]
            if invest_outcomes:
                recent_success_rate = sum(1 for o in invest_outcomes if o.get("success")) / len(invest_outcomes)
                ai_invest = [o for o in invest_outcomes if o.get("ai_used")]
                if ai_invest:
                    ai_success_rate = sum(1 for o in ai_invest if o.get("success")) / len(ai_invest)
                else:
                    ai_success_rate = recent_success_rate
                rois = []
                for outcome in invest_outcomes:
                    invested = outcome.get("investment_amount") or outcome.get("investment", {}).get("amount") or 0.0
                    returned = outcome.get("capital_returned", 0.0)
                    if invested > 0:
                        rois.append((returned - invested) / invested)
                if rois:
                    mean_roi = float(np.clip(np.mean(rois), -0.5, 0.5))

        competence = agent_traits.get("competence", 0.5)
        risk_tolerance = agent_traits.get("risk_tolerance", 0.5)
        raw_confidence = competence * np.exp(-total_uncertainty * 0.5) * (1 + ai_confidence_boost)
        raw_confidence = float(np.nan_to_num(raw_confidence, nan=0.5, posinf=0.5, neginf=0.5))
        experience_multiplier = np.clip(0.4 + 1.4 * recent_success_rate * competence, 0.2, 1.8)
        ai_effective_trust = float(np.clip(avg_ai_trust, 0.0, 1.0))
        ai_experience_multiplier = np.clip(0.4 + 1.4 * ai_success_rate * ai_effective_trust, 0.2, 1.8)
        roi_multiplier = np.clip(1 + mean_roi * (0.5 + risk_tolerance), 0.3, 1.7)
        raw_confidence *= experience_multiplier * ai_experience_multiplier * roi_multiplier
        raw_confidence = float(np.clip(raw_confidence, 0.01, 2.0))
        decision_confidence = stable_sigmoid(3.0 * (raw_confidence - 0.5))
        decision_confidence = float(np.nan_to_num(decision_confidence, nan=0.5))
        confidence_multiplier = 1 - 0.4 * hallucination_rate + 0.3 * avg_ai_trust
        decision_confidence *= float(np.clip(confidence_multiplier, 0.5, 1.3))
        decision_confidence = float(np.clip(decision_confidence, 0.02, 0.98))
        perception["decision_confidence"] = decision_confidence

        return perception
