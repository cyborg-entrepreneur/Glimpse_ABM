"""
Agent resources and decision logic for Glimpse ABM.

This module implements the entrepreneurial agent model, including resource
management, decision-making under uncertainty, and learning from outcomes.
The agent architecture is designed to capture key features of entrepreneurial
cognition and behavior under Knightian uncertainty with AI augmentation.

Theoretical Foundation
----------------------
The agent model operationalizes concepts from:

    Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).
    Are the futures computable? Knightian uncertainty & artificial intelligence.
    Academy of Management Review, 50(2), 415-440.

    Townsend, D. M., Hunt, R. A., & Rady, J. (2024). Chance, probability, &
    uncertainty at the edge of human reasoning: What is Knightian uncertainty?
    Strategic Entrepreneurship Journal, 18(3), 451-474.

Key theoretical mechanisms implemented:

1. **Uncertainty Response Profiles**: Agents learn heterogeneous responses
   to each of the four Knightian uncertainty dimensions based on their
   outcome history. This captures the adaptive nature of entrepreneurial
   cognition (Proposition 5 in Townsend et al., 2025).

2. **AI Tier Selection**: Agents dynamically select AI augmentation levels
   based on learned trust, cost considerations, and performance outcomes.
   Higher tiers offer better information quality but at higher costs and
   with potential side effects on competitive recursion.

3. **Action Selection Under Uncertainty**: The four-action framework
   (invest, innovate, explore, maintain) reflects the fundamental strategic
   choices entrepreneurs face, each with different uncertainty profiles:
   - Invest: Deploy capital into opportunities (practical indeterminism)
   - Innovate: Create new combinations (agentic novelty)
   - Explore: Reduce actor ignorance through search
   - Maintain: Preserve resources when uncertainty is high

4. **Performance Tracking**: ROIC and ROE metrics enable investigation of
   how AI augmentation affects entrepreneurial performance across action
   types and uncertainty conditions.

References
----------
Knight, F. H. (1921). Risk, uncertainty, and profit. Houghton Mifflin.

Sarasvathy, S. D. (2001). Causation and effectuation: Toward a theoretical
    shift from economic inevitability to entrepreneurial contingency.
    Academy of Management Review, 26(2), 243-263.

Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).
    Are the futures computable? Knightian uncertainty & artificial intelligence.
    Academy of Management Review, 50(2), 415-440.
"""

from __future__ import annotations

import collections
import math
from collections import defaultdict
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from .config import EmergentConfig
from .models import AILearningProfile, AIAnalysis, Information, Opportunity
from .utils import fast_mean, normalize_ai_label, safe_mean, stable_sigmoid

DEBUG_PORTFOLIO_LOG: List[Dict[str, Any]] = []
DEBUG_DECISION_LOG: List[Dict[str, Any]] = []

DEBUG_PORTFOLIO_LOG: List[Dict[str, Any]] = []

if TYPE_CHECKING:  # pragma: no cover
    from .information import InformationSystem
    from .knowledge import KnowledgeBase
    from .market import MarketEnvironment
    from .uncertainty import KnightianUncertaintyEnvironment
    from .innovation import InnovationEngine


@dataclass
class UncertaintyResponseProfile:
    """
    Tracks how an agent learns to respond to different uncertainty types.

    This class implements the adaptive uncertainty response mechanism described
    in Townsend et al. (2025), where entrepreneurs develop heterogeneous response
    patterns to each of the four Knightian uncertainty dimensions based on their
    experience and outcomes.

    The learning mechanism captures how entrepreneurs calibrate their responses
    to uncertainty through:
    - Outcome-based learning from past decisions
    - Exploratory experimentation with new response strategies
    - Weighted averaging that balances learning with stability

    Attributes
    ----------
    exploration_rate : float
        Probability of trying experimental response strategies rather than
        using learned weights. Captures entrepreneurial experimentation.
    memory_limit : int
        Number of past outcomes retained for learning. Reflects bounded
        rationality and recency effects in human cognition.
    learning_rate : float
        Speed of weight adjustment based on new outcomes.
    response_weights : Dict[str, float]
        Learned response weights for each uncertainty dimension:
        - actor_ignorance: How much to discount action value under ignorance
        - practical_indeterminism: How much to discount under execution risk
        - agentic_novelty: How much to value novelty potential
        - competitive_recursion: How much to discount under strategic interdependence
    outcome_history : Dict[str, List[Tuple[float, float, bool]]]
        History of (uncertainty_level, return, success) tuples for learning.

    Notes
    -----
    The response weights are initialized randomly to capture heterogeneity
    in initial entrepreneurial cognition. Through experience, agents learn
    which uncertainty dimensions are most predictive of success or failure
    in their specific context.

    References
    ----------
    Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).
        Are the futures computable? Knightian uncertainty & artificial
        intelligence. Academy of Management Review, 50(2), 415-440.
    """

    exploration_rate: float = 0.15
    memory_limit: int = 60
    learning_rate: float = 0.2
    response_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "actor_ignorance": np.random.uniform(0.3, 0.7),
            "practical_indeterminism": np.random.uniform(0.3, 0.7),
            "agentic_novelty": np.random.uniform(0.3, 0.7),
            "competitive_recursion": np.random.uniform(0.3, 0.7),
        }
    )
    outcome_history: Dict[str, List[Tuple[float, float, bool]]] = field(
        default_factory=lambda: {
            "actor_ignorance": [],
            "practical_indeterminism": [],
            "agentic_novelty": [],
            "competitive_recursion": [],
        }
    )

    def get_response_factor(
        self, uncertainty_type: str, uncertainty_level: float, explore: bool = True
    ) -> float:
        if explore and np.random.random() < self.exploration_rate:
            experimental_weight = np.random.uniform(0, 1)
            return 1 - (uncertainty_level * experimental_weight)
        weight = self.response_weights[uncertainty_type]
        return 1 - (uncertainty_level * weight)

    def update_from_outcome(
        self,
        uncertainty_perception: Dict,
        action: str,
        outcome: Dict,
        market_conditions: Dict,
    ) -> None:
        success = outcome.get("success", False)
        returns = outcome.get("capital_returned", 0) / max(
            outcome.get("investment_amount", 1), 1
        )
        for u_type in self.response_weights.keys():
            if u_type in uncertainty_perception:
                level = uncertainty_perception[u_type].get(
                    f"{u_type.split('_')[0]}_level",
                    uncertainty_perception[u_type].get("level", 0),
                )
                self.outcome_history[u_type].append((level, returns, success))
                if len(self.outcome_history[u_type]) > self.memory_limit:
                    self.outcome_history[u_type].pop(0)
                self._update_response_weight(u_type, market_conditions)

    def _update_response_weight(self, uncertainty_type: str, market_conditions: Dict) -> None:
        history = self.outcome_history[uncertainty_type]
        if not history:
            return

        recent = history[-self.memory_limit :]

        def mean_with_prior(data, prior_mean: float = 1.0, prior_weight: float = 2.0) -> float:
            if not data:
                return prior_mean
            returns = np.array([r for r, _ in data], dtype=float)
            mean = float(np.mean(returns)) if returns.size else prior_mean
            weight = len(returns)
            return float((mean * weight + prior_mean * prior_weight) / (weight + prior_weight))

        low_uncertainty = [(r, s) for l, r, s in recent if l < 0.33]
        high_uncertainty = [(r, s) for l, r, s in recent if l >= 0.67]

        perf_low = mean_with_prior(low_uncertainty, prior_mean=1.0)
        perf_high = mean_with_prior(high_uncertainty, prior_mean=perf_low)

        variability = np.std([r for _, r, _ in recent]) if recent else 0.0
        base_weight = self.response_weights[uncertainty_type]
        target_weight = base_weight

        if perf_high < perf_low * 0.85:
            target_weight = min(1.0, base_weight + self.learning_rate * (perf_low - perf_high))
        elif perf_low < perf_high * 0.9:
            target_weight = max(0.0, base_weight - self.learning_rate * (perf_high - perf_low))
        else:
            target_weight = base_weight * (1 - variability * 0.2)

        stability = np.clip(1.0 - variability, 0.3, 1.0)
        updated_weight = stability * target_weight + (1 - stability) * base_weight
        self.response_weights[uncertainty_type] = np.clip(updated_weight, 0, 1)


@dataclass
class PerformanceTracker:
    """Tracks capital deployment and returns by action and AI tier for ROE/ROIC metrics."""

    initial_equity: float
    deployed_by_action: Dict[str, float] = field(default_factory=lambda: collections.defaultdict(float))
    returned_by_action: Dict[str, float] = field(default_factory=lambda: collections.defaultdict(float))
    deployments_by_ai: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: collections.defaultdict(lambda: collections.defaultdict(float))
    )
    returns_by_ai: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: collections.defaultdict(lambda: collections.defaultdict(float))
    )
    roi_events: List[Dict[str, Any]] = field(default_factory=list)

    def _normalise_action(self, action: Optional[str]) -> str:
        return (action or "overall").lower()

    def _normalise_ai(self, ai_level: Optional[str]) -> str:
        return normalize_ai_label(ai_level) if ai_level else "none"

    def record_deployment(
        self, action: str, amount: float, ai_level: Optional[str] = "none", round_num: Optional[int] = None
    ) -> None:
        amount = float(amount)
        if amount <= 0:
            return
        action_key = self._normalise_action(action)
        ai_key = self._normalise_ai(ai_level)
        self.deployed_by_action[action_key] += amount
        self.deployed_by_action["overall"] += amount
        self.deployments_by_ai[action_key][ai_key] += amount
        self.deployments_by_ai["overall"][ai_key] += amount
        if round_num is not None:
            self.roi_events.append(
                {
                    "round": round_num,
                    "type": "deployment",
                    "action": action_key,
                    "ai_level": ai_key,
                    "amount": amount,
                }
            )

    def record_return(
        self, action: str, amount: float, ai_level: Optional[str] = "none", round_num: Optional[int] = None
    ) -> None:
        amount = float(amount)
        if amount < 0:
            amount = 0.0
        action_key = self._normalise_action(action)
        ai_key = self._normalise_ai(ai_level)
        self.returned_by_action[action_key] += amount
        self.returned_by_action["overall"] += amount
        self.returns_by_ai[action_key][ai_key] += amount
        self.returns_by_ai["overall"][ai_key] += amount
        if round_num is not None:
            self.roi_events.append(
                {
                    "round": round_num,
                    "type": "return",
                    "action": action_key,
                    "ai_level": ai_key,
                    "amount": amount,
                }
            )

    def compute_roic(self, action: Optional[str] = None) -> float:
        key = self._normalise_action(action)
        deployed = self.deployed_by_action.get(key, 0.0)
        if deployed <= 0:
            return 0.0
        returned = self.returned_by_action.get(key, 0.0)
        return (returned - deployed) / deployed

    def compute_roe(self, current_capital: float) -> float:
        denominator = max(self.initial_equity, 1e-9)
        return (current_capital - self.initial_equity) / denominator

    def snapshot(self, current_capital: float) -> Dict[str, float]:
        stats = {"roe": self.compute_roe(current_capital), "roic_overall": self.compute_roic(None)}
        for action in ["invest", "innovate", "explore"]:
            stats[f"roic_{action}"] = self.compute_roic(action)
            stats[f"capital_deployed_{action}"] = self.deployed_by_action.get(action, 0.0)
            stats[f"capital_returned_{action}"] = self.returned_by_action.get(action, 0.0)
        stats["capital_deployed_total"] = self.deployed_by_action.get("overall", 0.0)
        stats["capital_returned_total"] = self.returned_by_action.get("overall", 0.0)
        return stats


@dataclass
class AgentResources:
    """Represents an agent's multidimensional assets including cognitive constraints."""

    capital: float
    knowledge: Dict[str, float] = field(
        default_factory=lambda: {"tech": 0.1, "retail": 0.1, "service": 0.1, "manufacturing": 0.1}
    )
    capabilities: Dict[str, float] = field(
        default_factory=lambda: {
            "market_timing": 0.1,
            "opportunity_evaluation": 0.1,
            "innovation": 0.5,
            "uncertainty_management": 0.1,
        }
    )
    experience_units: int = 0
    performance: PerformanceTracker = field(default=None)
    knowledge_last_used: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.performance is None:
            self.performance = PerformanceTracker(initial_equity=self.capital)

        for sector in self.knowledge:
            self.knowledge_last_used.setdefault(sector, 0)

    def decay_resources(
        self,
        base_decay_rate: float,
        usage_weights: Optional[Dict[str, float]] = None,
        current_round: Optional[int] = None,
        inactivity_threshold: int = 15,
        shift_events: Optional[List[Dict[str, Any]]] = None,
        sector_pressure: Optional[Dict[str, float]] = None,
        config: Optional[Any] = None,
    ) -> None:
        """Apply decay to sector knowledge with activity, inactivity, and market-shift modifiers.

        Uses sector-specific decay rates from SectorProfile when available (calibrated from
        Ebbinghaus forgetting curve and industry skill depreciation research).

        When knowledge sits idle for 3× the inactivity threshold we perform an emergency reset
        (a coarse proxy for hard obsolescence), trimming the level and stamping the new round.
        """

        usage_weights = usage_weights or {}
        shift_events = shift_events or []

        # Build sector decay multipliers from config if available
        # Otherwise use legacy hardcoded values
        if config is not None and hasattr(config, 'SECTOR_PROFILES'):
            # Use sector-specific decay rates from SectorProfile (calibrated from learning research)
            sector_decay_multipliers = {}
            for sector in self.knowledge.keys():
                profile = config.SECTOR_PROFILES.get(sector, {})
                # knowledge_decay_rate is the absolute decay rate
                # Convert to multiplier relative to base_decay_rate
                sector_rate = profile.get('knowledge_decay_rate', base_decay_rate)
                # Multiplier = sector_rate / base_decay_rate
                sector_decay_multipliers[sector] = sector_rate / max(base_decay_rate, 0.001)
        else:
            # Legacy fallback
            sector_decay_multipliers = {
                "tech": 1.8,
                "retail": 1.05,
                "service": 0.85,
                "manufacturing": 0.6,
            }

        def _activity_modifier(sector: str) -> float:
            usage_factor = float(np.clip(usage_weights.get(sector, 0.0), 0.0, 1.0))
            return 1.0 - 0.5 * usage_factor

        def _inactivity_modifier(sector: str) -> float:
            if current_round is None:
                return 1.0
            last_used_round = self.knowledge_last_used.get(sector, current_round)
            rounds_idle = max(0, current_round - last_used_round)
            if rounds_idle >= inactivity_threshold * 3:
                # Emergency obsolescence reset: heavily discount knowledge that has gone stale.
                self.knowledge[sector] = max(0.01, self.knowledge[sector] * 0.3)
                self.knowledge_last_used[sector] = current_round
                return 1.0
            inactivity_penalty = max(0.0, rounds_idle - inactivity_threshold)
            extra = 0.0
            if rounds_idle >= inactivity_threshold * 2:
                extra = 0.15 * ((rounds_idle - 2 * inactivity_threshold) / max(1, inactivity_threshold))
            return 1.0 + 0.08 * (inactivity_penalty / max(1, inactivity_threshold)) + extra

        impacted_sectors: Dict[str, float] = {}
        for event in shift_events:
            sector = event.get("sector")
            severity = float(event.get("severity", 0.0) or 0.0)
            if sector:
                impacted_sectors[sector] = max(impacted_sectors.get(sector, 0.0), severity)

        pressure_map = sector_pressure or {}

        for sector in list(self.knowledge.keys()):
            multiplier = sector_decay_multipliers.get(sector, 1.0)
            noise = float(np.clip(np.random.normal(1.0, 0.1), 0.7, 1.3))
            activity_mod = _activity_modifier(sector)
            inactivity_mod = _inactivity_modifier(sector)
            shift_mod = 1.0 + impacted_sectors.get(sector, 0.0) * 2.4
            pressure_mod = 1.0 + np.clip(pressure_map.get(sector, 0.0) * 1.6, 0.0, 1.6)

            sector_decay_rate = base_decay_rate * multiplier * noise * activity_mod * inactivity_mod * shift_mod * pressure_mod
            sector_decay_rate = float(np.clip(sector_decay_rate, 0.0, 0.9))
            self.knowledge[sector] *= (1.0 - sector_decay_rate)
            self.knowledge[sector] = max(0.01, self.knowledge[sector])


@dataclass
class Portfolio:
    """Manages an agent's collection of investments with proper maturation tracking."""

    active_investments: Dict[str, Dict] = field(default_factory=dict)
    pending_investments: Dict[str, Dict] = field(default_factory=dict)
    matured_investments: List[Dict] = field(default_factory=list)
    past_investments: List[Dict] = field(default_factory=list)
    total_invested: float = 0.0
    locked_capital: float = 0.0
    diversification_score: float = 0.0
    config: EmergentConfig = field(default_factory=EmergentConfig, repr=False)

    def add_investment(
        self,
        opp_id: str,
        amount: float,
        sector: str,
        round_num: int,
        time_to_maturity: int,
        opportunity_obj: Opportunity,
        ai_level_used: str,
        ai_info: Information,
        decision_confidence: float = 0.5,
    ) -> None:
        maturation_round = round_num + time_to_maturity
        investment_record = {
            "amount": amount,
            "sector": sector,
            "entry_round": round_num,
            "maturation_round": maturation_round,
            "time_to_maturity": time_to_maturity,
            "returns": 0.0,
            "matured": False,
            "opportunity": opportunity_obj,
            "estimated_return_at_investment": ai_info.estimated_return,
            "estimated_uncertainty_at_investment": ai_info.estimated_uncertainty,
            "ai_confidence_at_investment": ai_info.confidence,
            "ai_level_used": ai_level_used,
            "ai_info_object": ai_info,
            "opportunity_id": opp_id,
            "decision_confidence": float(np.clip(decision_confidence, 0.05, 0.99)),
        }
        investment_key = opp_id
        if investment_key in self.pending_investments:
            suffix = 1
            while f"{opp_id}#{suffix}" in self.pending_investments:
                suffix += 1
            investment_key = f"{opp_id}#{suffix}"
        self.pending_investments[investment_key] = investment_record
        self.active_investments[investment_key] = investment_record
        self.total_invested += amount
        self.locked_capital += amount
        self._update_diversification()

    def check_matured_investments(self, current_round: int, market_conditions: Dict) -> List[Dict]:
        newly_matured = []
        for investment_key, investment in list(self.pending_investments.items()):
            if current_round >= investment["maturation_round"]:
                opp = investment["opportunity"]
                original_opp_id = investment.get("opportunity_id", investment_key.split("#")[0])

                raw_risk = getattr(opp, "latent_failure_potential", 0.5)
                risk = float(np.clip(raw_risk, 0.05, 0.95))
                ai_info = investment.get("ai_info_object")
                if ai_info is not None:
                    accuracy_val = getattr(ai_info, "actual_accuracy", None)
                    accuracy = float(accuracy_val) if accuracy_val is not None else 0.0
                    hallucination = 0.1 if getattr(ai_info, "contains_hallucination", False) else 0.0
                else:
                    accuracy = 0.0
                    hallucination = 0.0
                failure_adjustment = -0.15 * accuracy + hallucination
                sector_key = str(getattr(opp, "sector", investment.get("sector", "unknown")))
                demand_adjustments = (market_conditions.get("sector_demand_adjustments", {}) or {}).get(sector_key)
                sector_failure = 1.0
                if isinstance(demand_adjustments, dict):
                    sector_failure = float(demand_adjustments.get("failure", 1.0))
                regime_failure = float(market_conditions.get("regime_failure_multiplier", 1.0) or 1.0)
                base_failure = 0.05 + 0.5 * risk * sector_failure * regime_failure + failure_adjustment
                crowding_metrics = market_conditions.get("crowding_metrics", {}) or {}
                crowd_idx = float(crowding_metrics.get("crowding_index", 0.25) or 0.25)
                crowd_threshold = getattr(self.config, "RETURN_DEMAND_CROWDING_THRESHOLD", 0.35)
                crowd_multiplier = 1.0
                if crowd_idx > crowd_threshold:
                    crowd_multiplier += 0.25 * (crowd_idx - crowd_threshold)

                failure_chance = float(np.clip(base_failure * crowd_multiplier, 0.05, 0.9))
                success = np.random.random() >= failure_chance

                amount = float(investment.get("amount", 0.0))
                decision_confidence = float(np.clip(investment.get("decision_confidence", 0.5) or 0.5, 0.05, 0.99))
                if success:
                    realized_multiplier = opp.realized_return(
                        market_conditions,
                        investor_tier=normalize_ai_label(investment.get("ai_level_used", "none")),
                    )
                    capital_returned = amount * realized_multiplier
                    investment["defaulted"] = False
                else:
                    severity = np.interp(risk, [0.05, 0.95], [0.25, 0.9])
                    recovery_floor = np.interp(risk, [0.05, 0.95], [0.3, 0.02])
                    loss_noise = np.random.normal(0, 0.1)
                    recovery_ratio = float(np.clip(recovery_floor + loss_noise, 0.0, 0.4))
                    ai_tier = normalize_ai_label(investment.get("ai_level_used", "none"))
                    combo_hhi = float(market_conditions.get("combo_hhi", 0.0) or 0.0)
                    recovery_ratio *= np.clip(1.0 - combo_hhi * 0.8, 0.05, 1.0)
                    unc_state = market_conditions.get("uncertainty_state", {}) or {}
                    agentic_state = {}
                    if isinstance(unc_state, dict):
                        agentic_state = unc_state.get("agentic_novelty", unc_state)
                    scarcity_signal = float(agentic_state.get("component_scarcity", getattr(opp, "component_scarcity", 0.5)))
                    reuse_pressure = float(agentic_state.get("reuse_pressure", 0.0))
                    recovery_ratio *= np.clip(1.0 - reuse_pressure * 0.3, 0.35, 1.0)
                    recovery_ratio += 0.04 * max(0.0, scarcity_signal - 0.5)
                    tier_shares = market_conditions.get("tier_invest_share", {}) or {}
                    tier_share = float(tier_shares.get(ai_tier, 0.0))
                    if ai_tier != 'none':
                        accuracy_cushion = max(0.0, accuracy - 0.7) * (0.1 + 0.05 * scarcity_signal)
                        recovery_ratio = float(np.clip(recovery_ratio + accuracy_cushion, 0.0, 0.6))
                        if getattr(ai_info, "contains_hallucination", False):
                            recovery_ratio = float(np.clip(recovery_ratio - 0.07, 0.0, 0.6))
                    else:
                        recovery_ratio = float(np.clip(recovery_ratio - 0.05, 0.0, 0.4))
                    recovery_ratio *= np.clip(1.0 - tier_share * (1.0 + 0.3 * (1.0 - scarcity_signal)), 0.05, 1.0)
                    default_trigger = np.random.random() < (0.2 * severity * np.random.random())
                    if default_trigger:
                        recovery_ratio = 0.0
                        investment["defaulted"] = True
                    else:
                        investment["defaulted"] = False
                    realized_multiplier = recovery_ratio
                    capital_returned = amount * recovery_ratio
                investment["returns"] = capital_returned
                investment["matured"] = True
                investment["raw_success"] = bool(success)
                realized_multiplier = float(capital_returned / amount) if amount > 0 else 0.0
                investment["realized_multiplier"] = realized_multiplier
                investment["realized_roi"] = realized_multiplier - 1.0
                tier_label = normalize_ai_label(investment.get("ai_level_used", "none"))
                counterfactual_roi = realized_multiplier
                if tier_label != "none":
                    accuracy = getattr(ai_info, "actual_accuracy", 0.5) if ai_info is not None else 0.5
                    # Counterfactual: remove AI uplift proportional to accuracy above 0.5
                    counterfactual_roi = float(
                        np.clip(realized_multiplier - (accuracy - 0.5) * 0.35, 0.0, 2.5)
                    )
                investment["counterfactual_roi"] = counterfactual_roi
                investment["counterfactual_capital_returned"] = counterfactual_roi * amount
                roi_threshold = getattr(self.config, 'INVESTMENT_SUCCESS_ROI_THRESHOLD', 0.0)
                effective_success = bool(success and investment["realized_roi"] >= roi_threshold)
                investment["success"] = effective_success
                investment["maturation_market"] = market_conditions.copy()
                self.matured_investments.append(investment)
                del self.pending_investments[investment_key]
                if investment_key in self.active_investments:
                    del self.active_investments[investment_key]
                self.locked_capital -= investment["amount"]
                newly_matured.append(
                    {
                        "opportunity_id": original_opp_id,
                        "investment": investment,
                        "investment_amount": investment.get("amount", 0.0),
                        "capital_returned": capital_returned,
                        "success": effective_success,
                        "raw_success": success,
                        "defaulted": investment.get("defaulted", False),
                        "executed": True,
                        "outcome_pending": False,
                        "net_return": capital_returned - investment["amount"],
                        "ai_level_used": investment.get("ai_level_used", "none"),
                        "ai_info_at_investment": investment.get("ai_info_object"),
                        "realized_multiplier": realized_multiplier,
                        "decision_confidence": decision_confidence,
                        "counterfactual_roi": counterfactual_roi,
                        "counterfactual_capital_returned": investment.get("counterfactual_capital_returned"),
                    }
                )
        if newly_matured:
            self._update_diversification()
        return newly_matured

    def archive_matured_history(self) -> None:
        """Persist lightweight snapshots of matured deals and drop heavy references."""
        if not self.matured_investments:
            return

        for investment in self.matured_investments:
            snapshot = {
                "opportunity_id": investment.get("opportunity_id"),
                "sector": investment.get("sector"),
                "entry_round": investment.get("entry_round"),
                "maturation_round": investment.get("maturation_round"),
                "time_to_maturity": investment.get("time_to_maturity"),
                "investment_amount": investment.get("amount", 0.0),
                "capital_returned": investment.get("returns", 0.0),
                "success": bool(investment.get("success", False)),
                "ai_level_used": investment.get("ai_level_used", "none"),
            }
            self.past_investments.append(snapshot)
            investment.pop("opportunity", None)
            investment.pop("ai_info_object", None)
            investment.pop("maturation_market", None)

        self.matured_investments.clear()

    def _update_diversification(self) -> None:
        if not self.pending_investments and not self.active_investments:
            self.diversification_score = 0.0
            return
        all_investments: Dict[str, Dict] = {}
        all_investments.update(self.pending_investments)
        all_investments.update(self.active_investments)
        if not all_investments:
            self.diversification_score = 0.0
            return
        total_amount = 0.0
        sector_counts: Dict[str, float] = {}
        for inv_data in all_investments.values():
            sector = inv_data.get("sector", "unknown")
            amount = float(inv_data.get("amount", 0.0))
            if amount <= 0:
                continue
            sector_counts[sector] = sector_counts.get(sector, 0.0) + amount
            total_amount += amount
        if total_amount <= 0 or not sector_counts:
            self.diversification_score = 0.0
            return
        proportions = np.array([amount / total_amount for amount in sector_counts.values()], dtype=float)
        if not np.isfinite(proportions).all() or proportions.sum() <= 0:
            self.diversification_score = 0.0
            return
        max_possible_entropy = np.log(len(self.config.SECTORS)) if hasattr(self.config, "SECTORS") else np.log(4)
        if max_possible_entropy <= 0:
            self.diversification_score = 0.0
            return
        diversity = float(np.sum(-proportions * np.log(proportions + 1e-12)) / max_possible_entropy)
        if not np.isfinite(diversity) or diversity < 0:
            diversity = 0.0
        self.diversification_score = min(1.0, diversity)

    def get_available_capital(self, total_capital: float) -> float:
        return max(0.0, float(total_capital))

    def has_pending_investments(self) -> bool:
        return len(self.pending_investments) > 0

    @property
    def portfolio_size(self) -> int:
        return len(self.active_investments)


class EmergentAgent:
    def __init__(self, agent_id: int, initial_traits: Dict[str, float], config: EmergentConfig,
             knowledge_base: KnowledgeBase, innovation_engine: InnovationEngine, agent_type: str = 'emergent',
             initial_capital: Optional[float] = None, primary_sector: Optional[str] = None):
        self.id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.traits = initial_traits.copy()
        if 'entrepreneurial_drive' not in self.traits:
            self.traits['entrepreneurial_drive'] = float(
                np.clip(np.random.normal(0.55, 0.2), 0.0, 1.0)
            )
        if 'uncertainty_tolerance' not in self.traits:
            self.traits['uncertainty_tolerance'] = float(
                np.clip(np.random.beta(1.8, 1.1), 0.0, 1.0)
            )
        self.alive = True
        self.failure_round = None  # Track when agent fails for Kaplan-Meier analysis
        self.failure_reason = None  # Track why agent failed
        self.knowledge_base = knowledge_base
        self.innovation_engine = innovation_engine

        # Assign primary sector based on NVCA-weighted probabilities
        if primary_sector is not None:
            self.primary_sector = primary_sector
        else:
            self.primary_sector = self._sample_sector_weighted()

        # Sample initial capital from sector-specific range (NVCA 2024 calibrated)
        if initial_capital is not None:
            starting_capital = float(initial_capital)
        else:
            sector_profile = self.config.SECTOR_PROFILES.get(self.primary_sector, {})
            if 'initial_capital_range' in sector_profile:
                low, high = sector_profile['initial_capital_range']
            else:
                low, high = getattr(self.config, 'INITIAL_CAPITAL_RANGE', (self.config.INITIAL_CAPITAL, self.config.INITIAL_CAPITAL))
            starting_capital = float(np.random.uniform(low, high))
        self.resources = AgentResources(capital=starting_capital)
        # Override default knowledge to only include sectors from config
        config_sectors = getattr(config, 'SECTORS', None) or list(getattr(config, 'SECTOR_PROFILES', {}).keys())
        if config_sectors:
            self.resources.knowledge = {sector: 0.1 for sector in config_sectors}
            self.resources.knowledge_last_used = {sector: 0 for sector in config_sectors}
        # Boost knowledge in primary sector
        if self.primary_sector in self.resources.knowledge:
            self.resources.knowledge[self.primary_sector] = min(1.0, self.resources.knowledge[self.primary_sector] + 0.2)
        self.portfolio = Portfolio(config=config)
        self.trait_momentum = {trait: 0.0 for trait in initial_traits}
        history_depth = config.agent_history_depth if hasattr(config, 'agent_history_depth') else 20
        self.outcomes = collections.deque(maxlen=history_depth)
        self.ai_usage = collections.deque(maxlen=history_depth)
        self.innovations = [] 
        self.strategy_mode = "exploring"
        self.ai_learning = None
        self.ai_interactions_count = 0
        if agent_type and isinstance(agent_type, str):
            if agent_type.endswith("_ai"):
                initial_level = normalize_ai_label(agent_type.replace("_ai", ""))
            elif agent_type in {"basic", "advanced", "premium"}:
                initial_level = normalize_ai_label(agent_type)
            else:
                initial_level = "none"
        else:
            initial_level = "none"
        self.fixed_ai_level: Optional[str] = (
            initial_level if getattr(self.config, "AGENT_AI_MODE", "emergent") == "fixed" else "none"
        )
        self.current_ai_level: str = initial_level
        self.ai_learning_profile = AILearningProfile()
        self.ai_analysis_history = collections.deque(maxlen=history_depth)
        self.ai_domain_exposure = collections.Counter()
        self.market_volatility_history = collections.deque(maxlen=50)
        self._uncertainty_cache_round: int = -1
        self._uncertainty_cache: Dict[str, Any] = {}
        self.ai_switches = 0
        self._last_ai_tier_metrics: Dict[str, Any] = {}
        self._insolvency_streak: int = 0
        self.ai_success_cumulative: int = 0
        self.ai_attempt_cumulative: int = 0
        self.ai_roi_sum: float = 0.0
        self.ai_roi_count: int = 0
        self.ai_accuracy_cumulative: float = 0.0
        self.ai_accuracy_observations: int = 0
        # FIXED: Neutral priors for all tiers - agents learn from experience
        # Previously had built-in skepticism: none=0.52, basic=0.48, advanced=0.35, premium=0.26
        # Now all tiers start at 0.5 and agents update beliefs based on actual outcomes
        _prior_map = {
            'none': (2.0, 2.0),      # Prior mean = 0.50 (neutral)
            'basic': (2.0, 2.0),     # Prior mean = 0.50 (neutral)
            'advanced': (2.0, 2.0),  # Prior mean = 0.50 (neutral)
            'premium': (2.0, 2.0),   # Prior mean = 0.50 (neutral)
        }
        self.ai_tier_beliefs: Dict[str, Dict[str, float]] = {
            lvl: {"alpha": _prior_map.get(lvl, (2.5, 2.5))[0], "beta": _prior_map.get(lvl, (2.5, 2.5))[1]}
            for lvl in ['none', 'basic', 'advanced', 'premium']
        }
        self.ai_tier_usage: defaultdict[str, int] = defaultdict(int)
        if self.current_ai_level and self.current_ai_level != 'none':
            self.ai_tier_usage[self.current_ai_level] += 1
        self._last_innovation_round: Optional[int] = None
        bias_sigma = max(0.0, getattr(self.config, 'ACTION_BIAS_SIGMA', 0.05))
        self.action_bias = {
            'invest': float(np.random.normal(0.0, bias_sigma)),
            'innovate': float(np.random.normal(0.0, bias_sigma)),
            'explore': float(np.random.normal(0.0, bias_sigma)),
            'maintain': float(np.random.normal(0.0, bias_sigma)),
        }
        self._subscription_accounts: Dict[str, int] = collections.defaultdict(int)
        self._subscription_rates: Dict[str, float] = collections.defaultdict(float)
        self._subscription_deferral_remaining: Dict[str, int] = collections.defaultdict(int)
        self._subscription_last_usage_round: Dict[str, int] = {}
        self._last_subscription_charge: float = 0.0
        self._burn_history = collections.deque(maxlen=int(getattr(self.config, 'BURN_HISTORY_WINDOW', 3)))
        self._burn_streak: int = 0
        self._liquidity_streak: int = 0
        self._equity_streak: int = 0
        self.failure_reason: Optional[str] = None
        self._recent_actions: collections.deque[str] = collections.deque(maxlen=20)
        self.paradox_signal: float = 0.0
        self.paradox_history: collections.deque[Dict[str, float]] = collections.deque(maxlen=30)

    def _sample_sector_weighted(self) -> str:
        """Sample a sector based on NVCA-weighted probabilities."""
        sector_weights = getattr(self.config, 'SECTOR_WEIGHTS', None)
        if not sector_weights:
            # Fall back to uniform random from available sectors
            sectors = list(self.config.SECTOR_PROFILES.keys())
            return np.random.choice(sectors) if sectors else "tech"

        sectors = list(sector_weights.keys())
        weights = [sector_weights[s] for s in sectors]
        total = sum(weights)
        if total <= 0:
            return np.random.choice(sectors) if sectors else "tech"

        probs = [w / total for w in weights]
        return np.random.choice(sectors, p=probs)

    def _get_sector_survival_threshold(self) -> float:
        """Get survival threshold for agent based on primary sector (BLS/Fed calibrated)."""
        sector_profile = self.config.SECTOR_PROFILES.get(self.primary_sector, {})
        if 'survival_threshold' in sector_profile:
            return float(sector_profile['survival_threshold'])
        # Fallback to global threshold
        return float(self.config.SURVIVAL_THRESHOLD)

    def _get_sector_equity_ratio(self) -> float:
        """Get sector-specific minimum capital retention ratio (current/initial).

        Returns the minimum equity ratio (capital/initial_capital) that must be maintained
        to avoid failure. Calibrated to empirical survival patterns and investor shut-down
        thresholds from BLS, Carta, CB Insights, and sector-specific VC data (2015-2024).

        Sector ratios represent investor shut-down decision points:
        - Tech (0.28): High burn tolerance for growth; VCs patient if metrics show progress
        - Retail (0.38): Moderate burn; inventory volatility requires buffer
        - Service (0.52): Capital efficient model; higher retention expectations
        - Manufacturing (0.58): Asset-heavy; capital preservation critical for long payback

        Falls back to global SURVIVAL_CAPITAL_RATIO if sector-specific not defined.
        """
        sector_profile = self.config.SECTOR_PROFILES.get(self.primary_sector, {})
        if 'survival_equity_ratio' in sector_profile:
            return float(sector_profile['survival_equity_ratio'])
        # Fallback to global ratio
        return float(getattr(self.config, 'SURVIVAL_CAPITAL_RATIO', 0.40))

    def _get_uncertainty_cache(self, round_num: int) -> Dict[str, Any]:
        if self._uncertainty_cache_round != round_num:
            self._uncertainty_cache_round = round_num
            self._uncertainty_cache = {
                "avg_knowledge_level": None,
                "knowledge_deficit": None,
                "timing_pressure": None,
                "avg_timing_pressure": None,
            }
        return self._uncertainty_cache

    def _start_subscription_schedule(self, ai_level: str, ai_config: Dict[str, Any]) -> None:
        cycle = max(1, int(getattr(self.config, 'AI_SUBSCRIPTION_AMORTIZATION_ROUNDS', 20)))
        base_cost = float(ai_config.get('cost', 0.0))
        if base_cost <= 0:
            return
        self._subscription_accounts[ai_level] = cycle
        self._subscription_rates[ai_level] = base_cost / cycle

    def _compute_subscription_deferral_rounds(self, ai_level: str) -> int:
        base_grace = int(getattr(self.config, 'AI_SUBSCRIPTION_FLOAT_BASE_ROUNDS', 0))
        max_extra = max(0, int(getattr(self.config, 'AI_SUBSCRIPTION_FLOAT_MAX_ROUNDS', 3)))
        drive = float(self.traits.get('entrepreneurial_drive', 0.5))
        extra = int(round(max_extra * max(0.0, drive)))
        return base_grace + extra

    def _charge_subscription_installment(self, ai_level: str) -> float:
        remaining = self._subscription_accounts.get(ai_level, 0)
        rate = self._subscription_rates.get(ai_level, 0.0)
        if remaining <= 0 or rate <= 0:
            return 0.0
        deferral = self._subscription_deferral_remaining.get(ai_level, 0)
        if deferral > 0:
            next_deferral = deferral - 1
            if next_deferral > 0:
                self._subscription_deferral_remaining[ai_level] = next_deferral
            else:
                self._subscription_deferral_remaining.pop(ai_level, None)
            return 0.0
        self.resources.capital -= rate
        remaining -= 1
        if remaining <= 0:
            self._subscription_accounts.pop(ai_level, None)
            self._subscription_rates.pop(ai_level, None)
        else:
            self._subscription_accounts[ai_level] = remaining
        return rate

    def _apply_subscription_carry(self) -> float:
        total = 0.0
        credit_line = int(getattr(self.config, 'AI_CREDIT_LINE_ROUNDS', 0))
        current_round = getattr(self, 'current_round', None)
        for level in list(self._subscription_accounts.keys()):
            if (
                credit_line > 0
                and current_round is not None
                and current_round <= credit_line
                and level in {'advanced', 'premium'}
            ):
                continue
            total += self._charge_subscription_installment(level)
        self._last_subscription_charge = total
        return total

    def _evaluate_failure_conditions(self, capital_after: Optional[float] = None) -> Optional[str]:
        if not self.alive:
            return self.failure_reason
        if capital_after is None:
            capital_after = float(self.resources.capital)
        initial_equity = max(self.resources.performance.initial_equity, 1.0)
        grace_period = max(1, int(getattr(self.config, 'INSOLVENCY_GRACE_ROUNDS', 1)))

        reason: Optional[str] = None
        operating_cost = float(getattr(self, 'operating_cost_estimate', self.config.BASE_OPERATIONAL_COST))
        reserve_months = max(1, int(getattr(self.config, 'OPERATING_RESERVE_MONTHS', 3)))
        # Use sector-specific survival threshold (BLS/Fed calibrated)
        sector_survival_threshold = self._get_sector_survival_threshold()
        liquidity_floor = max(
            sector_survival_threshold,
            operating_cost * reserve_months,
        )
        ai_level = normalize_ai_label(getattr(self, 'current_ai_level', getattr(self, 'agent_type', 'none')))
        relief_factor = 1.0
        if ai_level != 'none':
            trust = float(np.clip(self.traits.get('ai_trust', 0.5), 0.0, 1.0))
            discount = float(np.clip(getattr(self.config, 'AI_TRUST_RESERVE_DISCOUNT', 0.25), 0.0, 0.9))
            relief_factor = float(np.clip(1.0 - trust * discount, 0.5, 1.0))
            liquidity_floor *= relief_factor
        if capital_after < liquidity_floor:
            self._liquidity_streak += 1
            if self._liquidity_streak >= grace_period:
                reason = 'liquidity_failure'
        else:
            self._liquidity_streak = 0

        # Use sector-specific equity ratio (empirically calibrated to investor shut-down thresholds)
        sector_equity_ratio = self._get_sector_equity_ratio()
        ratio_floor = sector_equity_ratio * relief_factor
        capital_ratio = capital_after / initial_equity
        if reason is None:
            if capital_ratio < ratio_floor:
                self._equity_streak += 1
                if self._equity_streak >= grace_period:
                    reason = 'equity_failure'
            else:
                self._equity_streak = 0

        if (
            reason is None
            and capital_ratio < ratio_floor * 0.85
            and random.random() < 0.05
        ):
            reason = 'funding_shock'

        burn_window = self._burn_history.maxlen or 3
        leverage_cap = float(getattr(self.config, 'BURN_LEVERAGE_CAP', 0.5))
        burn_threshold = float(getattr(self.config, 'BURN_FAILURE_THRESHOLD', 0.08))
        leverage = 0.0
        if capital_after > 0:
            leverage = float(self.portfolio.locked_capital / max(capital_after, 1e-6))
        if reason is None and len(self._burn_history) == burn_window and leverage >= leverage_cap:
            burn_avg = float(np.mean(self._burn_history))
            if burn_avg < -burn_threshold * initial_equity:
                self._burn_streak += 1
                if self._burn_streak >= grace_period:
                    reason = 'burnout_failure'
            else:
                self._burn_streak = 0
        elif len(self._burn_history) < burn_window or leverage < leverage_cap:
            self._burn_streak = 0

        return reason

    def _ensure_subscription_schedule(self, ai_level: str, ai_config: Dict[str, Any]) -> None:
        if not ai_config or float(ai_config.get('cost', 0.0)) <= 0:
            return
        if self._subscription_accounts.get(ai_level, 0) <= 0:
            self._start_subscription_schedule(ai_level, ai_config)
            grace_rounds = self._compute_subscription_deferral_rounds(ai_level)
            if grace_rounds > 0:
                self._subscription_deferral_remaining[ai_level] = grace_rounds

    def register_ai_usage(self, ai_level: str) -> None:
        normalized_level = normalize_ai_label(ai_level)
        if normalized_level == 'none':
            return
        ai_config = self.config.AI_LEVELS.get(normalized_level)
        if ai_config is None:
            return
        if ai_config.get('cost_type') == 'subscription':
            self._ensure_subscription_schedule(normalized_level, ai_config)
        current_round = getattr(self, 'current_round', None)
        if current_round is not None:
            self._subscription_last_usage_round[normalized_level] = current_round

    def _collect_neighbor_signals(self, neighbor_agents: Optional[List["EmergentAgent"]]) -> Dict[str, Any]:
        """Capture peer adoption, sentiment, and performance cues for social learning."""
        signals: Dict[str, Any] = {
            "ai_distribution": {},
            "ai_adoption_pressure": 0.0,
            "peer_roi_gap": 0.0,
            "opportunity_interest": {},
            "opportunity_sentiment": {},
            "sector_sentiment": {},
        }
        if not neighbor_agents:
            return signals

        ai_counts = collections.Counter()
        opportunity_interest = collections.Counter()
        opportunity_sentiment: Dict[str, List[float]] = collections.defaultdict(lambda: [0.0, 0.0])
        sector_sentiment: Dict[str, List[float]] = collections.defaultdict(lambda: [0.0, 0.0])
        sector_interest = collections.Counter()

        baseline_equity = max(self.resources.performance.initial_equity, 1.0)
        self_roi = self.resources.performance.compute_roic("invest")
        if self_roi is None or not math.isfinite(self_roi):
            self_roi = 0.0

        roi_weighted_sum = 0.0
        weight_total = 0.0

        def _last_n(seq: Any, n: int) -> List[Any]:
            if seq is None:
                return []
            if isinstance(seq, list):
                return seq[-n:]
            if isinstance(seq, collections.deque):
                return list(seq)[-n:]
            try:
                return list(seq)[-n:]
            except TypeError:
                return []

        for neighbor in neighbor_agents:
            if neighbor is None or not getattr(neighbor, "alive", False):
                continue

            last_usage = neighbor.ai_usage[-1] if neighbor.ai_usage else ("none", "none")
            ai_level = normalize_ai_label(last_usage[1])
            ai_counts[ai_level] += 1

            capital = float(getattr(neighbor.resources, "capital", 0.0) or 0.0)
            weight = max(0.1, capital / baseline_equity)
            weight_total += weight

            neighbor_roi = neighbor.resources.performance.compute_roic("invest")
            if neighbor_roi is None or not math.isfinite(neighbor_roi):
                neighbor_roi = neighbor.resources.performance.compute_roic(None)
            if neighbor_roi is None or not math.isfinite(neighbor_roi):
                neighbor_roi = 0.0
            roi_weighted_sum += weight * neighbor_roi

            portfolio = getattr(neighbor, "portfolio", None)
            if portfolio is not None:
                investments_snapshot = []
                try:
                    investments_snapshot = list(portfolio.active_investments.values())
                except RuntimeError:
                    investments_snapshot = list(dict(portfolio.active_investments).values())
                for investment in investments_snapshot:
                    opp_id = investment.get("opportunity_id")
                    if not opp_id:
                        opp_obj = investment.get("opportunity")
                        opp_id = getattr(opp_obj, "id", None)
                    if opp_id:
                        opportunity_interest[opp_id] += weight
                    sector = investment.get("sector")
                    if not sector:
                        opp_obj = investment.get("opportunity")
                        sector = getattr(opp_obj, "sector", None)
                    if sector:
                        sector_interest[sector] += weight * 0.5

            for outcome in _last_n(getattr(neighbor, "outcomes", []), 5):
                if not isinstance(outcome, dict) or outcome.get("action") != "invest":
                    continue
                details = outcome.get("chosen_opportunity_details")
                opp_id = None
                sector = None
                if isinstance(details, dict):
                    opp_id = details.get("id")
                    sector = details.get("sector")
                if opp_id is None:
                    opp_id = outcome.get("opportunity_id")
                if sector is None:
                    sector = outcome.get("sector")
                invested = outcome.get("investment_amount") or (outcome.get("investment") or {}).get("amount")
                returned = outcome.get("capital_returned")
                roi = outcome.get("roi")
                if roi is None and invested and returned is not None:
                    try:
                        roi = float(returned) / float(invested) if invested else None
                    except (TypeError, ValueError, ZeroDivisionError):
                        roi = None
                if roi is None or not math.isfinite(roi):
                    continue
                delta = float(np.clip(roi - 1.0, -1.0, 1.0))
                if opp_id:
                    totals = opportunity_sentiment[opp_id]
                    totals[0] += weight * delta
                    totals[1] += weight
                if sector:
                    totals = sector_sentiment[sector]
                    totals[0] += weight * delta
                    totals[1] += weight

        total_ai = sum(ai_counts.values())
        if total_ai > 0:
            signals["ai_distribution"] = {
                level: count / total_ai for level, count in ai_counts.items()
            }
            adoption_pressure = 1.0 - signals["ai_distribution"].get("none", 0.0)
            signals["ai_adoption_pressure"] = float(np.clip(adoption_pressure, 0.0, 1.0))

        if weight_total > 0:
            avg_neighbor_roi = roi_weighted_sum / weight_total
            peer_roi_gap = float(np.clip(avg_neighbor_roi - self_roi, -1.0, 1.0))
            signals["peer_roi_gap"] = peer_roi_gap
            herding_trait = 1.0 - self.traits.get('independent_thinking', 0.5)
            signals["ai_adoption_pressure"] = float(
                np.clip(
                    signals.get("ai_adoption_pressure", 0.0)
                    + np.clip(peer_roi_gap, -0.5, 0.5) * herding_trait,
                    -1.0,
                    1.5,
                )
            )

        if opportunity_interest:
            max_interest = max(opportunity_interest.values())
            if max_interest > 0:
                signals["opportunity_interest"] = {
                    opp_id: float(np.clip(val / max_interest, 0.0, 1.0))
                    for opp_id, val in opportunity_interest.items()
                }

        opportunity_sentiment_avg: Dict[str, float] = {}
        for opp_id, (delta_total, delta_weight) in opportunity_sentiment.items():
            if delta_weight > 0:
                opportunity_sentiment_avg[opp_id] = float(np.clip(delta_total / delta_weight, -1.0, 1.0))
        if opportunity_sentiment_avg:
            signals["opportunity_sentiment"] = opportunity_sentiment_avg

        sector_scores: Dict[str, float] = {}
        max_interest_weight = max(sector_interest.values()) if sector_interest else 0.0
        for sector, (delta_total, delta_weight) in sector_sentiment.items():
            sentiment = float(np.clip(delta_total / delta_weight, -1.0, 1.0)) if delta_weight > 0 else 0.0
            interest_component = 0.0
            if max_interest_weight > 0 and sector in sector_interest:
                interest_component = float(np.clip(sector_interest[sector] / max_interest_weight, 0.0, 1.0))
            sector_scores[sector] = 0.6 * sentiment + 0.4 * interest_component
        # Include purely interest-driven signals for sectors without sentiment
        for sector, val in sector_interest.items():
            if sector in sector_scores:
                continue
            if max_interest_weight > 0:
                sector_scores[sector] = float(np.clip(val / max_interest_weight, 0.0, 1.0))
        if sector_scores:
            signals["sector_sentiment"] = sector_scores

        return signals

    def make_decision(self, opportunities: List['Opportunity'], market_conditions: Dict, info_system: 'InformationSystem', market_environment: 'MarketEnvironment', round_num: int, all_agents: List['EmergentAgent'], uncertainty_env: 'KnightianUncertaintyEnvironment', neighbor_agents: Optional[List['EmergentAgent']] = None, ai_level_override: Optional[str] = None) -> Dict:
        if not self.alive:
            return {'action': 'none', 'ai_level_used': 'none', 'portfolio_size': self.portfolio.portfolio_size}

        subscription_carry = self._apply_subscription_carry()
        capital_before_action = float(self.resources.capital)
        grace_period = max(1, int(getattr(self.config, 'INSOLVENCY_GRACE_ROUNDS', 1)))
        initial_equity = max(self.resources.performance.initial_equity, 1.0)
        capital_ratio = self.resources.capital / initial_equity
        ratio_floor = float(getattr(self.config, 'SURVIVAL_CAPITAL_RATIO', 0.0))
        # Use sector-specific survival threshold (BLS/Fed calibrated)
        sector_threshold = self._get_sector_survival_threshold()
        below_floor = (
            self.resources.capital < sector_threshold
            or capital_ratio < ratio_floor
        )
        if below_floor:
            self._insolvency_streak += 1
            if self._insolvency_streak >= grace_period:
                self.alive = False
                self.failure_round = round_num  # Record when failure occurred
                self.failure_reason = 'bankruptcy'  # Record failure reason
                return {
                    'action': 'exit',
                    'reason': 'bankruptcy',
                    'ai_level_used': 'none',
                    'portfolio_size': self.portfolio.portfolio_size,
                }
            # Restructuring round: force maintain decision without AI spend.
            return {
                'action': 'maintain',
                'round': round_num,
                'reason': 'distress_restructuring',
                'ai_level_used': 'none',
                'ai_per_use_cost': 0.0,
                'ai_call_count': 0,
                'portfolio_size': self.portfolio.portfolio_size,
                'capital_before_action': capital_before_action,
                'capital_after_action': float(self.resources.capital),
            }
        else:
            if self._insolvency_streak:
                self._insolvency_streak = 0

        self._update_strategic_mode(market_conditions)

        neighbor_signals = self._collect_neighbor_signals(neighbor_agents)
        self._last_neighbor_signals = neighbor_signals

        ai_level = 'none'
        if ai_level_override is not None:
            ai_level = ai_level_override
        elif self.config.AGENT_AI_MODE == 'fixed':
            ai_level = self.fixed_ai_level
        elif self.config.AGENT_AI_MODE == 'emergent':
            ai_level = self._choose_ai_level(neighbor_agents, neighbor_signals)

        # --- FIX 1: Save the chosen AI level to the instance attribute ---
        self.current_ai_level = ai_level
        self.ai_usage.append((round_num, ai_level))
        if len(self.ai_usage) >= 2:
            prev_level = self.ai_usage[-2][1]
            if prev_level != ai_level:
                self.ai_switches += 1

        if getattr(self.config, 'ENABLE_DEBUG_LOGS', False):
            tier_metrics = dict(getattr(self, '_last_ai_tier_metrics', {}))
            tier_metrics.update({
                'log_type': 'ai_tier_selection',
                'agent_id': self.id,
                'round': round_num,
                'selected_level': ai_level,
            })
            DEBUG_DECISION_LOG.append(tier_metrics)

        ai_cfg = self.config.AI_LEVELS.get(ai_level, self.config.AI_LEVELS['none'])

        uncertainty_cache = self._get_uncertainty_cache(round_num)

        learning_profile = getattr(self, "ai_learning", None) or self.ai_learning_profile

        perception = uncertainty_env.perceive_uncertainty(
            agent_traits=self.traits,
            visible_opportunities=opportunities,
            market_conditions=market_conditions,
            ai_level=ai_level,
            # --- FIX 2: Pass the agent's AI learning attributes ---
            ai_learning_profile=learning_profile,
            ai_analysis_history=self.ai_analysis_history,
            agent_resources=self.resources,
            agent_knowledge=self.knowledge_base.agent_knowledge.get(self.id, set()),
            recent_outcomes=list(self.outcomes),
            cached_metrics=uncertainty_cache,
            agent_id=self.id,
            action_history=list(self._recent_actions),
        )
        perception['_round'] = round_num
        if neighbor_signals:
            perception['neighbor_signals'] = neighbor_signals
        self._last_perception = perception

        action_type = self._choose_action_type(
            opportunities, market_conditions, info_system,
            market_environment, uncertainty_env, ai_level, perception
        )

        decision = None
        if action_type == 'invest':
            decision = self._make_investment_decision(
                opportunities,
                info_system,
                market_conditions,
                market_environment,
                round_num,
                ai_level,
                perception,
            )
        elif action_type == 'innovate':
            decision = self._make_innovation_decision(market_conditions, round_num, ai_level, uncertainty_env, perception)
        elif action_type == 'explore':
            decision = self._make_exploration_decision(round_num, ai_level)
        else: # 'maintain'
            decision = self._make_maintain_decision(round_num, ai_level)

        if decision:
            decision['perception_at_decision'] = perception
        else:
            decision = {
                'action': 'maintain',
                'ai_level_used': ai_level,
                'portfolio_size': self.portfolio.portfolio_size,
                'ai_per_use_cost': 0.0,
                'ai_call_count': 0,
                'error': 'decision_method_returned_none',
                'perception_at_decision': perception
            }

        decision['subscription_cost'] = float(decision.get('subscription_cost', 0.0) or 0.0) + float(subscription_carry or 0.0)
        capital_after_action = float(self.resources.capital)
        decision['capital_before_action'] = capital_before_action
        decision['capital_after_action'] = capital_after_action
        self._burn_history.append(capital_after_action - capital_before_action)
        failure_reason = self._evaluate_failure_conditions(capital_after_action)
        if failure_reason:
            self.alive = False
            self.failure_round = round_num  # Record when failure occurred
            self.failure_reason = failure_reason  # Already set, but keeping consistent
            return {
                'action': 'exit',
                'reason': failure_reason,
                'ai_level_used': 'none',
                'portfolio_size': self.portfolio.portfolio_size,
                'capital_before_action': capital_before_action,
                'capital_after_action': capital_after_action,
            }
        self._recent_actions.append(decision.get('action'))
        return decision

    def _compute_ai_performance_metrics(self) -> Dict[str, float]:
        """Aggregate AI and non-AI investment outcomes for tier decisions."""

        def _numeric(value: Any) -> Optional[float]:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _compute_stats(records: List[Dict[str, Any]], *, default_success_rate: float) -> Dict[str, float]:
            total = len(records)
            successes = sum(1 for rec in records if rec.get('success'))
            roi_vals = [rec['roi'] for rec in records if rec.get('roi') is not None]
            accuracy_vals = [rec['accuracy'] for rec in records if rec.get('accuracy') is not None]
            amount_vals = [rec['amount'] for rec in records if rec.get('amount') is not None]
            net_cash_vals = [rec['net_cash'] for rec in records if rec.get('net_cash') is not None]
            stats = {
                'total': float(total),
                'successes': float(successes),
                'success_rate': float(successes / total) if total else float(default_success_rate),
                'avg_roi': float(np.mean(roi_vals)) if roi_vals else 1.0,
                'accuracy_rate': float(np.mean(accuracy_vals)) if accuracy_vals else 0.0,
                'avg_amount': float(np.mean(amount_vals)) if amount_vals else 0.0,
                'net_cash_total': float(sum(net_cash_vals)) if net_cash_vals else 0.0,
            }
            return stats

        recent_window = max(1, int(getattr(self.config, "AI_TIER_RECENT_WINDOW", 15)))

        ai_records: List[Dict[str, Any]] = []
        baseline_records: List[Dict[str, Any]] = []
        all_records: List[Dict[str, Any]] = []

        for outcome in self.outcomes:
            if not isinstance(outcome, dict):
                continue

            invest_amt = outcome.get('investment_amount')
            if invest_amt is None:
                invest_amt = (outcome.get('investment') or {}).get('amount')
            invest_amt = _numeric(invest_amt)
            capital_returned = _numeric(outcome.get('capital_returned'))
            roi = None
            net_cash = None
            if invest_amt and invest_amt > 0 and capital_returned is not None:
                roi = float(capital_returned / invest_amt)
                net_cash = float(capital_returned - invest_amt)

            record = {
                'success': bool(outcome.get('success', False)),
                'roi': roi,
                'accuracy': 1.0 if outcome.get('ai_was_accurate') else (0.0 if outcome.get('ai_was_accurate') is False else None),
                'amount': invest_amt,
                'net_cash': net_cash,
            }

            if net_cash is not None:
                all_records.append(record.copy())

            if outcome.get('ai_used'):
                ai_records.append(record)
            elif invest_amt is not None:
                baseline_records.append(record)

        ai_stats = _compute_stats(ai_records, default_success_rate=0.0)
        ai_recent_stats = _compute_stats(ai_records[-recent_window:], default_success_rate=0.0)

        baseline_stats = _compute_stats(baseline_records, default_success_rate=0.5)
        baseline_recent_stats = _compute_stats(baseline_records[-recent_window:], default_success_rate=0.5)

        all_stats = _compute_stats(all_records, default_success_rate=0.0)
        all_recent_stats = _compute_stats(all_records[-recent_window:], default_success_rate=0.0)

        cumulative_attempts = float(getattr(self, 'ai_attempt_cumulative', 0))
        if cumulative_attempts > 0:
            cumulative_successes = float(getattr(self, 'ai_success_cumulative', 0))
            ai_stats['total'] = cumulative_attempts
            ai_stats['successes'] = cumulative_successes
            ai_stats['success_rate'] = float(cumulative_successes / cumulative_attempts)
            roi_count = getattr(self, 'ai_roi_count', 0)
            if roi_count > 0:
                ai_stats['avg_roi'] = float(self.ai_roi_sum / roi_count)
            accuracy_obs = getattr(self, 'ai_accuracy_observations', 0)
            if accuracy_obs > 0:
                ai_stats['accuracy_rate'] = float(self.ai_accuracy_cumulative / accuracy_obs)

        metrics = {
            'total_records': ai_stats['total'],
            'total_successes': ai_stats['successes'],
            'success_rate': ai_stats['success_rate'],
            'avg_roi': ai_stats['avg_roi'],
            'accuracy_rate': ai_stats['accuracy_rate'],
            'recent_success_rate': ai_recent_stats['success_rate'],
            'recent_avg_roi': ai_recent_stats['avg_roi'],
            'recent_accuracy_rate': ai_recent_stats['accuracy_rate'],
            'baseline_success_rate': baseline_stats['success_rate'],
            'baseline_avg_roi': baseline_stats['avg_roi'],
            'baseline_recent_success_rate': baseline_recent_stats['success_rate'],
            'baseline_recent_avg_roi': baseline_recent_stats['avg_roi'],
            'success_rate_gain': ai_stats['success_rate'] - baseline_stats['success_rate'],
            'recent_success_rate_gain': ai_recent_stats['success_rate'] - baseline_recent_stats['success_rate'],
            'roi_gain': ai_stats['avg_roi'] - baseline_stats['avg_roi'],
            'recent_roi_gain': ai_recent_stats['avg_roi'] - baseline_recent_stats['avg_roi'],
            'recent_ai_activity': ai_recent_stats['total'],
            'recent_total_invests': all_recent_stats['total'],
            'average_investment': ai_stats['avg_amount'],
            'recent_cash_flow_total': all_recent_stats['net_cash_total'],
            'overall_cash_flow_total': all_stats['net_cash_total'],
        }

        if metrics['total_records'] == 0.0:
            metrics['success_rate'] = baseline_stats['success_rate']
            metrics['recent_success_rate'] = baseline_recent_stats['success_rate']
            metrics['avg_roi'] = baseline_stats['avg_roi']
            metrics['recent_avg_roi'] = baseline_recent_stats['avg_roi']
            metrics['accuracy_rate'] = 0.0
            metrics['recent_accuracy_rate'] = 0.0

        metrics.update({
            'baseline_success_rate': baseline_stats['success_rate'],
            'baseline_recent_success_rate': baseline_recent_stats['success_rate'],
            'baseline_avg_roi': baseline_stats['avg_roi'],
            'baseline_recent_avg_roi': baseline_recent_stats['avg_roi'],
        })
        metrics['success_rate_gain'] = metrics['success_rate'] - metrics['baseline_success_rate']
        metrics['recent_success_rate_gain'] = metrics['recent_success_rate'] - metrics['baseline_recent_success_rate']
        metrics['roi_gain'] = metrics['avg_roi'] - metrics['baseline_avg_roi']
        metrics['recent_roi_gain'] = metrics['recent_avg_roi'] - metrics['baseline_recent_avg_roi']
        metrics['roi_gain_ratio'] = (
            (metrics['avg_roi'] / max(metrics['baseline_avg_roi'], 1e-3)) - 1.0
        )
        metrics['recent_roi_gain_ratio'] = (
            (metrics['recent_avg_roi'] / max(metrics['baseline_recent_avg_roi'], 1e-3)) - 1.0
        )

        return metrics

    def _choose_ai_level(
        self,
        neighbor_agents: Optional[List['EmergentAgent']],
        neighbor_signals: Optional[Dict[str, Any]] = None,
    ) -> str:
        base_trust = float(np.clip(self.traits.get('ai_trust', 0.5), 0.0, 1.0))
        uncertainty_tolerance = float(np.clip(self.traits.get('uncertainty_tolerance', 0.5), 0.0, 1.0))
        avoidance = 1.0 - uncertainty_tolerance
        metrics = self._compute_ai_performance_metrics()
        order = ['none', 'basic', 'advanced', 'premium']
        # Neutral priors - agents learn from experience
        prior_map = {
            'none': (2.0, 2.0),
            'basic': (2.0, 2.0),
            'advanced': (2.0, 2.0),
            'premium': (2.0, 2.0),
        }
        posterior_means: Dict[str, float] = {}
        for tier in order:
            default_alpha, default_beta = prior_map.get(tier, (2.5, 2.5))
            belief = self.ai_tier_beliefs.setdefault(tier, {"alpha": default_alpha, "beta": default_beta})
            total = belief["alpha"] + belief["beta"]
            posterior_means[tier] = float(belief["alpha"] / total) if total > 0 else 0.5

        usage_by_tier = dict(self.ai_tier_usage)

        peer_roi_signal = float(neighbor_signals.get('peer_roi_gap', 0.0)) if neighbor_signals else 0.0
        adoption_pressure = float(neighbor_signals.get('ai_adoption_pressure', 0.0)) if neighbor_signals else 0.0
        peer_distribution = neighbor_signals.get('ai_distribution', {}) if neighbor_signals else {}

        cash_buffer = max(self.resources.capital, self.resources.performance.initial_equity, 1.0)
        operating_cost = float(getattr(self, 'operating_cost_estimate', None) or getattr(self, 'operating_cost', 0.0) or self.config.BASE_OPERATIONAL_COST)
        recent_activity = max(1.0, metrics.get('recent_ai_activity', 1.0))
        amortization_horizon = max(1, int(getattr(self.config, "AI_SUBSCRIPTION_AMORTIZATION_ROUNDS", 20)))
        ref_scale = max(operating_cost * 4.0, cash_buffer * 0.12, 1.0)
        # Scale costs by AI_COST_INTENSITY (for robustness testing)
        cost_intensity = getattr(self.config, 'AI_COST_INTENSITY', 1.0)
        cost_ratios: Dict[str, float] = {}
        for tier in order:
            if tier == 'none':
                cost_ratios[tier] = 0.0
                continue
            cfg = self.config.AI_LEVELS.get(tier, {})
            cost_type = cfg.get('cost_type', 'none')
            base_cost = float(cfg.get('cost', 0.0)) * cost_intensity
            per_use_cost = float(cfg.get('per_use_cost', 0.0)) * cost_intensity
            if cost_type == 'subscription':
                per_round = base_cost / amortization_horizon
                total_cost = per_round + per_use_cost * recent_activity
            elif cost_type == 'per_use':
                per_call = base_cost if base_cost > 0 else per_use_cost
                total_cost = per_call * recent_activity
            else:
                total_cost = per_use_cost * recent_activity
            cost_ratios[tier] = float(total_cost / ref_scale)

        current_level = getattr(self, 'current_ai_level', 'none')
        if current_level not in order:
            current_level = 'none'

        base_roi_gain = float(metrics.get('recent_roi_gain', 0.0))
        long_roi_gain = float(metrics.get('roi_gain', 0.0))
        recent_roi_ratio = float(metrics.get('recent_roi_gain_ratio', 0.0))
        roi_gain_ratio = float(metrics.get('roi_gain_ratio', 0.0))
        net_cash_total = float(metrics.get('overall_cash_flow_total', 0.0))
        net_cash_recent = float(metrics.get('recent_cash_flow_total', 0.0))
        initial_equity = max(self.resources.performance.initial_equity, 1.0)
        capital_health = float(self.resources.capital / initial_equity) - 1.0
        paradox_signal = float(getattr(self, 'paradox_signal', 0.0))
        reserve_haircut = {'basic': 0.0, 'advanced': 0.02, 'premium': 0.05}

        score_records: Dict[str, Dict[str, float]] = {}
        scores: Dict[str, float] = {}
        for tier in order:
            posterior = posterior_means[tier]
            trust_term = posterior * base_trust
            roi_signal = (
                0.3 * base_roi_gain
                + 0.4 * long_roi_gain
                + 0.45 * recent_roi_ratio
                + 0.25 * roi_gain_ratio
            )
            roi_term = posterior * roi_signal
            cash_term = (
                0.2 * np.clip(net_cash_total / initial_equity, -1.5, 1.5)
                + 0.1 * np.clip(net_cash_recent / initial_equity, -1.5, 1.5)
            )
            peer_term = 0.08 * peer_roi_signal + 0.12 * adoption_pressure + 0.05 * peer_distribution.get(tier, 0.0)
            tier_experience = float(usage_by_tier.get(tier, 0))
            learning_relief = float(np.clip(np.log1p(tier_experience) * 0.02, 0.0, 0.12))
            cost_term = 0.25 * cost_ratios.get(tier, 0.0) * (1.0 - learning_relief)
            if tier == 'premium':
                cost_term *= 0.85
            switch_penalty = max(0.0, self._compute_ai_switch_penalty(current_level, tier) - learning_relief)
            reserve_penalty = reserve_haircut.get(tier, 0.0) * max(0.0, 1.0 - capital_health)
            # REMOVED: Tier-specific paradox response
            # Previously: premium/advanced penalized by paradox, none rewarded
            # Now agents should learn from their own experience with paradox conditions
            # The paradox effect should emerge from actual outcomes, not hardcoded tier rules
            paradox_term = 0.0
            noise = np.random.gumbel() * 0.02
            total_score = (
                trust_term
                + roi_term
                + cash_term
                + 0.25 * capital_health
                + peer_term
                - cost_term
                - reserve_penalty
                - switch_penalty
                - 0.05 * avoidance
                + paradox_term
                + noise
            )
            scores[tier] = total_score
            score_records[tier] = {
                "posterior": posterior,
                "cost_ratio": cost_ratios.get(tier, 0.0),
                "switch_penalty": switch_penalty,
                "score": total_score,
            }

        target_level = max(order, key=lambda lvl: (scores.get(lvl, -1e9), -order.index(lvl)))
        self.ai_tier_usage[target_level] += 1
        metrics.update(
            {
                "posterior_means": posterior_means,
                "cost_ratios": cost_ratios,
                "tier_scores": score_records,
                "chosen_level": target_level,
                "peer_roi_signal": peer_roi_signal,
                "peer_adoption_pressure": adoption_pressure,
            }
        )
        self._last_ai_tier_metrics = metrics
        return target_level


    def _estimate_ai_cost(self, ai_level: str, expected_calls: float = 1.0) -> float:
        ai_config = self.config.AI_LEVELS.get(ai_level, self.config.AI_LEVELS['none'])
        cost_type = ai_config.get('cost_type', 'none')
        # Scale costs by AI_COST_INTENSITY (for robustness testing)
        cost_intensity = getattr(self.config, 'AI_COST_INTENSITY', 1.0)
        if cost_type == 'subscription':
            if ai_level == 'none':
                return 0.0
            if self._subscription_accounts.get(ai_level, 0) > 0:
                installment = self._subscription_rates.get(ai_level, 0.0) * cost_intensity
            else:
                installment = 0.0
            per_use_cost = ai_config.get('per_use_cost', 0.0) * cost_intensity
            return installment + per_use_cost * max(expected_calls, 0.0)
        if cost_type == 'per_use':
            return ai_config.get('cost', 0.0) * cost_intensity * max(expected_calls, 0.0)
        return 0.0

    def _compute_ai_switch_penalty(self, current_level: str, proposed_level: str) -> float:
        if proposed_level == current_level:
            return 0.0
        order = ['none', 'basic', 'advanced', 'premium']
        if proposed_level not in order or current_level not in order:
            return 0.1
        distance = abs(order.index(proposed_level) - order.index(current_level))
        base_penalty = 0.05 * distance + 0.03
        experience = self.ai_tier_usage.get(proposed_level, 0)
        experience_modifier = 1.0 / (1.0 + 0.15 * experience)
        fatigue_penalty = 0.02 * max(0, self.ai_switches - 3)
        learning_relief = float(np.clip(np.log1p(experience) * 0.015, 0.0, 0.1))
        return max(0.0, (base_penalty * experience_modifier) + fatigue_penalty - learning_relief)

    def _update_ai_tier_belief(self, tier: str, realized_multiplier: Optional[float]) -> None:
        tier = normalize_ai_label(tier)
        if realized_multiplier is None:
            return
        # Neutral priors - agents learn from experience
        prior_map = {
            'none': (2.0, 2.0),
            'basic': (2.0, 2.0),
            'advanced': (2.0, 2.0),
            'premium': (2.0, 2.0),
        }
        alpha_default, beta_default = prior_map.get(tier, (2.0, 2.0))
        belief = self.ai_tier_beliefs.setdefault(tier, {"alpha": alpha_default, "beta": beta_default})
        multiplier = float(np.clip(realized_multiplier, 0.0, 3.0))
        evidence = float(np.clip(stable_sigmoid((multiplier - 1.0) * 2.5), 0.02, 0.98))
        belief["alpha"] += evidence
        belief["beta"] += (1.0 - evidence)

    def _estimate_operational_costs(self, market: 'MarketEnvironment') -> float:
        """Estimate quarterly operating expenses plus competition pressure.

        Uses sector-specific operational costs calibrated to SBA/BLS data (2025).
        Agent's primary sector is determined by their highest knowledge area.
        """
        portfolio_competition = 0.0
        if market is not None and getattr(self.portfolio, 'active_investments', None):
            active_opp_ids = self.portfolio.active_investments.keys()
            opp_competitions = [
                market.opportunity_map[opp_id].competition
                for opp_id in active_opp_ids
                if opp_id in market.opportunity_map
            ]
            if opp_competitions:
                portfolio_competition = float(fast_mean(opp_competitions))

        # Determine agent's primary sector from their knowledge distribution
        sector_profiles = self.config.SECTOR_PROFILES
        agent_sector = None
        knowledge = getattr(self, 'knowledge', {})
        if knowledge:
            max_knowledge = 0.0
            for sector, level in knowledge.items():
                if level > max_knowledge and sector in sector_profiles:
                    max_knowledge = level
                    agent_sector = sector

        # Default fallback cost
        base_cost = self.config.BASE_OPERATIONAL_COST

        # Use sector-specific operational cost if available (SBA/BLS calibrated)
        if agent_sector and agent_sector in sector_profiles:
            sector_profile = sector_profiles[agent_sector]
            cost_range = sector_profile.get('operational_cost_range')
            if cost_range and isinstance(cost_range, (list, tuple)) and len(cost_range) == 2:
                cost_mid = 0.5 * (cost_range[0] + cost_range[1])
                # Higher competence = lower costs (better cost management)
                competence = self.traits.get('competence', 0.5) if isinstance(self.traits, dict) else 0.5
                cost_adj = 1.0 - 0.2 * competence  # 0.8 to 1.0 multiplier
                base_cost = cost_mid * cost_adj

        competition_cost = portfolio_competition * self.config.COMPETITION_COST_MULTIPLIER
        total_cost = base_cost + competition_cost

        # AI tier efficiency modifier
        tier = normalize_ai_label(getattr(self, "current_ai_level", getattr(self, "agent_type", "none")))
        tier_mod = {
            "none": 1.0,
            "basic": 1.08,
            "advanced": 0.94,
            "premium": 0.88,
        }.get(tier, 1.0)
        total_cost *= tier_mod
        self.operating_cost = total_cost
        return total_cost

    def _calculate_agent_cost_of_capital(self, capital_utilization: float, market_conditions: Dict) -> float:
        """Estimate a hurdle rate tailored to the agent and current market stress."""
        base_rate = self.config.COST_OF_CAPITAL
        uncertainty_penalty = (0.5 - self.traits.get('uncertainty_tolerance', 0.5)) * 0.05
        competence_bonus = (self.traits.get('competence', 0.5) - 0.5) * 0.03
        market_volatility = (market_conditions or {}).get('volatility', self.config.MARKET_VOLATILITY)
        volatility_penalty = max(0.0, market_volatility - self.config.MARKET_VOLATILITY) * 0.1
        liquidity_penalty = capital_utilization * 0.05
        cost = base_rate + uncertainty_penalty + volatility_penalty + liquidity_penalty - competence_bonus
        return float(max(0.02, cost))

    def _choose_action_type(self, opportunities: List[Opportunity], market_conditions: Dict, 
                            info_system: 'InformationSystem', market_environment: 'MarketEnvironment', 
                            uncertainty_env: 'KnightianUncertaintyEnvironment', ai_level: str,
                            perception: Dict) -> str: 
    
        op_cost = getattr(self, 'operating_cost_estimate', None)
        if op_cost is None:
            op_cost = self._estimate_operational_costs(market_environment)
            self.operating_cost_estimate = op_cost
        expected_calls = 1.0 if ai_level != "none" else 0.0
        ai_cost = self._estimate_ai_cost(ai_level, expected_calls)
        total_estimated_cost = op_cost + ai_cost

        # Calculate raw utilities
        raw_utilities = {
            'invest': self._calculate_investment_utility(opportunities, market_conditions, info_system, perception, ai_level),
            'innovate': self._calculate_innovation_utility(market_conditions, perception),
            'explore': self._calculate_exploration_utility(perception, ai_level),
            'maintain': self._calculate_maintain_utility(market_conditions, perception, total_estimated_cost)
        }

        # Apply configurable base weights to scale utilities
        # This allows calibration of action shares to empirical targets
        utilities = {
            'invest': raw_utilities['invest'] * getattr(self.config, 'ACTION_BASE_WEIGHT_INVEST', 0.40),
            'innovate': raw_utilities['innovate'] * getattr(self.config, 'ACTION_BASE_WEIGHT_INNOVATE', 0.35),
            'explore': raw_utilities['explore'] * getattr(self.config, 'ACTION_BASE_WEIGHT_EXPLORE', 0.25),
            'maintain': raw_utilities['maintain'] * getattr(self.config, 'ACTION_BASE_WEIGHT_MAINTAIN', 0.20)
        }
        locked_capital = max(0.0, self.portfolio.locked_capital)
        locked_ratio = locked_capital / max(1.0, self.resources.capital + locked_capital)
        utilities['invest'] -= locked_ratio * 0.25
        actions = list(utilities.keys())
        utils = np.array(list(utilities.values()), dtype=float)
        utils = np.nan_to_num(utils, nan=0.0, posinf=0.0, neginf=0.0)

        ai_profile = self.config.AI_LEVELS.get(ai_level) or self.config.AI_LEVELS.get("none", {})
        info_quality = float(ai_profile.get("info_quality", 0.0))
        info_breadth = float(ai_profile.get("info_breadth", 0.0))
        trait_trust = float(np.clip(self.traits.get('ai_trust', 0.5), 0.0, 1.0))
        trait_explore = float(np.clip(self.traits.get('exploration_tendency', 0.5), 0.0, 1.0))

        base_temperature = max(1e-3, float(getattr(self.config, 'ACTION_SELECTION_TEMPERATURE', 0.22)))
        util_variance = float(np.std(utils)) if utils.size else 0.0
        temperature = max(
            5e-4,
            base_temperature * (1.0 + 0.6 * np.clip(util_variance, 0.0, 1.0)),
        )
        temp_scale = max(0.3, 1.0 - 0.6 * info_quality)
        temp_scale /= max(0.5, 0.9 + 0.2 * info_breadth)
        temperature *= temp_scale
        temperature *= float(np.clip(0.8 + 0.5 * (1.0 - trait_trust), 0.6, 1.4))
        noise_base = max(0.0, float(getattr(self.config, 'ACTION_SELECTION_NOISE', 0.03)))
        noise_scale = noise_base * (1.15 - 0.8 * info_quality) * (1.0 + 0.25 * (1.0 - info_breadth))
        noise_scale = max(0.0, noise_scale)
        noise_scale *= float(np.clip(0.85 + 0.4 * trait_explore, 0.7, 1.6))
        bias_vec = np.array([self.action_bias.get(action, 0.0) for action in actions], dtype=float)
        noise = np.random.gumbel(loc=0.0, scale=noise_scale, size=len(actions)) if noise_scale > 0 else 0.0

        logits = (utils + bias_vec + noise) / temperature
        logits = np.nan_to_num(logits, nan=-np.inf)
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        sum_exp = exp_logits.sum()
        if not np.isfinite(sum_exp) or sum_exp <= 0:
            probabilities = np.ones(len(actions), dtype=float) / len(actions)
        else:
            probabilities = exp_logits / sum_exp

        if getattr(self.config, 'ENABLE_DEBUG_LOGS', False):
            DEBUG_DECISION_LOG.append({
                'agent_id': self.id,
                'round': perception.get('_round'),
                'utilities': dict(zip(actions, utils.tolist())),
                'bias': dict(zip(actions, bias_vec.tolist())),
                'noise_scale': noise_scale,
                'temperature': temperature,
                'locked_ratio': locked_ratio,
                'utility_std': util_variance,
                'probabilities': dict(zip(actions, probabilities.tolist())),
            })

        chosen_action = str(np.random.choice(actions, p=probabilities))

        # Lightly update bias to reflect reinforcement (keeps heterogeneity dynamic)
        if chosen_action in self.action_bias:
            self.action_bias[chosen_action] = float(np.clip(
                self.action_bias[chosen_action] * 0.95 + 0.05 * (utils[actions.index(chosen_action)] - utils.mean()),
                -0.5,
                0.5,
            ))

        return chosen_action


    def _calculate_investment_utility(self, opportunities, market_conditions, info_system, perception, ai_level):
        if not opportunities:
            return 0.0
    
        evals, _ = self._evaluate_portfolio_opportunities(opportunities, info_system, market_conditions, ai_level, perception)
        if not evals:
            return 0.0
    
        max_score = max(float(e.get('final_score', 0.0)) for e in evals)

        scaled_score = stable_sigmoid(max_score - 1.0)
        knowledge_signal = perception.get('knowledge_signal') or perception.get('actor_ignorance', {})
        execution_signal = perception.get('execution_risk') or perception.get('practical_indeterminism', {})
        innovation_signal = perception.get('innovation_signal') or perception.get('agentic_novelty', {})
        competition_signal = perception.get('competition_signal') or perception.get('competitive_recursion', {})

        ignorance_level = float(knowledge_signal.get('ignorance_level', knowledge_signal.get('level', 0.5)))
        confidence = float(knowledge_signal.get('confidence', 0.5))
        gap_pressure = float(knowledge_signal.get('gap_pressure', 0.0))
        clamped_ignorance = np.clip(ignorance_level - 1.5, -20.0, 20.0)
        ignorance_adjustment = 0.5 + 0.5 / (1 + np.exp(clamped_ignorance)) if np.isfinite(clamped_ignorance) else 1.0
        ignorance_adjustment *= float(np.clip(0.85 + 0.35 * confidence, 0.5, 1.4))
        ignorance_adjustment += 0.1 * max(0.0, 0.35 - gap_pressure)
        # Directly penalize invest utility by the four uncertainty components
        actor_unc = float(perception.get('actor_ignorance', {}).get('level', ignorance_level))
        practical_unc = float(perception.get('practical_indeterminism', {}).get('level', execution_signal.get('indeterminism_level', 0.5)))
        agentic_unc = float(perception.get('agentic_novelty', {}).get('level', innovation_signal.get('novelty_potential', 0.5)))
        recursive_unc = float(perception.get('competitive_recursion', {}).get('level', competition_signal.get('recursion_level', 0.5)))
        decision_conf = float(np.clip(perception.get('decision_confidence', 0.5), 0.1, 0.95))
        capital_ratio = self.resources.capital / max(self.resources.performance.initial_equity, 1.0)
        liquidity_boost = np.clip(capital_ratio - 0.8, 0.0, 1.5)
        opportunity_boost = np.clip(len(opportunities) / 6.0, 0.0, 1.0)

        value = float(scaled_score) * float(ignorance_adjustment) * decision_conf

        perf = self.resources.performance
        invest_roic = float(perf.compute_roic('invest'))
        innovate_roic = float(perf.compute_roic('innovate'))
        idle_capital = max(0.0, self.resources.capital - self.portfolio.locked_capital)
        idle_ratio = idle_capital / max(self.resources.performance.initial_equity, 1.0)
        innovation_loss_pressure = max(0.0, -innovate_roic)
        ai_roi_gain = 0.0
        metrics = getattr(self, '_last_ai_tier_metrics', {})
        if metrics:
            ai_roi_gain = float(metrics.get('roi_gain', 0.0))

        value *= 0.7 + 0.3 * liquidity_boost
        value += 0.2 * opportunity_boost
        roic_multiplier = float(np.clip(1.0 + 0.35 * invest_roic, 0.4, 1.4))
        value *= roic_multiplier
        value += max(0.0, idle_ratio) * 0.25
        value += innovation_loss_pressure * 0.2
        locked_capital = max(0.0, self.portfolio.locked_capital)
        locked_ratio = locked_capital / max(1.0, self.resources.capital + locked_capital)
        value -= locked_ratio * 0.25
        value += 0.3 * np.clip(ai_roi_gain, -0.5, 1.0)
        peer_roi_signal = metrics.get('peer_roi_signal_logged')
        if peer_roi_signal is not None:
            value += 0.15 * np.clip(peer_roi_signal, -0.5, 1.0)
        if getattr(self.portfolio, "portfolio_size", 0) <= 0:
            value += 0.15
        tier_combo = float(innovation_signal.get('combo_rate', innovation_signal.get('tier_combo_rate', 0.0)))
        tier_reuse = float(innovation_signal.get('reuse_pressure', innovation_signal.get('tier_reuse_pressure', 0.0)))
        tier_new_rate = float(innovation_signal.get('new_possibility_rate', innovation_signal.get('tier_new_possibility_rate', 0.0)))
        value += 0.15 * (tier_combo - tier_reuse) + 0.05 * tier_new_rate
        # Uncertainty component hooks: lean into novelty, stay wary of recursion (lighter penalty)
        value += 0.15 * agentic_unc
        value -= 0.06 * recursive_unc
        risk_tolerance = float(self.traits.get('uncertainty_tolerance', 0.5))
        avoidance = 1.0 - risk_tolerance
        value *= float(np.clip(0.85 + 0.3 * risk_tolerance, 0.5, 1.4))
        value -= avoidance * 0.12
        # REMOVED: Tier-specific utility biases
        # Previously: premium/advanced penalized for tier_reuse, basic/none rewarded for low combo
        # Now tier effects emerge from information quality affecting evaluations, not direct utility mods
        paradox_signal = getattr(self, 'paradox_signal', 0.0)
        if paradox_signal:
            trust = float(self.traits.get('ai_trust', 0.5))
            tolerance = float(self.traits.get('uncertainty_tolerance', 0.5))
            if trust >= 0.6 or tolerance >= 0.6:
                value += 0.25 * paradox_signal
            else:
                value -= 0.35 * paradox_signal
        recent_actions = list(getattr(self, '_recent_actions', []))
        if recent_actions:
            consecutive_invests = 0
            for act in reversed(recent_actions):
                if act == 'invest':
                    consecutive_invests += 1
                else:
                    break
            if consecutive_invests:
                value -= 0.12 * consecutive_invests
            invest_share = recent_actions.count('invest') / len(recent_actions)
            value -= invest_share * 0.15
        if idle_ratio < 0.05:
            value -= 0.12
        if getattr(self.portfolio, "portfolio_size", 0) >= 4:
            value -= 0.07 * (self.portfolio.portfolio_size - 3)
        return float(np.clip(value, 0.0, 1.0))

    def _calculate_innovation_utility(self, market_conditions, perception):
        """Innovation driven BY uncertainty, not reduced by it"""
        base_drive = self.traits['innovativeness'] * 0.5 + 0.05

        execution_signal = perception.get('execution_risk') or perception.get('practical_indeterminism', {})
        innovation_signal = perception.get('innovation_signal') or perception.get('agentic_novelty', {})

        indeterminism_level = float(execution_signal.get('risk_level', execution_signal.get('indeterminism_level', 0.5)))
        indeterminism_bonus = indeterminism_level * 0.4

        novelty_potential = float(innovation_signal.get('novelty_potential', innovation_signal.get('novelty_level', 0.5)))
        innovation_opportunity = novelty_potential * 0.25
        scarcity_signal = float(innovation_signal.get('component_scarcity', 0.5))
        scarcity_drag = max(0.0, 0.6 - scarcity_signal) * 0.8

        capital_ratio = self.resources.capital / max(self.resources.performance.initial_equity, 1.0)
        liquidity_penalty = np.clip(1.0 - capital_ratio, 0.0, 1.5)
        rd_deployed = self.resources.performance.deployed_by_action.get('innovate', 0.0)
        rd_burden = np.clip(rd_deployed / max(self.resources.performance.initial_equity, 1.0), 0.0, 2.0)
        perf = self.resources.performance
        innovate_roic = float(perf.compute_roic('innovate'))
        if rd_deployed > 0:
            net_flow = perf.returned_by_action.get('innovate', 0.0) - rd_deployed
            loss_ratio = max(0.0, -innovate_roic)
            cash_burn_ratio = max(0.0, -net_flow / rd_deployed)
        else:
            loss_ratio = 0.0
            cash_burn_ratio = 0.0

        net_flow = perf.returned_by_action.get('innovate', 0.0) - perf.deployed_by_action.get('innovate', 0.0)
        net_flow_penalty = np.clip(-net_flow / max(1.0, perf.deployed_by_action.get('innovate', 1.0)), 0.0, 1.0)

        risk_tolerance = float(self.traits.get('uncertainty_tolerance', 0.5))
        avoidance = 1.0 - risk_tolerance

        raw_score = (
            base_drive
            + indeterminism_bonus
            + innovation_opportunity
            - 0.25 * liquidity_penalty
            - 0.3 * rd_burden
            - scarcity_drag
            - 0.4 * loss_ratio
            - 0.3 * cash_burn_ratio
            + 0.2 * net_flow_penalty
            + 0.2 * (risk_tolerance - 0.5)
            - 0.15 * avoidance
        )
        paradox_signal = getattr(self, 'paradox_signal', 0.0)
        if paradox_signal:
            pivot_weight = max(0.0, 0.6 - risk_tolerance)
            surge_weight = max(0.0, risk_tolerance - 0.4)
            raw_score += 0.2 * pivot_weight * paradox_signal
            raw_score -= 0.15 * surge_weight * paradox_signal
        return float(np.clip(stable_sigmoid(raw_score - 0.6), 0.02, 0.98))

    def _calculate_exploration_utility(self, perception, ai_level):
        """Exploration utility reacts to live ignorance gaps, novelty droughts, and liquidity slack."""
        base_tendency = self.traits['exploration_tendency']

        knowledge_signal = perception.get('knowledge_signal') or perception.get('actor_ignorance', {})
        execution_signal = perception.get('execution_risk') or perception.get('practical_indeterminism', {})
        innovation_signal = perception.get('innovation_signal') or perception.get('agentic_novelty', {})
        competition_signal = perception.get('competition_signal') or perception.get('competitive_recursion', {})

        ignorance_level = float(knowledge_signal.get('ignorance_level', knowledge_signal.get('level', 0.5)))
        gap_pressure = float(knowledge_signal.get('gap_pressure', 0.0) or 0.0)
        ignorance_drive = ignorance_level * 0.35 + gap_pressure * 0.25

        cache = self._uncertainty_cache if self._uncertainty_cache_round == perception.get('_round') else None
        knowledge_deficit = None
        if cache is not None:
            knowledge_deficit = cache.get("knowledge_deficit")
        if knowledge_deficit is None:
            avg_knowledge = fast_mean(self.resources.knowledge.values())
            if not np.isfinite(avg_knowledge):
                avg_knowledge = 0.0
            knowledge_deficit = max(0.0, min(1.0, 1 - avg_knowledge))
            if cache is not None:
                cache["knowledge_deficit"] = knowledge_deficit
        else:
            if not np.isfinite(knowledge_deficit):
                knowledge_deficit = 1.0
            else:
                knowledge_deficit = float(np.clip(knowledge_deficit, 0.0, 1.0))
            if cache is not None:
                cache["knowledge_deficit"] = knowledge_deficit

        novelty_potential = float(innovation_signal.get('novelty_potential', innovation_signal.get('novelty_level', 0.5)))
        novelty_gap = max(0.0, 0.5 - novelty_potential)
        component_scarcity = float(innovation_signal.get('component_scarcity', 0.5) or 0.5)
        scarcity_push = max(0.0, 0.65 - component_scarcity)
        practical_volatility = float(execution_signal.get('volatility', 0.2))

        locked_capital = max(0.0, self.portfolio.locked_capital)
        capital_slack = max(0.0, (self.resources.capital - locked_capital) / max(self.resources.performance.initial_equity, 1.0))
        capital_ratio = self.resources.capital / max(self.resources.performance.initial_equity, 1.0)

        avg_knowledge = fast_mean(self.resources.knowledge.values())
        rich_knowledge_penalty = np.clip(avg_knowledge - 0.25, 0.0, 1.0)
        buffer_penalty = np.clip(capital_ratio - 1.3, 0.0, 1.0)

        idle_fraction = max(0.0, capital_slack - 0.1)

        ai_trust_level = float(self.traits.get('ai_trust', 0.5))
        trust_penalty = max(0.0, ai_trust_level - 0.5) * 0.45
        recursion_level = float(competition_signal.get('pressure_level', competition_signal.get('recursion_level', 0.5)))
        recursion_penalty = max(0.0, recursion_level - 0.5) * 0.2
        ai_usage_share = float(competition_signal.get('ai_usage_share', 0.0))
        capital_crowding = float(competition_signal.get('capital_crowding', 0.25))
        recursion_penalty += 0.08 * max(0.0, ai_usage_share - 0.45)
        recursion_penalty += 0.1 * max(0.0, capital_crowding - 0.4)
        # REMOVED: Tier-specific recursion penalty adjustments
        # Previously: premium/advanced penalized more when AI usage high, basic/none rewarded
        # Now tier effects emerge from information quality, not hardcoded utility modifications

        momentum_bonus = 0.0
        explore_roic = float(self.resources.performance.compute_roic('explore'))
        if np.isfinite(explore_roic):
            momentum_bonus = np.clip(explore_roic, -0.5, 0.5) * 0.2

        risk_tolerance = float(self.traits.get('uncertainty_tolerance', 0.5))
        avoidance = 1.0 - risk_tolerance
        recent_explore_penalty = 0.0
        invest_streak = 0
        invest_bias = 0.0
        history: List[str] = []
        if hasattr(self, "_recent_actions"):
            history = list(self._recent_actions)
            if history and history[-1] == 'explore':
                recent_explore_penalty += 0.15
            if len(history) >= 3:
                streak = sum(1 for act in history if act == 'explore')
                if streak >= len(history) - 1:
                    recent_explore_penalty += 0.25
            for act in reversed(history):
                if act == 'invest':
                    invest_streak += 1
                else:
                    break
            if history:
                invest_bias = history.count('invest') / len(history)

        raw_score = (
            base_tendency * 0.35
            + ignorance_drive
            + knowledge_deficit * 0.25
            + 0.2 * novelty_gap
            + 0.15 * scarcity_push
            + 0.08 * capital_slack
            - 0.25 * idle_fraction
            + 0.12 * practical_volatility
            + momentum_bonus
            - 0.35 * rich_knowledge_penalty
            - 0.2 * buffer_penalty
            - trust_penalty
            - recursion_penalty
            + 0.18 * avoidance
            - 0.08 * risk_tolerance
            + 0.15 * invest_streak
            + 0.08 * invest_bias
            - recent_explore_penalty
        )
        raw_score += 0.18 * recursion_level
        paradox_signal = getattr(self, 'paradox_signal', 0.0)
        if paradox_signal:
            pivot_weight = max(0.0, 0.6 - risk_tolerance)
            raw_score += 0.2 * pivot_weight * paradox_signal
        return float(np.clip(stable_sigmoid(raw_score - 0.45), 0.05, 0.95))

    def _calculate_maintain_utility(self, market_conditions, perception, estimated_cost):
        """Simplified maintain utility"""
        capital_buffer_in_rounds = self.resources.capital / (estimated_cost + 1e-9)
        buffer_pressure = stable_sigmoid(2.0 - capital_buffer_in_rounds)
    
        diversification = self.portfolio.diversification_score
        execution_signal = perception.get('execution_risk') or perception.get('practical_indeterminism', {})
        indeterminism_level = float(execution_signal.get('risk_level', execution_signal.get('indeterminism_level', 0.5)))
        uncertainty_penalty = indeterminism_level * 0.3
    
        idle_penalty = min(
            0.2,
            (self.resources.capital / max(self.config.INITIAL_CAPITAL, 1.0)) * 0.1,
        )
        avg_knowledge = fast_mean(self.resources.knowledge.values())
        knowledge_penalty = np.clip(avg_knowledge - 0.25, 0.0, 1.0) * 0.2
        surplus_buffer_penalty = np.clip(capital_buffer_in_rounds - 3.0, 0.0, 3.0) * 0.1
    
        maintain_utility = 0.2 + buffer_pressure * 0.5 + diversification * 0.3 - uncertainty_penalty - idle_penalty
        maintain_utility -= knowledge_penalty + surplus_buffer_penalty
        risk_tolerance = float(self.traits.get('uncertainty_tolerance', 0.5))
        avoidance = 1.0 - risk_tolerance
        maintain_utility += 0.2 * avoidance
        maintain_utility -= 0.1 * risk_tolerance
        paradox_signal = getattr(self, 'paradox_signal', 0.0)
        if paradox_signal:
            maintain_utility += 0.15 * max(0.0, avoidance) * paradox_signal
            maintain_utility -= 0.1 * max(0.0, -paradox_signal)
        return float(np.clip(maintain_utility, 0.05, 0.95))

    def _calculate_and_deduct_operational_costs(self, market: 'MarketEnvironment'):
        """
        Calculates and deducts the operational cost for the round using the estimator.
        """
        # Pass the 'market' object down to the estimator
        cost = self._estimate_operational_costs(market)
        self.resources.capital -= cost
        return cost

    def _make_innovation_decision(self, market_conditions: Dict, round_num: int, ai_level: str, uncertainty_env: Optional[Any], perception: Optional[Dict] = None) -> Dict:
        innovation = self.innovation_engine.attempt_innovation(self, market_conditions, round_num, ai_level)
        ai_config = self.config.AI_LEVELS.get(ai_level, self.config.AI_LEVELS['none'])
        per_use_cost = 0.0
        if ai_level != 'none' and ai_config.get('cost_type') == 'per_use':
            per_use_cost = float(ai_config.get('cost', ai_config.get('per_use_cost', 0.0)))

        max_rd_cap_fraction = float(getattr(self.config, 'INNOVATION_RD_CAP_FRACTION', 0.15))
        rd_cap = max_rd_cap_fraction * max(self.resources.performance.initial_equity, 1.0)

        perception = perception or getattr(self, '_last_perception', {}) or {}
        decision_confidence = float(np.clip(perception.get('decision_confidence', 0.5), 0.05, 0.99))
        innovation_signal = perception.get('innovation_signal') or perception.get('agentic_novelty', {})
        knowledge_signal = perception.get('knowledge_signal') or perception.get('actor_ignorance', {})

        if innovation:
            self.innovations.append(innovation)
            setattr(innovation, "ai_level_used", ai_level)
            deployed = self.resources.performance.deployed_by_action.get('innovate', 0.0)
            returned = self.resources.performance.returned_by_action.get('innovate', 0.0)
            net_flow = returned - deployed
            flow_ratio = net_flow / max(deployed, 1.0)
            recent_roic = self.resources.performance.compute_roic('innovate')
            ai_trust = float(self.traits.get('ai_trust', 0.5))
            agentic_view = innovation_signal
            novelty_level = float(agentic_view.get('novelty_potential', agentic_view.get('novelty_level', 0.5)))
            novelty_gap = max(0.0, 0.55 - novelty_level)
            new_poss_rate = float(agentic_view.get('new_possibility_rate', 0.0) or 0.0)
            new_poss_gap = max(0.0, 0.12 - new_poss_rate)
            gap_pressure = float(knowledge_signal.get('gap_pressure', 0.0) or 0.0)
            scarcity_signal = float(agentic_view.get('component_scarcity', 0.5) or 0.5)
            last_innovation_round = getattr(self, '_last_innovation_round', None)
            if last_innovation_round is None:
                rounds_since_innovation = round_num + 1  # force backlog pressure on first attempt
            else:
                rounds_since_innovation = max(0, round_num - last_innovation_round)
            backlog_threshold = max(4, int(getattr(self.config, 'INNOVATION_BACKLOG_ROUNDS', 10)))
            backlog_pressure = max(0, rounds_since_innovation - backlog_threshold)
            signal = (
                1.2 * np.clip(recent_roic, -1.0, 2.0)
                + 0.8 * np.clip(flow_ratio, -1.0, 2.0)
                + 0.4 * (ai_trust - 0.5)
                + 0.6 * novelty_gap
                + 0.5 * gap_pressure
                + 0.4 * new_poss_gap
            )
            if backlog_pressure > 0:
                signal += min(0.45, backlog_pressure * 0.04)
            # Uncertainty component hooks: lean into novelty, downweight recursion lightly
            agentic_unc = float(perception.get('agentic_novelty', {}).get('level', novelty_level))
            recursive_unc = float(perception.get('competitive_recursion', {}).get('level', perception.get('competition_signal', {}).get('recursion_level', 0.5)))
            signal += 0.4 * agentic_unc
            signal -= 0.05 * recursive_unc
            logistic_fraction = float(1 / (1 + np.exp(-signal)))
            adaptive_cap = np.clip(max_rd_cap_fraction + 0.15 * (novelty_gap + gap_pressure), 0.05, 0.25)
            target_fraction = np.clip(
                0.02
                + logistic_fraction * 0.18
                + 0.08 * (novelty_gap + new_poss_gap)
                + 0.05 * gap_pressure
                - 0.04 * max(0.0, 0.35 - scarcity_signal),
                0.0,
                adaptive_cap,
            )
            rd_investment = min(self.resources.capital * target_fraction, getattr(self.config, 'INNOVATION_MAX_SPEND', 10000.0))
            rd_investment = min(rd_investment, rd_cap)
            if backlog_pressure > 0:
                floor_fraction = float(getattr(self.config, 'INNOVATION_MIN_FLOOR_SPEND', 0.01))
                rd_investment = max(rd_investment, floor_fraction * self.resources.performance.initial_equity)
            rd_investment = min(rd_investment, self.resources.capital)
            if rd_investment <= 0:
                return self._make_maintain_decision(round_num, ai_level)
            self.resources.capital -= rd_investment
            self.resources.performance.record_deployment('innovate', rd_investment, ai_level, round_num)
            self.resources.knowledge_last_used[innovation.sector] = round_num
            self._last_innovation_round = round_num
        
            innovation_details = {
                'id': innovation.id,
                'type': innovation.type,
                'novelty': innovation.novelty,
                'quality': innovation.quality,
                'sector': innovation.sector
            }
        
            # The single, correct return for a successful innovation
            return {
                'action': 'innovate',
                'success': True,
                'innovation_details': innovation_details, # For saving to disk
                'innovation_obj': innovation,             # For the simulation loop to use
                'is_new_combination': bool(getattr(innovation, 'is_new_combination', False)),
                'combination_signature': getattr(innovation, 'combination_signature', None),
                'rd_investment': rd_investment,
                'capital_deployed': rd_investment,
                'ai_level_used': ai_level,
                'ai_estimated_return': None,
                'ai_estimated_uncertainty': None,
                'ai_confidence': None,
                'ai_contains_hallucination': False,
                'ai_analysis_domain': None,
                'ai_actual_accuracy': None,
                'ai_overconfidence_factor': None,
                'portfolio_size': self.portfolio.portfolio_size,
                'ai_per_use_cost': per_use_cost,
                'ai_call_count': 1 if per_use_cost > 0 else 0,
                'decision_confidence': decision_confidence,
                'paradox_signal': self.paradox_signal,
            }
        else:
            if rd_cap <= 0 or self.resources.capital <= 0:
                return self._make_maintain_decision(round_num, ai_level)
            return {
                'action': 'innovate', 
                'success': False, 
                'is_new_combination': False,
                'combination_signature': None,
                'rd_investment': 0.0,
                'capital_deployed': 0.0,
                'ai_level_used': ai_level,
                'ai_estimated_return': None,
                'ai_estimated_uncertainty': None,
                'ai_confidence': None,
                'ai_contains_hallucination': False,
                'ai_analysis_domain': None,
                'ai_actual_accuracy': None,
                'ai_overconfidence_factor': None,
                'portfolio_size': self.portfolio.portfolio_size,
                'ai_per_use_cost': per_use_cost,
                'ai_call_count': 1 if per_use_cost > 0 else 0,
                'decision_confidence': decision_confidence,
                'paradox_signal': self.paradox_signal,
            }
        
    def _make_investment_decision(
        self,
        opportunities: List[Opportunity],
        info_system: 'InformationSystem',
        market_conditions: Dict,
        market_environment: 'MarketEnvironment',
        round_num: int,
        ai_level: str,
        perception: Dict,
    ) -> Dict:
        
        evals, _ = self._evaluate_portfolio_opportunities(
            opportunities, info_system, market_conditions, ai_level, perception
        )
        return self._make_portfolio_decision(
            evals, market_environment, round_num, ai_level, perception, market_conditions
        )

    def _make_exploration_decision(self, round_num: int, ai_level: str) -> Dict:
        ai_level = normalize_ai_label(ai_level or getattr(self, "current_ai_level", getattr(self, "agent_type", "none")))
        # FIXED: Remove hardcoded breadth_multiplier - let effect emerge through info_breadth
        # Previously had direct tier bonuses (none=0.85, premium=1.35)
        # Now multiplier emerges from AI_LEVELS config info_breadth parameter
        ai_cfg = self.config.AI_LEVELS.get(ai_level, self.config.AI_LEVELS.get('none', {}))
        info_breadth = float(ai_cfg.get('info_breadth', 0.0))
        # Map info_breadth (0.0-0.85) to multiplier range (~0.85-1.36)
        breadth_multiplier = 0.85 + info_breadth * 0.6
        trait_factor = (0.02 + self.traits['exploration_tendency'] * 0.07) * breadth_multiplier
        uncertainty_cushion = max(0.02, 0.12 - self.traits['uncertainty_tolerance'] * 0.08)
        # Uncertainty hooks: high ignorance/recursion push explore up
        last_view = getattr(self, '_last_perception', {}) or {}
        actor_unc = float((last_view.get('actor_ignorance') or {}).get('level', 0.5))
        recursion_unc = float((last_view.get('competitive_recursion') or {}).get('level', 0.5))
        trait_factor *= float(np.clip(1.0 + 0.2 * actor_unc + 0.15 * recursion_unc, 0.6, 1.8))
        desired_cost = self.resources.capital * (trait_factor + uncertainty_cushion)
        desired_cost *= float(np.clip(np.random.normal(1.0, 0.25), 0.4, 1.8))
        cost = min(desired_cost, 5000)
        self.resources.capital -= cost
        self.resources.performance.record_deployment('explore', cost, 'none', round_num)

        # Uncertainty hooks: tilt explore propensity by component levels
        actor_unc = float((getattr(self, '_last_perception', {}) or {}).get('actor_ignorance', {}).get('level', 0.5))
        practical_unc = float((getattr(self, '_last_perception', {}) or {}).get('practical_indeterminism', {}).get('level', 0.5))
        agentic_unc = float((getattr(self, '_last_perception', {}) or {}).get('agentic_novelty', {}).get('level', 0.5))
        recursion_unc = float((getattr(self, '_last_perception', {}) or {}).get('competitive_recursion', {}).get('level', 0.5))
        trait_factor *= float(np.clip(1.0 + 0.15 * agentic_unc - 0.1 * recursion_unc, 0.5, 1.5))
        uncertainty_cushion *= float(np.clip(1.0 + 0.1 * actor_unc + 0.05 * practical_unc, 0.5, 1.6))

        # Determine exploration type based on market saturation
        if hasattr(self, '_last_perception'):
            last_view = getattr(self, '_last_perception', {}) or {}
            innovation_snapshot = last_view.get('innovation_signal') or last_view.get('agentic_novelty', {})
            innovation_saturation = 1.0 - float(innovation_snapshot.get('novelty_potential', 0.5))
        else:
            innovation_saturation = 0.5
    
        exploration_result = {}

        if np.random.random() < innovation_saturation:
            # High saturation: Explore for new market niches
            exploration_type = 'niche_discovery'
        
            # Create a new market niche/subsector
            existing_sectors = list(self.config.SECTOR_PROFILES.keys())
            base_sector = np.random.choice(existing_sectors)
        
            # Generate niche characteristics
            niche_modifiers = ['premium', 'budget', 'sustainable', 'digital', 'local', 'specialized']
            niche_modifier = np.random.choice(niche_modifiers)
            niche_id = f"{base_sector}_{niche_modifier}"
        
            # Add knowledge about this new niche
            if niche_id not in self.resources.knowledge:
                self.resources.knowledge[niche_id] = np.random.uniform(0.2, 0.4)
            else:
                self.resources.knowledge[niche_id] = min(1.0, self.resources.knowledge[niche_id] + 0.2)
            self.resources.knowledge_last_used[niche_id] = round_num
            self.resources.knowledge_last_used[base_sector] = round_num
        
            trait_amp = (0.7 + 0.6 * self.traits.get('exploration_tendency', 0.5)) * breadth_multiplier
            self.resources.knowledge[niche_id] = min(
                1.0, self.resources.knowledge[niche_id] * (0.9 + 0.2 * trait_amp)
            )
            exploration_result = {
                'discovered_niche': niche_id,
                'base_sector': base_sector,
                'niche_knowledge': self.resources.knowledge[niche_id]
            }

            # Higher chance of serendipitous discovery in new niches
            serendipity_chance = np.clip(
                (0.18 + 0.4 * (self.traits.get('exploration_tendency', 0.5) - 0.3)) * breadth_multiplier,
                0.05,
                0.7,
            )
            if np.random.random() < serendipity_chance:
                serendipity_reward = cost * np.random.uniform(2.0, 6.0)
                self.resources.capital += serendipity_reward
                self.resources.performance.record_return('explore', serendipity_reward, 'none', round_num)
                exploration_result['serendipity_reward'] = serendipity_reward
        else:
            # Low saturation: Traditional sector exploration
            exploration_type = 'sector_knowledge'
            domain = np.random.choice(list(self.config.SECTOR_PROFILES.keys()))
            trait_amp = (0.65 + 0.5 * self.traits.get('exploration_tendency', 0.5)) * breadth_multiplier
            knowledge_gain = np.random.uniform(0.08, 0.28) * trait_amp
            self.resources.knowledge[domain] = min(1.0, self.resources.knowledge.get(domain, 0) + knowledge_gain)
            self.resources.knowledge_last_used[domain] = round_num

            exploration_result = {
                'domain': domain,
                'knowledge_gained': True
            }

            # Standard serendipity chance
            serendipity_chance = np.clip(
                (0.12 + 0.25 * (self.traits.get('exploration_tendency', 0.5) - 0.4)) * breadth_multiplier,
                0.04,
                0.55,
            )
            if np.random.random() < serendipity_chance:
                serendipity_reward = cost * np.random.uniform(1.5, 4.0)
                self.resources.capital += serendipity_reward
                self.resources.performance.record_return('explore', serendipity_reward, 'none', round_num)
                exploration_result['serendipity_reward'] = serendipity_reward
    
        return {
            'action': 'explore',
            'exploration_type': exploration_type,
            'created_niche': bool(exploration_result.get('discovered_niche')),
            'cost': cost,
            'capital_deployed': cost,
            'ai_level_used': ai_level,
            'ai_estimated_return': None,
            'ai_estimated_uncertainty': None,
            'ai_confidence': None,
            'ai_contains_hallucination': False,
            'ai_analysis_domain': None,
            'ai_actual_accuracy': None,
            'ai_overconfidence_factor': None,
            'portfolio_size': self.portfolio.portfolio_size,
            'ai_per_use_cost': 0.0,
            'ai_call_count': 0,
            'paradox_signal': self.paradox_signal,
            **exploration_result,
        }
    
    def _make_maintain_decision(self, round_num: int, ai_level: str) -> Dict:
        # Use uncertainty to tilt maintain: hedge in high practical indeterminism, back off when recursion is high, slight lift when novelty is low.
        last_view = getattr(self, '_last_perception', {}) or {}
        actor_unc = float((last_view.get('actor_ignorance') or {}).get('level', 0.5))
        practical_unc = float((last_view.get('practical_indeterminism') or {}).get('level', 0.5))
        agentic_unc = float((last_view.get('agentic_novelty') or {}).get('level', 0.5))
        recursion_unc = float((last_view.get('competitive_recursion') or {}).get('level', 0.5))
        maintain_bias = 0.0
        maintain_bias += 0.10 * practical_unc
        maintain_bias += 0.08 * max(0.0, 0.5 - agentic_unc)
        maintain_bias -= 0.12 * recursion_unc
        return {
            'action': 'maintain', 
            'round': round_num, 
            'reason': 'managing_existing_portfolio', 
            'ai_level_used': ai_level, 
            'ai_estimated_return': None,
            'ai_estimated_uncertainty': None,
            'ai_confidence': None,
            'ai_contains_hallucination': False,
            'ai_analysis_domain': None,
            'ai_actual_accuracy': None,
            'ai_overconfidence_factor': None,
            'portfolio_size': self.portfolio.portfolio_size,
            'ai_per_use_cost': 0.0,
            'ai_call_count': 0,
            'paradox_signal': self.paradox_signal,
            'maintain_bias': maintain_bias,
        }
    
    def update_state_from_outcome(self, outcome: Dict, ai_was_accurate: Optional[bool] = None):
        """Extended to update uncertainty response learning"""

        # Original update logic
        if ai_was_accurate is not None:
            outcome['ai_was_accurate'] = bool(ai_was_accurate)
        self.outcomes.append(outcome)
        if 'capital_returned' in outcome:
            self.resources.capital += outcome['capital_returned']

        roi = outcome.get('roi')
        invest_amt = outcome.get('investment_amount')
        if invest_amt is None:
            invest_amt = (outcome.get('investment') or {}).get('amount')
        try:
            invest_amt = float(invest_amt) if invest_amt is not None else None
        except (TypeError, ValueError):
            invest_amt = None
        capital_returned = outcome.get('capital_returned')
        try:
            capital_returned = float(capital_returned) if capital_returned is not None else None
        except (TypeError, ValueError):
            capital_returned = None

        if roi is None and invest_amt and invest_amt > 0 and capital_returned is not None:
            roi = float(capital_returned / invest_amt)
            outcome['roi'] = roi

        if outcome.get('ai_used'):
            self.ai_attempt_cumulative += 1
            if outcome.get('success', False):
                self.ai_success_cumulative += 1
            if roi is not None:
                self.ai_roi_sum += roi
                self.ai_roi_count += 1
                tier_label = normalize_ai_label(outcome.get('ai_level_used', 'none'))
                self._update_ai_tier_belief(tier_label, roi)
            if ai_was_accurate is not None:
                self.ai_accuracy_observations += 1
                if ai_was_accurate:
                    self.ai_accuracy_cumulative += 1

        # Update traits as before
        was_successful = outcome.get('success', False)
        ai_was_used = outcome.get('ai_used', False)
        
        ai_analysis_obj = outcome.get('ai_analysis')
        if ai_was_used:
            opp_id = None
            if 'chosen_opportunity_details' in outcome:
                opp_id = outcome['chosen_opportunity_details'].get('id')
            if opp_id is None and 'opportunity_id' in outcome:
                opp_id = outcome.get('opportunity_id')

        # Call the trait evolution method with all necessary information
        self._evolve_traits_from_experience(
            successful_action=was_successful, 
            used_ai=ai_was_used, 
            ai_analysis=ai_analysis_obj,
            ai_was_accurate=ai_was_accurate
        )
    
    def record_paradox_observation(
        self,
        decision_confidence: Optional[float],
        realized_roi: Optional[float],
        ai_used: bool = False,
        counterfactual_roi: Optional[float] = None,
    ) -> None:
        if decision_confidence is None or realized_roi is None:
            return
        try:
            decision_conf = float(decision_confidence)
        except (TypeError, ValueError):
            decision_conf = 0.5
        decision_conf = float(np.clip(decision_conf, 0.05, 0.99))
        try:
            roi_value = float(realized_roi)
        except (TypeError, ValueError):
            roi_value = 1.0
        roi_value = float(np.clip(roi_value, 0.0, 3.0))

        def _roi_to_score(multiplier: float) -> float:
            return float(1.0 / (1.0 + np.exp(-3.0 * (np.clip(multiplier, 0.0, 3.0) - 1.0))))

        roi_score = _roi_to_score(roi_value)
        gap = decision_conf - roi_score
        cf_gap = 0.0
        if counterfactual_roi is not None and ai_used:
            cf_gap = roi_score - _roi_to_score(counterfactual_roi)

        weight = 1.2 if ai_used else 0.8
        combined = gap + 0.5 * cf_gap
        self.paradox_history.append(
            {
                "confidence": decision_conf,
                "roi": roi_value,
                "gap": combined,
            }
        )
        inertia = 0.85
        updated_signal = inertia * self.paradox_signal + (1 - inertia) * combined * weight
        self.paradox_signal = float(np.clip(updated_signal, -1.0, 1.0))
        self.last_paradox_gap = float(gap)
        self.last_paradox_cf_gap = float(cf_gap)

    def _evaluate_opportunity(self, opportunity: Opportunity, info: Information, market_conditions: Dict) -> float:
        # --- 1. Initial Quantitative Score Calculation ---
        expected_profit_margin = info.estimated_return - 1.0
        uncertainty_adjusted_margin = expected_profit_margin * (1 - info.estimated_uncertainty * 0.5)
        score = uncertainty_adjusted_margin * (1.0 + self.traits['uncertainty_tolerance'])
    
        # Apply market regime modifier
        regime_multiplier = {'boom': 1.2, 'growth': 1.1, 'normal': 1.0, 'volatile': 0.9, 'crisis': 0.7}.get(market_conditions.get('regime', 'normal'), 1.0)
        score *= regime_multiplier
    
        # Apply sector knowledge bonus
        if hasattr(opportunity, 'sector'):
            score *= (1 + (self.resources.knowledge.get(opportunity.sector, 0.1) * 0.5))

        # Sector crowding penalty - discount opportunities in crowded sectors
        # This encourages diversification across sectors
        clearing_index = market_conditions.get('sector_clearing_index', {}) or {}
        sector = getattr(opportunity, 'sector', 'unknown')
        sector_crowding = float(clearing_index.get(sector, 0.0) or 0.0)
        if sector_crowding > 1.0:
            # Apply penalty: score * (1 / (1 + 0.3 * excess_crowding))
            # At crowding=2 (1 excess): multiplier = 0.77
            # At crowding=4 (3 excess): multiplier = 0.53
            crowding_penalty = 1.0 / (1.0 + 0.3 * (sector_crowding - 1.0))
            score *= crowding_penalty

        # --- 2. Logic for Qualitative Insights ---
        # Define insight modifiers
        insight_modifiers = {
            "Requires specialized capabilities": {
                "effect": "complexity_modifier", "value": 1.25 # Increases perceived complexity
            },
            "Strong first-mover advantages": {
                "effect": "score_multiplier", "value": 1.1 # Makes it more attractive
            },
            "High competitive pressure detected": {
                "effect": "uncertainty_modifier", "value": 1.15 # Increases perceived uncertainty
            },
            # This is a potential hallucination from Cell 7
            "Untapped customer segment identified in emerging markets": {
                 "effect": "return_modifier", "value": 1.2 # Increases perceived return
            }
        }
    
        # Create temporary variables representing the agent's perception
        perceived_complexity = opportunity.complexity if hasattr(opportunity, 'complexity') else 0.5
        perceived_uncertainty = info.estimated_uncertainty
        perceived_return = info.estimated_return

        # In the section "Logic for Qualitative Insights"
        for insight in info.insights:
            if insight in insight_modifiers:
                mod = insight_modifiers[insight]
                # Introduce heterogeneity based on analytical_ability
                trait_multiplier = 1.0 + (self.traits['analytical_ability'] - 0.5) * 0.2

                if mod["effect"] == "complexity_modifier":
                    perceived_complexity *= mod["value"] * trait_multiplier
                elif mod["effect"] == "score_multiplier":
                    score *= mod["value"] * trait_multiplier
                elif mod["effect"] == "uncertainty_modifier":
                    perceived_uncertainty *= mod["value"] * (2.0 - trait_multiplier) # Inverse for uncertainty
                elif mod["effect"] == "return_modifier":
                    perceived_return *= mod["value"] * trait_multiplier

        # --- 3. Final Score Adjustment Based on Perceptions ---
        complexity_penalty = 1.0 - (perceived_complexity * (1 - self.traits['analytical_ability']) * 0.1)
    
        score *= complexity_penalty
    
        # The final score is scaled by the AI's confidence in its analysis
        # FIXED: Remove artificial floor that dampened confidence signal
        # Previously (0.5 + conf * 0.5) compressed 2.6x confidence diff to 1.44x
        # Now confidence has realistic impact on opportunity evaluation
        score *= (0.15 + info.confidence * 0.85)
    
        return max(0.1, score)

    def _evaluate_portfolio_opportunities(self, opportunities: List[Opportunity], info_system: 'InformationSystem', 
                                     market_conditions: Dict, ai_level: str, perception: Dict) -> Tuple[List[Dict], float]:
        evaluations = []
        evaluation_cache: Dict[Tuple[str, str], Dict] = {}

        ai_config = self.config.AI_LEVELS.get(ai_level, self.config.AI_LEVELS['none'])
        cost_type = ai_config.get('cost_type', 'none')
        profile = getattr(self, 'ai_learning', None)
        per_use_cost = 0.0

        opportunity_pool = list(opportunities or [])
        if not opportunity_pool:
            return [], 0.0

        def _opp_quality(opportunity: Opportunity) -> float:
            return float(getattr(opportunity, "latent_return_potential", 1.0) or 1.0)

        # REMOVED: Tier-based pre-filtering that gave premium/advanced first look at best opportunities
        # Previously: premium saw top 65% sorted by quality, none saw random 60%
        # Now ALL agents evaluate ALL opportunities - better AI tiers will make better selections
        # through more accurate information quality, not through pre-filtered access
        # This lets selection quality emerge from the information system, not hardcoded filtering

        for opp in opportunity_pool:
            domain = info_system._determine_domain(opp) if hasattr(info_system, '_determine_domain') else 'market_analysis'
            effective_ai_level = ai_level

            if ai_level != 'none' and profile is not None:
                capital_buffer = max(self.portfolio.get_available_capital(self.resources.capital), 0.0)
                usage = max(profile.usage_count.get(domain, 0), self.ai_domain_exposure.get(domain, 0))
                if usage >= 5:
                    expected_ai_cost = self._estimate_ai_cost(ai_level, expected_calls=1.0)
                    if not profile.should_use_ai_for_domain(domain, expected_ai_cost, capital_buffer, cost_type=cost_type):
                        effective_ai_level = 'none'

            opp_id = getattr(opp, 'id', None)
            cache_key = (opp_id, effective_ai_level)
            cached = evaluation_cache.get(cache_key)
            if cached is not None:
                evaluations.append(cached)
                continue

            info = info_system.get_information(opp, effective_ai_level, agent=self)
            if effective_ai_level != 'none' and cost_type == 'per_use':
                per_call = float(ai_config.get('cost', ai_config.get('per_use_cost', 0.0)) or 0.0)
                per_use_cost += per_call

            analysis_obj = getattr(info, '_source_analysis', None)
            if analysis_obj is not None:
                self.ai_analysis_history.append(analysis_obj)
                if hasattr(analysis_obj, 'domain'):
                    self.ai_domain_exposure[analysis_obj.domain] += 1
            elif getattr(info, 'domain', None):
                self.ai_domain_exposure[info.domain] += 1
            base_score = self._evaluate_opportunity(opp, info, market_conditions)
            final_score = self._apply_uncertainty_adjustments(base_score, opp, perception)
            neighbor_signals = perception.get('neighbor_signals', {}) if isinstance(perception, dict) else {}
            if neighbor_signals:
                interest_map = neighbor_signals.get('opportunity_interest', {})
                sentiment_map = neighbor_signals.get('opportunity_sentiment', {})
                sector_sentiment_map = neighbor_signals.get('sector_sentiment', {})
                interest_weight = float(getattr(self.config, 'NEIGHBOR_SOCIAL_WEIGHT', 0.1))
                sentiment_weight = float(getattr(self.config, 'NEIGHBOR_SENTIMENT_WEIGHT', 0.15))
                interest_boost = float(interest_map.get(opp_id, 0.0))
                sentiment_adjust = float(sentiment_map.get(opp_id, 0.0))
                sector_adjust = float(sector_sentiment_map.get(getattr(opp, 'sector', None), 0.0))
                final_score *= (1.0 + interest_weight * interest_boost)
                final_score += sentiment_weight * (sentiment_adjust + 0.5 * sector_adjust)
            final_score = max(0.01, float(final_score))
            record = {
                'opportunity': opp,
                'info': info,
                'final_score': final_score,
                'ai_level_used': effective_ai_level,
                'analysis_domain': domain,
            }
            evaluations.append(record)
            if opp_id is not None:
                evaluation_cache[cache_key] = record
        return evaluations, per_use_cost


    def _make_portfolio_decision(
        self,
        evaluated_opportunities: List[Dict],
        market_environment: 'MarketEnvironment',
        round_num: int,
        ai_level: str,
        perception: Dict,
        market_conditions: Dict,
    ) -> Dict:
        """Make portfolio decision based on evaluated opportunities with capital constraints."""

        current_capital = max(0.0, self.resources.capital)
        locked_capital = max(0.0, self.portfolio.locked_capital)
        base_capital = max(0.0, self.portfolio.get_available_capital(current_capital) - locked_capital)

        operating_estimate = getattr(self, 'operating_cost_estimate', None)
        if operating_estimate is None:
            operating_estimate = self._estimate_operational_costs(market_environment)
        self.operating_cost_estimate = operating_estimate

        reserve_months = float(getattr(self.config, 'OPERATING_RESERVE_MONTHS', 3))
        operating_reserve = operating_estimate * max(1.0, reserve_months)
        ai_reserve = 0.0
        ai_cfg = self.config.AI_LEVELS.get(ai_level, self.config.AI_LEVELS['none'])
        if ai_cfg.get('cost_type') == 'subscription':
            ai_reserve = ai_cfg.get('cost', 0.0)
        elif ai_cfg.get('cost_type') == 'per_use':
            ai_reserve = ai_cfg.get('cost', 0.0)

        reserve_fraction = float(getattr(self.config, 'LIQUIDITY_RESERVE_FRACTION', 0.3))
        # Use sector-specific survival threshold (BLS/Fed calibrated)
        sector_threshold = self._get_sector_survival_threshold()
        buffer_target = max(
            operating_reserve + ai_reserve + sector_threshold,
            current_capital * reserve_fraction,
        )
        liquidity_buffer = min(current_capital, buffer_target) + locked_capital * 0.2  # reserve some against commitments

        free_capital = current_capital - liquidity_buffer
        if free_capital <= locked_capital:
            if getattr(self.config, 'ENABLE_DEBUG_LOGS', False):
                DEBUG_PORTFOLIO_LOG.append({
                    'agent_id': self.id,
                    'round': round_num,
                    'reason': 'locked_capital_limit',
                    'available_capital': 0.0,
                    'liquidity_buffer': liquidity_buffer,
                    'locked_capital': locked_capital,
                    'current_capital': current_capital,
                    'ai_level_used': ai_level,
                })
            return self._make_maintain_decision(round_num, ai_level)

        available_capital = max(0.0, free_capital - locked_capital)

        ai_cfg = self.config.AI_LEVELS.get(ai_level, self.config.AI_LEVELS['none'])
        if ai_cfg.get('cost_type') == 'per_use':
            available_capital = max(0.0, available_capital - ai_cfg.get('cost', 0.0))

        # If too much capital is locked up, use a dynamic threshold based on recent scores
        total_commitments = current_capital + self.portfolio.locked_capital
        capital_utilization = self.portfolio.locked_capital / max(1, total_commitments)
        scores = np.array([e.get('final_score', 0.0) for e in evaluated_opportunities], dtype=float)
        volatility = float(market_conditions.get('volatility', self.config.MARKET_VOLATILITY))
        regime_failure = float(market_conditions.get('regime_failure_multiplier', 1.0) or 1.0)
        risk_aversion = max(0.0, 1.0 - float(self.traits.get('uncertainty_tolerance', 0.5)))
        gap_pressure = float((perception.get('knowledge_signal') or {}).get('gap_pressure', 0.0))
        stress = float(np.clip(0.4 * capital_utilization + 0.6 * volatility + 0.3 * (regime_failure - 1.0) + 0.4 * gap_pressure, 0.0, 1.8))
        if scores.size:
            base_q = 0.60
            q = float(np.clip(base_q + 0.25 * stress + 0.1 * risk_aversion, 0.6, 0.98))
            if scores.size == 1:
                baseline = float(scores[0])
            else:
                idx = (scores.size - 1) * q
                lower = int(np.floor(idx))
                upper = int(np.ceil(idx))
                if lower == upper:
                    baseline = float(np.partition(scores.copy(), lower)[lower])
                else:
                    partitioned = np.partition(scores.copy(), (lower, upper))
                    lower_val = float(partitioned[lower])
                    upper_val = float(partitioned[upper])
                    baseline = lower_val + (upper_val - lower_val) * (idx - lower)
        else:
            baseline = 0.0
        noise = np.random.normal(0, baseline * 0.18) if baseline > 0 else 0.0
        agent_cost_of_capital = self._calculate_agent_cost_of_capital(capital_utilization, market_conditions)
        dynamic_threshold = (baseline * (0.85 + 0.4 * capital_utilization + 0.35 * stress)) + noise
        hurdle = agent_cost_of_capital * (1.0 + 0.6 * stress + 0.4 * risk_aversion)
        decision_threshold = max(hurdle, dynamic_threshold, 0.02)

        best_eval = max(
            evaluated_opportunities, key=lambda item: item.get('final_score', 0.0)
        ) if evaluated_opportunities else None
        best_score = best_eval.get('final_score', 0) if best_eval else None

        if best_eval is None or best_score < decision_threshold:
            if getattr(self.config, 'ENABLE_DEBUG_LOGS', False):
                DEBUG_PORTFOLIO_LOG.append({
                    'agent_id': self.id,
                    'round': round_num,
                    'reason': 'threshold',
                    'best_score': best_score,
                    'decision_threshold': decision_threshold,
                    'available_capital': available_capital,
                    'capital_utilization': capital_utilization,
                    'ai_level_used': ai_level,
                })
            return self._make_maintain_decision(round_num, ai_level)

        # Select best opportunity
        effective_ai_level = best_eval.get('ai_level_used', ai_level)
        opp = best_eval.get('opportunity')
        info = best_eval.get('info')
        analysis_obj = getattr(info, '_source_analysis', None)

        if not opp or not info:
            return self._make_maintain_decision(round_num, effective_ai_level)

        # Calculate investment amount based on available capital
        confidence = perception.get('decision_confidence', 0.5)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.5
        if not np.isfinite(confidence):
            confidence = 0.5
        confidence = float(np.clip(confidence, 0.15, 0.95))
        signal_score = max(0.0, best_eval.get('final_score', 0))
        hurdle_rate = agent_cost_of_capital
        if signal_score < hurdle_rate:
            return self._make_maintain_decision(round_num, ai_level)
        max_fraction = float(getattr(self.config, 'MAX_INVESTMENT_FRACTION', 0.1))
        target_fraction = float(getattr(self.config, 'TARGET_INVESTMENT_FRACTION', 0.07))
        max_investment = available_capital * max_fraction
        desired_investment = available_capital * target_fraction * confidence * signal_score
        amount = min(desired_investment, max_investment)

        required_capital = float(getattr(opp, 'capital_requirements', 0) or 0.0)
        min_fraction = getattr(self.config, 'MIN_FUNDING_FRACTION', 0.25)
        min_required = required_capital * min_fraction if required_capital > 0 else 0.0

        if required_capital > 0 and amount < min_required and available_capital >= min_required:
            amount = min(max_investment, min_required)

        amount = min(amount, available_capital)
        amount = max(0.0, amount)

        if amount <= 0:
            if getattr(self.config, 'ENABLE_DEBUG_LOGS', False):
                DEBUG_PORTFOLIO_LOG.append({
                    'agent_id': self.id,
                    'round': round_num,
                    'reason': 'amount_zero',
                    'available_capital': available_capital,
                    'decision_confidence': confidence,
                    'signal_score': signal_score,
                    'hurdle_rate': hurdle_rate,
                    'computed_amount': amount,
                    'max_investment': max_investment,
                    'desired_investment': desired_investment,
                })
            return self._make_maintain_decision(round_num, ai_level)

        # Make the investment with maturation tracking
        time_to_maturity = getattr(opp, 'time_to_maturity', 12)
        self.resources.capital = max(0.0, self.resources.capital - amount)
        final_ai_cfg = self.config.AI_LEVELS.get(effective_ai_level, self.config.AI_LEVELS['none'])
        ai_call_count = 0
        per_use_cost = 0.0
        if final_ai_cfg.get('cost_type') == 'per_use' and effective_ai_level != 'none':
            used_ai = any(
                normalize_ai_label(record.get('ai_level_used', 'none')) != 'none'
                for record in evaluated_opportunities
            )
            if used_ai:
                per_call_cost = float(final_ai_cfg.get('cost', final_ai_cfg.get('per_use_cost', 0.0)))
                per_use_cost = per_call_cost
                ai_call_count = 1



        investment_sector = getattr(opp, 'sector', 'unknown')
        self.portfolio.add_investment(
            opp.id,
            amount,
            investment_sector,
            round_num,
            time_to_maturity,
            opp,
            effective_ai_level,
            info,
            decision_confidence=confidence,
        )
        self.resources.knowledge_last_used[investment_sector] = round_num
        if getattr(self.config, 'ENABLE_DEBUG_LOGS', False):
            DEBUG_PORTFOLIO_LOG.append({
                'agent_id': self.id,
                'round': round_num,
                'reason': 'invest',
                'amount': amount,
                'available_capital_post': self.resources.capital,
                'decision_confidence': confidence,
                'signal_score': signal_score,
                'hurdle_rate': hurdle_rate,
                'liquidity_buffer': liquidity_buffer,
                'locked_capital': self.portfolio.locked_capital,
                'ai_level_used': effective_ai_level,
            })
        self.resources.performance.record_deployment('invest', amount, effective_ai_level, round_num)

        funding_ratio = amount / max(required_capital, amount) if required_capital > 0 else 1.0

        opportunity_details = {
            'id': getattr(opp, 'id', 'unknown'),
            'sector': getattr(opp, 'sector', 'unknown'),
            'estimated_return': getattr(info, 'estimated_return', 1.0),
            'estimated_uncertainty': getattr(info, 'estimated_uncertainty', 0.5),
            'confidence': getattr(info, 'confidence', 0.5),
            'time_to_maturity': time_to_maturity,
            'maturation_round': round_num + time_to_maturity,
            'funding_ratio': funding_ratio
        }
        if best_eval.get('analysis_domain') is not None:
            opportunity_details['analysis_domain'] = best_eval['analysis_domain']

        derivative_origin = getattr(opp, 'origin_innovation_id', None)
        return {
            'action': 'invest',
            'executed': True,
            'success': None,
            'outcome_pending': True,
            'chosen_opportunity_obj': opp,
            'chosen_opportunity_details': opportunity_details,
            'invested_derivative': derivative_origin is not None,
            'derivative_origin': derivative_origin,
            'amount': amount,
            'amount_injected': amount,
            'investment_amount': amount,
            'capital_deployed': amount,
            'expected_return': opportunity_details['estimated_return'],
            'expected_uncertainty': opportunity_details['estimated_uncertainty'],
            'available_capital': available_capital,
            'locked_capital': self.portfolio.locked_capital,
            'ai_level_used': effective_ai_level,
            'ai_estimated_return': getattr(info, 'estimated_return', None),
            'ai_estimated_uncertainty': getattr(info, 'estimated_uncertainty', None),
            'ai_confidence': getattr(info, 'confidence', None),
            'ai_contains_hallucination': getattr(info, 'contains_hallucination', None),
            'ai_analysis_domain': getattr(analysis_obj, 'domain', getattr(info, 'domain', None)),
            'ai_actual_accuracy': getattr(analysis_obj, 'actual_accuracy', getattr(info, 'actual_accuracy', None)),
            'ai_overconfidence_factor': getattr(info, 'overconfidence_factor', None),
            'perception_at_decision': perception,
            'portfolio_size': self.portfolio.portfolio_size,
            'ai_per_use_cost': per_use_cost,
            'ai_call_count': ai_call_count,
            'decision_confidence': confidence,
            'paradox_signal': self.paradox_signal,
        }


    def _apply_uncertainty_adjustments(self, base_score: float, opportunity: Opportunity, perception: Dict) -> float:
        adjusted_score = base_score
        knowledge_signal = perception.get('knowledge_signal') or perception.get('actor_ignorance', {})
        execution_signal = perception.get('execution_risk') or perception.get('practical_indeterminism', {})
        knowledge_gaps = knowledge_signal.get('knowledge_gaps', perception.get('actor_ignorance', {}).get('knowledge_gaps', {}))
        timing_pressure = execution_signal.get('timing_pressure', perception.get('practical_indeterminism', {}).get('timing_pressure', {}))
        if opportunity.id in knowledge_gaps:
            adjusted_score *= (1 - knowledge_gaps[opportunity.id] * 0.5)
        if isinstance(timing_pressure, dict) and opportunity.id in timing_pressure:
            adjusted_score *= 0.8
        if opportunity.created_by is not None:
            adjusted_score *= 1.1
            if opportunity.created_by == self.id: adjusted_score *= 1.2

        social_proof_sensitivity = 1.0 - self.traits.get('analytical_ability', 0.5)
        social_proof_bonus = 1.0 + (opportunity.competition * social_proof_sensitivity * 0.25) # e.g., up to 12.5% bonus
        adjusted_score *= social_proof_bonus
        
        if opportunity.competition > 0.5:
            adjusted_score *= (1 - (self.traits.get('uncertainty_tolerance', 0.5) * opportunity.competition * 0.5))
        return max(0.01, adjusted_score)

    def _evolve_traits_from_experience(self, successful_action: bool, used_ai: bool, ai_analysis: Optional[AIAnalysis] = None, ai_was_accurate: Optional[bool] = None):
        """
        Evolves agent traits. Trust evolution is now based on the AI's analytical accuracy,
        while competence evolves based on the overall action success.
        """
        # --- Trust evolution logic ---
        if used_ai and ai_analysis and ai_was_accurate is not None:
            self.ai_interactions_count += 1
            current_trust = self.traits['ai_trust']
            base_adjustment_rate = self.config.AI_TRUST_ADJUSTMENT_RATE

            attribution_factor = 1.0 - self.traits['analytical_ability']
        
            surprise_factor = 1.0
            # A bad outcome from a high-confidence prediction is more damaging
            if not ai_was_accurate and ai_analysis.confidence > 0.75:
                surprise_factor = 1.5

            effective_adjustment_rate = base_adjustment_rate * attribution_factor * surprise_factor

            # THIS IS THE KEY CHANGE: Use the specific accuracy of the AI, not the general success of the action
            if ai_was_accurate:
                self.traits['ai_trust'] += (1.0 - current_trust) * effective_adjustment_rate
            else:
                self.traits['ai_trust'] -= current_trust * effective_adjustment_rate
        
            self.traits['ai_trust'] = np.clip(self.traits['ai_trust'], 0, 1)

        # --- Competence evolution logic (unchanged) ---
        # This correctly remains tied to the overall success of the action
        target_competence = 1.0 if successful_action else 0.0
        current_competence = self.traits['competence']
    
        base_momentum = self.traits['trait_momentum']
        cognitive_multiplier = self.traits['cognitive_style']
        effective_momentum = base_momentum * cognitive_multiplier

        self.traits['competence'] = (effective_momentum * current_competence) + \
                            ((1 - effective_momentum) * target_competence)
    
        self.traits['competence'] = np.clip(self.traits['competence'], 0, 1)

    def _update_strategic_mode(self, market_conditions: Dict):
        recent_success_rate = safe_mean([o.get('success', 0.5) for o in self.outcomes]) if self.outcomes else 0.5
        market_volatility = market_conditions.get('volatility', 0.2)
        self.market_volatility_history.append(market_volatility)

        success_values = np.array([o.get('success', 0.5) for o in self.outcomes], dtype=float) if self.outcomes else np.array([0.5])
        if success_values.size < 10:
            smoothing_prior = np.full(10, 0.5, dtype=float)
            success_values = np.concatenate([success_values, smoothing_prior])
        exploit_threshold = float(np.quantile(success_values, 0.7))
        diversify_threshold = float(np.quantile(success_values, 0.3))

        volatility_values = np.array(self.market_volatility_history, dtype=float)
        volatility_diversify_threshold = float(np.quantile(volatility_values, 0.65)) if volatility_values.size else 0.2

        if recent_success_rate >= exploit_threshold:
            self.strategy_mode = "exploiting"
        elif market_volatility >= volatility_diversify_threshold or recent_success_rate <= diversify_threshold:
            self.strategy_mode = "diversifying"
        else:
            self.strategy_mode = "exploring"


class IntegratedAgent(EmergentAgent):
    """Agent with fixed AI level representing specific adoption tier."""

    def __init__(
        self,
        agent_id,
        traits,
        config,
        knowledge_base,
        innovation_engine,
        agent_type="human",
    ):
        super().__init__(agent_id, traits, config, knowledge_base, innovation_engine, agent_type=agent_type)
        self.agent_type = agent_type
        self.fixed_ai_level = "none" if agent_type == "human" else agent_type.replace("_ai", "")

    def make_decision(
        self,
        opportunities,
        market_conditions,
        info_system,
        market_environment,
        round_num,
        all_agents,
        uncertainty_env,
        neighbor_agents=None,
        ai_level_override=None,
    ):
        return super().make_decision(
            opportunities=opportunities,
            market_conditions=market_conditions,
            info_system=info_system,
            market_environment=market_environment,
            round_num=round_num,
            all_agents=all_agents,
            uncertainty_env=uncertainty_env,
            neighbor_agents=neighbor_agents,
            ai_level_override=self.fixed_ai_level,
        )
