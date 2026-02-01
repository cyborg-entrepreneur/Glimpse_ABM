"""
Core dataclasses used across the Glimpse ABM simulation.

This module defines the fundamental data structures that represent the
entities and interactions in the agent-based model of entrepreneurship
under Knightian uncertainty with AI augmentation.

Theoretical Foundation
----------------------
The data structures in this module operationalize concepts from:

    Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).
    Are the futures computable? Knightian uncertainty & artificial intelligence.
    Academy of Management Review, 50(2), 415-440.

Key Concepts Operationalized
----------------------------
- **Opportunities**: Market opportunities with latent characteristics that
  manifest through entrepreneurial action, reflecting the generative nature
  of entrepreneurial futures (Sarasvathy, 2001).

- **Information**: AI-generated analyses that may contain hallucinations,
  biases, and overconfidence—operationalizing the "paradox of future knowledge"
  where AI improves information access while potentially creating new
  uncertainty sources.

- **Innovation**: Novel combinations of knowledge components, representing
  the "agentic novelty" dimension of Knightian uncertainty where genuinely
  new possibilities emerge through entrepreneurial creativity.

- **AILearningProfile**: Agent-specific learned understanding of AI capabilities,
  enabling adaptive trust calibration based on experience with AI accuracy
  and hallucination rates across domains.

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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import EmergentConfig
_DEFAULT_CONFIG: Optional[EmergentConfig] = None


def _get_default_config() -> EmergentConfig:
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = EmergentConfig()
    return _DEFAULT_CONFIG


@dataclass(slots=True)
class Information:
    """Information about an opportunity - Enhanced for AI learning."""

    estimated_return: float
    estimated_uncertainty: float
    confidence: float
    insights: Tuple[str, ...]
    hidden_factors: Dict[str, float]
    domain: Optional[str] = None
    actual_accuracy: Optional[float] = None
    contains_hallucination: bool = False
    bias_applied: float = 0.0
    overconfidence_factor: float = 1.0
    _source_analysis: Optional["AIAnalysis"] = field(default=None, repr=False, init=False)

    def quality_score(self) -> float:
        """Calculate information quality score."""
        base_quality = self.confidence * (1 - abs(self.hidden_factors.get("bias", 0)))
        return base_quality / max(1.0, self.overconfidence_factor)


@dataclass(slots=True)
class Opportunity:
    """
    Market opportunity with latent characteristics that manifest through action.

    This class operationalizes the entrepreneurial concept that opportunities
    are not simply "discovered" but are enacted through the interplay of
    entrepreneurial action and market conditions. The latent characteristics
    become realized returns only through the act of investment, reflecting
    Sarasvathy's (2001) effectuation logic and the generative view of
    entrepreneurial futures in Townsend et al. (2025).

    Attributes
    ----------
    id : str
        Unique identifier for the opportunity.
    latent_return_potential : float
        The underlying return multiple if the opportunity succeeds.
        This is NOT the realized return—actual returns depend on market
        conditions, competition, timing, and execution path.
    latent_failure_potential : float
        Base probability of failure, modified by market regime, competition,
        and lifecycle stage. Represents the inherent risk independent of
        the investor's actions.
    complexity : float
        Cognitive and operational complexity of the opportunity, affecting
        the difficulty of evaluation and execution.
    sector : str, optional
        Economic sector (tech, retail, service, manufacturing) determining
        risk-return profile and capital requirements.
    lifecycle_stage : str
        Current stage: "emerging", "growing", "mature", or "declining".
        Affects both return potential and failure probability.
    capital_requirements : float
        Minimum investment required to pursue the opportunity.
    time_to_maturity : int
        Rounds until the investment matures and returns are realized.
    combination_signature : str, optional
        Knowledge combination that created this opportunity (for innovations).
    component_scarcity : float
        Scarcity of the knowledge components underlying this opportunity.
        Higher scarcity increases potential returns (supply-demand dynamics)
        and represents the novelty dimension of uncertainty.

    Notes
    -----
    The separation between latent potential and realized returns is central
    to modeling Knightian uncertainty. The latent_return_potential represents
    what "could be" while the realized_return() method computes what actually
    manifests given the full context of market conditions, competitive dynamics,
    and stochastic execution paths.

    See Also
    --------
    realized_return : Method that converts latent potential into actual returns.
    """

    id: str
    latent_return_potential: float
    latent_failure_potential: float
    complexity: float
    discovered: bool = False
    created_by: Optional[int] = None
    discovery_round: Optional[int] = None
    config: Optional[EmergentConfig] = field(default=None, repr=False)
    competition: float = 0.0
    lifecycle_stage: str = "emerging"
    path_dependency: float = 0.0
    sector: Optional[str] = None
    capital_requirements: float = 10000
    capital_history: List[float] = field(default_factory=list)
    realized_returns: List[float] = field(default_factory=list)
    time_to_maturity: int = 12
    market_share: float = 1.0
    entry_barriers: float = 1.0
    age: int = 0
    base_failure_potential: Optional[float] = None
    truly_unknown: bool = False
    required_discovery_threshold: float = 0.0
    combination_uncertainty: float = 0.5
    combination_signature: Optional[str] = None
    market_impact: float = 0.0
    origin_innovation_id: Optional[str] = None
    crowding_penalty: float = 0.0
    component_scarcity: float = 0.5

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = _get_default_config()

        if self.latent_return_potential is None:
            sector_profile = None
            if (
                self.config
                and hasattr(self.config, "SECTOR_PROFILES")
                and self.sector in self.config.SECTOR_PROFILES
            ):
                sector_profile = self.config.SECTOR_PROFILES[self.sector]
            if sector_profile:
                low, high = sector_profile["return_range"]
                baseline = max((low + high) / 2, 0.2)
                sigma = 0.25 + np.clip(len(self.realized_returns), 0, 10) * 0.01
                self.latent_return_potential = float(
                    np.random.lognormal(mean=np.log(baseline), sigma=sigma)
                )
            else:
                low, high = getattr(self.config, "OPPORTUNITY_RETURN_RANGE", (0.8, 3.0))
                self.latent_return_potential = float(np.random.uniform(low, high))

        if self.latent_failure_potential is None:
            self.latent_failure_potential = 0.7

        if not hasattr(self, "lifecycle_stage"):
            self.lifecycle_stage = "emerging"
        if not hasattr(self, "age"):
            self.age = 0
        if not hasattr(self, "base_failure_potential"):
            self.base_failure_potential = self.latent_failure_potential

        if hasattr(self.config, "OPPORTUNITY_RETURN_RANGE"):
            min_ret, max_ret = self.config.OPPORTUNITY_RETURN_RANGE
            self.latent_return_potential = max(
                min_ret, min(max_ret, self.latent_return_potential)
            )

        if hasattr(self.config, "OPPORTUNITY_UNCERTAINTY_RANGE"):
            min_uncertainty, max_uncertainty = self.config.OPPORTUNITY_UNCERTAINTY_RANGE
            self.latent_failure_potential = max(
                min_uncertainty, min(max_uncertainty, self.latent_failure_potential)
            )

    def realized_return(self, market_conditions: Dict[str, Any], investor_tier: Optional[str] = None) -> float:
        """
        Convert latent return potential into a realized investment multiple.

        This method operationalizes the core insight that entrepreneurial returns
        are not predetermined but emerge through the interaction of opportunity
        characteristics, market conditions, competitive dynamics, and execution
        contingencies. The stochastic elements represent practical indeterminism—
        the irreducible unpredictability of execution paths even with perfect
        information (Townsend et al., 2025, Proposition 2).

        Parameters
        ----------
        market_conditions : Dict[str, Any]
            Current market state including:
            - regime: Macroeconomic regime (crisis/recession/normal/growth/boom)
            - regime_return_multiplier: Regime effect on returns
            - regime_failure_multiplier: Regime effect on failure probability
            - volatility: Current market volatility level
            - sector_clearing_index: Supply-demand clearing by sector
            - crowding_metrics: Capital concentration measures
            - uncertainty_state: Four-dimensional uncertainty measurements
            - tier_invest_share: Investment share by AI tier (for herding penalties)
        investor_tier : str, optional
            AI tier of the investing agent ("none", "basic", "advanced", "premium").
            Used to compute tier-specific crowding penalties when multiple agents
            with the same AI tier invest in similar opportunities.

        Returns
        -------
        float
            Realized investment multiple. Values interpretation:
            - < 1.0: Loss (capital destruction)
            - = 1.0: Break-even (capital returned)
            - > 1.0: Gain (value creation)
            - Can be negative for total wipeouts (limited by RETURN_LOWER_BOUND)

        Notes
        -----
        The return calculation proceeds through several stages:

        1. **Base Return Adjustment**: Latent return modified by macroeconomic
           regime (boom/crisis effects on venture returns).

        2. **Lifecycle and Resilience**: Stage-specific multipliers reflecting
           the S-curve of opportunity development.

        3. **Market Clearing Dynamics**: Supply-demand imbalances create over/
           undersupply adjustments. Crowded opportunities face return penalties;
           contrarian investments may earn premiums.

        4. **Novelty and Scarcity Effects**: Component scarcity (from agentic
           novelty dimension) affects return potential. Scarce, novel combinations
           command higher returns; commoditized patterns face compression.

        5. **Power Law Return Distribution**: Returns are drawn from a Pareto
           distribution matching empirical VC return patterns (Kaplan & Schoar,
           Korteweg & Sorensen). Most investments cluster near the minimum while
           rare "unicorns" drive portfolio returns. The shape parameter α controls
           tail heaviness: α ≈ 2.0 for early-stage VC, α ≈ 2.5 for growth equity.

        6. **Downside Risk Realization**: Tail risk materialization based on
           failure potential, market conditions, and oversupply pressure.

        The method explicitly models how AI-induced herding (through tier_invest_share)
        can reduce returns through correlated investment behavior, operationalizing
        the competitive recursion dimension of Knightian uncertainty.

        References
        ----------
        Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).
            Are the futures computable? Knightian uncertainty & artificial
            intelligence. Academy of Management Review, 50(2), 415-440.
        """
        base_multiple = float(np.clip(self.latent_return_potential or 1.0, 0.3, 10.0))
        regime_return = float(market_conditions.get("regime_return_multiplier", 1.0) or 1.0)
        base_multiple *= regime_return
        risk_signal = float(np.clip(self.latent_failure_potential, 0.05, 0.95))
        regime_failure = float(market_conditions.get("regime_failure_multiplier", 1.0) or 1.0)
        risk_signal = float(np.clip(risk_signal * regime_failure, 0.05, 0.95))
        lifecycle_multipliers = {
            "emerging": 1.12,
            "growing": 1.03,
            "mature": 0.9,
            "declining": 0.65,
        }
        lifecycle_factor = lifecycle_multipliers.get(self.lifecycle_stage, 1.0)
        resilience = np.clip(0.75 + 0.25 * (1.0 - risk_signal), 0.55, 1.45)

        clearing_index = market_conditions.get("sector_clearing_index", {}) or {}
        aggregate_ratio = float(market_conditions.get("aggregate_clearing_ratio", 1.0) or 1.0)
        sector_key = str(getattr(self, "sector", "unknown"))
        sector_adjustments = market_conditions.get("sector_demand_adjustments", {}) or {}
        sector_adjust = sector_adjustments.get(sector_key)
        if isinstance(sector_adjust, dict):
            base_multiple *= float(sector_adjust.get("return", 1.0))
            risk_signal = float(np.clip(risk_signal * float(sector_adjust.get("failure", 1.0)), 0.05, 0.95))
        sector_ratio = float(
            clearing_index.get(sector_key, aggregate_ratio) if clearing_index else aggregate_ratio
        )
        if not np.isfinite(sector_ratio):
            sector_ratio = 1.0
        oversupply = max(0.0, sector_ratio - 1.0)
        undersupply = max(0.0, 1.0 - sector_ratio)
        oversupply_penalty = getattr(self.config, "RETURN_OVERSUPPLY_PENALTY", 0.5)
        undersupply_bonus = getattr(self.config, "RETURN_UNDERSUPPLY_BONUS", 0.3)
        clearing_adjustment = 1.0 - oversupply_penalty * oversupply + undersupply_bonus * undersupply
        # Wider clearing band to allow stronger under/oversupply feedback
        clearing_adjustment = float(np.clip(clearing_adjustment, 0.1, 3.5))

        base_mean = base_multiple * lifecycle_factor * resilience * clearing_adjustment

        unc_state = market_conditions.get("uncertainty_state", {}) or {}
        agentic_state = {}
        if isinstance(unc_state, dict):
            agentic_state = unc_state.get("agentic_novelty", unc_state)
        novelty_signal = float(agentic_state.get("novelty_potential", agentic_state.get("level", 0.5) if isinstance(agentic_state, dict) else 0.5))
        stored_scarcity = float(getattr(self, "component_scarcity", 0.5) or 0.5)
        scarcity_signal = float(agentic_state.get("component_scarcity", stored_scarcity))
        scarcity_signal = float(np.clip(0.5 * scarcity_signal + 0.5 * stored_scarcity, 0.0, 1.0))
        reuse_pressure = float(agentic_state.get("reuse_pressure", 0.0) or 0.0)

        # Allow scarcity to lift the ceiling more
        scarcity_ceiling = 4.5 + 1.4 * scarcity_signal
        base_mean = float(np.clip(base_mean, 0.2, min(15.0, scarcity_ceiling)))

        combo_hhi = float(market_conditions.get("combo_hhi", 0.0) or 0.0)
        reuse_penalty = float(getattr(self, "crowding_penalty", 0.0) or 0.0)
        crowding_penalty = (0.25 * combo_hhi + 0.2 * reuse_penalty + 0.1 * reuse_pressure) * max(
            0.0, 1.0 - scarcity_signal
        )
        scarcity_bonus = 0.15 * scarcity_signal
        novelty_relief = max(0.0, novelty_signal - 0.5) * 0.3
        structural_adjustment = np.clip(
            1.0 - crowding_penalty + scarcity_bonus + novelty_relief,
            0.5,
            1.45,
        )
        base_mean *= structural_adjustment

        crowding_metrics = market_conditions.get("crowding_metrics", {}) or {}
        crowding_index = float(crowding_metrics.get("crowding_index", 0.25) or 0.25)
        crowd_threshold = getattr(self.config, "RETURN_DEMAND_CROWDING_THRESHOLD", 0.35)
        if crowding_index > crowd_threshold:
            crowd_penalty_extra = np.clip(1.0 - 0.25 * (crowding_index - crowd_threshold), 0.5, 1.0)
            base_mean *= crowd_penalty_extra

        tier_shares = market_conditions.get("tier_invest_share", {}) or {}
        tier_share = float(tier_shares.get(investor_tier, 0.0)) if investor_tier else 0.0
        if tier_share > 0.45:
            crowd_penalty = 1.0 - 0.6 * (tier_share - 0.45)
            base_mean *= float(np.clip(crowd_penalty, 0.35, 1.0))

        # =====================================================================
        # POWER LAW RETURN DISTRIBUTION
        # =====================================================================
        # Venture returns empirically follow a power law (Pareto) distribution:
        # - Most investments return near the minimum (many failures/zombies)
        # - A small number of "home runs" drive portfolio returns
        # - Matches VC empirics: ~65% return <1×, ~25% return 1-3×, ~10% are outliers
        #
        # We use a Pareto distribution with:
        # - x_m (minimum): scales with opportunity quality (base_mean)
        # - α (shape): from config.POWER_LAW_SHAPE_A (typically 2.0-2.5)
        #
        # Lower α = heavier tails = more extreme outliers
        # =====================================================================

        volatility = float(market_conditions.get("volatility", 0.0) or 0.0)
        regime = market_conditions.get("regime", "normal")

        # Power law shape parameter - lower = heavier tails
        alpha = float(getattr(self.config, "POWER_LAW_SHAPE_A", 2.5))

        # Adjust alpha based on regime (more extreme outcomes in volatile regimes)
        regime_alpha_adjust = {
            "crisis": -0.4,    # Heavier tails in crisis
            "recession": -0.2,
            "normal": 0.0,
            "growth": -0.1,    # Slightly heavier in growth (more unicorns)
            "boom": -0.3       # Heavier in boom (bubbles create outliers)
        }
        alpha += regime_alpha_adjust.get(regime, 0.0)
        alpha = float(np.clip(alpha, 1.5, 4.0))

        # Adjust alpha based on volatility (more volatile = heavier tails)
        alpha -= volatility * 0.3
        alpha = float(np.clip(alpha, 1.5, 4.0))

        # The minimum return (x_m) scales with opportunity quality
        # Increased floor (0.5 multiplier) to ensure median ~1× for sustainable survival
        x_min = float(np.clip(base_mean * 0.5, 0.3, 3.0))

        # Sample from Pareto distribution: X = x_m / U^(1/α)
        u = np.random.uniform(0, 1)
        u = float(np.clip(u, 1e-6, 1.0 - 1e-6))
        pareto_draw = x_min / (u ** (1.0 / alpha))

        # Apply quality scaling - better opportunities shift the distribution up
        quality_scale = base_mean / max(x_min, 0.1)
        scaled_return = pareto_draw * float(np.clip(quality_scale * 0.6, 0.6, 2.5))

        # Downside risk adjustment (oversupply, risk signal)
        downside_weight = float(getattr(self.config, "DOWNSIDE_OVERSUPPLY_WEIGHT", 0.65))
        downside = float(np.clip(0.2 + risk_signal * 0.4 + downside_weight * oversupply, 0.0, 2.0))

        # Beta shock for additional variance
        shock = np.random.beta(1.5, 2.5)
        scaled_return *= float(np.clip(1.0 - downside * shock * 0.5, 0.3, 1.2))

        # Final bounds - allow higher upper bound for power law outliers (unicorns)
        scarcity_headroom = 10.0 + 5.0 * max(0.0, scarcity_signal + novelty_signal - 0.8)
        upper_bound = float(np.clip(scarcity_headroom, 5.0, 50.0))  # Allow up to 50× for unicorns
        lower_bound = max(0.0, float(getattr(self.config, "RETURN_LOWER_BOUND", 0.0)))  # Floor at 0 for successful investments

        return float(np.clip(scaled_return, lower_bound, upper_bound))

    def update_lifecycle(self, adoption_rate: float) -> None:
        if self.lifecycle_stage == "emerging" and adoption_rate > 0.2:
            self.lifecycle_stage = "growing"
        elif self.lifecycle_stage == "growing" and adoption_rate > 0.6:
            self.lifecycle_stage = "mature"
        elif self.lifecycle_stage == "mature" and adoption_rate > 0.8:
            self.lifecycle_stage = "declining"


@dataclass(slots=True)
class Innovation:
    """Innovation artefact produced by the innovation engine."""

    id: str
    type: str
    knowledge_components: List[str]
    novelty: float
    quality: float
    round_created: int
    creator_id: int
    success: Optional[bool] = None
    market_impact: Optional[float] = None
    ai_assisted: bool = False
    ai_domains_used: List[str] = field(default_factory=list)
    sector: Optional[str] = None
    combination_signature: Optional[str] = None
    cash_multiple: Optional[float] = None
    scarcity: Optional[float] = None
    is_new_combination: bool = False
    ai_level_used: str = "none"

    def calculate_potential(self, market_conditions: Dict[str, Any]) -> float:
        """Estimate market potential based on innovation attributes."""
        base_potential = self.quality * (0.5 + 0.5 * self.novelty)
        type_modifiers = {
            "incremental": 0.7,
            "architectural": 0.9,
            "radical": 1.2,
            "disruptive": 1.5,
        }
        type_factor = type_modifiers.get(self.type, 1.0)
        regime = market_conditions.get("regime")
        if regime == "growth":
            market_factor = 1.2
        elif regime == "crisis":
            market_factor = 1.5 if self.type == "disruptive" else 0.7
        else:
            market_factor = 1.0
        ai_factor = 1.1 if self.ai_assisted else 1.0
        return base_potential * type_factor * market_factor * ai_factor


@dataclass(slots=True)
class Knowledge:
    """Represents a piece of knowledge or capability."""

    id: str
    domain: str
    level: float
    discovered_round: int
    discovered_by: Optional[int] = None
    parent_knowledge: List[str] = field(default_factory=list)

    def compatibility_with(self, other: "Knowledge") -> float:
        if self.domain == other.domain:
            return 0.8 + 0.2 * (1 - abs(self.level - other.level))
        domain_adjacency = {
            "technology": ["process", "market"],
            "market": ["technology", "business_model"],
            "process": ["technology", "business_model"],
            "business_model": ["market", "process"],
        }
        if other.domain in domain_adjacency.get(self.domain, []):
            return 0.4 + 0.2 * (1 - abs(self.level - other.level))
        return 0.1 + 0.1 * (1 - abs(self.level - other.level))


@dataclass(slots=True)
class AIAnalysis:
    """Enhanced AI analysis with potential errors and biases."""

    estimated_return: float
    estimated_uncertainty: float
    confidence: float
    insights: Tuple[str, ...]
    hidden_factors: Dict[str, float]
    actual_accuracy: float
    contains_hallucination: bool
    bias_applied: float
    domain: str
    false_insights: Tuple[str, ...]
    overconfidence_factor: float


@dataclass(slots=True)
class AILearningProfile:
    """Agent's learned understanding of AI capabilities."""

    domain_trust: Dict[str, float] = field(
        default_factory=lambda: {
            "market_analysis": 0.5,
            "technical_assessment": 0.5,
            "uncertainty_evaluation": 0.5,
            "innovation_potential": 0.5,
        }
    )
    accuracy_estimates: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "market_analysis": [],
            "technical_assessment": [],
            "uncertainty_evaluation": [],
            "innovation_potential": [],
        }
    )
    hallucination_experiences: Dict[str, int] = field(
        default_factory=lambda: {
            "market_analysis": 0,
            "technical_assessment": 0,
            "uncertainty_evaluation": 0,
            "innovation_potential": 0,
        }
    )
    usage_count: Dict[str, int] = field(
        default_factory=lambda: {
            "market_analysis": 0,
            "technical_assessment": 0,
            "uncertainty_evaluation": 0,
            "innovation_potential": 0,
        }
    )

    def update_trust(self, domain: str, was_accurate: bool, magnitude: float = 0.1) -> None:
        if was_accurate:
            self.domain_trust[domain] = min(1.0, self.domain_trust[domain] + magnitude)
        else:
            self.domain_trust[domain] = max(
                0.0, self.domain_trust[domain] - magnitude * 1.5
            )

    def should_use_ai_for_domain(
        self, domain: str, ai_cost: float, capital_buffer: float, cost_type: str = "per_use"
    ) -> bool:
        trust = self.domain_trust[domain]
        usage = self.usage_count[domain]
        exploration_bonus = 0.3 if usage < 5 else 0.0

        if cost_type == "subscription":
            cost_factor = 1.0
        else:
            available_buffer = max(capital_buffer - ai_cost, 1.0)
            cost_ratio = ai_cost / available_buffer
            cost_factor = max(0.0, 1 - cost_ratio)

        threshold = 0.4 - exploration_bonus
        decision_score = (trust + exploration_bonus) * cost_factor
        return decision_score > threshold

    def get_adjusted_confidence(self, ai_confidence: float, domain: str) -> float:
        trust = self.domain_trust[domain]
        hallucination_rate = self.hallucination_experiences[domain] / max(
            1, self.usage_count[domain]
        )
        adjustment = trust * (1 - hallucination_rate)
        return ai_confidence * adjustment
