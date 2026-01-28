"""
Market environment for Glimpse ABM.

This module implements the market dynamics that create the context for
entrepreneurial action under Knightian uncertainty. The market environment
generates the practical indeterminism and competitive recursion dimensions
of uncertainty through dynamic supply-demand clearing, regime transitions,
and crowding effects.

Theoretical Foundation
----------------------
The market model operationalizes concepts from:

    Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).
    Are the futures computable? Knightian uncertainty & artificial intelligence.
    Academy of Management Review, 50(2), 415-440.

Key mechanisms implemented:

1. **Regime Dynamics**: Markov-switching macroeconomic regimes (crisis,
   recession, normal, growth, boom) create path-dependent market conditions.
   This operationalizes practical indeterminism at the system level—even
   with perfect information about the current state, future transitions
   remain probabilistic.

2. **Supply-Demand Clearing**: Capital flows and opportunity capacity
   create endogenous price adjustments. Crowded sectors face return
   compression; underexplored sectors offer premiums. This models the
   market-clearing dynamics that affect realized returns.

3. **Sector Heterogeneity**: Four sectors (tech, retail, service,
   manufacturing) with distinct risk-return profiles enable investigation
   of how AI affects sector allocation decisions.

4. **Crowding and Herding**: AI-tier-specific capital concentration is
   tracked to model competitive recursion—when many AI-augmented agents
   converge on similar opportunities, returns deteriorate through
   crowding penalties.

5. **Opportunity Lifecycle**: Opportunities evolve through stages
   (emerging → growing → mature → declining), affecting both return
   potential and uncertainty characteristics.

The market environment interacts with the KnightianUncertaintyEnvironment
to provide the market_conditions context for uncertainty perception and
the conditions for realized return calculation.

References
----------
Knight, F. H. (1921). Risk, uncertainty, and profit. Houghton Mifflin.

Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).
    Are the futures computable? Knightian uncertainty & artificial intelligence.
    Academy of Management Review, 50(2), 415-440.
"""

from __future__ import annotations

import collections
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import EmergentConfig
from .knowledge import KnowledgeBase
from .models import Innovation, Opportunity
from .utils import normalize_ai_label, safe_mean


class MarketEnvironment:
    """
    Dynamic market with emergent properties and realistic return dynamics.

    This class manages the market context in which entrepreneurial agents
    operate, including opportunity generation, macroeconomic regime
    transitions, supply-demand clearing, and crowding dynamics.

    The market is designed to create the conditions for practical
    indeterminism and competitive recursion—two of the four Knightian
    uncertainty dimensions from Townsend et al. (2025).

    Attributes
    ----------
    opportunities : List[Opportunity]
        Currently available investment opportunities.
    market_regime : str
        Current macroeconomic state: "crisis", "recession", "normal",
        "growth", or "boom".
    volatility : float
        Current market volatility level affecting return uncertainty.
    regime_return_multiplier : float
        Multiplier applied to base returns based on current regime.
    regime_failure_multiplier : float
        Multiplier applied to failure probability based on regime.

    Notes
    -----
    The market environment operates at the system level while individual
    agents perceive market conditions through the uncertainty environment.
    This separation enables investigation of how AI affects the gap
    between objective market conditions and subjective perceptions.
    """

    def __init__(
        self,
        config: EmergentConfig,
        uncertainty_env: "KnightianUncertaintyEnvironment",
        innovation_engine: "InnovationEngine",
        knowledge_base: Optional[KnowledgeBase] = None,
    ):
        self.config = config
        self.uncertainty_env = uncertainty_env
        self.innovation_engine = innovation_engine
        self.knowledge_base = knowledge_base or KnowledgeBase(config=self.config)
        self.n_agents = self.config.N_AGENTS
        self.opportunities: List[Opportunity] = []
        self.opportunities_by_sector = collections.defaultdict(list)
        self.opportunity_map: Dict[str, Opportunity] = {}
        self._current_round = 0
        self.market_regime = "normal"
        self.volatility = self.config.MARKET_VOLATILITY
        self.trend = 0.0
        self.market_momentum = 0.0
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.competition_levels: Dict[str, float] = {}
        self.total_investment_by_round: List[float] = []
        self.failure_rate_by_round: List[float] = []
        self.new_ventures_by_round: List[int] = []
        self.innovations: List = []
        self.exploration_activity = 0
        self.sectors = self.config.SECTORS
        self.sector_profiles = self.config.SECTOR_PROFILES
        self._conditions_cache: Optional[Dict] = None
        self._cache_round = -1
        self._regime_change_count = 0
        self._last_regime = "normal"
        self._boom_streak = 0
        self._last_shift_event: Optional[Dict[str, Any]] = None
        self._last_crowding_metrics: Dict[str, Any] = {}
        self.sector_capital_history = collections.defaultdict(list)
        self.sector_performance_history = collections.defaultdict(list)
        min_ops = int(math.ceil(float(getattr(self.config, "MIN_OPPORTUNITIES", 0) or 0)))
        base_ops = int(math.ceil(float(getattr(self.config, "BASE_OPPORTUNITIES", min_ops) or min_ops)))
        self.target_opportunity_capacity = max(min_ops, base_ops)
        self._tier_invest_share: Dict[str, float] = {"none": 0.0, "basic": 0.0, "advanced": 0.0, "premium": 0.0}
        self._sector_clearing_index: Dict[str, float] = {}
        self._sector_flow_snapshot: Dict[str, float] = {}
        self._sector_capacity_snapshot: Dict[str, float] = {}
        self._aggregate_clearing_ratio: float = 1.0
        self._sector_flow_share: Dict[str, float] = {}
        self._total_sector_flow: float = 0.0
        self._tier_capital_flow: Dict[str, float] = {"none": 0.0, "basic": 0.0, "advanced": 0.0, "premium": 0.0}
        self._sector_demand_adjustments: Dict[str, Dict[str, float]] = {}
        self.regime_return_multiplier: float = 1.0
        self.regime_failure_multiplier: float = 1.0
        self._regime_states: List[str] = list(getattr(self.config, "MACRO_REGIME_STATES", ("normal",)))
        self._modifier_effects: Dict[str, Dict[str, float]] = {
            "premium": {"log_mu": 0.12, "log_sigma": -0.05, "failure_mean": -0.05},
            "budget": {"log_mu": -0.15, "log_sigma": 0.02, "failure_mean": 0.04},
            "digital": {"log_mu": 0.05, "log_sigma": -0.02, "failure_mean": -0.02},
            "local": {"log_mu": -0.05, "log_sigma": 0.03, "failure_mean": 0.01},
            "specialized": {"log_mu": 0.08, "log_sigma": 0.04, "failure_mean": -0.01},
            "sustainable": {"log_mu": 0.04, "log_sigma": 0.0, "failure_mean": -0.02},
        }
        self.branch_params: Dict[str, Dict[str, Any]] = {}
        self._initialize_branch_hierarchy()
        self._initialize_opportunities()
        self._last_uncertainty_state = None
        import threading

        self._lock = threading.Lock()
        self._opportunity_snapshot_round: Optional[int] = None
        self._opportunity_snapshot: Optional[
            Tuple[Tuple[Opportunity, ...], frozenset[str]]
        ] = None

    def _resolve_base_sector(self, sector: str) -> str:
        if sector in self.sector_profiles:
            return sector
        if "_" in sector:
            candidate = sector.split("_", 1)[0]
            if candidate in self.sector_profiles:
                return candidate
        return sector

    def _initialize_branch_hierarchy(self) -> None:
        for sector, profile in self.sector_profiles.items():
            self._initialize_branch(sector, profile)

    def _initialize_branch(self, name: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        if name in self.branch_params:
            return self.branch_params[name]
        return_range = profile.get("return_range", (1.0, 2.5))
        log_mu = float(
            profile.get("return_log_mu", math.log(max(return_range[0], 0.6)))
        )
        log_sigma = float(np.clip(profile.get("return_log_sigma", 0.35), 0.05, 1.0))
        failure_low, failure_high = profile.get("failure_range", (0.1, 0.7))
        failure_mean = float((failure_low + failure_high) / 2)
        failure_sigma = float(
            np.clip(np.mean(profile.get("failure_volatility_range", (0.04, 0.12))), 0.01, 0.5)
        )
        params = {
            "name": name,
            "root": name.split("_")[0],
            "profile": profile,
            "log_mu": log_mu,
            "log_sigma": log_sigma,
            "failure_mean": failure_mean,
            "failure_sigma": failure_sigma,
            "depth": 0,
        }
        self.branch_params[name] = params
        return params

    def _create_child_params(
        self,
        name: str,
        parent_params: Dict[str, Any],
        modifier: Optional[str] = None,
    ) -> Dict[str, Any]:
        root_sector = parent_params["root"]
        profile = self.sector_profiles[root_sector]
        mu_drift = np.random.normal(0.0, self.config.BRANCH_LOG_MEAN_DRIFT)
        sigma_drift = np.random.normal(0.0, self.config.BRANCH_LOG_SIGMA_DRIFT)
        failure_drift = np.random.normal(0.0, self.config.BRANCH_FAILURE_DRIFT)
        if modifier:
            effect = self._modifier_effects.get(modifier.lower(), {})
            mu_drift += effect.get("log_mu", 0.0)
            sigma_drift += effect.get("log_sigma", 0.0)
            failure_drift += effect.get("failure_mean", 0.0)
        log_mu = parent_params["log_mu"] + mu_drift
        log_sigma = max(0.05, parent_params["log_sigma"] + sigma_drift)
        return_range = profile.get("return_range", (1.0, 3.0))
        log_mu = float(np.clip(log_mu, math.log(return_range[0]), math.log(return_range[1])))
        failure_range = profile.get("failure_range", (0.1, 0.9))
        failure_mean = float(
            np.clip(parent_params["failure_mean"] + failure_drift, failure_range[0], failure_range[1])
        )
        failure_sigma = float(
            np.clip(
                parent_params["failure_sigma"] * np.random.uniform(0.85, 1.15),
                0.01,
                0.5,
            )
        )
        params = {
            "name": name,
            "root": root_sector,
            "profile": profile,
            "log_mu": log_mu,
            "log_sigma": log_sigma,
            "failure_mean": failure_mean,
            "failure_sigma": failure_sigma,
            "depth": parent_params.get("depth", 0) + 1,
        }
        return params

    def _ensure_branch(self, branch_name: str) -> Dict[str, Any]:
        branch_name = str(branch_name)
        if branch_name in self.branch_params:
            return self.branch_params[branch_name]
        parts = branch_name.split("_")
        base = parts[0]
        if base not in self.branch_params:
            profile = self.sector_profiles.get(base)
            if profile is None:
                raise KeyError(f"Unknown sector '{base}'")
            self._initialize_branch(base, profile)
        parent = base
        for depth in range(1, len(parts)):
            child = "_".join(parts[: depth + 1])
            if child in self.branch_params:
                parent = child
                continue
            parent_params = self.branch_params[parent]
            new_params = self._create_child_params(child, parent_params, modifier=parts[depth])
            self.branch_params[child] = new_params
            parent = child
        return self.branch_params[branch_name]

    def get_sector_competition_intensity(self, sector: str) -> float:
        """
        Get sector-specific competition intensity based on Census HHI data.

        Competition intensity values calibrated from Census Bureau Economic Census:
        - Tech: 1.2 (HHI 1500-2500, moderate concentration)
        - Retail: 0.7 (HHI 500-1000, fragmented)
        - Service: 0.9 (HHI 800-1500, moderately fragmented)
        - Manufacturing: 1.4 (HHI 1800-3000, concentrated)

        Returns base intensity scaled by global COMPETITION_SCALE_FACTOR for robustness testing.
        """
        profile = self.config.SECTOR_PROFILES.get(sector, {})
        base_intensity = float(profile.get('competition_intensity', 1.0))

        # Apply global competition scale factor for robustness testing
        return base_intensity * self.config.COMPETITION_SCALE_FACTOR

    def update_opportunity_competition(self, opp, delta: float) -> None:
        """Apply sector-specific competition intensity to opportunity competition updates."""
        intensity = self.get_sector_competition_intensity(opp.sector)
        opp.competition = float(np.clip(opp.competition + delta * intensity, 0.0, 1.0))

    def _sample_branch_characteristics(
        self, branch_name: str, quality_roll: Optional[float] = None
    ) -> Tuple[float, float, float, int]:
        params = self._ensure_branch(branch_name)
        profile = params["profile"]
        log_mu = params["log_mu"]
        log_sigma = params["log_sigma"]
        quality_roll = np.random.random() if quality_roll is None else quality_roll
        if quality_roll > 0.92:
            log_mu += 0.18
            log_sigma *= 0.9
        elif quality_roll < 0.28:
            log_mu -= 0.12
            log_sigma *= 1.08
        log_sigma = float(np.clip(log_sigma, 0.05, 1.0))
        latent_return = np.random.lognormal(mean=log_mu, sigma=log_sigma)
        latent_return = float(
            np.clip(latent_return, profile["return_range"][0], profile["return_range"][1])
        )
        failure_mean = params["failure_mean"]
        failure_sigma = params["failure_sigma"]
        if quality_roll > 0.85:
            failure_mean *= 0.92
        elif quality_roll < 0.3:
            failure_mean *= 1.08
        latent_failure = float(
            np.clip(
                np.random.normal(loc=failure_mean, scale=failure_sigma),
                profile["failure_range"][0],
                profile["failure_range"][1],
            )
        )
        capital_req = float(np.random.uniform(*profile["capital_range"]))
        maturity = int(np.random.randint(*profile["maturity_range"]))
        return latent_return, latent_failure, capital_req, maturity

    def _apply_branch_feedback(self, branch_name: str, mean_roi: float) -> None:
        params = self.branch_params.get(branch_name)
        if params is None:
            try:
                params = self._ensure_branch(branch_name)
            except KeyError:
                return
        profile = params["profile"]
        rate = float(getattr(self.config, "BRANCH_FEEDBACK_RATE", 0.02))
        feedback = float(np.clip(mean_roi, -1.0, 1.0)) * rate
        log_range = (math.log(profile["return_range"][0]), math.log(profile["return_range"][1]))
        params["log_mu"] = float(np.clip(params["log_mu"] + feedback, log_range[0], log_range[1]))
        failure_range = profile.get("failure_range", (0.1, 0.9))
        params["failure_mean"] = float(
            np.clip(params["failure_mean"] - feedback * 0.5, failure_range[0], failure_range[1])
        )

    def _get_demand_adjustments(self, sector: str) -> Dict[str, float]:
        if sector in self._sector_demand_adjustments:
            return self._sector_demand_adjustments[sector]
        base_sector = self._resolve_base_sector(sector)
        clearing_ratio = self._sector_clearing_index.get(sector)
        if clearing_ratio is None and base_sector != sector:
            clearing_ratio = self._sector_clearing_index.get(base_sector)
        if clearing_ratio is None or not np.isfinite(clearing_ratio):
            clearing_ratio = float(getattr(self, "_aggregate_clearing_ratio", 1.0) or 1.0)

        total_flow = getattr(self, "_total_sector_flow", 0.0)
        share_map = getattr(self, "_sector_flow_share", {}) or {}
        flow_share = 0.0
        if total_flow > 0.0:
            flow_share = float(share_map.get(sector, share_map.get(base_sector, 0.0)) or 0.0)
        else:
            crowding_metrics = getattr(self, "_last_crowding_metrics", {}) or {}
            flow_share = float(crowding_metrics.get("crowding_index", 0.25) or 0.25)
        crowd_threshold = getattr(self.config, "RETURN_DEMAND_CROWDING_THRESHOLD", 0.35)
        crowd_excess = max(0.0, flow_share - crowd_threshold)
        crowd_relief = max(0.0, crowd_threshold - flow_share)
        penalty_strength = getattr(self.config, "RETURN_DEMAND_CROWDING_PENALTY", 0.4)
        # Convex crowding penalty and relief
        return_penalty = 1.0 - penalty_strength * (crowd_excess ** 2 if crowd_excess > 0 else 0.0)
        return_penalty *= 1.0 + 0.35 * (crowd_relief ** 2 if crowd_relief > 0 else 0.0)
        failure_pressure = 1.0 + getattr(self.config, "FAILURE_DEMAND_PRESSURE", 0.25) * (crowd_excess ** 2 if crowd_excess > 0 else 0.0)
        failure_pressure *= 1.0 - 0.2 * (crowd_relief ** 2 if crowd_relief > 0 else 0.0)

        if not hasattr(self, "_recent_clearing_history"):
            self._recent_clearing_history = []
        self._recent_clearing_history.append(clearing_ratio)
        max_hist = 3
        if len(self._recent_clearing_history) > max_hist:
            self._recent_clearing_history = self._recent_clearing_history[-max_hist:]
        dispersion = float(np.std(self._recent_clearing_history)) if self._recent_clearing_history else 0.0
        scale = 1.0
        if dispersion > 0 and np.isfinite(dispersion):
            scale = float(np.clip(1.0 + 0.6 * abs(clearing_ratio - 1.0) / dispersion, 0.5, 3.0))

        if clearing_ratio > 1.0:
            undersupply = max(0.0, clearing_ratio - 1.0)
            return_penalty *= 1.0 + 0.45 * undersupply * scale
            failure_pressure *= 1.0 - 0.35 * undersupply * scale
        else:
            oversupply_gap = max(0.0, 1.0 - clearing_ratio)
            return_penalty *= 1.0 - 0.7 * oversupply_gap * scale
            failure_pressure *= 1.0 + 0.6 * oversupply_gap * scale

        pressure_map = getattr(self, "_sector_pressure", {}) or {}
        pressure_val = float(pressure_map.get(sector, pressure_map.get(base_sector, 0.0)) or 0.0)
        return_penalty *= float(np.clip(1.0 - 0.5 * pressure_val, 0.5, 1.3))
        failure_pressure *= float(np.clip(1.0 + 0.4 * pressure_val, 0.6, 1.6))

        adjustments = {
            "return": return_penalty,
            "failure": failure_pressure,
        }
        self._sector_demand_adjustments[sector] = adjustments
        return adjustments

    def _initialize_opportunities(self) -> None:
        n_opportunities = self.config.get_scaled_opportunities(self.n_agents)
        for i in range(n_opportunities):
            sector = str(np.random.choice(self.sectors))
            new_opp = self._create_realistic_opportunity(f"initial_{sector}_{i}", sector)
            self.opportunities.append(new_opp)
            self.opportunity_map[new_opp.id] = new_opp
            self._index_opportunity(new_opp)

    def _create_realistic_opportunity(
        self, opp_id: str, sector: str, innovator_capability: float = 0.5
    ) -> Opportunity:
        branch_name = str(sector)
        quality_roll = np.random.random()
        latent_return, latent_failure, capital_req, maturity = self._sample_branch_characteristics(
            branch_name, quality_roll=quality_roll
        )
        demand_adjustments = self._get_demand_adjustments(branch_name)
        latent_return *= demand_adjustments["return"] * self.regime_return_multiplier
        latent_failure *= demand_adjustments["failure"] * self.regime_failure_multiplier
        profile = self.sector_profiles[self._resolve_base_sector(branch_name)]
        fail_low, fail_high = profile["failure_range"]
        norm_uncertainty = (
            np.clip((latent_failure - fail_low) / (fail_high - fail_low), 0.0, 1.0)
            if fail_high > fail_low
            else 0.5
        )
        risk_return_shift = np.clip(0.7 + 0.8 * (norm_uncertainty - 0.5), 0.5, 1.5)
        latent_return *= risk_return_shift

        opp = Opportunity(
            id=opp_id,
            latent_return_potential=float(np.clip(latent_return, 0.5, 25.0)),
            latent_failure_potential=float(np.clip(latent_failure, 0.1, 0.95)),
            complexity=float(np.random.uniform(0.3, 1.2)),
            discovered=True,
            discovery_round=0,
            config=self.config,
            sector=branch_name,
            capital_requirements=float(capital_req),
            time_to_maturity=int(maturity),
        )
        return opp

    def step(
        self,
        round: int,
        agent_actions: List[Dict],
        innovations: List[Dict],
        matured_outcomes: Optional[List[Dict]] = None,
    ) -> Dict:
        self._current_round = round
        self._sector_demand_adjustments.clear()
        self.innovations.extend(innovations)
        matured_outcomes = matured_outcomes or []
        total_investment = sum(
            (a.get("amount") or a.get("capital_deployed") or 0.0)
            for a in agent_actions
            if a.get("action") == "invest"
        )
        self.total_investment_by_round.append(total_investment)

        if matured_outcomes:
            successes = sum(1 for o in matured_outcomes if o.get("success"))
            failures = sum(1 for o in matured_outcomes if not o.get("success"))
        else:
            successes = failures = 0

        total_matured = successes + failures

        opportunity_demand = collections.defaultdict(int)
        for action in agent_actions:
            if action.get("action") == "invest" and action.get("chosen_opportunity_obj"):
                opportunity_demand[action["chosen_opportunity_obj"].id] += 1

        for action in agent_actions:
            if action.get("action") == "invest" and action.get("chosen_opportunity_obj"):
                opp = action["chosen_opportunity_obj"]
                opp.competition = min(1.0, opp.competition + 0.1)
            elif action.get("action") == "explore":
                self.exploration_activity += 1

        if innovations:
            for innovation in innovations:
                if not innovation:
                    continue
                signature = getattr(innovation, "combination_signature", None)
                if signature:
                    self.competition_levels[signature] = self.competition_levels.get(signature, 0) * 0.95 + 0.05
        invest_actions = [a for a in agent_actions if a.get("action") == "invest"]
        ai_invests = sum(1 for action in invest_actions if str(action.get("ai_level_used", "none")).lower() != "none")
        ai_invest_share = ai_invests / max(1, len(invest_actions)) if invest_actions else 0.0

        self._ai_invest_share = ai_invest_share
        self._update_market_dynamics(agent_actions, total_investment, ai_invest_share)
        self._manage_opportunities(round, opportunity_demand, total_investment)
        self._update_clearing_metrics(agent_actions)

        sector_returns: Dict[str, List[float]] = collections.defaultdict(list)
        if matured_outcomes:
            for outcome in matured_outcomes:
                investment = outcome.get("investment") or {}
                opp = investment.get("opportunity")
                sector = getattr(opp, "sector", None) if opp is not None else investment.get("sector")
                invested = float(outcome.get("investment_amount", 0.0) or 0.0)
                returned = float(outcome.get("capital_returned", 0.0) or 0.0)
                if sector and invested > 0:
                    sector_returns[str(sector)].append((returned - invested) / invested)
        global_avg = None
        if sector_returns:
            all_vals = [val for vals in sector_returns.values() for val in vals]
            global_avg = float(np.mean(all_vals)) if all_vals else 0.0
        self._sector_pressure = {}
        if sector_returns:
            for sector, vals in sector_returns.items():
                if vals:
                    sector_avg = float(np.mean(vals))
                    pressure = max(0.0, (global_avg if global_avg is not None else 0.0) - sector_avg)
                    self._sector_pressure[sector] = pressure
                    self._apply_branch_feedback(sector, sector_avg)

        if total_matured > 0:
            self.failure_rate_by_round.append(failures / max(1, total_matured))
        else:
            total_investments = len([a for a in agent_actions if a.get("action") == "invest"])
            self.failure_rate_by_round.append(failures / max(1, total_investments) if total_investments > 0 else 0)

        total_actions = len(agent_actions)
        if total_actions > 0:
            action_counts = collections.Counter(action.get("action", "maintain") for action in agent_actions)
            share_invest = action_counts.get("invest", 0) / total_actions
            share_innovate = action_counts.get("innovate", 0) / total_actions
            share_explore = action_counts.get("explore", 0) / total_actions
            share_maintain = action_counts.get("maintain", 0) / total_actions
        else:
            share_invest = share_innovate = share_explore = 0.0
            share_maintain = 1.0
        ai_usage_share = (
            sum(1 for action in agent_actions if str(action.get("ai_level_used", "none")).lower() != "none")
            / max(1, total_actions)
        )
        crowding_index = share_invest**2 + share_innovate**2 + share_explore**2 + share_maintain**2
        crowding_base = float(getattr(self.config, "MARKET_SHIFT_CROWDING_BASE", 0.3))
        innovation_base = float(getattr(self.config, "MARKET_SHIFT_INNOVATION_BASE", 0.45))
        ai_base = float(getattr(self.config, "MARKET_SHIFT_AI_BASE", 0.35))
        crowding_pressure = max(0.0, crowding_index - crowding_base)
        innovation_pressure = max(0.0, share_innovate - innovation_base)
        ai_pressure = max(0.0, ai_usage_share - ai_base)
        self._last_crowding_metrics = {
            "crowding_index": crowding_index,
            "share_innovate": share_innovate,
            "share_invest": share_invest,
            "share_explore": share_explore,
            "share_maintain": share_maintain,
            "ai_usage_share": ai_usage_share,
        }

        shift_event = None
        base_shift_prob = float(getattr(self.config, "MARKET_SHIFT_PROBABILITY", 0.0) or 0.0)
        regime_multipliers = getattr(self.config, "MARKET_SHIFT_REGIME_MULTIPLIER", {})
        shift_prob = base_shift_prob * float(regime_multipliers.get(self.market_regime, 1.0))
        crowding_weight = float(getattr(self.config, "MARKET_SHIFT_CROWDING_WEIGHT", 0.6))
        innovation_weight = float(getattr(self.config, "MARKET_SHIFT_INNOVATION_WEIGHT", 0.4))
        ai_weight = float(getattr(self.config, "MARKET_SHIFT_AI_WEIGHT", 0.35))
        shift_prob *= 1.0 + crowding_weight * crowding_pressure + innovation_weight * innovation_pressure + ai_weight * ai_pressure
        shift_prob = float(np.clip(shift_prob, 0.0, 1.0))
        if shift_prob > 0 and self.sectors and np.random.random() < shift_prob:
            max_sectors = max(1, int(getattr(self.config, "MARKET_SHIFT_MAX_SECTORS", 2)))
            severity_range = getattr(self.config, "MARKET_SHIFT_SEVERITY_RANGE", (0.15, 0.45))
            if isinstance(severity_range, (int, float)):
                severity_low = severity_high = float(severity_range)
            elif isinstance(severity_range, (list, tuple)):
                if len(severity_range) >= 2:
                    severity_low, severity_high = float(severity_range[0]), float(severity_range[1])
                elif len(severity_range) == 1:
                    severity_low = severity_high = float(severity_range[0])
                else:
                    severity_low, severity_high = (0.15, 0.45)
            else:
                severity_low, severity_high = (0.15, 0.45)
            impacts: List[Dict[str, Any]] = []
            impact_count = int(np.random.randint(1, max_sectors + 1))
            chosen_sectors = np.random.choice(self.sectors, size=impact_count, replace=False if impact_count < len(self.sectors) else True)
            for shift_sector in chosen_sectors:
                severity = float(np.clip(np.random.uniform(severity_low, severity_high), 0.05, 0.95))
                impacts.append({"sector": str(shift_sector), "severity": severity})
            if impacts:
                shift_event = {"round": round, "impacts": impacts}
                self._last_shift_event = shift_event
        market_state = self.get_market_conditions()
        if shift_event:
            market_state["market_shift"] = shift_event
        else:
            market_state["market_shift"] = None
        market_state["sector_pressure"] = getattr(self, "_sector_pressure", {})
        market_state["crowding_metrics"] = getattr(self, "_last_crowding_metrics", {})
        market_state["shift_probability"] = shift_prob
        return market_state

    def _update_market_dynamics(self, agent_actions: List[Dict], total_investment: float, ai_invest_share: float) -> None:
        def _extract_expected_return(action: Dict[str, Any]) -> float:
            value = action.get("expected_return")
            if value is None:
                details = action.get("chosen_opportunity_details") or {}
                value = details.get("estimated_return")
            return float(value) if value is not None else float("nan")

        avg_investment_quality = safe_mean(
            [
                _extract_expected_return(action)
                for action in agent_actions
                if action.get("action") == "invest"
            ]
        )
        if np.isnan(avg_investment_quality):
            avg_investment_quality = 1.0
        quality_adjustment = (avg_investment_quality - 1.0) * 0.1
        ai_activity = np.clip(ai_invest_share - 0.5, -0.5, 0.5)
        quality_adjustment -= ai_activity * 0.05  # paradox: high AI share can degrade realized quality
        investment_activity = np.clip(total_investment / (self.n_agents * 250_000), 0, 2)
        self.market_momentum = self.market_momentum * 0.8 + 0.2 * (investment_activity + quality_adjustment + ai_activity * 0.1)
        if self.market_momentum > 1.5:
            self._boom_streak += 1
        else:
            self._boom_streak = max(0, self._boom_streak - 1)

        base_prob = self.config.BLACK_SWAN_PROBABILITY
        exponent = self.config.BOOM_TAIL_UNCERTAINTY_EXPONENT
        dynamic_black_swan_prob = base_prob * (exponent ** max(0, self._boom_streak - 1)) * (1 + ai_invest_share * 0.3)
        crowding_metrics = getattr(self, "_last_crowding_metrics", {}) or {}
        crowding_index = float(crowding_metrics.get("crowding_index", 0.25) or 0.25)
        signals = {
            "investment_activity": investment_activity,
            "ai_activity": ai_activity,
            "crowding_index": crowding_index,
            "quality_adjustment": quality_adjustment,
            "momentum": self.market_momentum,
        }
        self._update_macro_regime(signals, dynamic_black_swan_prob)

    def _update_macro_regime(self, signals: Dict[str, float], black_swan_prob: float) -> None:
        states = self._regime_states or ["normal"]
        transition_map = getattr(self.config, "MACRO_REGIME_TRANSITIONS", {})
        current_state = self.market_regime if self.market_regime in transition_map else states[0]
        base_probs = np.array([transition_map.get(current_state, {}).get(state, 0.0) for state in states], dtype=float)
        if base_probs.sum() <= 0:
            base_probs = np.ones(len(states), dtype=float)
        adjustments = np.zeros_like(base_probs)
        idx_map = {state: i for i, state in enumerate(states)}
        invest = float(signals.get("investment_activity", 1.0) or 1.0)
        momentum = float(signals.get("momentum", 0.0) or 0.0)
        crowding = float(signals.get("crowding_index", 0.25) or 0.25)
        ai_activity = float(signals.get("ai_activity", 0.0) or 0.0)
        quality = float(signals.get("quality_adjustment", 0.0) or 0.0)

        if invest > 1.1 or momentum > 0.8 or quality > 0:
            boost = 0.05 * max(invest - 1.1, 0.0) + 0.04 * max(momentum - 0.8, 0.0) + 0.03 * max(quality, 0.0)
            if "growth" in idx_map:
                adjustments[idx_map["growth"]] += boost
            if "boom" in idx_map:
                adjustments[idx_map["boom"]] += boost * 0.6
        if invest < 0.9 or momentum < -0.6:
            drag = 0.05 * max(0.9 - invest, 0.0) + 0.04 * max(-0.6 - momentum, 0.0)
            if "recession" in idx_map:
                adjustments[idx_map["recession"]] += drag
            if "crisis" in idx_map:
                adjustments[idx_map["crisis"]] += drag * 0.5
        crowd_threshold = getattr(self.config, "RETURN_DEMAND_CROWDING_THRESHOLD", 0.35)
        if crowding > crowd_threshold:
            penalty = 0.05 * (crowding - crowd_threshold)
            if "recession" in idx_map:
                adjustments[idx_map["recession"]] += penalty
            if "crisis" in idx_map:
                adjustments[idx_map["crisis"]] += penalty * 0.6
        else:
            relief = 0.03 * (crowd_threshold - crowding)
            if "normal" in idx_map:
                adjustments[idx_map["normal"]] += relief
            if "growth" in idx_map:
                adjustments[idx_map["growth"]] += relief * 0.5
        if ai_activity > 0.3 and "crisis" in idx_map:
            adjustments[idx_map["crisis"]] += 0.02 * ai_activity
        if "crisis" in idx_map:
            adjustments[idx_map["crisis"]] += black_swan_prob

        probs = base_probs + adjustments
        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            probs = np.ones(len(states), dtype=float) / float(len(states))
        else:
            probs = probs / total
        new_state = str(np.random.choice(states, p=probs))
        self.market_regime = new_state
        return_mod = getattr(self.config, "MACRO_REGIME_RETURN_MODIFIERS", {})
        failure_mod = getattr(self.config, "MACRO_REGIME_FAILURE_MODIFIERS", {})
        trend_map = getattr(self.config, "MACRO_REGIME_TREND", {})
        volatility_map = getattr(self.config, "MACRO_REGIME_VOLATILITY", {})
        self.regime_return_multiplier = float(return_mod.get(new_state, 1.0))
        self.regime_failure_multiplier = float(failure_mod.get(new_state, 1.0))
        target_trend = float(trend_map.get(new_state, 0.0))
        target_volatility = float(volatility_map.get(new_state, self.volatility))
        self.trend = float(np.clip(0.7 * getattr(self, "trend", 0.0) + 0.3 * target_trend, -1.0, 1.0))
        self.volatility = float(np.clip(0.6 * self.volatility + 0.4 * target_volatility, 0.05, 1.0))
        if new_state == "crisis":
            self._boom_streak = 0
            self.market_momentum = min(self.market_momentum, -1.5)

    def _manage_opportunities(
        self,
        round: int,
        opportunity_demand: Dict[str, int],
        total_investment: float,
    ) -> None:
        if total_investment > self.target_opportunity_capacity * 100000:
            self.target_opportunity_capacity = min(
                self.target_opportunity_capacity + 1, self.config.get_scaled_opportunities(self.n_agents)
            )
        elif total_investment < self.target_opportunity_capacity * 50000:
            self.target_opportunity_capacity = max(self.config.MIN_OPPORTUNITIES, self.target_opportunity_capacity - 1)

        desired_new_opps = self.target_opportunity_capacity - len(self.opportunities)
        desired_new_opps += int(
            np.random.poisson(max(0.0, len(self.opportunities) * 0.02)) - len(
                [o for o in self.opportunities if o.age < 5]
            )
        )
        desired_new_opps = max(0, desired_new_opps)
        self.new_ventures_by_round.append(desired_new_opps)

        for _ in range(desired_new_opps):
            probabilities = None
            opportunity_counts = collections.Counter(getattr(opp, "sector", "unknown") for opp in self.opportunities)
            if opportunity_counts:
                total = sum(opportunity_counts.values())
                if total > 0:
                    probabilities = np.array(
                        [
                            (total - opportunity_counts.get(sector, 0)) / total
                            for sector in self.sectors
                        ]
                    )
                    if probabilities.sum() > 0:
                        probabilities = probabilities / probabilities.sum()
                    else:
                        probabilities = None
            sector = np.random.choice(self.sectors, p=probabilities) if probabilities is not None else np.random.choice(self.sectors)
            new_opp = self._create_realistic_opportunity(
                f"market_{round}_{np.random.randint(1000)}", str(sector)
            )
            new_opp.discovery_round = round
            self.opportunities.append(new_opp)
            self.opportunity_map[new_opp.id] = new_opp
            self._index_opportunity(new_opp)

        for opp in self.opportunities:
            opp.age = getattr(opp, "age", 0) + 1
            opp.competition = max(0.0, min(1.0, opp.competition * 0.9))

        dead_opportunities = []
        for opp in self.opportunities:
            if opp.competition < 0.01 and opp.age > 20:
                dead_opportunities.append(opp)
            elif opp.lifecycle_stage == "declining" and opp.competition < 0.05:
                dead_opportunities.append(opp)
            elif opp.age > 10 and opportunity_demand.get(opp.id, 0) == 0:
                dead_opportunities.append(opp)

        for opp in dead_opportunities:
            if opp in self.opportunities:
                self.opportunities.remove(opp)
                self._remove_from_sector_index(opp)
            if opp.id in self.opportunity_map:
                del self.opportunity_map[opp.id]

        self.opportunities = [
            opp
            for opp in self.opportunities
            if not (opp.lifecycle_stage == "declining" and opp.age > 50 and opp.competition < 0.1)
        ]
        self._rebuild_sector_index()
        self.opportunity_map = {opp.id: opp for opp in self.opportunities}

    def _index_opportunity(self, opp: Opportunity) -> None:
        self.opportunities_by_sector[getattr(opp, "sector", "unknown")].append(opp)
        self._invalidate_opportunity_snapshot()

    def _remove_from_sector_index(self, opp: Opportunity) -> None:
        sector = getattr(opp, "sector", "unknown")
        if sector in self.opportunities_by_sector and opp in self.opportunities_by_sector[sector]:
            self.opportunities_by_sector[sector].remove(opp)
            self._invalidate_opportunity_snapshot()

    def clear_old_opportunities(self, round: int) -> None:
        self.opportunities = [
            opp
            for opp in self.opportunities
            if not (round - getattr(opp, "discovery_round", round) > 80 and opp.competition < 0.05)
        ]
        self._rebuild_sector_index()
        self._conditions_cache = None

    def get_market_conditions(self) -> Dict:
        """Return cached market snapshot for the current round."""
        current_round = getattr(self, "_current_round", -1)
        if self._cache_round == current_round and self._conditions_cache is not None:
            return self._conditions_cache

        total_opps = len(self.opportunities)
        if total_opps == 0:
            snapshot = {
                "regime": self.market_regime,
                "volatility": self.volatility,
                "trend": self.trend,
                "n_opportunities": 0,
                "avg_competition": 0.0,
                "market_saturation": 0.0,
                "lifecycle_distribution": {},
                "sector_distribution": {},
                "round": current_round,
            }
            self._conditions_cache = snapshot
            self._cache_round = current_round
            return snapshot

        lifecycle_counts = collections.Counter(
            getattr(opp, "lifecycle_stage", "emerging") for opp in self.opportunities
        )
        sector_counts = collections.Counter(
            getattr(opp, "sector", "unknown") for opp in self.opportunities
        )
        market_saturation = min(
            1.0,
            total_opps
            / max(1, self.n_agents * getattr(self.config, "OPPORTUNITIES_PER_CAPITA", 0.1)),
        )
        combo_hhi, sector_hhi = self.get_combination_diversity_metrics()
        demand_adjustments: Dict[str, Dict[str, float]] = {}
        demand_sectors = set(sector_counts.keys()) | set(getattr(self, "_sector_clearing_index", {}).keys())
        if not demand_sectors:
            demand_sectors = set(self.sectors)
        for sector_name in demand_sectors:
            demand_adjustments[sector_name] = dict(self._get_demand_adjustments(sector_name))
        snapshot = {
            "regime": self.market_regime,
            "volatility": self.volatility,
            "trend": self.trend,
            "n_opportunities": total_opps,
            "ai_invest_share": getattr(self, "_ai_invest_share", 0.0),
            "alive_ratio": float(getattr(self, "_current_alive_ratio", 1.0)),
            "avg_competition": safe_mean([opp.competition for opp in self.opportunities]),
            "market_saturation": market_saturation,
            "lifecycle_distribution": dict(lifecycle_counts),
            "sector_distribution": dict(sector_counts),
            "combo_hhi": combo_hhi,
            "sector_hhi": sector_hhi,
            "tier_invest_share": dict(getattr(self, "_tier_invest_share", {})),
            "sector_clearing_index": dict(getattr(self, "_sector_clearing_index", {})),
            "sector_capital_flow": dict(getattr(self, "_sector_flow_snapshot", {})),
            "sector_capacity": dict(getattr(self, "_sector_capacity_snapshot", {})),
            "aggregate_clearing_ratio": float(getattr(self, "_aggregate_clearing_ratio", 1.0)),
            "tier_capital_flow": dict(getattr(self, "_tier_capital_flow", {})),
            "sector_demand_adjustments": demand_adjustments,
            "regime_return_multiplier": float(self.regime_return_multiplier),
            "regime_failure_multiplier": float(self.regime_failure_multiplier),
            "crowding_metrics": dict(self._last_crowding_metrics or {}),
            "sector_pressure": dict(getattr(self, "_sector_pressure", {})),
            "round": current_round,
        }
        self._conditions_cache = snapshot
        self._cache_round = current_round
        return snapshot

    def _rebuild_sector_index(self) -> None:
        self.opportunities_by_sector = collections.defaultdict(list)
        for opp in self.opportunities:
            self.opportunities_by_sector[getattr(opp, "sector", "unknown")].append(opp)
        self._conditions_cache = None
        self._invalidate_opportunity_snapshot()

    def get_opportunities(self, agent: "EmergentAgent", info_level: float = 0.0) -> List[Opportunity]:
        """Return the opportunities an agent can currently access."""
        opportunity_snapshot, sectors_with_visible = self._get_opportunity_snapshot()

        visible_opps: List[Opportunity] = []
        undiscovered_by_sector: Dict[str, List[Opportunity]] = collections.defaultdict(list)
        for opp in opportunity_snapshot:
            if getattr(opp, "discovered", True):
                visible_opps.append(opp)
            else:
                sector_key = getattr(opp, "sector", "unknown")
                undiscovered_by_sector[sector_key].append(opp)

        ai_level = getattr(agent, "current_ai_level", "none")
        profile = self.config.AI_LEVELS.get(ai_level) or self.config.AI_LEVELS.get("none", {})
        info_quality = float(profile.get("info_quality", 0.0))
        info_breadth = float(profile.get("info_breadth", 0.0))
        ai_factor = 1.0 + info_quality * 2.0 + info_breadth * 1.5
        ai_factor = float(np.clip(ai_factor, 0.8, 4.5))

        base_exploration = agent.traits.get("exploration_tendency", 0.0)
        base_awareness = agent.traits.get("market_awareness", 0.0)
        knowledge_map = getattr(agent.resources, "knowledge", {})

        newly_discovered: List[Opportunity] = []
        base_prob = base_exploration * 0.2 + base_awareness * 0.3 + info_breadth * 0.25 + info_quality * 0.1
        for sector_key, opps in undiscovered_by_sector.items():
            sector_knowledge = knowledge_map.get(sector_key, 0.0)
            if sector_knowledge <= 0 and sector_key not in sectors_with_visible:
                continue

            for opp in opps:
                if hasattr(opp, "sector"):
                    sector_level = knowledge_map.get(getattr(opp, "sector", "unknown"), 0.0)
                    discovery_prob = base_prob * (1.0 + sector_level * 2.0)
                else:
                    discovery_prob = base_prob
                discovery_prob *= ai_factor
                if np.random.random() < discovery_prob:
                    newly_discovered.append(opp)
                    visible_opps.append(opp)

        if newly_discovered:
            with self._lock:
                for opp in newly_discovered:
                    if not getattr(opp, "discovered", False):
                        opp.discovered = True
                        opp.discovery_round = self._current_round
                        self._index_opportunity(opp)

        return self.get_perceived_opportunities(visible_opps, ai_level, agent)

    def get_perceived_opportunities(
        self,
        all_opportunities: List[Opportunity],
        ai_level: str,
        agent: "EmergentAgent",
    ) -> List[Opportunity]:
        """Filter and rank opportunities based on the agent's perspective."""
        if not all_opportunities:
            return []
        profile = self.config.AI_LEVELS.get(ai_level) or self.config.AI_LEVELS.get("none", {})
        info_quality = float(profile.get("info_quality", 0.0))
        info_breadth = float(profile.get("info_breadth", 0.0))
        base_visible = (3 + info_breadth * 60.0) * (1.0 + info_quality)
        max_visible = int(max(3, min(len(all_opportunities), round(base_visible))))

        knowledge_map = getattr(agent.resources, "knowledge", {})
        num_opps = len(all_opportunities)
        scores = np.empty(num_opps, dtype=float)
        for index, opp in enumerate(all_opportunities):
            relevance = 0.5
            if hasattr(opp, "sector"):
                relevance += knowledge_map.get(getattr(opp, "sector", "unknown"), 0.0) * 0.5
            scores[index] = relevance

        take = max_visible if num_opps > max_visible else num_opps
        if num_opps <= take:
            order = np.argsort(-scores, kind='mergesort')
            selected = order[:take]
        else:
            partition_idx = np.argpartition(-scores, take - 1)[:take]
            reordered = partition_idx[np.argsort(-scores[partition_idx], kind='mergesort')]
            selected = reordered
        return [all_opportunities[idx] for idx in selected]

    def _invalidate_opportunity_snapshot(self) -> None:
        self._opportunity_snapshot_round = None
        self._opportunity_snapshot = None

    def _get_opportunity_snapshot(self) -> Tuple[Tuple[Opportunity, ...], frozenset[str]]:
        current_round = self._current_round
        if (
            self._opportunity_snapshot_round == current_round
            and self._opportunity_snapshot is not None
        ):
            return self._opportunity_snapshot
        with self._lock:
            snapshot = tuple(self.opportunities)
            sector_keys = frozenset(
                sector for sector, items in self.opportunities_by_sector.items() if items
            )
        self._opportunity_snapshot = (snapshot, sector_keys)
        self._opportunity_snapshot_round = current_round
        return self._opportunity_snapshot

    def _calculate_market_gaps(self, agent_actions: List[Dict]) -> Dict[str, float]:
        """Estimate unmet demand by sector from recent activity."""
        sector_demand = collections.defaultdict(float)
        sector_supply = collections.defaultdict(float)

        for action in agent_actions:
            if action.get("action") == "invest" and not action.get("success", True):
                opp = action.get("chosen_opportunity_obj")
                if opp and hasattr(opp, "sector"):
                    sector_demand[getattr(opp, "sector", "unknown")] += 1

        for opp in self.opportunities:
            if hasattr(opp, "sector"):
                sector_supply[getattr(opp, "sector", "unknown")] += 1

        gaps: Dict[str, float] = {}
        for sector in set(sector_demand) | set(sector_supply):
            gaps[sector] = sector_demand.get(sector, 0.0) - sector_supply.get(sector, 0.0)
        return gaps

    def _estimate_sector_capacity(self) -> Dict[str, float]:
        """Estimate available capital absorption capacity by sector."""
        capacity = collections.defaultdict(float)
        for opp in self.opportunities:
            sector = getattr(opp, "sector", "unknown")
            requirement = float(getattr(opp, "capital_requirements", 0.0) or 0.0)
            maturity = max(1, int(getattr(opp, "time_to_maturity", 1)))
            capacity[sector] += requirement / maturity
        if not capacity:
            capacity["unknown"] = float(self.n_agents) * 50_000.0
        return capacity

    def _update_clearing_metrics(self, agent_actions: List[Dict]) -> None:
        """Track demand-vs-capacity ratios that drive market clearing."""
        sector_flows = collections.defaultdict(float)
        tier_flows = collections.defaultdict(float)
        for action in agent_actions:
            if action.get("action") != "invest":
                continue
            details = action.get("chosen_opportunity_details") or {}
            sector = details.get("sector")
            if not sector and action.get("chosen_opportunity_obj"):
                sector = getattr(action["chosen_opportunity_obj"], "sector", None)
            sector = str(sector or "unknown")
            capital = float(action.get("capital_deployed") or action.get("amount") or 0.0)
            if capital <= 0:
                continue
            sector_flows[sector] += capital
            tier = normalize_ai_label(action.get("ai_level_used", "none"))
            tier_flows[tier] = tier_flows.get(tier, 0.0) + capital

        capacity = self._estimate_sector_capacity()
        clearing_index: Dict[str, float] = {}
        for sector in set(capacity) | set(sector_flows):
            supply = max(capacity.get(sector, 0.0), 1.0)
            demand = sector_flows.get(sector, 0.0)
            clearing_index[sector] = float(demand / supply)

        total_capacity = max(float(sum(capacity.values())), 1.0)
        total_demand = float(sum(sector_flows.values()))
        self._sector_capacity_snapshot = dict(capacity)
        self._sector_flow_snapshot = dict(sector_flows)
        self._sector_clearing_index = dict(clearing_index)
        self._aggregate_clearing_ratio = float(total_demand / total_capacity)
        if total_demand > 0:
            self._sector_flow_share = {
                sector: float(demand / total_demand) for sector, demand in sector_flows.items()
            }
        else:
            self._sector_flow_share = {}
        self._total_sector_flow = total_demand
        self._tier_capital_flow = dict(tier_flows)
        self._conditions_cache = None

    def create_niche_opportunity(self, niche_id: str, discoverer_id: int, round_num: int) -> Opportunity:
        """Create a niche opportunity discovered via exploration."""
        if "_" in niche_id:
            base_sector, modifier = niche_id.rsplit("_", 1)
        else:
            base_sector, modifier = niche_id, "standard"

        modifier_effects = {
            "premium": {"return_mult": 1.15, "uncertainty_mult": 0.85, "capital_mult": 1.8},
            "budget": {"return_mult": 0.85, "uncertainty_mult": 1.15, "capital_mult": 0.6},
            "sustainable": {"return_mult": 1.05, "uncertainty_mult": 0.95, "capital_mult": 1.1},
            "digital": {"return_mult": 1.08, "uncertainty_mult": 1.0, "capital_mult": 0.8},
            "local": {"return_mult": 0.92, "uncertainty_mult": 0.85, "capital_mult": 0.7},
            "specialized": {"return_mult": 1.12, "uncertainty_mult": 1.05, "capital_mult": 1.3},
        }
        mods = modifier_effects.get(modifier, {"return_mult": 1.0, "uncertainty_mult": 1.0, "capital_mult": 1.0})

        branch_name = str(niche_id)
        self._ensure_branch(branch_name)
        latent_return, latent_failure, capital_req, maturity = self._sample_branch_characteristics(branch_name)
        latent_return *= mods["return_mult"]
        latent_failure *= mods["uncertainty_mult"]
        capital_req *= mods["capital_mult"]
        demand_adjustments = self._get_demand_adjustments(branch_name)
        latent_return *= demand_adjustments["return"] * self.regime_return_multiplier
        latent_failure *= demand_adjustments["failure"] * self.regime_failure_multiplier

        niche_opp = Opportunity(
            id=f"niche_{branch_name}_{round_num}_{np.random.randint(1000)}",
            latent_return_potential=float(np.clip(latent_return, 0.5, 25.0)),
            latent_failure_potential=float(np.clip(latent_failure, 0.1, 0.95)),
            complexity=float(np.random.uniform(0.4, 0.8)),
            discovered=False,
            discovery_round=round_num,
            created_by=discoverer_id,
            sector=branch_name,
            capital_requirements=float(capital_req),
            time_to_maturity=int(maturity),
            config=self.config,
        )

        with self._lock:
            self.opportunities.append(niche_opp)
            self.opportunity_map[niche_opp.id] = niche_opp
            self._index_opportunity(niche_opp)
        return niche_opp

    def record_innovation_outcome(
        self,
        opportunity: Opportunity,
        success: bool,
        return_achieved: float,
    ) -> None:
        """Log outcomes for opportunities created from innovations."""
        signature = getattr(opportunity, "combination_signature", None)
        if signature:
            try:
                self.innovation_engine.combination_tracker.record_outcome(
                    signature,
                    1.0 if success else 0.0,
                )
            except Exception:
                pass

        tracker = getattr(self.innovation_engine, "combination_tracker", None)
        reuse_ratio = 0.0
        if tracker and signature:
            reuse_ratio = tracker.get_reuse_ratio(signature)
        scarcity_context = float(getattr(opportunity, "component_scarcity", 0.5) or 0.5)
        if success:
            opportunity.market_impact = return_achieved
            intrinsic_gain = float(np.clip(return_achieved, 0.2, 3.5))
            scarcity_lift = 0.18 * max(0.0, scarcity_context - 0.4)
            reuse_drag = 0.4 * max(0.0, reuse_ratio - 0.2)
            adjustment = 1.0 + scarcity_lift - reuse_drag
            opportunity.latent_return_potential = float(
                np.clip(opportunity.latent_return_potential * adjustment + 0.05 * intrinsic_gain, 0.15, 4.0)
            )
            failure_drift = 1.0 + 0.5 * reuse_ratio - 0.3 * scarcity_context
            opportunity.latent_failure_potential = float(
                np.clip(opportunity.latent_failure_potential * failure_drift, 0.05, 0.95)
            )
            opportunity.crowding_penalty = reuse_ratio
            opportunity.component_scarcity = float(np.clip(0.5 * scarcity_context + 0.5 * (1.0 - reuse_ratio), 0.05, 1.0))
        else:
            opportunity.market_impact = 0.0
            failure_multiplier = 1.05 + 0.35 * reuse_ratio
            opportunity.latent_failure_potential = float(
                np.clip(opportunity.latent_failure_potential * failure_multiplier, 0.05, 0.99)
            )
            opportunity.component_scarcity = float(np.clip(opportunity.component_scarcity * 0.9, 0.02, 1.0))

    def spawn_opportunity_from_innovation(self, innovation: "Innovation", cash_multiple: float) -> None:
        """Create a derivative opportunity inspired by a successful innovation."""
        sector = getattr(innovation, "sector", "tech") or "tech"
        scarcity = float(getattr(innovation, "scarcity", 0.5) or 0.5)
        novelty = float(getattr(innovation, "novelty", 0.5) or 0.5)
        scarcity = float(np.clip(scarcity, 0.0, 1.0))
        novelty = float(np.clip(novelty, 0.0, 1.0))
        scarcity_scale = 0.85 + scarcity * 0.7
        novelty_scale = 0.9 + novelty * 0.6
        tracker = getattr(self.innovation_engine, "combination_tracker", None)
        signature = getattr(innovation, "combination_signature", None)
        reuse_ratio = tracker.get_reuse_ratio(signature) if tracker and signature else 0.0
        ai_domains = getattr(innovation, "ai_domains_used", []) or []
        ai_assisted = bool(getattr(innovation, "ai_assisted", False))

        intrinsic_multiple = float(
            np.clip(1.05 + (cash_multiple - 1.0) * 0.6, 0.8, 3.2)
            * scarcity_scale
            * novelty_scale
            * (0.9 + float(np.clip(getattr(innovation, "quality", 0.5), 0.0, 1.5)) * 0.2)
        )
        congestion_factor = float(np.clip(1.0 - reuse_ratio * (1.0 - scarcity), 0.35, 1.2))
        assistance_bonus = 1.0
        if ai_assisted:
            assistance_bonus += 0.08 * min(len(ai_domains), 3)
        tier_label = getattr(innovation, "ai_level_used", "none")
        if reuse_ratio > 0.5:
            assistance_bonus *= 0.7
        elif reuse_ratio < 0.15 and tier_label in {"basic", "none"}:
            assistance_bonus *= 1.12
        derived_multiplier = float(
            np.clip(intrinsic_multiple * congestion_factor * assistance_bonus, 0.65, 3.8)
        )
        base_failure_signal = float(
            np.clip(0.35 + reuse_ratio * 0.4 - scarcity * 0.2 - novelty * 0.1, 0.04, 0.9)
        )
        branch_name = str(sector)
        latent_return, latent_failure, capital_req, maturity = self._sample_branch_characteristics(branch_name)
        latent_return *= derived_multiplier
        latent_failure = float(
            np.clip(0.5 * latent_failure + 0.5 * base_failure_signal, 0.05, 0.95)
        )
        demand_adjustments = self._get_demand_adjustments(branch_name)
        latent_return *= demand_adjustments["return"] * self.regime_return_multiplier
        latent_failure *= demand_adjustments["failure"] * self.regime_failure_multiplier
        opp_id = f"spawn_{innovation.id}_{np.random.randint(1000)}"
        opportunity = Opportunity(
            id=opp_id,
            latent_return_potential=float(np.clip(latent_return, 0.5, 25.0)),
            latent_failure_potential=float(np.clip(latent_failure, 0.1, 0.95)),
            complexity=float(np.clip(0.3 + innovation.quality * 0.3, 0.3, 1.0)),
            discovered=False,
            discovery_round=self._current_round,
            created_by=innovation.creator_id,
            config=self.config,
            sector=branch_name,
            combination_signature=getattr(innovation, "combination_signature", None),
            origin_innovation_id=getattr(innovation, "id", None),
            crowding_penalty=reuse_ratio,
            component_scarcity=scarcity,
            capital_requirements=float(capital_req),
            time_to_maturity=int(maturity),
        )
        with self._lock:
            self.opportunities.append(opportunity)
            self.opportunity_map[opportunity.id] = opportunity
            self._index_opportunity(opportunity)

    def get_combination_diversity_metrics(self) -> Tuple[float, float]:
        combo_counts: Dict[str, int] = collections.Counter()
        sector_counts: Dict[str, int] = collections.Counter()
        for opp in self.opportunities:
            signature = getattr(opp, "combination_signature", None) or f"sector_{getattr(opp, 'sector', 'unknown')}"
            combo_counts[signature] += 1
            sector_counts[getattr(opp, "sector", "unknown")] += 1
        def _hhi(counter: Dict[str, int]) -> float:
            total = sum(counter.values())
            if total <= 0:
                return 0.0
            return float(sum((count / total) ** 2 for count in counter.values()))
        return _hhi(combo_counts), _hhi(sector_counts)
