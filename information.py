"""Information system components for Glimpse ABM."""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import numba
except ImportError:
    class _NumbaStub:  # type: ignore
        def jit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

    numba = _NumbaStub()  # type: ignore

from .config import EmergentConfig
from .models import AILearningProfile, AIAnalysis, Information, Opportunity


@numba.jit(nopython=True)
def get_enhanced_ai_analysis_numba(
    latent_return_potential,
    latent_failure_potential,
    complexity,
    base_accuracy,
    base_quality,
    hallucination_rate,
    bias,
    return_range,
    uncertainty_range,
    overconfidence_intensity=1.0,
):
    actual_accuracy = max(0.0, min(np.random.normal(base_accuracy, 0.1), 1.0))
    return_noise = np.random.normal(0, (1 - actual_accuracy) * 0.5)
    estimated_return = latent_return_potential + return_noise + bias * 0.3
    uncertainty_noise = np.random.normal(0, (1 - actual_accuracy) * 0.3)
    estimated_uncertainty = latent_failure_potential + uncertainty_noise - bias * 0.2

    contains_hallucination = np.random.random() < hallucination_rate
    if contains_hallucination:
        if np.random.random() < 0.5:
            estimated_return = np.random.uniform(return_range[0], return_range[1])
        else:
            estimated_uncertainty = np.random.uniform(uncertainty_range[0], uncertainty_range[1])

    estimated_return = max(return_range[0], min(estimated_return, return_range[1]))
    estimated_uncertainty = max(
        uncertainty_range[0], min(estimated_uncertainty, uncertainty_range[1])
    )

    true_confidence = actual_accuracy * (1 - complexity * (1 - actual_accuracy))
    # Scale overconfidence by intensity parameter (for robustness testing)
    base_overconfidence = (0.5 - base_quality) * 0.5 if base_quality < 0.5 else 0.0
    overconfidence_factor = 1.0 + base_overconfidence * overconfidence_intensity
    stated_confidence = max(0.1, min(true_confidence * overconfidence_factor, 0.95))

    return estimated_return, estimated_uncertainty, stated_confidence, contains_hallucination


class InformationSystem:
    """System for generating information about opportunities."""

    def __init__(self, config: EmergentConfig):
        self.config = config
        self.information_cache: Dict[Tuple[str, str], Information] = {}
        self.discovered_opportunities = set()
        self._cache_hits = 0
        self._cache_misses = 0
        self.domain_performance = {
            "market_analysis": [],
            "technical_assessment": [],
            "uncertainty_evaluation": [],
            "innovation_potential": [],
        }

    def get_information(
        self, opportunity: Opportunity, ai_level: str, agent_id: Optional[int] = None
    ) -> Information:
        cache_key = (opportunity.id, ai_level)
        if cache_key in self.information_cache:
            self._cache_hits += 1
            return self.information_cache[cache_key]

        self._cache_misses += 1
        ai_config = self.config.AI_LEVELS[ai_level]
        domain = self._determine_domain(opportunity)
        domain_cap = self.config.get_ai_domain_capability(ai_level, domain)

        # Quality-dependent accuracy scaling with lognormal distribution
        # (matches Julia's superior approach in information.jl:182-187)
        info_quality = ai_config["info_quality"]
        tier_noise = max(0.1, 1.2 - info_quality)
        base_accuracy = domain_cap["accuracy"]

        # Apply lognormal accuracy noise
        accuracy_noise_scale = 0.12 + 0.08 * tier_noise
        accuracy_noise_loc = 1.0 - 0.35 * (1.0 - info_quality)
        accuracy_noise = np.exp(np.random.normal(0, 1) * accuracy_noise_scale) * accuracy_noise_loc
        actual_accuracy = float(np.clip(base_accuracy * accuracy_noise, 0.35, 0.99))

        # 2-step hallucination rate chain (simplified version)
        # Step 1: Get base hallucination rate from domain capability
        base_hallucination_rate = domain_cap["hallucination_rate"]

        # Step 2: Apply intensity scaling (robustness parameter)
        hallucination_intensity = getattr(self.config, 'HALLUCINATION_INTENSITY', 1.0)
        hallucination_rate = base_hallucination_rate * hallucination_intensity

        bias = domain_cap["bias"]

        return_noise = np.random.normal(0, 1) * (1 - actual_accuracy) * 0.5
        estimated_return = opportunity.latent_return_potential + return_noise + bias * 0.3
        estimated_return = float(
            np.clip(estimated_return, *self.config.OPPORTUNITY_RETURN_RANGE)
        )

        uncertainty_noise = np.random.normal(0, 1) * (1 - actual_accuracy) * 0.3
        estimated_uncertainty = (
            opportunity.latent_failure_potential + uncertainty_noise - bias * 0.2
        )
        estimated_uncertainty = float(
            np.clip(estimated_uncertainty, *self.config.OPPORTUNITY_UNCERTAINTY_RANGE)
        )

        contains_hallucination = np.random.random() < hallucination_rate
        if contains_hallucination:
            if np.random.random() < 0.5:
                estimated_return = float(
                    np.random.uniform(*self.config.OPPORTUNITY_RETURN_RANGE)
                )
            else:
                estimated_uncertainty = float(
                    np.random.uniform(*self.config.OPPORTUNITY_UNCERTAINTY_RANGE)
                )

        true_confidence = actual_accuracy * (
            1 - opportunity.complexity * (1 - actual_accuracy)
        )
        # Scale overconfidence by intensity parameter (for robustness testing)
        overconfidence_intensity = getattr(self.config, 'OVERCONFIDENCE_INTENSITY', 1.0)
        base_overconfidence = (0.5 - ai_config["info_quality"]) * 0.5 if ai_config["info_quality"] < 0.5 else 0.0
        overconfidence_factor = 1.0 + base_overconfidence * overconfidence_intensity
        stated_confidence = float(np.clip(true_confidence * overconfidence_factor, 0.1, 0.95))

        info = Information(
            estimated_return=estimated_return,
            estimated_uncertainty=estimated_uncertainty,
            confidence=stated_confidence,
            insights=self._generate_insights(opportunity, ai_config["info_breadth"], domain),
            hidden_factors={
                "bias": bias,
                "unknown_uncertainty": (1 - ai_config["info_breadth"]) * opportunity.complexity,
                "market_shift_sensitivity": 1 - actual_accuracy,
                "hallucination_uncertainty": float(contains_hallucination),
            },
            domain=domain,
            actual_accuracy=actual_accuracy,
            contains_hallucination=contains_hallucination,
            bias_applied=bias,
            overconfidence_factor=overconfidence_factor,
        )
        self.information_cache[cache_key] = info
        return info

    def _determine_domain(self, opportunity: Opportunity) -> str:
        if hasattr(opportunity, "created_by") and opportunity.created_by is not None:
            return "innovation_potential"
        if opportunity.complexity > 0.7:
            return "technical_assessment"
        if hasattr(opportunity, "sector") and opportunity.sector in ["tech", "manufacturing"]:
            return "technical_assessment"
        if opportunity.competition > 0.5:
            return "market_analysis"
        return "uncertainty_evaluation"

    def _generate_insights(
        self, opportunity: Opportunity, breadth: float, domain: Optional[str] = None
    ) -> List[str]:
        insights: List[str] = []
        if breadth > 0.2:
            insights.append(f"Market lifecycle: {opportunity.lifecycle_stage}")
        if breadth > 0.4:
            if opportunity.competition > 0.5:
                insights.append("High competitive pressure detected")
            else:
                insights.append("Limited competition currently")
        if breadth > 0.6:
            if opportunity.path_dependency > 0.5:
                insights.append("Strong first-mover advantages")
            if opportunity.complexity > 0.7:
                insights.append("Requires specialized capabilities")
            if hasattr(opportunity, "sector"):
                insights.append(f"Sector: {opportunity.sector}")
        if breadth > 0.8:
            insights.append(f"Hidden uncertainty factors: ~{(1-breadth)*opportunity.complexity:.1%}")
            if opportunity.created_by is not None:
                insights.append("Novel opportunity (agent-created)")
            if domain:
                insights.append(f"Analysis domain: {domain}")
        return insights

    def clear_cache(self) -> None:
        self.information_cache.clear()


class EnhancedInformationSystem(InformationSystem):
    """Enhanced information system with AI learning integration."""

    ai_cache_clear_interval: int = 20

    def __init__(self, config: EmergentConfig, market_ref: "MarketEnvironment"):
        super().__init__(config)
        self.market = market_ref
        self.last_cache_clear_round = 0
        self.ai_capabilities = self.config.AI_DOMAIN_CAPABILITIES
        self.agent_learning_profiles: Dict[int, AILearningProfile] = {}
        self.ai_analysis_cache: Dict[Tuple[str, str, int], Information] = {}
        self.max_cache_size = config.max_cache_size if hasattr(config, "max_cache_size") else 10000
        self.cache_hits = 0
        self.cache_misses = 0
        self._lock = threading.Lock()

    def set_lock(self, lock) -> None:
        self._lock = lock if lock is not None else threading.Lock()

    def initialize_agent_learning(self, agent_id: int) -> None:
        if agent_id not in self.agent_learning_profiles:
            self.agent_learning_profiles[agent_id] = AILearningProfile()

    def get_stochastic_hallucination_rate(self, base_rate: float, domain: str) -> float:
        if base_rate < 0.1:
            alpha, beta = 2, 20
        elif base_rate < 0.2:
            alpha, beta = 3, 12
        else:
            alpha, beta = 4, 8
        stochastic_factor = np.random.uniform(0.0, 1.0)
        stochastic_rate = base_rate * (0.5 + stochastic_factor)
        if hasattr(self, "_hallucination_streak"):
            if np.random.random() < 0.7:
                stochastic_rate *= self._hallucination_streak
        else:
            self._hallucination_streak = np.random.uniform(0.8, 1.2)
        if np.random.random() < 0.1:
            self._hallucination_streak = np.random.uniform(0.8, 1.2)
        return float(np.clip(stochastic_rate, 0, 0.5))

    def update_agent_learning(self, agent_id: int, outcome: Dict, analysis: AIAnalysis) -> bool:
        if agent_id not in self.agent_learning_profiles:
            self.initialize_agent_learning(agent_id)
        profile = self.agent_learning_profiles[agent_id]
        domain = analysis.domain
        profile.usage_count[domain] += 1

        investment_amount = outcome.get("investment_amount")
        if investment_amount is None and "investment" in outcome:
            investment_amount = outcome["investment"].get("amount", 1)
        if investment_amount is None:
            investment_amount = 1
        actual_return = outcome.get("capital_returned", 0) / max(1, investment_amount)
        predicted_return = analysis.estimated_return
        return_error = abs(actual_return - predicted_return) / max(1, predicted_return)
        was_accurate = return_error < 0.3

        profile.accuracy_estimates[domain].append(1 - return_error)
        if len(profile.accuracy_estimates[domain]) > 20:
            profile.accuracy_estimates[domain].pop(0)

        if analysis.contains_hallucination and not was_accurate:
            profile.hallucination_experiences[domain] += 1

        if outcome.get("success", False):
            profile.update_trust(domain, was_accurate, magnitude=0.1)
        elif not was_accurate and analysis.confidence > 0.7:
            profile.update_trust(domain, False, magnitude=0.2)

        return was_accurate

    def maybe_clear_cache(self, round_num: int) -> None:
        interval = getattr(self.config, "AI_CACHE_CLEAR_INTERVAL", self.ai_cache_clear_interval)
        if round_num - self.last_cache_clear_round >= interval:
            self.clear_cache()
            self.ai_analysis_cache.clear()
            self.last_cache_clear_round = round_num

    def get_information(self, opportunity: Opportunity, ai_level: str, agent: "EmergentAgent") -> Information:
        cache_key = (ai_level, getattr(opportunity, 'id', None))
        if ai_level == "none":
            quality = (
                agent.traits.get("analytical_ability", 0.5) * 0.4
                + agent.traits.get("competence", 0.5) * 0.4
                + agent.traits.get("market_awareness", 0.5) * 0.2
            ) * 0.30
            breadth = (
                agent.traits.get("exploration_tendency", 0.5) * 0.5
                + agent.traits.get("market_awareness", 0.5) * 0.5
            ) * 0.25
            return_noise = np.random.normal(0, (1 - quality) * 0.7)
            estimated_return = opportunity.latent_return_potential + return_noise
            uncertainty_noise = np.random.normal(0, (1 - quality) * 0.5)
            estimated_uncertainty = opportunity.latent_failure_potential + uncertainty_noise
            confidence = float(
                np.clip(agent.traits.get("competence", 0.5) * 0.4, 0.05, 0.45)
            )
            return Information(
                estimated_return=float(
                    np.clip(estimated_return, *self.config.OPPORTUNITY_RETURN_RANGE)
                ),
                estimated_uncertainty=float(
                    np.clip(estimated_uncertainty, *self.config.OPPORTUNITY_UNCERTAINTY_RANGE)
                ),
                confidence=confidence,
                insights=tuple(self._generate_insights(opportunity, breadth, None)),
                hidden_factors={
                    "human_bias": (0.5 - agent.traits.get("analytical_ability", 0.5)),
                    "unknowns": 1 - quality,
                },
            )

        if agent is not None and ai_level != "none":
            try:
                agent.register_ai_usage(ai_level)
            except AttributeError:
                pass

        agent_id = agent.id
        cache_key = (opportunity.id, ai_level)

        cached_info = None
        if self._lock is not None:
            with self._lock:
                if agent_id not in self.agent_learning_profiles:
                    self.initialize_agent_learning(agent_id)
                cached_info = self.ai_analysis_cache.get(cache_key)
                if cached_info is not None:
                    self.cache_hits += 1
                    return cached_info
                self.cache_misses += 1
                if len(self.ai_analysis_cache) > self.max_cache_size:
                    keys_to_remove = list(self.ai_analysis_cache.keys())[
                        : int(self.max_cache_size * 0.2)
                    ]
                    for key in keys_to_remove:
                        del self.ai_analysis_cache[key]
        else:
            if agent_id not in self.agent_learning_profiles:
                self.initialize_agent_learning(agent_id)
            cached_info = self.ai_analysis_cache.get(cache_key)
            if cached_info is not None:
                self.cache_hits += 1
                return cached_info
            self.cache_misses += 1

        ai_config = self.config.AI_LEVELS.get(ai_level, self.config.AI_LEVELS["none"])
        domain = self._determine_domain(opportunity)
        domain_cap = self.ai_capabilities.get(ai_level, {}).get(domain, {})
        info_quality = ai_config.get("info_quality", 0.0)
        info_breadth = ai_config.get("info_breadth", 0.0)
        base_accuracy = domain_cap.get("accuracy", 0.5)

        # 4-step hallucination rate chain (matches Julia information.jl:189-204)
        # Step 1: Get base hallucination rate from domain capability
        base_hallucination_rate = domain_cap.get("hallucination_rate", 0.1)

        # Step 2: Apply stochastic modification
        hallucination_rate = self.get_stochastic_hallucination_rate(
            base_hallucination_rate, domain
        )

        # Step 3: Apply lognormal variance (quality-dependent noise)
        tier_noise = max(0.1, 1.2 - info_quality)
        lognormal_mu = 0.0
        lognormal_sigma = 0.25 * tier_noise
        lognormal_factor = float(np.clip(np.random.lognormal(mean=lognormal_mu, sigma=lognormal_sigma), 0.2, 3.0))
        hallucination_rate *= lognormal_factor

        # Step 4: Apply intensity scaling (robustness parameter)
        hallucination_intensity = getattr(self.config, 'HALLUCINATION_INTENSITY', 1.0)
        hallucination_rate = float(np.clip(hallucination_rate * hallucination_intensity, 0.0, 1.0))
        accuracy_noise = float(np.clip(np.random.normal(loc=1.0 - 0.35 * (1.0 - info_quality), scale=0.12 + 0.08 * tier_noise), 0.4, 1.2))
        base_accuracy = float(np.clip(base_accuracy * accuracy_noise, 0.35, 0.99))
        bias = domain_cap.get("bias", 0.0)

        # Scale overconfidence by intensity parameter (for robustness testing)
        overconfidence_intensity = getattr(self.config, 'OVERCONFIDENCE_INTENSITY', 1.0)

        est_return, est_uncertainty, stated_confidence, contains_hallucination = get_enhanced_ai_analysis_numba(
            opportunity.latent_return_potential,
            opportunity.latent_failure_potential,
            opportunity.complexity,
            base_accuracy,
            info_quality,
            hallucination_rate,
            bias,
            self.config.OPPORTUNITY_RETURN_RANGE,
            self.config.OPPORTUNITY_UNCERTAINTY_RANGE,
            overconfidence_intensity,
        )

        true_insights = self._generate_insights(opportunity, info_breadth, domain)
        false_insights = (
            self._generate_false_insights(opportunity, domain) if contains_hallucination else []
        )

        base_overconfidence = (0.5 - info_quality) * 0.5 if info_quality < 0.5 else 0.0
        overconfidence_factor = 1.0 + base_overconfidence * overconfidence_intensity

        analysis = AIAnalysis(
            estimated_return=est_return,
            estimated_uncertainty=est_uncertainty,
            confidence=stated_confidence,
            insights=tuple(true_insights + false_insights),
            hidden_factors={
                "bias": bias,
                "unknown_uncertainty": (1 - info_quality) * opportunity.complexity,
            },
            actual_accuracy=float(np.clip(np.random.normal(base_accuracy, 0.1), 0, 1)),
            contains_hallucination=contains_hallucination,
            bias_applied=bias,
            domain=domain,
            false_insights=tuple(false_insights),
            overconfidence_factor=overconfidence_factor,
        )
        info = self._convert_to_information(analysis)

        if self._lock is not None:
            with self._lock:
                if len(self.ai_analysis_cache) > self.max_cache_size:
                    keys_to_remove = list(self.ai_analysis_cache.keys())[
                        : int(self.max_cache_size * 0.2)
                    ]
                    for key in keys_to_remove:
                        del self.ai_analysis_cache[key]
                self.ai_analysis_cache[cache_key] = info
        else:
            if len(self.ai_analysis_cache) > self.max_cache_size:
                keys_to_remove = list(self.ai_analysis_cache.keys())[
                    : int(self.max_cache_size * 0.2)
                ]
                for key in keys_to_remove:
                    del self.ai_analysis_cache[key]
            self.ai_analysis_cache[cache_key] = info

        return info

    def _convert_to_information(self, analysis: AIAnalysis) -> Information:
        info = Information(
            estimated_return=analysis.estimated_return,
            estimated_uncertainty=analysis.estimated_uncertainty,
            confidence=analysis.confidence,
            insights=analysis.insights,
            hidden_factors=analysis.hidden_factors,
            actual_accuracy=analysis.actual_accuracy,
            contains_hallucination=analysis.contains_hallucination,
            bias_applied=analysis.bias_applied,
            domain=analysis.domain,
            overconfidence_factor=analysis.overconfidence_factor,
        )
        setattr(info, "_source_analysis", analysis)
        return info

    def _generate_false_insights(self, opportunity: Opportunity, domain: str) -> List[str]:
        false_insight_templates = {
            "market_analysis": [
                "Untapped customer segment identified in emerging markets",
                "Significant demand spike detected through alternative data",
                "Competitor withdrawal likely within 3 quarters",
            ],
            "technical_assessment": [
                "Breakthrough resolves key scalability challenges",
                "Prototype efficiency exceeds market benchmarks",
                "Regulatory path streamlined due to new standards",
            ],
            "uncertainty_evaluation": [
                "Black swan likelihood reduced by recent policy changes",
                "Risk contagion limited to adjacent sectors",
                "Dominant uncertainty source neutralized by supplier agreement",
            ],
            "innovation_potential": [
                "High cross-domain synergy with existing portfolio",
                "Productized knowledge base accelerates time-to-market",
                "Community adoption hurdle lower than industry peers",
            ],
        }
        domain_insights = false_insight_templates.get(domain, [])
        n_false = min(2, len(domain_insights))
        if n_false > 0 and len(domain_insights) > 0:
            return list(np.random.choice(domain_insights, n_false, replace=False))
        return []
