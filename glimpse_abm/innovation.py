"""Innovation system for Glimpse ABM."""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import random

import numpy as np

from .config import EmergentConfig
from .knowledge import KnowledgeBase
from .models import Innovation, Knowledge
from .utils import fast_mean, safe_mean, stable_sigmoid


DOMAIN_TO_INDEX = {
    "technology": 0,
    "market": 1,
    "process": 2,
    "business_model": 3,
}

ADJACENCY_MATRIX = np.array(
    [
        [False, True, True, False],
        [True, False, False, True],
        [True, False, False, True],
        [False, True, True, False],
    ],
    dtype=bool,
)


@dataclass
class CombinationTracker:
    """Tracks knowledge combinations used across the market."""

    market_ref: Optional["MarketEnvironment"] = None
    combination_history: Dict[str, List[str]] = field(default_factory=lambda: collections.defaultdict(list))
    combination_success: Dict[str, List[float]] = field(default_factory=lambda: collections.defaultdict(list))
    total_combinations: int = 0

    def is_new_signature(self, combination_signature: Optional[str]) -> bool:
        if not combination_signature:
            return False
        return combination_signature not in self.combination_history

    def record_combination(self, combination_signature: str, innovation_id: str) -> None:
        self.combination_history[combination_signature].append(innovation_id)
        self.total_combinations += 1

    def record_outcome(self, combination_signature: str, success: float) -> None:
        self.combination_success[combination_signature].append(success)

    def sample_signature(self, lookback: int = 100) -> Optional[str]:
        if not self.combination_history:
            return None
        signatures = list(self.combination_history.keys())
        if not signatures:
            return None
        signature = random.choice(signatures)
        history = self.combination_history.get(signature, [])
        if history and lookback > 0:
            history = history[-lookback:]
            if not history:
                return signature
        return signature

    def get_reuse_ratio(self, combination_signature: Optional[str]) -> float:
        if not combination_signature:
            return 0.0
        history = self.combination_history.get(combination_signature, [])
        if not history or self.total_combinations <= 0:
            return 0.0
        return float(np.clip(len(history) / max(1, self.total_combinations), 0.0, 1.0))




class InnovationEngine:
    """Handles the creation of new knowledge through combinations."""

    def __init__(self, config: EmergentConfig, knowledge_base: KnowledgeBase, combination_tracker: CombinationTracker):
        self.config = config
        self.knowledge_base = knowledge_base
        self.combination_tracker = combination_tracker
        self.innovations: Dict[str, Innovation] = {}
        self.innovation_history = collections.defaultdict(list)
        self.rd_investments = collections.defaultdict(float)
        self.ai_assisted_innovations = set()
        self.innovation_success_by_ai = {"ai_assisted": [], "human_only": []}

    def attempt_innovation(
        self,
        agent: "EmergentAgent",
        market_conditions: Dict,
        round: int,
        ai_level: str = "none",
        uncertainty_perception: Dict = None,
        decision_perception: Optional[Dict] = None,
    ) -> Optional[Innovation]:
        accessible_knowledge = self.knowledge_base.get_accessible_knowledge(
            agent.id, ai_level, agent.resources, agent.traits
        )
        if len(accessible_knowledge) < 2:
            return None
        # Use sector-specific innovation probability (NSF BRDIS/USPTO calibrated)
        sector_profile = self.config.SECTOR_PROFILES.get(agent.primary_sector, {})
        base_prob = sector_profile.get('innovation_probability', self.config.INNOVATION_PROBABILITY)
        competence_score = (
            agent.traits["innovativeness"] * 0.6 + agent.resources.capabilities.get("innovation", 0.1) * 0.4
        )
        # REMOVED: Hardcoded ai_bonus_map that gave direct tier-based bonuses
        # Previously: ai_bonus_map = {"none": 0.0, "basic": 0.12, "advanced": 0.25, "premium": 0.35}
        # Now AI bonus emerges purely from agent's learned trust and reliability estimates
        avg_trust = 0.5
        dynamic_bonus = 0.0
        clarity_signal = 0.0
        if decision_perception:
            ignorance = decision_perception.get("actor_ignorance", {}).get("ignorance_level", 0.5)
            indeterminism = decision_perception.get("practical_indeterminism", {}).get("indeterminism_level", 0.5)
            clarity_signal = ((stable_sigmoid(1.0 - ignorance) + stable_sigmoid(1.0 - indeterminism)) / 2.0) - 0.5

        if ai_level != "none" and getattr(agent, "ai_learning", None) is not None:
            learning_profile = agent.ai_learning
            trust_values = [
                learning_profile.domain_trust.get(dom, 0.5)
                for dom in ["technical_assessment", "innovation_potential"]
            ]
            avg_trust = float(fast_mean(trust_values)) if trust_values else 0.5
            reliability_signals = []
            for dom in ["technical_assessment", "innovation_potential"]:
                scores = learning_profile.accuracy_estimates.get(dom, [])
                if scores:
                    reliability_signals.append(float(fast_mean(scores[-5:])) - 0.5)
            reliability = float(fast_mean(reliability_signals)) if reliability_signals else 0.0
            # Dynamic bonus based on LEARNED trust and reliability (emergent)
            dynamic_bonus = (avg_trust - 0.5) * 0.2 + reliability * 0.25 + clarity_signal * 0.15

        # AI bonus now purely from emergent learning, no hardcoded tier bonuses
        ai_bonus = np.clip(dynamic_bonus, -0.2, 0.3)
        human_ingenuity_bonus = (
            agent.traits["exploration_tendency"] * 0.15
            + agent.traits["market_awareness"] * 0.15
            + agent.traits["innovativeness"] * 0.15
        )
        innovation_prob = (
            base_prob * 0.35 + competence_score * 0.45 + ai_bonus + human_ingenuity_bonus
        )
        innovation_prob = float(np.clip(innovation_prob, 0.05, 0.95))
        if np.random.random() > innovation_prob:
            return None

        ai_domains_used = []
        if ai_level != "none" and getattr(agent, "ai_learning", None) is not None:
            learning_profile = agent.ai_learning
            for domain in ["technical_assessment", "innovation_potential"]:
                trust = learning_profile.domain_trust.get(domain, 0.5)
                recent_scores = learning_profile.accuracy_estimates.get(domain, [])
                recent_positive = [score for score in recent_scores[-5:] if score >= 0.65]
                has_positive_outcome = len(recent_positive) > 0
                if trust > 0.45 and has_positive_outcome:
                    ai_domains_used.append(domain)

        experience_units = getattr(agent.resources, "experience_units", 0)
        innovation_type = self._determine_innovation_type(
            accessible_knowledge,
            agent.traits,
            market_conditions,
            ai_assisted=len(ai_domains_used) > 0,
            experience_units=experience_units,
        )
        n_components = self._get_component_count(innovation_type)
        reuse_prob = float(getattr(self.config, "INNOVATION_REUSE_PROBABILITY", 0.0) or 0.0)
        lookback = int(getattr(self.config, "INNOVATION_REUSE_LOOKBACK", 100))
        selected_knowledge = None
        reuse_signature = None
        # FIXED: Remove hardcoded tier_reuse_shift - let effect emerge through info_breadth
        # Previously had direct tier shifts (none=+0.05, premium=-0.08)
        # Now reuse probability emerges from info_breadth: broader info access → more novel
        # combinations available → lower tendency to reuse existing combinations
        ai_cfg = self.config.AI_LEVELS.get(ai_level, self.config.AI_LEVELS.get('none', {}))
        info_breadth = float(ai_cfg.get('info_breadth', 0.0))
        # Higher info_breadth reduces reuse (access to broader knowledge enables novel combinations)
        reuse_shift = -info_breadth * 0.12
        effective_reuse_prob = np.clip(reuse_prob + reuse_shift, 0.02, 0.75)
        if reuse_prob > 0 and np.random.random() < effective_reuse_prob:
            reuse_signature = self.combination_tracker.sample_signature(lookback=lookback)
            if reuse_signature:
                component_ids = reuse_signature.split("||")
                knowledge_lookup = self.knowledge_base.knowledge_pieces
                candidate = [knowledge_lookup[k_id] for k_id in component_ids if k_id in knowledge_lookup]
                if len(candidate) >= n_components:
                    selected_knowledge = candidate[:n_components]
        if selected_knowledge is None:
            selected_knowledge = self._select_knowledge_combination(
                accessible_knowledge,
                n_components,
                agent.traits,
                agent.ai_learning if hasattr(agent, "ai_learning") else None,
                ai_level=ai_level,
            )

        if not selected_knowledge:
            return None

        innovation_sector = self._determine_innovation_sector(agent, selected_knowledge)
        innovation = self._create_innovation(
            selected_knowledge,
            innovation_type,
            agent,
            round,
            ai_assisted=len(ai_domains_used) > 0,
            ai_domains_used=ai_domains_used,
            sector=innovation_sector,
        )
        if reuse_signature:
            innovation.combination_signature = reuse_signature
            innovation.novelty = float(np.clip(innovation.novelty * 0.6, 0.0, 1.0))

        self.knowledge_base.record_usage(innovation.knowledge_components)
        innovation.scarcity = self.knowledge_base.get_combination_scarcity(innovation.knowledge_components)

        self.innovations[innovation.id] = innovation
        self.innovation_history[agent.id].append(innovation)
        if innovation.ai_assisted:
            self.ai_assisted_innovations.add(innovation.id)
        return innovation

    def _determine_innovation_type(
        self,
        knowledge_pieces: List[Knowledge],
        agent_traits: Dict[str, float],
        market_conditions: Dict,
        ai_assisted: bool = False,
        experience_units: float = 0.0,
    ) -> str:
        base_probabilities = {
            "incremental": 0.4 + max(experience_units, 0) * 0.01,
            "architectural": 0.3 + agent_traits.get("trait_momentum", 0.1) * 0.35,
            "radical": 0.2 + agent_traits.get("innovativeness", 0.5) * 0.25,
            "disruptive": 0.1 + agent_traits.get("exploration_tendency", 0.5) * 0.2,
        }
        if market_conditions.get("regime") == "crisis":
            base_probabilities["disruptive"] += 0.05
        if ai_assisted:
            base_probabilities["architectural"] += 0.08
            base_probabilities["radical"] += 0.05
        total = sum(base_probabilities.values())
        for key in base_probabilities:
            base_probabilities[key] /= total
        types = list(base_probabilities.keys())
        probabilities = list(base_probabilities.values())
        return str(np.random.choice(types, p=probabilities))

    def _get_component_count(self, innovation_type: str) -> int:
        if innovation_type == "incremental":
            return int(np.random.choice([2, 3], p=[0.7, 0.3]))
        if innovation_type == "architectural":
            return int(np.random.choice([3, 4], p=[0.6, 0.4]))
        if innovation_type == "radical":
            return int(np.random.choice([3, 4, 5], p=[0.3, 0.4, 0.3]))
        return int(np.random.choice([4, 5], p=[0.5, 0.5]))

    def _select_knowledge_combination(
        self,
        accessible_knowledge: List[Knowledge],
        n_components: int,
        agent_traits: Dict[str, float],
        ai_learning_profile: Optional[AILearningProfile] = None,
        ai_level: str = "none",
    ) -> Optional[List[Knowledge]]:
        if not accessible_knowledge or len(accessible_knowledge) < n_components:
            return None
        accessible = list(accessible_knowledge)
        num_accessible = len(accessible)

        domain_indices_list: List[int] = []
        domain_recognised = True
        for item in accessible:
            try:
                domain_indices_list.append(DOMAIN_TO_INDEX[item.domain])
            except KeyError:
                domain_recognised = False
                break
        domain_indices = np.array(domain_indices_list, dtype=np.int8) if domain_recognised else None
        levels = np.array([k.level for k in accessible], dtype=float)

        first_idx = random.randrange(num_accessible)
        knowledge_components = [accessible[first_idx]]
        selected_indices = [first_idx]
        remaining_indices = [idx for idx in range(num_accessible) if idx != first_idx]
        if domain_indices is not None:
            selected_domains = [int(domain_indices[first_idx])]
            selected_levels = [float(levels[first_idx])]
        else:
            selected_domains = None
            selected_levels = None

        while len(knowledge_components) < n_components and remaining_indices:
            scores: List[float] = []
            len_selected = len(selected_indices)
            if domain_indices is not None:
                for idx in remaining_indices:
                    domain_idx = int(domain_indices[idx])
                    level_val = float(levels[idx])
                    total = 0.0
                    for dom, lvl in zip(selected_domains, selected_levels):
                        diff = abs(level_val - lvl)
                        if domain_idx == dom:
                            comp = 0.8 + 0.2 * (1.0 - diff)
                        elif ADJACENCY_MATRIX[domain_idx, dom]:
                            comp = 0.4 + 0.2 * (1.0 - diff)
                        else:
                            comp = 0.1 + 0.1 * (1.0 - diff)
                        total += comp
                    scores.append(total / len_selected if len_selected else 0.1)
            else:
                selected_objects = [accessible[s_idx] for s_idx in selected_indices]
                for idx in remaining_indices:
                    total = 0.0
                    for other in selected_objects:
                        total += self.knowledge_base.get_compatibility(accessible[idx], other)
                    scores.append(total / len_selected if len_selected else 0.1)

            weights = np.maximum(np.asarray(scores, dtype=float), 1e-6)
            total_weight = weights.sum()
            if total_weight <= 0:
                break
            weights /= total_weight

            choice_pos = int(np.random.choice(len(remaining_indices), p=weights))
            next_idx = remaining_indices.pop(choice_pos)
            if next_idx in selected_indices:
                continue
            selected_indices.append(next_idx)
            knowledge_components.append(accessible[next_idx])
            if domain_indices is not None:
                selected_domains.append(int(domain_indices[next_idx]))
                selected_levels.append(float(levels[next_idx]))

        if len(knowledge_components) < 2:
            return None
        usage_lookup = self.knowledge_base.knowledge_usage
        def _usage_score(piece: Knowledge) -> float:
            return float(usage_lookup.get(piece.id, 0))
        if ai_level in {"premium", "advanced"}:
            rare_pool = sorted(accessible, key=_usage_score)
            replace_idx = max(range(len(knowledge_components)), key=lambda i: _usage_score(knowledge_components[i]))
            for candidate in rare_pool:
                if candidate not in knowledge_components:
                    knowledge_components[replace_idx] = candidate
                    break
        elif ai_level == "basic":
            common_pool = sorted(accessible, key=_usage_score, reverse=True)
            for candidate in common_pool:
                if candidate not in knowledge_components:
                    replace_idx = random.randrange(len(knowledge_components))
                    knowledge_components[replace_idx] = candidate
                    break
        return knowledge_components

    def _determine_innovation_sector(
        self, agent: "EmergentAgent", selected_knowledge: List[Knowledge]
    ) -> str:
        sector_preferences = getattr(agent, "preferred_sectors", None)
        if sector_preferences:
            weights = np.array(list(sector_preferences.values()), dtype=float)
            weights = weights / weights.sum() if weights.sum() > 0 else None
            chosen_sector = np.random.choice(list(sector_preferences.keys()), p=weights)
            return str(chosen_sector)
        # Use knowledge base's domain-to-sector mapping (respects config.SECTORS)
        domain_to_sector = getattr(self.knowledge_base, 'DOMAIN_TO_SECTOR_MAP', {
            "technology": "tech",
            "market": "tech",
            "process": "tech",
            "business_model": "tech",
        })
        domains = collections.Counter(k.domain for k in selected_knowledge)
        dominant_domain = domains.most_common(1)[0][0]
        # Default to first available sector if domain not mapped
        available_sectors = getattr(self.config, 'SECTORS', ['tech']) if self.config else ['tech']
        default_sector = available_sectors[0] if available_sectors else 'tech'
        return domain_to_sector.get(dominant_domain, default_sector)

    def _create_innovation(
        self,
        knowledge_pieces: List[Knowledge],
        innovation_type: str,
        agent: "EmergentAgent",
        round: int,
        ai_assisted: bool = False,
        ai_domains_used: Optional[List[str]] = None,
        sector: Optional[str] = None,
    ) -> Innovation:
        knowledge_levels = [k.level for k in knowledge_pieces]
        base_quality = safe_mean(knowledge_levels) * 0.7 + np.random.uniform(0.2, 0.4)
        mix_penalty = np.std(knowledge_levels)
        quality = np.clip(base_quality * (1 - mix_penalty * 0.3), 0.1, 1.0)
        if ai_assisted:
            quality += 0.05
        quality = np.clip(quality + np.random.normal(0, 0.05), 0, 1)
        novelty = np.clip(
            fast_mean([abs(k.level - quality) for k in knowledge_pieces]) + np.random.uniform(0.1, 0.3),
            0,
            1,
        )
        if sector is None:
            sector = "tech"
        innovation_id = f"inn_{agent.id}_{round}_{np.random.randint(1000)}"
        innovation = Innovation(
            id=innovation_id,
            type=innovation_type,
            knowledge_components=[k.id for k in knowledge_pieces],
            novelty=novelty,
            quality=quality,
            round_created=round,
            creator_id=agent.id,
            ai_assisted=ai_assisted,
            ai_domains_used=ai_domains_used or [],
            sector=sector,
        )
        combination_signature = "||".join(sorted(innovation.knowledge_components))
        innovation.is_new_combination = self.combination_tracker.is_new_signature(combination_signature)
        self.combination_tracker.record_combination(combination_signature, innovation.id)
        innovation.combination_signature = combination_signature
        return innovation

    def invest_in_rd(self, agent_id: int, amount: float) -> None:
        self.rd_investments[agent_id] += amount

    def evaluate_innovation_success(
        self,
        innovation: Innovation,
        market_conditions: Dict,
        market_innovations: List[Innovation],
    ) -> Tuple[bool, float, float]:
        potential = innovation.calculate_potential(market_conditions)
        if not hasattr(innovation, "sector") or innovation.sector is None:
            innovation.sector = "tech"
        competing_innovations = [
            inn
            for inn in market_innovations
            if inn.id != innovation.id
            and inn.round_created >= innovation.round_created - 5
            and inn.sector is not None
            and innovation.sector is not None
            and inn.sector == innovation.sector
        ]
        # Get sector-specific competition intensity (Census HHI-calibrated)
        sector_profile = self.config.SECTOR_PROFILES.get(innovation.sector, {})
        sector_competition_intensity = sector_profile.get('competition_intensity', 1.0)

        if competing_innovations:
            competitor_strength = safe_mean(
                [c.quality * c.novelty for c in competing_innovations]
            )
            # Apply sector-specific intensity to competition effects
            competition_factor = 1 - min(0.5, competitor_strength * sector_competition_intensity)
        else:
            competition_factor = 1.0
        if innovation.novelty > 0.8:
            readiness_factor = 0.7
        elif innovation.novelty < 0.3:
            readiness_factor = 0.8
        else:
            readiness_factor = 1.0
        scarcity = float(getattr(innovation, "scarcity", 0.5) or 0.5)
        novelty = float(getattr(innovation, "novelty", 0.5) or 0.5)
        scarcity_boost = 1.0 + (scarcity - 0.5) * 0.35
        success_prob = float(np.clip(potential * competition_factor * readiness_factor * scarcity_boost, 0.0, 1.0))
        success = bool(np.random.random() < success_prob)
        impact = 0.0
        cash_multiple = 0.0
        if success:
            impact = innovation.quality * innovation.novelty * competition_factor
            if innovation.sector == "tech":
                impact *= 1.2
            elif innovation.sector == "manufacturing":
                impact *= 0.9
            impact = float(np.clip(impact * (1.0 + (novelty - 0.5) * 0.25), 0.05, 2.5))
            base_multiple = 1.25 + getattr(self.config, "INNOVATION_SUCCESS_BASE_RETURN", 0.25)
            # Use sector-specific innovation return multiplier (R&D intensity calibrated)
            sector_profile = self.config.SECTOR_PROFILES.get(innovation.sector, {})
            mult_range = sector_profile.get(
                'innovation_return_multiplier',
                getattr(self.config, "INNOVATION_SUCCESS_RETURN_MULTIPLIER", (1.8, 3.0))
            )
            if isinstance(mult_range, (list, tuple)) and len(mult_range) >= 2:
                low, high = float(mult_range[0]), float(mult_range[1])
            else:
                low = float(mult_range)
                high = low + 1.0
            if high < low:
                low, high = high, low
            impact_gain = np.random.uniform(low, high)
            scarcity_bonus = 1.0 + (scarcity - 0.5) * 0.8
            novelty_bonus = 1.0 + (novelty - 0.5) * 0.55
            cash_multiple = float(
                np.clip((base_multiple + impact * impact_gain) * scarcity_bonus * novelty_bonus, 1.1, 8.5)
            )
        else:
            recovery_ratio = getattr(self.config, 'INNOVATION_FAIL_RECOVERY_RATIO', 0.15)
            recovery_floor = np.interp(
                float(np.clip(innovation.novelty, 0.05, 0.95)),
                [0.05, 0.95],
                [0.78, 0.42],
            )
            salvage = max(recovery_ratio, recovery_floor - 0.12 * (scarcity - 0.5))
            cash_multiple = float(np.clip(salvage, 0.25, 0.65))
        innovation.success = success
        innovation.market_impact = impact
        if innovation.ai_assisted:
            self.innovation_success_by_ai["ai_assisted"].append(success)
        else:
            self.innovation_success_by_ai["human_only"].append(success)
        return success, impact, cash_multiple

    def get_innovation_metrics(self, agent_id: int) -> Dict:
        agent_innovations = self.innovation_history.get(agent_id, [])
        if not agent_innovations:
            return {
                "total_innovations": 0,
                "successful_innovations": 0,
                "success_rate": 0.0,
                "avg_quality": 0.0,
                "avg_novelty": 0.0,
                "total_impact": 0.0,
                "knowledge_pieces": 0,
                "ai_assisted_count": 0,
                "ai_success_rate": 0.0,
            }

        successful = [inn for inn in agent_innovations if inn.success]
        ai_assisted = [inn for inn in agent_innovations if inn.ai_assisted]
        ai_successful = [inn for inn in ai_assisted if inn.success]

        return {
            "total_innovations": len(agent_innovations),
            "successful_innovations": len(successful),
            "success_rate": len(successful) / len(agent_innovations),
            "avg_quality": safe_mean([inn.quality for inn in agent_innovations]),
            "avg_novelty": safe_mean([inn.novelty for inn in agent_innovations]),
            "total_impact": sum(inn.market_impact or 0 for inn in agent_innovations),
            "knowledge_pieces": len(self.knowledge_base.agent_knowledge.get(agent_id, set())),
            "ai_assisted_count": len(ai_assisted),
            "ai_success_rate": len(ai_successful) / len(ai_assisted) if ai_assisted else 0.0,
        }
