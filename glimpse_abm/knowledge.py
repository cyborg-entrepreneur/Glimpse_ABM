"""Knowledge base management for Glimpse ABM."""

from __future__ import annotations

import collections
import random
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from .config import EmergentConfig
from .models import Innovation, Knowledge, Opportunity
from .utils import normalize_ai_label

if TYPE_CHECKING:  # pragma: no cover - only for typing
    from .agents import EmergentAgent


class KnowledgeBase:
    """Manages collective knowledge in the system."""

    # Default domain-to-sector mapping (used when all 4 sectors available)
    _DEFAULT_DOMAIN_TO_SECTOR_MAP = {
        "technology": "tech",
        "market": "retail",
        "process": "service",
        "business_model": "manufacturing",
    }

    def __init__(self, config: Optional[EmergentConfig] = None) -> None:
        self.config = config
        # Build domain-to-sector map based on available sectors
        available_sectors = list(getattr(config, 'SECTORS', [])) if config else []
        if not available_sectors:
            available_sectors = list(self._DEFAULT_DOMAIN_TO_SECTOR_MAP.values())

        # If only one sector available, map all domains to it
        if len(available_sectors) == 1:
            self.DOMAIN_TO_SECTOR_MAP = {
                domain: available_sectors[0] for domain in self._DEFAULT_DOMAIN_TO_SECTOR_MAP.keys()
            }
        else:
            # Use default mapping, but fall back to first available sector if mapped sector missing
            self.DOMAIN_TO_SECTOR_MAP = {}
            for domain, default_sector in self._DEFAULT_DOMAIN_TO_SECTOR_MAP.items():
                if default_sector in available_sectors:
                    self.DOMAIN_TO_SECTOR_MAP[domain] = default_sector
                else:
                    self.DOMAIN_TO_SECTOR_MAP[domain] = available_sectors[0]

        # Build reverse mapping, keeping first domain for each sector
        # (important for single-sector mode where multiple domains map to one sector)
        self.SECTOR_TO_DOMAIN_MAP = {}
        for domain, sector in self.DOMAIN_TO_SECTOR_MAP.items():
            if sector not in self.SECTOR_TO_DOMAIN_MAP:
                self.SECTOR_TO_DOMAIN_MAP[sector] = domain
        self.knowledge_pieces: Dict[str, Knowledge] = {}
        self.domain_knowledge: Dict[str, List[Knowledge]] = collections.defaultdict(list)
        self.agent_knowledge: Dict[int, Set[str]] = collections.defaultdict(set)
        self.knowledge_graph: Dict[str, Set[str]] = collections.defaultdict(set)
        self.ai_discovered_knowledge: Set[str] = set()
        self._compatibility_cache: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
        self._knowledge_registry: List[Knowledge] = []
        self.knowledge_usage: Dict[str, int] = collections.defaultdict(int)
        self.agent_domain_beliefs = collections.defaultdict(
            lambda: collections.defaultdict(lambda: {"alpha": 2.0, "beta": 2.0})
        )
        self._initialize_base_knowledge()

    def _initialize_base_knowledge(self) -> None:
        base_domains = ["technology", "market", "process", "business_model"]
        for domain in base_domains:
            for level in [0.1, 0.2, 0.3]:
                knowledge = Knowledge(
                    id=f"base_{domain}_{level}",
                    domain=domain,
                    level=level,
                    discovered_round=0,
                    discovered_by=None,
                )
                self.add_knowledge(knowledge)
                self.knowledge_usage[knowledge.id] = 0

    def _get_ai_info_signals(self, ai_level: str) -> tuple[float, float]:
        if self.config is None:
            default_map = {
                "none": (0.0, 0.0),
                "basic": (0.35, 0.30),
                "advanced": (0.65, 0.55),
                "premium": (0.9, 0.8),
            }
            return default_map.get(normalize_ai_label(ai_level), (0.0, 0.0))
        profile = self.config.AI_LEVELS.get(ai_level) or self.config.AI_LEVELS.get("none", {})
        return float(profile.get("info_quality", 0.0)), float(profile.get("info_breadth", 0.0))

    def _reinforce_agent_resources(self, agent_resources: Optional["AgentResources"], knowledge: Knowledge, info_quality: float) -> None:
        if agent_resources is None or not hasattr(agent_resources, "knowledge"):
            return
        sector = self.DOMAIN_TO_SECTOR_MAP.get(knowledge.domain, knowledge.domain)
        knowledge_map = getattr(agent_resources, "knowledge", {})
        if sector not in knowledge_map:
            return
        delta = 0.025 + 0.08 * info_quality
        knowledge_map[sector] = float(np.clip(knowledge_map[sector] + delta, 0.01, 1.15))
        if hasattr(agent_resources, "knowledge_last_used"):
            agent_resources.knowledge_last_used[sector] = getattr(agent_resources, "current_round", 0)

    def _apply_tier_decay(
        self,
        agent_id: int,
        ai_level: str,
        agent_resources: Optional["AgentResources"] = None,
    ) -> None:
        tier = normalize_ai_label(ai_level)
        info_quality, _ = self._get_ai_info_signals(ai_level)
        # FIXED: Uniform base decay rate - tier differences emerge through info_quality
        # Previously had hardcoded tier-specific decay (premium=0.0, none=0.08)
        # Now all tiers have same base decay, modified by info_quality retention
        base_decay = 0.06  # Uniform base decay rate
        # Higher info_quality reduces decay (retention_modifier ranges 0.2 to 0.73)
        retention_modifier = float(np.clip(1.0 - 0.45 * info_quality, 0.2, 1.0))
        drop_prob = base_decay * retention_modifier
        knowledge_ids = self.agent_knowledge.get(agent_id, set())
        if not knowledge_ids or drop_prob <= 0.0:
            return
        if np.random.random() >= drop_prob:
            return
        drop_candidates = [kid for kid in knowledge_ids if not kid.startswith("base_")]
        if not drop_candidates:
            return
        drop_count = max(1, int(len(drop_candidates) * max(0.05, drop_prob)))
        removed = np.random.choice(drop_candidates, size=min(len(drop_candidates), drop_count), replace=False)
        if np.isscalar(removed):
            removed = [removed.item() if hasattr(removed, "item") else removed]
        for k_id in removed:
            knowledge_ids.discard(k_id)
            knowledge = self.knowledge_pieces.get(k_id)
            if knowledge is None or agent_resources is None or not hasattr(agent_resources, "knowledge"):
                continue
            sector = self.DOMAIN_TO_SECTOR_MAP.get(knowledge.domain, knowledge.domain)
            knowledge_map = agent_resources.knowledge
            if sector in knowledge_map:
                base_decay = 0.03 + 0.05 * max(drop_prob, 0.0)
                decay = base_decay * retention_modifier
                knowledge_map[sector] = max(0.01, knowledge_map[sector] * (1.0 - decay))

    def ensure_starter_knowledge(self, agent_id: int, pieces_per_domain: int = 1) -> None:
        starter_set = self.agent_knowledge[agent_id]
        if len(starter_set) >= 2:
            return

        base_domains = ["technology", "market", "process", "business_model"]
        primary_domain = base_domains[agent_id % len(base_domains)]
        domain_pieces = [
            k for k in self.domain_knowledge.get(primary_domain, []) if k.id.startswith("base_")
        ]
        if domain_pieces:
            starter_set.add(domain_pieces[0].id)

        if len(starter_set) >= 2:
            return

        base_candidates = [k.id for k in self.knowledge_pieces.values() if k.id.startswith("base_")]
        cross_domain_candidates = [
            kid
            for kid in base_candidates
            if not kid.startswith(f"base_{primary_domain}_") and kid not in starter_set
        ]
        if not cross_domain_candidates:
            cross_domain_candidates = [kid for kid in base_candidates if kid not in starter_set]
        if cross_domain_candidates:
            starter_set.add(str(np.random.choice(cross_domain_candidates)))

    def add_knowledge(self, knowledge: Knowledge, ai_discovered: bool = False) -> None:
        self.knowledge_pieces[knowledge.id] = knowledge
        self.domain_knowledge[knowledge.domain].append(knowledge)
        if ai_discovered:
            self.ai_discovered_knowledge.add(knowledge.id)
        for parent_id in knowledge.parent_knowledge:
            self.knowledge_graph[parent_id].add(knowledge.id)
        for other in self._knowledge_registry:
            score = knowledge.compatibility_with(other)
            self._compatibility_cache[knowledge.id][other.id] = score
            self._compatibility_cache[other.id][knowledge.id] = score
        self._compatibility_cache[knowledge.id][knowledge.id] = 1.0
        self._knowledge_registry.append(knowledge)
        self.knowledge_usage.setdefault(knowledge.id, 0)

    def get_accessible_knowledge(
        self,
        agent_id: int,
        ai_level: str = "none",
        agent_resources: Optional["AgentResources"] = None,
        agent_traits: Optional[dict] = None,
    ) -> List[Knowledge]:
        self.ensure_starter_knowledge(agent_id)
        agent_knowledge_ids = self.agent_knowledge.get(agent_id, set())

        self._apply_tier_decay(agent_id, ai_level, agent_resources)

        info_quality, info_breadth = self._get_ai_info_signals(ai_level)
        # FIXED: Remove hardcoded tier bonuses - let effects emerge through info_quality/info_breadth
        # Previously had double-dipping: flat tier bonus + info_quality effects
        # Now tier differences emerge purely from AI_LEVELS config (info_quality, info_breadth)
        bonus = info_quality * 0.55 + info_breadth * 0.45

        exploration_trait = 0.0
        if agent_traits:
            exploration_trait = agent_traits.get("exploration_tendency", 0.0)
            bonus += exploration_trait * 0.08

        network_strength = 0.0
        if agent_resources and hasattr(agent_resources, "network"):
            network_strength = float(agent_resources.network)
            bonus += network_strength * 0.25

        accessible = [self.knowledge_pieces[k_id] for k_id in agent_knowledge_ids if k_id in self.knowledge_pieces]
        other_knowledge = [k for k in self._knowledge_registry if k.id not in agent_knowledge_ids]

        acquired_now: Set[str] = set()
        if other_knowledge and bonus > 0:
            attempt_multiplier = 1.0 + info_breadth * 5.0 + network_strength * 0.6 + exploration_trait * 0.4
            attempts = max(1, int(round(attempt_multiplier)))
            for _ in range(attempts):
                knowledge_piece = other_knowledge[np.random.randint(len(other_knowledge))]
                novelty_bias = 1.0 + info_quality * (getattr(knowledge_piece, "level", 0.5) - 0.5)
                novelty_bias = float(np.clip(novelty_bias, 0.35, 1.5))
                threshold = float(np.clip(bonus * (1.0 - knowledge_piece.level) * novelty_bias, 0.0, 0.99))
                if threshold <= 0.0 or knowledge_piece.id in agent_knowledge_ids:
                    continue
                if np.random.random() < threshold:
                    agent_knowledge_ids.add(knowledge_piece.id)
                    acquired_now.add(knowledge_piece.id)
                    accessible.append(knowledge_piece)
                    self._reinforce_agent_resources(agent_resources, knowledge_piece, info_quality)

        return accessible

    def record_usage(self, knowledge_ids: List[str]) -> None:
        for k_id in knowledge_ids:
            self.knowledge_usage[k_id] = self.knowledge_usage.get(k_id, 0) + 1

    def get_combination_scarcity(self, knowledge_ids: List[str]) -> float:
        if not knowledge_ids:
            return 0.5
        scarcity_values = []
        for k_id in knowledge_ids:
            usage = self.knowledge_usage.get(k_id, 0)
            scarcity_values.append(1.0 / (1.0 + usage))
        return float(np.clip(np.mean(scarcity_values), 0.0, 1.0))

    def get_component_scarcity_metric(self) -> float:
        total_pieces = len(self.knowledge_pieces)
        if total_pieces <= 0:
            return 0.5

        usage_values = list(self.knowledge_usage.values())
        if usage_values:
            usage_array = np.asarray(usage_values, dtype=float)
            effective_peak = float(np.quantile(usage_array, 0.9)) or 1.0
            normalised = np.clip(usage_array / max(effective_peak, 1.0), 0.0, 1.0)
            rarity_component = float(np.clip(1.0 - float(normalised.mean()), 0.0, 1.0))
        else:
            rarity_component = 0.85

        used_pieces = set()
        for knowledge_ids in self.agent_knowledge.values():
            used_pieces.update(knowledge_ids)
        coverage = len(used_pieces) / total_pieces
        unused_share = float(np.clip(1.0 - coverage, 0.0, 1.0))

        scarcity = 0.6 * rarity_component + 0.4 * unused_share
        return float(np.clip(scarcity, 0.0, 1.0))

    def get_average_agent_knowledge(self) -> float:
        if not self.agent_knowledge:
            return 0.0
        counts = [len(kset) for kset in self.agent_knowledge.values()]
        if not counts:
            return 0.0
        return float(np.mean(counts))

    def get_compatibility(self, first: Knowledge, second: Knowledge) -> float:
        if first.id == second.id:
            return 1.0
        cached = self._compatibility_cache.get(first.id, {})
        if second.id in cached:
            return cached[second.id]
        score = first.compatibility_with(second)
        self._compatibility_cache[first.id][second.id] = score
        self._compatibility_cache[second.id][first.id] = score
        return score

    def learn_from_success(
        self,
        agent_id: int,
        innovation: Innovation,
        agent_resources: Optional["AgentResources"] = None,
    ) -> None:
        for k_id in innovation.knowledge_components:
            self.agent_knowledge[agent_id].add(k_id)

        if agent_resources and hasattr(agent_resources, "knowledge"):
            domain_counts = collections.Counter()
            for k_id in innovation.knowledge_components:
                if k_id in self.knowledge_pieces:
                    domain_counts[self.knowledge_pieces[k_id].domain] += 1
            if domain_counts:
                dominant_domain = domain_counts.most_common(1)[0][0]
                sector = self.DOMAIN_TO_SECTOR_MAP.get(dominant_domain, "tech")
                if sector in agent_resources.knowledge:
                    agent_resources.knowledge[sector] = min(
                        1.0, agent_resources.knowledge[sector] + 0.1
                    )
                    if hasattr(agent_resources, "knowledge_last_used"):
                        agent_resources.knowledge_last_used[sector] = innovation.round_created

        if innovation.success and innovation.quality > 0.7:
            new_knowledge = self._create_derived_knowledge(innovation)
            if new_knowledge:
                self.add_knowledge(new_knowledge, ai_discovered=innovation.ai_assisted)
                self.agent_knowledge[agent_id].add(new_knowledge.id)

    def learn_from_failure(
        self,
        agent_id: int,
        innovation: Innovation,
        agent_resources: Optional["AgentResources"] = None,
    ) -> None:
        for k_id in innovation.knowledge_components:
            if np.random.random() < 0.3:
                self.agent_knowledge[agent_id].add(k_id)

        if agent_resources and hasattr(agent_resources, "capabilities"):
            agent_resources.capabilities["uncertainty_management"] = min(
                1.0, agent_resources.capabilities["uncertainty_management"] + 0.02
            )

        if innovation.quality > 0.5 and np.random.random() < 0.2:
            failure_knowledge = Knowledge(
                id=f"failure_{innovation.id}",
                domain="process",
                level=0.3 + 0.2 * innovation.quality,
                discovered_round=innovation.round_created,
                discovered_by=agent_id,
                parent_knowledge=innovation.knowledge_components[:2],
            )
            self.add_knowledge(failure_knowledge)
            self.agent_knowledge[agent_id].add(failure_knowledge.id)
            if agent_resources and hasattr(agent_resources, "knowledge_last_used"):
                sector = self.DOMAIN_TO_SECTOR_MAP.get("process", "service")
                agent_resources.knowledge_last_used[sector] = innovation.round_created

    def apply_hallucination_penalty(
        self,
        agent: "EmergentAgent",
        domain: str,
        severity: float = 0.2,
    ) -> None:
        """Reduce effective knowledge trust when AI hallucinations mislead an agent."""
        if agent is None or getattr(agent, "resources", None) is None:
            return

        resources = agent.resources
        knowledge_map = getattr(resources, "knowledge", None)
        if not isinstance(knowledge_map, dict) or not knowledge_map:
            return

        # Get available sectors from agent's knowledge
        available_sectors = list(knowledge_map.keys())
        default_sector = available_sectors[0] if available_sectors else 'tech'

        # Map domains to available sectors (fall back to default if mapped sector unavailable)
        _base_domain_to_sector = {
            "market_analysis": "retail",
            "technical_assessment": "tech",
            "uncertainty_evaluation": "service",
            "innovation_potential": "manufacturing",
        }
        domain_to_sector = {}
        for d, s in _base_domain_to_sector.items():
            domain_to_sector[d] = s if s in available_sectors else default_sector

        # Build neighbor map only for available sectors
        sector_neighbors = {sector: tuple(s for s in available_sectors if s != sector) for sector in available_sectors}

        target_sector = domain_to_sector.get(domain)
        if target_sector is None or target_sector not in knowledge_map:
            return

        penalty = float(np.clip(0.1 + severity * 0.45, 0.1, 0.65))
        knowledge_map[target_sector] = max(0.01, knowledge_map[target_sector] * (1.0 - penalty))
        if hasattr(resources, "knowledge_last_used"):
            resources.knowledge_last_used[target_sector] = max(
                resources.knowledge_last_used.get(target_sector, 0), getattr(agent, "current_round", 0)
            )

        # Propagate a lighter penalty to adjacent sectors to reflect contagion of mistrust.
        neighbor_penalty = penalty * 0.35
        for neighbor in sector_neighbors.get(target_sector, ()):
            if neighbor in knowledge_map:
                knowledge_map[neighbor] = max(0.01, knowledge_map[neighbor] * (1.0 - neighbor_penalty))
                if hasattr(resources, "knowledge_last_used"):
                    resources.knowledge_last_used[neighbor] = max(
                        resources.knowledge_last_used.get(neighbor, 0), getattr(agent, "current_round", 0)
                    )

        # Nudge cognitive capabilities downward so repeated hallucinations lower execution quality.
        if hasattr(resources, "capabilities") and isinstance(resources.capabilities, dict):
            for key in ("uncertainty_management", "opportunity_evaluation"):
                if key in resources.capabilities:
                    resources.capabilities[key] = max(
                        0.01, resources.capabilities[key] * (1.0 - penalty * 0.25)
                    )

        # Suppress domain trust inside the AI learning profile to create longer shadow effects.
        profile = getattr(agent, "ai_learning", None)
        if profile is not None and domain in profile.domain_trust:
            profile.domain_trust[domain] = max(
                0.0, profile.domain_trust[domain] - penalty * 0.4
            )

        # Trigger targeted forgetting in the impacted sector.
        current_level = getattr(agent, "current_ai_level", None) or getattr(agent, "primary_ai_level", None) or "none"
        self._forget_sector_knowledge(agent, target_sector, severity, ai_level=current_level)

    def forget_stale_knowledge(
        self,
        agent: "EmergentAgent",
        current_round: int,
        max_size: Optional[int] = None,
        drop_fraction: float = 0.1,
    ) -> None:
        """Cull least-used knowledge when portfolios balloon."""
        knowledge_ids = list(self.agent_knowledge.get(agent.id, set()))
        if not knowledge_ids:
            return
        config_max = getattr(agent.config, "MAX_AGENT_KNOWLEDGE", 120)
        max_keep = int(max_size or config_max)
        if max_keep <= 0:
            max_keep = 60
        if len(knowledge_ids) <= max_keep:
            return

        drop_count = len(knowledge_ids) - max_keep
        drop_extra = max(1, int(max_keep * drop_fraction))
        total_drop = max(1, drop_count + drop_extra)

        def _keep_score(k_id: str) -> float:
            piece = self.knowledge_pieces.get(k_id)
            usage = self.knowledge_usage.get(k_id, 0)
            discovered_round = getattr(piece, "discovered_round", current_round)
            age = max(0, current_round - discovered_round)
            novelty_bonus = getattr(piece, "level", 0.0)
            return usage * 1.6 + novelty_bonus - age * 0.04

        sorted_ids = sorted(knowledge_ids, key=_keep_score)
        structured_drop = max(1, int(total_drop * 0.6))
        to_remove = list(sorted_ids[:structured_drop])
        remaining = total_drop - len(to_remove)
        if remaining > 0:
            random_pool = [kid for kid in knowledge_ids if kid not in to_remove]
            if random_pool:
                random_remove = random.sample(random_pool, min(len(random_pool), remaining))
                to_remove.extend(random_remove)
        for k_id in to_remove[:total_drop]:
            self._remove_agent_knowledge(agent, k_id)

    def prune_by_sector_strength(
        self,
        agent: "EmergentAgent",
        sector_levels: Optional[Dict[str, float]],
        threshold: float = 0.05,
    ) -> None:
        if agent is None or not sector_levels:
            return
        knowledge_ids = list(self.agent_knowledge.get(agent.id, set()))
        if not knowledge_ids:
            return
        for k_id in knowledge_ids:
            piece = self.knowledge_pieces.get(k_id)
            if piece is None:
                continue
            sector = self._knowledge_to_sector(piece)
            if sector_levels.get(sector, 1.0) < threshold:
                self._remove_agent_knowledge(agent, k_id)

    def _forget_sector_knowledge(
        self,
        agent: "EmergentAgent",
        sector: str,
        severity: float,
        ai_level: Optional[str] = None,
    ) -> None:
        if not sector:
            return
        agent_knowledge = self.agent_knowledge.get(agent.id)
        if not agent_knowledge:
            return
        level_key = ai_level or getattr(agent, "current_ai_level", None) or getattr(agent, "primary_ai_level", None) or "none"
        info_quality, _ = self._get_ai_info_signals(level_key)
        severity = float(np.clip(severity * (1.0 - 0.25 * info_quality), 0.05, 1.0))
        piece_ids = [
            k_id
            for k_id in list(agent_knowledge)
            if self.knowledge_pieces.get(k_id, None)
            and self._knowledge_to_sector(self.knowledge_pieces[k_id]) == sector
        ]
        if not piece_ids:
            return
        drop_count = max(1, int(len(piece_ids) * np.clip(severity * 0.4, 0.08, 0.65)))
        piece_ids.sort(key=lambda k: self.knowledge_usage.get(k, 0))
        for k_id in piece_ids[:drop_count]:
            self._remove_agent_knowledge(agent, k_id)

    def _knowledge_to_sector(self, knowledge: Knowledge) -> str:
        return self.DOMAIN_TO_SECTOR_MAP.get(knowledge.domain, "tech")

    def _sector_to_domain(self, sector: str) -> str:
        return self.SECTOR_TO_DOMAIN_MAP.get(sector, "process")

    def _update_domain_belief(
        self,
        agent_id: int,
        domain: str,
        evidence: float,
    ) -> float:
        evidence = float(np.clip(evidence, 0.0, 1.0))
        belief = self.agent_domain_beliefs[agent_id][domain]
        belief["alpha"] += evidence
        belief["beta"] += max(0.0, 1.0 - evidence)
        total = belief["alpha"] + belief["beta"]
        if total <= 0:
            return 0.5
        return float(belief["alpha"] / total)

    def get_domain_belief(self, agent_id: int, domain: str) -> float:
        belief = self.agent_domain_beliefs.get(agent_id, {}).get(domain)
        if not belief:
            return 0.5
        total = belief["alpha"] + belief["beta"]
        if total <= 0:
            return 0.5
        return float(belief["alpha"] / total)

    def _get_agent_domain_knowledge(self, agent_id: int, domain: str) -> List[str]:
        knowledge_ids = list(self.agent_knowledge.get(agent_id, set()))
        if not knowledge_ids:
            return []
        matches: List[str] = []
        for k_id in knowledge_ids:
            piece = self.knowledge_pieces.get(k_id)
            if piece is not None and piece.domain == domain:
                matches.append(k_id)
        return matches

    def _prune_domain_fragment(
        self,
        agent: "EmergentAgent",
        domain: str,
        count: int = 1,
    ) -> None:
        if count <= 0:
            return
        removable = self._get_agent_domain_knowledge(agent.id, domain)
        if not removable:
            return
        removable.sort(key=lambda k: self.knowledge_usage.get(k, 0))
        for k_id in removable[:count]:
            self._remove_agent_knowledge(agent, k_id)

    def _maybe_create_applied_knowledge(
        self,
        agent_id: int,
        domain: str,
        roi_signal: float,
        round_num: int,
        opportunity: Optional[Opportunity],
        ai_level: str,
    ) -> None:
        level = float(np.clip(0.25 + 0.4 * roi_signal, 0.1, 0.95))
        parent_refs: List[str] = []
        if opportunity is not None:
            combo_signature = getattr(opportunity, "combination_signature", None)
            if combo_signature:
                parent_refs = [combo_signature]
        identifier = f"invest_{domain}_{agent_id}_{round_num}_{np.random.randint(1_000_000)}"
        knowledge = Knowledge(
            id=identifier,
            domain=domain,
            level=level,
            discovered_round=round_num,
            discovered_by=agent_id,
            parent_knowledge=parent_refs,
        )
        ai_discovered = ai_level not in {"none", "", None}
        self.add_knowledge(knowledge, ai_discovered=ai_discovered)
        self.agent_knowledge[agent_id].add(knowledge.id)

    def apply_investment_outcome(
        self,
        agent: "EmergentAgent",
        sector: Optional[str],
        roi: float,
        success: bool,
        round_num: int,
        opportunity: Optional[Opportunity] = None,
        ai_level: str = "none",
    ) -> None:
        if agent is None or not sector:
            return
        resources = getattr(agent, "resources", None)
        if resources is None or not hasattr(resources, "knowledge"):
            return
        self.ensure_starter_knowledge(agent.id)
        knowledge_map = resources.knowledge
        knowledge_map.setdefault(sector, 0.05)
        domain = self._sector_to_domain(str(sector))
        info_quality, _ = self._get_ai_info_signals(ai_level)
        domain_piece_ids = self._get_agent_domain_knowledge(agent.id, domain)
        if domain_piece_ids:
            self.record_usage(domain_piece_ids)
        if hasattr(resources, "knowledge_last_used"):
            resources.knowledge_last_used[sector] = round_num

        roi = float(roi)
        success = bool(success)
        roi_clipped = float(np.clip(roi, -1.5, 1.5))
        evidence = 0.5 + 0.35 * np.tanh(roi_clipped)
        if success:
            evidence = max(evidence, 0.65)
        else:
            evidence = min(evidence, 0.35)
        posterior = self._update_domain_belief(agent.id, domain, evidence)
        belief_scale = 0.6 + 0.8 * posterior

        if success:
            roi_signal = float(np.clip(roi, 0.0, 1.5))
            gain = (0.04 + roi_signal * 0.2) * belief_scale
            if ai_level not in {"none", "", None}:
                gain += 0.01
            gain *= 1.0 + 0.35 * info_quality
            knowledge_map[sector] = float(np.clip(knowledge_map[sector] + gain, 0.01, 1.5))
            if roi_signal >= 0.1 and posterior >= 0.55:
                self._maybe_create_applied_knowledge(
                    agent.id,
                    domain,
                    roi_signal,
                    round_num,
                    opportunity,
                    ai_level,
                )
        else:
            loss_base = 0.03 + min(abs(roi), 1.5) * 0.18
            penalty_scale = 0.5 + 0.9 * (1.0 - posterior)
            loss = loss_base * penalty_scale
            loss *= (1.0 + 0.4 * (1.0 - info_quality))
            knowledge_map[sector] = float(np.clip(knowledge_map[sector] * (1.0 - loss), 0.01, 1.0))
            severity = float(np.clip(abs(roi), 0.1, 1.0))
            self._forget_sector_knowledge(agent, sector, severity, ai_level=ai_level)
            if abs(roi) >= 0.4:
                self._prune_domain_fragment(agent, domain, count=1)

    def _remove_agent_knowledge(self, agent: "EmergentAgent", knowledge_id: str) -> None:
        if knowledge_id not in self.agent_knowledge.get(agent.id, set()):
            return
        self.agent_knowledge[agent.id].discard(knowledge_id)
        piece = self.knowledge_pieces.get(knowledge_id)
        if piece is None:
            return
        sector = self._knowledge_to_sector(piece)
        resources = getattr(agent, "resources", None)
        if resources and hasattr(resources, "knowledge"):
            if sector in resources.knowledge:
                resources.knowledge[sector] = max(0.01, resources.knowledge[sector] * 0.9)
            if hasattr(resources, "knowledge_last_used") and sector in resources.knowledge_last_used:
                resources.knowledge_last_used[sector] = max(
                    resources.knowledge_last_used.get(sector, 0), getattr(agent, "current_round", 0)
                )

    def _create_derived_knowledge(self, innovation: Innovation) -> Optional[Knowledge]:
        if len(innovation.knowledge_components) < 2:
            return None

        component_domains = []
        max_level = 0.0
        for k_id in innovation.knowledge_components[:3]:
            if k_id in self.knowledge_pieces:
                k = self.knowledge_pieces[k_id]
                component_domains.append(k.domain)
                max_level = max(max_level, k.level)

        if not component_domains:
            return None

        domain_counts = collections.Counter(component_domains)
        new_domain = domain_counts.most_common(1)[0][0]
        level_advance = {
            "incremental": 0.1,
            "architectural": 0.15,
            "radical": 0.25,
            "disruptive": 0.3,
        }
        new_level = min(1.0, max_level + level_advance.get(innovation.type, 0.1))

        return Knowledge(
            id=f"derived_{innovation.id}",
            domain=new_domain,
            level=new_level,
            discovered_round=innovation.round_created,
            discovered_by=innovation.creator_id,
            parent_knowledge=innovation.knowledge_components[:3],
        )
