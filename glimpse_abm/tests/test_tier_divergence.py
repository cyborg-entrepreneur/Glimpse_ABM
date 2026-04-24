"""Regression test: AI-tier-aware ranking (Python parity with Julia).

Asserts that two agents facing the same opportunity set but with different
AI tiers produce different top-3 rankings in a non-trivial fraction of trials.
Mirrors julia/test/test_tier_divergence.jl line-for-line in spirit.

Pre-correctness Python ranking already used info.estimated_return (unlike
Julia v1, which discarded it), so this test should pass on the canonical
Python codebase. After applying Julia v2 Fix #5 (liquidity-relief tier field)
to Python, this test continues to pass and additionally guards the
InformationSystem-driven ranking from regressing into bypass behavior.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
for candidate in (PARENT, ROOT):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from glimpse_abm.agents import EmergentAgent
from glimpse_abm.config import EmergentConfig
from glimpse_abm.information import EnhancedInformationSystem
from glimpse_abm.innovation import CombinationTracker, InnovationEngine
from glimpse_abm.knowledge import KnowledgeBase
from glimpse_abm.market import MarketEnvironment
from glimpse_abm.uncertainty import KnightianUncertaintyEnvironment

N_TRIALS = 50
N_OPPS = 30


def _build_agent(agent_id: int, config: EmergentConfig, kb: KnowledgeBase,
                 ie: InnovationEngine, info_system: EnhancedInformationSystem,
                 ai_level: str, seed: int) -> EmergentAgent:
    np.random.seed(seed)
    traits = {
        "exploration_tendency": 0.5,
        "market_awareness": 0.5,
        "ai_trust": 0.5,
        "entrepreneurial_drive": 0.5,
        "uncertainty_tolerance": 0.5,
        "analytical_ability": 0.5,
        "cognitive_style": 0.5,
        "competence": 0.5,
        "innovativeness": 0.5,
        "trait_momentum": 0.5,
        "risk_tolerance": 0.5,
        "social_influence": 0.5,
        "learning_rate": 0.5,
        "experience": 0.5,
    }
    agent = EmergentAgent(
        agent_id=agent_id,
        initial_traits=traits,
        config=config,
        knowledge_base=kb,
        innovation_engine=ie,
        agent_type=ai_level,
        primary_sector="tech",
    )
    kb.ensure_starter_knowledge(agent.id)
    info_system.initialize_agent_learning(agent.id)
    agent.ai_learning = info_system.agent_learning_profiles.get(agent.id)
    return agent


def test_ai_tier_ranking_divergence() -> None:
    config = EmergentConfig()
    config.N_AGENTS = 2
    config.N_ROUNDS = 1
    config.AGENT_AI_MODE = "fixed"

    kb = KnowledgeBase(config=config)
    ct = CombinationTracker()
    ie = InnovationEngine(config=config, knowledge_base=kb, combination_tracker=ct)

    np.random.seed(1)
    uncertainty_env = KnightianUncertaintyEnvironment(config=config, knowledge_base=kb)
    market = MarketEnvironment(
        config=config,
        uncertainty_env=uncertainty_env,
        innovation_engine=ie,
        knowledge_base=kb,
    )
    info_system = EnhancedInformationSystem(config=config, market_ref=market)

    # Build a fixed opportunity set
    opportunities = []
    for i in range(N_OPPS):
        sector = np.random.choice(config.SECTORS)
        opp = market._create_realistic_opportunity(f"test_{i}", sector)
        opportunities.append(opp)

    market_conditions = {"regime": "normal", "uncertainty_state": {}}
    perception = {"neighbor_signals": {}}

    top1_match = 0
    top3_overlap_sum = 0.0
    rank_corr_sum = 0.0
    valid_corr_count = 0

    for trial in range(N_TRIALS):
        seed = 1000 + trial

        agent_none = _build_agent(1, config, kb, ie, info_system,
                                  ai_level="none", seed=seed)
        agent_premium = _build_agent(2, config, kb, ie, info_system,
                                     ai_level="premium", seed=seed)

        # Reset info-system cache between trials so tier-noise samples fresh
        if hasattr(info_system, "information_cache"):
            info_system.information_cache.clear()

        evals_none, _ = agent_none._evaluate_portfolio_opportunities(
            opportunities, info_system, market_conditions, "none", perception
        )

        if hasattr(info_system, "information_cache"):
            info_system.information_cache.clear()
        if hasattr(info_system, "ai_analysis_cache"):
            info_system.ai_analysis_cache.clear()

        evals_premium, _ = agent_premium._evaluate_portfolio_opportunities(
            opportunities, info_system, market_conditions, "premium", perception
        )

        # Sort by final_score (Python's _evaluate_portfolio_opportunities preserves
        # input order; downstream _make_portfolio_decision does the actual sort).
        evals_none_sorted = sorted(evals_none, key=lambda e: e["final_score"], reverse=True)
        evals_premium_sorted = sorted(evals_premium, key=lambda e: e["final_score"], reverse=True)

        ids_none = [e["opportunity"].id for e in evals_none_sorted]
        ids_premium = [e["opportunity"].id for e in evals_premium_sorted]

        top_none = ids_none[:3]
        top_premium = ids_premium[:3]

        if top_none and top_premium and top_none[0] == top_premium[0]:
            top1_match += 1
        top3_overlap_sum += len(set(top_none) & set(top_premium))

        # Spearman-style rank correlation across full ranking
        ranks_none = {opp_id: i for i, opp_id in enumerate(ids_none)}
        ranks_premium = {opp_id: i for i, opp_id in enumerate(ids_premium)}
        common_ids = list(set(ranks_none) & set(ranks_premium))
        if len(common_ids) > 1:
            rn = [ranks_none[i] for i in common_ids]
            rp = [ranks_premium[i] for i in common_ids]
            if statistics.stdev(rn) > 0 and statistics.stdev(rp) > 0:
                rank_corr_sum += np.corrcoef(rn, rp)[0, 1]
                valid_corr_count += 1

    top1_match_rate = top1_match / N_TRIALS
    avg_top3_overlap = top3_overlap_sum / N_TRIALS
    avg_rank_corr = rank_corr_sum / max(1, valid_corr_count)

    print(f"\nTier-divergence regression test:")
    print(f"  top1_match_rate  = {top1_match_rate:.3f}")
    print(f"  avg_top3_overlap = {avg_top3_overlap:.3f} / 3.0")
    print(f"  avg_rank_corr    = {avg_rank_corr:.3f}")

    # Same assertion thresholds as the Julia regression test
    assert top1_match_rate <= 0.60, (
        f"top1_match_rate = {top1_match_rate:.3f} too high — "
        "AI tiers always pick the same top opportunity, mechanism may be bypassed"
    )
    assert avg_top3_overlap < 2.8, (
        f"avg_top3_overlap = {avg_top3_overlap:.3f}/3.0 too high — "
        "AI tiers nearly always agree on top-3"
    )
    assert avg_rank_corr < 0.95, (
        f"avg_rank_corr = {avg_rank_corr:.3f} too high — "
        "AI tiers produce nearly-identical full rankings"
    )
