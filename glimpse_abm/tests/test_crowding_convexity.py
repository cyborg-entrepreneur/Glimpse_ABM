"""v3.1 regression: capital-saturation convexity crowding (Python parity).

Mirrors julia/test/test_crowding_convexity.jl. The core property is that
the crowding penalty depends on capital saturation (total_invested /
capacity), NOT on the count of competitors (opp.competition). Count-based
dependence was the bug the refactor targeted — ten $10k investments
should not penalize returns the same as one $10M investment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
for candidate in (PARENT, ROOT):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from glimpse_abm.config import EmergentConfig
from glimpse_abm.models import Opportunity


def _market_conditions() -> dict:
    return {
        "regime": "normal",
        "volatility": 0.2,
        "regime_return_multiplier": 1.0,
        "regime_failure_multiplier": 1.0,
        "tier_invest_share": {},
        "sector_clearing_index": {},
        "aggregate_clearing_ratio": 1.0,
        "crowding_metrics": {"crowding_index": 0.25},
        "sector_demand_adjustments": {},
        "avg_competition": 0.0,
        "uncertainty_state": {},
    }


def _build_opp(cfg: EmergentConfig, sat_ratio: float, competition: float = 1.0) -> Opportunity:
    cap = 1.0e7
    opp = Opportunity(
        id=f"probe_{sat_ratio}_{competition}",
        latent_return_potential=2.0,
        latent_failure_potential=0.3,
        complexity=0.5,
        discovered=True,
        config=cfg,
        sector="tech",
        capacity=cap,
        total_invested=sat_ratio * cap,
    )
    opp.competition = competition
    return opp


def _draw(opp: Opportunity, mc: dict, n: int, seed: int) -> list[float]:
    np.random.seed(seed)
    return [opp.realized_return(mc, investor_tier="none") for _ in range(n)]


def test_capital_saturation_convexity() -> None:
    cfg = EmergentConfig(N_AGENTS=10, N_ROUNDS=1, RANDOM_SEED=42)
    mc = _market_conditions()
    N = 400

    # Property 1: Below K_sat, penalty ~zero; above K_sat, returns fall.
    low = _build_opp(cfg, 0.3)
    high = _build_opp(cfg, 2.5)
    low_returns = _draw(low, mc, N, seed=42)
    high_returns = _draw(high, mc, N, seed=42)
    assert np.mean(low_returns) > np.mean(high_returns)
    assert np.mean(high_returns) / np.mean(low_returns) < 0.85

    # Property 2: Count-invariance — same saturation, different competition
    # → statistically indistinguishable returns. THE point of v3.1.
    few = _build_opp(cfg, 1.0, competition=1.0)
    many = _build_opp(cfg, 1.0, competition=50.0)
    few_returns = _draw(few, mc, N, seed=42)
    many_returns = _draw(many, mc, N, seed=42)
    ratio = np.mean(many_returns) / np.mean(few_returns)
    assert 0.95 < ratio < 1.05, f"count-dependence leaked: ratio={ratio}"

    # Property 3: Monotone in saturation (mean falls as sat rises).
    sat_levels = [0.5, 1.0, 1.5, 2.0, 3.0]
    means = []
    for s in sat_levels:
        opp = _build_opp(cfg, s)
        means.append(float(np.mean(_draw(opp, mc, N, seed=42))))
    assert means[0] > means[-1]
    assert means[0] > means[2]  # 0.5 vs 1.5
