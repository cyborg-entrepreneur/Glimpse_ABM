"""
Simulation configuration dataclass and calibration utilities for Glimpse ABM.

This module provides the comprehensive configuration system for the agent-based
model, including all tunable parameters, calibration profiles, and validation
utilities. The configuration structure is designed to support:

1. **Reproducibility**: Every simulation run captures a complete configuration
   snapshot, enabling exact replication of results.

2. **Calibration**: Built-in profiles anchor simulations to empirical benchmarks
   including sector-specific parameters from NVCA 2024 (capital), BLS/Fed SBCS
   (survival), NSF BRDIS/USPTO (innovation), NBER (market regimes), and Census
   Bureau HHI (competition).

3. **Sensitivity Analysis**: Parameter ranges and override mechanisms support
   Latin Hypercube Sampling and grid-based sensitivity sweeps.

Theoretical Foundation
----------------------
Parameter values and ranges are informed by the theoretical framework from:

    Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).
    Are the futures computable? Knightian uncertainty & artificial intelligence.
    Academy of Management Review, 50(2), 415-440.

    Townsend, D. M., Hunt, R. A., & Rady, J. (2024). Chance, probability, &
    uncertainty at the edge of human reasoning: What is Knightian uncertainty?
    Strategic Entrepreneurship Journal, 18(3), 451-474.

Key design decisions:

- **Four AI Tiers**: Operationalize the heterogeneity of AI capabilities
  available to entrepreneurs, from no AI to premium enterprise solutions.

- **Trait Distributions**: Capture the psychological heterogeneity of
  entrepreneurs, particularly their tolerance for uncertainty (Knight, 1921).

- **Sector Profiles**: Model realistic risk-return characteristics across
  tech, retail, service, and manufacturing ventures.

- **Recursion Weights**: Control the intensity of competitive recursion
  effects, enabling investigation of AI-induced strategic interdependence.

Usage
-----
Basic configuration:

    >>> config = EmergentConfig()
    >>> config = config.copy_with_overrides({"N_AGENTS": 500})

With calibration profile:

    >>> from glimpse_abm.config import get_calibration_profile, apply_calibration_profile
    >>> profile = get_calibration_profile("venture_baseline_2024")
    >>> config = apply_calibration_profile(EmergentConfig(), profile)

See Also
--------
docs/PARAMETER_GLOSSARY.md : Comprehensive parameter documentation with
    theoretical justification and empirical calibration targets.

References
----------
Knight, F. H. (1921). Risk, uncertainty, and profit. Houghton Mifflin.

Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).
    Are the futures computable? Knightian uncertainty & artificial intelligence.
    Academy of Management Review, 50(2), 415-440.
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from numbers import Number


@dataclass
class EmergentConfig:
    """Enhanced configuration for emergent simulation with all AI learning features."""

    # Agent configuration
    # =========================================================================
    # Capital settings calibrated for ~50% ± 8% survival rate (42-58% target range)
    # at 200 rounds with 30 agents. Calibration performed 2025-01 against
    # BLS Business Employment Dynamics benchmarks for 5-year startup survival.
    # Key relationship: capital / operational_cost = months of runway
    # At $1.5M / $50k = 30 months runway → ~45% survival (within target)
    # =========================================================================
    AGENT_AI_MODE: str = "emergent"
    N_AGENTS: int = 1000
    INITIAL_CAPITAL: float = 5_000_000.0
    INITIAL_CAPITAL_RANGE: Tuple[float, float] = (2_500_000.0, 10_000_000.0)
    SURVIVAL_THRESHOLD: float = 230_000.0
    SURVIVAL_CAPITAL_RATIO: float = 0.38
    INSOLVENCY_GRACE_ROUNDS: int = 7
    RANDOM_SEED: int = 42

    BASE_OPERATIONAL_COST: float = 10000.0  # Monthly - calibrated for ~55% 5-yr survival (BLS benchmark)
    COMPETITION_COST_MULTIPLIER: float = 50.0  # Monthly (was 150 quarterly)
    OPERATING_RESERVE_MONTHS: int = 3
    MAX_AGENT_KNOWLEDGE: int = 90
    SECTOR_STRENGTH_PRUNE_THRESHOLD: float = 0.1
    LIQUIDITY_RESERVE_FRACTION: float = 0.29
    MAX_INVESTMENT_FRACTION: float = 0.037  # Monthly (was 0.11 quarterly)
    TARGET_INVESTMENT_FRACTION: float = 0.033  # Monthly (was 0.10 quarterly)
    AI_CREDIT_LINE_ROUNDS: int = 24  # 24 months = 2 years (typical seed funding runway)
    AI_TRUST_RESERVE_DISCOUNT: float = 0.25

    # Robustness test parameters for scaling AI failure modes and costs
    HALLUCINATION_INTENSITY: float = 1.0  # Scale AI hallucination rates (0.0=none, 1.0=baseline, 2.0=double)
    OVERCONFIDENCE_INTENSITY: float = 1.0  # Scale AI overconfidence effects (0.0=none, 1.0=baseline, 2.0=double)
    AI_COST_INTENSITY: float = 1.0  # Scale AI subscription/usage costs (0.0=free, 1.0=baseline, 2.0=double)

    # Network Configuration
    USE_NETWORK_EFFECTS: bool = True
    NETWORK_N_NEIGHBORS: int = 4
    NETWORK_REWIRING_PROB: float = 0.1

    # Agent trait distributions
    TRAIT_DISTRIBUTIONS: Dict = field(
        default_factory=lambda: {
            # Tech entrepreneurs skew toward higher tolerance for ambiguity; Beta(1.05, 0.65) centers around ~0.62 with a long optimistic tail.
            "uncertainty_tolerance": {"dist": "beta", "params": {"a": 1.05, "b": 0.65}},
            "innovativeness": {"dist": "lognormal", "params": {"mean": 0.5, "sigma": 0.5}},
            "competence": {"dist": "uniform", "params": {"low": 0.1, "high": 0.8}},
            "ai_trust": {"dist": "normal_clipped", "params": {"mean": 0.5, "std": 0.38}},
            "trait_momentum": {"dist": "uniform", "params": {"low": 0.6, "high": 0.9}},
            "cognitive_style": {"dist": "uniform", "params": {"low": 0.8, "high": 1.2}},
            "analytical_ability": {"dist": "uniform", "params": {"low": 0.1, "high": 0.9}},
            "exploration_tendency": {"dist": "beta", "params": {"a": 0.85, "b": 0.85}},
            "market_awareness": {"dist": "uniform", "params": {"low": 0.1, "high": 0.9}},
            "entrepreneurial_drive": {"dist": "beta", "params": {"a": 2.2, "b": 1.8}},
        }
    )

    # Market configuration (monthly cadence: 120 rounds = 10 years)
    N_ROUNDS: int = 120  # Monthly (was 250 quarterly)
    N_RUNS: int = 50
    BASE_OPPORTUNITIES: int = 5
    DISCOVERY_PROBABILITY: float = 0.10  # Monthly (was 0.30 quarterly)
    INNOVATION_PROBABILITY: float = 0.14  # Monthly (was 0.42 quarterly)
    AI_HERDING_DECAY: float = 1.0
    AI_SIGNAL_HISTORY: int = 420  # 420 months history (was 140 quarters)

    # Innovation economics (monthly cadence)
    INNOVATION_BASE_SPEND_RATIO: float = 0.025
    INNOVATION_MAX_SPEND: float = 2667.0  # Monthly (was 8000 quarterly)
    INNOVATION_FAIL_RECOVERY_RATIO: float = 0.12
    INNOVATION_SUCCESS_BASE_RETURN: float = 0.08  # Monthly (was 0.25 quarterly)
    INNOVATION_SUCCESS_RETURN_MULTIPLIER: Tuple[float, float] = (1.8, 3.2)
    INNOVATION_RD_CAP_FRACTION: float = 0.04  # Monthly (was 0.12 quarterly)
    INNOVATION_REUSE_PROBABILITY: float = 0.07  # Monthly (was 0.22 quarterly)
    INNOVATION_REUSE_LOOKBACK: int = 300  # 300 months lookback (was 100 quarters)
    INVESTMENT_SUCCESS_ROI_THRESHOLD: float = 0.017  # Monthly (was 0.05 quarterly)
    BURN_HISTORY_WINDOW: int = 9  # Monthly (was 3 quarterly)
    BURN_FAILURE_THRESHOLD: float = 0.12
    BURN_LEVERAGE_CAP: float = 0.75
    RETURN_OVERSUPPLY_PENALTY: float = 0.35  # Reduced from 0.52 for better survival
    RETURN_UNDERSUPPLY_BONUS: float = 0.37
    RETURN_NOISE_SCALE: float = 0.38
    BRANCH_LOG_MEAN_DRIFT: float = 0.11
    BRANCH_LOG_SIGMA_DRIFT: float = 0.07
    BRANCH_FAILURE_DRIFT: float = 0.04
    BRANCH_FEEDBACK_RATE: float = 0.05

    # Opportunity characteristics
    OPPORTUNITY_RETURN_RANGE: Tuple[float, float] = (1.1, 25.0)
    OPPORTUNITY_UNCERTAINTY_RANGE: Tuple[float, float] = (0.12, 0.60)
    OPPORTUNITY_COMPLEXITY_RANGE: Tuple[float, float] = (0.0, 2.0)

    # =========================================================================
    # AI TOOL CONFIGURATION - 2027 Scaling Law Projections
    # =========================================================================
    # Temporal Framework: Simulation represents 2028-2038 (post-AGI launch)
    # Scaling Law: none (human) → basic (consumer AI) → advanced (SOTA) → premium (AGI)
    #
    # Tier definitions:
    # - none: Human decision-making only (no AI augmentation)
    # - basic: Consumer-level AI tools (ChatGPT Plus, Claude Pro) ~$20-30/month
    # - advanced: Current SOTA systems (GPT-4, Claude Opus, o1) ~$400/month heavy use
    # - premium: Projected AGI systems (December 2027 launch) ~$3,500/month early access
    #
    # SCALING LAW CALIBRATION (Hoffmann et al. 2022, Kaplan et al. 2020)
    # ─────────────────────────────────────────────────────────────────────
    # Core formula: info_quality = 0.25 + 0.09 × log₁₀(effective_compute)
    #
    # Derivation:
    #   - Human baseline (no AI): effective_compute = 10^0 = 1
    #     → info_quality = 0.25 + 0.09 × 0 = 0.25
    #   - Basic (2024 GPT-4 commoditized): effective_compute ≈ 10^2
    #     → info_quality = 0.25 + 0.09 × 2 = 0.43
    #   - Advanced (2026 frontier): effective_compute ≈ 10^5
    #     → info_quality = 0.25 + 0.09 × 5 = 0.70
    #   - Premium (2027 frontier): effective_compute ≈ 10^8
    #     → info_quality = 0.25 + 0.09 × 8 = 0.97
    #
    # COST SCALING
    # ─────────────────────────────────────────────────────────────────────
    # Based on historical trends: ~10-20x efficiency improvement from 2024
    # Cost tracks roughly with sqrt(compute) after efficiency gains
    # Monthly subscription: basic=$30, advanced=$400, premium=$3500
    #
    # INFO_BREADTH CALIBRATION
    # ─────────────────────────────────────────────────────────────────────
    # Breadth scales slightly below quality (training data coverage vs. depth)
    # Formula: info_breadth ≈ info_quality - 0.05 (capped at 0.92)
    #
    AI_LEVELS: Dict = field(
        default_factory=lambda: {
            "none": {
                "cost": 0,
                "cost_type": "none",
                "info_quality": 0.25,       # Human baseline (log10(1) = 0)
                "info_breadth": 0.20,
                "per_use_cost": 0.0,
                "effective_compute_log10": 0,  # Reference: human analytical capacity
            },
            "basic": {
                "cost": 30,                 # Monthly subscription (commoditized 2024 tech)
                "cost_type": "per_use",
                "info_quality": 0.43,       # log10(10^2) = 2 -> 0.25 + 0.09*2 = 0.43
                "info_breadth": 0.38,
                "per_use_cost": 3.0,        # Very cheap per-query
                "effective_compute_log10": 2,
            },
            "advanced": {
                "cost": 400,                # Monthly subscription
                "cost_type": "subscription",
                "info_quality": 0.70,       # log10(10^5) = 5 -> 0.25 + 0.09*5 = 0.70
                "info_breadth": 0.65,
                "per_use_cost": 35.0,
                "effective_compute_log10": 5,
            },
            "premium": {
                "cost": 3500,               # Monthly subscription (AGI - Dec 2027 launch)
                "cost_type": "subscription",
                "info_quality": 0.97,       # log10(10^8) = 8 -> 0.25 + 0.09*8 = 0.97
                "info_breadth": 0.92,
                "per_use_cost": 150.0,
                "effective_compute_log10": 8,
            },
        }
    )

    # AI DOMAIN CAPABILITIES - Aligned with 2027 Scaling Law Projections
    # ─────────────────────────────────────────────────────────────────────
    # ACCURACY: Tracks info_quality with domain-specific modifiers (±0.03)
    #   market_analysis: slightly higher (structured data)
    #   technical_assessment: highest (objective criteria)
    #   uncertainty_evaluation: lower (inherently harder)
    #   innovation_potential: lowest (most speculative)
    #
    # HALLUCINATION RATE: Scales inversely with capability
    #   Formula: hallucination ≈ 0.30 × (1 - info_quality)
    #   none: ~0.225-0.30, basic: ~0.17-0.22, advanced: ~0.09-0.12, premium: ~0.01-0.02
    #
    # BIAS: Approaches zero with increased capability
    #   Formula: bias ≈ ±0.08 × (1 - info_quality)
    #   Positive bias = overestimation, negative = underestimation
    AI_DOMAIN_CAPABILITIES: Dict = field(
        default_factory=lambda: {
            "none": {  # Human baseline (info_quality=0.25)
                "market_analysis": {
                    "accuracy": 0.38,
                    "hallucination_rate": 0.28,
                    "bias": 0.06,
                },
                "technical_assessment": {
                    "accuracy": 0.40,
                    "hallucination_rate": 0.26,
                    "bias": -0.04,
                },
                "uncertainty_evaluation": {
                    "accuracy": 0.35,
                    "hallucination_rate": 0.30,
                    "bias": -0.05,
                },
                "innovation_potential": {
                    "accuracy": 0.33,
                    "hallucination_rate": 0.29,
                    "bias": 0.07,
                },
            },
            "basic": {  # 2024 GPT-4 commoditized (info_quality=0.43)
                "market_analysis": {
                    "accuracy": 0.52,
                    "hallucination_rate": 0.20,
                    "bias": 0.04,
                },
                "technical_assessment": {
                    "accuracy": 0.54,
                    "hallucination_rate": 0.19,
                    "bias": -0.03,
                },
                "uncertainty_evaluation": {
                    "accuracy": 0.50,
                    "hallucination_rate": 0.21,
                    "bias": -0.04,
                },
                "innovation_potential": {
                    "accuracy": 0.48,
                    "hallucination_rate": 0.22,
                    "bias": 0.05,
                },
            },
            "advanced": {  # 2026 frontier (info_quality=0.70)
                "market_analysis": {
                    "accuracy": 0.78,
                    "hallucination_rate": 0.10,
                    "bias": 0.025,
                },
                "technical_assessment": {
                    "accuracy": 0.80,
                    "hallucination_rate": 0.09,
                    "bias": -0.015,
                },
                "uncertainty_evaluation": {
                    "accuracy": 0.76,
                    "hallucination_rate": 0.11,
                    "bias": -0.02,
                },
                "innovation_potential": {
                    "accuracy": 0.74,
                    "hallucination_rate": 0.12,
                    "bias": 0.02,
                },
            },
            "premium": {  # 2027 frontier (info_quality=0.97)
                "market_analysis": {
                    "accuracy": 0.96,
                    "hallucination_rate": 0.015,
                    "bias": 0.003,
                },
                "technical_assessment": {
                    "accuracy": 0.97,
                    "hallucination_rate": 0.012,
                    "bias": -0.002,
                },
                "uncertainty_evaluation": {
                    "accuracy": 0.95,
                    "hallucination_rate": 0.018,
                    "bias": -0.003,
                },
                "innovation_potential": {
                    "accuracy": 0.94,
                    "hallucination_rate": 0.02,
                    "bias": 0.004,
                },
            },
        }
    )

    # Investment & Innovation Penalties (monthly cadence)
    UNCERTAINTY_LEARNING_ENABLED: bool = True
    INITIAL_RESPONSE_VARIANCE: float = 0.3
    LEARNING_RATE_BASE: float = 0.033  # Monthly (was 0.1 quarterly)
    EXPLORATION_DECAY: float = 0.9983  # Monthly equivalent of 0.995 quarterly
    SOCIAL_LEARNING_WEIGHT: float = 0.2
    MIN_FUNDING_FRACTION: float = 0.25
    COST_OF_CAPITAL: float = 0.005  # Monthly (was 0.06 quarterly, ~6% annual)

    # Exploration & maintaining Sensitivity
    EXPLORATION_SENSITIVITY = 1.5
    MAINTAIN_UNCERTAINTY_SENSITIVITY = 0.5

    # Uncertainty and Market Dynamics (monthly cadence)
    BLACK_SWAN_PROBABILITY: float = 0.017  # Monthly (was 0.05 quarterly)
    BOOM_TAIL_UNCERTAINTY_EXPONENT: float = 1.08
    MARKET_VOLATILITY: float = 0.15  # Monthly volatility (lower than quarterly)
    COMPETITION_SCALE_FACTOR: float = 1.0  # Multiplier for sector-specific competition_intensity (0.0=no competition, 1.0=baseline, 2.0=double)
    MARKET_SHIFT_PROBABILITY: float = 0.03  # Monthly (was 0.09 quarterly)
    MARKET_SHIFT_SEVERITY_RANGE: Tuple[float, float] = (0.25, 0.75)
    MARKET_SHIFT_MAX_SECTORS: int = 3
    MARKET_SHIFT_REGIME_MULTIPLIER: Dict[str, float] = field(default_factory=lambda: {
        "boom": 1.5,
        "growth": 1.2,
        "normal": 1.0,
        "volatile": 1.3,
        "recession": 1.6,
        "crisis": 2.2,
    })
    MACRO_REGIME_STATES: Tuple[str, ...] = ("crisis", "recession", "normal", "growth", "boom")
    # NBER Business Cycle-calibrated transition matrix (quarterly rounds)
    # Source: NBER Business Cycle Dating Committee data 1945-2024
    # Average expansion: 64 months, average recession: 11 months
    # Crisis frequency: ~1 per decade, boom frequency: ~2-3 per decade
    # Market regime transitions - calibrated for stability with mean-reversion to normal
    # Adjusted to reduce inter-run variance while maintaining business cycle dynamics
    # Monthly regime transitions (converted from quarterly using p_monthly = 1 - (1-p_quarterly)/3)
    MACRO_REGIME_TRANSITIONS: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "crisis": {"crisis": 0.75, "recession": 0.12, "normal": 0.12, "growth": 0.01, "boom": 0.0},
            "recession": {"crisis": 0.01, "recession": 0.77, "normal": 0.17, "growth": 0.04, "boom": 0.01},
            "normal": {"crisis": 0.003, "recession": 0.027, "normal": 0.85, "growth": 0.09, "boom": 0.03},
            "growth": {"crisis": 0.003, "recession": 0.01, "normal": 0.07, "growth": 0.84, "boom": 0.077},
            "boom": {"crisis": 0.003, "recession": 0.013, "normal": 0.067, "growth": 0.13, "boom": 0.787},
        }
    )
    # Regime modifiers - balanced for realistic business cycle dynamics
    MACRO_REGIME_RETURN_MODIFIERS: Dict[str, float] = field(
        default_factory=lambda: {"crisis": 0.85, "recession": 0.98, "normal": 1.10, "growth": 1.28, "boom": 1.45}
    )
    MACRO_REGIME_FAILURE_MODIFIERS: Dict[str, float] = field(
        default_factory=lambda: {"crisis": 1.18, "recession": 1.05, "normal": 1.0, "growth": 0.90, "boom": 0.78}
    )
    MACRO_REGIME_TREND: Dict[str, float] = field(
        default_factory=lambda: {"crisis": -0.8, "recession": -0.35, "normal": 0.0, "growth": 0.35, "boom": 0.6}
    )
    MACRO_REGIME_VOLATILITY: Dict[str, float] = field(
        default_factory=lambda: {"crisis": 0.55, "recession": 0.35, "normal": 0.2, "growth": 0.25, "boom": 0.3}
    )
    RETURN_DEMAND_CROWDING_THRESHOLD: float = 0.42
    RETURN_DEMAND_CROWDING_PENALTY: float = 0.48
    FAILURE_DEMAND_PRESSURE: float = 0.20
    # Recursion weight controls (for sensitivity/LHS)
    RECURSION_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "crowd_weight": 0.35,
        "volatility_weight": 0.30,
        "ai_herd_weight": 0.40,
        "premium_reuse_weight": 0.20,
    })
    # AI uplift on agentic novelty
    AI_NOVELTY_UPLIFT: float = 0.08

    # ========================================================================
    # REFUTATION TEST PARAMETERS
    # These allow testing model robustness by modifying AI tier advantages
    # ========================================================================

    # Execution success multipliers by AI tier
    # Default: all 1.0 (no AI advantage in execution)
    AI_EXECUTION_SUCCESS_MULTIPLIERS: Dict[str, float] = field(
        default_factory=lambda: {
            "none": 1.0,
            "basic": 1.0,
            "advanced": 1.0,
            "premium": 1.0
        }
    )

    # Innovation quality boost by AI tier (additive, on 0-1 scale)
    # Default: 0.05 for all AI tiers (small boost for AI assistance)
    AI_QUALITY_BOOST: Dict[str, float] = field(
        default_factory=lambda: {
            "none": 0.0,
            "basic": 0.05,
            "advanced": 0.05,
            "premium": 0.05
        }
    )

    # AI cost multiplier (for testing zero/reduced AI costs)
    # Default: 1.0 (full costs); 0.0 = free AI
    AI_COST_MULTIPLIER: float = 1.0

    # AI_NOVELTY_CONSTRAINT_INTENSITY: Scales the negative effect of premium
    # AI on agentic novelty (0.0 to 1.0). Tests whether AI-induced anchoring
    # on historical patterns drives the paradox.
    AI_NOVELTY_CONSTRAINT_INTENSITY: float = 1.0

    # COMPETITION_INTENSITY: Scales competition effects from 0.0 (no competition
    # penalties) to 1.0 (full). Tests whether market-level competition dynamics
    # drive the paradox.
    COMPETITION_INTENSITY: float = 1.0

    # ========================================================================
    # NEW MARKET MECHANICS PARAMETERS (from Julia implementation)
    # ========================================================================

    # ========================================================================
    # CAPACITY-CONVEXITY CROWDING MODEL
    # ========================================================================
    # New unified crowding penalty using capacity + convexity approach:
    #   penalty = λ · max(0, C/K - 1)^γ
    #   net_return = base_return · exp(-penalty)
    #
    # This replaces the old scattered linear penalties with a theoretically
    # grounded model where:
    #   - No penalty until crowding exceeds capacity (K)
    #   - Penalty increases convexly beyond capacity (γ controls sharpness)
    #   - Exponential decay keeps returns positive
    # ========================================================================
    USE_CAPACITY_CONVEXITY_CROWDING: bool = True  # Enable new crowding model

    # K = Carrying capacity: competition level where penalties START
    CROWDING_CAPACITY_K: float = 1.5

    # γ = Convexity exponent: how sharply penalties increase beyond capacity
    # γ = 2: quadratic (convex - "a little crowded is OK, very crowded is brutal")
    CROWDING_CONVEXITY_GAMMA: float = 2.0

    # λ = Strength: maps crowding into payoff reduction
    # exp(-λ) gives the return multiplier at 2× capacity
    CROWDING_STRENGTH_LAMBDA: float = 0.50

    # Legacy parameters (used when USE_CAPACITY_CONVEXITY_CROWDING = False)
    DISABLE_COMPETITION_DYNAMICS: bool = False  # Master switch to disable competition effects
    OPPORTUNITY_COMPETITION_PENALTY: float = 0.5  # Return penalty from crowded opportunities
    OPPORTUNITY_COMPETITION_THRESHOLD: float = 0.2  # Crowding level at which penalties begin
    OPPORTUNITY_COMPETITION_FLOOR: float = 0.1  # Minimum return multiplier under max competition

    # Novelty disruption mechanics
    NOVELTY_DISRUPTION_ENABLED: bool = True  # Enable innovation disruption effects
    NOVELTY_DISRUPTION_THRESHOLD: float = 0.6  # Novelty level to trigger disruption
    NOVELTY_DISRUPTION_MAGNITUDE: float = 0.25  # Disruption impact on existing investments
    DISRUPTION_COMPETITION_THRESHOLD: float = 10.0  # Competition level triggering disruption vulnerability
    NOVELTY_NOISE_INVERSION_FACTOR: float = 0.4  # Noise scaling for novel opportunities

    # Opportunity capacity constraints
    OPPORTUNITY_CAPACITY_ENABLED: bool = True  # Enable capacity-based crowding
    OPPORTUNITY_BASE_CAPACITY: float = 500000.0  # Base investment capacity per opportunity
    OPPORTUNITY_CAPACITY_VARIANCE: float = 0.3  # Variance in opportunity capacities
    CAPACITY_PENALTY_START: float = 0.7  # Utilization level at which crowding penalty begins
    CAPACITY_PENALTY_MAX: float = 0.4  # Maximum penalty at full capacity

    # Sequential decision mechanics (early vs late movers)
    SEQUENTIAL_DECISIONS_ENABLED: bool = True  # Enable sequential decision rounds
    EARLY_DECISION_FRACTION: float = 0.3  # Fraction of agents making early decisions
    SIGNAL_VISIBILITY_WEIGHT: float = 0.15  # Weight of early signals for late movers

    # Downside oversupply weight in realized returns
    DOWNSIDE_OVERSUPPLY_WEIGHT: float = 0.45  # Balanced penalty for oversupply conditions
    # Lower bound on realized return multiple (0.0 = total loss, 0.5 = 50% back, 1.0 = break-even)
    RETURN_LOWER_BOUND: float = 0.0  # Minimum return multiple

    # Scaling parameters
    OPPORTUNITIES_PER_CAPITA: float = 0.01
    DISCOVERY_RATE_SCALING: float = 0.5
    MIN_OPPORTUNITIES: int = 5
    OPPORTUNITY_CAPITAL_REQUIREMENTS: float = 10000

    # Power law return distribution
    # Venture returns follow a power law (Pareto) distribution where most
    # investments return near the minimum while rare "unicorns" drive returns.
    #
    # POWER_LAW_SHAPE_A (α) controls tail heaviness:
    #   α = 2.0: Heavy tails (early-stage VC dynamics) - CALIBRATED DEFAULT
    #   α = 2.5: Moderate tails (growth equity)
    #   α = 3.0: Lighter tails (late-stage/buyout)
    #
    # Empirical VC benchmarks (Kaplan & Schoar, Korteweg & Sorensen):
    #   - ~65% of investments return <1× (losses)
    #   - ~25% return 1-3× (modest wins)
    #   - ~10% return 3×+ (winners that drive portfolio returns)
    #   - Top 1% can return 10-100×+ (unicorns)
    #
    # CALIBRATION NOTE (2026-01, n=10 runs):
    # α = 3.0 produces more consistent outcomes with lighter tails.
    # Less extreme than pure VC benchmarks (65% losses) because agents are
    # individual firms, not diversified portfolio managers.
    POWER_LAW_SHAPE_A: float = 3.0  # Lighter tails for more consistent outcomes

    # Learning parameters
    LEARNING_RATE: float = 0.05  # Learning rate for uncertainty response adaptation

    # Diagnostics
    ENABLE_DEBUG_LOGS: bool = False
    AI_TIER_REQUIREMENTS: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "basic": {
                "trust": 0.40,
                "min_successes": 0.0,
                "min_success_rate": 0.52,
                "min_roi": 0.95,
                "min_accuracy": 0.0,
                "max_cost_ratio": 0.08,
            },
            "advanced": {
                "trust": 0.58,
                "min_successes": 2.0,
                "min_success_rate": 0.54,
                "recent_min_success_rate": 0.52,
                "min_roi": 1.02,
                "recent_min_roi": 1.0,
                "min_accuracy": 0.50,
                "max_cost_ratio": 0.12,
            },
            "premium": {
                "trust": 0.68,
                "min_successes": 4.0,
                "min_success_rate": 0.60,
                "recent_min_success_rate": 0.58,
                "min_roi": 1.05,
                "recent_min_roi": 1.02,
                "min_accuracy": 0.55,
                "max_cost_ratio": 0.14,
            },
        }
    )
    AI_TIER_RECENT_WINDOW: int = 45  # Monthly (was 15 quarterly)
    AI_TIER_DEMOTE_MARGIN: float = 0.06
    AI_TIER_NEIGHBOR_INFLUENCE: float = 0.06
    AI_TIER_SCORING: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "basic": {
                "score_threshold": 0.0,
                "score_slope": 6.0,
                "trial_rounds": 12,  # Monthly (was 4 quarterly)
                "weights": {
                    "trust": 0.6,
                    "success": 0.3,
                    "success_gain": 0.2,
                    "recent_success_gain": 0.2,
                    "roi_gain": 0.15,
                    "recent_roi_gain": 0.15,
                    "cost": 0.25,
                    "accuracy": 0.10,
                    "usage": 0.08,
                    "bias": -0.25,
                },
            },
            "advanced": {
                "score_threshold": 0.08,
                "score_slope": 6.0,
                "trial_rounds": 12,  # Monthly (was 4 quarterly)
                "weights": {
                    "trust": 0.65,
                    "success": 0.32,
                    "success_gain": 0.27,
                    "recent_success_gain": 0.27,
                    "roi_gain": 0.22,
                    "recent_roi_gain": 0.22,
                    "cost": 0.18,
                    "accuracy": 0.15,
                    "usage": 0.12,
                    "bias": -0.28,
                },
            },
            "premium": {
                "score_threshold": 0.18,
                "score_slope": 6.5,
                "trial_rounds": 15,  # Monthly (was 5 quarterly)
                "weights": {
                    "trust": 0.70,
                    "success": 0.38,
                    "success_gain": 0.32,
                    "recent_success_gain": 0.32,
                    "roi_gain": 0.28,
                    "recent_roi_gain": 0.28,
                    "cost": 0.22,
                    "accuracy": 0.20,
                    "usage": 0.14,
                    "bias": -0.35,
                },
            },
        }
    )
    AI_TIER_SCORE_SMOOTHING: float = 0.25
    AI_TIER_DEMOTION_COOLDOWN: int = 36  # 36 months (was 12 quarters)
    TRAIT_MOMENTUM: float = 0.7
    AI_TRUST_ADJUSTMENT_RATE: float = 0.033  # Monthly adjustment (was 0.1 quarterly)
    AI_SUBSCRIPTION_AMORTIZATION_ROUNDS: int = 180  # Monthly (was 60 quarterly)
    AI_SUBSCRIPTION_FLOAT_BASE_ROUNDS: int = 0
    AI_SUBSCRIPTION_FLOAT_MAX_ROUNDS: int = 9  # 9 months (was 3 quarters)

    # Action selection controls
    ACTION_SELECTION_TEMPERATURE: float = 0.45
    ACTION_SELECTION_NOISE: float = 0.10
    ACTION_BIAS_SIGMA: float = 0.05

    # Base weights for action selection (multiplied against raw utility scores)
    # Calibrated 2025-01 to achieve:
    #   - Innovation share: ~41% (target: 40% ± 10%)
    #   - Survival rate: ~55% (target: 50% ± 8%)
    # Applied as multipliers in _choose_action_type() before softmax selection
    ACTION_BASE_WEIGHT_INVEST: float = 0.25
    ACTION_BASE_WEIGHT_INNOVATE: float = 0.55
    ACTION_BASE_WEIGHT_EXPLORE: float = 0.12
    ACTION_BASE_WEIGHT_MAINTAIN: float = 0.20

    # Uncertainty volatility controls (monthly cadence)
    UNCERTAINTY_SHORT_WINDOW: int = 18  # Monthly (was 6 quarterly)
    UNCERTAINTY_SHORT_DECAY: float = 0.0
    UNCERTAINTY_VOLATILITY_WINDOW: int = 42  # Monthly (was 14 quarterly)
    UNCERTAINTY_VOLATILITY_DECAY: float = 0.87  # Monthly equivalent (0.6^(1/3))
    UNCERTAINTY_VOLATILITY_SCALING: float = 0.45
    UNCERTAINTY_AI_SWITCH_WEIGHT: float = 0.09
    UNCERTAINTY_MARKET_RETURN_WEIGHT: float = 0.14
    UNCERTAINTY_ACTION_VARIANCE_WEIGHT: float = 0.14
    UNCERTAINTY_CROWDING_WEIGHT: float = 0.18
    UNCERTAINTY_COMPETITIVE_WEIGHT: float = 0.12

    # Knowledge & Innovation
    KNOWLEDGE_DECAY_RATE: float = 0.025  # Monthly decay (was 0.075 quarterly)
    SECTOR_KNOWLEDGE_PERSISTENCE: Dict[str, float] = field(
        default_factory=lambda: {
            "tech": 0.85,
            "retail": 0.92,
            "service": 0.95,
            "manufacturing": 0.97,
        }
    )
    SECTORS: List[str] = field(default_factory=list, init=False)

    # NVCA 2024-calibrated sector weights for agent initial sector assignment
    # Reflects dominant VC deal flow by sector
    SECTOR_WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {
            "tech": 0.60,           # 60% - dominant VC deal flow
            "service": 0.15,        # 15% - B2B services
            "manufacturing": 0.15,  # 15% - hardware/industrial
            "retail": 0.10,         # 10% - consumer/retail
        }
    )

    # Sector profiles with empirically-calibrated parameters
    # Sources: NVCA 2024, BLS BED, Fed SBCS, NSF BRDIS, USPTO, Census HHI
    SECTOR_PROFILES: Dict = field(
        default_factory=lambda: {
            "tech": {
                # Later-stage venture-backed software/hardware with differentiated upside
                "return_range": (1.60, 4.00),
                "return_log_mu": math.log(2.40),
                "return_log_sigma": 0.45,
                "return_volatility_range": (0.22, 0.38),
                "failure_range": (0.3, 0.5),
                "failure_volatility_range": (0.04, 0.12),
                "capital_range": (300000, 1200000),
                "maturity_range": (8, 20),  # Shortened for cash flow sustainability
                "gross_margin_range": (0.55, 0.85),
                "operating_margin_range": (0.08, 0.28),
                # Monthly operational costs (SBA/BLS calibrated 2025)
                # Tech: moderate employee costs, low physical overhead
                # BLS QCEW: Information sector converted to monthly
                "operational_cost_range": (20_000.0, 30_000.0),  # Monthly (was 60-90k quarterly)
                # Empirically-calibrated fields (scaled for 40-60 round runway)
                # Reflects Series A/B rounds with 24-36 month runway before profitability
                "initial_capital_range": (3_000_000.0, 6_000_000.0),  # 40-80 rounds runway
                "survival_threshold": 150_000.0,   # ~2 months operating expenses
                "survival_equity_ratio": 0.38,     # Tech: High burn tolerance (BLS/NVCA calibrated)
                "innovation_probability": 0.16,    # Monthly (was 0.48 quarterly)
                "innovation_return_multiplier": (2.0, 4.0),  # High tech upside
                "knowledge_decay_rate": 0.04,      # Monthly (was 0.12 quarterly)
                "competition_intensity": 1.2,      # HHI 1500-2500 moderate concentration
            },
            "retail": {
                # Multi-unit retail concepts with moderate upside and higher churn
                "return_range": (1.40, 2.80),
                "return_log_mu": math.log(1.85),
                "return_log_sigma": 0.32,
                "return_volatility_range": (0.18, 0.3),
                "failure_range": (0.2, 0.38),  # BLS: Retail failure 20-38% (5-year rate)
                "failure_volatility_range": (0.04, 0.1),
                "capital_range": (50000, 400000),
                "maturity_range": (4, 15),   # Empirical: 1-4 year investment maturity for retail
                "gross_margin_range": (0.18, 0.42),
                "operating_margin_range": (0.015, 0.08),
                # Monthly operational costs (BLS QCEW: Retail converted to monthly)
                "operational_cost_range": (13_000.0, 23_000.0),  # Monthly (was 40-70k quarterly)
                # Empirically-calibrated fields (Seed/Series A rounds with 18-24 month runway)
                "initial_capital_range": (2_200_000.0, 4_000_000.0),
                "survival_threshold": 130_000.0,   # ~3 months operating expenses
                "survival_equity_ratio": 0.38,     # Retail: Moderate burn (30-40% year-1; 12-month reserve requirement)
                "innovation_probability": 0.11,    # Monthly (was 0.32 quarterly)
                "innovation_return_multiplier": (1.6, 2.5),  # Moderate returns
                "knowledge_decay_rate": 0.023,     # Monthly (was 0.07 quarterly)
                "competition_intensity": 0.7,      # HHI 500-1000 fragmented
            },
            "service": {
                # B2B/B2C recurring service ventures with low capex and resilient margins
                "return_range": (1.50, 3.00),
                "return_log_mu": math.log(1.95),
                "return_log_sigma": 0.36,
                "return_volatility_range": (0.16, 0.28),
                "failure_range": (0.1, 0.28),  # BLS: Service failure 10-28% (5-year rate)
                "failure_volatility_range": (0.03, 0.08),
                "capital_range": (15000, 200000),
                "maturity_range": (3, 10),   # Empirical: 9mo-2.5yr investment maturity for services
                "gross_margin_range": (0.45, 0.75),
                "operating_margin_range": (0.12, 0.24),
                # Monthly operational costs (BLS QCEW: Services converted to monthly)
                "operational_cost_range": (8_300.0, 15_000.0),  # Monthly (was 25-45k quarterly)
                # Empirically-calibrated fields (Seed/Series A rounds with 16-28 month runway)
                "initial_capital_range": (1_400_000.0, 2_500_000.0),
                "survival_threshold": 70_000.0,    # ~2 months operating expenses
                "survival_equity_ratio": 0.52,     # Service: Capital efficient (higher retention expected; 80% fail if <20% at year 2)
                "innovation_probability": 0.13,    # Monthly (was 0.38 quarterly)
                "innovation_return_multiplier": (1.6, 2.5),  # Moderate returns
                "knowledge_decay_rate": 0.017,     # Monthly (was 0.05 quarterly)
                "competition_intensity": 0.9,      # HHI 800-1500 moderately fragmented
            },
            "manufacturing": {
                # Advanced manufacturing / industrial ventures with heavier capital loads
                "return_range": (1.60, 3.50),
                "return_log_mu": math.log(2.20),
                "return_log_sigma": 0.4,
                "return_volatility_range": (0.18, 0.3),
                "failure_range": (0.25, 0.42),  # BLS: Manufacturing failure 25-42% (5-year rate)
                "failure_volatility_range": (0.04, 0.1),
                "capital_range": (250000, 1500000),
                "maturity_range": (10, 28),  # Empirical: 2.5-7yr investment maturity for manufacturing
                "gross_margin_range": (0.28, 0.48),
                "operating_margin_range": (0.04, 0.18),
                # Monthly operational costs (BLS QCEW: Manufacturing converted to monthly)
                "operational_cost_range": (26_700.0, 40_000.0),  # Monthly (was 80-120k quarterly)
                # Empirically-calibrated fields (Series A/B rounds with 20-30 month runway)
                "initial_capital_range": (4_000_000.0, 7_500_000.0),
                "survival_threshold": 200_000.0,   # ~2 months operating expenses
                "survival_equity_ratio": 0.58,     # Manufacturing: Asset-heavy (capital preservation critical; 43.6% 10-year survival best)
                "innovation_probability": 0.17,    # Monthly (was 0.52 quarterly)
                "innovation_return_multiplier": (1.5, 2.8),  # Incremental improvements
                "knowledge_decay_rate": 0.01,      # Monthly (was 0.03 quarterly)
                "competition_intensity": 1.4,      # HHI 1800-3000 concentrated
            },
        }
    )

    calibration_targets: Dict[str, Any] = field(default_factory=dict)
    active_calibration: Optional[str] = None

    buffer_flush_interval: int = 5
    write_intermediate_batches: bool = True
    round_log_interval: int = 25
    enable_round_logging: bool = True
    max_cache_size: int = 100000
    agent_history_depth: int = 5
    preallocate_arrays: bool = True
    use_float32: bool = True
    use_parallel: bool = True
    parallel_threshold: int = 120
    parallel_chunk_size: int = 50
    # Agent-level thread workers (within a single simulation run)
    max_workers: int = (
        min(8, os.cpu_count() - 1) if os.cpu_count() and os.cpu_count() > 1 else 1
    )
    # Run-level process parallelism settings
    # PARALLEL_MODE: "max" uses CPUs-1 without workload caps; "safe" uses conservative workload-based caps
    PARALLEL_MODE: str = "max"
    # MAX_PARALLEL_RUNS: explicit cap on concurrent simulation processes (0 = use PARALLEL_MODE logic)
    MAX_PARALLEL_RUNS: int = 0

    def __post_init__(self) -> None:
        self.SECTORS = list(self.SECTOR_PROFILES.keys())

        if not hasattr(self, "STRATEGIC_MODE_THRESHOLDS"):
            self.STRATEGIC_MODE_THRESHOLDS = {
                "success_exploit": 0.7,
                "success_diversify": 0.3,
                "volatility_diversify": 0.4,
            }
        if self.N_AGENTS > 5000:
            self.buffer_flush_interval = 50
            self.max_cache_size = 100000
            self.agent_history_depth = 5
            self.preallocate_arrays = True
            self.use_float32 = True
    def get_scaled_opportunities(self, n_agents: int) -> int:
        """Compute integer opportunity targets even if overrides supply floats."""
        min_ops = int(math.ceil(float(self.MIN_OPPORTUNITIES)))
        per_capita = float(max(self.OPPORTUNITIES_PER_CAPITA, 0.0))
        scaled = int(math.ceil(n_agents * per_capita))
        base_ops = int(math.ceil(float(getattr(self, "BASE_OPPORTUNITIES", min_ops) or min_ops)))
        return max(min_ops, base_ops, scaled)

    def get_ai_domain_capability(self, ai_level: str, domain: str) -> Dict:
        if (
            ai_level in self.AI_DOMAIN_CAPABILITIES
            and domain in self.AI_DOMAIN_CAPABILITIES[ai_level]
        ):
            return self.AI_DOMAIN_CAPABILITIES[ai_level][domain]
        return {"accuracy": 0.5, "hallucination_rate": 0.1, "bias": 0.0}

    def snapshot(self) -> Dict[str, Any]:
        """Return a deep-copied, JSON-safe representation of the configuration."""
        return asdict(self)

    def copy_with_overrides(self, overrides: Optional[Dict[str, Any]] = None) -> "EmergentConfig":
        """Return a new config with the provided overrides merged in."""
        new_cfg = copy.deepcopy(self)
        if overrides:
            _apply_overrides(new_cfg, overrides)
        # Re-sync derived attributes after overrides are applied
        new_cfg.SECTORS = list(new_cfg.SECTOR_PROFILES.keys())
        return new_cfg


def _apply_overrides(config: EmergentConfig, overrides: Dict[str, Any]) -> None:
    """Recursively merge ``overrides`` into ``config``.

    Note: Certain keys are treated as full replacements rather than merges to support
    scenarios like reducing sectors from 4 to 1 (SECTOR_PROFILES) or completely
    redefining market regimes (MACRO_REGIME_TRANSITIONS).
    """
    # Keys that should be fully replaced rather than deep-merged
    REPLACE_KEYS = {"SECTOR_PROFILES", "MACRO_REGIME_TRANSITIONS", "TRAIT_DISTRIBUTIONS"}

    for key, value in overrides.items():
        # Support dotted notation for nested dict updates (e.g., "RECURSION_WEIGHTS.crowd_weight")
        if "." in key:
            top, *rest = key.split(".")
            if not hasattr(config, top):
                raise KeyError(f"Unknown configuration attribute '{top}' in calibration override.")
            current = getattr(config, top)
            if not isinstance(current, dict):
                raise KeyError(f"Attribute '{top}' is not a dictionary; cannot set '{key}'.")
            ref = current
            for part in rest[:-1]:
                if part not in ref or not isinstance(ref[part], dict):
                    ref[part] = {}
                ref = ref[part]
            ref[rest[-1]] = copy.deepcopy(value)
            setattr(config, top, current)
            continue
        if not hasattr(config, key):
            raise KeyError(f"Unknown configuration attribute '{key}' in calibration override.")
        current = getattr(config, key)
        # For designated keys, replace entirely rather than merge
        if key in REPLACE_KEYS:
            setattr(config, key, copy.deepcopy(value))
        elif isinstance(current, dict) and isinstance(value, dict):
            merged = _deep_merge_dict(current, value)
            setattr(config, key, merged)
        elif isinstance(current, tuple):
            coerced = _coerce_tuple(value, current)
            setattr(config, key, coerced)
        else:
            setattr(config, key, copy.deepcopy(value))


def _deep_merge_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries without mutating the originals."""
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _coerce_tuple(value: Any, template: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Ensure overrides targeting tuple parameters maintain tuple semantics."""
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    length = len(template) or 1
    if isinstance(value, Number):
        return tuple(value for _ in range(length))
    # Fallback: broadcast arbitrary objects to expected length
    return tuple(copy.deepcopy(value) for _ in range(length))


@dataclass(frozen=True)
class CalibrationProfile:
    """Reusable parameter bundle with empirical targets for reproducibility."""

    name: str
    description: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    target_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    source: str = "built-in"

    def to_metadata(self) -> Dict[str, Any]:
        """Return a serializable summary for run artefacts."""
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "overrides": copy.deepcopy(self.overrides),
            "target_metrics": copy.deepcopy(self.target_metrics),
        }


def apply_calibration_profile(config: EmergentConfig, profile: Optional[CalibrationProfile]) -> EmergentConfig:
    """Return a config with the calibration overrides applied."""
    if profile is None:
        return config
    updated = config.copy_with_overrides(profile.overrides)
    updated.active_calibration = profile.name
    updated.calibration_targets = copy.deepcopy(profile.target_metrics)
    return updated


def list_calibration_profiles() -> List[CalibrationProfile]:
    """Return the available built-in calibration profiles."""
    return list(CALIBRATION_LIBRARY.values())


def get_calibration_profile(name: str) -> CalibrationProfile:
    """Fetch a built-in calibration profile by name (case-insensitive)."""
    normalized = name.strip().lower()
    for profile in CALIBRATION_LIBRARY.values():
        if profile.name.lower() == normalized:
            return profile
    raise KeyError(f"Unknown calibration profile '{name}'. Available: {', '.join(CALIBRATION_LIBRARY.keys())}")


def load_calibration_profile(path: str | os.PathLike[str]) -> CalibrationProfile:
    """Load a calibration profile definition from disk."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    overrides = payload.get("overrides") or payload.get("parameters") or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"Calibration file {file_path} must define an 'overrides' dictionary.")
    target_metrics = payload.get("target_metrics", {})
    if target_metrics and not isinstance(target_metrics, dict):
        raise ValueError(f"'target_metrics' must be a dictionary in {file_path}.")
    name = payload.get("name") or file_path.stem
    description = payload.get("description", f"Custom calibration loaded from {file_path.name}")
    source = payload.get("source", str(file_path))
    return CalibrationProfile(
        name=name,
        description=description,
        overrides=overrides,
        target_metrics=target_metrics if isinstance(target_metrics, dict) else {},
        source=source,
    )


CALIBRATION_LIBRARY: Dict[str, CalibrationProfile] = {
    "minimal_causal": CalibrationProfile(
        name="minimal_causal",
        description=(
            "Minimal model for causal identification. Strips simulation to ~25 essential "
            "parameters to isolate AI tier effects on uncertainty dimensions. Use with "
            "--fixed-tier-sweep for cleanest causal interpretation. Disables network effects, "
            "simplifies trait distributions, and reduces stochastic noise to strengthen "
            "signal detection."
        ),
        overrides={
            # Core simulation parameters (5)
            "N_AGENTS": 500,
            "N_ROUNDS": 200,
            "N_RUNS": 30,
            "INITIAL_CAPITAL": 5_000_000.0,
            "SURVIVAL_CAPITAL_RATIO": 0.40,
            # Disable network effects to remove social contagion confound
            "USE_NETWORK_EFFECTS": False,
            # Simplified trait distributions - reduce heterogeneity to isolate AI effects
            "TRAIT_DISTRIBUTIONS": {
                "uncertainty_tolerance": {"dist": "uniform", "params": {"low": 0.3, "high": 0.7}},
                "innovativeness": {"dist": "uniform", "params": {"low": 0.3, "high": 0.7}},
                "competence": {"dist": "uniform", "params": {"low": 0.3, "high": 0.7}},
                "ai_trust": {"dist": "uniform", "params": {"low": 0.4, "high": 0.6}},
                "trait_momentum": {"dist": "uniform", "params": {"low": 0.7, "high": 0.8}},
                "cognitive_style": {"dist": "uniform", "params": {"low": 0.9, "high": 1.1}},
                "analytical_ability": {"dist": "uniform", "params": {"low": 0.4, "high": 0.6}},
                "exploration_tendency": {"dist": "uniform", "params": {"low": 0.4, "high": 0.6}},
                "market_awareness": {"dist": "uniform", "params": {"low": 0.4, "high": 0.6}},
                "entrepreneurial_drive": {"dist": "uniform", "params": {"low": 0.4, "high": 0.6}},
            },
            # Fix market regime to normal - removes regime confound
            "MACRO_REGIME_TRANSITIONS": {
                "crisis": {"crisis": 0.0, "recession": 0.0, "normal": 1.0, "growth": 0.0, "boom": 0.0},
                "recession": {"crisis": 0.0, "recession": 0.0, "normal": 1.0, "growth": 0.0, "boom": 0.0},
                "normal": {"crisis": 0.0, "recession": 0.0, "normal": 1.0, "growth": 0.0, "boom": 0.0},
                "growth": {"crisis": 0.0, "recession": 0.0, "normal": 1.0, "growth": 0.0, "boom": 0.0},
                "boom": {"crisis": 0.0, "recession": 0.0, "normal": 1.0, "growth": 0.0, "boom": 0.0},
            },
            # Reduce stochastic noise for cleaner signal
            "RETURN_NOISE_SCALE": 0.20,
            "BLACK_SWAN_PROBABILITY": 0.0,
            "MARKET_SHIFT_PROBABILITY": 0.0,
            # Clear recursion weights for interpretability
            "RECURSION_WEIGHTS": {
                "crowd_weight": 0.30,
                "volatility_weight": 0.30,
                "ai_herd_weight": 0.30,
                "premium_reuse_weight": 0.10,
            },
            # Disable knowledge decay to isolate AI information effects
            "KNOWLEDGE_DECAY_RATE": 0.0,
            # Simplified sector - single sector removes sector confound
            "SECTOR_PROFILES": {
                "tech": {
                    "return_range": (1.2, 2.5),
                    "return_log_mu": 0.5,
                    "return_log_sigma": 0.3,
                    "return_volatility_range": (0.2, 0.3),
                    "failure_range": (0.25, 0.40),
                    "failure_volatility_range": (0.05, 0.10),
                    "capital_range": (200000, 800000),
                    "maturity_range": (12, 24),
                    "gross_margin_range": (0.5, 0.7),
                    "operating_margin_range": (0.1, 0.2),
                },
            },
        },
        target_metrics={
            "causal_identification": {
                "description": "Minimal model for isolating AI tier causal effects",
                "fixed_tier_required": True,
                "source": "Internal validation - compare with full model for robustness",
            },
        },
    ),
    "venture_baseline_2024": CalibrationProfile(
        name="venture_baseline_2024",
        description=(
            "Anchors the simulation to US venture benchmarks: ~55% five-year survival, "
            "innovation share near 0.4, and modest 1.1x cash-on-cash investment multiples."
        ),
        overrides={
            "BASE_OPERATIONAL_COST": 70000.0,
            "SURVIVAL_CAPITAL_RATIO": 0.40,  # Updated for realistic survival rates
            "INSOLVENCY_GRACE_ROUNDS": 6,
            "DISCOVERY_PROBABILITY": 0.22,
            "INNOVATION_PROBABILITY": 0.37,
            "OPPORTUNITY_RETURN_RANGE": (0.85, 5.5),
            "INNOVATION_SUCCESS_RETURN_MULTIPLIER": (1.15, 2.4),
            "INVESTMENT_SUCCESS_ROI_THRESHOLD": 0.12,
            "MAX_INVESTMENT_FRACTION": 0.12,
            "AI_LEVELS": {
                "basic": {"cost": 250.0, "cost_type": "subscription", "per_use_cost": 0.0},
                "advanced": {"cost": 1400.0, "cost_type": "subscription", "per_use_cost": 0.0},
                "premium": {"cost": 15500.0, "cost_type": "subscription", "per_use_cost": 220.0},
            },
        },
        target_metrics={
            "survival_rate_round250": {
                "target": 0.55,
                "tolerance": 0.08,
                "source": "BLS Business Employment Dynamics (2019 cohort).",
            },
            "mean_investment_roi": {
                "target": 1.12,
                "tolerance": 0.2,
                "source": "PitchBook cash-on-cash outcomes for Series A-D (2020-2022).",
            },
            "innovation_share": {
                "target": 0.4,
                "tolerance": 0.1,
                "source": "NVCA 2024 innovation activity share estimates.",
            },
        },
    ),
    "deeptech_capital_constrained": CalibrationProfile(
        name="deeptech_capital_constrained",
        description=(
            "Represents capital-intensive deep-tech ecosystems where survival drops to ~35% "
            "but successful investments target 1.4-1.8x cash multiples."
        ),
        overrides={
            "BASE_OPERATIONAL_COST": 90000.0,
            "SURVIVAL_THRESHOLD": 400000.0,
            "SURVIVAL_CAPITAL_RATIO": 0.62,
            "OPERATING_RESERVE_MONTHS": 5,
            "MAX_INVESTMENT_FRACTION": 0.18,
            "TARGET_INVESTMENT_FRACTION": 0.12,
            "OPPORTUNITY_RETURN_RANGE": (0.95, 7.0),
            "INNOVATION_RD_CAP_FRACTION": 0.22,
            "INNOVATION_SUCCESS_RETURN_MULTIPLIER": (1.35, 3.1),
            "DISCOVERY_PROBABILITY": 0.18,
            "AI_LEVELS": {
                "basic": {"cost": 600.0, "cost_type": "subscription", "per_use_cost": 10.0},
                "advanced": {"cost": 2200.0, "per_use_cost": 12.0},
                "premium": {"cost": 20000.0, "per_use_cost": 400.0},
            },
        },
        target_metrics={
            "survival_rate_round250": {
                "target": 0.35,
                "tolerance": 0.06,
                "source": "NVCA deep-tech follow-on survival analyses (2017-2023).",
            },
            "mean_investment_roi": {
                "target": 1.48,
                "tolerance": 0.25,
                "source": "Corporate venture benchmarks for frontier tech (CB Insights, 2023).",
            },
            "innovation_share": {
                "target": 0.32,
                "tolerance": 0.08,
                "source": "OECD deep-tech investment mix 2022.",
            },
        },
    ),
    # Default calibration profile - documents the 50% survival target calibration
    # Calibration: 2025-01, 20 runs × 30 agents × 200 rounds
    # Result: 47.2% ± 39.8% mean survival (within target)
    # Note: High variance is inherent to stochastic startup survival model
    "default_50pct_survival": CalibrationProfile(
        name="default_50pct_survival",
        description=(
            "Default calibration targeting 50% ± 8% survival rate at 200 rounds. "
            "Calibrated 2025-01 using 20 runs × 30 agents × 200 rounds. "
            "High variance (±40%) is inherent to the stochastic startup survival model."
        ),
        overrides={
            # These are now the defaults - profile documents the calibration
            "INITIAL_CAPITAL": 1_500_000.0,
            "INITIAL_CAPITAL_RANGE": (1_200_000.0, 1_800_000.0),
            "BASE_OPERATIONAL_COST": 50_000.0,  # Default
        },
        target_metrics={
            "survival_rate_round200": {
                "target": 0.50,
                "tolerance": 0.08,
                "calibrated_result": 0.472,  # From 20-run verification
                "calibrated_std": 0.398,      # High variance inherent to model
                "source": "Internal calibration 2025-01, targeting BLS benchmarks.",
            },
        },
    ),
}
