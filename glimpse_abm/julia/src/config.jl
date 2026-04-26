"""
Configuration module for GlimpseABM.jl

Provides the comprehensive configuration system for the agent-based model,
including all tunable parameters, calibration profiles, and validation utilities.

Port of: glimpse_abm/config.py
"""

using Random
using JSON3

# ============================================================================
# TRAIT DISTRIBUTION SPECIFICATION
# ============================================================================

"""Distribution specification for agent traits."""
struct TraitDistribution
    dist::String
    params::Dict{String,Float64}
end

function TraitDistribution(; dist::String, params::Dict)
    TraitDistribution(dist, Dict{String,Float64}(String(k) => Float64(v) for (k,v) in params))
end

# ============================================================================
# SECTOR PROFILE
# ============================================================================

"""
Sector-specific economic parameters.

Calibration Sources:
- initial_capital_range: NVCA 2024 Yearbook, PitchBook seed/early stage data (scaled for 24-36 round runway)
- operational_cost_range: BLS QCEW quarterly costs by sector (SBA small business benchmarks 2024)
- survival_threshold: BLS Business Employment Dynamics, Fed SBCS 2024 (2-3 quarters operating expenses)
- innovation_probability: NSF BRDIS 2023, USPTO Patent Statistics grant rates
- innovation_return_multiplier: Industry R&D intensity studies
- knowledge_decay_rate: Ebbinghaus forgetting curve, industry skill depreciation research
- competition_intensity: Census Bureau Economic Census HHI data, DOJ HHI guidelines
"""
struct SectorProfile
    return_range::Tuple{Float64,Float64}
    return_log_mu::Float64
    return_log_sigma::Float64
    return_volatility_range::Tuple{Float64,Float64}
    failure_range::Tuple{Float64,Float64}
    failure_volatility_range::Tuple{Float64,Float64}
    capital_range::Tuple{Float64,Float64}
    maturity_range::Tuple{Int,Int}
    gross_margin_range::Tuple{Float64,Float64}
    operating_margin_range::Tuple{Float64,Float64}
    # Empirically-calibrated fields
    initial_capital_range::Tuple{Float64,Float64}       # NVCA 2024: sector-specific seed/Series A
    operational_cost_range::Tuple{Float64,Float64}      # BLS QCEW: quarterly operating costs
    survival_threshold::Float64                          # BLS/Fed: ~2 quarters operating expenses
    innovation_probability::Float64                      # NSF BRDIS 2023, USPTO grant rates
    innovation_return_multiplier::Tuple{Float64,Float64} # R&D intensity returns by sector
    knowledge_decay_rate::Float64                        # Skill depreciation half-life research
    competition_intensity::Float64                       # Census HHI-based competition intensity
end

# ============================================================================
# AI LEVEL CONFIGURATION
# ============================================================================

"""AI tier configuration."""
struct AILevelConfig
    cost::Float64
    cost_type::String
    info_quality::Float64
    info_breadth::Float64
    per_use_cost::Float64
end

"""AI domain capability specification."""
struct AIDomainCapability
    accuracy::Float64
    hallucination_rate::Float64
    bias::Float64
end

# ============================================================================
# MAIN CONFIGURATION STRUCT
# ============================================================================

"""
Enhanced configuration for emergent simulation with all AI learning features.

This is a direct port of the Python EmergentConfig dataclass with all 100+
parameters preserved for exact behavioral compatibility.
"""
@kwdef mutable struct EmergentConfig
    # ========================================================================
    # AGENT CONFIGURATION
    # ========================================================================
    # v2.7 default changed "emergent" → "fixed" for ATE-correctness of scripts
    # that assume fixed-tier (the main-analysis design). Emergent mode is the
    # robustness-check design — explicitly opt in per-run. Earlier "emergent"
    # default let run_fixed_tier_analysis + run_comprehensive_analysis silently
    # run in emergent mode because they never set AGENT_AI_MODE.
    AGENT_AI_MODE::String = "fixed"
    N_AGENTS::Int = 1000
    # NOTE on initial-capital precedence (corrected 2026-04-23):
    # The EmergentAgent constructor honors `initial_capital` kwarg first, then
    # falls back to sector_profile.initial_capital_range, then to
    # INITIAL_CAPITAL_RANGE below. The singular INITIAL_CAPITAL constant is NOT
    # consulted by the agent constructor — scripts that pass it via EmergentConfig
    # will be silently overridden by sector profiles. To force a uniform starting
    # capital across agents, pass it explicitly per-agent to EmergentAgent(...,
    # initial_capital=X). Field retained because production scripts reference it.
    INITIAL_CAPITAL::Float64 = 5_000_000.0
    # v2.10: explicit flag to force uniform starting capital across all agents.
    # When true, EmergentAgent constructor uses config.INITIAL_CAPITAL (scalar)
    # instead of sampling from sector_profile.initial_capital_range. Earlier
    # the sentinel-based detection (INITIAL_CAPITAL != 5M) failed when scripts
    # set it to 5M — the default — intending uniform capital.
    USE_UNIFORM_INITIAL_CAPITAL::Bool = false
    INITIAL_CAPITAL_RANGE::Tuple{Float64,Float64} = (2_500_000.0, 10_000_000.0)
    SURVIVAL_THRESHOLD::Float64 = 2_000_000.0
    # v2.10: explicit flag. When true, check_survival! uses the scalar
    # SURVIVAL_THRESHOLD instead of sector_profile.survival_threshold.
    USE_UNIFORM_SURVIVAL_THRESHOLD::Bool = false
    SURVIVAL_CAPITAL_RATIO::Float64 = 0.40  # equity_failure floor (v2.5: was 0.38, now 0.40)
    INSOLVENCY_GRACE_ROUNDS::Int = 6  # was 7 months; moderately tightened (v2.5)
    RANDOM_SEED::Int = 42
    USE_NUMPY_RNG::Bool = false  # Use NumpyRNG for cross-language reproducibility

    # Monthly operational cost — calibrated for ~55% 5-yr survival (BLS benchmark).
    # Raised 15K → 22.5K in v2.5 to reflect 2026-era compute / SaaS / cloud
    # infrastructure costs that all entrepreneurial ventures face. Premium-tier
    # subscription cost ($3500/month) stacks on top — those represent dedicated
    # AI tooling above the baseline compute environment.
    BASE_OPERATIONAL_COST::Float64 = 22500.0
    COMPETITION_COST_MULTIPLIER::Float64 = 50.0  # Monthly multiplier (was 150 quarterly)
    OPERATING_RESERVE_MONTHS::Int = 3
    MAX_AGENT_KNOWLEDGE::Int = 90
    SECTOR_STRENGTH_PRUNE_THRESHOLD::Float64 = 0.1
    LIQUIDITY_RESERVE_FRACTION::Float64 = 0.29
    MAX_INVESTMENT_FRACTION::Float64 = 0.037  # Monthly (was 0.11 quarterly) - invest 3x less per round
    TARGET_INVESTMENT_FRACTION::Float64 = 0.033  # Monthly (was 0.10 quarterly)
    # v3.5: high-conviction bet sizing for power-law right-tail outcomes.
    # When confidence × signal_score exceeds HIGH_CONVICTION_THRESHOLD,
    # the bet can scale up to MAX_HIGH_CONVICTION_FRACTION of capital
    # (vs the standard MAX_INVESTMENT_FRACTION). Real founders deploy
    # 10-30% of capital on conviction plays; the standard 3.7% cap
    # suppresses the right tail that produces venture-scale outcomes.
    # v3.5.2 calibration: a sequence of probes converged on 0.06 / 1.5.
    # Earlier 0.10 / 1.2 tanked survival 0.58 → 0.29; 0.08 / 1.5 brought
    # survival back to 0.44 but still hurt fixed-tier (0.53 → 0.37).
    # 0.06 / 1.5 produces meaningful right-tail (~5% of survivors >1.0×,
    # max growth ~3-4×) without much survival regression. Going beyond
    # this requires structural changes (exit events, longer compounding
    # horizons) — see Option C in the v3.5 plan if pursuing.
    MAX_HIGH_CONVICTION_FRACTION::Float64 = 0.06
    HIGH_CONVICTION_THRESHOLD::Float64 = 1.5
    AI_CREDIT_LINE_ROUNDS::Int = 24  # 24 months = 2 years (typical seed funding runway)
    AI_TRUST_RESERVE_DISCOUNT::Float64 = 0.25

    # Robustness test parameters for scaling AI failure modes and costs
    HALLUCINATION_INTENSITY::Float64 = 1.0  # Scale AI hallucination rates (0.0=none, 1.0=baseline, 2.0=double)
    OVERCONFIDENCE_INTENSITY::Float64 = 1.0  # Scale AI overconfidence effects (0.0=none, 1.0=baseline, 2.0=double)
    AI_COST_INTENSITY::Float64 = 1.0        # Scale AI subscription/usage costs (0.0=free, 1.0=baseline, 2.0=double)
    OPS_COST_INTENSITY::Float64 = 1.0       # Scale agent.operating_cost_estimate (0.0=free, 1.0=baseline, 2.0=double); refutation knob

    # ========================================================================
    # NETWORK CONFIGURATION
    # ========================================================================
    USE_NETWORK_EFFECTS::Bool = true
    NETWORK_N_NEIGHBORS::Int = 4
    NETWORK_REWIRING_PROB::Float64 = 0.1

    # ========================================================================
    # AGENT TRAIT DISTRIBUTIONS
    # ========================================================================
    TRAIT_DISTRIBUTIONS::Dict{String,TraitDistribution} = Dict(
        "uncertainty_tolerance" => TraitDistribution(dist="beta", params=Dict("a" => 1.05, "b" => 0.65)),
        "innovativeness" => TraitDistribution(dist="lognormal", params=Dict("mean" => 0.5, "sigma" => 0.5)),
        "competence" => TraitDistribution(dist="uniform", params=Dict("low" => 0.1, "high" => 0.8)),
        "ai_trust" => TraitDistribution(dist="normal_clipped", params=Dict("mean" => 0.5, "std" => 0.38)),
        "trait_momentum" => TraitDistribution(dist="uniform", params=Dict("low" => 0.6, "high" => 0.9)),
        "cognitive_style" => TraitDistribution(dist="uniform", params=Dict("low" => 0.8, "high" => 1.2)),
        "analytical_ability" => TraitDistribution(dist="uniform", params=Dict("low" => 0.1, "high" => 0.9)),
        "exploration_tendency" => TraitDistribution(dist="beta", params=Dict("a" => 0.85, "b" => 0.85)),
        "market_awareness" => TraitDistribution(dist="uniform", params=Dict("low" => 0.1, "high" => 0.9)),
        "entrepreneurial_drive" => TraitDistribution(dist="beta", params=Dict("a" => 2.2, "b" => 1.8)),
    )

    # ========================================================================
    # MARKET CONFIGURATION
    # ========================================================================
    N_ROUNDS::Int = 120  # 120 months = 10 years (monthly cadence)
    N_RUNS::Int = 50
    BASE_OPPORTUNITIES::Int = 5
    DISCOVERY_PROBABILITY::Float64 = 0.20  # Monthly probability (was 0.30 quarterly)
    INNOVATION_PROBABILITY::Float64 = 0.14  # Monthly probability (was 0.42 quarterly)
    AI_HERDING_DECAY::Float64 = 1.0
    AI_SIGNAL_HISTORY::Int = 420  # 420 months history (was 140 quarters)

    # ========================================================================
    # INNOVATION ECONOMICS
    # ========================================================================
    INNOVATION_BASE_SPEND_RATIO::Float64 = 0.025
    INNOVATION_MAX_SPEND::Float64 = 2667.0  # Monthly max (was 8000 quarterly)
    INNOVATION_FAIL_RECOVERY_RATIO::Float64 = 0.12
    INNOVATION_SUCCESS_BASE_RETURN::Float64 = 0.08  # Monthly return (was 0.25 quarterly)
    INNOVATION_SUCCESS_RETURN_MULTIPLIER::Tuple{Float64,Float64} = (1.8, 3.2)
    INNOVATION_RD_CAP_FRACTION::Float64 = 0.04  # Monthly cap (was 0.12 quarterly)
    INNOVATION_REUSE_PROBABILITY::Float64 = 0.07  # Monthly probability (was 0.22 quarterly)
    INNOVATION_REUSE_LOOKBACK::Int = 300  # 300 months lookback (was 100 quarters)

    # ========================================================================
    # REFUTATION TEST PARAMETERS
    # These allow testing model robustness by modifying AI tier advantages
    # ========================================================================

    # Execution success multipliers by AI tier
    # Default: all 1.0 (no AI advantage in execution)
    AI_EXECUTION_SUCCESS_MULTIPLIERS::Dict{String,Float64} = Dict(
        "none" => 1.0,
        "basic" => 1.0,
        "advanced" => 1.0,
        "premium" => 1.0
    )

    # Innovation quality boost by AI tier (additive, on 0-1 scale)
    # Default: 0.05 for all AI tiers (small boost for AI assistance)
    AI_QUALITY_BOOST::Dict{String,Float64} = Dict(
        "none" => 0.0,
        "basic" => 0.05,
        "advanced" => 0.05,
        "premium" => 0.05
    )

    # v2.7: AI_COST_MULTIPLIER DELETED — was duplicate of AI_COST_INTENSITY
    # (config.jl:135) but never actually read by src/ billing code. Robustness
    # scripts that set AI_COST_MULTIPLIER were silent no-ops. Scripts updated
    # to use AI_COST_INTENSITY in the same commit.
    INVESTMENT_SUCCESS_ROI_THRESHOLD::Float64 = 0.017  # Monthly threshold (was 0.05 quarterly)
    BURN_HISTORY_WINDOW::Int = 9  # 9 months (was 3 quarters)
    BURN_FAILURE_THRESHOLD::Float64 = 0.12
    BURN_LEVERAGE_CAP::Float64 = 0.75
    RETURN_OVERSUPPLY_PENALTY::Float64 = 0.35  # Reduced from 0.52 for better survival
    RETURN_UNDERSUPPLY_BONUS::Float64 = 0.37
    RETURN_NOISE_SCALE::Float64 = 0.38
    BRANCH_LOG_MEAN_DRIFT::Float64 = 0.11
    BRANCH_LOG_SIGMA_DRIFT::Float64 = 0.07
    BRANCH_FAILURE_DRIFT::Float64 = 0.04
    BRANCH_FEEDBACK_RATE::Float64 = 0.05

    # ========================================================================
    # OPPORTUNITY CHARACTERISTICS
    # ========================================================================
    OPPORTUNITY_RETURN_RANGE::Tuple{Float64,Float64} = (1.1, 25.0)
    OPPORTUNITY_UNCERTAINTY_RANGE::Tuple{Float64,Float64} = (0.12, 0.60)
    OPPORTUNITY_COMPLEXITY_RANGE::Tuple{Float64,Float64} = (0.0, 2.0)

    # ========================================================================
    # AI TOOL CONFIGURATION
    # ========================================================================
    # Temporal Framework: Simulation represents 2028-2038 (post-AGI launch)
    # Scaling Law: none (human) → basic (consumer AI) → advanced (SOTA) → premium (AGI)
    #
    # Tier definitions:
    # - none: Human decision-making only (no AI augmentation)
    # - basic: Consumer-level AI tools (ChatGPT Plus, Claude Pro) ~$20-30/month
    # - advanced: Current SOTA systems (GPT-4, Claude Opus, o1) ~$400/month heavy use
    # - premium: Projected AGI systems (December 2027 launch) ~$3,500/month early access
    AI_LEVELS::Dict{String,AILevelConfig} = Dict(
        "none" => AILevelConfig(0.0, "none", 0.25, 0.20, 0.0),           # Human baseline
        "basic" => AILevelConfig(30.0, "per_use", 0.43, 0.38, 3.0),      # Consumer AI
        "advanced" => AILevelConfig(400.0, "subscription", 0.70, 0.65, 35.0),  # SOTA (2024-2026)
        "premium" => AILevelConfig(3500.0, "subscription", 0.97, 0.92, 150.0), # AGI (Dec 2027)
    )

    AI_DOMAIN_CAPABILITIES::Dict{String,Dict{String,AIDomainCapability}} = Dict(
        "none" => Dict(
            "market_analysis" => AIDomainCapability(0.38, 0.28, 0.06),
            "technical_assessment" => AIDomainCapability(0.40, 0.26, -0.04),
            "uncertainty_evaluation" => AIDomainCapability(0.35, 0.30, -0.05),
            "innovation_potential" => AIDomainCapability(0.33, 0.29, 0.07),
        ),
        "basic" => Dict(
            "market_analysis" => AIDomainCapability(0.52, 0.20, 0.04),
            "technical_assessment" => AIDomainCapability(0.54, 0.19, -0.03),
            "uncertainty_evaluation" => AIDomainCapability(0.50, 0.21, -0.04),
            "innovation_potential" => AIDomainCapability(0.48, 0.22, 0.05),
        ),
        "advanced" => Dict(
            "market_analysis" => AIDomainCapability(0.78, 0.10, 0.025),
            "technical_assessment" => AIDomainCapability(0.80, 0.09, -0.015),
            "uncertainty_evaluation" => AIDomainCapability(0.76, 0.11, -0.02),
            "innovation_potential" => AIDomainCapability(0.74, 0.12, 0.02),
        ),
        "premium" => Dict(
            "market_analysis" => AIDomainCapability(0.96, 0.015, 0.003),
            "technical_assessment" => AIDomainCapability(0.97, 0.012, -0.002),
            "uncertainty_evaluation" => AIDomainCapability(0.95, 0.018, -0.003),
            "innovation_potential" => AIDomainCapability(0.94, 0.02, 0.004),
        ),
    )

    # ========================================================================
    # LEARNING PARAMETERS
    # ========================================================================
    UNCERTAINTY_LEARNING_ENABLED::Bool = true
    INITIAL_RESPONSE_VARIANCE::Float64 = 0.3
    LEARNING_RATE::Float64 = 0.05  # Learning rate for uncertainty response adaptation
    EXPLORATION_DECAY::Float64 = 0.9983  # Monthly decay (was 0.995 quarterly, same annual effect)
    SOCIAL_LEARNING_WEIGHT::Float64 = 0.2
    MIN_FUNDING_FRACTION::Float64 = 0.25
    COST_OF_CAPITAL::Float64 = 0.005  # Monthly cost (was 0.06 quarterly, ~6% annual)
    EXPLORATION_SENSITIVITY::Float64 = 1.5
    MAINTAIN_UNCERTAINTY_SENSITIVITY::Float64 = 0.5

    # ========================================================================
    # UNCERTAINTY AND MARKET DYNAMICS
    # ========================================================================
    BLACK_SWAN_PROBABILITY::Float64 = 0.017  # Monthly probability (was 0.05 quarterly)
    BOOM_TAIL_UNCERTAINTY_EXPONENT::Float64 = 1.08
    MARKET_VOLATILITY::Float64 = 0.15  # Monthly volatility (lower than quarterly)
    COMPETITION_SCALE_FACTOR::Float64 = 1.0  # Multiplier for sector-specific competition_intensity (0.0=no competition, 1.0=baseline, 2.0=double)
    DISABLE_COMPETITION_DYNAMICS::Bool = false  # Set to true to freeze competition at 0.0 (for counterfactual analysis)

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
    USE_CAPACITY_CONVEXITY_CROWDING::Bool = true  # Enable new crowding model

    # K = Carrying capacity: competition level where penalties START
    # Interpretation: "effective rivals" that can coexist without pain
    # At competition < K: no crowding penalty (healthy competition)
    # At competition = K: penalty just starts.
    #
    # v2.5 calibration: K = 1.5 (was 8.0 in earlier canonical drift).
    # Diagnostic at N=1000 with K=8 showed max competition only reached 2.57
    # across a 60-round run, so the convexity penalty was DEAD CODE — no
    # opportunity ever crossed K, exp(-penalty) was always 1.0. The
    # convergence-driven crowding mechanism (premium agents piling into the
    # same top-ranked opps → penalty on those opps' returns) couldn't engage.
    # Restored to v1's K=1.5 so the top ~5-15% of crowded opportunities now
    # incur a measurable convexity penalty.
    #
    # Population scaling multiplies K by √(N/N_ref), so K stays correlated
    # with the per-opp competition pressure that grows with N.
    #   N=1K:   K = 1.50
    #   N=10K:  K = 4.74
    #   N=100K: K = 15.0
    CROWDING_CAPACITY_K::Float64 = 1.5

    # γ = Convexity exponent: how sharply penalties increase beyond capacity
    # γ = 1: linear (gentle)
    # γ = 1.5: mildly convex (gradual ramp — realistic for competitive markets)
    # γ = 2: quadratic (convex - "a little crowded is OK, very crowded is brutal")
    # γ = 3: cubic (very sharp)
    # Calibrated: γ=1.5 produces gradual penalty from C=5 to C=15, avoiding
    # the cliff-like profile of γ=2 where C<K is fine but C>K is catastrophic.
    CROWDING_CONVEXITY_GAMMA::Float64 = 1.5

    # λ = Strength: maps crowding into payoff reduction
    # At 2× capacity (C/K = 2), penalty = λ · 1^γ = λ
    # exp(-λ) gives the return multiplier at 2× capacity:
    #   λ = 0.50 → ~39% drop at 2× capacity (exp(-0.50) ≈ 0.61)
    #   λ = 1.00 → ~63% drop at 2× capacity (exp(-1.00) ≈ 0.37)
    #   λ = 1.50 → ~78% drop at 2× capacity (exp(-1.50) ≈ 0.22)
    # Calibrated: λ=1.0 with K=5.0 and γ=1.5 produces portfolio mean ~1.06×
    # at observed competition levels (median C≈3.6, mean C≈6.7)
    CROWDING_STRENGTH_LAMBDA::Float64 = 1.5

    # v3.1 — Capital-saturation convexity threshold.
    # K_sat is the saturation ratio (total_invested / capacity) above which
    # the convexity penalty starts. Unlike CROWDING_CAPACITY_K above (which
    # operates on count-of-competitors and scales with √N), K_sat is a
    # capital-ratio and does NOT scale with N — capacity and invested capital
    # both scale together, so the ratio is population-invariant.
    #   K_sat = 1.0 → penalty starts at full capacity
    #   K_sat = 1.5 → penalty starts 50% above capacity (niche meaningfully oversubscribed)
    # Calibrated on N=1000 × 60 rounds × 4-tier fixed (seed=42):
    #   K_sat=1.5: none=0.39 basic=0.55 advanced=0.62 premium=0.43 (mean 0.50)
    #   Hits 50% aggregate target; preserves advanced > basic ordering. Premium
    #   recovers above v2.12 (0.43 vs 0.38) now that the tier-share penalty is
    #   gone and only capital-saturation applies.
    CROWDING_CAPACITY_RATIO_K::Float64 = 1.5

    # Legacy parameters (used when USE_CAPACITY_CONVEXITY_CROWDING = false)
    OPPORTUNITY_COMPETITION_PENALTY::Float64 = 0.5  # Return penalty per unit of opp.competition (0.5 = 50% max penalty at competition=1.0)
    OPPORTUNITY_COMPETITION_THRESHOLD::Float64 = 0.2  # Competition level above which penalty starts applying
    OPPORTUNITY_COMPETITION_FLOOR::Float64 = 0.1  # Minimum return multiplier after competition penalty (0.1 = 90% max reduction)
    MARKET_SHIFT_PROBABILITY::Float64 = 0.03  # Monthly probability (was 0.09 quarterly)
    MARKET_SHIFT_SEVERITY_RANGE::Tuple{Float64,Float64} = (0.25, 0.75)
    MARKET_SHIFT_MAX_SECTORS::Int = 3

    MARKET_SHIFT_REGIME_MULTIPLIER::Dict{String,Float64} = Dict(
        "boom" => 1.5,
        "growth" => 1.2,
        "normal" => 1.0,
        "volatile" => 1.3,
        "recession" => 1.6,
        "crisis" => 2.2,
    )

    MACRO_REGIME_STATES::Tuple{String,String,String,String,String} =
        ("crisis", "recession", "normal", "growth", "boom")

    # NBER Business Cycle-calibrated transition matrix (monthly rounds)
    # Source: NBER Business Cycle Dating Committee data 1945-2024
    # Average expansion: 64 months, average recession: 11 months
    # Crisis frequency: ~1 per decade, boom frequency: ~2-3 per decade
    # Adjusted for monthly cadence - higher self-transition to maintain regime duration
    # Formula: p_monthly = 1 - (1 - p_quarterly)/3
    MACRO_REGIME_TRANSITIONS::Dict{String,Dict{String,Float64}} = Dict(
        "crisis" => Dict("crisis" => 0.75, "recession" => 0.12, "normal" => 0.12, "growth" => 0.01, "boom" => 0.0),
        "recession" => Dict("crisis" => 0.01, "recession" => 0.77, "normal" => 0.17, "growth" => 0.04, "boom" => 0.01),
        "normal" => Dict("crisis" => 0.003, "recession" => 0.027, "normal" => 0.85, "growth" => 0.09, "boom" => 0.03),
        "growth" => Dict("crisis" => 0.003, "recession" => 0.01, "normal" => 0.07, "growth" => 0.84, "boom" => 0.077),
        "boom" => Dict("crisis" => 0.003, "recession" => 0.013, "normal" => 0.067, "growth" => 0.13, "boom" => 0.787),
    )

    # Regime modifiers - balanced for realistic business cycle dynamics
    MACRO_REGIME_RETURN_MODIFIERS::Dict{String,Float64} = Dict(
        "crisis" => 0.85, "recession" => 0.98, "normal" => 1.10, "growth" => 1.28, "boom" => 1.45
    )

    MACRO_REGIME_FAILURE_MODIFIERS::Dict{String,Float64} = Dict(
        "crisis" => 1.18, "recession" => 1.05, "normal" => 1.0, "growth" => 0.90, "boom" => 0.78
    )

    MACRO_REGIME_TREND::Dict{String,Float64} = Dict(
        "crisis" => -0.8, "recession" => -0.35, "normal" => 0.0, "growth" => 0.35, "boom" => 0.6
    )

    MACRO_REGIME_VOLATILITY::Dict{String,Float64} = Dict(
        "crisis" => 0.55, "recession" => 0.35, "normal" => 0.2, "growth" => 0.25, "boom" => 0.3
    )

    RETURN_DEMAND_CROWDING_THRESHOLD::Float64 = 0.42
    RETURN_DEMAND_CROWDING_PENALTY::Float64 = 0.48
    FAILURE_DEMAND_PRESSURE::Float64 = 0.20

    # ========================================================================
    # RECURSION WEIGHTS
    # ========================================================================
    RECURSION_WEIGHTS::Dict{String,Float64} = Dict(
        "crowd_weight" => 0.35,
        "volatility_weight" => 0.30,
        "ai_herd_weight" => 0.40,
        "premium_reuse_weight" => 0.20,
    )

    AI_NOVELTY_UPLIFT::Float64 = 0.08
    DOWNSIDE_OVERSUPPLY_WEIGHT::Float64 = 0.30  # Balanced penalty for oversupply conditions
    RETURN_LOWER_BOUND::Float64 = 0.0  # Minimum return multiple (0.0 = total loss, 0.5 = 50% back)

    # ========================================================================
    # NOVELTY DISRUPTION PARAMETERS (The "DeepSeek Effect")
    # When highly novel innovations occur, they can disrupt existing opportunities
    # ========================================================================
    NOVELTY_DISRUPTION_ENABLED::Bool = true
    NOVELTY_DISRUPTION_THRESHOLD::Float64 = 0.6       # Novelty level that triggers disruption
    NOVELTY_DISRUPTION_MAGNITUDE::Float64 = 0.25      # Max return reduction from disruption (0.25 = 25%)
    DISRUPTION_COMPETITION_THRESHOLD::Float64 = 10.0  # Competition level for vulnerability targeting
    NOVELTY_NOISE_INVERSION_FACTOR::Float64 = 0.4     # Premium's disadvantage on novel opportunities

    # ========================================================================
    # CAPACITY CONSTRAINT PARAMETERS
    # Capacity represents max concurrent capital an opportunity can absorb.
    # With 1000 agents at $5M, ~400 investing $185K/round across 30 opportunities,
    # outstanding capital per opportunity is ~$10-30M at steady state.
    # The capital-saturation convexity (models.jl) uses total_invested/capacity
    # as its crowding signal.
    # ========================================================================
    OPPORTUNITY_BASE_CAPACITY::Float64 = 15_000_000.0  # Base capacity in dollars
    OPPORTUNITY_CAPACITY_VARIANCE::Float64 = 0.3       # Random variance in capacity

    # ========================================================================
    # SEQUENTIAL DECISION PARAMETERS
    # Information cascades from observing early investors
    # ========================================================================
    SEQUENTIAL_DECISIONS_ENABLED::Bool = true
    EARLY_DECISION_FRACTION::Float64 = 0.3            # Fraction deciding in first wave
    SIGNAL_VISIBILITY_WEIGHT::Float64 = 0.15          # How much early signals influence later agents

    # ========================================================================
    # SCALING PARAMETERS
    # ========================================================================
    OPPORTUNITIES_PER_CAPITA::Float64 = 0.04
    DISCOVERY_RATE_SCALING::Float64 = 0.5
    MIN_OPPORTUNITIES::Int = 5
    POWER_LAW_SHAPE_A::Float64 = 2.2  # Heavier tails for VC-like power-law returns (finite mean since α>2)
    OPPORTUNITY_CAPITAL_REQUIREMENTS::Float64 = 10000.0

    # ========================================================================
    # POPULATION SCALING PARAMETERS
    # ========================================================================
    # Reference population for which all absolute parameters were calibrated.
    # When N_AGENTS differs, initialize!() applies scaling adjustments to keep
    # model dynamics comparable across population sizes.
    SCALE_REFERENCE_N::Int = 1000
    ENABLE_POPULATION_SCALING::Bool = true   # Set false to use raw (unscaled) parameters
    _POPULATION_SCALING_APPLIED::Bool = false  # Idempotency guard for initialize!
    COMPETITION_DECAY_RATE::Float64 = 0.02   # Per-round decay to prevent unbounded competition accumulation
    MAX_KNOWLEDGE_PIECES::Int = 5000         # Cap on knowledge registry size (age-based eviction)
    INNOVATION_HISTORY_RETENTION::Int = 20   # Keep only last N rounds of innovation history per sector

    # ========================================================================
    # DIAGNOSTICS
    # ========================================================================
    ENABLE_DEBUG_LOGS::Bool = false

    AI_TIER_RECENT_WINDOW::Int = 45  # 45 months (was 15 quarters)
    AI_TIER_DEMOTE_MARGIN::Float64 = 0.06
    AI_TIER_NEIGHBOR_INFLUENCE::Float64 = 0.06

    AI_TIER_SCORING::Dict{String,Dict{String,Any}} = Dict(
        "basic" => Dict(
            "score_threshold" => 0.0, "score_slope" => 6.0, "trial_rounds" => 12,  # 12 months (was 4 quarters)
            "weights" => Dict(
                "trust" => 0.6, "success" => 0.3, "success_gain" => 0.2,
                "recent_success_gain" => 0.2, "roi_gain" => 0.15, "recent_roi_gain" => 0.15,
                "cost" => 0.25, "accuracy" => 0.10, "usage" => 0.08, "bias" => -0.25,
            ),
        ),
        "advanced" => Dict(
            "score_threshold" => 0.08, "score_slope" => 6.0, "trial_rounds" => 12,  # 12 months (was 4 quarters)
            "weights" => Dict(
                "trust" => 0.65, "success" => 0.32, "success_gain" => 0.27,
                "recent_success_gain" => 0.27, "roi_gain" => 0.22, "recent_roi_gain" => 0.22,
                "cost" => 0.18, "accuracy" => 0.15, "usage" => 0.12, "bias" => -0.28,
            ),
        ),
        "premium" => Dict(
            "score_threshold" => 0.18, "score_slope" => 6.5, "trial_rounds" => 15,  # 15 months (was 5 quarters)
            "weights" => Dict(
                "trust" => 0.70, "success" => 0.38, "success_gain" => 0.32,
                "recent_success_gain" => 0.32, "roi_gain" => 0.28, "recent_roi_gain" => 0.28,
                "cost" => 0.22, "accuracy" => 0.20, "usage" => 0.14, "bias" => -0.35,
            ),
        ),
    )

    AI_TIER_SCORE_SMOOTHING::Float64 = 0.25
    AI_TIER_DEMOTION_COOLDOWN::Int = 36  # 36 months (was 12 quarters)
    TRAIT_MOMENTUM::Float64 = 0.7
    AI_TRUST_ADJUSTMENT_RATE::Float64 = 0.033  # Monthly adjustment (was 0.1 quarterly)
    AI_SUBSCRIPTION_AMORTIZATION_ROUNDS::Int = 180  # 180 months (was 60 quarters)
    # v3.4: months between emergent-mode tier re-evaluations. Real ChatGPT
    # users don't reconsider their subscription monthly; quarterly
    # cadence is closer to observed behavior. Pre-v3.4 emergent agents
    # re-decided every round, producing more switching than reality.
    AI_TIER_REVIEW_INTERVAL::Int = 3
    # v3.4.2: initial freeze period — defer the FIRST tier review until
    # this many rounds have elapsed. Lets investments mature so
    # tier_roi_history populates with real performance data before
    # agents make their first informed switch. Without this, the early
    # review fires while ROI=0 for all tiers and decisions devolve into
    # cost-vs-peer-signal cascades. Default 12 = one investment cycle.
    AI_TIER_INITIAL_FREEZE_ROUNDS::Int = 12
    AI_SUBSCRIPTION_FLOAT_BASE_ROUNDS::Int = 0
    AI_SUBSCRIPTION_FLOAT_MAX_ROUNDS::Int = 9  # 9 months (was 3 quarters)

    # ========================================================================
    # ACTION SELECTION CONTROLS
    # ========================================================================
    ACTION_SELECTION_TEMPERATURE::Float64 = 0.45
    ACTION_SELECTION_NOISE::Float64 = 0.10
    ACTION_BIAS_SIGMA::Float64 = 0.05

    # ========================================================================
    # UNCERTAINTY VOLATILITY CONTROLS
    # ========================================================================
    UNCERTAINTY_SHORT_WINDOW::Int = 18  # 18 months (was 6 quarters)
    UNCERTAINTY_SHORT_DECAY::Float64 = 0.0
    UNCERTAINTY_VOLATILITY_WINDOW::Int = 42  # 42 months (was 14 quarters)
    UNCERTAINTY_VOLATILITY_DECAY::Float64 = 0.87  # Monthly decay (was 0.6 quarterly)
    UNCERTAINTY_VOLATILITY_SCALING::Float64 = 0.45
    UNCERTAINTY_AI_SWITCH_WEIGHT::Float64 = 0.09
    UNCERTAINTY_MARKET_RETURN_WEIGHT::Float64 = 0.14
    UNCERTAINTY_ACTION_VARIANCE_WEIGHT::Float64 = 0.14
    UNCERTAINTY_CROWDING_WEIGHT::Float64 = 0.18
    UNCERTAINTY_COMPETITIVE_WEIGHT::Float64 = 0.12

    # ========================================================================
    # KNOWLEDGE & INNOVATION
    # ========================================================================
    KNOWLEDGE_DECAY_RATE::Float64 = 0.025  # Monthly decay (was 0.075 quarterly)

    SECTOR_KNOWLEDGE_PERSISTENCE::Dict{String,Float64} = Dict(
        "tech" => 0.85, "retail" => 0.92, "service" => 0.95, "manufacturing" => 0.97,
    )

    SECTORS::Vector{String} = String[]

    # NVCA 2024-calibrated sector weights for agent initial sector assignment
    # Reflects dominant VC deal flow by sector
    SECTOR_WEIGHTS::Dict{String,Float64} = Dict(
        "tech" => 0.60,           # 60% - dominant VC deal flow
        "service" => 0.15,        # 15% - B2B services
        "manufacturing" => 0.15,  # 15% - hardware/industrial
        "retail" => 0.10,         # 10% - consumer/retail
    )

    # Sector profiles with empirically-calibrated parameters
    # Sources: NVCA 2024, BLS BED/QCEW, Fed SBCS, NSF BRDIS, USPTO, Census HHI
    # Capital ranges calibrated for 24-36 month runway (monthly cadence)
    SECTOR_PROFILES::Dict{String,SectorProfile} = Dict(
        "tech" => SectorProfile(
            (1.60, 4.00), log(2.40), 0.45, (0.22, 0.38),           # return params (boosted for survival)
            (0.1, 0.17), (0.013, 0.04), (300000.0, 1200000.0),     # failure (monthly), capital
            (12, 36), (0.55, 0.85), (0.08, 0.28),                  # maturity: 12-36 months (fits 60-round sim)
            # Empirically-calibrated fields (scaled for 120-180 month runway):
            # Reflects Series A/B rounds with 24-36 month runway before profitability
            (3_000_000.0, 6_000_000.0),  # initial_capital_range: 120-240 months runway
            (20_000.0, 30_000.0),        # operational_cost_range: NET BURN RATE for tech (v3.5.9)
            # Reasoning: model's per-round op cost represents NET burn (gross opex - revenue
            # from existing operations), not gross opex. Gross opex from BLS QCEW NAICS 5415
            # wages ($130k/yr × 3-7 engineers / 12) × 1.4 overhead = $45-105k/mo, but going
            # concerns typically have revenue covering 60-80% of opex, leaving ~30% net burn:
            # 0.3 × ($45-105k) ≈ $13-32k. Center on $20-30k.
            1_950_000.0,                  # survival_threshold: ~65% of min sector capital (v2.5)
            0.16,                         # innovation_probability: monthly (was 0.48 quarterly)
            (2.0, 4.0),                   # innovation_return_multiplier: high tech upside
            0.04,                         # knowledge_decay_rate: monthly (was 0.12 quarterly)
            1.2                           # competition_intensity: HHI 1500-2500
        ),
        "retail" => SectorProfile(
            (1.40, 2.80), log(1.85), 0.32, (0.18, 0.3),            # return params (boosted for survival)
            (0.07, 0.13), (0.013, 0.033), (50000.0, 400000.0),     # failure (monthly), capital
            (6, 24), (0.18, 0.42), (0.015, 0.08),                  # maturity: 6-24 months (fits 60-round sim)
            # Empirically-calibrated fields (scaled for 120-180 month runway):
            (2_200_000.0, 4_000_000.0),  # initial_capital_range: 120-220 months runway
            (13_000.0, 23_000.0),        # operational_cost_range: net burn for retail; 0.3 × BLS QCEW NAICS 44-45 gross opex (v3.5.9)
            1_430_000.0,                  # survival_threshold: ~65% of min sector capital (v2.5)
            0.11,                         # innovation_probability: monthly (was 0.32 quarterly)
            (1.6, 2.5),                   # innovation_return_multiplier: moderate returns
            0.023,                        # knowledge_decay_rate: monthly (was 0.07 quarterly)
            0.7                           # competition_intensity: HHI 500-1000
        ),
        "service" => SectorProfile(
            (1.50, 3.00), log(1.95), 0.36, (0.16, 0.28),           # return params (boosted for survival)
            (0.033, 0.093), (0.01, 0.027), (15000.0, 200000.0),    # failure (monthly), capital
            (6, 18), (0.45, 0.75), (0.12, 0.24),                   # maturity: 6-18 months (fits 60-round sim)
            # Empirically-calibrated fields (scaled for 120-180 month runway):
            (1_400_000.0, 2_500_000.0),  # initial_capital_range: 120-213 months runway
            (8_300.0, 15_000.0),         # operational_cost_range: net burn for services; 0.3 × BLS QCEW NAICS 56/81 gross opex (v3.5.9)
            910_000.0,                    # survival_threshold: ~65% of min sector capital (v2.5)
            0.13,                         # innovation_probability: monthly (was 0.38 quarterly)
            (1.6, 2.5),                   # innovation_return_multiplier: moderate returns
            0.017,                        # knowledge_decay_rate: monthly (was 0.05 quarterly)
            0.9                           # competition_intensity: HHI 800-1500
        ),
        "manufacturing" => SectorProfile(
            (1.60, 3.50), log(2.20), 0.4, (0.18, 0.3),             # return params (boosted for survival)
            (0.083, 0.14), (0.013, 0.033), (250000.0, 1500000.0),  # failure (monthly), capital
            (18, 48), (0.28, 0.48), (0.04, 0.18),                  # maturity: 18-48 months (fits 60-round sim)
            # Empirically-calibrated fields (scaled for 120-180 month runway):
            (4_000_000.0, 7_500_000.0),  # initial_capital_range: 120-225 months runway
            (26_700.0, 40_000.0),        # operational_cost_range: net burn for manufacturing; 0.3 × BLS QCEW NAICS 31-33 gross opex (v3.5.9)
            2_600_000.0,                  # survival_threshold: ~65% of min sector capital (v2.5)
            0.17,                         # innovation_probability: monthly (was 0.52 quarterly)
            (1.5, 2.8),                   # innovation_return_multiplier: incremental improvements
            0.01,                         # knowledge_decay_rate: monthly (was 0.03 quarterly)
            1.4                           # competition_intensity: HHI 1800-3000
        ),
    )

    # ========================================================================
    # CALIBRATION
    # ========================================================================
    calibration_targets::Dict{String,Any} = Dict{String,Any}()
    active_calibration::Union{String,Nothing} = nothing

    # ========================================================================
    # PERFORMANCE / IO
    # ========================================================================
    buffer_flush_interval::Int = 15  # Every 15 months (was 5 quarters)
    write_intermediate_batches::Bool = true
    round_log_interval::Int = 12  # Log every year (12 months)
    enable_round_logging::Bool = true
    max_cache_size::Int = 100000
    agent_history_depth::Int = 5
    preallocate_arrays::Bool = true
    use_float32::Bool = true
    use_parallel::Bool = true
    parallel_threshold::Int = 120
    parallel_chunk_size::Int = 50
    max_workers::Int = max(1, Sys.CPU_THREADS - 1)
    PARALLEL_MODE::String = "max"
    MAX_PARALLEL_RUNS::Int = 0
end

# Post-initialization: sync SECTORS from SECTOR_PROFILES keys and apply population scaling
function initialize!(config::EmergentConfig)
    config.SECTORS = collect(keys(config.SECTOR_PROFILES))

    # Performance tuning for large populations
    if config.N_AGENTS > 5000
        config.buffer_flush_interval = 50
        config.max_cache_size = 100000
        config.agent_history_depth = 5
        config.preallocate_arrays = true
        config.use_float32 = true
    end

    # ── Population scaling ──────────────────────────────────────────
    # All absolute parameters were calibrated at SCALE_REFERENCE_N (default 1000).
    # When N_AGENTS differs, we adjust capacity/crowding thresholds so that
    # per-agent dynamics remain comparable.
    #
    # Scaling uses √(N/N_ref) for thresholds that gate on aggregate quantities.
    # Competition delta (market.jl) uses √N scaling with reference_population=100
    # to maintain calibrated per-capita pressure. These two scaling regimes
    # work together: √N thresholds absorb √N-scaled aggregate competition.
    if config.ENABLE_POPULATION_SCALING && config.N_AGENTS != config.SCALE_REFERENCE_N && !config._POPULATION_SCALING_APPLIED
        scale = config.N_AGENTS / config.SCALE_REFERENCE_N  # ratio (e.g., 10 for 10K, 100 for 100K)
        sqrt_scale = sqrt(scale)

        # v3.5.10 audit: do NOT scale CROWDING_CAPACITY_RATIO_K (the active
        # convex-crowding threshold). The threshold is an economic property
        # of the opportunity ("saturation 1.5× capacity triggers the
        # convex penalty"), not a population-level statistic. Scaling it
        # with √N collides with the x_min floor (models.jl:418) and pins
        # all penalty levels to the same return distribution. The legacy
        # CROWDING_CAPACITY_K (kept for backwards compat) is still scaled
        # but currently unread by any active code path.
        config.CROWDING_CAPACITY_K *= sqrt_scale

        # Opportunity base capacity: max capital an opportunity absorbs.
        # Total capital scales linearly with N, opportunities sub-linearly.
        # Scale by √N (not linearly) to maintain utilization pressure.
        config.OPPORTUNITY_BASE_CAPACITY *= sqrt_scale

        # Disruption competition threshold: absolute level, scale with K
        config.DISRUPTION_COMPETITION_THRESHOLD *= sqrt_scale

        # Network neighbors: scale logarithmically so agents maintain a
        # comparable fraction of social influence.
        config.NETWORK_N_NEIGHBORS = max(4, floor(Int, log(config.N_AGENTS) + 2))

        # Competition decay: at higher N, competition accumulates faster,
        # so increase decay proportionally to keep steady-state levels bounded.
        config.COMPETITION_DECAY_RATE = min(0.10, 0.02 * sqrt_scale)

        config._POPULATION_SCALING_APPLIED = true
        @info "Population scaling applied" N=config.N_AGENTS ref=config.SCALE_REFERENCE_N scale sqrt_scale K=config.CROWDING_CAPACITY_K capacity=config.OPPORTUNITY_BASE_CAPACITY neighbors=config.NETWORK_N_NEIGHBORS decay=config.COMPETITION_DECAY_RATE
    end

    return config
end

"""
Compute integer opportunity targets even if overrides supply floats.

Scaling policy: opportunities grow sub-linearly with population to model
realistic market saturation (not every additional entrepreneur creates a
proportional number of new markets).

  n_opps = max(MIN, BASE, N_ref * per_capita * (1 + ln(N/N_ref)))

This gives ~40 at 1K, ~132 at 10K, ~264 at 100K (vs. 4000 with linear scaling).
"""
function get_scaled_opportunities(config::EmergentConfig, n_agents::Int)::Int
    min_ops = ceil(Int, Float64(config.MIN_OPPORTUNITIES))
    base_ops = ceil(Int, Float64(config.BASE_OPPORTUNITIES))
    per_capita = max(config.OPPORTUNITIES_PER_CAPITA, 0.0)

    if config.ENABLE_POPULATION_SCALING
        ref_n = config.SCALE_REFERENCE_N
        # Sub-linear: reference-level linear + log surplus
        ref_opps = ref_n * per_capita
        log_scale = 1.0 + max(0.0, log(n_agents / ref_n))
        scaled = ceil(Int, ref_opps * log_scale)
    else
        scaled = ceil(Int, n_agents * per_capita)
    end

    return max(min_ops, base_ops, scaled)
end

"""
Get AI domain capability for a given tier and domain.
"""
function get_ai_domain_capability(config::EmergentConfig, ai_level::String, domain::String)::AIDomainCapability
    if haskey(config.AI_DOMAIN_CAPABILITIES, ai_level)
        tier_caps = config.AI_DOMAIN_CAPABILITIES[ai_level]
        if haskey(tier_caps, domain)
            return tier_caps[domain]
        end
    end
    return AIDomainCapability(0.5, 0.1, 0.0)
end

"""
Return a deep-copied, JSON-safe representation of the configuration.
"""
function snapshot(config::EmergentConfig)::Dict{String,Any}
    result = Dict{String,Any}()
    for field_name in fieldnames(EmergentConfig)
        result[String(field_name)] = deepcopy(getfield(config, field_name))
    end
    return result
end

"""
Return a new config with the provided overrides merged in.
"""
function copy_with_overrides(config::EmergentConfig, overrides::Dict{String,Any})::EmergentConfig
    new_cfg = deepcopy(config)
    apply_overrides!(new_cfg, overrides)
    new_cfg.SECTORS = collect(keys(new_cfg.SECTOR_PROFILES))
    return new_cfg
end

"""
Recursively merge overrides into config.
"""
function apply_overrides!(config::EmergentConfig, overrides::Dict{String,Any})
    for (key, value) in overrides
        key_sym = Symbol(key)
        if !hasproperty(config, key_sym)
            error("Unknown configuration attribute '$key' in calibration override.")
        end
        current = getfield(config, key_sym)
        if isa(current, Dict) && isa(value, Dict)
            merged = deep_merge_dict(current, value)
            setfield!(config, key_sym, merged)
        else
            setfield!(config, key_sym, deepcopy(value))
        end
    end
end

"""
Deep merge two dictionaries without mutating the originals.
"""
function deep_merge_dict(base::Dict, updates::Dict)::Dict
    merged = deepcopy(base)
    for (key, value) in updates
        if isa(value, Dict) && haskey(merged, key) && isa(merged[key], Dict)
            merged[key] = deep_merge_dict(merged[key], value)
        else
            merged[key] = deepcopy(value)
        end
    end
    return merged
end

# ============================================================================
# CALIBRATION PROFILES
# ============================================================================

"""
Reusable parameter bundle with empirical targets for reproducibility.
"""
struct CalibrationProfile
    name::String
    description::String
    overrides::Dict{String,Any}
    target_metrics::Dict{String,Dict{String,Any}}
    source::String
end

function CalibrationProfile(;
    name::String,
    description::String,
    overrides::Dict{String,Any} = Dict{String,Any}(),
    target_metrics::Dict{String,Dict{String,Any}} = Dict{String,Dict{String,Any}}(),
    source::String = "built-in"
)
    CalibrationProfile(name, description, overrides, target_metrics, source)
end

"""
Return a config with the calibration overrides applied.
"""
function apply_calibration_profile(config::EmergentConfig, profile::Union{CalibrationProfile,Nothing})::EmergentConfig
    if isnothing(profile)
        return config
    end
    updated = copy_with_overrides(config, profile.overrides)
    updated.active_calibration = profile.name
    updated.calibration_targets = deepcopy(profile.target_metrics)
    return updated
end

"""
Fetch a built-in calibration profile by name (case-insensitive).
"""
function get_calibration_profile(name::String)::CalibrationProfile
    normalized = lowercase(strip(name))
    for profile in values(CALIBRATION_LIBRARY)
        if lowercase(profile.name) == normalized
            return profile
        end
    end
    available = join(keys(CALIBRATION_LIBRARY), ", ")
    error("Unknown calibration profile '$name'. Available: $available")
end

"""
Load a calibration profile definition from disk.
"""
function load_calibration_profile(path::String)::CalibrationProfile
    file_path = abspath(expanduser(path))
    if !isfile(file_path)
        error("Calibration file not found: $file_path")
    end
    payload = JSON3.read(read(file_path, String))
    overrides = get(payload, :overrides, get(payload, :parameters, Dict{String,Any}()))
    target_metrics = get(payload, :target_metrics, Dict{String,Dict{String,Any}}())
    name = get(payload, :name, splitext(basename(file_path))[1])
    description = get(payload, :description, "Custom calibration loaded from $(basename(file_path))")
    source = get(payload, :source, file_path)
    return CalibrationProfile(
        name=name,
        description=description,
        overrides=overrides,
        target_metrics=target_metrics,
        source=source
    )
end

# ============================================================================
# BUILT-IN CALIBRATION LIBRARY
# ============================================================================

const CALIBRATION_LIBRARY = Dict{String,CalibrationProfile}(
    "minimal_causal" => CalibrationProfile(
        name="minimal_causal",
        description="Minimal model for causal identification. Strips simulation to ~25 essential parameters. Monthly cadence.",
        overrides=Dict{String,Any}(
            "N_AGENTS" => 500,
            "N_ROUNDS" => 600,  # 600 months = 50 years (was 200 quarters)
            "N_RUNS" => 30,
            "INITIAL_CAPITAL" => 5_000_000.0,
            "SURVIVAL_CAPITAL_RATIO" => 0.40,
            "USE_NETWORK_EFFECTS" => false,
            "RETURN_NOISE_SCALE" => 0.20,
            "BLACK_SWAN_PROBABILITY" => 0.0,
            "MARKET_SHIFT_PROBABILITY" => 0.0,
            "KNOWLEDGE_DECAY_RATE" => 0.0,
        ),
        target_metrics=Dict{String,Dict{String,Any}}(
            "causal_identification" => Dict{String,Any}(
                "description" => "Minimal model for isolating AI tier causal effects",
                "fixed_tier_required" => true,
            ),
        ),
    ),
    "venture_baseline_2024" => CalibrationProfile(
        name="venture_baseline_2024",
        description="Anchors the simulation to US venture benchmarks: ~55% five-year survival, ~25% ten-year survival. Monthly cadence.",
        overrides=Dict{String,Any}(
            "BASE_OPERATIONAL_COST" => 6000.0,  # Monthly: calibrated for ~20% 10-yr survival
            "SURVIVAL_CAPITAL_RATIO" => 0.56,
            "INSOLVENCY_GRACE_ROUNDS" => 6,  # 6 months grace period
            "DISCOVERY_PROBABILITY" => 0.073,  # Monthly (was 0.22 quarterly)
            "INNOVATION_PROBABILITY" => 0.123,  # Monthly (was 0.37 quarterly)
            "OPPORTUNITY_RETURN_RANGE" => (0.92, 6.0),  # Better returns for 10-yr survival
            "INVESTMENT_SUCCESS_ROI_THRESHOLD" => 0.04,  # Monthly (was 0.12 quarterly)
            "MAX_INVESTMENT_FRACTION" => 0.04,  # Monthly (was 0.12 quarterly)
        ),
        target_metrics=Dict{String,Dict{String,Any}}(
            "survival_rate_month60" => Dict{String,Any}(
                "target" => 0.55,
                "tolerance" => 0.08,
                "source" => "BLS Business Employment Dynamics (2019 cohort). 5-year survival.",
            ),
            "survival_rate_month120" => Dict{String,Any}(
                "target" => 0.25,
                "tolerance" => 0.10,
                "source" => "BLS 10-year survival estimates.",
            ),
        ),
    ),
)
