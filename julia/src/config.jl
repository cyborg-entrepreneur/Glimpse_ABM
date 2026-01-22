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
    AGENT_AI_MODE::String = "emergent"
    N_AGENTS::Int = 1000
    INITIAL_CAPITAL::Float64 = 5_000_000.0
    INITIAL_CAPITAL_RANGE::Tuple{Float64,Float64} = (2_500_000.0, 10_000_000.0)
    SURVIVAL_THRESHOLD::Float64 = 230_000.0
    SURVIVAL_CAPITAL_RATIO::Float64 = 0.38
    INSOLVENCY_GRACE_ROUNDS::Int = 7
    RANDOM_SEED::Int = 42
    USE_NUMPY_RNG::Bool = false  # Use NumpyRNG for cross-language reproducibility

    BASE_OPERATIONAL_COST::Float64 = 50000.0
    COMPETITION_COST_MULTIPLIER::Float64 = 150.0
    OPERATING_RESERVE_MONTHS::Int = 3
    MAX_AGENT_KNOWLEDGE::Int = 90
    SECTOR_STRENGTH_PRUNE_THRESHOLD::Float64 = 0.1
    LIQUIDITY_RESERVE_FRACTION::Float64 = 0.29
    MAX_INVESTMENT_FRACTION::Float64 = 0.11
    TARGET_INVESTMENT_FRACTION::Float64 = 0.10
    AI_CREDIT_LINE_ROUNDS::Int = 30
    AI_TRUST_RESERVE_DISCOUNT::Float64 = 0.25

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
    N_ROUNDS::Int = 250
    N_RUNS::Int = 50
    BASE_OPPORTUNITIES::Int = 5
    DISCOVERY_PROBABILITY::Float64 = 0.30
    INNOVATION_PROBABILITY::Float64 = 0.42
    AI_HERDING_DECAY::Float64 = 1.0
    AI_SIGNAL_HISTORY::Int = 140

    # ========================================================================
    # INNOVATION ECONOMICS
    # ========================================================================
    INNOVATION_BASE_SPEND_RATIO::Float64 = 0.025
    INNOVATION_MAX_SPEND::Float64 = 8000.0
    INNOVATION_FAIL_RECOVERY_RATIO::Float64 = 0.12
    INNOVATION_SUCCESS_BASE_RETURN::Float64 = 0.25
    INNOVATION_SUCCESS_RETURN_MULTIPLIER::Tuple{Float64,Float64} = (1.8, 3.2)
    INNOVATION_RD_CAP_FRACTION::Float64 = 0.12
    INNOVATION_REUSE_PROBABILITY::Float64 = 0.22
    INNOVATION_REUSE_LOOKBACK::Int = 100
    INVESTMENT_SUCCESS_ROI_THRESHOLD::Float64 = 0.05
    BURN_HISTORY_WINDOW::Int = 3
    BURN_FAILURE_THRESHOLD::Float64 = 0.12
    BURN_LEVERAGE_CAP::Float64 = 0.75
    RETURN_OVERSUPPLY_PENALTY::Float64 = 0.52
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
    AI_LEVELS::Dict{String,AILevelConfig} = Dict(
        "none" => AILevelConfig(0.0, "none", 0.2, 0.18, 0.0),
        "basic" => AILevelConfig(45.0, "per_use", 0.48, 0.38, 6.0),
        "advanced" => AILevelConfig(1500.0, "subscription", 0.78, 0.68, 60.0),
        "premium" => AILevelConfig(14000.0, "subscription", 0.93, 0.88, 240.0),
    )

    AI_DOMAIN_CAPABILITIES::Dict{String,Dict{String,AIDomainCapability}} = Dict(
        "none" => Dict(
            "market_analysis" => AIDomainCapability(0.45, 0.22, 0.05),
            "technical_assessment" => AIDomainCapability(0.48, 0.20, -0.03),
            "uncertainty_evaluation" => AIDomainCapability(0.42, 0.25, -0.04),
            "innovation_potential" => AIDomainCapability(0.40, 0.24, 0.06),
        ),
        "basic" => Dict(
            "market_analysis" => AIDomainCapability(0.65, 0.18, 0.03),
            "technical_assessment" => AIDomainCapability(0.66, 0.17, -0.02),
            "uncertainty_evaluation" => AIDomainCapability(0.62, 0.18, -0.03),
            "innovation_potential" => AIDomainCapability(0.60, 0.20, 0.04),
        ),
        "advanced" => Dict(
            "market_analysis" => AIDomainCapability(0.89, 0.05, 0.02),
            "technical_assessment" => AIDomainCapability(0.91, 0.035, -0.01),
            "uncertainty_evaluation" => AIDomainCapability(0.90, 0.045, -0.02),
            "innovation_potential" => AIDomainCapability(0.89, 0.05, 0.015),
        ),
        "premium" => Dict(
            "market_analysis" => AIDomainCapability(0.985, 0.008, 0.002),
            "technical_assessment" => AIDomainCapability(0.992, 0.005, -0.001),
            "uncertainty_evaluation" => AIDomainCapability(0.990, 0.006, -0.002),
            "innovation_potential" => AIDomainCapability(0.988, 0.007, 0.001),
        ),
    )

    # ========================================================================
    # LEARNING PARAMETERS
    # ========================================================================
    UNCERTAINTY_LEARNING_ENABLED::Bool = true
    INITIAL_RESPONSE_VARIANCE::Float64 = 0.3
    LEARNING_RATE_BASE::Float64 = 0.1
    EXPLORATION_DECAY::Float64 = 0.995
    SOCIAL_LEARNING_WEIGHT::Float64 = 0.2
    MIN_FUNDING_FRACTION::Float64 = 0.25
    COST_OF_CAPITAL::Float64 = 0.06
    EXPLORATION_SENSITIVITY::Float64 = 1.5
    MAINTAIN_UNCERTAINTY_SENSITIVITY::Float64 = 0.5

    # ========================================================================
    # UNCERTAINTY AND MARKET DYNAMICS
    # ========================================================================
    BLACK_SWAN_PROBABILITY::Float64 = 0.05
    BOOM_TAIL_UNCERTAINTY_EXPONENT::Float64 = 1.08
    MARKET_VOLATILITY::Float64 = 0.25
    COMPETITION_EFFECT::Float64 = 0.25
    MARKET_SHIFT_PROBABILITY::Float64 = 0.09
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

    # NBER Business Cycle-calibrated transition matrix (quarterly rounds)
    # Source: NBER Business Cycle Dating Committee data 1945-2024
    # Average expansion: 64 months, average recession: 11 months
    # Crisis frequency: ~1 per decade, boom frequency: ~2-3 per decade
    MACRO_REGIME_TRANSITIONS::Dict{String,Dict{String,Float64}} = Dict(
        "crisis" => Dict("crisis" => 0.35, "recession" => 0.40, "normal" => 0.20, "growth" => 0.05, "boom" => 0.0),
        "recession" => Dict("crisis" => 0.06, "recession" => 0.40, "normal" => 0.42, "growth" => 0.10, "boom" => 0.02),
        "normal" => Dict("crisis" => 0.02, "recession" => 0.10, "normal" => 0.52, "growth" => 0.28, "boom" => 0.08),
        "growth" => Dict("crisis" => 0.01, "recession" => 0.04, "normal" => 0.20, "growth" => 0.50, "boom" => 0.25),
        "boom" => Dict("crisis" => 0.02, "recession" => 0.06, "normal" => 0.18, "growth" => 0.38, "boom" => 0.36),
    )

    MACRO_REGIME_RETURN_MODIFIERS::Dict{String,Float64} = Dict(
        "crisis" => 0.82, "recession" => 0.97, "normal" => 1.08, "growth" => 1.25, "boom" => 1.45
    )

    MACRO_REGIME_FAILURE_MODIFIERS::Dict{String,Float64} = Dict(
        "crisis" => 1.25, "recession" => 1.08, "normal" => 1.0, "growth" => 0.88, "boom" => 0.72
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
    DOWNSIDE_OVERSUPPLY_WEIGHT::Float64 = 0.65
    RETURN_LOWER_BOUND::Float64 = -1.0

    # ========================================================================
    # SCALING PARAMETERS
    # ========================================================================
    OPPORTUNITIES_PER_CAPITA::Float64 = 0.01
    DISCOVERY_RATE_SCALING::Float64 = 0.5
    MIN_OPPORTUNITIES::Int = 5
    POWER_LAW_SHAPE_A::Float64 = 3.0
    OPPORTUNITY_CAPITAL_REQUIREMENTS::Float64 = 10000.0

    # ========================================================================
    # LEARNING RATE
    # ========================================================================
    LEARNING_RATE::Float64 = 0.02

    # ========================================================================
    # DIAGNOSTICS
    # ========================================================================
    ENABLE_DEBUG_LOGS::Bool = false

    AI_TIER_REQUIREMENTS::Dict{String,Dict{String,Float64}} = Dict(
        "basic" => Dict(
            "trust" => 0.40, "min_successes" => 0.0, "min_success_rate" => 0.52,
            "min_roi" => 0.95, "min_accuracy" => 0.0, "max_cost_ratio" => 0.08,
        ),
        "advanced" => Dict(
            "trust" => 0.58, "min_successes" => 2.0, "min_success_rate" => 0.54,
            "recent_min_success_rate" => 0.52, "min_roi" => 1.02, "recent_min_roi" => 1.0,
            "min_accuracy" => 0.50, "max_cost_ratio" => 0.12,
        ),
        "premium" => Dict(
            "trust" => 0.68, "min_successes" => 4.0, "min_success_rate" => 0.60,
            "recent_min_success_rate" => 0.58, "min_roi" => 1.05, "recent_min_roi" => 1.02,
            "min_accuracy" => 0.55, "max_cost_ratio" => 0.14,
        ),
    )

    AI_TIER_RECENT_WINDOW::Int = 15
    AI_TIER_DEMOTE_MARGIN::Float64 = 0.06
    AI_TIER_NEIGHBOR_INFLUENCE::Float64 = 0.06

    AI_TIER_SCORING::Dict{String,Dict{String,Any}} = Dict(
        "basic" => Dict(
            "score_threshold" => 0.0, "score_slope" => 6.0, "trial_rounds" => 4,
            "weights" => Dict(
                "trust" => 0.6, "success" => 0.3, "success_gain" => 0.2,
                "recent_success_gain" => 0.2, "roi_gain" => 0.15, "recent_roi_gain" => 0.15,
                "cost" => 0.25, "accuracy" => 0.10, "usage" => 0.08, "bias" => -0.25,
            ),
        ),
        "advanced" => Dict(
            "score_threshold" => 0.08, "score_slope" => 6.0, "trial_rounds" => 4,
            "weights" => Dict(
                "trust" => 0.65, "success" => 0.32, "success_gain" => 0.27,
                "recent_success_gain" => 0.27, "roi_gain" => 0.22, "recent_roi_gain" => 0.22,
                "cost" => 0.18, "accuracy" => 0.15, "usage" => 0.12, "bias" => -0.28,
            ),
        ),
        "premium" => Dict(
            "score_threshold" => 0.18, "score_slope" => 6.5, "trial_rounds" => 5,
            "weights" => Dict(
                "trust" => 0.70, "success" => 0.38, "success_gain" => 0.32,
                "recent_success_gain" => 0.32, "roi_gain" => 0.28, "recent_roi_gain" => 0.28,
                "cost" => 0.22, "accuracy" => 0.20, "usage" => 0.14, "bias" => -0.35,
            ),
        ),
    )

    AI_TIER_SCORE_SMOOTHING::Float64 = 0.25
    AI_TIER_DEMOTION_COOLDOWN::Int = 12
    TRAIT_MOMENTUM::Float64 = 0.7
    AI_TRUST_ADJUSTMENT_RATE::Float64 = 0.1
    AI_SUBSCRIPTION_AMORTIZATION_ROUNDS::Int = 60
    AI_SUBSCRIPTION_FLOAT_BASE_ROUNDS::Int = 0
    AI_SUBSCRIPTION_FLOAT_MAX_ROUNDS::Int = 3

    # ========================================================================
    # ACTION SELECTION CONTROLS
    # ========================================================================
    ACTION_SELECTION_TEMPERATURE::Float64 = 0.45
    ACTION_SELECTION_NOISE::Float64 = 0.10
    ACTION_BIAS_SIGMA::Float64 = 0.05

    # ========================================================================
    # UNCERTAINTY VOLATILITY CONTROLS
    # ========================================================================
    UNCERTAINTY_SHORT_WINDOW::Int = 6
    UNCERTAINTY_SHORT_DECAY::Float64 = 0.0
    UNCERTAINTY_VOLATILITY_WINDOW::Int = 14
    UNCERTAINTY_VOLATILITY_DECAY::Float64 = 0.6
    UNCERTAINTY_VOLATILITY_SCALING::Float64 = 0.45
    UNCERTAINTY_AI_SWITCH_WEIGHT::Float64 = 0.09
    UNCERTAINTY_MARKET_RETURN_WEIGHT::Float64 = 0.14
    UNCERTAINTY_ACTION_VARIANCE_WEIGHT::Float64 = 0.14
    UNCERTAINTY_CROWDING_WEIGHT::Float64 = 0.18
    UNCERTAINTY_COMPETITIVE_WEIGHT::Float64 = 0.12

    # ========================================================================
    # KNOWLEDGE & INNOVATION
    # ========================================================================
    KNOWLEDGE_DECAY_RATE::Float64 = 0.075

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
    # Capital ranges calibrated for 24-36 round runway (18-27 months at quarterly cadence)
    SECTOR_PROFILES::Dict{String,SectorProfile} = Dict(
        "tech" => SectorProfile(
            (1.35, 3.10), log(1.95), 0.45, (0.22, 0.38),           # return params
            (0.3, 0.5), (0.04, 0.12), (300000.0, 1200000.0),       # failure, capital
            (15, 40), (0.55, 0.85), (0.08, 0.28),                  # maturity, margins
            # Empirically-calibrated fields (scaled for 40-60 round runway):
            # Reflects Series A/B rounds with 24-36 month runway before profitability
            (3_000_000.0, 6_000_000.0),  # initial_capital_range: 40-80 rounds runway
            (60_000.0, 90_000.0),        # operational_cost_range: BLS tech sector quarterly
            150_000.0,                    # survival_threshold: ~2 quarters operating expenses
            0.48,                         # innovation_probability: NSF 15-25% R&D, 52% USPTO
            (2.0, 4.0),                   # innovation_return_multiplier: high tech upside
            0.12,                         # knowledge_decay_rate: 2-3 year half-life
            1.2                           # competition_intensity: HHI 1500-2500
        ),
        "retail" => SectorProfile(
            (1.15, 2.10), log(1.45), 0.32, (0.18, 0.3),            # return params
            (0.2, 0.38), (0.04, 0.1), (50000.0, 400000.0),         # failure, capital
            (9, 30), (0.18, 0.42), (0.015, 0.08),                  # maturity, margins
            # Empirically-calibrated fields (scaled for 40-60 round runway):
            (2_200_000.0, 4_000_000.0),  # initial_capital_range: 40-73 rounds runway
            (40_000.0, 70_000.0),        # operational_cost_range: BLS retail sector quarterly
            130_000.0,                    # survival_threshold: ~2 quarters operating expenses
            0.32,                         # innovation_probability: NSF 1-3% R&D, 35% USPTO
            (1.6, 2.5),                   # innovation_return_multiplier: moderate returns
            0.07,                         # knowledge_decay_rate: 4-5 year half-life
            0.7                           # competition_intensity: HHI 500-1000
        ),
        "service" => SectorProfile(
            (1.25, 2.20), log(1.53), 0.36, (0.16, 0.28),           # return params
            (0.1, 0.28), (0.03, 0.08), (15000.0, 200000.0),        # failure, capital
            (6, 20), (0.45, 0.75), (0.12, 0.24),                   # maturity, margins
            # Empirically-calibrated fields (scaled for 40-60 round runway):
            (1_400_000.0, 2_500_000.0),  # initial_capital_range: 40-71 rounds runway
            (25_000.0, 45_000.0),        # operational_cost_range: BLS services sector quarterly
            70_000.0,                     # survival_threshold: ~2 quarters operating expenses
            0.38,                         # innovation_probability: NSF 3-8% R&D, 40% USPTO
            (1.6, 2.5),                   # innovation_return_multiplier: moderate returns
            0.05,                         # knowledge_decay_rate: 5-7 year half-life
            0.9                           # competition_intensity: HHI 800-1500
        ),
        "manufacturing" => SectorProfile(
            (1.30, 2.65), log(1.78), 0.4, (0.18, 0.3),             # return params
            (0.25, 0.42), (0.04, 0.1), (250000.0, 1500000.0),      # failure, capital
            (24, 72), (0.28, 0.48), (0.04, 0.18),                  # maturity, margins
            # Empirically-calibrated fields (scaled for 40-60 round runway):
            (4_000_000.0, 7_500_000.0),  # initial_capital_range: 40-75 rounds runway
            (80_000.0, 120_000.0),       # operational_cost_range: BLS manufacturing quarterly
            200_000.0,                    # survival_threshold: ~2 quarters operating expenses
            0.52,                         # innovation_probability: NSF 8-15% R&D, 58% USPTO
            (1.5, 2.8),                   # innovation_return_multiplier: incremental improvements
            0.03,                         # knowledge_decay_rate: 7-10 year half-life
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
    buffer_flush_interval::Int = 5
    write_intermediate_batches::Bool = true
    round_log_interval::Int = 25
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

    # ========================================================================
    # STRATEGIC MODE THRESHOLDS
    # ========================================================================
    STRATEGIC_MODE_THRESHOLDS::Dict{String,Float64} = Dict(
        "success_exploit" => 0.7,
        "success_diversify" => 0.3,
        "volatility_diversify" => 0.4,
    )
end

# Post-initialization: sync SECTORS from SECTOR_PROFILES keys
function initialize!(config::EmergentConfig)
    config.SECTORS = collect(keys(config.SECTOR_PROFILES))
    if config.N_AGENTS > 5000
        config.buffer_flush_interval = 50
        config.max_cache_size = 100000
        config.agent_history_depth = 5
        config.preallocate_arrays = true
        config.use_float32 = true
    end
    return config
end

"""
Compute integer opportunity targets even if overrides supply floats.
"""
function get_scaled_opportunities(config::EmergentConfig, n_agents::Int)::Int
    min_ops = ceil(Int, Float64(config.MIN_OPPORTUNITIES))
    per_capita = max(config.OPPORTUNITIES_PER_CAPITA, 0.0)
    scaled = ceil(Int, n_agents * per_capita)
    base_ops = ceil(Int, Float64(config.BASE_OPPORTUNITIES))
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
        description="Minimal model for causal identification. Strips simulation to ~25 essential parameters.",
        overrides=Dict{String,Any}(
            "N_AGENTS" => 500,
            "N_ROUNDS" => 200,
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
        description="Anchors the simulation to US venture benchmarks: ~55% five-year survival.",
        overrides=Dict{String,Any}(
            "BASE_OPERATIONAL_COST" => 70000.0,
            "SURVIVAL_CAPITAL_RATIO" => 0.56,
            "INSOLVENCY_GRACE_ROUNDS" => 6,
            "DISCOVERY_PROBABILITY" => 0.22,
            "INNOVATION_PROBABILITY" => 0.37,
            "OPPORTUNITY_RETURN_RANGE" => (0.85, 5.5),
            "INVESTMENT_SUCCESS_ROI_THRESHOLD" => 0.12,
            "MAX_INVESTMENT_FRACTION" => 0.12,
        ),
        target_metrics=Dict{String,Dict{String,Any}}(
            "survival_rate_round250" => Dict{String,Any}(
                "target" => 0.55,
                "tolerance" => 0.08,
                "source" => "BLS Business Employment Dynamics (2019 cohort).",
            ),
        ),
    ),
)
