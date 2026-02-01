"""
Core dataclasses used across the Glimpse ABM simulation.

This module defines the fundamental data structures that represent the
entities and interactions in the agent-based model of entrepreneurship
under Knightian uncertainty with AI augmentation.

Port of: glimpse_abm/models.py
"""

using Random
using Statistics

# ============================================================================
# INFORMATION
# ============================================================================

"""
Information about an opportunity - Enhanced for AI learning.
"""
mutable struct Information
    estimated_return::Float64
    estimated_uncertainty::Float64
    confidence::Float64
    insights::Vector{String}
    hidden_factors::Dict{String,Float64}
    domain::Union{String,Nothing}
    actual_accuracy::Union{Float64,Nothing}
    contains_hallucination::Bool
    bias_applied::Float64
    overconfidence_factor::Float64
end

function Information(;
    estimated_return::Float64,
    estimated_uncertainty::Float64,
    confidence::Float64,
    insights::Vector{String} = String[],
    hidden_factors::Dict{String,Float64} = Dict{String,Float64}(),
    domain::Union{String,Nothing} = nothing,
    actual_accuracy::Union{Float64,Nothing} = nothing,
    contains_hallucination::Bool = false,
    bias_applied::Float64 = 0.0,
    overconfidence_factor::Float64 = 1.0
)
    Information(
        estimated_return, estimated_uncertainty, confidence, insights,
        hidden_factors, domain, actual_accuracy, contains_hallucination,
        bias_applied, overconfidence_factor
    )
end

"""
Calculate information quality score.
"""
function quality_score(info::Information)::Float64
    base_quality = info.confidence * (1.0 - abs(get(info.hidden_factors, "bias", 0.0)))
    return base_quality / max(1.0, info.overconfidence_factor)
end

# ============================================================================
# OPPORTUNITY
# ============================================================================

"""
Market opportunity with latent characteristics that manifest through action.

This struct operationalizes the entrepreneurial concept that opportunities
are not simply "discovered" but are enacted through the interplay of
entrepreneurial action and market conditions.
"""
mutable struct Opportunity
    id::String
    latent_return_potential::Float64
    latent_failure_potential::Float64
    complexity::Float64
    discovered::Bool
    created_by::Union{Int,Nothing}
    discovery_round::Union{Int,Nothing}
    config::Union{EmergentConfig,Nothing}
    competition::Float64
    lifecycle_stage::String
    path_dependency::Float64
    sector::Union{String,Nothing}
    capital_requirements::Float64
    capital_history::Vector{Float64}
    realized_returns::Vector{Float64}
    time_to_maturity::Int
    market_share::Float64
    entry_barriers::Float64
    age::Int
    base_failure_potential::Union{Float64,Nothing}
    truly_unknown::Bool
    required_discovery_threshold::Float64
    combination_uncertainty::Float64
    combination_signature::Union{String,Nothing}
    market_impact::Float64
    origin_innovation_id::Union{String,Nothing}
    crowding_penalty::Float64
    component_scarcity::Float64
    # New fields for paradox mechanisms
    novelty_score::Float64              # 0.0 = established, 1.0 = very novel
    total_invested::Float64             # Track total investment for capacity constraints
    capacity::Float64                   # Max investment capacity
    disrupted_count::Int                # How many times disrupted by innovations
end

function Opportunity(;
    id::String,
    latent_return_potential::Float64 = 1.5,
    latent_failure_potential::Float64 = 0.7,
    complexity::Float64 = 0.5,
    discovered::Bool = false,
    created_by::Union{Int,Nothing} = nothing,
    discovery_round::Union{Int,Nothing} = nothing,
    config::Union{EmergentConfig,Nothing} = nothing,
    competition::Float64 = 0.0,
    lifecycle_stage::String = "emerging",
    path_dependency::Float64 = 0.0,
    sector::Union{String,Nothing} = nothing,
    capital_requirements::Float64 = 10000.0,
    capital_history::Vector{Float64} = Float64[],
    realized_returns::Vector{Float64} = Float64[],
    time_to_maturity::Int = 12,
    market_share::Float64 = 1.0,
    entry_barriers::Float64 = 1.0,
    age::Int = 0,
    base_failure_potential::Union{Float64,Nothing} = nothing,
    truly_unknown::Bool = false,
    required_discovery_threshold::Float64 = 0.0,
    combination_uncertainty::Float64 = 0.5,
    combination_signature::Union{String,Nothing} = nothing,
    market_impact::Float64 = 0.0,
    origin_innovation_id::Union{String,Nothing} = nothing,
    crowding_penalty::Float64 = 0.0,
    component_scarcity::Float64 = 0.5,
    # New fields for paradox mechanisms
    novelty_score::Float64 = 0.0,       # 0.0 = established, 1.0 = very novel
    total_invested::Float64 = 0.0,      # Track total investment
    capacity::Float64 = 500000.0,       # Default capacity
    disrupted_count::Int = 0            # Disruption counter
)
    opp = Opportunity(
        id, latent_return_potential, latent_failure_potential, complexity,
        discovered, created_by, discovery_round, config, competition,
        lifecycle_stage, path_dependency, sector, capital_requirements,
        capital_history, realized_returns, time_to_maturity, market_share,
        entry_barriers, age, base_failure_potential, truly_unknown,
        required_discovery_threshold, combination_uncertainty,
        combination_signature, market_impact, origin_innovation_id,
        crowding_penalty, component_scarcity,
        novelty_score, total_invested, capacity, disrupted_count
    )

    # Initialize base_failure_potential if not set
    if isnothing(opp.base_failure_potential)
        opp.base_failure_potential = opp.latent_failure_potential
    end

    # Set capacity based on config if available
    if !isnothing(config) && hasfield(typeof(config), :OPPORTUNITY_BASE_CAPACITY)
        variance = hasfield(typeof(config), :OPPORTUNITY_CAPACITY_VARIANCE) ? config.OPPORTUNITY_CAPACITY_VARIANCE : 0.3
        opp.capacity = config.OPPORTUNITY_BASE_CAPACITY * (1.0 + (rand() - 0.5) * 2 * variance)
    end

    return opp
end

"""
Convert latent return potential into a realized investment multiple.

This method operationalizes the core insight that entrepreneurial returns
are not predetermined but emerge through the interaction of opportunity
characteristics, market conditions, competitive dynamics, and execution
contingencies.
"""
function realized_return(
    opp::Opportunity,
    market_conditions::Dict{String,Any},
    investor_tier::Union{String,Nothing} = nothing;
    rng::Random.AbstractRNG = Random.default_rng()
)::Float64
    # Base multiple from latent potential
    base_multiple = clamp(coalesce(opp.latent_return_potential, 1.0), 0.3, 10.0)

    # Regime effects
    regime_return = Float64(get(market_conditions, "regime_return_multiplier", 1.0))
    base_multiple *= regime_return

    risk_signal = clamp(opp.latent_failure_potential, 0.05, 0.95)
    regime_failure = Float64(get(market_conditions, "regime_failure_multiplier", 1.0))
    risk_signal = clamp(risk_signal * regime_failure, 0.05, 0.95)

    # Lifecycle multipliers
    lifecycle_multipliers = Dict(
        "emerging" => 1.12,
        "growing" => 1.03,
        "mature" => 0.9,
        "declining" => 0.65,
    )
    lifecycle_factor = get(lifecycle_multipliers, opp.lifecycle_stage, 1.0)
    resilience = clamp(0.75 + 0.25 * (1.0 - risk_signal), 0.55, 1.45)

    # Sector clearing dynamics
    clearing_index = get(market_conditions, "sector_clearing_index", Dict{String,Float64}())
    aggregate_ratio = Float64(get(market_conditions, "aggregate_clearing_ratio", 1.0))
    sector_key = coalesce(opp.sector, "unknown")

    sector_adjustments = get(market_conditions, "sector_demand_adjustments", Dict{String,Any}())
    sector_adjust = get(sector_adjustments, sector_key, nothing)
    if isa(sector_adjust, Dict)
        base_multiple *= Float64(get(sector_adjust, "return", 1.0))
        risk_signal = clamp(risk_signal * Float64(get(sector_adjust, "failure", 1.0)), 0.05, 0.95)
    end

    sector_ratio = if !isempty(clearing_index)
        Float64(get(clearing_index, sector_key, aggregate_ratio))
    else
        aggregate_ratio
    end
    if !isfinite(sector_ratio)
        sector_ratio = 1.0
    end

    oversupply = max(0.0, sector_ratio - 1.0)
    undersupply = max(0.0, 1.0 - sector_ratio)

    config = opp.config
    oversupply_penalty = isnothing(config) ? 0.5 : config.RETURN_OVERSUPPLY_PENALTY
    undersupply_bonus = isnothing(config) ? 0.3 : config.RETURN_UNDERSUPPLY_BONUS

    clearing_adjustment = 1.0 - oversupply_penalty * oversupply + undersupply_bonus * undersupply
    clearing_adjustment = clamp(clearing_adjustment, 0.1, 3.5)

    base_mean = base_multiple * lifecycle_factor * resilience * clearing_adjustment

    # Uncertainty state effects
    unc_state = get(market_conditions, "uncertainty_state", Dict{String,Any}())
    agentic_state = if isa(unc_state, Dict)
        get(unc_state, "agentic_novelty", unc_state)
    else
        Dict{String,Any}()
    end

    novelty_signal = if isa(agentic_state, Dict)
        Float64(get(agentic_state, "novelty_potential", get(agentic_state, "level", 0.5)))
    else
        0.5
    end

    stored_scarcity = coalesce(opp.component_scarcity, 0.5)
    scarcity_signal = if isa(agentic_state, Dict)
        Float64(get(agentic_state, "component_scarcity", stored_scarcity))
    else
        stored_scarcity
    end
    scarcity_signal = clamp(0.5 * scarcity_signal + 0.5 * stored_scarcity, 0.0, 1.0)

    reuse_pressure = if isa(agentic_state, Dict)
        Float64(get(agentic_state, "reuse_pressure", 0.0))
    else
        0.0
    end

    # Scarcity ceiling
    scarcity_ceiling = 4.5 + 1.4 * scarcity_signal
    base_mean = clamp(base_mean, 0.2, min(15.0, scarcity_ceiling))

    # =========================================================================
    # CROWDING PENALTY MODEL
    # =========================================================================
    # Check which crowding model to use
    use_capacity_convexity = !isnothing(config) &&
        hasfield(typeof(config), :USE_CAPACITY_CONVEXITY_CROWDING) &&
        config.USE_CAPACITY_CONVEXITY_CROWDING

    if use_capacity_convexity
        # =====================================================================
        # CAPACITY-CONVEXITY CROWDING MODEL (New unified approach)
        # =====================================================================
        # Formula: penalty = λ · max(0, C/K - 1)^γ
        #          net_return = base_return · exp(-penalty)
        #
        # Properties:
        # - No penalty until crowding exceeds capacity K (healthy competition OK)
        # - Convex penalty beyond capacity (γ=2: "a little crowded is OK, very crowded is brutal")
        # - Exponential decay keeps returns positive
        #
        # Theoretical basis:
        # - Carrying capacity (K): ecological/economic concept of sustainable competition
        # - Convexity (γ): marginal harm from crowding increases with crowding level
        # - Exponential form: standard in utility theory, ensures positive returns
        # =====================================================================
        K = hasfield(typeof(config), :CROWDING_CAPACITY_K) ? config.CROWDING_CAPACITY_K : 1.5
        γ = hasfield(typeof(config), :CROWDING_CONVEXITY_GAMMA) ? config.CROWDING_CONVEXITY_GAMMA : 2.0
        λ = hasfield(typeof(config), :CROWDING_STRENGTH_LAMBDA) ? config.CROWDING_STRENGTH_LAMBDA : 0.50

        # Competition from opportunity-level crowding
        C = opp.competition

        # Also incorporate market-level crowding (crowding_index)
        crowding_metrics = get(market_conditions, "crowding_metrics", Dict{String,Any}())
        crowding_index = Float64(get(crowding_metrics, "crowding_index", 0.25))

        # Combine opportunity and market-level crowding
        # Market crowding adds to effective competition when everyone is doing same thing
        effective_C = C + crowding_index * 0.5

        # Calculate penalty: λ · max(0, C/K - 1)^γ
        excess_crowding = max(0.0, effective_C / K - 1.0)
        crowding_penalty = λ * (excess_crowding ^ γ)

        # Apply exponential decay to returns: exp(-penalty)
        # This keeps returns positive and gives nice scaling properties
        crowding_multiplier = exp(-crowding_penalty)

        # Apply penalty to base_mean
        base_mean *= crowding_multiplier

        # Scarcity and novelty adjustments (reduced weight since crowding now unified)
        scarcity_bonus = 0.10 * scarcity_signal
        novelty_relief = max(0.0, novelty_signal - 0.5) * 0.15
        base_mean *= (1.0 + scarcity_bonus + novelty_relief)

    else
        # =====================================================================
        # LEGACY CROWDING MODEL (for backward compatibility)
        # =====================================================================
        # Crowding penalties (matching Python - no COMPETITION_INTENSITY scaling here)
        combo_hhi = Float64(get(market_conditions, "combo_hhi", 0.0))
        reuse_penalty = coalesce(opp.crowding_penalty, 0.0)
        crowding_penalty_legacy = (0.25 * combo_hhi + 0.2 * reuse_penalty + 0.1 * reuse_pressure) * max(0.0, 1.0 - scarcity_signal)
        scarcity_bonus = 0.15 * scarcity_signal
        novelty_relief = max(0.0, novelty_signal - 0.5) * 0.3

        structural_adjustment = clamp(
            1.0 - crowding_penalty_legacy + scarcity_bonus + novelty_relief,
            0.5,
            1.45
        )
        base_mean *= structural_adjustment

        # Market-level crowding
        crowding_metrics = get(market_conditions, "crowding_metrics", Dict{String,Any}())
        crowding_index = Float64(get(crowding_metrics, "crowding_index", 0.25))

        crowd_threshold = isnothing(config) ? 0.35 : config.RETURN_DEMAND_CROWDING_THRESHOLD
        if crowding_index > crowd_threshold
            crowd_penalty_extra = clamp(1.0 - 0.25 * (crowding_index - crowd_threshold), 0.5, 1.0)
            base_mean *= crowd_penalty_extra
        end

        # Opportunity-level competition penalty (legacy square root model)
        opp_comp_threshold = isnothing(config) ? 0.2 : config.OPPORTUNITY_COMPETITION_THRESHOLD
        opp_comp_penalty_rate = isnothing(config) ? 0.5 : config.OPPORTUNITY_COMPETITION_PENALTY
        opp_comp_floor = isnothing(config) ? 0.1 : config.OPPORTUNITY_COMPETITION_FLOOR

        if opp.competition > opp_comp_threshold
            excess_competition = opp.competition - opp_comp_threshold
            sqrt_factor = sqrt(excess_competition)
            competition_penalty = 1.0 - opp_comp_penalty_rate * sqrt_factor
            base_mean *= clamp(competition_penalty, opp_comp_floor, 1.0)
        end
    end

    # =========================================================================
    # CAPACITY CONSTRAINTS (Separate from Crowding)
    # =========================================================================
    # NOTE: This is COMPLEMENTARY to the crowding model above, not redundant.
    # - CROWDING: Penalizes based on NUMBER of competitors (competition field)
    # - CAPACITY: Penalizes based on CAPITAL INVESTED (total_invested field)
    #
    # Both can apply: an opportunity can have few competitors but be capital-saturated,
    # or many competitors with plenty of capacity remaining.
    #
    # Real opportunities have limited capacity. First movers get better terms.
    # As investment approaches capacity, returns diminish.
    capacity_enabled = !isnothing(config) &&
        hasfield(typeof(config), :OPPORTUNITY_CAPACITY_ENABLED) &&
        config.OPPORTUNITY_CAPACITY_ENABLED

    if capacity_enabled && hasfield(typeof(opp), :capacity) && opp.capacity > 0
        utilization = opp.total_invested / opp.capacity
        penalty_start = !isnothing(config) && hasfield(typeof(config), :CAPACITY_PENALTY_START) ?
            config.CAPACITY_PENALTY_START : 0.7
        penalty_max = !isnothing(config) && hasfield(typeof(config), :CAPACITY_PENALTY_MAX) ?
            config.CAPACITY_PENALTY_MAX : 0.4

        if utilization > penalty_start && penalty_start < 1.0
            # Linear penalty from penalty_start to 1.0
            # At 70% utilized: 0% capacity penalty
            # At 100% utilized: max penalty (e.g., 40%)
            denom = 1.0 - penalty_start
            excess = denom > 0.001 ? (utilization - penalty_start) / denom : 1.0
            capacity_penalty = 1.0 - clamp(excess, 0.0, 1.0) * penalty_max
            base_mean *= clamp(capacity_penalty, 1.0 - penalty_max, 1.0)
        end
    end

    # =========================================================================
    # POWER LAW RETURN DISTRIBUTION
    # =========================================================================
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
    # α ≈ 2.0: Very heavy tails (early-stage VC)
    # α ≈ 2.5: Moderate tails (growth equity)
    # α ≈ 3.0: Lighter tails (late-stage/buyout)
    # =========================================================================

    volatility = Float64(get(market_conditions, "volatility", 0.0))
    regime = get(market_conditions, "regime", "normal")

    # Power law shape parameter - lower = heavier tails
    alpha = isnothing(config) ? 2.5 : config.POWER_LAW_SHAPE_A

    # Adjust alpha based on regime (more extreme outcomes in volatile regimes)
    regime_alpha_adjust = Dict(
        "crisis" => -0.4,    # Heavier tails in crisis
        "recession" => -0.2,
        "normal" => 0.0,
        "growth" => -0.1,    # Slightly heavier in growth (more unicorns)
        "boom" => -0.3       # Heavier in boom (bubbles create outliers)
    )
    alpha += get(regime_alpha_adjust, regime, 0.0)
    alpha = clamp(alpha, 1.5, 4.0)  # Keep alpha in reasonable range

    # Adjust alpha based on volatility (more volatile = heavier tails)
    alpha -= volatility * 0.3
    alpha = clamp(alpha, 1.5, 4.0)

    # The minimum return (x_m) scales with opportunity quality
    # Higher quality opportunities have higher floor returns
    # Increased floor (0.5 multiplier) to ensure median ~1× for sustainable survival
    x_min = clamp(base_mean * 0.5, 0.3, 3.0)

    # Sample from Pareto distribution: X = x_m / U^(1/α)
    u = rand(rng)
    # Avoid division by zero and extreme values
    u = clamp(u, 1e-6, 1.0 - 1e-6)
    pareto_draw = x_min / (u ^ (1.0 / alpha))

    # Apply quality scaling - better opportunities shift the distribution up
    quality_scale = base_mean / max(x_min, 0.1)
    scaled_return = pareto_draw * clamp(quality_scale * 0.6, 0.6, 2.5)

    # Downside risk adjustment (oversupply, risk signal)
    downside_weight = isnothing(config) ? 0.65 : config.DOWNSIDE_OVERSUPPLY_WEIGHT
    downside = clamp(0.2 + risk_signal * 0.4 + downside_weight * oversupply, 0.0, 2.0)

    # Beta shock for additional variance
    shock = rand(rng, Beta(1.5, 2.5))
    scaled_return *= clamp(1.0 - downside * shock * 0.5, 0.3, 1.2)

    # Final bounds
    # Allow higher upper bound for power law outliers (unicorns)
    # Scarcity and novelty increase the ceiling for rare outcomes
    scarcity_headroom = 10.0 + 5.0 * max(0.0, scarcity_signal + novelty_signal - 0.8)
    upper_bound = clamp(scarcity_headroom, 5.0, 50.0)  # Allow up to 50× for true unicorns
    # Lower bound: 0.0 = total loss, 0.5 = 50% loss, 1.0 = break-even
    # Note: RETURN_LOWER_BOUND should be ≥0 to represent minimum return multiple
    lower_bound = isnothing(config) ? 0.0 : clamp(config.RETURN_LOWER_BOUND, 0.0, 1.0)

    return clamp(scaled_return, lower_bound, upper_bound)
end

"""
Update opportunity lifecycle based on adoption rate.
"""
function update_lifecycle!(opp::Opportunity, adoption_rate::Float64)
    if opp.lifecycle_stage == "emerging" && adoption_rate > 0.2
        opp.lifecycle_stage = "growing"
    elseif opp.lifecycle_stage == "growing" && adoption_rate > 0.6
        opp.lifecycle_stage = "mature"
    elseif opp.lifecycle_stage == "mature" && adoption_rate > 0.8
        opp.lifecycle_stage = "declining"
    end
end

# ============================================================================
# INNOVATION
# ============================================================================

"""
Innovation artifact produced by the innovation engine.
"""
mutable struct Innovation
    id::String
    type::String
    knowledge_components::Vector{String}
    novelty::Float64
    quality::Float64
    round_created::Int
    creator_id::Int
    success::Union{Bool,Nothing}
    market_impact::Union{Float64,Nothing}
    ai_assisted::Bool
    ai_domains_used::Vector{String}
    sector::Union{String,Nothing}
    combination_signature::Union{String,Nothing}
    cash_multiple::Union{Float64,Nothing}
    scarcity::Union{Float64,Nothing}
    is_new_combination::Bool
    ai_level_used::String
end

function Innovation(;
    id::String,
    type::String,
    knowledge_components::Vector{String},
    novelty::Float64,
    quality::Float64,
    round_created::Int,
    creator_id::Int,
    success::Union{Bool,Nothing} = nothing,
    market_impact::Union{Float64,Nothing} = nothing,
    ai_assisted::Bool = false,
    ai_domains_used::Vector{String} = String[],
    sector::Union{String,Nothing} = nothing,
    combination_signature::Union{String,Nothing} = nothing,
    cash_multiple::Union{Float64,Nothing} = nothing,
    scarcity::Union{Float64,Nothing} = nothing,
    is_new_combination::Bool = false,
    ai_level_used::String = "none"
)
    Innovation(
        id, type, knowledge_components, novelty, quality, round_created,
        creator_id, success, market_impact, ai_assisted, ai_domains_used,
        sector, combination_signature, cash_multiple, scarcity,
        is_new_combination, ai_level_used
    )
end

"""
Estimate market potential based on innovation attributes.
"""
function calculate_potential(innov::Innovation, market_conditions::Dict{String,Any})::Float64
    base_potential = innov.quality * (0.5 + 0.5 * innov.novelty)

    type_modifiers = Dict(
        "incremental" => 0.7,
        "architectural" => 0.9,
        "radical" => 1.2,
        "disruptive" => 1.5,
    )
    type_factor = get(type_modifiers, innov.type, 1.0)

    regime = get(market_conditions, "regime", "normal")
    market_factor = if regime == "growth"
        1.2
    elseif regime == "crisis"
        innov.type == "disruptive" ? 1.5 : 0.7
    else
        1.0
    end

    ai_factor = innov.ai_assisted ? 1.1 : 1.0

    return base_potential * type_factor * market_factor * ai_factor
end

# ============================================================================
# KNOWLEDGE
# ============================================================================

"""
Represents a piece of knowledge or capability.
"""
mutable struct Knowledge
    id::String
    domain::String
    level::Float64
    discovered_round::Int
    discovered_by::Union{Int,Nothing}
    parent_knowledge::Vector{String}
end

function Knowledge(;
    id::String,
    domain::String,
    level::Float64,
    discovered_round::Int,
    discovered_by::Union{Int,Nothing} = nothing,
    parent_knowledge::Vector{String} = String[]
)
    Knowledge(id, domain, level, discovered_round, discovered_by, parent_knowledge)
end

"""
Calculate compatibility between two knowledge pieces.
"""
function compatibility_with(k1::Knowledge, k2::Knowledge)::Float64
    if k1.domain == k2.domain
        return 0.8 + 0.2 * (1.0 - abs(k1.level - k2.level))
    end

    domain_adjacency = Dict(
        "technology" => ["process", "market"],
        "market" => ["technology", "business_model"],
        "process" => ["technology", "business_model"],
        "business_model" => ["market", "process"],
    )

    adjacent = get(domain_adjacency, k1.domain, String[])
    if k2.domain in adjacent
        return 0.4 + 0.2 * (1.0 - abs(k1.level - k2.level))
    end

    return 0.1 + 0.1 * (1.0 - abs(k1.level - k2.level))
end

# ============================================================================
# AI ANALYSIS
# ============================================================================

"""
Enhanced AI analysis with potential errors and biases.
"""
struct AIAnalysis
    estimated_return::Float64
    estimated_uncertainty::Float64
    confidence::Float64
    insights::Vector{String}
    hidden_factors::Dict{String,Float64}
    actual_accuracy::Float64
    contains_hallucination::Bool
    bias_applied::Float64
    domain::String
    false_insights::Vector{String}
    overconfidence_factor::Float64
end

function AIAnalysis(;
    estimated_return::Float64,
    estimated_uncertainty::Float64,
    confidence::Float64,
    insights::Vector{String} = String[],
    hidden_factors::Dict{String,Float64} = Dict{String,Float64}(),
    actual_accuracy::Float64,
    contains_hallucination::Bool,
    bias_applied::Float64,
    domain::String,
    false_insights::Vector{String} = String[],
    overconfidence_factor::Float64
)
    AIAnalysis(
        estimated_return, estimated_uncertainty, confidence, insights,
        hidden_factors, actual_accuracy, contains_hallucination,
        bias_applied, domain, false_insights, overconfidence_factor
    )
end

# ============================================================================
# AI LEARNING PROFILE
# ============================================================================

"""
Agent's learned understanding of AI capabilities.
"""
mutable struct AILearningProfile
    domain_trust::Dict{String,Float64}
    accuracy_estimates::Dict{String,Vector{Float64}}
    hallucination_experiences::Dict{String,Int}
    usage_count::Dict{String,Int}
    # Bayesian beliefs about AI tier effectiveness (alpha, beta parameters for Beta distribution)
    tier_beliefs::Dict{String,Dict{String,Float64}}
end

const DEFAULT_DOMAINS = ["market_analysis", "technical_assessment", "uncertainty_evaluation", "innovation_potential"]
const AI_TIERS = ["none", "basic", "advanced", "premium"]

# FIXED: Neutral priors for all tiers - agents learn from experience
# Previously had built-in skepticism: none=0.52, basic=0.48, advanced=0.35, premium=0.26
# Now all tiers start at 0.5 and agents update beliefs based on actual outcomes
# This lets tier preferences emerge from experience, not hardcoded prior bias
const DEFAULT_TIER_PRIORS = Dict(
    "none" => Dict("alpha" => 2.0, "beta" => 2.0),      # Prior mean = 0.50 (neutral)
    "basic" => Dict("alpha" => 2.0, "beta" => 2.0),     # Prior mean = 0.50 (neutral)
    "advanced" => Dict("alpha" => 2.0, "beta" => 2.0),  # Prior mean = 0.50 (neutral)
    "premium" => Dict("alpha" => 2.0, "beta" => 2.0)    # Prior mean = 0.50 (neutral)
)

function AILearningProfile()
    # Initialize tier beliefs with default priors
    tier_beliefs = Dict{String,Dict{String,Float64}}()
    for tier in AI_TIERS
        tier_beliefs[tier] = copy(DEFAULT_TIER_PRIORS[tier])
    end

    AILearningProfile(
        Dict(d => 0.5 for d in DEFAULT_DOMAINS),
        Dict(d => Float64[] for d in DEFAULT_DOMAINS),
        Dict(d => 0 for d in DEFAULT_DOMAINS),
        Dict(d => 0 for d in DEFAULT_DOMAINS),
        tier_beliefs
    )
end

"""
Update trust based on AI accuracy.
"""
function update_trust!(profile::AILearningProfile, domain::String, was_accurate::Bool; magnitude::Float64 = 0.1)
    if was_accurate
        profile.domain_trust[domain] = min(1.0, profile.domain_trust[domain] + magnitude)
    else
        profile.domain_trust[domain] = max(0.0, profile.domain_trust[domain] - magnitude * 1.5)
    end
end

"""
Update Bayesian beliefs about AI tier effectiveness based on realized return multiplier.

Uses continuous evidence derived from the return multiplier (matches Python implementation).
Higher returns provide stronger positive evidence, lower returns provide negative evidence.
The evidence is computed as: stable_sigmoid((multiplier - 1.0) * 2.5), clamped to [0.02, 0.98].
"""
function update_tier_belief!(profile::AILearningProfile, tier::String, realized_multiplier::Float64)
    if !haskey(profile.tier_beliefs, tier)
        # Initialize with default prior if tier not found
        prior = get(DEFAULT_TIER_PRIORS, tier, Dict("alpha" => 2.0, "beta" => 2.0))
        profile.tier_beliefs[tier] = copy(prior)
    end

    belief = profile.tier_beliefs[tier]

    # Compute continuous evidence from realized multiplier (matches Python lines 1551-1554)
    multiplier = clamp(realized_multiplier, 0.0, 3.0)
    evidence = clamp(stable_sigmoid((multiplier - 1.0) * 2.5), 0.02, 0.98)

    belief["alpha"] += evidence
    belief["beta"] += (1.0 - evidence)

    # Cap beliefs to prevent extreme certainty (keep some exploration)
    max_strength = 50.0
    total = belief["alpha"] + belief["beta"]
    if total > max_strength
        scale = max_strength / total
        belief["alpha"] *= scale
        belief["beta"] *= scale
    end
end

"""
Update Bayesian beliefs about AI tier effectiveness based on binary success/failure.

This is a convenience method that converts success to a multiplier estimate:
- Success: multiplier = 1.5 (positive evidence)
- Failure: multiplier = 0.5 (negative evidence)
"""
function update_tier_belief!(profile::AILearningProfile, tier::String, was_successful::Bool)
    # Convert binary outcome to estimated multiplier for continuous update
    estimated_multiplier = was_successful ? 1.5 : 0.5
    update_tier_belief!(profile, tier, estimated_multiplier)
end

"""
Get the posterior mean belief about a tier's effectiveness.
"""
function get_tier_belief_mean(profile::AILearningProfile, tier::String)::Float64
    if !haskey(profile.tier_beliefs, tier)
        return 0.5  # Neutral prior
    end
    belief = profile.tier_beliefs[tier]
    total = belief["alpha"] + belief["beta"]
    return total > 0 ? belief["alpha"] / total : 0.5
end

"""
Determine whether to use AI for a specific domain.
"""
function should_use_ai_for_domain(
    profile::AILearningProfile,
    domain::String,
    ai_cost::Float64,
    capital_buffer::Float64;
    cost_type::String = "per_use"
)::Bool
    trust = get(profile.domain_trust, domain, 0.5)
    usage = get(profile.usage_count, domain, 0)
    exploration_bonus = usage < 5 ? 0.3 : 0.0

    cost_factor = if cost_type == "subscription"
        1.0
    else
        available_buffer = max(capital_buffer - ai_cost, 1.0)
        cost_ratio = ai_cost / available_buffer
        max(0.0, 1.0 - cost_ratio)
    end

    threshold = 0.4 - exploration_bonus
    decision_score = (trust + exploration_bonus) * cost_factor
    return decision_score > threshold
end

"""
Get adjusted confidence based on learned AI trust.
"""
function get_adjusted_confidence(profile::AILearningProfile, ai_confidence::Float64, domain::String)::Float64
    trust = get(profile.domain_trust, domain, 0.5)
    hall_count = get(profile.hallucination_experiences, domain, 0)
    usage = get(profile.usage_count, domain, 1)
    hallucination_rate = hall_count / max(1, usage)
    adjustment = trust * (1.0 - hallucination_rate)
    return ai_confidence * adjustment
end

# ============================================================================
# PERFORMANCE TRACKER
# ============================================================================

"""
Tracks capital deployment and returns by action and AI tier for ROE/ROIC metrics.
"""
mutable struct PerformanceTracker
    initial_equity::Float64
    deployed_by_action::Dict{String,Float64}
    returned_by_action::Dict{String,Float64}
    deployments_by_ai::Dict{String,Dict{String,Float64}}
    returns_by_ai::Dict{String,Dict{String,Float64}}
    roi_events::Vector{Dict{String,Any}}
end

function PerformanceTracker(initial_equity::Float64)
    PerformanceTracker(
        initial_equity,
        Dict{String,Float64}(),
        Dict{String,Float64}(),
        Dict{String,Dict{String,Float64}}(),
        Dict{String,Dict{String,Float64}}(),
        Dict{String,Any}[]
    )
end

function normalize_action(action::Union{String,Nothing})::String
    return isnothing(action) ? "overall" : lowercase(action)
end

function normalize_ai_level_tracker(ai_level::Union{String,Nothing})::String
    return isnothing(ai_level) ? "none" : lowercase(strip(ai_level))
end

"""
Record capital deployment for an action.
"""
function record_deployment!(
    tracker::PerformanceTracker,
    action::String,
    amount::Float64;
    ai_level::String = "none",
    round_num::Union{Int,Nothing} = nothing
)
    if amount <= 0
        return
    end

    action_key = normalize_action(action)
    ai_key = normalize_ai_level_tracker(ai_level)

    tracker.deployed_by_action[action_key] = get(tracker.deployed_by_action, action_key, 0.0) + amount
    tracker.deployed_by_action["overall"] = get(tracker.deployed_by_action, "overall", 0.0) + amount

    if !haskey(tracker.deployments_by_ai, action_key)
        tracker.deployments_by_ai[action_key] = Dict{String,Float64}()
    end
    tracker.deployments_by_ai[action_key][ai_key] = get(tracker.deployments_by_ai[action_key], ai_key, 0.0) + amount

    if !haskey(tracker.deployments_by_ai, "overall")
        tracker.deployments_by_ai["overall"] = Dict{String,Float64}()
    end
    tracker.deployments_by_ai["overall"][ai_key] = get(tracker.deployments_by_ai["overall"], ai_key, 0.0) + amount

    if !isnothing(round_num)
        push!(tracker.roi_events, Dict{String,Any}(
            "round" => round_num,
            "type" => "deployment",
            "action" => action_key,
            "ai_level" => ai_key,
            "amount" => amount
        ))
    end
end

"""
Record capital return for an action.
"""
function record_return!(
    tracker::PerformanceTracker,
    action::String,
    amount::Float64;
    ai_level::String = "none",
    round_num::Union{Int,Nothing} = nothing
)
    amount = max(0.0, amount)

    action_key = normalize_action(action)
    ai_key = normalize_ai_level_tracker(ai_level)

    tracker.returned_by_action[action_key] = get(tracker.returned_by_action, action_key, 0.0) + amount
    tracker.returned_by_action["overall"] = get(tracker.returned_by_action, "overall", 0.0) + amount

    if !haskey(tracker.returns_by_ai, action_key)
        tracker.returns_by_ai[action_key] = Dict{String,Float64}()
    end
    tracker.returns_by_ai[action_key][ai_key] = get(tracker.returns_by_ai[action_key], ai_key, 0.0) + amount

    if !haskey(tracker.returns_by_ai, "overall")
        tracker.returns_by_ai["overall"] = Dict{String,Float64}()
    end
    tracker.returns_by_ai["overall"][ai_key] = get(tracker.returns_by_ai["overall"], ai_key, 0.0) + amount

    if !isnothing(round_num)
        push!(tracker.roi_events, Dict{String,Any}(
            "round" => round_num,
            "type" => "return",
            "action" => action_key,
            "ai_level" => ai_key,
            "amount" => amount
        ))
    end
end

"""
Compute Return on Invested Capital for an action.
"""
function compute_roic(tracker::PerformanceTracker, action::Union{String,Nothing} = nothing)::Float64
    key = normalize_action(action)
    deployed = get(tracker.deployed_by_action, key, 0.0)
    if deployed <= 0
        return 0.0
    end
    returned = get(tracker.returned_by_action, key, 0.0)
    return (returned - deployed) / deployed
end

"""
Compute Return on Equity.
"""
function compute_roe(tracker::PerformanceTracker, current_capital::Float64)::Float64
    denominator = max(tracker.initial_equity, 1e-9)
    return (current_capital - tracker.initial_equity) / denominator
end

"""
Get performance snapshot.
"""
function performance_snapshot(tracker::PerformanceTracker, current_capital::Float64)::Dict{String,Float64}
    stats = Dict{String,Float64}(
        "roe" => compute_roe(tracker, current_capital),
        "roic_overall" => compute_roic(tracker, nothing)
    )

    for action in ["invest", "innovate", "explore"]
        stats["roic_$(action)"] = compute_roic(tracker, action)
        stats["capital_deployed_$(action)"] = get(tracker.deployed_by_action, action, 0.0)
        stats["capital_returned_$(action)"] = get(tracker.returned_by_action, action, 0.0)
    end

    stats["capital_deployed_total"] = get(tracker.deployed_by_action, "overall", 0.0)
    stats["capital_returned_total"] = get(tracker.returned_by_action, "overall", 0.0)

    return stats
end

# ============================================================================
# PORTFOLIO
# ============================================================================

"""
Investment record within a portfolio.
"""
mutable struct InvestmentRecord
    opportunity_id::String
    amount::Float64
    sector::String
    entry_round::Int
    maturation_round::Int
    time_to_maturity::Int
    returns::Float64
    matured::Bool
    success::Bool
    raw_success::Bool
    defaulted::Bool
    ai_level_used::String
    estimated_return_at_investment::Float64
    estimated_uncertainty_at_investment::Float64
    ai_confidence_at_investment::Float64
    decision_confidence::Float64
    realized_multiplier::Float64
    realized_roi::Float64
    counterfactual_roi::Float64
end

function InvestmentRecord(;
    opportunity_id::String,
    amount::Float64,
    sector::String,
    entry_round::Int,
    maturation_round::Int,
    time_to_maturity::Int,
    ai_level_used::String = "none",
    estimated_return::Float64 = 0.0,
    estimated_uncertainty::Float64 = 0.5,
    ai_confidence::Float64 = 0.5,
    decision_confidence::Float64 = 0.5
)
    InvestmentRecord(
        opportunity_id, amount, sector, entry_round, maturation_round,
        time_to_maturity, 0.0, false, false, false, false, ai_level_used,
        estimated_return, estimated_uncertainty, ai_confidence,
        clamp(decision_confidence, 0.05, 0.99), 0.0, 0.0, 0.0
    )
end

"""
Manages an agent's collection of investments with proper maturation tracking.
"""
mutable struct Portfolio
    active_investments::Dict{String,InvestmentRecord}
    pending_investments::Dict{String,InvestmentRecord}
    matured_investments::Vector{InvestmentRecord}
    past_investments::Vector{Dict{String,Any}}
    total_invested::Float64
    locked_capital::Float64
    diversification_score::Float64
    config::Union{EmergentConfig,Nothing}
end

function Portfolio(config::Union{EmergentConfig,Nothing} = nothing)
    Portfolio(
        Dict{String,InvestmentRecord}(),
        Dict{String,InvestmentRecord}(),
        InvestmentRecord[],
        Dict{String,Any}[],
        0.0,
        0.0,
        0.0,
        config
    )
end

"""
Add a new investment to the portfolio.
"""
function add_investment!(
    portfolio::Portfolio,
    opp_id::String,
    amount::Float64,
    sector::String,
    round_num::Int,
    time_to_maturity::Int;
    ai_level_used::String = "none",
    estimated_return::Float64 = 0.0,
    estimated_uncertainty::Float64 = 0.5,
    ai_confidence::Float64 = 0.5,
    decision_confidence::Float64 = 0.5
)
    maturation_round = round_num + time_to_maturity

    investment = InvestmentRecord(
        opportunity_id = opp_id,
        amount = amount,
        sector = sector,
        entry_round = round_num,
        maturation_round = maturation_round,
        time_to_maturity = time_to_maturity,
        ai_level_used = ai_level_used,
        estimated_return = estimated_return,
        estimated_uncertainty = estimated_uncertainty,
        ai_confidence = ai_confidence,
        decision_confidence = decision_confidence
    )

    # Handle duplicate keys
    investment_key = opp_id
    if haskey(portfolio.pending_investments, investment_key)
        suffix = 1
        while haskey(portfolio.pending_investments, "$(opp_id)#$(suffix)")
            suffix += 1
        end
        investment_key = "$(opp_id)#$(suffix)"
    end

    portfolio.pending_investments[investment_key] = investment
    portfolio.active_investments[investment_key] = investment
    portfolio.total_invested += amount
    portfolio.locked_capital += amount

    update_diversification!(portfolio)
end

"""
Check for matured investments and process outcomes.
"""
function check_matured_investments!(
    portfolio::Portfolio,
    current_round::Int,
    market_conditions::Dict{String,Any};
    rng::Random.AbstractRNG = Random.default_rng()
)::Vector{Dict{String,Any}}
    newly_matured = Dict{String,Any}[]

    for (investment_key, investment) in collect(portfolio.pending_investments)
        if current_round >= investment.maturation_round
            # Calculate failure probability
            raw_risk = 0.5  # Default risk
            risk = clamp(raw_risk, 0.05, 0.95)

            # Market adjustments
            regime_failure = get(market_conditions, "regime_failure_multiplier", 1.0)
            base_failure = 0.05 + 0.5 * risk * regime_failure

            # Crowding adjustments
            crowding_metrics = get(market_conditions, "crowding_metrics", Dict{String,Any}())
            crowd_idx = get(crowding_metrics, "crowding_index", 0.25)
            crowd_threshold = 0.35
            crowd_multiplier = crowd_idx > crowd_threshold ? 1.0 + 0.25 * (crowd_idx - crowd_threshold) : 1.0

            failure_chance = clamp(base_failure * crowd_multiplier, 0.05, 0.9)
            success = rand(rng) >= failure_chance

            amount = investment.amount
            if success
                # Successful investment
                realized_multiplier = 1.0 + 0.2 * rand(rng)  # Simple return model
                capital_returned = amount * realized_multiplier
                investment.defaulted = false
                investment.raw_success = true
            else
                # Failed investment - partial recovery
                recovery_ratio = clamp(0.2 + 0.2 * rand(rng), 0.0, 0.4)
                realized_multiplier = recovery_ratio
                capital_returned = amount * recovery_ratio
                investment.defaulted = rand(rng) < 0.2 * risk
                investment.raw_success = false
            end

            investment.returns = capital_returned
            investment.matured = true
            investment.realized_multiplier = realized_multiplier
            investment.realized_roi = realized_multiplier - 1.0
            investment.success = success && investment.realized_roi >= 0.0

            # Counterfactual ROI
            investment.counterfactual_roi = investment.realized_roi

            # Move to matured
            push!(portfolio.matured_investments, investment)
            delete!(portfolio.pending_investments, investment_key)
            delete!(portfolio.active_investments, investment_key)
            portfolio.locked_capital -= investment.amount

            push!(newly_matured, Dict{String,Any}(
                "opportunity_id" => investment.opportunity_id,
                "investment_amount" => investment.amount,
                "capital_returned" => capital_returned,
                "success" => investment.success,
                "raw_success" => investment.raw_success,
                "defaulted" => investment.defaulted,
                "net_return" => capital_returned - investment.amount,
                "ai_level_used" => investment.ai_level_used,
                "realized_multiplier" => realized_multiplier,
                "decision_confidence" => investment.decision_confidence
            ))
        end
    end

    if !isempty(newly_matured)
        update_diversification!(portfolio)
    end

    return newly_matured
end

"""
Archive matured investments to lightweight snapshots.
"""
function archive_matured_history!(portfolio::Portfolio)
    for investment in portfolio.matured_investments
        snapshot = Dict{String,Any}(
            "opportunity_id" => investment.opportunity_id,
            "sector" => investment.sector,
            "entry_round" => investment.entry_round,
            "maturation_round" => investment.maturation_round,
            "time_to_maturity" => investment.time_to_maturity,
            "investment_amount" => investment.amount,
            "capital_returned" => investment.returns,
            "success" => investment.success,
            "ai_level_used" => investment.ai_level_used
        )
        push!(portfolio.past_investments, snapshot)
    end
    empty!(portfolio.matured_investments)
end

"""
Update portfolio diversification score.
"""
function update_diversification!(portfolio::Portfolio)
    all_investments = merge(portfolio.pending_investments, portfolio.active_investments)

    if isempty(all_investments)
        portfolio.diversification_score = 0.0
        return
    end

    sector_amounts = Dict{String,Float64}()
    total_amount = 0.0

    for inv in values(all_investments)
        amount = inv.amount
        if amount <= 0
            continue
        end
        sector_amounts[inv.sector] = get(sector_amounts, inv.sector, 0.0) + amount
        total_amount += amount
    end

    if total_amount <= 0 || isempty(sector_amounts)
        portfolio.diversification_score = 0.0
        return
    end

    # Compute entropy-based diversification
    proportions = [amt / total_amount for amt in values(sector_amounts)]
    n_sectors = length(sector_amounts)
    max_entropy = n_sectors > 1 ? log(n_sectors) : 1.0  # Use actual sector count

    if max_entropy <= 0
        portfolio.diversification_score = 0.0
        return
    end

    entropy = -sum(p * log(p + 1e-12) for p in proportions)
    diversity = entropy / max_entropy

    portfolio.diversification_score = clamp(diversity, 0.0, 1.0)
end

"""
Get available capital for new investments.
"""
function get_available_capital(portfolio::Portfolio, total_capital::Float64)::Float64
    return max(0.0, total_capital - portfolio.locked_capital)
end

"""
Check if portfolio has pending investments.
"""
function has_pending_investments(portfolio::Portfolio)::Bool
    return !isempty(portfolio.pending_investments)
end

"""
Get portfolio size (number of active investments).
"""
function portfolio_size(portfolio::Portfolio)::Int
    return length(portfolio.active_investments)
end

# ============================================================================
# UNCERTAINTY RESPONSE PROFILE
# ============================================================================

"""
Tracks how an agent learns to respond to different uncertainty types.

Implements adaptive uncertainty response mechanism where entrepreneurs develop
heterogeneous response patterns to each Knightian uncertainty dimension.
"""
mutable struct UncertaintyResponseProfile
    exploration_rate::Float64
    memory_limit::Int
    learning_rate::Float64
    response_weights::Dict{String,Float64}
    outcome_history::Dict{String,Vector{Tuple{Float64,Float64,Bool}}}
end

function UncertaintyResponseProfile(; rng::Random.AbstractRNG = Random.default_rng(), learning_rate::Float64 = 0.05)
    UncertaintyResponseProfile(
        0.15,
        60,
        learning_rate,  # Use configurable learning rate
        Dict{String,Float64}(
            "actor_ignorance" => 0.3 + 0.4 * rand(rng),
            "practical_indeterminism" => 0.3 + 0.4 * rand(rng),
            "agentic_novelty" => 0.3 + 0.4 * rand(rng),
            "competitive_recursion" => 0.3 + 0.4 * rand(rng)
        ),
        Dict{String,Vector{Tuple{Float64,Float64,Bool}}}(
            "actor_ignorance" => Tuple{Float64,Float64,Bool}[],
            "practical_indeterminism" => Tuple{Float64,Float64,Bool}[],
            "agentic_novelty" => Tuple{Float64,Float64,Bool}[],
            "competitive_recursion" => Tuple{Float64,Float64,Bool}[]
        )
    )
end

"""
Get response factor for an uncertainty type.
"""
function get_response_factor(
    profile::UncertaintyResponseProfile,
    uncertainty_type::String,
    uncertainty_level::Float64;
    explore::Bool = true,
    rng::Random.AbstractRNG = Random.default_rng()
)::Float64
    if explore && rand(rng) < profile.exploration_rate
        experimental_weight = rand(rng)
        return 1.0 - (uncertainty_level * experimental_weight)
    end

    weight = get(profile.response_weights, uncertainty_type, 0.5)
    return 1.0 - (uncertainty_level * weight)
end

"""
Update response profile from outcome.
"""
function update_from_outcome!(
    profile::UncertaintyResponseProfile,
    uncertainty_perception::Dict{String,Any},
    action::String,
    outcome::Dict{String,Any},
    market_conditions::Dict{String,Any}
)
    success = get(outcome, "success", false)
    investment_amount = max(get(outcome, "investment_amount", 1.0), 1.0)
    returns = get(outcome, "capital_returned", 0.0) / investment_amount

    for u_type in keys(profile.response_weights)
        if haskey(uncertainty_perception, u_type)
            u_data = uncertainty_perception[u_type]
            level = if isa(u_data, Dict)
                get(u_data, "level", get(u_data, "$(split(u_type, "_")[1])_level", 0.0))
            else
                0.0
            end

            push!(profile.outcome_history[u_type], (level, returns, success))

            # Trim to memory limit
            if length(profile.outcome_history[u_type]) > profile.memory_limit
                popfirst!(profile.outcome_history[u_type])
            end

            update_response_weight!(profile, u_type, market_conditions)
        end
    end
end

"""
Update response weight for a specific uncertainty type.
"""
function update_response_weight!(
    profile::UncertaintyResponseProfile,
    uncertainty_type::String,
    market_conditions::Dict{String,Any}
)
    history = profile.outcome_history[uncertainty_type]
    if isempty(history)
        return
    end

    recent = history[max(1, end - profile.memory_limit + 1):end]

    # Separate low and high uncertainty experiences
    low_uncertainty = [(r, s) for (l, r, s) in recent if l < 0.33]
    high_uncertainty = [(r, s) for (l, r, s) in recent if l >= 0.67]

    # Compute mean performance with prior
    function mean_with_prior(data, prior_mean = 1.0, prior_weight = 2.0)
        if isempty(data)
            return prior_mean
        end
        returns = [r for (r, _) in data]
        m = mean(returns)
        weight = length(returns)
        return (m * weight + prior_mean * prior_weight) / (weight + prior_weight)
    end

    perf_low = mean_with_prior(low_uncertainty)
    perf_high = mean_with_prior(high_uncertainty, perf_low)

    # Compute variability
    all_returns = [r for (_, r, _) in recent]
    variability = length(all_returns) > 1 ? std(all_returns) : 0.0

    base_weight = profile.response_weights[uncertainty_type]
    target_weight = base_weight

    if perf_high < perf_low * 0.85
        target_weight = min(1.0, base_weight + profile.learning_rate * (perf_low - perf_high))
    elseif perf_low < perf_high * 0.9
        target_weight = max(0.0, base_weight - profile.learning_rate * (perf_high - perf_low))
    else
        target_weight = base_weight * (1 - variability * 0.2)
    end

    stability = clamp(1.0 - variability, 0.3, 1.0)
    updated_weight = stability * target_weight + (1 - stability) * base_weight
    profile.response_weights[uncertainty_type] = clamp(updated_weight, 0.0, 1.0)
end

# ============================================================================
# AI TIER BELIEFS (Bayesian)
# ============================================================================

"""
Bayesian beliefs about AI tier performance.
"""
mutable struct AITierBeliefs
    alpha::Dict{String,Float64}  # Beta distribution alpha (successes + 1)
    beta::Dict{String,Float64}   # Beta distribution beta (failures + 1)
    usage_count::Dict{String,Int}
    success_count::Dict{String,Int}
    total_returns::Dict{String,Float64}
    total_invested::Dict{String,Float64}
end

function AITierBeliefs()
    tiers = ["none", "basic", "advanced", "premium"]
    AITierBeliefs(
        Dict(t => 1.0 for t in tiers),  # Uniform prior
        Dict(t => 1.0 for t in tiers),
        Dict(t => 0 for t in tiers),
        Dict(t => 0 for t in tiers),
        Dict(t => 0.0 for t in tiers),
        Dict(t => 0.0 for t in tiers)
    )
end

"""
Get expected performance for a tier (posterior mean of Beta distribution).
"""
function get_expected_performance(beliefs::AITierBeliefs, tier::String)::Float64
    a = get(beliefs.alpha, tier, 1.0)
    b = get(beliefs.beta, tier, 1.0)
    return a / (a + b)
end

"""
Get uncertainty about a tier (posterior variance).
"""
function get_tier_uncertainty(beliefs::AITierBeliefs, tier::String)::Float64
    a = get(beliefs.alpha, tier, 1.0)
    b = get(beliefs.beta, tier, 1.0)
    return (a * b) / ((a + b)^2 * (a + b + 1))
end

"""
Update beliefs based on outcome.
"""
function update_belief!(beliefs::AITierBeliefs, tier::String, success::Bool, investment::Float64, returns::Float64)
    tier_key = lowercase(tier)

    beliefs.usage_count[tier_key] = get(beliefs.usage_count, tier_key, 0) + 1
    beliefs.total_invested[tier_key] = get(beliefs.total_invested, tier_key, 0.0) + investment
    beliefs.total_returns[tier_key] = get(beliefs.total_returns, tier_key, 0.0) + returns

    if success
        beliefs.alpha[tier_key] = get(beliefs.alpha, tier_key, 1.0) + 1.0
        beliefs.success_count[tier_key] = get(beliefs.success_count, tier_key, 0) + 1
    else
        beliefs.beta[tier_key] = get(beliefs.beta, tier_key, 1.0) + 1.0
    end
end

"""
Select AI tier using Thompson Sampling.
"""
function select_tier_thompson(
    beliefs::AITierBeliefs;
    available_tiers::Vector{String} = ["none", "basic", "advanced", "premium"],
    rng::Random.AbstractRNG = Random.default_rng()
)::String
    best_tier = "none"
    best_sample = -Inf

    for tier in available_tiers
        a = get(beliefs.alpha, tier, 1.0)
        b = get(beliefs.beta, tier, 1.0)

        # Sample from Beta distribution
        sample = rand(rng, Distributions.Beta(a, b))

        if sample > best_sample
            best_sample = sample
            best_tier = tier
        end
    end

    return best_tier
end

"""
Select AI tier using Upper Confidence Bound.
"""
function select_tier_ucb(
    beliefs::AITierBeliefs;
    available_tiers::Vector{String} = ["none", "basic", "advanced", "premium"],
    exploration_weight::Float64 = 2.0
)::String
    total_usage = sum(get(beliefs.usage_count, t, 0) for t in available_tiers)

    if total_usage == 0
        return "none"
    end

    best_tier = "none"
    best_score = -Inf

    for tier in available_tiers
        usage = get(beliefs.usage_count, tier, 0)
        mean_perf = get_expected_performance(beliefs, tier)

        if usage == 0
            score = Inf  # Explore unused tiers
        else
            exploration_bonus = exploration_weight * sqrt(log(total_usage + 1) / usage)
            score = mean_perf + exploration_bonus
        end

        if score > best_score
            best_score = score
            best_tier = tier
        end
    end

    return best_tier
end
