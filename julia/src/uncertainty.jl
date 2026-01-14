"""
Knightian Uncertainty Environment for GlimpseABM.jl

This module implements the four-dimensional Knightian uncertainty framework
from Townsend et al. (2025, AMR).

Port of: glimpse_abm/uncertainty.py
"""

using Random
using Statistics

# ============================================================================
# KNIGHTIAN UNCERTAINTY ENVIRONMENT
# ============================================================================

"""
Environment tracking and measuring the four dimensions of Knightian uncertainty.

The four dimensions are:
1. Actor Ignorance - Information gaps about current states
2. Practical Indeterminism - Unpredictable execution outcomes
3. Agentic Novelty - Genuinely new possibilities from creative action
4. Competitive Recursion - Strategic interdependence effects
"""
mutable struct KnightianUncertaintyEnvironment
    config::EmergentConfig

    # Actor Ignorance State
    actor_ignorance_state::Dict{String,Any}

    # Practical Indeterminism State
    practical_indeterminism_state::Dict{String,Any}

    # Agentic Novelty State
    agentic_novelty_state::Dict{String,Any}

    # Competitive Recursion State
    competitive_recursion_state::Dict{String,Any}

    # Uncertainty evolution history
    uncertainty_evolution::Vector{Dict{String,Any}}

    # AI-specific signals
    ai_uncertainty_signals::Dict{String,Any}

    # Innovation tracking
    innovation_success_tracker::Vector{Dict{String,Any}}

    # Short-term buffers for volatility calculation
    short_term_buffers::Dict{String,Vector{Float64}}

    # Volatility state
    volatility_state::Dict{String,Any}

    rng::AbstractRNG
end

function KnightianUncertaintyEnvironment(
    config::EmergentConfig;
    rng::AbstractRNG = Random.default_rng()
)
    return KnightianUncertaintyEnvironment(
        config,
        Dict{String,Any}(
            "unknown_opportunities" => Dict{String,Float64}(),
            "knowledge_gaps" => Dict{String,Float64}(),
            "emergence_potential" => 0.0,
            "level" => 0.0
        ),
        Dict{String,Any}(
            "path_volatility" => 0.0,
            "timing_criticality" => Dict{String,Float64}(),
            "regime_instability" => 0.0,
            "level" => 0.0
        ),
        Dict{String,Any}(
            "creative_momentum" => 0.0,
            "disruption_potential" => Dict{String,Float64}(),
            "novelty_level" => 0.0,
            "level" => 0.0,
            "component_scarcity" => 0.5,
            "reuse_pressure" => 0.0
        ),
        Dict{String,Any}(
            "strategic_opacity" => 0.0,
            "herding_pressure" => Dict{String,Float64}(),
            "game_complexity" => 0.0,
            "level" => 0.0
        ),
        Dict{String,Any}[],
        Dict{String,Any}(
            "hallucination_events" => Dict{String,Any}[],
            "confidence_miscalibration" => Float64[],
            "ai_herding_patterns" => Dict{String,Float64}()
        ),
        Dict{String,Any}[],
        Dict{String,Vector{Float64}}(
            "actor_ignorance" => Float64[],
            "practical_indeterminism" => Float64[],
            "agentic_novelty" => Float64[],
            "competitive_recursion" => Float64[]
        ),
        Dict{String,Any}(
            "action_prev" => zeros(4),
            "ai_prev" => zeros(4),
            "market_prev" => 0.0,
            "volatility_metric" => 0.0
        ),
        rng
    )
end

"""
Record AI signals from agent actions.
"""
function record_ai_signals!(
    env::KnightianUncertaintyEnvironment,
    round_num::Int,
    agent_actions::Vector{Dict{String,Any}}
)
    if isempty(agent_actions)
        return
    end

    hallucinations = env.ai_uncertainty_signals["hallucination_events"]
    confidence_vals = env.ai_uncertainty_signals["confidence_miscalibration"]
    herding_patterns = env.ai_uncertainty_signals["ai_herding_patterns"]

    herding_counts = Dict{String,Int}()
    hallucinations_this_round = 0

    for action in agent_actions
        ai_level = lowercase(string(get(action, "ai_level_used", "none")))
        if ai_level == "none"
            continue
        end

        # Track hallucinations
        if get(action, "ai_contains_hallucination", false)
            hallucinations_this_round += 1
            push!(hallucinations, Dict(
                "round" => round_num,
                "agent_id" => get(action, "agent_id", 0),
                "ai_level" => ai_level
            ))
        end

        # Track confidence miscalibration
        confidence = get(action, "ai_confidence", nothing)
        accuracy = get(action, "ai_actual_accuracy", nothing)
        if !isnothing(confidence) && !isnothing(accuracy)
            miscal = Float64(confidence) - Float64(accuracy)
            if isfinite(miscal)
                push!(confidence_vals, miscal)
            end
        end

        # Track herding
        if get(action, "action", "") == "invest"
            details = get(action, "chosen_opportunity_details", Dict())
            opp_id = get(details, "id", get(action, "opportunity_id", nothing))
            if !isnothing(opp_id)
                herding_counts[string(opp_id)] = get(herding_counts, string(opp_id), 0) + 1
            end
        end
    end

    # Update herding patterns with decay
    decay = env.config.AI_HERDING_DECAY
    for key in keys(herding_patterns)
        herding_patterns[key] *= decay
    end
    for (opp_id, count) in herding_counts
        prior = get(herding_patterns, opp_id, 0.0) * decay
        herding_patterns[opp_id] = prior + count
    end

    # Update short-term buffers if hallucinations occurred
    if hallucinations_this_round > 0
        spike = clamp(0.15 + 0.05 * hallucinations_this_round, 0.0, 1.0)
        _update_short_term!(env, "actor_ignorance", spike)
        _update_short_term!(env, "agentic_novelty", spike * 0.6)
    end
end

"""
Update short-term buffer for a metric.
"""
function _update_short_term!(env::KnightianUncertaintyEnvironment, metric::String, value::Float64)
    if !isnan(value)
        push!(env.short_term_buffers[metric], value)
        # Keep buffer size limited
        max_size = env.config.UNCERTAINTY_SHORT_WINDOW
        if length(env.short_term_buffers[metric]) > max_size
            deleteat!(env.short_term_buffers[metric], 1)
        end
    end
end

"""
Get short-term average for a metric.
"""
function _get_short_term_average(env::KnightianUncertaintyEnvironment, metric::String; fallback::Float64 = 0.0)::Float64
    buffer = get(env.short_term_buffers, metric, Float64[])
    if isempty(buffer)
        return fallback
    end
    return mean(buffer)
end

"""
Track innovation outcomes to update novelty perceptions.

Parameters
----------
env : KnightianUncertaintyEnvironment
    The uncertainty environment to update.
opportunity_id : String
    Unique identifier for the opportunity/innovation.
success : Union{Bool,Nothing}
    Whether the innovation was successful.
impact : Union{Float64,Nothing}
    Impact/magnitude of the innovation.
combination_signature : Union{String,Nothing}
    Unique signature describing the combination of elements.
new_possibility_rate : Union{Float64,Nothing}
    Rate at which new possibilities are being discovered [0,1].
scarcity : Union{Float64,Nothing}
    Current resource/component scarcity signal [0,1].
"""
function register_innovation_event!(
    env::KnightianUncertaintyEnvironment,
    opportunity_id::String;
    success::Union{Bool,Nothing}=nothing,
    impact::Union{Float64,Nothing}=nothing,
    combination_signature::Union{String,Nothing}=nothing,
    new_possibility_rate::Union{Float64,Nothing}=nothing,
    scarcity::Union{Float64,Nothing}=nothing
)
    # Track the innovation record
    record = Dict{String,Any}(
        "opportunity_id" => opportunity_id,
        "success" => success,
        "impact" => impact,
        "combination_signature" => combination_signature
    )
    push!(env.innovation_success_tracker, record)

    # Update agentic novelty state
    if new_possibility_rate !== nothing
        env.agentic_novelty_state["recent_new_possibility_rate"] = clamp(new_possibility_rate, 0.0, 1.0)
    end

    if scarcity !== nothing
        env.agentic_novelty_state["recent_scarcity_signal"] = clamp(scarcity, 0.0, 1.0)
    end
end

"""
Measure environment-level uncertainty across the four Knightian dimensions.
"""
function measure_uncertainty_state!(
    env::KnightianUncertaintyEnvironment,
    market::MarketEnvironment,
    agent_actions::Vector{Dict{String,Any}},
    innovations::Vector{Innovation},
    round_num::Int
)::Dict{String,Dict{String,Any}}
    total_actions = length(agent_actions)

    # Count actions by type
    action_counts = Dict{String,Int}()
    for action in agent_actions
        act_type = get(action, "action", "maintain")
        action_counts[act_type] = get(action_counts, act_type, 0) + 1
    end

    # Count AI tier usage
    ai_counts = Dict{String,Int}("none" => 0, "basic" => 0, "advanced" => 0, "premium" => 0)
    for action in agent_actions
        tier = lowercase(string(get(action, "ai_level_used", "none")))
        if haskey(ai_counts, tier)
            ai_counts[tier] += 1
        end
    end

    if total_actions > 0
        action_shares = [
            get(action_counts, "invest", 0) / total_actions,
            get(action_counts, "innovate", 0) / total_actions,
            get(action_counts, "explore", 0) / total_actions,
            get(action_counts, "maintain", 0) / total_actions
        ]
        ai_shares = [
            ai_counts["none"] / total_actions,
            ai_counts["basic"] / total_actions,
            ai_counts["advanced"] / total_actions,
            ai_counts["premium"] / total_actions
        ]
    else
        action_shares = [0.0, 0.0, 0.0, 1.0]
        ai_shares = [1.0, 0.0, 0.0, 0.0]
    end

    ai_share_none = ai_shares[1]
    ai_usage_rate = 1.0 - ai_share_none

    # ========================================================================
    # ACTOR IGNORANCE
    # ========================================================================
    # Base ignorance from opportunities not yet discovered
    n_opportunities = length(market.opportunities)
    discovered = count(opp -> opp.discovered, market.opportunities)
    discovery_ratio = n_opportunities > 0 ? discovered / n_opportunities : 1.0

    # AI reduces ignorance
    ai_reduction = 0.0
    if ai_usage_rate > 0
        ai_reduction = ai_usage_rate * 0.3 * (1.0 + ai_shares[4] * 0.5)  # Premium more effective
    end

    base_ignorance = (1.0 - discovery_ratio) * 0.5
    actor_ignorance_level = max(0.0, base_ignorance - ai_reduction)

    # Add short-term volatility
    st_avg = _get_short_term_average(env, "actor_ignorance"; fallback=actor_ignorance_level)
    actor_ignorance_level = 0.7 * actor_ignorance_level + 0.3 * st_avg

    env.actor_ignorance_state["level"] = actor_ignorance_level

    # ========================================================================
    # PRACTICAL INDETERMINISM
    # ========================================================================
    # Base from market volatility and regime instability
    regime_instability = market.volatility + abs(market.trend) * 0.3

    # AI herding increases indeterminism
    herding_patterns = env.ai_uncertainty_signals["ai_herding_patterns"]
    ai_herding_intensity = if !isempty(herding_patterns)
        clamp(sum(values(herding_patterns)) / max(1.0, total_actions), 0.0, 1.0)
    else
        0.0
    end

    # Crowding increases indeterminism (scaled by competition intensity)
    crowding = get(market.crowding_metrics, "crowding_index", 0.25)
    competition_intensity = env.config.COMPETITION_INTENSITY

    # Scale the herding and crowding components by competition intensity
    practical_indet_level = (
        0.3 * regime_instability +
        0.3 * ai_herding_intensity * competition_intensity +  # Herding effect scaled
        0.2 * crowding * competition_intensity +              # Crowding effect scaled
        0.2 * (1.0 - ai_share_none) * 0.5  # AI increases execution uncertainty (not competitive)
    )
    practical_indet_level = clamp(practical_indet_level, 0.0, 1.0)

    env.practical_indeterminism_state["level"] = practical_indet_level
    env.practical_indeterminism_state["regime_instability"] = regime_instability

    # ========================================================================
    # AGENTIC NOVELTY
    # ========================================================================
    # Count new combinations
    new_combos = 0
    for action in agent_actions
        if get(action, "is_new_combination", false) ||
           get(action, "discovered_niche", false) ||
           get(action, "created_opportunity", false)
            new_combos += 1
        end
    end
    innovation_rate = total_actions > 0 ? new_combos / total_actions : 0.0

    # Innovation births
    innovation_births = length(innovations)

    # AI can both enable and constrain novelty
    # Premium AI may anchor on historical patterns
    # Scale the constraint effect by AI_NOVELTY_CONSTRAINT_INTENSITY (for robustness testing)
    novelty_constraint_intensity = env.config.AI_NOVELTY_CONSTRAINT_INTENSITY
    ai_novelty_effect = (
        0.3 * ai_shares[2] +  # Basic enables some novelty
        0.4 * ai_shares[3] -  # Advanced enables more
        0.1 * ai_shares[4] * novelty_constraint_intensity  # Premium constraint scaled
    )

    agentic_novelty_level = (
        0.4 * innovation_rate +
        0.3 * (innovation_births > 0 ? min(1.0, innovation_births / 10) : 0.0) +
        0.3 * max(0.0, ai_novelty_effect)
    )
    agentic_novelty_level = clamp(agentic_novelty_level, 0.0, 1.0)

    env.agentic_novelty_state["level"] = agentic_novelty_level
    env.agentic_novelty_state["novelty_potential"] = innovation_rate

    # ========================================================================
    # COMPETITIVE RECURSION
    # ========================================================================
    # Strategic interdependence from AI convergence
    invest_concentration = get(action_counts, "invest", 0) / max(1, total_actions)

    # AI tier concentration creates recursion
    premium_share = ai_shares[4]
    advanced_share = ai_shares[3]

    # Herding pressure (scaled by competition intensity)
    herding_pressure = ai_herding_intensity * competition_intensity

    # Competitive recursion - scale the crowding and herding components
    competitive_recursion_level = (
        0.3 * invest_concentration^2 * competition_intensity +  # Investment crowding scaled
        0.3 * herding_pressure +                                # Already scaled above
        0.2 * premium_share * 2.0 +  # Premium creates more recursion (not purely competitive)
        0.2 * advanced_share
    )
    competitive_recursion_level = clamp(competitive_recursion_level, 0.0, 1.0)

    env.competitive_recursion_state["level"] = competitive_recursion_level
    env.competitive_recursion_state["herding_pressure"] = herding_patterns

    # ========================================================================
    # COMPILE STATE
    # ========================================================================
    state = Dict{String,Dict{String,Any}}(
        "actor_ignorance" => Dict{String,Any}(
            "level" => actor_ignorance_level,
            "knowledge_gaps" => env.actor_ignorance_state["knowledge_gaps"],
            "ai_delta" => -ai_reduction
        ),
        "practical_indeterminism" => Dict{String,Any}(
            "level" => practical_indet_level,
            "regime_instability" => regime_instability,
            "ai_herding_intensity" => ai_herding_intensity,
            "ai_delta" => ai_herding_intensity * 0.5
        ),
        "agentic_novelty" => Dict{String,Any}(
            "level" => agentic_novelty_level,
            "novelty_potential" => innovation_rate,
            "component_scarcity" => env.agentic_novelty_state["component_scarcity"],
            "reuse_pressure" => env.agentic_novelty_state["reuse_pressure"],
            "ai_delta" => ai_novelty_effect
        ),
        "competitive_recursion" => Dict{String,Any}(
            "level" => competitive_recursion_level,
            "herding_pressure" => herding_pressure,
            "ai_premium_share" => premium_share,
            "ai_delta" => (premium_share + advanced_share) * 0.5
        )
    )

    # Record evolution
    push!(env.uncertainty_evolution, Dict(
        "round" => round_num,
        "actor_ignorance" => actor_ignorance_level,
        "practical_indeterminism" => practical_indet_level,
        "agentic_novelty" => agentic_novelty_level,
        "competitive_recursion" => competitive_recursion_level
    ))

    return state
end

"""
Get the current uncertainty state as a simple dictionary.
"""
function get_uncertainty_state(env::KnightianUncertaintyEnvironment)::Dict{String,Any}
    return Dict{String,Any}(
        "actor_ignorance" => env.actor_ignorance_state,
        "practical_indeterminism" => env.practical_indeterminism_state,
        "agentic_novelty" => env.agentic_novelty_state,
        "competitive_recursion" => env.competitive_recursion_state
    )
end

"""
Calculate composite uncertainty score.
"""
function get_composite_uncertainty(env::KnightianUncertaintyEnvironment)::Float64
    levels = [
        Float64(get(env.actor_ignorance_state, "level", 0.0)),
        Float64(get(env.practical_indeterminism_state, "level", 0.0)),
        Float64(get(env.agentic_novelty_state, "level", 0.0)),
        Float64(get(env.competitive_recursion_state, "level", 0.0))
    ]
    return mean(levels)
end

# ============================================================================
# AGENT PERCEPTION MODEL
# ============================================================================

"""
Compute an individual agent's subjective perception of uncertainty.

While measure_uncertainty_state() computes objective environment-level
uncertainty, this method models how individual agents perceive uncertainty
based on their traits, knowledge, AI augmentation level, and experience.

The gap between objective and perceived uncertainty is central to the
paradox of future knowledge (Townsend et al., 2025).

Parameters
----------
env : KnightianUncertaintyEnvironment
    The uncertainty environment
agent_traits : Dict
    Agent's psychological traits
visible_opportunities : Vector
    Opportunities currently visible to the agent
market_conditions : Dict
    Current market state
ai_level : String
    Agent's current AI augmentation tier
agent_knowledge : Set{String}
    Knowledge IDs known to the agent
sector_knowledge : Dict{String,Float64}
    Agent's sector knowledge levels

Returns
-------
Dict
    Agent's subjective uncertainty perception across all dimensions
"""
function perceive_uncertainty(
    env::KnightianUncertaintyEnvironment,
    agent_traits::Dict{String,Float64},
    visible_opportunities::Vector,
    market_conditions::Dict{String,Any};
    ai_level::String = "none",
    agent_knowledge::Set{String} = Set{String}(),
    sector_knowledge::Dict{String,Float64} = Dict{String,Float64}(),
    action_history::Vector{String} = String[],
    ai_learning_profile::Union{AILearningProfile,Nothing} = nothing
)::Dict{String,Any}
    perception = Dict{String,Any}()

    # Get AI configuration
    ai_config = get(env.config.AI_LEVELS, ai_level, env.config.AI_LEVELS["none"])
    info_quality = Float64(get(ai_config, "info_quality", 0.0))
    info_breadth = Float64(get(ai_config, "info_breadth", 0.0))

    # Extract agent traits
    competence_trait = get(agent_traits, "competence", 0.5)
    ai_trust_trait = get(agent_traits, "ai_trust", 0.5)
    exploration_trait = get(agent_traits, "exploration_tendency", 0.5)
    innovation_trait = get(agent_traits, "innovativeness", 0.5)
    uncertainty_tolerance = get(agent_traits, "uncertainty_tolerance", 0.5)
    analytical_ability = get(agent_traits, "analytical_ability", 0.5)

    # Compute action shares from history
    if !isempty(action_history)
        action_counts = Dict{String,Int}()
        for action in action_history
            action_counts[action] = get(action_counts, action, 0) + 1
        end
        total = length(action_history)
        share_invest = get(action_counts, "invest", 0) / total
        share_innovate = get(action_counts, "innovate", 0) / total
        share_explore = get(action_counts, "explore", 0) / total
        share_maintain = get(action_counts, "maintain", 0) / total
    else
        share_invest = share_innovate = share_explore = share_maintain = 0.25
    end

    # AI trust calibration
    avg_ai_trust = ai_trust_trait
    hallucination_rate = 0.0
    if !isnothing(ai_learning_profile)
        trusts = collect(values(ai_learning_profile.domain_trust))
        if !isempty(trusts)
            avg_ai_trust = clamp(mean(trusts), 0.0, 1.0)
        end
        total_hall = sum(values(ai_learning_profile.hallucination_experiences))
        total_usage = max(1, sum(values(ai_learning_profile.usage_count)))
        hallucination_rate = clamp(total_hall / total_usage, 0.0, 1.0)
    end
    avg_ai_trust = clamp(0.5 * ai_trust_trait + 0.5 * avg_ai_trust, 0.0, 1.0)

    # ========================================================================
    # ACTOR IGNORANCE PERCEPTION
    # ========================================================================

    # Personal knowledge assessment
    personal_knowledge = 0.0
    knowledge_span = 0.0
    if !isempty(sector_knowledge)
        knowledge_vals = collect(values(sector_knowledge))
        personal_knowledge = clamp(mean(knowledge_vals), 0.0, 1.0)
        knowledge_span = length(sector_knowledge) / max(1, length(env.config.SECTORS))
    end

    # Adjust for AI information quality
    personal_knowledge = clamp(personal_knowledge + info_quality * 0.15, 0.0, 1.0)
    knowledge_span = clamp(knowledge_span + info_breadth * 0.2, 0.0, 1.0)

    # Knowledge deficit
    avg_knowledge_level = personal_knowledge
    knowledge_deficit = max(0.0, 1.0 - avg_knowledge_level)
    knowledge_deficit = clamp(knowledge_deficit - 0.2 * info_breadth, 0.0, 1.2)

    # Discovery momentum
    discovery_momentum = 0.1 + 0.3 * share_explore

    # Base ignorance calculation
    discovery_shortfall = max(0.0, 1.0 - min(1.0, discovery_momentum + 0.2 * share_explore))
    base_ignorance = (
        0.25 * (1.0 - info_quality) +
        0.20 * (1.0 - info_breadth) +
        0.35 * knowledge_deficit +
        0.20 * discovery_shortfall
    )

    # Ignorance reduction from AI
    ignorance_reduction = info_quality * (0.12 + 0.25 * avg_ai_trust) + info_breadth * 0.08

    # Calculate knowledge gaps for visible opportunities
    knowledge_gaps = Dict{String,Float64}()
    analytical_modifier = 1 - analytical_ability * 0.4

    for opp in visible_opportunities
        sector = hasfield(typeof(opp), :sector) ? opp.sector : "unknown"
        sector_familiarity = get(sector_knowledge, sector, 0.0)

        # Component familiarity
        component_familiarity = sector_familiarity
        if hasfield(typeof(opp), :knowledge_components) && !isnothing(opp.knowledge_components)
            component_ids = Set(opp.knowledge_components)
            if !isempty(component_ids) && !isempty(agent_knowledge)
                familiar = length(intersect(agent_knowledge, component_ids))
                component_familiarity = familiar / max(1, length(component_ids))
            end
        end

        base_gap = 1.0 - max(sector_familiarity, component_familiarity)
        gap = clamp(base_gap * analytical_modifier * (1 - info_quality * 0.5), 0.0, 1.0)

        opp_id = hasfield(typeof(opp), :id) ? opp.id : "opp_unknown"
        if gap > 0.02
            knowledge_gaps[opp_id] = gap
        end
    end

    gap_pressure = isempty(knowledge_gaps) ? 0.0 : clamp(mean(values(knowledge_gaps)), 0.0, 1.0)

    # Final ignorance level
    volatility = Float64(get(env.practical_indeterminism_state, "volatility", 0.0))
    estimated_ignorance = clamp(
        base_ignorance -
        ignorance_reduction -
        0.08 * share_explore -
        0.05 * volatility +
        0.05 * share_maintain -
        0.4 * personal_knowledge -
        0.15 * knowledge_span -
        0.05 * ai_trust_trait +
        0.15 * (1.0 - competence_trait),
        0.01, 0.99
    )

    perception["actor_ignorance"] = Dict{String,Any}(
        "ignorance_level" => estimated_ignorance,
        "level" => estimated_ignorance,
        "knowledge_gaps" => knowledge_gaps,
        "gap_pressure" => gap_pressure,
        "info_quality" => info_quality,
        "info_breadth" => info_breadth,
        "personal_knowledge" => personal_knowledge,
        "knowledge_span" => knowledge_span,
        "discovery_momentum" => discovery_momentum
    )

    # ========================================================================
    # PRACTICAL INDETERMINISM PERCEPTION
    # ========================================================================

    market_volatility = Float64(get(market_conditions, "volatility", 0.2))
    regime = get(market_conditions, "regime", "normal")
    regime_uncertainty_map = Dict(
        "boom" => 0.3, "growth" => 0.2, "normal" => 0.1,
        "recession" => 0.4, "crisis" => 0.7
    )
    regime_uncertainty = get(regime_uncertainty_map, regime, 0.2)
    regime_uncertainty = clamp(regime_uncertainty * (1 - 0.3 * avg_ai_trust) + hallucination_rate * 0.2, 0.0, 1.0)

    # Path complexity
    visible_sectors = Set(hasfield(typeof(opp), :sector) ? opp.sector : "unknown" for opp in visible_opportunities)
    known_sectors = count(v -> v >= 0.4, values(sector_knowledge))
    path_complexity = max(0.0, length(visible_sectors) - known_sectors) / max(1, length(env.config.SECTORS))

    # Timing pressure
    timing_pressure = Dict{String,Float64}()
    for opp in visible_opportunities
        lifecycle = hasfield(typeof(opp), :lifecycle_stage) ? opp.lifecycle_stage : "emerging"
        competition = hasfield(typeof(opp), :competition) ? opp.competition : 0.0
        adoption = hasfield(typeof(opp), :market_share) ? opp.market_share : 0.0

        urgency = competition * 0.4 + adoption * 0.3
        urgency *= if lifecycle == "declining"
            0.3
        elseif lifecycle == "mature"
            0.6
        elseif lifecycle == "growing"
            1.2
        else
            1.0
        end

        opp_id = hasfield(typeof(opp), :id) ? opp.id : "opp_unknown"
        timing_pressure[opp_id] = urgency
    end
    avg_timing_pressure = isempty(timing_pressure) ? 0.0 : mean(values(timing_pressure))

    # Crowding and AI pressure from environment state
    crowding_pressure = Float64(get(env.practical_indeterminism_state, "crowding_pressure", 0.0))
    ai_pressure_level = Float64(get(env.practical_indeterminism_state, "ai_pressure", 0.0))
    ai_herding_intensity = Float64(get(env.practical_indeterminism_state, "ai_herding_intensity", 0.0))

    # Agent-specific path risk
    agent_path_risk = 0.18 + 0.22 * gap_pressure + 0.12 * share_invest + 0.10 * max(0.0, share_innovate - share_explore)
    agent_volatility_factor = clamp(abs(share_invest - share_maintain) + abs(share_innovate - share_explore), 0.0, 1.0)

    agent_uncertainty = (
        0.12 * (1.0 - competence_trait) +
        0.08 * (0.6 - personal_knowledge) +
        0.05 * max(0.0, 0.55 - ai_trust_trait) +
        0.08 * discovery_shortfall
    )

    system_term = (
        0.08 * market_volatility +
        0.07 * regime_uncertainty +
        0.06 * avg_timing_pressure +
        0.05 * path_complexity +
        0.05 * volatility
    )

    crowd_term = (
        0.06 * max(0.0, crowding_pressure) +
        0.04 * max(0.0, ai_pressure_level - 0.3) +
        0.05 * ai_herding_intensity
    )

    indeterminism_level = clamp(
        agent_path_risk + 0.15 * agent_volatility_factor + agent_uncertainty + system_term + crowd_term,
        0.0, 1.0
    )

    perception["practical_indeterminism"] = Dict{String,Any}(
        "indeterminism_level" => indeterminism_level,
        "level" => indeterminism_level,
        "timing_pressure" => timing_pressure,
        "market_regime_risk" => regime_uncertainty,
        "volatility" => volatility,
        "crowding_pressure" => crowding_pressure,
        "ai_pressure" => ai_pressure_level,
        "ai_herding_intensity" => ai_herding_intensity
    )

    # ========================================================================
    # AGENTIC NOVELTY PERCEPTION
    # ========================================================================

    novelty_state = env.agentic_novelty_state
    novelty_level = Float64(get(novelty_state, "level", 0.5))
    scarcity_signal = Float64(get(novelty_state, "component_scarcity", 0.5))
    reuse_pressure = Float64(get(novelty_state, "reuse_pressure", 0.0))
    new_possibility_rate = Float64(get(novelty_state, "new_possibility_rate", 0.0))

    # Tier-specific adjustments
    tier_combo_rate = Float64(get(get(novelty_state, "combo_rate_by_tier", Dict()), ai_level, 0.0))
    tier_reuse_pressure = Float64(get(get(novelty_state, "reuse_pressure_by_tier", Dict()), ai_level, reuse_pressure))

    novelty_adjustment = (
        0.25 * tier_combo_rate -
        0.2 * tier_reuse_pressure +
        0.15 * new_possibility_rate
    )

    novelty_level_agent = clamp(
        novelty_level + novelty_adjustment +
        0.22 * (innovation_trait - 0.5) +
        0.15 * (exploration_trait - 0.5) +
        0.10 * personal_knowledge,
        0.0, 1.0
    )

    disruption_potential = Dict{String,Float64}()
    for sector in env.config.SECTORS
        disruption_potential[sector] = 0.3 + 0.2 * novelty_level_agent
    end

    perception["agentic_novelty"] = Dict{String,Any}(
        "novelty_potential" => novelty_level_agent,
        "level" => novelty_level_agent,
        "creative_confidence" => novelty_level_agent,
        "disruption_potential" => disruption_potential,
        "component_scarcity" => scarcity_signal,
        "reuse_pressure" => reuse_pressure,
        "new_possibility_rate" => new_possibility_rate,
        "combo_rate" => tier_combo_rate
    )

    # ========================================================================
    # COMPETITIVE RECURSION PERCEPTION
    # ========================================================================

    recursion_state = env.competitive_recursion_state
    recursion_level = Float64(get(recursion_state, "level", 0.3))
    herding_pressure = Float64(get(recursion_state, "herding_pressure", 0.0))
    strategic_uncertainty = Float64(get(recursion_state, "strategic_uncertainty", 0.0))

    # AI tier shares
    ai_shares = get(market_conditions, "ai_tier_shares", Dict{String,Float64}())
    premium_share = Float64(get(ai_shares, "premium", 0.0))
    advanced_share = Float64(get(ai_shares, "advanced", 0.0))
    basic_share = Float64(get(ai_shares, "basic", 0.0))
    none_share = Float64(get(ai_shares, "none", 1.0))

    # Agent perception of competitive recursion
    # Higher AI tiers may perceive more recursion due to awareness
    tier_awareness_bonus = Dict(
        "none" => 0.0,
        "basic" => 0.05,
        "advanced" => 0.12,
        "premium" => 0.20
    )
    awareness = get(tier_awareness_bonus, ai_level, 0.0)

    # Own contribution to recursion
    own_tier_share = get(ai_shares, ai_level, 0.0)
    self_contribution = own_tier_share * 0.1

    perceived_recursion = clamp(
        recursion_level +
        awareness +
        self_contribution +
        0.1 * herding_pressure +
        0.05 * (premium_share + advanced_share) -
        0.1 * uncertainty_tolerance,
        0.0, 1.0
    )

    perception["competitive_recursion"] = Dict{String,Any}(
        "recursion_level" => perceived_recursion,
        "level" => perceived_recursion,
        "herding_pressure" => herding_pressure,
        "strategic_uncertainty" => strategic_uncertainty,
        "ai_premium_share" => premium_share,
        "ai_advanced_share" => advanced_share,
        "own_tier_share" => own_tier_share
    )

    # ========================================================================
    # DECISION CONFIDENCE
    # ========================================================================

    # Overall decision confidence integrating all dimensions
    uncertainty_penalty = (
        0.30 * estimated_ignorance +
        0.25 * indeterminism_level +
        0.20 * (1.0 - novelty_level_agent) +
        0.25 * perceived_recursion
    )

    ai_confidence_boost = info_quality * avg_ai_trust * 0.2
    hallucination_penalty = hallucination_rate * 0.15

    decision_confidence = clamp(
        0.5 +
        0.3 * competence_trait +
        0.2 * personal_knowledge +
        ai_confidence_boost -
        uncertainty_penalty -
        hallucination_penalty,
        0.1, 0.95
    )

    perception["decision_confidence"] = decision_confidence
    perception["overall_uncertainty"] = 1.0 - decision_confidence

    return perception
end

"""
Get AI tier-specific perception adjustments.
"""
function get_ai_perception_adjustments(
    env::KnightianUncertaintyEnvironment,
    ai_level::String
)::Dict{String,Float64}
    ai_config = get(env.config.AI_LEVELS, ai_level, env.config.AI_LEVELS["none"])

    return Dict{String,Float64}(
        "ignorance_reduction" => Float64(get(ai_config, "info_quality", 0.0)) * 0.3,
        "indeterminism_reduction" => Float64(get(ai_config, "info_quality", 0.0)) * 0.2,
        "novelty_boost" => Float64(get(ai_config, "info_breadth", 0.0)) * 0.15,
        "recursion_awareness" => ai_level == "premium" ? 0.2 : (ai_level == "advanced" ? 0.12 : 0.0)
    )
end
