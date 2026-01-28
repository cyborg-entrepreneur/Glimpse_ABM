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

    # Optional knowledge base reference
    knowledge_base::Union{Nothing,Any}

    # Actor Ignorance State
    actor_ignorance_state::Dict{String,Any}

    # Practical Indeterminism State
    practical_indeterminism_state::Dict{String,Any}

    # Agentic Novelty State
    agentic_novelty_state::Dict{String,Any}

    # Competitive Recursion State
    competitive_recursion_state::Dict{String,Any}

    # Uncertainty evolution history
    uncertainty_evolution::Vector{Any}

    # AI-specific signals
    ai_uncertainty_signals::Dict{String,Any}

    # Innovation tracking (circular buffer)
    innovation_success_tracker::Vector{Dict{String,Any}}

    # Exploration outcomes tracking
    exploration_outcomes::Vector{Dict{String,Any}}

    # Market regime history
    market_regime_history::Vector{String}

    # Opportunity discovery rate
    opportunity_discovery_rate::Float64

    # Global short-term buffers for volatility calculation
    _global_short_term_buffers::Dict{String,Vector{Float64}}

    # Per-agent short-term buffers
    _agent_short_term_buffers::Dict{Int,Dict{String,Vector{Float64}}}

    # Short-term window size
    short_term_window::Int

    # Short-term decay factor
    short_term_decay_factor::Float64

    # Volatility state
    _volatility_state::Dict{String,Any}

    # AI signal history limit
    _ai_signal_history::Int

    # Action history for tracking behavioral patterns
    _action_history::Vector{Dict{String,Any}}

    # Novelty diagnostics
    _novelty_diagnostics::Dict{String,Any}

    # Tier smoothing factors
    _tier_smoothing::Dict{String,Float64}

    # Last environment state for caching
    _last_environment_state::Dict{String,Any}

    # Last alive agents count
    _last_alive_agents::Int

    rng::Random.AbstractRNG
end

function KnightianUncertaintyEnvironment(
    config::EmergentConfig;
    knowledge_base::Union{Nothing,Any} = nothing,
    rng::Random.AbstractRNG = Random.default_rng()
)
    history_window = max(5, Int(getfield_default(config, :NOVELTY_HISTORY_WINDOW, 15)))
    short_term_window = Int(getfield_default(config, :UNCERTAINTY_SHORT_WINDOW, 5))
    short_term_decay = Float64(getfield_default(config, :UNCERTAINTY_SHORT_DECAY, 0.85))
    ai_signal_history = max(10, Int(getfield_default(config, :AI_SIGNAL_HISTORY, 200)))

    return KnightianUncertaintyEnvironment(
        config,
        knowledge_base,
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
            "level" => 0.0,
            "crowding_pressure" => 0.0,
            "ai_pressure" => 0.0,
            "ai_herding_intensity" => 0.0,
            "timing_variability" => 0.0
        ),
        Dict{String,Any}(
            "creative_momentum" => 0.0,
            "disruption_potential" => Dict{String,Float64}(),
            "novelty_level" => 0.0,
            "level" => 0.0,
            "component_scarcity" => 0.5,
            "scarcity_signal" => 0.5,
            "reuse_pressure" => 0.0,
            "new_possibilities" => 0,
            "new_possibility_rate" => 0.0,
            "innovation_births" => 0,
            "tier_drivers" => Dict{String,Any}(),
            "drivers" => Dict{String,Any}()
        ),
        Dict{String,Any}(
            "strategic_opacity" => 0.0,
            "herding_pressure" => Dict{String,Float64}(),
            "game_complexity" => 0.0,
            "level" => 0.0
        ),
        Any[],  # uncertainty_evolution (stores tuples)
        Dict{String,Any}(
            "hallucination_events" => Dict{String,Any}[],
            "confidence_miscalibration" => Float64[],
            "ai_herding_patterns" => Dict{String,Float64}()
        ),
        Dict{String,Any}[],  # innovation_success_tracker
        Dict{String,Any}[],  # exploration_outcomes
        String[],  # market_regime_history
        0.5,  # opportunity_discovery_rate
        Dict{String,Vector{Float64}}(
            "actor_ignorance" => Float64[],
            "practical_indeterminism" => Float64[],
            "agentic_novelty" => Float64[],
            "competitive_recursion" => Float64[]
        ),
        Dict{Int,Dict{String,Vector{Float64}}}(),  # _agent_short_term_buffers
        short_term_window,
        short_term_decay,
        Dict{String,Any}(
            "action_prev" => zeros(4),
            "ai_prev" => zeros(4),
            "market_prev" => 0.0,
            "volatility_metric" => 0.0,
            "history" => Float64[],
            "last_action_shares" => zeros(4),
            "last_ai_shares" => zeros(4)
        ),
        ai_signal_history,
        Dict{String,Any}[],  # _action_history
        Dict{String,Any}(),  # _novelty_diagnostics
        Dict{String,Float64}(
            "none" => 1.0,
            "basic" => 1.0,
            "advanced" => 1.0,
            "premium" => 1.0
        ),
        Dict{String,Any}(),  # _last_environment_state
        Int(getfield_default(config, :N_AGENTS, 100)),  # _last_alive_agents
        rng
    )
end

"""
Get short-term buffer for a metric (global or per-agent).
"""
function _short_buffer(env::KnightianUncertaintyEnvironment, metric::String, agent_id::Union{Int,Nothing})::Vector{Float64}
    if isnothing(agent_id)
        return env._global_short_term_buffers[metric]
    end
    if !haskey(env._agent_short_term_buffers, agent_id)
        env._agent_short_term_buffers[agent_id] = Dict{String,Vector{Float64}}(
            "actor_ignorance" => Float64[],
            "practical_indeterminism" => Float64[],
            "agentic_novelty" => Float64[],
            "competitive_recursion" => Float64[]
        )
    end
    return env._agent_short_term_buffers[agent_id][metric]
end

"""
Update short-term buffer for a metric.
"""
function _update_short_term!(env::KnightianUncertaintyEnvironment, metric::String, value::Float64; agent_id::Union{Int,Nothing} = nothing)
    if isnothing(value) || !isfinite(value)
        return
    end
    buffer = _short_buffer(env, metric, agent_id)
    push!(buffer, Float64(value))
    # Keep buffer size limited
    max_size = env.short_term_window
    while length(buffer) > max_size
        popfirst!(buffer)
    end
end

"""
Decay short-term buffer for a metric.
"""
function _decay_short_term!(env::KnightianUncertaintyEnvironment, metric::String, fallback::Float64; agent_id::Union{Int,Nothing} = nothing)
    buffer = _short_buffer(env, metric, agent_id)
    if !isempty(buffer)
        baseline = buffer[end]
    else
        baseline = isfinite(fallback) ? fallback : 0.0
    end
    if !isfinite(baseline)
        baseline = 0.0
    end
    # No decay: retain last signal strength when no new data arrives
    push!(buffer, baseline)
    max_size = env.short_term_window
    while length(buffer) > max_size
        popfirst!(buffer)
    end
end

"""
Normalize metric value (floor at zero).
"""
function _normalize_metric(env::KnightianUncertaintyEnvironment, metric::String, value::Float64)::Float64
    if !isfinite(value)
        return 0.0
    end
    return max(0.0, value)
end

"""
Get short-term average for a metric.
"""
function _get_short_term_average(env::KnightianUncertaintyEnvironment, metric::String; fallback::Float64 = 0.0, agent_id::Union{Int,Nothing} = nothing)::Float64
    # First try agent-specific buffer
    buffer = nothing
    if !isnothing(agent_id) && haskey(env._agent_short_term_buffers, agent_id)
        buffer = get(env._agent_short_term_buffers[agent_id], metric, nothing)
    end
    # Fall back to global buffer
    if isnothing(buffer)
        buffer = get(env._global_short_term_buffers, metric, Float64[])
    end
    if !isempty(buffer)
        avg = mean(buffer)
    else
        avg = isfinite(fallback) ? fallback : 0.0
    end
    return _normalize_metric(env, metric, avg)
end

"""
Update volatility state based on action and AI share changes.
"""
function _update_volatility_state!(env::KnightianUncertaintyEnvironment, action_shares::Vector{Float64}, ai_shares::Vector{Float64}, market)::Float64
    cfg = env.config

    action_prev = get(env._volatility_state, "action_prev", nothing)
    if isnothing(action_prev) || length(action_prev) != length(action_shares)
        action_prev = zeros(length(action_shares))
    end
    ai_prev = get(env._volatility_state, "ai_prev", nothing)
    if isnothing(ai_prev) || length(ai_prev) != length(ai_shares)
        ai_prev = zeros(length(ai_shares))
    end

    action_delta = mean(abs.(action_shares .- action_prev))
    ai_delta = mean(abs.(ai_shares .- ai_prev))

    market_momentum = hasfield(typeof(market), :market_momentum) ? Float64(market.market_momentum) : 0.0
    market_volatility = hasfield(typeof(market), :volatility) ? Float64(market.volatility) : 0.0
    market_signal = abs(market_momentum) + abs(market_volatility)
    market_prev = Float64(get(env._volatility_state, "market_prev", 0.0))
    market_delta = abs(market_signal - market_prev)

    action_weight = Float64(getfield_default(cfg, :UNCERTAINTY_ACTION_VARIANCE_WEIGHT, 0.10))
    ai_weight = Float64(getfield_default(cfg, :UNCERTAINTY_AI_SWITCH_WEIGHT, 0.07))
    market_weight = Float64(getfield_default(cfg, :UNCERTAINTY_MARKET_RETURN_WEIGHT, 0.12))

    raw = action_weight * action_delta + ai_weight * ai_delta + market_weight * market_delta
    decay = Float64(getfield_default(cfg, :UNCERTAINTY_VOLATILITY_DECAY, 0.85))
    prev_vol = Float64(get(env._volatility_state, "volatility_metric", 0.0))
    smoothed = prev_vol * decay + (1.0 - decay) * raw
    scaling = Float64(getfield_default(cfg, :UNCERTAINTY_VOLATILITY_SCALING, 0.18))
    volatility = smoothed * scaling

    env._volatility_state["action_prev"] = copy(action_shares)
    env._volatility_state["ai_prev"] = copy(ai_shares)
    env._volatility_state["market_prev"] = market_signal
    env._volatility_state["volatility_metric"] = volatility

    return volatility
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
        ai_level = normalize_ai_label(get(action, "ai_level_used", "none"))
        if ai_level == "none"
            continue
        end
        agent_id = get(action, "agent_id", nothing)
        domain = get(action, "ai_analysis_domain", nothing)

        # Track hallucinations
        if get(action, "ai_contains_hallucination", false)
            hallucinations_this_round += 1
            push!(hallucinations, Dict(
                "round" => round_num,
                "agent_id" => agent_id,
                "ai_level" => ai_level,
                "domain" => domain
            ))
        end

        # Track confidence miscalibration
        confidence = get(action, "ai_confidence", nothing)
        accuracy = get(action, "ai_actual_accuracy", nothing)
        try
            if !isnothing(confidence) && !isnothing(accuracy)
                miscal = Float64(confidence) - Float64(accuracy)
                if isfinite(miscal)
                    push!(confidence_vals, miscal)
                end
            end
        catch
            # Ignore type conversion errors
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
    decay = Float64(getfield_default(env.config, :AI_HERDING_DECAY, 1.0))
    for key in collect(keys(herding_patterns))
        herding_patterns[key] *= decay
    end
    for (opp_id, count) in herding_counts
        prior = get(herding_patterns, opp_id, 0.0) * decay
        herding_patterns[opp_id] = prior + count
    end

    # Limit history size
    max_history = env._ai_signal_history
    while length(hallucinations) > max_history
        popfirst!(hallucinations)
    end
    while length(confidence_vals) > max_history
        popfirst!(confidence_vals)
    end

    # Update short-term buffers if hallucinations occurred
    if hallucinations_this_round > 0
        spike = clamp(0.15 + 0.05 * hallucinations_this_round, 0.0, 1.0)
        _update_short_term!(env, "actor_ignorance", spike)
        _update_short_term!(env, "agentic_novelty", spike * 0.6)
    end
end

"""
Summarize agent actions for uncertainty calculation.
Matches Python's _summarize_actions method.
"""
function _summarize_actions(env::KnightianUncertaintyEnvironment, agent_actions::Vector{Dict{String,Any}}, market)::Dict{String,Any}
    summary = Dict{String,Any}(
        "new_combos" => 0,
        "innovate" => 0,
        "new_niches" => 0,
        "explore" => 0,
        "derivative_adoption" => 0,
        "invest" => 0,
        "combo_hhi" => 0.0,
        "sector_hhi" => 0.0,
        "invest_hhi" => 0.0,
        "herding_counts" => Dict{String,Int}(),
        "invest_by_ai" => Dict{String,Int}(),
        "tier_stats" => Dict{String,Any}(),
        "ai_action_correlation" => 0.0  # NEW: Track AI-induced action correlation
    )

    invest_counts = Dict{String,Int}()
    invest_by_ai = Dict{String,Int}()
    tiers = ["none", "basic", "advanced", "premium"]

    # NEW: Track actions and opportunities by tier for correlation calculation
    actions_by_tier = Dict{String,Vector{String}}(tier => String[] for tier in tiers)
    opportunities_by_tier = Dict{String,Dict{String,Int}}(tier => Dict{String,Int}() for tier in tiers)

    tier_template() = Dict{String,Int}(
        "total_actions" => 0,
        "innovate" => 0,
        "new_combos" => 0,
        "reuse_hits" => 0,
        "explore" => 0,
        "new_niches" => 0,
        "invest" => 0,
        "derivative_adoption" => 0
    )

    tier_stats = Dict{String,Dict{String,Int}}(tier => tier_template() for tier in tiers)

    for action in agent_actions
        act_type = get(action, "action", nothing)
        tier = normalize_ai_label(get(action, "ai_level_used", "none"))

        if !haskey(tier_stats, tier)
            tier_stats[tier] = tier_template()
        end
        tier_stats[tier]["total_actions"] += 1

        # NEW: Track actions for correlation calculation
        if !isnothing(act_type)
            push!(actions_by_tier[tier], act_type)
        end

        if act_type == "innovate"
            summary["innovate"] += 1
            if get(action, "is_new_combination", false)
                summary["new_combos"] += 1
                tier_stats[tier]["new_combos"] += 1
            elseif !isnothing(get(action, "combination_signature", nothing))
                tier_stats[tier]["reuse_hits"] += 1
            end
            tier_stats[tier]["innovate"] += 1
        elseif act_type == "explore"
            summary["explore"] += 1
            if get(action, "created_niche", false) || get(action, "discovered_niche", false)
                summary["new_niches"] += 1
                tier_stats[tier]["new_niches"] += 1
            end
            tier_stats[tier]["explore"] += 1
        elseif act_type == "invest"
            summary["invest"] += 1
            tier_stats[tier]["invest"] += 1
            if get(action, "invested_derivative", false)
                summary["derivative_adoption"] += 1
                tier_stats[tier]["derivative_adoption"] += 1
            end
            details = get(action, "chosen_opportunity_details", Dict())
            opp_id = get(details, "id", get(action, "opportunity_id", nothing))
            if !isnothing(opp_id)
                opp_id_str = string(opp_id)
                invest_counts[opp_id_str] = get(invest_counts, opp_id_str, 0) + 1
                # NEW: Track opportunity choices by tier
                opportunities_by_tier[tier][opp_id_str] = get(opportunities_by_tier[tier], opp_id_str, 0) + 1
            end
            invest_by_ai[tier] = get(invest_by_ai, tier, 0) + 1
        end
    end

    # Compute investment HHI
    invest_total = summary["invest"]
    if invest_total > 0 && !isempty(invest_counts)
        summary["invest_hhi"] = sum((count / invest_total)^2 for count in values(invest_counts))
    end
    summary["herding_counts"] = invest_counts
    summary["invest_by_ai"] = invest_by_ai

    # Compute tier-specific rates
    tier_combo_rate = Dict{String,Float64}()
    tier_reuse_pressure = Dict{String,Float64}()
    tier_new_poss_rate = Dict{String,Float64}()
    tier_adoption_rate = Dict{String,Float64}()

    for (tier, stats) in tier_stats
        innov = max(1, stats["innovate"])
        invest_count = max(1, stats["invest"])
        action_total = max(1, stats["total_actions"])
        tier_combo_rate[tier] = clamp(stats["new_combos"] / innov, 0.0, 1.0)
        tier_reuse_pressure[tier] = clamp(stats["reuse_hits"] / innov, 0.0, 1.0)
        tier_new_poss_rate[tier] = clamp((stats["new_combos"] + stats["new_niches"]) / action_total, 0.0, 1.0)
        tier_adoption_rate[tier] = clamp(stats["derivative_adoption"] / invest_count, 0.0, 1.0)
    end

    summary["tier_stats"] = tier_stats
    summary["tier_combo_rate"] = tier_combo_rate
    summary["tier_reuse_pressure"] = tier_reuse_pressure
    summary["tier_new_possibility_rate"] = tier_new_poss_rate
    summary["tier_adoption_rate"] = tier_adoption_rate

    # Get combination diversity metrics from market if available
    if !isnothing(market)
        combo_hhi, sector_hhi = get_combination_diversity_metrics(market)
        summary["combo_hhi"] = combo_hhi
        summary["sector_hhi"] = sector_hhi
    end

    # NEW: Calculate AI action correlation for competitive recursion
    # When agents with AI choose similar actions/opportunities, recursion increases
    ai_tiers = ["basic", "advanced", "premium"]
    ai_action_counts = sum(length(actions_by_tier[tier]) for tier in ai_tiers)

    if ai_action_counts >= 2
        # Calculate correlation based on opportunity clustering among AI agents
        ai_opportunity_hhi = 0.0
        total_ai_investments = sum(sum(values(opportunities_by_tier[tier])) for tier in ai_tiers)

        if total_ai_investments > 0
            # HHI of opportunities among AI agents (higher = more correlated)
            all_ai_opps = Dict{String,Int}()
            for tier in ai_tiers
                for (opp_id, count) in opportunities_by_tier[tier]
                    all_ai_opps[opp_id] = get(all_ai_opps, opp_id, 0) + count
                end
            end

            ai_opportunity_hhi = sum((count / total_ai_investments)^2 for count in values(all_ai_opps))
        end

        # Correlation ranges from baseline 0.3 (no AI) to 0.6+ (high AI adoption with clustering)
        # Scale by AI adoption rate and opportunity clustering
        ai_adoption_rate = ai_action_counts / max(1, length(agent_actions))
        summary["ai_action_correlation"] = clamp(
            0.30 + 0.30 * ai_adoption_rate * ai_opportunity_hhi,
            0.30,
            0.70
        )
    else
        summary["ai_action_correlation"] = 0.30  # Baseline correlation without AI
    end

    return summary
end

"""
Compute component scarcity from knowledge base.
"""
function _compute_component_scarcity(env::KnightianUncertaintyEnvironment)::Float64
    if !isnothing(env.knowledge_base)
        raw_scarcity = Float64(get_component_scarcity_metric(env.knowledge_base))
    else
        raw_scarcity = Float64(get(env.agentic_novelty_state, "scarcity_signal", 0.5))
    end
    prev = get(env.agentic_novelty_state, "scarcity_signal", nothing)
    if isnothing(prev)
        return raw_scarcity
    end
    blend = clamp(0.65 * prev + 0.35 * raw_scarcity, 0.0, 1.0)
    return blend
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
Matches Python implementation exactly.
"""
function measure_uncertainty_state!(
    env::KnightianUncertaintyEnvironment,
    market::MarketEnvironment,
    agent_actions::Vector{Dict{String,Any}},
    innovations::Vector{Innovation},
    round_num::Int
)::Dict{String,Dict{String,Any}}
    state = Dict{String,Dict{String,Any}}()
    ai_effects = Dict{String,Float64}()
    total_actions = length(agent_actions)

    # Count actions by type
    action_counter = Dict{String,Int}()
    for action in agent_actions
        act_type = get(action, "action", "maintain")
        action_counter[act_type] = get(action_counter, act_type, 0) + 1
    end

    if total_actions > 0
        action_shares = Float64[
            get(action_counter, "invest", 0) / total_actions,
            get(action_counter, "innovate", 0) / total_actions,
            get(action_counter, "explore", 0) / total_actions,
            get(action_counter, "maintain", 0) / total_actions
        ]
    else
        action_shares = Float64[0.0, 0.0, 0.0, 1.0]
    end

    # Count AI tier usage
    ai_counter = Dict{String,Int}()
    for action in agent_actions
        tier = normalize_ai_label(get(action, "ai_level_used", "none"))
        ai_counter[tier] = get(ai_counter, tier, 0) + 1
    end

    if total_actions > 0
        ai_shares = Float64[
            get(ai_counter, "none", 0) / total_actions,
            get(ai_counter, "basic", 0) / total_actions,
            get(ai_counter, "advanced", 0) / total_actions,
            get(ai_counter, "premium", 0) / total_actions
        ]
    else
        ai_shares = Float64[1.0, 0.0, 0.0, 0.0]
    end

    ai_share_none = ai_shares[1]

    # Summarize actions (tier-specific tracking)
    action_summary = _summarize_actions(env, agent_actions, market)

    # Update tier invest shares on market
    tier_invest_share = Dict{String,Float64}()
    invest_total = get(action_summary, "invest", 0)
    if invest_total > 0
        for (tier, count) in get(action_summary, "invest_by_ai", Dict())
            tier_invest_share[tier] = clamp(count / invest_total, 0.0, 1.0)
        end
    end
    if hasfield(typeof(market), :_tier_invest_share) && !isempty(tier_invest_share)
        market._tier_invest_share = tier_invest_share
    end
    if isempty(tier_invest_share)
        tier_invest_share = Dict{String,Float64}(
            "none" => ai_shares[1],
            "basic" => ai_shares[2],
            "advanced" => ai_shares[3],
            "premium" => ai_shares[4]
        )
    end

    # Record action history
    push!(env._action_history, action_summary)
    history_window = max(5, Int(getfield_default(env.config, :NOVELTY_HISTORY_WINDOW, 15)))
    while length(env._action_history) > history_window
        popfirst!(env._action_history)
    end

    # AI herding patterns
    ai_herding_patterns = get(env.ai_uncertainty_signals, "ai_herding_patterns", Dict())
    if !isempty(ai_herding_patterns)
        herding_total = sum(values(ai_herding_patterns))
        ai_herding_intensity = clamp(herding_total / max(1.0, total_actions), 0.0, 1.0)
    else
        ai_herding_intensity = 0.0
    end

    # Count new possibilities
    new_possibilities = 0
    for action in agent_actions
        if get(action, "discovered_niche", false) || get(action, "created_opportunity", false) || !isnothing(get(action, "new_opportunity_id", nothing))
            new_possibilities += 1
        end
    end
    innovation_births = count(inn -> !isnothing(inn), innovations)
    new_possibility_rate = new_possibilities / max(1, total_actions)

    # Update volatility state
    volatility = _update_volatility_state!(env, action_shares, ai_shares, market)
    env._volatility_state["last_action_shares"] = copy(action_shares)
    env._volatility_state["last_ai_shares"] = copy(ai_shares)

    share_invest, share_innovate, share_explore, share_maintain = action_shares

    # ========================================================================
    # ACTOR IGNORANCE
    # ========================================================================
    # Gather knowledge gaps from agent perceptions
    knowledge_gaps = Dict{String,Float64}()
    for action in agent_actions
        perception = get(action, "perception_at_decision", Dict())
        ignorance = get(perception, "actor_ignorance", Dict())
        gap_map = get(ignorance, "knowledge_gaps", Dict())
        if !isempty(gap_map)
            merge!(knowledge_gaps, gap_map)
        end
    end

    # Calculate knowledge-based metrics
    avg_knowledge = if !isnothing(env.knowledge_base)
        Float64(get_average_agent_knowledge(env.knowledge_base))
    else
        0.0
    end
    total_opportunities = length(market.opportunities)
    n_agents = Int(getfield_default(env.config, :N_AGENTS, 1))
    knowledge_norm = avg_knowledge / max(1.0, total_opportunities / max(1, n_agents / 4))

    gap_values = collect(values(knowledge_gaps))
    gap_pressure = !isempty(gap_values) ? clamp(safe_mean(gap_values), 0.0, 1.0) : 0.0
    gap_coverage = total_opportunities > 0 ? length(knowledge_gaps) / total_opportunities : 0.0
    # Use rolling window for hallucination rate so it can decay over time
    # hallucination_events is a Vector{Dict} where each entry is one event
    hallucination_events = env.ai_uncertainty_signals["hallucination_events"]
    window_size = min(100, env._ai_signal_history)
    recent_hallucinations = if length(hallucination_events) <= window_size
        length(hallucination_events)
    else
        # Count hallucinations in recent window (last window_size events)
        length(hallucination_events[max(1, end-window_size+1):end])
    end
    hallucination_rate = recent_hallucinations / max(1, window_size)
    knowledge_gap_term = 1.0 - clamp(knowledge_norm, 0.0, 1.0)

    # ACTOR IGNORANCE: Linear additive formula (consistent with other dimensions)
    # Base level + weighted drivers that can accumulate or decline
    # Theoretical rationale: Ignorance can spike during paradigm shifts or collapse
    # with learning breakthroughs - sigmoid would artificially constrain these dynamics

    actor_level = clamp(
        0.25 +                              # base level
        0.28 * knowledge_gap_term +         # knowledge gaps increase ignorance
        0.18 * gap_pressure +               # pressure from gaps
        0.12 * gap_coverage +               # breadth of gaps
        0.15 * hallucination_rate -         # AI hallucinations increase ignorance
        0.10 * share_explore +              # exploration reduces ignorance
        0.03 * share_maintain,              # maintaining slightly increases (not learning)
        0.0, 1.0
    )

    # Baseline (no AI) actor ignorance
    actor_level_no_ai = clamp(
        0.25 +
        0.28 * knowledge_gap_term +
        0.18 * gap_pressure +
        0.12 * gap_coverage -
        0.10 * share_explore +
        0.03 * share_maintain,
        0.0, 1.0
    )
    ai_effects["ai_ignorance_delta"] = actor_level - actor_level_no_ai

    _update_short_term!(env, "actor_ignorance", actor_level)
    env.actor_ignorance_state["level"] = actor_level

    state["actor_ignorance"] = Dict{String,Any}(
        "level" => actor_level,
        "knowledge_gaps" => knowledge_gaps,
        "gap_pressure" => gap_pressure,
        "volatility" => volatility,
        "ai_delta" => get(ai_effects, "ai_ignorance_delta", 0.0)
    )

    # ========================================================================
    # PRACTICAL INDETERMINISM
    # ========================================================================
    timing_levels = Float64[]
    timing_pressure = Dict{String,Float64}()
    if !isempty(market.opportunities)
        for opp in market.opportunities
            acceleration = Float64(opp.market_share) * Float64(opp.competition)
            timing_pressure[opp.id] = acceleration
            push!(timing_levels, acceleration)
        end
    end
    base_practical = !isempty(timing_levels) ? safe_mean(timing_levels) : 0.3
    timing_variability = !isempty(timing_levels) ? std(timing_levels) : 0.0

    market_volatility = Float64(market.volatility)
    regime = hasfield(typeof(market), :market_regime) ? market.market_regime : "normal"
    regime_uncertainty = get(Dict(
        "boom" => 0.35, "growth" => 0.22, "normal" => 0.18,
        "volatile" => 0.32, "recession" => 0.45, "crisis" => 0.65
    ), regime, 0.2)
    path_component = clamp(timing_variability * 1.5, 0.0, 1.0)

    # Get crowding metrics
    crowding_metrics = hasfield(typeof(market), :_last_crowding_metrics) ? market._last_crowding_metrics : Dict()
    crowding_index = Float64(get(crowding_metrics, "crowding_index", 0.0))
    ai_usage_pressure = Float64(get(crowding_metrics, "ai_usage_share", 0.0))
    crowding_baseline = 0.25
    crowding_pressure = max(0.0, crowding_index - crowding_baseline)
    ai_pressure_term = max(0.0, ai_usage_pressure - 0.30)

    crowd_weight = Float64(getfield_default(env.config, :UNCERTAINTY_CROWDING_WEIGHT, 0.12))
    competitive_weight = Float64(getfield_default(env.config, :UNCERTAINTY_COMPETITIVE_WEIGHT, 0.08))
    herding_weight = Float64(getfield_default(env.config, :UNCERTAINTY_AI_HERDING_WEIGHT, 0.1))

    practical_level = clamp(
        0.25 +
        0.30 * market_volatility +
        0.25 * regime_uncertainty +
        0.25 * clamp(base_practical, 0.0, 1.0) +
        0.20 * path_component +
        0.15 * share_invest +
        crowd_weight * crowding_pressure +
        competitive_weight * ai_pressure_term +
        herding_weight * ai_herding_intensity,
        0.0, 1.0
    )

    practical_level_no_ai = clamp(
        0.25 +
        0.30 * market_volatility +
        0.25 * regime_uncertainty +
        0.25 * clamp(base_practical, 0.0, 1.0) +
        0.20 * path_component +
        0.15 * share_invest,
        0.0, 1.0
    )
    ai_effects["ai_indeterminism_delta"] = practical_level - practical_level_no_ai

    _update_short_term!(env, "practical_indeterminism", practical_level)
    env.practical_indeterminism_state["path_volatility"] = path_component
    env.practical_indeterminism_state["timing_variability"] = timing_variability
    env.practical_indeterminism_state["regime_instability"] = regime_uncertainty
    env.practical_indeterminism_state["crowding_pressure"] = crowding_pressure
    env.practical_indeterminism_state["ai_pressure"] = ai_usage_pressure
    env.practical_indeterminism_state["ai_herding_intensity"] = ai_herding_intensity
    env.practical_indeterminism_state["level"] = practical_level

    state["practical_indeterminism"] = Dict{String,Any}(
        "level" => practical_level,
        "timing_pressure" => timing_pressure,
        "volatility" => volatility,
        "crowding_pressure" => crowding_pressure,
        "ai_herding_intensity" => ai_herding_intensity,
        "ai_delta" => get(ai_effects, "ai_indeterminism_delta", 0.0)
    )

    # ========================================================================
    # AGENTIC NOVELTY
    # ========================================================================
    history = env._action_history
    window_size = min(5, length(history))
    recent_history = window_size > 0 ? history[end-window_size+1:end] : []

    total_innov = sum(get(item, "innovate", 0) for item in recent_history)
    total_explore = sum(get(item, "explore", 0) for item in recent_history)
    total_invest = sum(get(item, "invest", 0) for item in recent_history)

    combo_rate = total_innov > 0 ? sum(get(item, "new_combos", 0) for item in recent_history) / total_innov : 0.0
    niche_rate = total_explore > 0 ? sum(get(item, "new_niches", 0) for item in recent_history) / total_explore : 0.0
    adoption_rate = total_invest > 0 ? sum(get(item, "derivative_adoption", 0) for item in recent_history) / total_invest : 0.0

    combo_hhi_avg = safe_mean([get(item, "combo_hhi", 0.0) for item in recent_history])
    sector_hhi_avg = safe_mean([get(item, "sector_hhi", 0.0) for item in recent_history])
    diversity_term = clamp(1.0 - combo_hhi_avg, 0.0, 1.0)
    sector_diversity = clamp(1.0 - sector_hhi_avg, 0.0, 1.0)

    scarcity_signal = _compute_component_scarcity(env)
    innovation_intensity = innovation_births / max(1, total_actions)

    # Disruption state with decay
    disruption_state = get(env.agentic_novelty_state, "disruption_potential", Dict{String,Float64}())
    if isempty(disruption_state) && hasfield(typeof(env.config), :SECTORS)
        for sector in env.config.SECTORS
            disruption_state[sector] = 0.3
        end
    end

    disruption_decay = Float64(getfield_default(env.config, :DISRUPTION_STATE_DECAY, 0.92))
    for sector in collect(keys(disruption_state))
        disruption_state[sector] = clamp(disruption_state[sector] * disruption_decay, 0.05, 0.95)
    end

    # Update disruption from innovations
    for innovation in innovations
        if isnothing(innovation)
            continue
        end
        sector = innovation.sector
        if isnothing(sector)
            continue
        end
        success_flag = innovation.success ? 1.0 : 0.0
        novelty_score = Float64(isnothing(innovation.novelty) ? 0.0 : innovation.novelty)
        impact_signal = Float64(hasfield(typeof(innovation), :market_impact) ? (isnothing(innovation.market_impact) ? 0.0 : innovation.market_impact) : 0.0)
        bump = 0.03 + 0.04 * novelty_score + 0.02 * max(impact_signal, 0.0) + 0.04 * success_flag
        disruption_state[sector] = clamp(get(disruption_state, sector, 0.3) + bump, 0.05, 0.95)
    end

    # Update from explore actions
    for action in agent_actions
        if get(action, "action", "") != "explore"
            continue
        end
        sector = get(action, "base_sector", get(action, "domain", nothing))
        if isnothing(sector)
            continue
        end
        bump = (get(action, "created_niche", false) || get(action, "discovered_niche", false)) ? 0.02 : 0.01
        disruption_state[sector] = clamp(get(disruption_state, sector, 0.3) + bump, 0.05, 0.95)
    end

    disruption_avg = !isempty(disruption_state) ? clamp(fast_mean(values(disruption_state)), 0.0, 1.0) : 0.3
    innovation_momentum = clamp(innovation_intensity + new_possibility_rate, 0.0, 1.0)
    reuse_pressure = 0.6 * combo_hhi_avg + 0.4 * sector_hhi_avg

    # AGENTIC NOVELTY: Linear additive formula (consistent with other dimensions)
    # Theoretical rationale: Novelty can collapse (combinatorial exhaustion) or
    # explode (paradigm shifts) - sigmoid would artificially prevent these valid extremes

    novelty_diagnostics = Dict{String,Any}(
        "combo_rate" => combo_rate,
        "niche_rate" => niche_rate,
        "adoption_rate" => adoption_rate,
        "diversity_term" => diversity_term,
        "sector_diversity" => sector_diversity,
        "scarcity" => scarcity_signal,
        "innovation_intensity" => innovation_intensity,
        "new_possibility_rate" => new_possibility_rate,
        "reuse_pressure" => reuse_pressure,
        "disruption_avg" => disruption_avg
    )

    # AI quality component
    ai_quality = clamp(ai_usage_pressure, 0.0, 1.0)
    ai_novelty_uplift = Float64(getfield_default(env.config, :AI_NOVELTY_UPLIFT, 0.08))

    # Linear additive: drivers that increase novelty potential minus drags
    # Positive: new combinations, new possibilities, niches, innovation, disruption, diversity
    # Negative: derivative adoption (known patterns), reuse pressure (combinatorial exhaustion)
    agentic_level = clamp(
        0.25 +                                  # base level
        0.20 * combo_rate +                     # new combinations increase novelty
        0.18 * new_possibility_rate +           # rate of new possibilities
        0.15 * niche_rate +                     # niche discovery
        0.12 * innovation_intensity +           # innovation activity
        0.10 * disruption_avg +                 # disruption potential
        0.08 * sector_diversity +               # diversity enables novelty
        -0.15 * adoption_rate +                 # derivative adoption REDUCES novelty
        -0.20 * reuse_pressure +                # reuse/exhaustion reduces novelty
        ai_novelty_uplift * ai_quality,         # AI effect on novelty
        0.0, 1.0
    )

    agentic_level_no_ai = clamp(
        0.25 +
        0.20 * combo_rate +
        0.18 * new_possibility_rate +
        0.15 * niche_rate +
        0.12 * innovation_intensity +
        0.10 * disruption_avg +
        0.08 * sector_diversity +
        -0.15 * adoption_rate +
        -0.20 * reuse_pressure,
        0.0, 1.0
    )

    _update_short_term!(env, "agentic_novelty", agentic_level)

    # Update agentic novelty state
    env.agentic_novelty_state["creative_momentum"] = agentic_level
    env.agentic_novelty_state["new_possibilities"] = new_possibilities
    env.agentic_novelty_state["new_possibility_rate"] = new_possibility_rate
    env.agentic_novelty_state["innovation_births"] = innovation_births
    env.agentic_novelty_state["scarcity_signal"] = scarcity_signal
    env.agentic_novelty_state["recent_scarcity_signal"] = scarcity_signal
    env.agentic_novelty_state["novelty_level"] = agentic_level
    env.agentic_novelty_state["drivers"] = novelty_diagnostics
    env.agentic_novelty_state["disruption_potential"] = disruption_state
    env.agentic_novelty_state["level"] = agentic_level
    env._novelty_diagnostics = novelty_diagnostics

    # Get tier-specific rates from action summary
    tier_combo_rate = get(action_summary, "tier_combo_rate", Dict{String,Float64}())
    tier_reuse_pressure = get(action_summary, "tier_reuse_pressure", Dict{String,Float64}())
    tier_new_poss_rate = get(action_summary, "tier_new_possibility_rate", Dict{String,Float64}())
    tier_adoption_rate = get(action_summary, "tier_adoption_rate", Dict{String,Float64}())

    tier_drivers = Dict{String,Any}(
        "combo_rate" => tier_combo_rate,
        "reuse_pressure" => tier_reuse_pressure,
        "new_possibility_rate" => tier_new_poss_rate,
        "adoption_rate" => tier_adoption_rate
    )
    env.agentic_novelty_state["tier_drivers"] = tier_drivers

    state["agentic_novelty"] = Dict{String,Any}(
        "level" => agentic_level,
        "novelty_potential" => agentic_level,
        "new_possibilities" => new_possibilities,
        "new_possibility_rate" => new_possibility_rate,
        "innovation_births" => innovation_births,
        "volatility" => volatility,
        "scarcity_signal" => scarcity_signal,
        "component_scarcity" => scarcity_signal,
        "disruption_potential" => disruption_state,
        "combo_rate" => combo_rate,
        "reuse_pressure" => reuse_pressure,
        "adoption_rate" => adoption_rate,
        "drivers" => novelty_diagnostics,
        "ai_delta" => agentic_level - agentic_level_no_ai,
        "tier_combo_rate" => tier_combo_rate,
        "tier_reuse_pressure" => tier_reuse_pressure,
        "tier_new_possibility_rate" => tier_new_poss_rate,
        "tier_adoption_rate" => tier_adoption_rate
    )

    # ========================================================================
    # COMPETITIVE RECURSION
    # ========================================================================
    invest_hhi = get(action_summary, "invest_hhi", 0.0)
    premium_share = ai_shares[4]
    tier_reuse_map = get(action_summary, "tier_reuse_pressure", Dict())
    premium_reuse = Float64(get(tier_reuse_map, "premium", 0.0))
    knowledge_overlap = Float64(get(action_summary, "sector_hhi", 0.0))

    agent_count = max(1, Int(getfield_default(env.config, :N_AGENTS, 1)))
    alive_agents = env._last_alive_agents
    population_factor = clamp(alive_agents / agent_count, 0.15, 1.0)

    # Track AI action correlation for analysis (no direct effect on recursion)
    ai_correlation = Float64(get(action_summary, "ai_action_correlation", 0.30))

    # Recursion weights from config
    rw = getfield_default(env.config, :RECURSION_WEIGHTS, Dict())
    crowd_w = Float64(get(rw, "crowd_weight", 0.35))
    vol_w = Float64(get(rw, "volatility_weight", 0.30))
    herd_w = Float64(get(rw, "ai_herd_weight", 0.40))
    premium_reuse_w = Float64(get(rw, "premium_reuse_weight", 0.20))

    # Crowding component based on BEHAVIORAL convergence, not just tier adoption
    # Reduced direct premium_share effect (was 0.35) - recursion should emerge from
    # actual convergent behavior, not merely from AI tier selection
    # The herding/convergence effects will still emerge through invest_hhi and
    # ai_herding_intensity (which measures actual opportunity clustering)
    crowding_component = (
        0.45 * invest_hhi +            # Concentration of investments (behavioral)
        0.15 * premium_share +          # Reduced from 0.35 - mild anticipation effect
        crowd_w * crowding_pressure +   # Actual crowding pressure (behavioral)
        0.12 * knowledge_overlap        # Knowledge concentration
    )

    scale = 0.5 + 0.5 * population_factor
    recursion_level = clamp(
        crowding_component * scale +
        vol_w * volatility +
        herd_w * ai_herding_intensity +
        premium_reuse_w * premium_reuse,
        0.0, 1.0
    )

    recursion_level_no_ai = clamp(
        crowding_component * scale +
        0.15 * volatility,
        0.0, 1.0
    )
    ai_effects["ai_recursion_delta"] = recursion_level - recursion_level_no_ai

    _update_short_term!(env, "competitive_recursion", recursion_level)

    herding_pressure_map = Dict{String,Float64}()
    herding_counts = get(action_summary, "herding_counts", Dict())
    invest_count = max(1, get(action_summary, "invest", 1))
    for (opp_id, count) in herding_counts
        herding_pressure_map[opp_id] = count / invest_count
    end

    env.competitive_recursion_state["level"] = recursion_level
    env.competitive_recursion_state["herding_pressure"] = herding_pressure_map

    state["competitive_recursion"] = Dict{String,Any}(
        "level" => recursion_level,
        "herding_pressure" => herding_pressure_map,
        "volatility" => volatility,
        "ai_premium_share" => premium_share,
        "ai_herding_intensity" => ai_herding_intensity,
        "ai_action_correlation" => ai_correlation,  # NEW: Track correlation metric
        "ai_delta" => get(ai_effects, "ai_recursion_delta", 0.0)
    )

    # Record evolution as tuple (round, state)
    push!(env.uncertainty_evolution, (round_num, state))
    env._last_environment_state = state

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
action_history : Vector{String}
    Agent's recent action history for behavioral pattern analysis
ai_learning_profile : AILearningProfile
    Agent's learned understanding of AI capabilities
agent_id : Int
    Agent identifier for agent-specific short-term buffering
agent_resources : Any
    Agent's current resource state
cached_metrics : Dict
    Pre-computed metrics to avoid redundant calculation
recent_outcomes : Vector{Dict}
    Recent investment/innovation outcomes

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
    ai_learning_profile::Union{AILearningProfile,Nothing} = nothing,
    agent_id::Union{Int,Nothing} = nothing,
    agent_resources::Any = nothing,
    cached_metrics::Union{Dict{String,Any},Nothing} = nothing,
    recent_outcomes::Vector{Dict{String,Any}} = Dict{String,Any}[]
)::Dict{String,Any}
    perception = Dict{String,Any}()
    env_state = env._last_environment_state
    current_round = get(market_conditions, "round", 0)
    volatility = Float64(get(env._volatility_state, "volatility_metric", 0.0))

    # Get global action shares from volatility state
    global_action_shares = get(env._volatility_state, "last_action_shares", zeros(4))
    if isnothing(global_action_shares) || length(global_action_shares) != 4
        global_share_invest = global_share_innovate = global_share_explore = global_share_maintain = 0.0
    else
        global_share_invest, global_share_innovate, global_share_explore, global_share_maintain = Float64.(global_action_shares)
    end

    # Compute agent-specific action shares from history
    agent_counts = Dict{String,Int}()
    for action in action_history
        agent_counts[action] = get(agent_counts, action, 0) + 1
    end
    agent_total = length(action_history)

    if agent_total > 0
        agent_share_invest = get(agent_counts, "invest", 0) / agent_total
        agent_share_innovate = get(agent_counts, "innovate", 0) / agent_total
        agent_share_explore = get(agent_counts, "explore", 0) / agent_total
        agent_share_maintain = get(agent_counts, "maintain", 0) / agent_total
    else
        agent_share_invest = global_share_invest
        agent_share_innovate = global_share_innovate
        agent_share_explore = global_share_explore
        agent_share_maintain = global_share_maintain
    end

    # Blend agent-specific and global action shares (matching Python's 0.85 blend)
    blend = 0.85
    share_invest = blend * agent_share_invest + (1.0 - blend) * global_share_invest
    share_innovate = blend * agent_share_innovate + (1.0 - blend) * global_share_innovate
    share_explore = blend * agent_share_explore + (1.0 - blend) * global_share_explore
    share_maintain = blend * agent_share_maintain + (1.0 - blend) * global_share_maintain

    # Get AI configuration
    ai_config = get(env.config.AI_LEVELS, ai_level, env.config.AI_LEVELS["none"])
    info_quality = Float64(hasfield(typeof(ai_config), :info_quality) ? ai_config.info_quality : get(ai_config, "info_quality", 0.0))
    info_breadth = Float64(hasfield(typeof(ai_config), :info_breadth) ? ai_config.info_breadth : get(ai_config, "info_breadth", 0.0))

    # Get global AI shares
    last_ai_shares = get(env._volatility_state, "last_ai_shares", zeros(4))
    if isnothing(last_ai_shares) || length(last_ai_shares) != 4
        ai_shares = Float64[1.0, 0.0, 0.0, 0.0]
    else
        ai_shares = Float64.(last_ai_shares)
    end

    # Extract agent traits
    competence_trait = Float64(get(agent_traits, "competence", 0.5))
    ai_trust_trait = Float64(get(agent_traits, "ai_trust", 0.5))
    exploration_trait = Float64(get(agent_traits, "exploration_tendency", 0.5))
    innovation_trait = Float64(get(agent_traits, "innovativeness", 0.5))
    uncertainty_tolerance = Float64(get(agent_traits, "uncertainty_tolerance", 0.5))
    analytical_ability = Float64(get(agent_traits, "analytical_ability", 0.5))
    risk_tolerance = Float64(get(agent_traits, "risk_tolerance", 0.5))

    # AI trust calibration
    base_ai_trust = ai_trust_trait
    avg_ai_trust = base_ai_trust
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
    avg_ai_trust = clamp(0.5 * base_ai_trust + 0.5 * avg_ai_trust, 0.0, 1.0)

    # Tier smoothing factor (matching Python)
    tier_key = normalize_ai_label(ai_level)
    tier_factor = get(env._tier_smoothing, tier_key, 1.0)

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
    knowledge_deficit = clamp(knowledge_deficit - 0.2 * info_breadth, 0.0, 1.0)

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

    gap_pressure = isempty(knowledge_gaps) ? 0.0 : clamp(fast_mean(values(knowledge_gaps)), 0.0, 1.0)

    # Final ignorance level
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

    # Short-term buffer integration (matching Python)
    short_actor = _get_short_term_average(env, "actor_ignorance"; fallback=estimated_ignorance, agent_id=agent_id)
    quality_term = 0.3 * info_quality + 0.2 * info_breadth
    actor_mix_base = 0.45 + quality_term - 0.25 * knowledge_deficit
    short_weight = clamp(0.32 - 0.28 * info_quality + 0.12 * tier_factor, 0.03, 0.45)
    actor_mix = clamp(actor_mix_base, 0.05, 0.95)
    ignorance_level = (1.0 - short_weight) * actor_mix * estimated_ignorance + short_weight * short_actor
    ignorance_level = clamp(ignorance_level, 0.0, 1.0)

    # Update short-term buffer
    _update_short_term!(env, "actor_ignorance", ignorance_level; agent_id=agent_id)

    # Knowledge signal (matching Python structure)
    knowledge_signal = Dict{String,Any}(
        "confidence" => max(0.0, 1.0 - ignorance_level),
        "ignorance_level" => ignorance_level,
        "knowledge_gaps" => knowledge_gaps,
        "gap_pressure" => gap_pressure,
        "info_quality" => info_quality,
        "info_breadth" => info_breadth,
        "personal_knowledge" => personal_knowledge,
        "knowledge_span" => knowledge_span,
        "discovery_momentum" => discovery_momentum,
        "volatility" => volatility
    )
    perception["knowledge_signal"] = knowledge_signal

    perception["actor_ignorance"] = Dict{String,Any}(
        "ignorance_level" => ignorance_level,
        "level" => ignorance_level,
        "knowledge_gaps" => knowledge_gaps,
        "info_sufficiency" => info_quality,
        "discovery_momentum" => discovery_momentum,
        "volatility" => volatility,
        "gap_pressure" => gap_pressure
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

    indeterminism_value = clamp(
        agent_path_risk + 0.15 * agent_volatility_factor + agent_uncertainty + system_term + crowd_term,
        0.0, 1.0
    )

    # Short-term buffer integration (matching Python)
    short_practical = _get_short_term_average(env, "practical_indeterminism"; fallback=indeterminism_value, agent_id=agent_id)
    short_weight_practical = clamp(0.34 - 0.30 * info_quality + 0.15 * tier_factor, 0.04, 0.5)
    practical_mix = clamp(0.25 + 0.45 * info_quality + 0.25 * info_breadth - 0.2 * tier_factor, 0.1, 0.9)
    indeterminism_level = clamp(
        (1.0 - short_weight_practical) * practical_mix * indeterminism_value + short_weight_practical * short_practical,
        0.0, 1.0
    )

    # Update short-term buffer
    _update_short_term!(env, "practical_indeterminism", indeterminism_level; agent_id=agent_id)

    # Execution risk signal (matching Python structure)
    execution_signal = Dict{String,Any}(
        "risk_level" => indeterminism_level,
        "timing_pressure" => timing_pressure,
        "market_regime_risk" => regime_uncertainty,
        "volatility" => volatility,
        "crowding_pressure" => crowding_pressure,
        "ai_pressure" => ai_pressure_level,
        "ai_herding_intensity" => ai_herding_intensity
    )
    perception["execution_risk"] = execution_signal

    perception["practical_indeterminism"] = Dict{String,Any}(
        "indeterminism_level" => indeterminism_level,
        "level" => indeterminism_level,
        "timing_criticality" => timing_pressure,
        "timing_pressure" => timing_pressure,
        "market_regime_risk" => regime_uncertainty,
        "regime_stability" => regime_uncertainty,
        "volatility" => volatility,
        "crowding_pressure" => crowding_pressure,
        "ai_pressure" => ai_pressure_level,
        "ai_herding_intensity" => ai_herding_intensity
    )

    # ========================================================================
    # AGENTIC NOVELTY PERCEPTION
    # ========================================================================

    novelty_state = env.agentic_novelty_state
    recent_new_possibility_rate = Float64(get(novelty_state, "new_possibility_rate", 0.0))
    recent_innovation_births = Int(get(novelty_state, "innovation_births", 0))
    recent_new_possibilities = Int(get(novelty_state, "new_possibilities", 0))
    scarcity_signal = Float64(get(novelty_state, "recent_scarcity_signal", get(novelty_state, "scarcity_signal", 0.5)))

    novelty_level_global = Float64(get(novelty_state, "novelty_level", 0.5))
    driver_snapshot = get(novelty_state, "drivers", Dict())

    # Get base rates from driver snapshot
    base_combo_rate = Float64(get(driver_snapshot, "combo_rate", 0.0))
    base_reuse_pressure = Float64(get(driver_snapshot, "reuse_pressure", 0.0))
    base_new_poss_rate = Float64(get(driver_snapshot, "new_possibility_rate", recent_new_possibility_rate))
    base_adoption_rate = Float64(get(driver_snapshot, "adoption_rate", 0.0))

    # Tier-specific rates from tier_drivers
    tier_driver_map = get(novelty_state, "tier_drivers", Dict())
    tier_combo_rate = Float64(get(get(tier_driver_map, "combo_rate", Dict()), ai_level, base_combo_rate))
    tier_reuse_pressure = Float64(get(get(tier_driver_map, "reuse_pressure", Dict()), ai_level, base_reuse_pressure))
    tier_new_poss_rate = Float64(get(get(tier_driver_map, "new_possibility_rate", Dict()), ai_level, base_new_poss_rate))
    tier_adoption_rate = Float64(get(get(tier_driver_map, "adoption_rate", Dict()), ai_level, base_adoption_rate))

    # Compute novelty adjustment (matching Python exactly)
    novelty_adjustment = (
        0.25 * (tier_combo_rate - base_combo_rate) -
        0.2 * (tier_reuse_pressure - base_reuse_pressure) +
        0.15 * (tier_new_poss_rate - base_new_poss_rate) +
        0.1 * (tier_adoption_rate - base_adoption_rate)
    )

    novelty_level_agent = clamp(novelty_level_global + novelty_adjustment, 0.05, 0.95)
    novelty_level_agent = clamp(
        novelty_level_agent +
        0.22 * (innovation_trait - 0.5) +
        0.15 * (exploration_trait - 0.5) +
        0.10 * personal_knowledge,
        0.0, 1.0
    )

    # Short-term buffer integration (matching Python)
    short_novelty = _get_short_term_average(env, "agentic_novelty"; fallback=novelty_level_agent, agent_id=agent_id)
    short_weight_novelty = clamp(0.30 - 0.26 * info_quality + 0.12 * tier_factor, 0.03, 0.45)
    novelty_mix = clamp(0.2 + 0.4 * info_quality + 0.25 * info_breadth - 0.15 * tier_factor, 0.1, 0.95)
    novelty_level_agent = clamp(
        (1.0 - short_weight_novelty) * novelty_mix * novelty_level_agent + short_weight_novelty * short_novelty,
        0.0, 1.0
    )

    # Update short-term buffer
    _update_short_term!(env, "agentic_novelty", novelty_level_agent; agent_id=agent_id)

    creative_momentum = novelty_level_agent

    # Get disruption potential from state
    disruption_potential = Dict{String,Float64}(get(novelty_state, "disruption_potential", Dict()))
    if isempty(disruption_potential) && hasfield(typeof(env.config), :SECTORS)
        for sector in env.config.SECTORS
            disruption_potential[sector] = 0.3
        end
    end

    # Innovation signal (matching Python structure)
    innovation_signal = Dict{String,Any}(
        "novelty_potential" => novelty_level_agent,
        "creative_confidence" => creative_momentum,
        "component_scarcity" => scarcity_signal,
        "new_possibility_rate" => recent_new_possibility_rate,
        "innovation_births" => recent_innovation_births,
        "new_possibilities" => recent_new_possibilities,
        "combo_rate" => tier_combo_rate,
        "reuse_pressure" => tier_reuse_pressure,
        "adoption_rate" => tier_adoption_rate
    )
    perception["innovation_signal"] = innovation_signal

    perception["agentic_novelty"] = Dict{String,Any}(
        "novelty_potential" => novelty_level_agent,
        "level" => novelty_level_agent,
        "creative_confidence" => creative_momentum,
        "disruption_potential" => disruption_potential,
        "volatility" => volatility,
        "new_possibility_rate" => recent_new_possibility_rate,
        "innovation_births" => recent_innovation_births,
        "new_possibilities" => recent_new_possibilities,
        "component_scarcity" => scarcity_signal,
        "drivers" => driver_snapshot,
        "combo_rate" => tier_combo_rate,
        "reuse_pressure" => tier_reuse_pressure,
        "adoption_rate" => tier_adoption_rate,
        "tier_combo_rate" => tier_combo_rate,
        "tier_reuse_pressure" => tier_reuse_pressure,
        "tier_new_possibility_rate" => tier_new_poss_rate,
        "tier_adoption_rate" => tier_adoption_rate,
        "global_combo_rate" => base_combo_rate,
        "global_reuse_pressure" => base_reuse_pressure,
        "tier_novelty_adjustment" => novelty_adjustment
    )

    # ========================================================================
    # COMPETITIVE RECURSION PERCEPTION
    # ========================================================================

    # Get herding patterns from AI uncertainty signals
    herding_patterns = get(env.ai_uncertainty_signals, "ai_herding_patterns", Dict())

    # Compute herding pressure from visible opportunities
    herding_pressure_map = Dict{String,Float64}()
    for opp in visible_opportunities
        demand = hasfield(typeof(opp), :competition) ? Float64(opp.competition) : 0.0
        opp_id = hasfield(typeof(opp), :id) ? opp.id : "opp_$(objectid(opp))"
        herding_pressure_map[opp_id] = clamp(demand, 0.0, 1.0)
    end

    pressure_vals = collect(values(herding_pressure_map))
    avg_pressure = !isempty(pressure_vals) ? fast_mean(pressure_vals) : 0.0
    if !isfinite(avg_pressure)
        avg_pressure = 0.0
    end

    # Get confidence miscalibration
    confidence_vals = get(env.ai_uncertainty_signals, "confidence_miscalibration", Float64[])
    confidence_miscalibration = !isempty(confidence_vals) ? safe_mean(confidence_vals) : 0.0
    if !isfinite(confidence_miscalibration)
        confidence_miscalibration = 0.0
    end

    ai_herding_intensity = Float64(get(env.practical_indeterminism_state, "ai_herding_intensity", 0.0))
    capital_crowding = Float64(get(get(market_conditions, "crowding_metrics", Dict()), "crowding_index", 0.0))

    # Raw recursion level (matching Python)
    recursion_raw = (
        0.12 +
        avg_pressure * 0.45 +
        confidence_miscalibration * 0.18 +
        hallucination_rate * 0.12 +
        volatility * 0.08 +
        share_invest * 0.08 +
        ai_herding_intensity * 0.15 +
        max(0.0, capital_crowding) * 0.08
    )
    recursion_raw += (
        0.10 * max(0.0, knowledge_span - 0.35) -
        0.10 * (exploration_trait - 0.5) +
        0.08 * max(0.0, 0.55 - personal_knowledge)
    )
    recursion_level = clamp(recursion_raw, 0.0, 1.0)

    # Update strategic opacity
    hallucination_events = length(get(env.ai_uncertainty_signals, "hallucination_events", []))
    strategic_opacity = clamp(
        Float64(get(env.competitive_recursion_state, "strategic_opacity", 0.0)) * 0.9 +
        hallucination_events * 0.03 +
        avg_pressure * 0.1 +
        volatility * 0.1,
        0.0, 1.0
    )
    env.competitive_recursion_state["strategic_opacity"] = strategic_opacity

    # Short-term buffer integration (matching Python)
    short_recursion = _get_short_term_average(env, "competitive_recursion"; fallback=recursion_level, agent_id=agent_id)
    recursion_mix = clamp(0.25 + 0.45 * info_quality + 0.25 * info_breadth - 0.2 * tier_factor, 0.15, 0.9)
    recursion_short_weight = clamp(0.32 - 0.28 * info_quality + 0.15 * tier_factor, 0.04, 0.5)
    recursion_level = clamp(
        (1.0 - recursion_short_weight) * recursion_mix * recursion_level + recursion_short_weight * short_recursion,
        0.0, 1.0
    )

    # Update short-term buffer
    _update_short_term!(env, "competitive_recursion", recursion_level; agent_id=agent_id)

    ai_usage_share = 1.0 - ai_shares[1]  # 1 - none share

    # Competition signal (matching Python structure)
    ai_tier_shares_from_conditions = get(market_conditions, "ai_tier_shares", Dict{String,Float64}())
    competition_signal = Dict{String,Any}(
        "pressure_level" => recursion_level,
        "herding_awareness" => avg_pressure,
        "herding_pressure" => herding_pressure_map,
        "ai_usage_share" => ai_usage_share,
        "capital_crowding" => capital_crowding,
        "ai_herding_intensity" => ai_herding_intensity
    )
    perception["competition_signal"] = competition_signal

    perception["competitive_recursion"] = Dict{String,Any}(
        "recursion_level" => recursion_level,
        "level" => recursion_level,
        "herding_awareness" => avg_pressure,
        "herding_pressure" => herding_pressure_map,
        "ai_herding_patterns" => Dict{String,Float64}(herding_patterns),
        "ai_herding_intensity" => ai_herding_intensity,
        "strategic_opacity" => strategic_opacity,
        "volatility" => volatility,
        "ai_usage_share" => ai_usage_share
    )

    perception["crowding_metrics"] = get(market_conditions, "crowding_metrics", Dict())
    perception["volatility_metric"] = volatility
    perception["action_profile"] = Dict{String,Float64}(
        "invest" => share_invest,
        "innovate" => share_innovate,
        "explore" => share_explore,
        "maintain" => share_maintain
    )

    # ========================================================================
    # DECISION CONFIDENCE (matching Python exactly)
    # ========================================================================

    # Overall uncertainty from all dimensions
    total_uncertainty = (
        ignorance_level * 0.35 +
        indeterminism_level * 0.25 +
        (1.0 - novelty_level_agent) * 0.20 +
        recursion_level * 0.20
    )

    # FIXED: Increase coefficient to allow info_quality to meaningfully impact decision confidence
    # Previously 0.2 was too weak - Premium AI (info_quality=0.95) only gave 0.19 boost
    # Now Premium gives 0.38 boost vs None (0.45) giving 0.18 - proper differentiation
    ai_confidence_boost = info_quality * 0.4

    # Calculate experience-based multipliers from recent outcomes
    recent_success_rate = 0.5
    ai_success_rate = 0.5
    mean_roi = 0.0
    if !isempty(recent_outcomes)
        invest_outcomes = filter(o -> get(o, "action", "") == "invest", recent_outcomes)
        if !isempty(invest_outcomes)
            recent_success_rate = count(o -> get(o, "success", false), invest_outcomes) / length(invest_outcomes)
            ai_invest = filter(o -> get(o, "ai_used", false), invest_outcomes)
            if !isempty(ai_invest)
                ai_success_rate = count(o -> get(o, "success", false), ai_invest) / length(ai_invest)
            else
                ai_success_rate = recent_success_rate
            end
            rois = Float64[]
            for outcome in invest_outcomes
                invested = Float64(get(outcome, "investment_amount", get(get(outcome, "investment", Dict()), "amount", 0.0)))
                returned = Float64(get(outcome, "capital_returned", 0.0))
                if invested > 0
                    push!(rois, (returned - invested) / invested)
                end
            end
            if !isempty(rois)
                mean_roi = clamp(mean(rois), -0.5, 0.5)
            end
        end
    end

    # Calculate raw confidence (matching Python's formula)
    raw_confidence = competence_trait * exp(-total_uncertainty * 0.5) * (1 + ai_confidence_boost)
    raw_confidence = isfinite(raw_confidence) ? raw_confidence : 0.5

    experience_multiplier = clamp(0.4 + 1.4 * recent_success_rate * competence_trait, 0.2, 1.8)
    ai_effective_trust = clamp(avg_ai_trust, 0.0, 1.0)
    ai_experience_multiplier = clamp(0.4 + 1.4 * ai_success_rate * ai_effective_trust, 0.2, 1.8)
    roi_multiplier = clamp(1 + mean_roi * (0.5 + risk_tolerance), 0.3, 1.7)

    raw_confidence *= experience_multiplier * ai_experience_multiplier * roi_multiplier
    raw_confidence = clamp(raw_confidence, 0.01, 2.0)

    decision_confidence = stable_sigmoid(3.0 * (raw_confidence - 0.5))
    decision_confidence = isfinite(decision_confidence) ? decision_confidence : 0.5

    confidence_multiplier = 1 - 0.4 * hallucination_rate + 0.3 * avg_ai_trust
    decision_confidence *= clamp(confidence_multiplier, 0.5, 1.3)
    decision_confidence = clamp(decision_confidence, 0.02, 0.98)

    perception["decision_confidence"] = decision_confidence

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

    # FIXED: All adjustments now emerge from config parameters (info_quality, info_breadth)
    # Previously recursion_awareness was hardcoded by tier (premium=0.2, advanced=0.12, others=0.0)
    # Now it emerges from info_quality: better info → better understanding of competitive dynamics
    return Dict{String,Float64}(
        "ignorance_reduction" => ai_config.info_quality * 0.3,
        "indeterminism_reduction" => ai_config.info_quality * 0.2,
        "novelty_boost" => ai_config.info_breadth * 0.15,
        "recursion_awareness" => ai_config.info_quality * 0.22
    )
end
