"""
Agent resources and decision logic for GlimpseABM.jl

This module implements the entrepreneurial agent model, including resource
management, decision-making under uncertainty, and learning from outcomes.

Port of: glimpse_abm/agents.py

Note: UncertaintyResponseProfile and PerformanceTracker are defined in models.jl
"""

using Random
using Statistics
using Distributions

# ============================================================================
# AGENT RESOURCES
# ============================================================================

"""
Represents an agent's multidimensional assets including cognitive constraints.
"""
mutable struct AgentResources
    capital::Float64
    knowledge::Dict{String,Float64}
    capabilities::Dict{String,Float64}
    experience_units::Int
    performance::PerformanceTracker
    knowledge_last_used::Dict{String,Int}
end

function AgentResources(capital::Float64)
    return AgentResources(
        capital,
        Dict("tech" => 0.1, "retail" => 0.1, "service" => 0.1, "manufacturing" => 0.1),
        Dict(
            "market_timing" => 0.1,
            "opportunity_evaluation" => 0.1,
            "innovation" => 0.5,
            "uncertainty_management" => 0.1
        ),
        0,
        PerformanceTracker(capital),
        Dict{String,Int}()
    )
end

# ============================================================================
# EMERGENT AGENT
# ============================================================================

"""
Entrepreneurial agent with emergent AI adoption behavior.

The agent learns how to respond to uncertainty, selects AI tools adaptively,
and makes strategic decisions about investing, innovating, exploring, or
maintaining resources.
"""
mutable struct EmergentAgent
    id::Int
    resources::AgentResources
    config::EmergentConfig

    # Agent traits
    traits::Dict{String,Float64}
    uncertainty_tolerance::Float64
    innovativeness::Float64
    competence::Float64
    ai_trust::Float64
    trait_momentum::Float64

    # AI state
    current_ai_level::String
    fixed_ai_level::Union{String,Nothing}
    ai_learning::AILearningProfile
    ai_usage_count::Int
    ai_tier_history::Vector{String}

    # Performance state
    alive::Bool
    survival_rounds::Int
    insolvency_rounds::Int
    total_invested::Float64
    total_returned::Float64
    success_count::Int
    failure_count::Int
    innovation_count::Int

    # Decision state
    uncertainty_response::UncertaintyResponseProfile
    action_history::Vector{String}
    last_action::String
    last_outcome::Dict{String,Any}

    # Portfolio
    active_investments::Vector{Dict{String,Any}}

    # RNG
    rng::AbstractRNG
end

function EmergentAgent(
    id::Int,
    config::EmergentConfig;
    initial_capital::Union{Float64,Nothing} = nothing,
    fixed_ai_level::Union{String,Nothing} = nothing,
    rng::AbstractRNG = Random.default_rng()
)
    # Sample initial capital
    capital = if isnothing(initial_capital)
        rand(rng, Uniform(config.INITIAL_CAPITAL_RANGE...))
    else
        initial_capital
    end

    # Sample traits
    traits = sample_all_traits(config; rng=rng)

    return EmergentAgent(
        id,
        AgentResources(capital),
        config,
        traits,
        get(traits, "uncertainty_tolerance", 0.5),
        get(traits, "innovativeness", 0.5),
        get(traits, "competence", 0.5),
        get(traits, "ai_trust", 0.5),
        get(traits, "trait_momentum", 0.7),
        "none",  # current_ai_level
        fixed_ai_level,
        AILearningProfile(),
        0,  # ai_usage_count
        String[],  # ai_tier_history
        true,  # alive
        0,  # survival_rounds
        0,  # insolvency_rounds
        0.0,  # total_invested
        0.0,  # total_returned
        0,  # success_count
        0,  # failure_count
        0,  # innovation_count
        UncertaintyResponseProfile(rng=rng),
        String[],  # action_history
        "maintain",  # last_action
        Dict{String,Any}(),  # last_outcome
        Dict{String,Any}[],  # active_investments
        rng
    )
end

"""
Get the agent's current capital.
"""
function get_capital(agent::EmergentAgent)::Float64
    return agent.resources.capital
end

"""
Set the agent's capital.
"""
function set_capital!(agent::EmergentAgent, value::Float64)
    agent.resources.capital = max(0.0, value)
end

"""
Check if agent is alive and above survival threshold.
"""
function check_survival!(agent::EmergentAgent, round::Int)::Bool
    if !agent.alive
        return false
    end

    # Use explicit SURVIVAL_THRESHOLD (matches Python behavior)
    survival_threshold = agent.config.SURVIVAL_THRESHOLD

    if agent.resources.capital < survival_threshold
        agent.insolvency_rounds += 1
        if agent.insolvency_rounds >= agent.config.INSOLVENCY_GRACE_ROUNDS
            agent.alive = false
            return false
        end
    else
        agent.insolvency_rounds = 0
        agent.survival_rounds = round
    end

    return true
end

"""
Get the AI level to use (fixed or adaptive).
"""
function get_ai_level(agent::EmergentAgent)::String
    if !isnothing(agent.fixed_ai_level)
        return agent.fixed_ai_level
    end
    return agent.current_ai_level
end

"""
Select an action based on uncertainty perception and agent state.
"""
function select_action(
    agent::EmergentAgent,
    market_conditions::Dict{String,Any},
    uncertainty_state::Dict{String,Any};
    available_opportunities::Vector{Opportunity} = Opportunity[]
)::String
    if !agent.alive
        return "maintain"
    end

    capital = get_capital(agent)
    survival_threshold = agent.config.INITIAL_CAPITAL * agent.config.SURVIVAL_CAPITAL_RATIO

    # Capital pressure
    capital_ratio = capital / agent.config.INITIAL_CAPITAL
    under_pressure = capital < survival_threshold * 1.5

    # Extract uncertainty levels
    actor_ignorance = get(get(uncertainty_state, "actor_ignorance", Dict()), "level", 0.0)
    practical_indet = get(get(uncertainty_state, "practical_indeterminism", Dict()), "level", 0.0)
    agentic_novelty = get(get(uncertainty_state, "agentic_novelty", Dict()), "level", 0.0)
    competitive_rec = get(get(uncertainty_state, "competitive_recursion", Dict()), "level", 0.0)

    composite_uncertainty = (actor_ignorance + practical_indet + agentic_novelty + competitive_rec) / 4

    # Base action scores
    scores = Dict(
        "invest" => 0.0,
        "innovate" => 0.0,
        "explore" => 0.0,
        "maintain" => 0.0
    )

    # Investment score - higher when uncertainty is moderate and capital is good
    if !isempty(available_opportunities)
        invest_factor = get_response_factor(
            agent.uncertainty_response, "practical_indeterminism", practical_indet;
            rng=agent.rng
        )
        scores["invest"] = 0.4 * invest_factor * capital_ratio * (1.0 - competitive_rec * 0.5)
    end

    # Innovation score - higher when agentic novelty potential is high
    innovate_factor = get_response_factor(
        agent.uncertainty_response, "agentic_novelty", agentic_novelty;
        rng=agent.rng
    )
    scores["innovate"] = 0.3 * agent.innovativeness * innovate_factor * (1.0 + agentic_novelty * 0.3)

    # Exploration score - higher when ignorance is high
    explore_factor = get_response_factor(
        agent.uncertainty_response, "actor_ignorance", actor_ignorance;
        rng=agent.rng
    )
    scores["explore"] = 0.25 * explore_factor * (1.0 + actor_ignorance * 0.5)

    # Maintain score - higher under high uncertainty or capital pressure
    maintain_base = 0.2
    if under_pressure
        maintain_base += 0.3
    end
    if composite_uncertainty > 0.6
        maintain_base += 0.2
    end
    scores["maintain"] = maintain_base

    # Apply temperature and select
    temperature = agent.config.ACTION_SELECTION_TEMPERATURE
    noise = agent.config.ACTION_SELECTION_NOISE

    # Add noise and apply softmax-like selection
    for action in keys(scores)
        scores[action] += noise * randn(agent.rng)
        scores[action] = max(0.0, scores[action])
    end

    # Normalize to probabilities
    total = sum(values(scores))
    if total > 0
        for action in keys(scores)
            scores[action] /= total
        end
    else
        scores["maintain"] = 1.0
    end

    # Sample action
    actions = collect(keys(scores))
    weights = [scores[a] for a in actions]
    return weighted_choice(actions, weights; rng=agent.rng)
end

"""
Execute an action and return the outcome.
"""
function execute_action!(
    agent::EmergentAgent,
    action::String,
    market::MarketEnvironment,
    round::Int;
    opportunity::Union{Opportunity,Nothing} = nothing
)::Dict{String,Any}
    outcome = Dict{String,Any}(
        "action" => action,
        "agent_id" => agent.id,
        "round" => round,
        "ai_level_used" => get_ai_level(agent),
        "capital_before" => get_capital(agent)
    )

    if action == "invest" && !isnothing(opportunity)
        outcome = _execute_invest!(agent, opportunity, market, round, outcome)
    elseif action == "innovate"
        outcome = _execute_innovate!(agent, market, round, outcome)
    elseif action == "explore"
        outcome = _execute_explore!(agent, market, round, outcome)
    else  # maintain
        outcome = _execute_maintain!(agent, round, outcome)
    end

    # Record action
    push!(agent.action_history, action)
    agent.last_action = action
    agent.last_outcome = outcome

    outcome["capital_after"] = get_capital(agent)
    return outcome
end

function _execute_invest!(
    agent::EmergentAgent,
    opportunity::Opportunity,
    market::MarketEnvironment,
    round::Int,
    outcome::Dict{String,Any}
)::Dict{String,Any}
    capital = get_capital(agent)
    max_invest = capital * agent.config.MAX_INVESTMENT_FRACTION
    invest_amount = min(max_invest, opportunity.capital_requirements)

    if invest_amount <= 0 || invest_amount > capital
        outcome["success"] = false
        outcome["reason"] = "insufficient_capital"
        return outcome
    end

    # Deduct capital
    set_capital!(agent, capital - invest_amount)
    agent.total_invested += invest_amount

    # Record investment
    investment = Dict{String,Any}(
        "opportunity_id" => opportunity.id,
        "opportunity" => opportunity,
        "amount" => invest_amount,
        "round_invested" => round,
        "maturity_round" => round + opportunity.time_to_maturity,
        "ai_level" => get_ai_level(agent)
    )
    push!(agent.active_investments, investment)

    # Track performance
    record_deployment!(agent.resources.performance, "invest", invest_amount;
                       ai_level=get_ai_level(agent), round_num=round)

    outcome["success"] = true
    outcome["amount"] = invest_amount
    outcome["opportunity_id"] = opportunity.id
    outcome["chosen_opportunity_obj"] = opportunity
    outcome["maturity_round"] = investment["maturity_round"]

    return outcome
end

function _execute_innovate!(
    agent::EmergentAgent,
    market::MarketEnvironment,
    round::Int,
    outcome::Dict{String,Any}
)::Dict{String,Any}
    capital = get_capital(agent)

    # Innovation cost
    rd_spend = min(
        capital * agent.config.INNOVATION_BASE_SPEND_RATIO,
        agent.config.INNOVATION_MAX_SPEND
    )

    if rd_spend > capital * 0.5
        rd_spend = capital * 0.2  # Cap R&D if capital constrained
    end

    set_capital!(agent, capital - rd_spend)

    # Innovation success probability
    base_prob = agent.config.INNOVATION_PROBABILITY
    competence_factor = agent.competence
    innovativeness_factor = agent.innovativeness

    success_prob = base_prob * (0.5 + 0.5 * competence_factor) * (0.7 + 0.3 * innovativeness_factor)
    success = rand(agent.rng) < success_prob

    if success
        # Create innovation value
        base_return = rd_spend * agent.config.INNOVATION_SUCCESS_BASE_RETURN
        multiplier = rand(agent.rng, Uniform(agent.config.INNOVATION_SUCCESS_RETURN_MULTIPLIER...))
        innovation_return = base_return * multiplier

        set_capital!(agent, get_capital(agent) + innovation_return)
        agent.innovation_count += 1
        agent.success_count += 1

        outcome["success"] = true
        outcome["rd_spend"] = rd_spend
        outcome["innovation_return"] = innovation_return
        outcome["is_new_combination"] = rand(agent.rng) < 0.3
    else
        # Partial recovery on failure
        recovery = rd_spend * agent.config.INNOVATION_FAIL_RECOVERY_RATIO
        set_capital!(agent, get_capital(agent) + recovery)

        outcome["success"] = false
        outcome["rd_spend"] = rd_spend
        outcome["recovery"] = recovery
    end

    record_deployment!(agent.resources.performance, "innovate", rd_spend;
                       ai_level=get_ai_level(agent), round_num=round)

    return outcome
end

function _execute_explore!(
    agent::EmergentAgent,
    market::MarketEnvironment,
    round::Int,
    outcome::Dict{String,Any}
)::Dict{String,Any}
    # Exploration cost (minimal)
    explore_cost = get_capital(agent) * 0.01
    set_capital!(agent, get_capital(agent) - explore_cost)

    # Knowledge gain
    discovery_prob = agent.config.DISCOVERY_PROBABILITY
    discovered = rand(agent.rng) < discovery_prob

    if discovered
        # Increase knowledge in a random sector
        sector = rand(agent.rng, agent.config.SECTORS)
        current_knowledge = get(agent.resources.knowledge, sector, 0.1)
        knowledge_gain = rand(agent.rng, Uniform(0.05, 0.15))
        agent.resources.knowledge[sector] = min(1.0, current_knowledge + knowledge_gain)
        agent.resources.knowledge_last_used[sector] = round

        outcome["success"] = true
        outcome["discovered_sector"] = sector
        outcome["knowledge_gain"] = knowledge_gain
    else
        outcome["success"] = false
    end

    outcome["explore_cost"] = explore_cost
    record_deployment!(agent.resources.performance, "explore", explore_cost;
                       ai_level=get_ai_level(agent), round_num=round)

    return outcome
end

function _execute_maintain!(
    agent::EmergentAgent,
    round::Int,
    outcome::Dict{String,Any}
)::Dict{String,Any}
    # Maintenance action - preserve capital
    outcome["success"] = true
    outcome["action_type"] = "preserve"
    return outcome
end

"""
Process matured investments and return outcomes.
"""
function process_matured_investments!(
    agent::EmergentAgent,
    market::MarketEnvironment,
    round::Int
)::Vector{Dict{String,Any}}
    if !agent.alive
        return Dict{String,Any}[]
    end

    matured_outcomes = Dict{String,Any}[]
    remaining_investments = Dict{String,Any}[]

    market_conditions = get_market_conditions(market)

    for investment in agent.active_investments
        if investment["maturity_round"] <= round
            # Investment matures
            opp = investment["opportunity"]
            invested_amount = Float64(investment["amount"])

            # Calculate realized return
            ret_multiple = realized_return(opp, market_conditions, get_ai_level(agent); rng=agent.rng)
            capital_returned = invested_amount * ret_multiple

            # Apply to agent capital
            set_capital!(agent, get_capital(agent) + capital_returned)
            agent.total_returned += capital_returned

            success = ret_multiple >= 1.0
            if success
                agent.success_count += 1
            else
                agent.failure_count += 1
            end

            # Record return
            record_return!(agent.resources.performance, "invest", capital_returned;
                          ai_level=investment["ai_level"], round_num=round)

            push!(matured_outcomes, Dict{String,Any}(
                "investment" => investment,
                "investment_amount" => invested_amount,
                "capital_returned" => capital_returned,
                "return_multiple" => ret_multiple,
                "success" => success,
                "round_matured" => round
            ))
        else
            push!(remaining_investments, investment)
        end
    end

    agent.active_investments = remaining_investments
    return matured_outcomes
end

"""
Apply operational costs for the round.
"""
function apply_operational_costs!(agent::EmergentAgent, round::Int)
    if !agent.alive
        return
    end

    cost = agent.config.BASE_OPERATIONAL_COST
    current = get_capital(agent)
    set_capital!(agent, current - cost)
end

"""
Get a snapshot of the agent's current state.
"""
function snapshot(agent::EmergentAgent, round::Int)::Dict{String,Any}
    return Dict{String,Any}(
        "id" => agent.id,
        "round" => round,
        "capital" => get_capital(agent),
        "alive" => agent.alive,
        "ai_level" => get_ai_level(agent),
        "survival_rounds" => agent.survival_rounds,
        "success_count" => agent.success_count,
        "failure_count" => agent.failure_count,
        "innovation_count" => agent.innovation_count,
        "total_invested" => agent.total_invested,
        "total_returned" => agent.total_returned,
        "active_investments" => length(agent.active_investments),
        "uncertainty_tolerance" => agent.uncertainty_tolerance,
        "innovativeness" => agent.innovativeness,
        "competence" => agent.competence,
        "ai_trust" => agent.ai_trust
    )
end

# ============================================================================
# VECTORIZED AGENT STATE (for batch operations)
# ============================================================================

"""
Vectorized agent state for efficient batch operations.
"""
mutable struct VectorizedAgentState
    n_agents::Int
    capitals::Vector{Float64}
    alive::BitVector
    ai_levels::Vector{String}
    survival_rounds::Vector{Int}
    success_counts::Vector{Int}
    failure_counts::Vector{Int}
    innovation_counts::Vector{Int}
end

function VectorizedAgentState(agents::Vector{EmergentAgent})
    n = length(agents)
    return VectorizedAgentState(
        n,
        [get_capital(a) for a in agents],
        BitVector([a.alive for a in agents]),
        [get_ai_level(a) for a in agents],
        [a.survival_rounds for a in agents],
        [a.success_count for a in agents],
        [a.failure_count for a in agents],
        [a.innovation_count for a in agents]
    )
end

"""
Sync vectorized state back to individual agents.
"""
function sync_to_agents!(state::VectorizedAgentState, agents::Vector{EmergentAgent})
    for i in 1:min(state.n_agents, length(agents))
        set_capital!(agents[i], state.capitals[i])
        agents[i].alive = state.alive[i]
        agents[i].survival_rounds = state.survival_rounds[i]
        agents[i].success_count = state.success_counts[i]
        agents[i].failure_count = state.failure_counts[i]
        agents[i].innovation_count = state.innovation_counts[i]
    end
end

"""
Update vectorized state from individual agents.
"""
function sync_from_agents!(state::VectorizedAgentState, agents::Vector{EmergentAgent})
    for i in 1:min(state.n_agents, length(agents))
        state.capitals[i] = get_capital(agents[i])
        state.alive[i] = agents[i].alive
        state.ai_levels[i] = get_ai_level(agents[i])
        state.survival_rounds[i] = agents[i].survival_rounds
        state.success_counts[i] = agents[i].success_count
        state.failure_counts[i] = agents[i].failure_count
        state.innovation_counts[i] = agents[i].innovation_count
    end
end

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Note: stable_sigmoid is defined in innovation.jl

"""
Fast mean for collections, returns 0.0 for empty.
"""
function fast_mean(vals)::Float64
    if isempty(vals)
        return 0.0
    end
    s = 0.0
    n = 0
    for v in vals
        if !ismissing(v) && isfinite(v)
            s += v
            n += 1
        end
    end
    return n > 0 ? s / n : 0.0
end

# ============================================================================
# AI PERFORMANCE METRICS
# ============================================================================

"""
Compute AI performance metrics for AI level selection.
"""
function compute_ai_performance_metrics(agent::EmergentAgent)::Dict{String,Any}
    perf = agent.resources.performance
    metrics = Dict{String,Any}()

    # Basic ROI metrics
    overall_roic = compute_roic(perf)
    invest_roic = compute_roic(perf; action="invest")
    innovate_roic = compute_roic(perf; action="innovate")
    explore_roic = compute_roic(perf; action="explore")

    metrics["overall_roic"] = overall_roic
    metrics["invest_roic"] = invest_roic
    metrics["innovate_roic"] = innovate_roic
    metrics["explore_roic"] = explore_roic

    # Cash flow metrics
    deployed_total = get(perf.deployed_by_action, "overall", 0.0)
    returned_total = get(perf.returned_by_action, "overall", 0.0)
    metrics["overall_cash_flow_total"] = returned_total - deployed_total

    # Recent activity (approximate)
    n_actions = length(agent.action_history)
    recent_window = min(20, n_actions)
    metrics["recent_ai_activity"] = max(1.0, Float64(recent_window) / 5.0)

    # ROI gain ratio
    if deployed_total > 0
        metrics["roi_gain"] = (returned_total - deployed_total) / deployed_total
        metrics["roi_gain_ratio"] = metrics["roi_gain"]
    else
        metrics["roi_gain"] = 0.0
        metrics["roi_gain_ratio"] = 0.0
    end

    metrics["recent_roi_gain"] = metrics["roi_gain"]
    metrics["recent_roi_gain_ratio"] = metrics["roi_gain_ratio"]
    metrics["recent_cash_flow_total"] = metrics["overall_cash_flow_total"]

    return metrics
end

# ============================================================================
# AI LEVEL SELECTION
# ============================================================================

"""
Estimate AI cost for a given level and expected calls.
"""
function estimate_ai_cost(
    agent::EmergentAgent,
    ai_level::String;
    expected_calls::Float64 = 1.0
)::Float64
    if ai_level == "none"
        return 0.0
    end

    ai_config = get(agent.config.AI_LEVELS, ai_level, agent.config.AI_LEVELS["none"])
    cost_type = get(ai_config, "cost_type", "none")

    # Scale costs by AI_COST_INTENSITY (for robustness testing)
    cost_intensity = agent.config.AI_COST_INTENSITY

    if cost_type == "subscription"
        base_cost = Float64(get(ai_config, "cost", 0.0)) * cost_intensity
        per_use = Float64(get(ai_config, "per_use_cost", 0.0)) * cost_intensity
        # Amortize subscription over rounds
        amort_rounds = max(1, get(agent.config.AI_SUBSCRIPTION_AMORTIZATION_ROUNDS, 20))
        return base_cost / amort_rounds + per_use * max(expected_calls, 0.0)
    elseif cost_type == "per_use"
        return Float64(get(ai_config, "cost", 0.0)) * cost_intensity * max(expected_calls, 0.0)
    end

    return 0.0
end

"""
Compute penalty for switching AI levels.
"""
function compute_ai_switch_penalty(
    agent::EmergentAgent,
    current_level::String,
    proposed_level::String
)::Float64
    if proposed_level == current_level
        return 0.0
    end

    order = ["none", "basic", "advanced", "premium"]
    current_idx = findfirst(==(current_level), order)
    proposed_idx = findfirst(==(proposed_level), order)

    if isnothing(current_idx) || isnothing(proposed_idx)
        return 0.1
    end

    distance = abs(proposed_idx - current_idx)
    base_penalty = 0.05 * distance + 0.03

    # Experience with proposed level reduces penalty
    experience = count(==(proposed_level), agent.ai_tier_history)
    experience_modifier = 1.0 / (1.0 + 0.15 * experience)

    # Fatigue from frequent switching
    n_switches = 0
    for i in 2:length(agent.ai_tier_history)
        if agent.ai_tier_history[i] != agent.ai_tier_history[i-1]
            n_switches += 1
        end
    end
    fatigue_penalty = 0.02 * max(0, n_switches - 3)

    learning_relief = clamp(log1p(experience) * 0.015, 0.0, 0.1)

    return max(0.0, (base_penalty * experience_modifier) + fatigue_penalty - learning_relief)
end

"""
Choose AI level based on Bayesian beliefs and performance metrics.
"""
function choose_ai_level(
    agent::EmergentAgent;
    neighbor_signals::Dict{String,Any} = Dict{String,Any}()
)::String
    # If fixed AI level, return it
    if !isnothing(agent.fixed_ai_level)
        return agent.fixed_ai_level
    end

    base_trust = clamp(get(agent.traits, "ai_trust", 0.5), 0.0, 1.0)
    uncertainty_tolerance = clamp(get(agent.traits, "uncertainty_tolerance", 0.5), 0.0, 1.0)
    avoidance = 1.0 - uncertainty_tolerance

    metrics = compute_ai_performance_metrics(agent)
    order = ["none", "basic", "advanced", "premium"]

    # Prior beliefs for each tier
    prior_map = Dict(
        "none" => (2.6, 2.4),
        "basic" => (2.4, 2.6),
        "advanced" => (1.7, 3.1),
        "premium" => (1.2, 3.4)
    )

    # Compute posterior means from AI learning profile
    posterior_means = Dict{String,Float64}()
    for tier in order
        beliefs = agent.ai_learning.tier_beliefs
        if haskey(beliefs, tier)
            belief = beliefs[tier]
            total = belief["alpha"] + belief["beta"]
            posterior_means[tier] = total > 0 ? belief["alpha"] / total : 0.5
        else
            default_alpha, default_beta = prior_map[tier]
            posterior_means[tier] = default_alpha / (default_alpha + default_beta)
        end
    end

    # Compute cost ratios
    cash_buffer = max(get_capital(agent), agent.resources.performance.initial_equity, 1.0)
    operating_cost = agent.config.BASE_OPERATIONAL_COST
    recent_activity = max(1.0, Float64(get(metrics, "recent_ai_activity", 1.0)))
    amort_horizon = max(1, get(agent.config.AI_SUBSCRIPTION_AMORTIZATION_ROUNDS, 20))
    ref_scale = max(operating_cost * 4.0, cash_buffer * 0.12, 1.0)

    # Scale costs by AI_COST_INTENSITY (for robustness testing)
    cost_intensity = agent.config.AI_COST_INTENSITY

    cost_ratios = Dict{String,Float64}()
    for tier in order
        if tier == "none"
            cost_ratios[tier] = 0.0
            continue
        end
        cfg = get(agent.config.AI_LEVELS, tier, Dict())
        cost_type = get(cfg, "cost_type", "none")
        base_cost = Float64(get(cfg, "cost", 0.0)) * cost_intensity
        per_use_cost = Float64(get(cfg, "per_use_cost", 0.0)) * cost_intensity

        total_cost = if cost_type == "subscription"
            per_round = base_cost / amort_horizon
            per_round + per_use_cost * recent_activity
        elseif cost_type == "per_use"
            per_call = base_cost > 0 ? base_cost : per_use_cost
            per_call * recent_activity
        else
            per_use_cost * recent_activity
        end

        cost_ratios[tier] = total_cost / ref_scale
    end

    current_level = agent.current_ai_level
    if !(current_level in order)
        current_level = "none"
    end

    # Extract performance signals
    roi_gain = Float64(get(metrics, "roi_gain", 0.0))
    roi_gain_ratio = Float64(get(metrics, "roi_gain_ratio", 0.0))
    net_cash_total = Float64(get(metrics, "overall_cash_flow_total", 0.0))
    initial_equity = max(agent.resources.performance.initial_equity, 1.0)
    capital_health = (get_capital(agent) / initial_equity) - 1.0

    # Peer signals
    peer_roi_signal = Float64(get(neighbor_signals, "peer_roi_gap", 0.0))
    adoption_pressure = Float64(get(neighbor_signals, "ai_adoption_pressure", 0.0))
    peer_distribution = get(neighbor_signals, "ai_distribution", Dict{String,Float64}())

    reserve_haircut = Dict("basic" => 0.0, "advanced" => 0.02, "premium" => 0.05)

    # Score each tier
    scores = Dict{String,Float64}()
    for tier in order
        posterior = posterior_means[tier]
        trust_term = posterior * base_trust

        roi_signal = 0.3 * roi_gain + 0.45 * roi_gain_ratio
        roi_term = posterior * roi_signal

        cash_term = 0.2 * clamp(net_cash_total / initial_equity, -1.5, 1.5)

        peer_term = 0.08 * peer_roi_signal + 0.12 * adoption_pressure +
                   0.05 * Float64(get(peer_distribution, tier, 0.0))

        # Experience with tier
        tier_experience = count(==(tier), agent.ai_tier_history)
        learning_relief = clamp(log1p(tier_experience) * 0.02, 0.0, 0.12)

        cost_term = 0.25 * get(cost_ratios, tier, 0.0) * (1.0 - learning_relief)
        if tier == "premium"
            cost_term *= 0.85
        end

        switch_penalty = max(0.0, compute_ai_switch_penalty(agent, current_level, tier) - learning_relief)
        reserve_penalty = get(reserve_haircut, tier, 0.0) * max(0.0, 1.0 - capital_health)

        # Gumbel noise for stochastic selection
        noise = -log(-log(rand(agent.rng))) * 0.02  # Gumbel(0,1) * 0.02

        total_score = (
            trust_term +
            roi_term +
            cash_term +
            0.25 * capital_health +
            peer_term -
            cost_term -
            reserve_penalty -
            switch_penalty -
            0.05 * avoidance +
            noise
        )

        scores[tier] = total_score
    end

    # Select tier with highest score
    best_tier = "none"
    best_score = -Inf
    for (i, tier) in enumerate(order)
        score = scores[tier]
        # Tie-break by order (prefer lower tiers)
        if score > best_score || (score == best_score && i < findfirst(==(best_tier), order))
            best_score = score
            best_tier = tier
        end
    end

    # Update tracking
    push!(agent.ai_tier_history, best_tier)
    agent.current_ai_level = best_tier

    return best_tier
end

# ============================================================================
# UTILITY CALCULATIONS
# ============================================================================

"""
Calculate investment utility given opportunities and perception.
"""
function calculate_investment_utility(
    agent::EmergentAgent,
    opportunities::Vector{Opportunity},
    market_conditions::Dict{String,Any},
    perception::Dict{String,Any};
    ai_level::String = "none"
)::Float64
    if isempty(opportunities)
        return 0.0
    end

    # Get best opportunity score
    max_score = 0.0
    for opp in opportunities
        score = evaluate_opportunity_basic(agent, opp, market_conditions)
        max_score = max(max_score, score)
    end

    scaled_score = stable_sigmoid(max_score - 1.0)

    # Extract uncertainty levels from perception
    actor_unc = Float64(get(get(perception, "actor_ignorance", Dict()), "level", 0.5))
    practical_unc = Float64(get(get(perception, "practical_indeterminism", Dict()), "level", 0.5))
    agentic_unc = Float64(get(get(perception, "agentic_novelty", Dict()), "level", 0.5))
    recursive_unc = Float64(get(get(perception, "competitive_recursion", Dict()), "level", 0.5))
    decision_conf = clamp(Float64(get(perception, "decision_confidence", 0.5)), 0.1, 0.95)

    # Capital metrics
    capital_ratio = get_capital(agent) / max(agent.resources.performance.initial_equity, 1.0)
    liquidity_boost = clamp(capital_ratio - 0.8, 0.0, 1.5)
    opportunity_boost = clamp(length(opportunities) / 6.0, 0.0, 1.0)

    # Ignorance adjustment
    ignorance_adjustment = 0.5 + 0.5 / (1.0 + exp(clamp(actor_unc - 1.5, -20.0, 20.0)))

    value = scaled_score * ignorance_adjustment * decision_conf

    # Performance adjustments
    perf = agent.resources.performance
    invest_roic = compute_roic(perf; action="invest")

    # Locked capital
    locked = sum(inv["amount"] for inv in agent.active_investments; init=0.0)
    idle_capital = max(0.0, get_capital(agent) - locked)
    idle_ratio = idle_capital / max(agent.resources.performance.initial_equity, 1.0)

    value *= 0.7 + 0.3 * liquidity_boost
    value += 0.2 * opportunity_boost

    roic_multiplier = clamp(1.0 + 0.35 * invest_roic, 0.4, 1.4)
    value *= roic_multiplier

    value += max(0.0, idle_ratio) * 0.25

    # Portfolio size penalty
    if length(agent.active_investments) >= 4
        value -= 0.07 * (length(agent.active_investments) - 3)
    end

    # Uncertainty hooks
    value += 0.15 * agentic_unc
    value -= 0.06 * recursive_unc

    # Risk tolerance
    risk_tolerance = Float64(get(agent.traits, "uncertainty_tolerance", 0.5))
    value *= clamp(0.85 + 0.3 * risk_tolerance, 0.5, 1.4)
    value -= (1.0 - risk_tolerance) * 0.12

    # Recent action penalty
    recent_invests = count(==("invest"), agent.action_history[max(1, end-9):end])
    value -= 0.1 * recent_invests / 10.0

    return clamp(value, 0.0, 1.0)
end

"""
Calculate innovation utility.
"""
function calculate_innovation_utility(
    agent::EmergentAgent,
    market_conditions::Dict{String,Any},
    perception::Dict{String,Any}
)::Float64
    base_drive = get(agent.traits, "innovativeness", 0.5) * 0.5 + 0.05

    # Extract signals
    practical_unc = Float64(get(get(perception, "practical_indeterminism", Dict()), "level", 0.5))
    agentic_unc = Float64(get(get(perception, "agentic_novelty", Dict()), "level", 0.5))

    indeterminism_bonus = practical_unc * 0.4
    innovation_opportunity = agentic_unc * 0.25

    # Capital constraints
    capital_ratio = get_capital(agent) / max(agent.resources.performance.initial_equity, 1.0)
    liquidity_penalty = clamp(1.0 - capital_ratio, 0.0, 1.5)

    # R&D burden
    rd_deployed = get(agent.resources.performance.deployed_by_action, "innovate", 0.0)
    rd_burden = clamp(rd_deployed / max(agent.resources.performance.initial_equity, 1.0), 0.0, 2.0)

    innovate_roic = compute_roic(agent.resources.performance; action="innovate")
    loss_ratio = max(0.0, -innovate_roic)

    risk_tolerance = Float64(get(agent.traits, "uncertainty_tolerance", 0.5))
    avoidance = 1.0 - risk_tolerance

    raw_score = (
        base_drive +
        indeterminism_bonus +
        innovation_opportunity -
        0.25 * liquidity_penalty -
        0.3 * rd_burden -
        0.4 * loss_ratio +
        0.2 * (risk_tolerance - 0.5) -
        0.15 * avoidance
    )

    return clamp(stable_sigmoid(raw_score - 0.6), 0.02, 0.98)
end

"""
Calculate exploration utility.
"""
function calculate_exploration_utility(
    agent::EmergentAgent,
    perception::Dict{String,Any};
    ai_level::String = "none"
)::Float64
    base_tendency = get(agent.traits, "exploration_tendency", 0.3)

    # Extract signals
    actor_unc = Float64(get(get(perception, "actor_ignorance", Dict()), "level", 0.5))
    agentic_unc = Float64(get(get(perception, "agentic_novelty", Dict()), "level", 0.5))
    recursive_unc = Float64(get(get(perception, "competitive_recursion", Dict()), "level", 0.5))

    ignorance_drive = actor_unc * 0.35

    # Knowledge deficit
    avg_knowledge = fast_mean(values(agent.resources.knowledge))
    knowledge_deficit = clamp(1.0 - avg_knowledge, 0.0, 1.0)

    novelty_gap = max(0.0, 0.5 - agentic_unc)

    # Capital slack
    locked = sum(inv["amount"] for inv in agent.active_investments; init=0.0)
    capital_slack = max(0.0, (get_capital(agent) - locked) / max(agent.resources.performance.initial_equity, 1.0))

    rich_knowledge_penalty = clamp(avg_knowledge - 0.25, 0.0, 1.0)

    ai_trust_level = Float64(get(agent.traits, "ai_trust", 0.5))
    trust_penalty = max(0.0, ai_trust_level - 0.5) * 0.45

    recursion_penalty = max(0.0, recursive_unc - 0.5) * 0.2
    if ai_level in ["premium", "advanced"]
        recursion_penalty += 0.12 * max(0.0, recursive_unc - 0.5)
    end

    explore_roic = compute_roic(agent.resources.performance; action="explore")
    momentum_bonus = clamp(explore_roic, -0.5, 0.5) * 0.2

    risk_tolerance = Float64(get(agent.traits, "uncertainty_tolerance", 0.5))
    avoidance = 1.0 - risk_tolerance

    # Recent explore penalty
    recent_explores = count(==("explore"), agent.action_history[max(1, end-4):end])
    recent_explore_penalty = recent_explores > 2 ? 0.25 : 0.0

    raw_score = (
        base_tendency * 0.35 +
        ignorance_drive +
        knowledge_deficit * 0.25 +
        0.2 * novelty_gap +
        0.08 * capital_slack +
        momentum_bonus -
        0.35 * rich_knowledge_penalty -
        trust_penalty -
        recursion_penalty +
        0.18 * avoidance -
        0.08 * risk_tolerance +
        0.18 * recursive_unc -
        recent_explore_penalty
    )

    return clamp(stable_sigmoid(raw_score - 0.45), 0.05, 0.95)
end

"""
Calculate maintain utility.
"""
function calculate_maintain_utility(
    agent::EmergentAgent,
    market_conditions::Dict{String,Any},
    perception::Dict{String,Any};
    estimated_cost::Float64 = 0.0
)::Float64
    cost = estimated_cost > 0 ? estimated_cost : agent.config.BASE_OPERATIONAL_COST
    capital_buffer_in_rounds = get_capital(agent) / (cost + 1e-9)
    buffer_pressure = stable_sigmoid(2.0 - capital_buffer_in_rounds)

    # Diversification (use active investments as proxy)
    n_sectors = length(unique(get(inv, "opportunity", Opportunity()).sector for inv in agent.active_investments))
    diversification = min(1.0, n_sectors / 4.0)

    practical_unc = Float64(get(get(perception, "practical_indeterminism", Dict()), "level", 0.5))
    uncertainty_penalty = practical_unc * 0.3

    idle_penalty = min(0.2, (get_capital(agent) / max(agent.config.INITIAL_CAPITAL, 1.0)) * 0.1)

    avg_knowledge = fast_mean(values(agent.resources.knowledge))
    knowledge_penalty = clamp(avg_knowledge - 0.25, 0.0, 1.0) * 0.2
    surplus_buffer_penalty = clamp(capital_buffer_in_rounds - 3.0, 0.0, 3.0) * 0.1

    maintain_utility = 0.2 + buffer_pressure * 0.5 + diversification * 0.3 - uncertainty_penalty - idle_penalty
    maintain_utility -= knowledge_penalty + surplus_buffer_penalty

    risk_tolerance = Float64(get(agent.traits, "uncertainty_tolerance", 0.5))
    avoidance = 1.0 - risk_tolerance
    maintain_utility += 0.2 * avoidance - 0.1 * risk_tolerance

    return clamp(maintain_utility, 0.05, 0.95)
end

# ============================================================================
# OPPORTUNITY EVALUATION
# ============================================================================

"""
Basic opportunity evaluation without full information system.
"""
function evaluate_opportunity_basic(
    agent::EmergentAgent,
    opportunity::Opportunity,
    market_conditions::Dict{String,Any}
)::Float64
    # Expected profit margin
    expected_margin = opportunity.expected_return - 1.0
    uncertainty_adjusted = expected_margin * (1.0 - opportunity.uncertainty * 0.5)

    score = uncertainty_adjusted * (1.0 + get(agent.traits, "uncertainty_tolerance", 0.5))

    # Regime multiplier
    regime = get(market_conditions, "regime", "normal")
    regime_mult = Dict("boom" => 1.2, "growth" => 1.1, "normal" => 1.0, "volatile" => 0.9, "crisis" => 0.7)
    score *= get(regime_mult, regime, 1.0)

    # Sector knowledge
    sector_knowledge = get(agent.resources.knowledge, opportunity.sector, 0.1)
    score *= 1.0 + sector_knowledge * 0.5

    return max(0.1, score)
end

"""
Apply uncertainty adjustments to opportunity score.
"""
function apply_uncertainty_adjustments(
    agent::EmergentAgent,
    base_score::Float64,
    opportunity::Opportunity,
    perception::Dict{String,Any}
)::Float64
    adjusted = base_score

    # Competition adjustment
    if opportunity.competition > 0.5
        tolerance = get(agent.traits, "uncertainty_tolerance", 0.5)
        adjusted *= 1.0 - (tolerance * opportunity.competition * 0.5)
    end

    # Social proof (herding tendency)
    analytical = get(agent.traits, "analytical_ability", 0.5)
    social_sensitivity = 1.0 - analytical
    social_bonus = 1.0 + opportunity.competition * social_sensitivity * 0.25
    adjusted *= social_bonus

    # Creator bonus
    if opportunity.creator_id != -1
        adjusted *= 1.1
        if opportunity.creator_id == agent.id
            adjusted *= 1.2
        end
    end

    return max(0.01, adjusted)
end

"""
Evaluate portfolio opportunities and return ranked list.
"""
function evaluate_portfolio_opportunities(
    agent::EmergentAgent,
    opportunities::Vector{Opportunity},
    market_conditions::Dict{String,Any},
    perception::Dict{String,Any};
    ai_level::String = "none"
)::Vector{Dict{String,Any}}
    if isempty(opportunities)
        return Dict{String,Any}[]
    end

    evaluations = Dict{String,Any}[]

    # Filter opportunities based on AI level
    opp_pool = copy(opportunities)
    if length(opp_pool) > 1
        if ai_level in ["premium", "advanced"]
            # Sort by quality, take top portion
            sort!(opp_pool, by=o -> o.latent_return_potential, rev=true)
            cutoff = max(1, Int(floor(length(opp_pool) * (ai_level == "premium" ? 0.65 : 0.8))))
            opp_pool = opp_pool[1:cutoff]
        else
            # Random shuffle, discard some
            shuffle!(agent.rng, opp_pool)
            discard = ai_level == "basic" ? 0.25 : 0.4
            keep = max(1, Int(floor(length(opp_pool) * (1.0 - discard))))
            opp_pool = opp_pool[1:keep]
        end
    end

    for opp in opp_pool
        base_score = evaluate_opportunity_basic(agent, opp, market_conditions)
        final_score = apply_uncertainty_adjustments(agent, base_score, opp, perception)

        push!(evaluations, Dict{String,Any}(
            "opportunity" => opp,
            "final_score" => final_score,
            "ai_level_used" => ai_level
        ))
    end

    # Sort by score descending
    sort!(evaluations, by=e -> e["final_score"], rev=true)

    return evaluations
end

# ============================================================================
# DECISION METHODS
# ============================================================================

"""
Make a complete decision including AI level selection and action choice.
"""
function make_decision!(
    agent::EmergentAgent,
    opportunities::Vector{Opportunity},
    market_conditions::Dict{String,Any},
    market::MarketEnvironment,
    round_num::Int;
    uncertainty_env::Union{KnightianUncertaintyEnvironment,Nothing} = nothing,
    neighbor_agents::Vector{EmergentAgent} = EmergentAgent[],
    ai_level_override::Union{String,Nothing} = nothing
)::Dict{String,Any}
    if !agent.alive
        return Dict{String,Any}(
            "action" => "maintain",
            "agent_id" => agent.id,
            "round" => round_num,
            "success" => false,
            "reason" => "agent_inactive"
        )
    end

    # Choose AI level
    ai_level = if !isnothing(ai_level_override)
        ai_level_override
    elseif !isnothing(agent.fixed_ai_level)
        agent.fixed_ai_level
    else
        neighbor_signals = collect_neighbor_signals(agent, neighbor_agents)
        choose_ai_level(agent; neighbor_signals=neighbor_signals)
    end

    # Build perception
    perception = if !isnothing(uncertainty_env)
        agent_traits = agent.traits
        visible_opps = opportunities
        perceive_uncertainty(
            uncertainty_env,
            agent_traits,
            visible_opps,
            market_conditions;
            ai_level=ai_level,
            agent_knowledge=Set(keys(agent.resources.knowledge)),
            sector_knowledge=agent.resources.knowledge,
            action_history=agent.action_history,
            ai_learning_profile=agent.ai_learning
        )
    else
        Dict{String,Any}(
            "decision_confidence" => 0.5,
            "actor_ignorance" => Dict("level" => 0.5),
            "practical_indeterminism" => Dict("level" => 0.5),
            "agentic_novelty" => Dict("level" => 0.5),
            "competitive_recursion" => Dict("level" => 0.5)
        )
    end

    # Calculate utilities for each action
    estimated_cost = agent.config.BASE_OPERATIONAL_COST

    invest_utility = calculate_investment_utility(agent, opportunities, market_conditions, perception; ai_level=ai_level)
    innovate_utility = calculate_innovation_utility(agent, market_conditions, perception)
    explore_utility = calculate_exploration_utility(agent, perception; ai_level=ai_level)
    maintain_utility = calculate_maintain_utility(agent, market_conditions, perception; estimated_cost=estimated_cost)

    # Select action
    utilities = Dict(
        "invest" => invest_utility,
        "innovate" => innovate_utility,
        "explore" => explore_utility,
        "maintain" => maintain_utility
    )

    # Can't invest without opportunities
    if isempty(opportunities)
        utilities["invest"] = 0.0
    end

    # Add noise and select
    noisy_utilities = Dict{String,Float64}()
    for (action, util) in utilities
        noise = agent.config.ACTION_SELECTION_NOISE * randn(agent.rng)
        noisy_utilities[action] = max(0.0, util + noise)
    end

    # Softmax selection
    temperature = agent.config.ACTION_SELECTION_TEMPERATURE
    total = sum(exp(u / temperature) for u in values(noisy_utilities))
    if total > 0
        probs = Dict(a => exp(u / temperature) / total for (a, u) in noisy_utilities)
    else
        probs = Dict("maintain" => 1.0, "invest" => 0.0, "innovate" => 0.0, "explore" => 0.0)
    end

    # Sample action
    r = rand(agent.rng)
    cumsum = 0.0
    chosen_action = "maintain"
    for action in ["invest", "innovate", "explore", "maintain"]
        cumsum += get(probs, action, 0.0)
        if r <= cumsum
            chosen_action = action
            break
        end
    end

    # Execute chosen action
    outcome = if chosen_action == "invest" && !isempty(opportunities)
        evals = evaluate_portfolio_opportunities(agent, opportunities, market_conditions, perception; ai_level=ai_level)
        if !isempty(evals)
            best_opp = evals[1]["opportunity"]
            execute_action!(agent, "invest", market, round_num; opportunity=best_opp)
        else
            execute_action!(agent, "maintain", market, round_num)
        end
    else
        execute_action!(agent, chosen_action, market, round_num)
    end

    outcome["ai_level_used"] = ai_level
    outcome["utilities"] = utilities
    outcome["perception"] = perception

    return outcome
end

# ============================================================================
# NEIGHBOR SIGNALS AND SOCIAL INFLUENCE
# ============================================================================

"""
Collect signals from neighbor agents for social influence.
"""
function collect_neighbor_signals(
    agent::EmergentAgent,
    neighbors::Vector{EmergentAgent}
)::Dict{String,Any}
    signals = Dict{String,Any}()

    if isempty(neighbors)
        return signals
    end

    alive_neighbors = filter(n -> n.alive && n.id != agent.id, neighbors)
    if isempty(alive_neighbors)
        return signals
    end

    # AI distribution among neighbors
    ai_counts = Dict{String,Int}()
    for n in alive_neighbors
        level = get_ai_level(n)
        ai_counts[level] = get(ai_counts, level, 0) + 1
    end
    total_neighbors = length(alive_neighbors)
    ai_distribution = Dict(k => v / total_neighbors for (k, v) in ai_counts)
    signals["ai_distribution"] = ai_distribution

    # AI adoption pressure (fraction using non-none AI)
    ai_users = count(n -> get_ai_level(n) != "none", alive_neighbors)
    signals["ai_adoption_pressure"] = ai_users / total_neighbors

    # Peer ROI comparison
    agent_roi = compute_roic(agent.resources.performance)
    neighbor_rois = [compute_roic(n.resources.performance) for n in alive_neighbors]
    avg_neighbor_roi = isempty(neighbor_rois) ? 0.0 : mean(neighbor_rois)
    signals["peer_roi_gap"] = avg_neighbor_roi - agent_roi

    # Average capital
    signals["avg_neighbor_capital"] = mean(get_capital(n) for n in alive_neighbors)

    return signals
end

# ============================================================================
# OUTCOME PROCESSING AND TRAIT EVOLUTION
# ============================================================================

"""
Update agent state from an outcome.
"""
function update_state_from_outcome!(
    agent::EmergentAgent,
    outcome::Dict{String,Any};
    ai_was_accurate::Union{Bool,Nothing} = nothing
)
    # Update capital if returned
    if haskey(outcome, "capital_returned")
        capital_returned = Float64(outcome["capital_returned"])
        set_capital!(agent, get_capital(agent) + capital_returned)
    end

    # Track AI accuracy
    ai_used = get(outcome, "ai_used", false)
    if ai_used && !isnothing(ai_was_accurate)
        update_ai_trust!(agent, ai_was_accurate, outcome)
    end

    # Update traits from experience
    was_successful = get(outcome, "success", false)
    evolve_traits_from_experience!(agent, was_successful, ai_used; ai_was_accurate=ai_was_accurate)

    # Store outcome
    agent.last_outcome = outcome
end

"""
Update AI trust based on accuracy.
"""
function update_ai_trust!(
    agent::EmergentAgent,
    ai_was_accurate::Bool,
    outcome::Dict{String,Any}
)
    current_trust = agent.ai_trust
    base_rate = agent.config.AI_TRUST_ADJUSTMENT_RATE

    # Attribution based on analytical ability
    attribution = 1.0 - get(agent.traits, "analytical_ability", 0.5)

    # Surprise factor for inaccurate high-confidence predictions
    surprise = 1.0
    confidence = get(outcome, "ai_confidence", 0.5)
    if !ai_was_accurate && confidence > 0.75
        surprise = 1.5
    end

    effective_rate = base_rate * attribution * surprise

    if ai_was_accurate
        agent.ai_trust = current_trust + (1.0 - current_trust) * effective_rate
    else
        agent.ai_trust = current_trust - current_trust * effective_rate
    end

    agent.ai_trust = clamp(agent.ai_trust, 0.0, 1.0)
    agent.traits["ai_trust"] = agent.ai_trust
end

"""
Evolve agent traits based on experience.
"""
function evolve_traits_from_experience!(
    agent::EmergentAgent,
    successful_action::Bool,
    used_ai::Bool;
    ai_was_accurate::Union{Bool,Nothing} = nothing
)
    # Competence evolution
    target_competence = successful_action ? 1.0 : 0.0
    current_competence = get(agent.traits, "competence", 0.5)

    base_momentum = agent.trait_momentum
    cognitive_mult = get(agent.traits, "cognitive_style", 0.8)
    effective_momentum = base_momentum * cognitive_mult

    new_competence = effective_momentum * current_competence + (1.0 - effective_momentum) * target_competence
    agent.traits["competence"] = clamp(new_competence, 0.0, 1.0)
    agent.competence = agent.traits["competence"]

    # Innovativeness evolves toward successful innovation
    if agent.last_action == "innovate"
        current_innov = agent.innovativeness
        target_innov = successful_action ? min(1.0, current_innov + 0.1) : max(0.0, current_innov - 0.05)
        agent.innovativeness = 0.9 * current_innov + 0.1 * target_innov
        agent.traits["innovativeness"] = agent.innovativeness
    end
end

"""
Record a paradox observation (confidence vs outcome gap).
"""
function record_paradox_observation!(
    agent::EmergentAgent,
    decision_confidence::Float64,
    realized_roi::Float64;
    ai_used::Bool = false
)
    decision_conf = clamp(decision_confidence, 0.05, 0.99)
    roi_value = clamp(realized_roi, 0.0, 3.0)

    # Convert ROI to score
    roi_score = 1.0 / (1.0 + exp(-3.0 * (roi_value - 1.0)))
    gap = decision_conf - roi_score

    # Weight by AI usage
    weight = ai_used ? 1.2 : 0.8

    # Update paradox signal with inertia
    inertia = 0.85
    if !haskey(agent.last_outcome, "paradox_signal")
        agent.last_outcome["paradox_signal"] = 0.0
    end
    old_signal = Float64(get(agent.last_outcome, "paradox_signal", 0.0))
    new_signal = inertia * old_signal + (1.0 - inertia) * gap * weight
    agent.last_outcome["paradox_signal"] = clamp(new_signal, -1.0, 1.0)
end

# ============================================================================
# OPERATIONAL COSTS
# ============================================================================

"""
Estimate operational costs including competition pressure.
"""
function estimate_operational_costs(
    agent::EmergentAgent,
    market::MarketEnvironment
)::Float64
    base_cost = agent.config.BASE_OPERATIONAL_COST

    # Competition pressure from active investments
    if !isempty(agent.active_investments)
        competitions = Float64[]
        for inv in agent.active_investments
            opp = get(inv, "opportunity", nothing)
            if !isnothing(opp) && opp isa Opportunity
                push!(competitions, opp.competition)
            end
        end
        if !isempty(competitions)
            avg_competition = mean(competitions)
            base_cost *= 1.0 + avg_competition * 0.2
        end
    end

    return base_cost
end
