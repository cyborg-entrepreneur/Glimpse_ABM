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

    # Distress tracking (matches Python _evaluate_failure_conditions)
    liquidity_streak::Int       # Rounds below liquidity floor
    equity_streak::Int          # Rounds below equity floor
    burn_streak::Int            # Rounds with high leverage + negative burn
    burn_history::Vector{Float64}  # Last N rounds of capital deltas (max BURN_HISTORY_WINDOW)
    failure_reason::Union{String,Nothing}  # Reason for failure if dead
    operating_cost_estimate::Float64  # Latest operating cost estimate for liquidity floor calculation
    insolvency_streak::Int      # Pre-decision insolvency check (matches Python make_decision)

    # Decision state
    uncertainty_response::UncertaintyResponseProfile
    action_history::Vector{String}
    last_action::String
    last_outcome::Dict{String,Any}

    # Portfolio
    active_investments::Vector{Dict{String,Any}}
    locked_capital::Float64

    # Action selection state (matches Python)
    action_bias::Dict{String,Float64}
    paradox_signal::Float64
    recent_actions::Vector{String}  # Rolling window of recent actions

    # Subscription tracking (matches Python)
    subscription_accounts::Dict{String,Int}  # tier => remaining rounds
    subscription_rates::Dict{String,Float64}  # tier => per-round rate
    subscription_deferral_remaining::Dict{String,Int}  # tier => deferral rounds left
    last_subscription_charge::Float64

    # Last perception for action execution (matches Python _last_perception)
    last_perception::Dict{String,Any}

    # RNG
    rng::Random.AbstractRNG
end

function EmergentAgent(
    id::Int,
    config::EmergentConfig;
    initial_capital::Union{Float64,Nothing} = nothing,
    fixed_ai_level::Union{String,Nothing} = nothing,
    initial_ai_level::String = "none",
    rng::Random.AbstractRNG = Random.default_rng()
)
    # Sample initial capital
    capital = if isnothing(initial_capital)
        rand(rng, Uniform(config.INITIAL_CAPITAL_RANGE...))
    else
        initial_capital
    end

    # Sample traits
    traits = sample_all_traits(config; rng=rng)

    # Sample action biases (matches Python implementation)
    bias_sigma = max(0.0, config.ACTION_BIAS_SIGMA)
    action_bias = Dict{String,Float64}(
        "invest" => randn(rng) * bias_sigma,
        "innovate" => randn(rng) * bias_sigma,
        "explore" => randn(rng) * bias_sigma,
        "maintain" => randn(rng) * bias_sigma
    )

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
        initial_ai_level,  # current_ai_level (can be set to distribute across tiers)
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
        0,  # liquidity_streak (distress tracking)
        0,  # equity_streak (distress tracking)
        0,  # burn_streak (distress tracking)
        Float64[],  # burn_history (distress tracking)
        nothing,  # failure_reason
        config.BASE_OPERATIONAL_COST,  # operating_cost_estimate
        0,  # insolvency_streak (pre-decision check)
        UncertaintyResponseProfile(rng=rng),
        String[],  # action_history
        "maintain",  # last_action
        Dict{String,Any}(),  # last_outcome
        Dict{String,Any}[],  # active_investments
        0.0,  # locked_capital
        action_bias,  # action_bias
        0.0,  # paradox_signal
        String[],  # recent_actions
        Dict{String,Int}(),  # subscription_accounts
        Dict{String,Float64}(),  # subscription_rates
        Dict{String,Int}(),  # subscription_deferral_remaining
        0.0,  # last_subscription_charge
        Dict{String,Any}(),  # last_perception
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
Evaluate failure conditions matching Python _evaluate_failure_conditions exactly.

Four failure modes:
1. liquidity_failure: capital < liquidity_floor for grace_period rounds
2. equity_failure: capital_ratio < ratio_floor for grace_period rounds
3. funding_shock: 5% random failure when severely depressed
4. burnout_failure: high leverage + negative burn rate for grace_period rounds

Returns failure reason string or nothing if agent survives.
"""
function evaluate_failure_conditions!(agent::EmergentAgent, capital_after::Union{Float64,Nothing}=nothing)::Union{String,Nothing}
    if !agent.alive
        return agent.failure_reason
    end

    if isnothing(capital_after)
        capital_after = Float64(agent.resources.capital)
    end

    initial_equity = max(agent.resources.performance.initial_equity, 1.0)
    grace_period = max(1, agent.config.INSOLVENCY_GRACE_ROUNDS)

    reason::Union{String,Nothing} = nothing

    # Get operating cost estimate (default to BASE_OPERATIONAL_COST if not set)
    operating_cost = agent.operating_cost_estimate
    if operating_cost <= 0.0
        operating_cost = agent.config.BASE_OPERATIONAL_COST
    end

    reserve_months = max(1, agent.config.OPERATING_RESERVE_MONTHS)

    # Liquidity floor: max of survival threshold and operating cost reserve
    liquidity_floor = max(
        agent.config.SURVIVAL_THRESHOLD,
        operating_cost * reserve_months
    )

    # AI trust relief factor
    ai_level = get_ai_level(agent)
    relief_factor = 1.0
    if ai_level != "none"
        trust = clamp(get(agent.traits, "ai_trust", 0.5), 0.0, 1.0)
        discount = clamp(agent.config.AI_TRUST_RESERVE_DISCOUNT, 0.0, 0.9)
        relief_factor = clamp(1.0 - trust * discount, 0.5, 1.0)
        liquidity_floor *= relief_factor
    end

    # Check 1: Liquidity failure
    if capital_after < liquidity_floor
        agent.liquidity_streak += 1
        if agent.liquidity_streak >= grace_period
            reason = "liquidity_failure"
        end
    else
        agent.liquidity_streak = 0
    end

    # Check 2: Equity failure (capital ratio)
    ratio_floor = agent.config.SURVIVAL_CAPITAL_RATIO * relief_factor
    capital_ratio = capital_after / initial_equity

    if isnothing(reason)
        if capital_ratio < ratio_floor
            agent.equity_streak += 1
            if agent.equity_streak >= grace_period
                reason = "equity_failure"
            end
        else
            agent.equity_streak = 0
        end
    end

    # Check 3: Funding shock (5% random failure when severely depressed)
    if isnothing(reason) && capital_ratio < ratio_floor * 0.85 && rand(agent.rng) < 0.05
        reason = "funding_shock"
    end

    # Check 4: Burnout failure (high leverage + negative burn rate)
    burn_window = agent.config.BURN_HISTORY_WINDOW
    leverage_cap = agent.config.BURN_LEVERAGE_CAP
    burn_threshold = agent.config.BURN_FAILURE_THRESHOLD

    leverage = 0.0
    if capital_after > 0
        leverage = agent.locked_capital / max(capital_after, 1e-6)
    end

    if isnothing(reason) && length(agent.burn_history) == burn_window && leverage >= leverage_cap
        burn_avg = mean(agent.burn_history)
        if burn_avg < -burn_threshold * initial_equity
            agent.burn_streak += 1
            if agent.burn_streak >= grace_period
                reason = "burnout_failure"
            end
        else
            agent.burn_streak = 0
        end
    elseif length(agent.burn_history) < burn_window || leverage < leverage_cap
        agent.burn_streak = 0
    end

    return reason
end

"""
Update burn history with capital delta for this round.
Should be called each round with the change in capital.
"""
function update_burn_history!(agent::EmergentAgent, capital_delta::Float64)
    burn_window = agent.config.BURN_HISTORY_WINDOW
    push!(agent.burn_history, capital_delta)
    # Keep only the last burn_window entries
    while length(agent.burn_history) > burn_window
        popfirst!(agent.burn_history)
    end
end

"""
Check if agent is alive and above survival threshold.
Now uses evaluate_failure_conditions! for full Python compatibility.
"""
function check_survival!(agent::EmergentAgent, round::Int)::Bool
    if !agent.alive
        return false
    end

    reason = evaluate_failure_conditions!(agent)

    if !isnothing(reason)
        agent.alive = false
        agent.failure_reason = reason
        return false
    end

    agent.survival_rounds = round
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
    opportunity::Union{Opportunity,Nothing} = nothing,
    investment_amount::Union{Float64,Nothing} = nothing,  # Pre-computed position size
    innovation_engine::Union{InnovationEngine,Nothing} = nothing,
    market_conditions::Union{Dict{String,Any},Nothing} = nothing,
    ai_info::Union{Dict{String,Any},Nothing} = nothing
)::Dict{String,Any}
    outcome = Dict{String,Any}(
        "action" => action,
        "agent_id" => agent.id,
        "round" => round,
        "ai_level_used" => get_ai_level(agent),
        "capital_before" => get_capital(agent)
    )

    if action == "invest" && !isnothing(opportunity)
        outcome = _execute_invest!(agent, opportunity, market, round, outcome;
                                   sized_amount=investment_amount,
                                   ai_info=ai_info)
    elseif action == "innovate"
        outcome = _execute_innovate!(agent, market, round, outcome;
                                     innovation_engine=innovation_engine,
                                     market_conditions=market_conditions)
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

"""
Execute investment action with optional pre-computed position size.
If sized_amount is provided, use it (from make_portfolio_decision).
Otherwise, fall back to simple fraction-based sizing.
"""
function _execute_invest!(
    agent::EmergentAgent,
    opportunity::Opportunity,
    market::MarketEnvironment,
    round::Int,
    outcome::Dict{String,Any};
    sized_amount::Union{Float64,Nothing} = nothing,
    ai_info::Union{Dict{String,Any},Nothing} = nothing
)::Dict{String,Any}
    capital = get_capital(agent)

    # Use pre-computed sized_amount if available (from make_portfolio_decision)
    # Otherwise fall back to simple fraction-based sizing
    # NOTE: Python does NOT cap at capital_requirements - amount is determined by position sizing
    invest_amount = if !isnothing(sized_amount) && sized_amount > 0
        # Use the confidence-scaled position size from portfolio decision
        min(sized_amount, capital)
    else
        # Legacy fallback: simple fraction-based sizing
        max_invest = capital * agent.config.MAX_INVESTMENT_FRACTION
        min(max_invest, capital)
    end

    if invest_amount <= 0 || invest_amount > capital
        outcome["success"] = false
        outcome["reason"] = "insufficient_capital"
        return outcome
    end

    # Deduct capital and track locked capital
    set_capital!(agent, capital - invest_amount)
    agent.total_invested += invest_amount
    agent.locked_capital += invest_amount

    # Record investment
    investment = Dict{String,Any}(
        "opportunity_id" => opportunity.id,
        "opportunity" => opportunity,
        "amount" => invest_amount,
        "round_invested" => round,
        "maturity_round" => round + opportunity.time_to_maturity,
        "ai_level" => get_ai_level(agent)
    )
    # Attach AI info for maturation outcomes (matches Python accuracy/hallucination adjustments)
    if !isnothing(ai_info)
        investment["ai_info"] = ai_info
    end
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
    outcome::Dict{String,Any};
    innovation_engine::Union{InnovationEngine,Nothing} = nothing,
    market_conditions::Union{Dict{String,Any},Nothing} = nothing
)::Dict{String,Any}
    capital = get_capital(agent)
    ai_level = get_ai_level(agent)

    # Innovation cost (R&D spend)
    rd_spend = min(
        capital * agent.config.INNOVATION_BASE_SPEND_RATIO,
        agent.config.INNOVATION_MAX_SPEND
    )

    if rd_spend > capital * 0.5
        rd_spend = capital * 0.2  # Cap R&D if capital constrained
    end

    set_capital!(agent, capital - rd_spend)

    # Get perception for clarity signal
    perception = agent.last_perception
    mkt_conds = isnothing(market_conditions) ? Dict{String,Any}() : market_conditions

    # ========================================================================
    # USE FULL INNOVATIONENGINE IF AVAILABLE (matches Python exactly)
    # ========================================================================
    if !isnothing(innovation_engine)
        # Attempt innovation using the full InnovationEngine
        innovation = attempt_innovation!(
            innovation_engine,
            agent,
            mkt_conds,
            round;
            ai_level=ai_level,
            uncertainty_perception=perception,
            decision_perception=perception,
            rng=agent.rng
        )

        if isnothing(innovation)
            # Innovation attempt failed (not enough knowledge, etc.)
            # Return partial R&D cost
            recovery = rd_spend * agent.config.INNOVATION_FAIL_RECOVERY_RATIO
            set_capital!(agent, get_capital(agent) + recovery)

            outcome["success"] = false
            outcome["rd_spend"] = rd_spend
            outcome["recovery"] = recovery
            outcome["reason"] = "insufficient_knowledge"
            outcome["ai_bonus_applied"] = 0.0

            record_deployment!(agent.resources.performance, "innovate", rd_spend;
                               ai_level=ai_level, round_num=round)
            return outcome
        end

        # Record R&D investment in engine
        invest_in_rd!(innovation_engine, agent.id, rd_spend)

        # Get all recent innovations for competition evaluation
        market_innovations = collect(values(innovation_engine.innovations))

        # Evaluate innovation success using full engine logic
        success, impact, cash_multiple = evaluate_innovation_success!(
            innovation_engine,
            innovation,
            mkt_conds,
            market_innovations;
            rng=agent.rng
        )

        if success
            innovation_return = rd_spend * cash_multiple
            set_capital!(agent, get_capital(agent) + innovation_return)
            agent.innovation_count += 1
            agent.success_count += 1

            # Record the innovation return
            record_return!(agent.resources.performance, "innovate", innovation_return;
                          ai_level=ai_level, round_num=round)

            outcome["success"] = true
            outcome["rd_spend"] = rd_spend
            outcome["innovation_return"] = innovation_return
            outcome["cash_multiple"] = cash_multiple
            outcome["novelty"] = innovation.novelty
            outcome["scarcity"] = something(innovation.scarcity, 0.5)
            outcome["quality"] = innovation.quality
            outcome["innovation_type"] = innovation.type
            outcome["is_new_combination"] = innovation.is_new_combination
            outcome["market_impact"] = impact
            outcome["innovation_id"] = innovation.id
            outcome["knowledge_components"] = innovation.knowledge_components
            outcome["ai_assisted"] = innovation.ai_assisted
            outcome["ai_domains_used"] = innovation.ai_domains_used
            outcome["combination_signature"] = something(innovation.combination_signature, "")
        else
            # Failure: apply recovery based on innovation properties
            novelty = innovation.novelty
            scarcity = something(innovation.scarcity, 0.5)
            # Higher novelty = lower recovery (riskier bets fail harder)
            novelty_clamped = clamp(novelty, 0.05, 0.95)
            recovery_floor = 0.78 - (novelty_clamped - 0.05) / (0.95 - 0.05) * (0.78 - 0.42)
            salvage = max(agent.config.INNOVATION_FAIL_RECOVERY_RATIO, recovery_floor - 0.12 * (scarcity - 0.5))
            recovery = rd_spend * clamp(salvage, 0.25, 0.65)

            set_capital!(agent, get_capital(agent) + recovery)

            # Record the innovation recovery as return
            record_return!(agent.resources.performance, "innovate", recovery;
                          ai_level=ai_level, round_num=round)

            outcome["success"] = false
            outcome["rd_spend"] = rd_spend
            outcome["recovery"] = recovery
            outcome["novelty"] = novelty
            outcome["scarcity"] = scarcity
            outcome["quality"] = innovation.quality
            outcome["innovation_type"] = innovation.type
            outcome["is_new_combination"] = innovation.is_new_combination
            outcome["innovation_id"] = innovation.id
            outcome["knowledge_components"] = innovation.knowledge_components
        end

        record_deployment!(agent.resources.performance, "innovate", rd_spend;
                           ai_level=ai_level, round_num=round)
        return outcome
    end

    # ========================================================================
    # FALLBACK: INLINE IMPLEMENTATION (when no InnovationEngine provided)
    # This matches Python InnovationEngine logic for use without full wiring
    # ========================================================================

    # Calculate accessible knowledge level
    knowledge_values = collect(values(agent.resources.knowledge))
    avg_knowledge = isempty(knowledge_values) ? 0.1 : mean(knowledge_values)
    knowledge_breadth = length(filter(v -> v > 0.2, knowledge_values))
    knowledge_factor = clamp(avg_knowledge * 0.4 + knowledge_breadth * 0.05, 0.0, 0.3)

    # Base probability with competence
    base_prob = agent.config.INNOVATION_PROBABILITY
    competence_score = (
        get(agent.traits, "innovativeness", 0.5) * 0.6 +
        get(agent.resources.capabilities, "innovation", 0.1) * 0.4
    )

    # REMOVED: Hardcoded ai_bonus_map that gave direct tier-based bonuses
    # Previously: ai_bonus_map = Dict("none" => 0.0, "basic" => 0.12, "advanced" => 0.25, "premium" => 0.35)
    # Now AI bonus emerges purely from agent's learned trust and reliability estimates

    # Compute clarity signal from perception
    clarity_signal = 0.0
    if !isempty(perception)
        ignorance = get(get(perception, "actor_ignorance", Dict()), "level", 0.5)
        indeterminism = get(get(perception, "practical_indeterminism", Dict()), "level", 0.5)
        clarity_signal = ((stable_sigmoid(1.0 - ignorance) + stable_sigmoid(1.0 - indeterminism)) / 2.0) - 0.5
    end

    # AI learning profile adjustments - dynamic bonus based on LEARNED trust and reliability (emergent)
    avg_trust = 0.5
    dynamic_bonus = 0.0
    if ai_level != "none"
        learning_profile = agent.ai_learning
        trust_values = [
            get(learning_profile.domain_trust, dom, 0.5)
            for dom in ["technical_assessment", "innovation_potential"]
        ]
        avg_trust = isempty(trust_values) ? 0.5 : mean(trust_values)

        reliability_signals = Float64[]
        for dom in ["technical_assessment", "innovation_potential"]
            scores = get(learning_profile.accuracy_estimates, dom, Float64[])
            if !isempty(scores)
                push!(reliability_signals, mean(scores[max(1, length(scores)-4):end]) - 0.5)
            end
        end
        reliability = isempty(reliability_signals) ? 0.0 : mean(reliability_signals)
        dynamic_bonus = (avg_trust - 0.5) * 0.2 + reliability * 0.25 + clarity_signal * 0.15
    end

    # AI bonus now purely from emergent learning, no hardcoded tier bonuses
    ai_bonus = clamp(dynamic_bonus, -0.2, 0.3)

    human_bonus = (
        get(agent.traits, "exploration_tendency", 0.5) * 0.15 +
        get(agent.traits, "market_awareness", 0.5) * 0.15 +
        get(agent.traits, "innovativeness", 0.5) * 0.15
    )

    success_prob = clamp(base_prob * 0.35 + competence_score * 0.45 + ai_bonus + human_bonus + knowledge_factor, 0.05, 0.95)

    # FIXED: Remove hardcoded tier_reuse_shift - let effect emerge through info_breadth
    # Previously had direct tier shifts (none=+0.05, premium=-0.08)
    # Now reuse probability emerges from info_breadth: broader info access → more novel
    # combinations available → lower tendency to reuse existing combinations
    ai_cfg = get(agent.config.AI_LEVELS, ai_level, get(agent.config.AI_LEVELS, "none", Dict()))
    info_breadth_reuse = Float64(ai_cfg.info_breadth)
    # Higher info_breadth reduces reuse (access to broader knowledge enables novel combinations)
    reuse_shift = -info_breadth_reuse * 0.12
    reuse_prob = clamp(0.3 + reuse_shift, 0.02, 0.75)
    is_reused_combination = rand(agent.rng) < reuse_prob

    # Innovation type determination
    experience_units = agent.resources.experience_units
    type_probs = Dict{String,Float64}(
        "incremental" => 0.4 + max(experience_units, 0) * 0.001,
        "architectural" => 0.3 + get(agent.traits, "trait_momentum", 0.1) * 0.35,
        "radical" => 0.2 + get(agent.traits, "innovativeness", 0.5) * 0.25,
        "disruptive" => 0.1 + get(agent.traits, "exploration_tendency", 0.5) * 0.2
    )
    if ai_level != "none"
        type_probs["architectural"] += 0.08
        type_probs["radical"] += 0.05
    end

    total_type_prob = sum(values(type_probs))
    r = rand(agent.rng) * total_type_prob
    cumsum_type = 0.0
    innovation_type = "incremental"
    for (itype, prob) in type_probs
        cumsum_type += prob
        if r <= cumsum_type
            innovation_type = itype
            break
        end
    end

    success = rand(agent.rng) < success_prob

    # Novelty and scarcity
    base_novelty = rand(agent.rng, Uniform(0.3, 0.9))
    if innovation_type == "disruptive"
        base_novelty = clamp(base_novelty + 0.2, 0.0, 1.0)
    elseif innovation_type == "radical"
        base_novelty = clamp(base_novelty + 0.1, 0.0, 1.0)
    end
    novelty = is_reused_combination ? base_novelty * 0.6 : base_novelty
    scarcity = clamp(1.0 - avg_knowledge * 0.5 - knowledge_breadth * 0.02, 0.1, 0.9)
    quality = clamp(avg_knowledge * 0.5 + get(agent.resources.capabilities, "innovation", 0.1) * 0.3 + rand(agent.rng) * 0.2, 0.1, 1.0)

    if success
        base_return = rd_spend * agent.config.INNOVATION_SUCCESS_BASE_RETURN
        mult_range = agent.config.INNOVATION_SUCCESS_RETURN_MULTIPLIER
        base_multiplier = rand(agent.rng, Uniform(mult_range[1], mult_range[2]))

        scarcity_bonus = 1.0 + (scarcity - 0.5) * 0.8
        novelty_bonus = 1.0 + (novelty - 0.5) * 0.55
        type_bonus = Dict("incremental" => 1.0, "architectural" => 1.15, "radical" => 1.35, "disruptive" => 1.6)

        cash_multiple = clamp(base_multiplier * scarcity_bonus * novelty_bonus * get(type_bonus, innovation_type, 1.0), 1.1, 8.5)
        innovation_return = base_return * cash_multiple

        set_capital!(agent, get_capital(agent) + innovation_return)
        agent.innovation_count += 1
        agent.success_count += 1

        # Record the innovation return
        record_return!(agent.resources.performance, "innovate", innovation_return;
                      ai_level=ai_level, round_num=round)

        outcome["success"] = true
        outcome["rd_spend"] = rd_spend
        outcome["innovation_return"] = innovation_return
        outcome["cash_multiple"] = cash_multiple
        outcome["novelty"] = novelty
        outcome["scarcity"] = scarcity
        outcome["quality"] = quality
        outcome["innovation_type"] = innovation_type
        outcome["is_new_combination"] = !is_reused_combination
        outcome["ai_bonus_applied"] = ai_bonus
        outcome["clarity_signal"] = clarity_signal
    else
        novelty_clamped = clamp(novelty, 0.05, 0.95)
        recovery_floor = 0.78 - (novelty_clamped - 0.05) / (0.95 - 0.05) * (0.78 - 0.42)
        salvage = max(agent.config.INNOVATION_FAIL_RECOVERY_RATIO, recovery_floor - 0.12 * (scarcity - 0.5))
        recovery = rd_spend * clamp(salvage, 0.25, 0.65)

        set_capital!(agent, get_capital(agent) + recovery)

        # Record the innovation recovery as return
        record_return!(agent.resources.performance, "innovate", recovery;
                      ai_level=ai_level, round_num=round)

        outcome["success"] = false
        outcome["rd_spend"] = rd_spend
        outcome["recovery"] = recovery
        outcome["novelty"] = novelty
        outcome["scarcity"] = scarcity
        outcome["innovation_type"] = innovation_type
        outcome["is_new_combination"] = !is_reused_combination
        outcome["ai_bonus_applied"] = ai_bonus
        outcome["clarity_signal"] = clarity_signal
    end

    record_deployment!(agent.resources.performance, "innovate", rd_spend;
                       ai_level=ai_level, round_num=round)

    return outcome
end

function _execute_explore!(
    agent::EmergentAgent,
    market::MarketEnvironment,
    round::Int,
    outcome::Dict{String,Any}
)::Dict{String,Any}
    ai_level = get_ai_level(agent)

    # FIXED: Remove hardcoded tier_multiplier - let effect emerge through info_breadth
    # Previously had direct tier bonuses (none=0.85, premium=1.35)
    # Now multiplier emerges from AI_LEVELS config info_breadth parameter
    ai_cfg = get(agent.config.AI_LEVELS, ai_level, get(agent.config.AI_LEVELS, "none", Dict()))
    info_breadth = Float64(ai_cfg.info_breadth)
    # Map info_breadth (0.0-0.85) to multiplier range (~0.85-1.36)
    breadth_multiplier = 0.85 + info_breadth * 0.6

    # Exploration cost scales with traits and AI tier
    exploration_tendency = get(agent.traits, "exploration_tendency", 0.3)
    uncertainty_tolerance = get(agent.traits, "uncertainty_tolerance", 0.5)

    trait_factor = (0.02 + exploration_tendency * 0.07) * breadth_multiplier
    uncertainty_cushion = max(0.02, 0.12 - uncertainty_tolerance * 0.08)

    # Get uncertainty values from perception (matches Python uncertainty hooks)
    perception = agent.last_perception
    actor_unc = get(get(perception, "actor_ignorance", Dict()), "level", 0.5)
    recursion_unc = get(get(perception, "competitive_recursion", Dict()), "level", 0.5)
    agentic_unc = get(get(perception, "agentic_novelty", Dict()), "level", 0.5)
    practical_unc = get(get(perception, "practical_indeterminism", Dict()), "level", 0.5)

    # First uncertainty adjustment (matches Python)
    trait_factor *= clamp(1.0 + 0.2 * actor_unc + 0.15 * recursion_unc, 0.6, 1.8)

    # Second uncertainty adjustment (matches Python)
    trait_factor *= clamp(1.0 + 0.15 * agentic_unc - 0.1 * recursion_unc, 0.5, 1.5)
    uncertainty_cushion *= clamp(1.0 + 0.1 * actor_unc + 0.05 * practical_unc, 0.5, 1.6)

    desired_cost = get_capital(agent) * (trait_factor + uncertainty_cushion)
    # Add some randomness to cost (matches Python np.random.normal(1.0, 0.25))
    cost_noise = clamp(randn(agent.rng) * 0.25 + 1.0, 0.4, 1.8)
    # Match Python: cap at 5000 only, no 10% capital cap
    explore_cost = min(desired_cost * cost_noise, 5000.0)

    set_capital!(agent, get_capital(agent) - explore_cost)

    # Discovery probability - effect emerges through info_quality
    base_discovery_prob = agent.config.DISCOVERY_PROBABILITY

    # FIXED: Remove hardcoded discovery_bonus - let effect emerge through info_quality
    # Previously had direct tier bonuses (none=0.0, premium=0.18)
    # Now discovery bonus emerges from AI_LEVELS config info_quality parameter
    info_quality = Float64(ai_cfg.info_quality)
    # Better info_quality → better ability to identify discovery opportunities
    ai_discovery_bonus = info_quality * 0.2

    discovery_prob = clamp(base_discovery_prob + ai_discovery_bonus, 0.1, 0.9)
    discovered = rand(agent.rng) < discovery_prob

    # Determine exploration type based on innovation saturation (matches Python)
    # Calculate innovation saturation from market innovations
    recent_innovations = length(market.innovations)
    innovation_saturation = clamp(recent_innovations / max(1, market.n_agents * 0.5), 0.1, 0.7)
    is_niche_discovery = rand(agent.rng) < innovation_saturation

    if discovered
        if is_niche_discovery
            # Niche discovery with modifiers (matches Python)
            outcome["exploration_type"] = "niche_discovery"
            existing_sectors = collect(agent.config.SECTORS)
            base_sector = rand(agent.rng, existing_sectors)

            niche_modifiers = ["premium", "budget", "sustainable", "digital", "local", "specialized"]
            niche_modifier = rand(agent.rng, niche_modifiers)
            niche_id = "$(base_sector)_$(niche_modifier)"

            # Add knowledge about this new niche
            if !haskey(agent.resources.knowledge, niche_id)
                agent.resources.knowledge[niche_id] = rand(agent.rng, Uniform(0.2, 0.4))
            else
                agent.resources.knowledge[niche_id] = min(1.0, agent.resources.knowledge[niche_id] + 0.2)
            end

            # Apply amplification based on exploration tendency and AI tier
            trait_amp = (0.7 + 0.6 * exploration_tendency) * breadth_multiplier
            agent.resources.knowledge[niche_id] = min(
                1.0, agent.resources.knowledge[niche_id] * (0.9 + 0.2 * trait_amp)
            )
            agent.resources.knowledge_last_used[niche_id] = round

            outcome["success"] = true
            outcome["discovered_sector"] = niche_id
            outcome["knowledge_gain"] = agent.resources.knowledge[niche_id]
            outcome["niche_modifier"] = niche_modifier

            # Serendipity rewards for niche discovery (higher multiplier)
            serendipity_chance = clamp(
                (0.18 + 0.4 * (exploration_tendency - 0.3)) * breadth_multiplier,
                0.05,
                0.7
            )
            if rand(agent.rng) < serendipity_chance
                serendipity_reward = explore_cost * rand(agent.rng, Uniform(2.0, 6.0))
                set_capital!(agent, get_capital(agent) + serendipity_reward)
                record_return!(agent.resources.performance, "explore", serendipity_reward; ai_level="none", round_num=round)
                outcome["serendipity_reward"] = serendipity_reward
            end
        else
            # Standard sector exploration
            outcome["exploration_type"] = "sector_knowledge"
            sector = rand(agent.rng, agent.config.SECTORS)
            current_knowledge = get(agent.resources.knowledge, sector, 0.1)

            # Knowledge gain also scales with AI tier
            base_gain = rand(agent.rng, Uniform(0.05, 0.15))
            knowledge_gain = base_gain * breadth_multiplier
            agent.resources.knowledge[sector] = min(1.0, current_knowledge + knowledge_gain)
            agent.resources.knowledge_last_used[sector] = round

            outcome["success"] = true
            outcome["discovered_sector"] = sector
            outcome["knowledge_gain"] = knowledge_gain

            # Serendipity rewards for sector exploration (lower multiplier)
            serendipity_chance = clamp(
                (0.12 + 0.25 * (exploration_tendency - 0.4)) * breadth_multiplier,
                0.04,
                0.55
            )
            if rand(agent.rng) < serendipity_chance
                serendipity_reward = explore_cost * rand(agent.rng, Uniform(1.5, 4.0))
                set_capital!(agent, get_capital(agent) + serendipity_reward)
                record_return!(agent.resources.performance, "explore", serendipity_reward; ai_level="none", round_num=round)
                outcome["serendipity_reward"] = serendipity_reward
            end
        end
    else
        outcome["success"] = false
        outcome["exploration_type"] = "failed"
    end

    outcome["explore_cost"] = explore_cost
    outcome["breadth_multiplieriplier_applied"] = breadth_multiplier
    record_deployment!(agent.resources.performance, "explore", explore_cost;
                       ai_level=ai_level, round_num=round)

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
Matches Python check_matured_investments with binary success/failure and severe loss on failure.
"""
function process_matured_investments!(
    agent::EmergentAgent,
    market::MarketEnvironment,
    round::Int;
    market_conditions::Union{Dict{String,Any},Nothing} = nothing
)::Vector{Dict{String,Any}}
    if !agent.alive
        return Dict{String,Any}[]
    end

    matured_outcomes = Dict{String,Any}[]
    remaining_investments = Dict{String,Any}[]

    # Use provided market_conditions if given, otherwise get from market
    market_conditions = isnothing(market_conditions) ? get_market_conditions(market) : market_conditions

    for investment in agent.active_investments
        if investment["maturity_round"] <= round
            # Investment matures
            opp = investment["opportunity"]
            invested_amount = Float64(investment["amount"])
            ai_tier = get(investment, "ai_level", "none")
            ai_info = get(investment, "ai_info", nothing)

            # ================================================================
            # STEP 1: Calculate failure chance (matches Python exactly)
            # ================================================================
            raw_risk = coalesce(opp.latent_failure_potential, 0.5)
            risk = clamp(raw_risk, 0.05, 0.95)

            # AI accuracy reduces failure chance
            accuracy = 0.0
            hallucination = 0.0
            if !isnothing(ai_info) && isa(ai_info, Dict)
                accuracy = Float64(get(ai_info, "actual_accuracy", 0.0))
                hallucination = get(ai_info, "contains_hallucination", false) ? 0.1 : 0.0
            end
            failure_adjustment = -0.15 * accuracy + hallucination

            # Sector failure multiplier
            sector_key = coalesce(opp.sector, "unknown")
            sector_adjustments = get(market_conditions, "sector_demand_adjustments", Dict{String,Any}())
            sector_adjust = get(sector_adjustments, sector_key, nothing)
            sector_failure = 1.0
            if isa(sector_adjust, Dict)
                sector_failure = Float64(get(sector_adjust, "failure", 1.0))
            end

            # Regime failure multiplier
            regime_failure = Float64(get(market_conditions, "regime_failure_multiplier", 1.0))

            # Base failure probability
            base_failure = 0.05 + 0.5 * risk * sector_failure * regime_failure + failure_adjustment

            # Crowding increases failure
            crowding_metrics = get(market_conditions, "crowding_metrics", Dict{String,Any}())
            crowd_idx = Float64(get(crowding_metrics, "crowding_index", 0.25))
            crowd_threshold = agent.config.RETURN_DEMAND_CROWDING_THRESHOLD
            crowd_multiplier = 1.0
            if crowd_idx > crowd_threshold
                crowd_multiplier += 0.25 * (crowd_idx - crowd_threshold)
            end

            failure_chance = clamp(base_failure * crowd_multiplier, 0.05, 0.9)

            # ================================================================
            # STEP 2: Binary success/failure determination
            # ================================================================
            success = rand(agent.rng) >= failure_chance

            # ================================================================
            # STEP 3: Calculate return based on success/failure
            # ================================================================
            ret_multiple = 0.0
            defaulted = false

            if success
                # SUCCESS: Use realized_return for multiplier
                # NOTE: Unlike previous Julia version, we do NOT enforce min 1.0
                # Python allows realized_return < 1.0 for successful investments
                # due to stochastic noise, which causes capital loss even on "success"
                ret_multiple = realized_return(opp, market_conditions, ai_tier; rng=agent.rng)
            else
                # FAILURE: Calculate recovery_ratio (0.0 to 0.4 typically)
                # Severity and recovery floor based on risk
                severity = 0.25 + (risk - 0.05) / (0.95 - 0.05) * (0.9 - 0.25)
                recovery_floor = 0.3 - (risk - 0.05) / (0.95 - 0.05) * (0.3 - 0.02)
                loss_noise = randn(agent.rng) * 0.1
                recovery_ratio = clamp(recovery_floor + loss_noise, 0.0, 0.4)

                # HHI adjustment
                combo_hhi = Float64(get(market_conditions, "combo_hhi", 0.0))
                recovery_ratio *= clamp(1.0 - combo_hhi * 0.8, 0.05, 1.0)

                # Scarcity and reuse adjustments
                unc_state = get(market_conditions, "uncertainty_state", Dict{String,Any}())
                agentic_state = isa(unc_state, Dict) ? get(unc_state, "agentic_novelty", unc_state) : Dict{String,Any}()
                scarcity_signal = if isa(agentic_state, Dict)
                    Float64(get(agentic_state, "component_scarcity", coalesce(opp.component_scarcity, 0.5)))
                else
                    coalesce(opp.component_scarcity, 0.5)
                end
                reuse_pressure = isa(agentic_state, Dict) ? Float64(get(agentic_state, "reuse_pressure", 0.0)) : 0.0

                recovery_ratio *= clamp(1.0 - reuse_pressure * 0.3, 0.35, 1.0)
                recovery_ratio += 0.04 * max(0.0, scarcity_signal - 0.5)

                # AI tier effects on recovery
                tier_shares = get(market_conditions, "tier_invest_share", Dict{String,Float64}())
                tier_share = Float64(get(tier_shares, ai_tier, 0.0))

                if ai_tier != "none"
                    accuracy_cushion = max(0.0, accuracy - 0.7) * (0.1 + 0.05 * scarcity_signal)
                    recovery_ratio = clamp(recovery_ratio + accuracy_cushion, 0.0, 0.6)
                    if !isnothing(ai_info) && isa(ai_info, Dict) && get(ai_info, "contains_hallucination", false)
                        recovery_ratio = clamp(recovery_ratio - 0.07, 0.0, 0.6)
                    end
                else
                    recovery_ratio = clamp(recovery_ratio - 0.05, 0.0, 0.4)
                end

                # Tier share penalty
                recovery_ratio *= clamp(1.0 - tier_share * (1.0 + 0.3 * (1.0 - scarcity_signal)), 0.05, 1.0)

                # Default trigger: can cause total loss
                default_trigger = rand(agent.rng) < (0.2 * severity * rand(agent.rng))
                if default_trigger
                    recovery_ratio = 0.0
                    defaulted = true
                end

                ret_multiple = recovery_ratio
            end

            # ================================================================
            # STEP 4: Apply to agent
            # ================================================================
            capital_returned = invested_amount * ret_multiple

            set_capital!(agent, get_capital(agent) + capital_returned)
            agent.total_returned += capital_returned
            agent.locked_capital = max(0.0, agent.locked_capital - invested_amount)

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
                "defaulted" => defaulted,
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
Matches Python _estimate_operational_costs + apply_operational_costs logic.
"""
function apply_operational_costs!(agent::EmergentAgent, round::Int; market::Union{MarketEnvironment,Nothing}=nothing)
    if !agent.alive
        return
    end

    # Calculate operational costs matching Python implementation
    cost = estimate_operational_costs(agent, market)
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
    invest_roic = compute_roic(perf, "invest")
    innovate_roic = compute_roic(perf, "innovate")
    explore_roic = compute_roic(perf, "explore")

    metrics["overall_roic"] = overall_roic
    metrics["invest_roic"] = invest_roic
    metrics["innovate_roic"] = innovate_roic
    metrics["explore_roic"] = explore_roic

    # Separate AI-assisted vs baseline (non-AI) performance (matches Python)
    overall_by_ai = get(perf.deployments_by_ai, "overall", Dict{String,Float64}())
    returns_by_ai = get(perf.returns_by_ai, "overall", Dict{String,Float64}())

    # AI-assisted totals (basic, advanced, premium)
    ai_deployed = 0.0
    ai_returned = 0.0
    for tier in ["basic", "advanced", "premium"]
        ai_deployed += get(overall_by_ai, tier, 0.0)
        ai_returned += get(returns_by_ai, tier, 0.0)
    end

    # Baseline (none) totals
    baseline_deployed = get(overall_by_ai, "none", 0.0)
    baseline_returned = get(returns_by_ai, "none", 0.0)

    # Compute AI ROI
    ai_avg_roi = if ai_deployed > 0
        ai_returned / ai_deployed
    else
        1.0
    end

    # Compute baseline ROI
    baseline_avg_roi = if baseline_deployed > 0
        baseline_returned / baseline_deployed
    else
        1.0  # Default to 1.0 (break-even) if no baseline data
    end

    # Cash flow metrics
    deployed_total = get(perf.deployed_by_action, "overall", 0.0)
    returned_total = get(perf.returned_by_action, "overall", 0.0)
    metrics["overall_cash_flow_total"] = returned_total - deployed_total

    # Compute recent metrics from roi_events (matches Python recent_window)
    recent_window = max(1, agent.config.AI_TIER_RECENT_WINDOW)
    recent_events = perf.roi_events
    n_events = length(recent_events)

    # Recent AI and baseline tracking
    recent_ai_deployed = 0.0
    recent_ai_returned = 0.0
    recent_baseline_deployed = 0.0
    recent_baseline_returned = 0.0
    recent_total_deployed = 0.0
    recent_total_returned = 0.0

    # Process recent events (last N events, matching deployments with returns)
    recent_start = max(1, n_events - recent_window * 2 + 1)  # Approximate window
    for i in recent_start:n_events
        event = recent_events[i]
        ai_level = get(event, "ai_level", "none")
        amount = get(event, "amount", 0.0)
        event_type = get(event, "type", "")

        is_ai = ai_level in ("basic", "advanced", "premium")

        if event_type == "deployment"
            recent_total_deployed += amount
            if is_ai
                recent_ai_deployed += amount
            else
                recent_baseline_deployed += amount
            end
        elseif event_type == "return"
            recent_total_returned += amount
            if is_ai
                recent_ai_returned += amount
            else
                recent_baseline_returned += amount
            end
        end
    end

    # Recent ROI calculations
    recent_ai_avg_roi = if recent_ai_deployed > 0
        recent_ai_returned / recent_ai_deployed
    else
        ai_avg_roi  # Fall back to overall
    end

    recent_baseline_avg_roi = if recent_baseline_deployed > 0
        recent_baseline_returned / recent_baseline_deployed
    else
        baseline_avg_roi  # Fall back to overall
    end

    # ROI gain metrics (AI performance vs baseline) - matches Python exactly
    metrics["roi_gain"] = ai_avg_roi - baseline_avg_roi
    metrics["recent_roi_gain"] = recent_ai_avg_roi - recent_baseline_avg_roi

    # ROI gain ratio (relative improvement)
    metrics["roi_gain_ratio"] = baseline_avg_roi > 1e-3 ? (ai_avg_roi / baseline_avg_roi) - 1.0 : 0.0
    metrics["recent_roi_gain_ratio"] = recent_baseline_avg_roi > 1e-3 ? (recent_ai_avg_roi / recent_baseline_avg_roi) - 1.0 : 0.0

    # Recent cash flow
    metrics["recent_cash_flow_total"] = recent_total_returned - recent_total_deployed

    # Activity metrics
    n_actions = length(agent.action_history)
    metrics["recent_ai_activity"] = max(1.0, Float64(min(recent_window, n_actions)) / 5.0)

    # Additional metrics for compatibility
    metrics["avg_roi"] = ai_deployed > 0 ? ai_avg_roi : baseline_avg_roi
    metrics["baseline_avg_roi"] = baseline_avg_roi
    metrics["recent_avg_roi"] = recent_ai_deployed > 0 ? recent_ai_avg_roi : recent_baseline_avg_roi
    metrics["baseline_recent_avg_roi"] = recent_baseline_avg_roi

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
    cost_type = ai_config.cost_type

    # Scale costs by AI_COST_INTENSITY (for robustness testing)
    cost_intensity = agent.config.AI_COST_INTENSITY

    if cost_type == "subscription"
        base_cost = ai_config.cost * cost_intensity
        per_use = ai_config.per_use_cost * cost_intensity
        # Amortize subscription over rounds
        amort_rounds = max(1, agent.config.AI_SUBSCRIPTION_AMORTIZATION_ROUNDS)
        return base_cost / amort_rounds + per_use * max(expected_calls, 0.0)
    elseif cost_type == "per_use"
        return ai_config.cost * cost_intensity * max(expected_calls, 0.0)
    end

    return 0.0
end

"""
Start a subscription schedule for an AI tier (matches Python _start_subscription_schedule).
"""
function start_subscription_schedule!(agent::EmergentAgent, ai_level::String)
    if ai_level == "none"
        return
    end

    ai_config = get(agent.config.AI_LEVELS, ai_level, nothing)
    if isnothing(ai_config)
        return
    end

    cycle = max(1, agent.config.AI_SUBSCRIPTION_AMORTIZATION_ROUNDS)
    base_cost = ai_config.cost * agent.config.AI_COST_INTENSITY
    if base_cost <= 0
        return
    end

    agent.subscription_accounts[ai_level] = cycle
    agent.subscription_rates[ai_level] = base_cost / cycle
end

"""
Compute subscription deferral rounds based on entrepreneurial drive (matches Python).
"""
function compute_subscription_deferral_rounds(agent::EmergentAgent, ai_level::String)::Int
    base_grace = agent.config.AI_SUBSCRIPTION_FLOAT_BASE_ROUNDS
    max_extra = max(0, agent.config.AI_SUBSCRIPTION_FLOAT_MAX_ROUNDS)
    drive = get(agent.traits, "entrepreneurial_drive", 0.5)
    extra = round(Int, max_extra * max(0.0, drive))
    return base_grace + extra
end

"""
Charge one subscription installment (matches Python _charge_subscription_installment).
Returns the amount charged.
"""
function charge_subscription_installment!(agent::EmergentAgent, ai_level::String)::Float64
    remaining = get(agent.subscription_accounts, ai_level, 0)
    rate = get(agent.subscription_rates, ai_level, 0.0)

    if remaining <= 0 || rate <= 0
        return 0.0
    end

    # Check deferral
    deferral = get(agent.subscription_deferral_remaining, ai_level, 0)
    if deferral > 0
        next_deferral = deferral - 1
        if next_deferral > 0
            agent.subscription_deferral_remaining[ai_level] = next_deferral
        else
            delete!(agent.subscription_deferral_remaining, ai_level)
        end
        return 0.0
    end

    # Charge the subscription
    set_capital!(agent, get_capital(agent) - rate)
    remaining -= 1

    if remaining <= 0
        delete!(agent.subscription_accounts, ai_level)
        delete!(agent.subscription_rates, ai_level)
    else
        agent.subscription_accounts[ai_level] = remaining
    end

    return rate
end

"""
Apply all active subscription charges (matches Python _apply_subscription_carry).
Returns total amount charged.
"""
function apply_subscription_carry!(agent::EmergentAgent, current_round::Int)::Float64
    total = 0.0
    credit_line = agent.config.AI_CREDIT_LINE_ROUNDS

    for level in collect(keys(agent.subscription_accounts))
        # Skip advanced/premium during credit line period
        if credit_line > 0 && current_round < credit_line && level in ("advanced", "premium")
            continue
        end
        total += charge_subscription_installment!(agent, level)
    end

    agent.last_subscription_charge = total
    return total
end

"""
Ensure a subscription schedule exists for the given AI level (matches Python _ensure_subscription_schedule).
Called when AI is actually used, not just selected.
"""
function ensure_subscription_schedule!(agent::EmergentAgent, ai_level::String)
    if ai_level == "none"
        return
    end

    ai_config = get(agent.config.AI_LEVELS, ai_level, nothing)
    if isnothing(ai_config) || ai_config.cost <= 0
        return
    end

    # Only start if no active subscription exists
    if get(agent.subscription_accounts, ai_level, 0) <= 0
        start_subscription_schedule!(agent, ai_level)
        grace_rounds = compute_subscription_deferral_rounds(agent, ai_level)
        if grace_rounds > 0
            agent.subscription_deferral_remaining[ai_level] = grace_rounds
        end
    end
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

    # Neutral priors - agents learn from experience (matches Python/DEFAULT_TIER_PRIORS)
    prior_map = Dict(
        "none" => (2.0, 2.0),      # Prior mean = 0.50 (neutral)
        "basic" => (2.0, 2.0),     # Prior mean = 0.50 (neutral)
        "advanced" => (2.0, 2.0),  # Prior mean = 0.50 (neutral)
        "premium" => (2.0, 2.0)    # Prior mean = 0.50 (neutral)
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
    amort_horizon = max(1, agent.config.AI_SUBSCRIPTION_AMORTIZATION_ROUNDS)
    ref_scale = max(operating_cost * 4.0, cash_buffer * 0.12, 1.0)

    # Scale costs by AI_COST_INTENSITY (for robustness testing)
    cost_intensity = agent.config.AI_COST_INTENSITY

    cost_ratios = Dict{String,Float64}()
    for tier in order
        if tier == "none"
            cost_ratios[tier] = 0.0
            continue
        end
        cfg = get(agent.config.AI_LEVELS, tier, nothing)
        if isnothing(cfg)
            cost_ratios[tier] = 0.0
            continue
        end
        cost_type = cfg.cost_type
        base_cost = cfg.cost * cost_intensity
        per_use_cost = cfg.per_use_cost * cost_intensity

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

    # Extract performance signals (matches Python _choose_ai_level)
    base_roi_gain = Float64(get(metrics, "recent_roi_gain", 0.0))
    long_roi_gain = Float64(get(metrics, "roi_gain", 0.0))
    recent_roi_ratio = Float64(get(metrics, "recent_roi_gain_ratio", 0.0))
    roi_gain_ratio = Float64(get(metrics, "roi_gain_ratio", 0.0))
    net_cash_total = Float64(get(metrics, "overall_cash_flow_total", 0.0))
    net_cash_recent = Float64(get(metrics, "recent_cash_flow_total", 0.0))
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

        # ROI signal (matches Python exactly)
        roi_signal = (
            0.3 * base_roi_gain +
            0.4 * long_roi_gain +
            0.45 * recent_roi_ratio +
            0.25 * roi_gain_ratio
        )
        roi_term = posterior * roi_signal

        # Cash term includes both total and recent (matches Python)
        cash_term = (
            0.2 * clamp(net_cash_total / initial_equity, -1.5, 1.5) +
            0.1 * clamp(net_cash_recent / initial_equity, -1.5, 1.5)
        )

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

        # REMOVED: Tier-specific paradox response
        # Previously: premium/advanced penalized by paradox, none rewarded
        # Now agents should learn from their own experience with paradox conditions
        # The paradox effect should emerge from actual outcomes, not hardcoded tier rules
        paradox_term = 0.0

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
            paradox_term +
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
# HURDLE RATE AND COST OF CAPITAL (matches Python _calculate_agent_cost_of_capital)
# ============================================================================

"""
Estimate a hurdle rate tailored to the agent and current market stress.
Matches Python _calculate_agent_cost_of_capital exactly.
"""
function calculate_agent_cost_of_capital(
    agent::EmergentAgent,
    capital_utilization::Float64,
    market_conditions::Dict{String,Any}
)::Float64
    base_rate = agent.config.COST_OF_CAPITAL
    uncertainty_penalty = (0.5 - get(agent.traits, "uncertainty_tolerance", 0.5)) * 0.05
    competence_bonus = (get(agent.traits, "competence", 0.5) - 0.5) * 0.03
    market_volatility = Float64(get(market_conditions, "volatility", agent.config.MARKET_VOLATILITY))
    volatility_penalty = max(0.0, market_volatility - agent.config.MARKET_VOLATILITY) * 0.1
    liquidity_penalty = capital_utilization * 0.05
    cost = base_rate + uncertainty_penalty + volatility_penalty + liquidity_penalty - competence_bonus
    return Float64(max(0.02, cost))
end

"""
Estimate operational costs for the agent (matches Python _estimate_operational_costs).
Now accepts optional market parameter for competition calculation.
"""
function estimate_operational_costs(agent::EmergentAgent, market::Union{MarketEnvironment,Nothing}=nothing)::Float64
    # Portfolio competition pressure
    portfolio_competition = 0.0
    if !isempty(agent.active_investments)
        competitions = Float64[]
        for inv in agent.active_investments
            # Try to get opportunity from investment or from market
            opp = get(inv, "opportunity", nothing)
            if isnothing(opp) && !isnothing(market)
                opp_id = get(inv, "opportunity_id", nothing)
                if !isnothing(opp_id)
                    opp = get(market.opportunity_map, opp_id, nothing)
                end
            end
            if !isnothing(opp)
                push!(competitions, Float64(opp.competition))
            end
        end
        if !isempty(competitions)
            portfolio_competition = mean(competitions)
        end
    end

    base_cost = agent.config.BASE_OPERATIONAL_COST

    # Get sector operating cost range from SECTOR_PROFILES (matches Python)
    agent_sector = get(agent.traits, "sector", nothing)
    sector_profiles = agent.config.SECTOR_PROFILES
    sector_profile = get(sector_profiles, agent_sector, Dict{String,Any}())
    operating_range = get(sector_profile, "operating_cost_range", (0.05, 0.20))

    # Handle tuple/vector for operating range
    if isa(operating_range, Tuple) && length(operating_range) >= 2
        sector_mult = 0.5 * (operating_range[1] + operating_range[2])
    elseif isa(operating_range, Vector) && length(operating_range) >= 2
        sector_mult = 0.5 * (operating_range[1] + operating_range[2])
    else
        sector_mult = 0.125  # Default fallback
    end

    monthly_base_cost = base_cost * sector_mult
    competition_cost = portfolio_competition * agent.config.COMPETITION_COST_MULTIPLIER
    total_cost = monthly_base_cost + competition_cost

    # AI tier efficiency modifier (matches Python exactly)
    ai_level = get_ai_level(agent)
    ai_efficiency = Dict("none" => 1.0, "basic" => 1.08, "advanced" => 0.94, "premium" => 0.88)
    eff_mult = get(ai_efficiency, ai_level, 1.0)

    total_cost *= eff_mult
    return Float64(total_cost)
end

# ============================================================================
# UTILITY CALCULATIONS
# ============================================================================

"""
Calculate investment utility given opportunities and perception (matches Python _calculate_investment_utility).
"""
function calculate_investment_utility(
    agent::EmergentAgent,
    opportunities::Vector{Opportunity},
    market_conditions::Dict{String,Any},
    perception::Dict{String,Any};
    ai_level::String = "none",
    info_system::Union{InformationSystem,Nothing} = nothing
)::Float64
    if isempty(opportunities)
        return 0.0
    end

    # Get best opportunity score (matches Python: evaluate + apply_uncertainty_adjustments)
    max_score = 0.0
    for opp in opportunities
        # Use InformationSystem if available (matches Python _evaluate_portfolio_opportunities)
        if !isnothing(info_system)
            info = get_information(info_system, opp, ai_level; agent_id=agent.id, rng=agent.rng)
            base_score = evaluate_opportunity_with_info(agent, opp, info, market_conditions)
        else
            base_score = evaluate_opportunity_basic(agent, opp, market_conditions)
        end
        # Apply uncertainty adjustments like Python does
        adjusted_score = apply_uncertainty_adjustments(agent, base_score, opp, perception)
        max_score = max(max_score, adjusted_score)
    end

    scaled_score = stable_sigmoid(max_score - 1.0)

    # Extract signals (matches Python signal extraction)
    knowledge_signal = get(perception, "knowledge_signal", get(perception, "actor_ignorance", Dict{String,Any}()))
    innovation_signal = get(perception, "innovation_signal", get(perception, "agentic_novelty", Dict{String,Any}()))
    competition_signal = get(perception, "competition_signal", get(perception, "competitive_recursion", Dict{String,Any}()))

    # Extract uncertainty levels
    actor_unc = Float64(get(get(perception, "actor_ignorance", Dict()), "level", 0.5))
    agentic_unc = Float64(get(get(perception, "agentic_novelty", Dict()), "level", 0.5))
    recursive_unc = Float64(get(get(perception, "competitive_recursion", Dict()), "level", 0.5))
    decision_conf = clamp(Float64(get(perception, "decision_confidence", 0.5)), 0.1, 0.95)

    # Ignorance adjustment with confidence and gap_pressure (matches Python lines 1704-1710)
    ignorance_level = Float64(get(knowledge_signal, "ignorance_level", get(knowledge_signal, "level", 0.5)))
    confidence = Float64(get(knowledge_signal, "confidence", 0.5))
    gap_pressure = Float64(get(knowledge_signal, "gap_pressure", 0.0))
    clamped_ignorance = clamp(ignorance_level - 1.5, -20.0, 20.0)
    ignorance_adjustment = isfinite(clamped_ignorance) ? 0.5 + 0.5 / (1.0 + exp(clamped_ignorance)) : 1.0
    ignorance_adjustment *= clamp(0.85 + 0.35 * confidence, 0.5, 1.4)
    ignorance_adjustment += 0.1 * max(0.0, 0.35 - gap_pressure)

    # Capital metrics
    capital_ratio = get_capital(agent) / max(agent.resources.performance.initial_equity, 1.0)
    liquidity_boost = clamp(capital_ratio - 0.8, 0.0, 1.5)
    opportunity_boost = clamp(length(opportunities) / 6.0, 0.0, 1.0)

    value = scaled_score * ignorance_adjustment * decision_conf

    # Performance adjustments
    perf = agent.resources.performance
    invest_roic = compute_roic(perf, "invest")
    innovate_roic = compute_roic(perf, "innovate")

    # Locked/idle capital
    locked_capital = max(0.0, agent.locked_capital)
    idle_capital = max(0.0, get_capital(agent) - locked_capital)
    idle_ratio = idle_capital / max(agent.resources.performance.initial_equity, 1.0)

    # Innovation loss pressure (matches Python line 1728-1739)
    innovation_loss_pressure = max(0.0, -innovate_roic)

    value *= 0.7 + 0.3 * liquidity_boost
    value += 0.2 * opportunity_boost

    roic_multiplier = clamp(1.0 + 0.35 * invest_roic, 0.4, 1.4)
    value *= roic_multiplier

    value += max(0.0, idle_ratio) * 0.25
    value += innovation_loss_pressure * 0.2  # Was missing

    # Locked ratio penalty (matches Python lines 1740-1742)
    locked_ratio = locked_capital / max(1.0, get_capital(agent) + locked_capital)
    value -= locked_ratio * 0.25

    # AI ROI gain (matches Python lines 1729-1746)
    # Note: last_ai_tier_metrics comes from choose_ai_level which stores it in last_outcome
    ai_tier_metrics = get(agent.last_outcome, "ai_tier_metrics", Dict{String,Any}())
    ai_roi_gain = Float64(get(ai_tier_metrics, "roi_gain", 0.0))
    value += 0.3 * clamp(ai_roi_gain, -0.5, 1.0)
    peer_roi_signal = get(ai_tier_metrics, "peer_roi_signal_logged", nothing)
    if !isnothing(peer_roi_signal)
        value += 0.15 * clamp(Float64(peer_roi_signal), -0.5, 1.0)
    end

    # Empty portfolio bonus (matches Python lines 1747-1748)
    if isempty(agent.active_investments)
        value += 0.15
    end

    # Tier combo/reuse effects (matches Python lines 1749-1752)
    tier_combo = Float64(get(innovation_signal, "combo_rate", get(innovation_signal, "tier_combo_rate", 0.0)))
    tier_reuse = Float64(get(innovation_signal, "reuse_pressure", get(innovation_signal, "tier_reuse_pressure", 0.0)))
    tier_new_rate = Float64(get(innovation_signal, "new_possibility_rate", get(innovation_signal, "tier_new_possibility_rate", 0.0)))
    value += 0.15 * (tier_combo - tier_reuse) + 0.05 * tier_new_rate

    # Uncertainty hooks
    value += 0.15 * agentic_unc
    value -= 0.06 * recursive_unc

    # Risk tolerance
    risk_tolerance = Float64(get(agent.traits, "uncertainty_tolerance", 0.5))
    avoidance = 1.0 - risk_tolerance
    value *= clamp(0.85 + 0.3 * risk_tolerance, 0.5, 1.4)
    value -= avoidance * 0.12

    # REMOVED: Tier-specific utility biases
    # Previously: premium/advanced penalized for tier_reuse, basic/none rewarded for low combo
    # Now tier effects emerge from information quality affecting evaluations, not direct utility mods

    # Paradox signal effect (matches Python implementation)
    paradox_signal = agent.paradox_signal
    if paradox_signal != 0.0
        trust = Float64(get(agent.traits, "ai_trust", 0.5))
        tolerance = Float64(get(agent.traits, "uncertainty_tolerance", 0.5))
        if trust >= 0.6 || tolerance >= 0.6
            value += 0.25 * paradox_signal
        else
            value -= 0.35 * paradox_signal
        end
    end

    # Recent action penalties (matches Python lines 1774-1785)
    recent_actions = agent.recent_actions
    if !isempty(recent_actions)
        consecutive_invests = 0
        for act in reverse(recent_actions)
            if act == "invest"
                consecutive_invests += 1
            else
                break
            end
        end
        if consecutive_invests > 0
            value -= 0.12 * consecutive_invests
        end
        invest_share = count(==("invest"), recent_actions) / length(recent_actions)
        value -= invest_share * 0.15
    end

    # Low idle ratio penalty (matches Python lines 1786-1787)
    if idle_ratio < 0.05
        value -= 0.12
    end

    # Portfolio size penalty (matches Python lines 1788-1789)
    if length(agent.active_investments) >= 4
        value -= 0.07 * (length(agent.active_investments) - 3)
    end

    return clamp(value, 0.0, 1.0)
end

"""
Calculate innovation utility (matches Python _calculate_innovation_utility exactly).
"""
function calculate_innovation_utility(
    agent::EmergentAgent,
    market_conditions::Dict{String,Any},
    perception::Dict{String,Any}
)::Float64
    base_drive = get(agent.traits, "innovativeness", 0.5) * 0.5 + 0.05

    # Extract signals (matches Python lines 1796-1797)
    execution_signal = get(perception, "execution_risk", get(perception, "practical_indeterminism", Dict{String,Any}()))
    innovation_signal = get(perception, "innovation_signal", get(perception, "agentic_novelty", Dict{String,Any}()))

    # Indeterminism bonus (matches Python lines 1799-1800)
    indeterminism_level = Float64(get(execution_signal, "risk_level", get(execution_signal, "indeterminism_level", 0.5)))
    indeterminism_bonus = indeterminism_level * 0.4

    # Innovation opportunity and scarcity (matches Python lines 1802-1805)
    novelty_potential = Float64(get(innovation_signal, "novelty_potential", get(innovation_signal, "novelty_level", 0.5)))
    innovation_opportunity = novelty_potential * 0.25
    scarcity_signal = Float64(get(innovation_signal, "component_scarcity", 0.5))
    scarcity_drag = max(0.0, 0.6 - scarcity_signal) * 0.8

    # Capital constraints (matches Python lines 1807-1808)
    capital_ratio = get_capital(agent) / max(agent.resources.performance.initial_equity, 1.0)
    liquidity_penalty = clamp(1.0 - capital_ratio, 0.0, 1.5)

    # R&D burden and ROIC (matches Python lines 1809-1819)
    rd_deployed = get(agent.resources.performance.deployed_by_action, "innovate", 0.0)
    rd_burden = clamp(rd_deployed / max(agent.resources.performance.initial_equity, 1.0), 0.0, 2.0)

    perf = agent.resources.performance
    innovate_roic = compute_roic(perf, "innovate")

    # Cash burn calculation (matches Python lines 1813-1819)
    if rd_deployed > 0
        net_flow_innov = get(perf.returned_by_action, "innovate", 0.0) - rd_deployed
        loss_ratio = max(0.0, -innovate_roic)
        cash_burn_ratio = max(0.0, -net_flow_innov / rd_deployed)
    else
        loss_ratio = 0.0
        cash_burn_ratio = 0.0
    end

    # Net flow penalty (matches Python lines 1821-1822)
    net_flow = get(perf.returned_by_action, "innovate", 0.0) - get(perf.deployed_by_action, "innovate", 0.0)
    net_flow_penalty = clamp(-net_flow / max(1.0, get(perf.deployed_by_action, "innovate", 1.0)), 0.0, 1.0)

    risk_tolerance = Float64(get(agent.traits, "uncertainty_tolerance", 0.5))
    avoidance = 1.0 - risk_tolerance

    # Raw score calculation (matches Python lines 1827-1839)
    raw_score = (
        base_drive +
        indeterminism_bonus +
        innovation_opportunity -
        0.25 * liquidity_penalty -
        0.3 * rd_burden -
        scarcity_drag -
        0.4 * loss_ratio -
        0.3 * cash_burn_ratio +
        0.2 * net_flow_penalty +
        0.2 * (risk_tolerance - 0.5) -
        0.15 * avoidance
    )

    # Paradox signal effect (matches Python lines 1840-1845)
    paradox_signal = agent.paradox_signal
    if paradox_signal != 0.0
        pivot_weight = max(0.0, 0.6 - risk_tolerance)
        surge_weight = max(0.0, risk_tolerance - 0.4)
        raw_score += 0.2 * pivot_weight * paradox_signal
        raw_score -= 0.15 * surge_weight * paradox_signal
    end

    return clamp(stable_sigmoid(raw_score - 0.6), 0.02, 0.98)
end

"""
Calculate exploration utility (matches Python _calculate_exploration_utility exactly).
"""
function calculate_exploration_utility(
    agent::EmergentAgent,
    perception::Dict{String,Any};
    ai_level::String = "none"
)::Float64
    base_tendency = get(agent.traits, "exploration_tendency", 0.3)

    # Extract signals (matches Python signal extraction)
    knowledge_signal = get(perception, "knowledge_signal", get(perception, "actor_ignorance", Dict{String,Any}()))
    execution_signal = get(perception, "execution_risk", get(perception, "practical_indeterminism", Dict{String,Any}()))
    innovation_signal = get(perception, "innovation_signal", get(perception, "agentic_novelty", Dict{String,Any}()))
    competition_signal = get(perception, "competition_signal", get(perception, "competitive_recursion", Dict{String,Any}()))

    # Ignorance drive with gap_pressure (matches Python line 1859)
    ignorance_level = Float64(get(knowledge_signal, "ignorance_level", get(knowledge_signal, "level", 0.5)))
    gap_pressure = Float64(get(knowledge_signal, "gap_pressure", 0.0))
    ignorance_drive = ignorance_level * 0.35 + gap_pressure * 0.25

    # Knowledge deficit
    avg_knowledge = fast_mean(values(agent.resources.knowledge))
    if !isfinite(avg_knowledge)
        avg_knowledge = 0.0
    end
    knowledge_deficit = clamp(1.0 - avg_knowledge, 0.0, 1.0)

    # Novelty and scarcity from innovation signal (matches Python lines 1880-1883)
    novelty_potential = Float64(get(innovation_signal, "novelty_potential", get(innovation_signal, "novelty_level", 0.5)))
    novelty_gap = max(0.0, 0.5 - novelty_potential)
    component_scarcity = Float64(get(innovation_signal, "component_scarcity", 0.5))
    scarcity_push = max(0.0, 0.65 - component_scarcity)

    # Practical volatility from execution signal (matches Python line 1884)
    practical_volatility = Float64(get(execution_signal, "volatility", 0.2))

    # Capital metrics (matches Python lines 1886-1894)
    locked_capital = max(0.0, agent.locked_capital)
    capital_slack = max(0.0, (get_capital(agent) - locked_capital) / max(agent.resources.performance.initial_equity, 1.0))
    capital_ratio = get_capital(agent) / max(agent.resources.performance.initial_equity, 1.0)

    rich_knowledge_penalty = clamp(avg_knowledge - 0.25, 0.0, 1.0)
    buffer_penalty = clamp(capital_ratio - 1.3, 0.0, 1.0)
    idle_fraction = max(0.0, capital_slack - 0.1)

    # Trust penalty
    ai_trust_level = Float64(get(agent.traits, "ai_trust", 0.5))
    trust_penalty = max(0.0, ai_trust_level - 0.5) * 0.45

    # Recursion penalty with full competitive dynamics (matches Python lines 1898-1907)
    recursion_level = Float64(get(competition_signal, "pressure_level", get(competition_signal, "recursion_level", 0.5)))
    recursion_penalty = max(0.0, recursion_level - 0.5) * 0.2
    ai_usage_share = Float64(get(competition_signal, "ai_usage_share", 0.0))
    capital_crowding = Float64(get(competition_signal, "capital_crowding", 0.25))
    recursion_penalty += 0.08 * max(0.0, ai_usage_share - 0.45)
    recursion_penalty += 0.1 * max(0.0, capital_crowding - 0.4)
    # REMOVED: Tier-specific recursion penalty adjustments
    # Previously: premium/advanced penalized more when AI usage high, basic/none rewarded
    # Now tier effects emerge from information quality, not hardcoded utility modifications

    # Momentum bonus from explore ROIC
    explore_roic = compute_roic(agent.resources.performance, "explore")
    momentum_bonus = isfinite(explore_roic) ? clamp(explore_roic, -0.5, 0.5) * 0.2 : 0.0

    risk_tolerance = Float64(get(agent.traits, "uncertainty_tolerance", 0.5))
    avoidance = 1.0 - risk_tolerance

    # Recent explore penalty and invest streak (matches Python logic)
    recent_explore_penalty = 0.0
    invest_streak = 0
    invest_bias = 0.0
    history = agent.recent_actions
    if !isempty(history)
        if history[end] == "explore"
            recent_explore_penalty += 0.15
        end
        if length(history) >= 3
            streak = count(==("explore"), history)
            if streak >= length(history) - 1
                recent_explore_penalty += 0.25
            end
        end
        # Count consecutive invests from the end (Python logic)
        for act in reverse(history)
            if act == "invest"
                invest_streak += 1
            else
                break
            end
        end
        invest_bias = count(==("invest"), history) / length(history)
    end

    # Raw score with ALL Python terms (matches Python lines 1936-1955)
    raw_score = (
        base_tendency * 0.35 +
        ignorance_drive +
        knowledge_deficit * 0.25 +
        0.2 * novelty_gap +
        0.15 * scarcity_push +          # Was missing
        0.08 * capital_slack -
        0.25 * idle_fraction +          # Was missing
        0.12 * practical_volatility +   # Was missing
        momentum_bonus -
        0.35 * rich_knowledge_penalty -
        0.2 * buffer_penalty -          # Was missing
        trust_penalty -
        recursion_penalty +
        0.18 * avoidance -
        0.08 * risk_tolerance +
        0.15 * invest_streak +
        0.08 * invest_bias -
        recent_explore_penalty
    )

    # Add recursion level effect (Python line 1956)
    raw_score += 0.18 * recursion_level

    # Paradox signal effect (matches Python implementation)
    paradox_signal = agent.paradox_signal
    if paradox_signal != 0.0
        pivot_weight = max(0.0, 0.6 - risk_tolerance)
        raw_score += 0.2 * pivot_weight * paradox_signal
    end

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

    # Diversification (entropy-based, matching Python portfolio.diversification_score)
    sector_amounts = Dict{String,Float64}()
    total_amount = 0.0
    for inv in agent.active_investments
        sector = get(inv, "sector", nothing)
        if isnothing(sector)
            opp = get(inv, "opportunity", nothing)
            if !isnothing(opp)
                sector = hasfield(typeof(opp), :sector) ? opp.sector : nothing
            end
        end
        if isnothing(sector)
            sector = "unknown"
        end
        amount = Float64(get(inv, "amount", 0.0))
        if amount > 0
            sector_amounts[sector] = get(sector_amounts, sector, 0.0) + amount
            total_amount += amount
        end
    end

    diversification = 0.0
    if total_amount > 0 && !isempty(sector_amounts)
        # Compute Shannon entropy normalized by max entropy (log(4) for 4 sectors)
        max_entropy = log(4)
        if max_entropy > 0
            entropy = 0.0
            for amt in values(sector_amounts)
                p = amt / total_amount
                if p > 0
                    entropy -= p * log(p + 1e-12)
                end
            end
            diversification = clamp(entropy / max_entropy, 0.0, 1.0)
        end
    end

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

    # Paradox signal effect (matches Python implementation)
    paradox_signal = agent.paradox_signal
    if paradox_signal != 0.0
        maintain_utility += 0.15 * max(0.0, avoidance) * paradox_signal
        maintain_utility -= 0.1 * max(0.0, -paradox_signal)
    end

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
    # Use latent_return_potential instead of expected_return
    expected_return = coalesce(opportunity.latent_return_potential, 1.0)
    expected_margin = expected_return - 1.0

    # Use complexity as proxy for uncertainty
    opportunity_uncertainty = coalesce(opportunity.complexity, 0.5)
    uncertainty_adjusted = expected_margin * (1.0 - opportunity_uncertainty * 0.5)

    score = uncertainty_adjusted * (1.0 + get(agent.traits, "uncertainty_tolerance", 0.5))

    # Regime multiplier
    regime = get(market_conditions, "regime", "normal")
    regime_mult = Dict("boom" => 1.2, "growth" => 1.1, "normal" => 1.0, "volatile" => 0.9, "crisis" => 0.7)
    score *= get(regime_mult, regime, 1.0)

    # Sector knowledge
    sector = coalesce(opportunity.sector, "unknown")
    sector_knowledge = get(agent.resources.knowledge, sector, 0.1)
    score *= 1.0 + sector_knowledge * 0.5

    # Sector crowding penalty - discount opportunities in crowded sectors
    # This encourages diversification across sectors
    clearing_index = get(market_conditions, "sector_clearing_index", Dict{String,Float64}())
    sector_crowding = Float64(get(clearing_index, sector, 0.0))
    if sector_crowding > 1.0
        # Apply penalty: score * (1 / (1 + 0.3 * excess_crowding))
        # At crowding=2 (1 excess): multiplier = 0.77
        # At crowding=4 (3 excess): multiplier = 0.53
        crowding_penalty = 1.0 / (1.0 + 0.3 * (sector_crowding - 1.0))
        score *= crowding_penalty
    end

    return max(0.1, score)
end

"""
Evaluate opportunity using InformationSystem (matches Python _evaluate_opportunity).
Uses info.estimated_return instead of raw latent_return_potential.
"""
function evaluate_opportunity_with_info(
    agent::EmergentAgent,
    opportunity::Opportunity,
    info::Information,
    market_conditions::Dict{String,Any}
)::Float64
    # Use estimated_return from info (matches Python line 2420)
    expected_profit_margin = info.estimated_return - 1.0
    uncertainty_adjusted_margin = expected_profit_margin * (1.0 - info.estimated_uncertainty * 0.5)
    score = uncertainty_adjusted_margin * (1.0 + get(agent.traits, "uncertainty_tolerance", 0.5))

    # Apply market regime modifier
    regime = get(market_conditions, "regime", "normal")
    regime_mult = Dict("boom" => 1.2, "growth" => 1.1, "normal" => 1.0, "volatile" => 0.9, "crisis" => 0.7)
    score *= get(regime_mult, regime, 1.0)

    # Apply sector knowledge bonus
    sector = coalesce(opportunity.sector, "unknown")
    sector_knowledge = get(agent.resources.knowledge, sector, 0.1)
    score *= 1.0 + sector_knowledge * 0.5

    # --- Logic for Qualitative Insights (matches Python lines 2432-2469) ---
    perceived_complexity = coalesce(opportunity.complexity, 0.5)
    perceived_uncertainty = info.estimated_uncertainty
    perceived_return = info.estimated_return
    analytical_ability = get(agent.traits, "analytical_ability", 0.5)

    for insight in info.insights
        trait_multiplier = 1.0 + (analytical_ability - 0.5) * 0.2

        if insight == "Requires specialized capabilities"
            perceived_complexity *= 1.25 * trait_multiplier
        elseif insight == "Strong first-mover advantages"
            score *= 1.1 * trait_multiplier
        elseif insight == "High competitive pressure detected"
            perceived_uncertainty *= 1.15 * (2.0 - trait_multiplier)
        elseif insight == "Untapped customer segment identified in emerging markets"
            perceived_return *= 1.2 * trait_multiplier
        end
    end

    # Final score adjustment based on perceptions (matches Python lines 2471-2477)
    complexity_penalty = 1.0 - (perceived_complexity * (1.0 - analytical_ability) * 0.1)
    score *= complexity_penalty

    # Scale by AI's confidence in its analysis
    # FIXED: Remove artificial floor that dampened confidence signal
    # Previously (0.5 + conf * 0.5) compressed 2.6x confidence diff to 1.44x
    # Now confidence has realistic impact on opportunity evaluation
    score *= 0.15 + info.confidence * 0.85

    return max(0.1, score)
end

"""
Apply uncertainty adjustments to opportunity score.
Matches Python _apply_uncertainty_adjustments with knowledge gaps and timing pressure.
"""
function apply_uncertainty_adjustments(
    agent::EmergentAgent,
    base_score::Float64,
    opportunity::Opportunity,
    perception::Dict{String,Any}
)::Float64
    adjusted = base_score

    # Extract perception signals (matches Python perception structure)
    knowledge_signal = get(perception, "knowledge_signal", get(perception, "actor_ignorance", Dict{String,Any}()))
    execution_signal = get(perception, "execution_risk", get(perception, "practical_indeterminism", Dict{String,Any}()))

    # Knowledge gaps adjustment (Python: if opportunity.id in knowledge_gaps)
    knowledge_gaps = get(knowledge_signal, "knowledge_gaps", Dict{String,Any}())
    if isa(knowledge_gaps, Dict)
        opp_id = opportunity.id
        if haskey(knowledge_gaps, opp_id)
            gap_value = Float64(get(knowledge_gaps, opp_id, 0.0))
            adjusted *= (1.0 - gap_value * 0.5)
        end
    end

    # Timing pressure adjustment (Python: if opportunity.id in timing_pressure)
    timing_pressure = get(execution_signal, "timing_pressure", Dict{String,Any}())
    if isa(timing_pressure, Dict)
        opp_id = opportunity.id
        if haskey(timing_pressure, opp_id)
            adjusted *= 0.8
        end
    end

    # Creator bonus (use created_by field) - matches Python exactly
    created_by = coalesce(opportunity.created_by, -1)
    if created_by != -1
        adjusted *= 1.1
        if created_by == agent.id
            adjusted *= 1.2
        end
    end

    # Social proof sensitivity - matches Python
    social_proof_sensitivity = 1.0 - get(agent.traits, "analytical_ability", 0.5)
    social_proof_bonus = 1.0 + (opportunity.competition * social_proof_sensitivity * 0.25)
    adjusted *= social_proof_bonus

    # Competition adjustment (Python: if competition > 0.5)
    if opportunity.competition > 0.5
        tolerance = get(agent.traits, "uncertainty_tolerance", 0.5)
        adjusted *= 1.0 - (tolerance * opportunity.competition * 0.5)
    end

    return max(0.01, adjusted)
end

"""
Evaluate portfolio opportunities and return ranked list.
Uses InformationSystem for proper AI analysis with domain-specific accuracy,
hallucination modeling, and per-use cost tracking.

Matches Python _evaluate_opportunities flow:
1. Call get_information() for each opportunity (or get_human_information for none)
2. Track per-use costs for AI tiers with per_use cost type
3. Use Information object to guide scoring
4. Store AI analysis metadata for outcome-based learning
"""
function evaluate_portfolio_opportunities(
    agent::EmergentAgent,
    opportunities::Vector{Opportunity},
    market_conditions::Dict{String,Any},
    perception::Dict{String,Any};
    ai_level::String = "none",
    info_system::Union{InformationSystem,Nothing} = nothing
)::Vector{Dict{String,Any}}
    if isempty(opportunities)
        return Dict{String,Any}[]
    end

    evaluations = Dict{String,Any}[]
    total_per_use_cost = 0.0

    # Get AI tier configuration
    ai_cfg = get(agent.config.AI_LEVELS, ai_level, agent.config.AI_LEVELS["none"])
    info_quality = Float64(ai_cfg.info_quality)
    cost_intensity = agent.config.AI_COST_INTENSITY

    # Per-use cost tracking (matches Python)
    per_use_cost_per_call = 0.0
    if ai_level != "none" && ai_cfg.cost_type == "per_use"
        per_use_cost_per_call = ai_cfg.per_use_cost * cost_intensity
    end

    # Fallback hallucination risk when no info_system
    fallback_hallucination_risk = max(0.0, 0.5 - info_quality * 0.5)

    # REMOVED: Tier-based pre-filtering that gave premium/advanced first look at best opportunities
    # Previously: premium saw top 65% sorted by quality, none saw random 60%
    # Now ALL agents evaluate ALL opportunities - better AI tiers will make better selections
    # through more accurate information quality, not through pre-filtered access
    # This lets selection quality emerge from the information system, not hardcoded filtering
    opp_pool = copy(opportunities)

    for opp in opp_pool
        # ========================================================================
        # GET AI INFORMATION (matches Python information.get_information flow)
        # ========================================================================
        info::Union{Information,Nothing} = nothing
        contains_hallucination = false
        actual_accuracy = 0.5
        ai_estimated_return = opp.latent_return_potential
        ai_estimated_uncertainty = opp.latent_failure_potential
        ai_confidence = 0.5
        analysis_domain = "uncertainty_evaluation"

        if !isnothing(info_system)
            # Use proper InformationSystem (matches Python)
            if ai_level == "none"
                info = get_human_information(info_system, opp, agent.traits; rng=agent.rng)
            else
                info = get_information(info_system, opp, ai_level; agent_id=agent.id, rng=agent.rng)
                # Track per-use cost for each evaluation
                total_per_use_cost += per_use_cost_per_call
            end

            # Extract information fields
            ai_estimated_return = info.estimated_return
            ai_estimated_uncertainty = info.estimated_uncertainty
            ai_confidence = info.confidence
            contains_hallucination = info.contains_hallucination
            actual_accuracy = info.actual_accuracy
            analysis_domain = info.domain
        end

        # Base score from opportunity characteristics
        base_score = evaluate_opportunity_basic(agent, opp, market_conditions)
        final_score = apply_uncertainty_adjustments(agent, base_score, opp, perception)

        # ========================================================================
        # INFORMATION-BASED ADJUSTMENTS
        # ========================================================================
        if !isnothing(info)
            # Use AI-estimated return to adjust score (matches Python)
            return_adjustment = (ai_estimated_return - 1.0) * ai_confidence * 0.5
            final_score += return_adjustment

            # Uncertainty penalty (higher AI-estimated uncertainty = lower score)
            uncertainty_penalty = ai_estimated_uncertainty * (1.0 - ai_confidence) * 0.3
            final_score -= uncertainty_penalty

            # Hallucination effect (when present, can inflate or deflate)
            if contains_hallucination
                analytical_ability = Float64(get(agent.traits, "analytical_ability", 0.5))
                susceptibility = 1.0 - analytical_ability * 0.5

                if rand(agent.rng) < susceptibility
                    # Hallucination modifier (biased optimistic, matches Python)
                    halluc_modifier = if rand(agent.rng) < 0.65
                        1.0 + rand(agent.rng) * 0.25
                    else
                        1.0 - rand(agent.rng) * 0.15
                    end
                    final_score *= halluc_modifier
                end
            end
        else
            # Fallback: use simplified hallucination modeling (no info_system)
            if ai_level != "none" && fallback_hallucination_risk > 0
                analytical_ability = Float64(get(agent.traits, "analytical_ability", 0.5))
                susceptibility = 1.0 - analytical_ability * 0.5

                if rand(agent.rng) < fallback_hallucination_risk * susceptibility
                    halluc_modifier = if rand(agent.rng) < 0.65
                        1.0 + rand(agent.rng) * 0.25
                    else
                        1.0 - rand(agent.rng) * 0.15
                    end
                    final_score *= halluc_modifier
                end

                hidden_uncertainty = fallback_hallucination_risk * (1.0 - info_quality) * opp.complexity
                final_score *= (1.0 - hidden_uncertainty * 0.2)
            end
        end

        final_score = max(0.01, final_score)

        # Build evaluation record with AI analysis metadata (for outcome tracking)
        eval_record = Dict{String,Any}(
            "opportunity" => opp,
            "final_score" => final_score,
            "ai_level_used" => ai_level,
            "ai_estimated_return" => ai_estimated_return,
            "ai_estimated_uncertainty" => ai_estimated_uncertainty,
            "ai_confidence" => ai_confidence,
            "ai_contains_hallucination" => contains_hallucination,
            "ai_actual_accuracy" => actual_accuracy,
            "ai_analysis_domain" => analysis_domain
        )

        # Store Information object for outcome-based learning (matches Python _source_analysis)
        if !isnothing(info)
            eval_record["_source_info"] = info
            # Compact AI info dict for investment outcomes (used at maturity)
            eval_record["ai_info"] = Dict{String,Any}(
                "actual_accuracy" => actual_accuracy,
                "contains_hallucination" => contains_hallucination,
                "domain" => analysis_domain,
                "estimated_return" => ai_estimated_return,
                "estimated_uncertainty" => ai_estimated_uncertainty,
                "confidence" => ai_confidence
            )
        end

        push!(evaluations, eval_record)
    end

    # Sort by score descending
    sort!(evaluations, by=e -> e["final_score"], rev=true)

    # Store total per-use cost in first evaluation (to be extracted by caller)
    if !isempty(evaluations)
        evaluations[1]["ai_per_use_cost"] = total_per_use_cost
    end

    return evaluations
end

"""
Make portfolio investment decision with capital constraints and position sizing.
Matches Python _make_portfolio_decision with:
- Operating reserve and liquidity buffer calculations
- Hurdle rate based on agent cost of capital
- Stress-based thresholds
- Confidence-scaled position sizing

Returns either an investment decision or a maintain decision.
"""
function make_portfolio_decision(
    agent::EmergentAgent,
    evaluated_opportunities::Vector{Dict{String,Any}},
    market::MarketEnvironment,
    round_num::Int,
    ai_level::String,
    perception::Dict{String,Any},
    market_conditions::Dict{String,Any}
)::Dict{String,Any}
    current_capital = max(0.0, get_capital(agent))
    locked_capital = max(0.0, agent.locked_capital)

    # Estimate operational costs
    operating_estimate = estimate_operational_costs(agent, market)

    # Operating reserve (matches Python OPERATING_RESERVE_MONTHS)
    reserve_months = Float64(getfield_default(agent.config, :OPERATING_RESERVE_MONTHS, 3.0))
    operating_reserve = operating_estimate * max(1.0, reserve_months)

    # AI reserve (subscription or per-use cost)
    ai_reserve = 0.0
    ai_cfg = get(agent.config.AI_LEVELS, ai_level, agent.config.AI_LEVELS["none"])
    cost_type = ai_cfg.cost_type
    if cost_type == "subscription"
        ai_reserve = Float64(ai_cfg.cost)
    elseif cost_type == "per_use"
        ai_reserve = Float64(ai_cfg.cost)
    end

    # Liquidity buffer calculation (matches Python)
    reserve_fraction = Float64(getfield_default(agent.config, :LIQUIDITY_RESERVE_FRACTION, 0.3))
    buffer_target = max(
        operating_reserve + ai_reserve + agent.config.SURVIVAL_THRESHOLD,
        current_capital * reserve_fraction
    )
    liquidity_buffer = min(current_capital, buffer_target) + locked_capital * 0.2

    # Check if locked capital exceeds free capital
    free_capital = current_capital - liquidity_buffer
    if free_capital <= locked_capital
        return _make_maintain_decision(agent, round_num, ai_level)
    end

    available_capital = max(0.0, free_capital - locked_capital)

    # Deduct per-use AI cost from available capital
    if cost_type == "per_use"
        available_capital = max(0.0, available_capital - Float64(ai_cfg.cost))
    end

    # Calculate stress and dynamic threshold (matches Python)
    total_commitments = current_capital + agent.locked_capital
    capital_utilization = agent.locked_capital / max(1.0, total_commitments)

    # Get volatility and regime failure multiplier
    volatility = Float64(get(market_conditions, "volatility", agent.config.MARKET_VOLATILITY))
    regime_failure = Float64(get(market_conditions, "regime_failure_multiplier", 1.0))

    # Risk aversion from traits
    risk_aversion = max(0.0, 1.0 - Float64(get(agent.traits, "uncertainty_tolerance", 0.5)))

    # Gap pressure from perception
    knowledge_signal = get(perception, "knowledge_signal", Dict{String,Any}())
    gap_pressure = Float64(get(knowledge_signal, "gap_pressure", 0.0))

    # Stress calculation (matches Python exactly)
    stress = clamp(
        0.4 * capital_utilization + 0.6 * volatility + 0.3 * (regime_failure - 1.0) + 0.4 * gap_pressure,
        0.0, 1.8
    )

    # Calculate decision threshold
    scores = [Float64(get(e, "final_score", 0.0)) for e in evaluated_opportunities]
    if !isempty(scores)
        base_q = 0.60
        q = clamp(base_q + 0.25 * stress + 0.1 * risk_aversion, 0.6, 0.98)
        baseline = quantile(scores, q)
    else
        baseline = 0.0
    end

    noise = baseline > 0 ? randn(agent.rng) * baseline * 0.18 : 0.0
    agent_cost_of_capital = calculate_agent_cost_of_capital(agent, capital_utilization, market_conditions)
    dynamic_threshold = (baseline * (0.85 + 0.4 * capital_utilization + 0.35 * stress)) + noise
    hurdle = agent_cost_of_capital * (1.0 + 0.6 * stress + 0.4 * risk_aversion)
    decision_threshold = max(hurdle, dynamic_threshold, 0.02)

    # Find best opportunity
    if isempty(evaluated_opportunities)
        return _make_maintain_decision(agent, round_num, ai_level)
    end

    best_eval = evaluated_opportunities[1]  # Already sorted
    best_score = Float64(get(best_eval, "final_score", 0.0))
    ai_info_payload = get(best_eval, "ai_info", nothing)

    # Check against threshold
    if best_score < decision_threshold
        return _make_maintain_decision(agent, round_num, ai_level)
    end

    # Get opportunity
    opp = get(best_eval, "opportunity", nothing)
    if isnothing(opp)
        return _make_maintain_decision(agent, round_num, ai_level)
    end

    # Calculate investment amount with confidence scaling (matches Python exactly)
    confidence = Float64(get(perception, "decision_confidence", 0.5))
    confidence = clamp(confidence, 0.15, 0.95)

    signal_score = max(0.0, best_score)
    hurdle_rate = agent_cost_of_capital
    if signal_score < hurdle_rate
        return _make_maintain_decision(agent, round_num, ai_level)
    end

    # Position sizing (matches Python)
    max_fraction = Float64(getfield_default(agent.config, :MAX_INVESTMENT_FRACTION, 0.1))
    target_fraction = Float64(getfield_default(agent.config, :TARGET_INVESTMENT_FRACTION, 0.07))
    max_investment = available_capital * max_fraction

    # KEY: Confidence-scaled position sizing (Python: desired_investment = available_capital * target_fraction * confidence * signal_score)
    desired_investment = available_capital * target_fraction * confidence * signal_score
    amount = min(desired_investment, max_investment)

    # Check minimum funding fraction
    required_capital = Float64(coalesce(opp.capital_requirements, 0.0))
    min_fraction = Float64(getfield_default(agent.config, :MIN_FUNDING_FRACTION, 0.25))
    min_required = required_capital > 0 ? required_capital * min_fraction : 0.0

    # Check minimum funding vs available capital (Python lines 2711-2712)
    if required_capital > 0 && amount < min_required && available_capital >= min_required
        amount = min(max_investment, min_required)
    end

    # Final caps (matches Python lines 2714-2715)
    amount = min(amount, available_capital)
    amount = max(0.0, amount)

    if amount <= 0
        return _make_maintain_decision(agent, round_num, ai_level)
    end

    # Return investment decision
    return Dict{String,Any}(
        "action" => "invest",
        "opportunity" => opp,
        "amount" => amount,
        "ai_level_used" => ai_level,
        "decision_confidence" => confidence,
        "signal_score" => signal_score,
        "hurdle_rate" => hurdle_rate,
        "decision_threshold" => decision_threshold,
        "stress" => stress,
        "capital_utilization" => capital_utilization,
        "ai_info" => ai_info_payload
    )
end

"""
Helper to create a maintain decision.
"""
function _make_maintain_decision(agent::EmergentAgent, round_num::Int, ai_level::String)::Dict{String,Any}
    return Dict{String,Any}(
        "action" => "maintain",
        "agent_id" => agent.id,
        "round" => round_num,
        "ai_level_used" => ai_level,
        "reason" => "threshold_or_capital_constraint"
    )
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
    ai_level_override::Union{String,Nothing} = nothing,
    innovation_engine::Union{InnovationEngine,Nothing} = nothing,
    info_system::Union{InformationSystem,Nothing} = nothing
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

    # Apply subscription charges at start of decision (matches Python make_decision)
    apply_subscription_carry!(agent, round_num)

    # Pre-decision insolvency check (matches Python make_decision lines 1085-1117)
    capital_before_action = Float64(agent.resources.capital)
    grace_period = max(1, agent.config.INSOLVENCY_GRACE_ROUNDS)
    initial_equity = max(agent.resources.performance.initial_equity, 1.0)
    capital_ratio = agent.resources.capital / initial_equity
    ratio_floor = agent.config.SURVIVAL_CAPITAL_RATIO
    below_floor = (agent.resources.capital < agent.config.SURVIVAL_THRESHOLD) || (capital_ratio < ratio_floor)

    if below_floor
        agent.insolvency_streak += 1
        if agent.insolvency_streak >= grace_period
            agent.alive = false
            agent.failure_reason = "bankruptcy"
            return Dict{String,Any}(
                "action" => "exit",
                "agent_id" => agent.id,
                "round" => round_num,
                "reason" => "bankruptcy",
                "ai_level_used" => "none",
                "portfolio_size" => length(agent.active_investments)
            )
        end
        # Restructuring round: force maintain decision without AI spend
        return Dict{String,Any}(
            "action" => "maintain",
            "agent_id" => agent.id,
            "round" => round_num,
            "reason" => "distress_restructuring",
            "ai_level_used" => "none",
            "ai_per_use_cost" => 0.0,
            "ai_call_count" => 0,
            "portfolio_size" => length(agent.active_investments),
            "capital_before_action" => capital_before_action,
            "capital_after_action" => Float64(agent.resources.capital)
        )
    else
        if agent.insolvency_streak > 0
            agent.insolvency_streak = 0
        end
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

    # Start subscription if using a paid AI tier (matches Python register_ai_usage)
    if ai_level != "none"
        ensure_subscription_schedule!(agent, ai_level)
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

    # Store perception for action execution (matches Python _last_perception)
    agent.last_perception = perception

    # Calculate utilities for each action
    # Match Python: use estimated operating costs + AI cost (not raw BASE_OPERATIONAL_COST)
    op_cost = estimate_operational_costs(agent, market)
    expected_calls = ai_level != "none" ? 1.0 : 0.0
    ai_cost_est = estimate_ai_cost(agent, ai_level; expected_calls=expected_calls)
    estimated_cost = op_cost + ai_cost_est

    invest_utility = calculate_investment_utility(agent, opportunities, market_conditions, perception; ai_level=ai_level, info_system=info_system)
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

    # Apply locked capital penalty to invest utility (matches Python)
    locked_capital = max(0.0, agent.locked_capital)
    locked_ratio = locked_capital / max(1.0, get_capital(agent) + locked_capital)
    utilities["invest"] -= locked_ratio * 0.25

    # Get AI profile for temperature annealing (matches Python)
    ai_config = get(agent.config.AI_LEVELS, ai_level, agent.config.AI_LEVELS["none"])
    info_quality = ai_config.info_quality
    info_breadth = ai_config.info_breadth
    trait_trust = clamp(get(agent.traits, "ai_trust", 0.5), 0.0, 1.0)
    trait_explore = clamp(get(agent.traits, "exploration_tendency", 0.5), 0.0, 1.0)

    # Temperature annealing (matches Python softmax with temperature scaling)
    actions = ["invest", "innovate", "explore", "maintain"]
    utils = [utilities[a] for a in actions]
    util_variance = length(utils) > 0 ? std(utils) : 0.0

    base_temperature = max(1e-3, agent.config.ACTION_SELECTION_TEMPERATURE)
    temperature = max(5e-4, base_temperature * (1.0 + 0.6 * clamp(util_variance, 0.0, 1.0)))
    temp_scale = max(0.3, 1.0 - 0.6 * info_quality)
    temp_scale /= max(0.5, 0.9 + 0.2 * info_breadth)
    temperature *= temp_scale
    temperature *= clamp(0.8 + 0.5 * (1.0 - trait_trust), 0.6, 1.4)

    # Noise scaling by AI quality (matches Python Gumbel noise approach)
    noise_base = max(0.0, agent.config.ACTION_SELECTION_NOISE)
    noise_scale = noise_base * (1.15 - 0.8 * info_quality) * (1.0 + 0.25 * (1.0 - info_breadth))
    noise_scale = max(0.0, noise_scale)
    noise_scale *= clamp(0.85 + 0.4 * trait_explore, 0.7, 1.6)

    # Get action bias and add Gumbel noise
    bias_vec = [get(agent.action_bias, a, 0.0) for a in actions]
    noise_vec = noise_scale > 0 ? [rand(agent.rng, Gumbel(0.0, noise_scale)) for _ in actions] : zeros(4)

    # Compute logits with bias and noise
    logits = [(utils[i] + bias_vec[i] + noise_vec[i]) / temperature for i in 1:4]
    logits = [isfinite(l) ? l : -Inf for l in logits]
    max_logit = maximum(logits)
    logits = [l - max_logit for l in logits]
    exp_logits = [exp(l) for l in logits]
    sum_exp = sum(exp_logits)

    if !isfinite(sum_exp) || sum_exp <= 0
        probs = Dict("maintain" => 1.0, "invest" => 0.0, "innovate" => 0.0, "explore" => 0.0)
    else
        probs = Dict(actions[i] => exp_logits[i] / sum_exp for i in 1:4)
    end

    # Sample action
    r = rand(agent.rng)
    cumsum_prob = 0.0
    chosen_action = "maintain"
    for action in actions
        cumsum_prob += get(probs, action, 0.0)
        if r <= cumsum_prob
            chosen_action = action
            break
        end
    end

    # Update action bias to keep heterogeneity dynamic (matches Python)
    mean_util = mean(utils)
    chosen_idx = findfirst(==(chosen_action), actions)
    if !isnothing(chosen_idx)
        old_bias = get(agent.action_bias, chosen_action, 0.0)
        new_bias = 0.95 * old_bias + 0.05 * (utils[chosen_idx] - mean_util)
        agent.action_bias[chosen_action] = clamp(new_bias, -0.5, 0.5)
    end

    # Update recent actions (rolling window of 20)
    push!(agent.recent_actions, chosen_action)
    while length(agent.recent_actions) > 20
        popfirst!(agent.recent_actions)
    end

    # Track AI per-use cost
    ai_per_use_cost_tracked = Ref(0.0)

    # Execute chosen action
    outcome = if chosen_action == "invest" && !isempty(opportunities)
        # Use full portfolio decision with capital constraints and position sizing
        evals = evaluate_portfolio_opportunities(agent, opportunities, market_conditions, perception;
                                                 ai_level=ai_level, info_system=info_system)
        # Extract per-use cost from evaluations (stored in first eval by evaluate_portfolio_opportunities)
        ai_per_use_cost_tracked[] = !isempty(evals) ? Float64(get(evals[1], "ai_per_use_cost", 0.0)) : 0.0

            if !isempty(evals)
                # Use make_portfolio_decision for hurdle rate and confidence-scaled sizing
                portfolio_decision = make_portfolio_decision(
                    agent, evals, market, round_num, ai_level, perception, market_conditions
                )

            if get(portfolio_decision, "action", "maintain") == "invest"
                # Execute with the computed investment amount
                opp = portfolio_decision["opportunity"]
                sized_amount = Float64(get(portfolio_decision, "amount", 0.0))
                ai_info_payload = get(portfolio_decision, "ai_info", nothing)
                execute_action!(agent, "invest", market, round_num;
                               opportunity=opp,
                               investment_amount=sized_amount,
                               ai_info=ai_info_payload,
                               innovation_engine=innovation_engine,
                               market_conditions=market_conditions)
            else
                # Portfolio decision returned maintain (threshold/capital constraint)
                execute_action!(agent, "maintain", market, round_num;
                               innovation_engine=innovation_engine,
                               market_conditions=market_conditions)
            end
        else
            execute_action!(agent, "maintain", market, round_num;
                           innovation_engine=innovation_engine,
                           market_conditions=market_conditions)
        end
    else
        execute_action!(agent, chosen_action, market, round_num;
                       innovation_engine=innovation_engine,
                       market_conditions=market_conditions)
    end

    outcome["ai_level_used"] = ai_level
    outcome["utilities"] = utilities
    outcome["perception"] = perception

    # Add per-use cost tracking if we evaluated with AI (matches Python ai_per_use_cost)
    if ai_per_use_cost_tracked[] > 0
        outcome["ai_per_use_cost"] = ai_per_use_cost_tracked[]
        # Deduct per-use cost from capital (matches Python simulation phase 1)
        agent.resources.capital = max(0.0, agent.resources.capital - ai_per_use_cost_tracked[])
    end

    # Update burn history and evaluate failure conditions (matches Python lines 1216-1220)
    capital_after_action = Float64(agent.resources.capital)
    update_burn_history!(agent, capital_after_action - capital_before_action)
    outcome["capital_before_action"] = capital_before_action
    outcome["capital_after_action"] = capital_after_action

    failure_reason = evaluate_failure_conditions!(agent, capital_after_action)
    if !isnothing(failure_reason)
        agent.alive = false
        agent.failure_reason = failure_reason
        # Return early with exit action (matches Python lines 1221-1230)
        return Dict{String,Any}(
            "action" => "exit",
            "agent_id" => agent.id,
            "reason" => "failure_$failure_reason",
            "ai_level_used" => ai_level,
            "capital_before_action" => capital_before_action,
            "capital_after_action" => capital_after_action
        )
    end

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

    # Update paradox signal with inertia (matches Python implementation)
    inertia = 0.85
    old_signal = agent.paradox_signal
    new_signal = inertia * old_signal + (1.0 - inertia) * gap * weight
    agent.paradox_signal = clamp(new_signal, -1.0, 1.0)
end

# ============================================================================
# OPERATIONAL COSTS
# ============================================================================
# NOTE: estimate_operational_costs is defined earlier (lines ~1711) with
# sector-aware implementation. That single definition handles both cases
# (with or without market parameter) via Union type.
