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
# AGENT UNCERTAINTY METRICS (Emergent, Agent-Level)
# ============================================================================

"""
Tracks agent-level uncertainty metrics that EMERGE from actual outcomes.
These replace the formula-based environment-level metrics.

The four Knightian dimensions are computed from what actually happens to agents:
- Actor Ignorance: How wrong were the agent's estimates? (estimation error)
- Practical Indeterminism: How variable were outcomes? (return variance)
- Agentic Novelty: How much new stuff did the agent create? (creative output)
- Competitive Recursion: How crowded were investments? (competition experienced)
"""
mutable struct AgentUncertaintyMetrics
    # Actor Ignorance - tracks estimation accuracy
    # |estimated_return - actual_return| / max(0.01, |actual_return|)
    estimation_errors::Vector{Float64}

    # Practical Indeterminism - tracks outcome variance
    # Realized return multiples from investments
    return_history::Vector{Float64}

    # Agentic Novelty - tracks creative output
    new_combinations_created::Int
    niches_discovered::Int
    derivative_adoptions::Int  # Following existing patterns vs creating
    total_actions::Int

    # Competitive Recursion - tracks crowding experienced
    # Competition levels on opportunities when invested
    competition_levels::Vector{Float64}

    # Round tracking for temporal analysis
    round_metrics::Vector{Dict{String,Float64}}

    # Knowledge Recombination Metrics - tracks innovation quality from knowledge system
    innovation_qualities::Vector{Float64}    # Quality score (0-1) of each innovation
    innovation_novelties::Vector{Float64}    # Novelty score (0-1) of each innovation
    innovation_scarcities::Vector{Float64}   # Scarcity bonus (0-1) of each innovation
end

function AgentUncertaintyMetrics()
    return AgentUncertaintyMetrics(
        Float64[],  # estimation_errors
        Float64[],  # return_history
        0,          # new_combinations_created
        0,          # niches_discovered
        0,          # derivative_adoptions
        0,          # total_actions
        Float64[],  # competition_levels
        Dict{String,Float64}[],  # round_metrics
        Float64[],  # innovation_qualities
        Float64[],  # innovation_novelties
        Float64[]   # innovation_scarcities
    )
end

"""
Compute the four uncertainty dimensions from tracked metrics.
Returns values in [0, 1] range for comparison.
"""
function compute_emergent_uncertainty(metrics::AgentUncertaintyMetrics)::Dict{String,Float64}
    # Actor Ignorance: mean estimation error (0 = perfect, 1 = very wrong)
    actor_ignorance = if isempty(metrics.estimation_errors)
        0.5  # No data, neutral
    else
        clamp(mean(metrics.estimation_errors), 0.0, 1.0)
    end

    # Practical Indeterminism: coefficient of variation of returns
    practical_indeterminism = if length(metrics.return_history) < 2
        0.5  # Not enough data
    else
        μ = mean(metrics.return_history)
        σ = std(metrics.return_history)
        # CV normalized to [0, 1] - higher variance = higher indeterminism
        clamp(σ / max(0.1, abs(μ)), 0.0, 1.0)
    end

    # Agentic Novelty: ratio of creative to total actions
    agentic_novelty = if metrics.total_actions == 0
        0.5  # No data
    else
        creative_actions = metrics.new_combinations_created + metrics.niches_discovered
        following_actions = metrics.derivative_adoptions
        total_creative_relevant = creative_actions + following_actions
        if total_creative_relevant == 0
            0.5
        else
            clamp(creative_actions / total_creative_relevant, 0.0, 1.0)
        end
    end

    # Competitive Recursion: mean competition level experienced
    competitive_recursion = if isempty(metrics.competition_levels)
        0.0  # No investments, no recursion
    else
        # HISTORICAL competition: mean of accumulated competition levels across investments
        # This tracks agent's EXPERIENCE with crowded markets over time
        # Competition accumulates per investment: delta ~0.1-0.2 per agent investment
        # Over many rounds, mean can reach 10-50+ on popular opportunities
        # Use /20.0 so moderate (10) → 0.5, high (20+) → 1.0
        # NOTE: Different scale from CURRENT opportunity.competition (÷2.0 elsewhere)
        clamp(mean(metrics.competition_levels) / 20.0, 0.0, 1.0)
    end

    return Dict{String,Float64}(
        "actor_ignorance" => actor_ignorance,
        "practical_indeterminism" => practical_indeterminism,
        "agentic_novelty" => agentic_novelty,
        "competitive_recursion" => competitive_recursion
    )
end

"""
Record an investment outcome for uncertainty tracking.
"""
function record_investment_outcome!(
    metrics::AgentUncertaintyMetrics,
    estimated_return::Float64,
    actual_return::Float64,
    competition_level::Float64
)
    # Track estimation error (Actor Ignorance)
    if isfinite(estimated_return) && isfinite(actual_return)
        error = abs(estimated_return - actual_return) / max(0.01, abs(actual_return))
        push!(metrics.estimation_errors, clamp(error, 0.0, 5.0))  # Cap at 5x error
    end

    # Track return (Practical Indeterminism)
    if isfinite(actual_return)
        push!(metrics.return_history, actual_return)
    end

    # Track competition (Competitive Recursion)
    if isfinite(competition_level)
        push!(metrics.competition_levels, competition_level)
    end

    # Keep history bounded (last 50 investments)
    max_history = 50
    if length(metrics.estimation_errors) > max_history
        metrics.estimation_errors = metrics.estimation_errors[end-max_history+1:end]
    end
    if length(metrics.return_history) > max_history
        metrics.return_history = metrics.return_history[end-max_history+1:end]
    end
    if length(metrics.competition_levels) > max_history
        metrics.competition_levels = metrics.competition_levels[end-max_history+1:end]
    end
end

"""
Record a creative action for novelty tracking.
"""
function record_creative_action!(
    metrics::AgentUncertaintyMetrics;
    new_combination::Bool = false,
    niche_discovered::Bool = false,
    derivative_adoption::Bool = false
)
    metrics.total_actions += 1
    if new_combination
        metrics.new_combinations_created += 1
    end
    if niche_discovered
        metrics.niches_discovered += 1
    end
    if derivative_adoption
        metrics.derivative_adoptions += 1
    end
end

"""
Get the emergent uncertainty dimensions for an agent.
Returns Dict with actor_ignorance, practical_indeterminism, agentic_novelty, competitive_recursion.
"""
function get_emergent_uncertainty(agent)::Dict{String,Float64}
    return compute_emergent_uncertainty(agent.uncertainty_metrics)
end

"""
Aggregate emergent uncertainty across multiple agents by AI tier.
Returns Dict[tier => Dict[dimension => value]].
Uses fixed_ai_level if set, otherwise current_ai_level.
"""
function aggregate_emergent_uncertainty_by_tier(agents::Vector)::Dict{String,Dict{String,Float64}}
    tier_metrics = Dict{String,Vector{Dict{String,Float64}}}()

    for agent in agents
        if !agent.alive
            continue
        end
        # Use fixed_ai_level if set, otherwise current_ai_level
        tier = if !isnothing(agent.fixed_ai_level)
            agent.fixed_ai_level
        else
            agent.current_ai_level
        end
        if !haskey(tier_metrics, tier)
            tier_metrics[tier] = Dict{String,Float64}[]
        end
        push!(tier_metrics[tier], get_emergent_uncertainty(agent))
    end

    # Aggregate by taking means
    result = Dict{String,Dict{String,Float64}}()
    for (tier, metrics_list) in tier_metrics
        if isempty(metrics_list)
            continue
        end
        result[tier] = Dict{String,Float64}(
            "actor_ignorance" => mean(m["actor_ignorance"] for m in metrics_list),
            "practical_indeterminism" => mean(m["practical_indeterminism"] for m in metrics_list),
            "agentic_novelty" => mean(m["agentic_novelty"] for m in metrics_list),
            "competitive_recursion" => mean(m["competitive_recursion"] for m in metrics_list),
            "n_agents" => Float64(length(metrics_list))
        )
    end

    return result
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
    operating_cost_estimate::Float64  # Estimated operating cost per round

    # Primary sector (assigned at creation based on NVCA sector weights)
    primary_sector::String

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

    # AI subscription tracking (for cost charging)
    subscription_accounts::Dict{String,Int}  # tier → rounds remaining
    subscription_rates::Dict{String,Float64}  # tier → cost per round
    subscription_deferral_remaining::Dict{String,Int}  # tier → grace rounds
    last_subscription_charge::Float64

    # Performance state
    alive::Bool
    survival_rounds::Int
    insolvency_rounds::Int
    total_invested::Float64
    total_returned::Float64
    success_count::Int
    failure_count::Int
    innovation_count::Int
    failure_round::Union{Int,Nothing}  # Track when agent fails (for Kaplan-Meier survival analysis)
    failure_reason::Union{String,Nothing}  # Why agent failed (capital/insolvency)

    # Decision state
    uncertainty_response::UncertaintyResponseProfile
    action_history::Vector{String}
    last_action::String
    last_outcome::Dict{String,Any}

    # Portfolio
    active_investments::Vector{Dict{String,Any}}

    # Emergent uncertainty metrics (agent-level)
    uncertainty_metrics::AgentUncertaintyMetrics

    # RNG
    rng::AbstractRNG
end

function EmergentAgent(
    id::Int,
    config::EmergentConfig;
    initial_capital::Union{Float64,Nothing} = nothing,
    primary_sector::Union{String,Nothing} = nothing,
    fixed_ai_level::Union{String,Nothing} = nothing,
    rng::AbstractRNG = Random.default_rng()
)
    # Select primary sector based on NVCA-weighted probabilities
    sector = if isnothing(primary_sector)
        _sample_sector_weighted(config, rng)
    else
        primary_sector
    end

    # Sample initial capital from sector-specific range (NVCA 2024 calibrated)
    capital = if isnothing(initial_capital)
        sector_profile = get(config.SECTOR_PROFILES, sector, nothing)
        if !isnothing(sector_profile) && hasproperty(sector_profile, :initial_capital_range)
            rand(rng, Uniform(sector_profile.initial_capital_range...))
        else
            rand(rng, Uniform(config.INITIAL_CAPITAL_RANGE...))
        end
    else
        initial_capital
    end

    # Sample traits
    traits = sample_all_traits(config; rng=rng)

    # Create agent resources with sector-boosted knowledge
    resources = AgentResources(capital)
    # Boost knowledge in primary sector
    resources.knowledge[sector] = min(1.0, get(resources.knowledge, sector, 0.1) + 0.2)

    # Initialize operating cost estimate from sector profile
    sector_profile = get(config.SECTOR_PROFILES, sector, nothing)
    initial_cost_estimate = if !isnothing(sector_profile) && hasproperty(sector_profile, :operational_cost_range)
        (sector_profile.operational_cost_range[1] + sector_profile.operational_cost_range[2]) / 2.0
    else
        config.BASE_OPERATIONAL_COST
    end

    return EmergentAgent(
        id,
        resources,
        config,
        initial_cost_estimate,  # operating_cost_estimate
        sector,  # primary_sector
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
        Dict{String,Int}(),  # subscription_accounts
        Dict{String,Float64}(),  # subscription_rates
        Dict{String,Int}(),  # subscription_deferral_remaining
        0.0,  # last_subscription_charge
        true,  # alive
        0,  # survival_rounds
        0,  # insolvency_rounds
        0.0,  # total_invested
        0.0,  # total_returned
        0,  # success_count
        0,  # failure_count
        0,  # innovation_count
        nothing,  # failure_round
        nothing,  # failure_reason
        UncertaintyResponseProfile(rng=rng),
        String[],  # action_history
        "maintain",  # last_action
        Dict{String,Any}(),  # last_outcome
        Dict{String,Any}[],  # active_investments
        AgentUncertaintyMetrics(),  # uncertainty_metrics (emergent, agent-level)
        rng
    )
end

"""
Sample a sector based on NVCA-weighted probabilities.
"""
function _sample_sector_weighted(config::EmergentConfig, rng::AbstractRNG)::String
    sectors = collect(keys(config.SECTOR_WEIGHTS))
    weights = [config.SECTOR_WEIGHTS[s] for s in sectors]

    # Normalize weights
    total = sum(weights)
    if total <= 0
        return rand(rng, sectors)
    end

    probs = weights ./ total
    r = rand(rng)
    cumsum = 0.0
    for (i, prob) in enumerate(probs)
        cumsum += prob
        if r <= cumsum
            return sectors[i]
        end
    end
    return sectors[end]
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
Uses sector-specific survival thresholds calibrated from BLS/Fed data.
"""
function check_survival!(agent::EmergentAgent, round::Int)::Bool
    if !agent.alive
        return false
    end

    # Use sector-specific survival threshold (BLS/Fed calibrated)
    survival_threshold = _get_sector_survival_threshold(agent)

    # Apply AI trust-based liquidity relief (matches Python agents.py:930-933)
    # Agents with high AI trust can maintain lower liquidity reserves
    liquidity_floor = survival_threshold
    ai_level = agent.current_ai_level
    if ai_level != "none"
        trust = clamp(agent.traits["ai_trust"], 0.0, 1.0)
        discount = clamp(agent.config.AI_TRUST_RESERVE_DISCOUNT, 0.0, 0.9)
        relief_factor = clamp(1.0 - trust * discount, 0.5, 1.0)
        liquidity_floor *= relief_factor
    end

    if agent.resources.capital < liquidity_floor
        agent.insolvency_rounds += 1
        if agent.insolvency_rounds >= agent.config.INSOLVENCY_GRACE_ROUNDS
            agent.alive = false
            agent.failure_round = round  # Track when agent fails (for Kaplan-Meier)
            agent.failure_reason = "liquidity_failure"  # Failed due to insufficient capital
            return false
        end
    else
        agent.insolvency_rounds = 0
        agent.survival_rounds = round
    end

    return true
end

"""
Get the survival threshold for an agent based on their primary sector.
Calibrated from BLS Business Employment Dynamics and Fed SBCS 2024.
"""
function _get_sector_survival_threshold(agent::EmergentAgent)::Float64
    sector_profile = get(agent.config.SECTOR_PROFILES, agent.primary_sector, nothing)
    if !isnothing(sector_profile) && hasproperty(sector_profile, :survival_threshold)
        return sector_profile.survival_threshold
    end
    # Fallback to global threshold
    return agent.config.SURVIVAL_THRESHOLD
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
    # Use sector-specific survival threshold (aligned with check_survival!)
    survival_threshold = _get_sector_survival_threshold(agent)

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
    # Decreases when competitive recursion is high (crowded opportunities)
    if !isempty(available_opportunities)
        invest_factor = get_response_factor(
            agent.uncertainty_response, "practical_indeterminism", practical_indet;
            rng=agent.rng
        )
        scores["invest"] = 0.4 * invest_factor * capital_ratio * (1.0 - competitive_rec * 0.5)
    end

    # Innovation score - higher when agentic novelty potential is high
    # ALSO increases when competitive recursion is high - natural response to crowding
    # is to create new opportunities rather than compete for existing ones
    innovate_factor = get_response_factor(
        agent.uncertainty_response, "agentic_novelty", agentic_novelty;
        rng=agent.rng
    )
    competition_innovation_boost = competitive_rec * 0.4  # Boost innovation when competition is high
    scores["innovate"] = 0.3 * agent.innovativeness * innovate_factor * (1.0 + agentic_novelty * 0.3 + competition_innovation_boost)

    # Exploration score - higher when ignorance is high
    # Also increases with competition - seek new niches when existing ones are crowded
    explore_factor = get_response_factor(
        agent.uncertainty_response, "actor_ignorance", actor_ignorance;
        rng=agent.rng
    )
    competition_explore_boost = competitive_rec * 0.25  # Explore more when competition is high
    scores["explore"] = 0.25 * explore_factor * (1.0 + actor_ignorance * 0.5 + competition_explore_boost)

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
    estimated_return::Union{Float64,Nothing} = nothing,
    innovation_engine::Union{InnovationEngine,Nothing} = nothing,
    market_conditions::Union{Dict{String,Any},Nothing} = nothing,
    uncertainty_perception::Union{Dict{String,Any},Nothing} = nothing
)::Dict{String,Any}
    outcome = Dict{String,Any}(
        "action" => action,
        "agent_id" => agent.id,
        "round" => round,
        "ai_level_used" => get_ai_level(agent),
        "capital_before" => get_capital(agent)
    )

    if action == "invest" && !isnothing(opportunity)
        outcome = _execute_invest!(agent, opportunity, market, round, outcome; estimated_return=estimated_return)
    elseif action == "innovate"
        outcome = _execute_innovate!(agent, market, round, outcome;
            innovation_engine=innovation_engine,
            market_conditions=market_conditions,
            uncertainty_perception=uncertainty_perception)
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
    outcome::Dict{String,Any};
    estimated_return::Union{Float64,Nothing} = nothing
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

    # Track total investment on opportunity (for capacity constraints)
    if hasfield(typeof(opportunity), :total_invested)
        opportunity.total_invested += invest_amount
    end

    # Store estimated return (use latent if not provided)
    est_ret = isnothing(estimated_return) ? opportunity.latent_return_potential : estimated_return

    # Record investment with estimated return for uncertainty tracking
    investment = Dict{String,Any}(
        "opportunity_id" => opportunity.id,
        "opportunity" => opportunity,
        "amount" => invest_amount,
        "round_invested" => round,
        "maturity_round" => round + opportunity.time_to_maturity,
        "ai_level" => get_ai_level(agent),
        "estimated_return" => est_ret,
        "competition_at_entry" => hasfield(typeof(opportunity), :competition) ? opportunity.competition : 0.0
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

    # Track whether investing in derivative (known pattern) vs novel opportunity
    # Connected to information system: higher info quality → MORE contrarian (can spot mispricing)
    # CURRENT opportunity competition (not historical): 0=none, 1.0=normal, 3.0=severe
    # Use /2.0 so normal competition (1.0) → 0.5, severe (3.0) → clamped to 1.0
    # NOTE: Different scale from HISTORICAL competition in get_emergent_uncertainty (÷20.0)
    competition_signal = clamp(opportunity.competition / 2.0, 0.0, 1.0)

    # Get AI information parameters
    ai_tier = get_ai_level(agent)
    ai_config = get(agent.config.AI_LEVELS, ai_tier, nothing)
    info_quality = isnothing(ai_config) ? 0.25 : Float64(ai_config.info_quality)

    # FIXED: Derivative probability components (inverted from before)
    # 1. Competition signal: high competition = crowded = derivative investment
    # 2. Info quality effect: INVERTED - higher quality = MORE contrarian
    #    - Low info_quality: can't identify mispricing → follows the crowd (higher derivative prob)
    #    - High info_quality: can spot when crowd is wrong → contrarian (lower derivative prob)
    #    This creates the AI paradox: premium agents diverge more, sometimes finding novel winners
    competition_effect = competition_signal * 0.35
    contrarian_ability = (1.0 - info_quality) * 0.25  # 0.19 (none) to 0.01 (premium)

    # None:    0.20 + 0.35*comp + 0.19 = 0.39 + 0.35*comp (more following)
    # Premium: 0.20 + 0.35*comp + 0.01 = 0.21 + 0.35*comp (more contrarian)
    derivative_prob = 0.20 + competition_effect + contrarian_ability
    is_derivative = rand(agent.rng) < derivative_prob
    outcome["invested_derivative"] = is_derivative
    outcome["competition_at_investment"] = opportunity.competition
    outcome["info_quality_used"] = info_quality
    outcome["estimated_return"] = est_ret

    # Track for emergent agentic novelty (derivative = following, not novel = creating)
    record_creative_action!(agent.uncertainty_metrics; derivative_adoption=is_derivative)

    return outcome
end

function _execute_innovate!(
    agent::EmergentAgent,
    market::MarketEnvironment,
    round::Int,
    outcome::Dict{String,Any};
    innovation_engine::Union{InnovationEngine,Nothing} = nothing,
    market_conditions::Union{Dict{String,Any},Nothing} = nothing,
    uncertainty_perception::Union{Dict{String,Any},Nothing} = nothing
)::Dict{String,Any}
    capital = get_capital(agent)
    ai_tier = get_ai_level(agent)

    # Innovation cost (R&D spend)
    rd_spend = min(
        capital * agent.config.INNOVATION_BASE_SPEND_RATIO,
        agent.config.INNOVATION_MAX_SPEND
    )

    if rd_spend > capital * 0.5
        rd_spend = capital * 0.2  # Cap R&D if capital constrained
    end

    set_capital!(agent, capital - rd_spend)
    outcome["rd_spend"] = rd_spend
    outcome["ai_tier_used"] = ai_tier

    # =========================================================================
    # USE KNOWLEDGE-BASED INNOVATION SYSTEM (if available)
    # =========================================================================
    if !isnothing(innovation_engine)
        # Use the full knowledge recombination system
        mkt_cond = isnothing(market_conditions) ? Dict{String,Any}("regime" => "normal") : market_conditions

        # Attempt to create an innovation through knowledge recombination
        innovation = attempt_innovation!(
            innovation_engine,
            agent,
            mkt_cond,
            round;
            ai_level=ai_tier,
            uncertainty_perception=uncertainty_perception,
            decision_perception=uncertainty_perception,
            rng=agent.rng
        )

        if !isnothing(innovation)
            # Innovation was created through knowledge recombination
            # Now evaluate its success and calculate returns

            # Get market innovations for competition calculation
            market_innovations = collect(values(innovation_engine.innovations))

            # Evaluate innovation success based on quality, novelty, scarcity, competition
            success, impact, cash_multiple = evaluate_innovation_success!(
                innovation_engine,
                innovation,
                mkt_cond,
                market_innovations;
                rng=agent.rng
            )

            # Calculate actual returns based on R&D spend and knowledge-derived multiplier
            innovation_return = rd_spend * cash_multiple

            set_capital!(agent, get_capital(agent) + innovation_return)

            if success
                agent.innovation_count += 1
                agent.success_count += 1
            else
                agent.failure_count += 1
            end

            # Record detailed outcome
            outcome["success"] = success
            outcome["innovation_return"] = innovation_return
            outcome["cash_multiple"] = cash_multiple
            outcome["market_impact"] = impact
            outcome["innovation_id"] = innovation.id
            outcome["innovation_type"] = innovation.type
            outcome["innovation_quality"] = innovation.quality
            outcome["innovation_novelty"] = innovation.novelty
            outcome["innovation_scarcity"] = something(innovation.scarcity, 0.5)
            outcome["is_new_combination"] = innovation.is_new_combination
            outcome["knowledge_components"] = innovation.knowledge_components
            outcome["ai_assisted"] = innovation.ai_assisted
            outcome["ai_domains_used"] = innovation.ai_domains_used

            # Track for emergent agentic novelty
            record_creative_action!(agent.uncertainty_metrics; new_combination=innovation.is_new_combination)

            # Record knowledge recombination metrics for analysis
            push!(agent.uncertainty_metrics.innovation_qualities, innovation.quality)
            push!(agent.uncertainty_metrics.innovation_novelties, innovation.novelty)
            push!(agent.uncertainty_metrics.innovation_scarcities, something(innovation.scarcity, 0.5))

            # Get AI info parameters for tracking
            ai_config = get(agent.config.AI_LEVELS, ai_tier, nothing)
            outcome["info_quality_used"] = isnothing(ai_config) ? 0.25 : Float64(ai_config.info_quality)
            outcome["info_breadth_used"] = isnothing(ai_config) ? 0.20 : Float64(ai_config.info_breadth)
        else
            # Failed to create innovation (couldn't find good knowledge combination)
            # Partial recovery of R&D spend
            recovery = rd_spend * agent.config.INNOVATION_FAIL_RECOVERY_RATIO
            set_capital!(agent, get_capital(agent) + recovery)

            outcome["success"] = false
            outcome["recovery"] = recovery
            outcome["failure_reason"] = "no_viable_combination"
            outcome["is_new_combination"] = false

            agent.failure_count += 1
        end
    else
        # =====================================================================
        # FALLBACK: Simple probability-based innovation (legacy behavior)
        # This path is used when innovation_engine is not provided
        # =====================================================================
        base_prob = agent.config.INNOVATION_PROBABILITY
        competence_factor = agent.competence
        innovativeness_factor = agent.innovativeness

        success_prob = base_prob * (0.5 + 0.5 * competence_factor) * (0.7 + 0.3 * innovativeness_factor)
        success = rand(agent.rng) < success_prob

        if success
            base_return = rd_spend * agent.config.INNOVATION_SUCCESS_BASE_RETURN
            multiplier = rand(agent.rng, Uniform(agent.config.INNOVATION_SUCCESS_RETURN_MULTIPLIER...))
            innovation_return = base_return * multiplier

            set_capital!(agent, get_capital(agent) + innovation_return)
            agent.innovation_count += 1
            agent.success_count += 1

            outcome["success"] = true
            outcome["innovation_return"] = innovation_return

            # Simple new combination tracking
            ai_config = get(agent.config.AI_LEVELS, ai_tier, nothing)
            info_quality = isnothing(ai_config) ? 0.25 : Float64(ai_config.info_quality)
            info_breadth = isnothing(ai_config) ? 0.20 : Float64(ai_config.info_breadth)

            breadth_bonus = info_breadth * 0.12
            accuracy_adjustment = -info_quality * 0.08
            new_combo_base = 0.22 + breadth_bonus + accuracy_adjustment
            new_combo_prob = new_combo_base * (0.7 + 0.6 * agent.innovativeness)
            is_new_combo = rand(agent.rng) < new_combo_prob

            outcome["is_new_combination"] = is_new_combo
            outcome["info_breadth_used"] = info_breadth
            outcome["info_quality_used"] = info_quality
            outcome["fallback_mode"] = true

            record_creative_action!(agent.uncertainty_metrics; new_combination=is_new_combo)
        else
            recovery = rd_spend * agent.config.INNOVATION_FAIL_RECOVERY_RATIO
            set_capital!(agent, get_capital(agent) + recovery)

            outcome["success"] = false
            outcome["recovery"] = recovery
            outcome["fallback_mode"] = true
        end
    end

    record_deployment!(agent.resources.performance, "innovate", rd_spend;
                       ai_level=ai_tier, round_num=round)

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
        # Get AI information parameters
        ai_tier = get_ai_level(agent)
        ai_config = get(agent.config.AI_LEVELS, ai_tier, nothing)
        info_quality = isnothing(ai_config) ? 0.25 : Float64(ai_config.info_quality)
        info_breadth = isnothing(ai_config) ? 0.20 : Float64(ai_config.info_breadth)

        # Increase knowledge in a random sector
        sector = rand(agent.rng, agent.config.SECTORS)
        current_knowledge = get(agent.resources.knowledge, sector, 0.1)

        # Knowledge gain scales with info_breadth (broader access = faster learning)
        # Range: 0.20 (none) to 0.92 (premium)
        # None: base 0.05-0.15, Premium: base 0.05-0.15 * 1.43
        breadth_multiplier = 0.7 + 0.6 * info_breadth  # 0.82 (none) to 1.25 (premium)
        knowledge_gain = rand(agent.rng, Uniform(0.05, 0.15)) * breadth_multiplier

        agent.resources.knowledge[sector] = min(1.0, current_knowledge + knowledge_gain)
        agent.resources.knowledge_last_used[sector] = round

        outcome["success"] = true
        outcome["discovered_sector"] = sector
        outcome["knowledge_gain"] = knowledge_gain
        outcome["base_sector"] = sector

        # Niche discovery: trade-off between serendipity and systematic search
        #
        # SERENDIPITY (human advantage):
        # - Random exploration can stumble on truly novel niches ("unknown unknowns")
        # - Less constrained by existing mental models
        # - Higher variance, occasionally finds things AI would never look for
        #
        # SYSTEMATIC (AI advantage):
        # - Better at finding niches in analyzed/mapped territory ("known unknowns")
        # - More thorough coverage of possibility space
        # - Lower variance, reliably finds opportunities within scope
        #
        # Trade-off: neither is strictly better
        serendipity_factor = (1.0 - info_breadth) * 0.12  # 0.096 (none) to 0.01 (premium)
        systematic_factor = info_breadth * 0.08           # 0.016 (none) to 0.074 (premium)

        # Net niche discovery base:
        # None:    0.12 + 0.096 + 0.016 = 0.232 (serendipity advantage)
        # Premium: 0.12 + 0.010 + 0.074 = 0.204 (slightly lower, but more reliable)
        niche_discovery_base = 0.12 + serendipity_factor + systematic_factor

        # Uncertainty tolerance still affects niche discovery (risk-takers explore edges)
        niche_discovery_prob = niche_discovery_base * (0.6 + 0.8 * agent.uncertainty_tolerance)
        discovered_niche = rand(agent.rng) < niche_discovery_prob
        created_niche = discovered_niche && knowledge_gain > 0.12  # Strong discovery creates new niche
        outcome["discovered_niche"] = discovered_niche
        outcome["created_niche"] = created_niche
        outcome["info_breadth_used"] = info_breadth
        outcome["serendipity_factor"] = serendipity_factor
        outcome["systematic_factor"] = systematic_factor

        # Track for emergent agentic novelty
        record_creative_action!(agent.uncertainty_metrics; niche_discovered=created_niche)
    else
        outcome["success"] = false
        outcome["discovered_niche"] = false
        outcome["created_niche"] = false
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
    round::Int;
    market_conditions::Union{Dict{String,Any},Nothing} = nothing
)::Vector{Dict{String,Any}}
    if !agent.alive
        return Dict{String,Any}[]
    end

    matured_outcomes = Dict{String,Any}[]
    remaining_investments = Dict{String,Any}[]

    # Use provided market_conditions or fetch from market
    market_conditions = isnothing(market_conditions) ? get_market_conditions(market) : market_conditions

    for investment in agent.active_investments
        if investment["maturity_round"] <= round
            # Investment matures
            opp = get(investment, "opportunity", nothing)
            if isnothing(opp)
                # Skip malformed investment - push to remaining and continue
                push!(remaining_investments, investment)
                continue
            end
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

            # Track emergent uncertainty metrics
            estimated_return = Float64(get(investment, "estimated_return", ret_multiple))
            competition_level = hasfield(typeof(opp), :competition) ? Float64(opp.competition) : 0.0
            record_investment_outcome!(
                agent.uncertainty_metrics,
                estimated_return,
                ret_multiple,
                competition_level
            )

            push!(matured_outcomes, Dict{String,Any}(
                "investment" => investment,
                "investment_amount" => invested_amount,
                "capital_returned" => capital_returned,
                "return_multiple" => ret_multiple,
                "success" => success,
                "round_matured" => round,
                "estimated_return" => estimated_return,
                "competition_at_maturity" => competition_level
            ))
        else
            push!(remaining_investments, investment)
        end
    end

    agent.active_investments = remaining_investments
    return matured_outcomes
end

"""
Apply operational costs for the round using sector-specific costs.
"""
function apply_operational_costs!(agent::EmergentAgent, round::Int)
    if !agent.alive
        return
    end

    # Use sector-specific operational cost from SECTOR_PROFILES
    sector_profile = get(agent.config.SECTOR_PROFILES, agent.primary_sector, nothing)
    cost = if !isnothing(sector_profile) && hasproperty(sector_profile, :operational_cost_range)
        # Use agent's stored estimate (mid-range from sector profile)
        agent.operating_cost_estimate
    else
        agent.config.BASE_OPERATIONAL_COST
    end

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

    ai_config = get(agent.config.AI_LEVELS, ai_level, nothing)
    if isnothing(ai_config)
        return 0.0
    end

    cost_type = ai_config.cost_type
    # Scale costs by AI_COST_INTENSITY (for robustness testing)
    cost_intensity = agent.config.AI_COST_INTENSITY

    if cost_type == "subscription"
        base_cost = Float64(ai_config.cost) * cost_intensity
        per_use = Float64(ai_config.per_use_cost) * cost_intensity
        # Amortize subscription over rounds
        amort_rounds = max(1, agent.config.AI_SUBSCRIPTION_AMORTIZATION_ROUNDS)
        return base_cost / amort_rounds + per_use * max(expected_calls, 0.0)
    elseif cost_type == "per_use"
        return Float64(ai_config.cost) * cost_intensity * max(expected_calls, 0.0)
    end

    return 0.0
end

"""
Start a subscription schedule for a given AI level.
Charges the full quarterly cost per round (monthly cost × 3 months).
Since rounds = quarters and AI costs are documented as monthly fees,
we convert: Premium at 3500/month = 10500 per quarter.
"""
function start_subscription_schedule!(
    agent::EmergentAgent,
    ai_level::String,
    ai_config::AILevelConfig
)
    monthly_cost = Float64(ai_config.cost)

    if monthly_cost <= 0
        return
    end

    # Convert monthly cost to quarterly cost (rounds = quarters = 3 months)
    quarterly_cost = monthly_cost * 3.0

    # Scale costs by AI_COST_INTENSITY (for robustness testing)
    cost_intensity = agent.config.AI_COST_INTENSITY
    quarterly_cost = quarterly_cost * cost_intensity

    # Set up subscription to charge EVERY round (no amortization)
    agent.subscription_accounts[ai_level] = 1  # Active subscription
    agent.subscription_rates[ai_level] = quarterly_cost  # Full quarterly cost
    agent.subscription_deferral_remaining[ai_level] = 0
end

"""
Charge one subscription installment for a given AI level.
Deducts the full quarterly cost from agent capital EVERY round.
Returns the amount charged.
"""
function charge_subscription_installment!(
    agent::EmergentAgent,
    ai_level::String
)::Float64
    active = get(agent.subscription_accounts, ai_level, 0)
    rate = get(agent.subscription_rates, ai_level, 0.0)

    if active <= 0 || rate <= 0
        return 0.0
    end

    # Check deferral (credit line grace period)
    deferral = get(agent.subscription_deferral_remaining, ai_level, 0)
    if deferral > 0
        agent.subscription_deferral_remaining[ai_level] = deferral - 1
        return 0.0
    end

    # CHARGE THE SUBSCRIPTION - deduct full quarterly cost from capital
    set_capital!(agent, get_capital(agent) - rate)

    # Subscription remains active (charge EVERY round, no amortization)
    # subscription_accounts[ai_level] stays at 1 to indicate active subscription

    return rate
end

"""
Apply subscription charges for all active AI subscriptions.
Called once per round to charge all subscription installments.
Returns total amount charged.
"""
function apply_subscription_carry!(
    agent::EmergentAgent,
    round::Int
)::Float64
    total = 0.0
    credit_line = agent.config.AI_CREDIT_LINE_ROUNDS

    for level in collect(keys(agent.subscription_accounts))
        # Allow credit line grace period for advanced/premium tiers
        if credit_line > 0 && round <= credit_line && level in ("advanced", "premium")
            continue
        end

        total += charge_subscription_installment!(agent, level)
    end

    agent.last_subscription_charge = total
    return total
end

"""
Ensure subscription schedule is active for the given AI level.
Call this whenever an agent adopts or switches to a subscription-based AI tier.
"""
function ensure_subscription_schedule!(
    agent::EmergentAgent,
    ai_level::String
)
    ai_config = get(agent.config.AI_LEVELS, ai_level, nothing)

    if isnothing(ai_config)
        return
    end

    if ai_config.cost_type != "subscription" || ai_config.cost <= 0
        return
    end

    # Start schedule if not already active
    if get(agent.subscription_accounts, ai_level, 0) <= 0
        start_subscription_schedule!(agent, ai_level, ai_config)
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
    amort_horizon = max(1, agent.config.AI_SUBSCRIPTION_AMORTIZATION_ROUNDS)
    ref_scale = max(operating_cost * 4.0, cash_buffer * 0.12, 1.0)

    cost_ratios = Dict{String,Float64}()
    # Scale costs by AI_COST_INTENSITY (for robustness testing)
    cost_intensity = agent.config.AI_COST_INTENSITY

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
        base_cost = Float64(cfg.cost) * cost_intensity
        per_use_cost = Float64(cfg.per_use_cost) * cost_intensity

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

    # Start subscription schedule if this is a subscription tier
    ensure_subscription_schedule!(agent, best_tier)

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

    # Get AI tier parameters - NOW USED for opportunity evaluation
    ai_config = get(agent.config.AI_LEVELS, ai_level, nothing)
    info_quality = isnothing(ai_config) ? 0.25 : Float64(ai_config.info_quality)
    info_breadth = isnothing(ai_config) ? 0.20 : Float64(ai_config.info_breadth)

    # Get best opportunity score with AI-enhanced evaluation
    max_score = 0.0
    avg_score = 0.0
    for opp in opportunities
        base_score = evaluate_opportunity_basic(agent, opp, market_conditions)

        # AI tier advantage: better info_quality = more accurate opportunity assessment
        # Higher info_quality reduces noise in evaluation (sees true potential better)
        noise_reduction = info_quality * 0.3  # Up to 30% noise reduction for premium
        ai_adjusted_score = base_score * (1.0 + noise_reduction * (opp.latent_return_potential - 1.0))

        # info_breadth helps identify opportunities across more sectors
        breadth_bonus = info_breadth * 0.1 * (1.0 - opp.complexity)  # Simpler opps benefit more from broad view
        ai_adjusted_score += breadth_bonus

        max_score = max(max_score, ai_adjusted_score)
        avg_score += ai_adjusted_score
    end
    avg_score /= length(opportunities)

    # AI advantage: premium can better distinguish best from average opportunities
    score_spread = max_score - avg_score
    ai_selection_bonus = info_quality * score_spread * 0.2  # Better at picking winners

    scaled_score = stable_sigmoid(max_score + ai_selection_bonus - 1.0)

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
    invest_roic = compute_roic(perf, "invest")

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
Now uses AI tier for decision-making advantage.
"""
function calculate_innovation_utility(
    agent::EmergentAgent,
    market_conditions::Dict{String,Any},
    perception::Dict{String,Any};
    ai_level::String = "none"
)::Float64
    base_drive = get(agent.traits, "innovativeness", 0.5) * 0.5 + 0.05

    # Get AI tier parameters - NOW USED for innovation decision
    ai_config = get(agent.config.AI_LEVELS, ai_level, nothing)
    info_quality = isnothing(ai_config) ? 0.25 : Float64(ai_config.info_quality)
    info_breadth = isnothing(ai_config) ? 0.20 : Float64(ai_config.info_breadth)

    # Extract signals
    practical_unc = Float64(get(get(perception, "practical_indeterminism", Dict()), "level", 0.5))
    agentic_unc = Float64(get(get(perception, "agentic_novelty", Dict()), "level", 0.5))

    indeterminism_bonus = practical_unc * 0.4
    innovation_opportunity = agentic_unc * 0.25

    # AI advantage for innovation decisions:
    # 1. info_breadth: broader knowledge access enables identifying novel combinations
    breadth_innovation_bonus = info_breadth * 0.15  # Up to +14% for premium

    # 2. info_quality: better at assessing innovation feasibility
    #    High quality helps identify which innovations are worth pursuing
    feasibility_assessment = info_quality * 0.10  # Up to +10% for premium

    # 3. AI helps reduce perceived risk of innovation (better analysis of outcomes)
    ai_risk_reduction = info_quality * 0.08  # Reduces avoidance effect

    # Capital constraints
    capital_ratio = get_capital(agent) / max(agent.resources.performance.initial_equity, 1.0)
    liquidity_penalty = clamp(1.0 - capital_ratio, 0.0, 1.5)

    # R&D burden
    rd_deployed = get(agent.resources.performance.deployed_by_action, "innovate", 0.0)
    rd_burden = clamp(rd_deployed / max(agent.resources.performance.initial_equity, 1.0), 0.0, 2.0)

    innovate_roic = compute_roic(agent.resources.performance, "innovate")
    loss_ratio = max(0.0, -innovate_roic)

    risk_tolerance = Float64(get(agent.traits, "uncertainty_tolerance", 0.5))
    avoidance = 1.0 - risk_tolerance

    raw_score = (
        base_drive +
        indeterminism_bonus +
        innovation_opportunity +
        breadth_innovation_bonus +      # NEW: AI breadth bonus
        feasibility_assessment -         # NEW: AI quality bonus
        0.25 * liquidity_penalty -
        0.3 * rd_burden -
        0.4 * loss_ratio +
        0.2 * (risk_tolerance - 0.5) -
        0.15 * avoidance * (1.0 - ai_risk_reduction)  # MODIFIED: AI reduces avoidance
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
    # No tier-specific modification - let behavior emerge naturally

    explore_roic = compute_roic(agent.resources.performance, "explore")
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
    sectors_in_portfolio = String[]
    for inv in agent.active_investments
        opp = get(inv, "opportunity", nothing)
        if !isnothing(opp) && !isnothing(opp.sector)
            push!(sectors_in_portfolio, opp.sector)
        end
    end
    n_sectors = length(unique(sectors_in_portfolio))
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
    # Expected profit margin (using latent return potential)
    expected_margin = opportunity.latent_return_potential - 1.0
    # Use complexity as a proxy for uncertainty
    uncertainty = opportunity.complexity
    uncertainty_adjusted = expected_margin * (1.0 - uncertainty * 0.5)

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
    if !isnothing(opportunity.created_by)
        adjusted *= 1.1
        if opportunity.created_by == agent.id
            adjusted *= 1.2
        end
    end

    return max(0.01, adjusted)
end

"""
Evaluate portfolio opportunities and return ranked list.

Sequential Decision Signals:
If early_signals is provided (from sequential decision making), opportunities that
early investors chose receive a boost. The effect scales with info_quality - Premium
agents trust visible signals more (rational herding / information cascades).
"""
function evaluate_portfolio_opportunities(
    agent::EmergentAgent,
    opportunities::Vector{Opportunity},
    market_conditions::Dict{String,Any},
    perception::Dict{String,Any};
    ai_level::String = "none",
    early_signals::Union{Dict{String,Int},Nothing} = nothing,
    signal_weight::Float64 = 0.15
)::Vector{Dict{String,Any}}
    if isempty(opportunities)
        return Dict{String,Any}[]
    end

    evaluations = Dict{String,Any}[]

    # Evaluate ALL opportunities - no artificial cutoffs
    # Herding emerges NATURALLY from information quality:
    # - Premium agents have low noise → all rank the SAME opportunities as best → convergence
    # - None agents have high noise → rank DIFFERENT opportunities as best → distribution
    #
    # NOVELTY INVERSION: Premium's advantage is INVERTED for novel opportunities
    # - For established opportunities: Premium has low noise (accurate)
    # - For novel opportunities: Premium has HIGH noise (patterns don't apply)
    # This reflects "the more important the innovation, the less predictable it is" (Rescher)
    opp_pool = copy(opportunities)
    if length(opp_pool) > 1
        # Get actual AI config parameters
        ai_config = get(agent.config.AI_LEVELS, ai_level, nothing)
        info_quality = isnothing(ai_config) ? 0.25 : Float64(ai_config.info_quality)

        # Base noise scale inversely proportional to info quality
        # Premium (0.97): base_noise = 0.015 → accurate for established opportunities
        # None (0.25):    base_noise = 0.375 → noisy for all opportunities
        base_noise_scale = 0.5 * (1.0 - info_quality)

        # Novelty noise inversion factor from config
        inversion_factor = getfield_default(agent.config, :NOVELTY_NOISE_INVERSION_FACTOR, 0.4)

        # Sort by ESTIMATED returns with novelty-dependent noise
        estimated_returns = Dict{String,Float64}()
        for opp in opp_pool
            # Get opportunity novelty score (0.0 = established, 1.0 = very novel)
            opp_novelty = hasfield(typeof(opp), :novelty_score) ? opp.novelty_score : 0.0

            # For NOVEL opportunities, Premium's advantage inverts:
            # - Established (novelty=0): effective_noise = base_noise_scale (Premium accurate)
            # - Novel (novelty=1): effective_noise = base_noise_scale + info_quality * inversion_factor
            #   Premium at novelty=1: 0.015 + 0.97 * 0.4 = 0.403 (HIGHER than None's 0.375!)
            #   None at novelty=1: 0.375 + 0.25 * 0.4 = 0.475 (slightly higher)
            novelty_noise_penalty = opp_novelty * info_quality * inversion_factor

            effective_noise = base_noise_scale + novelty_noise_penalty
            noise = randn(agent.rng) * effective_noise
            base_estimate = opp.latent_return_potential + noise

            # Apply early signals boost (information cascade / sequential decision visibility)
            # All tiers respond uniformly to early investor signals - let herding emerge naturally
            if !isnothing(early_signals)
                signal_count = get(early_signals, opp.id, 0)
                if signal_count > 0
                    # Uniform signal boost for all tiers - no hardcoded herding susceptibility
                    signal_boost = signal_count * signal_weight * 0.1
                    base_estimate *= (1.0 + signal_boost)
                end
            end

            estimated_returns[opp.id] = base_estimate
        end
        sort!(opp_pool, by=o -> estimated_returns[o.id], rev=true)

        # NO CUTOFF - agents evaluate all opportunities and pick the best according to their estimates
    end

    for opp in opp_pool
        base_score = evaluate_opportunity_basic(agent, opp, market_conditions)
        final_score = apply_uncertainty_adjustments(agent, base_score, opp, perception)

        # Get estimated return (if computed during sorting, otherwise use latent return)
        est_return = if @isdefined(estimated_returns) && haskey(estimated_returns, opp.id)
            estimated_returns[opp.id]
        else
            opp.latent_return_potential
        end

        push!(evaluations, Dict{String,Any}(
            "opportunity" => opp,
            "final_score" => final_score,
            "ai_level_used" => ai_level,
            "estimated_return" => est_return,
            "competition_at_evaluation" => hasfield(typeof(opp), :competition) ? opp.competition : 0.0
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
    ai_level_override::Union{String,Nothing} = nothing,
    innovation_engine::Union{InnovationEngine,Nothing} = nothing,
    info_system::Union{InformationSystem,Nothing} = nothing,
    early_signals::Union{Dict{String,Int},Nothing} = nothing,
    signal_weight::Float64 = 0.15
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
    innovate_utility = calculate_innovation_utility(agent, market_conditions, perception; ai_level=ai_level)
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
        evals = evaluate_portfolio_opportunities(
            agent, opportunities, market_conditions, perception;
            ai_level=ai_level,
            early_signals=early_signals,
            signal_weight=signal_weight
        )
        if !isempty(evals)
            best_eval = evals[1]
            best_opp = best_eval["opportunity"]
            estimated_return = get(best_eval, "estimated_return", best_opp.latent_return_potential)
            execute_action!(agent, "invest", market, round_num;
                opportunity=best_opp,
                estimated_return=estimated_return,
                innovation_engine=innovation_engine,
                market_conditions=market_conditions,
                uncertainty_perception=perception)
        else
            execute_action!(agent, "maintain", market, round_num;
                innovation_engine=innovation_engine,
                market_conditions=market_conditions,
                uncertainty_perception=perception)
        end
    else
        execute_action!(agent, chosen_action, market, round_num;
            innovation_engine=innovation_engine,
            market_conditions=market_conditions,
            uncertainty_perception=perception)
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
