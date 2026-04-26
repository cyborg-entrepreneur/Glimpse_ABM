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
    last_tier_review_round::Int  # v3.4: round of most-recent choose_ai_level eval

    # Performance state
    alive::Bool
    survival_rounds::Int
    insolvency_rounds::Int
    equity_streak::Int  # consecutive rounds below sector equity ratio (v2.5 — equity_failure mechanism)
    total_invested::Float64
    total_returned::Float64
    success_count::Int              # total successes (invest + innovate)
    failure_count::Int              # total failures (invest + innovate)
    innovation_count::Int           # total innovation attempts (not just successes)
    # v3.3.4: separate counters so analysis scripts don't conflate channels.
    # Prior code incremented success_count on BOTH innovation-success and
    # investment-maturity-success, then run_fixed_tier_analysis.jl:546 used
    # the mix as Innovation_Success_Rate — the reported rate was really
    # "any success rate." Keep success_count/failure_count as totals for
    # backward-compat; use these new counters for channel-specific metrics.
    innovation_success_count::Int
    innovation_failure_count::Int
    investment_success_count::Int
    investment_failure_count::Int
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

    # Sample initial capital. Precedence (v2.10):
    # 1. Explicit `initial_capital` kwarg (per-agent override)
    # 2. config.INITIAL_CAPITAL scalar when USE_UNIFORM_INITIAL_CAPITAL=true
    #    (scripts set the flag + the scalar to force identical starting capital)
    # 3. sector_profile.initial_capital_range (default sector-specific sample)
    # 4. config.INITIAL_CAPITAL_RANGE (fallback if sector profile lacks the field)
    #
    # v2.9's sentinel-based detection (INITIAL_CAPITAL != 5M default) failed
    # when scripts set it to the default value intending uniform capital —
    # the explicit flag unambiguously opts in.
    use_uniform_cap = getfield_default(config, :USE_UNIFORM_INITIAL_CAPITAL, false)
    capital = if !isnothing(initial_capital)
        initial_capital
    elseif use_uniform_cap
        config.INITIAL_CAPITAL
    else
        sector_profile = get(config.SECTOR_PROFILES, sector, nothing)
        if !isnothing(sector_profile) && hasproperty(sector_profile, :initial_capital_range)
            rand(rng, Uniform(sector_profile.initial_capital_range...))
        else
            rand(rng, Uniform(config.INITIAL_CAPITAL_RANGE...))
        end
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
        # current_ai_level mirrors fixed_ai_level for fixed-tier agents so direct
        # reads (analysis scripts, telemetry) classify them correctly. Earlier
        # this hardcoded "none" and relied on get_ai_level() to mask it, which
        # worked in main code paths but misled any code that introspected the
        # field directly.
        something(fixed_ai_level, "none"),
        fixed_ai_level,
        AILearningProfile(),
        0,  # ai_usage_count
        String[],  # ai_tier_history
        Dict{String,Int}(),  # subscription_accounts
        Dict{String,Float64}(),  # subscription_rates
        Dict{String,Int}(),  # subscription_deferral_remaining
        0.0,  # last_subscription_charge
        -100,  # last_tier_review_round (v3.4 — sentinel ensures first-round eval fires)
        true,  # alive
        0,  # survival_rounds
        0,  # insolvency_rounds
        0,  # equity_streak (v2.5 — equity_failure mechanism)
        0.0,  # total_invested
        0.0,  # total_returned
        0,  # success_count
        0,  # failure_count
        0,  # innovation_count
        0,  # innovation_success_count  (v3.3.4)
        0,  # innovation_failure_count  (v3.3.4)
        0,  # investment_success_count  (v3.3.4)
        0,  # investment_failure_count  (v3.3.4)
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
    # Use get_ai_level() to honor fixed_ai_level for fixed-tier runs
    # (current_ai_level is "none" until first make_decision! call, which
    # happens AFTER check_survival! in the step path — so fixed-tier
    # premium agents were silently missing this relief)
    liquidity_floor = survival_threshold
    ai_level = get_ai_level(agent)
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

    # v2.5: equity_failure check (mirrors Python agents.py:_evaluate_failure_conditions).
    # Agent fails when capital_ratio (current / initial) drops below the sector
    # equity ratio for INSOLVENCY_GRACE_ROUNDS consecutive rounds. Earlier
    # versions of Julia only had liquidity_failure, so any agent above the
    # liquidity floor was effectively immortal regardless of how much equity
    # they had burned. This is the second leg of the failure mechanism.
    initial_equity = max(agent.resources.performance.initial_equity, 1.0)
    capital_ratio = agent.resources.capital / initial_equity
    sector_equity_ratio = _get_sector_equity_ratio(agent)
    # Same AI-trust relief that liquidity uses
    ai_relief = 1.0
    if ai_level != "none"
        trust = clamp(agent.traits["ai_trust"], 0.0, 1.0)
        discount = clamp(agent.config.AI_TRUST_RESERVE_DISCOUNT, 0.0, 0.9)
        ai_relief = clamp(1.0 - trust * discount, 0.5, 1.0)
    end
    ratio_floor = sector_equity_ratio * ai_relief
    if capital_ratio < ratio_floor
        agent.equity_streak += 1
        if agent.equity_streak >= agent.config.INSOLVENCY_GRACE_ROUNDS
            agent.alive = false
            agent.failure_round = round
            agent.failure_reason = "equity_failure"
            return false
        end
    else
        agent.equity_streak = 0
    end

    return true
end

"""
Get the survival threshold for an agent based on their primary sector.
Calibrated from BLS Business Employment Dynamics and Fed SBCS 2024.
"""
function _get_sector_survival_threshold(agent::EmergentAgent)::Float64
    # v2.10: honor config.USE_UNIFORM_SURVIVAL_THRESHOLD flag. When true,
    # return the scalar (bypassing sector profiles). Earlier scripts that
    # set config.SURVIVAL_THRESHOLD expecting a uniform 10K floor were
    # silently overridden by sector profiles ($700K-$2.6M depending on
    # sector) — survival rates looked artificially low in those scripts.
    if getfield_default(agent.config, :USE_UNIFORM_SURVIVAL_THRESHOLD, false)
        return agent.config.SURVIVAL_THRESHOLD
    end
    sector_profile = get(agent.config.SECTOR_PROFILES, agent.primary_sector, nothing)
    if !isnothing(sector_profile) && hasproperty(sector_profile, :survival_threshold)
        return sector_profile.survival_threshold
    end
    # Fallback to global threshold
    return agent.config.SURVIVAL_THRESHOLD
end

"""
Get the minimum capital-retention ratio (capital / initial_equity) below
which the agent is considered equity-impaired. Falls back to global
SURVIVAL_CAPITAL_RATIO if the sector profile doesn't expose one.
"""
function _get_sector_equity_ratio(agent::EmergentAgent)::Float64
    sector_profile = get(agent.config.SECTOR_PROFILES, agent.primary_sector, nothing)
    if !isnothing(sector_profile) && hasproperty(sector_profile, :survival_equity_ratio)
        return sector_profile.survival_equity_ratio
    end
    return agent.config.SURVIVAL_CAPITAL_RATIO
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
    market_conditions::MarketConditions,
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
    ai_info::Union{Information,Nothing} = nothing,
    innovation_engine::Union{InnovationEngine,Nothing} = nothing,
    market_conditions::Union{MarketConditions,Nothing} = nothing,
    uncertainty_perception::Union{Dict{String,Any},Nothing} = nothing,
    # v3.2: confidence and signal_score flow from the decision layer
    # (perception + evaluate_portfolio_opportunities) into invest sizing.
    confidence::Union{Float64,Nothing} = nothing,
    signal_score::Union{Float64,Nothing} = nothing
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
            estimated_return=estimated_return, ai_info=ai_info,
            confidence=confidence, signal_score=signal_score)
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
    estimated_return::Union{Float64,Nothing} = nothing,
    ai_info::Union{Information,Nothing} = nothing,
    # v3.2: confidence and signal_score scale the dollar amount so that
    # agents who perceive less uncertainty (higher confidence) and higher
    # expected quality (higher signal_score) deploy more capital per bet,
    # mirroring Python agents.py:2881. nothing = legacy flat sizing.
    confidence::Union{Float64,Nothing} = nothing,
    signal_score::Union{Float64,Nothing} = nothing
)::Dict{String,Any}
    capital = get_capital(agent)
    max_fraction = agent.config.MAX_INVESTMENT_FRACTION
    target_fraction = hasfield(typeof(agent.config), :TARGET_INVESTMENT_FRACTION) ?
        agent.config.TARGET_INVESTMENT_FRACTION : max_fraction
    min_fraction = hasfield(typeof(agent.config), :MIN_FUNDING_FRACTION) ?
        agent.config.MIN_FUNDING_FRACTION : 0.25
    # v3.5: high-conviction bets allowed to exceed the standard
    # max_fraction cap when both confidence AND signal_score are well above
    # baseline. Real founders sometimes deploy 10-30% of capital on
    # conviction plays — the standard 3.7% cap suppresses the right tail
    # that produces venture-scale ("unicorn") outcomes.
    max_high_conviction_fraction = hasfield(typeof(agent.config), :MAX_HIGH_CONVICTION_FRACTION) ?
        agent.config.MAX_HIGH_CONVICTION_FRACTION : 0.10
    high_conviction_threshold = hasfield(typeof(agent.config), :HIGH_CONVICTION_THRESHOLD) ?
        agent.config.HIGH_CONVICTION_THRESHOLD : 1.2

    max_invest = capital * max_fraction

    if isnothing(confidence) || isnothing(signal_score)
        # Legacy path (pre-v3.2): flat sizing. Preserved for internal callers
        # that don't have the evaluation context handy (tests, replay paths).
        invest_amount = min(max_invest, opportunity.capital_requirements)
    else
        # v3.2 confidence × signal sizing (mirrors Python):
        #   desired = capital · target_fraction · confidence · signal_score
        #   amount  = min(desired, max_invest)
        # Confidence range [0.15, 0.95] matches Python's perception clip.
        c = clamp(confidence, 0.15, 0.95)
        s = max(0.0, signal_score)
        desired = capital * target_fraction * c * s
        # v3.5: lift the max-cap when both confidence and signal_score are
        # high enough to qualify as a "high-conviction" bet. The product
        # (c × s) gates this so small confidence or weak signal can't
        # trigger oversizing.
        effective_max = if (c * s) > high_conviction_threshold
            capital * max_high_conviction_fraction
        else
            max_invest
        end
        invest_amount = min(desired, effective_max)

        # Minimum funding floor: if the opportunity needs a minimum stake and
        # the agent can afford it, don't under-size below the stake floor.
        required = opportunity.capital_requirements
        min_required = required > 0.0 ? required * min_fraction : 0.0
        if required > 0.0 && invest_amount < min_required && capital >= min_required
            invest_amount = min(max_invest, min_required)
        end

        invest_amount = min(invest_amount, capital)
        invest_amount = max(0.0, invest_amount)
    end

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

    # Propagate InformationSystem metadata so record_ai_signals! and
    # downstream analytics can see per-tier hallucination / confidence
    # patterns. Without this the uncertainty-telemetry layer read zero
    # hallucinations for every tier regardless of actual info quality.
    if !isnothing(ai_info)
        outcome["ai_contains_hallucination"] = ai_info.contains_hallucination
        outcome["ai_confidence"] = ai_info.confidence
        outcome["ai_actual_accuracy"] = ai_info.actual_accuracy
        outcome["ai_analysis_domain"] = ai_info.domain
    end

    # Flag for uncertainty.jl filters that count "AI-using" actions.
    # ai_level_used != "none" already implies AI use, but the consumer
    # at uncertainty.jl:1785 reads this as a separate boolean — propagate
    # explicitly so the filter doesn't always return zero.
    outcome["ai_used"] = get_ai_level(agent) != "none"

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
    market_conditions::Union{MarketConditions,Nothing} = nothing,
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

    # v2.9: charge per-use AI cost when an innovation attempt is made by a
    # per_use tier (Basic). Innovation consumes AI info_quality + info_breadth
    # downstream; earlier Basic agents got those effects for free. Mirrors
    # _execute_explore! and evaluate_portfolio_opportunities.
    ai_cfg = get(agent.config.AI_LEVELS, ai_tier, nothing)
    if !isnothing(ai_cfg) && ai_cfg.cost_type == "per_use" && ai_tier != "none"
        per_use_charge = Float64(ai_cfg.per_use_cost) * agent.config.AI_COST_INTENSITY
        if per_use_charge > 0
            set_capital!(agent, get_capital(agent) - per_use_charge)
        end
    end

    # =========================================================================
    # USE KNOWLEDGE-BASED INNOVATION SYSTEM (if available)
    # =========================================================================
    if !isnothing(innovation_engine)
        # v3.3.2: if the caller didn't pass a MarketConditions snapshot,
        # construct one from the current market state. Prior fallback used
        # Dict("regime"=>"normal"), which crashed attempt_innovation!
        # (signature: ::MarketConditions) with a MethodError. We have the
        # live market handle here — use it.
        mkt_cond = isnothing(market_conditions) ? get_market_conditions(market) : market_conditions

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

            # innovation_count counts innovations CREATED (success+failure), per
            # the field comment "total innovation attempts (not just successes)".
            # innovation_success_count is the success-only sub-counter.
            agent.innovation_count += 1

            if success
                agent.success_count += 1
                agent.innovation_success_count += 1  # v3.3.4 channel-specific
                # v3.3.4: wire knowledge learning. learn_from_success! adds the
                # innovation's components to the agent's knowledge set and (for
                # high-quality innovations) creates derived knowledge that
                # enters the shared base. This was dormant — the function was
                # defined in knowledge.jl:499 but never called, so the
                # knowledge pool stayed at its initial 12 components across
                # hundreds of innovations (reviewer probe). The agentic
                # scarcity mechanism (v3.3.3) depends on this: without new
                # components, scarcity signal saturates near zero as existing
                # components saturate in usage.
                innovation.success = true  # field is read by create_derived_knowledge
                learn_from_success!(innovation_engine.knowledge_base, agent.id, innovation)
            else
                agent.failure_count += 1
                agent.innovation_failure_count += 1  # v3.3.4 channel-specific
                innovation.success = false
                learn_from_failure!(innovation_engine.knowledge_base, agent.id, innovation; rng=agent.rng)
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
            # Propagate remaining Innovation fields so simulation.jl can fully
            # reconstruct the Innovation object without losing semantic content.
            outcome["innovation_sector"] = innovation.sector
            outcome["combination_signature"] = innovation.combination_signature

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

            # Count this as an innovation attempt (no Innovation object was
            # created, but the agent did select :innovate and burn R&D). Without
            # this increment, innovation_count was below action-selection counts
            # by ~10% (probe: 370 actions vs 337 counted innovations).
            agent.innovation_count += 1
            agent.innovation_failure_count += 1
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

        # innovation_count counts attempts; innovation_success_count is the
        # success-only sub-counter. Bug fix 2026-04-25.
        agent.innovation_count += 1

        if success
            base_return = rd_spend * agent.config.INNOVATION_SUCCESS_BASE_RETURN
            multiplier = rand(agent.rng, Uniform(agent.config.INNOVATION_SUCCESS_RETURN_MULTIPLIER...))
            innovation_return = base_return * multiplier

            set_capital!(agent, get_capital(agent) + innovation_return)
            agent.innovation_success_count += 1
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

            # Mirror the engine path's failure accounting (agents.jl:1029-1033).
            # innovation_count was already incremented above the if-branch;
            # this path just needs the failure-side counters.
            agent.innovation_failure_count += 1
            agent.failure_count += 1
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

        # v2.9: charge per-use AI cost when exploration uses AI info_quality /
        # info_breadth. Earlier Basic tier got AI exploration benefits for
        # free — per-use billing only fired in the investment-information
        # path. Mirrors the same pattern used in evaluate_portfolio_opportunities.
        if !isnothing(ai_config) && ai_config.cost_type == "per_use" && ai_tier != "none"
            per_use_charge = Float64(ai_config.per_use_cost) * agent.config.AI_COST_INTENSITY
            if per_use_charge > 0
                set_capital!(agent, get_capital(agent) - per_use_charge)
            end
        end

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
        # Tag the outcome so simulation.jl:459 (which checks for
        # exploration_type == "niche_discovery") actually fires create_niche_opportunity.
        # Previously only `created_niche` Bool was set; the trigger string was missing,
        # so niche creation was wired but never fired.
        if created_niche
            outcome["exploration_type"] = "niche_discovery"
            # Telemetry flag for uncertainty.jl niche-creation accounting.
            # The consumer at uncertainty.jl:733 also looks for "new_opportunity_id"
            # — that is set later by simulation.jl after market.create_niche_opportunity
            # succeeds (since the new opp's id isn't known until creation).
            outcome["created_opportunity"] = true
        end

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
    market_conditions::Union{MarketConditions,Nothing} = nothing
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

            # Release invested capital from opportunity capacity tracking
            # (total_invested should reflect current outstanding capital, not lifetime cumulative)
            if hasfield(typeof(opp), :total_invested)
                opp.total_invested = max(0.0, opp.total_invested - invested_amount)
            end

            success = ret_multiple >= 1.0
            if success
                agent.success_count += 1
                agent.investment_success_count += 1  # v3.3.4 channel-specific
            else
                agent.failure_count += 1
                agent.investment_failure_count += 1  # v3.3.4 channel-specific
            end

            # v3.4 [D]: append to tier-specific ROI history. Maintained as a
            # sliding window of the last 12 outcomes per tier so
            # choose_ai_level can read tier-specific roi rather than a
            # global signal that credited every tier for the agent's
            # overall ROI regardless of which tier generated it.
            tier_used = String(get(investment, "ai_level", get_ai_level(agent)))
            if haskey(agent.ai_learning.tier_roi_history, tier_used)
                push!(agent.ai_learning.tier_roi_history[tier_used], ret_multiple - 1.0)
                while length(agent.ai_learning.tier_roi_history[tier_used]) > 12
                    popfirst!(agent.ai_learning.tier_roi_history[tier_used])
                end
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

            invested_tier = get(investment, "ai_level", get_ai_level(agent))
            push!(matured_outcomes, Dict{String,Any}(
                "investment" => investment,
                "investment_amount" => invested_amount,
                "capital_returned" => capital_returned,
                "return_multiple" => ret_multiple,
                "success" => success,
                "round_matured" => round,
                "estimated_return" => estimated_return,
                "competition_at_maturity" => competition_level,
                # v2.7: emit tier-at-investment so simulation.jl:274's
                # update_tier_belief! attributes the outcome to the tier that
                # actually made the investment (not the agent's current tier).
                "ai_level" => invested_tier,
                # v2.8: emit ai_used for update_state_from_outcome! which
                # gates AI-trust learning on this key. Without it,
                # update_ai_trust! never fires for matured outcomes even when
                # the investment was made using AI.
                "ai_used" => invested_tier != "none",
            ))
        else
            push!(remaining_investments, investment)
        end
    end

    agent.active_investments = remaining_investments
    return matured_outcomes
end

# v3.5.13: apply_operational_costs! deleted — was unused. Production op-cost
# charging happens in simulation.jl's apply_operating_costs! via
# estimate_operational_costs(agent, market) which applies sector cost +
# OPS_COST_INTENSITY + active-investment competition multiplier + severity.
# The deleted helper bypassed all four; if anyone restored it, charges
# would diverge from the actual production path.

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

    # Recent activity — count actual invest/innovate/explore actions in the
    # last 20 rounds (true sliding window). Pre-v3.3.5 used
    # `min(20, length(action_history)) / 5.0`, which is cumulative: an agent
    # with 50 lifetime actions scored the same as one with 20, regardless of
    # when those actions occurred. Per-use cost in choose_ai_level reads
    # this metric — cumulative values biased per-use tier adoption estimates
    # (inactive agents looked as active as recently-active ones).
    window_size = 20
    recent_slice = agent.action_history[max(1, end - window_size + 1):end]
    recent_ai_using = count(a -> a in ("invest", "innovate", "explore"), recent_slice)
    metrics["recent_ai_activity"] = max(1.0, Float64(recent_ai_using) / 5.0)

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
        # Match actual charging: full monthly subscription cost per round
        # (see start_subscription_schedule! at agents.jl:~1345 — the monthly
        # cost is billed once per round, no per-use rider). Previously this
        # estimator returned `base_cost / amort_rounds + per_use * expected_calls`,
        # which told the planner the subscription was effectively free for
        # subscription-type tiers; then the biller charged the full monthly
        # cost each round. Agents adopted premium because "cheap" and then
        # got surprised by a 180x higher bill.
        return Float64(ai_config.cost) * cost_intensity
    elseif cost_type == "per_use"
        # Per-use tiers charge `per_use_cost` per call (basic: $3/call), not
        # the subscription `cost` field. Previously this used .cost, billing
        # basic agents 10× the documented per-use rate.
        return Float64(ai_config.per_use_cost) * cost_intensity * max(expected_calls, 0.0)
    end

    return 0.0
end

"""
Start a subscription schedule for a given AI level.

Charges the documented monthly subscription cost per round. The simulation
runs on a monthly cadence (config.jl: N_ROUNDS = 120 months = 10 years;
operational costs and discount rates throughout are explicitly monthly), so
each round = one month of subscription. Premium at \$3500/month therefore
charges \$3500 per round.

Corrected 2026-04-23: previous implementation multiplied by 3.0 ("convert
monthly to quarterly"), under the incorrect assumption that rounds were
quarters. That assumption conflicted with config-level documentation and
overcharged premium agents 3× (\$10,500 vs \$3,500 per round). The 3.0
multiplier has been removed.
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

    # Scale costs by AI_COST_INTENSITY (for robustness testing)
    cost_intensity = agent.config.AI_COST_INTENSITY
    per_round_cost = monthly_cost * cost_intensity

    # Set up subscription to charge EVERY round (no amortization)
    # Each round = one month per config.N_ROUNDS documentation
    agent.subscription_accounts[ai_level] = 1  # Active subscription
    agent.subscription_rates[ai_level] = per_round_cost  # Monthly cost per round
    agent.subscription_deferral_remaining[ai_level] = 0
end

"""
Charge one subscription installment for a given AI level.
Deducts the documented monthly cost from agent capital each round.
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

    # CHARGE THE SUBSCRIPTION - deduct documented monthly cost from capital
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

    # v2.8: removed 24-round credit-line grace period. Planning
    # (estimate_ai_cost, choose_ai_level) returns full monthly cost from
    # round 1; this path used to waive advanced/premium charges for the
    # first AI_CREDIT_LINE_ROUNDS=24 rounds, creating plan-vs-bill
    # inconsistency and distorting early-round dynamic tier selection.
    # Agents now pay subscription from round 1 matching the "series-A/B
    # rounds with 24-36 month runway" calibration (runway comes from
    # initial capital, not from free AI).
    for level in collect(keys(agent.subscription_accounts))
        total += charge_subscription_installment!(agent, level)
    end

    agent.last_subscription_charge = total
    return total
end

"""
Cancel the subscription schedule for a specific AI level.
"""
function cancel_subscription_schedule!(agent::EmergentAgent, ai_level::String)
    delete!(agent.subscription_accounts, ai_level)
    delete!(agent.subscription_rates, ai_level)
    delete!(agent.subscription_deferral_remaining, ai_level)
end

"""
Ensure subscription schedule is active for the given AI level.
Call this whenever an agent adopts or switches to a subscription-based AI tier.

For emergent-mode agents, this also cancels any previously-active subscription
to a DIFFERENT tier — an agent who drops from premium to advanced should stop
being charged for premium. Previously the agent accumulated simultaneous
subscriptions across every tier they ever used.
"""
function ensure_subscription_schedule!(
    agent::EmergentAgent,
    ai_level::String
)
    ai_config = get(agent.config.AI_LEVELS, ai_level, nothing)

    if isnothing(ai_config)
        return
    end

    # Cancel any active subscription to a DIFFERENT tier.
    for active_tier in collect(keys(agent.subscription_accounts))
        if active_tier != ai_level && get(agent.subscription_accounts, active_tier, 0) > 0
            cancel_subscription_schedule!(agent, active_tier)
        end
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

v3.4: asymmetric penalty — downgrades carry larger friction than upgrades.
Real downgrades involve data/integration loss and the social signal of
"giving up on the expensive tool"; upgrades have a learning-curve cost
but typically come with vendor onboarding. Loss-aversion literature
puts the asymmetry at roughly 2× — downgrade per step costs ~2× an
upgrade per step. Pre-v3.4 the cost was symmetric, treating
premium → none and none → premium as identical decisions.
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
    # v3.4: asymmetric base penalty
    is_downgrade = proposed_idx < current_idx
    per_step = is_downgrade ? 0.10 : 0.05
    base_penalty = per_step * distance + 0.03

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
Choose AI level based on Bayesian beliefs, performance metrics, peer
signals, and cost. v3.4 rewrite addresses 8 realism gaps documented in
the deep-debugging-review of the emergent path:

  [A] Cost matters: ref_scale was operating_cost*4 + cash_buffer*0.12,
      making cost_term a rounding error (~0.0006 for premium subscription).
      Now scaled to operating_cost so cost is felt as multiples of
      monthly burn rate. Premium cost_term ≈ 0.15 — same magnitude as
      trust_term.
  [B] Neutralize priors: hard-coded prior_map biased against premium
      (posterior 0.26 vs none 0.52). Removed the local override; AI
      learning profile already initializes neutral 0.50 priors for all
      tiers (DEFAULT_TIER_PRIORS in models.jl).
  [C] Status-quo tie-breaking: ties → keep current tier (was: lower-
      tier wins, anti-premium bias).
  [D] Tier-specific ROI: roi_term used a global roi_gain that credited
      every tier for the agent's overall performance. Now reads the
      tier's own roi_history (maintained in
      ai_learning.tier_roi_history by process_matured_investments!).
  [E] Sticky re-evaluation: only re-decide every AI_TIER_REVIEW_INTERVAL
      rounds (default 3 = quarterly). Real subscribers don't reconsider
      monthly.
  [G] Stronger peer effects: peer_distribution weight raised 0.05 → 0.20
      so observation of peers' tier choices meaningfully shifts beliefs.
  [H] Tier-specific trust: a low-base-trust agent should be MORE skeptical
      of expensive tiers (premium > advanced > basic). Modulates trust by
      tier-rank attenuation.

Skipped per scope decision: reserve haircut at high capital, larger noise,
budget heuristic, trial mechanism. (See REVISION_PLAN_v1.md.)
"""
function choose_ai_level(
    agent::EmergentAgent;
    neighbor_signals::Dict{String,Any} = Dict{String,Any}(),
    current_round::Int = 0
)::String
    # If fixed AI level, return it
    if !isnothing(agent.fixed_ai_level)
        return agent.fixed_ai_level
    end

    # [E] Sticky re-evaluation: defer the first tier review until
    # investments have matured (initial_freeze rounds), then review at
    # the ongoing cadence. v3.4 added review_interval; v3.4.2 added the
    # explicit initial-freeze separation:
    #
    #   * Without initial freeze: the first review fires at round 1-3
    #     while tier_roi_history is empty (no investments matured yet).
    #     Decision devolves into cost vs peer-signal cascades, locking
    #     in a winner-take-all distribution before any agent has tier-
    #     specific performance data.
    #   * With initial freeze (12 rounds = one investment maturity cycle):
    #     agents stay at their starting tier for the first cycle, build
    #     tier_roi data, then make an informed first switch.
    #
    # Pre-v3.4.2 sentinel `last_tier_review_round = -100` made the
    # gating arithmetic produce `rounds_since = current_round + 100`,
    # always > review_interval — freeze never applied. New logic:
    # sentinel <0 means never reviewed → wait until initial_freeze
    # rounds have elapsed; thereafter respect review_interval.
    review_interval = max(1, getfield_default(agent.config, :AI_TIER_REVIEW_INTERVAL, 3))
    initial_freeze = max(0, getfield_default(agent.config, :AI_TIER_INITIAL_FREEZE_ROUNDS, 12))
    if agent.last_tier_review_round < 0
        if current_round < initial_freeze
            return agent.current_ai_level
        end
    else
        rounds_since_review = current_round - agent.last_tier_review_round
        if rounds_since_review < review_interval && current_round > 0
            return agent.current_ai_level
        end
    end

    base_trust = clamp(get(agent.traits, "ai_trust", 0.5), 0.0, 1.0)
    uncertainty_tolerance = clamp(get(agent.traits, "uncertainty_tolerance", 0.5), 0.0, 1.0)
    avoidance = 1.0 - uncertainty_tolerance

    metrics = compute_ai_performance_metrics(agent)
    order = ["none", "basic", "advanced", "premium"]

    # [B] Neutralize priors. Removed the local prior_map override — read
    # only from ai_learning.tier_beliefs, which initializes uniformly at
    # alpha=2,beta=2 (posterior 0.5) for every tier (DEFAULT_TIER_PRIORS).
    # Pre-v3.4 the local prior_map biased premium's posterior to 0.26
    # before any data, hard-coding the conclusion the model was supposed
    # to discover from data.
    posterior_means = Dict{String,Float64}()
    for tier in order
        beliefs = agent.ai_learning.tier_beliefs
        if haskey(beliefs, tier)
            belief = beliefs[tier]
            total = belief["alpha"] + belief["beta"]
            posterior_means[tier] = total > 0 ? belief["alpha"] / total : 0.5
        else
            posterior_means[tier] = 0.5
        end
    end

    # Compute cost ratios. v3.4 [A]: rescale relative to operating_cost so
    # cost is felt at monthly-burn-rate scale rather than the huge ref_scale
    # = max(operating_cost*4, cash_buffer*0.12) used pre-v3.4.
    # v3.5.9: use agent.operating_cost_estimate (sector-derived) rather than
    # global BASE_OPERATIONAL_COST so the agent's perceived cost environment
    # matches the actual sector-specific charge in estimate_operational_costs.
    # v3.5.10: also apply OPS_COST_INTENSITY so refutation knob shifts both
    # actual burn and perceived burn coherently.
    # v3.5.13: also apply the same active-investment competition multiplier
    # estimate_operational_costs uses (1 + 0.2 × avg_competition). Earlier
    # choose_ai_level read raw operating_cost_estimate * intensity but
    # production charge added competition pressure on top. Replicating
    # the formula here (rather than calling estimate_operational_costs)
    # because choose_ai_level doesn't have market in scope and the
    # multiplier depends only on agent.active_investments.
    raw_burn = agent.operating_cost_estimate * agent.config.OPS_COST_INTENSITY
    if !isempty(agent.active_investments)
        comps = Float64[]
        for inv in agent.active_investments
            opp = get(inv, "opportunity", nothing)
            if !isnothing(opp) && opp isa Opportunity
                push!(comps, opp.competition)
            end
        end
        if !isempty(comps)
            raw_burn *= 1.0 + mean(comps) * 0.2
        end
    end
    operating_cost = max(raw_burn, 1.0)
    recent_activity = max(1.0, Float64(get(metrics, "recent_ai_activity", 1.0)))

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
            # v2.7 parity with actual billing (start_subscription_schedule!):
            # subscription charges the full monthly cost per round, so the
            # planner's per-round cost is base_cost (not base_cost/amort_horizon).
            # Earlier path returned ~$19/month for premium while biller charged
            # $3500/month — a 180× under-estimate that biased emergent-mode
            # tier choice toward premium.
            #
            # v3.3.3: billing doesn't charge a per-use rider for subscription
            # tiers (charge_subscription_installment! bills only the monthly
            # rate). Adding `per_use_cost * recent_activity` here made the
            # planner over-estimate subscription cost, biasing emergent tier
            # choice AWAY from premium/advanced by a factor of activity.
            base_cost
        elseif cost_type == "per_use"
            # Use the per_use_cost field directly. Earlier ternary preferred
            # base_cost (the subscription field) when nonzero, which charged
            # basic at $30/use instead of the documented $3/use.
            per_use_cost * recent_activity
        else
            per_use_cost * recent_activity
        end

        cost_ratios[tier] = total_cost / operating_cost
    end

    current_level = agent.current_ai_level
    if !(current_level in order)
        current_level = "none"
    end

    # Extract performance signals
    net_cash_total = Float64(get(metrics, "overall_cash_flow_total", 0.0))
    initial_equity = max(agent.resources.performance.initial_equity, 1.0)
    # v3.3.5: back out the already-paid subscription for THIS round when
    # computing capital_health (sunk cost shouldn't influence forward-
    # looking decision).
    effective_capital = get_capital(agent) + agent.last_subscription_charge
    capital_health = (effective_capital / initial_equity) - 1.0

    # Peer signals
    peer_roi_signal = Float64(get(neighbor_signals, "peer_roi_gap", 0.0))
    adoption_pressure = Float64(get(neighbor_signals, "ai_adoption_pressure", 0.0))
    peer_distribution = get(neighbor_signals, "ai_distribution", Dict{String,Float64}())

    # v3.5.5 audit: removed three asymmetric tier-specific tunings that had
    # no academic justification — `reserve_haircut` (basic 0/advanced 0.02/
    # premium 0.05), `tier_skepticism_factor` (none 0/basic 0.05/advanced
    # 0.10/premium 0.20), and the `cost_term *= 0.85` premium-only subsidy
    # below. These were tuned to prevent premium extinction in v3.4 but
    # represented unprincipled fingers-on-the-scale. Tier capability is now
    # captured solely by AI_LEVELS (cost, info_quality, info_breadth,
    # hallucination, per_use_cost) and AI_DOMAIN_CAPABILITIES.

    # Score each tier
    scores = Dict{String,Float64}()
    for tier in order
        posterior = posterior_means[tier]
        # Trust term: posterior expectation × base_trust (no tier attenuation)
        trust_term = posterior * base_trust

        # [D] Tier-specific ROI: use this tier's own roi_history when
        # available (maintained by process_matured_investments!). Falls
        # back to global ROI signal only if no tier-specific data exists.
        tier_roi_window = get(agent.ai_learning.tier_roi_history, tier, Float64[])
        tier_specific_roi = if !isempty(tier_roi_window)
            mean(tier_roi_window)
        else
            # No tier-specific data; use 0 (neutral) rather than letting
            # global ROI leak across tiers as in pre-v3.4
            0.0
        end
        roi_term = posterior * (0.30 * tier_specific_roi + 0.45 * tier_specific_roi)

        cash_term = 0.2 * clamp(net_cash_total / initial_equity, -1.5, 1.5)

        # [G] Stronger peer effects: peer_distribution weight raised
        # 0.05 → 0.10. Peer adoption is a meaningful social-proof signal
        # for tier choice; the prior weight made it a near-rounding-error.
        # (Initial v3.4 tried 0.20 — too strong, produced winner-take-all
        # herding into basic. 0.10 keeps the signal meaningful without
        # cascading.)
        peer_term = 0.10 * peer_roi_signal + 0.15 * adoption_pressure +
                   0.10 * Float64(get(peer_distribution, tier, 0.0))

        # Experience with tier
        tier_experience = count(==(tier), agent.ai_tier_history)
        learning_relief = clamp(log1p(tier_experience) * 0.02, 0.0, 0.12)

        # [A] Make cost matter. Coefficient raised from 0.25 to 0.6 AND
        # ref_scale tightened from huge to operating_cost (above), so
        # cost_term magnitude scales with monthly burn rate. Premium
        # cost_term ≈ 0.6 × 0.156 = 0.09 — comparable to trust_term but
        # not dominating. (Initial v3.4 tried 1.5 — premium went extinct
        # in 1 round; 0.6 lets cost matter meaningfully without forcing
        # the corner solution.)
        cost_term = 0.6 * get(cost_ratios, tier, 0.0) * (1.0 - learning_relief)

        switch_penalty = max(0.0, compute_ai_switch_penalty(agent, current_level, tier) - learning_relief)

        # Gumbel noise for stochastic selection
        noise = -log(-log(rand(agent.rng))) * 0.02  # Gumbel(0,1) * 0.02

        total_score = (
            trust_term +
            roi_term +
            cash_term +
            0.25 * capital_health +
            peer_term -
            cost_term -
            switch_penalty -
            0.05 * avoidance +
            noise
        )

        scores[tier] = total_score
    end

    # Select tier with highest score. v3.4 [C]: status-quo tie-breaking —
    # ties go to the agent's CURRENT tier rather than the lowest-index
    # tier (which had a systematic anti-premium bias). If current tier
    # isn't in the score set, fall back to lowest-index.
    best_tier = current_level
    best_score = scores[current_level]
    for (i, tier) in enumerate(order)
        score = scores[tier]
        if score > best_score
            best_score = score
            best_tier = tier
        end
    end

    # Update tracking
    push!(agent.ai_tier_history, best_tier)
    agent.current_ai_level = best_tier
    agent.last_tier_review_round = current_round  # v3.4 [E] sticky re-eval

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
    market_conditions::MarketConditions,
    perception::Dict{String,Any};
    ai_level::String = "none",
    info_system::Union{InformationSystem,Nothing} = nothing
)::Float64
    if isempty(opportunities)
        return 0.0
    end

    # Get AI tier parameters - NOW USED for opportunity evaluation
    ai_config = get(agent.config.AI_LEVELS, ai_level, nothing)
    info_quality = isnothing(ai_config) ? 0.25 : Float64(ai_config.info_quality)
    info_breadth = isnothing(ai_config) ? 0.20 : Float64(ai_config.info_breadth)

    # Get best opportunity score with AI-enhanced evaluation.
    # Use the agent's PERCEIVED return (estimated_return from InformationSystem)
    # when available, falling back to latent only for diagnostic paths.
    # Previously this loop scored against opp.latent_return_potential directly,
    # which meant the should-I-invest-at-all decision was ground-truth-aware
    # even after the v2 ranking fix — same bypass, different code path.
    #
    # Per-use AI cost deduction (mirrors evaluate_portfolio_opportunities):
    # for per_use tiers (Basic), each get_information call is a chargeable AI
    # invocation. Earlier this loop hit get_information without deducting,
    # so utility evaluation gave Basic tier free AI while portfolio ranking
    # charged it — split bill from the same evaluator.
    ai_config_pe = get(agent.config.AI_LEVELS, ai_level, nothing)
    cost_intensity_pe = agent.config.AI_COST_INTENSITY
    per_use_charge = if !isnothing(ai_config_pe) && ai_config_pe.cost_type == "per_use"
        Float64(ai_config_pe.per_use_cost) * cost_intensity_pe
    else
        0.0
    end

    max_score = 0.0
    avg_score = 0.0
    for opp in opportunities
        # v2.7: propagate the full Information triple (est_return, est_uncertainty,
        # confidence, contains_hallucination) so evaluate_opportunity_basic can use
        # the tier-aware uncertainty + confidence in its scoring. Earlier only
        # estimated_return was used, understating premium's info-quality advantage.
        est_return = opp.latent_return_potential
        est_uncertainty = nothing
        conf = nothing
        contains_halluc = false
        if !isnothing(info_system)
            # Only deduct per-use cost on cache MISS. Re-reading a cached info
            # object is not a new API call — the agent already paid for it.
            cached_before = haskey(info_system.information_cache,
                                   (opp.id, ai_level, agent.id))
            info = get_information(info_system, opp, ai_level;
                                   agent_id=agent.id, rng=agent.rng)
            if !cached_before && per_use_charge > 0
                set_capital!(agent, get_capital(agent) - per_use_charge)
            end
            est_return = info.estimated_return
            est_uncertainty = info.estimated_uncertainty
            conf = info.confidence
            contains_halluc = info.contains_hallucination
        end

        base_score = evaluate_opportunity_basic(agent, opp, market_conditions;
                                                estimated_return=est_return,
                                                estimated_uncertainty=est_uncertainty,
                                                confidence=conf,
                                                contains_hallucination=contains_halluc)

        # AI tier advantage now reflects perceived (not latent) return.
        noise_reduction = info_quality * 0.3
        ai_adjusted_score = base_score * (1.0 + noise_reduction * (est_return - 1.0))

        breadth_bonus = info_breadth * 0.1 * (1.0 - opp.complexity)
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

    # v3.3: Ignorance adjustment — replaces a sigmoid centered at actor_unc=1.5
    # whose flat-tail operating range (agents typically perceive 0.05–0.25)
    # produced ~1.7% utility variance across tiers even though perception
    # itself differed by 4.5×. Linear response in [0.2, 1.0] now makes the
    # tier-differentiated Knightian-ignorance perception actually translate
    # to utility — premium's low perceived ignorance earns a real utility
    # premium, none's higher perceived ignorance a real penalty.
    ignorance_adjustment = clamp(1.0 - actor_unc * 0.8, 0.2, 1.0)

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
    # v3.3: competitive_recursion coefficient raised from 0.06 → 0.25. The
    # environment correctly reports massive recursion under premium (~0.98)
    # because high-tier agents identify the same top opportunities and pile
    # in — the paper's equilibrium-trap mechanism. Prior coefficient gave
    # only 6% utility drag at max recursion; agents saw the crowded niches
    # but didn't behaviorally pivot away. Stronger coefficient now feeds
    # the trap into the pre-decision utility, not just the post-maturation
    # return via v3.1 convexity. Calibrate K_sat downstream.
    value -= 0.25 * recursive_unc

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
    market_conditions::MarketConditions,
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
    market_conditions::MarketConditions,
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
Basic opportunity evaluation.

By default the agent's *perceived* return (`estimated_return`) drives the
score. When `estimated_return` is omitted, falls back to
`opportunity.latent_return_potential` (the hidden ground truth) — kept for
diagnostic / test paths that don't want AI-tier noise injected.

Corrected 2026-04-23: previously this function ALWAYS used
`opportunity.latent_return_potential`, which meant agents of every AI tier
ranked opportunities by the same hidden ground truth and the
information-quality mechanism the paper claimed to test was bypassed
entirely. The `estimated_return` parameter (set by
`evaluate_portfolio_opportunities` from `InformationSystem.get_information`)
is now the canonical input.
"""
function evaluate_opportunity_basic(
    agent::EmergentAgent,
    opportunity::Opportunity,
    market_conditions::MarketConditions;
    estimated_return::Union{Float64,Nothing} = nothing,
    estimated_uncertainty::Union{Float64,Nothing} = nothing,
    confidence::Union{Float64,Nothing} = nothing,
    contains_hallucination::Bool = false,
)::Float64
    # Expected profit margin: use AI-tier-aware estimate when provided,
    # else fall back to the hidden latent value (diagnostic path only).
    expected_return = isnothing(estimated_return) ? opportunity.latent_return_potential : estimated_return
    expected_margin = expected_return - 1.0

    # v2.7: Use the AI-tier-aware uncertainty estimate when provided. Premium
    # AI doesn't just produce a better return estimate — it also produces a
    # better assessment of HOW uncertain that estimate is. High-quality AI
    # narrows perceived uncertainty; low-quality AI (or no AI) falls back to
    # opp.complexity as a crude proxy. Ignoring estimated_uncertainty in
    # pre-v2.7 scoring understated premium's info-quality advantage.
    perceived_uncertainty = if !isnothing(estimated_uncertainty)
        # Blend AI-estimated uncertainty with opp.complexity (which captures
        # intrinsic difficulty independent of AI assessment).
        0.6 * estimated_uncertainty + 0.4 * opportunity.complexity
    else
        opportunity.complexity
    end
    uncertainty_adjusted = expected_margin * (1.0 - perceived_uncertainty * 0.5)

    score = uncertainty_adjusted * (1.0 + get(agent.traits, "uncertainty_tolerance", 0.5))

    # v2.7: confidence weighting. The agent's conviction in their estimate
    # scales the score's magnitude — low-confidence signals should have
    # less influence on decisions (uncertainty_tolerance captures the agent's
    # general appetite for ambiguity, confidence captures how much they trust
    # THIS specific AI call). Default 0.5 when no confidence info available.
    conf = isnothing(confidence) ? 0.5 : clamp(confidence, 0.05, 0.99)
    # At conf = 0.5, no change. At conf = 0.99, +15% score. At conf = 0.05, -25%.
    score *= 0.75 + 0.5 * conf

    # v3.5.13 — hidden-truth-leakage fix. Earlier this branch read
    # contains_hallucination (the hidden ground truth) and only applied the
    # discount when the information actually was a hallucination. Probe:
    # "score 1.767 without flag vs mean 0.993 with flag" — agents had
    # privileged access to the truth. Real-world agents can't know in
    # advance whether a specific AI call hallucinated; they apply expected-
    # hallucination discounts based on (a) their tier's known hallucination
    # rate and (b) their own analytical ability.
    #
    # Replace the truth-flag check with a tier-aware EXPECTED discount that
    # fires on EVERY evaluation regardless of whether this specific call
    # was actually hallucinated. Realized outcomes still punish agents that
    # acted on hallucinated information through the maturity path.
    ai_tier = get_ai_level(agent)
    tier_capabilities = get(agent.config.AI_DOMAIN_CAPABILITIES, ai_tier, nothing)
    if !isnothing(tier_capabilities)
        # Average hallucination rate across the four cognitive domains for this tier
        domain_halls = [get(tier_capabilities, dom, nothing)
                        for dom in ["market_analysis", "technical_assessment",
                                    "uncertainty_evaluation", "innovation_potential"]]
        rates = [d.hallucination_rate for d in domain_halls if !isnothing(d)]
        expected_hall_rate = isempty(rates) ? 0.1 : mean(rates)
    else
        expected_hall_rate = 0.1
    end
    analytical = Float64(get(agent.traits, "analytical_ability", 0.5))
    # Expected discount = expected_hall_rate × (cost of an undetected hall × undetected fraction).
    # Per-call discount magnitude calibrated to roughly match the prior
    # truth-flag-detected discount in expectation, while applying uniformly.
    expected_discount = expected_hall_rate * (0.75 - 0.35 * analytical)
    score *= clamp(1.0 - expected_discount, 0.5, 1.0)

    # Regime multiplier
    regime = market_conditions.regime
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
    # v2.9: INVERSION FIX. Earlier `adjusted *= 1.0 - (tolerance * competition * 0.5)`
    # meant high-tolerance agents received LARGER penalties, which is backwards:
    # uncertainty_tolerance captures appetite for ambiguity, so high-tolerance
    # agents should be LESS bothered by competition. Corrected to (1.0 - tolerance).
    if opportunity.competition > 0.5
        tolerance = get(agent.traits, "uncertainty_tolerance", 0.5)
        intolerance = 1.0 - tolerance
        adjusted *= 1.0 - (intolerance * opportunity.competition * 0.5)
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
    market_conditions::MarketConditions,
    perception::Dict{String,Any};
    ai_level::String = "none",
    info_system::Union{InformationSystem,Nothing} = nothing,
    early_signals::Union{Dict{String,Int},Nothing} = nothing,
    signal_weight::Float64 = 0.15
)::Vector{Dict{String,Any}}
    if isempty(opportunities)
        return Dict{String,Any}[]
    end

    evaluations = Dict{String,Any}[]
    opp_pool = copy(opportunities)

    # ─────────────────────────────────────────────────────────────────
    # Compute AI-tier-aware per-opportunity estimates (corrected 2026-04-23)
    #
    # Previously this block computed `estimated_returns` then sorted the opp
    # pool by estimate, and the evaluation loop below used `latent_return_potential`
    # via `evaluate_opportunity_basic` — so the tier-noisy estimates were
    # discarded and ALL tiers ranked by the same hidden ground truth. The
    # information-quality mechanism the paper claims to test was bypassed.
    #
    # Now: when `info_system` is provided (production path), use
    # `get_information(opp, ai_level)` to produce hallucination-injected,
    # tier-noisy estimates from the InformationSystem. When `info_system` is
    # nothing (test/diagnostic path), use the inline tier-noise model below
    # which preserves the previous numerical behavior except that the
    # estimates now ACTUALLY drive `final_score` via `evaluate_opportunity_basic`.
    #
    # Herding emerges NATURALLY from information quality:
    # - Premium agents have low noise → rank similar opportunities as best
    # - None agents have high noise → rank diverse opportunities as best
    # NOVELTY INVERSION: Premium's advantage inverts for novel opportunities
    # (the more important the innovation, the less predictable it is — Rescher)
    # ─────────────────────────────────────────────────────────────────

    estimated_returns = Dict{String,Float64}()
    # Stash Information objects for per-opp propagation into invest-action outcomes.
    # record_ai_signals! (uncertainty.jl:347) reads ai_contains_hallucination /
    # ai_confidence / ai_actual_accuracy from the action dict; earlier these
    # fields were never populated so hallucination telemetry saw zero regardless
    # of tier.
    info_cache_local = Dict{String,Information}()

    if !isnothing(info_system)
        # Production path: full InformationSystem with hallucination chain.
        # For per_use tiers (Basic), deduct the documented per-use AI cost
        # for each get_information call. Previously per_use_cost was only
        # referenced in cost-estimation/planning helpers but never actually
        # deducted from agent.capital — Basic tier ($30/use documented)
        # effectively got free AI.
        ai_config_pe = get(agent.config.AI_LEVELS, ai_level, nothing)
        cost_intensity_pe = agent.config.AI_COST_INTENSITY
        per_use_charge = if !isnothing(ai_config_pe) && ai_config_pe.cost_type == "per_use"
            Float64(ai_config_pe.per_use_cost) * cost_intensity_pe
        else
            0.0
        end

        for opp in opp_pool
            # Deduct per-use cost on cache MISS only (see calculate_investment_utility
            # for rationale — prevents double-charging across utility + ranking).
            cached_before = haskey(info_system.information_cache,
                                   (opp.id, ai_level, agent.id))
            info = get_information(info_system, opp, ai_level;
                                   agent_id=agent.id, rng=agent.rng)
            info_cache_local[opp.id] = info
            est = info.estimated_return
            if !isnothing(early_signals)
                signal_count = get(early_signals, opp.id, 0)
                if signal_count > 0
                    signal_boost = signal_count * signal_weight * 0.1
                    est *= (1.0 + signal_boost)
                end
            end
            estimated_returns[opp.id] = est

            if !cached_before && per_use_charge > 0
                set_capital!(agent, get_capital(agent) - per_use_charge)
            end
        end
    elseif length(opp_pool) > 1
        # Diagnostic/test path: inline tier-noise estimate (no hallucinations)
        ai_config = get(agent.config.AI_LEVELS, ai_level, nothing)
        info_quality = isnothing(ai_config) ? 0.25 : Float64(ai_config.info_quality)
        base_noise_scale = 0.5 * (1.0 - info_quality)
        inversion_factor = getfield_default(agent.config, :NOVELTY_NOISE_INVERSION_FACTOR, 0.4)
        for opp in opp_pool
            opp_novelty = hasfield(typeof(opp), :novelty_score) ? opp.novelty_score : 0.0
            novelty_noise_penalty = opp_novelty * info_quality * inversion_factor
            effective_noise = base_noise_scale + novelty_noise_penalty
            noise = randn(agent.rng) * effective_noise
            est = opp.latent_return_potential + noise
            if !isnothing(early_signals)
                signal_count = get(early_signals, opp.id, 0)
                if signal_count > 0
                    signal_boost = signal_count * signal_weight * 0.1
                    est *= (1.0 + signal_boost)
                end
            end
            estimated_returns[opp.id] = est
        end
    else
        # Single-opportunity edge case: no tier noise needed
        for opp in opp_pool
            estimated_returns[opp.id] = opp.latent_return_potential
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Evaluate each opportunity using the AI-tier-aware estimate
    # ─────────────────────────────────────────────────────────────────

    for opp in opp_pool
        est_return = get(estimated_returns, opp.id, opp.latent_return_potential)
        # v2.7: pull the full Information object when available so scoring uses
        # AI-tier-aware uncertainty + confidence + hallucination flags.
        est_unc = nothing
        conf = nothing
        contains_halluc = false
        if haskey(info_cache_local, opp.id)
            inf = info_cache_local[opp.id]
            est_unc = inf.estimated_uncertainty
            conf = inf.confidence
            contains_halluc = inf.contains_hallucination
        end
        base_score = evaluate_opportunity_basic(agent, opp, market_conditions;
                                                estimated_return=est_return,
                                                estimated_uncertainty=est_unc,
                                                confidence=conf,
                                                contains_hallucination=contains_halluc)
        final_score = apply_uncertainty_adjustments(agent, base_score, opp, perception)

        eval_record = Dict{String,Any}(
            "opportunity" => opp,
            "final_score" => final_score,
            "ai_level_used" => ai_level,
            "estimated_return" => est_return,
            "competition_at_evaluation" => hasfield(typeof(opp), :competition) ? opp.competition : 0.0,
        )
        if haskey(info_cache_local, opp.id)
            inf = info_cache_local[opp.id]
            eval_record["ai_info"] = inf
            eval_record["ai_contains_hallucination"] = inf.contains_hallucination
            eval_record["ai_confidence"] = inf.confidence
            eval_record["ai_actual_accuracy"] = inf.actual_accuracy
        end
        push!(evaluations, eval_record)
    end

    # Single sort by final_score (which now reflects AI-tier-aware estimate)
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
    market_conditions::MarketConditions,
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
        # v3.4: pass current_round for sticky re-evaluation gating
        choose_ai_level(agent; neighbor_signals=neighbor_signals, current_round=round_num)
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

    # Calculate utilities for each action. v3.5.11: use estimate_operational_costs
    # so the agent's perceived cost reflects (a) sector-specific base via
    # agent.operating_cost_estimate, (b) OPS_COST_INTENSITY refutation knob,
    # and (c) competition multiplier from agent's own active investments —
    # all three components of what's actually charged. (The round-local
    # severity multiplier in simulation.jl is bounded [0.7, 1.9] and is a
    # market-wide stochastic variable not visible at decision time, so it's
    # not modeled in perceived cost; this represents agents reacting to
    # expected burn, with realized burn fluctuating ±50% via severity.)
    estimated_cost = estimate_operational_costs(agent, market)

    invest_utility = calculate_investment_utility(agent, opportunities, market_conditions, perception; ai_level=ai_level, info_system=info_system)
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
            info_system=info_system,
            early_signals=early_signals,
            signal_weight=signal_weight
        )
        if !isempty(evals)
            best_eval = evals[1]
            best_opp = best_eval["opportunity"]
            estimated_return = get(best_eval, "estimated_return", best_opp.latent_return_potential)
            ai_info_for_invest = get(best_eval, "ai_info", nothing)
            # v3.2: extract confidence (from perception) and signal_score
            # (from evaluation) so _execute_invest! can size the bet by them.
            decision_conf = Float64(get(perception, "decision_confidence", 0.5))
            final_score = Float64(get(best_eval, "final_score", 0.0))
            execute_action!(agent, "invest", market, round_num;
                opportunity=best_opp,
                estimated_return=estimated_return,
                ai_info=ai_info_for_invest isa Information ? ai_info_for_invest : nothing,
                innovation_engine=innovation_engine,
                market_conditions=market_conditions,
                uncertainty_perception=perception,
                confidence=decision_conf,
                signal_score=final_score)
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
    # Alias for the consumer at uncertainty.jl:753 which reads
    # action["perception_at_decision"] when computing knowledge gaps.
    # Without this alias the knowledge_gaps merge always saw an empty Dict.
    outcome["perception_at_decision"] = perception

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
    # v2.8: capital credit REMOVED from this function. The caller is
    # responsible for applying capital changes BEFORE calling this — in the
    # matured-investment path process_matured_investments! already credits
    # capital_returned at agents.jl:~1151. Earlier (v2.7) this function
    # credited capital AGAIN, causing double-counting on every matured
    # investment. The function's job is purely state evolution (AI-trust
    # learning + trait evolution), not bookkeeping.

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
    # Use the sector-specific cost initialized in the agent constructor
    # (agents.jl:402, sector midpoint of operational_cost_range), not the
    # global BASE_OPERATIONAL_COST. Earlier this hardcoded the global
    # value, so e.g. tech agents (initialized at $37,500) were charged
    # $22,500 like services — sector heterogeneity in operating costs
    # was effectively dead in the production charge path.
    base_cost = agent.operating_cost_estimate

    # Apply OPS_COST_INTENSITY scaling (refutation knob; default 1.0).
    # This is THE op-cost knob: refutation OPS_COST_50% sets it to 0.5,
    # OPS_COST_25% to 0.25, etc. Earlier refutations overrode
    # BASE_OPERATIONAL_COST, but production reads agent.operating_cost_estimate
    # (sector-derived at construction), so those overrides were no-ops on
    # actual burn.
    base_cost *= agent.config.OPS_COST_INTENSITY

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
