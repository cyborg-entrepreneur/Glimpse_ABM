"""
Market environment for GlimpseABM.jl

This module implements the market dynamics that create the context for
entrepreneurial action under Knightian uncertainty.

Port of: glimpse_abm/market.py
"""

using Random
using Statistics
using Distributions

# ============================================================================
# MARKET ENVIRONMENT
# ============================================================================

"""
Dynamic market with emergent properties and realistic return dynamics.

This struct manages the market context in which entrepreneurial agents
operate, including opportunity generation, macroeconomic regime
transitions, supply-demand clearing, and crowding dynamics.
"""
mutable struct MarketEnvironment
    config::EmergentConfig
    n_agents::Int
    opportunities::Vector{Opportunity}
    opportunities_by_sector::Dict{String,Vector{Opportunity}}
    opportunity_map::Dict{String,Opportunity}
    current_round::Int
    market_regime::String
    volatility::Float64
    trend::Float64
    market_momentum::Float64
    price_history::Vector{Float64}
    volume_history::Vector{Float64}
    competition_levels::Dict{String,Float64}
    total_investment_by_round::Vector{Float64}
    failure_rate_by_round::Vector{Float64}
    new_ventures_by_round::Vector{Int}
    innovations::Vector{Innovation}
    exploration_activity::Int
    sectors::Vector{String}
    sector_profiles::Dict{String,SectorProfile}
    regime_return_multiplier::Float64
    regime_failure_multiplier::Float64
    tier_invest_share::Dict{String,Float64}
    sector_clearing_index::Dict{String,Float64}
    aggregate_clearing_ratio::Float64
    sector_demand_adjustments::Dict{String,Dict{String,Float64}}
    branch_params::Dict{String,Dict{String,Any}}
    crowding_metrics::Dict{String,Float64}
    rng::AbstractRNG
end

function MarketEnvironment(
    config::EmergentConfig;
    rng::AbstractRNG = Random.default_rng()
)
    initialize!(config)  # Ensure SECTORS is populated

    market = MarketEnvironment(
        config,
        config.N_AGENTS,
        Opportunity[],
        Dict{String,Vector{Opportunity}}(),
        Dict{String,Opportunity}(),
        0,
        "normal",
        config.MARKET_VOLATILITY,
        0.0,
        0.0,
        Float64[],
        Float64[],
        Dict{String,Float64}(),
        Float64[],
        Float64[],
        Int[],
        Innovation[],
        0,
        config.SECTORS,
        config.SECTOR_PROFILES,
        1.0,
        1.0,
        Dict("none" => 0.0, "basic" => 0.0, "advanced" => 0.0, "premium" => 0.0),
        Dict{String,Float64}(),
        1.0,
        Dict{String,Dict{String,Float64}}(),
        Dict{String,Dict{String,Any}}(),
        Dict{String,Float64}(),
        rng
    )

    # Initialize branch parameters for each sector
    for (sector, profile) in market.sector_profiles
        _initialize_branch!(market, sector, profile)
    end

    # Initialize opportunities
    _initialize_opportunities!(market)

    return market
end

"""
Initialize branch parameters for a sector.
"""
function _initialize_branch!(market::MarketEnvironment, name::String, profile::SectorProfile)
    log_mu = profile.return_log_mu
    log_sigma = clamp(profile.return_log_sigma, 0.05, 1.0)
    failure_mean = (profile.failure_range[1] + profile.failure_range[2]) / 2
    failure_sigma = clamp(mean(profile.failure_volatility_range), 0.01, 0.5)

    market.branch_params[name] = Dict{String,Any}(
        "name" => name,
        "root" => split(name, "_")[1],
        "profile" => profile,
        "log_mu" => log_mu,
        "log_sigma" => log_sigma,
        "failure_mean" => failure_mean,
        "failure_sigma" => failure_sigma,
        "depth" => 0
    )
end

"""
Sample opportunity characteristics from a branch.
"""
function _sample_branch_characteristics(
    market::MarketEnvironment,
    branch_name::String;
    quality_roll::Union{Float64,Nothing} = nothing
)::Tuple{Float64,Float64,Float64,Int}
    params = get(market.branch_params, branch_name, nothing)
    if isnothing(params)
        # Create default params if branch doesn't exist
        profile = get(market.sector_profiles, branch_name, first(values(market.sector_profiles)))
        _initialize_branch!(market, branch_name, profile)
        params = market.branch_params[branch_name]
    end

    profile = params["profile"]
    log_mu = params["log_mu"]
    log_sigma = params["log_sigma"]

    quality_roll = isnothing(quality_roll) ? rand(market.rng) : quality_roll

    # Quality adjustments
    if quality_roll > 0.92
        log_mu += 0.18
        log_sigma *= 0.9
    elseif quality_roll < 0.28
        log_mu -= 0.12
        log_sigma *= 1.08
    end
    log_sigma = clamp(log_sigma, 0.05, 1.0)

    # Sample latent return
    latent_return = exp(log_mu + log_sigma * randn(market.rng))
    latent_return = clamp(latent_return, profile.return_range[1], profile.return_range[2])

    # Sample failure probability
    failure_mean = params["failure_mean"]
    failure_sigma = params["failure_sigma"]
    if quality_roll > 0.85
        failure_mean *= 0.92
    elseif quality_roll < 0.3
        failure_mean *= 1.08
    end

    latent_failure = failure_mean + failure_sigma * randn(market.rng)
    latent_failure = clamp(latent_failure, profile.failure_range[1], profile.failure_range[2])

    # Sample capital and maturity
    capital_req = rand(market.rng, Uniform(profile.capital_range...))
    maturity = rand(market.rng, profile.maturity_range[1]:profile.maturity_range[2])

    return latent_return, latent_failure, capital_req, maturity
end

"""
Create a realistic opportunity in a given sector.
"""
function _create_realistic_opportunity(
    market::MarketEnvironment,
    opp_id::String,
    sector::String;
    innovator_capability::Float64 = 0.5
)::Opportunity
    quality_roll = rand(market.rng)
    latent_return, latent_failure, capital_req, maturity = _sample_branch_characteristics(
        market, sector; quality_roll=quality_roll
    )

    # Apply regime modifiers
    latent_return *= market.regime_return_multiplier
    latent_failure *= market.regime_failure_multiplier

    # Clamp values
    latent_return = clamp(latent_return, 0.5, 25.0)
    latent_failure = clamp(latent_failure, 0.1, 0.95)

    return Opportunity(
        id=opp_id,
        latent_return_potential=latent_return,
        latent_failure_potential=latent_failure,
        complexity=rand(market.rng, Uniform(0.3, 1.2)),
        discovered=true,
        discovery_round=0,
        config=market.config,
        sector=sector,
        capital_requirements=capital_req,
        time_to_maturity=maturity
    )
end

"""
Initialize the initial set of opportunities.
"""
function _initialize_opportunities!(market::MarketEnvironment)
    n_opportunities = get_scaled_opportunities(market.config, market.n_agents)

    for i in 1:n_opportunities
        sector = rand(market.rng, market.sectors)
        opp = _create_realistic_opportunity(market, "initial_$(sector)_$(i)", sector)
        push!(market.opportunities, opp)
        market.opportunity_map[opp.id] = opp

        # Index by sector
        if !haskey(market.opportunities_by_sector, sector)
            market.opportunities_by_sector[sector] = Opportunity[]
        end
        push!(market.opportunities_by_sector[sector], opp)
    end
end

"""
Transition to next market regime based on transition probabilities.
"""
function _transition_regime!(market::MarketEnvironment)
    transitions = market.config.MACRO_REGIME_TRANSITIONS
    if !haskey(transitions, market.market_regime)
        return
    end

    probs = transitions[market.market_regime]
    regimes = collect(keys(probs))
    weights = [probs[r] for r in regimes]

    # Normalize weights
    total = sum(weights)
    if total > 0
        weights ./= total
    else
        return
    end

    # Sample next regime
    cumsum_weights = cumsum(weights)
    r = rand(market.rng)
    idx = clamp(searchsortedfirst(cumsum_weights, r), 1, length(regimes))
    market.market_regime = regimes[idx]

    # Update modifiers
    market.regime_return_multiplier = get(market.config.MACRO_REGIME_RETURN_MODIFIERS, market.market_regime, 1.0)
    market.regime_failure_multiplier = get(market.config.MACRO_REGIME_FAILURE_MODIFIERS, market.market_regime, 1.0)
    market.trend = get(market.config.MACRO_REGIME_TREND, market.market_regime, 0.0)
    market.volatility = get(market.config.MACRO_REGIME_VOLATILITY, market.market_regime, 0.2)
end

"""
Step the market environment forward one round.
"""
function step!(
    market::MarketEnvironment,
    round::Int,
    agent_actions::Vector{Dict{String,Any}},
    innovations::Vector{Innovation};
    matured_outcomes::Vector{Dict{String,Any}} = Dict{String,Any}[]
)::Dict{String,Any}
    market.current_round = round
    empty!(market.sector_demand_adjustments)

    # Record innovations
    append!(market.innovations, innovations)

    # Calculate total investment
    invest_actions = filter(a -> get(a, "action", "") == "invest", agent_actions)
    total_investment = isempty(invest_actions) ? 0.0 : sum(
        Float64(get(a, "amount", get(a, "capital_deployed", 0.0)))
        for a in invest_actions
    )
    push!(market.total_investment_by_round, total_investment)

    # Update competition levels (uses COMPETITION_SCALE_FACTOR for robustness testing)
    # Normalize competition delta by population size to keep consistent pressure across scales
    # Reference population is 100 agents; with 1000 agents, delta is 10x smaller
    reference_population = 100.0
    actual_population = Float64(market.config.N_AGENTS)
    population_normalized_delta = 0.1 * (reference_population / actual_population)

    for action in agent_actions
        if get(action, "action", "") == "invest"
            opp = get(action, "chosen_opportunity_obj", nothing)
            if !isnothing(opp) && isa(opp, Opportunity)
                update_opportunity_competition!(market, opp, population_normalized_delta)
            end
        elseif get(action, "action", "") == "explore"
            market.exploration_activity += 1
        end
    end

    # Update regime with some probability
    if rand(market.rng) < 0.1
        _transition_regime!(market)
    end

    # Calculate crowding metrics
    total_actions = length(agent_actions)
    if total_actions > 0
        action_counts = Dict{String,Int}()
        for action in agent_actions
            act_type = get(action, "action", "maintain")
            action_counts[act_type] = get(action_counts, act_type, 0) + 1
        end

        share_invest = get(action_counts, "invest", 0) / total_actions
        share_innovate = get(action_counts, "innovate", 0) / total_actions
        share_explore = get(action_counts, "explore", 0) / total_actions
        share_maintain = get(action_counts, "maintain", 0) / total_actions

        crowding_index = share_invest^2 + share_innovate^2 + share_explore^2 + share_maintain^2

        # Calculate AI usage share (proportion of agents using AI)
        ai_usage_share = count(a -> get(a, "ai_level_used", "none") != "none", agent_actions) / max(1, total_actions)

        market.crowding_metrics = Dict(
            "crowding_index" => crowding_index,
            "share_invest" => share_invest,
            "share_innovate" => share_innovate,
            "share_explore" => share_explore,
            "share_maintain" => share_maintain,
            "ai_usage_share" => ai_usage_share
        )
    end

    # Calculate tier investment shares
    invest_actions = [a for a in agent_actions if get(a, "action", "") == "invest"]
    if !isempty(invest_actions)
        tier_counts = Dict("none" => 0, "basic" => 0, "advanced" => 0, "premium" => 0)
        for action in invest_actions
            tier = lowercase(string(get(action, "ai_level_used", "none")))
            if haskey(tier_counts, tier)
                tier_counts[tier] += 1
            end
        end
        n_invest = length(invest_actions)
        for tier in keys(tier_counts)
            market.tier_invest_share[tier] = tier_counts[tier] / n_invest
        end
    end

    return get_market_conditions(market)
end

"""
Get current market conditions dictionary.
"""
function get_market_conditions(market::MarketEnvironment)::Dict{String,Any}
    return Dict{String,Any}(
        "regime" => market.market_regime,
        "volatility" => market.volatility,
        "trend" => market.trend,
        "momentum" => market.market_momentum,
        "regime_return_multiplier" => market.regime_return_multiplier,
        "regime_failure_multiplier" => market.regime_failure_multiplier,
        "n_opportunities" => length(market.opportunities),
        "exploration_activity" => market.exploration_activity,
        "tier_invest_share" => market.tier_invest_share,
        "sector_clearing_index" => market.sector_clearing_index,
        "aggregate_clearing_ratio" => market.aggregate_clearing_ratio,
        "crowding_metrics" => market.crowding_metrics,
        "round" => market.current_round
    )
end

"""
Get available opportunities for an agent.
"""
function get_available_opportunities(market::MarketEnvironment)::Vector{Opportunity}
    return [opp for opp in market.opportunities if opp.discovered]
end

"""
Add a new opportunity to the market.
"""
function add_opportunity!(market::MarketEnvironment, opp::Opportunity)
    push!(market.opportunities, opp)
    market.opportunity_map[opp.id] = opp

    if !isnothing(opp.sector)
        if !haskey(market.opportunities_by_sector, opp.sector)
            market.opportunities_by_sector[opp.sector] = Opportunity[]
        end
        push!(market.opportunities_by_sector[opp.sector], opp)
    end
end

# ============================================================================
# SECTOR-SPECIFIC COMPETITION INTENSITY
# ============================================================================

"""
Get sector-specific competition intensity based on Census HHI data.

Competition intensity values calibrated from Census Bureau Economic Census:
- Tech: 1.2 (HHI 1500-2500, moderate concentration)
- Retail: 0.7 (HHI 500-1000, fragmented)
- Service: 0.9 (HHI 800-1500, moderately fragmented)
- Manufacturing: 1.4 (HHI 1800-3000, concentrated)

Returns base intensity scaled by global COMPETITION_SCALE_FACTOR for robustness testing.
"""
function get_sector_competition_intensity(market::MarketEnvironment, sector::String)::Float64
    profile = get(market.sector_profiles, sector, nothing)
    base_intensity = if !isnothing(profile) && hasproperty(profile, :competition_intensity)
        profile.competition_intensity
    else
        1.0  # Default intensity
    end

    # Apply global competition scale factor for robustness testing
    return base_intensity * market.config.COMPETITION_SCALE_FACTOR
end

"""
Apply sector-specific competition intensity to opportunity competition updates.
"""
function update_opportunity_competition!(market::MarketEnvironment, opp::Opportunity, delta::Float64)
    # Counterfactual mode: disable competition dynamics
    if market.config.DISABLE_COMPETITION_DYNAMICS
        opp.competition = 0.0
        return
    end

    # Normal mode: update competition based on agent activity
    #
    # Competition is modeled as a "competitive pressure index" that can exceed 1.0.
    # This is analogous to Herfindahl-Hirschman Index (HHI) which ranges 0-10,000,
    # or price impact in market microstructure which scales with trade size.
    #
    # Interpretation:
    # - competition = 0.0: No other investors (monopoly-like position)
    # - competition = 1.0: Normal competitive pressure (~10 investors)
    # - competition = 3.0: Severe crowding (~30 investors, 3x normal)
    #
    # This unbounded representation is essential for the AI Information Paradox:
    # - When 33 Premium AI agents pile into ONE opportunity, competition ~3.3
    # - When 15 Human agents invest (spread across opportunities), competition ~1.5
    # - The penalty difference reflects the real economic cost of crowding
    #
    # Theoretical basis: Cournot competition (profits ∝ 1/n²) and
    # market microstructure (Kyle 1985: price impact ∝ √order_flow)
    intensity = get_sector_competition_intensity(market, opp.sector)
    opp.competition = max(0.0, opp.competition + delta * intensity)
end

# ============================================================================
# BRANCH FEEDBACK AND DEMAND ADJUSTMENTS
# ============================================================================

"""
Modifier effects for branch parameters.
"""
const MODIFIER_EFFECTS = Dict{String,Dict{String,Float64}}(
    "premium" => Dict("log_mu" => 0.12, "log_sigma" => -0.05, "failure_mean" => -0.05),
    "budget" => Dict("log_mu" => -0.15, "log_sigma" => 0.02, "failure_mean" => 0.04),
    "digital" => Dict("log_mu" => 0.05, "log_sigma" => -0.02, "failure_mean" => -0.02),
    "local" => Dict("log_mu" => -0.05, "log_sigma" => 0.03, "failure_mean" => 0.01),
    "specialized" => Dict("log_mu" => 0.08, "log_sigma" => 0.04, "failure_mean" => -0.01),
    "sustainable" => Dict("log_mu" => 0.04, "log_sigma" => 0.0, "failure_mean" => -0.02)
)

"""
Apply feedback to branch parameters based on observed ROI.
"""
function apply_branch_feedback!(market::MarketEnvironment, branch_name::String, mean_roi::Float64)
    params = get(market.branch_params, branch_name, nothing)
    if isnothing(params)
        return
    end

    profile = params["profile"]
    rate = market.config.BRANCH_FEEDBACK_RATE
    feedback = clamp(mean_roi, -1.0, 1.0) * rate

    # Adjust log_mu based on ROI
    log_range = (log(profile.return_range[1]), log(profile.return_range[2]))
    params["log_mu"] = clamp(params["log_mu"] + feedback, log_range[1], log_range[2])

    # Adjust failure_mean inversely
    failure_range = profile.failure_range
    params["failure_mean"] = clamp(
        params["failure_mean"] - feedback * 0.5,
        failure_range[1], failure_range[2]
    )
end

"""
Create child branch parameters with drift.
"""
function create_child_params(
    market::MarketEnvironment,
    name::String,
    parent_params::Dict{String,Any};
    modifier::Union{String,Nothing} = nothing
)::Dict{String,Any}
    root_sector = parent_params["root"]
    profile = market.sector_profiles[root_sector]

    # Apply drift
    mu_drift = market.config.BRANCH_LOG_MEAN_DRIFT * randn(market.rng)
    sigma_drift = market.config.BRANCH_LOG_SIGMA_DRIFT * randn(market.rng)
    failure_drift = market.config.BRANCH_FAILURE_DRIFT * randn(market.rng)

    # Apply modifier effects
    if !isnothing(modifier)
        effect = get(MODIFIER_EFFECTS, lowercase(modifier), Dict{String,Float64}())
        mu_drift += get(effect, "log_mu", 0.0)
        sigma_drift += get(effect, "log_sigma", 0.0)
        failure_drift += get(effect, "failure_mean", 0.0)
    end

    log_mu = parent_params["log_mu"] + mu_drift
    log_sigma = max(0.05, parent_params["log_sigma"] + sigma_drift)

    log_mu = clamp(log_mu, log(profile.return_range[1]), log(profile.return_range[2]))
    failure_mean = clamp(
        parent_params["failure_mean"] + failure_drift,
        profile.failure_range[1], profile.failure_range[2]
    )
    failure_sigma = clamp(
        parent_params["failure_sigma"] * rand(market.rng, Uniform(0.85, 1.15)),
        0.01, 0.5
    )

    return Dict{String,Any}(
        "name" => name,
        "root" => root_sector,
        "profile" => profile,
        "log_mu" => log_mu,
        "log_sigma" => log_sigma,
        "failure_mean" => failure_mean,
        "failure_sigma" => failure_sigma,
        "depth" => get(parent_params, "depth", 0) + 1
    )
end

"""
Ensure a branch exists, creating it if necessary.
"""
function ensure_branch!(market::MarketEnvironment, branch_name::String)::Dict{String,Any}
    if haskey(market.branch_params, branch_name)
        return market.branch_params[branch_name]
    end

    parts = split(branch_name, "_")
    base = String(parts[1])

    if !haskey(market.branch_params, base)
        profile = get(market.sector_profiles, base, nothing)
        if isnothing(profile)
            error("Unknown sector '$base'")
        end
        _initialize_branch!(market, base, profile)
    end

    parent = base
    for depth in 2:length(parts)
        child = join(parts[1:depth], "_")
        if haskey(market.branch_params, child)
            parent = child
            continue
        end
        parent_params = market.branch_params[parent]
        new_params = create_child_params(market, child, parent_params; modifier=String(parts[depth]))
        market.branch_params[child] = new_params
        parent = child
    end

    return market.branch_params[branch_name]
end

"""
Get demand adjustments for a sector based on clearing metrics.
"""
function get_demand_adjustments(market::MarketEnvironment, sector::String)::Dict{String,Float64}
    if haskey(market.sector_demand_adjustments, sector)
        return market.sector_demand_adjustments[sector]
    end

    # Get clearing ratio
    clearing_ratio = get(market.sector_clearing_index, sector, market.aggregate_clearing_ratio)
    if !isfinite(clearing_ratio)
        clearing_ratio = market.aggregate_clearing_ratio
    end

    # Crowding calculations
    crowd_threshold = market.config.RETURN_DEMAND_CROWDING_THRESHOLD
    flow_share = 0.0
    if haskey(market.crowding_metrics, "share_invest")
        flow_share = get(market.crowding_metrics, "share_invest", 0.25)
    end

    crowd_excess = max(0.0, flow_share - crowd_threshold)
    crowd_relief = max(0.0, crowd_threshold - flow_share)

    penalty_strength = market.config.RETURN_DEMAND_CROWDING_PENALTY

    # Convex crowding penalty and relief
    return_penalty = 1.0 - penalty_strength * crowd_excess^2
    return_penalty *= 1.0 + 0.35 * crowd_relief^2

    failure_pressure = 1.0 + market.config.FAILURE_DEMAND_PRESSURE * crowd_excess^2
    failure_pressure *= 1.0 - 0.2 * crowd_relief^2

    # Supply/demand adjustments
    if clearing_ratio > 1.0
        undersupply = clearing_ratio - 1.0
        return_penalty *= 1.0 + 0.45 * undersupply
        failure_pressure *= 1.0 - 0.35 * undersupply
    else
        oversupply_gap = 1.0 - clearing_ratio
        return_penalty *= 1.0 - 0.7 * oversupply_gap
        failure_pressure *= 1.0 + 0.6 * oversupply_gap
    end

    adjustments = Dict{String,Float64}(
        "return" => return_penalty,
        "failure" => failure_pressure
    )
    market.sector_demand_adjustments[sector] = adjustments
    return adjustments
end

# ============================================================================
# MARKET DYNAMICS
# ============================================================================

"""
Update market dynamics based on agent actions.
"""
function update_market_dynamics!(
    market::MarketEnvironment,
    agent_actions::Vector{Dict{String,Any}},
    total_investment::Float64,
    ai_invest_share::Float64
)
    # Calculate average investment quality
    invest_actions = filter(a -> get(a, "action", "") == "invest", agent_actions)
    qualities = Float64[]
    for action in invest_actions
        expected = get(action, "expected_return", nothing)
        if isnothing(expected)
            details = get(action, "chosen_opportunity_details", Dict())
            expected = get(details, "estimated_return", nothing)
        end
        if !isnothing(expected) && isfinite(expected)
            push!(qualities, Float64(expected))
        end
    end

    avg_quality = isempty(qualities) ? 1.0 : mean(qualities)
    quality_adjustment = (avg_quality - 1.0) * 0.1

    # AI activity adjustment (paradox: high AI share can degrade realized quality)
    ai_activity = clamp(ai_invest_share - 0.5, -0.5, 0.5)
    quality_adjustment -= ai_activity * 0.05

    # Investment activity
    investment_activity = clamp(total_investment / (market.n_agents * 250_000), 0, 2)

    # Update momentum
    market.market_momentum = market.market_momentum * 0.8 +
        0.2 * (investment_activity + quality_adjustment + ai_activity * 0.1)

    # Track boom streak
    boom_streak = get(market.crowding_metrics, "boom_streak", 0.0)
    if market.market_momentum > 1.5
        boom_streak += 1
    else
        boom_streak = max(0, boom_streak - 1)
    end
    market.crowding_metrics["boom_streak"] = boom_streak

    # Dynamic black swan probability
    base_prob = market.config.BLACK_SWAN_PROBABILITY
    exponent = market.config.BOOM_TAIL_UNCERTAINTY_EXPONENT
    dynamic_black_swan = base_prob * (exponent^max(0, boom_streak - 1)) * (1 + ai_invest_share * 0.3)

    crowding_index = get(market.crowding_metrics, "crowding_index", 0.25)

    signals = Dict{String,Float64}(
        "investment_activity" => investment_activity,
        "ai_activity" => ai_activity,
        "crowding_index" => crowding_index,
        "quality_adjustment" => quality_adjustment,
        "momentum" => market.market_momentum
    )

    update_macro_regime!(market, signals, dynamic_black_swan)
end

"""
Update macro regime based on market signals.
"""
function update_macro_regime!(
    market::MarketEnvironment,
    signals::Dict{String,Float64},
    black_swan_prob::Float64
)
    states = market.config.MACRO_REGIME_STATES
    transition_map = market.config.MACRO_REGIME_TRANSITIONS
    current_state = haskey(transition_map, market.market_regime) ? market.market_regime : states[1]

    # Base transition probabilities
    base_probs = [get(get(transition_map, current_state, Dict()), state, 0.0) for state in states]
    if sum(base_probs) <= 0
        base_probs = ones(length(states))
    end

    adjustments = zeros(length(states))
    idx_map = Dict(state => i for (i, state) in enumerate(states))

    invest = get(signals, "investment_activity", 1.0)
    momentum = get(signals, "momentum", 0.0)
    crowding = get(signals, "crowding_index", 0.25)
    ai_activity = get(signals, "ai_activity", 0.0)
    quality = get(signals, "quality_adjustment", 0.0)

    # Growth/boom boost
    if invest > 1.1 || momentum > 0.8 || quality > 0
        boost = 0.05 * max(invest - 1.1, 0.0) + 0.04 * max(momentum - 0.8, 0.0) + 0.03 * max(quality, 0.0)
        if haskey(idx_map, "growth")
            adjustments[idx_map["growth"]] += boost
        end
        if haskey(idx_map, "boom")
            adjustments[idx_map["boom"]] += boost * 0.6
        end
    end

    # Recession/crisis drag
    if invest < 0.9 || momentum < -0.6
        drag = 0.05 * max(0.9 - invest, 0.0) + 0.04 * max(-0.6 - momentum, 0.0)
        if haskey(idx_map, "recession")
            adjustments[idx_map["recession"]] += drag
        end
        if haskey(idx_map, "crisis")
            adjustments[idx_map["crisis"]] += drag * 0.5
        end
    end

    # Crowding effects
    crowd_threshold = market.config.RETURN_DEMAND_CROWDING_THRESHOLD
    if crowding > crowd_threshold
        penalty = 0.05 * (crowding - crowd_threshold)
        if haskey(idx_map, "recession")
            adjustments[idx_map["recession"]] += penalty
        end
        if haskey(idx_map, "crisis")
            adjustments[idx_map["crisis"]] += penalty * 0.6
        end
    else
        relief = 0.03 * (crowd_threshold - crowding)
        if haskey(idx_map, "normal")
            adjustments[idx_map["normal"]] += relief
        end
        if haskey(idx_map, "growth")
            adjustments[idx_map["growth"]] += relief * 0.5
        end
    end

    # AI activity can increase crisis risk
    if ai_activity > 0.3 && haskey(idx_map, "crisis")
        adjustments[idx_map["crisis"]] += 0.02 * ai_activity
    end

    # Black swan effect
    if haskey(idx_map, "crisis")
        adjustments[idx_map["crisis"]] += black_swan_prob
    end

    # Compute final probabilities
    probs = base_probs .+ adjustments
    probs = max.(probs, 0.0)
    total = sum(probs)

    if !isfinite(total) || total <= 0
        probs = ones(length(states)) / length(states)
    else
        probs ./= total
    end

    # Sample new state
    cumsum_probs = cumsum(probs)
    r = rand(market.rng)
    idx = clamp(searchsortedfirst(cumsum_probs, r), 1, length(states))
    new_state = states[idx]

    market.market_regime = new_state

    # Update modifiers
    market.regime_return_multiplier = get(market.config.MACRO_REGIME_RETURN_MODIFIERS, new_state, 1.0)
    market.regime_failure_multiplier = get(market.config.MACRO_REGIME_FAILURE_MODIFIERS, new_state, 1.0)

    target_trend = get(market.config.MACRO_REGIME_TREND, new_state, 0.0)
    target_volatility = get(market.config.MACRO_REGIME_VOLATILITY, new_state, market.volatility)

    market.trend = clamp(0.7 * market.trend + 0.3 * target_trend, -1.0, 1.0)
    market.volatility = clamp(0.6 * market.volatility + 0.4 * target_volatility, 0.05, 1.0)

    # Reset boom streak on crisis
    if new_state == "crisis"
        market.crowding_metrics["boom_streak"] = 0.0
        market.market_momentum = min(market.market_momentum, -1.5)
    end
end

# ============================================================================
# OPPORTUNITY LIFECYCLE MANAGEMENT
# ============================================================================

"""
Manage opportunity creation, aging, and removal.
"""
function manage_opportunities!(
    market::MarketEnvironment,
    round::Int,
    opportunity_demand::Dict{String,Int},
    total_investment::Float64
)
    # Calculate target capacity based on agent count
    # NOTE: Previously this code incorrectly modified market.n_agents (agent count)
    # to track opportunity targets. Now we just use target_cap directly.
    target_cap = get_scaled_opportunities(market.config, market.n_agents)

    # Calculate desired new opportunities
    young_opps = count(o -> o.age < 5, market.opportunities)
    poisson_new = rand(market.rng, Poisson(max(0.0, length(market.opportunities) * 0.02)))
    desired_new = max(0, target_cap - length(market.opportunities) + poisson_new - young_opps)
    push!(market.new_ventures_by_round, desired_new)

    # Create new opportunities with sector balancing
    for _ in 1:desired_new
        # Calculate sector probabilities (favor underrepresented sectors)
        sector_counts = Dict{String,Int}()
        for opp in market.opportunities
            sector_counts[opp.sector] = get(sector_counts, opp.sector, 0) + 1
        end

        total_opps = length(market.opportunities)
        if total_opps > 0
            probs = Float64[]
            for sector in market.sectors
                count = get(sector_counts, sector, 0)
                push!(probs, (total_opps - count) / total_opps)
            end
            if sum(probs) > 0
                probs ./= sum(probs)
            else
                probs = ones(length(market.sectors)) / length(market.sectors)
            end

            # Sample sector
            cumsum_probs = cumsum(probs)
            r = rand(market.rng)
            idx = clamp(searchsortedfirst(cumsum_probs, r), 1, length(market.sectors))
            sector = market.sectors[idx]
        else
            sector = rand(market.rng, market.sectors)
        end

        new_opp = _create_realistic_opportunity(
            market, "market_$(round)_$(rand(market.rng, 1000:9999))", sector
        )
        new_opp.discovery_round = round
        push!(market.opportunities, new_opp)
        market.opportunity_map[new_opp.id] = new_opp
        _index_opportunity!(market, new_opp)
    end

    # Age all opportunities and decay competition
    # Competition decays as investors exit or market conditions normalize
    for opp in market.opportunities
        opp.age += 1
        opp.competition = max(0.0, opp.competition * 0.9)
    end

    # Remove dead opportunities
    dead_opps = Opportunity[]
    for opp in market.opportunities
        if opp.competition < 0.01 && opp.age > 20
            push!(dead_opps, opp)
        elseif opp.lifecycle_stage == "declining" && opp.competition < 0.05
            push!(dead_opps, opp)
        elseif opp.age > 10 && get(opportunity_demand, opp.id, 0) == 0
            push!(dead_opps, opp)
        end
    end

    for opp in dead_opps
        _remove_opportunity!(market, opp)
    end

    # Additional cleanup of very old declining opportunities
    filter!(market.opportunities) do opp
        !(opp.lifecycle_stage == "declining" && opp.age > 50 && opp.competition < 0.1)
    end

    _rebuild_sector_index!(market)
end

"""
Index an opportunity by sector.
"""
function _index_opportunity!(market::MarketEnvironment, opp::Opportunity)
    sector = isnothing(opp.sector) ? "unknown" : opp.sector
    if !haskey(market.opportunities_by_sector, sector)
        market.opportunities_by_sector[sector] = Opportunity[]
    end
    push!(market.opportunities_by_sector[sector], opp)
end

"""
Remove an opportunity from the market.
"""
function _remove_opportunity!(market::MarketEnvironment, opp::Opportunity)
    filter!(o -> o.id != opp.id, market.opportunities)

    # Remove from sector index
    sector = isnothing(opp.sector) ? "unknown" : opp.sector
    if haskey(market.opportunities_by_sector, sector)
        filter!(o -> o.id != opp.id, market.opportunities_by_sector[sector])
    end

    # Remove from map
    delete!(market.opportunity_map, opp.id)
end

"""
Rebuild the sector index from scratch.
"""
function _rebuild_sector_index!(market::MarketEnvironment)
    empty!(market.opportunities_by_sector)
    for opp in market.opportunities
        _index_opportunity!(market, opp)
    end
    market.opportunity_map = Dict(opp.id => opp for opp in market.opportunities)
end

# ============================================================================
# CLEARING METRICS
# ============================================================================

"""
Estimate sector capacity for capital absorption.
"""
function estimate_sector_capacity(market::MarketEnvironment)::Dict{String,Float64}
    capacity = Dict{String,Float64}()
    for opp in market.opportunities
        sector = isnothing(opp.sector) ? "unknown" : opp.sector
        req = opp.capital_requirements
        maturity = max(1, opp.time_to_maturity)
        capacity[sector] = get(capacity, sector, 0.0) + req / maturity
    end

    if isempty(capacity)
        capacity["unknown"] = Float64(market.n_agents) * 50_000.0
    end

    return capacity
end

"""
Update clearing metrics from agent actions.
"""
function update_clearing_metrics!(market::MarketEnvironment, agent_actions::Vector{Dict{String,Any}})
    sector_flows = Dict{String,Float64}()
    tier_flows = Dict{String,Float64}()

    for action in agent_actions
        if get(action, "action", "") != "invest"
            continue
        end

        # Extract sector
        details = get(action, "chosen_opportunity_details", Dict())
        sector = get(details, "sector", nothing)
        if isnothing(sector)
            opp = get(action, "chosen_opportunity_obj", nothing)
            if !isnothing(opp) && isa(opp, Opportunity)
                sector = opp.sector
            end
        end
        sector = isnothing(sector) ? "unknown" : string(sector)

        # Extract capital
        capital = Float64(get(action, "capital_deployed", get(action, "amount", 0.0)))
        if capital <= 0
            continue
        end

        sector_flows[sector] = get(sector_flows, sector, 0.0) + capital

        # Track by tier
        tier = normalize_ai_label(get(action, "ai_level_used", "none"))
        tier_flows[tier] = get(tier_flows, tier, 0.0) + capital
    end

    # Calculate clearing index
    capacity = estimate_sector_capacity(market)
    clearing_index = Dict{String,Float64}()
    for sector in union(keys(capacity), keys(sector_flows))
        supply = max(get(capacity, sector, 0.0), 1.0)
        demand = get(sector_flows, sector, 0.0)
        clearing_index[sector] = demand / supply
    end

    total_capacity = max(sum(values(capacity)), 1.0)
    total_demand = sum(values(sector_flows))

    market.sector_clearing_index = clearing_index
    market.aggregate_clearing_ratio = total_demand / total_capacity

    # Update tier invest shares
    if total_demand > 0
        for (tier, flow) in tier_flows
            market.tier_invest_share[tier] = flow / total_demand
        end
    end

    # Clear demand adjustments cache
    empty!(market.sector_demand_adjustments)
end

# ============================================================================
# OPPORTUNITY ACCESS AND PERCEPTION
# ============================================================================

"""
Get opportunities available to an agent based on AI level and traits.
Note: agent is typed as Any to avoid circular dependency with agents.jl
"""
function get_opportunities_for_agent(
    market::MarketEnvironment,
    agent
)::Vector{Opportunity}
    # Start with discovered opportunities
    visible_opps = [opp for opp in market.opportunities if opp.discovered]

    # AI-based discovery of undiscovered opportunities
    ai_level = get_ai_level(agent)
    profile = get(market.config.AI_LEVELS, ai_level, market.config.AI_LEVELS["none"])
    info_quality = Float64(get(profile, "info_quality", 0.0))
    info_breadth = Float64(get(profile, "info_breadth", 0.0))
    ai_factor = 1.0 + info_quality * 2.0 + info_breadth * 1.5
    ai_factor = clamp(ai_factor, 0.8, 4.5)

    base_exploration = get(agent.traits, "exploration_tendency", 0.0)
    base_awareness = get(agent.traits, "market_awareness", 0.0)
    knowledge_map = agent.resources.knowledge

    base_prob = base_exploration * 0.2 + base_awareness * 0.3 + info_breadth * 0.25 + info_quality * 0.1

    for opp in market.opportunities
        if opp.discovered
            continue
        end

        sector_knowledge = get(knowledge_map, opp.sector, 0.0)
        if sector_knowledge <= 0
            continue
        end

        discovery_prob = base_prob * (1.0 + sector_knowledge * 2.0) * ai_factor
        if rand(market.rng) < discovery_prob
            opp.discovered = true
            opp.discovery_round = market.current_round
            push!(visible_opps, opp)
        end
    end

    # Filter and rank by agent perspective
    return get_perceived_opportunities(market, visible_opps, ai_level, agent)
end

"""
Filter and rank opportunities based on agent's perspective.
"""
function get_perceived_opportunities(
    market::MarketEnvironment,
    all_opportunities::Vector{Opportunity},
    ai_level::String,
    agent  # EmergentAgent - untyped to avoid circular dependency
)::Vector{Opportunity}
    if isempty(all_opportunities)
        return Opportunity[]
    end

    profile = get(market.config.AI_LEVELS, ai_level, market.config.AI_LEVELS["none"])
    info_quality = Float64(get(profile, "info_quality", 0.0))
    info_breadth = Float64(get(profile, "info_breadth", 0.0))

    base_visible = (3 + info_breadth * 60.0) * (1.0 + info_quality)
    max_visible = Int(max(3, min(length(all_opportunities), round(base_visible))))

    knowledge_map = agent.resources.knowledge

    # Score opportunities by relevance
    scores = Float64[]
    for opp in all_opportunities
        relevance = 0.5
        if !isnothing(opp.sector)
            relevance += get(knowledge_map, opp.sector, 0.0) * 0.5
        end
        push!(scores, relevance)
    end

    # Get top opportunities
    perm = sortperm(scores, rev=true)
    take = min(max_visible, length(all_opportunities))
    selected_indices = perm[1:take]

    return [all_opportunities[i] for i in selected_indices]
end

# ============================================================================
# NICHE AND INNOVATION OPPORTUNITIES
# ============================================================================

"""
Create a niche opportunity discovered via exploration.
"""
function create_niche_opportunity(
    market::MarketEnvironment,
    niche_id::String,
    discoverer_id::Int,
    round_num::Int
)::Opportunity
    # Parse niche_id
    if occursin("_", niche_id)
        parts = rsplit(niche_id, "_"; limit=2)
        base_sector = String(parts[1])
        modifier = String(parts[2])
    else
        base_sector = niche_id
        modifier = "standard"
    end

    modifier_effects = Dict{String,Dict{String,Float64}}(
        "premium" => Dict("return_mult" => 1.15, "uncertainty_mult" => 0.85, "capital_mult" => 1.8),
        "budget" => Dict("return_mult" => 0.85, "uncertainty_mult" => 1.15, "capital_mult" => 0.6),
        "sustainable" => Dict("return_mult" => 1.05, "uncertainty_mult" => 0.95, "capital_mult" => 1.1),
        "digital" => Dict("return_mult" => 1.08, "uncertainty_mult" => 1.0, "capital_mult" => 0.8),
        "local" => Dict("return_mult" => 0.92, "uncertainty_mult" => 0.85, "capital_mult" => 0.7),
        "specialized" => Dict("return_mult" => 1.12, "uncertainty_mult" => 1.05, "capital_mult" => 1.3)
    )

    mods = get(modifier_effects, modifier, Dict("return_mult" => 1.0, "uncertainty_mult" => 1.0, "capital_mult" => 1.0))

    branch_name = niche_id
    ensure_branch!(market, branch_name)
    latent_return, latent_failure, capital_req, maturity = _sample_branch_characteristics(market, branch_name)

    latent_return *= mods["return_mult"]
    latent_failure *= mods["uncertainty_mult"]
    capital_req *= mods["capital_mult"]

    demand_adjustments = get_demand_adjustments(market, branch_name)
    latent_return *= demand_adjustments["return"] * market.regime_return_multiplier
    latent_failure *= demand_adjustments["failure"] * market.regime_failure_multiplier

    niche_opp = Opportunity(
        id="niche_$(branch_name)_$(round_num)_$(rand(market.rng, 1000:9999))",
        latent_return_potential=clamp(latent_return, 0.5, 25.0),
        latent_failure_potential=clamp(latent_failure, 0.1, 0.95),
        complexity=rand(market.rng, Uniform(0.4, 0.8)),
        discovered=false,
        discovery_round=round_num,
        creator_id=discoverer_id,
        sector=branch_name,
        capital_requirements=capital_req,
        time_to_maturity=maturity,
        config=market.config
    )

    push!(market.opportunities, niche_opp)
    market.opportunity_map[niche_opp.id] = niche_opp
    _index_opportunity!(market, niche_opp)

    return niche_opp
end

"""
Record innovation outcome and update opportunity characteristics.
"""
function record_innovation_outcome!(
    market::MarketEnvironment,
    opportunity::Opportunity,
    success::Bool,
    return_achieved::Float64
)
    if success
        opportunity.market_impact = return_achieved
        intrinsic_gain = clamp(return_achieved, 0.2, 3.5)

        # Adjust return potential
        opportunity.latent_return_potential = clamp(
            opportunity.latent_return_potential + 0.05 * intrinsic_gain,
            0.15, 4.0
        )
    else
        opportunity.market_impact = 0.0
        opportunity.latent_failure_potential = clamp(
            opportunity.latent_failure_potential * 1.05,
            0.05, 0.99
        )
    end
end

"""
Spawn a derivative opportunity from a successful innovation.
"""
function spawn_opportunity_from_innovation!(
    market::MarketEnvironment,
    innovation::Innovation,
    cash_multiple::Float64
)
    sector = isnothing(innovation.sector) ? "tech" : innovation.sector
    # Handle potential Nothing values with defaults
    scarcity = clamp(something(innovation.scarcity, 0.5), 0.0, 1.0)
    novelty = clamp(something(innovation.novelty, 0.5), 0.0, 1.0)

    scarcity_scale = 0.85 + scarcity * 0.7
    novelty_scale = 0.9 + novelty * 0.6

    intrinsic_multiple = clamp(
        1.05 + (cash_multiple - 1.0) * 0.6,
        0.8, 3.2
    ) * scarcity_scale * novelty_scale * (0.9 + clamp(innovation.quality, 0.0, 1.5) * 0.2)

    derived_multiplier = clamp(intrinsic_multiple, 0.65, 3.8)
    base_failure_signal = clamp(0.35 - scarcity * 0.2 - novelty * 0.1, 0.04, 0.9)

    branch_name = sector
    latent_return, latent_failure, capital_req, maturity = _sample_branch_characteristics(market, branch_name)
    latent_return *= derived_multiplier
    latent_failure = clamp(0.5 * latent_failure + 0.5 * base_failure_signal, 0.05, 0.95)

    demand_adjustments = get_demand_adjustments(market, branch_name)
    latent_return *= demand_adjustments["return"] * market.regime_return_multiplier
    latent_failure *= demand_adjustments["failure"] * market.regime_failure_multiplier

    opp_id = "spawn_$(innovation.id)_$(rand(market.rng, 1000:9999))"

    # Novelty score inherited from innovation (novel opportunities are harder to evaluate)
    innov_novelty = clamp(something(innovation.novelty, 0.5), 0.0, 1.0)

    # Capacity based on config
    base_capacity = hasfield(typeof(market.config), :OPPORTUNITY_BASE_CAPACITY) ?
        market.config.OPPORTUNITY_BASE_CAPACITY : 500000.0
    capacity_variance = hasfield(typeof(market.config), :OPPORTUNITY_CAPACITY_VARIANCE) ?
        market.config.OPPORTUNITY_CAPACITY_VARIANCE : 0.3
    opp_capacity = base_capacity * (1.0 + (rand(market.rng) - 0.5) * 2 * capacity_variance)

    opportunity = Opportunity(
        id=opp_id,
        latent_return_potential=clamp(latent_return, 0.5, 25.0),
        latent_failure_potential=clamp(latent_failure, 0.1, 0.95),
        complexity=clamp(0.3 + innovation.quality * 0.3, 0.3, 1.0),
        discovered=false,
        discovery_round=market.current_round,
        created_by=innovation.creator_id,
        config=market.config,
        sector=branch_name,
        capital_requirements=capital_req,
        time_to_maturity=maturity,
        novelty_score=innov_novelty,  # Novel opportunities from innovations
        capacity=opp_capacity
    )

    push!(market.opportunities, opportunity)
    market.opportunity_map[opportunity.id] = opportunity
    _index_opportunity!(market, opportunity)

    return opportunity
end

# ============================================================================
# NOVELTY DISRUPTION (The "DeepSeek Effect")
# ============================================================================

"""
Apply novelty disruption when a highly novel innovation occurs.
Crowded opportunities in the same sector lose value - this implements
the mechanism where unexpected innovations disrupt established opportunities.

This is grounded in the theoretical insight that "the more important the
innovation, the less predictable it is" (Rescher, 2016). Novel innovations
create new possibilities that can make existing opportunities less valuable.
"""
function apply_novelty_disruption!(
    market::MarketEnvironment,
    innovation::Innovation
)::Int
    # Check if disruption is enabled
    if !getfield_default(market.config, :NOVELTY_DISRUPTION_ENABLED, true)
        return 0
    end

    novelty = clamp(something(innovation.novelty, 0.0), 0.0, 1.0)
    threshold = getfield_default(market.config, :NOVELTY_DISRUPTION_THRESHOLD, 0.6)

    if novelty < threshold
        return 0  # No disruption for low-novelty innovations
    end

    sector = something(innovation.sector, "tech")
    magnitude = getfield_default(market.config, :NOVELTY_DISRUPTION_MAGNITUDE, 0.25)
    comp_threshold = getfield_default(market.config, :DISRUPTION_COMPETITION_THRESHOLD, 10.0)

    # Base disruption scales with how novel the innovation is
    # At threshold: 0% disruption, at novelty=1.0: full magnitude
    base_disruption = (novelty - threshold) / (1.0 - threshold) * magnitude

    disrupted_count = 0
    for opp in market.opportunities
        # Only disrupt opportunities in the same sector with high competition
        if opp.sector == sector && opp.competition > comp_threshold
            # Vulnerability scales with competition (crowded = fragile)
            # At competition=0: vulnerability = 1.0
            # Competition scale: 0=none, 1.0=normal, 3.0=severe crowding
            # At competition=1.0 (normal): vulnerability = 1.25
            # At competition=2.0 (crowded): vulnerability = 1.5
            # At competition=3.0 (severe): vulnerability = 1.75
            vulnerability = 1.0 + (opp.competition / 2.0) * 0.5
            reduction = base_disruption * vulnerability

            # Apply disruption - reduce latent return potential
            opp.latent_return_potential *= (1.0 - clamp(reduction, 0.0, 0.5))
            opp.disrupted_count += 1
            disrupted_count += 1
        end
    end

    return disrupted_count
end

# ============================================================================
# DIVERSITY METRICS
# ============================================================================

"""
Compute combination and sector HHI diversity metrics.
"""
function get_combination_diversity_metrics(market::MarketEnvironment)::Tuple{Float64,Float64}
    combo_counts = Dict{String,Int}()
    sector_counts = Dict{String,Int}()

    for opp in market.opportunities
        signature = !isnothing(opp.combination_signature) ? opp.combination_signature : "sector_$(opp.sector)"
        combo_counts[signature] = get(combo_counts, signature, 0) + 1
        sector_counts[opp.sector] = get(sector_counts, opp.sector, 0) + 1
    end

    function compute_hhi(counter::Dict{String,Int})::Float64
        total = sum(values(counter))
        if total <= 0
            return 0.0
        end
        return sum((count / total)^2 for count in values(counter))
    end

    return (compute_hhi(combo_counts), compute_hhi(sector_counts))
end

"""
Clear old opportunities that have been around too long with low competition.
"""
function clear_old_opportunities!(market::MarketEnvironment, round::Int)
    filter!(market.opportunities) do opp
        age = round - opp.discovery_round
        !(age > 80 && opp.competition < 0.05)
    end
    _rebuild_sector_index!(market)
end
