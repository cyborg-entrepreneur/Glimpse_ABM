"""
Main simulation orchestrator for GlimpseABM.jl

This module implements the EmergentSimulation that coordinates agents,
market, and uncertainty environments.

Port of: glimpse_abm/simulation.py
"""

using Random
using Statistics
using DataFrames
using Dates
using SHA

# ============================================================================
# EMERGENT SIMULATION
# ============================================================================

"""
Main simulation class that orchestrates the agent-based model.
"""
mutable struct EmergentSimulation
    config::EmergentConfig
    agents::Vector{EmergentAgent}
    market::MarketEnvironment
    uncertainty_env::KnightianUncertaintyEnvironment
    knowledge_base::KnowledgeBase
    innovation_engine::InnovationEngine
    info_system::InformationSystem  # AI information generation system
    current_round::Int
    history::Vector{Dict{String,Any}}
    run_id::String
    output_dir::String
    rng::Random.AbstractRNG
    start_time::DateTime
    # Uncertainty transformation tracking
    previous_uncertainty_levels::Dict{String,Float64}
    baseline_uncertainty_levels::Dict{String,Float64}
end

"""
Initialize a new simulation.
"""
function EmergentSimulation(;
    config::EmergentConfig = EmergentConfig(),
    output_dir::String = "results",
    run_id::String = "run_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
    seed::Union{Int,Nothing} = nothing,
    initial_tier_distribution::Union{Dict{String,Float64},Nothing} = nothing
)
    # Initialize configuration
    initialize!(config)

    # Set up RNG - matches Python's seed derivation with run_id hashing
    # Python: seed = (base_seed + SHA256(run_id) % 1_000_000) % (2^32 - 1)
    actual_seed = if !isnothing(seed)
        seed
    elseif !isnothing(config.RANDOM_SEED) && config.RANDOM_SEED > 0
        # Deterministic hash (avoid Julia's randomized hash)
        run_hash = reinterpret(UInt64, sha256(run_id)[1:8])[1] % 1_000_000
        mod(config.RANDOM_SEED + run_hash, 2^32 - 1)
    else
        # Fallback to random seed
        rand(1:2^31-1)
    end

    # Create RNG - use NumpyRNG for cross-language reproducibility if configured
    rng = if config.USE_NUMPY_RNG
        NumpyRNG(actual_seed)
    else
        MersenneTwister(actual_seed)
    end

    # Create uncertainty environment
    uncertainty_env = KnightianUncertaintyEnvironment(config; rng=rng)

    # Create market environment
    market = MarketEnvironment(config; rng=rng)

    # Create knowledge base and innovation engine (matches Python)
    knowledge_base = KnowledgeBase(config)
    combination_tracker = CombinationTracker()
    innovation_engine = InnovationEngine(config, knowledge_base, combination_tracker)

    # Create information system for AI-assisted analysis
    info_system = InformationSystem(config)

    # Determine initial AI tier distribution
    # Default: 100% none. Can specify e.g. Dict("none"=>0.25, "basic"=>0.25, "advanced"=>0.25, "premium"=>0.25)
    tier_order = ["none", "basic", "advanced", "premium"]
    tier_probs = if isnothing(initial_tier_distribution)
        [1.0, 0.0, 0.0, 0.0]  # Default: all start at none
    else
        [get(initial_tier_distribution, t, 0.0) for t in tier_order]
    end
    # Normalize probabilities
    total_prob = sum(tier_probs)
    if total_prob > 0
        tier_probs = tier_probs ./ total_prob
    else
        tier_probs = [1.0, 0.0, 0.0, 0.0]
    end

    # Create agents with distributed initial tiers
    agents = EmergentAgent[]
    for i in 1:config.N_AGENTS
        # Sample initial tier based on distribution
        r = rand(rng)
        cumsum = 0.0
        initial_tier = "none"
        for (j, tier) in enumerate(tier_order)
            cumsum += tier_probs[j]
            if r <= cumsum
                initial_tier = tier
                break
            end
        end
        agent = EmergentAgent(i, config; rng=rng, fixed_ai_level=initial_tier)
        # Initialize subscription schedule for fixed AI tier (ensures costs are charged)
        if initial_tier != "none"
            ensure_subscription_schedule!(agent, initial_tier)
        end
        push!(agents, agent)
    end

    return EmergentSimulation(
        config,
        agents,
        market,
        uncertainty_env,
        knowledge_base,
        innovation_engine,
        info_system,
        0,
        Dict{String,Any}[],
        run_id,
        output_dir,
        rng,
        now(),
        Dict{String,Float64}(),  # previous_uncertainty_levels
        Dict{String,Float64}()   # baseline_uncertainty_levels
    )
end

"""
Initialize agents with a fixed AI level (for causal analysis).
"""
function initialize_agents!(sim::EmergentSimulation; fixed_ai_level::Union{String,Nothing} = nothing)
    for agent in sim.agents
        if !isnothing(fixed_ai_level)
            agent.fixed_ai_level = fixed_ai_level
            agent.current_ai_level = fixed_ai_level
            # Start subscription schedule if this is a subscription tier
            ensure_subscription_schedule!(agent, fixed_ai_level)
        end
    end
end

"""
Run the full simulation.
"""
function run!(sim::EmergentSimulation)
    println("[$(sim.run_id)] Starting simulation...")

    for round in 1:sim.config.N_ROUNDS
        step!(sim, round)

        # Log progress periodically
        if sim.config.enable_round_logging && round % sim.config.round_log_interval == 0
            alive_count = count(a -> a.alive, sim.agents)
            capitals = [get_capital(a) for a in sim.agents if a.alive]
            mean_capital = isempty(capitals) ? 0.0 : mean(capitals)
            # println("[$(sim.run_id)] Round $round: $(alive_count)/$(sim.config.N_AGENTS) alive, mean capital: \$$(round(Int, mean_capital))")
        end
    end

    println("[$(sim.run_id)] Simulation finished.")
    return sim
end

"""
Execute a single simulation round.
Matches Python _step order: operating costs -> matured investments -> decisions -> survival checks
"""
function step!(sim::EmergentSimulation, round::Int)
    sim.current_round = round

    # Get current uncertainty state and market conditions
    uncertainty_state = get_uncertainty_state(sim.uncertainty_env)
    market_conditions = get_market_conditions(sim.market)
    market_conditions["uncertainty_state"] = uncertainty_state

    # Get alive agents
    alive_agents = filter(a -> a.alive, sim.agents)

    # Phase 1: Apply operational costs FIRST with severity multiplier (matching Python order)
    # Python applies this before matured investments and decisions
    avg_comp = Float64(get(market_conditions, "avg_competition", 0.0))
    volatility = Float64(get(market_conditions, "volatility", sim.config.MARKET_VOLATILITY))
    base_vol = Float64(sim.config.MARKET_VOLATILITY)
    severity = 1.0 + avg_comp * 0.35 + max(0.0, volatility - base_vol) * 0.45
    severity = clamp(severity, 0.7, 1.9)

    for agent in sim.agents
        if !agent.alive
            continue
        end
        # Calculate base cost
        estimated_cost = estimate_operational_costs(agent, sim.market)
        # Apply severity multiplier (matching Python _apply_operating_costs)
        operating_cost = max(0.0, estimated_cost * severity)
        agent.operating_cost_estimate = operating_cost
        if operating_cost > 0.0
            set_capital!(agent, get_capital(agent) - operating_cost)
            # Check survival after operating costs (matches Python line 189)
            check_survival!(agent, round)
        end
    end

    # Phase 1.5: Charge AI subscription installments (matching Python _apply_subscription_carry)
    for agent in sim.agents
        if !agent.alive
            continue
        end
        subscription_cost = apply_subscription_carry!(agent, round)
        # Check survival after subscription charges
        if subscription_cost > 0.0
            check_survival!(agent, round)
        end
    end

    # Phase 2: Process matured investments (matching Python BLOCK 1)
    # Pass market_conditions with uncertainty_state (matches Python line 913)
    all_matured = Dict{String,Any}[]
    for agent in sim.agents
        if !agent.alive
            continue
        end
        matured = process_matured_investments!(agent, sim.market, round; market_conditions=market_conditions)
        for m in matured
            # Update tier beliefs based on investment outcomes
            ai_tier = get(m, "ai_level", get_ai_level(agent))
            success = get(m, "success", false)
            update_tier_belief!(agent.ai_learning, ai_tier, success)
        end
        append!(all_matured, matured)
        # Check survival after matured investments (matches Python line 1003)
        check_survival!(agent, round)
    end

    # Get available opportunities (after applying costs, matching Python timing)
    available_opportunities = get_available_opportunities(sim.market)

    # Refresh alive agents after survival checks
    alive_agents = filter(a -> a.alive, sim.agents)

    # Phase 3: AI level selection and action decisions (matching Python BLOCK 2)
    # Check if sequential decisions are enabled
    sequential_enabled = hasfield(typeof(sim.config), :SEQUENTIAL_DECISIONS_ENABLED) ?
        sim.config.SEQUENTIAL_DECISIONS_ENABLED : false

    agent_actions = Dict{String,Any}[]

    if sequential_enabled && length(alive_agents) > 1
        # Sequential decision making: early agents decide first, their choices become visible signals
        early_fraction = hasfield(typeof(sim.config), :EARLY_DECISION_FRACTION) ?
            sim.config.EARLY_DECISION_FRACTION : 0.3
        signal_weight = hasfield(typeof(sim.config), :SIGNAL_VISIBILITY_WEIGHT) ?
            sim.config.SIGNAL_VISIBILITY_WEIGHT : 0.15

        n_early = max(1, Int(floor(length(alive_agents) * early_fraction)))

        # Shuffle and split into early and late deciders
        shuffled_agents = shuffle(sim.rng, collect(alive_agents))
        early_agents = shuffled_agents[1:n_early]
        late_agents = shuffled_agents[n_early+1:end]

        # Phase 3a: Early agents decide (no visibility signals)
        early_signals = Dict{String,Int}()  # Track which opportunities early agents invested in

        for agent in early_agents
            neighbor_agents = EmergentAgent[]
            other_agents = filter(a -> a.id != agent.id && a.alive, alive_agents)
            if !isempty(other_agents)
                n_neighbors = min(5, length(other_agents))
                neighbor_agents = rand(sim.rng, other_agents, n_neighbors)
            end

            outcome = make_decision!(
                agent,
                available_opportunities,
                market_conditions,
                sim.market,
                round;
                uncertainty_env=sim.uncertainty_env,
                neighbor_agents=neighbor_agents,
                innovation_engine=sim.innovation_engine,
                info_system=sim.info_system
            )

            push!(agent_actions, outcome)

            # Record visible signal for invest actions
            if get(outcome, "action", "") == "invest"
                opp_id = string(get(outcome, "opportunity_id", ""))
                if !isempty(opp_id)
                    early_signals[opp_id] = get(early_signals, opp_id, 0) + 1
                end
            end
        end

        # Phase 3b: Late agents decide with visible signals
        for agent in late_agents
            neighbor_agents = EmergentAgent[]
            other_agents = filter(a -> a.id != agent.id && a.alive, alive_agents)
            if !isempty(other_agents)
                n_neighbors = min(5, length(other_agents))
                neighbor_agents = rand(sim.rng, other_agents, n_neighbors)
            end

            outcome = make_decision!(
                agent,
                available_opportunities,
                market_conditions,
                sim.market,
                round;
                uncertainty_env=sim.uncertainty_env,
                neighbor_agents=neighbor_agents,
                innovation_engine=sim.innovation_engine,
                info_system=sim.info_system,
                early_signals=early_signals,
                signal_weight=signal_weight
            )

            push!(agent_actions, outcome)
        end
    else
        # Original simultaneous decision logic
        for agent in sim.agents
            if !agent.alive
                continue
            end

            # Collect neighbor agents for social influence
            neighbor_agents = EmergentAgent[]
            if length(alive_agents) > 1
                other_agents = filter(a -> a.id != agent.id && a.alive, alive_agents)
                if !isempty(other_agents)
                    n_neighbors = min(5, length(other_agents))
                    neighbor_agents = rand(sim.rng, other_agents, n_neighbors)
                end
            end

            # Use make_decision! which properly integrates AI level effects
            outcome = make_decision!(
                agent,
                available_opportunities,
                market_conditions,
                sim.market,
                round;
                uncertainty_env=sim.uncertainty_env,
                neighbor_agents=neighbor_agents,
                innovation_engine=sim.innovation_engine,
                info_system=sim.info_system
            )

            push!(agent_actions, outcome)
        end
    end

    # Update tier beliefs from immediate action outcomes (innovate, explore)
    for action in agent_actions
        agent_id = get(action, "agent_id", 0)
        if agent_id < 1 || agent_id > length(sim.agents)
            continue
        end
        agent = sim.agents[agent_id]
        if !agent.alive
            continue
        end
        action_type = get(action, "action", "maintain")
        if action_type in ["innovate", "explore"]
            success = get(action, "success", false)
            ai_tier = get(action, "ai_level_used", get_ai_level(agent))
            update_tier_belief!(agent.ai_learning, ai_tier, success)
        end
    end

    # Phase 4: Final survival check for all agents (matches Python lines 1077-1078)
    for agent in sim.agents
        check_survival!(agent, round)
    end

    # Phase 5: Update market
    innovations = Innovation[]  # Collect innovations from agent actions
    for action in agent_actions
        if get(action, "action", "") == "innovate" && get(action, "success", false)
            innov = Innovation(
                id=generate_innovation_id(round, get(action, "agent_id", 0), 0),
                type="incremental",
                knowledge_components=String[],
                novelty=rand(sim.rng),
                quality=rand(sim.rng),
                round_created=round,
                creator_id=get(action, "agent_id", 0),
                ai_level_used=get(action, "ai_level_used", "none")
            )
            push!(innovations, innov)
            # Spawn derivative opportunity from successful innovation (Python match)
            cash_multiple = Float64(get(action, "cash_multiple", 1.5))
            spawn_opportunity_from_innovation!(sim.market, innov, cash_multiple)
        end
    end

    # Phase 5b: Apply novelty disruption for high-novelty innovations
    # This implements the "DeepSeek Effect" - novel innovations disrupt crowded opportunities
    novelty_disruption_enabled = hasfield(typeof(sim.config), :NOVELTY_DISRUPTION_ENABLED) ?
        sim.config.NOVELTY_DISRUPTION_ENABLED : true
    if novelty_disruption_enabled
        for innov in innovations
            disrupted_count = apply_novelty_disruption!(sim.market, innov)
            if disrupted_count > 0
                # Track disruption in action for the innovator (optional: for analysis)
                for action in agent_actions
                    if get(action, "agent_id", 0) == innov.creator_id &&
                       get(action, "action", "") == "innovate"
                        action["disrupted_opportunities"] = disrupted_count
                        break
                    end
                end
            end
        end
    end

    # Create niche opportunities from exploration discoveries (Python match)
    for action in agent_actions
        if get(action, "action", "") == "explore" && get(action, "exploration_type", "") == "niche_discovery"
            agent_id = get(action, "agent_id", 0)
            niche_id = get(action, "discovered_sector", nothing)
            if !isnothing(niche_id)
                # Create 1-3 new opportunities in the discovered niche (matches Python)
                n_niche_opps = rand(sim.rng, 1:3)
                for _ in 1:n_niche_opps
                    create_niche_opportunity(sim.market, string(niche_id), agent_id, round)
                end
            end
        end
    end

    market_state = step!(sim.market, round, agent_actions, innovations; matured_outcomes=all_matured)

    # Phase 6: Update uncertainty measurements
    record_ai_signals!(sim.uncertainty_env, round, agent_actions)
    uncertainty_state = measure_uncertainty_state!(
        sim.uncertainty_env,
        sim.market,
        agent_actions,
        innovations,
        round
    )

    # Phase 7: Record history
    round_stats = compile_round_stats(sim, round, agent_actions, all_matured, uncertainty_state)
    push!(sim.history, round_stats)

    return round_stats
end

"""
Compile statistics for a round.
"""
function compile_round_stats(
    sim::EmergentSimulation,
    round::Int,
    agent_actions::Vector{Dict{String,Any}},
    matured_outcomes::Vector{Dict{String,Any}},
    uncertainty_state::Dict{String,Dict{String,Any}}
)::Dict{String,Any}
    alive_agents = [a for a in sim.agents if a.alive]
    n_alive = length(alive_agents)
    n_total = length(sim.agents)

    # Capital statistics
    capitals = [get_capital(a) for a in alive_agents]
    mean_capital = isempty(capitals) ? 0.0 : mean(capitals)
    std_capital = isempty(capitals) || length(capitals) < 2 ? 0.0 : std(capitals)
    median_capital = isempty(capitals) ? 0.0 : median(capitals)

    # Action counts and capital tracking by action type
    action_counts = Dict{String,Int}()
    ai_usage = Dict{String,Int}("none" => 0, "basic" => 0, "advanced" => 0, "premium" => 0)

    # Track capital deployed/returned by action type for ROIC calculation
    capital_deployed = Dict{String,Float64}("invest" => 0.0, "innovate" => 0.0, "explore" => 0.0)
    capital_returned = Dict{String,Float64}("invest" => 0.0, "innovate" => 0.0, "explore" => 0.0)

    # Track opportunity IDs for HHI calculation
    opportunity_ids = String[]
    invest_confidences = Float64[]

    for action in agent_actions
        act_type = get(action, "action", "maintain")
        action_counts[act_type] = get(action_counts, act_type, 0) + 1

        ai_level = lowercase(string(get(action, "ai_level_used", "none")))
        if haskey(ai_usage, ai_level)
            ai_usage[ai_level] += 1
        end

        # Track capital deployed by action type
        if act_type == "invest"
            amount = Float64(get(action, "amount", 0.0))
            capital_deployed["invest"] += amount
            opp_id = get(action, "opportunity_id", nothing)
            if !isnothing(opp_id)
                push!(opportunity_ids, string(opp_id))
            end
            # Track decision confidence for invest actions
            perception = get(action, "perception", Dict{String,Any}())
            conf = Float64(get(perception, "decision_confidence", 0.5))
            push!(invest_confidences, conf)
        elseif act_type == "innovate"
            rd_spend = Float64(get(action, "rd_spend", 0.0))
            capital_deployed["innovate"] += rd_spend
            if get(action, "success", false)
                ret = Float64(get(action, "innovation_return", 0.0))
                capital_returned["innovate"] += ret
            else
                rec = Float64(get(action, "recovery", 0.0))
                capital_returned["innovate"] += rec
            end
        elseif act_type == "explore"
            cost = Float64(get(action, "cost", 0.0))
            capital_deployed["explore"] += cost
            # Exploration doesn't have direct capital return
        end
    end

    # Add matured investment returns to capital_returned["invest"]
    for outcome in matured_outcomes
        ret = Float64(get(outcome, "capital_returned", 0.0))
        capital_returned["invest"] += ret
    end

    # Calculate ROIC by action type (matches Python)
    mean_roic_invest = capital_deployed["invest"] > 0 ?
        (capital_returned["invest"] - capital_deployed["invest"]) / capital_deployed["invest"] : 0.0
    mean_roic_innovate = capital_deployed["innovate"] > 0 ?
        (capital_returned["innovate"] - capital_deployed["innovate"]) / capital_deployed["innovate"] : 0.0
    mean_roic_explore = 0.0  # Explore doesn't have direct return

    # Net capital flow by action type
    net_capital_flow_invest = capital_returned["invest"] - capital_deployed["invest"]
    net_capital_flow_innovate = capital_returned["innovate"] - capital_deployed["innovate"]

    # Calculate HHI (Herfindahl-Hirschman Index) for investment concentration
    overall_hhi = 0.0
    if !isempty(opportunity_ids)
        opp_counts = Dict{String,Int}()
        for opp_id in opportunity_ids
            opp_counts[opp_id] = get(opp_counts, opp_id, 0) + 1
        end
        total_invests = length(opportunity_ids)
        overall_hhi = sum((count / total_invests)^2 for count in values(opp_counts))
    end

    # Action shares
    total_actions = sum(values(action_counts))
    action_shares = Dict(
        "invest" => total_actions > 0 ? get(action_counts, "invest", 0) / total_actions : 0.0,
        "innovate" => total_actions > 0 ? get(action_counts, "innovate", 0) / total_actions : 0.0,
        "explore" => total_actions > 0 ? get(action_counts, "explore", 0) / total_actions : 0.0,
        "maintain" => total_actions > 0 ? get(action_counts, "maintain", 0) / total_actions : 0.0
    )

    # AI tier shares
    total_ai = sum(values(ai_usage))
    ai_shares = Dict(
        "none" => total_ai > 0 ? ai_usage["none"] / total_ai : 0.0,
        "basic" => total_ai > 0 ? ai_usage["basic"] / total_ai : 0.0,
        "advanced" => total_ai > 0 ? ai_usage["advanced"] / total_ai : 0.0,
        "premium" => total_ai > 0 ? ai_usage["premium"] / total_ai : 0.0
    )

    # Innovation stats
    innovation_attempts = get(action_counts, "innovate", 0)
    innovation_successes = count(a -> get(a, "action", "") == "innovate" && get(a, "success", false), agent_actions)
    innovation_success_rate = innovation_attempts > 0 ? innovation_successes / innovation_attempts : 0.0

    # Mean confidence for invest actions
    mean_confidence_invest = isempty(invest_confidences) ? 0.0 : mean(invest_confidences)

    # Matured investment stats
    n_matured = length(matured_outcomes)
    n_success = count(o -> get(o, "success", false), matured_outcomes)
    n_failure = n_matured - n_success

    # Uncertainty levels (formula-based, environment level - kept for backwards compatibility)
    actor_ignorance = Float64(get(get(uncertainty_state, "actor_ignorance", Dict()), "level", 0.0))
    practical_indet = Float64(get(get(uncertainty_state, "practical_indeterminism", Dict()), "level", 0.0))
    agentic_novelty = Float64(get(get(uncertainty_state, "agentic_novelty", Dict()), "level", 0.0))
    competitive_rec = Float64(get(get(uncertainty_state, "competitive_recursion", Dict()), "level", 0.0))

    # EMERGENT uncertainty (agent-level, computed from actual outcomes)
    # These metrics emerge from what actually happens to agents, not from formulas
    emergent_by_tier = aggregate_emergent_uncertainty_by_tier(sim.agents)

    # Get simulation's AI tier (all agents have same fixed tier in this simulation design)
    # Use fixed_ai_level if set, otherwise current_ai_level
    sim_tier = if !isempty(sim.agents)
        first_agent = sim.agents[1]
        if !isnothing(first_agent.fixed_ai_level)
            first_agent.fixed_ai_level
        else
            first_agent.current_ai_level
        end
    else
        "none"
    end

    # Get emergent metrics for this tier
    tier_emergent = get(emergent_by_tier, sim_tier, Dict{String,Float64}(
        "actor_ignorance" => 0.5,
        "practical_indeterminism" => 0.5,
        "agentic_novelty" => 0.5,
        "competitive_recursion" => 0.0
    ))

    emergent_actor_ignorance = Float64(get(tier_emergent, "actor_ignorance", 0.5))
    emergent_practical_indet = Float64(get(tier_emergent, "practical_indeterminism", 0.5))
    emergent_agentic_novelty = Float64(get(tier_emergent, "agentic_novelty", 0.5))
    emergent_competitive_rec = Float64(get(tier_emergent, "competitive_recursion", 0.0))

    # --- Uncertainty Transformation Metrics ---
    # Store baseline on first round (or first round with uncertainty data)
    if isempty(sim.baseline_uncertainty_levels)
        sim.baseline_uncertainty_levels["actor_ignorance"] = actor_ignorance
        sim.baseline_uncertainty_levels["practical_indeterminism"] = practical_indet
        sim.baseline_uncertainty_levels["agentic_novelty"] = agentic_novelty
        sim.baseline_uncertainty_levels["competitive_recursion"] = competitive_rec
    end

    # Get previous levels (default to current if first round)
    prev_actor = get(sim.previous_uncertainty_levels, "actor_ignorance", actor_ignorance)
    prev_practical = get(sim.previous_uncertainty_levels, "practical_indeterminism", practical_indet)
    prev_agentic = get(sim.previous_uncertainty_levels, "agentic_novelty", agentic_novelty)
    prev_competitive = get(sim.previous_uncertainty_levels, "competitive_recursion", competitive_rec)

    # Get baseline levels
    base_actor = get(sim.baseline_uncertainty_levels, "actor_ignorance", actor_ignorance)
    base_practical = get(sim.baseline_uncertainty_levels, "practical_indeterminism", practical_indet)
    base_agentic = get(sim.baseline_uncertainty_levels, "agentic_novelty", agentic_novelty)
    base_competitive = get(sim.baseline_uncertainty_levels, "competitive_recursion", competitive_rec)

    # Compute delta (round-over-round change)
    delta_actor = actor_ignorance - prev_actor
    delta_practical = practical_indet - prev_practical
    delta_agentic = agentic_novelty - prev_agentic
    delta_competitive = competitive_rec - prev_competitive

    # Compute cumulative delta (change from baseline)
    cumulative_delta_actor = actor_ignorance - base_actor
    cumulative_delta_practical = practical_indet - base_practical
    cumulative_delta_agentic = agentic_novelty - base_agentic
    cumulative_delta_competitive = competitive_rec - base_competitive

    # Compute portfolio composition (shares)
    uncertainty_total = actor_ignorance + practical_indet + agentic_novelty + competitive_rec
    total_safe = max(uncertainty_total, 0.001)  # Avoid division by zero
    share_actor = actor_ignorance / total_safe
    share_practical = practical_indet / total_safe
    share_agentic = agentic_novelty / total_safe
    share_competitive = competitive_rec / total_safe

    # Compute HHI (Herfindahl-Hirschman Index) - concentration measure
    uncertainty_hhi = share_actor^2 + share_practical^2 + share_agentic^2 + share_competitive^2

    # Compute entropy - diversity measure (avoid log(0))
    eps = 1e-10
    uncertainty_entropy = -(
        share_actor * log(share_actor + eps) +
        share_practical * log(share_practical + eps) +
        share_agentic * log(share_agentic + eps) +
        share_competitive * log(share_competitive + eps)
    )

    # Update previous levels for next round
    sim.previous_uncertainty_levels["actor_ignorance"] = actor_ignorance
    sim.previous_uncertainty_levels["practical_indeterminism"] = practical_indet
    sim.previous_uncertainty_levels["agentic_novelty"] = agentic_novelty
    sim.previous_uncertainty_levels["competitive_recursion"] = competitive_rec

    # Mean AI trust
    trust_values = [Float64(get(a.traits, "ai_trust", 0.5)) for a in alive_agents]
    mean_trust = isempty(trust_values) ? 0.5 : mean(trust_values)
    std_trust = length(trust_values) < 2 ? 0.0 : std(trust_values)

    return Dict{String,Any}(
        "round" => round,
        "n_alive" => n_alive,
        "n_total" => n_total,
        "survival_rate" => n_total > 0 ? n_alive / n_total : 0.0,
        "mean_capital" => mean_capital,
        "median_capital" => median_capital,
        "std_capital" => std_capital,
        "total_capital" => sum(capitals),
        # Action counts
        "invest_count" => get(action_counts, "invest", 0),
        "innovate_count" => get(action_counts, "innovate", 0),
        "explore_count" => get(action_counts, "explore", 0),
        "maintain_count" => get(action_counts, "maintain", 0),
        # Action shares
        "action_share_invest" => action_shares["invest"],
        "action_share_innovate" => action_shares["innovate"],
        "action_share_explore" => action_shares["explore"],
        "action_share_maintain" => action_shares["maintain"],
        # AI tier counts and shares
        "ai_none_count" => ai_usage["none"],
        "ai_basic_count" => ai_usage["basic"],
        "ai_advanced_count" => ai_usage["advanced"],
        "ai_premium_count" => ai_usage["premium"],
        "ai_share_none" => ai_shares["none"],
        "ai_share_basic" => ai_shares["basic"],
        "ai_share_advanced" => ai_shares["advanced"],
        "ai_share_premium" => ai_shares["premium"],
        # Capital deployed/returned by action type (matches Python)
        "total_capital_deployed" => sum(values(capital_deployed)),
        "total_capital_returned" => sum(values(capital_returned)),
        "total_capital_deployed_invest" => capital_deployed["invest"],
        "total_capital_deployed_innovate" => capital_deployed["innovate"],
        "total_capital_deployed_explore" => capital_deployed["explore"],
        "total_capital_returned_invest" => capital_returned["invest"],
        "total_capital_returned_innovate" => capital_returned["innovate"],
        "total_capital_returned_explore" => capital_returned["explore"],
        "net_capital_flow_invest" => net_capital_flow_invest,
        "net_capital_flow_innovate" => net_capital_flow_innovate,
        # ROIC by action type (matches Python)
        "mean_roic_invest" => mean_roic_invest,
        "mean_roic_innovate" => mean_roic_innovate,
        "mean_roic_explore" => mean_roic_explore,
        # HHI and sector metrics (matches Python)
        "overall_hhi" => overall_hhi,
        # Innovation metrics
        "innovation_attempts" => innovation_attempts,
        "innovation_successes" => innovation_successes,
        "innovation_success_rate" => innovation_success_rate,
        # Confidence metrics
        "mean_confidence_invest" => mean_confidence_invest,
        "mean_ai_trust" => mean_trust,
        "ai_trust_std" => std_trust,
        # Matured investment stats
        "n_matured" => n_matured,
        "n_success" => n_success,
        "n_failure" => n_failure,
        "success_rate" => n_matured > 0 ? n_success / n_matured : 0.0,
        # Uncertainty levels (formula-based, kept for backwards compatibility)
        "actor_ignorance" => actor_ignorance,
        "practical_indeterminism" => practical_indet,
        "agentic_novelty" => agentic_novelty,
        "competitive_recursion" => competitive_rec,
        # EMERGENT uncertainty (agent-level, from actual outcomes)
        "emergent_actor_ignorance" => emergent_actor_ignorance,
        "emergent_practical_indeterminism" => emergent_practical_indet,
        "emergent_agentic_novelty" => emergent_agentic_novelty,
        "emergent_competitive_recursion" => emergent_competitive_rec,
        # Uncertainty transformation metrics
        "delta_actor_ignorance" => delta_actor,
        "delta_practical_indeterminism" => delta_practical,
        "delta_agentic_novelty" => delta_agentic,
        "delta_competitive_recursion" => delta_competitive,
        "cumulative_delta_actor" => cumulative_delta_actor,
        "cumulative_delta_practical" => cumulative_delta_practical,
        "cumulative_delta_agentic" => cumulative_delta_agentic,
        "cumulative_delta_competitive" => cumulative_delta_competitive,
        "uncertainty_total" => uncertainty_total,
        "share_actor_ignorance" => share_actor,
        "share_practical_indeterminism" => share_practical,
        "share_agentic_novelty" => share_agentic,
        "share_competitive_recursion" => share_competitive,
        "uncertainty_hhi" => uncertainty_hhi,
        "uncertainty_entropy" => uncertainty_entropy,
        # Agent counts
        "alive_agents" => n_alive,
        "dead_agents" => n_total - n_alive
    )
end

"""
Convert simulation history to DataFrame.
"""
function history_to_dataframe(sim::EmergentSimulation)::DataFrame
    if isempty(sim.history)
        return DataFrame()
    end

    # Get all column names from first entry
    cols = collect(keys(sim.history[1]))

    # Create DataFrame
    df = DataFrame()
    for col in cols
        df[!, Symbol(col)] = [get(h, col, missing) for h in sim.history]
    end

    return df
end

"""
Get final agent data as DataFrame.
"""
function agents_to_dataframe(sim::EmergentSimulation)::DataFrame
    data = [snapshot(agent, sim.current_round) for agent in sim.agents]

    if isempty(data)
        return DataFrame()
    end

    cols = collect(keys(data[1]))
    df = DataFrame()
    for col in cols
        df[!, Symbol(col)] = [get(d, col, missing) for d in data]
    end

    return df
end

"""
Get summary statistics for the simulation.
"""
function summary_stats(sim::EmergentSimulation)::Dict{String,Any}
    alive_agents = [a for a in sim.agents if a.alive]

    # Final survival rate
    survival_rate = length(alive_agents) / length(sim.agents)

    # Capital statistics
    final_capitals = [get_capital(a) for a in alive_agents]
    mean_final_capital = isempty(final_capitals) ? 0.0 : mean(final_capitals)

    # AI tier distribution at end
    ai_distribution = Dict{String,Int}("none" => 0, "basic" => 0, "advanced" => 0, "premium" => 0)
    for agent in sim.agents
        tier = get_ai_level(agent)
        if haskey(ai_distribution, tier)
            ai_distribution[tier] += 1
        end
    end

    # Success/failure totals
    total_successes = sum(a.success_count for a in sim.agents)
    total_failures = sum(a.failure_count for a in sim.agents)
    total_innovations = sum(a.innovation_count for a in sim.agents)

    # Uncertainty averages from history
    if !isempty(sim.history)
        mean_actor_ignorance = mean(get(h, "actor_ignorance", 0.0) for h in sim.history)
        mean_practical_indet = mean(get(h, "practical_indeterminism", 0.0) for h in sim.history)
        mean_agentic_novelty = mean(get(h, "agentic_novelty", 0.0) for h in sim.history)
        mean_competitive_rec = mean(get(h, "competitive_recursion", 0.0) for h in sim.history)
    else
        mean_actor_ignorance = 0.0
        mean_practical_indet = 0.0
        mean_agentic_novelty = 0.0
        mean_competitive_rec = 0.0
    end

    return Dict{String,Any}(
        "run_id" => sim.run_id,
        "n_agents" => length(sim.agents),
        "n_rounds" => sim.config.N_ROUNDS,
        "final_survival_rate" => survival_rate,
        "n_survivors" => length(alive_agents),
        "mean_final_capital" => mean_final_capital,
        "total_successes" => total_successes,
        "total_failures" => total_failures,
        "total_innovations" => total_innovations,
        "ai_none_count" => ai_distribution["none"],
        "ai_basic_count" => ai_distribution["basic"],
        "ai_advanced_count" => ai_distribution["advanced"],
        "ai_premium_count" => ai_distribution["premium"],
        "mean_actor_ignorance" => mean_actor_ignorance,
        "mean_practical_indeterminism" => mean_practical_indet,
        "mean_agentic_novelty" => mean_agentic_novelty,
        "mean_competitive_recursion" => mean_competitive_rec,
        "elapsed_seconds" => (now() - sim.start_time).value / 1000.0
    )
end

"""
Save simulation results to disk.
"""
function save_results!(sim::EmergentSimulation)
    mkpath(sim.output_dir)

    # Save history
    history_df = history_to_dataframe(sim)
    if nrow(history_df) > 0
        save_dataframe_csv(history_df, joinpath(sim.output_dir, "history.csv"))
        save_dataframe_arrow(history_df, joinpath(sim.output_dir, "history.arrow"))
    end

    # Save agent data
    agents_df = agents_to_dataframe(sim)
    if nrow(agents_df) > 0
        save_dataframe_csv(agents_df, joinpath(sim.output_dir, "final_agents.csv"))
    end

    # Save config
    save_config_snapshot(sim.config, joinpath(sim.output_dir, "config_snapshot.json"))

    # Save summary
    stats = summary_stats(sim)
    open(joinpath(sim.output_dir, "summary.json"), "w") do io
        JSON3.write(io, stats)
    end

    println("[$(sim.run_id)] Results saved to $(sim.output_dir)")
end

# ============================================================================
# BATCH SIMULATION UTILITIES
# ============================================================================

"""
Run multiple simulations with different configurations.
"""
function run_batch(;
    base_config::EmergentConfig = EmergentConfig(),
    n_runs::Int = 10,
    output_base::String = "results",
    fixed_ai_levels::Vector{String} = String[],
    parallel::Bool = false
)::Vector{EmergentSimulation}
    results = EmergentSimulation[]

    if isempty(fixed_ai_levels)
        # Run with adaptive AI
        for run_idx in 1:n_runs
            config = deepcopy(base_config)
            config.RANDOM_SEED = base_config.RANDOM_SEED + run_idx

            run_id = "run_$(run_idx)"
            output_dir = joinpath(output_base, run_id)

            sim = EmergentSimulation(
                config=config,
                output_dir=output_dir,
                run_id=run_id,
                seed=config.RANDOM_SEED
            )

            run!(sim)
            save_results!(sim)
            push!(results, sim)
        end
    else
        # Run fixed AI tier sweep
        for (tier_idx, ai_level) in enumerate(fixed_ai_levels)
            for run_idx in 1:n_runs
                config = deepcopy(base_config)
                config.RANDOM_SEED = base_config.RANDOM_SEED + (tier_idx - 1) * n_runs + run_idx

                run_id = "Fixed_AI_Level_$(ai_level)_run_$(run_idx)"
                output_dir = joinpath(output_base, run_id)

                sim = EmergentSimulation(
                    config=config,
                    output_dir=output_dir,
                    run_id=run_id,
                    seed=config.RANDOM_SEED
                )

                # Set fixed AI level for all agents
                initialize_agents!(sim; fixed_ai_level=ai_level)

                run!(sim)
                save_results!(sim)
                push!(results, sim)
            end
        end
    end

    return results
end

"""
Aggregate results from multiple simulations.
"""
function aggregate_results(simulations::Vector{EmergentSimulation})::DataFrame
    all_stats = Dict{String,Any}[]

    for sim in simulations
        stats = summary_stats(sim)
        push!(all_stats, stats)
    end

    if isempty(all_stats)
        return DataFrame()
    end

    cols = collect(keys(all_stats[1]))
    df = DataFrame()
    for col in cols
        df[!, Symbol(col)] = [get(s, col, missing) for s in all_stats]
    end

    return df
end

# ============================================================================
# ENHANCED VECTORIZED AGENT STATE
# ============================================================================

"""
Enhanced vectorized agent state with trait tracking for efficient batch operations.
This extends the basic VectorizedAgentState in agents.jl with additional fields.
"""
mutable struct EnhancedVectorizedState
    n_agents::Int
    trait_names::Vector{String}
    trait_indices::Dict{String,Int}

    # Core state arrays
    alive::BitVector
    initial_capital::Vector{Float64}
    capital::Vector{Float64}
    traits::Matrix{Float64}
    ai_level::Vector{Int}  # 0=none, 1=basic, 2=advanced, 3=premium

    # Performance tracking
    capital_growth::Vector{Float64}
    innovation_count::Vector{Int}
    portfolio_size::Vector{Int}
    roe::Vector{Float64}
end

"""
Create a new EnhancedVectorizedState.
"""
function EnhancedVectorizedState(n_agents::Int, trait_names::Vector{String}, config::EmergentConfig)
    trait_indices = Dict(name => i for (i, name) in enumerate(trait_names))

    # Initial capital range
    initial_capital_low = config.INITIAL_CAPITAL
    initial_capital_high = config.INITIAL_CAPITAL
    if hasfield(typeof(config), :INITIAL_CAPITAL_RANGE)
        range_val = config.INITIAL_CAPITAL_RANGE
        if !isnothing(range_val) && length(range_val) >= 2
            initial_capital_low, initial_capital_high = range_val[1], range_val[2]
        end
    end

    initial_capital = rand(n_agents) .* (initial_capital_high - initial_capital_low) .+ initial_capital_low

    return EnhancedVectorizedState(
        n_agents,
        trait_names,
        trait_indices,
        trues(n_agents),  # alive
        initial_capital,
        copy(initial_capital),  # capital
        zeros(Float64, n_agents, length(trait_names)),  # traits
        zeros(Int, n_agents),  # ai_level
        ones(Float64, n_agents),  # capital_growth
        zeros(Int, n_agents),  # innovation_count
        zeros(Int, n_agents),  # portfolio_size
        zeros(Float64, n_agents)  # roe
    )
end

"""
Vectorized capital update.
"""
function batch_update_capital!(state::EnhancedVectorizedState, agent_ids::Vector{Int}, amounts::Vector{Float64}, survival_threshold::Float64)
    for (i, agent_id) in enumerate(agent_ids)
        state.capital[agent_id] += amounts[i]
        if state.capital[agent_id] < survival_threshold
            state.alive[agent_id] = false
        end
    end
end

"""
Get indices of alive agents.
"""
function get_alive_indices(state::EnhancedVectorizedState)::Vector{Int}
    return findall(state.alive)
end

"""
Get trait vector for a specific trait.
"""
function get_trait_vector(state::EnhancedVectorizedState, trait_name::String)::Vector{Float64}
    idx = state.trait_indices[trait_name]
    return state.traits[:, idx]
end

# ============================================================================
# DATA BUFFERING SYSTEM
# ============================================================================

"""
Data buffer for efficient batch writes.
"""
mutable struct DataBuffer
    decisions::Vector{Dict{String,Any}}
    market::Vector{Dict{String,Any}}
    uncertainty::Vector{Dict{String,Any}}
    uncertainty_details::Vector{Dict{String,Any}}  # Per-action uncertainty perception data
    innovations::Vector{Dict{String,Any}}
    knowledge::Vector{Dict{String,Any}}
    matured::Vector{Dict{String,Any}}
    summary::Vector{Dict{String,Any}}
    flush_interval::Int
    write_intermediate::Bool
end

"""
Create a new data buffer.
"""
function DataBuffer(; flush_interval::Int=10, write_intermediate::Bool=true)
    return DataBuffer(
        Dict{String,Any}[],
        Dict{String,Any}[],
        Dict{String,Any}[],
        Dict{String,Any}[],  # uncertainty_details
        Dict{String,Any}[],
        Dict{String,Any}[],
        Dict{String,Any}[],
        Dict{String,Any}[],
        flush_interval,
        write_intermediate
    )
end

"""
Clear all buffers.
"""
function clear_buffers!(buffer::DataBuffer)
    empty!(buffer.decisions)
    empty!(buffer.market)
    empty!(buffer.uncertainty)
    empty!(buffer.uncertainty_details)
    empty!(buffer.innovations)
    empty!(buffer.knowledge)
    empty!(buffer.matured)
    empty!(buffer.summary)
end

# ============================================================================
# AGENT NETWORK (SOCIAL NETWORK)
# ============================================================================

"""
Simple agent network structure (Watts-Strogatz style).
"""
mutable struct AgentNetwork
    n_agents::Int
    adjacency::Vector{Set{Int}}
    n_neighbors::Int
    rewiring_prob::Float64
end

"""
Create a Watts-Strogatz style agent network.
"""
function AgentNetwork(n_agents::Int; n_neighbors::Int=6, rewiring_prob::Float64=0.1, rng::Random.AbstractRNG=Random.default_rng())
    adjacency = [Set{Int}() for _ in 1:n_agents]

    # Create ring lattice
    k = div(n_neighbors, 2)
    for i in 1:n_agents
        for j in 1:k
            neighbor = mod(i + j - 1, n_agents) + 1
            push!(adjacency[i], neighbor)
            push!(adjacency[neighbor], i)
        end
    end

    # Rewire edges with probability p
    for i in 1:n_agents
        neighbors_list = collect(adjacency[i])
        for j in neighbors_list
            if rand(rng) < rewiring_prob && j > i
                # Remove edge
                delete!(adjacency[i], j)
                delete!(adjacency[j], i)

                # Add new random edge
                candidates = setdiff(1:n_agents, adjacency[i], [i])
                if !isempty(candidates)
                    new_neighbor = rand(rng, collect(candidates))
                    push!(adjacency[i], new_neighbor)
                    push!(adjacency[new_neighbor], i)
                end
            end
        end
    end

    return AgentNetwork(n_agents, adjacency, n_neighbors, rewiring_prob)
end

"""
Get neighbors of an agent.
"""
function get_neighbors(network::AgentNetwork, agent_id::Int)::Vector{Int}
    return collect(network.adjacency[agent_id])
end

# ============================================================================
# ENHANCED SIMULATION WITH FULL FEATURES
# ============================================================================

"""
Enhanced simulation with vectorized state and data buffering.
"""
mutable struct EnhancedSimulation
    config::EmergentConfig
    agents::Vector{EmergentAgent}
    agent_state::EnhancedVectorizedState
    market::MarketEnvironment
    uncertainty_env::KnightianUncertaintyEnvironment
    knowledge_base::KnowledgeBase
    innovation_engine::InnovationEngine
    info_system::InformationSystem  # AI information generation system
    agent_network::Union{AgentNetwork,Nothing}
    current_round::Int
    history::Vector{Dict{String,Any}}
    data_buffer::DataBuffer
    run_id::String
    output_dir::String
    data_paths::Dict{String,String}
    rng::Random.AbstractRNG
    start_time::DateTime
    last_uncertainty_state::Union{Dict{String,Any},Nothing}
    previous_uncertainty_levels::Dict{String,Float64}
    baseline_uncertainty_levels::Dict{String,Float64}  # Tracks round 1 levels for cumulative delta
end

"""
Initialize an enhanced simulation with all features.
"""
function EnhancedSimulation(;
    config::EmergentConfig = EmergentConfig(),
    output_dir::String = "results",
    run_id::String = "run_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
    seed::Union{Int,Nothing} = nothing,
    initial_tier_distribution::Union{Dict{String,Float64},Nothing} = nothing
)
    initialize!(config)

    # Seed derivation (match Python pattern with run_id hash)
    actual_seed = if !isnothing(seed)
        seed
    elseif !isnothing(config.RANDOM_SEED) && config.RANDOM_SEED > 0
        run_hash = reinterpret(UInt64, sha256(run_id)[1:8])[1] % 1_000_000
        mod(config.RANDOM_SEED + run_hash, 2^32 - 1)
    else
        rand(1:2^31-1)
    end

    # Create RNG - use NumpyRNG for cross-language reproducibility if configured
    rng = if config.USE_NUMPY_RNG
        NumpyRNG(actual_seed)
    else
        MersenneTwister(actual_seed)
    end

    # Create components
    uncertainty_env = KnightianUncertaintyEnvironment(config; rng=rng)
    market = MarketEnvironment(config; rng=rng)

    # Create knowledge base and innovation engine (matches Python)
    knowledge_base = KnowledgeBase(config)
    combination_tracker = CombinationTracker()
    innovation_engine = InnovationEngine(config, knowledge_base, combination_tracker)

    # Create information system for AI-assisted analysis
    info_system = InformationSystem(config)

    # Determine initial AI tier distribution
    tier_order = ["none", "basic", "advanced", "premium"]
    tier_probs = if isnothing(initial_tier_distribution)
        [1.0, 0.0, 0.0, 0.0]  # Default: all start at none
    else
        [get(initial_tier_distribution, t, 0.0) for t in tier_order]
    end
    total_prob = sum(tier_probs)
    if total_prob > 0
        tier_probs = tier_probs ./ total_prob
    else
        tier_probs = [1.0, 0.0, 0.0, 0.0]
    end

    # Create agents with distributed initial tiers
    agents = EmergentAgent[]
    trait_names = collect(keys(config.TRAIT_DISTRIBUTIONS))

    for i in 1:config.N_AGENTS
        # Sample initial tier based on distribution
        r = rand(rng)
        cumsum = 0.0
        initial_tier = "none"
        for (j, tier) in enumerate(tier_order)
            cumsum += tier_probs[j]
            if r <= cumsum
                initial_tier = tier
                break
            end
        end
        agent = EmergentAgent(i, config; rng=rng, fixed_ai_level=initial_tier)
        # Initialize subscription schedule for fixed AI tier (ensures costs are charged)
        if initial_tier != "none"
            ensure_subscription_schedule!(agent, initial_tier)
        end
        push!(agents, agent)
    end

    # Create vectorized state
    agent_state = EnhancedVectorizedState(config.N_AGENTS, trait_names, config)

    # Sync vectorized state with agents
    for (i, agent) in enumerate(agents)
        agent_state.capital[i] = get_capital(agent)
        agent_state.alive[i] = agent.alive
        for (trait_name, value) in agent.traits
            if haskey(agent_state.trait_indices, trait_name)
                idx = agent_state.trait_indices[trait_name]
                agent_state.traits[i, idx] = value
            end
        end
    end

    # Create agent network if enabled
    agent_network = nothing
    if config.USE_NETWORK_EFFECTS
        agent_network = AgentNetwork(
            config.N_AGENTS;
            n_neighbors=config.NETWORK_N_NEIGHBORS,
            rewiring_prob=config.NETWORK_REWIRING_PROB,
            rng=rng
        )
    end

    # Setup data paths
    data_paths = setup_data_directories(output_dir, run_id)

    # Create data buffer
    buffer_interval = getfield(config, :buffer_flush_interval)
    write_intermediate = getfield(config, :write_intermediate_batches)
    data_buffer = DataBuffer(; flush_interval=buffer_interval, write_intermediate=write_intermediate)

    return EnhancedSimulation(
        config,
        agents,
        agent_state,
        market,
        uncertainty_env,
        knowledge_base,
        innovation_engine,
        info_system,
        agent_network,
        0,
        Dict{String,Any}[],
        data_buffer,
        run_id,
        output_dir,
        data_paths,
        rng,
        now(),
        nothing,
        Dict{String,Float64}(),
        Dict{String,Float64}()  # baseline_uncertainty_levels - populated on round 1
    )
end

"""
Setup data directories for output.
"""
function setup_data_directories(base_dir::String, run_id::String)::Dict{String,String}
    run_dir = joinpath(base_dir, run_id)
    paths = Dict{String,String}(
        "base" => run_dir,
        "decisions" => joinpath(run_dir, "decisions"),
        "market" => joinpath(run_dir, "market"),
        "uncertainty" => joinpath(run_dir, "uncertainty"),
        "uncertainty_details" => joinpath(run_dir, "uncertainty_details"),
        "innovations" => joinpath(run_dir, "innovations"),
        "knowledge" => joinpath(run_dir, "knowledge"),
        "matured" => joinpath(run_dir, "matured"),
        "summary" => joinpath(run_dir, "summary")
    )

    for path in values(paths)
        mkpath(path)
    end

    return paths
end

"""
Enforce survival threshold for an agent.
Uses evaluate_failure_conditions! for full Python compatibility with all 4 failure modes:
liquidity_failure, equity_failure, funding_shock, burnout_failure.
"""
function enforce_survival_threshold!(agent::EmergentAgent, config::EmergentConfig, round_num::Int)
    if !agent.alive
        return
    end

    # Use full failure evaluation logic (matches Python _evaluate_failure_conditions)
    reason = evaluate_failure_conditions!(agent)
    if !isnothing(reason)
        agent.alive = false
        agent.failure_round = round_num  # Track when agent fails (for Kaplan-Meier)
        agent.failure_reason = reason
    end
end

"""
Apply operating costs to agents based on market conditions.
Also updates burn history tracking for distress detection.
"""
function apply_operating_costs!(
    agents::Vector{EmergentAgent},
    market::MarketEnvironment,
    market_conditions::Dict{String,Any},
    config::EmergentConfig,
    round_num::Int
)
    avg_competition = Float64(get(market_conditions, "avg_competition", 0.0))
    volatility = Float64(get(market_conditions, "volatility", config.MARKET_VOLATILITY))
    base_vol = config.MARKET_VOLATILITY

    # Compute severity multiplier
    severity = 1.0 + avg_competition * 0.35 + max(0.0, volatility - base_vol) * 0.45
    severity = clamp(severity, 0.7, 1.9)

    for agent in agents
        if !agent.alive
            continue
        end

        # Estimate operational cost and store for distress tracking
        estimated_cost = estimate_operational_costs(agent, market)
        agent.operating_cost_estimate = estimated_cost
        operating_cost = max(0.0, estimated_cost * severity)

        # Track capital before cost for burn history
        capital_before = agent.resources.capital

        if operating_cost > 0.0
            agent.resources.capital -= operating_cost
        end

        # Update burn history with capital delta
        capital_delta = agent.resources.capital - capital_before
        update_burn_history!(agent, capital_delta)

        enforce_survival_threshold!(agent, config, round_num)
    end
end

"""
Calculate outcome for an action.
"""
function calculate_action_outcome(
    action::Dict{String,Any},
    market_conditions::Dict{String,Any},
    round_num::Int
)::Dict{String,Any}
    action_type = get(action, "action", "maintain")
    ai_used = get(action, "ai_level_used", "none") != "none"

    outcome = Dict{String,Any}(
        "action" => action_type,
        "round" => round_num,
        "ai_used" => ai_used
    )

    if action_type == "invest"
        outcome["executed"] = get(action, "executed", false)
        outcome["success"] = get(action, "success", nothing)
        outcome["capital_returned"] = 0.0  # No immediate return
        outcome["investment_locked"] = get(action, "investment_amount", 0.0)
        outcome["outcome_pending"] = get(action, "outcome_pending", true)

    elseif action_type == "innovate"
        success = get(action, "success", false)
        outcome["success"] = success
        rd_spend = Float64(get(action, "rd_investment", 0.0))
        capital_returned = Float64(get(action, "capital_returned", 0.0))
        outcome["capital_returned"] = capital_returned
        outcome["net_return"] = capital_returned - rd_spend

    elseif action_type == "explore"
        outcome["success"] = true
        serendipity = get(action, "serendipity_reward", 0.0)
        outcome["capital_returned"] = serendipity

    else  # maintain
        outcome["success"] = true
        outcome["capital_returned"] = 0.0
    end

    return outcome
end

"""
Run the enhanced simulation with detailed step processing.
"""
function run!(sim::EnhancedSimulation)
    println("[$(sim.run_id)] Starting enhanced simulation...")

    for round in 1:sim.config.N_ROUNDS
        enhanced_step!(sim, round)

        # Flush buffers periodically
        if sim.data_buffer.write_intermediate && round % sim.data_buffer.flush_interval == 0
            flush_buffers!(sim, round)
        end
    end

    # Final flush
    flush_buffers!(sim, sim.config.N_ROUNDS)
    save_final_agent_state!(sim)

    println("[$(sim.run_id)] Enhanced simulation finished.")
    return sim
end

"""
Enhanced simulation step with full features.
"""
function enhanced_step!(sim::EnhancedSimulation, round::Int)
    sim.current_round = round

    # Get alive agents
    alive_indices = get_alive_indices(sim.agent_state)
    alive_agents = [sim.agents[i] for i in alive_indices]

    if isempty(alive_agents)
        return
    end

    # Get market conditions
    market_conditions = get_market_conditions(sim.market)
    if !isnothing(sim.last_uncertainty_state)
        market_conditions["uncertainty_state"] = sim.last_uncertainty_state
    end

    # Apply operating costs
    apply_operating_costs!(alive_agents, sim.market, market_conditions, sim.config, round)

    # Refresh alive agents after cost application
    alive_agents = [a for a in alive_agents if a.alive]
    if isempty(alive_agents)
        return
    end

    # Get available opportunities
    available_opportunities = get_available_opportunities(sim.market)
    uncertainty_state = get_uncertainty_state(sim.uncertainty_env)

    # Phase 1: Process matured investments
    all_matured = Dict{String,Any}[]
    for agent in alive_agents
        matured = process_matured_investments!(agent, sim.market, round)
        for m in matured
            m["agent_id"] = agent.id
            push!(all_matured, m)
        end
    end

    # Process matured outcomes - update agent states
    for outcome in all_matured
        agent_id = outcome["agent_id"]
        agent = sim.agents[agent_id]
        if !agent.alive
            continue
        end

        # Update agent state from outcome
        update_state_from_outcome!(agent, outcome)
        enforce_survival_threshold!(agent, sim.config, round)
    end

    # Refresh alive agents
    alive_agents = [a for a in alive_agents if a.alive]
    if isempty(alive_agents)
        return
    end

    # Phase 2: Collect agent decisions
    # Check if sequential decisions are enabled
    sequential_enabled = hasfield(typeof(sim.config), :SEQUENTIAL_DECISIONS_ENABLED) ?
        sim.config.SEQUENTIAL_DECISIONS_ENABLED : false

    agent_actions = Dict{String,Any}[]

    if sequential_enabled && length(alive_agents) > 1
        # Sequential decision making: early agents decide first, their choices become visible signals
        early_fraction = hasfield(typeof(sim.config), :EARLY_DECISION_FRACTION) ?
            sim.config.EARLY_DECISION_FRACTION : 0.3
        signal_weight = hasfield(typeof(sim.config), :SIGNAL_VISIBILITY_WEIGHT) ?
            sim.config.SIGNAL_VISIBILITY_WEIGHT : 0.15

        n_early = max(1, Int(floor(length(alive_agents) * early_fraction)))

        # Shuffle and split into early and late deciders
        shuffled_agents = shuffle(sim.rng, collect(alive_agents))
        early_agents = shuffled_agents[1:n_early]
        late_agents = shuffled_agents[n_early+1:end]

        # Phase 2a: Early agents decide (no visibility signals)
        early_signals = Dict{String,Int}()

        for agent in early_agents
            neighbor_agents = EmergentAgent[]
            if !isnothing(sim.agent_network)
                neighbor_ids = get_neighbors(sim.agent_network, agent.id)
                if length(neighbor_ids) > 5
                    neighbor_ids = rand(sim.rng, neighbor_ids, 5)
                end
                neighbor_agents = [sim.agents[nid] for nid in neighbor_ids if sim.agents[nid].alive]
            end

            agent_opportunities = get_opportunities_for_agent(sim.market, agent)
            if isempty(agent_opportunities)
                agent_opportunities = available_opportunities
            end

            decision = make_decision!(
                agent,
                agent_opportunities,
                market_conditions,
                sim.market,
                round;
                uncertainty_env=sim.uncertainty_env,
                neighbor_agents=neighbor_agents,
                innovation_engine=sim.innovation_engine,
                info_system=sim.info_system
            )

            decision["agent_id"] = agent.id
            push!(agent_actions, decision)

            # Record visible signal for invest actions
            if get(decision, "action", "") == "invest"
                opp_id = string(get(decision, "opportunity_id", ""))
                if !isempty(opp_id)
                    early_signals[opp_id] = get(early_signals, opp_id, 0) + 1
                end
            end
        end

        # Phase 2b: Late agents decide with visible signals
        for agent in late_agents
            neighbor_agents = EmergentAgent[]
            if !isnothing(sim.agent_network)
                neighbor_ids = get_neighbors(sim.agent_network, agent.id)
                if length(neighbor_ids) > 5
                    neighbor_ids = rand(sim.rng, neighbor_ids, 5)
                end
                neighbor_agents = [sim.agents[nid] for nid in neighbor_ids if sim.agents[nid].alive]
            end

            agent_opportunities = get_opportunities_for_agent(sim.market, agent)
            if isempty(agent_opportunities)
                agent_opportunities = available_opportunities
            end

            decision = make_decision!(
                agent,
                agent_opportunities,
                market_conditions,
                sim.market,
                round;
                uncertainty_env=sim.uncertainty_env,
                neighbor_agents=neighbor_agents,
                innovation_engine=sim.innovation_engine,
                info_system=sim.info_system,
                early_signals=early_signals,
                signal_weight=signal_weight
            )

            decision["agent_id"] = agent.id
            push!(agent_actions, decision)
        end
    else
        # Original simultaneous decision logic
        for agent in alive_agents
            # Get neighbor agents from network
            neighbor_agents = EmergentAgent[]
            if !isnothing(sim.agent_network)
                neighbor_ids = get_neighbors(sim.agent_network, agent.id)
                if length(neighbor_ids) > 5
                    neighbor_ids = rand(sim.rng, neighbor_ids, 5)
                end
                neighbor_agents = [sim.agents[nid] for nid in neighbor_ids if sim.agents[nid].alive]
            end

            # Get opportunities available to this agent
            agent_opportunities = get_opportunities_for_agent(sim.market, agent)
            if isempty(agent_opportunities)
                agent_opportunities = available_opportunities
            end

            # Make decision with full InnovationEngine and InformationSystem integration (matches Python)
            decision = make_decision!(
                agent,
                agent_opportunities,
                market_conditions,
                sim.market,
                round;
                uncertainty_env=sim.uncertainty_env,
                neighbor_agents=neighbor_agents,
                innovation_engine=sim.innovation_engine,
                info_system=sim.info_system
            )

            decision["agent_id"] = agent.id
            push!(agent_actions, decision)
        end
    end

    # Record AI signals
    record_ai_signals!(sim.uncertainty_env, round, agent_actions)

    # Phase 3: Process immediate action outcomes (non-invest)
    innovations_this_round = Innovation[]

    for action in agent_actions
        agent_id = action["agent_id"]
        agent = sim.agents[agent_id]
        if !agent.alive
            continue
        end

        action_type = get(action, "action", "maintain")

        if action_type == "innovate"
            # Process innovation
            success = get(action, "success", false)
            rd_spend = Float64(get(action, "rd_investment", 0.0))
            cash_multiple = Float64(get(action, "cash_multiple", success ? 1.5 : 0.15))

            capital_return = rd_spend * cash_multiple
            action["capital_returned"] = capital_return
            action["net_return"] = capital_return - rd_spend

            if success
                innov = Innovation(
                    id=generate_innovation_id(round, agent_id, length(innovations_this_round) + 1),
                    type="incremental",
                    knowledge_components=String[],
                    novelty=rand(sim.rng),
                    quality=rand(sim.rng),
                    round_created=round,
                    creator_id=agent_id,
                    ai_level_used=get(action, "ai_level_used", "none")
                )
                push!(innovations_this_round, innov)
                # Spawn derivative opportunity from successful innovation (Python match)
                spawn_opportunity_from_innovation!(sim.market, innov, cash_multiple)
            end
        end

        if action_type != "invest"
            outcome = calculate_action_outcome(action, market_conditions, round)
            update_state_from_outcome!(agent, outcome)
            enforce_survival_threshold!(agent, sim.config, round)
        end
    end

    # Phase 3b: Apply novelty disruption for high-novelty innovations
    # This implements the "DeepSeek Effect" - novel innovations disrupt crowded opportunities
    novelty_disruption_enabled = hasfield(typeof(sim.config), :NOVELTY_DISRUPTION_ENABLED) ?
        sim.config.NOVELTY_DISRUPTION_ENABLED : true
    if novelty_disruption_enabled
        for innov in innovations_this_round
            disrupted_count = apply_novelty_disruption!(sim.market, innov)
            if disrupted_count > 0
                # Track disruption in action for the innovator
                for action in agent_actions
                    if get(action, "agent_id", 0) == innov.creator_id &&
                       get(action, "action", "") == "innovate"
                        action["disrupted_opportunities"] = disrupted_count
                        break
                    end
                end
            end
        end
    end

    # Phase 4: Update vectorized state
    for i in 1:sim.config.N_AGENTS
        agent = sim.agents[i]
        sim.agent_state.capital[i] = get_capital(agent)
        sim.agent_state.alive[i] = agent.alive
        sim.agent_state.portfolio_size[i] = agent.portfolio_size
        sim.agent_state.innovation_count[i] = agent.innovation_count

        initial_equity = sim.agent_state.initial_capital[i]
        if initial_equity > 0
            sim.agent_state.capital_growth[i] = sim.agent_state.capital[i] / initial_equity
        end
    end

    # Phase 5: Update market
    market_state = step!(sim.market, round, agent_actions, innovations_this_round; matured_outcomes=all_matured)

    # Phase 6: Update uncertainty
    uncertainty_state = measure_uncertainty_state!(
        sim.uncertainty_env,
        sim.market,
        agent_actions,
        innovations_this_round,
        round
    )
    sim.last_uncertainty_state = uncertainty_state

    # Phase 7: Record to buffer
    record_round_to_buffer!(sim, round, agent_actions, market_state, uncertainty_state,
                           innovations_this_round, all_matured, alive_agents)

    # Phase 8: Compile and store history
    round_stats = compile_enhanced_round_stats(sim, round, agent_actions, all_matured, uncertainty_state)
    push!(sim.history, round_stats)
end

"""
Record round data to buffer for batch writing.
"""
function record_round_to_buffer!(
    sim::EnhancedSimulation,
    round::Int,
    decisions::Vector{Dict{String,Any}},
    market_state::Dict{String,Any},
    uncertainty_state::Dict{String,Dict{String,Any}},
    innovations::Vector{Innovation},
    matured_outcomes::Vector{Dict{String,Any}},
    alive_agents::Vector{EmergentAgent}
)
    # Buffer decisions
    for action in decisions
        clean_action = Dict{String,Any}(
            "run_id" => sim.run_id,
            "round" => round,
            "action" => get(action, "action", "maintain"),
            "agent_id" => get(action, "agent_id", 0),
            "success" => get(action, "success", nothing),
            "ai_level_used" => get(action, "ai_level_used", "none"),
            "portfolio_size" => get(action, "portfolio_size", 0)
        )

        for field in ["investment_amount", "rd_investment", "cost", "capital_deployed", "capital_returned"]
            if haskey(action, field)
                clean_action[field] = action[field]
            end
        end

        push!(sim.data_buffer.decisions, clean_action)
    end

    # Buffer market state
    if !isempty(market_state)
        market_record = Dict{String,Any}(
            "run_id" => sim.run_id,
            "round" => round
        )
        merge!(market_record, market_state)
        push!(sim.data_buffer.market, market_record)
    end

    # Buffer uncertainty state with transformation metrics
    if !isempty(uncertainty_state)
        # Extract current levels
        actor_level = Float64(get(get(uncertainty_state, "actor_ignorance", Dict()), "level", 0.0))
        practical_level = Float64(get(get(uncertainty_state, "practical_indeterminism", Dict()), "level", 0.0))
        agentic_level = Float64(get(get(uncertainty_state, "agentic_novelty", Dict()), "level", 0.0))
        competitive_level = Float64(get(get(uncertainty_state, "competitive_recursion", Dict()), "level", 0.0))

        # Store baseline on round 1 (or first round with uncertainty data)
        if isempty(sim.baseline_uncertainty_levels)
            sim.baseline_uncertainty_levels["actor_ignorance"] = actor_level
            sim.baseline_uncertainty_levels["practical_indeterminism"] = practical_level
            sim.baseline_uncertainty_levels["agentic_novelty"] = agentic_level
            sim.baseline_uncertainty_levels["competitive_recursion"] = competitive_level
        end

        # Get previous levels (default to current if first round)
        prev_actor = get(sim.previous_uncertainty_levels, "actor_ignorance", actor_level)
        prev_practical = get(sim.previous_uncertainty_levels, "practical_indeterminism", practical_level)
        prev_agentic = get(sim.previous_uncertainty_levels, "agentic_novelty", agentic_level)
        prev_competitive = get(sim.previous_uncertainty_levels, "competitive_recursion", competitive_level)

        # Get baseline levels
        base_actor = get(sim.baseline_uncertainty_levels, "actor_ignorance", actor_level)
        base_practical = get(sim.baseline_uncertainty_levels, "practical_indeterminism", practical_level)
        base_agentic = get(sim.baseline_uncertainty_levels, "agentic_novelty", agentic_level)
        base_competitive = get(sim.baseline_uncertainty_levels, "competitive_recursion", competitive_level)

        # Compute delta (round-over-round change)
        delta_actor = actor_level - prev_actor
        delta_practical = practical_level - prev_practical
        delta_agentic = agentic_level - prev_agentic
        delta_competitive = competitive_level - prev_competitive

        # Compute cumulative delta (change from baseline)
        cumulative_delta_actor = actor_level - base_actor
        cumulative_delta_practical = practical_level - base_practical
        cumulative_delta_agentic = agentic_level - base_agentic
        cumulative_delta_competitive = competitive_level - base_competitive

        # Compute portfolio composition (shares)
        uncertainty_total = actor_level + practical_level + agentic_level + competitive_level
        total_safe = max(uncertainty_total, 0.001)  # Avoid division by zero
        share_actor = actor_level / total_safe
        share_practical = practical_level / total_safe
        share_agentic = agentic_level / total_safe
        share_competitive = competitive_level / total_safe

        # Compute HHI (Herfindahl-Hirschman Index) - concentration measure
        uncertainty_hhi = share_actor^2 + share_practical^2 + share_agentic^2 + share_competitive^2

        # Compute entropy - diversity measure (avoid log(0))
        eps = 1e-10
        uncertainty_entropy = -(
            share_actor * log(share_actor + eps) +
            share_practical * log(share_practical + eps) +
            share_agentic * log(share_agentic + eps) +
            share_competitive * log(share_competitive + eps)
        )

        flat_uncertainty = Dict{String,Any}(
            "run_id" => sim.run_id,
            "round" => round,
            # Current levels
            "actor_ignorance_level" => actor_level,
            "practical_indeterminism_level" => practical_level,
            "agentic_novelty_level" => agentic_level,
            "competitive_recursion_level" => competitive_level,
            # Round-over-round deltas (transformation rate)
            "delta_actor_ignorance" => delta_actor,
            "delta_practical_indeterminism" => delta_practical,
            "delta_agentic_novelty" => delta_agentic,
            "delta_competitive_recursion" => delta_competitive,
            # Cumulative change from baseline (total transformation)
            "cumulative_delta_actor" => cumulative_delta_actor,
            "cumulative_delta_practical" => cumulative_delta_practical,
            "cumulative_delta_agentic" => cumulative_delta_agentic,
            "cumulative_delta_competitive" => cumulative_delta_competitive,
            # Portfolio composition
            "uncertainty_total" => uncertainty_total,
            "share_actor_ignorance" => share_actor,
            "share_practical_indeterminism" => share_practical,
            "share_agentic_novelty" => share_agentic,
            "share_competitive_recursion" => share_competitive,
            # Concentration and diversity measures
            "uncertainty_hhi" => uncertainty_hhi,
            "uncertainty_entropy" => uncertainty_entropy
        )
        push!(sim.data_buffer.uncertainty, flat_uncertainty)

        # Update previous levels for next round
        sim.previous_uncertainty_levels["actor_ignorance"] = actor_level
        sim.previous_uncertainty_levels["practical_indeterminism"] = practical_level
        sim.previous_uncertainty_levels["agentic_novelty"] = agentic_level
        sim.previous_uncertainty_levels["competitive_recursion"] = competitive_level
    end

    # Buffer innovations
    for inn in innovations
        push!(sim.data_buffer.innovations, Dict{String,Any}(
            "run_id" => sim.run_id,
            "round" => round,
            "type" => inn.type,
            "quality" => inn.quality,
            "novelty" => inn.novelty,
            "creator_id" => inn.creator_id
        ))
    end

    # Buffer matured outcomes
    for outcome in matured_outcomes
        push!(sim.data_buffer.matured, Dict{String,Any}(
            "run_id" => sim.run_id,
            "round" => round,
            "agent_id" => get(outcome, "agent_id", 0),
            "success" => get(outcome, "success", false),
            "capital_returned" => get(outcome, "capital_returned", 0.0),
            "ai_level_used" => get(outcome, "ai_level_used", "none")
        ))
    end

    # Buffer summary
    alive_count = length(alive_agents)
    capitals = [get_capital(a) for a in alive_agents]
    mean_capital = isempty(capitals) ? 0.0 : mean(capitals)

    ai_shares = Dict{String,Float64}("none" => 0.0, "basic" => 0.0, "advanced" => 0.0, "premium" => 0.0)
    for agent in alive_agents
        level = get_ai_level(agent)
        if haskey(ai_shares, level)
            ai_shares[level] += 1.0
        end
    end
    total_alive = max(1, alive_count)
    for k in keys(ai_shares)
        ai_shares[k] /= total_alive
    end

    action_counts = Dict{String,Int}()
    for action in decisions
        act = get(action, "action", "maintain")
        action_counts[act] = get(action_counts, act, 0) + 1
    end
    total_actions = max(1, length(decisions))

    summary_entry = Dict{String,Any}(
        "round" => round,
        "alive_agents" => alive_count,
        "dead_agents" => sim.config.N_AGENTS - alive_count,
        "mean_capital" => mean_capital,
        "ai_share_none" => ai_shares["none"],
        "ai_share_basic" => ai_shares["basic"],
        "ai_share_advanced" => ai_shares["advanced"],
        "ai_share_premium" => ai_shares["premium"],
        "action_share_invest" => get(action_counts, "invest", 0) / total_actions,
        "action_share_innovate" => get(action_counts, "innovate", 0) / total_actions,
        "action_share_explore" => get(action_counts, "explore", 0) / total_actions,
        "action_share_maintain" => get(action_counts, "maintain", 0) / total_actions
    )
    push!(sim.data_buffer.summary, summary_entry)

    # Buffer uncertainty_details (per-action perception data - matches Python)
    for action in decisions
        agent_id = get(action, "agent_id", 0)
        if agent_id < 1 || agent_id > length(sim.agents)
            continue
        end
        agent = sim.agents[agent_id]
        perception = get(action, "perception", Dict{String,Any}())

        detail_entry = Dict{String,Any}(
            "run_id" => sim.run_id,
            "round" => round,
            "agent_id" => agent_id,
            "action" => get(action, "action", "maintain"),
            "ai_level_used" => get(action, "ai_level_used", "none"),
            # Decision confidence from perception
            "decision_confidence" => Float64(get(perception, "decision_confidence", 0.5)),
            # Four dimensions of perceived uncertainty
            "actor_ignorance_perceived" => Float64(get(perception, "actor_ignorance", 0.5)),
            "practical_indeterminism_perceived" => Float64(get(perception, "practical_indeterminism", 0.5)),
            "agentic_novelty_perceived" => Float64(get(perception, "agentic_novelty", 0.5)),
            "competitive_recursion_perceived" => Float64(get(perception, "competitive_recursion", 0.5)),
            # Additional perception signals
            "knowledge_signal" => Float64(get(perception, "knowledge_signal", 0.5)),
            "execution_risk" => Float64(get(perception, "execution_risk", 0.5)),
            "innovation_signal" => Float64(get(perception, "innovation_signal", 0.5)),
            "competition_signal" => Float64(get(perception, "competition_signal", 0.5)),
            # Agent state
            "ai_trust" => Float64(get(agent.traits, "ai_trust", 0.5)),
            "analytical_ability" => Float64(get(agent.traits, "analytical_ability", 0.5)),
            "exploration_tendency" => Float64(get(agent.traits, "exploration_tendency", 0.5)),
            # Investment-specific data (if applicable)
            "amount" => Float64(get(action, "amount", 0.0)),
            "opportunity_id" => string(get(action, "opportunity_id", "")),
            "success" => get(action, "success", nothing)
        )
        push!(sim.data_buffer.uncertainty_details, detail_entry)
    end
end

"""
Flush data buffers to disk.
"""
function flush_buffers!(sim::EnhancedSimulation, round::Int)
    base_path = sim.data_paths["base"]

    # Write decisions
    if !isempty(sim.data_buffer.decisions)
        df = DataFrame(sim.data_buffer.decisions)
        save_dataframe_arrow(df, joinpath(sim.data_paths["decisions"], "batch_$(round).arrow"))
        empty!(sim.data_buffer.decisions)
    end

    # Write market
    if !isempty(sim.data_buffer.market)
        df = DataFrame(sim.data_buffer.market)
        save_dataframe_arrow(df, joinpath(sim.data_paths["market"], "batch_$(round).arrow"))
        empty!(sim.data_buffer.market)
    end

    # Write uncertainty
    if !isempty(sim.data_buffer.uncertainty)
        df = DataFrame(sim.data_buffer.uncertainty)
        save_dataframe_arrow(df, joinpath(sim.data_paths["uncertainty"], "batch_$(round).arrow"))
        empty!(sim.data_buffer.uncertainty)
    end

    # Write uncertainty_details (per-action perception data)
    if !isempty(sim.data_buffer.uncertainty_details)
        df = DataFrame(sim.data_buffer.uncertainty_details)
        save_dataframe_arrow(df, joinpath(sim.data_paths["uncertainty_details"], "batch_$(round).arrow"))
        empty!(sim.data_buffer.uncertainty_details)
    end

    # Write innovations
    if !isempty(sim.data_buffer.innovations)
        df = DataFrame(sim.data_buffer.innovations)
        save_dataframe_arrow(df, joinpath(sim.data_paths["innovations"], "batch_$(round).arrow"))
        empty!(sim.data_buffer.innovations)
    end

    # Write matured
    if !isempty(sim.data_buffer.matured)
        df = DataFrame(sim.data_buffer.matured)
        save_dataframe_arrow(df, joinpath(sim.data_paths["matured"], "batch_$(round).arrow"))
        empty!(sim.data_buffer.matured)
    end

    # Write summary
    if !isempty(sim.data_buffer.summary)
        df = DataFrame(sim.data_buffer.summary)
        save_dataframe_arrow(df, joinpath(sim.data_paths["summary"], "batch_$(round).arrow"))
        empty!(sim.data_buffer.summary)
    end
end

"""
Save final agent state to disk.
"""
function save_final_agent_state!(sim::EnhancedSimulation)
    agent_records = Dict{String,Any}[]

    for agent in sim.agents
        initial_equity = sim.agent_state.initial_capital[agent.id]
        capital_growth = initial_equity > 0 ? get_capital(agent) / initial_equity : 0.0

        record = Dict{String,Any}(
            "run_id" => sim.run_id,
            "agent_id" => agent.id,
            "survived" => agent.alive,
            "final_capital" => get_capital(agent),
            "capital_growth" => capital_growth,
            "primary_ai_level" => get_ai_level(agent),
            "innovations" => agent.innovation_count,
            "portfolio_size" => agent.portfolio_size
        )

        # Add traits
        for (trait_name, value) in agent.traits
            record[trait_name] = value
        end

        push!(agent_records, record)
    end

    df = DataFrame(agent_records)
    save_dataframe_csv(df, joinpath(sim.data_paths["base"], "final_agents.csv"))
    save_dataframe_arrow(df, joinpath(sim.data_paths["base"], "final_agents.arrow"))
end

"""
Compile enhanced round statistics.
"""
function compile_enhanced_round_stats(
    sim::EnhancedSimulation,
    round::Int,
    agent_actions::Vector{Dict{String,Any}},
    matured_outcomes::Vector{Dict{String,Any}},
    uncertainty_state::Dict{String,Dict{String,Any}}
)::Dict{String,Any}
    alive_agents = [a for a in sim.agents if a.alive]
    n_alive = length(alive_agents)
    n_total = length(sim.agents)

    # Capital statistics
    capitals = [get_capital(a) for a in alive_agents]
    mean_capital = isempty(capitals) ? 0.0 : mean(capitals)
    median_capital = isempty(capitals) ? 0.0 : median(capitals)
    std_capital = isempty(capitals) || length(capitals) < 2 ? 0.0 : std(capitals)

    # Action counts
    action_counts = Dict{String,Int}()
    ai_usage = Dict{String,Int}("none" => 0, "basic" => 0, "advanced" => 0, "premium" => 0)

    total_capital_deployed = 0.0
    total_capital_returned = 0.0

    for action in agent_actions
        act_type = get(action, "action", "maintain")
        action_counts[act_type] = get(action_counts, act_type, 0) + 1

        ai_level = lowercase(string(get(action, "ai_level_used", "none")))
        if haskey(ai_usage, ai_level)
            ai_usage[ai_level] += 1
        end

        total_capital_deployed += Float64(get(action, "capital_deployed", 0.0))
        total_capital_returned += Float64(get(action, "capital_returned", 0.0))
    end

    # Matured investment stats
    n_matured = length(matured_outcomes)
    n_success = count(o -> get(o, "success", false), matured_outcomes)
    n_failure = n_matured - n_success

    for outcome in matured_outcomes
        total_capital_returned += Float64(get(outcome, "capital_returned", 0.0))
    end

    # AI trust and diversity
    mean_ai_trust = 0.0
    mean_diversity = 0.0
    if !isempty(alive_agents)
        trust_vals = [get(a.traits, "ai_trust", 0.5) for a in alive_agents]
        mean_ai_trust = mean(trust_vals)

        diversity_vals = [a.portfolio_diversity for a in alive_agents if isfinite(a.portfolio_diversity)]
        mean_diversity = isempty(diversity_vals) ? 0.0 : mean(diversity_vals)
    end

    # ROE/ROIC
    mean_roe = 0.0
    if !isempty(alive_agents)
        roe_vals = Float64[]
        for agent in alive_agents
            initial = sim.agent_state.initial_capital[agent.id]
            if initial > 0
                push!(roe_vals, (get_capital(agent) - initial) / initial)
            end
        end
        mean_roe = isempty(roe_vals) ? 0.0 : mean(roe_vals)
    end

    # Innovation stats
    innovation_attempts = get(action_counts, "innovate", 0)
    innovation_successes = count(a -> get(a, "action", "") == "innovate" && get(a, "success", false), agent_actions)
    innovation_success_rate = innovation_attempts > 0 ? innovation_successes / innovation_attempts : 0.0

    # Uncertainty levels
    actor_ignorance = get(get(uncertainty_state, "actor_ignorance", Dict()), "level", 0.0)
    practical_indet = get(get(uncertainty_state, "practical_indeterminism", Dict()), "level", 0.0)
    agentic_novelty = get(get(uncertainty_state, "agentic_novelty", Dict()), "level", 0.0)
    competitive_rec = get(get(uncertainty_state, "competitive_recursion", Dict()), "level", 0.0)

    return Dict{String,Any}(
        "round" => round,
        "n_alive" => n_alive,
        "n_total" => n_total,
        "survival_rate" => n_total > 0 ? n_alive / n_total : 0.0,
        "mean_capital" => mean_capital,
        "median_capital" => median_capital,
        "std_capital" => std_capital,
        "total_capital" => sum(capitals),
        "total_capital_deployed" => total_capital_deployed,
        "total_capital_returned" => total_capital_returned,
        "net_capital_flow" => total_capital_returned - total_capital_deployed,
        "invest_count" => get(action_counts, "invest", 0),
        "innovate_count" => get(action_counts, "innovate", 0),
        "explore_count" => get(action_counts, "explore", 0),
        "maintain_count" => get(action_counts, "maintain", 0),
        "ai_none_count" => ai_usage["none"],
        "ai_basic_count" => ai_usage["basic"],
        "ai_advanced_count" => ai_usage["advanced"],
        "ai_premium_count" => ai_usage["premium"],
        "n_matured" => n_matured,
        "n_success" => n_success,
        "n_failure" => n_failure,
        "success_rate" => n_matured > 0 ? n_success / n_matured : 0.0,
        "mean_ai_trust" => mean_ai_trust,
        "mean_portfolio_diversity" => mean_diversity,
        "mean_roe" => mean_roe,
        "innovation_attempts" => innovation_attempts,
        "innovation_successes" => innovation_successes,
        "innovation_success_rate" => innovation_success_rate,
        "actor_ignorance" => actor_ignorance,
        "practical_indeterminism" => practical_indet,
        "agentic_novelty" => agentic_novelty,
        "competitive_recursion" => competitive_rec
    )
end

"""
Get summary statistics for enhanced simulation.
"""
function summary_stats(sim::EnhancedSimulation)::Dict{String,Any}
    alive_agents = [a for a in sim.agents if a.alive]

    survival_rate = length(alive_agents) / length(sim.agents)

    final_capitals = [get_capital(a) for a in alive_agents]
    mean_final_capital = isempty(final_capitals) ? 0.0 : mean(final_capitals)

    ai_distribution = Dict{String,Int}("none" => 0, "basic" => 0, "advanced" => 0, "premium" => 0)
    for agent in sim.agents
        tier = get_ai_level(agent)
        if haskey(ai_distribution, tier)
            ai_distribution[tier] += 1
        end
    end

    total_successes = sum(a.success_count for a in sim.agents)
    total_failures = sum(a.failure_count for a in sim.agents)
    total_innovations = sum(a.innovation_count for a in sim.agents)

    # Uncertainty averages
    if !isempty(sim.history)
        mean_actor_ignorance = mean(get(h, "actor_ignorance", 0.0) for h in sim.history)
        mean_practical_indet = mean(get(h, "practical_indeterminism", 0.0) for h in sim.history)
        mean_agentic_novelty = mean(get(h, "agentic_novelty", 0.0) for h in sim.history)
        mean_competitive_rec = mean(get(h, "competitive_recursion", 0.0) for h in sim.history)
    else
        mean_actor_ignorance = 0.0
        mean_practical_indet = 0.0
        mean_agentic_novelty = 0.0
        mean_competitive_rec = 0.0
    end

    return Dict{String,Any}(
        "run_id" => sim.run_id,
        "n_agents" => length(sim.agents),
        "n_rounds" => sim.config.N_ROUNDS,
        "final_survival_rate" => survival_rate,
        "n_survivors" => length(alive_agents),
        "mean_final_capital" => mean_final_capital,
        "total_successes" => total_successes,
        "total_failures" => total_failures,
        "total_innovations" => total_innovations,
        "ai_none_count" => ai_distribution["none"],
        "ai_basic_count" => ai_distribution["basic"],
        "ai_advanced_count" => ai_distribution["advanced"],
        "ai_premium_count" => ai_distribution["premium"],
        "mean_actor_ignorance" => mean_actor_ignorance,
        "mean_practical_indeterminism" => mean_practical_indet,
        "mean_agentic_novelty" => mean_agentic_novelty,
        "mean_competitive_recursion" => mean_competitive_rec,
        "elapsed_seconds" => (now() - sim.start_time).value / 1000.0
    )
end

# ============================================================================
# PARALLEL BATCH SIMULATION
# ============================================================================

"""
Run batch simulations with optional parallelization.
"""
function run_parallel_batch(;
    base_config::EmergentConfig = EmergentConfig(),
    n_runs::Int = 10,
    output_base::String = "results",
    fixed_ai_levels::Vector{String} = String[],
    parallel::Bool = true,
    n_threads::Int = Threads.nthreads()
)::Vector{Dict{String,Any}}
    results = Dict{String,Any}[]

    if isempty(fixed_ai_levels)
        # Adaptive AI runs
        configs = [(i, nothing) for i in 1:n_runs]
    else
        # Fixed AI tier sweep
        configs = [(run_idx, ai_level)
                   for ai_level in fixed_ai_levels
                   for run_idx in 1:n_runs]
    end

    if parallel && n_threads > 1
        # Parallel execution
        results_lock = ReentrantLock()

        Threads.@threads for (idx, ai_level) in collect(configs)
            config = deepcopy(base_config)

            if isnothing(ai_level)
                config.RANDOM_SEED = base_config.RANDOM_SEED + idx
                run_id = "run_$(idx)"
            else
                tier_offset = findfirst(==(ai_level), fixed_ai_levels)
                config.RANDOM_SEED = base_config.RANDOM_SEED + (tier_offset - 1) * n_runs + idx
                run_id = "Fixed_AI_Level_$(ai_level)_run_$(idx)"
            end

            output_dir = joinpath(output_base, run_id)

            sim = EnhancedSimulation(
                config=config,
                output_dir=output_dir,
                run_id=run_id,
                seed=config.RANDOM_SEED
            )

            if !isnothing(ai_level)
                for agent in sim.agents
                    agent.fixed_ai_level = ai_level
                    agent.current_ai_level = ai_level
                    # Start subscription schedule if this is a subscription tier
                    ensure_subscription_schedule!(agent, ai_level)
                end
            end

            run!(sim)
            stats = summary_stats(sim)

            lock(results_lock) do
                push!(results, stats)
            end
        end
    else
        # Sequential execution
        for (idx, ai_level) in configs
            config = deepcopy(base_config)

            if isnothing(ai_level)
                config.RANDOM_SEED = base_config.RANDOM_SEED + idx
                run_id = "run_$(idx)"
            else
                tier_offset = findfirst(==(ai_level), fixed_ai_levels)
                config.RANDOM_SEED = base_config.RANDOM_SEED + (tier_offset - 1) * n_runs + idx
                run_id = "Fixed_AI_Level_$(ai_level)_run_$(idx)"
            end

            output_dir = joinpath(output_base, run_id)

            sim = EnhancedSimulation(
                config=config,
                output_dir=output_dir,
                run_id=run_id,
                seed=config.RANDOM_SEED
            )

            if !isnothing(ai_level)
                for agent in sim.agents
                    agent.fixed_ai_level = ai_level
                    agent.current_ai_level = ai_level
                    # Start subscription schedule if this is a subscription tier
                    ensure_subscription_schedule!(agent, ai_level)
                end
            end

            run!(sim)
            push!(results, summary_stats(sim))
        end
    end

    return results
end

"""
Aggregate parallel batch results into DataFrame.
"""
function aggregate_parallel_results(results::Vector{Dict{String,Any}})::DataFrame
    if isempty(results)
        return DataFrame()
    end

    cols = collect(keys(results[1]))
    df = DataFrame()
    for col in cols
        df[!, Symbol(col)] = [get(r, col, missing) for r in results]
    end

    return df
end
