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
    current_round::Int
    history::Vector{Dict{String,Any}}
    run_id::String
    output_dir::String
    rng::AbstractRNG
    start_time::DateTime
end

"""
Initialize a new simulation.
"""
function EmergentSimulation(;
    config::EmergentConfig = EmergentConfig(),
    output_dir::String = "results",
    run_id::String = "run_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
    seed::Union{Int,Nothing} = nothing
)
    # Initialize configuration
    initialize!(config)

    # Set up RNG
    actual_seed = isnothing(seed) ? config.RANDOM_SEED : seed
    rng = MersenneTwister(actual_seed)

    # Create uncertainty environment
    uncertainty_env = KnightianUncertaintyEnvironment(config; rng=rng)

    # Create market environment
    market = MarketEnvironment(config; rng=rng)

    # Create agents
    agents = EmergentAgent[]
    for i in 1:config.N_AGENTS
        agent = EmergentAgent(i, config; rng=rng)
        push!(agents, agent)
    end

    return EmergentSimulation(
        config,
        agents,
        market,
        uncertainty_env,
        0,
        Dict{String,Any}[],
        run_id,
        output_dir,
        rng,
        now()
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
            mean_capital = mean([get_capital(a) for a in sim.agents if a.alive])
            # println("[$(sim.run_id)] Round $round: $(alive_count)/$(sim.config.N_AGENTS) alive, mean capital: \$$(round(Int, mean_capital))")
        end
    end

    println("[$(sim.run_id)] Simulation finished.")
    return sim
end

"""
Execute a single simulation round.
"""
function step!(sim::EmergentSimulation, round::Int)
    sim.current_round = round

    # Get available opportunities
    available_opportunities = get_available_opportunities(sim.market)

    # Get current uncertainty state
    uncertainty_state = get_uncertainty_state(sim.uncertainty_env)

    # Phase 1: Collect agent decisions
    agent_actions = Dict{String,Any}[]
    for agent in sim.agents
        if !agent.alive
            continue
        end

        # Select action
        action = select_action(
            agent,
            get_market_conditions(sim.market),
            uncertainty_state;
            available_opportunities=available_opportunities
        )

        # Select opportunity if investing
        opportunity = nothing
        if action == "invest" && !isempty(available_opportunities)
            opportunity = rand(sim.rng, available_opportunities)
        end

        # Execute action
        outcome = execute_action!(agent, action, sim.market, round; opportunity=opportunity)
        push!(agent_actions, outcome)
    end

    # Phase 2: Process matured investments
    all_matured = Dict{String,Any}[]
    for agent in sim.agents
        matured = process_matured_investments!(agent, sim.market, round)
        append!(all_matured, matured)
    end

    # Phase 3: Apply operational costs
    for agent in sim.agents
        apply_operational_costs!(agent, round)
    end

    # Phase 4: Check survival
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

    # Action counts
    action_counts = Dict{String,Int}()
    ai_usage = Dict{String,Int}("none" => 0, "basic" => 0, "advanced" => 0, "premium" => 0)
    for action in agent_actions
        act_type = get(action, "action", "maintain")
        action_counts[act_type] = get(action_counts, act_type, 0) + 1

        ai_level = lowercase(string(get(action, "ai_level_used", "none")))
        if haskey(ai_usage, ai_level)
            ai_usage[ai_level] += 1
        end
    end

    # Matured investment stats
    n_matured = length(matured_outcomes)
    n_success = count(o -> get(o, "success", false), matured_outcomes)
    n_failure = n_matured - n_success

    # Uncertainty levels
    actor_ignorance = get(get(uncertainty_state, "actor_ignorance", Dict()), "level", 0.0)
    practical_indet = get(get(uncertainty_state, "practical_indeterminism", Dict()), "level", 0.0)
    agentic_novelty = get(get(uncertainty_state, "agentic_novelty", Dict()), "level", 0.0)
    competitive_rec = get(get(uncertainty_state, "competitive_recursion", Dict()), "level", 0.0)

    return Dict{String,Any}(
        "round" => round,
        "n_alive" => n_alive,
        "n_total" => n_total,
        "survival_rate" => n_alive / n_total,
        "mean_capital" => mean_capital,
        "std_capital" => std_capital,
        "total_capital" => sum(capitals),
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
        "actor_ignorance" => actor_ignorance,
        "practical_indeterminism" => practical_indet,
        "agentic_novelty" => agentic_novelty,
        "competitive_recursion" => competitive_rec
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
function AgentNetwork(n_agents::Int; n_neighbors::Int=6, rewiring_prob::Float64=0.1, rng::AbstractRNG=Random.default_rng())
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
    agent_network::Union{AgentNetwork,Nothing}
    current_round::Int
    history::Vector{Dict{String,Any}}
    data_buffer::DataBuffer
    run_id::String
    output_dir::String
    data_paths::Dict{String,String}
    rng::AbstractRNG
    start_time::DateTime
    last_uncertainty_state::Union{Dict{String,Any},Nothing}
    previous_uncertainty_levels::Dict{String,Float64}
end

"""
Initialize an enhanced simulation with all features.
"""
function EnhancedSimulation(;
    config::EmergentConfig = EmergentConfig(),
    output_dir::String = "results",
    run_id::String = "run_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
    seed::Union{Int,Nothing} = nothing
)
    initialize!(config)

    actual_seed = isnothing(seed) ? config.RANDOM_SEED : seed
    rng = MersenneTwister(actual_seed)

    # Create components
    uncertainty_env = KnightianUncertaintyEnvironment(config; rng=rng)
    market = MarketEnvironment(config; rng=rng)

    # Create agents
    agents = EmergentAgent[]
    trait_names = collect(keys(config.TRAIT_DISTRIBUTIONS))

    for i in 1:config.N_AGENTS
        agent = EmergentAgent(i, config; rng=rng)
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
    buffer_interval = get(config.parameters, "buffer_flush_interval", 10)
    write_intermediate = get(config.parameters, "write_intermediate_batches", true)
    data_buffer = DataBuffer(; flush_interval=buffer_interval, write_intermediate=write_intermediate)

    return EnhancedSimulation(
        config,
        agents,
        agent_state,
        market,
        uncertainty_env,
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
        Dict{String,Float64}()
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
"""
function enforce_survival_threshold!(agent::EmergentAgent, config::EmergentConfig)
    if !agent.alive
        return
    end

    capital = get_capital(agent)
    survival_threshold = config.SURVIVAL_THRESHOLD

    if capital < survival_threshold
        agent.alive = false
        agent.failure_round = agent.failure_round > 0 ? agent.failure_round : -1  # Mark as failed
    end
end

"""
Apply operating costs to agents based on market conditions.
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

        # Estimate operational cost
        estimated_cost = estimate_operational_costs(agent, market)
        operating_cost = max(0.0, estimated_cost * severity)

        if operating_cost > 0.0
            agent.resources.capital -= operating_cost
        end

        enforce_survival_threshold!(agent, config)
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
        enforce_survival_threshold!(agent, sim.config)
    end

    # Refresh alive agents
    alive_agents = [a for a in alive_agents if a.alive]
    if isempty(alive_agents)
        return
    end

    # Phase 2: Collect agent decisions
    agent_actions = Dict{String,Any}[]

    for agent in alive_agents
        # Get neighbor signals if network is enabled
        neighbor_signals = Dict{String,Any}()
        if !isnothing(sim.agent_network)
            neighbor_ids = get_neighbors(sim.agent_network, agent.id)
            if length(neighbor_ids) > 5
                neighbor_ids = rand(sim.rng, neighbor_ids, 5)
            end
            neighbors = [sim.agents[nid] for nid in neighbor_ids if sim.agents[nid].alive]
            neighbor_signals = collect_neighbor_signals(agent, neighbors)
        end

        # Get opportunities available to this agent
        agent_opportunities = get_opportunities_for_agent(sim.market, agent)
        if isempty(agent_opportunities)
            agent_opportunities = available_opportunities
        end

        # Make decision
        decision = make_decision!(
            agent,
            agent_opportunities,
            market_conditions,
            sim.market,
            round;
            neighbor_signals=neighbor_signals
        )

        decision["agent_id"] = agent.id
        push!(agent_actions, decision)
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
            end
        end

        if action_type != "invest"
            outcome = calculate_action_outcome(action, market_conditions, round)
            update_state_from_outcome!(agent, outcome)
            enforce_survival_threshold!(agent, sim.config)
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

    # Buffer uncertainty state
    if !isempty(uncertainty_state)
        flat_uncertainty = Dict{String,Any}(
            "run_id" => sim.run_id,
            "round" => round,
            "actor_ignorance_level" => get(get(uncertainty_state, "actor_ignorance", Dict()), "level", 0.0),
            "practical_indeterminism_level" => get(get(uncertainty_state, "practical_indeterminism", Dict()), "level", 0.0),
            "agentic_novelty_level" => get(get(uncertainty_state, "agentic_novelty", Dict()), "level", 0.0),
            "competitive_recursion_level" => get(get(uncertainty_state, "competitive_recursion", Dict()), "level", 0.0)
        )
        push!(sim.data_buffer.uncertainty, flat_uncertainty)
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
        "survival_rate" => n_alive / n_total,
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
