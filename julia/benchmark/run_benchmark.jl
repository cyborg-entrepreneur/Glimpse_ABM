"""
Self-contained benchmark for Julia ABM performance.

Tests core computational patterns used in GLIMPSE ABM.
"""

using Random
using Statistics
using Printf

# Core simulation logic (mirrors key patterns from GlimpseABM)
module BenchmarkSim

using Random
using Statistics

export run_simulation, AgentState

mutable struct AgentState
    id::Int
    capital::Float64
    alive::Bool
    ai_level::Int  # 0=none, 1=basic, 2=advanced, 3=premium
    risk_tolerance::Float64
    adaptability::Float64
    sector_experience::Vector{Float64}
    beliefs::Dict{String,Float64}
    portfolio_value::Float64
    innovation_count::Int
end

function AgentState(id::Int, rng::AbstractRNG)
    AgentState(
        id,
        500_000.0,
        true,
        rand(rng, 0:3),
        rand(rng),
        rand(rng),
        rand(rng, 5),
        Dict("market" => 0.5, "tech" => 0.5, "demand" => 0.5),
        0.0,
        0
    )
end

struct Opportunity
    id::Int
    expected_return::Float64
    failure_prob::Float64
    capital_required::Float64
    sector::Int
    maturity::Int
end

function generate_opportunities(rng::AbstractRNG, n::Int=10)
    [Opportunity(
        i,
        randn(rng) * 0.15 + 0.05,
        rand(rng) * 0.3 + 0.1,
        rand(rng) * 100_000 + 50_000,
        rand(rng, 1:5),
        rand(rng, 3:12)
    ) for i in 1:n]
end

function evaluate_opportunity(agent::AgentState, opp::Opportunity, rng::AbstractRNG)::Float64
    if !agent.alive
        return -Inf
    end

    # Base expected value
    ev = opp.expected_return * (1 - opp.failure_prob)

    # AI accuracy bonus
    ai_bonus = agent.ai_level * 0.015

    # Risk adjustment
    risk_adj = agent.risk_tolerance * (1 - opp.failure_prob)

    # Sector experience
    sector_exp = agent.sector_experience[opp.sector]

    # Belief adjustment
    belief_adj = get(agent.beliefs, "market", 0.5) * 0.1

    # Add noise
    noise = randn(rng) * 0.02

    return ev + ai_bonus + risk_adj * 0.1 + sector_exp * 0.05 + belief_adj + noise
end

function make_decision!(agent::AgentState, opportunities::Vector{Opportunity}, rng::AbstractRNG)
    if !agent.alive
        return
    end

    # Evaluate all opportunities
    scores = [evaluate_opportunity(agent, opp, rng) for opp in opportunities]

    # Find best opportunity
    best_idx = argmax(scores)
    best_opp = opportunities[best_idx]

    # Check if we can afford it
    if best_opp.capital_required > agent.capital * 0.3
        return  # Skip investment
    end

    # Simulate investment outcome
    if rand(rng) < best_opp.failure_prob
        # Failed
        agent.capital -= best_opp.capital_required * 0.5
    else
        # Success
        agent.capital += best_opp.capital_required * best_opp.expected_return
        agent.innovation_count += 1
    end
end

function update_agent!(agent::AgentState, rng::AbstractRNG)
    if !agent.alive
        return
    end

    # Operational costs
    agent.capital -= 80_000 / 12

    # Market returns on remaining capital
    market_return = randn(rng) * 0.02
    agent.capital *= (1 + market_return)

    # Update beliefs (Bayesian-like update)
    for key in keys(agent.beliefs)
        agent.beliefs[key] = clamp(agent.beliefs[key] + randn(rng) * 0.01, 0.0, 1.0)
    end

    # Check survival
    if agent.capital < 200_000
        agent.alive = false
    end
end

function run_simulation(n_agents::Int, n_rounds::Int; seed::Int=42)
    rng = MersenneTwister(seed)

    # Initialize agents
    agents = [AgentState(i, rng) for i in 1:n_agents]

    # Run simulation
    for round in 1:n_rounds
        # Generate opportunities
        opportunities = generate_opportunities(rng)

        # Agent decision loop
        for agent in agents
            make_decision!(agent, opportunities, rng)
            update_agent!(agent, rng)
        end
    end

    # Return statistics
    survivors = count(a -> a.alive, agents)
    mean_capital = mean(a.capital for a in agents if a.alive)
    total_innovations = sum(a.innovation_count for a in agents)

    return (survivors=survivors, mean_capital=mean_capital, innovations=total_innovations)
end

end # module BenchmarkSim

using .BenchmarkSim

function run_benchmark(n_agents::Int, n_rounds::Int; n_trials::Int=5, warmup::Int=2)
    # Warmup (JIT compilation)
    for _ in 1:warmup
        run_simulation(n_agents, n_rounds)
    end

    # Timed runs
    times = Float64[]
    for trial in 1:n_trials
        t = @elapsed run_simulation(n_agents, n_rounds; seed=42+trial)
        push!(times, t)
    end

    return times
end

function main()
    println("=" ^ 70)
    println("GLIMPSE ABM - JULIA BENCHMARK")
    println("=" ^ 70)
    println()
    println("Testing core simulation patterns (agent decisions, market dynamics)")
    println()

    # Benchmark configurations
    configs = [
        (100, 50, "Small"),
        (500, 100, "Medium"),
        (1000, 200, "Standard"),
        (2000, 200, "Large"),
    ]

    results = []

    for (n_agents, n_rounds, label) in configs
        agent_rounds = n_agents * n_rounds

        print("[$label] $n_agents agents × $n_rounds rounds...")
        times = run_benchmark(n_agents, n_rounds; n_trials=5, warmup=2)

        mean_time = mean(times)
        std_time = std(times)
        throughput = agent_rounds / mean_time

        push!(results, (label=label, agents=n_agents, rounds=n_rounds,
                       mean=mean_time, std=std_time, throughput=throughput))

        println(" done")
        @printf("  Time: %.4f s (±%.4f s)\n", mean_time, std_time)
        println("  Throughput: $(Int(round(throughput))) agent-rounds/sec")
        println()
    end

    # Summary table
    println("=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println()
    @printf("%-10s %8s %8s %12s %18s\n", "Config", "Agents", "Rounds", "Time (s)", "Throughput")
    println("-" ^ 60)
    for r in results
        @printf("%-10s %8d %8d %12.4f %15d /s\n",
                r.label, r.agents, r.rounds, r.mean, Int(round(r.throughput)))
    end

    return results
end

# Run benchmark
results = main()
