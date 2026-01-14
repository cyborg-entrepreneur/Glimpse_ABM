"""
Benchmark comparison between Julia and Python implementations.

Measures execution time for equivalent simulation workloads.
"""

using Printf
using Statistics
using Random

# Add parent directory to load path
push!(LOAD_PATH, dirname(@__DIR__))

using GlimpseABM

"""
Run a single simulation benchmark.
"""
function benchmark_simulation(n_agents::Int, n_rounds::Int; seed::Int=42)
    # Create configuration
    config = EmergentConfig(
        N_AGENTS=n_agents,
        N_ROUNDS=n_rounds,
        RANDOM_SEED=seed
    )

    # Create market
    rng = MersenneTwister(seed)
    market = MarketEnvironment(config, rng)

    # Create agents
    agents = [EmergentAgent(i, config, rng) for i in 1:n_agents]

    # Create uncertainty environment
    uncertainty = KnightianUncertaintyEnvironment(config)

    # Simulate rounds
    for round in 1:n_rounds
        # Generate opportunities
        opportunities = generate_opportunities!(market, round)

        # Agent decision loop
        for agent in agents
            if agent.alive
                # Evaluate opportunities
                for opp in opportunities
                    evaluate_opportunity(agent, opp, market, uncertainty, config)
                end

                # Make decision
                make_decision!(agent, opportunities, market, uncertainty, config, round)

                # Update capital
                update_capital!(agent, config)
            end
        end

        # Step market
        step!(market)
    end

    # Count survivors
    survivors = count(a -> a.alive, agents)
    return survivors
end

"""
Run multiple trials and compute statistics.
"""
function run_benchmark(n_agents::Int, n_rounds::Int; n_trials::Int=5, warmup::Int=1)
    times = Float64[]

    # Warmup runs (JIT compilation)
    for _ in 1:warmup
        benchmark_simulation(n_agents, n_rounds)
    end

    # Timed runs
    for trial in 1:n_trials
        t = @elapsed benchmark_simulation(n_agents, n_rounds; seed=42+trial)
        push!(times, t)
    end

    return times
end

"""
Main benchmark function.
"""
function main()
    println("=" ^ 70)
    println("GLIMPSE ABM JULIA BENCHMARK")
    println("=" ^ 70)
    println()

    # Benchmark configurations
    configs = [
        (100, 50),    # Small
        (500, 100),   # Medium
        (1000, 200),  # Standard
    ]

    results = Dict{Tuple{Int,Int}, Dict{String,Float64}}()

    for (n_agents, n_rounds) in configs
        println("Benchmarking: $n_agents agents × $n_rounds rounds")

        times = run_benchmark(n_agents, n_rounds; n_trials=5, warmup=2)

        results[(n_agents, n_rounds)] = Dict(
            "mean" => mean(times),
            "std" => std(times),
            "min" => minimum(times),
            "max" => maximum(times)
        )

        @printf("  Mean: %.3f s (±%.3f s)\n", mean(times), std(times))
        @printf("  Range: [%.3f, %.3f] s\n", minimum(times), maximum(times))

        # Throughput
        agent_rounds = n_agents * n_rounds
        throughput = agent_rounds / mean(times)
        @printf("  Throughput: %.0f agent-rounds/second\n", throughput)
        println()
    end

    return results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
