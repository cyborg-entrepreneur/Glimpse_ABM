"""
Run Julia GLIMPSE ABM simulation and output validation metrics.
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using GlimpseABM
using Statistics
using JSON3
using Random

function run_validation()
    # Fixed configuration for reproducibility - MUST match Python
    n_agents = 100
    n_rounds = 50
    seed = 42

    println("Running Julia validation simulation...")
    println("  Agents: $n_agents")
    println("  Rounds: $n_rounds")
    println("  Seed: $seed")

    config = EmergentConfig(
        N_AGENTS=n_agents,
        N_ROUNDS=n_rounds,
        RANDOM_SEED=seed,
        enable_round_logging=false
    )

    # Run simulation
    sim = EmergentSimulation(
        config=config,
        output_dir="validation_results",
        run_id="validation",
        seed=seed
    )
    run!(sim)

    # Collect validation metrics
    agents = sim.agents

    # Survival metrics
    alive_agents = filter(a -> a.alive, agents)
    survival_count = length(alive_agents)
    survival_rate = survival_count / length(agents)

    # Capital metrics
    if !isempty(alive_agents)
        capitals = [a.resources.capital for a in alive_agents]
        avg_capital = mean(capitals)
        std_capital = std(capitals)
        min_capital = minimum(capitals)
        max_capital = maximum(capitals)
    else
        avg_capital = std_capital = min_capital = max_capital = 0.0
    end

    # AI tier distribution
    ai_counts = Dict("none" => 0, "basic" => 0, "advanced" => 0, "premium" => 0)
    for a in agents
        tier = a.current_ai_level
        if haskey(ai_counts, tier)
            ai_counts[tier] += 1
        end
    end

    # Innovation metrics
    total_innovations = sum(a.innovation_count for a in agents)
    total_investments = sum(a.total_invested for a in agents)

    # Success/failure counts
    total_successes = sum(a.success_count for a in agents)
    total_failures = sum(a.failure_count for a in agents)

    results = Dict(
        "language" => "julia",
        "config" => Dict(
            "n_agents" => n_agents,
            "n_rounds" => n_rounds,
            "seed" => seed
        ),
        "survival" => Dict(
            "count" => survival_count,
            "rate" => round(survival_rate, digits=6)
        ),
        "capital" => Dict(
            "mean" => round(avg_capital, digits=2),
            "std" => round(std_capital, digits=2),
            "min" => round(min_capital, digits=2),
            "max" => round(max_capital, digits=2)
        ),
        "ai_distribution" => ai_counts,
        "activity" => Dict(
            "total_innovations" => total_innovations,
            "total_investments" => round(total_investments, digits=2),
            "total_successes" => total_successes,
            "total_failures" => total_failures
        )
    )

    # Output results
    println("\n" * "=" ^ 60)
    println("VALIDATION RESULTS (Julia)")
    println("=" ^ 60)
    println(JSON3.pretty(results))

    # Save to file
    output_path = joinpath(@__DIR__, "julia_results.json")
    open(output_path, "w") do f
        JSON3.pretty(f, results)
    end
    println("\nResults saved to: $output_path")

    return results
end

# Run validation
results = run_validation()
