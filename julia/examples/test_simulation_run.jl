"""
Test a real simulation run using the GlimpseABM.jl package.
"""

using Pkg
Pkg.activate(".")

using GlimpseABM
using Random
using Statistics
using Printf
using Dates

println("=" ^ 70)
println("GLIMPSE ABM - REAL SIMULATION TEST")
println("=" ^ 70)
println()

# Configuration
n_agents = 500
n_rounds = 100
seed = 42

println("Configuration:")
println("  Agents: $n_agents")
println("  Rounds: $n_rounds")
println("  Seed: $seed")
println()

# Create configuration
config = EmergentConfig(
    N_AGENTS=n_agents,
    N_ROUNDS=n_rounds,
    RANDOM_SEED=seed,
    enable_round_logging=true,
    round_log_interval=20
)

println("Creating simulation...")
start_time = time()

# Create simulation using the high-level API
sim = EmergentSimulation(
    config=config,
    output_dir="test_results",
    run_id="benchmark_test",
    seed=seed
)

println("  ✓ Simulation created")

# Count initial AI tier distribution
ai_counts = Dict("none" => 0, "basic" => 0, "advanced" => 0, "premium" => 0)
for agent in sim.agents
    tier = agent.current_ai_level
    if haskey(ai_counts, tier)
        ai_counts[tier] += 1
    end
end

println("\nInitial AI tier distribution:")
for (tier, count) in sort(collect(ai_counts))
    pct = round(100 * count / n_agents, digits=1)
    println("  $tier: $count ($pct%)")
end

println("\n" * "-" ^ 70)
println("Running simulation...")
println("-" ^ 70)

# Run the simulation
run!(sim)

elapsed = time() - start_time

println("-" ^ 70)
@printf("Simulation completed in %.3f seconds\n", elapsed)
println("-" ^ 70)

# Final statistics
println("\n" * "=" ^ 70)
println("FINAL RESULTS")
println("=" ^ 70)

# Survival by AI tier
println("\nSurvival by AI Tier:")
println("-" ^ 40)

tier_stats = Dict{String, Dict{String, Any}}()
for tier in ["none", "basic", "advanced", "premium"]
    tier_agents = filter(a -> a.current_ai_level == tier, sim.agents)
    survivors = count(a -> a.alive, tier_agents)
    total = length(tier_agents)
    survival_rate = total > 0 ? 100 * survivors / total : 0.0

    alive_tier_agents = filter(a -> a.alive, tier_agents)
    avg_capital = !isempty(alive_tier_agents) ? mean(a.resources.capital for a in alive_tier_agents) : 0.0
    total_innovations = !isempty(tier_agents) ? sum(a.innovation_count for a in tier_agents) : 0

    tier_stats[tier] = Dict(
        "total" => total,
        "survivors" => survivors,
        "survival_rate" => survival_rate,
        "avg_capital" => avg_capital,
        "innovations" => total_innovations
    )

    if total > 0
        @printf("  %-10s: %3d/%3d survived (%.1f%%), avg capital: \$%.0f, innovations: %d\n",
                titlecase(tier), survivors, total, survival_rate, avg_capital, total_innovations)
    end
end

# Overall statistics
println("\nOverall Statistics:")
println("-" ^ 40)

total_survivors = count(a -> a.alive, sim.agents)
overall_survival = 100 * total_survivors / n_agents
alive_agents = filter(a -> a.alive, sim.agents)
avg_final_capital = !isempty(alive_agents) ? mean(a.resources.capital for a in alive_agents) : 0.0
total_innovations = sum(a.innovation_count for a in sim.agents)

@printf("  Total survivors: %d/%d (%.1f%%)\n", total_survivors, n_agents, overall_survival)
@printf("  Average final capital: \$%.0f\n", avg_final_capital)
@printf("  Total innovations: %d\n", total_innovations)
@printf("  Throughput: %.0f agent-rounds/second\n", (n_agents * n_rounds) / elapsed)

# AI tier effect analysis
println("\nAI Tier Effect Analysis:")
println("-" ^ 40)

baseline_rate = get(tier_stats, "none", Dict())
if haskey(baseline_rate, "survival_rate")
    baseline = baseline_rate["survival_rate"]
    for tier in ["basic", "advanced", "premium"]
        stats = get(tier_stats, tier, Dict())
        if haskey(stats, "survival_rate") && stats["total"] > 0
            diff = stats["survival_rate"] - baseline
            sign = diff >= 0 ? "+" : ""
            @printf("  %s vs none: %s%.1f percentage points\n", titlecase(tier), sign, diff)
        end
    end
end

# Performance summary
println("\n" * "=" ^ 70)
println("PERFORMANCE SUMMARY")
println("=" ^ 70)
@printf("  Execution time: %.3f seconds\n", elapsed)
@printf("  Agent-rounds: %d\n", n_agents * n_rounds)
@printf("  Throughput: %.0f agent-rounds/second\n", (n_agents * n_rounds) / elapsed)

# Compare to Python estimate
python_estimate = (n_agents * n_rounds) / 70000  # ~70k agent-rounds/sec from benchmark
@printf("\n  Estimated Python time: %.1f seconds\n", python_estimate)
@printf("  Julia speedup: %.1fx faster\n", python_estimate / elapsed)

println("\n✓ Simulation test completed successfully!")
