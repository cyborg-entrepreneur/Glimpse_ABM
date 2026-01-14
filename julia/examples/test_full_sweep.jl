"""
Test a full fixed-tier sweep using the GlimpseABM.jl package.
Runs simulations with each AI tier to measure causal effects.
"""

using Pkg
Pkg.activate(".")

using GlimpseABM
using Random
using Statistics
using Printf
using Dates

println("=" ^ 70)
println("GLIMPSE ABM - FIXED-TIER SWEEP TEST")
println("=" ^ 70)
println()

# Configuration
n_agents = 1000
n_rounds = 200
seed = 42
ai_tiers = ["none", "basic", "advanced", "premium"]

println("Configuration:")
println("  Agents per run: $n_agents")
println("  Rounds per run: $n_rounds")
println("  AI tiers: $(join(ai_tiers, ", "))")
println("  Total agent-rounds: $(n_agents * n_rounds * length(ai_tiers))")
println()

# Results storage
results = Dict{String, Dict{String, Any}}()

total_start = time()

for (i, tier) in enumerate(ai_tiers)
    println("-" ^ 70)
    println("[$i/$(length(ai_tiers))] Running simulation with AI tier: $(uppercase(tier))")
    println("-" ^ 70)

    # Create configuration
    config = EmergentConfig(
        N_AGENTS=n_agents,
        N_ROUNDS=n_rounds,
        RANDOM_SEED=seed + i,  # Different seed per tier for independence
        enable_round_logging=false
    )

    # Create simulation
    tier_start = time()
    sim = EmergentSimulation(
        config=config,
        output_dir="test_sweep_results",
        run_id="sweep_$(tier)",
        seed=seed + i
    )

    # Set fixed AI level for all agents
    for agent in sim.agents
        agent.fixed_ai_level = tier
        agent.current_ai_level = tier
    end

    # Run simulation
    run!(sim)
    tier_elapsed = time() - tier_start

    # Collect statistics
    survivors = count(a -> a.alive, sim.agents)
    survival_rate = 100 * survivors / n_agents
    alive_agents = filter(a -> a.alive, sim.agents)
    avg_capital = !isempty(alive_agents) ? mean(a.resources.capital for a in alive_agents) : 0.0
    total_innovations = sum(a.innovation_count for a in sim.agents)

    results[tier] = Dict(
        "survivors" => survivors,
        "survival_rate" => survival_rate,
        "avg_capital" => avg_capital,
        "innovations" => total_innovations,
        "elapsed" => tier_elapsed
    )

    @printf("  Survivors: %d/%d (%.1f%%)\n", survivors, n_agents, survival_rate)
    @printf("  Avg capital: \$%.0f\n", avg_capital)
    @printf("  Innovations: %d\n", total_innovations)
    @printf("  Time: %.2f seconds (%.0f agent-rounds/sec)\n",
            tier_elapsed, (n_agents * n_rounds) / tier_elapsed)
    println()
end

total_elapsed = time() - total_start

# Summary
println("=" ^ 70)
println("SWEEP RESULTS SUMMARY")
println("=" ^ 70)
println()

println("Survival Rates by AI Tier:")
println("-" ^ 50)
@printf("%-12s %12s %15s %12s\n", "AI Tier", "Survival %", "Avg Capital", "Innovations")
println("-" ^ 50)

baseline_survival = results["none"]["survival_rate"]

for tier in ai_tiers
    r = results[tier]
    diff = r["survival_rate"] - baseline_survival
    diff_str = tier == "none" ? "(baseline)" : @sprintf("%+.1f pp", diff)
    @printf("%-12s %11.1f%% %14.0f %12d  %s\n",
            titlecase(tier), r["survival_rate"], r["avg_capital"], r["innovations"], diff_str)
end

println()
println("Causal Effect Estimates (vs no AI):")
println("-" ^ 50)

for tier in ["basic", "advanced", "premium"]
    r = results[tier]
    ate = r["survival_rate"] - baseline_survival

    # Simple effect size (diff / baseline variability)
    @printf("  %s: ATE = %+.1f percentage points\n", titlecase(tier), ate)
end

println()
println("=" ^ 70)
println("PERFORMANCE SUMMARY")
println("=" ^ 70)
total_agent_rounds = n_agents * n_rounds * length(ai_tiers)
@printf("  Total execution time: %.2f seconds\n", total_elapsed)
@printf("  Total agent-rounds: %d\n", total_agent_rounds)
@printf("  Overall throughput: %.0f agent-rounds/second\n", total_agent_rounds / total_elapsed)
println()

# Comparison with Python
python_time_estimate = total_agent_rounds / 70000  # ~70k from our benchmark
@printf("  Estimated Python time: %.1f seconds\n", python_time_estimate)
@printf("  Julia speedup: %.1fx faster\n", python_time_estimate / total_elapsed)

println()
println("✓ Fixed-tier sweep completed successfully!")
