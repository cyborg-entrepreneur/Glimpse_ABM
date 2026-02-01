#!/usr/bin/env julia
"""
Fixed-Tier Causal Model Run
1000 agents, 60 rounds, 50 runs per tier
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GlimpseABM
using Statistics
using Printf
using Random

const N_AGENTS = 1000
const N_ROUNDS = 60
const N_RUNS = 50
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const BASE_SEED = 20260131

println("="^70)
println("  FIXED-TIER CAUSAL MODEL (Cleaned & Calibrated)")
println("="^70)
println("  Parameters: $N_AGENTS agents, $N_ROUNDS rounds, $N_RUNS runs per tier")
println("  Crowding Model: Capacity-Convexity (K=1.5, γ=2.0, λ=0.50)")
println("  Operational Cost: 10,000 (calibrated for ~55% 5-yr survival)")
println("="^70)

results = Dict{String, Vector{Float64}}()
for tier in AI_TIERS
    results[tier] = Float64[]
end

total_runs = N_RUNS * length(AI_TIERS)
completed = Threads.Atomic{Int}(0)
all_results = Vector{Tuple{String, Float64}}(undef, total_runs)
start_time = time()

Threads.@threads for idx in 1:total_runs
    tier_idx = ((idx - 1) ÷ N_RUNS) + 1
    run_idx = ((idx - 1) % N_RUNS) + 1
    tier = AI_TIERS[tier_idx]
    seed = BASE_SEED + tier_idx * 1000 + run_idx

    config = EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        INITIAL_CAPITAL=5_000_000.0,
        SURVIVAL_THRESHOLD=10_000.0
    )

    tier_dist = Dict(t => (t == tier ? 1.0 : 0.0) for t in AI_TIERS)
    sim = GlimpseABM.EmergentSimulation(config=config, initial_tier_distribution=tier_dist, seed=seed)

    for round in 1:N_ROUNDS
        GlimpseABM.step!(sim, round)
    end

    alive = count(a -> a.alive, sim.agents)
    all_results[idx] = (tier, alive / N_AGENTS)

    c = Threads.atomic_add!(completed, 1)
    if c % 50 == 0 || c == total_runs
        elapsed = time() - start_time
        @printf("  Progress: %d/%d runs (%.1fs)\n", c, total_runs, elapsed)
    end
end

# Collect results
for (tier, rate) in all_results
    push!(results[tier], rate)
end

# Print results
println("\n" * "="^70)
println("  RESULTS")
println("="^70)
@printf("\n  %-12s %12s %12s %12s\n", "Tier", "Survival", "Std Dev", "N")
println("-"^50)
for tier in AI_TIERS
    rates = results[tier]
    @printf("  %-12s %11.1f%% %11.2f%% %12d\n",
            tier, mean(rates)*100, std(rates)*100, length(rates))
end

# Treatment effects
println("\n  TREATMENT EFFECTS (vs None baseline)")
println("-"^50)
baseline = mean(results["none"])
for tier in AI_TIERS[2:end]
    te = (mean(results[tier]) - baseline) * 100
    @printf("  %-12s %+10.2f pp\n", tier, te)
end

println("\n" * "="^70)
elapsed = time() - start_time
@printf("  Total time: %.1f seconds\n", elapsed)
println("="^70)
