#!/usr/bin/env julia
"""
DETAILED DYNAMIC AI ADOPTION ANALYSIS
=====================================

Collects round-by-round AI tier adoption rates across 50 runs
for generating graphs with proper error terms.

Output: CSV with mean and std for each tier at each round
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using GlimpseABM
using Statistics
using Printf
using Dates
using Random
using DataFrames
using CSV
using Base.Threads

# ============================================================================
# CONFIGURATION
# ============================================================================

const N_AGENTS = 1000
const N_ROUNDS = 60
const N_RUNS = 50
const BASE_SEED = 20260131
const AI_TIERS = ["none", "basic", "advanced", "premium"]

const OUTPUT_DIR = joinpath(dirname(@__DIR__), "results", "dynamic_adoption_detailed")
mkpath(OUTPUT_DIR)

# ============================================================================
# HEADER
# ============================================================================

println("="^80)
println("DETAILED DYNAMIC AI ADOPTION ANALYSIS")
println("="^80)
println("Configuration:")
println("  Threads:     $(Threads.nthreads())")
println("  Agents:      $N_AGENTS")
println("  Rounds:      $N_ROUNDS")
println("  Runs:        $N_RUNS")
println("  Output:      $OUTPUT_DIR")
println("="^80)
flush(stdout)

start_time = time()

# ============================================================================
# DATA STORAGE
# ============================================================================

# Store tier shares for each run at each round
# Dimensions: [run, round, tier]
tier_shares = zeros(N_RUNS, N_ROUNDS + 1, 4)  # +1 for initial state (round 0)

results_lock = ReentrantLock()

# ============================================================================
# RUN SIMULATIONS
# ============================================================================

println("\nRunning $N_RUNS dynamic-adoption simulations with round-by-round tracking...")
flush(stdout)

Threads.@threads for run_idx in 1:N_RUNS
    seed = BASE_SEED + run_idx

    # Create simulation with 25% initial distribution
    config = EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        INITIAL_CAPITAL=5_000_000.0,
        SURVIVAL_THRESHOLD=10_000.0
    )

    tier_dist = Dict(t => 0.25 for t in AI_TIERS)
    sim = EmergentSimulation(config=config, initial_tier_distribution=tier_dist)

    # Enable dynamic adoption by clearing fixed_ai_level
    for agent in sim.agents
        current = agent.fixed_ai_level
        agent.fixed_ai_level = nothing
        agent.current_ai_level = current
    end

    # Record initial distribution (round 0)
    for (tier_idx, tier) in enumerate(AI_TIERS)
        count = sum(1 for a in sim.agents if get_ai_level(a) == tier)
        tier_shares[run_idx, 1, tier_idx] = count / N_AGENTS
    end

    # Run simulation and record at each round
    for r in 1:N_ROUNDS
        GlimpseABM.step!(sim, r)

        # Count alive agents in each tier
        alive_agents = [a for a in sim.agents if a.alive]
        n_alive = length(alive_agents)

        if n_alive > 0
            for (tier_idx, tier) in enumerate(AI_TIERS)
                count = sum(1 for a in alive_agents if get_ai_level(a) == tier)
                tier_shares[run_idx, r + 1, tier_idx] = count / n_alive
            end
        else
            # If no agents alive, keep previous distribution
            for tier_idx in 1:4
                tier_shares[run_idx, r + 1, tier_idx] = tier_shares[run_idx, r, tier_idx]
            end
        end
    end

    # Progress update
    lock(results_lock) do
        completed = sum(tier_shares[:, end, 1] .> 0)  # Count completed runs
        if run_idx % 10 == 0
            elapsed = time() - start_time
            @printf("  Completed %d/%d runs (%.1fs elapsed)\n", run_idx, N_RUNS, elapsed)
            flush(stdout)
        end
    end
end

println("\nAll simulations complete. Computing statistics...")
flush(stdout)

# ============================================================================
# COMPUTE STATISTICS
# ============================================================================

# Calculate mean and std for each tier at each round
rounds = 0:N_ROUNDS
tier_labels = ["No AI", "Basic AI", "Advanced AI", "Premium AI"]

# Create DataFrame for round-by-round data
rows = []
for (round_idx, round) in enumerate(rounds)
    for (tier_idx, tier_label) in enumerate(tier_labels)
        values = tier_shares[:, round_idx, tier_idx]
        push!(rows, (
            round = round,
            tier = tier_label,
            mean_share = mean(values) * 100,
            std_share = std(values) * 100,
            se_share = std(values) / sqrt(N_RUNS) * 100,
            ci_lo = (mean(values) - 1.96 * std(values) / sqrt(N_RUNS)) * 100,
            ci_hi = (mean(values) + 1.96 * std(values) / sqrt(N_RUNS)) * 100,
            min_share = minimum(values) * 100,
            max_share = maximum(values) * 100
        ))
    end
end

df = DataFrame(rows)
CSV.write(joinpath(OUTPUT_DIR, "round_by_round_adoption.csv"), df)
println("  Saved: round_by_round_adoption.csv")

# Also create a wide-format summary for easy plotting
summary_rows = []
for round in rounds
    row_data = Dict{Symbol, Any}(:round => round)
    for (tier_idx, tier_label) in enumerate(tier_labels)
        tier_key = replace(lowercase(tier_label), " " => "_")
        values = tier_shares[:, round + 1, tier_idx]
        row_data[Symbol("$(tier_key)_mean")] = mean(values) * 100
        row_data[Symbol("$(tier_key)_std")] = std(values) * 100
        row_data[Symbol("$(tier_key)_se")] = std(values) / sqrt(N_RUNS) * 100
    end
    push!(summary_rows, row_data)
end

summary_df = DataFrame(summary_rows)
CSV.write(joinpath(OUTPUT_DIR, "adoption_summary_wide.csv"), summary_df)
println("  Saved: adoption_summary_wide.csv")

# ============================================================================
# PRINT KEY RESULTS
# ============================================================================

println("\n" * "="^80)
println("ROUND-BY-ROUND ADOPTION SUMMARY")
println("="^80)

# Print at key time points
key_rounds = [0, 12, 24, 36, 48, 60]
println("\nTier Distribution (%) at Key Time Points:")
println("-"^70)
@printf("%-12s", "Tier")
for r in key_rounds
    @printf("%10s", "Round $r")
end
println()
println("-"^70)

for (tier_idx, tier_label) in enumerate(tier_labels)
    @printf("%-12s", tier_label)
    for r in key_rounds
        m = mean(tier_shares[:, r + 1, tier_idx]) * 100
        se = std(tier_shares[:, r + 1, tier_idx]) / sqrt(N_RUNS) * 100
        @printf("%7.1f±%.1f", m, se)
    end
    println()
end

# ============================================================================
# FINISH
# ============================================================================

elapsed = time() - start_time
println("\n" * "="^80)
@printf("Analysis complete: %.1f minutes\n", elapsed / 60)
println("Output directory: $OUTPUT_DIR")
println("="^80)
