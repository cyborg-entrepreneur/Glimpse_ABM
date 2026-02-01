#!/usr/bin/env julia
"""
MIXED TIER AI PARADOX ANALYSIS
==============================

Runs the analysis with mixed AI tiers (25% each tier).
Agents are assigned a tier at initialization and keep it throughout.

Configuration: 1000 agents × 60 rounds × 50 runs
Tier distribution: 25% none, 25% basic, 25% advanced, 25% premium

Usage:
    julia --threads=auto --project=. scripts/run_mixed_tier_analysis.jl [output_dir]

Expected runtime: ~30-45 minutes
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
# GLOBAL CONFIGURATION
# ============================================================================

const N_AGENTS = 1000
const N_ROUNDS = 60   # 5 years (monthly rounds)
const N_RUNS = 50
const BASE_SEED = 20260131
const AI_TIERS = ["none", "basic", "advanced", "premium"]

# Check for output directory argument
const OUTPUT_DIR = if length(ARGS) > 0
    ARGS[1]
else
    joinpath(dirname(@__DIR__), "results",
        "mixed_tier_analysis_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
end

mkpath(OUTPUT_DIR)

const TIER_LABELS = Dict(
    "none" => "No AI",
    "basic" => "Basic AI",
    "advanced" => "Advanced AI",
    "premium" => "Premium AI"
)

# ============================================================================
# HEADER
# ============================================================================

println("="^80)
println("MIXED TIER AI PARADOX ANALYSIS")
println("="^80)
println("Configuration:")
println("  Threads:     $(Threads.nthreads())")
println("  Agents:      $N_AGENTS")
println("  Rounds:      $N_ROUNDS (5 years)")
println("  Runs:        $N_RUNS")
println("  Tier Mix:    25% each (none, basic, advanced, premium)")
println("  Output:      $OUTPUT_DIR")
println("="^80)

start_time = time()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

"""
Run a single simulation with mixed tiers (25% each).
Returns survival rates by tier.
"""
function run_mixed_simulation(; seed=42, n_rounds=N_ROUNDS)
    config = EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=n_rounds,
        RANDOM_SEED=seed,
        INITIAL_CAPITAL=5_000_000.0,
        SURVIVAL_THRESHOLD=10_000.0
    )

    # Mixed tier distribution: 25% each
    tier_dist = Dict(t => 0.25 for t in AI_TIERS)
    sim = EmergentSimulation(config=config, initial_tier_distribution=tier_dist)

    for r in 1:n_rounds
        GlimpseABM.step!(sim, r)
    end

    # Calculate survival by tier
    tier_survival = Dict{String, Tuple{Int, Int}}()  # (survived, total)
    for t in AI_TIERS
        tier_survival[t] = (0, 0)
    end

    for agent in sim.agents
        tier = get_ai_level(agent)
        survived_count, total_count = tier_survival[tier]
        tier_survival[tier] = (
            survived_count + (agent.resources.capital > config.SURVIVAL_THRESHOLD ? 1 : 0),
            total_count + 1
        )
    end

    return tier_survival
end

"""
Calculate survival rate from (survived, total) tuple.
"""
function survival_rate(counts::Tuple{Int, Int})::Float64
    survived, total = counts
    return total > 0 ? survived / total : 0.0
end

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

println("\nRunning $N_RUNS mixed-tier simulations...")

# Store results
results = Dict{String, Vector{Float64}}()
for t in AI_TIERS
    results[t] = Float64[]
end

# Thread-safe storage
results_lock = ReentrantLock()

# Run simulations in parallel
Threads.@threads for run_idx in 1:N_RUNS
    seed = BASE_SEED + run_idx
    tier_survival = run_mixed_simulation(seed=seed)

    lock(results_lock) do
        for t in AI_TIERS
            push!(results[t], survival_rate(tier_survival[t]))
        end
    end

    # Progress update (thread-safe)
    if run_idx % 10 == 0
        lock(results_lock) do
            elapsed = time() - start_time
            pct = run_idx / N_RUNS * 100
            eta = elapsed / run_idx * (N_RUNS - run_idx)
            @printf("\r  Progress: %d/%d (%.0f%%) | ETA: %.0fs    ", run_idx, N_RUNS, pct, eta)
        end
    end
end

println("\n")

# ============================================================================
# COMPUTE STATISTICS
# ============================================================================

println("--- MIXED-TIER RESULTS (25% each tier) ---")
println("-"^60)

# Create summary dataframe
rows = []
baseline_survival = mean(results["none"])

for tier in AI_TIERS
    survival_vals = results[tier]
    mean_surv = mean(survival_vals)
    std_surv = std(survival_vals)
    ci_lo = mean_surv - 1.96 * std_surv / sqrt(N_RUNS)
    ci_hi = mean_surv + 1.96 * std_surv / sqrt(N_RUNS)
    ate = tier == "none" ? NaN : (mean_surv - baseline_survival) * 100

    push!(rows, (
        Tier = TIER_LABELS[tier],
        tier_code = tier,
        Survival_Mean = mean_surv * 100,
        Survival_Std = std_surv * 100,
        CI_Lo = ci_lo * 100,
        CI_Hi = ci_hi * 100,
        ATE_pp = ate,
        N_Runs = N_RUNS
    ))

    if tier == "none"
        @printf("  %-12s: %.1f%% [%.1f, %.1f] | ATE: —\n",
                TIER_LABELS[tier], mean_surv*100, ci_lo*100, ci_hi*100)
    else
        @printf("  %-12s: %.1f%% [%.1f, %.1f] | ATE: %+.1f pp\n",
                TIER_LABELS[tier], mean_surv*100, ci_lo*100, ci_hi*100, ate)
    end
end

# Save summary
summary_df = DataFrame(rows)
CSV.write(joinpath(OUTPUT_DIR, "mixed_tier_summary.csv"), summary_df)

# ============================================================================
# DETAILED RUN DATA
# ============================================================================

# Save individual run data
run_rows = []
for run_idx in 1:N_RUNS
    for tier in AI_TIERS
        push!(run_rows, (
            run = run_idx,
            tier = tier,
            tier_label = TIER_LABELS[tier],
            survival_rate = results[tier][run_idx]
        ))
    end
end
run_df = DataFrame(run_rows)
CSV.write(joinpath(OUTPUT_DIR, "mixed_tier_runs.csv"), run_df)

# ============================================================================
# COMPARISON METRICS
# ============================================================================

println("\n--- COMPARISON METRICS ---")
println("-"^60)

# Premium vs None effect
none_mean = mean(results["none"])
premium_mean = mean(results["premium"])
ate_premium = (premium_mean - none_mean) * 100

# Statistical test (two-sample t-test approximation)
pooled_std = sqrt((var(results["none"]) + var(results["premium"])) / 2)
se = pooled_std * sqrt(2/N_RUNS)
t_stat = (premium_mean - none_mean) / se
significant = abs(t_stat) > 1.96

@printf("  Premium ATE:     %+.1f pp\n", ate_premium)
@printf("  Standard Error:  %.2f pp\n", se * 100)
@printf("  t-statistic:     %.2f\n", t_stat)
@printf("  Significant:     %s (α=0.05)\n", significant ? "YES" : "NO")

# Save comparison metrics
comparison = DataFrame(
    metric = ["premium_ate_pp", "standard_error_pp", "t_statistic", "significant",
              "none_mean", "premium_mean", "n_runs"],
    value = [ate_premium, se * 100, t_stat, significant ? 1.0 : 0.0,
             none_mean * 100, premium_mean * 100, Float64(N_RUNS)]
)
CSV.write(joinpath(OUTPUT_DIR, "mixed_tier_comparison.csv"), comparison)

# ============================================================================
# FINISH
# ============================================================================

elapsed = time() - start_time
println("\n" * "="^80)
@printf("Mixed-tier analysis complete: %.1f minutes\n", elapsed/60)
println("Output directory: $OUTPUT_DIR")
println("  mixed_tier_summary.csv")
println("  mixed_tier_runs.csv")
println("  mixed_tier_comparison.csv")
println("="^80)
