#!/usr/bin/env julia
"""
DYNAMIC AI ADOPTION ANALYSIS
============================

Runs the analysis with dynamic AI tier adoption.
Agents start with 25% distribution but can switch tiers based on
their choose_ai_level logic.

Configuration: 1000 agents × 60 rounds × 50 runs
Initial tier distribution: 25% each (but agents can switch)

Usage:
    julia --threads=auto --project=. scripts/run_dynamic_adoption.jl
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
const N_ROUNDS = 60
const N_RUNS = 50
const BASE_SEED = 20260131
const AI_TIERS = ["none", "basic", "advanced", "premium"]

const OUTPUT_DIR = joinpath(dirname(@__DIR__), "results",
    "dynamic_adoption_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")

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
println("DYNAMIC AI ADOPTION ANALYSIS")
println("="^80)
println("Configuration:")
println("  Threads:     $(Threads.nthreads())")
println("  Agents:      $N_AGENTS")
println("  Rounds:      $N_ROUNDS (5 years)")
println("  Runs:        $N_RUNS")
println("  Initial:     25% each tier (dynamic switching enabled)")
println("  Output:      $OUTPUT_DIR")
println("="^80)

start_time = time()

# ============================================================================
# RUN SIMULATION WITH DYNAMIC ADOPTION
# ============================================================================

"""
Run a simulation where agents can dynamically switch AI tiers.
Uses the existing EmergentSimulation but modifies agents to enable switching.
"""
function run_dynamic_simulation(; seed=42)
    config = EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        INITIAL_CAPITAL=5_000_000.0,
        SURVIVAL_THRESHOLD=10_000.0
    )

    # Create simulation with 25% initial distribution
    tier_dist = Dict(t => 0.25 for t in AI_TIERS)
    sim = EmergentSimulation(config=config, initial_tier_distribution=tier_dist)

    # Record initial tier distribution
    initial_tiers = Dict{String, Int}(t => 0 for t in AI_TIERS)
    for agent in sim.agents
        tier = get_ai_level(agent)
        initial_tiers[tier] += 1
    end

    # ENABLE DYNAMIC ADOPTION: Clear fixed_ai_level but keep current_ai_level
    for agent in sim.agents
        current = agent.fixed_ai_level  # Save current tier
        agent.fixed_ai_level = nothing  # Allow switching
        agent.current_ai_level = current  # Keep starting tier
    end

    # Run simulation
    for r in 1:N_ROUNDS
        GlimpseABM.step!(sim, r)
    end

    # Record final tier distribution and survival
    final_tiers = Dict{String, Int}(t => 0 for t in AI_TIERS)
    tier_survival = Dict{String, Tuple{Int, Int}}(t => (0, 0) for t in AI_TIERS)

    for agent in sim.agents
        tier = get_ai_level(agent)
        final_tiers[tier] += 1

        survived_count, total_count = tier_survival[tier]
        survived = agent.resources.capital > config.SURVIVAL_THRESHOLD
        tier_survival[tier] = (survived_count + (survived ? 1 : 0), total_count + 1)
    end

    return (
        tier_survival = tier_survival,
        initial_tiers = initial_tiers,
        final_tiers = final_tiers
    )
end

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

println("\nRunning $N_RUNS dynamic-adoption simulations...")

# Storage
results = Dict{String, Vector{Float64}}(t => Float64[] for t in AI_TIERS)
initial_dist = Dict{String, Vector{Float64}}(t => Float64[] for t in AI_TIERS)
final_dist = Dict{String, Vector{Float64}}(t => Float64[] for t in AI_TIERS)

results_lock = ReentrantLock()

# Run simulations
Threads.@threads for run_idx in 1:N_RUNS
    seed = BASE_SEED + run_idx
    result = run_dynamic_simulation(seed=seed)

    lock(results_lock) do
        for t in AI_TIERS
            survived, total = result.tier_survival[t]
            rate = total > 0 ? survived / total : NaN
            push!(results[t], rate)
            push!(initial_dist[t], result.initial_tiers[t] / N_AGENTS)
            push!(final_dist[t], result.final_tiers[t] / N_AGENTS)
        end
    end

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

println("--- DYNAMIC ADOPTION RESULTS ---")
println("-"^60)

# Tier distribution changes
println("\nTIER DISTRIBUTION (Initial → Final):")
for tier in AI_TIERS
    init_mean = mean(initial_dist[tier]) * 100
    final_mean = mean(final_dist[tier]) * 100
    change = final_mean - init_mean
    @printf("  %-12s: %.1f%% → %.1f%% (%+.1f pp)\n",
            TIER_LABELS[tier], init_mean, final_mean, change)
end

# Survival by final tier
println("\nSURVIVAL BY FINAL TIER:")
rows = []
valid_none = filter(!isnan, results["none"])
baseline_survival = !isempty(valid_none) ? mean(valid_none) : 0.5

for tier in AI_TIERS
    valid_vals = filter(!isnan, results[tier])
    if isempty(valid_vals)
        @printf("  %-12s: No agents at this tier\n", TIER_LABELS[tier])
        continue
    end

    mean_surv = mean(valid_vals)
    std_surv = std(valid_vals)
    n_valid = length(valid_vals)
    ci_lo = mean_surv - 1.96 * std_surv / sqrt(n_valid)
    ci_hi = mean_surv + 1.96 * std_surv / sqrt(n_valid)
    ate = tier == "none" ? NaN : (mean_surv - baseline_survival) * 100

    push!(rows, (
        Tier = TIER_LABELS[tier],
        tier_code = tier,
        Survival_Mean = mean_surv * 100,
        Survival_Std = std_surv * 100,
        CI_Lo = ci_lo * 100,
        CI_Hi = ci_hi * 100,
        ATE_pp = ate,
        Initial_Share = mean(initial_dist[tier]) * 100,
        Final_Share = mean(final_dist[tier]) * 100,
        N_Runs = n_valid
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
if !isempty(rows)
    summary_df = DataFrame(rows)
    CSV.write(joinpath(OUTPUT_DIR, "dynamic_adoption_summary.csv"), summary_df)
end

# ============================================================================
# COMPARISON METRICS
# ============================================================================

println("\n--- COMPARISON METRICS ---")
println("-"^60)

none_vals = filter(!isnan, results["none"])
prem_vals = filter(!isnan, results["premium"])

if !isempty(none_vals) && !isempty(prem_vals)
    none_mean = mean(none_vals)
    premium_mean = mean(prem_vals)
    ate_premium = (premium_mean - none_mean) * 100

    pooled_std = sqrt((var(none_vals) + var(prem_vals)) / 2)
    se = pooled_std * sqrt(1/length(none_vals) + 1/length(prem_vals))
    t_stat = (premium_mean - none_mean) / se
    significant = abs(t_stat) > 1.96

    @printf("  Premium ATE:     %+.1f pp\n", ate_premium)
    @printf("  Standard Error:  %.2f pp\n", se * 100)
    @printf("  t-statistic:     %.2f\n", t_stat)
    @printf("  Significant:     %s (α=0.05)\n", significant ? "YES" : "NO")

    println("\n--- ADOPTION DYNAMICS ---")
    for tier in AI_TIERS
        init_mean = mean(initial_dist[tier]) * 100
        final_mean = mean(final_dist[tier]) * 100
        if abs(final_mean - init_mean) > 1.0
            direction = final_mean > init_mean ? "↑" : "↓"
            @printf("  %s adoption %s %.1f pp\n", TIER_LABELS[tier], direction, abs(final_mean - init_mean))
        end
    end

    comparison = DataFrame(
        metric = ["premium_ate_pp", "standard_error_pp", "t_statistic", "significant",
                  "none_mean", "premium_mean", "n_runs"],
        value = [ate_premium, se * 100, t_stat, significant ? 1.0 : 0.0,
                 none_mean * 100, premium_mean * 100, Float64(N_RUNS)]
    )
    CSV.write(joinpath(OUTPUT_DIR, "dynamic_comparison.csv"), comparison)
end

# ============================================================================
# FINISH
# ============================================================================

elapsed = time() - start_time
println("\n" * "="^80)
@printf("Dynamic adoption analysis complete: %.1f minutes\n", elapsed/60)
println("Output directory: $OUTPUT_DIR")
println("="^80)
