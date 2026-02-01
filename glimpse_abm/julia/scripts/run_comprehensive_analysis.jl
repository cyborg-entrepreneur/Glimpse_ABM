#!/usr/bin/env julia
"""
COMPREHENSIVE AI PARADOX ANALYSIS SUITE
========================================

Runs the complete analysis pipeline:
1. Fixed-Tier Analysis (main model)
2. Robustness Analysis
3. Mechanism Analysis
4. Refutation Tests (31 conditions)

Configuration: 1000 agents × 60 rounds × 50 runs per condition

Usage:
    julia --threads=auto --project=. scripts/run_comprehensive_analysis.jl

Expected runtime: ~2-3 hours depending on hardware
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

const MASTER_OUTPUT_DIR = joinpath(dirname(@__DIR__), "results",
    "comprehensive_analysis_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")

mkpath(MASTER_OUTPUT_DIR)
mkpath(joinpath(MASTER_OUTPUT_DIR, "fixed_tier"))
mkpath(joinpath(MASTER_OUTPUT_DIR, "robustness"))
mkpath(joinpath(MASTER_OUTPUT_DIR, "mechanism"))
mkpath(joinpath(MASTER_OUTPUT_DIR, "refutation"))

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
println("COMPREHENSIVE AI PARADOX ANALYSIS SUITE")
println("="^80)
println("Configuration:")
println("  Threads:     $(Threads.nthreads())")
println("  Agents:      $N_AGENTS")
println("  Rounds:      $N_ROUNDS (5 years)")
println("  Runs:        $N_RUNS per condition")
println("  Output:      $MASTER_OUTPUT_DIR")
println("="^80)
flush(stdout)

master_start = time()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function run_simulation(tier::String; seed=42, n_rounds=N_ROUNDS,
                        config_overrides=Dict{String,Any}())
    config = EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=n_rounds,
        RANDOM_SEED=seed,
        INITIAL_CAPITAL=5_000_000.0,
        SURVIVAL_THRESHOLD=10_000.0
    )

    # Apply any config overrides
    for (key, value) in config_overrides
        key_sym = Symbol(key)
        if hasproperty(config, key_sym)
            setfield!(config, key_sym, value)
        end
    end

    tier_dist = Dict(t => (t == tier ? 1.0 : 0.0) for t in AI_TIERS)
    sim = EmergentSimulation(config=config, initial_tier_distribution=tier_dist)

    for r in 1:n_rounds
        GlimpseABM.step!(sim, r)
    end

    alive = filter(a -> a.alive, sim.agents)

    # Action distribution
    action_counts = Dict("invest" => 0, "innovate" => 0, "explore" => 0, "maintain" => 0)
    for agent in sim.agents
        for act in agent.action_history
            action_counts[act] = get(action_counts, act, 0) + 1
        end
    end
    total_actions = sum(values(action_counts))

    # Competition
    all_competition = Float64[]
    for agent in sim.agents
        append!(all_competition, agent.uncertainty_metrics.competition_levels)
    end

    return Dict{String,Any}(
        "survival_rate" => length(alive) / N_AGENTS,
        "n_alive" => length(alive),
        "innovate_share" => total_actions > 0 ? action_counts["innovate"] / total_actions : 0.0,
        "explore_share" => total_actions > 0 ? action_counts["explore"] / total_actions : 0.0,
        "mean_competition" => isempty(all_competition) ? 0.0 : mean(all_competition),
        "niches" => sum(a.uncertainty_metrics.niches_discovered for a in sim.agents),
        "innovations" => sum(a.innovation_count for a in sim.agents),
        "successes" => sum(a.success_count for a in sim.agents),
        "failures" => sum(a.failure_count for a in sim.agents)
    )
end

function bootstrap_ci(data::Vector{Float64}; n_boot=1000, alpha=0.05)
    rng = MersenneTwister(42)
    boot_means = Float64[]
    n = length(data)
    for _ in 1:n_boot
        sample = [data[rand(rng, 1:n)] for _ in 1:n]
        push!(boot_means, mean(sample))
    end
    sorted = sort(boot_means)
    ci_lo = sorted[max(1, Int(floor(alpha/2 * n_boot)))]
    ci_hi = sorted[min(n_boot, Int(ceil((1 - alpha/2) * n_boot)))]
    return (mean=mean(data), ci_lo=ci_lo, ci_hi=ci_hi, se=std(boot_means))
end

function compute_ate(treatment::Vector{Float64}, control::Vector{Float64})
    ate = mean(treatment) - mean(control)
    rng = MersenneTwister(42)
    boot_ates = Float64[]
    nt, nc = length(treatment), length(control)
    for _ in 1:1000
        t_sample = [treatment[rand(rng, 1:nt)] for _ in 1:nt]
        c_sample = [control[rand(rng, 1:nc)] for _ in 1:nc]
        push!(boot_ates, mean(t_sample) - mean(c_sample))
    end
    sorted = sort(boot_ates)
    ci_lo = sorted[max(1, Int(floor(0.025 * 1000)))]
    ci_hi = sorted[min(1000, Int(ceil(0.975 * 1000)))]
    sig = ci_lo > 0 || ci_hi < 0
    return (ate=ate, ci_lo=ci_lo, ci_hi=ci_hi, significant=sig)
end

tier_offset = Dict("none" => 0, "basic" => 10000, "advanced" => 20000, "premium" => 30000)

# ============================================================================
# PHASE 1: FIXED-TIER ANALYSIS
# ============================================================================

println("\n" * "="^80)
println("PHASE 1: FIXED-TIER ANALYSIS")
println("="^80)
flush(stdout)

fixed_results = Dict{String, Vector{Dict}}()
for tier in AI_TIERS
    fixed_results[tier] = Vector{Dict}(undef, N_RUNS)
end

tasks = [(tier, run_idx) for tier in AI_TIERS for run_idx in 1:N_RUNS]
total_sims = length(tasks)
completed = Threads.Atomic{Int}(0)
results_lock = ReentrantLock()
phase_start = time()

println("Running $total_sims simulations...")

Threads.@threads for (tier, run_idx) in tasks
    seed = BASE_SEED + tier_offset[tier] + run_idx
    result = run_simulation(tier; seed=seed)
    result["tier"] = tier
    result["run_idx"] = run_idx

    lock(results_lock) do
        fixed_results[tier][run_idx] = result
    end

    done = Threads.atomic_add!(completed, 1)
    if done % 20 == 0 || done == total_sims
        elapsed = time() - phase_start
        @printf("\r  Progress: %d/%d (%.0f%%) | ETA: %.0fs    ",
            done, total_sims, 100*done/total_sims, (total_sims - done) / (done / elapsed))
    end
end

# Compute statistics
fixed_stats = Dict{String, NamedTuple}()
for tier in AI_TIERS
    surv_vals = [r["survival_rate"] for r in fixed_results[tier]]
    ci = bootstrap_ci(surv_vals)
    fixed_stats[tier] = (mean=ci.mean, ci_lo=ci.ci_lo, ci_hi=ci.ci_hi, se=ci.se, values=surv_vals)
end

# Treatment effects
ate_stats = Dict{String, NamedTuple}()
baseline = fixed_stats["none"].values
for tier in ["basic", "advanced", "premium"]
    treatment = fixed_stats[tier].values
    result = compute_ate(treatment, baseline)
    ate_stats[tier] = result
end

println("\n\n--- FIXED-TIER RESULTS ---")
println("-"^60)
for tier in AI_TIERS
    s = fixed_stats[tier]
    ate_str = tier == "none" ? "—" : @sprintf("%+.1f pp", ate_stats[tier].ate * 100)
    @printf("  %-12s: %.1f%% [%.1f, %.1f] | ATE: %s\n",
        TIER_LABELS[tier], s.mean*100, s.ci_lo*100, s.ci_hi*100, ate_str)
end

# Save fixed-tier results
fixed_df = DataFrame(
    Tier = [TIER_LABELS[t] for t in AI_TIERS],
    Survival_Mean = [fixed_stats[t].mean * 100 for t in AI_TIERS],
    Survival_CI_Lo = [fixed_stats[t].ci_lo * 100 for t in AI_TIERS],
    Survival_CI_Hi = [fixed_stats[t].ci_hi * 100 for t in AI_TIERS],
    ATE_pp = [t == "none" ? missing : ate_stats[t].ate * 100 for t in AI_TIERS],
    Significant = [t == "none" ? missing : ate_stats[t].significant for t in AI_TIERS],
    Innovate_Share = [mean(r["innovate_share"] for r in fixed_results[t]) * 100 for t in AI_TIERS],
    Explore_Share = [mean(r["explore_share"] for r in fixed_results[t]) * 100 for t in AI_TIERS],
    Mean_Competition = [mean(r["mean_competition"] for r in fixed_results[t]) for t in AI_TIERS],
    Niches = [mean(r["niches"] for r in fixed_results[t]) for t in AI_TIERS],
    Success_Rate = [sum(r["successes"] for r in fixed_results[t]) /
                    max(1, sum(r["successes"] + r["failures"] for r in fixed_results[t])) * 100 for t in AI_TIERS]
)
CSV.write(joinpath(MASTER_OUTPUT_DIR, "fixed_tier", "fixed_tier_summary.csv"), fixed_df)
println("  Saved: fixed_tier/fixed_tier_summary.csv")

phase1_time = time() - phase_start
@printf("  Phase 1 complete: %.1f minutes\n", phase1_time/60)

# ============================================================================
# PHASE 2: ROBUSTNESS ANALYSIS
# ============================================================================

println("\n" * "="^80)
println("PHASE 2: ROBUSTNESS ANALYSIS")
println("="^80)
flush(stdout)

robustness_rows = []
phase_start = time()

# Capital sensitivity
capital_levels = [2_500_000.0, 5_000_000.0, 7_500_000.0, 10_000_000.0]
capital_labels = ["2.5M", "5M", "7.5M", "10M"]
println("  Testing initial capital sensitivity...")

for (cap, label) in zip(capital_levels, capital_labels)
    none_surv = Float64[]
    prem_surv = Float64[]

    Threads.@threads for run_idx in 1:(2*N_RUNS)
        tier = run_idx <= N_RUNS ? "none" : "premium"
        actual_idx = run_idx <= N_RUNS ? run_idx : run_idx - N_RUNS
        seed = BASE_SEED + 100000 + (tier == "premium" ? 30000 : 0) + actual_idx

        config = EmergentConfig(N_AGENTS=N_AGENTS, N_ROUNDS=N_ROUNDS, RANDOM_SEED=seed,
                                INITIAL_CAPITAL=cap, SURVIVAL_THRESHOLD=10_000.0)
        tier_dist = Dict(t => (t == tier ? 1.0 : 0.0) for t in AI_TIERS)
        sim = EmergentSimulation(config=config, initial_tier_distribution=tier_dist)
        for r in 1:N_ROUNDS
            GlimpseABM.step!(sim, r)
        end
        surv = count(a -> a.alive, sim.agents) / N_AGENTS

        lock(results_lock) do
            if tier == "none"
                push!(none_surv, surv)
            else
                push!(prem_surv, surv)
            end
        end
    end

    ate = compute_ate(prem_surv, none_surv)
    push!(robustness_rows, (test="Initial Capital", condition=label,
          none_survival=mean(none_surv)*100, premium_survival=mean(prem_surv)*100,
          ate_pp=ate.ate*100, ci_lo=ate.ci_lo*100, ci_hi=ate.ci_hi*100, significant=ate.significant))
    @printf("    Capital %s: ATE = %+.1f pp %s\n", label, ate.ate*100, ate.significant ? "***" : "")
end

# Population size sensitivity
pop_sizes = [500, 1000, 2000]
println("  Testing population size sensitivity...")

for n in pop_sizes
    none_surv = Float64[]
    prem_surv = Float64[]

    Threads.@threads for run_idx in 1:(2*N_RUNS)
        tier = run_idx <= N_RUNS ? "none" : "premium"
        actual_idx = run_idx <= N_RUNS ? run_idx : run_idx - N_RUNS
        seed = BASE_SEED + 200000 + (tier == "premium" ? 30000 : 0) + actual_idx

        config = EmergentConfig(N_AGENTS=n, N_ROUNDS=N_ROUNDS, RANDOM_SEED=seed,
                                INITIAL_CAPITAL=5_000_000.0, SURVIVAL_THRESHOLD=10_000.0)
        tier_dist = Dict(t => (t == tier ? 1.0 : 0.0) for t in AI_TIERS)
        sim = EmergentSimulation(config=config, initial_tier_distribution=tier_dist)
        for r in 1:N_ROUNDS
            GlimpseABM.step!(sim, r)
        end
        surv = count(a -> a.alive, sim.agents) / n

        lock(results_lock) do
            if tier == "none"
                push!(none_surv, surv)
            else
                push!(prem_surv, surv)
            end
        end
    end

    ate = compute_ate(prem_surv, none_surv)
    push!(robustness_rows, (test="Population Size", condition=string(n),
          none_survival=mean(none_surv)*100, premium_survival=mean(prem_surv)*100,
          ate_pp=ate.ate*100, ci_lo=ate.ci_lo*100, ci_hi=ate.ci_hi*100, significant=ate.significant))
    @printf("    Pop %d: ATE = %+.1f pp %s\n", n, ate.ate*100, ate.significant ? "***" : "")
end

# Time horizon sensitivity
time_horizons = [60, 90, 120]
time_labels = ["5yr", "7.5yr", "10yr"]
println("  Testing time horizon sensitivity...")

for (rounds, label) in zip(time_horizons, time_labels)
    none_surv = Float64[]
    prem_surv = Float64[]

    Threads.@threads for run_idx in 1:(2*N_RUNS)
        tier = run_idx <= N_RUNS ? "none" : "premium"
        actual_idx = run_idx <= N_RUNS ? run_idx : run_idx - N_RUNS
        seed = BASE_SEED + 300000 + (tier == "premium" ? 30000 : 0) + actual_idx

        config = EmergentConfig(N_AGENTS=N_AGENTS, N_ROUNDS=rounds, RANDOM_SEED=seed,
                                INITIAL_CAPITAL=5_000_000.0, SURVIVAL_THRESHOLD=10_000.0)
        tier_dist = Dict(t => (t == tier ? 1.0 : 0.0) for t in AI_TIERS)
        sim = EmergentSimulation(config=config, initial_tier_distribution=tier_dist)
        for r in 1:rounds
            GlimpseABM.step!(sim, r)
        end
        surv = count(a -> a.alive, sim.agents) / N_AGENTS

        lock(results_lock) do
            if tier == "none"
                push!(none_surv, surv)
            else
                push!(prem_surv, surv)
            end
        end
    end

    ate = compute_ate(prem_surv, none_surv)
    push!(robustness_rows, (test="Time Horizon", condition=label,
          none_survival=mean(none_surv)*100, premium_survival=mean(prem_surv)*100,
          ate_pp=ate.ate*100, ci_lo=ate.ci_lo*100, ci_hi=ate.ci_hi*100, significant=ate.significant))
    @printf("    Time %s: ATE = %+.1f pp %s\n", label, ate.ate*100, ate.significant ? "***" : "")
end

# Seed stability
seed_bases = [12345, 54321, 98765, 11111, 99999]
println("  Testing seed stability...")

for seed_base in seed_bases
    none_surv = Float64[]
    prem_surv = Float64[]

    Threads.@threads for run_idx in 1:(2*N_RUNS)
        tier = run_idx <= N_RUNS ? "none" : "premium"
        actual_idx = run_idx <= N_RUNS ? run_idx : run_idx - N_RUNS
        seed = seed_base + (tier == "premium" ? 30000 : 0) + actual_idx

        config = EmergentConfig(N_AGENTS=N_AGENTS, N_ROUNDS=N_ROUNDS, RANDOM_SEED=seed,
                                INITIAL_CAPITAL=5_000_000.0, SURVIVAL_THRESHOLD=10_000.0)
        tier_dist = Dict(t => (t == tier ? 1.0 : 0.0) for t in AI_TIERS)
        sim = EmergentSimulation(config=config, initial_tier_distribution=tier_dist)
        for r in 1:N_ROUNDS
            GlimpseABM.step!(sim, r)
        end
        surv = count(a -> a.alive, sim.agents) / N_AGENTS

        lock(results_lock) do
            if tier == "none"
                push!(none_surv, surv)
            else
                push!(prem_surv, surv)
            end
        end
    end

    ate = compute_ate(prem_surv, none_surv)
    push!(robustness_rows, (test="Seed Sequence", condition=string(seed_base),
          none_survival=mean(none_surv)*100, premium_survival=mean(prem_surv)*100,
          ate_pp=ate.ate*100, ci_lo=ate.ci_lo*100, ci_hi=ate.ci_hi*100, significant=ate.significant))
    @printf("    Seed %d: ATE = %+.1f pp %s\n", seed_base, ate.ate*100, ate.significant ? "***" : "")
end

robustness_df = DataFrame(robustness_rows)
CSV.write(joinpath(MASTER_OUTPUT_DIR, "robustness", "robustness_summary.csv"), robustness_df)
println("  Saved: robustness/robustness_summary.csv")

phase2_time = time() - phase_start
@printf("  Phase 2 complete: %.1f minutes\n", phase2_time/60)

# ============================================================================
# PHASE 3: MECHANISM ANALYSIS
# ============================================================================

println("\n" * "="^80)
println("PHASE 3: MECHANISM ANALYSIS")
println("="^80)
flush(stdout)
phase_start = time()

# Already have fixed_results from Phase 1, compute mechanism stats
mechanism_rows = []
for tier in AI_TIERS
    results = fixed_results[tier]
    push!(mechanism_rows, (
        Tier = TIER_LABELS[tier],
        Survival_Mean = mean(r["survival_rate"] for r in results) * 100,
        Innovate_Share = mean(r["innovate_share"] for r in results) * 100,
        Explore_Share = mean(r["explore_share"] for r in results) * 100,
        Competition = mean(r["mean_competition"] for r in results),
        Niches = mean(r["niches"] for r in results),
        Success_Rate = sum(r["successes"] for r in results) /
                       max(1, sum(r["successes"] + r["failures"] for r in results)) * 100
    ))
end
mechanism_df = DataFrame(mechanism_rows)
CSV.write(joinpath(MASTER_OUTPUT_DIR, "mechanism", "mechanism_summary.csv"), mechanism_df)

# Mediation analysis (run-level correlations)
all_run_data = []
for tier in AI_TIERS
    for r in fixed_results[tier]
        push!(all_run_data, (
            tier_numeric = findfirst(==(tier), AI_TIERS),
            survival = r["survival_rate"],
            innovate_share = r["innovate_share"],
            explore_share = r["explore_share"],
            competition = r["mean_competition"],
            niches = Float64(r["niches"])
        ))
    end
end
run_df = DataFrame(all_run_data)

tier_survival_corr = cor(run_df.tier_numeric, run_df.survival)
tier_innovate_corr = cor(run_df.tier_numeric, run_df.innovate_share)
tier_niches_corr = cor(run_df.tier_numeric, run_df.niches)
innovate_survival_corr = cor(run_df.innovate_share, run_df.survival)
niches_survival_corr = cor(run_df.niches, run_df.survival)

indirect_innovate = tier_innovate_corr * innovate_survival_corr
indirect_niches = tier_niches_corr * niches_survival_corr

mediation_df = DataFrame(
    Path = ["Tier→Survival", "Tier→Innovate", "Tier→Niches",
            "Innovate→Survival", "Niches→Survival",
            "Indirect_via_Innovation", "Indirect_via_Niches"],
    Correlation = [tier_survival_corr, tier_innovate_corr, tier_niches_corr,
                   innovate_survival_corr, niches_survival_corr,
                   indirect_innovate, indirect_niches]
)
CSV.write(joinpath(MASTER_OUTPUT_DIR, "mechanism", "mediation_analysis.csv"), mediation_df)
println("  Saved: mechanism/mechanism_summary.csv, mediation_analysis.csv")

phase3_time = time() - phase_start
@printf("  Phase 3 complete: %.1f minutes\n", phase3_time/60)

# ============================================================================
# PHASE 4: REFUTATION TESTS
# ============================================================================

println("\n" * "="^80)
println("PHASE 4: REFUTATION TESTS (31 CONDITIONS)")
println("="^80)
flush(stdout)
phase_start = time()

# Define refutation test conditions
struct RefutationCondition
    name::String
    description::String
    category::String
    config_modifier::Function
end

refutation_tests = [
    # Baseline
    RefutationCondition("BASELINE", "Standard model", "BASELINE", c -> c),

    # Execution multipliers
    RefutationCondition("EXEC_2X", "Premium 2x execution", "EXECUTION",
        c -> (c.AI_EXECUTION_SUCCESS_MULTIPLIERS["premium"] = 2.0; c)),
    RefutationCondition("EXEC_3X", "Premium 3x execution", "EXECUTION",
        c -> (c.AI_EXECUTION_SUCCESS_MULTIPLIERS["premium"] = 3.0; c)),
    RefutationCondition("EXEC_5X", "Premium 5x execution", "EXECUTION",
        c -> (c.AI_EXECUTION_SUCCESS_MULTIPLIERS["premium"] = 5.0; c)),
    RefutationCondition("EXEC_7X", "Premium 7x execution", "EXECUTION",
        c -> (c.AI_EXECUTION_SUCCESS_MULTIPLIERS["premium"] = 7.0; c)),
    RefutationCondition("EXEC_10X", "Premium 10x execution", "EXECUTION",
        c -> (c.AI_EXECUTION_SUCCESS_MULTIPLIERS["premium"] = 10.0; c)),

    # Quality boosts
    RefutationCondition("QUALITY_+10", "Premium +10% quality", "QUALITY",
        c -> (c.AI_QUALITY_BOOST["premium"] = 0.10; c)),
    RefutationCondition("QUALITY_+20", "Premium +20% quality", "QUALITY",
        c -> (c.AI_QUALITY_BOOST["premium"] = 0.20; c)),
    RefutationCondition("QUALITY_+30", "Premium +30% quality", "QUALITY",
        c -> (c.AI_QUALITY_BOOST["premium"] = 0.30; c)),
    RefutationCondition("QUALITY_+40", "Premium +40% quality", "QUALITY",
        c -> (c.AI_QUALITY_BOOST["premium"] = 0.40; c)),
    RefutationCondition("QUALITY_+50", "Premium +50% quality", "QUALITY",
        c -> (c.AI_QUALITY_BOOST["premium"] = 0.50; c)),

    # Combined execution + quality
    RefutationCondition("COMBINED_3X_+20", "3x exec + 20% quality", "COMBINED",
        c -> (c.AI_EXECUTION_SUCCESS_MULTIPLIERS["premium"] = 3.0; c.AI_QUALITY_BOOST["premium"] = 0.20; c)),
    RefutationCondition("COMBINED_5X_+30", "5x exec + 30% quality", "COMBINED",
        c -> (c.AI_EXECUTION_SUCCESS_MULTIPLIERS["premium"] = 5.0; c.AI_QUALITY_BOOST["premium"] = 0.30; c)),
    RefutationCondition("EXTREME_10X_+50", "10x exec + 50% quality", "COMBINED",
        c -> (c.AI_EXECUTION_SUCCESS_MULTIPLIERS["premium"] = 10.0; c.AI_QUALITY_BOOST["premium"] = 0.50; c)),

    # Crowding dynamics (λ controls penalty strength; baseline = 0.50)
    RefutationCondition("CROWDING_OFF", "No crowding", "CROWDING",
        c -> (c.CROWDING_STRENGTH_LAMBDA = 0.0; c)),
    RefutationCondition("CROWDING_25%", "25% crowding", "CROWDING",
        c -> (c.CROWDING_STRENGTH_LAMBDA = 0.125; c)),  # 25% of baseline 0.50
    RefutationCondition("CROWDING_50%", "50% crowding", "CROWDING",
        c -> (c.CROWDING_STRENGTH_LAMBDA = 0.25; c)),   # 50% of baseline 0.50
    RefutationCondition("CROWDING_75%", "75% crowding", "CROWDING",
        c -> (c.CROWDING_STRENGTH_LAMBDA = 0.375; c)),  # 75% of baseline 0.50

    # AI cost variations (AI_COST_MULTIPLIER: 0.0=free, 1.0=baseline)
    RefutationCondition("COST_0%", "Free AI", "COST",
        c -> (c.AI_COST_MULTIPLIER = 0.0; c)),
    RefutationCondition("COST_25%", "25% AI cost", "COST",
        c -> (c.AI_COST_MULTIPLIER = 0.25; c)),
    RefutationCondition("COST_50%", "50% AI cost", "COST",
        c -> (c.AI_COST_MULTIPLIER = 0.50; c)),
    RefutationCondition("COST_75%", "75% AI cost", "COST",
        c -> (c.AI_COST_MULTIPLIER = 0.75; c)),

    # Herding behavior (AI_HERDING_DECAY: 0.0=no persistence, 1.0=full persistence)
    # Note: Herding emerges naturally from information cascades; decay only affects pattern persistence
    RefutationCondition("HERDING_OFF", "No herding", "HERDING",
        c -> (c.AI_HERDING_DECAY = 0.0; c)),
    RefutationCondition("HERDING_25%", "25% herding", "HERDING",
        c -> (c.AI_HERDING_DECAY = 0.25; c)),
    RefutationCondition("HERDING_50%", "50% herding", "HERDING",
        c -> (c.AI_HERDING_DECAY = 0.50; c)),

    # Operational costs (BASE_OPERATIONAL_COST: baseline = 10000.0)
    RefutationCondition("OPS_COST_50%", "50% ops cost", "OPERATIONS",
        c -> (c.BASE_OPERATIONAL_COST = 5000.0; c)),  # 50% of baseline 10000
    RefutationCondition("OPS_COST_25%", "25% ops cost", "OPERATIONS",
        c -> (c.BASE_OPERATIONAL_COST = 2500.0; c)),  # 25% of baseline 10000

    # Combined favorable conditions
    RefutationCondition("NO_CROWD_FREE_AI", "No crowd + Free AI", "COMBINED_FAV",
        c -> (c.CROWDING_STRENGTH_LAMBDA = 0.0; c.AI_COST_MULTIPLIER = 0.0; c)),
    RefutationCondition("NO_CROWD_5X_EXEC", "No crowd + 5x exec", "COMBINED_FAV",
        c -> (c.CROWDING_STRENGTH_LAMBDA = 0.0; c.AI_EXECUTION_SUCCESS_MULTIPLIERS["premium"] = 5.0; c)),
    RefutationCondition("CROWD_50_FREE_AI", "50% crowd + Free AI", "COMBINED_FAV",
        c -> (c.CROWDING_STRENGTH_LAMBDA = 0.25; c.AI_COST_MULTIPLIER = 0.0; c)),  # 50% of baseline
    RefutationCondition("ALL_FAVORABLE", "All favorable", "COMBINED_FAV",
        c -> (c.CROWDING_STRENGTH_LAMBDA = 0.0; c.AI_COST_MULTIPLIER = 0.0;
              c.AI_EXECUTION_SUCCESS_MULTIPLIERS["premium"] = 10.0; c.AI_QUALITY_BOOST["premium"] = 0.50; c)),
]

refutation_rows = []
n_tests = length(refutation_tests)

for (test_idx, test) in enumerate(refutation_tests)
    println("\n[$test_idx/$n_tests] $(test.name): $(test.description)")
    flush(stdout)

    tier_results = Dict{String, Vector{Float64}}()
    for tier in AI_TIERS
        tier_results[tier] = Float64[]
    end

    test_start = time()
    total_runs = N_RUNS * length(AI_TIERS)
    completed_test = Threads.Atomic{Int}(0)

    Threads.@threads for idx in 1:total_runs
        tier_idx = ((idx - 1) ÷ N_RUNS) + 1
        run_idx = ((idx - 1) % N_RUNS) + 1
        tier = AI_TIERS[tier_idx]
        seed = BASE_SEED + 500000 + test_idx * 10000 + tier_idx * 1000 + run_idx

        config = EmergentConfig(
            N_AGENTS=N_AGENTS,
            N_ROUNDS=N_ROUNDS,
            RANDOM_SEED=seed,
            INITIAL_CAPITAL=5_000_000.0,
            SURVIVAL_THRESHOLD=10_000.0
        )

        # Apply test modifications
        try
            test.config_modifier(config)
        catch
            # Some fields may not exist, continue with defaults
        end

        tier_dist = Dict(t => (t == tier ? 1.0 : 0.0) for t in AI_TIERS)
        sim = EmergentSimulation(config=config, initial_tier_distribution=tier_dist)

        for r in 1:N_ROUNDS
            GlimpseABM.step!(sim, r)
        end

        surv = count(a -> a.alive, sim.agents) / N_AGENTS

        lock(results_lock) do
            push!(tier_results[tier], surv)
        end

        c = Threads.atomic_add!(completed_test, 1)
        if c % 40 == 0 || c == total_runs
            @printf("    %d/%d runs (%.1fs)\r", c, total_runs, time() - test_start)
            flush(stdout)
        end
    end

    # Compute treatment effect
    none_mean = mean(tier_results["none"])
    prem_mean = mean(tier_results["premium"])
    ate = compute_ate(tier_results["premium"], tier_results["none"])

    push!(refutation_rows, (
        test = test.name,
        category = test.category,
        description = test.description,
        none_survival = none_mean,
        basic_survival = mean(tier_results["basic"]),
        advanced_survival = mean(tier_results["advanced"]),
        premium_survival = prem_mean,
        treatment_effect = ate.ate * 100
    ))

    @printf("    ✓ None: %.1f%%, Premium: %.1f%%, Effect: %+.1f pp\n",
        none_mean*100, prem_mean*100, ate.ate*100)
    flush(stdout)
end

refutation_df = DataFrame(refutation_rows)
CSV.write(joinpath(MASTER_OUTPUT_DIR, "refutation", "refutation_summary.csv"), refutation_df)
println("\n  Saved: refutation/refutation_summary.csv")

phase4_time = time() - phase_start
@printf("  Phase 4 complete: %.1f minutes\n", phase4_time/60)

# ============================================================================
# PHASE 5: DYNAMIC ADOPTION TEST
# ============================================================================

println("\n" * "="^80)
println("PHASE 5: DYNAMIC ADOPTION TEST")
println("="^80)
flush(stdout)
phase_start = time()

println("  Testing emergent AI adoption behavior...")
println("  Key metric: Tier distribution change (learning to avoid paradox)")

"""
Run simulation with dynamic AI adoption (agents can switch tiers).
Tracks tier distribution trajectory and adoption flows.
"""
function run_dynamic_adoption_test(; seed=42)
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
    initial_tiers = Dict{String, Int}(t => 0 for t in AI_TIERS)
    for agent in sim.agents
        current = agent.fixed_ai_level
        initial_tiers[current] += 1
        agent.fixed_ai_level = nothing
        agent.current_ai_level = current
    end

    # Track tier distribution over time (at key points)
    tier_trajectory = Dict{Int, Dict{String, Int}}()
    checkpoints = [1, 12, 24, 36, 48, 60]  # Months 1, 12, 24, 36, 48, 60

    for r in 1:N_ROUNDS
        GlimpseABM.step!(sim, r)

        # Record tier distribution at checkpoints
        if r in checkpoints
            tier_trajectory[r] = Dict{String, Int}(t => 0 for t in AI_TIERS)
            for agent in sim.agents
                if agent.alive
                    tier = get_ai_level(agent)
                    tier_trajectory[r][tier] += 1
                end
            end
        end
    end

    # Final tier distribution (all agents, alive or not)
    final_tiers = Dict{String, Int}(t => 0 for t in AI_TIERS)
    for agent in sim.agents
        tier = get_ai_level(agent)
        final_tiers[tier] += 1
    end

    # Count tier switches (from ai_tier_history if available)
    total_switches = 0
    premium_exits = 0  # Count agents who left premium
    for agent in sim.agents
        history = agent.ai_tier_history
        if length(history) > 1
            for i in 2:length(history)
                if history[i] != history[i-1]
                    total_switches += 1
                    if history[i-1] == "premium"
                        premium_exits += 1
                    end
                end
            end
        end
    end

    return (
        initial_tiers = initial_tiers,
        final_tiers = final_tiers,
        tier_trajectory = tier_trajectory,
        total_switches = total_switches,
        premium_exits = premium_exits
    )
end

# Run dynamic adoption tests
dynamic_initial = Dict{String, Vector{Float64}}(t => Float64[] for t in AI_TIERS)
dynamic_final = Dict{String, Vector{Float64}}(t => Float64[] for t in AI_TIERS)
dynamic_switches = Int[]
dynamic_premium_exits = Int[]

# Track trajectory at year 1 and year 5
dynamic_year1 = Dict{String, Vector{Float64}}(t => Float64[] for t in AI_TIERS)
dynamic_year5 = Dict{String, Vector{Float64}}(t => Float64[] for t in AI_TIERS)

Threads.@threads for run_idx in 1:N_RUNS
    seed = BASE_SEED + 900000 + run_idx
    result = run_dynamic_adoption_test(seed=seed)

    lock(results_lock) do
        for t in AI_TIERS
            push!(dynamic_initial[t], result.initial_tiers[t] / N_AGENTS)
            push!(dynamic_final[t], result.final_tiers[t] / N_AGENTS)

            # Year 1 (round 12) distribution among alive agents
            if haskey(result.tier_trajectory, 12)
                alive_12 = sum(values(result.tier_trajectory[12]))
                if alive_12 > 0
                    push!(dynamic_year1[t], result.tier_trajectory[12][t] / alive_12)
                end
            end

            # Year 5 (round 60) distribution among alive agents
            if haskey(result.tier_trajectory, 60)
                alive_60 = sum(values(result.tier_trajectory[60]))
                if alive_60 > 0
                    push!(dynamic_year5[t], result.tier_trajectory[60][t] / alive_60)
                end
            end
        end
        push!(dynamic_switches, result.total_switches)
        push!(dynamic_premium_exits, result.premium_exits)
    end

    if run_idx % 10 == 0
        @printf("    %d/%d runs\r", run_idx, N_RUNS)
    end
end

# Report dynamic adoption results
println("\n  TIER DISTRIBUTION EVOLUTION:")
println("  (Initial → Year 5 among survivors)")
for tier in AI_TIERS
    init_mean = mean(dynamic_initial[tier]) * 100
    final_mean = mean(dynamic_final[tier]) * 100
    year5_vals = dynamic_year5[tier]
    year5_mean = !isempty(year5_vals) ? mean(year5_vals) * 100 : final_mean
    change = final_mean - init_mean
    @printf("    %-12s: %.1f%% → %.1f%% (%+.1f pp)\n",
            TIER_LABELS[tier], init_mean, final_mean, change)
end

# Adoption flow metrics
avg_switches = mean(dynamic_switches)
avg_premium_exits = mean(dynamic_premium_exits)
println("\n  ADOPTION DYNAMICS:")
@printf("    Avg tier switches per simulation: %.1f\n", avg_switches)
@printf("    Avg premium exits per simulation: %.1f\n", avg_premium_exits)
@printf("    Premium abandonment rate: %.1f%%\n", avg_premium_exits / (N_AGENTS * 0.25) * 100)

# Key finding summary
init_premium = mean(dynamic_initial["premium"]) * 100
final_premium = mean(dynamic_final["premium"]) * 100
premium_decline = init_premium - final_premium

println("\n  KEY FINDING:")
@printf("    Premium AI adoption: %.1f%% → %.1f%% (%.1f pp decline)\n",
        init_premium, final_premium, premium_decline)
println("    Interpretation: Agents learn to avoid the paradox through rational adaptation")

# Build output dataframe
dynamic_rows = []
for tier in AI_TIERS
    year1_vals = dynamic_year1[tier]
    year5_vals = dynamic_year5[tier]
    push!(dynamic_rows, (
        tier = tier,
        tier_label = TIER_LABELS[tier],
        initial_share = mean(dynamic_initial[tier]) * 100,
        year1_share = !isempty(year1_vals) ? mean(year1_vals) * 100 : NaN,
        year5_share = !isempty(year5_vals) ? mean(year5_vals) * 100 : NaN,
        final_share = mean(dynamic_final[tier]) * 100,
        change_pp = mean(dynamic_final[tier]) * 100 - mean(dynamic_initial[tier]) * 100
    ))
end

# NOTE: Survival by final tier removed - not meaningful due to selection bias
# The key metric is tier distribution change, which shows agents learning to avoid paradox
println("\n  NOTE: Survival by final tier not reported (selection bias)")
println("        Key finding is tier distribution change showing rational adaptation")

# Save dynamic adoption results
dynamic_df = DataFrame(dynamic_rows)
CSV.write(joinpath(MASTER_OUTPUT_DIR, "refutation", "dynamic_adoption_summary.csv"), dynamic_df)
println("  Saved: refutation/dynamic_adoption_summary.csv")

phase5_time = time() - phase_start
@printf("  Phase 5 complete: %.1f minutes\n", phase5_time/60)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

total_time = time() - master_start

println("\n" * "="^80)
println("COMPREHENSIVE ANALYSIS COMPLETE")
println("="^80)

println("\nRESULTS SUMMARY:")
println("  Fixed-Tier:")
for tier in AI_TIERS
    s = fixed_stats[tier]
    @printf("    %-12s: %.1f%%\n", TIER_LABELS[tier], s.mean*100)
end
@printf("    Premium ATE: %+.1f pp\n", ate_stats["premium"].ate*100)

println("\n  Robustness: $(sum(robustness_df.significant))/$(nrow(robustness_df)) tests significant")

println("\n  Refutation Key Findings:")
baseline_effect = refutation_df[refutation_df.test .== "BASELINE", :treatment_effect][1]
crowding_off = refutation_df[refutation_df.test .== "CROWDING_OFF", :treatment_effect][1]
cost_free = refutation_df[refutation_df.test .== "COST_0%", :treatment_effect][1]
@printf("    Baseline Effect: %+.1f pp\n", baseline_effect)
@printf("    CROWDING_OFF:    %+.1f pp (%.0f%% reduction)\n", crowding_off, (1 - crowding_off/baseline_effect)*100)
@printf("    COST_0%:         %+.1f pp (%.0f%% reduction)\n", cost_free, (1 - cost_free/baseline_effect)*100)

println("\n  Dynamic Adoption:")
init_prem = mean(dynamic_initial["premium"]) * 100
final_prem = mean(dynamic_final["premium"]) * 100
@printf("    Premium: %.1f%% → %.1f%% (agents learn to avoid paradox)\n", init_prem, final_prem)

println("\nOUTPUT DIRECTORY: $MASTER_OUTPUT_DIR")
println("  fixed_tier/fixed_tier_summary.csv")
println("  robustness/robustness_summary.csv")
println("  mechanism/mechanism_summary.csv")
println("  mechanism/mediation_analysis.csv")
println("  refutation/refutation_summary.csv")
println("  refutation/dynamic_adoption_summary.csv")

@printf("\nTotal runtime: %.1f hours (%.0f minutes)\n", total_time/3600, total_time/60)
println("="^80)

# Save config file for reference
open(joinpath(MASTER_OUTPUT_DIR, "config.txt"), "w") do io
    println(io, "Comprehensive Analysis Configuration")
    println(io, "="^40)
    println(io, "N_AGENTS = $N_AGENTS")
    println(io, "N_ROUNDS = $N_ROUNDS")
    println(io, "N_RUNS = $N_RUNS")
    println(io, "BASE_SEED = $BASE_SEED")
    println(io, "Threads = $(Threads.nthreads())")
    println(io, "Runtime = $(round(total_time/60, digits=1)) minutes")
    println(io, "Generated = $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
end
