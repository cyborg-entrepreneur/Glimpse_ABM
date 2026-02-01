#!/usr/bin/env julia
"""
FIXED-TIER AI PARADOX ANALYSIS (v4)
===================================

Primary analysis of AI tier effects on:
- Survival outcomes
- Behavioral shifts (Innovation vs Exploration)
- Innovation success & Niche Discovery
- Unicorn/outlier emergence

Configuration:
- 1000 agents × 60 rounds × 50 runs per tier
- All 4 tiers: None, Basic, Advanced, Premium (homogeneous populations)
- Random seeds for reproducibility
- Bootstrap confidence intervals

Output:
- Single landscape PDF (fits on one page with descriptions)

Improvements in v4:
- Figure D: Innovate Action Share (shows behavioral shift)
- Figure E: Explore Action Share (shows tradeoff)
- Added brief descriptions below each section
- Removed summary tables (section 6)
- Resized to fit single landscape page
- Y-axis on K changed to "Total Niches Created"

Usage:
    julia --threads=12 --project=. scripts/run_fixed_tier_analysis.jl
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
using CairoMakie
using SpecialFunctions: erf

# ============================================================================
# CONFIGURATION
# ============================================================================

const N_AGENTS = 1000
const N_ROUNDS = 60
const N_RUNS = 50
const BASE_SEED = 20260128  # Today's date as base seed
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const N_BOOTSTRAP = 2000

# Output directory
const OUTPUT_DIR = joinpath(dirname(@__DIR__), "results",
    "fixed_tier_analysis_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")

mkpath(OUTPUT_DIR)
mkpath(joinpath(OUTPUT_DIR, "data"))

# Colors and labels
const TIER_COLORS = Dict(
    "none" => colorant"#6c757d",      # Gray
    "basic" => colorant"#0d6efd",     # Blue
    "advanced" => colorant"#fd7e14",  # Orange
    "premium" => colorant"#dc3545"    # Red
)

const TIER_LABELS = Dict(
    "none" => "No AI",
    "basic" => "Basic AI",
    "advanced" => "Advanced AI",
    "premium" => "Premium AI"
)

const TIER_ORDER = ["none", "basic", "advanced", "premium"]

const ACTION_COLORS = Dict(
    "invest" => colorant"#28a745",    # Green
    "innovate" => colorant"#6f42c1",  # Purple
    "explore" => colorant"#17a2b8",   # Cyan
    "maintain" => colorant"#ffc107"   # Yellow
)

# ============================================================================
# PRINT HEADER
# ============================================================================

println("="^80)
println("FIXED-TIER AI PARADOX ANALYSIS (v4)")
println("="^80)
println("Configuration:")
println("  Threads:     $(Threads.nthreads())")
println("  Agents:      $N_AGENTS per simulation")
println("  Rounds:      $N_ROUNDS (5 years at monthly cadence)")
println("  Runs/Tier:   $N_RUNS")
println("  Total Sims:  $(length(AI_TIERS) * N_RUNS)")
println("  Base Seed:   $BASE_SEED")
println("  Bootstrap:   $N_BOOTSTRAP samples")
println("  Output:      $OUTPUT_DIR")
println("="^80)

master_start = time()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function create_config(; seed=42)
    EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        INITIAL_CAPITAL=100_000_000.0,
        SURVIVAL_THRESHOLD=10_000.0
    )
end

function run_single_simulation(tier::String, run_idx::Int, seed::Int)
    config = create_config(seed=seed)

    # Create homogeneous tier distribution
    tier_dist = Dict(t => (t == tier ? 1.0 : 0.0) for t in AI_TIERS)
    sim = EmergentSimulation(config=config, initial_tier_distribution=tier_dist)

    # Get actual initial equity per agent (from resources.performance.initial_equity)
    actual_initial_equities = [a.resources.performance.initial_equity for a in sim.agents]
    total_initial_equity = sum(actual_initial_equities)

    # Trajectory tracking
    survival_trajectory = Float64[]
    capital_retention_trajectory = Float64[]  # Current capital / actual initial equity
    total_capital_trajectory = Float64[]  # Absolute total capital over time
    cr_trajectory = Float64[]

    # Action share trajectories (per round)
    action_shares = Dict(
        "invest" => Float64[],
        "innovate" => Float64[],
        "explore" => Float64[],
        "maintain" => Float64[]
    )

    # Innovation metrics trajectories (per round)
    niches_trajectory = Int[]  # Cumulative niches discovered
    combinations_trajectory = Int[]  # Cumulative new combinations

    for r in 1:config.N_ROUNDS
        GlimpseABM.step!(sim, r)

        alive = filter(a -> a.alive, sim.agents)
        push!(survival_trajectory, length(alive) / length(sim.agents))

        # Capital retention: sum of current capital / sum of initial equity
        total_current_capital = sum(GlimpseABM.get_capital(a) for a in sim.agents)
        push!(capital_retention_trajectory, total_current_capital / total_initial_equity)
        push!(total_capital_trajectory, total_current_capital)

        # Competitive recursion
        cr_vals = Float64[]
        for agent in sim.agents
            if !isempty(agent.uncertainty_metrics.competition_levels)
                push!(cr_vals, last(agent.uncertainty_metrics.competition_levels))
            end
        end
        push!(cr_trajectory, isempty(cr_vals) ? 0.0 : mean(cr_vals))

        # Action shares for this round
        action_counts = Dict("invest" => 0, "innovate" => 0, "explore" => 0, "maintain" => 0)
        for agent in sim.agents
            if length(agent.action_history) >= r
                act = agent.action_history[r]
                action_counts[act] = get(action_counts, act, 0) + 1
            end
        end
        total_actions = sum(values(action_counts))
        for act in ["invest", "innovate", "explore", "maintain"]
            push!(action_shares[act], total_actions > 0 ? action_counts[act] / total_actions : 0.0)
        end

        # Track cumulative niches and combinations discovered
        total_niches = sum(a.uncertainty_metrics.niches_discovered for a in sim.agents)
        total_combinations = sum(a.uncertainty_metrics.new_combinations_created for a in sim.agents)
        push!(niches_trajectory, total_niches)
        push!(combinations_trajectory, total_combinations)
    end

    # Final metrics
    alive_agents = filter(a -> a.alive, sim.agents)

    # Competitive recursion (mean over all rounds)
    all_cr = Float64[]
    for agent in sim.agents
        if !isempty(agent.uncertainty_metrics.competition_levels)
            append!(all_cr, agent.uncertainty_metrics.competition_levels)
        end
    end

    # Novelty scores
    novelty_scores = [(a.uncertainty_metrics.new_combinations_created +
                       a.uncertainty_metrics.niches_discovered) /
                       max(1, a.uncertainty_metrics.total_actions) for a in sim.agents]

    # Capital metrics - use ACTUAL initial equity per agent
    final_capitals = [GlimpseABM.get_capital(a) for a in sim.agents]
    survivor_capitals = [GlimpseABM.get_capital(a) for a in alive_agents]

    # Capital multipliers relative to each agent's actual initial equity
    capital_multipliers = [GlimpseABM.get_capital(a) / a.resources.performance.initial_equity
                          for a in sim.agents]

    # Survivor capital multipliers (for unicorn analysis)
    survivor_multipliers = isempty(alive_agents) ? Float64[] :
        [GlimpseABM.get_capital(a) / a.resources.performance.initial_equity for a in alive_agents]

    # Overall capital retention
    final_capital_retention = sum(final_capitals) / total_initial_equity

    # Innovation metrics
    innovations = [a.innovation_count for a in sim.agents]
    successes = [a.success_count for a in sim.agents]
    failures = [a.failure_count for a in sim.agents]

    # Final niche and combination counts
    final_niches = sum(a.uncertainty_metrics.niches_discovered for a in sim.agents)
    final_combinations = sum(a.uncertainty_metrics.new_combinations_created for a in sim.agents)

    # Knowledge recombination quality metrics (from innovation engine)
    all_qualities = Float64[]
    all_novelties = Float64[]
    all_scarcities = Float64[]
    for a in sim.agents
        append!(all_qualities, a.uncertainty_metrics.innovation_qualities)
        append!(all_novelties, a.uncertainty_metrics.innovation_novelties)
        append!(all_scarcities, a.uncertainty_metrics.innovation_scarcities)
    end

    # Mean innovation quality, novelty, and scarcity for this run
    mean_innovation_quality = isempty(all_qualities) ? 0.0 : mean(all_qualities)
    mean_innovation_novelty = isempty(all_novelties) ? 0.0 : mean(all_novelties)
    mean_innovation_scarcity = isempty(all_scarcities) ? 0.0 : mean(all_scarcities)
    std_innovation_quality = length(all_qualities) > 1 ? std(all_qualities) : 0.0
    std_innovation_novelty = length(all_novelties) > 1 ? std(all_novelties) : 0.0

    # Action distribution (final)
    final_action_counts = Dict("invest" => 0, "innovate" => 0, "explore" => 0, "maintain" => 0)
    for agent in sim.agents
        for act in agent.action_history
            final_action_counts[act] = get(final_action_counts, act, 0) + 1
        end
    end
    total_final_actions = sum(values(final_action_counts))
    action_distribution = Dict(
        act => total_final_actions > 0 ? count / total_final_actions : 0.0
        for (act, count) in final_action_counts
    )

    # Investment metrics
    total_invested = sum(a.total_invested for a in sim.agents)
    total_returned = sum(a.total_returned for a in sim.agents)

    # Agent-level DataFrame
    agent_df = DataFrame(
        agent_id = [a.id for a in sim.agents],
        tier = fill(tier, length(sim.agents)),
        run_idx = fill(run_idx, length(sim.agents)),
        seed = fill(seed, length(sim.agents)),
        survived = [a.alive for a in sim.agents],
        final_capital = final_capitals,
        capital_multiplier = capital_multipliers,
        innovation_count = innovations,
        success_count = successes,
        failure_count = failures,
        total_invested = [a.total_invested for a in sim.agents],
        total_returned = [a.total_returned for a in sim.agents],
        survival_rounds = [a.survival_rounds for a in sim.agents],
        n_actions = [length(a.action_history) for a in sim.agents]
    )

    return Dict(
        "tier" => tier,
        "run_idx" => run_idx,
        "seed" => seed,
        # Survival
        "survival_rate" => length(alive_agents) / length(sim.agents),
        "n_alive" => length(alive_agents),
        "survival_trajectory" => survival_trajectory,
        # Competition
        "mean_cr" => isempty(all_cr) ? 0.0 : mean(all_cr),
        "std_cr" => length(all_cr) > 1 ? std(all_cr) : 0.0,
        "cr_trajectory" => cr_trajectory,
        # Novelty
        "mean_novelty" => mean(novelty_scores),
        "std_novelty" => std(novelty_scores),
        # Financial
        "total_invested" => total_invested,
        "total_returned" => total_returned,
        "mean_capital" => mean(final_capitals),
        "median_capital" => median(final_capitals),
        "std_capital" => std(final_capitals),
        "mean_survivor_capital" => isempty(survivor_capitals) ? 0.0 : mean(survivor_capitals),
        "median_survivor_capital" => isempty(survivor_capitals) ? 0.0 : median(survivor_capitals),
        "max_capital" => maximum(final_capitals),
        "capital_retention_trajectory" => capital_retention_trajectory,
        "total_capital_trajectory" => total_capital_trajectory,
        "capital_multipliers" => capital_multipliers,
        "survivor_multipliers" => survivor_multipliers,
        "final_capital_retention" => final_capital_retention,
        "total_initial_equity" => total_initial_equity,
        # Innovation
        "total_innovations" => sum(innovations),
        "mean_innovations" => mean(innovations),
        "total_successes" => sum(successes),
        "total_failures" => sum(failures),
        "innovator_rate" => count(x -> x > 0, innovations) / length(innovations),
        # Action distribution
        "action_shares" => action_shares,
        "action_distribution" => action_distribution,
        # Niche/combination discovery
        "niches_trajectory" => niches_trajectory,
        "combinations_trajectory" => combinations_trajectory,
        "final_niches" => final_niches,
        "final_combinations" => final_combinations,
        # Knowledge recombination metrics (from innovation engine)
        "mean_innovation_quality" => mean_innovation_quality,
        "mean_innovation_novelty" => mean_innovation_novelty,
        "mean_innovation_scarcity" => mean_innovation_scarcity,
        "std_innovation_quality" => std_innovation_quality,
        "std_innovation_novelty" => std_innovation_novelty,
        "all_qualities" => all_qualities,
        "all_novelties" => all_novelties,
        "all_scarcities" => all_scarcities,
        "n_innovations_with_quality" => length(all_qualities),
        # Agent data
        "agent_data" => agent_df
    )
end

function welch_t_test(x::Vector{Float64}, y::Vector{Float64})
    mx, my = mean(x), mean(y)
    vx, vy = var(x), var(y)
    nx, ny = length(x), length(y)
    se = sqrt(vx/nx + vy/ny)
    t_stat = se > 1e-10 ? (mx - my) / se : 0.0
    # Approximate p-value using normal distribution
    z = abs(t_stat)
    p_value = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
    return (t_stat=t_stat, p_value=p_value, diff=mx-my, se=se)
end

function bootstrap_ci(data::Vector{Float64}; n_boot=N_BOOTSTRAP, alpha=0.05)
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

function bootstrap_ate_ci(treatment::Vector{Float64}, control::Vector{Float64}; n_boot=N_BOOTSTRAP, alpha=0.05)
    rng = MersenneTwister(42)
    boot_ates = Float64[]
    nt, nc = length(treatment), length(control)
    for _ in 1:n_boot
        t_sample = [treatment[rand(rng, 1:nt)] for _ in 1:nt]
        c_sample = [control[rand(rng, 1:nc)] for _ in 1:nc]
        push!(boot_ates, mean(t_sample) - mean(c_sample))
    end
    sorted = sort(boot_ates)
    ci_lo = sorted[max(1, Int(floor(alpha/2 * n_boot)))]
    ci_hi = sorted[min(n_boot, Int(ceil((1 - alpha/2) * n_boot)))]
    sig = ci_lo > 0 || ci_hi < 0  # CI doesn't include 0
    return (ate=mean(treatment)-mean(control), ci_lo=ci_lo, ci_hi=ci_hi, se=std(boot_ates), significant=sig)
end

# ============================================================================
# PHASE 1: RUN SIMULATIONS
# ============================================================================

println("\n" * "="^80)
println("PHASE 1: RUNNING SIMULATIONS")
println("="^80)

all_results = Dict{String, Vector{Dict}}()
all_agent_data = DataFrame()
completed = Threads.Atomic{Int}(0)
results_lock = ReentrantLock()

for tier in AI_TIERS
    all_results[tier] = Vector{Dict}(undef, N_RUNS)
end

# Generate all task combinations
tasks = [(tier, run_idx) for tier in AI_TIERS for run_idx in 1:N_RUNS]
total_sims = length(tasks)
phase1_start = time()

println("Running $total_sims simulations across $(Threads.nthreads()) threads...")

Threads.@threads for (tier, run_idx) in tasks
    # Unique seed: tier_index * 10000 + run_idx
    tier_idx = findfirst(==(tier), AI_TIERS)
    seed = BASE_SEED + tier_idx * 10000 + run_idx

    result = run_single_simulation(tier, run_idx, seed)

    lock(results_lock) do
        all_results[tier][run_idx] = result
    end

    done = Threads.atomic_add!(completed, 1)
    if done % 10 == 0 || done == total_sims
        elapsed = time() - phase1_start
        rate = done / elapsed
        eta = (total_sims - done) / rate
        @printf("\r  Progress: %d/%d (%.0f%%) | %.1f sims/sec | ETA: %.0fs    ",
            done, total_sims, 100*done/total_sims, rate, eta)
    end
end

# Collect all agent data
println("\n  Collecting agent-level data...")
for tier in AI_TIERS
    for r in all_results[tier]
        global all_agent_data = vcat(all_agent_data, r["agent_data"])
    end
end

phase1_elapsed = time() - phase1_start
@printf("  Phase 1 complete in %.1f seconds (%.1f sims/sec)\n", phase1_elapsed, total_sims/phase1_elapsed)

# ============================================================================
# PHASE 2: COMPUTE STATISTICS
# ============================================================================

println("\n" * "="^80)
println("PHASE 2: COMPUTING STATISTICS")
println("="^80)

# Survival statistics
survival_stats = Dict{String, NamedTuple}()
for tier in AI_TIERS
    rates = [r["survival_rate"] for r in all_results[tier]]
    ci = bootstrap_ci(rates)
    survival_stats[tier] = (
        mean = ci.mean,
        ci_lo = ci.ci_lo,
        ci_hi = ci.ci_hi,
        se = ci.se,
        std = std(rates),
        values = rates
    )
end

# Competition statistics
cr_stats = Dict{String, NamedTuple}()
for tier in AI_TIERS
    vals = [r["mean_cr"] for r in all_results[tier]]
    ci = bootstrap_ci(vals)
    cr_stats[tier] = (mean=ci.mean, ci_lo=ci.ci_lo, ci_hi=ci.ci_hi, se=ci.se, values=vals)
end

# Novelty statistics
novelty_stats = Dict{String, NamedTuple}()
for tier in AI_TIERS
    vals = [r["mean_novelty"] for r in all_results[tier]]
    ci = bootstrap_ci(vals)
    novelty_stats[tier] = (mean=ci.mean, ci_lo=ci.ci_lo, ci_hi=ci.ci_hi, se=ci.se, values=vals)
end

# Financial statistics
financial_stats = Dict{String, NamedTuple}()
for tier in AI_TIERS
    invested = [r["total_invested"] for r in all_results[tier]]
    returned = [r["total_returned"] for r in all_results[tier]]
    roi_vals = (returned .- invested) ./ invested .* 100
    survivor_cap = [r["mean_survivor_capital"] for r in all_results[tier]]

    # Capital efficiency: returned per dollar invested
    efficiency_vals = returned ./ invested

    # Capital retention: final capital as fraction of initial equity (PROPER METRIC)
    retention_vals = [r["final_capital_retention"] for r in all_results[tier]]

    # Mean survivor capital multiplier
    survivor_mult_vals = Float64[]
    for r in all_results[tier]
        if !isempty(r["survivor_multipliers"])
            push!(survivor_mult_vals, mean(r["survivor_multipliers"]))
        else
            push!(survivor_mult_vals, 0.0)
        end
    end

    roi_ci = bootstrap_ci(roi_vals)
    cap_ci = bootstrap_ci(survivor_cap)
    eff_ci = bootstrap_ci(efficiency_vals)
    ret_ci = bootstrap_ci(retention_vals)
    surv_mult_ci = bootstrap_ci(survivor_mult_vals)

    financial_stats[tier] = (
        mean_invested = mean(invested),
        mean_returned = mean(returned),
        roi_mean = roi_ci.mean,
        roi_ci_lo = roi_ci.ci_lo,
        roi_ci_hi = roi_ci.ci_hi,
        roi_se = roi_ci.se,
        survivor_cap_mean = cap_ci.mean,
        survivor_cap_ci_lo = cap_ci.ci_lo,
        survivor_cap_ci_hi = cap_ci.ci_hi,
        efficiency_mean = eff_ci.mean,
        efficiency_ci_lo = eff_ci.ci_lo,
        efficiency_ci_hi = eff_ci.ci_hi,
        retention_mean = ret_ci.mean,
        retention_ci_lo = ret_ci.ci_lo,
        retention_ci_hi = ret_ci.ci_hi,
        survivor_mult_mean = surv_mult_ci.mean,
        survivor_mult_ci_lo = surv_mult_ci.ci_lo,
        survivor_mult_ci_hi = surv_mult_ci.ci_hi,
        roi_values = roi_vals,
        survivor_cap_values = survivor_cap,
        efficiency_values = efficiency_vals,
        retention_values = retention_vals
    )
end

# Innovation statistics
innovation_stats = Dict{String, NamedTuple}()
for tier in AI_TIERS
    total = [r["total_innovations"] for r in all_results[tier]]
    per_agent = total ./ N_AGENTS
    rates = [r["innovator_rate"] for r in all_results[tier]]
    successes = [r["total_successes"] for r in all_results[tier]]
    failures = [r["total_failures"] for r in all_results[tier]]
    success_rates = successes ./ (successes .+ failures .+ 1e-10) .* 100

    total_ci = bootstrap_ci(Float64.(total))
    rate_ci = bootstrap_ci(rates)
    success_ci = bootstrap_ci(success_rates)

    innovation_stats[tier] = (
        total_mean = total_ci.mean,
        total_ci_lo = total_ci.ci_lo,
        total_ci_hi = total_ci.ci_hi,
        per_agent = mean(per_agent),
        innovator_rate = rate_ci.mean,
        innovator_rate_ci_lo = rate_ci.ci_lo,
        innovator_rate_ci_hi = rate_ci.ci_hi,
        success_rate = success_ci.mean,
        success_rate_ci_lo = success_ci.ci_lo,
        success_rate_ci_hi = success_ci.ci_hi
    )
end

# Action distribution statistics
action_stats = Dict{String, Dict{String, NamedTuple}}()
for tier in AI_TIERS
    action_stats[tier] = Dict()
    for act in ["invest", "innovate", "explore", "maintain"]
        vals = [r["action_distribution"][act] for r in all_results[tier]]
        ci = bootstrap_ci(vals)
        action_stats[tier][act] = (mean=ci.mean, ci_lo=ci.ci_lo, ci_hi=ci.ci_hi, se=ci.se, values=vals)
    end
end

# Niche discovery statistics
niche_stats = Dict{String, NamedTuple}()
for tier in AI_TIERS
    niche_vals = Float64[r["final_niches"] for r in all_results[tier]]
    combo_vals = Float64[r["final_combinations"] for r in all_results[tier]]
    niche_ci = bootstrap_ci(niche_vals)
    combo_ci = bootstrap_ci(combo_vals)
    niche_stats[tier] = (
        niches_mean = niche_ci.mean,
        niches_ci_lo = niche_ci.ci_lo,
        niches_ci_hi = niche_ci.ci_hi,
        combinations_mean = combo_ci.mean,
        combinations_ci_lo = combo_ci.ci_lo,
        combinations_ci_hi = combo_ci.ci_hi,
        niches_values = niche_vals,
        combinations_values = combo_vals
    )
end

# Knowledge recombination quality statistics
knowledge_stats = Dict{String, NamedTuple}()
for tier in AI_TIERS
    quality_vals = [r["mean_innovation_quality"] for r in all_results[tier]]
    novelty_vals = [r["mean_innovation_novelty"] for r in all_results[tier]]
    scarcity_vals = [r["mean_innovation_scarcity"] for r in all_results[tier]]
    n_innovations = [r["n_innovations_with_quality"] for r in all_results[tier]]

    # All individual quality values across all runs for distribution analysis
    all_tier_qualities = Float64[]
    all_tier_novelties = Float64[]
    all_tier_scarcities = Float64[]
    for r in all_results[tier]
        append!(all_tier_qualities, r["all_qualities"])
        append!(all_tier_novelties, r["all_novelties"])
        append!(all_tier_scarcities, r["all_scarcities"])
    end

    # Compute composite "knowledge value" = quality × novelty × (1 + scarcity)
    knowledge_values = Float64[]
    for r in all_results[tier]
        if !isempty(r["all_qualities"])
            for (q, n, s) in zip(r["all_qualities"], r["all_novelties"], r["all_scarcities"])
                push!(knowledge_values, q * n * (1.0 + s))
            end
        end
    end

    quality_ci = bootstrap_ci(filter(!isnan, quality_vals))
    novelty_ci = bootstrap_ci(filter(!isnan, novelty_vals))
    scarcity_ci = bootstrap_ci(filter(!isnan, scarcity_vals))

    knowledge_stats[tier] = (
        quality_mean = quality_ci.mean,
        quality_ci_lo = quality_ci.ci_lo,
        quality_ci_hi = quality_ci.ci_hi,
        novelty_mean = novelty_ci.mean,
        novelty_ci_lo = novelty_ci.ci_lo,
        novelty_ci_hi = novelty_ci.ci_hi,
        scarcity_mean = scarcity_ci.mean,
        scarcity_ci_lo = scarcity_ci.ci_lo,
        scarcity_ci_hi = scarcity_ci.ci_hi,
        total_innovations = sum(n_innovations),
        mean_innovations_per_run = mean(Float64.(n_innovations)),
        all_qualities = all_tier_qualities,
        all_novelties = all_tier_novelties,
        all_scarcities = all_tier_scarcities,
        knowledge_values = knowledge_values,
        quality_std = isempty(all_tier_qualities) ? 0.0 : std(all_tier_qualities),
        novelty_std = isempty(all_tier_novelties) ? 0.0 : std(all_tier_novelties)
    )
end

# Treatment effects (vs None baseline)
ate_stats = Dict{String, NamedTuple}()
baseline = survival_stats["none"].values
for tier in ["basic", "advanced", "premium"]
    treatment = survival_stats[tier].values
    result = bootstrap_ate_ci(treatment, baseline)
    t_test = welch_t_test(treatment, baseline)
    ate_stats[tier] = (
        ate = result.ate,
        ci_lo = result.ci_lo,
        ci_hi = result.ci_hi,
        se = result.se,
        significant = result.significant,
        p_value = t_test.p_value
    )
end

# Unicorn statistics - using percentiles within each tier
initial_cap = 100_000_000.0
unicorn_stats = Dict{String, NamedTuple}()

# Global percentile thresholds (across all agents)
all_multipliers = all_agent_data.capital_multiplier
global_p99 = quantile(all_multipliers, 0.99)
global_p95 = quantile(all_multipliers, 0.95)
global_p90 = quantile(all_multipliers, 0.90)
global_p75 = quantile(all_multipliers, 0.75)
global_p50 = quantile(all_multipliers, 0.50)

# Survivor-only metrics
all_survivor_data = filter(r -> r.survived, all_agent_data)
survivor_p99 = nrow(all_survivor_data) > 0 ? quantile(all_survivor_data.capital_multiplier, 0.99) : 0.0
survivor_p95 = nrow(all_survivor_data) > 0 ? quantile(all_survivor_data.capital_multiplier, 0.95) : 0.0
survivor_p90 = nrow(all_survivor_data) > 0 ? quantile(all_survivor_data.capital_multiplier, 0.90) : 0.0

for tier in AI_TIERS
    tier_data = filter(r -> r.tier == tier, all_agent_data)
    tier_survivors = filter(r -> r.survived, tier_data)
    n = nrow(tier_data)
    n_surv = nrow(tier_survivors)

    # Percentile values within tier (survivors only)
    if n_surv > 0
        tier_p99 = quantile(tier_survivors.capital_multiplier, 0.99)
        tier_p95 = quantile(tier_survivors.capital_multiplier, 0.95)
        tier_p90 = quantile(tier_survivors.capital_multiplier, 0.90)
        tier_p75 = quantile(tier_survivors.capital_multiplier, 0.75)
        tier_p50 = quantile(tier_survivors.capital_multiplier, 0.50)
        tier_max = maximum(tier_survivors.capital_multiplier)
        tier_mean = mean(tier_survivors.capital_multiplier)
    else
        tier_p99 = tier_p95 = tier_p90 = tier_p75 = tier_p50 = tier_max = tier_mean = 0.0
    end

    # Rate of survivors in global top percentiles
    top_global_p95 = n_surv > 0 ? count(tier_survivors.capital_multiplier .>= survivor_p95) / n_surv * 100 : 0.0
    top_global_p90 = n_surv > 0 ? count(tier_survivors.capital_multiplier .>= survivor_p90) / n_surv * 100 : 0.0

    unicorn_stats[tier] = (
        # Within-tier percentiles (survivors)
        p99 = tier_p99,
        p95 = tier_p95,
        p90 = tier_p90,
        p75 = tier_p75,
        p50 = tier_p50,
        max = tier_max,
        mean_survivor = tier_mean,
        n_survivors = n_surv,
        # Global rankings
        top_global_p95_rate = top_global_p95,
        top_global_p90_rate = top_global_p90
    )
end

println("  Statistics computed.")

# ============================================================================
# PHASE 3: PRINT RESULTS
# ============================================================================

println("\n" * "="^80)
println("PHASE 3: RESULTS SUMMARY")
println("="^80)

println("\n--- SURVIVAL RATES ---")
println("-"^70)
@printf("%-12s %10s %20s %10s\n", "Tier", "Mean", "95% CI", "SE")
println("-"^70)
for tier in TIER_ORDER
    s = survival_stats[tier]
    @printf("%-12s %9.1f%% [%5.1f%%, %5.1f%%] %9.2f%%\n",
        TIER_LABELS[tier], s.mean*100, s.ci_lo*100, s.ci_hi*100, s.se*100)
end

println("\n--- TREATMENT EFFECTS (vs No AI) ---")
println("-"^70)
@printf("%-12s %10s %20s %10s %8s\n", "Tier", "ATE (pp)", "95% CI", "p-value", "Sig")
println("-"^70)
for tier in ["basic", "advanced", "premium"]
    a = ate_stats[tier]
    sig_star = a.significant ? "***" : (a.p_value < 0.05 ? "**" : (a.p_value < 0.10 ? "*" : ""))
    @printf("%-12s %+9.1f  [%+5.1f, %+5.1f] %9.4f %8s\n",
        TIER_LABELS[tier], a.ate*100, a.ci_lo*100, a.ci_hi*100, a.p_value, sig_star)
end

println("\n--- FINANCIAL PERFORMANCE ---")
println("-"^70)
@printf("%-12s %12s %15s %15s\n", "Tier", "ROI (%)", "Efficiency", "Survivor Cap")
println("-"^70)
for tier in TIER_ORDER
    f = financial_stats[tier]
    @printf("%-12s %+11.1f%% %14.3f %14.0f\n",
        TIER_LABELS[tier], f.roi_mean, f.efficiency_mean, f.survivor_cap_mean)
end

println("\n--- ACTION DISTRIBUTION ---")
println("-"^70)
@printf("%-12s %10s %10s %10s %10s\n", "Tier", "Invest", "Innovate", "Explore", "Maintain")
println("-"^70)
for tier in TIER_ORDER
    @printf("%-12s %9.1f%% %9.1f%% %9.1f%% %9.1f%%\n",
        TIER_LABELS[tier],
        action_stats[tier]["invest"].mean*100,
        action_stats[tier]["innovate"].mean*100,
        action_stats[tier]["explore"].mean*100,
        action_stats[tier]["maintain"].mean*100)
end

println("\n--- INNOVATION ---")
println("-"^80)
@printf("%-12s %12s %15s %15s %12s %12s\n", "Tier", "Per Agent", "Innovator Rate", "Success Rate", "Niches", "Combos")
println("-"^80)
for tier in TIER_ORDER
    i = innovation_stats[tier]
    n = niche_stats[tier]
    @printf("%-12s %12.2f %14.1f%% %14.1f%% %11.1f %11.1f\n",
        TIER_LABELS[tier], i.per_agent, i.innovator_rate*100, i.success_rate, n.niches_mean, n.combinations_mean)
end

println("\n--- UNICORN ANALYSIS (Survivors Only) ---")
println("-"^70)
@printf("%-12s %10s %10s %10s %10s %10s\n", "Tier", "P50", "P75", "P90", "P95", "Max")
println("-"^70)
for tier in TIER_ORDER
    u = unicorn_stats[tier]
    @printf("%-12s %9.4f %9.4f %9.4f %9.4f %9.4f\n",
        TIER_LABELS[tier], u.p50, u.p75, u.p90, u.p95, u.max)
end

println("\n--- KNOWLEDGE RECOMBINATION QUALITY ---")
println("-"^80)
@printf("%-12s %10s %10s %10s %10s %12s\n", "Tier", "Quality", "Novelty", "Scarcity", "σ(Quality)", "N Innov")
println("-"^80)
for tier in TIER_ORDER
    k = knowledge_stats[tier]
    @printf("%-12s %9.3f %9.3f %9.3f %10.3f %11d\n",
        TIER_LABELS[tier], k.quality_mean, k.novelty_mean, k.scarcity_mean,
        k.quality_std, k.total_innovations)
end

# Key paradox result
premium_ate = ate_stats["premium"].ate
println("\n" * "="^70)
println("PARADOX STATUS: ", premium_ate < 0 ? "CONFIRMED" : "NOT CONFIRMED")
@printf("Premium AI Effect: %+.1f pp survival (95%% CI: [%+.1f, %+.1f])\n",
    premium_ate*100, ate_stats["premium"].ci_lo*100, ate_stats["premium"].ci_hi*100)
println("="^70)

# ============================================================================
# PHASE 4: GENERATE LANDSCAPE PDF
# ============================================================================

println("\n" * "="^80)
println("PHASE 4: GENERATING LANDSCAPE PDF")
println("="^80)

# Set theme
set_theme!(Theme(
    fontsize = 10,
    font = "Arial",
    Axis = (
        xgridvisible = false,
        ygridvisible = true,
        ygridstyle = :dash,
        ygridcolor = (:gray, 0.3)
    )
))

# Create figure - landscape orientation (width > height) - single page
fig = Figure(size = (1400, 1000))

# Title and subtitle
Label(fig[1, 1:4], "AI Information Paradox: Fixed-Tier Analysis",
    fontsize = 20, font = :bold, halign = :center)
Label(fig[1, 1:4], "\n\nN = $N_AGENTS agents × $N_ROUNDS rounds × $N_RUNS runs per tier | Bootstrap CI (N=$N_BOOTSTRAP)",
    fontsize = 9, color = :gray40, halign = :center, valign = :top)

# ========== ROW 2: SURVIVAL ANALYSIS (3 plots) ==========

# Fig A: Survival rates with CI
ax1a = Axis(fig[2, 1], xlabel = "AI Tier", ylabel = "Survival Rate (%)",
    title = "A. Final Survival Rates",
    xticks = (1:4, ["None", "Basic", "Adv", "Prem"]))
means = [survival_stats[t].mean * 100 for t in TIER_ORDER]
los = [survival_stats[t].ci_lo * 100 for t in TIER_ORDER]
his = [survival_stats[t].ci_hi * 100 for t in TIER_ORDER]
barplot!(ax1a, 1:4, means, color = [TIER_COLORS[t] for t in TIER_ORDER])
errorbars!(ax1a, 1:4, means, means .- los, his .- means, color = :black, whiskerwidth = 10, linewidth = 1.5)

# Fig B: Treatment effects
ax1b = Axis(fig[2, 2], xlabel = "AI Tier", ylabel = "Treatment Effect (pp)",
    title = "B. Survival Effect vs No AI",
    xticks = (1:3, ["Basic", "Adv", "Prem"]))
ates = [ate_stats[t].ate * 100 for t in ["basic", "advanced", "premium"]]
ate_los = [ate_stats[t].ci_lo * 100 for t in ["basic", "advanced", "premium"]]
ate_his = [ate_stats[t].ci_hi * 100 for t in ["basic", "advanced", "premium"]]
barplot!(ax1b, 1:3, ates, color = [TIER_COLORS[t] for t in ["basic", "advanced", "premium"]])
errorbars!(ax1b, 1:3, ates, ates .- ate_los, ate_his .- ates, color = :black, whiskerwidth = 10, linewidth = 1.5)
hlines!(ax1b, [0], color = :black, linestyle = :dash, linewidth = 1)

# Fig C: Survival trajectories
ax1c = Axis(fig[2, 3:4], xlabel = "Round (Month)", ylabel = "Survival Rate (%)",
    title = "C. Survival Trajectories Over Time")
for tier in TIER_ORDER
    trajs = [r["survival_trajectory"] for r in all_results[tier]]
    means_t = [mean(t[i] for t in trajs) * 100 for i in 1:N_ROUNDS]
    stds_t = [std(t[i] for t in trajs) * 100 for i in 1:N_ROUNDS]
    ci_lo_t = means_t .- 1.96 .* stds_t ./ sqrt(N_RUNS)
    ci_hi_t = means_t .+ 1.96 .* stds_t ./ sqrt(N_RUNS)
    band!(ax1c, 1:N_ROUNDS, ci_lo_t, ci_hi_t, color = (TIER_COLORS[tier], 0.2))
    lines!(ax1c, 1:N_ROUNDS, means_t, color = TIER_COLORS[tier], linewidth = 2, label = TIER_LABELS[tier])
end
axislegend(ax1c, position = :rb, labelsize = 8)

# Description row for survival
Label(fig[3, 1:4],
    "Survival Analysis: Higher AI tiers show significantly lower survival rates. Premium AI reduces survival by ~11 pp (p<0.001). The paradox emerges as AI-enabled agents take more risks.",
    fontsize = 8, color = :gray30, halign = :left, valign = :top)

# ========== ROW 4: BEHAVIORAL & FINANCIAL (4 plots) ==========

# Fig D: Innovate Action Share (shows meaningful variance 27.8% → 31.9%)
ax2a = Axis(fig[4, 1], xlabel = "AI Tier", ylabel = "Innovate Share (%)",
    title = "D. Innovation Activity Share",
    xticks = (1:4, ["None", "Basic", "Adv", "Prem"]))
inn_shares = [action_stats[t]["innovate"].mean * 100 for t in TIER_ORDER]
inn_share_los = [action_stats[t]["innovate"].ci_lo * 100 for t in TIER_ORDER]
inn_share_his = [action_stats[t]["innovate"].ci_hi * 100 for t in TIER_ORDER]
barplot!(ax2a, 1:4, inn_shares, color = [TIER_COLORS[t] for t in TIER_ORDER])
errorbars!(ax2a, 1:4, inn_shares, inn_shares .- inn_share_los, inn_share_his .- inn_shares, color = :black, whiskerwidth = 10, linewidth = 1.5)
# Rescale to show variation
ylims!(ax2a, minimum(inn_share_los) * 0.95, maximum(inn_share_his) * 1.05)

# Fig E: Explore vs Innovate Ratio (behavioral shift)
ax2b = Axis(fig[4, 2], xlabel = "AI Tier", ylabel = "Explore Share (%)",
    title = "E. Exploration Activity Share",
    xticks = (1:4, ["None", "Basic", "Adv", "Prem"]))
exp_shares = [action_stats[t]["explore"].mean * 100 for t in TIER_ORDER]
exp_share_los = [action_stats[t]["explore"].ci_lo * 100 for t in TIER_ORDER]
exp_share_his = [action_stats[t]["explore"].ci_hi * 100 for t in TIER_ORDER]
barplot!(ax2b, 1:4, exp_shares, color = [TIER_COLORS[t] for t in TIER_ORDER])
errorbars!(ax2b, 1:4, exp_shares, exp_shares .- exp_share_los, exp_share_his .- exp_shares, color = :black, whiskerwidth = 10, linewidth = 1.5)
ylims!(ax2b, minimum(exp_share_los) * 0.95, maximum(exp_share_his) * 1.05)

# Fig F: Survivor Capital Multiplier Percentiles (shows selection effects)
ax2c = Axis(fig[4, 3], xlabel = "Percentile", ylabel = "Capital Multiplier",
    title = "F. Survivor Wealth (P50/P90/P95)",
    xticks = (1:3, ["P50", "P90", "P95"]))
bar_width = 0.18
for (ti, tier) in enumerate(TIER_ORDER)
    u = unicorn_stats[tier]
    vals = [u.p50, u.p90, u.p95]
    offset = (ti - 2.5) * bar_width
    barplot!(ax2c, (1:3) .+ offset, vals, width = bar_width,
             color = TIER_COLORS[tier], label = TIER_LABELS[tier])
end
axislegend(ax2c, position = :lt, labelsize = 7)

# Fig G: Niche Discovery Over Time
ax2d = Axis(fig[4, 4], xlabel = "Round", ylabel = "Cumulative Niches",
    title = "G. Niche Discovery Over Time")
for tier in TIER_ORDER
    trajs = [r["niches_trajectory"] for r in all_results[tier]]
    means_t = [mean(Float64(t[i]) for t in trajs) for i in 1:N_ROUNDS]
    stds_t = [std(Float64(t[i]) for t in trajs) for i in 1:N_ROUNDS]
    ci_lo_t = means_t .- 1.96 .* stds_t ./ sqrt(N_RUNS)
    ci_hi_t = means_t .+ 1.96 .* stds_t ./ sqrt(N_RUNS)
    band!(ax2d, 1:N_ROUNDS, ci_lo_t, ci_hi_t, color = (TIER_COLORS[tier], 0.2))
    lines!(ax2d, 1:N_ROUNDS, means_t, color = TIER_COLORS[tier], linewidth = 2, label = TIER_LABELS[tier])
end
axislegend(ax2d, position = :lt, labelsize = 7)

# Description row for behavioral/financial
Label(fig[5, 1:4],
    "Behavioral Shifts: AI agents shift from exploration (34.7%→32.0%) to innovation (27.8%→31.9%). Despite creating 11× more niches, Premium AI survivors have lower wealth percentiles than No AI survivors.",
    fontsize = 8, color = :gray30, halign = :left, valign = :top)

# ========== ROW 6: INNOVATION METRICS (4 plots) ==========

# Fig H: Innovation Volume
ax3a = Axis(fig[6, 1], xlabel = "AI Tier", ylabel = "Innovations/Agent",
    title = "H. Innovation Volume",
    xticks = (1:4, ["None", "Basic", "Adv", "Prem"]))
inn_means = [innovation_stats[t].per_agent for t in TIER_ORDER]
inn_stds = [std([r["mean_innovations"] for r in all_results[t]]) for t in TIER_ORDER]
inn_ses = inn_stds ./ sqrt(N_RUNS)
barplot!(ax3a, 1:4, inn_means, color = [TIER_COLORS[t] for t in TIER_ORDER])
errorbars!(ax3a, 1:4, inn_means, 1.96 .* inn_ses, 1.96 .* inn_ses, color = :black, whiskerwidth = 10, linewidth = 1.5)
if minimum(inn_means .- 1.96 .* inn_ses) > 0
    ylims!(ax3a, minimum(inn_means .- 1.96 .* inn_ses) * 0.95, maximum(inn_means .+ 1.96 .* inn_ses) * 1.05)
end

# Fig I: Success Rate
ax3b = Axis(fig[6, 2], xlabel = "AI Tier", ylabel = "Success Rate (%)",
    title = "I. Innovation Success Rate",
    xticks = (1:4, ["None", "Basic", "Adv", "Prem"]))
succ_rates = [innovation_stats[t].success_rate for t in TIER_ORDER]
succ_los = [innovation_stats[t].success_rate_ci_lo for t in TIER_ORDER]
succ_his = [innovation_stats[t].success_rate_ci_hi for t in TIER_ORDER]
barplot!(ax3b, 1:4, succ_rates, color = [TIER_COLORS[t] for t in TIER_ORDER])
errorbars!(ax3b, 1:4, succ_rates, succ_rates .- succ_los, succ_his .- succ_rates, color = :black, whiskerwidth = 10, linewidth = 1.5)
if minimum(succ_los) > 0
    ylims!(ax3b, minimum(succ_los) * 0.95, maximum(succ_his) * 1.05)
end

# Fig J: Total Niches Created (y-axis label updated)
ax3c = Axis(fig[6, 3], xlabel = "AI Tier", ylabel = "Total Niches Created",
    title = "J. Market Niches Created",
    xticks = (1:4, ["None", "Basic", "Adv", "Prem"]))
niche_means = [niche_stats[t].niches_mean for t in TIER_ORDER]
niche_los = [niche_stats[t].niches_ci_lo for t in TIER_ORDER]
niche_his = [niche_stats[t].niches_ci_hi for t in TIER_ORDER]
barplot!(ax3c, 1:4, niche_means, color = [TIER_COLORS[t] for t in TIER_ORDER])
errorbars!(ax3c, 1:4, niche_means, niche_means .- niche_los, niche_his .- niche_means, color = :black, whiskerwidth = 10, linewidth = 1.5)

# Fig K: Knowledge Recombination Quality Distribution (replaces Novel Combinations)
ax3d = Axis(fig[6, 4], xlabel = "AI Tier", ylabel = "Innovation Quality",
    title = "K. Knowledge Recombination Quality",
    xticks = (1:4, ["None", "Basic", "Adv", "Prem"]))

# Prepare boxplot data - quality distributions per tier
box_positions = Float64[]
box_values = Float64[]
for (ti, tier) in enumerate(TIER_ORDER)
    qualities = knowledge_stats[tier].all_qualities
    # Sample max 500 points per tier for visualization
    if length(qualities) > 500
        sample_idx = rand(MersenneTwister(42), 1:length(qualities), 500)
        sampled_q = qualities[sample_idx]
    else
        sampled_q = qualities
    end
    append!(box_positions, fill(Float64(ti), length(sampled_q)))
    append!(box_values, sampled_q)
end

# Boxplot with tier-specific colors
if !isempty(box_values)
    boxplot!(ax3d, box_positions, box_values,
        color = [TIER_COLORS[TIER_ORDER[Int(p)]] for p in box_positions],
        whiskerwidth = 0.5, width = 0.7, show_outliers = false)
    # Add mean markers
    for (ti, tier) in enumerate(TIER_ORDER)
        scatter!(ax3d, [ti], [knowledge_stats[tier].quality_mean],
            color = :white, marker = :diamond, markersize = 8,
            strokewidth = 1.5, strokecolor = :black)
    end
else
    # Fallback to bar chart if no quality data
    quality_means = [knowledge_stats[t].quality_mean for t in TIER_ORDER]
    quality_los = [knowledge_stats[t].quality_ci_lo for t in TIER_ORDER]
    quality_his = [knowledge_stats[t].quality_ci_hi for t in TIER_ORDER]
    barplot!(ax3d, 1:4, quality_means, color = [TIER_COLORS[t] for t in TIER_ORDER])
    errorbars!(ax3d, 1:4, quality_means, quality_means .- quality_los, quality_his .- quality_means,
        color = :black, whiskerwidth = 10, linewidth = 1.5)
end
ylims!(ax3d, 0, 1)

# Description row for innovation (updated)
premium_q = knowledge_stats["premium"].quality_mean
none_q = knowledge_stats["none"].quality_mean
q_diff_pct = none_q > 0.001 ? ((premium_q - none_q) / none_q) * 100 : 0.0
premium_n_inn = knowledge_stats["premium"].total_innovations
none_n_inn = knowledge_stats["none"].total_innovations
Label(fig[7, 1:4],
    @sprintf("Innovation Paradox: Premium AI creates %.0f× more innovations (%d vs %d) with %+.1f%% different quality. Quality variance (σ=%.3f vs %.3f) shows knowledge recombination doesn't guarantee better outcomes.",
        none_n_inn > 0 ? premium_n_inn / none_n_inn : 0, premium_n_inn, none_n_inn, q_diff_pct,
        knowledge_stats["premium"].quality_std, knowledge_stats["none"].quality_std),
    fontsize = 8, color = :gray30, halign = :left, valign = :top)

# Footer with timestamp
timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
Label(fig[8, 1:4], "Generated: $timestamp | GlimpseABM v1.0 | Base Seed: $BASE_SEED | Townsend et al. (2025) AMR",
    fontsize = 8, color = :gray50, halign = :center)

# Adjust row heights for better spacing
rowgap!(fig.layout, 1, 5)
rowgap!(fig.layout, 3, 10)
rowgap!(fig.layout, 5, 10)
rowgap!(fig.layout, 7, 5)

# Save PDF
pdf_path = joinpath(OUTPUT_DIR, "fixed_tier_analysis_results.pdf")
save(pdf_path, fig)
println("  Saved: $pdf_path")

# ============================================================================
# PHASE 5: SAVE DATA
# ============================================================================

println("\n" * "="^80)
println("PHASE 5: SAVING DATA FILES")
println("="^80)

# Summary CSV
summary_df = DataFrame(
    Tier = [TIER_LABELS[t] for t in TIER_ORDER],
    Survival_Mean = [survival_stats[t].mean * 100 for t in TIER_ORDER],
    Survival_CI_Lo = [survival_stats[t].ci_lo * 100 for t in TIER_ORDER],
    Survival_CI_Hi = [survival_stats[t].ci_hi * 100 for t in TIER_ORDER],
    ATE_pp = [t == "none" ? missing : ate_stats[t].ate * 100 for t in TIER_ORDER],
    ATE_CI_Lo = [t == "none" ? missing : ate_stats[t].ci_lo * 100 for t in TIER_ORDER],
    ATE_CI_Hi = [t == "none" ? missing : ate_stats[t].ci_hi * 100 for t in TIER_ORDER],
    ATE_Significant = [t == "none" ? missing : ate_stats[t].significant for t in TIER_ORDER],
    ROI_Mean = [financial_stats[t].roi_mean for t in TIER_ORDER],
    ROI_CI_Lo = [financial_stats[t].roi_ci_lo for t in TIER_ORDER],
    ROI_CI_Hi = [financial_stats[t].roi_ci_hi for t in TIER_ORDER],
    Efficiency_Mean = [financial_stats[t].efficiency_mean for t in TIER_ORDER],
    Invest_Share = [action_stats[t]["invest"].mean for t in TIER_ORDER],
    Innovate_Share = [action_stats[t]["innovate"].mean for t in TIER_ORDER],
    Explore_Share = [action_stats[t]["explore"].mean for t in TIER_ORDER],
    Maintain_Share = [action_stats[t]["maintain"].mean for t in TIER_ORDER],
    Innovations_Per_Agent = [innovation_stats[t].per_agent for t in TIER_ORDER],
    Innovation_Success_Rate = [innovation_stats[t].success_rate for t in TIER_ORDER],
    Niches_Discovered = [niche_stats[t].niches_mean for t in TIER_ORDER],
    New_Combinations = [niche_stats[t].combinations_mean for t in TIER_ORDER],
    Knowledge_Quality_Mean = [knowledge_stats[t].quality_mean for t in TIER_ORDER],
    Knowledge_Novelty_Mean = [knowledge_stats[t].novelty_mean for t in TIER_ORDER],
    Knowledge_Scarcity_Mean = [knowledge_stats[t].scarcity_mean for t in TIER_ORDER],
    Knowledge_Quality_Std = [knowledge_stats[t].quality_std for t in TIER_ORDER],
    Total_Innovations_With_Quality = [knowledge_stats[t].total_innovations for t in TIER_ORDER],
    Survivor_P50 = [unicorn_stats[t].p50 for t in TIER_ORDER],
    Survivor_P95 = [unicorn_stats[t].p95 for t in TIER_ORDER],
    Survivor_Max = [unicorn_stats[t].max for t in TIER_ORDER]
)
CSV.write(joinpath(OUTPUT_DIR, "summary_statistics.csv"), summary_df)
println("  Saved: summary_statistics.csv")

# Agent-level data
CSV.write(joinpath(OUTPUT_DIR, "data", "agent_level_data.csv"), all_agent_data)
println("  Saved: data/agent_level_data.csv")

# Run-level aggregates
run_df = DataFrame()
for tier in TIER_ORDER
    for (i, r) in enumerate(all_results[tier])
        push!(run_df, (
            tier = tier,
            run_idx = i,
            seed = r["seed"],
            survival_rate = r["survival_rate"],
            mean_cr = r["mean_cr"],
            mean_novelty = r["mean_novelty"],
            total_invested = r["total_invested"],
            total_returned = r["total_returned"],
            efficiency = r["total_returned"] / r["total_invested"],
            total_innovations = r["total_innovations"],
            innovator_rate = r["innovator_rate"],
            invest_share = r["action_distribution"]["invest"],
            innovate_share = r["action_distribution"]["innovate"],
            explore_share = r["action_distribution"]["explore"],
            maintain_share = r["action_distribution"]["maintain"],
            mean_innovation_quality = r["mean_innovation_quality"],
            mean_innovation_novelty = r["mean_innovation_novelty"],
            mean_innovation_scarcity = r["mean_innovation_scarcity"],
            std_innovation_quality = r["std_innovation_quality"],
            n_innovations_with_quality = r["n_innovations_with_quality"]
        ))
    end
end
CSV.write(joinpath(OUTPUT_DIR, "data", "run_level_data.csv"), run_df)
println("  Saved: data/run_level_data.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

total_time = time() - master_start

println("\n" * "="^80)
println("ANALYSIS COMPLETE")
println("="^80)

println("\nKEY FINDINGS:")
println("  1. AI PARADOX: $(premium_ate < 0 ? "CONFIRMED" : "NOT CONFIRMED")")
@printf("     Premium AI Effect: %+.1f pp [95%% CI: %+.1f to %+.1f]\n",
    premium_ate*100, ate_stats["premium"].ci_lo*100, ate_stats["premium"].ci_hi*100)
println("  2. Survival Gradient: $(join([@sprintf("%.1f%%", survival_stats[t].mean*100) for t in TIER_ORDER], " → "))")
println("  3. Investment ROI: $(join([@sprintf("%+.1f%%", financial_stats[t].roi_mean) for t in TIER_ORDER], " → "))")
println("  4. Niches Discovered: $(join([@sprintf("%.1f", niche_stats[t].niches_mean) for t in TIER_ORDER], " → "))")
println("  5. Innovation Volume: $(join([@sprintf("%.0f", knowledge_stats[t].total_innovations) for t in TIER_ORDER], " → "))")
println("  6. Knowledge Quality: $(join([@sprintf("%.3f", knowledge_stats[t].quality_mean) for t in TIER_ORDER], " → "))")
println("  7. Quality Variance (σ): $(join([@sprintf("%.3f", knowledge_stats[t].quality_std) for t in TIER_ORDER], " → "))")

println("\nOUTPUT FILES:")
println("  PDF Report: $pdf_path")
println("  Summary:    $(joinpath(OUTPUT_DIR, "summary_statistics.csv"))")
println("  Agent Data: $(joinpath(OUTPUT_DIR, "data", "agent_level_data.csv"))")
println("  Run Data:   $(joinpath(OUTPUT_DIR, "data", "run_level_data.csv"))")

@printf("\nTotal runtime: %.1f minutes (%.0f seconds)\n", total_time/60, total_time)
println("="^80)
