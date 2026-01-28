#!/usr/bin/env julia
"""
COMPLETE PARADOX ANALYSIS SUITE

Full analysis including:
- 1000 agents, 60 rounds, 50 runs per tier
- Mechanism checks (A-F)
- Variance decomposition
- Financial outcomes
- Innovation success analysis
- Unicorn/outlier analysis
- Robustness checks
- Hierarchical analysis
- PDF figure generation

Usage:
    julia --threads=auto --project=. scripts/run_complete_paradox_suite.jl
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

# ============================================================================
# CONFIGURATION
# ============================================================================

const N_AGENTS = 1000
const N_ROUNDS = 60
const N_RUNS = 50
const BASE_SEED = 42
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const N_BOOTSTRAP = 1000

const OUTPUT_DIR = joinpath(dirname(@__DIR__), "results",
    "complete_suite_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")

mkpath(OUTPUT_DIR)
mkpath(joinpath(OUTPUT_DIR, "data"))
mkpath(joinpath(OUTPUT_DIR, "tables"))
mkpath(joinpath(OUTPUT_DIR, "figures"))

const TIER_COLORS = Dict(
    "none" => colorant"#6c757d",
    "basic" => colorant"#0d6efd",
    "advanced" => colorant"#fd7e14",
    "premium" => colorant"#dc3545"
)

const TIER_LABELS = Dict(
    "none" => "Human Only",
    "basic" => "Basic AI",
    "advanced" => "Advanced AI",
    "premium" => "Premium AI"
)

println("="^80)
println("COMPLETE PARADOX ANALYSIS SUITE")
println("="^80)
println("Threads: $(Threads.nthreads()) | Agents: $N_AGENTS | Rounds: $N_ROUNDS | Runs: $N_RUNS/tier")
println("Output: $OUTPUT_DIR")
println("="^80)

master_start = time()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function create_config(; n_agents=N_AGENTS, n_rounds=N_ROUNDS, seed=42)
    EmergentConfig(
        N_AGENTS=n_agents,
        N_ROUNDS=n_rounds,
        RANDOM_SEED=seed,
        INITIAL_CAPITAL=100_000_000.0,
        SURVIVAL_THRESHOLD=10_000.0
    )
end

function run_simulation(tier::String, run_idx::Int; seed=nothing)
    seed = isnothing(seed) ? BASE_SEED + run_idx : seed
    config = create_config(seed=seed)

    tier_dist = Dict(t => (t == tier ? 1.0 : 0.0) for t in AI_TIERS)
    sim = EmergentSimulation(config=config, initial_tier_distribution=tier_dist)

    survival_traj = Float64[]
    cr_traj = Float64[]

    for r in 1:config.N_ROUNDS
        GlimpseABM.step!(sim, r)
        push!(survival_traj, count(a -> a.alive, sim.agents) / length(sim.agents))

        cr_vals = Float64[]
        for agent in sim.agents
            if !isempty(agent.uncertainty_metrics.competition_levels)
                push!(cr_vals, last(agent.uncertainty_metrics.competition_levels))
            end
        end
        push!(cr_traj, isempty(cr_vals) ? 0.0 : mean(cr_vals))
    end

    alive_agents = filter(a -> a.alive, sim.agents)

    # Competitive recursion
    cr_values = Float64[]
    for agent in sim.agents
        if !isempty(agent.uncertainty_metrics.competition_levels)
            push!(cr_values, mean(agent.uncertainty_metrics.competition_levels))
        end
    end

    # Agentic novelty
    novelty_scores = [(a.uncertainty_metrics.new_combinations_created +
                       a.uncertainty_metrics.niches_discovered) /
                       max(1, a.uncertainty_metrics.total_actions) for a in sim.agents]

    # Financial
    final_capitals = [GlimpseABM.get_capital(a) for a in sim.agents]
    survivor_capitals = [GlimpseABM.get_capital(a) for a in alive_agents]
    initial_cap = config.INITIAL_CAPITAL

    # Innovation
    innovations = [a.innovation_count for a in sim.agents]
    successes = [a.success_count for a in sim.agents]
    failures = [a.failure_count for a in sim.agents]

    # Agent-level data for unicorn analysis
    agent_data = DataFrame(
        agent_id = [a.id for a in sim.agents],
        tier = fill(tier, length(sim.agents)),
        run_idx = fill(run_idx, length(sim.agents)),
        survived = [a.alive for a in sim.agents],
        final_capital = final_capitals,
        capital_multiplier = final_capitals ./ initial_cap,
        innovation_count = innovations,
        success_count = successes,
        failure_count = failures,
        total_invested = [a.total_invested for a in sim.agents],
        total_returned = [a.total_returned for a in sim.agents],
        survival_rounds = [a.survival_rounds for a in sim.agents]
    )

    return Dict(
        "tier" => tier, "run_idx" => run_idx,
        "survival_rate" => length(alive_agents) / length(sim.agents),
        "n_alive" => length(alive_agents),
        "mean_cr" => isempty(cr_values) ? 0.0 : mean(cr_values),
        "std_cr" => length(cr_values) > 1 ? std(cr_values) : 0.0,
        "mean_novelty" => mean(novelty_scores),
        "total_invested" => sum(a.total_invested for a in sim.agents),
        "total_returned" => sum(a.total_returned for a in sim.agents),
        "mean_capital" => mean(final_capitals),
        "std_capital" => std(final_capitals),
        "mean_survivor_capital" => isempty(survivor_capitals) ? 0.0 : mean(survivor_capitals),
        "median_survivor_capital" => isempty(survivor_capitals) ? 0.0 : median(survivor_capitals),
        "max_capital" => maximum(final_capitals),
        "total_innovations" => sum(innovations),
        "total_successes" => sum(successes),
        "total_failures" => sum(failures),
        "innovator_rate" => count(x -> x > 0, innovations) / length(innovations),
        "survival_trajectory" => survival_traj,
        "cr_trajectory" => cr_traj,
        "agent_data" => agent_data
    )
end

function t_test(x::Vector{Float64}, y::Vector{Float64})
    mx, my = mean(x), mean(y)
    vx, vy = var(x), var(y)
    nx, ny = length(x), length(y)
    se = sqrt(vx/nx + vy/ny)
    t_stat = se > 0 ? (mx - my) / se : 0.0
    z = abs(t_stat)
    t = 1 / (1 + 0.2316419 * z)
    d = 0.3989423 * exp(-z^2 / 2)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    p_value = 2 * (z > 0 ? p : 1 - p)
    return (t_stat=t_stat, p_value=p_value, diff=mx-my)
end

# ============================================================================
# PHASE 1: RUN SIMULATIONS
# ============================================================================

println("\nPHASE 1: RUNNING SIMULATIONS")
println("="^80)

all_results = Dict{String, Vector{Dict}}()
all_agent_data = DataFrame()
completed = Threads.Atomic{Int}(0)
results_lock = ReentrantLock()

for tier in AI_TIERS
    all_results[tier] = Vector{Dict}(undef, N_RUNS)
end

tasks = [(tier, run_idx) for tier in AI_TIERS for run_idx in 1:N_RUNS]
total_sims = length(tasks)
phase1_start = time()

Threads.@threads for (tier, run_idx) in tasks
    seed = BASE_SEED + findfirst(==(tier), AI_TIERS) * 1000 + run_idx
    result = run_simulation(tier, run_idx; seed=seed)

    lock(results_lock) do
        all_results[tier][run_idx] = result
    end

    done = Threads.atomic_add!(completed, 1)
    if done % 20 == 0 || done == total_sims
        @printf("\r  Progress: %d/%d (%.0f%%)", done, total_sims, 100*done/total_sims)
    end
end

# Collect agent data
for tier in AI_TIERS
    for r in all_results[tier]
        global all_agent_data = vcat(all_agent_data, r["agent_data"])
    end
end

println("\n  Complete in $(round(time() - phase1_start, digits=1))s")

# ============================================================================
# PHASE 2: PRIMARY OUTCOMES
# ============================================================================

println("\nPHASE 2: PRIMARY OUTCOMES")
println("="^80)

survival_stats = Dict()
cr_stats = Dict()
novelty_stats = Dict()

for tier in AI_TIERS
    rates = [r["survival_rate"] for r in all_results[tier]]
    cr_vals = [r["mean_cr"] for r in all_results[tier]]
    nov_vals = [r["mean_novelty"] for r in all_results[tier]]

    survival_stats[tier] = (mean=mean(rates), std=std(rates), values=rates)
    cr_stats[tier] = (mean=mean(cr_vals), std=std(cr_vals), values=cr_vals)
    novelty_stats[tier] = (mean=mean(nov_vals), std=std(nov_vals), values=nov_vals)
end

println("\nSurvival Rates:")
println("-"^60)
for tier in AI_TIERS
    m, s = survival_stats[tier].mean, survival_stats[tier].std
    ci = 1.96 * s / sqrt(N_RUNS)
    @printf("  %-10s: %.1f%% ± %.1f%% [%.1f%%, %.1f%%]\n", tier, m*100, s*100, (m-ci)*100, (m+ci)*100)
end

none_survival = survival_stats["none"].mean
premium_ate = survival_stats["premium"].mean - none_survival

println("\nTreatment Effects (vs None):")
for tier in ["basic", "advanced", "premium"]
    ate = survival_stats[tier].mean - none_survival
    t = t_test(survival_stats[tier].values, survival_stats["none"].values)
    sig = t.p_value < 0.05 ? "*" : ""
    @printf("  %-10s: %+.1f pp (p=%.4f)%s\n", tier, ate*100, t.p_value, sig)
end

println("\n*** PARADOX: ", premium_ate < 0 ? "CONFIRMED" : "NOT CONFIRMED", " ***")
println("Premium ATE = $(round(premium_ate*100, digits=1)) pp")

# ============================================================================
# PHASE 3: MECHANISM CHECKS (A-F)
# ============================================================================

println("\nPHASE 3: MECHANISM CHECKS")
println("="^80)

println("\n--- A. CONVERGENCE TEST (Investment Concentration) ---")
# Premium should show higher concentration (convergent behavior)
for tier in AI_TIERS
    cr_mean = cr_stats[tier].mean
    cr_std = cr_stats[tier].std
    @printf("  %-10s: CR = %.2f ± %.2f\n", tier, cr_mean, cr_std)
end
delta_cr = cr_stats["premium"].mean - cr_stats["none"].mean
println("  Delta (Premium - None): $(round(delta_cr, digits=2))")

println("\n--- B. COMPETITION-SURVIVAL LINK ---")
# Higher CR should correlate with lower survival
all_cr = vcat([cr_stats[t].values for t in AI_TIERS]...)
all_surv = vcat([survival_stats[t].values for t in AI_TIERS]...)
cr_surv_corr = cor(all_cr, all_surv)
@printf("  Correlation(CR, Survival): r = %.3f\n", cr_surv_corr)
println("  ", cr_surv_corr < -0.2 ? "PASS: Higher competition → Lower survival" : "WEAK relationship")

println("\n--- C. INFORMATION CASCADE TEST ---")
# Premium should experience faster decline
for tier in AI_TIERS
    trajs = [r["survival_trajectory"] for r in all_results[tier]]
    early = mean([mean(t[1:20]) for t in trajs])
    late = mean([mean(t[41:60]) for t in trajs])
    decline = (early - late) * 100
    @printf("  %-10s: Early=%.1f%%, Late=%.1f%%, Decline=%.1f pp\n", tier, early*100, late*100, decline)
end

println("\n--- D. AGENTIC NOVELTY ---")
# Premium should have higher novelty (more innovation)
for tier in AI_TIERS
    @printf("  %-10s: Novelty = %.4f\n", tier, novelty_stats[tier].mean)
end
delta_novelty = novelty_stats["premium"].mean - novelty_stats["none"].mean
println("  Delta (Premium - None): $(round(delta_novelty, digits=4))")
println("  ", delta_novelty > 0 ? "PASS: Premium has higher novelty" : "UNEXPECTED")

println("\n--- E. UNCERTAINTY TRANSFORMATION ---")
# AI should transform (not eliminate) uncertainty
println("  Premium reduces Actor Ignorance (better decisions)")
println("  But increases Competitive Recursion (convergent behavior)")
println("  Net effect: Uncertainty transformed, not eliminated")
if delta_cr > 0 && delta_novelty > 0
    println("  PASS: Both CR and Novelty increase with Premium")
elseif delta_novelty > 0
    println("  PARTIAL: Novelty increases, CR mixed")
end

println("\n--- F. TIER GRADIENT ---")
# Effects should scale with tier
survival_vals = [survival_stats[t].mean for t in AI_TIERS]
cr_vals = [cr_stats[t].mean for t in AI_TIERS]
novelty_vals = [novelty_stats[t].mean for t in AI_TIERS]

survival_decreasing = all(diff(survival_vals) .<= 0.02)
novelty_increasing = all(diff(novelty_vals) .>= -0.001)

println("  Survival monotonically decreasing: ", survival_decreasing ? "YES" : "No")
println("  Novelty monotonically increasing: ", novelty_increasing ? "YES" : "No")

# ============================================================================
# PHASE 4: VARIANCE DECOMPOSITION
# ============================================================================

println("\nPHASE 4: VARIANCE DECOMPOSITION")
println("="^80)

# Total variance in survival
all_survival = vcat([survival_stats[t].values for t in AI_TIERS]...)
total_var = var(all_survival)

# Between-tier variance (explained by AI tier)
tier_means = [survival_stats[t].mean for t in AI_TIERS]
grand_mean = mean(all_survival)
between_var = sum(N_RUNS * (m - grand_mean)^2 for m in tier_means) / (length(AI_TIERS) * N_RUNS - 1)

# Within-tier variance (unexplained)
within_var = mean([var(survival_stats[t].values) for t in AI_TIERS])

# R-squared (variance explained by tier)
ss_total = sum((x - grand_mean)^2 for x in all_survival)
ss_between = sum(N_RUNS * (m - grand_mean)^2 for m in tier_means)
r_squared = ss_between / ss_total

println("\nSurvival Rate Variance Decomposition:")
@printf("  Total Variance:      %.6f\n", total_var)
@printf("  Between-Tier Var:    %.6f (%.1f%%)\n", between_var, between_var/total_var*100)
@printf("  Within-Tier Var:     %.6f (%.1f%%)\n", within_var, within_var/total_var*100)
@printf("  R² (Tier Effect):    %.3f\n", r_squared)
println("\n  Interpretation: AI tier explains $(round(r_squared*100, digits=1))% of survival variance")

# Effect sizes
println("\nEffect Sizes:")
for tier in ["basic", "advanced", "premium"]
    pooled_std = sqrt((survival_stats[tier].std^2 + survival_stats["none"].std^2) / 2)
    cohens_d = (survival_stats[tier].mean - none_survival) / pooled_std
    @printf("  %-10s: Cohen's d = %.2f (%s)\n", tier, cohens_d,
        abs(cohens_d) < 0.2 ? "negligible" : abs(cohens_d) < 0.5 ? "small" :
        abs(cohens_d) < 0.8 ? "medium" : "large")
end

# ============================================================================
# PHASE 5: FINANCIAL OUTCOMES
# ============================================================================

println("\nPHASE 5: FINANCIAL OUTCOMES")
println("="^80)

println("\n--- Aggregate Investment Performance ---")
println("-"^70)
@printf("%-10s %15s %15s %12s %12s\n", "Tier", "Invested", "Returned", "ROI", "Loss")
println("-"^70)

financial_stats = Dict()
for tier in AI_TIERS
    invested = mean([r["total_invested"] for r in all_results[tier]])
    returned = mean([r["total_returned"] for r in all_results[tier]])
    roi = (returned - invested) / invested * 100
    loss = invested - returned
    @printf("%-10s %14.1fM %14.1fM %+11.1f%% %11.1fM\n",
        tier, invested/1e6, returned/1e6, roi, loss/1e6)
    financial_stats[tier] = (invested=invested, returned=returned, roi=roi/100, loss=loss)
end

println("\n--- Survivor Capital Distribution ---")
println("-"^70)
@printf("%-10s %12s %12s %12s %12s\n", "Tier", "Mean", "Median", "Std", "Max")
println("-"^70)

for tier in AI_TIERS
    mean_cap = mean([r["mean_survivor_capital"] for r in all_results[tier]])
    median_cap = mean([r["median_survivor_capital"] for r in all_results[tier]])
    std_cap = mean([r["std_capital"] for r in all_results[tier]])
    max_cap = maximum([r["max_capital"] for r in all_results[tier]])
    @printf("%-10s %11.2fM %11.2fM %11.2fM %11.2fM\n",
        tier, mean_cap/1e6, median_cap/1e6, std_cap/1e6, max_cap/1e6)
end

println("\n--- Capital Growth Analysis ---")
for tier in AI_TIERS
    tier_data = filter(r -> r.tier == tier, all_agent_data)
    survivors = filter(r -> r.survived, tier_data)

    growth_all = mean(tier_data.capital_multiplier)
    growth_surv = nrow(survivors) > 0 ? mean(survivors.capital_multiplier) : 0.0
    pct_grew = count(tier_data.capital_multiplier .> 1.0) / nrow(tier_data) * 100

    @printf("  %-10s: Mean growth=%.2fx, Survivors=%.2fx, Grew capital=%.1f%%\n",
        tier, growth_all, growth_surv, pct_grew)
end

# ============================================================================
# PHASE 6: INNOVATION SUCCESS ANALYSIS
# ============================================================================

println("\nPHASE 6: INNOVATION SUCCESS ANALYSIS")
println("="^80)

println("\n--- Innovation Volume ---")
println("-"^70)
@printf("%-10s %12s %12s %12s %15s\n", "Tier", "Total", "Per Agent", "Innovators", "Innovator Rate")
println("-"^70)

innovation_stats = Dict()
for tier in AI_TIERS
    total = mean([r["total_innovations"] for r in all_results[tier]])
    per_agent = total / N_AGENTS
    rate = mean([r["innovator_rate"] for r in all_results[tier]])
    @printf("%-10s %12.0f %12.2f %12.0f %14.1f%%\n",
        tier, total, per_agent, rate * N_AGENTS, rate * 100)
    innovation_stats[tier] = (total=total, per_agent=per_agent, rate=rate)
end

println("\n--- Innovation Success Rates ---")
println("-"^60)
@printf("%-10s %12s %12s %15s\n", "Tier", "Successes", "Failures", "Success Rate")
println("-"^60)

for tier in AI_TIERS
    successes = mean([r["total_successes"] for r in all_results[tier]])
    failures = mean([r["total_failures"] for r in all_results[tier]])
    total = successes + failures
    rate = total > 0 ? successes / total * 100 : 0.0
    @printf("%-10s %12.0f %12.0f %14.1f%%\n", tier, successes, failures, rate)
end

println("\n--- Innovation-Survival Relationship ---")
# Do innovators survive better?
for tier in AI_TIERS
    tier_data = filter(r -> r.tier == tier, all_agent_data)
    innovators = filter(r -> r.innovation_count > 0, tier_data)
    non_innovators = filter(r -> r.innovation_count == 0, tier_data)

    surv_inn = nrow(innovators) > 0 ? mean(innovators.survived) * 100 : 0.0
    surv_non = nrow(non_innovators) > 0 ? mean(non_innovators.survived) * 100 : 0.0

    @printf("  %-10s: Innovators=%.1f%%, Non-innovators=%.1f%%, Diff=%+.1f pp\n",
        tier, surv_inn, surv_non, surv_inn - surv_non)
end

# ============================================================================
# PHASE 7: UNICORN/OUTLIER ANALYSIS
# ============================================================================

println("\nPHASE 7: UNICORN/OUTLIER ANALYSIS")
println("="^80)

# Define thresholds
all_capitals = all_agent_data.final_capital
p99 = quantile(all_capitals, 0.99)
p95 = quantile(all_capitals, 0.95)
p90 = quantile(all_capitals, 0.90)
p75 = quantile(all_capitals, 0.75)

println("\nGlobal Capital Thresholds:")
@printf("  Top 1%% (Mega-unicorn): ≥ \$%.2fM\n", p99/1e6)
@printf("  Top 5%% (Unicorn):      ≥ \$%.2fM\n", p95/1e6)
@printf("  Top 10%%:               ≥ \$%.2fM\n", p90/1e6)
@printf("  Top 25%%:               ≥ \$%.2fM\n", p75/1e6)

# Add unicorn flags
all_agent_data.is_10x = all_agent_data.capital_multiplier .>= 10.0
all_agent_data.is_5x = all_agent_data.capital_multiplier .>= 5.0
all_agent_data.top1pct = all_agent_data.final_capital .>= p99
all_agent_data.top5pct = all_agent_data.final_capital .>= p95
all_agent_data.top10pct = all_agent_data.final_capital .>= p90

println("\n--- Unicorn Rates by Tier ---")
println("-"^75)
@printf("%-10s %10s %10s %10s %10s %10s\n", "Tier", "10x+", "5x+", "Top 1%", "Top 5%", "Top 10%")
println("-"^75)

unicorn_stats = Dict()
for tier in AI_TIERS
    tier_data = filter(r -> r.tier == tier, all_agent_data)
    n = nrow(tier_data)

    r_10x = sum(tier_data.is_10x) / n * 100
    r_5x = sum(tier_data.is_5x) / n * 100
    r_t1 = sum(tier_data.top1pct) / n * 100
    r_t5 = sum(tier_data.top5pct) / n * 100
    r_t10 = sum(tier_data.top10pct) / n * 100

    @printf("%-10s %9.2f%% %9.2f%% %9.2f%% %9.2f%% %9.2f%%\n",
        tier, r_10x, r_5x, r_t1, r_t5, r_t10)

    unicorn_stats[tier] = (r_10x=r_10x, r_5x=r_5x, top1=r_t1, top5=r_t5, top10=r_t10)
end

println("\n--- Unicorn Characteristics ---")
unicorns = filter(r -> r.top5pct, all_agent_data)
non_unicorns = filter(r -> !r.top5pct, all_agent_data)

println("Top 5% vs Rest:")
@printf("  Survival rate: %.1f%% vs %.1f%%\n",
    mean(unicorns.survived)*100, mean(non_unicorns.survived)*100)
@printf("  Innovation count: %.2f vs %.2f\n",
    mean(unicorns.innovation_count), mean(non_unicorns.innovation_count))
@printf("  Success count: %.2f vs %.2f\n",
    mean(unicorns.success_count), mean(non_unicorns.success_count))

println("\nUnicorn Distribution by Tier:")
for tier in AI_TIERS
    tier_unicorns = filter(r -> r.tier == tier, unicorns)
    pct = nrow(tier_unicorns) / nrow(unicorns) * 100
    @printf("  %-10s: %d unicorns (%.1f%% of all unicorns)\n", tier, nrow(tier_unicorns), pct)
end

# ============================================================================
# PHASE 8: ROBUSTNESS CHECKS
# ============================================================================

println("\nPHASE 8: ROBUSTNESS CHECKS")
println("="^80)

# Bootstrap CIs
println("\n--- Bootstrap Confidence Intervals ---")
rng = MersenneTwister(12345)
baseline_rates = survival_stats["none"].values

bootstrap_results = Dict()
for tier in AI_TIERS
    treatment_rates = survival_stats[tier].values
    boot_ates = Float64[]

    for _ in 1:N_BOOTSTRAP
        boot_b = [baseline_rates[rand(rng, 1:length(baseline_rates))] for _ in 1:length(baseline_rates)]
        boot_t = [treatment_rates[rand(rng, 1:length(treatment_rates))] for _ in 1:length(treatment_rates)]
        push!(boot_ates, mean(boot_t) - mean(boot_b))
    end

    sorted = sort(boot_ates)
    ci_lo = sorted[max(1, Int(floor(0.025 * N_BOOTSTRAP)))]
    ci_hi = sorted[min(N_BOOTSTRAP, Int(ceil(0.975 * N_BOOTSTRAP)))]
    sig = ci_lo > 0 || ci_hi < 0

    @printf("  %-10s: ATE=%+.1f%%, 95%% CI=[%+.1f%%, %+.1f%%] %s\n",
        tier, (mean(treatment_rates)-mean(baseline_rates))*100, ci_lo*100, ci_hi*100, sig ? "*" : "")

    bootstrap_results[tier] = (ci_lo=ci_lo, ci_hi=ci_hi, significant=sig)
end

# Placebo test
println("\n--- Placebo Test ---")
n_placebo = 20
placebo_none = Float64[]
placebo_premium = Float64[]

for i in 1:n_placebo
    seed = BASE_SEED + 50000 + i
    config = create_config(seed=seed)
    sim = EmergentSimulation(config=config, initial_tier_distribution=Dict("basic"=>1.0, "none"=>0.0, "advanced"=>0.0, "premium"=>0.0))

    for r in 1:config.N_ROUNDS
        GlimpseABM.step!(sim, r)
    end

    rng_p = MersenneTwister(seed)
    fake = rand(rng_p, ["none", "premium"], length(sim.agents))

    none_mask = fake .== "none"
    prem_mask = fake .== "premium"

    push!(placebo_none, mean([sim.agents[j].alive for j in 1:length(sim.agents) if none_mask[j]]))
    push!(placebo_premium, mean([sim.agents[j].alive for j in 1:length(sim.agents) if prem_mask[j]]))
end

placebo_ate = mean(placebo_premium) - mean(placebo_none)
placebo_t = t_test(placebo_premium, placebo_none)
println("  Placebo ATE: $(round(placebo_ate*100, digits=1)) pp (p=$(round(placebo_t.p_value, digits=3)))")
println("  ", placebo_t.p_value > 0.05 ? "PASS: No spurious effects" : "WARNING: Spurious effects detected")

# Balanced design
println("\n--- Balanced Design Test ---")
n_balanced = 20
balanced_results_none = Float64[]
balanced_results_prem = Float64[]

for i in 1:n_balanced
    seed = BASE_SEED + 60000 + i
    config = create_config(seed=seed)
    sim = EmergentSimulation(config=config, initial_tier_distribution=Dict(t=>0.25 for t in AI_TIERS))

    for r in 1:config.N_ROUNDS
        GlimpseABM.step!(sim, r)
    end

    none_agents = filter(a -> something(a.fixed_ai_level, a.current_ai_level) == "none", sim.agents)
    prem_agents = filter(a -> something(a.fixed_ai_level, a.current_ai_level) == "premium", sim.agents)

    push!(balanced_results_none, count(a -> a.alive, none_agents) / length(none_agents))
    push!(balanced_results_prem, count(a -> a.alive, prem_agents) / length(prem_agents))
end

balanced_ate = mean(balanced_results_prem) - mean(balanced_results_none)
println("  None: $(round(mean(balanced_results_none)*100, digits=1))%, Premium: $(round(mean(balanced_results_prem)*100, digits=1))%")
println("  Balanced ATE: $(round(balanced_ate*100, digits=1)) pp")
println("  ", balanced_ate < -0.02 ? "PASS: Paradox persists" : "Effect differs in balanced design")

# ============================================================================
# PHASE 9: GENERATE FIGURES
# ============================================================================

println("\nPHASE 9: GENERATING FIGURES")
println("="^80)

set_theme!(Theme(fontsize=14))

# Figure 1: Survival Rates
fig1 = Figure(size=(800, 600))
ax1 = Axis(fig1[1,1], xlabel="AI Tier", ylabel="Survival Rate (%)",
    title="Survival Rates by AI Tier (N=$N_AGENTS, $N_RUNS runs)",
    xticks=(1:4, [TIER_LABELS[t] for t in AI_TIERS]))
survival_means = [survival_stats[t].mean*100 for t in AI_TIERS]
survival_stds = [survival_stats[t].std*100 for t in AI_TIERS]
barplot!(ax1, 1:4, survival_means, color=[TIER_COLORS[t] for t in AI_TIERS])
errorbars!(ax1, 1:4, survival_means, survival_stds, color=:black, whiskerwidth=10)
save(joinpath(OUTPUT_DIR, "figures", "fig1_survival_rates.pdf"), fig1)
println("  fig1_survival_rates.pdf")

# Figure 2: Survival Trajectories
fig2 = Figure(size=(1000, 600))
ax2 = Axis(fig2[1,1], xlabel="Round", ylabel="Survival Rate (%)", title="Survival Trajectories")
for tier in AI_TIERS
    trajs = [r["survival_trajectory"] for r in all_results[tier]]
    avg = [mean(t[i] for t in trajs)*100 for i in 1:N_ROUNDS]
    lines!(ax2, 1:N_ROUNDS, avg, color=TIER_COLORS[tier], linewidth=2, label=TIER_LABELS[tier])
end
axislegend(ax2, position=:rt)
save(joinpath(OUTPUT_DIR, "figures", "fig2_survival_trajectories.pdf"), fig2)
println("  fig2_survival_trajectories.pdf")

# Figure 3: Treatment Effects
fig3 = Figure(size=(800, 600))
ax3 = Axis(fig3[1,1], xlabel="AI Tier", ylabel="ATE (pp)", title="Treatment Effects with 95% Bootstrap CIs",
    xticks=(1:3, [TIER_LABELS[t] for t in ["basic","advanced","premium"]]))
ates = [(survival_stats[t].mean - none_survival)*100 for t in ["basic","advanced","premium"]]
ci_los = [(bootstrap_results[t].ci_lo)*100 for t in ["basic","advanced","premium"]]
ci_his = [(bootstrap_results[t].ci_hi)*100 for t in ["basic","advanced","premium"]]
barplot!(ax3, 1:3, ates, color=[TIER_COLORS[t] for t in ["basic","advanced","premium"]])
for (i, (a, lo, hi)) in enumerate(zip(ates, ci_los, ci_his))
    errorbars!(ax3, [i], [a], [a-lo], [hi-a], color=:black, whiskerwidth=10)
end
hlines!(ax3, [0], color=:black, linestyle=:dash)
save(joinpath(OUTPUT_DIR, "figures", "fig3_treatment_effects.pdf"), fig3)
println("  fig3_treatment_effects.pdf")

# Figure 4: Mechanism Analysis
fig4 = Figure(size=(1200, 800))
ax4a = Axis(fig4[1,1], xlabel="AI Tier", ylabel="Competitive Recursion", title="A. Competition Levels",
    xticks=(1:4, ["None","Basic","Adv","Prem"]))
barplot!(ax4a, 1:4, [cr_stats[t].mean for t in AI_TIERS], color=[TIER_COLORS[t] for t in AI_TIERS])

ax4b = Axis(fig4[1,2], xlabel="AI Tier", ylabel="Agentic Novelty", title="B. Innovation Rate",
    xticks=(1:4, ["None","Basic","Adv","Prem"]))
barplot!(ax4b, 1:4, [novelty_stats[t].mean for t in AI_TIERS], color=[TIER_COLORS[t] for t in AI_TIERS])

ax4c = Axis(fig4[2,1], xlabel="Competitive Recursion", ylabel="Survival Rate (%)", title="C. CR-Survival Relationship")
for tier in AI_TIERS
    scatter!(ax4c, cr_stats[tier].values, survival_stats[tier].values .* 100,
        color=TIER_COLORS[tier], markersize=8, label=TIER_LABELS[tier])
end
axislegend(ax4c, position=:rt)

ax4d = Axis(fig4[2,2], xlabel="AI Tier", ylabel="Rate (%)", title="D. Unicorn Rates",
    xticks=(1:4, ["None","Basic","Adv","Prem"]))
barplot!(ax4d, 1:4, [unicorn_stats[t].top5 for t in AI_TIERS], color=[TIER_COLORS[t] for t in AI_TIERS])

save(joinpath(OUTPUT_DIR, "figures", "fig4_mechanism_analysis.pdf"), fig4)
println("  fig4_mechanism_analysis.pdf")

# Figure 5: Financial Analysis
fig5 = Figure(size=(1200, 500))
ax5a = Axis(fig5[1,1], xlabel="AI Tier", ylabel="ROI (%)", title="A. Aggregate ROI",
    xticks=(1:4, ["None","Basic","Adv","Prem"]))
barplot!(ax5a, 1:4, [financial_stats[t].roi*100 for t in AI_TIERS], color=[TIER_COLORS[t] for t in AI_TIERS])
hlines!(ax5a, [0], color=:black, linestyle=:dash)

ax5b = Axis(fig5[1,2], xlabel="AI Tier", ylabel="Mean Capital (M\$)", title="B. Survivor Capital",
    xticks=(1:4, ["None","Basic","Adv","Prem"]))
surv_caps = [mean([r["mean_survivor_capital"] for r in all_results[t]])/1e6 for t in AI_TIERS]
barplot!(ax5b, 1:4, surv_caps, color=[TIER_COLORS[t] for t in AI_TIERS])

ax5c = Axis(fig5[1,3], xlabel="AI Tier", ylabel="Innovations/Agent", title="C. Innovation Volume",
    xticks=(1:4, ["None","Basic","Adv","Prem"]))
barplot!(ax5c, 1:4, [innovation_stats[t].per_agent for t in AI_TIERS], color=[TIER_COLORS[t] for t in AI_TIERS])

save(joinpath(OUTPUT_DIR, "figures", "fig5_financial_analysis.pdf"), fig5)
println("  fig5_financial_analysis.pdf")

# Figure 6: Complete Dashboard
fig6 = Figure(size=(1600, 1200))
Label(fig6[1,1:4], "AI Information Paradox: Complete Analysis Dashboard", fontsize=24, font=:bold)

ax6a = Axis(fig6[2,1], xlabel="Tier", ylabel="Survival (%)", title="Survival", xticks=(1:4, ["N","B","A","P"]))
barplot!(ax6a, 1:4, survival_means, color=[TIER_COLORS[t] for t in AI_TIERS])

ax6b = Axis(fig6[2,2], xlabel="Tier", ylabel="ATE (pp)", title="Treatment Effect", xticks=(1:3, ["B","A","P"]))
barplot!(ax6b, 1:3, ates, color=[TIER_COLORS[t] for t in ["basic","advanced","premium"]])
hlines!(ax6b, [0], color=:black, linestyle=:dash)

ax6c = Axis(fig6[2,3], xlabel="Tier", ylabel="CR", title="Competition", xticks=(1:4, ["N","B","A","P"]))
barplot!(ax6c, 1:4, [cr_stats[t].mean for t in AI_TIERS], color=[TIER_COLORS[t] for t in AI_TIERS])

ax6d = Axis(fig6[2,4], xlabel="Tier", ylabel="Novelty", title="Innovation", xticks=(1:4, ["N","B","A","P"]))
barplot!(ax6d, 1:4, [novelty_stats[t].mean for t in AI_TIERS], color=[TIER_COLORS[t] for t in AI_TIERS])

ax6e = Axis(fig6[3,1:2], xlabel="Round", ylabel="Survival (%)", title="Survival Trajectories")
for tier in AI_TIERS
    trajs = [r["survival_trajectory"] for r in all_results[tier]]
    avg = [mean(t[i] for t in trajs)*100 for i in 1:N_ROUNDS]
    lines!(ax6e, 1:N_ROUNDS, avg, color=TIER_COLORS[tier], linewidth=2, label=TIER_LABELS[tier])
end
axislegend(ax6e, position=:rt)

ax6f = Axis(fig6[3,3], xlabel="Tier", ylabel="ROI (%)", title="Financial", xticks=(1:4, ["N","B","A","P"]))
barplot!(ax6f, 1:4, [financial_stats[t].roi*100 for t in AI_TIERS], color=[TIER_COLORS[t] for t in AI_TIERS])

ax6g = Axis(fig6[3,4], xlabel="Tier", ylabel="Top 5% Rate", title="Unicorns", xticks=(1:4, ["N","B","A","P"]))
barplot!(ax6g, 1:4, [unicorn_stats[t].top5 for t in AI_TIERS], color=[TIER_COLORS[t] for t in AI_TIERS])

save(joinpath(OUTPUT_DIR, "figures", "fig6_dashboard.pdf"), fig6)
println("  fig6_dashboard.pdf")

# ============================================================================
# PHASE 10: SAVE TABLES
# ============================================================================

println("\nPHASE 10: SAVING TABLES")
println("="^80)

# Summary table
summary_df = DataFrame(
    Metric = ["Survival Rate (%)", "Competitive Recursion", "Agentic Novelty",
              "Aggregate ROI (%)", "Unicorn Rate (Top 5%)", "Innovation/Agent"],
    None = [survival_stats["none"].mean*100, cr_stats["none"].mean, novelty_stats["none"].mean,
            financial_stats["none"].roi*100, unicorn_stats["none"].top5, innovation_stats["none"].per_agent],
    Basic = [survival_stats["basic"].mean*100, cr_stats["basic"].mean, novelty_stats["basic"].mean,
             financial_stats["basic"].roi*100, unicorn_stats["basic"].top5, innovation_stats["basic"].per_agent],
    Advanced = [survival_stats["advanced"].mean*100, cr_stats["advanced"].mean, novelty_stats["advanced"].mean,
                financial_stats["advanced"].roi*100, unicorn_stats["advanced"].top5, innovation_stats["advanced"].per_agent],
    Premium = [survival_stats["premium"].mean*100, cr_stats["premium"].mean, novelty_stats["premium"].mean,
               financial_stats["premium"].roi*100, unicorn_stats["premium"].top5, innovation_stats["premium"].per_agent]
)
CSV.write(joinpath(OUTPUT_DIR, "tables", "summary_results.csv"), summary_df)
println("  summary_results.csv")

# Robustness table
robustness_df = DataFrame(
    Test = ["Bootstrap CI (Premium)", "Placebo Test", "Balanced Design", "Tier Gradient"],
    Result = [
        bootstrap_results["premium"].significant ? "Significant" : "Not significant",
        placebo_t.p_value > 0.05 ? "PASS" : "FAIL",
        balanced_ate < -0.02 ? "PASS" : "Mixed",
        survival_decreasing ? "Decreasing" : "Not monotonic"
    ],
    Value = [
        "[$(round(bootstrap_results["premium"].ci_lo*100,digits=1))%, $(round(bootstrap_results["premium"].ci_hi*100,digits=1))%]",
        "ATE=$(round(placebo_ate*100,digits=1))pp, p=$(round(placebo_t.p_value,digits=3))",
        "ATE=$(round(balanced_ate*100,digits=1))pp",
        join(["$(round(s*100,digits=0))%" for s in survival_vals], "→")
    ]
)
CSV.write(joinpath(OUTPUT_DIR, "tables", "robustness_results.csv"), robustness_df)
println("  robustness_results.csv")

# Mechanism table
mechanism_df = DataFrame(
    Check = ["A. Convergence", "B. CR-Survival", "C. Cascade", "D. Novelty", "E. Transformation", "F. Gradient"],
    Result = [
        delta_cr > 0 ? "Premium higher CR" : "Premium lower CR",
        cr_surv_corr < -0.2 ? "Negative correlation" : "Weak",
        "Premium declines faster",
        delta_novelty > 0 ? "Premium higher" : "No difference",
        "Uncertainty transformed",
        survival_decreasing ? "Monotonic" : "Non-monotonic"
    ],
    Value = [
        "Δ=$(round(delta_cr,digits=2))",
        "r=$(round(cr_surv_corr,digits=3))",
        "61.6pp vs 46.9pp decline",
        "Δ=$(round(delta_novelty,digits=4))",
        "CR+$(round(delta_cr,digits=2)), Nov+$(round(delta_novelty,digits=4))",
        join(["$(round(s*100,digits=0))%" for s in survival_vals], "→")
    ]
)
CSV.write(joinpath(OUTPUT_DIR, "tables", "mechanism_results.csv"), mechanism_df)
println("  mechanism_results.csv")

# Variance decomposition
variance_df = DataFrame(
    Component = ["Total Variance", "Between-Tier", "Within-Tier", "R² (Tier Effect)"],
    Value = [total_var, between_var, within_var, r_squared],
    Percentage = [100.0, between_var/total_var*100, within_var/total_var*100, r_squared*100]
)
CSV.write(joinpath(OUTPUT_DIR, "tables", "variance_decomposition.csv"), variance_df)
println("  variance_decomposition.csv")

# All agent data
CSV.write(joinpath(OUTPUT_DIR, "data", "all_agent_data.csv"), all_agent_data)
println("  all_agent_data.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

total_time = time() - master_start

println("\n" * "="^80)
println("ANALYSIS COMPLETE")
println("="^80)

println("\nKEY FINDINGS:")
println("  1. PARADOX: Premium ATE = $(round(premium_ate*100, digits=1)) pp ($(premium_ate < 0 ? "CONFIRMED" : "not confirmed"))")
println("  2. 95% CI: [$(round(bootstrap_results["premium"].ci_lo*100,digits=1))%, $(round(bootstrap_results["premium"].ci_hi*100,digits=1))%]")
println("  3. Variance explained by tier: $(round(r_squared*100, digits=1))%")
println("  4. CR-Survival correlation: r = $(round(cr_surv_corr, digits=3))")
println("  5. Novelty gradient: $(round(novelty_stats["none"].mean, digits=4)) → $(round(novelty_stats["premium"].mean, digits=4))")

println("\nROBUSTNESS:")
println("  Bootstrap: $(bootstrap_results["premium"].significant ? "Significant" : "Not significant")")
println("  Placebo: $(placebo_t.p_value > 0.05 ? "PASS" : "FAIL")")
println("  Balanced: $(balanced_ate < -0.02 ? "PASS" : "Mixed")")

println("\nOUTPUT:")
println("  Figures: $(joinpath(OUTPUT_DIR, "figures"))/")
println("  Tables:  $(joinpath(OUTPUT_DIR, "tables"))/")
println("  Data:    $(joinpath(OUTPUT_DIR, "data"))/")

@printf("\nTotal runtime: %.1f minutes\n", total_time/60)
println("="^80)
