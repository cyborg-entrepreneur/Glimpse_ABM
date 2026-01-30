#!/usr/bin/env julia
"""
MECHANISM ANALYSIS FOR AI INFORMATION PARADOX
==============================================

Examines the causal pathways explaining WHY Premium AI leads to lower survival:

1. Behavioral Mechanism: How AI changes agent decision-making
   - Action distribution shifts (explore → innovate)
   - Risk-taking behavior changes

2. Competition Mechanism: How AI creates crowding effects
   - Herding on similar opportunities
   - Competitive recursion levels

3. Financial Mechanism: How AI affects capital dynamics
   - Subscription costs draining capital
   - Investment patterns and returns

4. Innovation Mechanism: How AI affects creative output
   - Niche discovery rates
   - Innovation success/failure patterns

5. Mediation Analysis: Testing causal pathways
   - Does innovation activity mediate the survival effect?
   - Does competition mediate the survival effect?

Configuration: 1000 agents × 60 rounds × 50 runs per tier (full runs)

Output: Single landscape PDF with mechanism analysis results

Usage:
    julia --threads=12 --project=. scripts/run_mechanism_analysis.jl
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
const N_RUNS = 50  # Full runs for mechanism analysis
const BASE_SEED = 20260128
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const N_BOOTSTRAP = 1000

const OUTPUT_DIR = joinpath(dirname(@__DIR__), "results",
    "mechanism_analysis_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")

mkpath(OUTPUT_DIR)

const TIER_COLORS = Dict(
    "none" => colorant"#6c757d",
    "basic" => colorant"#0d6efd",
    "advanced" => colorant"#fd7e14",
    "premium" => colorant"#dc3545"
)

const TIER_LABELS = Dict(
    "none" => "No AI",
    "basic" => "Basic AI",
    "advanced" => "Advanced AI",
    "premium" => "Premium AI"
)

const TIER_ORDER = ["none", "basic", "advanced", "premium"]

# ============================================================================
# PRINT HEADER
# ============================================================================

println("="^80)
println("MECHANISM ANALYSIS FOR AI INFORMATION PARADOX")
println("="^80)
println("Configuration:")
println("  Threads:     $(Threads.nthreads())")
println("  Agents:      $N_AGENTS per simulation")
println("  Rounds:      $N_ROUNDS (5 years)")
println("  Runs/Tier:   $N_RUNS (full runs)")
println("  Output:      $OUTPUT_DIR")
println("="^80)

master_start = time()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function run_mechanism_sim(tier::String, run_idx::Int, seed::Int)
    config = EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        INITIAL_CAPITAL=5_000_000.0,
        SURVIVAL_THRESHOLD=10_000.0
    )

    tier_dist = Dict(t => (t == tier ? 1.0 : 0.0) for t in AI_TIERS)
    sim = EmergentSimulation(config=config, initial_tier_distribution=tier_dist)

    # Track mechanisms over time
    survival_traj = Float64[]
    innovate_share_traj = Float64[]
    explore_share_traj = Float64[]
    competition_traj = Float64[]
    niches_traj = Int[]
    capital_traj = Float64[]

    # Per-round tracking
    for r in 1:N_ROUNDS
        GlimpseABM.step!(sim, r)

        alive = filter(a -> a.alive, sim.agents)
        push!(survival_traj, length(alive) / length(sim.agents))

        # Action shares this round
        action_counts = Dict("invest" => 0, "innovate" => 0, "explore" => 0, "maintain" => 0)
        for agent in sim.agents
            if length(agent.action_history) >= r
                act = agent.action_history[r]
                action_counts[act] = get(action_counts, act, 0) + 1
            end
        end
        total_actions = sum(values(action_counts))
        push!(innovate_share_traj, total_actions > 0 ? action_counts["innovate"] / total_actions : 0.0)
        push!(explore_share_traj, total_actions > 0 ? action_counts["explore"] / total_actions : 0.0)

        # Competition levels
        cr_vals = Float64[]
        for agent in sim.agents
            if !isempty(agent.uncertainty_metrics.competition_levels)
                push!(cr_vals, last(agent.uncertainty_metrics.competition_levels))
            end
        end
        push!(competition_traj, isempty(cr_vals) ? 0.0 : mean(cr_vals))

        # Cumulative niches
        push!(niches_traj, sum(a.uncertainty_metrics.niches_discovered for a in sim.agents))

        # Total capital
        push!(capital_traj, sum(GlimpseABM.get_capital(a) for a in sim.agents))
    end

    # Final metrics
    alive_agents = filter(a -> a.alive, sim.agents)

    # Behavioral metrics
    final_action_counts = Dict("invest" => 0, "innovate" => 0, "explore" => 0, "maintain" => 0)
    for agent in sim.agents
        for act in agent.action_history
            final_action_counts[act] = get(final_action_counts, act, 0) + 1
        end
    end
    total_final_actions = sum(values(final_action_counts))

    # Competition metrics
    all_competition = Float64[]
    for agent in sim.agents
        append!(all_competition, agent.uncertainty_metrics.competition_levels)
    end

    # Innovation metrics
    innovations = [a.innovation_count for a in sim.agents]
    successes = [a.success_count for a in sim.agents]
    failures = [a.failure_count for a in sim.agents]

    # Financial metrics
    total_invested = sum(a.total_invested for a in sim.agents)
    total_returned = sum(a.total_returned for a in sim.agents)
    final_capitals = [GlimpseABM.get_capital(a) for a in sim.agents]

    # Niche discovery
    final_niches = sum(a.uncertainty_metrics.niches_discovered for a in sim.agents)
    final_combinations = sum(a.uncertainty_metrics.new_combinations_created for a in sim.agents)

    # AI trust evolution (for AI tiers)
    ai_trust_final = [a.ai_trust for a in sim.agents]

    return Dict(
        "tier" => tier,
        "run_idx" => run_idx,
        "seed" => seed,
        # Survival
        "survival_rate" => length(alive_agents) / length(sim.agents),
        "survival_traj" => survival_traj,
        # Behavioral
        "innovate_share" => total_final_actions > 0 ? final_action_counts["innovate"] / total_final_actions : 0.0,
        "explore_share" => total_final_actions > 0 ? final_action_counts["explore"] / total_final_actions : 0.0,
        "invest_share" => total_final_actions > 0 ? final_action_counts["invest"] / total_final_actions : 0.0,
        "maintain_share" => total_final_actions > 0 ? final_action_counts["maintain"] / total_final_actions : 0.0,
        "innovate_share_traj" => innovate_share_traj,
        "explore_share_traj" => explore_share_traj,
        # Competition
        "mean_competition" => isempty(all_competition) ? 0.0 : mean(all_competition),
        "max_competition" => isempty(all_competition) ? 0.0 : maximum(all_competition),
        "competition_traj" => competition_traj,
        # Innovation
        "total_innovations" => sum(innovations),
        "innovations_per_agent" => mean(innovations),
        "success_rate" => sum(successes) / max(1, sum(successes) + sum(failures)),
        "final_niches" => final_niches,
        "final_combinations" => final_combinations,
        "niches_traj" => niches_traj,
        # Financial
        "total_invested" => total_invested,
        "total_returned" => total_returned,
        "roi" => (total_returned - total_invested) / total_invested,
        "mean_final_capital" => mean(final_capitals),
        "capital_traj" => capital_traj,
        # AI Trust
        "mean_ai_trust" => mean(ai_trust_final),
        "std_ai_trust" => std(ai_trust_final)
    )
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

# ============================================================================
# PHASE 1: RUN SIMULATIONS
# ============================================================================

println("\n" * "="^80)
println("PHASE 1: RUNNING MECHANISM SIMULATIONS")
println("="^80)

all_results = Dict{String, Vector{Dict}}()
for tier in AI_TIERS
    all_results[tier] = Vector{Dict}(undef, N_RUNS)
end

tasks = [(tier, run_idx) for tier in AI_TIERS for run_idx in 1:N_RUNS]
total_sims = length(tasks)
results_lock = ReentrantLock()
completed = Threads.Atomic{Int}(0)

tier_offset = Dict("none" => 0, "basic" => 10000, "advanced" => 20000, "premium" => 30000)

println("Running $total_sims simulations across $(Threads.nthreads()) threads...")
phase1_start = time()

Threads.@threads for (tier, run_idx) in tasks
    seed = BASE_SEED + tier_offset[tier] + run_idx
    result = run_mechanism_sim(tier, run_idx, seed)

    lock(results_lock) do
        all_results[tier][run_idx] = result
    end

    done = Threads.atomic_add!(completed, 1)
    if done % 20 == 0 || done == total_sims
        elapsed = time() - phase1_start
        rate = done / elapsed
        eta = (total_sims - done) / rate
        @printf("\r  Progress: %d/%d (%.0f%%) | %.1f sims/sec | ETA: %.0fs    ",
            done, total_sims, 100*done/total_sims, rate, eta)
    end
end

phase1_elapsed = time() - phase1_start
println("\n  Phase 1 complete in $(round(phase1_elapsed, digits=1))s")

# ============================================================================
# PHASE 2: COMPUTE MECHANISM STATISTICS
# ============================================================================

println("\n" * "="^80)
println("PHASE 2: COMPUTING MECHANISM STATISTICS")
println("="^80)

# Aggregate statistics by tier
mechanism_stats = Dict{String, Dict{String, NamedTuple}}()

for tier in TIER_ORDER
    mechanism_stats[tier] = Dict()

    # Survival
    surv_vals = [r["survival_rate"] for r in all_results[tier]]
    mechanism_stats[tier]["survival"] = bootstrap_ci(surv_vals)

    # Behavioral
    innov_share = [r["innovate_share"] for r in all_results[tier]]
    explr_share = [r["explore_share"] for r in all_results[tier]]
    mechanism_stats[tier]["innovate_share"] = bootstrap_ci(innov_share)
    mechanism_stats[tier]["explore_share"] = bootstrap_ci(explr_share)

    # Competition
    comp_vals = [r["mean_competition"] for r in all_results[tier]]
    mechanism_stats[tier]["competition"] = bootstrap_ci(comp_vals)

    # Innovation
    niche_vals = Float64[r["final_niches"] for r in all_results[tier]]
    combo_vals = Float64[r["final_combinations"] for r in all_results[tier]]
    success_vals = [r["success_rate"] for r in all_results[tier]]
    mechanism_stats[tier]["niches"] = bootstrap_ci(niche_vals)
    mechanism_stats[tier]["combinations"] = bootstrap_ci(combo_vals)
    mechanism_stats[tier]["success_rate"] = bootstrap_ci(success_vals)

    # Financial
    roi_vals = [r["roi"] for r in all_results[tier]]
    mechanism_stats[tier]["roi"] = bootstrap_ci(roi_vals)

    # AI Trust
    trust_vals = [r["mean_ai_trust"] for r in all_results[tier]]
    mechanism_stats[tier]["ai_trust"] = bootstrap_ci(trust_vals)
end

# Print mechanism summary
println("\n--- MECHANISM SUMMARY ---")
println("-"^90)
@printf("%-12s %10s %12s %12s %10s %12s %10s\n",
    "Tier", "Survival", "Innovate%", "Explore%", "Compet.", "Niches", "ROI%")
println("-"^90)
for tier in TIER_ORDER
    m = mechanism_stats[tier]
    @printf("%-12s %9.1f%% %11.1f%% %11.1f%% %9.3f %11.1f %9.1f%%\n",
        TIER_LABELS[tier],
        m["survival"].mean * 100,
        m["innovate_share"].mean * 100,
        m["explore_share"].mean * 100,
        m["competition"].mean,
        m["niches"].mean,
        m["roi"].mean * 100)
end

# ============================================================================
# PHASE 3: MEDIATION ANALYSIS
# ============================================================================

println("\n" * "="^80)
println("PHASE 3: MEDIATION ANALYSIS")
println("="^80)

# Collect agent-level data for correlation analysis
all_run_data = []
for tier in TIER_ORDER
    for r in all_results[tier]
        push!(all_run_data, (
            tier = tier,
            tier_numeric = findfirst(==(tier), TIER_ORDER),
            survival = r["survival_rate"],
            innovate_share = r["innovate_share"],
            explore_share = r["explore_share"],
            competition = r["mean_competition"],
            niches = r["final_niches"],
            combinations = r["final_combinations"],
            roi = r["roi"]
        ))
    end
end
run_df = DataFrame(all_run_data)

# Simple correlations
println("  Correlation Analysis (run-level, N=$(nrow(run_df))):")

# Tier → Survival (total effect)
tier_survival_corr = cor(run_df.tier_numeric, run_df.survival)
@printf("    Tier → Survival:      r = %.3f (total effect)\n", tier_survival_corr)

# Tier → Mediators
tier_innovate_corr = cor(run_df.tier_numeric, run_df.innovate_share)
tier_competition_corr = cor(run_df.tier_numeric, run_df.competition)
tier_niches_corr = cor(run_df.tier_numeric, Float64.(run_df.niches))
@printf("    Tier → Innovate:      r = %.3f\n", tier_innovate_corr)
@printf("    Tier → Competition:   r = %.3f\n", tier_competition_corr)
@printf("    Tier → Niches:        r = %.3f\n", tier_niches_corr)

# Mediators → Survival
innovate_survival_corr = cor(run_df.innovate_share, run_df.survival)
competition_survival_corr = cor(run_df.competition, run_df.survival)
niches_survival_corr = cor(Float64.(run_df.niches), run_df.survival)
@printf("    Innovate → Survival:  r = %.3f\n", innovate_survival_corr)
@printf("    Competition → Surv:   r = %.3f\n", competition_survival_corr)
@printf("    Niches → Survival:    r = %.3f\n", niches_survival_corr)

# Sobel-like mediation estimate (simplified)
# Indirect effect = a * b where a = Tier→Mediator, b = Mediator→Survival
indirect_innovate = tier_innovate_corr * innovate_survival_corr
indirect_competition = tier_competition_corr * competition_survival_corr
indirect_niches = tier_niches_corr * niches_survival_corr

println("\n  Mediation Estimates (indirect effects):")
@printf("    Via Innovation:   %.3f (%.0f%% of total)\n", indirect_innovate, 100*abs(indirect_innovate)/abs(tier_survival_corr))
@printf("    Via Competition:  %.3f (%.0f%% of total)\n", indirect_competition, 100*abs(indirect_competition)/abs(tier_survival_corr))
@printf("    Via Niches:       %.3f (%.0f%% of total)\n", indirect_niches, 100*abs(indirect_niches)/abs(tier_survival_corr))

# ============================================================================
# PHASE 4: GENERATE PDF
# ============================================================================

println("\n" * "="^80)
println("PHASE 4: GENERATING MECHANISM PDF")
println("="^80)

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

fig = Figure(size = (1400, 1000))

# Title
Label(fig[1, 1:4], "Mechanism Analysis: Why Does Premium AI Reduce Survival?",
    fontsize = 20, font = :bold, halign = :center)
Label(fig[1, 1:4], "\n\nExamining causal pathways: Behavioral shifts, competition dynamics, and innovation patterns",
    fontsize = 9, color = :gray40, halign = :center, valign = :top)

# ========== ROW 2: BEHAVIORAL MECHANISM ==========

# Fig A: Innovate Share by Tier
ax1a = Axis(fig[2, 1], xlabel = "AI Tier", ylabel = "Innovate Share (%)",
    title = "A. Innovation Activity by Tier",
    xticks = (1:4, ["None", "Basic", "Adv", "Prem"]))
innov_means = [mechanism_stats[t]["innovate_share"].mean * 100 for t in TIER_ORDER]
innov_los = [mechanism_stats[t]["innovate_share"].ci_lo * 100 for t in TIER_ORDER]
innov_his = [mechanism_stats[t]["innovate_share"].ci_hi * 100 for t in TIER_ORDER]
barplot!(ax1a, 1:4, innov_means, color = [TIER_COLORS[t] for t in TIER_ORDER])
errorbars!(ax1a, 1:4, innov_means, innov_means .- innov_los, innov_his .- innov_means,
    color = :black, whiskerwidth = 10, linewidth = 1.5)
ylims!(ax1a, minimum(innov_los) * 0.95, maximum(innov_his) * 1.05)

# Fig B: Explore Share by Tier
ax1b = Axis(fig[2, 2], xlabel = "AI Tier", ylabel = "Explore Share (%)",
    title = "B. Exploration Activity by Tier",
    xticks = (1:4, ["None", "Basic", "Adv", "Prem"]))
explr_means = [mechanism_stats[t]["explore_share"].mean * 100 for t in TIER_ORDER]
explr_los = [mechanism_stats[t]["explore_share"].ci_lo * 100 for t in TIER_ORDER]
explr_his = [mechanism_stats[t]["explore_share"].ci_hi * 100 for t in TIER_ORDER]
barplot!(ax1b, 1:4, explr_means, color = [TIER_COLORS[t] for t in TIER_ORDER])
errorbars!(ax1b, 1:4, explr_means, explr_means .- explr_los, explr_his .- explr_means,
    color = :black, whiskerwidth = 10, linewidth = 1.5)
ylims!(ax1b, minimum(explr_los) * 0.95, maximum(explr_his) * 1.05)

# Fig C: Innovate Share Over Time
ax1c = Axis(fig[2, 3:4], xlabel = "Round (Month)", ylabel = "Innovate Share (%)",
    title = "C. Innovation Activity Over Time by Tier")
for tier in TIER_ORDER
    trajs = [r["innovate_share_traj"] for r in all_results[tier]]
    means_t = [mean(t[i] for t in trajs) * 100 for i in 1:N_ROUNDS]
    stds_t = [std(t[i] for t in trajs) * 100 for i in 1:N_ROUNDS]
    ci_lo_t = means_t .- 1.96 .* stds_t ./ sqrt(N_RUNS)
    ci_hi_t = means_t .+ 1.96 .* stds_t ./ sqrt(N_RUNS)
    band!(ax1c, 1:N_ROUNDS, ci_lo_t, ci_hi_t, color = (TIER_COLORS[tier], 0.2))
    lines!(ax1c, 1:N_ROUNDS, means_t, color = TIER_COLORS[tier], linewidth = 2, label = TIER_LABELS[tier])
end
axislegend(ax1c, position = :rb, labelsize = 8)

# Description
Label(fig[3, 1:4],
    "Behavioral Mechanism: AI agents shift from exploration ($(Printf.@sprintf("%.1f", explr_means[1]))% → $(Printf.@sprintf("%.1f", explr_means[4]))%) to innovation ($(Printf.@sprintf("%.1f", innov_means[1]))% → $(Printf.@sprintf("%.1f", innov_means[4]))%). This behavioral shift toward risky creative activity is a key driver of the survival penalty.",
    fontsize = 8, color = :gray30, halign = :left, valign = :top)

# ========== ROW 4: COMPETITION & INNOVATION MECHANISM ==========

# Fig D: Competition Levels by Tier
ax2a = Axis(fig[4, 1], xlabel = "AI Tier", ylabel = "Mean Competition Level",
    title = "D. Competition Intensity by Tier",
    xticks = (1:4, ["None", "Basic", "Adv", "Prem"]))
comp_means = [mechanism_stats[t]["competition"].mean for t in TIER_ORDER]
comp_los = [mechanism_stats[t]["competition"].ci_lo for t in TIER_ORDER]
comp_his = [mechanism_stats[t]["competition"].ci_hi for t in TIER_ORDER]
barplot!(ax2a, 1:4, comp_means, color = [TIER_COLORS[t] for t in TIER_ORDER])
errorbars!(ax2a, 1:4, comp_means, comp_means .- comp_los, comp_his .- comp_means,
    color = :black, whiskerwidth = 10, linewidth = 1.5)

# Fig E: Niches Discovered by Tier
ax2b = Axis(fig[4, 2], xlabel = "AI Tier", ylabel = "Total Niches Created",
    title = "E. Market Niches Created by Tier",
    xticks = (1:4, ["None", "Basic", "Adv", "Prem"]))
niche_means = [mechanism_stats[t]["niches"].mean for t in TIER_ORDER]
niche_los = [mechanism_stats[t]["niches"].ci_lo for t in TIER_ORDER]
niche_his = [mechanism_stats[t]["niches"].ci_hi for t in TIER_ORDER]
barplot!(ax2b, 1:4, niche_means, color = [TIER_COLORS[t] for t in TIER_ORDER])
errorbars!(ax2b, 1:4, niche_means, niche_means .- niche_los, niche_his .- niche_means,
    color = :black, whiskerwidth = 10, linewidth = 1.5)

# Fig F: Niche Discovery Over Time
ax2c = Axis(fig[4, 3:4], xlabel = "Round (Month)", ylabel = "Cumulative Niches",
    title = "F. Cumulative Niche Discovery Over Time")
for tier in TIER_ORDER
    trajs = [r["niches_traj"] for r in all_results[tier]]
    means_t = [mean(Float64(t[i]) for t in trajs) for i in 1:N_ROUNDS]
    stds_t = [std(Float64(t[i]) for t in trajs) for i in 1:N_ROUNDS]
    ci_lo_t = means_t .- 1.96 .* stds_t ./ sqrt(N_RUNS)
    ci_hi_t = means_t .+ 1.96 .* stds_t ./ sqrt(N_RUNS)
    band!(ax2c, 1:N_ROUNDS, ci_lo_t, ci_hi_t, color = (TIER_COLORS[tier], 0.2))
    lines!(ax2c, 1:N_ROUNDS, means_t, color = TIER_COLORS[tier], linewidth = 2, label = TIER_LABELS[tier])
end
axislegend(ax2c, position = :lt, labelsize = 8)

# Description
Label(fig[5, 1:4],
    "Competition & Innovation Mechanism: Premium AI creates $(Printf.@sprintf("%.0f", niche_means[4]/niche_means[1]))× more market niches ($(Printf.@sprintf("%.0f", niche_means[1])) → $(Printf.@sprintf("%.0f", niche_means[4]))), but this creative output doesn't translate to survival. Competition levels remain similar across tiers, suggesting the mechanism is risk-taking rather than crowding.",
    fontsize = 8, color = :gray30, halign = :left, valign = :top)

# ========== ROW 6: MEDIATION ANALYSIS ==========

# Fig G: Correlation Path Diagram (simplified as bar chart)
ax3a = Axis(fig[6, 1:2], xlabel = "Correlation Path", ylabel = "Correlation (r)",
    title = "G. Mediation Pathways: Correlation Analysis",
    xticks = (1:6, ["Tier→\nSurv", "Tier→\nInnov", "Tier→\nNiche", "Innov→\nSurv", "Niche→\nSurv", "Comp→\nSurv"]))
corr_vals = [tier_survival_corr, tier_innovate_corr, tier_niches_corr,
             innovate_survival_corr, niches_survival_corr, competition_survival_corr]
corr_colors = [corr_vals[i] < 0 ? TIER_COLORS["premium"] : TIER_COLORS["basic"] for i in 1:6]
barplot!(ax3a, 1:6, corr_vals, color = corr_colors)
hlines!(ax3a, [0], color = :black, linestyle = :dash, linewidth = 1)

# Fig H: Indirect Effects (Mediation)
ax3b = Axis(fig[6, 3], xlabel = "Mediator", ylabel = "Indirect Effect",
    title = "H. Mediation: Indirect Effects",
    xticks = (1:3, ["Innovation", "Niches", "Compet."]))
indirect_vals = [indirect_innovate, indirect_niches, indirect_competition]
indirect_colors = [v < 0 ? TIER_COLORS["premium"] : TIER_COLORS["basic"] for v in indirect_vals]
barplot!(ax3b, 1:3, indirect_vals, color = indirect_colors)
hlines!(ax3b, [0], color = :black, linestyle = :dash, linewidth = 1)

# Fig I: Survival vs Innovate Share Scatter
ax3c = Axis(fig[6, 4], xlabel = "Innovate Share (%)", ylabel = "Survival Rate (%)",
    title = "I. Survival vs Innovation (by Tier)")
for tier in TIER_ORDER
    innov_data = [r["innovate_share"] * 100 for r in all_results[tier]]
    surv_data = [r["survival_rate"] * 100 for r in all_results[tier]]
    scatter!(ax3c, innov_data, surv_data, color = (TIER_COLORS[tier], 0.6),
        markersize = 6, label = TIER_LABELS[tier])
end
axislegend(ax3c, position = :rt, labelsize = 7)

# Description
pct_via_innovation = abs(indirect_innovate) / abs(tier_survival_corr) * 100
pct_via_niches = abs(indirect_niches) / abs(tier_survival_corr) * 100
Label(fig[7, 1:4],
    "Mediation Analysis: The Tier→Survival effect (r=$(Printf.@sprintf("%.2f", tier_survival_corr))) is partially mediated by innovation activity ($(Printf.@sprintf("%.0f", pct_via_innovation))% indirect) and niche creation ($(Printf.@sprintf("%.0f", pct_via_niches))% indirect). Higher AI tiers increase innovation, but innovation is negatively associated with survival (r=$(Printf.@sprintf("%.2f", innovate_survival_corr))), creating the paradox.",
    fontsize = 8, color = :gray30, halign = :left, valign = :top)

# Footer
timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
Label(fig[8, 1:4], "Generated: $timestamp | GlimpseABM Mechanism Analysis | N=$N_AGENTS × $N_ROUNDS × $N_RUNS runs | Townsend et al. (2025) AMR",
    fontsize = 8, color = :gray50, halign = :center)

# Adjust spacing
rowgap!(fig.layout, 1, 5)
rowgap!(fig.layout, 3, 10)
rowgap!(fig.layout, 5, 10)
rowgap!(fig.layout, 7, 5)

# Save PDF
pdf_path = joinpath(OUTPUT_DIR, "mechanism_analysis_results.pdf")
save(pdf_path, fig)
println("  Saved: $pdf_path")

# ============================================================================
# SAVE DATA
# ============================================================================

println("\n" * "="^80)
println("SAVING DATA FILES")
println("="^80)

# Summary by tier
summary_df = DataFrame(
    Tier = [TIER_LABELS[t] for t in TIER_ORDER],
    Survival_Mean = [mechanism_stats[t]["survival"].mean * 100 for t in TIER_ORDER],
    Innovate_Share = [mechanism_stats[t]["innovate_share"].mean * 100 for t in TIER_ORDER],
    Explore_Share = [mechanism_stats[t]["explore_share"].mean * 100 for t in TIER_ORDER],
    Competition = [mechanism_stats[t]["competition"].mean for t in TIER_ORDER],
    Niches = [mechanism_stats[t]["niches"].mean for t in TIER_ORDER],
    Combinations = [mechanism_stats[t]["combinations"].mean for t in TIER_ORDER],
    Success_Rate = [mechanism_stats[t]["success_rate"].mean * 100 for t in TIER_ORDER],
    ROI = [mechanism_stats[t]["roi"].mean * 100 for t in TIER_ORDER]
)
CSV.write(joinpath(OUTPUT_DIR, "mechanism_summary.csv"), summary_df)
println("  Saved: mechanism_summary.csv")

# Correlation/mediation results
mediation_df = DataFrame(
    Path = ["Tier→Survival", "Tier→Innovate", "Tier→Niches", "Tier→Competition",
            "Innovate→Survival", "Niches→Survival", "Competition→Survival",
            "Indirect_via_Innovation", "Indirect_via_Niches", "Indirect_via_Competition"],
    Correlation = [tier_survival_corr, tier_innovate_corr, tier_niches_corr, tier_competition_corr,
                   innovate_survival_corr, niches_survival_corr, competition_survival_corr,
                   indirect_innovate, indirect_niches, indirect_competition]
)
CSV.write(joinpath(OUTPUT_DIR, "mediation_analysis.csv"), mediation_df)
println("  Saved: mediation_analysis.csv")

# Run-level data
CSV.write(joinpath(OUTPUT_DIR, "run_level_data.csv"), run_df)
println("  Saved: run_level_data.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

total_time = time() - master_start

println("\n" * "="^80)
println("MECHANISM ANALYSIS COMPLETE")
println("="^80)

println("\nKEY MECHANISM FINDINGS:")
println("  1. BEHAVIORAL SHIFT: Innovate share increases $(Printf.@sprintf("%.1f", innov_means[1]))% → $(Printf.@sprintf("%.1f", innov_means[4]))% (+$(Printf.@sprintf("%.0f", (innov_means[4]-innov_means[1])/innov_means[1]*100))%)")
println("  2. EXPLORATION DECLINE: Explore share decreases $(Printf.@sprintf("%.1f", explr_means[1]))% → $(Printf.@sprintf("%.1f", explr_means[4]))% ($(Printf.@sprintf("%.0f", (explr_means[4]-explr_means[1])/explr_means[1]*100))%)")
println("  3. NICHE CREATION: $(Printf.@sprintf("%.0f", niche_means[4]/niche_means[1]))× more niches created by Premium AI")
println("  4. SURVIVAL PARADOX: More innovation (r=$(Printf.@sprintf("%.2f", tier_innovate_corr))) → Lower survival (r=$(Printf.@sprintf("%.2f", innovate_survival_corr)))")
println("  5. MEDIATION: $(Printf.@sprintf("%.0f", pct_via_innovation))% of tier effect mediated via innovation behavior")

println("\nOUTPUT FILES:")
println("  PDF Report: $pdf_path")
println("  Summary:    $(joinpath(OUTPUT_DIR, "mechanism_summary.csv"))")
println("  Mediation:  $(joinpath(OUTPUT_DIR, "mediation_analysis.csv"))")

@printf("\nTotal runtime: %.1f minutes (%.0f seconds)\n", total_time/60, total_time)
println("="^80)
