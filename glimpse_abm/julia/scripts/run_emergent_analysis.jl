#!/usr/bin/env julia
"""
EMERGENT AI ADOPTION ANALYSIS
=============================

Analysis where agents DYNAMICALLY SELECT their AI tier each round based on:
- Bayesian beliefs about tier effectiveness
- ROI signals and performance metrics
- Peer pressure and adoption rates
- Cost considerations and switching penalties

This captures realistic AI adoption dynamics where agents learn and adapt.

Tracks:
- AI adoption patterns over time
- AI trust evolution
- Survival, financial, innovation outcomes by adopted tier
- Unicorn/outlier emergence

Output: Single landscape PDF with key findings

Usage:
    julia --threads=12 --project=. scripts/run_emergent_analysis.jl
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
const BASE_SEED = 20260128
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const N_BOOTSTRAP = 2000

const OUTPUT_DIR = joinpath(dirname(@__DIR__), "results",
    "emergent_analysis_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")

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
println("EMERGENT AI ADOPTION ANALYSIS")
println("="^80)
println("Configuration:")
println("  Threads:     $(Threads.nthreads())")
println("  Agents:      $N_AGENTS per simulation")
println("  Rounds:      $N_ROUNDS (5 years)")
println("  Runs:        $N_RUNS")
println("  AI Selection: DYNAMIC (agents choose each round)")
println("  Output:      $OUTPUT_DIR")
println("="^80)

master_start = time()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function run_emergent_simulation(run_idx::Int, seed::Int)
    config = EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        INITIAL_CAPITAL=100_000_000.0,
        SURVIVAL_THRESHOLD=10_000.0
    )

    # Equal initial distribution - agents will dynamically choose
    initial_dist = Dict("none" => 0.25, "basic" => 0.25, "advanced" => 0.25, "premium" => 0.25)
    sim = EmergentSimulation(config=config, initial_tier_distribution=initial_dist)

    # Clear fixed_ai_level to enable dynamic selection
    for agent in sim.agents
        agent.fixed_ai_level = nothing
    end

    # Trajectories
    survival_trajectory = Float64[]
    tier_distribution_trajectory = [Dict{String,Float64}() for _ in 1:N_ROUNDS]
    ai_trust_trajectory = Float64[]
    ai_trust_by_tier = Dict(t => Float64[] for t in TIER_ORDER)

    # Action shares
    action_shares = Dict("invest" => Float64[], "innovate" => Float64[],
                         "explore" => Float64[], "maintain" => Float64[])

    # Innovation trajectories
    niches_trajectory = Int[]

    for r in 1:N_ROUNDS
        GlimpseABM.step!(sim, r)

        alive = filter(a -> a.alive, sim.agents)
        push!(survival_trajectory, length(alive) / length(sim.agents))

        # Tier distribution among alive agents
        tier_counts = Dict(t => 0 for t in TIER_ORDER)
        for a in alive
            tier = a.current_ai_level
            tier_counts[tier] = get(tier_counts, tier, 0) + 1
        end
        n_alive = length(alive)
        for t in TIER_ORDER
            tier_distribution_trajectory[r][t] = n_alive > 0 ? tier_counts[t] / n_alive : 0.0
        end

        # AI trust (mean across alive agents)
        if !isempty(alive)
            trust_vals = [a.ai_trust for a in alive]
            push!(ai_trust_trajectory, mean(trust_vals))

            # Trust by current tier
            for t in TIER_ORDER
                tier_agents = filter(a -> a.current_ai_level == t, alive)
                if !isempty(tier_agents)
                    push!(ai_trust_by_tier[t], mean(a.ai_trust for a in tier_agents))
                end
            end
        else
            push!(ai_trust_trajectory, 0.5)
        end

        # Action shares
        action_counts = Dict("invest" => 0, "innovate" => 0, "explore" => 0, "maintain" => 0)
        for agent in sim.agents
            if length(agent.action_history) >= r
                act = agent.action_history[r]
                action_counts[act] = get(action_counts, act, 0) + 1
            end
        end
        total_actions = sum(values(action_counts))
        for act in keys(action_shares)
            push!(action_shares[act], total_actions > 0 ? action_counts[act] / total_actions : 0.0)
        end

        # Niches
        total_niches = sum(a.uncertainty_metrics.niches_discovered for a in sim.agents)
        push!(niches_trajectory, total_niches)
    end

    # Final metrics by FINAL tier (what tier the agent ended up using)
    alive_agents = filter(a -> a.alive, sim.agents)

    # Determine "dominant" tier for each agent (mode of their history)
    function dominant_tier(agent)
        if isempty(agent.ai_tier_history)
            return agent.current_ai_level
        end
        tier_counts = Dict(t => count(==(t), agent.ai_tier_history) for t in TIER_ORDER)
        return argmax(tier_counts)
    end

    # Metrics by dominant tier
    survival_by_tier = Dict{String, Vector{Bool}}()
    capital_by_tier = Dict{String, Vector{Float64}}()
    roi_by_tier = Dict{String, Vector{Float64}}()
    innovations_by_tier = Dict{String, Vector{Int}}()
    niches_by_tier = Dict{String, Vector{Int}}()

    for t in TIER_ORDER
        survival_by_tier[t] = Bool[]
        capital_by_tier[t] = Float64[]
        roi_by_tier[t] = Float64[]
        innovations_by_tier[t] = Int[]
        niches_by_tier[t] = Int[]
    end

    for agent in sim.agents
        tier = dominant_tier(agent)
        push!(survival_by_tier[tier], agent.alive)
        push!(capital_by_tier[tier], GlimpseABM.get_capital(agent))

        # ROI
        invested = agent.total_invested
        returned = agent.total_returned
        roi = invested > 0 ? (returned - invested) / invested : 0.0
        push!(roi_by_tier[tier], roi)

        push!(innovations_by_tier[tier], agent.innovation_count)
        push!(niches_by_tier[tier], agent.uncertainty_metrics.niches_discovered)
    end

    # Capital multipliers for unicorn analysis
    capital_multipliers = [GlimpseABM.get_capital(a) / a.resources.performance.initial_equity
                          for a in sim.agents]
    survivor_multipliers = [GlimpseABM.get_capital(a) / a.resources.performance.initial_equity
                           for a in alive_agents]

    # Tier switching stats
    total_switches = 0
    for agent in sim.agents
        for i in 2:length(agent.ai_tier_history)
            if agent.ai_tier_history[i] != agent.ai_tier_history[i-1]
                total_switches += 1
            end
        end
    end

    return Dict(
        "run_idx" => run_idx,
        "seed" => seed,
        "survival_rate" => length(alive_agents) / length(sim.agents),
        "survival_trajectory" => survival_trajectory,
        "tier_distribution_trajectory" => tier_distribution_trajectory,
        "ai_trust_trajectory" => ai_trust_trajectory,
        "ai_trust_by_tier" => ai_trust_by_tier,
        "action_shares" => action_shares,
        "niches_trajectory" => niches_trajectory,
        "survival_by_tier" => survival_by_tier,
        "capital_by_tier" => capital_by_tier,
        "roi_by_tier" => roi_by_tier,
        "innovations_by_tier" => innovations_by_tier,
        "niches_by_tier" => niches_by_tier,
        "capital_multipliers" => capital_multipliers,
        "survivor_multipliers" => survivor_multipliers,
        "total_switches" => total_switches,
        "final_tier_distribution" => tier_distribution_trajectory[end]
    )
end

function bootstrap_ci(data::Vector{Float64}; n_boot=N_BOOTSTRAP, alpha=0.05)
    if isempty(data)
        return (mean=0.0, ci_lo=0.0, ci_hi=0.0)
    end
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
    return (mean=mean(data), ci_lo=ci_lo, ci_hi=ci_hi)
end

# ============================================================================
# PHASE 1: RUN SIMULATIONS
# ============================================================================

println("\n" * "="^80)
println("PHASE 1: RUNNING EMERGENT SIMULATIONS")
println("="^80)

all_results = Vector{Dict}(undef, N_RUNS)
completed = Threads.Atomic{Int}(0)
results_lock = ReentrantLock()

phase1_start = time()
println("Running $N_RUNS simulations with dynamic AI selection...")

Threads.@threads for run_idx in 1:N_RUNS
    seed = BASE_SEED + run_idx
    result = run_emergent_simulation(run_idx, seed)

    lock(results_lock) do
        all_results[run_idx] = result
    end

    done = Threads.atomic_add!(completed, 1)
    if done % 10 == 0 || done == N_RUNS
        elapsed = time() - phase1_start
        rate = done / elapsed
        eta = (N_RUNS - done) / rate
        @printf("\r  Progress: %d/%d (%.0f%%) | %.1f sims/sec | ETA: %.0fs    ",
            done, N_RUNS, 100*done/N_RUNS, rate, eta)
    end
end

println("\n  Phase 1 complete: $(round(time() - phase1_start, digits=1))s")

# ============================================================================
# PHASE 2: AGGREGATE RESULTS
# ============================================================================

println("\n" * "="^80)
println("PHASE 2: AGGREGATING RESULTS")
println("="^80)

# Aggregate trajectories
mean_survival_traj = zeros(N_ROUNDS)
mean_trust_traj = zeros(N_ROUNDS)
mean_tier_dist = [Dict(t => 0.0 for t in TIER_ORDER) for _ in 1:N_ROUNDS]
mean_action_shares = Dict(act => zeros(N_ROUNDS) for act in ["invest", "innovate", "explore", "maintain"])
mean_niches_traj = zeros(N_ROUNDS)

for res in all_results
    mean_survival_traj .+= res["survival_trajectory"] ./ N_RUNS
    mean_trust_traj .+= res["ai_trust_trajectory"] ./ N_RUNS
    mean_niches_traj .+= res["niches_trajectory"] ./ N_RUNS

    for r in 1:N_ROUNDS
        for t in TIER_ORDER
            mean_tier_dist[r][t] += get(res["tier_distribution_trajectory"][r], t, 0.0) / N_RUNS
        end
    end

    for act in keys(mean_action_shares)
        mean_action_shares[act] .+= res["action_shares"][act] ./ N_RUNS
    end
end

# Aggregate by-tier metrics
agg_survival_by_tier = Dict(t => Float64[] for t in TIER_ORDER)
agg_capital_by_tier = Dict(t => Float64[] for t in TIER_ORDER)
agg_roi_by_tier = Dict(t => Float64[] for t in TIER_ORDER)
agg_niches_by_tier = Dict(t => Float64[] for t in TIER_ORDER)

for res in all_results
    for t in TIER_ORDER
        surv_data = res["survival_by_tier"][t]
        if !isempty(surv_data)
            push!(agg_survival_by_tier[t], mean(surv_data))
            push!(agg_capital_by_tier[t], mean(res["capital_by_tier"][t]))
            push!(agg_roi_by_tier[t], mean(res["roi_by_tier"][t]))
            push!(agg_niches_by_tier[t], mean(res["niches_by_tier"][t]))
        end
    end
end

# Aggregate capital multipliers for unicorn analysis
all_multipliers = Float64[]
all_survivor_multipliers = Float64[]
for res in all_results
    append!(all_multipliers, res["capital_multipliers"])
    append!(all_survivor_multipliers, res["survivor_multipliers"])
end

# Summary stats
overall_survival = mean(r["survival_rate"] for r in all_results)
total_switches = sum(r["total_switches"] for r in all_results)
mean_switches_per_agent = total_switches / (N_RUNS * N_AGENTS)

println("  Overall survival rate: $(round(overall_survival*100, digits=1))%")
println("  Mean tier switches/agent: $(round(mean_switches_per_agent, digits=2))")

# Final tier distribution
final_dist = Dict(t => 0.0 for t in TIER_ORDER)
for res in all_results
    for t in TIER_ORDER
        final_dist[t] += get(res["final_tier_distribution"], t, 0.0) / N_RUNS
    end
end
println("  Final tier distribution:")
for t in TIER_ORDER
    @printf("    %s: %.1f%%\n", TIER_LABELS[t], final_dist[t]*100)
end

# ============================================================================
# PHASE 3: GENERATE PDF
# ============================================================================

println("\n" * "="^80)
println("PHASE 3: GENERATING PDF")
println("="^80)

set_theme!(Theme(
    fontsize = 9,
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
Label(fig[1, 1:4], "Emergent AI Adoption Analysis: Dynamic Tier Selection",
    fontsize = 18, font = :bold, halign = :center)
Label(fig[1, 1:4], "\n\nAgents dynamically choose AI tiers based on Bayesian learning, ROI signals, peer pressure, and costs",
    fontsize = 9, color = :gray40, halign = :center, valign = :top)

# ========== ROW 2: ADOPTION DYNAMICS ==========

# Fig A: AI Tier Adoption Over Time (stacked area)
ax_a = Axis(fig[2, 1], xlabel = "Month", ylabel = "Share of Alive Agents",
    title = "A. AI Tier Adoption Over Time")
rounds = 1:N_ROUNDS

# Create stacked area data
none_vals = [mean_tier_dist[r]["none"] for r in rounds]
basic_vals = [mean_tier_dist[r]["basic"] for r in rounds]
advanced_vals = [mean_tier_dist[r]["advanced"] for r in rounds]
premium_vals = [mean_tier_dist[r]["premium"] for r in rounds]

# Stack from bottom
band!(ax_a, rounds, zeros(N_ROUNDS), none_vals, color = (TIER_COLORS["none"], 0.7), label = "No AI")
band!(ax_a, rounds, none_vals, none_vals .+ basic_vals, color = (TIER_COLORS["basic"], 0.7), label = "Basic")
band!(ax_a, rounds, none_vals .+ basic_vals, none_vals .+ basic_vals .+ advanced_vals,
      color = (TIER_COLORS["advanced"], 0.7), label = "Advanced")
band!(ax_a, rounds, none_vals .+ basic_vals .+ advanced_vals,
      none_vals .+ basic_vals .+ advanced_vals .+ premium_vals,
      color = (TIER_COLORS["premium"], 0.7), label = "Premium")
axislegend(ax_a, position = :rt, labelsize = 7, framevisible = false)

# Fig B: AI Trust Over Time
ax_b = Axis(fig[2, 2], xlabel = "Month", ylabel = "Mean AI Trust",
    title = "B. AI Trust Evolution Over Time")
lines!(ax_b, rounds, mean_trust_traj, color = :black, linewidth = 2, label = "Overall")
# Add by-tier trust if we have data
for t in TIER_ORDER
    tier_trust = Float64[]
    for r in 1:N_ROUNDS
        tier_vals = Float64[]
        for res in all_results
            if r <= length(res["ai_trust_by_tier"][t]) && !isempty(res["ai_trust_by_tier"][t])
                # Approximate: use overall trajectory length
            end
        end
    end
end
hlines!(ax_b, [0.5], color = :gray, linestyle = :dash, linewidth = 1)
ylims!(ax_b, 0.3, 0.7)

# Fig C: Survival by Dominant Tier
ax_c = Axis(fig[2, 3], xlabel = "Dominant AI Tier", ylabel = "Survival Rate (%)",
    title = "C. Survival by Adopted Tier",
    xticks = (1:4, [TIER_LABELS[t] for t in TIER_ORDER]))
surv_means = [isempty(agg_survival_by_tier[t]) ? 0.0 : mean(agg_survival_by_tier[t]) * 100 for t in TIER_ORDER]
surv_ci = [isempty(agg_survival_by_tier[t]) ? (mean=0.0, ci_lo=0.0, ci_hi=0.0) :
           bootstrap_ci(agg_survival_by_tier[t] .* 100) for t in TIER_ORDER]
barplot!(ax_c, 1:4, surv_means, color = [TIER_COLORS[t] for t in TIER_ORDER])
errorbars!(ax_c, 1:4, surv_means,
    [surv_means[i] - surv_ci[i].ci_lo for i in 1:4],
    [surv_ci[i].ci_hi - surv_means[i] for i in 1:4],
    color = :black, whiskerwidth = 8, linewidth = 1.5)

# Fig D: Survival Trajectory
ax_d = Axis(fig[2, 4], xlabel = "Month", ylabel = "Survival Rate (%)",
    title = "D. Survival Trajectory")
lines!(ax_d, rounds, mean_survival_traj .* 100, color = :black, linewidth = 2)
# Add CI band
surv_traj_by_run = hcat([r["survival_trajectory"] for r in all_results]...)
surv_lo = [quantile(surv_traj_by_run[r, :], 0.025) * 100 for r in 1:N_ROUNDS]
surv_hi = [quantile(surv_traj_by_run[r, :], 0.975) * 100 for r in 1:N_ROUNDS]
band!(ax_d, rounds, surv_lo, surv_hi, color = (:gray, 0.3))
lines!(ax_d, rounds, mean_survival_traj .* 100, color = :black, linewidth = 2)

# Description Row 2
Label(fig[3, 1:4],
    "Adoption Dynamics: Agents dynamically select AI tiers based on learned beliefs and performance. AI trust evolves based on prediction accuracy. Survival varies by adopted tier - agents who predominantly use higher AI tiers show different survival patterns than those who stick with lower tiers.",
    fontsize = 8, color = :gray30, halign = :left, valign = :top)

# ========== ROW 4: FINANCIAL & BEHAVIORAL ==========

# Fig E: Mean ROI by Tier
ax_e = Axis(fig[4, 1], xlabel = "Dominant AI Tier", ylabel = "Mean ROI (%)",
    title = "E. Investment ROI by Tier",
    xticks = (1:4, [TIER_LABELS[t] for t in TIER_ORDER]))
roi_means = [isempty(agg_roi_by_tier[t]) ? 0.0 : mean(agg_roi_by_tier[t]) * 100 for t in TIER_ORDER]
roi_ci = [isempty(agg_roi_by_tier[t]) ? (mean=0.0, ci_lo=0.0, ci_hi=0.0) :
          bootstrap_ci(agg_roi_by_tier[t] .* 100) for t in TIER_ORDER]
barplot!(ax_e, 1:4, roi_means, color = [TIER_COLORS[t] for t in TIER_ORDER])
errorbars!(ax_e, 1:4, roi_means,
    [roi_means[i] - roi_ci[i].ci_lo for i in 1:4],
    [roi_ci[i].ci_hi - roi_means[i] for i in 1:4],
    color = :black, whiskerwidth = 8, linewidth = 1.5)
hlines!(ax_e, [0], color = :black, linestyle = :dash, linewidth = 1)

# Fig F: Innovate Action Share Over Time
ax_f = Axis(fig[4, 2], xlabel = "Month", ylabel = "Action Share (%)",
    title = "F. Innovate vs Explore Actions Over Time")
lines!(ax_f, rounds, mean_action_shares["innovate"] .* 100, color = colorant"#6f42c1",
       linewidth = 2, label = "Innovate")
lines!(ax_f, rounds, mean_action_shares["explore"] .* 100, color = colorant"#17a2b8",
       linewidth = 2, label = "Explore")
axislegend(ax_f, position = :rt, labelsize = 8)

# Fig G: Mean Niches by Tier
ax_g = Axis(fig[4, 3], xlabel = "Dominant AI Tier", ylabel = "Mean Niches Discovered",
    title = "G. Niche Discovery by Tier",
    xticks = (1:4, [TIER_LABELS[t] for t in TIER_ORDER]))
niche_means = [isempty(agg_niches_by_tier[t]) ? 0.0 : mean(agg_niches_by_tier[t]) for t in TIER_ORDER]
niche_ci = [isempty(agg_niches_by_tier[t]) ? (mean=0.0, ci_lo=0.0, ci_hi=0.0) :
            bootstrap_ci(agg_niches_by_tier[t]) for t in TIER_ORDER]
barplot!(ax_g, 1:4, niche_means, color = [TIER_COLORS[t] for t in TIER_ORDER])
errorbars!(ax_g, 1:4, niche_means,
    [niche_means[i] - niche_ci[i].ci_lo for i in 1:4],
    [niche_ci[i].ci_hi - niche_means[i] for i in 1:4],
    color = :black, whiskerwidth = 8, linewidth = 1.5)

# Fig H: Cumulative Niches Over Time
ax_h = Axis(fig[4, 4], xlabel = "Month", ylabel = "Cumulative Niches",
    title = "H. Cumulative Niche Discovery")
lines!(ax_h, rounds, mean_niches_traj, color = :black, linewidth = 2)
niche_traj_by_run = hcat([Float64.(r["niches_trajectory"]) for r in all_results]...)
niche_lo = [quantile(niche_traj_by_run[r, :], 0.025) for r in 1:N_ROUNDS]
niche_hi = [quantile(niche_traj_by_run[r, :], 0.975) for r in 1:N_ROUNDS]
band!(ax_h, rounds, niche_lo, niche_hi, color = (:gray, 0.3))
lines!(ax_h, rounds, mean_niches_traj, color = :black, linewidth = 2)

# Description Row 4
Label(fig[5, 1:4],
    "Financial & Innovation: ROI varies by adopted tier. Innovation vs exploration tradeoffs emerge dynamically. Agents who adopt higher AI tiers discover more niches but may face different financial outcomes. Mean switches per agent: $(round(mean_switches_per_agent, digits=2)).",
    fontsize = 8, color = :gray30, halign = :left, valign = :top)

# ========== ROW 6: UNICORN & OUTCOMES ==========

# Fig I: Capital Distribution (Histogram)
ax_i = Axis(fig[6, 1], xlabel = "Capital Multiplier (Final/Initial)", ylabel = "Frequency",
    title = "I. Capital Multiplier Distribution")
hist!(ax_i, clamp.(all_multipliers, 0, 3), bins = 50, color = (:gray, 0.7),
      strokewidth = 1, strokecolor = :gray40)
vlines!(ax_i, [1.0], color = :black, linestyle = :dash, linewidth = 1.5, label = "Break-even")
vlines!(ax_i, [mean(all_multipliers)], color = :blue, linewidth = 2, label = "Mean")

# Fig J: Unicorn Rates (5x, 10x)
ax_j = Axis(fig[6, 2], xlabel = "Threshold", ylabel = "Rate (%)",
    title = "J. Outlier Achievement Rates")
rate_5x = count(m -> m >= 5.0, all_survivor_multipliers) / max(1, length(all_survivor_multipliers)) * 100
rate_10x = count(m -> m >= 10.0, all_survivor_multipliers) / max(1, length(all_survivor_multipliers)) * 100
rate_2x = count(m -> m >= 2.0, all_survivor_multipliers) / max(1, length(all_survivor_multipliers)) * 100
barplot!(ax_j, [1, 2, 3], [rate_2x, rate_5x, rate_10x],
         color = [colorant"#28a745", colorant"#fd7e14", colorant"#dc3545"])
ax_j.xticks = ([1, 2, 3], ["2x (Double)", "5x (High)", "10x (Unicorn)"])

# Fig K: Final Tier Distribution
ax_k = Axis(fig[6, 3], xlabel = "AI Tier", ylabel = "Share of Survivors (%)",
    title = "K. Final Tier Distribution (Survivors)")
final_shares = [final_dist[t] * 100 for t in TIER_ORDER]
barplot!(ax_k, 1:4, final_shares, color = [TIER_COLORS[t] for t in TIER_ORDER])
ax_k.xticks = (1:4, [TIER_LABELS[t] for t in TIER_ORDER])

# Fig L: Tier Switching Behavior
ax_l = Axis(fig[6, 4], xlabel = "Run", ylabel = "Total Tier Switches",
    title = "L. Tier Switching Activity")
switches_per_run = [r["total_switches"] for r in all_results]
barplot!(ax_l, 1:N_RUNS, switches_per_run, color = (:gray, 0.6))
hlines!(ax_l, [mean(switches_per_run)], color = :red, linewidth = 2, linestyle = :dash)

# Description Row 6
unicorn_summary = @sprintf("Outlier rates among survivors: 2x=%.1f%%, 5x=%.1f%%, 10x=%.1f%%",
                           rate_2x, rate_5x, rate_10x)
Label(fig[7, 1:4],
    "Unicorn & Outcomes: $unicorn_summary. Final tier distribution shows which AI levels survivors converged to. Tier switching indicates how actively agents adapted their AI strategy.",
    fontsize = 8, color = :gray30, halign = :left, valign = :top)

# Footer
timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
Label(fig[8, 1:4],
    "Generated: $timestamp | GlimpseABM Emergent Analysis | N=$N_AGENTS agents × $N_ROUNDS rounds × $N_RUNS runs | Dynamic AI Selection",
    fontsize = 8, color = :gray50, halign = :center)

# Adjust spacing
rowgap!(fig.layout, 1, 5)
rowgap!(fig.layout, 3, 10)
rowgap!(fig.layout, 5, 10)
rowgap!(fig.layout, 7, 5)

# Save PDF
pdf_path = joinpath(OUTPUT_DIR, "emergent_analysis_results.pdf")
save(pdf_path, fig)
println("  Saved: $pdf_path")

# ============================================================================
# SAVE DATA
# ============================================================================

println("\n" * "="^80)
println("SAVING DATA FILES")
println("="^80)

# Summary statistics
summary_df = DataFrame(
    metric = [
        "Overall Survival Rate",
        "Mean Switches/Agent",
        "Final None Share",
        "Final Basic Share",
        "Final Advanced Share",
        "Final Premium Share",
        "2x Rate (Survivors)",
        "5x Rate (Survivors)",
        "10x Rate (Survivors)",
        "Mean AI Trust (Final)"
    ],
    value = [
        overall_survival * 100,
        mean_switches_per_agent,
        final_dist["none"] * 100,
        final_dist["basic"] * 100,
        final_dist["advanced"] * 100,
        final_dist["premium"] * 100,
        rate_2x,
        rate_5x,
        rate_10x,
        mean_trust_traj[end]
    ]
)
CSV.write(joinpath(OUTPUT_DIR, "summary_statistics.csv"), summary_df)
println("  Saved: summary_statistics.csv")

# By-tier statistics
tier_stats = DataFrame(
    tier = TIER_ORDER,
    tier_label = [TIER_LABELS[t] for t in TIER_ORDER],
    survival_mean = surv_means,
    roi_mean = roi_means,
    niches_mean = niche_means,
    final_share = [final_dist[t] * 100 for t in TIER_ORDER]
)
CSV.write(joinpath(OUTPUT_DIR, "tier_statistics.csv"), tier_stats)
println("  Saved: tier_statistics.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

total_time = time() - master_start

println("\n" * "="^80)
println("EMERGENT ANALYSIS COMPLETE")
println("="^80)

println("\nKEY FINDINGS:")
@printf("  1. Overall Survival: %.1f%%\n", overall_survival * 100)
@printf("  2. Mean AI Trust (End): %.3f\n", mean_trust_traj[end])
@printf("  3. Mean Tier Switches/Agent: %.2f\n", mean_switches_per_agent)
println("  4. Final Tier Distribution:")
for t in TIER_ORDER
    @printf("       %s: %.1f%%\n", TIER_LABELS[t], final_dist[t] * 100)
end
@printf("  5. Unicorn (10x) Rate: %.2f%% of survivors\n", rate_10x)

println("\nOUTPUT FILES:")
println("  PDF Report: $pdf_path")
println("  Summary:    $(joinpath(OUTPUT_DIR, "summary_statistics.csv"))")
println("  Tier Stats: $(joinpath(OUTPUT_DIR, "tier_statistics.csv"))")

@printf("\nTotal runtime: %.1f minutes (%.0f seconds)\n", total_time/60, total_time)
println("="^80)
