#!/usr/bin/env julia
"""
Mixed-Population Fixed-Tier Experiment - Full Analysis Version

Collects comprehensive metrics for Table 3-style visualizations:
- Survival rates and trajectories
- Innovation activity share
- Exploration activity share
- Survivor wealth distributions
- Niche creation
- Innovation volume and success rates
- Knowledge recombination quality

Usage:
    julia --threads=auto --project=. scripts/run_mixed_tier_analysis_full.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GlimpseABM
using Statistics
using Random
using DataFrames
using CSV
using Dates
using Printf

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================

const N_AGENTS = 2000
const N_ROUNDS = 60
const N_RUNS = 50
const AGENTS_PER_TIER = N_AGENTS ÷ 4
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const BASE_SEED = 20260425

const OUTPUT_DIR = joinpath(@__DIR__, "..", "results", "mixed_tier_full_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function create_tier_assignments(n_agents::Int, rng::AbstractRNG)
    assignments = String[]
    agents_per_tier = n_agents ÷ length(AI_TIERS)
    for tier in AI_TIERS
        append!(assignments, fill(tier, agents_per_tier))
    end
    remainder = n_agents - length(assignments)
    for i in 1:remainder
        push!(assignments, AI_TIERS[mod1(i, length(AI_TIERS))])
    end
    shuffle!(rng, assignments)
    return assignments
end

function create_config(; seed::Int=42)
    # Inherit BLS-calibrated defaults: SURVIVAL_THRESHOLD=$2M,
    # INITIAL_CAPITAL_RANGE=($2.5M, $10M), heterogeneous capital + threshold.
    # Match v3.3.4 reval (ARC 5135198) so mixed-tier survival is directly
    # comparable to the single-population baseline (mean 0.540, BLS 50-55%).
    EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        AGENT_AI_MODE="fixed",
    )
end

"""
Run a single mixed-tier simulation with comprehensive metric collection.
"""
function run_single_mixed_simulation(run_idx::Int, seed::Int)
    rng = Random.MersenneTwister(seed)
    config = create_config(seed=seed)
    tier_assignments = create_tier_assignments(N_AGENTS, rng)

    initial_dist = Dict("none" => 0.25, "basic" => 0.25, "advanced" => 0.25, "premium" => 0.25)
    sim = GlimpseABM.EmergentSimulation(
        config=config,
        initial_tier_distribution=initial_dist,
        seed=seed
    )

    # Override with fixed tier assignments.
    # v2.9: when overriding tier after construction, we must reset subscriptions
    # to match the new tier. The constructor started a subscription for each
    # agent's originally-sampled tier; leaving those in place while switching
    # fixed_ai_level corrupts billing (agent pays for tier X while using tier Y).
    for (i, agent) in enumerate(sim.agents)
        old_tier = agent.fixed_ai_level
        new_tier = tier_assignments[i]
        agent.fixed_ai_level = new_tier
        agent.current_ai_level = new_tier
        # Cancel all prior subscriptions; ensure_subscription_schedule! will
        # also cancel non-matching tiers internally, but this call pattern is
        # explicit and safer.
        for tier in collect(keys(agent.subscription_accounts))
            GlimpseABM.cancel_subscription_schedule!(agent, tier)
        end
        if new_tier != "none"
            GlimpseABM.ensure_subscription_schedule!(agent, new_tier)
        end
    end

    # Per-round survival trajectories per tier
    trajectories = Dict{String, Vector{Float64}}(tier => Float64[] for tier in AI_TIERS)

    # Run simulation, tracking survival each round
    for round in 1:N_ROUNDS
        GlimpseABM.step!(sim, round)
        for tier in AI_TIERS
            tier_agents = filter(a -> a.fixed_ai_level == tier, sim.agents)
            alive_count = count(a -> a.alive, tier_agents)
            push!(trajectories[tier], alive_count / length(tier_agents))
        end
    end

    # End-of-run aggregation. Action shares are computed from agent.action_history
    # over ALL agents (alive + dead), not just survivors at the end. Earlier
    # passes used last_action on agents alive after each step!, which excluded
    # actions taken by agents that died that round — biasing shares toward the
    # actions of survivors. Using action_history over all agents fixes that.
    tier_stats = Dict{String, Dict{String, Any}}()

    for tier in AI_TIERS
        tier_agents = filter(a -> a.fixed_ai_level == tier, sim.agents)
        alive = filter(a -> a.alive, tier_agents)

        survivor_capitals = [a.resources.capital for a in alive]

        # Action counts from full action_history (no survivorship bias)
        action_counts = Dict("invest" => 0, "innovate" => 0, "explore" => 0, "maintain" => 0)
        for agent in tier_agents
            for act in agent.action_history
                haskey(action_counts, act) && (action_counts[act] += 1)
            end
        end
        total_actions = sum(values(action_counts))
        innovate_share = total_actions > 0 ? action_counts["innovate"] / total_actions : 0.0
        explore_share = total_actions > 0 ? action_counts["explore"] / total_actions : 0.0

        # Innovation metrics — agent struct fields (v3.3.4+, post-bug-fix)
        tier_innovation_count = sum(a.innovation_count for a in tier_agents)
        tier_innovation_successes = sum(a.innovation_success_count for a in tier_agents)
        tier_innovation_success_rate = tier_innovation_count > 0 ?
            tier_innovation_successes / tier_innovation_count : 0.0

        # Niches discovered — read from agent uncertainty metrics
        tier_total_niches = sum(a.uncertainty_metrics.niches_discovered for a in tier_agents)
        tier_combinations = sum(a.uncertainty_metrics.new_combinations_created for a in tier_agents)

        tier_stats[tier] = Dict(
            "total" => length(tier_agents),
            "survived" => length(alive),
            "failed" => length(tier_agents) - length(alive),
            "survival_rate" => length(alive) / length(tier_agents),
            "trajectory" => trajectories[tier],

            "mean_survivor_capital" => isempty(survivor_capitals) ? 0.0 : mean(survivor_capitals),
            "median_survivor_capital" => isempty(survivor_capitals) ? 0.0 : median(survivor_capitals),
            "p50_capital" => isempty(survivor_capitals) ? 0.0 : quantile(survivor_capitals, 0.5),
            "p90_capital" => isempty(survivor_capitals) ? 0.0 : quantile(survivor_capitals, 0.9),
            "p95_capital" => isempty(survivor_capitals) ? 0.0 : quantile(survivor_capitals, 0.95),

            "innovate_share" => innovate_share,
            "explore_share" => explore_share,
            "action_counts" => action_counts,

            "innovations_per_agent" => tier_innovation_count / length(tier_agents),
            "innovation_successes_per_agent" => tier_innovation_successes / length(tier_agents),
            "innovation_success_rate" => tier_innovation_success_rate,
            "total_niches" => tier_total_niches,
            "niches_per_agent" => tier_total_niches / length(tier_agents),
            "combinations_per_agent" => tier_combinations / length(tier_agents),
        )
    end

    # Capacity-saturation diagnostics — extract from sim.market.opportunities
    # to verify the convex crowding mechanism is active. saturation_max,
    # quantiles, and fraction above K_sat tell us how often the penalty fires.
    K_sat = hasproperty(sim.config, :CROWDING_CAPACITY_RATIO_K) ?
        sim.config.CROWDING_CAPACITY_RATIO_K : 1.5
    live_sats = Float64[]
    for o in sim.market.opportunities
        if o.capacity > 0 && o.total_invested > 0
            push!(live_sats, o.total_invested / o.capacity)
        end
    end
    sat_diag = if isempty(live_sats)
        Dict("n_active" => 0, "max" => 0.0, "p50" => 0.0, "p75" => 0.0,
             "p90" => 0.0, "p95" => 0.0, "p99" => 0.0,
             "frac_above_K" => 0.0, "frac_above_2K" => 0.0)
    else
        Dict(
            "n_active" => length(live_sats),
            "max" => maximum(live_sats),
            "p50" => quantile(live_sats, 0.5),
            "p75" => quantile(live_sats, 0.75),
            "p90" => quantile(live_sats, 0.9),
            "p95" => quantile(live_sats, 0.95),
            "p99" => quantile(live_sats, 0.99),
            "frac_above_K" => count(s -> s > K_sat, live_sats) / length(live_sats),
            "frac_above_2K" => count(s -> s > 2 * K_sat, live_sats) / length(live_sats),
        )
    end

    return Dict(
        "run_idx" => run_idx,
        "seed" => seed,
        "tier_stats" => tier_stats,
        "saturation" => sat_diag,
        "status" => "completed"
    )
end

"""
Aggregate results and compute summary statistics.
"""
function aggregate_results(all_results::Vector{Dict})
    tier_data = Dict{String, Dict{String, Vector{Float64}}}()
    for tier in AI_TIERS
        tier_data[tier] = Dict(
            "survival_rate" => Float64[],
            "mean_capital" => Float64[],
            "p50_capital" => Float64[],
            "p90_capital" => Float64[],
            "p95_capital" => Float64[],
            "innovate_share" => Float64[],
            "explore_share" => Float64[],
            "innovations_per_agent" => Float64[],
            "innovation_successes_per_agent" => Float64[],
            "innovation_success_rate" => Float64[],
            "niches_per_agent" => Float64[],
            "combinations_per_agent" => Float64[]
        )
    end

    # Collect trajectories
    all_trajectories = Dict{String, Vector{Vector{Float64}}}(tier => Vector{Float64}[] for tier in AI_TIERS)

    successful_runs = 0
    for result in all_results
        if result["status"] == "completed"
            successful_runs += 1
            for tier in AI_TIERS
                stats = result["tier_stats"][tier]
                push!(tier_data[tier]["survival_rate"], stats["survival_rate"])
                push!(tier_data[tier]["mean_capital"], stats["mean_survivor_capital"])
                push!(tier_data[tier]["p50_capital"], stats["p50_capital"])
                push!(tier_data[tier]["p90_capital"], stats["p90_capital"])
                push!(tier_data[tier]["p95_capital"], stats["p95_capital"])
                push!(tier_data[tier]["innovate_share"], stats["innovate_share"])
                push!(tier_data[tier]["explore_share"], stats["explore_share"])
                push!(tier_data[tier]["innovations_per_agent"], stats["innovations_per_agent"])
                push!(tier_data[tier]["innovation_successes_per_agent"], stats["innovation_successes_per_agent"])
                push!(tier_data[tier]["innovation_success_rate"], stats["innovation_success_rate"])
                push!(tier_data[tier]["niches_per_agent"], stats["niches_per_agent"])
                push!(tier_data[tier]["combinations_per_agent"], stats["combinations_per_agent"])
                push!(all_trajectories[tier], stats["trajectory"])
            end
        end
    end

    # Compute mean trajectories
    mean_trajectories = Dict{String, Vector{Float64}}()
    for tier in AI_TIERS
        if !isempty(all_trajectories[tier])
            n_rounds = length(all_trajectories[tier][1])
            mean_traj = zeros(n_rounds)
            for traj in all_trajectories[tier]
                mean_traj .+= traj
            end
            mean_traj ./= length(all_trajectories[tier])
            mean_trajectories[tier] = mean_traj
        end
    end

    # Compute summary statistics
    summary_stats = Dict{String, Dict{String, Any}}()
    for tier in AI_TIERS
        d = tier_data[tier]
        summary_stats[tier] = Dict(
            "mean_survival_rate" => mean(d["survival_rate"]),
            "std_survival_rate" => std(d["survival_rate"]),
            "mean_p50_capital" => mean(d["p50_capital"]),
            "mean_p90_capital" => mean(d["p90_capital"]),
            "mean_p95_capital" => mean(d["p95_capital"]),
            "mean_innovate_share" => mean(d["innovate_share"]),
            "mean_explore_share" => mean(d["explore_share"]),
            "mean_innovations_per_agent" => mean(d["innovations_per_agent"]),
            "mean_innovation_successes_per_agent" => mean(d["innovation_successes_per_agent"]),
            "mean_innovation_success_rate" => mean(d["innovation_success_rate"]),
            "mean_niches_per_agent" => mean(d["niches_per_agent"]),
            "mean_combinations_per_agent" => mean(d["combinations_per_agent"]),
            "n_runs" => length(d["survival_rate"]),
            "trajectory" => get(mean_trajectories, tier, Float64[])
        )
    end

    # Aggregate saturation diagnostics across runs
    sat_keys = ["max", "p50", "p75", "p90", "p95", "p99",
                "frac_above_K", "frac_above_2K", "n_active"]
    sat_summary = Dict{String, Float64}()
    for key in sat_keys
        vals = Float64[]
        for r in all_results
            r["status"] == "completed" || continue
            haskey(r, "saturation") || continue
            push!(vals, Float64(r["saturation"][key]))
        end
        sat_summary["mean_" * key] = isempty(vals) ? 0.0 : mean(vals)
        sat_summary["std_" * key] = isempty(vals) ? 0.0 : std(vals)
    end

    return Dict(
        "summary_stats" => summary_stats,
        "tier_data" => tier_data,
        "mean_trajectories" => mean_trajectories,
        "saturation_summary" => sat_summary,
        "successful_runs" => successful_runs
    )
end

"""
Save comprehensive results to CSV files.
"""
function save_results(summary::Dict, all_results::Vector{Dict}, output_dir::String)
    mkpath(output_dir)
    stats = summary["summary_stats"]

    # 1. Summary statistics
    summary_rows = []
    for tier in AI_TIERS
        s = stats[tier]
        push!(summary_rows, (
            tier=tier,
            mean_survival_rate=s["mean_survival_rate"],
            std_survival_rate=s["std_survival_rate"],
            mean_p50_capital=s["mean_p50_capital"],
            mean_p90_capital=s["mean_p90_capital"],
            mean_p95_capital=s["mean_p95_capital"],
            mean_innovate_share=s["mean_innovate_share"],
            mean_explore_share=s["mean_explore_share"],
            mean_innovations_per_agent=s["mean_innovations_per_agent"],
            mean_innovation_successes_per_agent=s["mean_innovation_successes_per_agent"],
            mean_innovation_success_rate=s["mean_innovation_success_rate"],
            mean_niches_per_agent=s["mean_niches_per_agent"],
            mean_combinations_per_agent=s["mean_combinations_per_agent"],
            n_runs=s["n_runs"]
        ))
    end
    CSV.write(joinpath(output_dir, "summary_stats.csv"), DataFrame(summary_rows))

    # 2. Per-run detailed data
    run_rows = []
    for result in all_results
        if result["status"] == "completed"
            for tier in AI_TIERS
                s = result["tier_stats"][tier]
                push!(run_rows, (
                    run_idx=result["run_idx"],
                    tier=tier,
                    survived=s["survived"],
                    failed=s["failed"],
                    total=s["total"],
                    survival_rate=s["survival_rate"],
                    mean_capital=s["mean_survivor_capital"],
                    p50_capital=s["p50_capital"],
                    p90_capital=s["p90_capital"],
                    p95_capital=s["p95_capital"],
                    innovate_share=s["innovate_share"],
                    explore_share=s["explore_share"],
                    innovations_per_agent=s["innovations_per_agent"],
                    innovation_successes_per_agent=s["innovation_successes_per_agent"],
                    innovation_success_rate=s["innovation_success_rate"],
                    niches_per_agent=s["niches_per_agent"],
                    combinations_per_agent=s["combinations_per_agent"],
                ))
            end
        end
    end
    CSV.write(joinpath(output_dir, "per_run_data.csv"), DataFrame(run_rows))

    # 3. Survival trajectories
    trajectories = summary["mean_trajectories"]
    if !isempty(first(values(trajectories)))
        traj_df = DataFrame(
            round=1:N_ROUNDS,
            none=trajectories["none"],
            basic=trajectories["basic"],
            advanced=trajectories["advanced"],
            premium=trajectories["premium"]
        )
        CSV.write(joinpath(output_dir, "survival_trajectories.csv"), traj_df)
    end

    # 4. Treatment effects
    baseline = stats["none"]["mean_survival_rate"]
    effects = [(
        tier=tier,
        survival_rate=stats[tier]["mean_survival_rate"],
        treatment_effect=(stats[tier]["mean_survival_rate"] - baseline) * 100,
        std_error=stats[tier]["std_survival_rate"] / sqrt(stats[tier]["n_runs"])
    ) for tier in AI_TIERS]
    CSV.write(joinpath(output_dir, "treatment_effects.csv"), DataFrame(effects))

    # 5. Saturation diagnostics — single-row summary
    if haskey(summary, "saturation_summary")
        sat = summary["saturation_summary"]
        sat_row = [(
            mean_n_active = get(sat, "mean_n_active", 0.0),
            mean_max = get(sat, "mean_max", 0.0),
            mean_p50 = get(sat, "mean_p50", 0.0),
            mean_p75 = get(sat, "mean_p75", 0.0),
            mean_p90 = get(sat, "mean_p90", 0.0),
            mean_p95 = get(sat, "mean_p95", 0.0),
            mean_p99 = get(sat, "mean_p99", 0.0),
            mean_frac_above_K = get(sat, "mean_frac_above_K", 0.0),
            mean_frac_above_2K = get(sat, "mean_frac_above_2K", 0.0),
            std_max = get(sat, "std_max", 0.0),
            std_p95 = get(sat, "std_p95", 0.0),
            std_frac_above_K = get(sat, "std_frac_above_K", 0.0),
        )]
        CSV.write(joinpath(output_dir, "saturation_diagnostics.csv"), DataFrame(sat_row))
    end

    # 6. Per-run saturation (for distribution plots)
    sat_rows = []
    for r in all_results
        r["status"] == "completed" || continue
        haskey(r, "saturation") || continue
        s = r["saturation"]
        push!(sat_rows, (
            run_idx = r["run_idx"],
            n_active = s["n_active"],
            sat_max = s["max"],
            sat_p50 = s["p50"],
            sat_p75 = s["p75"],
            sat_p90 = s["p90"],
            sat_p95 = s["p95"],
            sat_p99 = s["p99"],
            frac_above_K = s["frac_above_K"],
            frac_above_2K = s["frac_above_2K"],
        ))
    end
    if !isempty(sat_rows)
        CSV.write(joinpath(output_dir, "per_run_saturation.csv"), DataFrame(sat_rows))
    end

    println("Results saved to: $output_dir")
end

"""
Print formatted results summary.
"""
function print_results(summary::Dict)
    stats = summary["summary_stats"]
    baseline = stats["none"]["mean_survival_rate"]

    println("\n" * "="^80)
    println("  MIXED-TIER EXPERIMENT RESULTS (Table 3 Equivalent)")
    println("="^80)

    # Panel A: Survival Rates
    println("\n  A. Final Survival Rates")
    println("-"^60)
    for tier in AI_TIERS
        s = stats[tier]
        @printf("    %-12s: %6.2f%% (±%.2f%%)\n",
                uppercasefirst(tier), s["mean_survival_rate"]*100, s["std_survival_rate"]*100)
    end

    # Panel B: Treatment Effects
    println("\n  B. Treatment Effects vs No AI")
    println("-"^60)
    for tier in ["basic", "advanced", "premium"]
        effect = (stats[tier]["mean_survival_rate"] - baseline) * 100
        @printf("    %-12s: %+.2f percentage points\n", uppercasefirst(tier), effect)
    end

    # Panel D & E: Activity Shares
    println("\n  D. Innovation Activity Share")
    println("-"^60)
    for tier in AI_TIERS
        @printf("    %-12s: %6.2f%%\n", uppercasefirst(tier), stats[tier]["mean_innovate_share"]*100)
    end

    println("\n  E. Exploration Activity Share")
    println("-"^60)
    for tier in AI_TIERS
        @printf("    %-12s: %6.2f%%\n", uppercasefirst(tier), stats[tier]["mean_explore_share"]*100)
    end

    # Panel F: Survivor Wealth
    println("\n  F. Survivor Wealth Percentiles")
    println("-"^60)
    @printf("    %-12s %12s %12s %12s\n", "Tier", "P50", "P90", "P95")
    for tier in AI_TIERS
        s = stats[tier]
        @printf("    %-12s %12.0f %12.0f %12.0f\n",
                uppercasefirst(tier), s["mean_p50_capital"], s["mean_p90_capital"], s["mean_p95_capital"])
    end

    # Panel H: Innovation Volume
    println("\n  H. Innovations Per Agent")
    println("-"^60)
    for tier in AI_TIERS
        @printf("    %-12s: %.3f\n", uppercasefirst(tier), stats[tier]["mean_innovations_per_agent"])
    end

    # Panel I: Capacity-saturation diagnostics
    if haskey(summary, "saturation_summary")
        sat = summary["saturation_summary"]
        println("\n  I. Capacity-Saturation Diagnostics (across active opportunities)")
        println("-"^60)
        @printf("    Active opps per run:    %.0f ± %.0f\n",
                get(sat, "mean_n_active", 0.0), get(sat, "std_n_active", 0.0))
        @printf("    Saturation max:         %.2f ± %.2f\n",
                get(sat, "mean_max", 0.0), get(sat, "std_max", 0.0))
        @printf("    Saturation p95:         %.2f ± %.2f\n",
                get(sat, "mean_p95", 0.0), get(sat, "std_p95", 0.0))
        @printf("    Saturation p50/p75/p90: %.2f / %.2f / %.2f\n",
                get(sat, "mean_p50", 0.0), get(sat, "mean_p75", 0.0),
                get(sat, "mean_p90", 0.0))
        @printf("    Frac above K_sat:       %.1f%% ± %.1f%%\n",
                100*get(sat, "mean_frac_above_K", 0.0),
                100*get(sat, "std_frac_above_K", 0.0))
        @printf("    Frac above 2×K_sat:     %.1f%%\n",
                100*get(sat, "mean_frac_above_2K", 0.0))
    end

    println("\n" * "="^80)
end

# ============================================================================
# MAIN
# ============================================================================

function main()
    println("="^80)
    println("  MIXED-POPULATION FIXED-TIER EXPERIMENT - FULL ANALYSIS")
    println("="^80)
    println("  Agents per run:     $N_AGENTS")
    println("  Agents per tier:    $AGENTS_PER_TIER")
    println("  Number of runs:     $N_RUNS")
    println("  Rounds per run:     $N_ROUNDS")
    println("  Threads available:  $(Threads.nthreads())")
    println("  Output directory:   $OUTPUT_DIR")
    println("="^80)

    mkpath(OUTPUT_DIR)

    println("\nStarting $N_RUNS simulation runs...")
    start_time = time()

    all_results = Vector{Dict}(undef, N_RUNS)
    completed = Threads.Atomic{Int}(0)

    Threads.@threads for run_idx in 1:N_RUNS
        seed = BASE_SEED + run_idx
        try
            all_results[run_idx] = run_single_mixed_simulation(run_idx, seed)
        catch e
            all_results[run_idx] = Dict("run_idx" => run_idx, "status" => "error: $e")
            @warn "Run $run_idx failed: $e"
        end

        c = Threads.atomic_add!(completed, 1)
        if c % 10 == 0 || c == N_RUNS
            @printf("  Completed %d/%d runs (%.1fs elapsed)\n", c, N_RUNS, time() - start_time)
        end
    end

    @printf("\nAll runs complete! Total time: %.1f seconds\n", time() - start_time)

    println("\nAggregating results...")
    summary = aggregate_results(all_results)

    print_results(summary)
    save_results(summary, all_results, OUTPUT_DIR)

    return summary, all_results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
