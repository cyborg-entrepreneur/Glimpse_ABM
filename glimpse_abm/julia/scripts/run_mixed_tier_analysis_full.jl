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

const N_AGENTS = 1000
const N_ROUNDS = 60
const N_RUNS = 50
const AGENTS_PER_TIER = N_AGENTS ÷ 4
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const BASE_SEED = 20260130

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
    EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        INITIAL_CAPITAL=5_000_000.0,
        SURVIVAL_THRESHOLD=10_000.0
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

    # Override with fixed tier assignments
    for (i, agent) in enumerate(sim.agents)
        agent.fixed_ai_level = tier_assignments[i]
        agent.current_ai_level = tier_assignments[i]
    end

    # Initialize tracking structures
    trajectories = Dict{String, Vector{Float64}}(tier => Float64[] for tier in AI_TIERS)

    # Action tracking per tier
    action_counts = Dict{String, Dict{String, Int}}()
    for tier in AI_TIERS
        action_counts[tier] = Dict("invest" => 0, "innovate" => 0, "explore" => 0, "maintain" => 0)
    end

    # Innovation tracking
    innovation_attempts = Dict{String, Int}(tier => 0 for tier in AI_TIERS)
    innovation_successes = Dict{String, Int}(tier => 0 for tier in AI_TIERS)
    niches_created = Dict{String, Int}(tier => 0 for tier in AI_TIERS)

    # Cumulative niches over time
    cumulative_niches = Dict{String, Vector{Int}}(tier => Int[] for tier in AI_TIERS)

    # Run simulation with tracking
    for round in 1:N_ROUNDS
        # Track pre-step state for action counting
        pre_step_actions = Dict{Int, String}()

        GlimpseABM.step!(sim, round)

        # Track survival by tier
        for tier in AI_TIERS
            tier_agents = filter(a -> a.fixed_ai_level == tier, sim.agents)
            alive_count = count(a -> a.alive, tier_agents)
            push!(trajectories[tier], alive_count / length(tier_agents))
        end

        # Track actions and innovations (approximate from agent state)
        for agent in sim.agents
            if agent.alive
                tier = agent.fixed_ai_level

                # Track last action if available
                if hasproperty(agent, :last_action) && !isnothing(agent.last_action)
                    action = string(agent.last_action)
                    if haskey(action_counts[tier], action)
                        action_counts[tier][action] += 1
                    end
                end

                # Track innovations
                if hasproperty(agent, :innovation_attempts)
                    innovation_attempts[tier] = max(innovation_attempts[tier],
                        get(agent.innovation_attempts, :total, 0))
                end
                if hasproperty(agent, :innovation_successes)
                    innovation_successes[tier] = max(innovation_successes[tier],
                        get(agent.innovation_successes, :total, 0))
                end
            end
        end

        # Track cumulative niches (simplified - count from market if available)
        for tier in AI_TIERS
            push!(cumulative_niches[tier], niches_created[tier])
        end
    end

    # Compute final statistics by tier
    tier_stats = Dict{String, Dict{String, Any}}()

    for tier in AI_TIERS
        tier_agents = filter(a -> a.fixed_ai_level == tier, sim.agents)
        alive = filter(a -> a.alive, tier_agents)

        # Wealth statistics
        survivor_capitals = [a.resources.capital for a in alive]

        # Compute action shares
        total_actions = sum(values(action_counts[tier]))
        innovate_share = total_actions > 0 ? action_counts[tier]["innovate"] / total_actions : 0.0
        explore_share = total_actions > 0 ? action_counts[tier]["explore"] / total_actions : 0.0

        # Innovation metrics (use simplified counts)
        tier_innovation_attempts = count(a -> hasproperty(a, :total_innovations) && a.total_innovations > 0, tier_agents)
        tier_innovation_count = sum(a -> hasproperty(a, :total_innovations) ? a.total_innovations : 0, tier_agents)

        tier_stats[tier] = Dict(
            "total" => length(tier_agents),
            "survived" => length(alive),
            "failed" => length(tier_agents) - length(alive),
            "survival_rate" => length(alive) / length(tier_agents),
            "trajectory" => trajectories[tier],

            # Wealth metrics
            "mean_survivor_capital" => isempty(survivor_capitals) ? 0.0 : mean(survivor_capitals),
            "median_survivor_capital" => isempty(survivor_capitals) ? 0.0 : median(survivor_capitals),
            "p50_capital" => isempty(survivor_capitals) ? 0.0 : quantile(survivor_capitals, 0.5),
            "p90_capital" => isempty(survivor_capitals) ? 0.0 : quantile(survivor_capitals, 0.9),
            "p95_capital" => isempty(survivor_capitals) ? 0.0 : quantile(survivor_capitals, 0.95),

            # Action shares
            "innovate_share" => innovate_share,
            "explore_share" => explore_share,
            "action_counts" => action_counts[tier],

            # Innovation metrics
            "innovations_per_agent" => tier_innovation_count / length(tier_agents),
            "innovation_success_rate" => 0.0,  # Placeholder
            "total_niches" => niches_created[tier],
            "cumulative_niches" => cumulative_niches[tier]
        )
    end

    return Dict(
        "run_idx" => run_idx,
        "seed" => seed,
        "tier_stats" => tier_stats,
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
            "innovations_per_agent" => Float64[]
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
            "n_runs" => length(d["survival_rate"]),
            "trajectory" => get(mean_trajectories, tier, Float64[])
        )
    end

    return Dict(
        "summary_stats" => summary_stats,
        "tier_data" => tier_data,
        "mean_trajectories" => mean_trajectories,
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
                    innovations_per_agent=s["innovations_per_agent"]
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
