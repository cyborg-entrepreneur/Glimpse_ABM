#!/usr/bin/env julia
"""
REFUTATION TEST: Does AI Improve Execution Success?

This test challenges the core assumption of the AI paradox: that AI improves
information quality but NOT execution success rates.

Design:
- BASELINE: All tiers have same execution multiplier (1.0)
- TREATMENT: AI tiers get scaled execution success multipliers
    None:     1.0x (baseline)
    Basic:    1.2x (20% boost)
    Advanced: 1.5x (50% boost)
    Premium:  2.0x (100% boost - double success rate)

Predictions:
- If paradox is ROBUST: Effect should attenuate but may persist
- If paradox BREAKS: Premium AI should have HIGHER survival (paradox reverses)

This is a Popperian "severe test" targeting the model's core mechanism.

Usage:
    julia --threads=auto --project=. scripts/refutation_test_execution_success.jl
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
# TEST PARAMETERS
# ============================================================================

const N_AGENTS = 1000
const N_ROUNDS = 60
const N_RUNS = 50
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const BASE_SEED = 20260130

const OUTPUT_DIR = joinpath(@__DIR__, "..", "results", "refutation_test_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

"""
Create config with specified execution success multipliers.
"""
function create_config(; seed::Int=42, execution_multipliers::Dict{String,Float64}=Dict())
    config = EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        INITIAL_CAPITAL=5_000_000.0,
        SURVIVAL_THRESHOLD=10_000.0
    )

    # Override execution success multipliers if provided
    if !isempty(execution_multipliers)
        for (tier, mult) in execution_multipliers
            config.AI_EXECUTION_SUCCESS_MULTIPLIERS[tier] = mult
        end
    end

    return config
end

"""
Run a single fixed-tier simulation.
"""
function run_single_simulation(
    tier::String,
    run_idx::Int,
    seed::Int;
    execution_multipliers::Dict{String,Float64}=Dict()
)
    config = create_config(seed=seed, execution_multipliers=execution_multipliers)

    # Create all agents with the same fixed tier
    tier_dist = Dict(t => (t == tier ? 1.0 : 0.0) for t in AI_TIERS)
    sim = GlimpseABM.EmergentSimulation(
        config=config,
        initial_tier_distribution=tier_dist,
        seed=seed
    )

    # Track metrics
    action_counts = Dict("invest" => 0, "innovate" => 0, "explore" => 0, "maintain" => 0)

    # Run simulation
    for round in 1:N_ROUNDS
        GlimpseABM.step!(sim, round)
    end

    # Collect final statistics
    alive = filter(a -> a.alive, sim.agents)
    survivor_capitals = [a.resources.capital for a in alive]

    # Count innovation successes
    innovation_attempts = 0
    innovation_successes = 0

    if hasproperty(sim, :innovation_engine) && !isnothing(sim.innovation_engine)
        for (id, inn) in sim.innovation_engine.innovations
            innovation_attempts += 1
            if something(inn.success, false)
                innovation_successes += 1
            end
        end
    end

    return Dict(
        "tier" => tier,
        "run_idx" => run_idx,
        "survived" => length(alive),
        "failed" => N_AGENTS - length(alive),
        "survival_rate" => length(alive) / N_AGENTS,
        "mean_capital" => isempty(survivor_capitals) ? 0.0 : mean(survivor_capitals),
        "median_capital" => isempty(survivor_capitals) ? 0.0 : median(survivor_capitals),
        "innovation_attempts" => innovation_attempts,
        "innovation_successes" => innovation_successes,
        "innovation_success_rate" => innovation_attempts > 0 ? innovation_successes / innovation_attempts : 0.0
    )
end

"""
Run a complete experimental condition (all tiers, all runs).
"""
function run_condition(
    condition_name::String;
    execution_multipliers::Dict{String,Float64}=Dict()
)
    println("\n📊 Running: $condition_name")
    println("-"^60)

    if !isempty(execution_multipliers)
        for tier in AI_TIERS
            mult = get(execution_multipliers, tier, 1.0)
            println("    $tier: $(mult)x execution success")
        end
    else
        println("    All tiers: 1.0x (baseline)")
    end

    results = Dict{String, Vector{Dict}}()
    for tier in AI_TIERS
        results[tier] = Dict[]
    end

    total_runs = N_RUNS * length(AI_TIERS)
    all_results = Vector{Dict}(undef, total_runs)
    completed = Threads.Atomic{Int}(0)
    start_time = time()

    Threads.@threads for idx in 1:total_runs
        tier_idx = ((idx - 1) ÷ N_RUNS) + 1
        run_idx = ((idx - 1) % N_RUNS) + 1
        tier = AI_TIERS[tier_idx]
        seed = BASE_SEED + tier_idx * 1000 + run_idx

        result = run_single_simulation(tier, run_idx, seed;
                                        execution_multipliers=execution_multipliers)
        all_results[idx] = result

        c = Threads.atomic_add!(completed, 1)
        if c % 40 == 0 || c == total_runs
            @printf("    Completed %d/%d runs (%.1fs)\n", c, total_runs, time() - start_time)
        end
    end

    # Organize by tier
    for result in all_results
        push!(results[result["tier"]], result)
    end

    return results
end

"""
Compute summary statistics from results.
"""
function compute_stats(results::Dict{String, Vector{Dict}})
    stats = Dict{String, Dict{String, Float64}}()
    for tier in AI_TIERS
        rates = [r["survival_rate"] for r in results[tier]]
        success_rates = [r["innovation_success_rate"] for r in results[tier]]
        capitals = [r["mean_capital"] for r in results[tier]]

        stats[tier] = Dict(
            "mean_survival" => mean(rates),
            "std_survival" => std(rates),
            "mean_innovation_success" => mean(success_rates),
            "mean_capital" => mean(capitals),
            "n" => Float64(length(rates))
        )
    end
    return stats
end

"""
Print comparison results.
"""
function print_comparison(baseline_stats, treatment_stats)
    println("\n" * "="^80)
    println("  RESULTS COMPARISON")
    println("="^80)

    # Survival rates
    println("\n  SURVIVAL RATES BY CONDITION")
    println("-"^70)
    @printf("  %-12s %15s %15s %15s\n", "Tier", "Baseline", "AI Boost (2x)", "Change")
    println("-"^70)

    for tier in AI_TIERS
        base = baseline_stats[tier]["mean_survival"] * 100
        treat = treatment_stats[tier]["mean_survival"] * 100
        diff = treat - base
        @printf("  %-12s %14.2f%% %14.2f%% %+14.2f pp\n",
                uppercasefirst(tier), base, treat, diff)
    end

    # Treatment effects within each condition
    println("\n  AI PARADOX: Treatment Effect (vs No AI baseline)")
    println("-"^70)
    @printf("  %-12s %15s %15s %18s\n", "Tier", "Baseline TE", "AI Boost TE", "Paradox Status")
    println("-"^70)

    baseline_none = baseline_stats["none"]["mean_survival"]
    treatment_none = treatment_stats["none"]["mean_survival"]

    for tier in ["basic", "advanced", "premium"]
        base_te = (baseline_stats[tier]["mean_survival"] - baseline_none) * 100
        treat_te = (treatment_stats[tier]["mean_survival"] - treatment_none) * 100

        status = if treat_te > 1.0
            "✅ REVERSED"
        elseif treat_te > -1.0
            "🟡 ELIMINATED"
        elseif abs(treat_te) < abs(base_te) * 0.5
            "🟠 ATTENUATED"
        else
            "🔴 PERSISTS"
        end

        @printf("  %-12s %+14.2f pp %+14.2f pp %18s\n",
                uppercasefirst(tier), base_te, treat_te, status)
    end

    # Innovation success rates
    println("\n  INNOVATION SUCCESS RATES")
    println("-"^70)
    @printf("  %-12s %15s %15s %15s\n", "Tier", "Baseline", "AI Boost", "Change")
    println("-"^70)

    for tier in AI_TIERS
        base_isr = baseline_stats[tier]["mean_innovation_success"] * 100
        treat_isr = treatment_stats[tier]["mean_innovation_success"] * 100
        diff = treat_isr - base_isr
        @printf("  %-12s %14.2f%% %14.2f%% %+14.2f pp\n",
                uppercasefirst(tier), base_isr, treat_isr, diff)
    end

    # Final verdict
    println("\n" * "="^80)
    println("  VERDICT")
    println("="^80)

    premium_base_te = (baseline_stats["premium"]["mean_survival"] - baseline_none) * 100
    premium_treat_te = (treatment_stats["premium"]["mean_survival"] - treatment_none) * 100

    if premium_treat_te > 1.0
        println("\n  🔴 PARADOX REVERSED!")
        println("     Premium AI now has HIGHER survival than No AI.")
        println("     The model's prediction critically depends on the assumption")
        println("     that AI does NOT improve execution success rates.")
        println("\n  IMPLICATION: If future AI can improve execution (not just assessment),")
        println("               the AI paradox would not hold.")
    elseif premium_treat_te > -1.0
        println("\n  🟡 PARADOX ELIMINATED")
        println("     Premium AI survival is now statistically equivalent to No AI.")
        println("     A 2x execution success advantage roughly offsets the behavioral risk.")
    elseif abs(premium_treat_te) < abs(premium_base_te) * 0.5
        println("\n  🟠 PARADOX SIGNIFICANTLY ATTENUATED")
        println("     Premium AI still underperforms, but the penalty is halved.")
        println("     Execution success partially compensates for behavioral risk.")
    else
        println("\n  🟢 PARADOX PERSISTS")
        println("     Even with 2x execution success, Premium AI still underperforms.")
        println("     The behavioral shift mechanism DOMINATES the execution advantage.")
        println("\n  STRONG VALIDATION: The model's core insight is robust.")
        println("     Agents with better AI take so much more risk that doubling")
        println("     their success rate cannot compensate.")
    end

    @printf("\n  Premium AI Paradox Effect:")
    @printf("\n    Baseline:  %+.2f pp", premium_base_te)
    @printf("\n    With 2x:   %+.2f pp", premium_treat_te)
    @printf("\n    Change:    %+.2f pp\n", premium_treat_te - premium_base_te)
    println("="^80)
end

"""
Save results to CSV.
"""
function save_results(baseline_results, treatment_results, output_dir)
    mkpath(output_dir)

    all_rows = []
    for (condition, results) in [("baseline", baseline_results), ("ai_boost_2x", treatment_results)]
        for tier in AI_TIERS
            for r in results[tier]
                push!(all_rows, (
                    condition=condition,
                    tier=tier,
                    run_idx=r["run_idx"],
                    survival_rate=r["survival_rate"],
                    survived=r["survived"],
                    failed=r["failed"],
                    mean_capital=r["mean_capital"],
                    innovation_success_rate=r["innovation_success_rate"]
                ))
            end
        end
    end

    df = DataFrame(all_rows)
    path = joinpath(output_dir, "refutation_test_results.csv")
    CSV.write(path, df)
    println("\nResults saved to: $path")
end

# ============================================================================
# MAIN
# ============================================================================

function main()
    println("="^80)
    println("  REFUTATION TEST: AI Execution Success Multiplier")
    println("="^80)
    println("\n  HYPOTHESIS UNDER TEST:")
    println("  The AI paradox depends on AI improving ASSESSMENT but not EXECUTION.")
    println("\n  TEST DESIGN:")
    println("  - Baseline: All tiers have 1.0x execution success (standard model)")
    println("  - Treatment: Premium AI gets 2.0x execution success probability")
    println("\n  PREDICTIONS:")
    println("  - If paradox ROBUST: Premium still underperforms (behavior dominates)")
    println("  - If paradox BREAKS: Premium outperforms (execution compensates)")
    println("="^80)

    mkpath(OUTPUT_DIR)

    # Condition 1: Baseline (standard model)
    baseline_multipliers = Dict(
        "none" => 1.0,
        "basic" => 1.0,
        "advanced" => 1.0,
        "premium" => 1.0
    )
    baseline_results = run_condition("BASELINE (Standard Model)",
                                      execution_multipliers=baseline_multipliers)

    # Condition 2: AI Execution Boost
    treatment_multipliers = Dict(
        "none" => 1.0,
        "basic" => 1.2,
        "advanced" => 1.5,
        "premium" => 2.0
    )
    treatment_results = run_condition("TREATMENT (Premium 2x Execution Success)",
                                       execution_multipliers=treatment_multipliers)

    # Compute statistics
    baseline_stats = compute_stats(baseline_results)
    treatment_stats = compute_stats(treatment_results)

    # Print comparison
    print_comparison(baseline_stats, treatment_stats)

    # Save results
    save_results(baseline_results, treatment_results, OUTPUT_DIR)

    return baseline_stats, treatment_stats
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
