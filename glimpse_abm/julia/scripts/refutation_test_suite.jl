#!/usr/bin/env julia
"""
COMPREHENSIVE REFUTATION TEST SUITE

This suite runs multiple severe tests designed to break the AI paradox finding.
Each test targets a different assumption in the model.

Tests:
1. EXECUTION SUCCESS SCALING (2x, 3x, 4x, 5x)
   - What if AI improves execution success, not just assessment?

2. NICHE QUALITY DISCOVERY
   - What if Premium AI finds/creates higher quality innovations?

3. COMBINED ADVANTAGES
   - What if Premium AI gets both execution AND quality advantages?

Usage:
    julia --threads=auto --project=. scripts/refutation_test_suite.jl
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

const OUTPUT_DIR = joinpath(@__DIR__, "..", "results", "refutation_suite_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")

# ============================================================================
# CORE SIMULATION FUNCTION
# ============================================================================

function run_single_simulation(
    tier::String,
    run_idx::Int,
    seed::Int;
    execution_multipliers::Dict{String,Float64}=Dict(),
    quality_boosts::Dict{String,Float64}=Dict()
)
    config = EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        INITIAL_CAPITAL=5_000_000.0,
        SURVIVAL_THRESHOLD=10_000.0
    )

    # Apply execution success multipliers
    for (t, mult) in execution_multipliers
        config.AI_EXECUTION_SUCCESS_MULTIPLIERS[t] = mult
    end

    # Apply quality boosts
    for (t, boost) in quality_boosts
        config.AI_QUALITY_BOOST[t] = boost
    end

    # Create simulation with all agents at same tier
    tier_dist = Dict(t => (t == tier ? 1.0 : 0.0) for t in AI_TIERS)
    sim = GlimpseABM.EmergentSimulation(
        config=config,
        initial_tier_distribution=tier_dist,
        seed=seed
    )

    # Run simulation
    for round in 1:N_ROUNDS
        GlimpseABM.step!(sim, round)
    end

    # Collect statistics
    alive = filter(a -> a.alive, sim.agents)
    survivor_capitals = [a.resources.capital for a in alive]

    # Innovation metrics
    innovation_count = 0
    innovation_successes = 0
    total_quality = 0.0

    if hasproperty(sim, :innovation_engine) && !isnothing(sim.innovation_engine)
        for (id, inn) in sim.innovation_engine.innovations
            innovation_count += 1
            total_quality += inn.quality
            if something(inn.success, false)
                innovation_successes += 1
            end
        end
    end

    return Dict(
        "tier" => tier,
        "run_idx" => run_idx,
        "survival_rate" => length(alive) / N_AGENTS,
        "survived" => length(alive),
        "mean_capital" => isempty(survivor_capitals) ? 0.0 : mean(survivor_capitals),
        "innovation_count" => innovation_count,
        "innovation_success_rate" => innovation_count > 0 ? innovation_successes / innovation_count : 0.0,
        "avg_quality" => innovation_count > 0 ? total_quality / innovation_count : 0.0
    )
end

function run_condition(
    name::String;
    execution_multipliers::Dict{String,Float64}=Dict(),
    quality_boosts::Dict{String,Float64}=Dict()
)
    println("\n📊 $name")
    println("-"^60)

    results = Dict{String, Vector{Dict}}(tier => Dict[] for tier in AI_TIERS)
    total_runs = N_RUNS * length(AI_TIERS)
    all_results = Vector{Dict}(undef, total_runs)
    completed = Threads.Atomic{Int}(0)
    start_time = time()

    Threads.@threads for idx in 1:total_runs
        tier_idx = ((idx - 1) ÷ N_RUNS) + 1
        run_idx = ((idx - 1) % N_RUNS) + 1
        tier = AI_TIERS[tier_idx]
        seed = BASE_SEED + tier_idx * 1000 + run_idx

        all_results[idx] = run_single_simulation(
            tier, run_idx, seed;
            execution_multipliers=execution_multipliers,
            quality_boosts=quality_boosts
        )

        c = Threads.atomic_add!(completed, 1)
        if c % 50 == 0 || c == total_runs
            @printf("    %d/%d runs (%.1fs)\n", c, total_runs, time() - start_time)
        end
    end

    for result in all_results
        push!(results[result["tier"]], result)
    end

    return results
end

function compute_stats(results::Dict{String, Vector{Dict}})
    stats = Dict{String, Dict{String, Float64}}()
    for tier in AI_TIERS
        rates = [r["survival_rate"] for r in results[tier]]
        qualities = [r["avg_quality"] for r in results[tier]]
        success_rates = [r["innovation_success_rate"] for r in results[tier]]

        stats[tier] = Dict(
            "survival" => mean(rates),
            "survival_std" => std(rates),
            "quality" => mean(qualities),
            "success_rate" => mean(success_rates)
        )
    end
    return stats
end

function compute_treatment_effect(stats::Dict{String, Dict{String, Float64}})
    baseline = stats["none"]["survival"]
    return (stats["premium"]["survival"] - baseline) * 100
end

# ============================================================================
# TEST DEFINITIONS
# ============================================================================

struct RefutationTest
    name::String
    description::String
    execution_multipliers::Dict{String,Float64}
    quality_boosts::Dict{String,Float64}
end

function get_test_suite()
    tests = RefutationTest[]

    # Baseline
    push!(tests, RefutationTest(
        "BASELINE",
        "Standard model (no AI advantages)",
        Dict("none" => 1.0, "basic" => 1.0, "advanced" => 1.0, "premium" => 1.0),
        Dict("none" => 0.0, "basic" => 0.05, "advanced" => 0.05, "premium" => 0.05)
    ))

    # Execution success scaling tests
    for mult in [2.0, 3.0, 4.0, 5.0]
        push!(tests, RefutationTest(
            "EXEC_$(Int(mult))X",
            "Premium AI gets $(Int(mult))x execution success",
            Dict("none" => 1.0, "basic" => 1.0 + (mult-1)*0.1,
                 "advanced" => 1.0 + (mult-1)*0.25, "premium" => mult),
            Dict("none" => 0.0, "basic" => 0.05, "advanced" => 0.05, "premium" => 0.05)
        ))
    end

    # Quality boost tests
    for boost in [0.10, 0.20, 0.30]
        push!(tests, RefutationTest(
            "QUALITY_+$(Int(boost*100))",
            "Premium AI gets +$(Int(boost*100))% quality boost",
            Dict("none" => 1.0, "basic" => 1.0, "advanced" => 1.0, "premium" => 1.0),
            Dict("none" => 0.0, "basic" => 0.05, "advanced" => 0.08, "premium" => boost)
        ))
    end

    # Combined advantages (execution + quality)
    push!(tests, RefutationTest(
        "COMBINED_2X_+20",
        "Premium: 2x execution + 20% quality",
        Dict("none" => 1.0, "basic" => 1.1, "advanced" => 1.25, "premium" => 2.0),
        Dict("none" => 0.0, "basic" => 0.05, "advanced" => 0.10, "premium" => 0.20)
    ))

    push!(tests, RefutationTest(
        "COMBINED_3X_+30",
        "Premium: 3x execution + 30% quality",
        Dict("none" => 1.0, "basic" => 1.15, "advanced" => 1.4, "premium" => 3.0),
        Dict("none" => 0.0, "basic" => 0.08, "advanced" => 0.15, "premium" => 0.30)
    ))

    push!(tests, RefutationTest(
        "EXTREME_5X_+40",
        "Premium: 5x execution + 40% quality (extreme)",
        Dict("none" => 1.0, "basic" => 1.2, "advanced" => 1.6, "premium" => 5.0),
        Dict("none" => 0.0, "basic" => 0.10, "advanced" => 0.20, "premium" => 0.40)
    ))

    return tests
end

# ============================================================================
# MAIN
# ============================================================================

function main()
    println("="^80)
    println("  COMPREHENSIVE REFUTATION TEST SUITE")
    println("="^80)
    println("\n  Testing multiple hypotheses that could break the AI paradox:")
    println("  1. Execution success scaling (2x-5x)")
    println("  2. Innovation quality discovery")
    println("  3. Combined advantages")
    println("\n  Parameters: $N_AGENTS agents, $N_ROUNDS rounds, $N_RUNS runs per condition")
    println("="^80)

    mkpath(OUTPUT_DIR)

    tests = get_test_suite()
    all_results = Dict{String, Dict{String, Any}}()

    for test in tests
        results = run_condition(
            test.name;
            execution_multipliers=test.execution_multipliers,
            quality_boosts=test.quality_boosts
        )
        stats = compute_stats(results)
        te = compute_treatment_effect(stats)

        all_results[test.name] = Dict(
            "description" => test.description,
            "stats" => stats,
            "treatment_effect" => te,
            "raw_results" => results
        )
    end

    # Print summary table
    println("\n" * "="^80)
    println("  REFUTATION TEST RESULTS SUMMARY")
    println("="^80)
    println("\n  Premium AI Treatment Effect (vs No AI baseline)")
    println("-"^80)
    @printf("  %-20s %-35s %12s %12s\n", "Test", "Description", "Effect (pp)", "Status")
    println("-"^80)

    baseline_te = all_results["BASELINE"]["treatment_effect"]

    for test in tests
        te = all_results[test.name]["treatment_effect"]

        status = if te > 1.0
            "✅ REVERSED"
        elseif te > -1.0
            "🟡 ELIMINATED"
        elseif te > baseline_te * 0.5
            "🟠 ATTENUATED"
        else
            "🔴 PERSISTS"
        end

        @printf("  %-20s %-35s %+11.2f  %12s\n",
                test.name, test.description[1:min(35, length(test.description))], te, status)
    end
    println("-"^80)

    # Detailed breakdown
    println("\n  SURVIVAL RATES BY TIER")
    println("-"^80)
    @printf("  %-20s %12s %12s %12s %12s\n", "Test", "None", "Basic", "Advanced", "Premium")
    println("-"^80)

    for test in tests
        stats = all_results[test.name]["stats"]
        @printf("  %-20s %11.1f%% %11.1f%% %11.1f%% %11.1f%%\n",
                test.name,
                stats["none"]["survival"] * 100,
                stats["basic"]["survival"] * 100,
                stats["advanced"]["survival"] * 100,
                stats["premium"]["survival"] * 100)
    end
    println("-"^80)

    # Final verdict
    println("\n" * "="^80)
    println("  VERDICT")
    println("="^80)

    # Check if any test reversed the paradox
    reversed_tests = [t.name for t in tests if all_results[t.name]["treatment_effect"] > 1.0]
    eliminated_tests = [t.name for t in tests if
                        all_results[t.name]["treatment_effect"] > -1.0 &&
                        all_results[t.name]["treatment_effect"] <= 1.0]

    if !isempty(reversed_tests)
        println("\n  🔴 PARADOX REVERSED in $(length(reversed_tests)) test(s):")
        for name in reversed_tests
            println("     - $name: $(all_results[name]["description"])")
        end
        println("\n  The model's predictions depend on specific assumptions.")
    elseif !isempty(eliminated_tests)
        println("\n  🟡 PARADOX ELIMINATED (but not reversed) in $(length(eliminated_tests)) test(s):")
        for name in eliminated_tests
            println("     - $name")
        end
        println("\n  Sufficient AI advantages can neutralize the paradox.")
    else
        println("\n  🟢 PARADOX PERSISTS across ALL $(length(tests)) tests!")
        println("\n  STRONG VALIDATION: The AI paradox is extremely robust.")
        println("  Even with 5x execution success and 40% quality boost,")
        println("  Premium AI still underperforms relative to No AI.")
        println("\n  The behavioral shift mechanism dominates ALL tested advantages.")
    end

    # Find threshold
    println("\n  THRESHOLD ANALYSIS:")
    for test in tests
        te = all_results[test.name]["treatment_effect"]
        if te > baseline_te * 0.5 && te < 0
            println("     - $(test.name): Treatment effect reduced to $(round(te, digits=2)) pp")
            println("       ($(round((1 - te/baseline_te) * 100, digits=0))% attenuation)")
        end
    end

    println("="^80)

    # Save results
    save_results(tests, all_results, OUTPUT_DIR)

    return all_results
end

function save_results(tests, all_results, output_dir)
    # Summary CSV
    rows = []
    for test in tests
        r = all_results[test.name]
        stats = r["stats"]
        push!(rows, (
            test=test.name,
            description=test.description,
            none_survival=stats["none"]["survival"],
            basic_survival=stats["basic"]["survival"],
            advanced_survival=stats["advanced"]["survival"],
            premium_survival=stats["premium"]["survival"],
            treatment_effect=r["treatment_effect"],
            premium_quality=stats["premium"]["quality"],
            premium_success_rate=stats["premium"]["success_rate"]
        ))
    end

    df = DataFrame(rows)
    path = joinpath(output_dir, "refutation_suite_summary.csv")
    CSV.write(path, df)
    println("\nResults saved to: $path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
