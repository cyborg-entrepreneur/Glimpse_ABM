#!/usr/bin/env julia
"""
EXTENDED REFUTATION TEST SUITE (V2)

Adds crowding, cost, and herding refutation tests to the original suite.

NEW TESTS:
1. CROWDING_OFF - Disable all competition dynamics
2. CROWDING_REDUCED_50 - Halve all crowding penalties
3. ZERO_COST - Make AI free (no subscription/usage costs)
4. HERDING_REDUCED - Reduce AI herding weight by 75%
5. PREMIUM_ANTI_CROWDING - Premium AI gets 50% crowding immunity

Usage:
    julia --threads=auto --project=. scripts/refutation_test_suite_v2.jl
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

const OUTPUT_DIR = joinpath(@__DIR__, "..", "results", "refutation_suite_v2_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")

# ============================================================================
# EXTENDED TEST STRUCTURE
# ============================================================================

struct RefutationTestV2
    name::String
    description::String
    execution_multipliers::Dict{String,Float64}
    quality_boosts::Dict{String,Float64}
    config_overrides::Dict{String,Any}
end

function RefutationTestV2(
    name::String,
    description::String;
    execution_multipliers::Dict{String,Float64}=Dict("none"=>1.0,"basic"=>1.0,"advanced"=>1.0,"premium"=>1.0),
    quality_boosts::Dict{String,Float64}=Dict("none"=>0.0,"basic"=>0.05,"advanced"=>0.05,"premium"=>0.05),
    config_overrides::Dict{String,Any}=Dict{String,Any}()
)
    RefutationTestV2(name, description, execution_multipliers, quality_boosts, config_overrides)
end

# ============================================================================
# CORE SIMULATION FUNCTION (Extended)
# ============================================================================

function run_single_simulation(
    tier::String,
    run_idx::Int,
    seed::Int,
    test::RefutationTestV2
)
    config = EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        INITIAL_CAPITAL=5_000_000.0,
        SURVIVAL_THRESHOLD=10_000.0
    )

    # Apply execution success multipliers
    for (t, mult) in test.execution_multipliers
        config.AI_EXECUTION_SUCCESS_MULTIPLIERS[t] = mult
    end

    # Apply quality boosts
    for (t, boost) in test.quality_boosts
        config.AI_QUALITY_BOOST[t] = boost
    end

    # Apply config overrides
    for (key, value) in test.config_overrides
        key_sym = Symbol(key)
        if hasproperty(config, key_sym)
            setfield!(config, key_sym, value)
        end
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

function run_condition(test::RefutationTestV2)
    println("\n📊 $(test.name)")
    println("   $(test.description)")
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

        all_results[idx] = run_single_simulation(tier, run_idx, seed, test)

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
# EXTENDED TEST SUITE
# ============================================================================

function get_test_suite_v2()
    tests = RefutationTestV2[]

    # ========================================================================
    # ORIGINAL TESTS
    # ========================================================================

    # Baseline
    push!(tests, RefutationTestV2(
        "BASELINE",
        "Standard model (no special AI advantages)"
    ))

    # Execution success scaling tests
    for mult in [2.0, 3.0, 4.0, 5.0]
        push!(tests, RefutationTestV2(
            "EXEC_$(Int(mult))X",
            "Premium AI gets $(Int(mult))x execution success";
            execution_multipliers=Dict(
                "none" => 1.0,
                "basic" => 1.0 + (mult-1)*0.1,
                "advanced" => 1.0 + (mult-1)*0.25,
                "premium" => mult
            )
        ))
    end

    # Quality boost tests
    for boost in [0.10, 0.20, 0.30]
        push!(tests, RefutationTestV2(
            "QUALITY_+$(Int(boost*100))",
            "Premium AI gets +$(Int(boost*100))% quality boost";
            quality_boosts=Dict(
                "none" => 0.0,
                "basic" => 0.05,
                "advanced" => 0.08,
                "premium" => boost
            )
        ))
    end

    # Combined advantages
    push!(tests, RefutationTestV2(
        "COMBINED_2X_+20",
        "Premium: 2x execution + 20% quality";
        execution_multipliers=Dict("none"=>1.0,"basic"=>1.1,"advanced"=>1.25,"premium"=>2.0),
        quality_boosts=Dict("none"=>0.0,"basic"=>0.05,"advanced"=>0.10,"premium"=>0.20)
    ))

    push!(tests, RefutationTestV2(
        "COMBINED_3X_+30",
        "Premium: 3x execution + 30% quality";
        execution_multipliers=Dict("none"=>1.0,"basic"=>1.15,"advanced"=>1.4,"premium"=>3.0),
        quality_boosts=Dict("none"=>0.0,"basic"=>0.08,"advanced"=>0.15,"premium"=>0.30)
    ))

    push!(tests, RefutationTestV2(
        "EXTREME_5X_+40",
        "Premium: 5x execution + 40% quality (extreme)";
        execution_multipliers=Dict("none"=>1.0,"basic"=>1.2,"advanced"=>1.6,"premium"=>5.0),
        quality_boosts=Dict("none"=>0.0,"basic"=>0.10,"advanced"=>0.20,"premium"=>0.40)
    ))

    # ========================================================================
    # NEW CROWDING/COMPETITION TESTS
    # ========================================================================

    # CROWDING_OFF: Disable all competition dynamics
    push!(tests, RefutationTestV2(
        "CROWDING_OFF",
        "Disable all competition/crowding dynamics";
        config_overrides=Dict{String,Any}(
            "DISABLE_COMPETITION_DYNAMICS" => true,
            "COMPETITION_SCALE_FACTOR" => 0.0
        )
    ))

    # CROWDING_REDUCED_50: Halve crowding penalties
    push!(tests, RefutationTestV2(
        "CROWDING_50%",
        "Halve all crowding/competition penalties";
        config_overrides=Dict{String,Any}(
            "COMPETITION_SCALE_FACTOR" => 0.5,
            "RETURN_DEMAND_CROWDING_PENALTY" => 0.24,  # Half of 0.48
            "OPPORTUNITY_COMPETITION_PENALTY" => 0.25  # Half of 0.5
        )
    ))

    # ========================================================================
    # COST TESTS
    # ========================================================================

    # ZERO_COST: Make AI free
    push!(tests, RefutationTestV2(
        "ZERO_COST",
        "AI is free (no subscription or usage costs)";
        config_overrides=Dict{String,Any}(
            "AI_COST_MULTIPLIER" => 0.0,
            "AI_COST_INTENSITY" => 0.0
        )
    ))

    # HALF_COST: Reduce AI costs by 50%
    push!(tests, RefutationTestV2(
        "HALF_COST",
        "AI costs reduced by 50%";
        config_overrides=Dict{String,Any}(
            "AI_COST_MULTIPLIER" => 0.5,
            "AI_COST_INTENSITY" => 0.5
        )
    ))

    # ========================================================================
    # HERDING TESTS
    # ========================================================================

    # HERDING_OFF: Disable AI herding effects
    push!(tests, RefutationTestV2(
        "HERDING_OFF",
        "Disable AI herding weight entirely";
        config_overrides=Dict{String,Any}(
            "RECURSION_WEIGHTS" => Dict(
                "crowd_weight" => 0.35,
                "volatility_weight" => 0.30,
                "ai_herd_weight" => 0.0,  # Was 0.40
                "premium_reuse_weight" => 0.20
            )
        )
    ))

    # HERDING_REDUCED: Reduce herding by 75%
    push!(tests, RefutationTestV2(
        "HERDING_25%",
        "AI herding weight reduced by 75%";
        config_overrides=Dict{String,Any}(
            "RECURSION_WEIGHTS" => Dict(
                "crowd_weight" => 0.35,
                "volatility_weight" => 0.30,
                "ai_herd_weight" => 0.10,  # Was 0.40
                "premium_reuse_weight" => 0.20
            )
        )
    ))

    # ========================================================================
    # COMBINED NEW TESTS
    # ========================================================================

    # Everything favorable: crowding off + zero cost + herding off + 5x exec + 40% quality
    push!(tests, RefutationTestV2(
        "ALL_FAVORABLE",
        "All favorable: no crowd/herd, free AI, 5x exec, +40% quality";
        execution_multipliers=Dict("none"=>1.0,"basic"=>1.2,"advanced"=>1.6,"premium"=>5.0),
        quality_boosts=Dict("none"=>0.0,"basic"=>0.10,"advanced"=>0.20,"premium"=>0.40),
        config_overrides=Dict{String,Any}(
            "DISABLE_COMPETITION_DYNAMICS" => true,
            "COMPETITION_SCALE_FACTOR" => 0.0,
            "AI_COST_MULTIPLIER" => 0.0,
            "AI_COST_INTENSITY" => 0.0,
            "RECURSION_WEIGHTS" => Dict(
                "crowd_weight" => 0.0,
                "volatility_weight" => 0.30,
                "ai_herd_weight" => 0.0,
                "premium_reuse_weight" => 0.0
            )
        )
    ))

    return tests
end

# ============================================================================
# MAIN
# ============================================================================

function main()
    println("="^80)
    println("  EXTENDED REFUTATION TEST SUITE (V2)")
    println("="^80)
    println("\n  NEW TESTS:")
    println("  - Crowding/Competition dynamics")
    println("  - AI Cost elimination")
    println("  - Herding reduction")
    println("  - Combined favorable conditions")
    println("\n  Parameters: $N_AGENTS agents, $N_ROUNDS rounds, $N_RUNS runs per condition")
    println("="^80)

    mkpath(OUTPUT_DIR)

    tests = get_test_suite_v2()
    all_results = Dict{String, Dict{String, Any}}()

    for test in tests
        results = run_condition(test)
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
    println("  REFUTATION TEST RESULTS SUMMARY (V2)")
    println("="^80)
    println("\n  Premium AI Treatment Effect (vs No AI baseline)")
    println("-"^80)
    @printf("  %-20s %-40s %10s %10s\n", "Test", "Description", "Effect", "Status")
    println("-"^80)

    baseline_te = all_results["BASELINE"]["treatment_effect"]

    for test in tests
        te = all_results[test.name]["treatment_effect"]

        status = if te > 1.0
            "✅ REVERSED"
        elseif te > -1.0
            "🟡 NEUTRAL"
        elseif te > baseline_te * 0.5
            "🟠 REDUCED"
        else
            "🔴 PERSISTS"
        end

        desc_short = length(test.description) > 40 ? test.description[1:37] * "..." : test.description
        @printf("  %-20s %-40s %+9.2f  %10s\n", test.name, desc_short, te, status)
    end
    println("-"^80)

    # Survival rates by tier
    println("\n  SURVIVAL RATES BY TIER")
    println("-"^80)
    @printf("  %-20s %10s %10s %10s %10s\n", "Test", "None", "Basic", "Advanced", "Premium")
    println("-"^80)

    for test in tests
        stats = all_results[test.name]["stats"]
        @printf("  %-20s %9.1f%% %9.1f%% %9.1f%% %9.1f%%\n",
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

    reversed = [t.name for t in tests if all_results[t.name]["treatment_effect"] > 1.0]
    neutral = [t.name for t in tests if
               all_results[t.name]["treatment_effect"] > -1.0 &&
               all_results[t.name]["treatment_effect"] <= 1.0]
    reduced = [t.name for t in tests if
               all_results[t.name]["treatment_effect"] > baseline_te * 0.5 &&
               all_results[t.name]["treatment_effect"] <= -1.0]

    if !isempty(reversed)
        println("\n  🔴 PARADOX REVERSED in $(length(reversed)) test(s):")
        for name in reversed
            te = all_results[name]["treatment_effect"]
            println("     - $name: +$(round(te, digits=1)) pp")
        end
    end

    if !isempty(neutral)
        println("\n  🟡 PARADOX NEUTRALIZED in $(length(neutral)) test(s):")
        for name in neutral
            te = all_results[name]["treatment_effect"]
            println("     - $name: $(round(te, digits=1)) pp")
        end
    end

    if !isempty(reduced)
        println("\n  🟠 PARADOX ATTENUATED in $(length(reduced)) test(s):")
        for name in reduced
            te = all_results[name]["treatment_effect"]
            pct_reduction = (1 - te/baseline_te) * 100
            println("     - $name: $(round(te, digits=1)) pp ($(round(pct_reduction, digits=0))% reduction)")
        end
    end

    persists = length(tests) - length(reversed) - length(neutral)
    if persists > 0
        println("\n  🟢 PARADOX PERSISTS in $persists test(s)")
    end

    # Key finding
    all_favorable_te = all_results["ALL_FAVORABLE"]["treatment_effect"]
    println("\n  KEY FINDING:")
    if all_favorable_te > 0
        println("     With ALL advantages combined (no crowding, free AI, no herding,")
        println("     5x execution, +40% quality), Premium AI finally OUTPERFORMS baseline.")
        println("     Effect: +$(round(all_favorable_te, digits=1)) pp")
    elseif all_favorable_te > -1.0
        println("     Even with ALL advantages combined, paradox is only NEUTRALIZED.")
        println("     Effect: $(round(all_favorable_te, digits=1)) pp")
    else
        println("     Even with ALL advantages combined, paradox PERSISTS!")
        println("     Effect: $(round(all_favorable_te, digits=1)) pp")
        println("     This indicates deep structural issues beyond crowding/cost/herding.")
    end

    println("="^80)

    # Save results
    save_results(tests, all_results, OUTPUT_DIR)

    return all_results
end

function save_results(tests, all_results, output_dir)
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
    path = joinpath(output_dir, "refutation_suite_v2_summary.csv")
    CSV.write(path, df)
    println("\nResults saved to: $path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
