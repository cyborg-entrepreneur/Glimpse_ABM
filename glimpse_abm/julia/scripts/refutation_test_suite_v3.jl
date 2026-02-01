#!/usr/bin/env julia
"""
COMPREHENSIVE REFUTATION TEST SUITE (V3)

Extended test suite with 30+ refutation conditions testing all potential
mechanisms that could explain or eliminate the AI paradox.

TEST CATEGORIES:
1. BASELINE - Standard model
2. EXECUTION - AI execution success multipliers (2X-10X)
3. QUALITY - AI quality boosts (+10% to +50%)
4. COMBINED - Execution + Quality combinations
5. CROWDING - Competition/crowding dynamics (OFF, 25%, 50%, 75%)
6. COST - AI cost variations (0%, 25%, 50%, 75%)
7. HERDING - Herding behavior (OFF, 25%, 50%)
8. INFORMATION - Information quality/accuracy bonuses
9. OPERATIONS - Operational cost reductions
10. COMBINED FAVORABLE - Multiple advantages combined

Usage:
    julia --threads=auto --project=. scripts/refutation_test_suite_v3.jl
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
const N_RUNS = 30  # Reduced for faster iteration with more tests
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const BASE_SEED = 20260131

const OUTPUT_DIR = joinpath(@__DIR__, "..", "results", "refutation_suite_v3_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")

# ============================================================================
# TEST STRUCTURE
# ============================================================================

struct RefutationTest
    name::String
    description::String
    category::String
    execution_multipliers::Dict{String,Float64}
    quality_boosts::Dict{String,Float64}
    config_overrides::Dict{String,Any}
end

function RefutationTest(
    name::String,
    description::String,
    category::String;
    execution_multipliers::Dict{String,Float64}=Dict("none"=>1.0,"basic"=>1.0,"advanced"=>1.0,"premium"=>1.0),
    quality_boosts::Dict{String,Float64}=Dict("none"=>0.0,"basic"=>0.05,"advanced"=>0.05,"premium"=>0.05),
    config_overrides::Dict{String,Any}=Dict{String,Any}()
)
    RefutationTest(name, description, category, execution_multipliers, quality_boosts, config_overrides)
end

# ============================================================================
# SIMULATION FUNCTION
# ============================================================================

function run_single_simulation(tier::String, run_idx::Int, seed::Int, test::RefutationTest)
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

    # Create simulation
    tier_dist = Dict(t => (t == tier ? 1.0 : 0.0) for t in AI_TIERS)
    sim = GlimpseABM.EmergentSimulation(config=config, initial_tier_distribution=tier_dist, seed=seed)

    # Run simulation
    for round in 1:N_ROUNDS
        GlimpseABM.step!(sim, round)
    end

    # Collect statistics
    alive = filter(a -> a.alive, sim.agents)

    return Dict(
        "tier" => tier,
        "run_idx" => run_idx,
        "survival_rate" => length(alive) / N_AGENTS,
        "survived" => length(alive)
    )
end

function run_condition(test::RefutationTest)
    println("\n📊 [$(test.category)] $(test.name)")
    println("   $(test.description)")

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
        if c % 40 == 0 || c == total_runs
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
        stats[tier] = Dict(
            "survival" => mean(rates),
            "survival_std" => std(rates)
        )
    end
    return stats
end

function compute_treatment_effect(stats::Dict{String, Dict{String, Float64}})
    baseline = stats["none"]["survival"]
    return (stats["premium"]["survival"] - baseline) * 100
end

# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================

function get_comprehensive_test_suite()
    tests = RefutationTest[]

    # ========================================================================
    # 1. BASELINE
    # ========================================================================
    push!(tests, RefutationTest("BASELINE", "Standard model (no modifications)", "BASELINE"))

    # ========================================================================
    # 2. EXECUTION SUCCESS TESTS
    # ========================================================================
    for mult in [2.0, 3.0, 5.0, 7.0, 10.0]
        push!(tests, RefutationTest(
            "EXEC_$(Int(mult))X",
            "Premium AI gets $(Int(mult))x execution success",
            "EXECUTION";
            execution_multipliers=Dict(
                "none" => 1.0,
                "basic" => 1.0 + (mult-1)*0.1,
                "advanced" => 1.0 + (mult-1)*0.25,
                "premium" => mult
            )
        ))
    end

    # ========================================================================
    # 3. QUALITY BOOST TESTS
    # ========================================================================
    for boost in [0.10, 0.20, 0.30, 0.40, 0.50]
        push!(tests, RefutationTest(
            "QUALITY_+$(Int(boost*100))",
            "Premium AI gets +$(Int(boost*100))% quality boost",
            "QUALITY";
            quality_boosts=Dict(
                "none" => 0.0,
                "basic" => 0.05,
                "advanced" => 0.08,
                "premium" => boost
            )
        ))
    end

    # ========================================================================
    # 4. COMBINED ADVANTAGES
    # ========================================================================
    push!(tests, RefutationTest(
        "COMBINED_3X_+20",
        "Premium: 3x execution + 20% quality",
        "COMBINED";
        execution_multipliers=Dict("none"=>1.0,"basic"=>1.15,"advanced"=>1.4,"premium"=>3.0),
        quality_boosts=Dict("none"=>0.0,"basic"=>0.05,"advanced"=>0.10,"premium"=>0.20)
    ))

    push!(tests, RefutationTest(
        "COMBINED_5X_+30",
        "Premium: 5x execution + 30% quality",
        "COMBINED";
        execution_multipliers=Dict("none"=>1.0,"basic"=>1.2,"advanced"=>1.6,"premium"=>5.0),
        quality_boosts=Dict("none"=>0.0,"basic"=>0.08,"advanced"=>0.15,"premium"=>0.30)
    ))

    push!(tests, RefutationTest(
        "EXTREME_10X_+50",
        "Premium: 10x execution + 50% quality (extreme)",
        "COMBINED";
        execution_multipliers=Dict("none"=>1.0,"basic"=>1.5,"advanced"=>2.5,"premium"=>10.0),
        quality_boosts=Dict("none"=>0.0,"basic"=>0.15,"advanced"=>0.30,"premium"=>0.50)
    ))

    # ========================================================================
    # 5. CROWDING/COMPETITION TESTS
    # ========================================================================
    push!(tests, RefutationTest(
        "CROWDING_OFF",
        "Disable all competition/crowding dynamics",
        "CROWDING";
        config_overrides=Dict{String,Any}(
            "DISABLE_COMPETITION_DYNAMICS" => true,
            "COMPETITION_SCALE_FACTOR" => 0.0
        )
    ))

    push!(tests, RefutationTest(
        "CROWDING_25%",
        "Crowding penalties reduced by 75%",
        "CROWDING";
        config_overrides=Dict{String,Any}(
            "COMPETITION_SCALE_FACTOR" => 0.25
        )
    ))

    push!(tests, RefutationTest(
        "CROWDING_50%",
        "Crowding penalties reduced by 50%",
        "CROWDING";
        config_overrides=Dict{String,Any}(
            "COMPETITION_SCALE_FACTOR" => 0.5
        )
    ))

    push!(tests, RefutationTest(
        "CROWDING_75%",
        "Crowding penalties reduced by 25%",
        "CROWDING";
        config_overrides=Dict{String,Any}(
            "COMPETITION_SCALE_FACTOR" => 0.75
        )
    ))

    # ========================================================================
    # 6. AI COST TESTS
    # ========================================================================
    push!(tests, RefutationTest(
        "COST_0%",
        "AI is completely free (no costs)",
        "COST";
        config_overrides=Dict{String,Any}(
            "AI_COST_MULTIPLIER" => 0.0,
            "AI_COST_INTENSITY" => 0.0
        )
    ))

    push!(tests, RefutationTest(
        "COST_25%",
        "AI costs reduced to 25%",
        "COST";
        config_overrides=Dict{String,Any}(
            "AI_COST_MULTIPLIER" => 0.25,
            "AI_COST_INTENSITY" => 0.25
        )
    ))

    push!(tests, RefutationTest(
        "COST_50%",
        "AI costs reduced to 50%",
        "COST";
        config_overrides=Dict{String,Any}(
            "AI_COST_MULTIPLIER" => 0.5,
            "AI_COST_INTENSITY" => 0.5
        )
    ))

    push!(tests, RefutationTest(
        "COST_75%",
        "AI costs reduced to 75%",
        "COST";
        config_overrides=Dict{String,Any}(
            "AI_COST_MULTIPLIER" => 0.75,
            "AI_COST_INTENSITY" => 0.75
        )
    ))

    # ========================================================================
    # 7. HERDING TESTS
    # ========================================================================
    push!(tests, RefutationTest(
        "HERDING_OFF",
        "AI herding completely disabled",
        "HERDING";
        config_overrides=Dict{String,Any}(
            "RECURSION_WEIGHTS" => Dict(
                "crowd_weight" => 0.35,
                "volatility_weight" => 0.30,
                "ai_herd_weight" => 0.0,
                "premium_reuse_weight" => 0.20
            )
        )
    ))

    push!(tests, RefutationTest(
        "HERDING_25%",
        "AI herding reduced to 25%",
        "HERDING";
        config_overrides=Dict{String,Any}(
            "RECURSION_WEIGHTS" => Dict(
                "crowd_weight" => 0.35,
                "volatility_weight" => 0.30,
                "ai_herd_weight" => 0.10,
                "premium_reuse_weight" => 0.20
            )
        )
    ))

    push!(tests, RefutationTest(
        "HERDING_50%",
        "AI herding reduced to 50%",
        "HERDING";
        config_overrides=Dict{String,Any}(
            "RECURSION_WEIGHTS" => Dict(
                "crowd_weight" => 0.35,
                "volatility_weight" => 0.30,
                "ai_herd_weight" => 0.20,
                "premium_reuse_weight" => 0.20
            )
        )
    ))

    # ========================================================================
    # 8. OPERATIONAL COST TESTS
    # ========================================================================
    push!(tests, RefutationTest(
        "OPS_COST_50%",
        "Operational costs halved",
        "OPERATIONS";
        config_overrides=Dict{String,Any}(
            "BASE_OPERATIONAL_COST" => 5000.0  # Half of 10000
        )
    ))

    push!(tests, RefutationTest(
        "OPS_COST_25%",
        "Operational costs reduced to 25%",
        "OPERATIONS";
        config_overrides=Dict{String,Any}(
            "BASE_OPERATIONAL_COST" => 2500.0
        )
    ))

    # ========================================================================
    # 9. COMBINED FAVORABLE CONDITIONS
    # ========================================================================
    push!(tests, RefutationTest(
        "NO_CROWD_FREE_AI",
        "No crowding + Free AI",
        "COMBINED_FAV";
        config_overrides=Dict{String,Any}(
            "DISABLE_COMPETITION_DYNAMICS" => true,
            "COMPETITION_SCALE_FACTOR" => 0.0,
            "AI_COST_MULTIPLIER" => 0.0,
            "AI_COST_INTENSITY" => 0.0
        )
    ))

    push!(tests, RefutationTest(
        "NO_CROWD_5X_EXEC",
        "No crowding + 5x execution",
        "COMBINED_FAV";
        execution_multipliers=Dict("none"=>1.0,"basic"=>1.2,"advanced"=>1.6,"premium"=>5.0),
        config_overrides=Dict{String,Any}(
            "DISABLE_COMPETITION_DYNAMICS" => true,
            "COMPETITION_SCALE_FACTOR" => 0.0
        )
    ))

    push!(tests, RefutationTest(
        "CROWD_50_FREE_AI",
        "50% crowding + Free AI",
        "COMBINED_FAV";
        config_overrides=Dict{String,Any}(
            "COMPETITION_SCALE_FACTOR" => 0.5,
            "AI_COST_MULTIPLIER" => 0.0,
            "AI_COST_INTENSITY" => 0.0
        )
    ))

    push!(tests, RefutationTest(
        "ALL_FAVORABLE",
        "All favorable: no crowd, free AI, 10x exec, +50% quality",
        "COMBINED_FAV";
        execution_multipliers=Dict("none"=>1.0,"basic"=>1.5,"advanced"=>2.5,"premium"=>10.0),
        quality_boosts=Dict("none"=>0.0,"basic"=>0.15,"advanced"=>0.30,"premium"=>0.50),
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
    println("  COMPREHENSIVE REFUTATION TEST SUITE (V3)")
    println("="^80)
    println("\n  Parameters: $N_AGENTS agents, $N_ROUNDS rounds, $N_RUNS runs per condition")
    println("  Threads: $(Threads.nthreads())")
    println("="^80)

    mkpath(OUTPUT_DIR)

    tests = get_comprehensive_test_suite()
    println("\n  Running $(length(tests)) refutation tests...")

    all_results = Dict{String, Dict{String, Any}}()
    master_start = time()

    for (i, test) in enumerate(tests)
        println("\n[$i/$(length(tests))]")
        results = run_condition(test)
        stats = compute_stats(results)
        te = compute_treatment_effect(stats)

        all_results[test.name] = Dict(
            "description" => test.description,
            "category" => test.category,
            "stats" => stats,
            "treatment_effect" => te
        )
    end

    # ========================================================================
    # PRINT SUMMARY
    # ========================================================================

    println("\n" * "="^80)
    println("  REFUTATION TEST RESULTS SUMMARY (V3)")
    println("="^80)

    baseline_te = all_results["BASELINE"]["treatment_effect"]

    # Group by category
    categories = unique([t.category for t in tests])

    for category in categories
        println("\n  [$category]")
        println("-"^70)
        @printf("  %-20s %10s %10s %10s %10s %10s\n",
                "Test", "None", "Premium", "Effect", "Δ Base", "Status")
        println("-"^70)

        for test in filter(t -> t.category == category, tests)
            r = all_results[test.name]
            stats = r["stats"]
            te = r["treatment_effect"]
            delta = te - baseline_te

            status = if te > 1.0
                "✅ REVERSED"
            elseif te > -3.0
                "🟡 NEUTRAL"
            elseif te > baseline_te * 0.6
                "🟠 REDUCED"
            else
                "🔴 PERSISTS"
            end

            @printf("  %-20s %9.1f%% %9.1f%% %+9.1f %+9.1f %12s\n",
                    test.name,
                    stats["none"]["survival"] * 100,
                    stats["premium"]["survival"] * 100,
                    te, delta, status)
        end
    end

    # ========================================================================
    # VERDICT
    # ========================================================================

    println("\n" * "="^80)
    println("  VERDICT")
    println("="^80)

    reversed = [t.name for t in tests if all_results[t.name]["treatment_effect"] > 1.0]
    neutral = [t.name for t in tests if -3.0 < all_results[t.name]["treatment_effect"] <= 1.0]
    reduced = [t.name for t in tests if baseline_te * 0.6 < all_results[t.name]["treatment_effect"] <= -3.0]
    persists = [t.name for t in tests if all_results[t.name]["treatment_effect"] <= baseline_te * 0.6]

    println("\n  ✅ PARADOX REVERSED ($(length(reversed)) tests): $(join(reversed, ", "))")
    println("  🟡 PARADOX NEUTRAL ($(length(neutral)) tests): $(join(neutral, ", "))")
    println("  🟠 PARADOX REDUCED ($(length(reduced)) tests): $(join(reduced, ", "))")
    println("  🔴 PARADOX PERSISTS ($(length(persists)) tests): $(join(persists, ", "))")

    println("\n  BASELINE EFFECT: $(round(baseline_te, digits=1)) pp")

    # Key findings
    println("\n  KEY FINDINGS:")

    crowding_off_te = all_results["CROWDING_OFF"]["treatment_effect"]
    cost_0_te = all_results["COST_0%"]["treatment_effect"]
    all_fav_te = all_results["ALL_FAVORABLE"]["treatment_effect"]

    println("    1. CROWDING_OFF: $(round(crowding_off_te, digits=1)) pp ($(round((1-crowding_off_te/baseline_te)*100, digits=0))% reduction)")
    println("    2. FREE AI:      $(round(cost_0_te, digits=1)) pp ($(round((1-cost_0_te/baseline_te)*100, digits=0))% reduction)")
    println("    3. ALL FAVORABLE: $(round(all_fav_te, digits=1)) pp")

    println("\n" * "="^80)
    @printf("  Total runtime: %.1f minutes\n", (time() - master_start) / 60)
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
            category=test.category,
            description=test.description,
            none_survival=stats["none"]["survival"],
            basic_survival=stats["basic"]["survival"],
            advanced_survival=stats["advanced"]["survival"],
            premium_survival=stats["premium"]["survival"],
            treatment_effect=r["treatment_effect"]
        ))
    end

    df = DataFrame(rows)
    path = joinpath(output_dir, "refutation_suite_v3_summary.csv")
    CSV.write(path, df)
    println("\nResults saved to: $path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
