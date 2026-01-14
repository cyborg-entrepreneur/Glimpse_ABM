#!/usr/bin/env julia
"""
Competition Intensity Robustness Test

This script tests whether the AI information paradox persists when competition
penalties are systematically disabled.

The COMPETITION_INTENSITY parameter scales ALL competition-related mechanisms:
- Crowding penalties on returns
- Failure pressure from crowding
- AI tier concentration penalties
- Herding effects on uncertainty
- Competitive recursion
- Regime shift effects from crowding

By running at COMPETITION_INTENSITY = 0.0, we test whether the paradox is driven
by competition effects or by other mechanisms (e.g., hallucination costs,
overconfidence, reduced novelty).

If the paradox PERSISTS at 0% competition: Competition is NOT the driver
If the paradox DISAPPEARS at 0% competition: Competition IS the driver
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using GlimpseABM
using Random
using Statistics
using Printf
using Dates

# ============================================================================
# CONFIGURATION
# ============================================================================

const N_AGENTS = 1000
const N_ROUNDS = 120
const N_RUNS_PER_TIER = 20  # Fewer runs since we're testing multiple intensities
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const BASE_SEED = 42

# Competition intensity levels to test
const COMPETITION_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

"""
Create a config with the specified competition intensity.
"""
function create_config(competition_intensity::Float64, seed::Int)
    config = EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        COMPETITION_INTENSITY=competition_intensity,
        enable_round_logging=false
    )
    return config
end

"""
Run a single simulation with fixed AI tier.
"""
function run_single_simulation(config::EmergentConfig, tier::String, run_id::String)
    sim = EmergentSimulation(
        config=config,
        output_dir=tempdir(),
        run_id=run_id,
        seed=config.RANDOM_SEED
    )

    # Set fixed AI level for all agents
    for agent in sim.agents
        agent.fixed_ai_level = tier
        agent.current_ai_level = tier
    end

    # Run simulation
    run!(sim)

    # Collect results
    survivors = count(a -> a.alive, sim.agents)
    alive_agents = filter(a -> a.alive, sim.agents)

    return Dict(
        "survivors" => survivors,
        "survival_rate" => survivors / config.N_AGENTS,
        "avg_capital" => !isempty(alive_agents) ? mean(a.resources.capital for a in alive_agents) : 0.0,
        "innovations" => sum(a.innovation_count for a in sim.agents)
    )
end

"""
Calculate effect sizes and statistics.
"""
function calculate_effects(results::Dict{String, Vector{Dict{String,Any}}})
    baseline_rates = [r["survival_rate"] for r in results["none"]]
    baseline_mean = mean(baseline_rates)
    baseline_std = std(baseline_rates)

    effects = Dict{String, Dict{String,Any}}()

    for tier in AI_TIERS
        rates = [r["survival_rate"] for r in results[tier]]
        treatment_mean = mean(rates)
        treatment_std = std(rates)

        # Average Treatment Effect
        ate = treatment_mean - baseline_mean

        # Cohen's d
        pooled_std = sqrt((baseline_std^2 + treatment_std^2) / 2)
        cohens_d = pooled_std > 0 ? ate / pooled_std : 0.0

        effects[tier] = Dict(
            "mean_survival" => treatment_mean,
            "std_survival" => treatment_std,
            "ate" => ate,
            "cohens_d" => cohens_d,
            "n_runs" => length(rates)
        )
    end

    return effects
end

# ============================================================================
# MAIN EXECUTION
# ============================================================================

function main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    output_base = "competition_intensity_sweep_$(timestamp)"
    mkpath(output_base)

    println("=" ^ 80)
    println("COMPETITION INTENSITY ROBUSTNESS TEST")
    println("=" ^ 80)
    println()
    println("Purpose: Test if the AI information paradox persists without competition")
    println()
    println("Configuration:")
    println("  Agents per run: $N_AGENTS")
    println("  Rounds per run: $N_ROUNDS")
    println("  Runs per tier: $N_RUNS_PER_TIER")
    println("  AI tiers: $(join(AI_TIERS, ", "))")
    println("  Competition levels: $(join(COMPETITION_LEVELS, ", "))")
    println()

    total_runs = length(COMPETITION_LEVELS) * length(AI_TIERS) * N_RUNS_PER_TIER
    println("  Total simulations: $total_runs")
    println("  Output directory: $output_base")
    println()

    all_results = Dict{Float64, Dict{String, Dict{String,Any}}}()

    sweep_start = time()

    for (level_idx, competition_level) in enumerate(COMPETITION_LEVELS)
        level_start = time()

        println("=" ^ 80)
        println("[$level_idx/$(length(COMPETITION_LEVELS))] Competition Intensity: $(Int(competition_level * 100))%")
        println("=" ^ 80)
        println()

        level_results = Dict{String, Vector{Dict{String,Any}}}()

        for (tier_idx, tier) in enumerate(AI_TIERS)
            tier_start = time()
            print("  [$tier_idx/$(length(AI_TIERS))] AI Tier: $(uppercase(tier)) ... ")

            level_results[tier] = Dict{String,Any}[]

            for run_idx in 1:N_RUNS_PER_TIER
                seed = BASE_SEED + Int(hash((competition_level, tier, run_idx)) % 10000)
                config = create_config(competition_level, seed)

                run_id = "comp_$(Int(competition_level*100))_$(tier)_run_$(run_idx-1)"
                result = run_single_simulation(config, tier, run_id)
                push!(level_results[tier], result)
            end

            tier_elapsed = time() - tier_start
            survival_rates = [r["survival_rate"] for r in level_results[tier]]
            @printf("done (%.1fs) - Mean survival: %.1f%% ± %.1f%%\n",
                    tier_elapsed, 100*mean(survival_rates), 100*std(survival_rates))
        end

        # Calculate effects for this competition level
        effects = calculate_effects(level_results)
        all_results[competition_level] = effects

        level_elapsed = time() - level_start

        # Print level summary
        println()
        println("  Results at $(Int(competition_level * 100))% competition:")
        println("  " * "-" ^ 60)
        @printf("  %-12s %12s %12s %12s\n", "AI Tier", "Survival", "ATE", "Cohen's d")
        println("  " * "-" ^ 60)

        for tier in AI_TIERS
            e = effects[tier]
            ate_str = tier == "none" ? "(baseline)" : @sprintf("%+.1f pp", 100*e["ate"])
            d_str = tier == "none" ? "-" : @sprintf("%.3f", e["cohens_d"])
            @printf("  %-12s %11.1f%% %12s %12s\n",
                    titlecase(tier), 100*e["mean_survival"], ate_str, d_str)
        end

        @printf("\n  Level time: %.1f seconds\n\n", level_elapsed)
    end

    sweep_elapsed = time() - sweep_start

    # ========================================================================
    # FINAL ANALYSIS: DOES THE PARADOX PERSIST?
    # ========================================================================

    println()
    println("=" ^ 80)
    println("CRITICAL ANALYSIS: Does the Information Paradox Persist?")
    println("=" ^ 80)
    println()

    # Summary table
    println("Average Treatment Effects by Competition Intensity:")
    println("-" ^ 80)
    @printf("%-20s ", "Competition Level")
    for tier in ["basic", "advanced", "premium"]
        @printf("%18s ", "$(titlecase(tier)) ATE")
    end
    println()
    println("-" ^ 80)

    paradox_persists_at_zero = true

    for level in COMPETITION_LEVELS
        effects = all_results[level]
        @printf("%-20s ", "$(Int(level * 100))%")
        for tier in ["basic", "advanced", "premium"]
            ate = effects[tier]["ate"]
            @printf("%+17.1f%% ", 100*ate)

            # Check if paradox persists at 0% competition
            if level == 0.0 && ate >= 0.0
                paradox_persists_at_zero = false
            end
        end
        println()
    end
    println("-" ^ 80)

    # Key finding
    println()
    println("=" ^ 80)
    println("KEY FINDING")
    println("=" ^ 80)
    println()

    # Get effects at 0% and 100% competition
    effects_0 = all_results[0.0]
    effects_100 = all_results[1.0]

    premium_ate_0 = effects_0["premium"]["ate"]
    premium_ate_100 = effects_100["premium"]["ate"]

    if premium_ate_0 < 0
        println("✓ PARADOX PERSISTS at 0% competition intensity")
        println()
        println("  Premium AI ATE at   0% competition: $(@sprintf("%+.1f%%", 100*premium_ate_0))")
        println("  Premium AI ATE at 100% competition: $(@sprintf("%+.1f%%", 100*premium_ate_100))")
        println()
        println("  CONCLUSION: Competition penalties are NOT the primary driver of the paradox.")
        println("  The negative effects of AI persist even without any crowding/herding penalties.")
        println()
        println("  Alternative mechanisms driving the paradox:")
        println("  - AI hallucination costs (false positive recommendations)")
        println("  - Overconfidence from AI signals")
        println("  - Reduced agentic novelty (anchoring on historical patterns)")
        println("  - Opportunity costs from AI subscription fees")
    else
        println("✗ PARADOX DISAPPEARS at 0% competition intensity")
        println()
        println("  Premium AI ATE at   0% competition: $(@sprintf("%+.1f%%", 100*premium_ate_0))")
        println("  Premium AI ATE at 100% competition: $(@sprintf("%+.1f%%", 100*premium_ate_100))")
        println()
        println("  CONCLUSION: Competition penalties ARE the primary driver of the paradox.")
        println("  When crowding/herding effects are removed, AI no longer harms survival.")
    end

    # Calculate how much of the effect is due to competition
    competition_contribution = if premium_ate_0 != 0
        (premium_ate_100 - premium_ate_0) / abs(premium_ate_100) * 100
    else
        100.0
    end

    println()
    println("  Competition contribution to paradox: $(@sprintf("%.0f%%", abs(competition_contribution)))")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    results_file = joinpath(output_base, "competition_intensity_results.csv")
    open(results_file, "w") do f
        println(f, "competition_level,tier,mean_survival,std_survival,ate,cohens_d,n_runs")
        for level in COMPETITION_LEVELS
            for tier in AI_TIERS
                e = all_results[level][tier]
                @printf(f, "%.2f,%s,%.6f,%.6f,%.6f,%.6f,%d\n",
                        level, tier, e["mean_survival"], e["std_survival"],
                        e["ate"], e["cohens_d"], e["n_runs"])
            end
        end
    end
    println()
    println("Results saved to: $results_file")

    # Performance summary
    println()
    println("=" ^ 80)
    println("PERFORMANCE")
    println("=" ^ 80)
    @printf("  Total execution time: %.1f minutes (%.0f seconds)\n", sweep_elapsed/60, sweep_elapsed)
    @printf("  Total simulations: %d\n", total_runs)

    println()
    println("=" ^ 80)
    println("✓ COMPETITION INTENSITY SWEEP COMPLETED")
    println("=" ^ 80)

    return all_results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
