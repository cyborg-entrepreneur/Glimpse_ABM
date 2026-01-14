#!/usr/bin/env julia
"""
Multi-Mechanism Robustness Test

This script tests which mechanisms drive the AI information paradox by systematically
disabling each mechanism while keeping others active.

MECHANISMS TESTED:
1. COMPETITION_INTENSITY - Crowding, herding, tier concentration penalties
2. HALLUCINATION_INTENSITY - AI misinformation/false positive recommendations
3. OVERCONFIDENCE_INTENSITY - Inflated confidence estimates from low-quality AI
4. AI_NOVELTY_CONSTRAINT_INTENSITY - Premium AI anchoring on historical patterns
5. AI_COST_INTENSITY - Subscription fees and opportunity costs

For each mechanism:
- Run with mechanism at 0% (disabled) and 100% (enabled)
- Compare ATEs to identify which mechanisms drive the paradox

If the paradox PERSISTS when mechanism is disabled: NOT the primary driver
If the paradox DISAPPEARS when mechanism is disabled: IS a primary driver
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
const N_RUNS_PER_TIER = 20
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const BASE_SEED = 42

# Mechanisms to test - each entry is (name, config_key, description)
const MECHANISMS = [
    ("Competition", :COMPETITION_INTENSITY, "Crowding/herding penalties"),
    ("Hallucination", :HALLUCINATION_INTENSITY, "AI misinformation"),
    ("Overconfidence", :OVERCONFIDENCE_INTENSITY, "Inflated confidence"),
    ("NoveltyConstraint", :AI_NOVELTY_CONSTRAINT_INTENSITY, "Historical anchoring"),
    ("AICost", :AI_COST_INTENSITY, "Subscription fees")
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

"""
Create a config with specified mechanism intensities.
"""
function create_config(overrides::Dict{Symbol,Float64}, seed::Int)
    config = EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        enable_round_logging=false
    )

    # Apply overrides
    for (key, value) in overrides
        setproperty!(config, key, value)
    end

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

"""
Run tests for a single mechanism at different intensities.
"""
function test_mechanism(mechanism_name::String, mechanism_key::Symbol, description::String)
    println()
    println("=" ^ 80)
    println("Testing: $mechanism_name ($description)")
    println("=" ^ 80)
    println()

    results = Dict{Float64, Dict{String, Dict{String,Any}}}()

    for intensity in [0.0, 1.0]
        intensity_start = time()
        intensity_label = intensity == 0.0 ? "DISABLED" : "ENABLED"

        println("  [$intensity_label] $mechanism_name at $(Int(intensity * 100))%")

        # Create config with this mechanism at specified intensity
        overrides = Dict{Symbol,Float64}(mechanism_key => intensity)

        level_results = Dict{String, Vector{Dict{String,Any}}}()

        for (tier_idx, tier) in enumerate(AI_TIERS)
            tier_start = time()
            print("    [$tier_idx/$(length(AI_TIERS))] AI Tier: $(uppercase(tier)) ... ")

            level_results[tier] = Dict{String,Any}[]

            for run_idx in 1:N_RUNS_PER_TIER
                seed = BASE_SEED + Int(hash((mechanism_name, intensity, tier, run_idx)) % 10000)
                config = create_config(overrides, seed)

                run_id = "mech_$(mechanism_name)_$(Int(intensity*100))_$(tier)_run_$(run_idx-1)"
                result = run_single_simulation(config, tier, run_id)
                push!(level_results[tier], result)
            end

            tier_elapsed = time() - tier_start
            survival_rates = [r["survival_rate"] for r in level_results[tier]]
            @printf("done (%.1fs) - Mean survival: %.1f%% ± %.1f%%\n",
                    tier_elapsed, 100*mean(survival_rates), 100*std(survival_rates))
        end

        effects = calculate_effects(level_results)
        results[intensity] = effects

        intensity_elapsed = time() - intensity_start
        @printf("    Intensity test time: %.1f seconds\n\n", intensity_elapsed)
    end

    return results
end

# ============================================================================
# MAIN EXECUTION
# ============================================================================

function main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    output_base = "mechanism_robustness_sweep_$(timestamp)"
    mkpath(output_base)

    println("=" ^ 80)
    println("MULTI-MECHANISM ROBUSTNESS TEST")
    println("=" ^ 80)
    println()
    println("Purpose: Identify which mechanisms drive the AI information paradox")
    println()
    println("Configuration:")
    println("  Agents per run: $N_AGENTS")
    println("  Rounds per run: $N_ROUNDS")
    println("  Runs per tier: $N_RUNS_PER_TIER")
    println("  AI tiers: $(join(AI_TIERS, ", "))")
    println("  Mechanisms: $(length(MECHANISMS))")
    println()

    total_runs = length(MECHANISMS) * 2 * length(AI_TIERS) * N_RUNS_PER_TIER
    println("  Total simulations: $total_runs")
    println("  Output directory: $output_base")
    println()

    all_results = Dict{String, Dict{Float64, Dict{String, Dict{String,Any}}}}()

    sweep_start = time()

    for (mech_idx, (name, key, desc)) in enumerate(MECHANISMS)
        println()
        println("[$mech_idx/$(length(MECHANISMS))] MECHANISM: $name")
        results = test_mechanism(name, key, desc)
        all_results[name] = results
    end

    sweep_elapsed = time() - sweep_start

    # ========================================================================
    # SUMMARY ANALYSIS
    # ========================================================================

    println()
    println("=" ^ 80)
    println("MECHANISM CONTRIBUTION ANALYSIS")
    println("=" ^ 80)
    println()

    # Create summary table
    println("Premium AI Treatment Effects by Mechanism State:")
    println("-" ^ 90)
    @printf("%-20s %15s %15s %15s %15s\n",
            "Mechanism", "ATE (0%)", "ATE (100%)", "Δ ATE", "Contribution")
    println("-" ^ 90)

    mechanism_contributions = Dict{String, Float64}()

    for (name, key, desc) in MECHANISMS
        results = all_results[name]
        ate_0 = results[0.0]["premium"]["ate"]
        ate_100 = results[1.0]["premium"]["ate"]
        delta = ate_100 - ate_0

        # Calculate contribution as the portion of the effect explained by this mechanism
        # If ATE at 100% is more negative than at 0%, the mechanism contributes to the paradox
        contribution = if ate_100 != 0
            delta / ate_100 * 100
        else
            0.0
        end
        mechanism_contributions[name] = contribution

        @printf("%-20s %+14.1f%% %+14.1f%% %+14.1f%% %14.0f%%\n",
                name, 100*ate_0, 100*ate_100, 100*delta, abs(contribution))
    end

    println("-" ^ 90)

    # Key findings
    println()
    println("=" ^ 80)
    println("KEY FINDINGS")
    println("=" ^ 80)
    println()

    # Identify mechanisms where paradox persists at 0%
    persists_at_zero = String[]
    disappears_at_zero = String[]

    for (name, key, desc) in MECHANISMS
        ate_0 = all_results[name][0.0]["premium"]["ate"]
        if ate_0 < -0.01  # Negative ATE threshold
            push!(persists_at_zero, name)
        else
            push!(disappears_at_zero, name)
        end
    end

    println("Mechanisms where paradox PERSISTS when disabled (NOT primary drivers):")
    for name in persists_at_zero
        ate = all_results[name][0.0]["premium"]["ate"]
        println("  ✓ $name (Premium ATE = $(@sprintf("%+.1f%%", 100*ate)) at 0%)")
    end

    println()
    println("Mechanisms where paradox DISAPPEARS when disabled (PRIMARY drivers):")
    if isempty(disappears_at_zero)
        println("  (None - paradox persists across all mechanism ablations)")
    else
        for name in disappears_at_zero
            ate = all_results[name][0.0]["premium"]["ate"]
            println("  ✗ $name (Premium ATE = $(@sprintf("%+.1f%%", 100*ate)) at 0%)")
        end
    end

    # Rank mechanisms by contribution
    println()
    println("Mechanism Contribution Ranking (by magnitude of delta):")
    sorted_mechs = sort(collect(mechanism_contributions), by=x->abs(x[2]), rev=true)
    for (rank, (name, contrib)) in enumerate(sorted_mechs)
        println("  $rank. $name: $(@sprintf("%.0f%%", abs(contrib))) contribution")
    end

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    results_file = joinpath(output_base, "mechanism_robustness_results.csv")
    open(results_file, "w") do f
        println(f, "mechanism,intensity,tier,mean_survival,std_survival,ate,cohens_d,n_runs")
        for (name, key, desc) in MECHANISMS
            for intensity in [0.0, 1.0]
                for tier in AI_TIERS
                    e = all_results[name][intensity][tier]
                    @printf(f, "%s,%.2f,%s,%.6f,%.6f,%.6f,%.6f,%d\n",
                            name, intensity, tier, e["mean_survival"], e["std_survival"],
                            e["ate"], e["cohens_d"], e["n_runs"])
                end
            end
        end
    end
    println()
    println("Results saved to: $results_file")

    # Summary file
    summary_file = joinpath(output_base, "mechanism_summary.csv")
    open(summary_file, "w") do f
        println(f, "mechanism,ate_at_0pct,ate_at_100pct,delta_ate,contribution_pct,paradox_persists")
        for (name, key, desc) in MECHANISMS
            ate_0 = all_results[name][0.0]["premium"]["ate"]
            ate_100 = all_results[name][1.0]["premium"]["ate"]
            delta = ate_100 - ate_0
            contrib = mechanism_contributions[name]
            persists = ate_0 < -0.01
            @printf(f, "%s,%.6f,%.6f,%.6f,%.2f,%s\n",
                    name, ate_0, ate_100, delta, contrib, persists ? "yes" : "no")
        end
    end
    println("Summary saved to: $summary_file")

    # Performance summary
    println()
    println("=" ^ 80)
    println("PERFORMANCE")
    println("=" ^ 80)
    @printf("  Total execution time: %.1f minutes (%.0f seconds)\n", sweep_elapsed/60, sweep_elapsed)
    @printf("  Total simulations: %d\n", total_runs)

    println()
    println("=" ^ 80)
    println("✓ MULTI-MECHANISM ROBUSTNESS SWEEP COMPLETED")
    println("=" ^ 80)

    return all_results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
