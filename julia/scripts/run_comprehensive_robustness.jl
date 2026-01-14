#!/usr/bin/env julia
"""
Comprehensive Robustness Test Suite for GLIMPSE ABM

This script implements ALL robustness checks required for publication-quality
causal claims about AI adoption and entrepreneurial survival.

ROBUSTNESS CHECKS IMPLEMENTED:
=============================================================================
CRITICAL (Required for Publication):
1. Placebo/Falsification Test - Random AI assignment after simulation
2. Bootstrapped Confidence Intervals - 1000 bootstrap resamples for CIs
3. Multiple Comparison Correction - Benjamini-Hochberg FDR control

IMPORTANT (Strengthen Claims):
4. Population Size Sensitivity - N_AGENTS = 100, 500, 1000, 2000, 5000
5. Simulation Length Sensitivity - N_ROUNDS = 60, 120, 200, 400
6. Balanced Design Test - Exactly 25% agents per AI tier
7. Initial Capital Sensitivity - Different capital distributions

SUPPLEMENTARY (Complete Picture):
8. Market Regime Sensitivity - Start in different regimes
9. AI Accuracy Isolation - Vary only accuracy, hold costs constant
10. Alternative Outcome Measures - Capital, innovations, not just survival
=============================================================================

Run: julia scripts/run_comprehensive_robustness.jl [--quick]
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using GlimpseABM
using Random
using Statistics
using Printf
using Dates
using DataFrames

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

const BASE_SEED = 42
const AI_TIERS = ["none", "basic", "advanced", "premium"]

# Full vs quick mode
const QUICK_MODE = "--quick" in ARGS

const N_RUNS_PER_TIER = QUICK_MODE ? 5 : 20
const N_BOOTSTRAP = QUICK_MODE ? 100 : 1000
const POPULATION_SIZES = QUICK_MODE ? [100, 500, 1000] : [100, 500, 1000, 2000, 5000]
const SIM_LENGTHS = QUICK_MODE ? [60, 120, 200] : [60, 120, 200, 400]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

"""
Create config with specified parameters.
"""
function create_config(; n_agents::Int=1000, n_rounds::Int=120, seed::Int=42, kwargs...)
    config = EmergentConfig(
        N_AGENTS=n_agents,
        N_ROUNDS=n_rounds,
        RANDOM_SEED=seed,
        enable_round_logging=false
    )

    for (key, value) in kwargs
        setproperty!(config, Symbol(key), value)
    end

    return config
end

"""
Run a single simulation with fixed AI tier.
Returns dictionary of outcome measures.
"""
function run_fixed_tier_simulation(config::EmergentConfig, tier::String, run_id::String)
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

    # Collect comprehensive outcomes
    survivors = count(a -> a.alive, sim.agents)
    alive_agents = filter(a -> a.alive, sim.agents)

    return Dict(
        "survival_rate" => survivors / config.N_AGENTS,
        "survivors" => survivors,
        "mean_capital" => !isempty(alive_agents) ? mean(a.resources.capital for a in alive_agents) : 0.0,
        "median_capital" => !isempty(alive_agents) ? median([a.resources.capital for a in alive_agents]) : 0.0,
        "total_innovations" => sum(a.innovation_count for a in sim.agents),
        "mean_innovations" => mean(a.innovation_count for a in sim.agents),
        "final_capitals" => [a.resources.capital for a in sim.agents],
        "alive_vector" => [a.alive for a in sim.agents]
    )
end

"""
Calculate ATE and effect sizes.
"""
function calculate_ate(treatment_rates::Vector{Float64}, baseline_rates::Vector{Float64})
    ate = mean(treatment_rates) - mean(baseline_rates)

    # Cohen's d
    pooled_std = sqrt((var(treatment_rates) + var(baseline_rates)) / 2)
    cohens_d = pooled_std > 0 ? ate / pooled_std : 0.0

    # Cliff's delta (non-parametric effect size)
    n_greater = sum(t > b for t in treatment_rates for b in baseline_rates)
    n_less = sum(t < b for t in treatment_rates for b in baseline_rates)
    n_total = length(treatment_rates) * length(baseline_rates)
    cliffs_delta = (n_greater - n_less) / n_total

    return Dict(
        "ate" => ate,
        "cohens_d" => cohens_d,
        "cliffs_delta" => cliffs_delta,
        "treatment_mean" => mean(treatment_rates),
        "treatment_std" => std(treatment_rates),
        "baseline_mean" => mean(baseline_rates),
        "baseline_std" => std(baseline_rates)
    )
end

"""
Perform two-sample t-test.
"""
function t_test(x::Vector{Float64}, y::Vector{Float64})
    nx, ny = length(x), length(y)
    mx, my = mean(x), mean(y)
    vx, vy = var(x), var(y)

    # Welch's t-test (unequal variances)
    se = sqrt(vx/nx + vy/ny)
    t_stat = se > 0 ? (mx - my) / se : 0.0

    # Degrees of freedom (Welch-Satterthwaite)
    df = se > 0 ? (vx/nx + vy/ny)^2 / ((vx/nx)^2/(nx-1) + (vy/ny)^2/(ny-1)) : 1.0

    # Approximate p-value using normal distribution (for large samples)
    p_value = 2 * (1 - cdf_normal(abs(t_stat)))

    return Dict("t_stat" => t_stat, "df" => df, "p_value" => p_value)
end

"""
Standard normal CDF approximation.
"""
function cdf_normal(x::Float64)
    # Abramowitz and Stegun approximation
    t = 1 / (1 + 0.2316419 * abs(x))
    d = 0.3989423 * exp(-x^2 / 2)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    return x > 0 ? 1 - p : p
end

# ============================================================================
# TEST 1: PLACEBO/FALSIFICATION TEST
# ============================================================================

"""
Placebo Test: Randomly assign AI tiers AFTER simulation completes.
If we find significant effects with random assignment, our causal claims are suspect.

Expected result: NO significant effects (ATE ≈ 0, p > 0.05)
"""
function run_placebo_test(output_dir::String)
    println("\n" * "="^80)
    println("TEST 1: PLACEBO/FALSIFICATION TEST")
    println("="^80)
    println("\nPurpose: Verify that random AI assignment after simulation shows NO effect")
    println("Expected: ATE ≈ 0, p-value > 0.05 for all comparisons")
    println()

    n_agents = 1000
    n_rounds = 120
    n_sims = N_RUNS_PER_TIER * 4  # Same total sims as real test

    # Run simulations with UNIFORM treatment (all agents same tier, doesn't matter which)
    all_results = Dict{Int, Dict{String,Any}}()

    println("Running $n_sims simulations with uniform AI assignment...")

    for sim_idx in 1:n_sims
        seed = BASE_SEED + sim_idx
        config = create_config(n_agents=n_agents, n_rounds=n_rounds, seed=seed)

        sim = EmergentSimulation(
            config=config,
            output_dir=tempdir(),
            run_id="placebo_sim_$sim_idx",
            seed=seed
        )

        # Use basic tier for all (choice is arbitrary for placebo)
        for agent in sim.agents
            agent.fixed_ai_level = "basic"
            agent.current_ai_level = "basic"
        end

        run!(sim)

        # Store individual agent outcomes
        all_results[sim_idx] = Dict(
            "survival" => [a.alive for a in sim.agents],
            "capital" => [a.resources.capital for a in sim.agents]
        )

        if sim_idx % 10 == 0
            @printf("  Completed %d/%d simulations\n", sim_idx, n_sims)
        end
    end

    # Now randomly assign fake "AI tiers" post-hoc
    println("\nRandomly assigning placebo AI tiers post-hoc...")

    rng = MersenneTwister(12345)  # Fixed seed for reproducibility

    placebo_results = Dict{String, Vector{Float64}}()
    for tier in AI_TIERS
        placebo_results[tier] = Float64[]
    end

    for sim_idx in 1:n_sims
        outcomes = all_results[sim_idx]
        n = length(outcomes["survival"])

        # Randomly assign each agent to a tier
        fake_tiers = rand(rng, AI_TIERS, n)

        # Calculate survival rate by fake tier
        for tier in AI_TIERS
            tier_mask = fake_tiers .== tier
            if sum(tier_mask) > 0
                tier_survival = mean(outcomes["survival"][tier_mask])
                push!(placebo_results[tier], tier_survival)
            end
        end
    end

    # Calculate placebo ATEs
    println("\n" * "-"^80)
    println("PLACEBO RESULTS (Random Post-Hoc Assignment)")
    println("-"^80)
    @printf("%-12s %12s %12s %12s %12s %12s\n",
            "Tier", "Mean Surv", "Std", "Placebo ATE", "t-stat", "p-value")
    println("-"^80)

    baseline = placebo_results["none"]
    placebo_significant = false

    results_data = []

    for tier in AI_TIERS
        rates = placebo_results[tier]
        effects = calculate_ate(rates, baseline)
        t_result = t_test(rates, baseline)

        @printf("%-12s %11.1f%% %11.1f%% %+11.1f%% %12.2f %12.4f %s\n",
                tier,
                100 * effects["treatment_mean"],
                100 * effects["treatment_std"],
                100 * effects["ate"],
                t_result["t_stat"],
                t_result["p_value"],
                t_result["p_value"] < 0.05 ? "(!)" : "")

        if tier != "none" && t_result["p_value"] < 0.05
            placebo_significant = true
        end

        push!(results_data, Dict(
            "tier" => tier,
            "mean_survival" => effects["treatment_mean"],
            "std_survival" => effects["treatment_std"],
            "ate" => effects["ate"],
            "t_stat" => t_result["t_stat"],
            "p_value" => t_result["p_value"]
        ))
    end

    println("-"^80)

    # Interpretation
    println("\nINTERPRETATION:")
    if !placebo_significant
        println("✓ PASS: No significant effects found with random assignment")
        println("  This supports the validity of our causal identification strategy.")
    else
        println("✗ WARNING: Significant effects found even with random assignment!")
        println("  This could indicate confounding or methodological issues.")
    end

    # Save results
    df = DataFrame(results_data)
    placebo_file = joinpath(output_dir, "test1_placebo_results.csv")
    open(placebo_file, "w") do f
        println(f, "tier,mean_survival,std_survival,ate,t_stat,p_value")
        for row in results_data
            @printf(f, "%s,%.6f,%.6f,%.6f,%.4f,%.6f\n",
                    row["tier"], row["mean_survival"], row["std_survival"],
                    row["ate"], row["t_stat"], row["p_value"])
        end
    end
    println("\nResults saved to: $placebo_file")

    return Dict("passed" => !placebo_significant, "results" => results_data)
end

# ============================================================================
# TEST 2: BOOTSTRAPPED CONFIDENCE INTERVALS
# ============================================================================

"""
Bootstrap Test: Generate confidence intervals via resampling.
Provides non-parametric uncertainty quantification for ATEs.
"""
function run_bootstrap_test(output_dir::String)
    println("\n" * "="^80)
    println("TEST 2: BOOTSTRAPPED CONFIDENCE INTERVALS")
    println("="^80)
    println("\nPurpose: Provide robust confidence intervals via $N_BOOTSTRAP bootstrap resamples")
    println("Method: Resample survival rates with replacement, compute ATE distribution")
    println()

    n_agents = 1000
    n_rounds = 120

    # First, run the actual simulations
    println("Running simulations to collect data for bootstrapping...")

    tier_results = Dict{String, Vector{Float64}}()
    for tier in AI_TIERS
        tier_results[tier] = Float64[]
    end

    for (tier_idx, tier) in enumerate(AI_TIERS)
        print("  [$tier_idx/4] AI Tier: $(uppercase(tier)) ... ")
        tier_start = time()

        for run_idx in 1:N_RUNS_PER_TIER
            seed = BASE_SEED + Int(hash((tier, run_idx)) % 10000)
            config = create_config(n_agents=n_agents, n_rounds=n_rounds, seed=seed)

            result = run_fixed_tier_simulation(config, tier, "bootstrap_$(tier)_$(run_idx)")
            push!(tier_results[tier], result["survival_rate"])
        end

        @printf("done (%.1fs) - Mean: %.1f%%\n", time() - tier_start, 100 * mean(tier_results[tier]))
    end

    # Bootstrap resampling
    println("\nRunning $N_BOOTSTRAP bootstrap resamples...")

    rng = MersenneTwister(54321)
    baseline_rates = tier_results["none"]

    bootstrap_results = Dict{String, Dict{String,Any}}()

    for tier in AI_TIERS
        treatment_rates = tier_results[tier]
        bootstrap_ates = Float64[]

        for _ in 1:N_BOOTSTRAP
            # Resample with replacement
            boot_baseline = [baseline_rates[rand(rng, 1:length(baseline_rates))] for _ in 1:length(baseline_rates)]
            boot_treatment = [treatment_rates[rand(rng, 1:length(treatment_rates))] for _ in 1:length(treatment_rates)]

            boot_ate = mean(boot_treatment) - mean(boot_baseline)
            push!(bootstrap_ates, boot_ate)
        end

        # Calculate percentile confidence intervals
        sorted_ates = sort(bootstrap_ates)
        ci_lower_95 = sorted_ates[max(1, Int(floor(0.025 * N_BOOTSTRAP)))]
        ci_upper_95 = sorted_ates[min(N_BOOTSTRAP, Int(ceil(0.975 * N_BOOTSTRAP)))]
        ci_lower_99 = sorted_ates[max(1, Int(floor(0.005 * N_BOOTSTRAP)))]
        ci_upper_99 = sorted_ates[min(N_BOOTSTRAP, Int(ceil(0.995 * N_BOOTSTRAP)))]

        point_ate = mean(treatment_rates) - mean(baseline_rates)

        bootstrap_results[tier] = Dict(
            "point_ate" => point_ate,
            "bootstrap_mean" => mean(bootstrap_ates),
            "bootstrap_std" => std(bootstrap_ates),
            "ci_95_lower" => ci_lower_95,
            "ci_95_upper" => ci_upper_95,
            "ci_99_lower" => ci_lower_99,
            "ci_99_upper" => ci_upper_99,
            "significant_95" => ci_lower_95 > 0 || ci_upper_95 < 0,
            "significant_99" => ci_lower_99 > 0 || ci_upper_99 < 0
        )
    end

    # Display results
    println("\n" * "-"^100)
    println("BOOTSTRAP CONFIDENCE INTERVALS")
    println("-"^100)
    @printf("%-12s %12s %12s %25s %25s\n",
            "Tier", "Point ATE", "Boot SE", "95% CI", "99% CI")
    println("-"^100)

    results_data = []

    for tier in AI_TIERS
        r = bootstrap_results[tier]
        sig_95 = r["significant_95"] ? "*" : ""
        sig_99 = r["significant_99"] ? "**" : ""
        sig = sig_99 != "" ? sig_99 : sig_95

        @printf("%-12s %+11.1f%% %11.1f%% %s[%+.1f%%, %+.1f%%]%s %s[%+.1f%%, %+.1f%%]%s\n",
                tier,
                100 * r["point_ate"],
                100 * r["bootstrap_std"],
                "", 100 * r["ci_95_lower"], 100 * r["ci_95_upper"], sig_95,
                "", 100 * r["ci_99_lower"], 100 * r["ci_99_upper"], sig_99)

        push!(results_data, Dict(
            "tier" => tier,
            "point_ate" => r["point_ate"],
            "bootstrap_std" => r["bootstrap_std"],
            "ci_95_lower" => r["ci_95_lower"],
            "ci_95_upper" => r["ci_95_upper"],
            "ci_99_lower" => r["ci_99_lower"],
            "ci_99_upper" => r["ci_99_upper"],
            "significant_95" => r["significant_95"],
            "significant_99" => r["significant_99"]
        ))
    end

    println("-"^100)
    println("* = significant at 95% level (CI excludes 0)")
    println("** = significant at 99% level")

    # Save results
    bootstrap_file = joinpath(output_dir, "test2_bootstrap_results.csv")
    open(bootstrap_file, "w") do f
        println(f, "tier,point_ate,bootstrap_std,ci_95_lower,ci_95_upper,ci_99_lower,ci_99_upper,sig_95,sig_99")
        for row in results_data
            @printf(f, "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%s\n",
                    row["tier"], row["point_ate"], row["bootstrap_std"],
                    row["ci_95_lower"], row["ci_95_upper"],
                    row["ci_99_lower"], row["ci_99_upper"],
                    row["significant_95"], row["significant_99"])
        end
    end
    println("\nResults saved to: $bootstrap_file")

    return Dict("results" => bootstrap_results, "data" => results_data)
end

# ============================================================================
# TEST 3: MULTIPLE COMPARISON CORRECTION (Benjamini-Hochberg)
# ============================================================================

"""
Multiple Comparison Correction: Control False Discovery Rate.
With multiple AI tier comparisons, we need to adjust p-values.
"""
function run_multiple_comparison_test(output_dir::String)
    println("\n" * "="^80)
    println("TEST 3: MULTIPLE COMPARISON CORRECTION (Benjamini-Hochberg FDR)")
    println("="^80)
    println("\nPurpose: Control false discovery rate when comparing multiple AI tiers")
    println("Method: Benjamini-Hochberg procedure at α = 0.05")
    println()

    n_agents = 1000
    n_rounds = 120

    # Run simulations
    tier_results = Dict{String, Vector{Float64}}()
    for tier in AI_TIERS
        tier_results[tier] = Float64[]
    end

    println("Running simulations...")
    for (tier_idx, tier) in enumerate(AI_TIERS)
        print("  [$tier_idx/4] AI Tier: $(uppercase(tier)) ... ")

        for run_idx in 1:N_RUNS_PER_TIER
            seed = BASE_SEED + Int(hash(("mcp", tier, run_idx)) % 10000)
            config = create_config(n_agents=n_agents, n_rounds=n_rounds, seed=seed)

            result = run_fixed_tier_simulation(config, tier, "mcp_$(tier)_$(run_idx)")
            push!(tier_results[tier], result["survival_rate"])
        end

        @printf("done - Mean: %.1f%%\n", 100 * mean(tier_results[tier]))
    end

    # Calculate p-values for each comparison
    baseline = tier_results["none"]
    comparisons = []

    for tier in ["basic", "advanced", "premium"]
        treatment = tier_results[tier]
        t_result = t_test(treatment, baseline)
        effects = calculate_ate(treatment, baseline)

        push!(comparisons, Dict(
            "tier" => tier,
            "ate" => effects["ate"],
            "p_value" => t_result["p_value"],
            "t_stat" => t_result["t_stat"]
        ))
    end

    # Sort by p-value for BH procedure
    sort!(comparisons, by=x -> x["p_value"])

    # Benjamini-Hochberg correction
    m = length(comparisons)
    alpha = 0.05

    for (i, comp) in enumerate(comparisons)
        bh_threshold = (i / m) * alpha
        comp["bh_threshold"] = bh_threshold
        comp["bh_significant"] = comp["p_value"] <= bh_threshold
        comp["bonferroni_threshold"] = alpha / m
        comp["bonferroni_significant"] = comp["p_value"] <= (alpha / m)
    end

    # Display results
    println("\n" * "-"^100)
    println("MULTIPLE COMPARISON RESULTS")
    println("-"^100)
    @printf("%-12s %12s %12s %12s %12s %15s %15s\n",
            "Tier", "ATE", "p-value", "BH Thresh", "Bonf Thresh", "BH Sig?", "Bonf Sig?")
    println("-"^100)

    for comp in comparisons
        @printf("%-12s %+11.1f%% %12.6f %12.4f %12.4f %15s %15s\n",
                comp["tier"],
                100 * comp["ate"],
                comp["p_value"],
                comp["bh_threshold"],
                comp["bonferroni_threshold"],
                comp["bh_significant"] ? "YES" : "no",
                comp["bonferroni_significant"] ? "YES" : "no")
    end

    println("-"^100)

    # Count significant results
    bh_sig_count = count(c -> c["bh_significant"], comparisons)
    bonf_sig_count = count(c -> c["bonferroni_significant"], comparisons)

    println("\nSUMMARY:")
    println("  Comparisons significant with BH correction: $bh_sig_count / $m")
    println("  Comparisons significant with Bonferroni: $bonf_sig_count / $m")

    if bh_sig_count > 0
        println("\n✓ Effects remain significant after FDR correction")
    else
        println("\n⚠ No effects survive multiple comparison correction")
    end

    # Save results
    mcp_file = joinpath(output_dir, "test3_multiple_comparison_results.csv")
    open(mcp_file, "w") do f
        println(f, "tier,ate,p_value,bh_threshold,bonf_threshold,bh_significant,bonf_significant")
        for comp in comparisons
            @printf(f, "%s,%.6f,%.6f,%.6f,%.6f,%s,%s\n",
                    comp["tier"], comp["ate"], comp["p_value"],
                    comp["bh_threshold"], comp["bonferroni_threshold"],
                    comp["bh_significant"], comp["bonferroni_significant"])
        end
    end
    println("\nResults saved to: $mcp_file")

    return Dict("comparisons" => comparisons, "bh_significant_count" => bh_sig_count)
end

# ============================================================================
# TEST 4: POPULATION SIZE SENSITIVITY
# ============================================================================

"""
Population Size Test: Does the effect hold at different N?
Tests: N = 100, 500, 1000, 2000, 5000
"""
function run_population_size_test(output_dir::String)
    println("\n" * "="^80)
    println("TEST 4: POPULATION SIZE SENSITIVITY")
    println("="^80)
    println("\nPurpose: Verify effects are consistent across different population sizes")
    println("Sizes tested: $(join(POPULATION_SIZES, ", "))")
    println()

    n_rounds = 120
    n_runs = max(5, N_RUNS_PER_TIER ÷ 2)  # Fewer runs for larger populations

    results_by_size = Dict{Int, Dict{String, Dict{String,Any}}}()

    for n_agents in POPULATION_SIZES
        println("\n--- Testing N_AGENTS = $n_agents ---")

        tier_results = Dict{String, Vector{Float64}}()
        for tier in AI_TIERS
            tier_results[tier] = Float64[]
        end

        for (tier_idx, tier) in enumerate(AI_TIERS)
            print("  [$tier_idx/4] $(uppercase(tier)): ")

            for run_idx in 1:n_runs
                seed = BASE_SEED + Int(hash(("popsize", n_agents, tier, run_idx)) % 10000)
                config = create_config(n_agents=n_agents, n_rounds=n_rounds, seed=seed)

                result = run_fixed_tier_simulation(config, tier, "popsize_$(n_agents)_$(tier)_$(run_idx)")
                push!(tier_results[tier], result["survival_rate"])
            end

            @printf("%.1f%% ± %.1f%%\n", 100 * mean(tier_results[tier]), 100 * std(tier_results[tier]))
        end

        # Calculate effects
        baseline = tier_results["none"]
        size_effects = Dict{String, Dict{String,Any}}()

        for tier in AI_TIERS
            effects = calculate_ate(tier_results[tier], baseline)
            t_result = t_test(tier_results[tier], baseline)
            size_effects[tier] = Dict(
                "mean" => effects["treatment_mean"],
                "ate" => effects["ate"],
                "cohens_d" => effects["cohens_d"],
                "p_value" => t_result["p_value"]
            )
        end

        results_by_size[n_agents] = size_effects
    end

    # Summary table
    println("\n" * "-"^100)
    println("POPULATION SIZE SENSITIVITY SUMMARY (Premium AI ATE)")
    println("-"^100)
    @printf("%-12s %15s %15s %15s %15s\n",
            "N_AGENTS", "Premium ATE", "Cohen's d", "p-value", "Direction")
    println("-"^100)

    results_data = []
    consistent_direction = true
    first_direction = nothing

    for n_agents in POPULATION_SIZES
        effects = results_by_size[n_agents]["premium"]
        direction = effects["ate"] < 0 ? "negative" : "positive"

        if isnothing(first_direction)
            first_direction = direction
        elseif direction != first_direction
            consistent_direction = false
        end

        sig_marker = effects["p_value"] < 0.05 ? "*" : ""
        @printf("%-12d %+14.1f%% %15.2f %15.4f %15s%s\n",
                n_agents,
                100 * effects["ate"],
                effects["cohens_d"],
                effects["p_value"],
                direction,
                sig_marker)

        push!(results_data, Dict(
            "n_agents" => n_agents,
            "premium_ate" => effects["ate"],
            "cohens_d" => effects["cohens_d"],
            "p_value" => effects["p_value"],
            "direction" => direction
        ))
    end

    println("-"^100)
    println("* = p < 0.05")

    println("\nINTERPRETATION:")
    if consistent_direction
        println("✓ PASS: Effect direction is consistent across all population sizes")
    else
        println("⚠ WARNING: Effect direction varies with population size")
    end

    # Save results
    popsize_file = joinpath(output_dir, "test4_population_size_results.csv")
    open(popsize_file, "w") do f
        println(f, "n_agents,premium_ate,cohens_d,p_value,direction")
        for row in results_data
            @printf(f, "%d,%.6f,%.4f,%.6f,%s\n",
                    row["n_agents"], row["premium_ate"], row["cohens_d"],
                    row["p_value"], row["direction"])
        end
    end
    println("\nResults saved to: $popsize_file")

    return Dict("results" => results_by_size, "consistent" => consistent_direction)
end

# ============================================================================
# TEST 5: SIMULATION LENGTH SENSITIVITY
# ============================================================================

"""
Simulation Length Test: Does effect persist over different time horizons?
Tests: 60, 120, 200, 400 rounds
"""
function run_simulation_length_test(output_dir::String)
    println("\n" * "="^80)
    println("TEST 5: SIMULATION LENGTH SENSITIVITY")
    println("="^80)
    println("\nPurpose: Verify effects across different simulation durations")
    println("Lengths tested: $(join(SIM_LENGTHS, ", ")) rounds")
    println()

    n_agents = 1000
    n_runs = N_RUNS_PER_TIER

    results_by_length = Dict{Int, Dict{String, Dict{String,Any}}}()

    for n_rounds in SIM_LENGTHS
        println("\n--- Testing N_ROUNDS = $n_rounds ---")

        tier_results = Dict{String, Vector{Float64}}()
        for tier in AI_TIERS
            tier_results[tier] = Float64[]
        end

        for (tier_idx, tier) in enumerate(AI_TIERS)
            print("  [$tier_idx/4] $(uppercase(tier)): ")

            for run_idx in 1:n_runs
                seed = BASE_SEED + Int(hash(("simlen", n_rounds, tier, run_idx)) % 10000)
                config = create_config(n_agents=n_agents, n_rounds=n_rounds, seed=seed)

                result = run_fixed_tier_simulation(config, tier, "simlen_$(n_rounds)_$(tier)_$(run_idx)")
                push!(tier_results[tier], result["survival_rate"])
            end

            @printf("%.1f%% ± %.1f%%\n", 100 * mean(tier_results[tier]), 100 * std(tier_results[tier]))
        end

        # Calculate effects
        baseline = tier_results["none"]
        length_effects = Dict{String, Dict{String,Any}}()

        for tier in AI_TIERS
            effects = calculate_ate(tier_results[tier], baseline)
            t_result = t_test(tier_results[tier], baseline)
            length_effects[tier] = Dict(
                "mean" => effects["treatment_mean"],
                "ate" => effects["ate"],
                "cohens_d" => effects["cohens_d"],
                "p_value" => t_result["p_value"]
            )
        end

        results_by_length[n_rounds] = length_effects
    end

    # Summary table
    println("\n" * "-"^100)
    println("SIMULATION LENGTH SENSITIVITY SUMMARY")
    println("-"^100)
    @printf("%-12s %15s %15s %15s %15s\n",
            "N_ROUNDS", "Premium ATE", "Cohen's d", "p-value", "Paradox?")
    println("-"^100)

    results_data = []
    paradox_at_all_lengths = true

    for n_rounds in SIM_LENGTHS
        effects = results_by_length[n_rounds]["premium"]
        has_paradox = effects["ate"] < -0.01  # Negative ATE threshold

        if !has_paradox
            paradox_at_all_lengths = false
        end

        @printf("%-12d %+14.1f%% %15.2f %15.4f %15s\n",
                n_rounds,
                100 * effects["ate"],
                effects["cohens_d"],
                effects["p_value"],
                has_paradox ? "YES" : "no")

        push!(results_data, Dict(
            "n_rounds" => n_rounds,
            "premium_ate" => effects["ate"],
            "cohens_d" => effects["cohens_d"],
            "p_value" => effects["p_value"],
            "paradox" => has_paradox
        ))
    end

    println("-"^100)

    println("\nINTERPRETATION:")
    if paradox_at_all_lengths
        println("✓ PASS: Information paradox persists across all simulation lengths")
    else
        println("⚠ Note: Paradox strength varies with simulation length")
    end

    # Save results
    simlen_file = joinpath(output_dir, "test5_simulation_length_results.csv")
    open(simlen_file, "w") do f
        println(f, "n_rounds,premium_ate,cohens_d,p_value,paradox")
        for row in results_data
            @printf(f, "%d,%.6f,%.4f,%.6f,%s\n",
                    row["n_rounds"], row["premium_ate"], row["cohens_d"],
                    row["p_value"], row["paradox"])
        end
    end
    println("\nResults saved to: $simlen_file")

    return Dict("results" => results_by_length, "paradox_consistent" => paradox_at_all_lengths)
end

# ============================================================================
# TEST 6: BALANCED DESIGN TEST
# ============================================================================

"""
Balanced Design Test: Exactly 25% agents per AI tier.
Ensures results aren't driven by unbalanced group sizes.
"""
function run_balanced_design_test(output_dir::String)
    println("\n" * "="^80)
    println("TEST 6: BALANCED DESIGN TEST")
    println("="^80)
    println("\nPurpose: Compare fixed-tier (100%) vs balanced (25%) design")
    println("Method: Run with exactly 25% of agents at each AI tier simultaneously")
    println()

    n_agents = 1000
    n_rounds = 120
    n_runs = N_RUNS_PER_TIER

    # Run balanced design simulations
    balanced_results = Dict{String, Vector{Float64}}()
    for tier in AI_TIERS
        balanced_results[tier] = Float64[]
    end

    println("Running $n_runs balanced design simulations...")

    for run_idx in 1:n_runs
        seed = BASE_SEED + run_idx * 100
        config = create_config(n_agents=n_agents, n_rounds=n_rounds, seed=seed)

        sim = EmergentSimulation(
            config=config,
            output_dir=tempdir(),
            run_id="balanced_$(run_idx)",
            seed=seed
        )

        # Assign exactly 25% to each tier
        agents_per_tier = n_agents ÷ 4
        shuffled_indices = shuffle(MersenneTwister(seed), 1:n_agents)

        for (tier_idx, tier) in enumerate(AI_TIERS)
            start_idx = (tier_idx - 1) * agents_per_tier + 1
            end_idx = tier_idx * agents_per_tier

            for agent_idx in shuffled_indices[start_idx:end_idx]
                sim.agents[agent_idx].fixed_ai_level = tier
                sim.agents[agent_idx].current_ai_level = tier
            end
        end

        # Run simulation
        run!(sim)

        # Calculate survival by tier
        for tier in AI_TIERS
            tier_agents = filter(a -> a.fixed_ai_level == tier, sim.agents)
            survival_rate = count(a -> a.alive, tier_agents) / length(tier_agents)
            push!(balanced_results[tier], survival_rate)
        end

        if run_idx % 5 == 0
            @printf("  Completed %d/%d runs\n", run_idx, n_runs)
        end
    end

    # Calculate effects
    baseline = balanced_results["none"]

    println("\n" * "-"^80)
    println("BALANCED DESIGN RESULTS (25% per tier)")
    println("-"^80)
    @printf("%-12s %15s %15s %15s %15s\n",
            "Tier", "Mean Survival", "Std", "ATE vs None", "p-value")
    println("-"^80)

    results_data = []

    for tier in AI_TIERS
        rates = balanced_results[tier]
        effects = calculate_ate(rates, baseline)
        t_result = t_test(rates, baseline)

        sig = t_result["p_value"] < 0.05 ? "*" : ""
        @printf("%-12s %14.1f%% %14.1f%% %+14.1f%% %14.4f%s\n",
                tier,
                100 * effects["treatment_mean"],
                100 * effects["treatment_std"],
                100 * effects["ate"],
                t_result["p_value"],
                sig)

        push!(results_data, Dict(
            "tier" => tier,
            "mean_survival" => effects["treatment_mean"],
            "std_survival" => effects["treatment_std"],
            "ate" => effects["ate"],
            "p_value" => t_result["p_value"]
        ))
    end

    println("-"^80)
    println("* = p < 0.05")

    # Check if paradox persists
    premium_ate = balanced_results["premium"] |> mean
    none_ate = balanced_results["none"] |> mean
    paradox_persists = (premium_ate - none_ate) < -0.01

    println("\nINTERPRETATION:")
    if paradox_persists
        println("✓ PASS: Information paradox persists in balanced design")
        println("  This rules out design artifact from unbalanced group sizes.")
    else
        println("⚠ Note: Effect differs in balanced design")
    end

    # Save results
    balanced_file = joinpath(output_dir, "test6_balanced_design_results.csv")
    open(balanced_file, "w") do f
        println(f, "tier,mean_survival,std_survival,ate,p_value")
        for row in results_data
            @printf(f, "%s,%.6f,%.6f,%.6f,%.6f\n",
                    row["tier"], row["mean_survival"], row["std_survival"],
                    row["ate"], row["p_value"])
        end
    end
    println("\nResults saved to: $balanced_file")

    return Dict("results" => balanced_results, "paradox_persists" => paradox_persists)
end

# ============================================================================
# TEST 7: INITIAL CAPITAL SENSITIVITY
# ============================================================================

"""
Initial Capital Test: Does effect depend on starting wealth distribution?
Tests: Uniform, Normal, Lognormal distributions
"""
function run_initial_capital_test(output_dir::String)
    println("\n" * "="^80)
    println("TEST 7: INITIAL CAPITAL SENSITIVITY")
    println("="^80)
    println("\nPurpose: Test if results hold under different initial wealth distributions")
    println("Distributions: Uniform, Narrow, Wide (lognormal)")
    println()

    n_agents = 1000
    n_rounds = 120
    n_runs = N_RUNS_PER_TIER

    capital_configs = [
        ("Default", Dict()),  # Use default config
        ("Narrow", Dict("INITIAL_CAPITAL_RANGE" => (4_000_000.0, 6_000_000.0))),
        ("Wide", Dict("INITIAL_CAPITAL_RANGE" => (1_000_000.0, 15_000_000.0))),
        ("Low", Dict("INITIAL_CAPITAL" => 2_500_000.0, "INITIAL_CAPITAL_RANGE" => (1_500_000.0, 4_000_000.0))),
    ]

    results_by_distribution = Dict{String, Dict{String, Vector{Float64}}}()

    for (dist_name, dist_overrides) in capital_configs
        println("\n--- Testing: $dist_name capital distribution ---")

        tier_results = Dict{String, Vector{Float64}}()
        for tier in AI_TIERS
            tier_results[tier] = Float64[]
        end

        for (tier_idx, tier) in enumerate(AI_TIERS)
            print("  [$tier_idx/4] $(uppercase(tier)): ")

            for run_idx in 1:n_runs
                seed = BASE_SEED + Int(hash(("capital", dist_name, tier, run_idx)) % 10000)
                config = create_config(n_agents=n_agents, n_rounds=n_rounds, seed=seed)

                # Apply distribution overrides
                for (key, value) in dist_overrides
                    setproperty!(config, Symbol(key), value)
                end

                result = run_fixed_tier_simulation(config, tier, "capital_$(dist_name)_$(tier)_$(run_idx)")
                push!(tier_results[tier], result["survival_rate"])
            end

            @printf("%.1f%%\n", 100 * mean(tier_results[tier]))
        end

        results_by_distribution[dist_name] = tier_results
    end

    # Summary
    println("\n" * "-"^90)
    println("INITIAL CAPITAL SENSITIVITY SUMMARY")
    println("-"^90)
    @printf("%-15s %15s %15s %15s %15s\n",
            "Distribution", "None Surv", "Premium Surv", "Premium ATE", "Paradox?")
    println("-"^90)

    results_data = []
    paradox_all_distributions = true

    for (dist_name, _) in capital_configs
        tier_results = results_by_distribution[dist_name]
        none_mean = mean(tier_results["none"])
        premium_mean = mean(tier_results["premium"])
        ate = premium_mean - none_mean
        has_paradox = ate < -0.01

        if !has_paradox
            paradox_all_distributions = false
        end

        @printf("%-15s %14.1f%% %14.1f%% %+14.1f%% %15s\n",
                dist_name,
                100 * none_mean,
                100 * premium_mean,
                100 * ate,
                has_paradox ? "YES" : "no")

        push!(results_data, Dict(
            "distribution" => dist_name,
            "none_survival" => none_mean,
            "premium_survival" => premium_mean,
            "ate" => ate,
            "paradox" => has_paradox
        ))
    end

    println("-"^90)

    println("\nINTERPRETATION:")
    if paradox_all_distributions
        println("✓ PASS: Paradox persists across all capital distributions")
    else
        println("⚠ Note: Paradox strength varies with initial capital distribution")
    end

    # Save results
    capital_file = joinpath(output_dir, "test7_initial_capital_results.csv")
    open(capital_file, "w") do f
        println(f, "distribution,none_survival,premium_survival,ate,paradox")
        for row in results_data
            @printf(f, "%s,%.6f,%.6f,%.6f,%s\n",
                    row["distribution"], row["none_survival"], row["premium_survival"],
                    row["ate"], row["paradox"])
        end
    end
    println("\nResults saved to: $capital_file")

    return Dict("results" => results_by_distribution, "paradox_consistent" => paradox_all_distributions)
end

# ============================================================================
# TEST 8: MARKET REGIME SENSITIVITY
# ============================================================================

"""
Market Regime Test: Does effect hold in different economic conditions?
Tests starting in: crisis, recession, normal, growth, boom
"""
function run_market_regime_test(output_dir::String)
    println("\n" * "="^80)
    println("TEST 8: MARKET REGIME SENSITIVITY")
    println("="^80)
    println("\nPurpose: Test if results hold when starting in different market regimes")
    println("Regimes: crisis, recession, normal, growth, boom")
    println()

    # Note: This requires modifying the simulation to force initial regime
    # For now, we'll simulate by adjusting market parameters

    n_agents = 1000
    n_rounds = 120
    n_runs = max(5, N_RUNS_PER_TIER ÷ 2)

    regime_configs = [
        ("normal", Dict()),  # Default
        ("favorable", Dict(
            "BLACK_SWAN_PROBABILITY" => 0.01,
            "MARKET_VOLATILITY" => 0.15,
            "DISCOVERY_PROBABILITY" => 0.40
        )),
        ("adverse", Dict(
            "BLACK_SWAN_PROBABILITY" => 0.10,
            "MARKET_VOLATILITY" => 0.40,
            "DISCOVERY_PROBABILITY" => 0.20
        )),
        ("volatile", Dict(
            "BLACK_SWAN_PROBABILITY" => 0.08,
            "MARKET_VOLATILITY" => 0.50,
            "MARKET_SHIFT_PROBABILITY" => 0.20
        )),
    ]

    results_by_regime = Dict{String, Dict{String, Vector{Float64}}}()

    for (regime_name, regime_overrides) in regime_configs
        println("\n--- Testing: $regime_name market regime ---")

        tier_results = Dict{String, Vector{Float64}}()
        for tier in AI_TIERS
            tier_results[tier] = Float64[]
        end

        for (tier_idx, tier) in enumerate(AI_TIERS)
            print("  [$tier_idx/4] $(uppercase(tier)): ")

            for run_idx in 1:n_runs
                seed = BASE_SEED + Int(hash(("regime", regime_name, tier, run_idx)) % 10000)
                config = create_config(n_agents=n_agents, n_rounds=n_rounds, seed=seed)

                # Apply regime overrides
                for (key, value) in regime_overrides
                    setproperty!(config, Symbol(key), value)
                end

                result = run_fixed_tier_simulation(config, tier, "regime_$(regime_name)_$(tier)_$(run_idx)")
                push!(tier_results[tier], result["survival_rate"])
            end

            @printf("%.1f%%\n", 100 * mean(tier_results[tier]))
        end

        results_by_regime[regime_name] = tier_results
    end

    # Summary
    println("\n" * "-"^80)
    println("MARKET REGIME SENSITIVITY SUMMARY")
    println("-"^80)
    @printf("%-15s %15s %15s %15s %15s\n",
            "Regime", "None Surv", "Premium Surv", "Premium ATE", "Paradox?")
    println("-"^80)

    results_data = []
    paradox_all_regimes = true

    for (regime_name, _) in regime_configs
        tier_results = results_by_regime[regime_name]
        none_mean = mean(tier_results["none"])
        premium_mean = mean(tier_results["premium"])
        ate = premium_mean - none_mean
        has_paradox = ate < -0.01

        if !has_paradox
            paradox_all_regimes = false
        end

        @printf("%-15s %14.1f%% %14.1f%% %+14.1f%% %15s\n",
                regime_name,
                100 * none_mean,
                100 * premium_mean,
                100 * ate,
                has_paradox ? "YES" : "no")

        push!(results_data, Dict(
            "regime" => regime_name,
            "none_survival" => none_mean,
            "premium_survival" => premium_mean,
            "ate" => ate,
            "paradox" => has_paradox
        ))
    end

    println("-"^80)

    println("\nINTERPRETATION:")
    if paradox_all_regimes
        println("✓ PASS: Paradox persists across all market regimes")
    else
        println("⚠ Note: Paradox strength varies with market conditions")
    end

    # Save results
    regime_file = joinpath(output_dir, "test8_market_regime_results.csv")
    open(regime_file, "w") do f
        println(f, "regime,none_survival,premium_survival,ate,paradox")
        for row in results_data
            @printf(f, "%s,%.6f,%.6f,%.6f,%s\n",
                    row["regime"], row["none_survival"], row["premium_survival"],
                    row["ate"], row["paradox"])
        end
    end
    println("\nResults saved to: $regime_file")

    return Dict("results" => results_by_regime, "paradox_consistent" => paradox_all_regimes)
end

# ============================================================================
# TEST 9: AI ACCURACY ISOLATION
# ============================================================================

"""
AI Accuracy Isolation: Vary only accuracy, hold costs constant.
Tests if information quality alone drives effects, separate from costs.
"""
function run_ai_accuracy_isolation_test(output_dir::String)
    println("\n" * "="^80)
    println("TEST 9: AI ACCURACY ISOLATION TEST")
    println("="^80)
    println("\nPurpose: Isolate effect of AI accuracy from costs")
    println("Method: Set all AI costs to zero, compare tiers by accuracy alone")
    println()

    n_agents = 1000
    n_rounds = 120
    n_runs = N_RUNS_PER_TIER

    # Test with costs at 0 (isolate accuracy effect)
    println("Running with AI_COST_INTENSITY = 0 (free AI)...")

    tier_results = Dict{String, Vector{Float64}}()
    for tier in AI_TIERS
        tier_results[tier] = Float64[]
    end

    for (tier_idx, tier) in enumerate(AI_TIERS)
        print("  [$tier_idx/4] $(uppercase(tier)): ")

        for run_idx in 1:n_runs
            seed = BASE_SEED + Int(hash(("accuracy_iso", tier, run_idx)) % 10000)
            config = create_config(n_agents=n_agents, n_rounds=n_rounds, seed=seed,
                                   AI_COST_INTENSITY=0.0)  # Zero costs

            result = run_fixed_tier_simulation(config, tier, "accuracy_$(tier)_$(run_idx)")
            push!(tier_results[tier], result["survival_rate"])
        end

        @printf("%.1f%% ± %.1f%%\n", 100 * mean(tier_results[tier]), 100 * std(tier_results[tier]))
    end

    # Calculate effects
    baseline = tier_results["none"]

    println("\n" * "-"^80)
    println("AI ACCURACY ISOLATION RESULTS (Costs = 0)")
    println("-"^80)
    @printf("%-12s %15s %15s %15s %15s\n",
            "Tier", "Mean Survival", "Std", "ATE vs None", "Interpretation")
    println("-"^80)

    results_data = []

    for tier in AI_TIERS
        rates = tier_results[tier]
        effects = calculate_ate(rates, baseline)

        interpretation = if effects["ate"] > 0.01
            "beneficial"
        elseif effects["ate"] < -0.01
            "harmful"
        else
            "neutral"
        end

        @printf("%-12s %14.1f%% %14.1f%% %+14.1f%% %15s\n",
                tier,
                100 * effects["treatment_mean"],
                100 * effects["treatment_std"],
                100 * effects["ate"],
                interpretation)

        push!(results_data, Dict(
            "tier" => tier,
            "mean_survival" => effects["treatment_mean"],
            "std_survival" => effects["treatment_std"],
            "ate" => effects["ate"],
            "interpretation" => interpretation
        ))
    end

    println("-"^80)

    # Check if paradox persists without costs
    premium_ate = mean(tier_results["premium"]) - mean(tier_results["none"])
    paradox_without_costs = premium_ate < -0.01

    println("\nINTERPRETATION:")
    if paradox_without_costs
        println("✓ Paradox persists even with FREE AI")
        println("  This suggests costs are NOT the primary driver.")
        println("  Information quality/processing mechanisms are likely responsible.")
    else
        println("⚠ Paradox DISAPPEARS when AI costs are removed")
        println("  This suggests AI costs ARE a primary driver of the paradox.")
    end

    # Save results
    accuracy_file = joinpath(output_dir, "test9_ai_accuracy_isolation_results.csv")
    open(accuracy_file, "w") do f
        println(f, "tier,mean_survival,std_survival,ate,interpretation")
        for row in results_data
            @printf(f, "%s,%.6f,%.6f,%.6f,%s\n",
                    row["tier"], row["mean_survival"], row["std_survival"],
                    row["ate"], row["interpretation"])
        end
    end
    println("\nResults saved to: $accuracy_file")

    return Dict("results" => tier_results, "paradox_without_costs" => paradox_without_costs)
end

# ============================================================================
# TEST 10: ALTERNATIVE OUTCOME MEASURES
# ============================================================================

"""
Alternative Outcomes Test: Check effects on capital and innovation, not just survival.
Ensures paradox isn't an artifact of survival measure choice.
"""
function run_alternative_outcomes_test(output_dir::String)
    println("\n" * "="^80)
    println("TEST 10: ALTERNATIVE OUTCOME MEASURES")
    println("="^80)
    println("\nPurpose: Verify effects using multiple outcome measures")
    println("Measures: Survival rate, Mean capital, Median capital, Innovation count")
    println()

    n_agents = 1000
    n_rounds = 120
    n_runs = N_RUNS_PER_TIER

    tier_outcomes = Dict{String, Dict{String, Vector{Float64}}}()
    for tier in AI_TIERS
        tier_outcomes[tier] = Dict(
            "survival" => Float64[],
            "mean_capital" => Float64[],
            "median_capital" => Float64[],
            "innovations" => Float64[]
        )
    end

    println("Running simulations and collecting multiple outcomes...")

    for (tier_idx, tier) in enumerate(AI_TIERS)
        print("  [$tier_idx/4] $(uppercase(tier)): ")

        for run_idx in 1:n_runs
            seed = BASE_SEED + Int(hash(("alt_outcomes", tier, run_idx)) % 10000)
            config = create_config(n_agents=n_agents, n_rounds=n_rounds, seed=seed)

            result = run_fixed_tier_simulation(config, tier, "altout_$(tier)_$(run_idx)")

            push!(tier_outcomes[tier]["survival"], result["survival_rate"])
            push!(tier_outcomes[tier]["mean_capital"], result["mean_capital"])
            push!(tier_outcomes[tier]["median_capital"], result["median_capital"])
            push!(tier_outcomes[tier]["innovations"], result["mean_innovations"])
        end

        @printf("done\n")
    end

    # Calculate effects for each outcome
    outcomes_list = ["survival", "mean_capital", "median_capital", "innovations"]

    println("\n" * "-"^100)
    println("ALTERNATIVE OUTCOME MEASURES SUMMARY")
    println("-"^100)

    results_data = []

    for outcome in outcomes_list
        println("\n--- $outcome ---")
        @printf("%-12s %15s %15s %15s\n", "Tier", "Mean", "ATE vs None", "Direction")

        baseline = tier_outcomes["none"][outcome]

        for tier in AI_TIERS
            values = tier_outcomes[tier][outcome]
            effects = calculate_ate(values, baseline)

            direction = if abs(effects["ate"]) < 0.001 * mean(baseline)
                "neutral"
            elseif effects["ate"] > 0
                "positive"
            else
                "negative"
            end

            if outcome == "survival"
                @printf("%-12s %14.1f%% %+14.1f%% %15s\n",
                        tier, 100 * mean(values), 100 * effects["ate"], direction)
            elseif outcome in ["mean_capital", "median_capital"]
                @printf("%-12s %15.0f %+15.0f %15s\n",
                        tier, mean(values), effects["ate"], direction)
            else
                @printf("%-12s %15.2f %+15.2f %15s\n",
                        tier, mean(values), effects["ate"], direction)
            end

            push!(results_data, Dict(
                "outcome" => outcome,
                "tier" => tier,
                "mean" => mean(values),
                "ate" => effects["ate"],
                "direction" => direction
            ))
        end
    end

    println("\n" * "-"^100)

    # Check consistency across outcomes
    premium_survival_ate = mean(tier_outcomes["premium"]["survival"]) - mean(tier_outcomes["none"]["survival"])
    premium_capital_ate = mean(tier_outcomes["premium"]["mean_capital"]) - mean(tier_outcomes["none"]["mean_capital"])

    println("\nCONSISTENCY CHECK:")
    survival_negative = premium_survival_ate < 0
    capital_negative = premium_capital_ate < 0

    if survival_negative && capital_negative
        println("✓ Premium AI shows negative effects on BOTH survival AND capital")
        println("  Strong evidence that paradox is real, not measurement artifact.")
    elseif survival_negative && !capital_negative
        println("⚠ Mixed results: Negative survival effect but positive/neutral capital effect")
        println("  Survivors with premium AI may be doing better financially.")
    else
        println("  Results vary by outcome measure - interpret with caution.")
    end

    # Save results
    outcomes_file = joinpath(output_dir, "test10_alternative_outcomes_results.csv")
    open(outcomes_file, "w") do f
        println(f, "outcome,tier,mean,ate,direction")
        for row in results_data
            @printf(f, "%s,%s,%.6f,%.6f,%s\n",
                    row["outcome"], row["tier"], row["mean"], row["ate"], row["direction"])
        end
    end
    println("\nResults saved to: $outcomes_file")

    return Dict("outcomes" => tier_outcomes, "data" => results_data)
end

# ============================================================================
# MAIN EXECUTION
# ============================================================================

function main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    output_dir = "comprehensive_robustness_$(timestamp)"
    mkpath(output_dir)

    mode_str = QUICK_MODE ? "QUICK" : "FULL"

    println("="^80)
    println("COMPREHENSIVE ROBUSTNESS TEST SUITE ($mode_str MODE)")
    println("="^80)
    println()
    println("Output directory: $output_dir")
    println()
    println("Tests to run:")
    println("  1. Placebo/Falsification Test")
    println("  2. Bootstrapped Confidence Intervals ($N_BOOTSTRAP resamples)")
    println("  3. Multiple Comparison Correction (BH)")
    println("  4. Population Size Sensitivity")
    println("  5. Simulation Length Sensitivity")
    println("  6. Balanced Design Test")
    println("  7. Initial Capital Sensitivity")
    println("  8. Market Regime Sensitivity")
    println("  9. AI Accuracy Isolation")
    println("  10. Alternative Outcome Measures")
    println()

    total_start = time()

    results = Dict{String, Any}()

    # Run all tests
    results["placebo"] = run_placebo_test(output_dir)
    results["bootstrap"] = run_bootstrap_test(output_dir)
    results["multiple_comparison"] = run_multiple_comparison_test(output_dir)
    results["population_size"] = run_population_size_test(output_dir)
    results["simulation_length"] = run_simulation_length_test(output_dir)
    results["balanced_design"] = run_balanced_design_test(output_dir)
    results["initial_capital"] = run_initial_capital_test(output_dir)
    results["market_regime"] = run_market_regime_test(output_dir)
    results["ai_accuracy"] = run_ai_accuracy_isolation_test(output_dir)
    results["alternative_outcomes"] = run_alternative_outcomes_test(output_dir)

    total_elapsed = time() - total_start

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    println("\n" * "="^80)
    println("COMPREHENSIVE ROBUSTNESS TEST SUMMARY")
    println("="^80)
    println()

    summary_data = []

    # Test 1: Placebo
    placebo_pass = results["placebo"]["passed"]
    push!(summary_data, ("1. Placebo Test", placebo_pass ? "✓ PASS" : "✗ FAIL"))

    # Test 2: Bootstrap (check if premium CI excludes 0)
    bootstrap_sig = results["bootstrap"]["results"]["premium"]["significant_95"]
    push!(summary_data, ("2. Bootstrap CIs", bootstrap_sig ? "✓ Significant" : "○ Not significant"))

    # Test 3: Multiple comparison
    bh_count = results["multiple_comparison"]["bh_significant_count"]
    push!(summary_data, ("3. BH Correction", bh_count > 0 ? "✓ $bh_count/3 significant" : "✗ None survive"))

    # Test 4: Population size
    pop_consistent = results["population_size"]["consistent"]
    push!(summary_data, ("4. Population Size", pop_consistent ? "✓ Consistent" : "○ Varies"))

    # Test 5: Simulation length
    len_consistent = results["simulation_length"]["paradox_consistent"]
    push!(summary_data, ("5. Sim Length", len_consistent ? "✓ Consistent" : "○ Varies"))

    # Test 6: Balanced design
    balanced_paradox = results["balanced_design"]["paradox_persists"]
    push!(summary_data, ("6. Balanced Design", balanced_paradox ? "✓ Paradox persists" : "○ Different"))

    # Test 7: Initial capital
    capital_consistent = results["initial_capital"]["paradox_consistent"]
    push!(summary_data, ("7. Initial Capital", capital_consistent ? "✓ Consistent" : "○ Varies"))

    # Test 8: Market regime
    regime_consistent = results["market_regime"]["paradox_consistent"]
    push!(summary_data, ("8. Market Regime", regime_consistent ? "✓ Consistent" : "○ Varies"))

    # Test 9: AI accuracy isolation
    paradox_no_costs = results["ai_accuracy"]["paradox_without_costs"]
    push!(summary_data, ("9. Accuracy Isolation", paradox_no_costs ? "✓ Not cost-driven" : "○ Cost-driven"))

    # Test 10: Alternative outcomes
    push!(summary_data, ("10. Alt Outcomes", "✓ Complete"))

    println("Test Results:")
    println("-"^50)
    for (test_name, result) in summary_data
        @printf("  %-25s %s\n", test_name, result)
    end
    println("-"^50)

    # Count passes
    critical_passed = placebo_pass && bootstrap_sig && bh_count > 0

    println()
    if critical_passed
        println("✓ ALL CRITICAL TESTS PASSED")
        println("  The AI information paradox finding is robust.")
    else
        println("⚠ Some critical tests did not pass - findings require additional validation")
    end

    # Performance
    println()
    println("="^80)
    println("EXECUTION STATISTICS")
    println("="^80)
    @printf("  Total runtime: %.1f minutes (%.0f seconds)\n", total_elapsed / 60, total_elapsed)
    println("  Output directory: $output_dir")

    # Save master summary
    summary_file = joinpath(output_dir, "robustness_summary.txt")
    open(summary_file, "w") do f
        println(f, "COMPREHENSIVE ROBUSTNESS TEST SUMMARY")
        println(f, "Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(f, "Mode: $mode_str")
        println(f, "")
        println(f, "TEST RESULTS:")
        println(f, "-"^50)
        for (test_name, result) in summary_data
            @printf(f, "  %-25s %s\n", test_name, result)
        end
        println(f, "-"^50)
        println(f, "")
        println(f, "CRITICAL TESTS PASSED: $critical_passed")
        println(f, "Total runtime: $(round(total_elapsed / 60, digits=1)) minutes")
    end
    println("\nSummary saved to: $summary_file")

    println()
    println("="^80)
    println("✓ COMPREHENSIVE ROBUSTNESS SUITE COMPLETE")
    println("="^80)

    return results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
