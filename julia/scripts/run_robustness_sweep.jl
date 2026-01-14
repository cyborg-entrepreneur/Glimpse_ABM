#!/usr/bin/env julia
"""
Full Robustness Sweep for GLIMPSE ABM - Julia Implementation

Runs fixed-tier causal analysis across 8 parameter configurations:
1. baseline - Default minimal_causal parameters
2. high_cost - BASE_OPERATIONAL_COST=90000
3. low_cost - BASE_OPERATIONAL_COST=40000
4. high_threshold - SURVIVAL_CAPITAL_RATIO=0.55
5. low_threshold - SURVIVAL_CAPITAL_RATIO=0.25
6. high_noise - RETURN_NOISE_SCALE=0.35
7. low_noise - RETURN_NOISE_SCALE=0.10
8. multi_sector - 4 sectors (ecological validity)

For each configuration, runs 30 simulations per AI tier (none, basic, advanced, premium).
Total: 8 configs × 4 tiers × 30 runs = 960 simulations
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
# CONFIGURATION
# ============================================================================

const N_AGENTS = 1000
const N_ROUNDS = 120
const N_RUNS_PER_TIER = 30
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const BASE_SEED = 42

# Robustness configurations
const ROBUSTNESS_CONFIGS = Dict(
    "baseline" => Dict{Symbol,Any}(),
    "high_cost" => Dict{Symbol,Any}(:BASE_OPERATIONAL_COST => 90000.0),
    "low_cost" => Dict{Symbol,Any}(:BASE_OPERATIONAL_COST => 40000.0),
    "high_threshold" => Dict{Symbol,Any}(:SURVIVAL_CAPITAL_RATIO => 0.55),
    "low_threshold" => Dict{Symbol,Any}(:SURVIVAL_CAPITAL_RATIO => 0.25),
    "high_noise" => Dict{Symbol,Any}(:RETURN_NOISE_SCALE => 0.35),
    "low_noise" => Dict{Symbol,Any}(:RETURN_NOISE_SCALE => 0.10),
    "multi_sector" => Dict{Symbol,Any}(:SECTORS => ["tech", "healthcare", "finance", "retail"])
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

"""
Create a config with the specified parameter overrides.
"""
function create_config(overrides::Dict{Symbol,Any}, seed::Int)
    config = EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        enable_round_logging=false
    )

    # Apply overrides
    for (key, value) in overrides
        if hasfield(typeof(config), key)
            setfield!(config, key, value)
        end
    end

    return config
end

"""
Run a single simulation with fixed AI tier.
"""
function run_single_simulation(config::EmergentConfig, tier::String, run_id::String, output_dir::String)
    sim = EmergentSimulation(
        config=config,
        output_dir=output_dir,
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
        "total_capital" => sum(a.resources.capital for a in sim.agents),
        "innovations" => sum(a.innovation_count for a in sim.agents),
        "n_agents" => config.N_AGENTS
    )
end

"""
Run all simulations for a single configuration.
"""
function run_config_sweep(config_name::String, overrides::Dict{Symbol,Any}, output_base::String)
    config_dir = joinpath(output_base, config_name)
    mkpath(config_dir)

    results = Dict{String, Vector{Dict{String,Any}}}()

    for tier in AI_TIERS
        results[tier] = Dict{String,Any}[]

        for run_idx in 1:N_RUNS_PER_TIER
            seed = BASE_SEED + hash((config_name, tier, run_idx)) % 10000
            config = create_config(overrides, seed)

            run_id = "Fixed_AI_Level_$(tier)_run_$(run_idx-1)"
            run_dir = joinpath(config_dir, run_id)

            result = run_single_simulation(config, tier, run_id, run_dir)
            push!(results[tier], result)
        end
    end

    return results
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

        # Standard error and confidence interval
        se = sqrt(baseline_std^2/length(baseline_rates) + treatment_std^2/length(rates))
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        # Welch's t-test
        n1, n2 = length(baseline_rates), length(rates)
        se_t = sqrt(baseline_std^2/n1 + treatment_std^2/n2)
        t_stat = se_t > 0 ? ate / se_t : 0.0

        effects[tier] = Dict(
            "mean_survival" => treatment_mean,
            "std_survival" => treatment_std,
            "ate" => ate,
            "cohens_d" => cohens_d,
            "ci_lower" => ci_lower,
            "ci_upper" => ci_upper,
            "t_stat" => t_stat,
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
    output_base = "glimpse_robustness_julia_$(timestamp)"
    mkpath(output_base)

    println("=" ^ 80)
    println("GLIMPSE ABM - FULL ROBUSTNESS SWEEP (Julia)")
    println("=" ^ 80)
    println()
    println("Configuration:")
    println("  Agents per run: $N_AGENTS")
    println("  Rounds per run: $N_ROUNDS")
    println("  Runs per tier: $N_RUNS_PER_TIER")
    println("  AI tiers: $(join(AI_TIERS, ", "))")
    println("  Configurations: $(length(ROBUSTNESS_CONFIGS))")
    println()

    total_runs = length(ROBUSTNESS_CONFIGS) * length(AI_TIERS) * N_RUNS_PER_TIER
    total_agent_rounds = total_runs * N_AGENTS * N_ROUNDS
    println("  Total simulations: $total_runs")
    println("  Total agent-rounds: $(round(total_agent_rounds/1e9, digits=2)) billion")
    println("  Output directory: $output_base")
    println()

    # Estimate time
    estimated_throughput = 5_000_000  # agent-rounds/sec from benchmark
    estimated_time_sec = total_agent_rounds / estimated_throughput
    println("  Estimated time: $(round(estimated_time_sec/60, digits=1)) minutes")
    println()

    all_results = Dict{String, Dict{String, Dict{String,Any}}}()

    sweep_start = time()

    config_names = sort(collect(keys(ROBUSTNESS_CONFIGS)))
    for (config_idx, config_name) in enumerate(config_names)
        overrides = ROBUSTNESS_CONFIGS[config_name]
        config_start = time()

        println("=" ^ 80)
        println("[$config_idx/$(length(ROBUSTNESS_CONFIGS))] Configuration: $(uppercase(config_name))")
        println("=" ^ 80)

        if !isempty(overrides)
            println("  Overrides: $overrides")
        else
            println("  Overrides: (none - baseline)")
        end
        println()

        # Run all tiers for this config
        config_results = Dict{String, Vector{Dict{String,Any}}}()

        for (tier_idx, tier) in enumerate(AI_TIERS)
            tier_start = time()
            print("  [$tier_idx/$(length(AI_TIERS))] AI Tier: $(uppercase(tier)) ... ")

            config_results[tier] = Dict{String,Any}[]

            for run_idx in 1:N_RUNS_PER_TIER
                seed = BASE_SEED + Int(hash((config_name, tier, run_idx)) % 10000)
                config = create_config(overrides, seed)

                run_id = "Fixed_AI_Level_$(tier)_run_$(run_idx-1)"
                run_dir = joinpath(output_base, config_name, run_id)

                result = run_single_simulation(config, tier, run_id, run_dir)
                push!(config_results[tier], result)
            end

            tier_elapsed = time() - tier_start
            survival_rates = [r["survival_rate"] for r in config_results[tier]]
            @printf("done (%.1fs) - Mean survival: %.1f%% ± %.1f%%\n",
                    tier_elapsed, 100*mean(survival_rates), 100*std(survival_rates))
        end

        # Calculate effects for this config
        effects = calculate_effects(config_results)
        all_results[config_name] = effects

        config_elapsed = time() - config_start

        # Print config summary
        println()
        println("  Results for $config_name:")
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

        @printf("\n  Config time: %.1f seconds\n\n", config_elapsed)
    end

    sweep_elapsed = time() - sweep_start

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    println()
    println("=" ^ 80)
    println("ROBUSTNESS SWEEP COMPLETE - SUMMARY")
    println("=" ^ 80)
    println()

    # Create summary table
    println("Average Treatment Effects (ATE) Across All Configurations:")
    println("-" ^ 80)
    @printf("%-15s ", "Configuration")
    for tier in ["basic", "advanced", "premium"]
        @printf("%18s ", "$(titlecase(tier)) ATE")
    end
    println()
    println("-" ^ 80)

    for config_name in sort(collect(keys(all_results)))
        effects = all_results[config_name]
        @printf("%-15s ", config_name)
        for tier in ["basic", "advanced", "premium"]
            @printf("%+17.1f%% ", 100*effects[tier]["ate"])
        end
        println()
    end
    println("-" ^ 80)

    # Calculate means across configs
    @printf("%-15s ", "MEAN")
    for tier in ["basic", "advanced", "premium"]
        ates = [all_results[c][tier]["ate"] for c in keys(all_results)]
        @printf("%+17.1f%% ", 100*mean(ates))
    end
    println()

    @printf("%-15s ", "STD")
    for tier in ["basic", "advanced", "premium"]
        ates = [all_results[c][tier]["ate"] for c in keys(all_results)]
        @printf("%17.1f%% ", 100*std(ates))
    end
    println()

    # Cohen's d summary
    println()
    println("Cohen's d Effect Sizes Across All Configurations:")
    println("-" ^ 80)
    @printf("%-15s ", "Configuration")
    for tier in ["basic", "advanced", "premium"]
        @printf("%18s ", "$(titlecase(tier)) d")
    end
    println()
    println("-" ^ 80)

    for config_name in sort(collect(keys(all_results)))
        effects = all_results[config_name]
        @printf("%-15s ", config_name)
        for tier in ["basic", "advanced", "premium"]
            @printf("%18.3f ", effects[tier]["cohens_d"])
        end
        println()
    end
    println("-" ^ 80)

    # Robustness bounds
    println()
    println("ROBUSTNESS BOUNDS (Range of Effects Across All Configurations):")
    println("-" ^ 60)

    for tier in ["basic", "advanced", "premium"]
        ates = [all_results[c][tier]["ate"] for c in keys(all_results)]
        ds = [all_results[c][tier]["cohens_d"] for c in keys(all_results)]

        println("$(titlecase(tier)) vs None:")
        @printf("  ATE Range:      [%+.1f%%, %+.1f%%]\n", 100*minimum(ates), 100*maximum(ates))
        @printf("  ATE Mean ± SD:  %+.1f%% ± %.1f%%\n", 100*mean(ates), 100*std(ates))
        @printf("  Cohen's d Range: [%.3f, %.3f]\n", minimum(ds), maximum(ds))

        # Check direction consistency
        all_positive = all(ates .> 0)
        all_negative = all(ates .< 0)
        direction = all_positive ? "YES (all positive)" : (all_negative ? "YES (all negative)" : "NO (mixed)")
        println("  Direction consistent: $direction")
        println()
    end

    # Performance summary
    println("=" ^ 80)
    println("PERFORMANCE")
    println("=" ^ 80)
    @printf("  Total execution time: %.1f minutes (%.0f seconds)\n", sweep_elapsed/60, sweep_elapsed)
    @printf("  Total agent-rounds: %.2f billion\n", total_agent_rounds/1e9)
    @printf("  Throughput: %.2f million agent-rounds/second\n", total_agent_rounds/sweep_elapsed/1e6)
    println()

    # Estimated Python time
    python_throughput = 70_000  # from benchmark
    python_time = total_agent_rounds / python_throughput
    @printf("  Estimated Python time: %.1f hours\n", python_time/3600)
    @printf("  Julia speedup: %.0fx faster\n", python_time / sweep_elapsed)

    # Save results to CSV
    results_file = joinpath(output_base, "robustness_summary.csv")
    open(results_file, "w") do f
        println(f, "config,tier,mean_survival,std_survival,ate,cohens_d,ci_lower,ci_upper,n_runs")
        for config_name in sort(collect(keys(all_results)))
            for tier in AI_TIERS
                e = all_results[config_name][tier]
                @printf(f, "%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n",
                        config_name, tier, e["mean_survival"], e["std_survival"],
                        e["ate"], e["cohens_d"], e["ci_lower"], e["ci_upper"], e["n_runs"])
            end
        end
    end
    println()
    println("Results saved to: $results_file")

    println()
    println("=" ^ 80)
    println("✓ ROBUSTNESS SWEEP COMPLETED SUCCESSFULLY")
    println("=" ^ 80)

    return all_results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
