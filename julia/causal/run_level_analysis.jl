"""
Run-Level Statistical Analysis for GlimpseABM.jl

This module implements proper unit of analysis handling for nested data structure
where agents are clustered within simulation runs. Addresses AMJ reviewer concerns
about independence assumptions.

Port of: glimpse_abm/run_level_analysis.py

Key Functions:
- aggregate_to_run_level: Aggregate agent data to run statistics
- run_level_anova: Test differences across AI tiers using runs as unit
- calculate_icc: Quantify within-run clustering
- bootstrap_run_ci: Proper confidence intervals via run-level resampling
- pairwise_run_tests: All comparisons with FDR correction
"""

using Statistics
using Random
using DataFrames
using CSV
using Printf
using HypothesisTests

# ============================================================================
# RUN-LEVEL AGGREGATION
# ============================================================================

"""
    aggregate_to_run_level(sim_results::Dict) -> DataFrame

Aggregate agent-level data to run-level statistics.

This is the PRIMARY unit of analysis for causal inference. Each simulation
run is treated as one independent observation.

Arguments:
- sim_results: Dict mapping run_id => simulation object or agent array

Returns:
- DataFrame with run-level statistics
"""
function aggregate_to_run_level(sim_results::Dict)
    run_data = []

    for (run_id, agents) in sim_results
        # Get AI tier (should be same for all agents in fixed-tier design)
        ai_tier = agents[1].primary_ai_level

        # Compute run-level statistics
        n_agents = length(agents)
        survival_rate = mean(a.alive for a in agents)

        run_stats = Dict(
            "run_id" => run_id,
            "ai_tier" => ai_tier,
            "n_agents" => n_agents,
            "survival_rate" => survival_rate,
            "mean_capital" => mean(a.capital for a in agents),
            "median_capital" => median(a.capital for a in agents),
            "mean_capital_survivors" => let survivors = filter(a -> a.alive, agents)
                isempty(survivors) ? missing : mean(a.capital for a in survivors)
            end
        )

        push!(run_data, run_stats)
    end

    run_df = DataFrame(run_data)

    println("\n✓ Aggregated $(nrow(run_df)) runs to run-level statistics")
    tier_counts = combine(groupby(run_df, :ai_tier), nrow => :count)
    println("  AI tiers: ", Dict(zip(tier_counts.ai_tier, tier_counts.count)))

    return run_df
end

# ============================================================================
# RUN-LEVEL HYPOTHESIS TESTS
# ============================================================================

"""
    run_level_anova(run_df::DataFrame, outcome::Symbol=:survival_rate) -> Dict

Test for differences across AI tiers using ANOVA at run level.

This is the PRIMARY test for AI effects. Treats each run as one
independent observation, avoiding inflated significance from agent-level analysis.
"""
function run_level_anova(run_df::DataFrame, outcome::Symbol=:survival_rate)
    # Extract groups
    groups = Dict{String,Vector{Float64}}()
    for tier in unique(run_df.ai_tier)
        tier_data = filter(row -> row.ai_tier == tier, run_df)
        groups[tier] = collect(skipmissing(tier_data[!, outcome]))
    end

    # ANOVA using HypothesisTests.jl
    group_arrays = [groups[tier] for tier in sort(collect(keys(groups)))]
    anova_result = OneWayANOVATest(group_arrays...)

    f_stat = anova_result.F
    p_value = pvalue(anova_result)

    # Effect size (eta-squared)
    grand_mean = mean(run_df[!, outcome])
    ss_between = sum(length(g) * (mean(g) - grand_mean)^2 for g in values(groups))
    ss_total = sum((run_df[!, outcome] .- grand_mean).^2)
    eta_squared = ss_between / ss_total

    # Interpret effect size
    effect_interpretation = if eta_squared < 0.01
        "negligible"
    elseif eta_squared < 0.06
        "small"
    elseif eta_squared < 0.14
        "medium"
    else
        "large"
    end

    results = Dict(
        "test" => "One-Way ANOVA (Run-Level)",
        "outcome" => String(outcome),
        "f_statistic" => f_stat,
        "p_value" => p_value,
        "eta_squared" => eta_squared,
        "effect_interpretation" => effect_interpretation,
        "df_between" => length(groups) - 1,
        "df_within" => nrow(run_df) - length(groups),
        "n_runs" => nrow(run_df),
        "group_means" => Dict(tier => mean(data) for (tier, data) in groups),
        "group_sds" => Dict(tier => std(data) for (tier, data) in groups),
        "n_runs_per_group" => Dict(tier => length(data) for (tier, data) in groups),
    )

    return results
end

"""
    pairwise_run_tests(run_df::DataFrame, outcome::Symbol=:survival_rate) -> DataFrame

All pairwise comparisons between AI tiers at run level with Bonferroni correction.
"""
function pairwise_run_tests(run_df::DataFrame, outcome::Symbol=:survival_rate)
    tiers = sort(unique(run_df.ai_tier))
    results = []

    for i in 1:length(tiers)
        for j in (i+1):length(tiers)
            tier1 = tiers[i]
            tier2 = tiers[j]

            data1 = filter(row -> row.ai_tier == tier1, run_df)[!, outcome]
            data2 = filter(row -> row.ai_tier == tier2, run_df)[!, outcome]

            # t-test
            t_test = EqualVarianceTTest(data1, data2)
            t_stat = t_test.t
            p_val = pvalue(t_test)

            # Mean difference
            mean_diff = mean(data2) - mean(data1)

            # Cohen's d
            pooled_std = sqrt((var(data1) + var(data2)) / 2)
            cohens_d = pooled_std > 0 ? mean_diff / pooled_std : 0.0

            # 95% CI for difference
            se_diff = sqrt(var(data1)/length(data1) + var(data2)/length(data2))
            ci_lower = mean_diff - 1.96 * se_diff
            ci_upper = mean_diff + 1.96 * se_diff

            push!(results, Dict(
                "comparison" => "$tier1 vs $tier2",
                "tier1" => tier1,
                "tier2" => tier2,
                "mean_diff" => mean_diff,
                "t_statistic" => t_stat,
                "p_value" => p_val,
                "ci_lower" => ci_lower,
                "ci_upper" => ci_upper,
                "cohens_d" => cohens_d,
                "n1" => length(data1),
                "n2" => length(data2),
            ))
        end
    end

    results_df = DataFrame(results)

    # Bonferroni correction
    n_comparisons = nrow(results_df)
    results_df.p_adjusted = min.(results_df.p_value .* n_comparisons, 1.0)
    results_df.correction = fill("Bonferroni", nrow(results_df))

    return results_df
end

# ============================================================================
# INTRACLASS CORRELATION (ICC)
# ============================================================================

"""
    calculate_icc(agent_df::DataFrame, outcome::Symbol=:survived,
                  run_id_col::Symbol=:run_id) -> Dict

Calculate intraclass correlation coefficient (ICC) to quantify clustering.

ICC measures the proportion of total variance that is between runs vs
within runs. High ICC means agents within runs are similar (clustering),
justifying run-level analysis.
"""
function calculate_icc(agent_df::DataFrame, outcome::Symbol=:survived,
                       run_id_col::Symbol=:run_id)
    # Convert boolean to numeric if needed
    if eltype(agent_df[!, outcome]) == Bool
        y = Float64.(agent_df[!, outcome])
    else
        y = agent_df[!, outcome]
    end

    run_ids = agent_df[!, run_id_col]

    # Grand mean
    grand_mean = mean(y)

    # Between-run variance
    run_stats = combine(groupby(agent_df, run_id_col)) do df
        DataFrame(
            run_mean = mean(df[!, outcome]),
            run_size = nrow(df)
        )
    end

    between_var = sum((run_stats.run_mean .- grand_mean).^2 .* run_stats.run_size) / (nrow(run_stats) - 1)

    # Within-run variance
    within_var = 0.0
    for run_id in unique(run_ids)
        run_data = filter(row -> row[run_id_col] == run_id, agent_df)[!, outcome]
        run_data_numeric = eltype(run_data) == Bool ? Float64.(run_data) : run_data
        run_mean = mean(run_data_numeric)
        within_var += sum((run_data_numeric .- run_mean).^2)
    end
    within_var /= (nrow(agent_df) - nrow(run_stats))

    # ICC
    avg_cluster_size = nrow(agent_df) / nrow(run_stats)
    icc = between_var / (between_var + within_var)

    # Interpret ICC
    interpretation = if icc < 0.05
        "negligible clustering (agent-level analysis may be acceptable)"
    elseif icc < 0.10
        "small clustering (run-level analysis recommended)"
    elseif icc < 0.20
        "moderate clustering (run-level analysis strongly recommended)"
    else
        "substantial clustering (run-level analysis required)"
    end

    # Design effect
    design_effect = 1 + (avg_cluster_size - 1) * icc

    return Dict(
        "icc" => icc,
        "between_run_variance" => between_var,
        "within_run_variance" => within_var,
        "total_variance" => between_var + within_var,
        "interpretation" => interpretation,
        "n_runs" => nrow(run_stats),
        "mean_agents_per_run" => avg_cluster_size,
        "design_effect" => design_effect,
        "note" => "ICC > 0.05 suggests run-level analysis is more appropriate than agent-level"
    )
end

# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS (RUN-LEVEL)
# ============================================================================

"""
    bootstrap_run_ci(run_df::DataFrame, tier1::String, tier2::String,
                     outcome::Symbol=:survival_rate; n_bootstrap::Int=10000,
                     confidence_level::Float64=0.95) -> Dict

Bootstrap confidence interval for difference in means, resampling RUNS.

This is the correct way to compute CIs given nested data structure.
Resamples runs (not agents) to respect clustering.
"""
function bootstrap_run_ci(run_df::DataFrame, tier1::String, tier2::String,
                         outcome::Symbol=:survival_rate; n_bootstrap::Int=10000,
                         confidence_level::Float64=0.95)
    # Extract run-level data
    data1 = filter(row -> row.ai_tier == tier1, run_df)[!, outcome]
    data2 = filter(row -> row.ai_tier == tier2, run_df)[!, outcome]

    # Observed difference
    observed_diff = mean(data2) - mean(data1)

    # Bootstrap resampling (resample RUNS, not agents)
    boot_diffs = zeros(n_bootstrap)
    rng = MersenneTwister(42)

    for i in 1:n_bootstrap
        # Resample runs with replacement
        boot1 = sample(rng, data1, length(data1), replace=true)
        boot2 = sample(rng, data2, length(data2), replace=true)
        boot_diffs[i] = mean(boot2) - mean(boot1)
    end

    # CI from percentiles
    alpha = 1 - confidence_level
    ci_lower = quantile(boot_diffs, alpha/2)
    ci_upper = quantile(boot_diffs, 1 - alpha/2)

    # Bootstrap SE
    se_boot = std(boot_diffs)

    return Dict(
        "comparison" => "$tier1 vs $tier2",
        "outcome" => String(outcome),
        "mean_diff" => observed_diff,
        "ci_lower" => ci_lower,
        "ci_upper" => ci_upper,
        "se_boot" => se_boot,
        "confidence_level" => confidence_level,
        "n_bootstrap" => n_bootstrap,
        "n_runs_tier1" => length(data1),
        "n_runs_tier2" => length(data2),
    )
end

# ============================================================================
# COMPLETE RUN-LEVEL ANALYSIS
# ============================================================================

"""
    run_complete_run_level_analysis(sim_results::Dict, output_dir::String="") -> Dict

Execute complete run-level analysis pipeline.

This is the MAIN function to call for publication-quality analysis.

Steps:
1. Aggregate agents to run level
2. Run-level ANOVA for primary outcomes
3. Pairwise comparisons with correction
4. Calculate ICCs to justify run-level approach
5. Bootstrap CIs for key comparisons
6. Generate summary tables
"""
function run_complete_run_level_analysis(sim_results::Dict, output_dir::String="output_run_level")
    println("\n" * "="^70)
    println("RUN-LEVEL STATISTICAL ANALYSIS")
    println("="^70)
    println("\nThis analysis treats simulation RUNS (not agents) as the unit of analysis,")
    println("properly accounting for clustering of agents within runs.\n")

    # Create output directory
    mkpath(output_dir)

    all_results = Dict()

    # Step 1: Aggregate to run level
    println("\n[1/5] Aggregating to run level...")
    run_df = aggregate_to_run_level(sim_results)
    CSV.write(joinpath(output_dir, "run_level_data.csv"), run_df)
    all_results["run_data"] = run_df

    # Step 2: Run-level ANOVA for key outcomes
    println("\n[2/5] Run-level ANOVA tests...")
    outcomes = [:survival_rate, :mean_capital]

    anova_results = []
    for outcome in outcomes
        if outcome in propertynames(run_df) && !all(ismissing, run_df[!, outcome])
            result = run_level_anova(run_df, outcome)
            push!(anova_results, result)
            @printf("  ✓ %s: F=%.2f, p=%.4f, η²=%.3f (%s)\n",
                   result["outcome"], result["f_statistic"], result["p_value"],
                   result["eta_squared"], result["effect_interpretation"])
        end
    end

    all_results["anova"] = anova_results
    CSV.write(joinpath(output_dir, "run_level_anova.csv"), DataFrame(anova_results))

    # Step 3: Pairwise comparisons
    println("\n[3/5] Pairwise comparisons with Bonferroni correction...")
    pairwise_results = Dict()
    for outcome in outcomes[1:2]  # Primary outcomes only
        if outcome in propertynames(run_df)
            pw_df = pairwise_run_tests(run_df, outcome)
            pairwise_results[String(outcome)] = pw_df
            println("  ✓ $(outcome): $(nrow(pw_df)) comparisons")
            CSV.write(joinpath(output_dir, "pairwise_$(outcome).csv"), pw_df)
        end
    end

    all_results["pairwise"] = pairwise_results

    # Step 4: ICC calculations (skip if agent-level data not available)
    println("\n[4/5] ICC calculations...")
    println("  ℹ️  ICC requires agent-level data (compute separately if needed)")
    all_results["icc"] = []

    # Step 5: Bootstrap CIs for key comparisons
    println("\n[5/5] Bootstrap confidence intervals (run-level resampling)...")
    bootstrap_results = []

    # Key comparisons
    tiers = sort(unique(run_df.ai_tier))
    if "none" in tiers && "premium" in tiers
        boot_result = bootstrap_run_ci(run_df, "none", "premium", :survival_rate)
        push!(bootstrap_results, boot_result)
        @printf("  ✓ none vs premium: Δ=%.3f, 95%% CI=[%.3f, %.3f]\n",
               boot_result["mean_diff"], boot_result["ci_lower"], boot_result["ci_upper"])
    end
    if "basic" in tiers && "none" in tiers
        boot_result = bootstrap_run_ci(run_df, "basic", "none", :survival_rate)
        push!(bootstrap_results, boot_result)
        @printf("  ✓ basic vs none: Δ=%.3f, 95%% CI=[%.3f, %.3f]\n",
               boot_result["mean_diff"], boot_result["ci_lower"], boot_result["ci_upper"])
    end

    all_results["bootstrap"] = bootstrap_results
    CSV.write(joinpath(output_dir, "bootstrap_cis.csv"), DataFrame(bootstrap_results))

    println("\n" * "="^70)
    println("✅ Run-level analysis complete!")
    println("   Results saved to: $output_dir")
    println("="^70)

    return all_results
end

export aggregate_to_run_level, run_level_anova, pairwise_run_tests,
       calculate_icc, bootstrap_run_ci, run_complete_run_level_analysis
