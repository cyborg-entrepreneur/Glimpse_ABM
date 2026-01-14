"""
Analyze robustness sweep results and generate publication-ready summary.

Port of: glimpse_abm/scripts/analyze_robustness.py
"""

module AnalyzeRobustness

using DataFrames
using Statistics
using Arrow
using Serialization

export analyze_robustness, generate_robustness_report

"""
Load agent data from a run directory.
"""
function load_agents_from_run(run_dir::String)::Union{DataFrame,Nothing}
    # Try different file formats
    for (filename, loader) in [
        ("final_agents.arrow", path -> DataFrame(Arrow.Table(path))),
        ("final_agents.jld2", path -> begin
            jld = JLD2.jldopen(path, "r")
            df = jld["agents"]
            close(jld)
            df
        end)
    ]
        path = joinpath(run_dir, filename)
        if isfile(path)
            try
                return loader(path)
            catch e
                continue
            end
        end
    end
    return nothing
end

"""
Calculate Cohen's d effect size.
"""
function cohens_d(group1::Vector{Float64}, group2::Vector{Float64})::Float64
    n1, n2 = length(group1), length(group2)
    if n1 < 2 || n2 < 2
        return 0.0
    end

    v1, v2 = var(group1), var(group2)
    pooled_std = sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))

    if pooled_std ≈ 0
        return 0.0
    end

    return (mean(group1) - mean(group2)) / pooled_std
end

"""
Two-sample t-test (Welch's t-test).
"""
function welch_ttest(group1::Vector{Float64}, group2::Vector{Float64})::Tuple{Float64,Float64}
    n1, n2 = length(group1), length(group2)
    if n1 < 2 || n2 < 2
        return (NaN, NaN)
    end

    m1, m2 = mean(group1), mean(group2)
    v1, v2 = var(group1), var(group2)

    se = sqrt(v1/n1 + v2/n2)
    if se ≈ 0
        return (0.0, 1.0)
    end

    t_stat = (m1 - m2) / se

    # Welch-Satterthwaite degrees of freedom
    df = (v1/n1 + v2/n2)^2 / ((v1/n1)^2/(n1-1) + (v2/n2)^2/(n2-1))

    # Approximate p-value using normal distribution (good for large df)
    p_value = 2 * (1 - cdf(Normal(), abs(t_stat)))

    return (t_stat, p_value)
end

using Distributions: Normal, cdf

"""
Analyze all robustness configurations and generate summary tables.
"""
function analyze_robustness(results_base::String="glimpse_robustness_sweep")::Union{DataFrame,Nothing}
    results_dir = results_base

    if !isdir(results_dir)
        println("Error: Results directory '$results_dir' not found.")
        return nothing
    end

    configs = [d for d in readdir(results_dir, join=true) if isdir(d)]

    if isempty(configs)
        println("Error: No configuration directories found in '$results_dir'")
        return nothing
    end

    println("=" ^ 80)
    println("ROBUSTNESS ANALYSIS: Fixed-Tier Causal Effects Across Parameter Space")
    println("=" ^ 80)
    println("\nFound $(length(configs)) configurations to analyze.")

    # Collect results from all configurations
    all_results = Dict{String,Any}[]

    for config_dir in sort(configs)
        config_name = basename(config_dir)

        # Load agent data from all runs
        run_dirs = [d for d in readdir(config_dir, join=true)
                    if isdir(d) && startswith(basename(d), "Fixed_AI_Level_")]

        if isempty(run_dirs)
            println("  Warning: No runs found in $config_name")
            continue
        end

        agents_list = DataFrame[]
        for run_dir in run_dirs
            agents_df = load_agents_from_run(run_dir)
            if agents_df !== nothing
                tier = split(basename(run_dir), "_")[4]
                agents_df[!, :ai_tier] .= tier
                agents_df[!, :config] .= config_name
                push!(agents_list, agents_df)
            end
        end

        if isempty(agents_list)
            continue
        end

        agents_df = vcat(agents_list...)

        # Ensure survived column exists and is numeric
        if !("survived" in names(agents_df))
            if "final_status" in names(agents_df)
                agents_df[!, :survived] = agents_df.final_status .== "active"
            elseif "alive" in names(agents_df)
                agents_df[!, :survived] = agents_df.alive
            else
                continue
            end
        end

        # Get baseline (no AI)
        baseline_mask = agents_df.ai_tier .== "none"
        baseline = Float64.(coalesce.(agents_df[baseline_mask, :survived], false))

        if isempty(baseline)
            continue
        end

        baseline_mean = mean(baseline)
        baseline_std = std(baseline)

        # Calculate effects for each AI tier
        for tier in ["basic", "advanced", "premium"]
            treatment_mask = agents_df.ai_tier .== tier
            treatment = Float64.(coalesce.(agents_df[treatment_mask, :survived], false))

            if isempty(treatment)
                continue
            end

            # Average Treatment Effect
            ate = mean(treatment) - baseline_mean

            # Cohen's d
            pooled_std = sqrt((baseline_std^2 + std(treatment)^2) / 2)
            d = pooled_std > 0 ? ate / pooled_std : 0.0

            # T-test for p-value
            _, p_value = welch_ttest(treatment, baseline)

            # Confidence interval
            se = sqrt(var(baseline)/length(baseline) + var(treatment)/length(treatment))
            ci_lower = ate - 1.96 * se
            ci_upper = ate + 1.96 * se

            push!(all_results, Dict{String,Any}(
                "config" => config_name,
                "comparison" => "$(tier)_vs_none",
                "baseline_survival" => baseline_mean,
                "treatment_survival" => mean(treatment),
                "ate" => ate,
                "ci_lower" => ci_lower,
                "ci_upper" => ci_upper,
                "cohens_d" => d,
                "p_value" => p_value,
                "n_treatment" => length(treatment),
                "n_control" => length(baseline)
            ))
        end

        println("  Processed: $config_name ($(length(run_dirs)) runs)")
    end

    if isempty(all_results)
        println("Error: No results to analyze.")
        return nothing
    end

    results_df = DataFrame(all_results)

    # Print summary table
    println("\n" * "=" ^ 80)
    println("TABLE: Robustness of Causal Effects Across Parameter Configurations")
    println("=" ^ 80)

    # Summary by comparison
    println("\nAverage Treatment Effects (ATE):")
    println("-" ^ 60)

    for comp in unique(results_df.comparison)
        subset = results_df[results_df.comparison .== comp, :]
        println("$comp:")
        println("  Mean ATE: $(round(mean(subset.ate), digits=4))")
        println("  ATE Range: [$(round(minimum(subset.ate), digits=4)), $(round(maximum(subset.ate), digits=4))]")
    end

    println("\n\nCohen's d Effect Sizes:")
    println("-" ^ 60)

    for comp in unique(results_df.comparison)
        subset = results_df[results_df.comparison .== comp, :]
        println("$comp:")
        println("  Mean d: $(round(mean(subset.cohens_d), digits=3))")
        println("  d Range: [$(round(minimum(subset.cohens_d), digits=3)), $(round(maximum(subset.cohens_d), digits=3))]")
    end

    # Robustness bounds
    println("\n" * "=" ^ 80)
    println("ROBUSTNESS BOUNDS (Range of Effects Across All Configurations)")
    println("=" ^ 80)

    for comparison in ["basic_vs_none", "advanced_vs_none", "premium_vs_none"]
        subset = results_df[results_df.comparison .== comparison, :]

        if isempty(subset)
            continue
        end

        println("\n$comparison:")
        println("  ATE Range:      [$(round(minimum(subset.ate), digits=3)), $(round(maximum(subset.ate), digits=3))]")
        println("  ATE Mean ± SD:  $(round(mean(subset.ate), digits=3)) ± $(round(std(subset.ate), digits=3))")
        println("  Cohen's d Range: [$(round(minimum(subset.cohens_d), digits=3)), $(round(maximum(subset.cohens_d), digits=3))]")

        # Check if effect direction is consistent
        all_negative = all(subset.ate .< 0)
        println("  Direction consistent: $(all_negative ? "YES (all negative)" : "NO (some positive)")")
    end

    # Save results
    output_file = joinpath(results_dir, "robustness_summary.csv")
    # Using simple CSV writing
    open(output_file, "w") do f
        # Header
        cols = names(results_df)
        println(f, join(cols, ","))
        # Data
        for row in eachrow(results_df)
            values = [string(row[c]) for c in cols]
            println(f, join(values, ","))
        end
    end
    println("\n\nResults saved to: $output_file")

    # Create publication table
    pub_table = Dict{String,Any}[]
    for config in unique(results_df.config)
        config_data = results_df[results_df.config .== config, :]
        row = Dict{String,Any}("Configuration" => config)
        for r in eachrow(config_data)
            comp = replace(r.comparison, "_vs_none" => "")
            row["$(comp)_ATE"] = round(r.ate, digits=3)
            row["$(comp)_d"] = round(r.cohens_d, digits=2)
        end
        push!(pub_table, row)
    end

    pub_df = DataFrame(pub_table)
    pub_file = joinpath(results_dir, "robustness_publication_table.csv")
    open(pub_file, "w") do f
        cols = names(pub_df)
        println(f, join(cols, ","))
        for row in eachrow(pub_df)
            values = [string(row[c]) for c in cols]
            println(f, join(values, ","))
        end
    end
    println("Publication table saved to: $pub_file")

    return results_df
end

"""
Generate a summary report string.
"""
function generate_robustness_report(results_df::DataFrame)::String
    lines = String[]

    push!(lines, "=" ^ 60)
    push!(lines, "ROBUSTNESS ANALYSIS SUMMARY")
    push!(lines, "=" ^ 60)
    push!(lines, "")

    push!(lines, "Configurations analyzed: $(length(unique(results_df.config)))")
    push!(lines, "")

    for comparison in ["basic_vs_none", "advanced_vs_none", "premium_vs_none"]
        subset = results_df[results_df.comparison .== comparison, :]
        if isempty(subset)
            continue
        end

        push!(lines, "$comparison:")
        push!(lines, "  ATE: $(round(mean(subset.ate), digits=4)) [$(round(minimum(subset.ate), digits=4)), $(round(maximum(subset.ate), digits=4))]")
        push!(lines, "  Cohen's d: $(round(mean(subset.cohens_d), digits=3))")

        sig_count = count(subset.p_value .< 0.05)
        push!(lines, "  Significant (p<0.05): $sig_count / $(nrow(subset)) configs")
        push!(lines, "")
    end

    return join(lines, "\n")
end

"""
Main entry point for command-line usage.
"""
function main(args=ARGS)
    if isempty(args)
        results_base = "glimpse_robustness_sweep"
    else
        results_base = args[1]
    end

    analyze_robustness(results_base)
end

end # module AnalyzeRobustness
