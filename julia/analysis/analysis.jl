"""
Analysis Framework for GlimpseABM.jl

Comprehensive analysis utilities for simulation results:
- Data loading and standardization
- Statistical summaries
- Cross-run aggregation
- Effect size computations
- Performance and uncertainty analysis
- Research table export

Port of: glimpse_abm/analysis.py (ComprehensiveAnalysisFramework)

Analysis Version: 2025.01
"""

module Analysis

using Statistics
using DataFrames
using Dates
using Random

# Import from parent package
using ..GlimpseABM: load_dataframe_arrow, load_agents_jld2, load_config_snapshot
using ..GlimpseABM.Causal: cohens_d, cliffs_delta

export AnalysisFramework, load_results, run_full_analysis
export compute_survival_summary, compute_capital_summary
export compute_ai_tier_comparison, aggregate_sweep_results
export analyze_performance_outcomes, analyze_ai_augmentation_effects
export analyze_uncertainty_dynamics, analyze_emergent_behaviors
export export_research_tables, normalize_ai_label, canonical_to_display
# Statistical tests
export kruskal_wallis_test, mann_whitney_u_test, run_statistical_tests
# Effect sizes (cohens_d and cliffs_delta are in Causal module)
export compute_effect_sizes
# Clustering
export cluster_agent_strategies
# Innovation analysis
export analyze_innovation_equilibrium
# Information paradox
export InformationParadoxAnalyzer, run_paradox_analysis
export analyze_temporal_reversal, analyze_herding_dynamics, analyze_none_advantages
# Advanced analysis
export run_advanced_analysis, run_complete_analysis

# ============================================================================
# AI LEVEL NORMALIZATION
# ============================================================================

"""
Canonical AI level labels used throughout the system.
"""
const CANONICAL_AI_LEVELS = ["none", "basic", "advanced", "premium"]

"""
Mapping from various AI level representations to canonical form.
"""
const AI_LABEL_MAP = Dict(
    "human" => "none",
    "human_only" => "none",
    "no_ai" => "none",
    "none" => "none",
    "basic" => "basic",
    "basic_ai" => "basic",
    "standard" => "basic",
    "advanced" => "advanced",
    "advanced_ai" => "advanced",
    "premium" => "premium",
    "premium_ai" => "premium",
    "full" => "premium"
)

"""
Display names for AI levels.
"""
const AI_DISPLAY_NAMES = Dict(
    "none" => "Human Only",
    "basic" => "Basic AI",
    "advanced" => "Advanced AI",
    "premium" => "Premium AI"
)

"""
Normalize AI level label to canonical form.
"""
function normalize_ai_label(label::Union{String,Missing,Nothing})::String
    if ismissing(label) || isnothing(label)
        return "none"
    end
    label_lower = lowercase(strip(string(label)))
    return get(AI_LABEL_MAP, label_lower, "none")
end

"""
Convert canonical AI label to display form.
"""
function canonical_to_display(label::String)::String
    return get(AI_DISPLAY_NAMES, label, label)
end

# ============================================================================
# ANALYSIS FRAMEWORK
# ============================================================================

"""
Comprehensive analysis framework for GlimpseABM simulation results.

Analyzes ABM results by reading data from disk and providing:
- Performance outcomes analysis
- AI augmentation effects
- Uncertainty dynamics
- Emergent behavior patterns
"""
mutable struct AnalysisFramework
    results_dir::String
    agent_df::DataFrame
    decision_df::DataFrame
    market_df::DataFrame
    uncertainty_df::DataFrame
    innovation_df::DataFrame
    knowledge_df::DataFrame
    summary_df::DataFrame
    matured_df::DataFrame
    uncertainty_detail_df::DataFrame
    analyses::Dict{String,Any}
    figure_output_dir::String
end

"""
Load analysis framework from results directory.
"""
function AnalysisFramework(results_dir::String)
    # Load all DataFrames
    agent_df = _load_dataframe(results_dir, "final_agents")
    decision_df = _load_partitioned_data(results_dir, "decisions")
    market_df = _load_partitioned_data(results_dir, "market")
    uncertainty_df = _load_partitioned_data(results_dir, "uncertainty")
    innovation_df = _load_partitioned_data(results_dir, "innovations")
    knowledge_df = _load_partitioned_data(results_dir, "knowledge")
    summary_df = _load_partitioned_data(results_dir, "summary")
    matured_df = _load_partitioned_data(results_dir, "matured")
    uncertainty_detail_df = _load_partitioned_data(results_dir, "uncertainty_details")

    # Standardize DataFrames
    agent_df = _standardize_agent_df(agent_df)
    decision_df = _standardize_decision_df(decision_df)
    market_df = _standardize_market_df(market_df)
    uncertainty_df = _standardize_uncertainty_df(uncertainty_df)
    summary_df = _standardize_summary_df(summary_df)
    matured_df = _standardize_matured_df(matured_df)

    # Create behavioral groups for emergent data
    agent_df = _create_behavioral_groups(agent_df, decision_df)

    # Create summary if empty
    if isempty(summary_df)
        summary_df = _create_summary_df(agent_df, decision_df)
    end

    # Set up figure output directory
    figure_output_dir = joinpath(results_dir, "figures")
    isdir(figure_output_dir) || mkpath(figure_output_dir)

    return AnalysisFramework(
        results_dir, agent_df, decision_df, market_df, uncertainty_df,
        innovation_df, knowledge_df, summary_df, matured_df,
        uncertainty_detail_df, Dict{String,Any}(), figure_output_dir
    )
end

"""
Load a single DataFrame (for final_agents, etc.).
"""
function _load_dataframe(results_dir::String, name::String)::DataFrame
    # Try Arrow first
    arrow_path = joinpath(results_dir, "$name.arrow")
    if isfile(arrow_path)
        try
            return load_dataframe_arrow(arrow_path)
        catch
            # Continue to other formats
        end
    end

    # Try CSV
    csv_path = joinpath(results_dir, "$name.csv")
    if isfile(csv_path)
        return _load_csv(csv_path)
    end

    # Try JLD2
    jld2_path = joinpath(results_dir, "$name.jld2")
    if isfile(jld2_path)
        try
            return DataFrame(load_agents_jld2(jld2_path))
        catch
            # Return empty
        end
    end

    return DataFrame()
end

"""
Load partitioned data from subdirectory (for decisions, market, etc.).
"""
function _load_partitioned_data(results_dir::String, data_type::String)::DataFrame
    # Check for subdirectory with partitioned data
    sub_dir = joinpath(results_dir, data_type)
    if isdir(sub_dir)
        # Find all CSV/Arrow files
        files = String[]
        for f in readdir(sub_dir)
            path = joinpath(sub_dir, f)
            if isfile(path) && (endswith(f, ".csv") || endswith(f, ".arrow"))
                push!(files, path)
            end
        end

        if !isempty(files)
            dfs = DataFrame[]
            for f in sort(files)
                try
                    if endswith(f, ".arrow")
                        push!(dfs, load_dataframe_arrow(f))
                    else
                        push!(dfs, _load_csv(f))
                    end
                catch e
                    @warn "Failed to load $f" exception=e
                end
            end
            if !isempty(dfs)
                return vcat(dfs...)
            end
        end
    end

    # Try single file
    return _load_dataframe(results_dir, data_type)
end

"""
Simple CSV loader.
"""
function _load_csv(path::String)::DataFrame
    lines = readlines(path)
    if isempty(lines)
        return DataFrame()
    end

    headers = split(lines[1], ",") .|> strip .|> String
    df = DataFrame([Symbol(h) => String[] for h in headers])

    for line in lines[2:end]
        values = split(line, ",")
        row = Dict{Symbol,String}()
        for (i, h) in enumerate(headers)
            row[Symbol(h)] = i <= length(values) ? strip(String(values[i])) : ""
        end
        push!(df, row)
    end

    # Try to convert numeric columns
    for col in names(df)
        try
            df[!, col] = parse.(Float64, df[!, col])
        catch
            # Keep as string
        end
    end

    return df
end

"""
Standardize history DataFrame.
"""
function _standardize_history_df(df::DataFrame)::DataFrame
    if isempty(df)
        return df
    end

    df = copy(df)

    # Ensure round column
    if !hasproperty(df, :round)
        if hasproperty(df, :step)
            rename!(df, :step => :round)
        else
            df.round = 1:nrow(df)
        end
    end

    # Convert numeric columns
    numeric_cols = [:round, :n_alive, :n_total, :survival_rate,
                    :mean_capital, :std_capital, :total_capital,
                    :actor_ignorance, :practical_indeterminism,
                    :agentic_novelty, :competitive_recursion,
                    :success_rate, :invest_count, :innovate_count,
                    :ai_none_count, :ai_basic_count, :ai_advanced_count, :ai_premium_count]

    for col in numeric_cols
        if hasproperty(df, col)
            df[!, col] = _safe_numeric(df[!, col])
        end
    end

    return df
end

"""
Standardize agent DataFrame.
"""
function _standardize_agent_df(df::DataFrame)::DataFrame
    if isempty(df)
        return df
    end

    df = copy(df)

    # Rename columns for consistency
    rename_map = Dict(
        :id => :agent_id,
        :ai_level => :primary_ai_level,
        :capital => :final_capital,
        :alive => :survived
    )

    for (old_name, new_name) in rename_map
        if hasproperty(df, old_name) && !hasproperty(df, new_name)
            rename!(df, old_name => new_name)
        end
    end

    # Ensure agent_id
    if !hasproperty(df, :agent_id)
        df.agent_id = 1:nrow(df)
    end

    # Ensure survived is numeric
    if hasproperty(df, :survived)
        df.survived = _safe_numeric(df.survived)
    end

    # Ensure final_capital is numeric
    if hasproperty(df, :final_capital)
        df.final_capital = _safe_numeric(df.final_capital)
    end

    return df
end

"""
Create summary DataFrame from agent and decision data.
"""
function _create_summary_df(agent_df::DataFrame, decision_df::DataFrame)::DataFrame
    summary = DataFrame()

    if !isempty(agent_df)
        summary[!, :n_agents] = [nrow(agent_df)]
        if hasproperty(agent_df, :survived)
            n_survivors = sum(agent_df.survived .== 1)
            summary[!, :n_survivors] = [n_survivors]
            summary[!, :survival_rate] = [n_survivors / nrow(agent_df)]
        end
        if hasproperty(agent_df, :final_capital)
            summary[!, :mean_capital] = [_safe_mean(agent_df.final_capital)]
        end
    end

    if !isempty(decision_df) && hasproperty(decision_df, :round)
        summary[!, :n_rounds] = [maximum(decision_df.round)]
        summary[!, :n_decisions] = [nrow(decision_df)]
    end

    return summary
end

"""
Safely convert to numeric.
"""
function _safe_numeric(x)
    try
        if eltype(x) <: Number
            return x
        else
            return parse.(Float64, string.(x))
        end
    catch
        return zeros(length(x))
    end
end

"""
Safe mean that handles empty arrays.
"""
function _safe_mean(x)
    clean = filter(v -> !ismissing(v) && !isnan(v), collect(x))
    isempty(clean) ? NaN : mean(clean)
end

"""
Standardize decision DataFrame.
"""
function _standardize_decision_df(df::DataFrame)::DataFrame
    if isempty(df)
        return df
    end

    df = copy(df)

    # Rename columns for consistency
    rename_map = Dict(
        :ai_level => :ai_level_used,
        :chosen_ai_level => :ai_level_used,
        :decision_round => :round,
        :round_idx => :round,
        :step => :round,
        :time => :round,
        :action_type => :action,
        :opportunity => :opportunity_id,
        :agent => :agent_id,
        :actor_id => :agent_id
    )

    for (old_name, new_name) in rename_map
        if hasproperty(df, old_name) && !hasproperty(df, new_name)
            rename!(df, old_name => new_name)
        end
    end

    # Ensure round column
    if hasproperty(df, :round)
        df[!, :round] = _safe_numeric(df[!, :round])
    else
        df.round = 1:nrow(df)
    end

    # Ensure run_id
    if !hasproperty(df, :run_id)
        df.run_id = fill("default_run", nrow(df))
    end

    # Ensure agent_id
    if !hasproperty(df, :agent_id)
        df.agent_id = 1:nrow(df)
    end

    # Normalize AI level
    if hasproperty(df, :ai_level_used)
        df.ai_level_used = normalize_ai_label.(df.ai_level_used)
    else
        df.ai_level_used = fill("none", nrow(df))
    end

    # Add ai_used flag
    df.ai_used = df.ai_level_used .!= "none"

    return df
end

"""
Standardize market DataFrame.
"""
function _standardize_market_df(df::DataFrame)::DataFrame
    if isempty(df)
        return df
    end

    df = copy(df)

    # Rename columns
    rename_map = Dict(
        :round_idx => :round,
        :step => :round,
        :time => :round,
        :market_regime => :regime
    )

    for (old_name, new_name) in rename_map
        if hasproperty(df, old_name) && !hasproperty(df, new_name)
            rename!(df, old_name => new_name)
        end
    end

    # Ensure round column
    if hasproperty(df, :round)
        df[!, :round] = _safe_numeric(df[!, :round])
    else
        df.round = 1:nrow(df)
    end

    return df
end

"""
Standardize uncertainty DataFrame.
"""
function _standardize_uncertainty_df(df::DataFrame)::DataFrame
    if isempty(df)
        return df
    end

    df = copy(df)

    # Rename columns
    rename_map = Dict(
        :round_idx => :round,
        :time => :round,
        :actor_ignorance => :actor_ignorance_level,
        :practical_indeterminism => :practical_indeterminism_level,
        :agentic_novelty => :agentic_novelty_level,
        :competitive_recursion => :competitive_recursion_level
    )

    for (old_name, new_name) in rename_map
        if hasproperty(df, old_name) && !hasproperty(df, new_name)
            rename!(df, old_name => new_name)
        end
    end

    # Ensure round column
    if hasproperty(df, :round)
        df[!, :round] = _safe_numeric(df[!, :round])
    else
        df.round = 1:nrow(df)
    end

    return df
end

"""
Standardize summary DataFrame.
"""
function _standardize_summary_df(df::DataFrame)::DataFrame
    if isempty(df)
        return df
    end

    df = copy(df)

    # Rename columns
    rename_map = Dict(
        :round_idx => :round,
        :step => :round,
        :time => :round,
        :ai_share_human => :ai_share_none,
        :ai_share_basic_ai => :ai_share_basic,
        :ai_share_advanced_ai => :ai_share_advanced,
        :ai_share_premium_ai => :ai_share_premium
    )

    for (old_name, new_name) in rename_map
        if hasproperty(df, old_name) && !hasproperty(df, new_name)
            rename!(df, old_name => new_name)
        end
    end

    # Ensure round column
    if hasproperty(df, :round)
        df[!, :round] = _safe_numeric(df[!, :round])
    end

    # Ensure AI share columns exist
    for col in [:ai_share_none, :ai_share_basic, :ai_share_advanced, :ai_share_premium]
        if !hasproperty(df, col)
            df[!, col] = zeros(nrow(df))
        end
    end

    return df
end

"""
Standardize matured DataFrame.
"""
function _standardize_matured_df(df::DataFrame)::DataFrame
    if isempty(df)
        return df
    end

    df = copy(df)

    # Convert numeric columns
    numeric_cols = [:round, :agent_id, :entry_round, :maturation_round,
                    :time_to_maturity, :investment_amount, :capital_returned,
                    :net_return, :ai_estimated_return, :ai_estimated_uncertainty,
                    :ai_confidence, :ai_actual_accuracy, :ai_overconfidence_factor]

    for col in numeric_cols
        if hasproperty(df, col)
            df[!, col] = _safe_numeric(df[!, col])
        end
    end

    # Normalize AI level
    if hasproperty(df, :ai_level_used)
        df.ai_level_used = normalize_ai_label.(df.ai_level_used)
    end

    return df
end

"""
Create behavioral groups based on emergent AI usage patterns.
"""
function _create_behavioral_groups(agent_df::DataFrame, decision_df::DataFrame)::DataFrame
    if isempty(agent_df)
        return agent_df
    end

    df = copy(agent_df)

    # Default behavioral group
    if !hasproperty(df, :behavioral_group)
        df.behavioral_group = fill("Unknown", nrow(df))
    end

    if isempty(decision_df)
        # Use primary AI level as behavioral group
        if hasproperty(df, :primary_ai_level)
            df.behavioral_group = string.(df.primary_ai_level)
        end
        return df
    end

    # Filter to emergent decisions (if run_id contains "emergent")
    emergent_decisions = filter(row -> occursin("emergent", lowercase(string(get(row, :run_id, "")))), decision_df)

    if isempty(emergent_decisions)
        # Use primary AI level as behavioral group
        if hasproperty(df, :primary_ai_level)
            df.behavioral_group = string.(df.primary_ai_level)
        end
        return df
    end

    # Calculate AI usage proportions for each agent
    # Group by run_id and agent_id, count AI level usage
    usage_counts = combine(
        groupby(emergent_decisions, [:run_id, :agent_id, :ai_level_used]),
        nrow => :count
    )

    # Pivot to get proportions
    if hasproperty(df, :agent_id) && hasproperty(df, :run_id)
        # Create usage proportion DataFrame
        agent_groups = Dict{Tuple{String,Int},String}()

        for (key, group) in pairs(groupby(usage_counts, [:run_id, :agent_id]))
            total = sum(group.count)
            proportions = Dict{String,Float64}()
            for row in eachrow(group)
                proportions[row.ai_level_used] = row.count / total
            end

            # Classify agent
            behavior = _classify_behavior(proportions)
            agent_groups[(string(key.run_id), key.agent_id)] = behavior
        end

        # Apply to agent DataFrame
        for i in 1:nrow(df)
            key = (string(df.run_id[i]), df.agent_id[i])
            if haskey(agent_groups, key)
                df.behavioral_group[i] = agent_groups[key]
            elseif hasproperty(df, :primary_ai_level)
                df.behavioral_group[i] = string(df.primary_ai_level[i])
            end
        end
    end

    return df
end

"""
Classify agent behavior based on AI usage proportions.
"""
function _classify_behavior(proportions::Dict{String,Float64})::String
    none_pct = get(proportions, "none", 0.0)
    basic_pct = get(proportions, "basic", 0.0)
    advanced_pct = get(proportions, "advanced", 0.0)
    premium_pct = get(proportions, "premium", 0.0)

    if none_pct > 0.8
        return "AI Skeptic"
    elseif premium_pct > 0.6 || advanced_pct > 0.6
        return "AI Devotee"
    elseif basic_pct > 0.7
        return "Cautious Adopter"
    else
        # Count number of levels used significantly
        levels_used = sum([none_pct > 0.1, basic_pct > 0.1, advanced_pct > 0.1, premium_pct > 0.1])
        if levels_used >= 3
            return "Adaptive User"
        end
    end
    return "Standard User"
end

# ============================================================================
# PERFORMANCE ANALYSIS
# ============================================================================

"""
Analyze performance outcomes by AI tier and behavioral group.
"""
function analyze_performance_outcomes(framework::AnalysisFramework)::Dict{String,Any}
    results = Dict{String,Any}()
    agent_df = framework.agent_df

    if isempty(agent_df)
        return results
    end

    # Survival rate by AI tier
    if hasproperty(agent_df, :primary_ai_level) && hasproperty(agent_df, :survived)
        survival_by_ai = Dict{String,Float64}()
        for tier in unique(agent_df.primary_ai_level)
            tier_data = filter(row -> row.primary_ai_level == tier, agent_df)
            if nrow(tier_data) > 0
                survival_by_ai[normalize_ai_label(tier)] = mean(tier_data.survived)
            end
        end
        results["survival_rate_by_ai"] = survival_by_ai
    end

    # Survival rate by behavioral group
    if hasproperty(agent_df, :behavioral_group) && hasproperty(agent_df, :survived)
        survival_by_group = Dict{String,Float64}()
        for group in unique(agent_df.behavioral_group)
            group_data = filter(row -> row.behavioral_group == group, agent_df)
            if nrow(group_data) > 0
                survival_by_group[string(group)] = mean(group_data.survived)
            end
        end
        results["survival_rate_by_behavioral_group"] = survival_by_group
    end

    # Wealth distribution by AI tier
    if hasproperty(agent_df, :primary_ai_level) && hasproperty(agent_df, :final_capital)
        wealth_by_ai = Dict{String,Dict{String,Float64}}()
        for tier in unique(agent_df.primary_ai_level)
            tier_data = filter(row -> row.primary_ai_level == tier, agent_df)
            capitals = collect(skipmissing(tier_data.final_capital))
            if !isempty(capitals)
                wealth_by_ai[normalize_ai_label(tier)] = Dict(
                    "mean" => mean(capitals),
                    "std" => length(capitals) > 1 ? std(capitals) : 0.0,
                    "median" => median(capitals)
                )
            end
        end
        results["wealth_distribution_by_ai"] = wealth_by_ai
    end

    # Performance drivers correlation
    corr_cols = [:capital_growth, :uncertainty_tolerance, :innovativeness,
                 :exploration_tendency, :ai_trust, :innovations, :portfolio_diversity]
    available_cols = filter(c -> hasproperty(agent_df, c), corr_cols)

    if length(available_cols) >= 2 && :capital_growth in available_cols
        correlations = Dict{String,Float64}()
        target = collect(skipmissing(agent_df.capital_growth))

        for col in available_cols
            if col != :capital_growth
                other = collect(skipmissing(agent_df[!, col]))
                if length(target) == length(other) && length(target) > 2
                    correlations[string(col)] = cor(Float64.(target), Float64.(other))
                end
            end
        end
        results["performance_drivers_correlation"] = correlations
    end

    return results
end

"""
Analyze AI augmentation effects across different metrics.
"""
function analyze_ai_augmentation_effects(framework::AnalysisFramework)::Dict{String,Any}
    results = Dict{String,Any}()
    agent_df = framework.agent_df
    decision_df = framework.decision_df
    matured_df = framework.matured_df

    if !isempty(agent_df)
        ai_levels = CANONICAL_AI_LEVELS
        metrics = [:capital_growth, :final_capital, :innovations, :portfolio_diversity]

        tier_col = hasproperty(agent_df, :primary_ai_canonical) ? :primary_ai_canonical : :primary_ai_level

        for metric in metrics
            if hasproperty(agent_df, metric)
                values = Dict{String,Float64}()
                for level in ai_levels
                    mask = agent_df[!, tier_col] .== level
                    if any(mask)
                        level_data = agent_df[mask, metric]
                        values[level] = _safe_mean(level_data)
                    end
                end
                if !isempty(values)
                    results[string(metric)] = values
                end
            end
        end
    end

    # Investment success rate from matured investments
    if !isempty(matured_df) && hasproperty(matured_df, :investment_amount)
        valid_matured = filter(row -> row.investment_amount > 0, matured_df)
        if !isempty(valid_matured) && hasproperty(valid_matured, :success)
            success_by_ai = Dict{String,Float64}()
            for level in unique(valid_matured.ai_level_used)
                level_data = filter(row -> row.ai_level_used == level, valid_matured)
                if nrow(level_data) > 0
                    success_by_ai[string(level)] = mean(level_data.success)
                end
            end
            results["investment_success_rate_by_ai"] = success_by_ai
        end
    end

    return results
end

"""
Analyze Knightian uncertainty dynamics.
"""
function analyze_uncertainty_dynamics(framework::AnalysisFramework)::Dict{String,Any}
    results = Dict{String,Any}()
    uncertainty_df = framework.uncertainty_df
    market_df = framework.market_df

    if !isempty(uncertainty_df)
        # Evolution of uncertainty dimensions over time
        uncertainty_cols = [:actor_ignorance_level, :practical_indeterminism_level,
                           :agentic_novelty_level, :competitive_recursion_level]
        available_cols = filter(c -> hasproperty(uncertainty_df, c), uncertainty_cols)

        if !isempty(available_cols) && hasproperty(uncertainty_df, :round)
            evolution = Dict{Int,Dict{String,Float64}}()
            for (key, group) in pairs(groupby(uncertainty_df, :round))
                round_num = key.round
                round_vals = Dict{String,Float64}()
                for col in available_cols
                    round_vals[string(col)] = _safe_mean(group[!, col])
                end
                evolution[Int(round_num)] = round_vals
            end
            results["uncertainty_evolution"] = evolution
        end
    end

    # Market regime frequency
    if !isempty(market_df) && hasproperty(market_df, :regime)
        regime_counts = Dict{String,Int}()
        for regime in market_df.regime
            regime_str = string(regime)
            regime_counts[regime_str] = get(regime_counts, regime_str, 0) + 1
        end
        total = sum(values(regime_counts))
        results["market_regime_frequency"] = Dict(k => v/total for (k, v) in regime_counts)
    end

    return results
end

"""
Analyze emergent behaviors including AI adoption trends.
"""
function analyze_emergent_behaviors(framework::AnalysisFramework)::Dict{String,Any}
    results = Dict{String,Any}()
    summary_df = framework.summary_df
    decision_df = framework.decision_df

    if !isempty(summary_df) && hasproperty(summary_df, :round)
        # AI adoption trends
        adoption_cols = [:ai_share_none, :ai_share_basic, :ai_share_advanced, :ai_share_premium]
        available_cols = filter(c -> hasproperty(summary_df, c), adoption_cols)

        if !isempty(available_cols)
            adoption_trends = Dict{Int,Dict{String,Float64}}()
            for (key, group) in pairs(groupby(summary_df, :round))
                round_num = key.round
                round_vals = Dict{String,Float64}()
                for col in available_cols
                    round_vals[string(col)] = _safe_mean(group[!, col])
                end
                adoption_trends[Int(round_num)] = round_vals
            end
            results["ai_adoption_trends"] = adoption_trends
        end
    end

    # Action counts
    if !isempty(decision_df) && hasproperty(decision_df, :action)
        action_counts = Dict{String,Int}()
        for action in decision_df.action
            action_str = string(action)
            action_counts[action_str] = get(action_counts, action_str, 0) + 1
        end
        results["action_counts"] = action_counts
    end

    return results
end

# ============================================================================
# RESEARCH TABLE EXPORT
# ============================================================================

"""
Export research tables for AMJ-style publication.
"""
function export_research_tables(framework::AnalysisFramework; output_dir::Union{String,Nothing}=nothing)::Dict{String,String}
    exported = Dict{String,String}()

    if isnothing(output_dir)
        output_dir = joinpath(framework.results_dir, "tables")
    end
    isdir(output_dir) || mkpath(output_dir)

    # Helper to write table
    function write_table(df::DataFrame, filename::String)
        if isempty(df)
            return nothing
        end
        path = joinpath(output_dir, filename)
        open(path, "w") do io
            # Write header
            println(io, join(names(df), ","))
            # Write rows
            for row in eachrow(df)
                println(io, join([string(v) for v in row], ","))
            end
        end
        return path
    end

    # Survival summary by AI tier
    survival_summary = compute_survival_summary(framework.agent_df)
    path = write_table(survival_summary, "survival_by_ai_tier.csv")
    if !isnothing(path)
        exported["survival_by_ai_tier"] = path
    end

    # Capital summary by AI tier
    capital_summary = compute_capital_summary(framework.agent_df)
    path = write_table(capital_summary, "capital_by_ai_tier.csv")
    if !isnothing(path)
        exported["capital_by_ai_tier"] = path
    end

    # AI tier comparison
    tier_comparison = compute_ai_tier_comparison(framework.agent_df)
    path = write_table(tier_comparison, "ai_tier_comparison.csv")
    if !isnothing(path)
        exported["ai_tier_comparison"] = path
    end

    println("Exported $(length(exported)) research tables to $output_dir")
    return exported
end

# ============================================================================
# SURVIVAL ANALYSIS
# ============================================================================

"""
Compute survival summary by AI tier.
"""
function compute_survival_summary(agent_df::DataFrame; tier_col::Symbol=:primary_ai_level)
    if isempty(agent_df) || !hasproperty(agent_df, tier_col) || !hasproperty(agent_df, :survived)
        return DataFrame()
    end

    results = DataFrame(
        ai_tier=String[],
        n_agents=Int[],
        n_survivors=Int[],
        survival_rate=Float64[],
        se=Float64[],
        ci_lower=Float64[],
        ci_upper=Float64[]
    )

    for tier in unique(agent_df[!, tier_col])
        tier_df = filter(row -> row[tier_col] == tier, agent_df)
        n = nrow(tier_df)

        if n == 0
            continue
        end

        survived = sum(tier_df.survived .== 1)
        rate = survived / n

        # Wilson score interval for proportion
        z = 1.96
        denominator = 1 + z^2/n
        center = (rate + z^2/(2*n)) / denominator
        spread = z * sqrt((rate*(1-rate) + z^2/(4*n))/n) / denominator
        ci_lower = max(0, center - spread)
        ci_upper = min(1, center + spread)
        se = spread / z

        push!(results, (
            ai_tier=string(tier),
            n_agents=n,
            n_survivors=survived,
            survival_rate=rate,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper
        ))
    end

    return sort(results, :ai_tier)
end

# ============================================================================
# CAPITAL ANALYSIS
# ============================================================================

"""
Compute capital summary by AI tier.
"""
function compute_capital_summary(agent_df::DataFrame; tier_col::Symbol=:primary_ai_level)
    if isempty(agent_df) || !hasproperty(agent_df, tier_col) || !hasproperty(agent_df, :final_capital)
        return DataFrame()
    end

    results = DataFrame(
        ai_tier=String[],
        n_agents=Int[],
        mean_capital=Float64[],
        median_capital=Float64[],
        std_capital=Float64[],
        min_capital=Float64[],
        max_capital=Float64[]
    )

    for tier in unique(agent_df[!, tier_col])
        tier_df = filter(row -> row[tier_col] == tier, agent_df)

        if nrow(tier_df) == 0
            continue
        end

        capitals = collect(skipmissing(tier_df.final_capital))

        if isempty(capitals)
            continue
        end

        push!(results, (
            ai_tier=string(tier),
            n_agents=length(capitals),
            mean_capital=mean(capitals),
            median_capital=median(capitals),
            std_capital=length(capitals) > 1 ? std(capitals) : 0.0,
            min_capital=minimum(capitals),
            max_capital=maximum(capitals)
        ))
    end

    return sort(results, :ai_tier)
end

# ============================================================================
# AI TIER COMPARISON
# ============================================================================

"""
Compute comprehensive AI tier comparison.
"""
function compute_ai_tier_comparison(agent_df::DataFrame)
    survival_summary = compute_survival_summary(agent_df)
    capital_summary = compute_capital_summary(agent_df)

    if isempty(survival_summary) || isempty(capital_summary)
        return DataFrame()
    end

    # Join summaries
    comparison = leftjoin(survival_summary, capital_summary, on=:ai_tier)

    return comparison
end

# ============================================================================
# SWEEP AGGREGATION
# ============================================================================

"""
Aggregate results from multiple simulation runs.
"""
function aggregate_sweep_results(run_dirs::Vector{String})::DataFrame
    all_results = DataFrame[]

    for dir in run_dirs
        if !isdir(dir)
            continue
        end

        try
            framework = AnalysisFramework(dir)

            if !isempty(framework.agent_df)
                comparison = compute_ai_tier_comparison(framework.agent_df)
                if !isempty(comparison)
                    comparison.run_id .= basename(dir)
                    push!(all_results, comparison)
                end
            end
        catch e
            @warn "Failed to load results from $dir" exception=e
        end
    end

    if isempty(all_results)
        return DataFrame()
    end

    return vcat(all_results...)
end

"""
Compute summary statistics across sweep runs.
"""
function summarize_sweep_results(sweep_df::DataFrame)
    if isempty(sweep_df) || !hasproperty(sweep_df, :ai_tier) || !hasproperty(sweep_df, :survival_rate)
        return DataFrame()
    end

    results = DataFrame(
        ai_tier=String[],
        n_runs=Int[],
        mean_survival=Float64[],
        std_survival=Float64[],
        mean_capital=Float64[],
        std_capital=Float64[]
    )

    for tier in unique(sweep_df.ai_tier)
        tier_data = filter(row -> row.ai_tier == tier, sweep_df)

        if nrow(tier_data) == 0
            continue
        end

        survival_rates = collect(skipmissing(tier_data.survival_rate))
        capitals = hasproperty(tier_data, :mean_capital) ?
                   collect(skipmissing(tier_data.mean_capital)) : Float64[]

        push!(results, (
            ai_tier=tier,
            n_runs=length(survival_rates),
            mean_survival=mean(survival_rates),
            std_survival=length(survival_rates) > 1 ? std(survival_rates) : 0.0,
            mean_capital=isempty(capitals) ? NaN : mean(capitals),
            std_capital=length(capitals) > 1 ? std(capitals) : 0.0
        ))
    end

    return sort(results, :ai_tier)
end

# ============================================================================
# FULL ANALYSIS
# ============================================================================

"""
Run full analysis on results directory.

This is the main entry point for comprehensive analysis of simulation results.
"""
function run_full_analysis(results_dir::String)::Dict{String,Any}
    println("\n" * "="^70)
    println("COMPREHENSIVE ANALYSIS FRAMEWORK")
    println("="^70)
    println("Results directory: $results_dir")
    println("="^70 * "\n")

    framework = AnalysisFramework(results_dir)
    results = Dict{String,Any}()

    # Check data availability
    if isempty(framework.agent_df) && isempty(framework.decision_df)
        @warn "Agent and decision data not found. Skipping analysis."
        return results
    end

    # 1. Survival analysis
    println("Computing survival summary...")
    results["survival_summary"] = compute_survival_summary(framework.agent_df)
    if !isempty(results["survival_summary"])
        println(results["survival_summary"])
    end

    # 2. Capital analysis
    println("\nComputing capital summary...")
    results["capital_summary"] = compute_capital_summary(framework.agent_df)
    if !isempty(results["capital_summary"])
        println(results["capital_summary"])
    end

    # 3. AI tier comparison
    println("\nComputing AI tier comparison...")
    results["tier_comparison"] = compute_ai_tier_comparison(framework.agent_df)

    # 4. Performance outcomes
    println("\nAnalyzing performance outcomes...")
    results["performance_outcomes"] = analyze_performance_outcomes(framework)

    # 5. AI augmentation effects
    println("\nAnalyzing AI augmentation effects...")
    results["ai_augmentation_effects"] = analyze_ai_augmentation_effects(framework)

    # 6. Uncertainty dynamics
    println("\nAnalyzing uncertainty dynamics...")
    results["uncertainty_dynamics"] = analyze_uncertainty_dynamics(framework)

    # 7. Emergent behaviors
    println("\nAnalyzing emergent behaviors...")
    results["emergent_behaviors"] = analyze_emergent_behaviors(framework)

    # Summary statistics
    results["summary_df"] = framework.summary_df

    # Store analyses in framework
    framework.analyses["performance_outcomes"] = results["performance_outcomes"]
    framework.analyses["ai_augmentation_effects"] = results["ai_augmentation_effects"]
    framework.analyses["uncertainty_dynamics"] = results["uncertainty_dynamics"]
    framework.analyses["emergent_behaviors"] = results["emergent_behaviors"]

    println("\n" * "="^70)
    println("ANALYSIS COMPLETE")
    println("="^70)

    return results
end

"""
Run full analysis and export research tables.
"""
function run_full_analysis_with_export(results_dir::String; output_dir::Union{String,Nothing}=nothing)::Tuple{Dict{String,Any},Dict{String,String}}
    results = run_full_analysis(results_dir)
    framework = AnalysisFramework(results_dir)
    exported = export_research_tables(framework; output_dir=output_dir)
    return (results, exported)
end

# ============================================================================
# STATISTICAL TESTS
# ============================================================================

"""
Perform Kruskal-Wallis H-test across AI levels for a given metric.
Non-parametric test for comparing distributions.
"""
function kruskal_wallis_test(data::DataFrame, metric::Symbol; group_col::Symbol=:primary_ai_level)::Dict{String,Any}
    if isempty(data) || !hasproperty(data, metric) || !hasproperty(data, group_col)
        return Dict{String,Any}("status" => "Insufficient data")
    end

    groups = Dict{String,Vector{Float64}}()
    for tier in CANONICAL_AI_LEVELS
        tier_data = filter(row -> normalize_ai_label(string(row[group_col])) == tier, data)
        if nrow(tier_data) > 2
            values = collect(skipmissing(tier_data[!, metric]))
            valid_values = filter(isfinite, values)
            if length(valid_values) > 2
                groups[tier] = Float64.(valid_values)
            end
        end
    end

    if length(groups) < 2
        return Dict{String,Any}("status" => "Less than 2 groups with sufficient data")
    end

    # Compute Kruskal-Wallis H statistic
    all_values = Float64[]
    group_labels = String[]
    for (label, vals) in groups
        append!(all_values, vals)
        append!(group_labels, fill(label, length(vals)))
    end

    n = length(all_values)
    ranks = tiedrank(all_values)

    # Compute rank sums for each group
    rank_sums = Dict{String,Float64}()
    group_sizes = Dict{String,Int}()
    for (label, vals) in groups
        group_sizes[label] = length(vals)
        rank_sums[label] = 0.0
    end

    idx = 1
    for (label, vals) in groups
        for _ in vals
            rank_sums[label] += ranks[idx]
            idx += 1
        end
    end

    # H statistic: H = (12 / (n(n+1))) * sum(R_i^2 / n_i) - 3(n+1)
    h_stat = 0.0
    for (label, r_sum) in rank_sums
        n_i = group_sizes[label]
        h_stat += (r_sum^2) / n_i
    end
    h_stat = (12.0 / (n * (n + 1))) * h_stat - 3.0 * (n + 1)

    # Degrees of freedom
    k = length(groups)
    df = k - 1

    # P-value approximation using chi-squared distribution
    # For large samples, H follows chi-squared with k-1 df
    p_value = 1.0 - cdf_chi_squared(h_stat, df)

    return Dict{String,Any}(
        "test" => "Kruskal-Wallis H-test",
        "metric" => string(metric),
        "h_statistic" => h_stat,
        "df" => df,
        "p_value" => p_value,
        "n_groups" => k,
        "group_sizes" => group_sizes,
        "is_significant" => p_value < 0.05
    )
end

"""
Chi-squared CDF approximation for p-value calculation.
"""
function cdf_chi_squared(x::Float64, df::Int)::Float64
    if x <= 0.0 || df <= 0
        return 0.0
    end
    # Use incomplete gamma function approximation
    # P(X <= x) = gamma(df/2, x/2) / Gamma(df/2)
    # Simple approximation for common cases
    k = df / 2.0
    theta = 2.0

    # Wilson-Hilferty approximation for chi-squared CDF
    z = ((x / df)^(1/3) - (1 - 2/(9*df))) / sqrt(2/(9*df))
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))
end

"""
Compute tied ranks for Kruskal-Wallis test.
"""
function tiedrank(x::Vector{Float64})::Vector{Float64}
    n = length(x)
    order = sortperm(x)
    ranks = zeros(Float64, n)

    i = 1
    while i <= n
        j = i
        # Find ties
        while j < n && x[order[j+1]] == x[order[j]]
            j += 1
        end
        # Assign average rank to ties
        avg_rank = (i + j) / 2.0
        for k in i:j
            ranks[order[k]] = avg_rank
        end
        i = j + 1
    end

    return ranks
end

"""
Perform Mann-Whitney U test between two groups.
"""
function mann_whitney_u_test(group1::Vector{Float64}, group2::Vector{Float64})::Dict{String,Any}
    n1 = length(group1)
    n2 = length(group2)

    if n1 < 2 || n2 < 2
        return Dict{String,Any}("status" => "Insufficient data in groups")
    end

    # Combine and rank
    combined = vcat(group1, group2)
    ranks = tiedrank(combined)

    # Sum of ranks for group 1
    r1 = sum(ranks[1:n1])

    # U statistic
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    # Normal approximation for p-value (large sample)
    mean_u = n1 * n2 / 2
    std_u = sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    z = (u - mean_u) / std_u
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))

    return Dict{String,Any}(
        "test" => "Mann-Whitney U",
        "u_statistic" => u,
        "z_score" => z,
        "p_value" => p_value,
        "n1" => n1,
        "n2" => n2,
        "is_significant" => p_value < 0.05
    )
end

"""
Run all statistical tests on framework data.
"""
function run_statistical_tests(framework::AnalysisFramework)::Dict{String,Any}
    results = Dict{String,Any}()

    agent_df = framework.agent_df

    if !isempty(agent_df)
        # Test survival rates across AI levels
        if hasproperty(agent_df, :survived)
            results["survival_kruskal"] = kruskal_wallis_test(agent_df, :survived)
        end

        # Test capital growth across AI levels
        if hasproperty(agent_df, :capital_growth)
            results["capital_growth_kruskal"] = kruskal_wallis_test(agent_df, :capital_growth)
        end

        # Test final capital across AI levels
        if hasproperty(agent_df, :final_capital)
            results["final_capital_kruskal"] = kruskal_wallis_test(agent_df, :final_capital)
        end

        # Pairwise comparisons for significant results
        if hasproperty(agent_df, :primary_ai_level) && hasproperty(agent_df, :capital_growth)
            pairwise = Dict{String,Any}()
            for i in 1:length(CANONICAL_AI_LEVELS)
                for j in (i+1):length(CANONICAL_AI_LEVELS)
                    tier1 = CANONICAL_AI_LEVELS[i]
                    tier2 = CANONICAL_AI_LEVELS[j]

                    g1 = filter(row -> normalize_ai_label(string(row.primary_ai_level)) == tier1, agent_df)
                    g2 = filter(row -> normalize_ai_label(string(row.primary_ai_level)) == tier2, agent_df)

                    if nrow(g1) > 2 && nrow(g2) > 2
                        v1 = Float64.(collect(skipmissing(g1.capital_growth)))
                        v2 = Float64.(collect(skipmissing(g2.capital_growth)))
                        v1 = filter(isfinite, v1)
                        v2 = filter(isfinite, v2)

                        if length(v1) > 2 && length(v2) > 2
                            pairwise["$(tier1)_vs_$(tier2)"] = mann_whitney_u_test(v1, v2)
                        end
                    end
                end
            end
            results["pairwise_capital_growth"] = pairwise
        end
    end

    return results
end

# ============================================================================
# EFFECT SIZE CALCULATIONS
# ============================================================================

# Note: cohens_d and cliffs_delta are imported from Causal module

"""
Calculate effect sizes between AI tiers for a metric.
"""
function compute_effect_sizes(data::DataFrame, metric::Symbol; group_col::Symbol=:primary_ai_level)::Dict{String,Any}
    results = Dict{String,Any}()

    if isempty(data) || !hasproperty(data, metric) || !hasproperty(data, group_col)
        return results
    end

    # Extract groups
    groups = Dict{String,Vector{Float64}}()
    for tier in CANONICAL_AI_LEVELS
        tier_data = filter(row -> normalize_ai_label(string(row[group_col])) == tier, data)
        if nrow(tier_data) > 0
            values = Float64.(collect(skipmissing(tier_data[!, metric])))
            groups[tier] = filter(isfinite, values)
        end
    end

    # Pairwise effect sizes
    cohens = Dict{String,Float64}()
    cliffs = Dict{String,Float64}()

    for i in 1:length(CANONICAL_AI_LEVELS)
        for j in (i+1):length(CANONICAL_AI_LEVELS)
            tier1 = CANONICAL_AI_LEVELS[i]
            tier2 = CANONICAL_AI_LEVELS[j]

            if haskey(groups, tier1) && haskey(groups, tier2)
                g1, g2 = groups[tier1], groups[tier2]
                if length(g1) > 2 && length(g2) > 2
                    key = "$(tier1)_vs_$(tier2)"
                    cohens[key] = cohens_d(g1, g2)
                    cliffs[key] = cliffs_delta(g1, g2)
                end
            end
        end
    end

    results["cohens_d"] = cohens
    results["cliffs_delta"] = cliffs

    return results
end

# ============================================================================
# CLUSTERING ANALYSIS
# ============================================================================

"""
Perform k-means clustering on agent strategies.
Simple implementation without external ML dependencies.
"""
function cluster_agent_strategies(agent_df::DataFrame; k::Int=4, max_iter::Int=100)::Dict{String,Any}
    if isempty(agent_df) || nrow(agent_df) < k * 3
        return Dict{String,Any}("status" => "Insufficient data for clustering")
    end

    # Select features for clustering
    feature_cols = Symbol[]
    for col in [:uncertainty_tolerance, :innovativeness, :exploration_tendency, :portfolio_diversity, :ai_trust]
        if hasproperty(agent_df, col)
            push!(feature_cols, col)
        end
    end

    if length(feature_cols) < 2
        return Dict{String,Any}("status" => "Insufficient features for clustering")
    end

    # Extract feature matrix
    n = nrow(agent_df)
    m = length(feature_cols)
    X = zeros(Float64, n, m)

    for (j, col) in enumerate(feature_cols)
        vals = agent_df[!, col]
        for i in 1:n
            X[i, j] = ismissing(vals[i]) || !isfinite(vals[i]) ? 0.0 : Float64(vals[i])
        end
    end

    # Standardize features
    for j in 1:m
        col_mean = mean(X[:, j])
        col_std = std(X[:, j])
        if col_std > 0
            X[:, j] = (X[:, j] .- col_mean) ./ col_std
        end
    end

    # Simple k-means clustering
    labels, centroids = kmeans_simple(X, k; max_iter=max_iter)

    # Compute silhouette score approximation
    silhouette = compute_silhouette_score(X, labels, k)

    # Create cluster profiles
    cluster_profiles = Dict{Int,Dict{String,Float64}}()
    for cluster in 1:k
        cluster_mask = labels .== cluster
        profile = Dict{String,Float64}()

        for (j, col) in enumerate(feature_cols)
            cluster_vals = X[cluster_mask, j]
            profile[string(col)] = isempty(cluster_vals) ? 0.0 : mean(cluster_vals)
        end

        # Add outcome metrics if available
        cluster_agents = agent_df[cluster_mask, :]
        if hasproperty(agent_df, :capital_growth)
            cg = collect(skipmissing(cluster_agents.capital_growth))
            profile["capital_growth"] = isempty(cg) ? 0.0 : mean(cg)
        end
        if hasproperty(agent_df, :survived)
            surv = collect(skipmissing(cluster_agents.survived))
            profile["survival_rate"] = isempty(surv) ? 0.0 : mean(surv)
        end

        cluster_profiles[cluster] = profile
    end

    return Dict{String,Any}(
        "n_clusters" => k,
        "silhouette_score" => silhouette,
        "cluster_sizes" => [sum(labels .== i) for i in 1:k],
        "cluster_profiles" => cluster_profiles,
        "features_used" => string.(feature_cols)
    )
end

"""
Simple k-means clustering implementation.
"""
function kmeans_simple(X::Matrix{Float64}, k::Int; max_iter::Int=100)
    n, m = size(X)

    # Initialize centroids using k-means++ style
    centroids = zeros(Float64, k, m)
    centroids[1, :] = X[rand(1:n), :]

    for i in 2:k
        # Compute distances to nearest centroid
        dists = zeros(Float64, n)
        for j in 1:n
            min_dist = Inf
            for c in 1:(i-1)
                d = sum((X[j, :] .- centroids[c, :]).^2)
                min_dist = min(min_dist, d)
            end
            dists[j] = min_dist
        end

        # Select next centroid with probability proportional to distance
        probs = dists ./ sum(dists)
        cumprobs = cumsum(probs)
        r = rand()
        idx = searchsortedfirst(cumprobs, r)
        centroids[i, :] = X[min(idx, n), :]
    end

    # Iterate
    labels = zeros(Int, n)
    for _ in 1:max_iter
        # Assign labels
        old_labels = copy(labels)
        for i in 1:n
            min_dist = Inf
            min_label = 1
            for c in 1:k
                d = sum((X[i, :] .- centroids[c, :]).^2)
                if d < min_dist
                    min_dist = d
                    min_label = c
                end
            end
            labels[i] = min_label
        end

        # Check convergence
        if labels == old_labels
            break
        end

        # Update centroids
        for c in 1:k
            cluster_points = X[labels .== c, :]
            if size(cluster_points, 1) > 0
                centroids[c, :] = mean(cluster_points, dims=1)
            end
        end
    end

    return labels, centroids
end

"""
Compute silhouette score for clustering quality.
"""
function compute_silhouette_score(X::Matrix{Float64}, labels::Vector{Int}, k::Int)::Float64
    n = size(X, 1)
    if n < 2 || k < 2
        return 0.0
    end

    silhouettes = zeros(Float64, n)

    for i in 1:n
        # Compute a(i) - mean distance to same cluster
        same_cluster = findall(labels .== labels[i])
        if length(same_cluster) <= 1
            silhouettes[i] = 0.0
            continue
        end

        a_i = 0.0
        for j in same_cluster
            if j != i
                a_i += sqrt(sum((X[i, :] .- X[j, :]).^2))
            end
        end
        a_i /= (length(same_cluster) - 1)

        # Compute b(i) - min mean distance to other clusters
        b_i = Inf
        for c in 1:k
            if c == labels[i]
                continue
            end
            other_cluster = findall(labels .== c)
            if isempty(other_cluster)
                continue
            end
            mean_dist = 0.0
            for j in other_cluster
                mean_dist += sqrt(sum((X[i, :] .- X[j, :]).^2))
            end
            mean_dist /= length(other_cluster)
            b_i = min(b_i, mean_dist)
        end

        if b_i == Inf
            b_i = 0.0
        end

        # Silhouette coefficient
        max_ab = max(a_i, b_i)
        silhouettes[i] = max_ab > 0 ? (b_i - a_i) / max_ab : 0.0
    end

    return mean(silhouettes)
end

# ============================================================================
# INNOVATION EQUILIBRIUM ANALYSIS
# ============================================================================

"""
Analyze innovation equilibrium trap patterns.
"""
function analyze_innovation_equilibrium(framework::AnalysisFramework)::Dict{String,Any}
    results = Dict{String,Any}()
    matured_df = framework.matured_df
    summary_df = framework.summary_df

    # Matured investment analysis
    if !isempty(matured_df) && hasproperty(matured_df, :investment_amount)
        valid_matured = filter(row -> row.investment_amount > 0, matured_df)

        if !isempty(valid_matured)
            # Add return multiple
            return_multiples = valid_matured.capital_returned ./ valid_matured.investment_amount

            # Success rate by AI level
            if hasproperty(valid_matured, :ai_level_used) && hasproperty(valid_matured, :success)
                success_by_ai = Dict{String,Float64}()
                return_by_ai = Dict{String,Float64}()

                for tier in CANONICAL_AI_LEVELS
                    tier_data = filter(row -> normalize_ai_label(string(get(row, :ai_level_used, "none"))) == tier, valid_matured)
                    if nrow(tier_data) > 0
                        success_by_ai[tier] = mean(tier_data.success)
                        tier_returns = tier_data.capital_returned ./ tier_data.investment_amount
                        return_by_ai[tier] = mean(filter(isfinite, tier_returns))
                    end
                end

                results["matured_success_rate_by_ai"] = success_by_ai
                results["matured_return_multiple_by_ai"] = return_by_ai
            end

            # Sector success matrix
            if hasproperty(valid_matured, :sector) && hasproperty(valid_matured, :ai_level_used)
                sector_success = Dict{String,Dict{String,Float64}}()
                for sector in unique(valid_matured.sector)
                    sector_data = filter(row -> row.sector == sector, valid_matured)
                    sector_success[string(sector)] = Dict{String,Float64}()
                    for tier in CANONICAL_AI_LEVELS
                        tier_data = filter(row -> normalize_ai_label(string(get(row, :ai_level_used, "none"))) == tier, sector_data)
                        if nrow(tier_data) > 0
                            sector_success[string(sector)][tier] = mean(tier_data.success)
                        end
                    end
                end
                results["sector_success_matrix"] = sector_success
            end
        end
    end

    # Innovation success trend
    if !isempty(summary_df) && hasproperty(summary_df, :innovation_success_rate) && hasproperty(summary_df, :round)
        trend = Dict{Int,Float64}()
        for row in eachrow(summary_df)
            round_num = Int(row.round)
            val = row.innovation_success_rate
            if !ismissing(val) && isfinite(val)
                trend[round_num] = val
            end
        end
        results["innovation_success_trend"] = trend
    end

    # Premium AI share trend
    if !isempty(summary_df) && hasproperty(summary_df, :ai_share_premium) && hasproperty(summary_df, :round)
        trend = Dict{Int,Float64}()
        for row in eachrow(summary_df)
            round_num = Int(row.round)
            val = row.ai_share_premium
            if !ismissing(val) && isfinite(val)
                trend[round_num] = val
            end
        end
        results["premium_ai_share_trend"] = trend
    end

    return results
end

# ============================================================================
# INFORMATION PARADOX ANALYZER
# ============================================================================

"""
Analyze information paradox patterns in AI-augmented decision making.
"""
mutable struct InformationParadoxAnalyzer
    results_dir::String
    framework::AnalysisFramework
    stage_boundaries::Dict{String,Tuple{Int,Int}}
    paradox_metrics::Dict{String,Any}
end

"""
Create an InformationParadoxAnalyzer from results directory.
"""
function InformationParadoxAnalyzer(results_dir::String; n_rounds::Int=200)
    framework = AnalysisFramework(results_dir)

    # Define stage boundaries
    stage_boundaries = Dict{String,Tuple{Int,Int}}(
        "early" => (0, div(n_rounds, 3)),
        "middle" => (div(n_rounds, 3), 2 * div(n_rounds, 3)),
        "late" => (2 * div(n_rounds, 3), n_rounds)
    )

    return InformationParadoxAnalyzer(
        results_dir,
        framework,
        stage_boundaries,
        Dict{String,Any}()
    )
end

"""
Analyze temporal performance reversal patterns.
"""
function analyze_temporal_reversal(analyzer::InformationParadoxAnalyzer)::Dict{String,Any}
    results = Dict{String,Any}()
    decision_df = analyzer.framework.decision_df

    if isempty(decision_df) || !hasproperty(decision_df, :round)
        return results
    end

    for (stage_name, (start_round, end_round)) in analyzer.stage_boundaries
        stage_data = filter(row -> start_round <= row.round < end_round, decision_df)

        if isempty(stage_data)
            continue
        end

        stage_results = Dict{String,Float64}()

        # Performance by AI level in this stage
        if hasproperty(stage_data, :ai_level_used) && hasproperty(stage_data, :success)
            for tier in CANONICAL_AI_LEVELS
                tier_data = filter(row -> normalize_ai_label(string(get(row, :ai_level_used, "none"))) == tier, stage_data)
                if nrow(tier_data) > 0
                    stage_results[tier] = mean(skipmissing(tier_data.success))
                end
            end
        end

        results[stage_name] = stage_results
    end

    # Calculate reversal metric (early advantage becoming late disadvantage)
    if haskey(results, "early") && haskey(results, "late")
        reversal = Dict{String,Float64}()
        for tier in CANONICAL_AI_LEVELS
            early_perf = get(results["early"], tier, NaN)
            late_perf = get(results["late"], tier, NaN)
            if !isnan(early_perf) && !isnan(late_perf)
                reversal[tier] = late_perf - early_perf  # Negative = reversal
            end
        end
        results["performance_reversal"] = reversal
    end

    analyzer.paradox_metrics["temporal_reversal"] = results
    return results
end

"""
Analyze herding dynamics and diversity loss.
"""
function analyze_herding_dynamics(analyzer::InformationParadoxAnalyzer)::Dict{String,Any}
    results = Dict{String,Any}()
    decision_df = analyzer.framework.decision_df

    if isempty(decision_df) || !hasproperty(decision_df, :ai_level_used)
        return results
    end

    # Calculate HHI for opportunity concentration by AI level
    if hasproperty(decision_df, :opportunity_id) && hasproperty(decision_df, :round)
        herding_by_level = Dict{String,Vector{Float64}}()

        for tier in CANONICAL_AI_LEVELS
            tier_data = filter(row -> normalize_ai_label(string(get(row, :ai_level_used, "none"))) == tier, decision_df)

            if nrow(tier_data) < 10
                continue
            end

            # Calculate HHI per round
            hhi_values = Float64[]
            for (key, group) in pairs(groupby(tier_data, :round))
                opp_counts = Dict{Any,Int}()
                for opp in group.opportunity_id
                    if !ismissing(opp)
                        opp_counts[opp] = get(opp_counts, opp, 0) + 1
                    end
                end

                total = sum(values(opp_counts))
                if total > 0
                    hhi = sum((count / total)^2 for count in values(opp_counts))
                    push!(hhi_values, hhi)
                end
            end

            if !isempty(hhi_values)
                herding_by_level[tier] = hhi_values
            end
        end

        # Summarize herding metrics
        herding_summary = Dict{String,Dict{String,Float64}}()
        for (tier, hhi_vals) in herding_by_level
            herding_summary[tier] = Dict(
                "mean_hhi" => mean(hhi_vals),
                "max_hhi" => maximum(hhi_vals),
                "trend" => length(hhi_vals) > 1 ? (hhi_vals[end] - hhi_vals[1]) : 0.0
            )
        end

        results["herding_by_ai_level"] = herding_summary
    end

    analyzer.paradox_metrics["herding_dynamics"] = results
    return results
end

"""
Analyze non-AI agent advantages in avoiding paradox.
"""
function analyze_none_advantages(analyzer::InformationParadoxAnalyzer)::Dict{String,Any}
    advantages = Dict{String,Float64}()

    # Diversity preservation
    herding = get(analyzer.paradox_metrics, "herding_dynamics", Dict())
    if haskey(herding, "herding_by_ai_level")
        herding_by_level = herding["herding_by_ai_level"]
        none_hhi = get(get(herding_by_level, "none", Dict()), "mean_hhi", NaN)
        ai_hhis = Float64[]
        for tier in ["basic", "advanced", "premium"]
            hhi = get(get(herding_by_level, tier, Dict()), "mean_hhi", NaN)
            if !isnan(hhi)
                push!(ai_hhis, hhi)
            end
        end
        if !isnan(none_hhi) && !isempty(ai_hhis)
            advantages["anti_herding"] = mean(ai_hhis) - none_hhi  # Positive = advantage
        end
    end

    # Performance reversal resilience
    reversal = get(analyzer.paradox_metrics, "temporal_reversal", Dict())
    if haskey(reversal, "performance_reversal")
        perf_reversal = reversal["performance_reversal"]
        none_reversal = get(perf_reversal, "none", NaN)
        ai_reversals = Float64[]
        for tier in ["basic", "advanced", "premium"]
            rev = get(perf_reversal, tier, NaN)
            if !isnan(rev)
                push!(ai_reversals, rev)
            end
        end
        if !isnan(none_reversal) && !isempty(ai_reversals)
            advantages["reversal_resilience"] = none_reversal - mean(ai_reversals)
        end
    end

    analyzer.paradox_metrics["none_advantages"] = advantages
    return advantages
end

"""
Run full information paradox analysis.
"""
function run_paradox_analysis(analyzer::InformationParadoxAnalyzer)::Dict{String,Any}
    println("Running Information Paradox Analysis...")

    # 1. Temporal performance reversal
    analyze_temporal_reversal(analyzer)

    # 2. Herding dynamics
    analyze_herding_dynamics(analyzer)

    # 3. Non-AI advantages
    analyze_none_advantages(analyzer)

    return analyzer.paradox_metrics
end

# ============================================================================
# COMPREHENSIVE ADVANCED ANALYSIS
# ============================================================================

"""
Run comprehensive advanced analysis on framework.
"""
function run_advanced_analysis(framework::AnalysisFramework)::Dict{String,Any}
    results = Dict{String,Any}()

    println("Running advanced statistical analysis...")

    # 1. Statistical tests
    results["statistical_tests"] = run_statistical_tests(framework)

    # 2. Effect sizes
    if !isempty(framework.agent_df) && hasproperty(framework.agent_df, :capital_growth)
        results["effect_sizes"] = compute_effect_sizes(framework.agent_df, :capital_growth)
    end

    # 3. Clustering analysis
    results["clustering"] = cluster_agent_strategies(framework.agent_df)

    # 4. Innovation equilibrium
    results["innovation_equilibrium"] = analyze_innovation_equilibrium(framework)

    # 5. Information paradox (if we have decision data)
    if !isempty(framework.decision_df)
        paradox_analyzer = InformationParadoxAnalyzer(framework.results_dir)
        results["information_paradox"] = run_paradox_analysis(paradox_analyzer)
    end

    return results
end

"""
Run full analysis including advanced methods.
"""
function run_complete_analysis(results_dir::String)::Dict{String,Any}
    println("\n" * "="^70)
    println("COMPLETE ANALYSIS FRAMEWORK (WITH ADVANCED METHODS)")
    println("="^70)

    # Run basic analysis
    basic_results = run_full_analysis(results_dir)

    # Run advanced analysis
    framework = AnalysisFramework(results_dir)
    advanced_results = run_advanced_analysis(framework)

    # Merge results
    all_results = merge(basic_results, advanced_results)

    println("\n" * "="^70)
    println("COMPLETE ANALYSIS FINISHED")
    println("="^70)

    return all_results
end

end # module Analysis
