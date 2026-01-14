"""
Visualization utilities for GlimpseABM.jl using CairoMakie

Port of: glimpse_abm/analysis.py (ComprehensiveVisualizationSuite)
"""

module Visualization

using Statistics
using DataFrames

# Conditional CairoMakie loading
const HAS_MAKIE = Ref(false)

function __init__()
    try
        @eval using CairoMakie
        HAS_MAKIE[] = true
    catch
        @warn "CairoMakie not available. Visualization functions will be disabled."
    end
end

# Core plotting functions
export plot_survival_by_tier, plot_capital_distribution, plot_ai_adoption_over_time
export plot_uncertainty_dynamics, plot_effect_sizes, plot_survival_curves
export create_summary_dashboard, save_all_figures

# Dashboard exports
export create_performance_dashboard, create_temporal_dynamics_dashboard
export create_innovation_dashboard, create_concentration_dashboard
export create_ai_uncertainty_dashboard, create_perception_storyboard
export create_decision_storyboard, create_market_storyboard
export create_all_visualizations

# Comprehensive suite export
export ComprehensiveVisualizationSuite

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================

# Color palette for AI tiers (matched to Python)
const AI_TIER_COLORS = Dict(
    "none" => "#95a5a6",      # Gray
    "basic" => "#3498db",     # Blue
    "advanced" => "#f39c12",  # Orange
    "premium" => "#e74c3c"    # Red
)

# Behavioral group colors
const BEHAVIORAL_GROUP_COLORS = Dict(
    "AI Skeptic" => "#95a5a6",
    "Cautious Adopter" => "#3498db",
    "Standard User" => "#f39c12",
    "Adaptive User" => "#2ecc71",
    "AI Devotee" => "#e74c3c",
    "Unknown" => "#7f8c8d"
)

# Uncertainty dimension colors
const UNCERTAINTY_COLORS = Dict(
    "actor_ignorance" => "#9b59b6",
    "practical_indeterminism" => "#34495e",
    "agentic_novelty" => "#27ae60",
    "competitive_recursion" => "#d35400"
)

const AI_TIER_ORDER = ["none", "basic", "advanced", "premium"]

"""
Get color for AI tier.
"""
function tier_color(tier::String)
    return get(AI_TIER_COLORS, lowercase(tier), "#000000")
end

"""
Get color for uncertainty dimension.
"""
function uncertainty_color(dim::String)
    return get(UNCERTAINTY_COLORS, lowercase(dim), "#333333")
end

"""
Canonical AI level to display name.
"""
function canonical_to_display(level::String)::String
    mapping = Dict(
        "none" => "None",
        "basic" => "Basic",
        "advanced" => "Advanced",
        "premium" => "Premium"
    )
    return get(mapping, lowercase(level), titlecase(level))
end

# ============================================================================
# SURVIVAL RATE PLOTS
# ============================================================================

"""
Plot survival rates by AI tier.

Parameters
----------
survival_rates : Dict{String,Float64} - Map of tier -> survival rate
output_path : String - Path to save figure (optional)

Returns
-------
Figure object (if CairoMakie available)
"""
function plot_survival_by_tier(
    survival_rates::Dict{String,Float64};
    output_path::Union{String,Nothing}=nothing,
    title::String="Survival Rate by AI Tier"
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    # Sort by tier order
    tiers = String[]
    rates = Float64[]
    colors = String[]

    for tier in AI_TIER_ORDER
        if haskey(survival_rates, tier)
            push!(tiers, titlecase(tier))
            push!(rates, survival_rates[tier] * 100)
            push!(colors, tier_color(tier))
        end
    end

    if isempty(tiers)
        return nothing
    end

    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1],
        xlabel="AI Tier",
        ylabel="Survival Rate (%)",
        title=title,
        xticks=(1:length(tiers), tiers)
    )

    barplot!(ax, 1:length(tiers), rates, color=colors)
    ylims!(ax, 0, 100)

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

"""
Plot survival rates with confidence intervals.
"""
function plot_survival_with_ci(
    survival_data::DataFrame;
    tier_col::Symbol=:ai_tier,
    rate_col::Symbol=:survival_rate,
    ci_lower_col::Symbol=:ci_lower,
    ci_upper_col::Symbol=:ci_upper,
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1],
        xlabel="AI Tier",
        ylabel="Survival Rate (%)",
        title="Survival Rate by AI Tier (95% CI)"
    )

    tiers = String.(survival_data[!, tier_col])
    rates = survival_data[!, rate_col] .* 100
    ci_low = survival_data[!, ci_lower_col] .* 100
    ci_high = survival_data[!, ci_upper_col] .* 100

    # Error bars
    scatter!(ax, 1:length(tiers), rates, markersize=15, color=:steelblue)
    rangebars!(ax, 1:length(tiers), ci_low, ci_high, color=:steelblue, whiskerwidth=10)

    ax.xticks = (1:length(tiers), titlecase.(tiers))
    ylims!(ax, 0, 100)

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

# ============================================================================
# CAPITAL DISTRIBUTION PLOTS
# ============================================================================

"""
Plot capital distribution by AI tier.
"""
function plot_capital_distribution(
    df::DataFrame;
    tier_col::Symbol=:primary_ai_level,
    capital_col::Symbol=:final_capital,
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    if !hasproperty(df, tier_col) || !hasproperty(df, capital_col)
        @warn "Missing required columns"
        return nothing
    end

    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1],
        xlabel="Final Capital",
        ylabel="Density",
        title="Capital Distribution by AI Tier"
    )

    for tier in AI_TIER_ORDER
        tier_data = filter(row -> lowercase(string(row[tier_col])) == tier, df)
        if nrow(tier_data) > 10
            capitals = tier_data[!, capital_col]
            # Simple histogram approximation using density
            density!(ax, capitals, label=titlecase(tier), color=(tier_color(tier), 0.6))
        end
    end

    axislegend(ax, position=:rt)

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

"""
Plot boxplot of capital by AI tier.
"""
function plot_capital_boxplot(
    df::DataFrame;
    tier_col::Symbol=:primary_ai_level,
    capital_col::Symbol=:final_capital,
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    if !hasproperty(df, tier_col) || !hasproperty(df, capital_col)
        return nothing
    end

    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1],
        xlabel="AI Tier",
        ylabel="Final Capital",
        title="Final Capital by AI Tier"
    )

    positions = Float64[]
    data_vectors = Vector{Float64}[]
    labels = String[]

    for (i, tier) in enumerate(AI_TIER_ORDER)
        tier_data = filter(row -> lowercase(string(row[tier_col])) == tier, df)
        if nrow(tier_data) > 0
            push!(positions, Float64(i))
            push!(data_vectors, collect(skipmissing(tier_data[!, capital_col])))
            push!(labels, titlecase(tier))
        end
    end

    if isempty(data_vectors)
        return nothing
    end

    boxplot!(ax, repeat(positions, inner=[length(d) for d in data_vectors]),
             vcat(data_vectors...))

    ax.xticks = (positions, labels)

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

# ============================================================================
# TIME SERIES PLOTS
# ============================================================================

"""
Plot AI adoption over time.
"""
function plot_ai_adoption_over_time(
    history_df::DataFrame;
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    # Look for AI share columns
    share_cols = [:ai_none_count, :ai_basic_count, :ai_advanced_count, :ai_premium_count]
    has_shares = all(hasproperty(history_df, col) for col in share_cols)

    if !has_shares || !hasproperty(history_df, :round)
        @warn "Missing required columns for adoption plot"
        return nothing
    end

    rounds = history_df.round
    n_alive = hasproperty(history_df, :n_alive) ? history_df.n_alive : ones(length(rounds))

    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1],
        xlabel="Round",
        ylabel="Share of Agents",
        title="AI Adoption Over Time"
    )

    for (tier, col) in zip(AI_TIER_ORDER, share_cols)
        if hasproperty(history_df, col)
            shares = history_df[!, col] ./ max.(n_alive, 1)
            lines!(ax, rounds, shares, label=titlecase(tier), color=tier_color(tier), linewidth=2)
        end
    end

    axislegend(ax, position=:rt)
    ylims!(ax, 0, 1)

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

"""
Plot survival rate over time.
"""
function plot_survival_over_time(
    history_df::DataFrame;
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    if !hasproperty(history_df, :survival_rate) || !hasproperty(history_df, :round)
        return nothing
    end

    fig = Figure(size=(700, 400))
    ax = Axis(fig[1, 1],
        xlabel="Round",
        ylabel="Survival Rate",
        title="Agent Survival Over Time"
    )

    rounds = history_df.round
    survival = history_df.survival_rate

    lines!(ax, rounds, survival, color=:steelblue, linewidth=2)
    ylims!(ax, 0, 1)

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

# ============================================================================
# UNCERTAINTY DYNAMICS PLOTS
# ============================================================================

"""
Plot Knightian uncertainty dimensions over time.
"""
function plot_uncertainty_dynamics(
    history_df::DataFrame;
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    uncertainty_cols = [
        (:actor_ignorance, "Actor Ignorance", "#E41A1C"),
        (:practical_indeterminism, "Practical Indeterminism", "#377EB8"),
        (:agentic_novelty, "Agentic Novelty", "#4DAF4A"),
        (:competitive_recursion, "Competitive Recursion", "#984EA3")
    ]

    # Filter to available columns
    available = filter(x -> hasproperty(history_df, x[1]), uncertainty_cols)

    if isempty(available) || !hasproperty(history_df, :round)
        @warn "Missing uncertainty columns"
        return nothing
    end

    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1],
        xlabel="Round",
        ylabel="Uncertainty Level",
        title="Knightian Uncertainty Dynamics"
    )

    rounds = history_df.round

    for (col, label, color) in available
        values = history_df[!, col]
        lines!(ax, rounds, values, label=label, color=color, linewidth=2)
    end

    axislegend(ax, position=:rt)
    ylims!(ax, 0, 1)

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

# ============================================================================
# EFFECT SIZE PLOTS
# ============================================================================

"""
Plot effect sizes (Cohen's d) for AI tiers vs baseline.
"""
function plot_effect_sizes(
    effects::Dict{String,Any};
    metric_name::String="Metric",
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    if isempty(effects)
        return nothing
    end

    tiers = String[]
    d_values = Float64[]
    ci_lows = Float64[]
    ci_highs = Float64[]

    for tier in ["basic", "advanced", "premium"]
        if haskey(effects, tier)
            result = effects[tier]
            push!(tiers, titlecase(tier))
            push!(d_values, result.value)
            push!(ci_lows, result.ci_lower)
            push!(ci_highs, result.ci_upper)
        end
    end

    if isempty(tiers)
        return nothing
    end

    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1],
        xlabel="AI Tier",
        ylabel="Cohen's d (vs None)",
        title="Effect Sizes: $metric_name"
    )

    positions = 1:length(tiers)
    colors = [tier_color(lowercase(t)) for t in tiers]

    # Bars
    barplot!(ax, positions, d_values, color=colors)

    # Error bars
    rangebars!(ax, positions, ci_lows, ci_highs, color=:black, whiskerwidth=10)

    # Reference line at 0
    hlines!(ax, [0], color=:gray, linestyle=:dash)

    # Effect size thresholds
    hlines!(ax, [0.2, -0.2], color=:lightgray, linestyle=:dot, label="Small")
    hlines!(ax, [0.5, -0.5], color=:lightgray, linestyle=:dash, label="Medium")
    hlines!(ax, [0.8, -0.8], color=:lightgray, linestyle=:solid, label="Large")

    ax.xticks = (positions, tiers)

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

# ============================================================================
# SURVIVAL CURVES
# ============================================================================

"""
Plot Kaplan-Meier style survival curves by AI tier.
"""
function plot_survival_curves(
    df::DataFrame;
    tier_col::Symbol=:primary_ai_level,
    failure_round_col::Symbol=:failure_step,
    max_rounds::Int=250,
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    if !hasproperty(df, tier_col)
        return nothing
    end

    fig = Figure(size=(700, 500))
    ax = Axis(fig[1, 1],
        xlabel="Round",
        ylabel="Survival Probability",
        title="Survival Curves by AI Tier"
    )

    rounds = 0:max_rounds

    for tier in AI_TIER_ORDER
        tier_df = filter(row -> lowercase(string(row[tier_col])) == tier, df)

        if nrow(tier_df) < 5
            continue
        end

        n_total = nrow(tier_df)

        # Compute survival at each round
        survival_probs = Float64[]

        for r in rounds
            if hasproperty(df, failure_round_col)
                # Count agents still alive at round r
                failures = tier_df[!, failure_round_col]
                n_alive = count(f -> ismissing(f) || f > r, failures)
            else
                # Use final survival status scaled by round
                # (simplified - assumes uniform failure)
                survived = hasproperty(tier_df, :survived) ? tier_df.survived : ones(nrow(tier_df))
                survival_rate = mean(survived)
                n_alive = round(Int, n_total * (1 - r/max_rounds * (1 - survival_rate)))
            end
            push!(survival_probs, n_alive / n_total)
        end

        lines!(ax, collect(rounds), survival_probs,
               label=titlecase(tier), color=tier_color(tier), linewidth=2)
    end

    axislegend(ax, position=:rb)
    ylims!(ax, 0, 1)

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

# ============================================================================
# SUMMARY DASHBOARD
# ============================================================================

"""
Create a summary dashboard with multiple plots.
"""
function create_summary_dashboard(
    history_df::DataFrame,
    agent_df::DataFrame;
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    fig = Figure(size=(1200, 800))

    # Plot 1: Survival over time
    ax1 = Axis(fig[1, 1], title="Survival Over Time", xlabel="Round", ylabel="Rate")
    if hasproperty(history_df, :survival_rate) && hasproperty(history_df, :round)
        lines!(ax1, history_df.round, history_df.survival_rate, color=:steelblue, linewidth=2)
        ylims!(ax1, 0, 1)
    end

    # Plot 2: Capital over time
    ax2 = Axis(fig[1, 2], title="Mean Capital Over Time", xlabel="Round", ylabel="Capital")
    if hasproperty(history_df, :mean_capital) && hasproperty(history_df, :round)
        lines!(ax2, history_df.round, history_df.mean_capital, color=:forestgreen, linewidth=2)
    end

    # Plot 3: Survival by tier (if agent_df has tier info)
    ax3 = Axis(fig[2, 1], title="Survival by AI Tier", xlabel="Tier", ylabel="Rate (%)")
    if hasproperty(agent_df, :primary_ai_level) && hasproperty(agent_df, :survived)
        tiers = String[]
        rates = Float64[]
        for tier in AI_TIER_ORDER
            tier_df = filter(row -> lowercase(string(row.primary_ai_level)) == tier, agent_df)
            if nrow(tier_df) > 0
                push!(tiers, titlecase(tier))
                push!(rates, mean(tier_df.survived) * 100)
            end
        end
        if !isempty(tiers)
            colors = [tier_color(lowercase(t)) for t in tiers]
            barplot!(ax3, 1:length(tiers), rates, color=colors)
            ax3.xticks = (1:length(tiers), tiers)
            ylims!(ax3, 0, 100)
        end
    end

    # Plot 4: Investment success over time
    ax4 = Axis(fig[2, 2], title="Investment Success Rate", xlabel="Round", ylabel="Rate")
    if hasproperty(history_df, :success_rate) && hasproperty(history_df, :round)
        lines!(ax4, history_df.round, history_df.success_rate, color=:purple, linewidth=2)
        ylims!(ax4, 0, 1)
    end

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

# ============================================================================
# BATCH FIGURE GENERATION
# ============================================================================

"""
Save all standard figures for a simulation result.
"""
function save_all_figures(
    history_df::DataFrame,
    agent_df::DataFrame,
    output_dir::String
)
    mkpath(output_dir)

    figures_saved = String[]

    # Survival over time
    try
        fig = plot_survival_over_time(history_df,
            output_path=joinpath(output_dir, "survival_over_time.png"))
        if !isnothing(fig)
            push!(figures_saved, "survival_over_time.png")
        end
    catch e
        @warn "Failed to create survival_over_time plot" exception=e
    end

    # AI adoption over time
    try
        fig = plot_ai_adoption_over_time(history_df,
            output_path=joinpath(output_dir, "ai_adoption.png"))
        if !isnothing(fig)
            push!(figures_saved, "ai_adoption.png")
        end
    catch e
        @warn "Failed to create ai_adoption plot" exception=e
    end

    # Uncertainty dynamics
    try
        fig = plot_uncertainty_dynamics(history_df,
            output_path=joinpath(output_dir, "uncertainty_dynamics.png"))
        if !isnothing(fig)
            push!(figures_saved, "uncertainty_dynamics.png")
        end
    catch e
        @warn "Failed to create uncertainty_dynamics plot" exception=e
    end

    # Summary dashboard
    try
        fig = create_summary_dashboard(history_df, agent_df,
            output_path=joinpath(output_dir, "summary_dashboard.png"))
        if !isnothing(fig)
            push!(figures_saved, "summary_dashboard.png")
        end
    catch e
        @warn "Failed to create summary_dashboard" exception=e
    end

    println("Saved $(length(figures_saved)) figures to $output_dir")
    return figures_saved
end

# ============================================================================
# PERFORMANCE DASHBOARD
# ============================================================================

"""
Plot wealth/capital distribution by AI tier using violin plots.
"""
function plot_wealth_distribution(
    agent_df::DataFrame;
    tier_col::Symbol=:primary_ai_level,
    capital_col::Symbol=:final_capital,
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    if !hasproperty(agent_df, tier_col) || !hasproperty(agent_df, capital_col)
        return nothing
    end

    fig = Figure(size=(700, 500))
    ax = Axis(fig[1, 1],
        xlabel="AI Tier",
        ylabel="Final Capital",
        title="Final Capital Distribution by AI Tier"
    )

    positions = Float64[]
    data_all = Float64[]
    pos_all = Float64[]
    labels = String[]

    for (i, tier) in enumerate(AI_TIER_ORDER)
        tier_data = filter(row -> lowercase(string(row[tier_col])) == tier, agent_df)
        if nrow(tier_data) > 0
            capitals = collect(skipmissing(tier_data[!, capital_col]))
            if !isempty(capitals)
                push!(positions, Float64(i))
                push!(labels, canonical_to_display(tier))
                append!(data_all, capitals)
                append!(pos_all, fill(Float64(i), length(capitals)))
            end
        end
    end

    if isempty(data_all)
        return nothing
    end

    # Use violin plot
    violin!(ax, pos_all, data_all, color=(:steelblue, 0.6))
    ax.xticks = (positions, labels)

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

"""
Plot success factors correlation (traits vs capital growth).
"""
function plot_success_factors(
    agent_df::DataFrame;
    correlation_data::Dict{String,Float64}=Dict{String,Float64}(),
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    # If no correlation data provided, compute from agent_df
    if isempty(correlation_data) && hasproperty(agent_df, :capital_growth)
        trait_cols = [:risk_tolerance, :adaptability, :innovation_propensity,
                      :ai_trust, :learning_rate, :exploration_rate]
        for col in trait_cols
            if hasproperty(agent_df, col)
                valid = dropmissing(agent_df[!, [col, :capital_growth]])
                if nrow(valid) > 10
                    correlation_data[string(col)] = cor(valid[!, col], valid.capital_growth)
                end
            end
        end
    end

    if isempty(correlation_data)
        return nothing
    end

    # Remove capital_growth from correlation data if present
    delete!(correlation_data, "capital_growth")

    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1],
        xlabel="Spearman Correlation",
        ylabel="Trait",
        title="Correlation of Traits with Capital Growth"
    )

    # Sort by value
    sorted_items = sort(collect(correlation_data), by=x->x[2])
    traits = [x[1] for x in sorted_items]
    values = [x[2] for x in sorted_items]

    colors = [v > 0 ? tier_color("premium") : tier_color("none") for v in values]

    barplot!(ax, 1:length(traits), values, direction=:x, color=colors)
    ax.yticks = (1:length(traits), traits)

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

"""
Plot capital growth vs number of innovations scatter.
"""
function plot_performance_vs_innovations(
    agent_df::DataFrame;
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    if !hasproperty(agent_df, :capital_growth) || !hasproperty(agent_df, :innovations)
        return nothing
    end

    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1],
        xlabel="Total Innovations",
        ylabel="Capital Growth",
        title="Capital Growth vs. Number of Innovations"
    )

    tier_col = hasproperty(agent_df, :primary_ai_level) ? :primary_ai_level : nothing

    if !isnothing(tier_col)
        for tier in AI_TIER_ORDER
            tier_df = filter(row -> lowercase(string(row[tier_col])) == tier, agent_df)
            if nrow(tier_df) > 0
                scatter!(ax, tier_df.innovations, tier_df.capital_growth,
                        label=canonical_to_display(tier),
                        color=(tier_color(tier), 0.6),
                        markersize=8)
            end
        end
        axislegend(ax, position=:rt)
    else
        scatter!(ax, agent_df.innovations, agent_df.capital_growth,
                color=(:steelblue, 0.6), markersize=8)
    end

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

"""
Create a 2x2 performance dashboard.
"""
function create_performance_dashboard(
    agent_df::DataFrame;
    survival_data::Dict{String,Float64}=Dict{String,Float64}(),
    correlation_data::Dict{String,Float64}=Dict{String,Float64}(),
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    fig = Figure(size=(1400, 1000))
    Label(fig[0, 1:2], "Performance and Outcomes Analysis", fontsize=24, font=:bold)

    # Plot 1: Wealth distribution (violin)
    ax1 = Axis(fig[1, 1], title="Final Capital Distribution by AI Tier",
               xlabel="AI Tier", ylabel="Final Capital")

    if hasproperty(agent_df, :primary_ai_level) && hasproperty(agent_df, :final_capital)
        positions = Float64[]
        data_all = Float64[]
        pos_all = Float64[]
        labels = String[]

        for (i, tier) in enumerate(AI_TIER_ORDER)
            tier_data = filter(row -> lowercase(string(row.primary_ai_level)) == tier, agent_df)
            if nrow(tier_data) > 0
                capitals = collect(skipmissing(tier_data.final_capital))
                if !isempty(capitals)
                    push!(positions, Float64(i))
                    push!(labels, canonical_to_display(tier))
                    append!(data_all, capitals)
                    append!(pos_all, fill(Float64(i), length(capitals)))
                end
            end
        end
        if !isempty(data_all)
            violin!(ax1, pos_all, data_all, color=(:steelblue, 0.6))
            ax1.xticks = (positions, labels)
        end
    end

    # Plot 2: Survival rates by AI tier
    ax2 = Axis(fig[1, 2], title="Survival Rate by AI Tier",
               xlabel="AI Tier", ylabel="Survival Rate")

    # Compute survival if not provided
    if isempty(survival_data) && hasproperty(agent_df, :survived) && hasproperty(agent_df, :primary_ai_level)
        for tier in AI_TIER_ORDER
            tier_df = filter(row -> lowercase(string(row.primary_ai_level)) == tier, agent_df)
            if nrow(tier_df) > 0
                survival_data[tier] = mean(tier_df.survived)
            end
        end
    end

    if !isempty(survival_data)
        tiers = String[]
        rates = Float64[]
        colors_list = String[]
        for tier in AI_TIER_ORDER
            if haskey(survival_data, tier)
                push!(tiers, canonical_to_display(tier))
                push!(rates, survival_data[tier])
                push!(colors_list, tier_color(tier))
            end
        end
        if !isempty(tiers)
            barplot!(ax2, 1:length(tiers), rates, color=colors_list)
            ax2.xticks = (1:length(tiers), tiers)
            ylims!(ax2, 0, 1)
        end
    end

    # Plot 3: Success factors correlation
    ax3 = Axis(fig[2, 1], title="Correlation of Traits with Capital Growth",
               xlabel="Spearman Correlation", ylabel="Trait")

    if isempty(correlation_data) && hasproperty(agent_df, :capital_growth)
        trait_cols = [:risk_tolerance, :adaptability, :innovation_propensity, :ai_trust]
        for col in trait_cols
            if hasproperty(agent_df, col)
                valid = dropmissing(agent_df[!, [col, :capital_growth]])
                if nrow(valid) > 10
                    correlation_data[string(col)] = cor(valid[!, col], valid.capital_growth)
                end
            end
        end
    end

    if !isempty(correlation_data)
        sorted_items = sort(collect(correlation_data), by=x->x[2])
        traits = [x[1] for x in sorted_items]
        values = [x[2] for x in sorted_items]
        colors_corr = [v > 0 ? tier_color("premium") : tier_color("none") for v in values]
        barplot!(ax3, 1:length(traits), values, direction=:x, color=colors_corr)
        ax3.yticks = (1:length(traits), traits)
    end

    # Plot 4: Performance vs innovations
    ax4 = Axis(fig[2, 2], title="Capital Growth vs. Innovations",
               xlabel="Total Innovations", ylabel="Capital Growth")

    if hasproperty(agent_df, :capital_growth) && hasproperty(agent_df, :innovations)
        tier_col = hasproperty(agent_df, :primary_ai_level) ? :primary_ai_level : nothing
        if !isnothing(tier_col)
            for tier in AI_TIER_ORDER
                tier_df = filter(row -> lowercase(string(row[tier_col])) == tier, agent_df)
                if nrow(tier_df) > 0
                    scatter!(ax4, tier_df.innovations, tier_df.capital_growth,
                            label=canonical_to_display(tier),
                            color=(tier_color(tier), 0.6),
                            markersize=6)
                end
            end
            axislegend(ax4, position=:rt)
        else
            scatter!(ax4, agent_df.innovations, agent_df.capital_growth,
                    color=(:steelblue, 0.6), markersize=6)
        end
    end

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

# ============================================================================
# TEMPORAL DYNAMICS DASHBOARD
# ============================================================================

"""
Plot action distribution over time (stacked area).
"""
function plot_action_distribution_over_time(
    decision_df::DataFrame;
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    if !hasproperty(decision_df, :action) || !hasproperty(decision_df, :round)
        return nothing
    end

    fig = Figure(size=(700, 500))
    ax = Axis(fig[1, 1],
        xlabel="Simulation Round",
        ylabel="Proportion of Actions",
        title="Action Distribution Over Time"
    )

    # Group by round and action
    rounds = sort(unique(decision_df.round))
    actions = ["invest", "innovate", "explore", "maintain"]
    action_colors = ["#1abc9c", "#f39c12", "#8e44ad", "#34495e"]

    action_shares = Dict{String,Vector{Float64}}()
    for action in actions
        action_shares[action] = Float64[]
    end

    for r in rounds
        round_data = filter(row -> row.round == r, decision_df)
        total = nrow(round_data)
        for action in actions
            count = count(row -> row.action == action, eachrow(round_data))
            push!(action_shares[action], total > 0 ? count / total : 0.0)
        end
    end

    # Create stacked area plot
    cumulative = zeros(length(rounds))
    for (i, (action, color)) in enumerate(zip(actions, action_colors))
        shares = action_shares[action]
        band!(ax, rounds, cumulative, cumulative .+ shares,
              color=(color, 0.7), label=titlecase(action))
        cumulative .+= shares
    end

    axislegend(ax, position=:rt)
    ylims!(ax, 0, 1)

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

"""
Create temporal dynamics dashboard with 4 plots.
"""
function create_temporal_dynamics_dashboard(
    history_df::DataFrame,
    decision_df::DataFrame=DataFrame();
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    fig = Figure(size=(1400, 1000))
    Label(fig[0, 1:2], "Temporal Dynamics of AI-Augmented Entrepreneurship", fontsize=24, font=:bold)

    # Plot 1: Mean capital over time
    ax1 = Axis(fig[1, 1], title="Average Capital Over Time",
               xlabel="Round", ylabel="Mean Capital")

    if hasproperty(history_df, :mean_capital) && hasproperty(history_df, :round)
        lines!(ax1, history_df.round, history_df.mean_capital, color=:steelblue, linewidth=2)
    end

    # Plot 2: AI adoption share over time
    ax2 = Axis(fig[1, 2], title="AI Adoption Share Over Time",
               xlabel="Round", ylabel="Share of Active Agents")

    share_cols = [:ai_share_none, :ai_share_basic, :ai_share_advanced, :ai_share_premium]
    has_shares = any(hasproperty(history_df, col) for col in share_cols)

    if has_shares && hasproperty(history_df, :round)
        for (tier, col) in zip(AI_TIER_ORDER, share_cols)
            if hasproperty(history_df, col)
                lines!(ax2, history_df.round, history_df[!, col],
                      label=canonical_to_display(tier),
                      color=tier_color(tier), linewidth=2)
            end
        end
        axislegend(ax2, position=:rt)
        ylims!(ax2, 0, 1)
    end

    # Plot 3: Action mix over time
    ax3 = Axis(fig[2, 1], title="Action Mix Over Time",
               xlabel="Round", ylabel="Share of Actions")

    action_cols = [:action_share_invest, :action_share_innovate,
                   :action_share_explore, :action_share_maintain]
    action_colors_map = Dict(
        :action_share_invest => "#1abc9c",
        :action_share_innovate => "#f39c12",
        :action_share_explore => "#8e44ad",
        :action_share_maintain => "#34495e"
    )

    has_actions = any(hasproperty(history_df, col) for col in action_cols)
    if has_actions && hasproperty(history_df, :round)
        for col in action_cols
            if hasproperty(history_df, col)
                action_name = replace(string(col), "action_share_" => "")
                lines!(ax3, history_df.round, history_df[!, col],
                      label=titlecase(action_name),
                      color=action_colors_map[col], linewidth=2)
            end
        end
        axislegend(ax3, position=:rt)
        ylims!(ax3, 0, 1)
    end

    # Plot 4: Innovation success rate
    ax4 = Axis(fig[2, 2], title="Innovation Success Rate Over Time",
               xlabel="Round", ylabel="Success Rate")

    if hasproperty(history_df, :innovation_success_rate) && hasproperty(history_df, :round)
        lines!(ax4, history_df.round, history_df.innovation_success_rate,
               color=:forestgreen, linewidth=2)
        ylims!(ax4, 0, 1)
    elseif hasproperty(history_df, :success_rate) && hasproperty(history_df, :round)
        lines!(ax4, history_df.round, history_df.success_rate,
               color=:forestgreen, linewidth=2)
        ylims!(ax4, 0, 1)
    end

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

# ============================================================================
# INNOVATION DASHBOARD
# ============================================================================

"""
Create innovation dynamics dashboard.
"""
function create_innovation_dashboard(
    history_df::DataFrame,
    innovation_df::DataFrame=DataFrame();
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    fig = Figure(size=(1400, 1000))
    Label(fig[0, 1:2], "Innovation Dynamics and Knowledge Diversity", fontsize=24, font=:bold)

    # Plot 1: Innovation impact distribution
    ax1 = Axis(fig[1, 1], title="Distribution of Innovation Impact",
               xlabel="Market Impact", ylabel="Count")

    if !isempty(innovation_df) && hasproperty(innovation_df, :market_impact)
        impacts = collect(skipmissing(innovation_df.market_impact))
        if !isempty(impacts)
            hist!(ax1, impacts, bins=30, color=(:green, 0.6))
        end
    elseif !isempty(innovation_df) && hasproperty(innovation_df, :quality) && hasproperty(innovation_df, :novelty)
        impacts = innovation_df.quality .* innovation_df.novelty
        hist!(ax1, impacts, bins=30, color=(:green, 0.6))
        ax1.xlabel = "Impact Proxy (quality × novelty)"
    end

    # Plot 2: Innovation type frequency
    ax2 = Axis(fig[1, 2], title="Innovation Type Frequency",
               xlabel="Innovation Type", ylabel="Count")

    if !isempty(innovation_df) && hasproperty(innovation_df, :type)
        type_counts = combine(groupby(innovation_df, :type), nrow => :count)
        if nrow(type_counts) > 0
            barplot!(ax2, 1:nrow(type_counts), type_counts.count, color=:steelblue)
            ax2.xticks = (1:nrow(type_counts), string.(type_counts.type))
        end
    end

    # Plot 3: Portfolio diversity over time
    ax3 = Axis(fig[2, 1], title="Average Portfolio Diversity Over Time",
               xlabel="Round", ylabel="Diversification Score")

    if hasproperty(history_df, :mean_portfolio_diversity) && hasproperty(history_df, :round)
        lines!(ax3, history_df.round, history_df.mean_portfolio_diversity,
               color=:purple, linewidth=2)
    end

    # Plot 4: AI trust over time
    ax4 = Axis(fig[2, 2], title="Average AI Trust Over Time",
               xlabel="Round", ylabel="Mean Trust")

    if hasproperty(history_df, :mean_ai_trust) && hasproperty(history_df, :round)
        lines!(ax4, history_df.round, history_df.mean_ai_trust,
               color=:crimson, linewidth=2)
    end

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

# ============================================================================
# CONCENTRATION DASHBOARD
# ============================================================================

"""
Create concentration and fragility metrics dashboard.
"""
function create_concentration_dashboard(
    history_df::DataFrame,
    market_df::DataFrame=DataFrame(),
    decision_df::DataFrame=DataFrame();
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    fig = Figure(size=(1400, 1000))
    Label(fig[0, 1:2], "Concentration and Fragility Metrics", fontsize=24, font=:bold)

    # Plot 1: HHI over time
    ax1 = Axis(fig[1, 1], title="Investment Concentration (HHI) Over Time",
               xlabel="Round", ylabel="HHI")

    if hasproperty(history_df, :overall_hhi) && hasproperty(history_df, :round)
        lines!(ax1, history_df.round, history_df.overall_hhi,
               color=:darkblue, linewidth=2)
    end

    # Plot 2: Top sector share over time
    ax2 = Axis(fig[1, 2], title="Top Sector Market Share Over Time",
               xlabel="Round", ylabel="Share")

    if hasproperty(history_df, :top_sector_share) && hasproperty(history_df, :round)
        lines!(ax2, history_df.round, history_df.top_sector_share,
               color=:orange, linewidth=2)
        ylims!(ax2, 0, 1)
    end

    # Plot 3: Market volatility
    ax3 = Axis(fig[2, 1], title="Market Volatility Over Time",
               xlabel="Round", ylabel="Volatility")

    if !isempty(market_df) && hasproperty(market_df, :volatility) && hasproperty(market_df, :round)
        market_agg = combine(groupby(market_df, :round), :volatility => mean => :volatility)
        sort!(market_agg, :round)
        lines!(ax3, market_agg.round, market_agg.volatility,
               color=:steelblue, linewidth=2)
    end

    # Plot 4: Premium AI opportunity diversity
    ax4 = Axis(fig[2, 2], title="Premium AI Opportunity Diversity",
               xlabel="Round", ylabel="Unique Opps / Total")

    if !isempty(decision_df) && hasproperty(decision_df, :ai_level_used) &&
       hasproperty(decision_df, :opportunity_id) && hasproperty(decision_df, :round)
        premium_df = filter(row -> row.ai_level_used == "premium", decision_df)
        if nrow(premium_df) > 0
            diversity_data = combine(groupby(premium_df, :round)) do gdf
                unique_count = length(unique(skipmissing(gdf.opportunity_id)))
                total_count = nrow(gdf)
                (diversity = total_count > 0 ? unique_count / total_count : 0.0,)
            end
            sort!(diversity_data, :round)
            lines!(ax4, diversity_data.round, diversity_data.diversity,
                   color=tier_color("premium"), linewidth=2)
        end
    end

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

# ============================================================================
# AI VS UNCERTAINTY DASHBOARD
# ============================================================================

"""
Create AI vs Knightian uncertainty dashboard.
"""
function create_ai_uncertainty_dashboard(
    uncertainty_df::DataFrame;
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    fig = Figure(size=(1400, 1000))
    Label(fig[0, 1:2], "AI Augmentation vs. Perceived Knightian Uncertainty", fontsize=24, font=:bold)

    uncertainty_dims = [
        (:actor_ignorance_level, "Actor Ignorance"),
        (:practical_indeterminism_level, "Practical Indeterminism"),
        (:agentic_novelty_level, "Agentic Novelty"),
        (:competitive_recursion_level, "Competitive Recursion")
    ]

    axes = [Axis(fig[r, c]) for r in 1:2, c in 1:2]

    for (idx, (col, title)) in enumerate(uncertainty_dims)
        ax = axes[idx]
        ax.title = "AI vs. $title"
        ax.xlabel = "AI Level"
        ax.ylabel = "Average Perceived Level"

        if !hasproperty(uncertainty_df, col) || !hasproperty(uncertainty_df, :ai_level_used)
            continue
        end

        # Aggregate by AI level
        ai_means = combine(groupby(uncertainty_df, :ai_level_used), col => mean => :mean_val)

        tiers = String[]
        means = Float64[]
        colors_list = String[]

        for tier in AI_TIER_ORDER
            tier_row = filter(row -> row.ai_level_used == tier, ai_means)
            if nrow(tier_row) > 0
                push!(tiers, canonical_to_display(tier))
                push!(means, tier_row.mean_val[1])
                push!(colors_list, tier_color(tier))
            end
        end

        if !isempty(tiers)
            barplot!(ax, 1:length(tiers), means, color=colors_list)
            ax.xticks = (1:length(tiers), tiers)
        end
    end

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

# ============================================================================
# PERCEPTION STORYBOARD
# ============================================================================

"""
Create perception storyboard showing uncertainty perception by AI tier over time.
"""
function create_perception_storyboard(
    uncertainty_df::DataFrame;
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    if !hasproperty(uncertainty_df, :round) || !hasproperty(uncertainty_df, :ai_level_used)
        return nothing
    end

    fig = Figure(size=(1400, 1000))
    Label(fig[0, 1:2], "Perception Storyboard: Uncertainty by AI Tier", fontsize=24, font=:bold)

    metrics = [
        (:actor_ignorance_level, "Actor Ignorance"),
        (:practical_indeterminism_level, "Practical Indeterminism"),
        (:agentic_novelty_level, "Agentic Novelty"),
        (:competitive_recursion_level, "Competitive Recursion")
    ]

    axes = [Axis(fig[r, c]) for r in 1:2, c in 1:2]

    for (idx, (col, title)) in enumerate(metrics)
        ax = axes[idx]
        ax.title = title
        ax.xlabel = "Simulation Round"
        ax.ylabel = "Perceived Level"

        if !hasproperty(uncertainty_df, col)
            continue
        end

        # Aggregate by round and AI level
        agg_data = combine(groupby(uncertainty_df, [:round, :ai_level_used]),
                          col => mean => :mean_val,
                          col => std => :std_val)

        for tier in AI_TIER_ORDER
            tier_data = filter(row -> row.ai_level_used == tier, agg_data)
            if nrow(tier_data) > 0
                sort!(tier_data, :round)
                lines!(ax, tier_data.round, tier_data.mean_val,
                      label=canonical_to_display(tier),
                      color=tier_color(tier), linewidth=2)

                # Add confidence band if std available
                if hasproperty(tier_data, :std_val)
                    std_vals = coalesce.(tier_data.std_val, 0.0)
                    band!(ax, tier_data.round,
                          tier_data.mean_val .- std_vals,
                          tier_data.mean_val .+ std_vals,
                          color=(tier_color(tier), 0.15))
                end
            end
        end
    end

    # Add shared legend
    Legend(fig[3, 1:2], axes[1], orientation=:horizontal, tellwidth=false, tellheight=true)

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

# ============================================================================
# DECISION STORYBOARD
# ============================================================================

"""
Create decision storyboard showing decision patterns.
"""
function create_decision_storyboard(
    decision_df::DataFrame,
    history_df::DataFrame=DataFrame();
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    fig = Figure(size=(1400, 1000))
    Label(fig[0, 1:2], "Decision Storyboard", fontsize=24, font=:bold)

    # Plot 1: Action distribution over time
    ax1 = Axis(fig[1, 1], title="Action Distribution Over Time",
               xlabel="Simulation Round", ylabel="Proportion of Actions")

    if hasproperty(decision_df, :action) && hasproperty(decision_df, :round)
        rounds = sort(unique(decision_df.round))
        actions = ["invest", "innovate", "explore", "maintain"]
        action_colors = ["#1abc9c", "#f39c12", "#8e44ad", "#34495e"]

        for (action, color) in zip(actions, action_colors)
            shares = Float64[]
            for r in rounds
                round_data = filter(row -> row.round == r, decision_df)
                action_count = count(row -> row.action == action, eachrow(round_data))
                total = nrow(round_data)
                push!(shares, total > 0 ? action_count / total : 0.0)
            end
            lines!(ax1, rounds, shares, label=titlecase(action), color=color, linewidth=2)
        end
        axislegend(ax1, position=:rt)
        ylims!(ax1, 0, 1)
    end

    # Plot 2: AI adoption over time
    ax2 = Axis(fig[1, 2], title="AI Adoption Over Time",
               xlabel="Simulation Round", ylabel="Share of Active Decisions")

    if hasproperty(decision_df, :ai_level_used) && hasproperty(decision_df, :round)
        rounds = sort(unique(decision_df.round))
        for tier in AI_TIER_ORDER
            shares = Float64[]
            for r in rounds
                round_data = filter(row -> row.round == r, decision_df)
                tier_count = count(row -> row.ai_level_used == tier, eachrow(round_data))
                total = nrow(round_data)
                push!(shares, total > 0 ? tier_count / total : 0.0)
            end
            lines!(ax2, rounds, shares, label=canonical_to_display(tier),
                  color=tier_color(tier), linewidth=2)
        end
        axislegend(ax2, position=:rt)
        ylims!(ax2, 0, 1)
    end

    # Plot 3: Decision mix by AI tier (stacked bar)
    ax3 = Axis(fig[2, 1], title="Decision Mix by AI Level",
               xlabel="AI Level", ylabel="Decision Share")

    if hasproperty(decision_df, :action) && hasproperty(decision_df, :ai_level_used)
        actions = ["invest", "innovate", "explore", "maintain"]
        action_colors = ["#1abc9c", "#f39c12", "#8e44ad", "#34495e"]

        positions = Float64[]
        labels = String[]

        for (i, tier) in enumerate(AI_TIER_ORDER)
            tier_data = filter(row -> row.ai_level_used == tier, decision_df)
            if nrow(tier_data) > 0
                push!(positions, Float64(i))
                push!(labels, canonical_to_display(tier))

                cumulative = 0.0
                total = nrow(tier_data)
                for (action, color) in zip(actions, action_colors)
                    action_count = count(row -> row.action == action, eachrow(tier_data))
                    share = action_count / total
                    barplot!(ax3, [Float64(i)], [share], offset=cumulative,
                            color=color, width=0.6)
                    cumulative += share
                end
            end
        end
        ax3.xticks = (positions, labels)
        ylims!(ax3, 0, 1)
    end

    # Plot 4: Knowledge accumulation
    ax4 = Axis(fig[2, 2], title="Knowledge Accumulation",
               xlabel="Simulation Round", ylabel="Mean Knowledge Pieces")

    if hasproperty(history_df, :mean_knowledge_count) && hasproperty(history_df, :round)
        lines!(ax4, history_df.round, history_df.mean_knowledge_count,
               color=:green, linewidth=2)
    end

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

# ============================================================================
# MARKET STORYBOARD
# ============================================================================

"""
Create market dynamics storyboard.
"""
function create_market_storyboard(
    uncertainty_df::DataFrame,
    market_df::DataFrame=DataFrame(),
    decision_df::DataFrame=DataFrame();
    output_path::Union{String,Nothing}=nothing
)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available"
        return nothing
    end

    fig = Figure(size=(1400, 1000))
    Label(fig[0, 1:2], "Market & Uncertainty Storyboard", fontsize=24, font=:bold)

    # Plot 1: Uncertainty evolution
    ax1 = Axis(fig[1, 1], title="Evolution of Knightian Uncertainty",
               xlabel="Simulation Round", ylabel="Average Level")

    uncertainty_cols = [
        (:actor_ignorance_level, "Actor Ignorance"),
        (:practical_indeterminism_level, "Practical Indeterminism"),
        (:agentic_novelty_level, "Agentic Novelty"),
        (:competitive_recursion_level, "Competitive Recursion")
    ]

    if hasproperty(uncertainty_df, :round)
        agg_df = combine(groupby(uncertainty_df, :round)) do gdf
            result = (round=first(gdf.round),)
            for (col, _) in uncertainty_cols
                if hasproperty(gdf, col)
                    result = merge(result, NamedTuple{(col,)}((mean(skipmissing(gdf[!, col])),)))
                end
            end
            result
        end
        sort!(agg_df, :round)

        for (col, label) in uncertainty_cols
            if hasproperty(agg_df, col)
                dim_name = replace(string(col), "_level" => "")
                lines!(ax1, agg_df.round, agg_df[!, col],
                      label=label, color=uncertainty_color(dim_name), linewidth=2)
            end
        end
        axislegend(ax1, position=:rt)
    end

    # Plot 2: Market regime frequency (pie chart simulation with bars)
    ax2 = Axis(fig[1, 2], title="Market Regime Frequency",
               xlabel="Regime", ylabel="Frequency")

    if !isempty(market_df) && hasproperty(market_df, :regime)
        regime_counts = combine(groupby(market_df, :regime), nrow => :count)
        total = sum(regime_counts.count)
        regime_counts.freq = regime_counts.count ./ total

        regimes = string.(regime_counts.regime)
        freqs = regime_counts.freq

        barplot!(ax2, 1:length(regimes), freqs, color=:steelblue)
        ax2.xticks = (1:length(regimes), titlecase.(regimes))
        ylims!(ax2, 0, 1)
    end

    # Plot 3: Herding evolution
    ax3 = Axis(fig[2, 1], title="Evolution of Herding Behavior",
               xlabel="Simulation Round", ylabel="Average Herding Intensity")

    herding_cols = [:competitive_recursion_herding_intensity, :herding_intensity,
                    :competitive_recursion_level]
    herding_col = nothing
    for col in herding_cols
        if hasproperty(uncertainty_df, col)
            herding_col = col
            break
        end
    end

    if !isnothing(herding_col) && hasproperty(uncertainty_df, :round)
        herding_agg = combine(groupby(uncertainty_df, :round),
                             herding_col => mean => :herding)
        sort!(herding_agg, :round)
        lines!(ax3, herding_agg.round, herding_agg.herding,
               color=uncertainty_color("competitive_recursion"), linewidth=2)
        band!(ax3, herding_agg.round, zeros(nrow(herding_agg)), herding_agg.herding,
              color=(uncertainty_color("competitive_recursion"), 0.3))
    end

    # Plot 4: Action distribution over time
    ax4 = Axis(fig[2, 2], title="Agent Action Distribution Over Time",
               xlabel="Simulation Round", ylabel="Proportion of Actions")

    if !isempty(decision_df) && hasproperty(decision_df, :action) && hasproperty(decision_df, :round)
        rounds = sort(unique(decision_df.round))
        actions = ["invest", "innovate", "explore", "maintain"]
        action_colors = ["#1abc9c", "#f39c12", "#8e44ad", "#34495e"]

        for (action, color) in zip(actions, action_colors)
            shares = Float64[]
            for r in rounds
                round_data = filter(row -> row.round == r, decision_df)
                action_count = count(row -> row.action == action, eachrow(round_data))
                total = nrow(round_data)
                push!(shares, total > 0 ? action_count / total : 0.0)
            end
            lines!(ax4, rounds, shares, label=titlecase(action), color=color, linewidth=2)
        end
        axislegend(ax4, position=:rt)
        ylims!(ax4, 0, 1)
    end

    if !isnothing(output_path)
        save(output_path, fig)
    end

    return fig
end

# ============================================================================
# COMPREHENSIVE VISUALIZATION SUITE
# ============================================================================

"""
Comprehensive visualization suite for GlimpseABM results.

Port of Python ComprehensiveVisualizationSuite class.
"""
mutable struct ComprehensiveVisualizationSuite
    agent_df::DataFrame
    market_df::DataFrame
    uncertainty_df::DataFrame
    innovation_df::DataFrame
    summary_df::DataFrame
    decision_df::DataFrame
    history_df::DataFrame
    figure_output_dir::String
    saved_figures::Vector{String}
end

"""
Create a ComprehensiveVisualizationSuite from DataFrames.
"""
function ComprehensiveVisualizationSuite(;
    agent_df::DataFrame=DataFrame(),
    market_df::DataFrame=DataFrame(),
    uncertainty_df::DataFrame=DataFrame(),
    innovation_df::DataFrame=DataFrame(),
    summary_df::DataFrame=DataFrame(),
    decision_df::DataFrame=DataFrame(),
    history_df::DataFrame=DataFrame(),
    figure_output_dir::String=""
)
    return ComprehensiveVisualizationSuite(
        agent_df, market_df, uncertainty_df, innovation_df,
        summary_df, decision_df, history_df, figure_output_dir,
        String[]
    )
end

"""
Create all visualization dashboards.
"""
function create_all_visualizations(suite::ComprehensiveVisualizationSuite)
    if !HAS_MAKIE[]
        @warn "CairoMakie not available. Skipping visualizations."
        return
    end

    println("\n📊 CREATING COMPREHENSIVE VISUALIZATION SUITE")
    println("=" * 70)

    if isempty(suite.agent_df)
        println("⚠️ Agent data not found. Skipping visualizations.")
        return
    end

    output_dir = suite.figure_output_dir
    if !isempty(output_dir)
        mkpath(output_dir)
    end

    # Create each dashboard
    dashboards = [
        ("perception_storyboard", () -> create_perception_storyboard(
            suite.uncertainty_df,
            output_path=isempty(output_dir) ? nothing : joinpath(output_dir, "perception_storyboard.png")
        )),
        ("decision_storyboard", () -> create_decision_storyboard(
            suite.decision_df, suite.history_df,
            output_path=isempty(output_dir) ? nothing : joinpath(output_dir, "decision_storyboard.png")
        )),
        ("performance_dashboard", () -> create_performance_dashboard(
            suite.agent_df,
            output_path=isempty(output_dir) ? nothing : joinpath(output_dir, "performance_dashboard.png")
        )),
        ("market_storyboard", () -> create_market_storyboard(
            suite.uncertainty_df, suite.market_df, suite.decision_df,
            output_path=isempty(output_dir) ? nothing : joinpath(output_dir, "market_storyboard.png")
        )),
        ("temporal_dynamics", () -> create_temporal_dynamics_dashboard(
            suite.history_df, suite.decision_df,
            output_path=isempty(output_dir) ? nothing : joinpath(output_dir, "temporal_dynamics.png")
        )),
        ("innovation_dashboard", () -> create_innovation_dashboard(
            suite.history_df, suite.innovation_df,
            output_path=isempty(output_dir) ? nothing : joinpath(output_dir, "innovation_dashboard.png")
        )),
        ("concentration_dashboard", () -> create_concentration_dashboard(
            suite.history_df, suite.market_df, suite.decision_df,
            output_path=isempty(output_dir) ? nothing : joinpath(output_dir, "concentration_dashboard.png")
        )),
        ("ai_uncertainty_dashboard", () -> create_ai_uncertainty_dashboard(
            suite.uncertainty_df,
            output_path=isempty(output_dir) ? nothing : joinpath(output_dir, "ai_uncertainty_dashboard.png")
        ))
    ]

    for (name, create_fn) in dashboards
        print("   - Generating $(replace(name, "_" => " "))...")
        try
            fig = create_fn()
            if !isnothing(fig) && !isempty(output_dir)
                push!(suite.saved_figures, "$name.png")
            end
            println(" ✓")
        catch e
            println(" ✗")
            @warn "Failed to create $name" exception=e
        end
    end

    println("\n✅ Visualization suite complete!")
    println("Saved $(length(suite.saved_figures)) figures to $output_dir")
end

"""
Convenience function to create all visualizations from DataFrames.
"""
function create_all_visualizations(;
    agent_df::DataFrame=DataFrame(),
    market_df::DataFrame=DataFrame(),
    uncertainty_df::DataFrame=DataFrame(),
    innovation_df::DataFrame=DataFrame(),
    summary_df::DataFrame=DataFrame(),
    decision_df::DataFrame=DataFrame(),
    history_df::DataFrame=DataFrame(),
    output_dir::String=""
)
    suite = ComprehensiveVisualizationSuite(
        agent_df=agent_df,
        market_df=market_df,
        uncertainty_df=uncertainty_df,
        innovation_df=innovation_df,
        summary_df=summary_df,
        decision_df=decision_df,
        history_df=history_df,
        figure_output_dir=output_dir
    )
    create_all_visualizations(suite)
    return suite
end

end # module Visualization
