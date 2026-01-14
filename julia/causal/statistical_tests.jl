"""
Rigorous Statistical Testing Framework for GlimpseABM.jl

This module provides publication-quality statistical analyses suitable for
top-tier management journals like the Academy of Management Journal (AMJ).
All tests include effect sizes, confidence intervals, assumption checks,
and multiple comparison corrections.

Port of: glimpse_abm/statistical_tests.py

Theoretical Foundation
----------------------
The statistical framework supports empirical investigation of the theoretical
propositions from:

    Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).
    Are the futures computable? Knightian uncertainty & artificial intelligence.
    Academy of Management Review, 50(2), 415-440.
"""

using Statistics
using Random
using DataFrames
using Distributions

# ============================================================================
# FAST STATS MODE FOR LARGE SWEEPS
# ============================================================================

const FAST_STATS_MODE = Ref(false)

"""
Enable/disable fast stats mode (reduced bootstrap iterations).
"""
function set_fast_stats_mode(enabled::Bool=true)
    FAST_STATS_MODE[] = enabled
    if enabled
        println("[Stats] Fast stats mode ENABLED (500 bootstrap iterations)")
    end
end

"""
Get bootstrap iterations based on current mode.
"""
function get_bootstrap_iterations(full_iterations::Int=5000)::Int
    if FAST_STATS_MODE[]
        return min(500, full_iterations)
    end
    return full_iterations
end

# ============================================================================
# STATISTICAL TEST RESULT TYPES
# ============================================================================

"""
Container for a single statistical test result with full reporting.
"""
struct StatisticalTestResult
    test_name::String
    test_statistic::Float64
    p_value::Float64
    p_value_adjusted::Union{Float64,Nothing}
    effect_size::Union{Float64,Nothing}
    effect_size_type::Union{String,Nothing}
    effect_size_ci::Union{Tuple{Float64,Float64},Nothing}
    effect_interpretation::Union{String,Nothing}
    sample_sizes::Dict{String,Int}
    assumptions_met::Dict{String,Bool}
    assumptions_details::Dict{String,String}
    conclusion::String
end

function StatisticalTestResult(;
    test_name::String,
    test_statistic::Float64,
    p_value::Float64,
    p_value_adjusted::Union{Float64,Nothing}=nothing,
    effect_size::Union{Float64,Nothing}=nothing,
    effect_size_type::Union{String,Nothing}=nothing,
    effect_size_ci::Union{Tuple{Float64,Float64},Nothing}=nothing,
    effect_interpretation::Union{String,Nothing}=nothing,
    sample_sizes::Dict{String,Int}=Dict{String,Int}(),
    assumptions_met::Dict{String,Bool}=Dict{String,Bool}(),
    assumptions_details::Dict{String,String}=Dict{String,String}(),
    conclusion::String=""
)
    StatisticalTestResult(
        test_name, test_statistic, p_value, p_value_adjusted,
        effect_size, effect_size_type, effect_size_ci, effect_interpretation,
        sample_sizes, assumptions_met, assumptions_details, conclusion
    )
end

"""
Convert StatisticalTestResult to dictionary for DataFrame creation.
"""
function to_dict(result::StatisticalTestResult)::Dict{String,Any}
    Dict{String,Any}(
        "test_name" => result.test_name,
        "test_statistic" => result.test_statistic,
        "p_value" => result.p_value,
        "p_value_adjusted" => result.p_value_adjusted,
        "effect_size" => result.effect_size,
        "effect_size_type" => result.effect_size_type,
        "effect_size_ci_lower" => isnothing(result.effect_size_ci) ? nothing : result.effect_size_ci[1],
        "effect_size_ci_upper" => isnothing(result.effect_size_ci) ? nothing : result.effect_size_ci[2],
        "effect_interpretation" => result.effect_interpretation,
        "n_total" => isempty(result.sample_sizes) ? nothing : sum(values(result.sample_sizes)),
        "assumptions_met" => isempty(result.assumptions_met) ? nothing : all(values(result.assumptions_met)),
        "conclusion" => result.conclusion
    )
end

"""
Container for mixed-effects model results.
"""
struct MixedEffectsResult
    model_name::String
    dependent_variable::String
    fixed_effects::Dict{String,Dict{String,Float64}}
    random_effects_variance::Dict{String,Float64}
    model_fit::Dict{String,Float64}
    n_observations::Int
    n_groups::Dict{String,Int}
    convergence::Bool
    interpretation::String
end

"""
Result of a causal effect estimation from fixed-tier design.
"""
struct CausalEffectEstimate
    treatment::String
    outcome::String
    ate::Float64
    ate_se::Float64
    ate_ci_lower::Float64
    ate_ci_upper::Float64
    cohens_d::Float64
    cohens_d_ci_lower::Float64
    cohens_d_ci_upper::Float64
    n_treatment::Int
    n_control::Int
    p_value::Float64
    identification::String
    robustness_check::String
end

# ============================================================================
# EFFECT SIZE CALCULATIONS
# ============================================================================

"""
Calculate Cohen's d effect size with bootstrap confidence interval.
"""
function cohens_d_with_ci(
    group1::AbstractVector{<:Real},
    group2::AbstractVector{<:Real};
    n_bootstrap::Int=10000,
    confidence::Float64=0.95
)::Tuple{Float64,String,Tuple{Float64,Float64}}
    n1, n2 = length(group1), length(group2)

    if n1 < 2 || n2 < 2
        return (NaN, "undefined", (NaN, NaN))
    end

    v1, v2 = var(group1), var(group2)
    pooled_std = sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))

    if pooled_std ≈ 0
        return (0.0, "undefined", (0.0, 0.0))
    end

    d = (mean(group1) - mean(group2)) / pooled_std

    # Interpretation
    abs_d = abs(d)
    interpretation = if abs_d < 0.2
        "negligible"
    elseif abs_d < 0.5
        "small"
    elseif abs_d < 0.8
        "medium"
    else
        "large"
    end

    # Bootstrap CI
    rng = MersenneTwister(42)
    d_boots = Float64[]
    n_iter = get_bootstrap_iterations(n_bootstrap)

    for _ in 1:n_iter
        boot1 = group1[rand(rng, 1:n1, n1)]
        boot2 = group2[rand(rng, 1:n2, n2)]

        v1_b, v2_b = var(boot1), var(boot2)
        pooled_std_b = sqrt(((n1 - 1) * v1_b + (n2 - 1) * v2_b) / (n1 + n2 - 2))

        if pooled_std_b > 0
            push!(d_boots, (mean(boot1) - mean(boot2)) / pooled_std_b)
        end
    end

    alpha = 1 - confidence
    ci_lower = quantile(d_boots, alpha / 2)
    ci_upper = quantile(d_boots, 1 - alpha / 2)

    return (d, interpretation, (ci_lower, ci_upper))
end

"""
Calculate eta-squared for k independent groups (ANOVA effect size).

η² = SS_between / SS_total
"""
function eta_squared(groups::Vector{<:AbstractVector{<:Real}})::Tuple{Float64,String}
    all_data = vcat(groups...)
    grand_mean = mean(all_data)

    # Between-group sum of squares
    ss_between = sum(length(g) * (mean(g) - grand_mean)^2 for g in groups)

    # Total sum of squares
    ss_total = sum((x - grand_mean)^2 for x in all_data)

    if ss_total ≈ 0
        return (0.0, "undefined")
    end

    eta_sq = ss_between / ss_total

    # Interpretation (Cohen, 1988)
    interpretation = if eta_sq < 0.01
        "negligible"
    elseif eta_sq < 0.06
        "small"
    elseif eta_sq < 0.14
        "medium"
    else
        "large"
    end

    return (eta_sq, interpretation)
end

"""
Calculate epsilon-squared for Kruskal-Wallis test.

ε² = H / (n - 1), where H is the Kruskal-Wallis statistic.
"""
function epsilon_squared(h_statistic::Float64, n_total::Int, k_groups::Int)::Tuple{Float64,String}
    if n_total <= 1
        return (0.0, "undefined")
    end

    eps_sq = h_statistic / (n_total - 1)

    interpretation = if eps_sq < 0.01
        "negligible"
    elseif eps_sq < 0.06
        "small"
    elseif eps_sq < 0.14
        "medium"
    else
        "large"
    end

    return (eps_sq, interpretation)
end

"""
Calculate Cramér's V effect size for chi-square test.
"""
function cramers_v(chi2::Float64, n::Int, min_dim::Int)::Tuple{Float64,String}
    if min_dim <= 0 || n <= 0
        return (0.0, "undefined")
    end

    v = sqrt(chi2 / (n * min_dim))

    interpretation = if v < 0.1
        "negligible"
    elseif v < 0.3
        "small"
    elseif v < 0.5
        "medium"
    else
        "large"
    end

    return (v, interpretation)
end

# ============================================================================
# ASSUMPTION TESTING
# ============================================================================

"""
Test normality using Shapiro-Wilk-like approach.

Note: Julia doesn't have built-in Shapiro-Wilk, so we use a simplified
D'Agostino-Pearson-style test based on skewness and kurtosis.
"""
function test_normality(data::AbstractVector{<:Real}; alpha::Float64=0.05)::Tuple{Bool,String}
    n = length(data)

    if n < 8
        return (false, "Insufficient data (n < 8)")
    end

    # Use a subsample for very large datasets
    if n > 5000
        rng = MersenneTwister(42)
        data = data[randperm(rng, n)[1:5000]]
        n = 5000
    end

    # Compute skewness and kurtosis
    m = mean(data)
    s = std(data)

    if s ≈ 0
        return (false, "No variance in data")
    end

    z = (data .- m) ./ s
    skew = mean(z.^3)
    kurt = mean(z.^4) - 3  # Excess kurtosis

    # Jarque-Bera test statistic
    jb = n / 6 * (skew^2 + kurt^2 / 4)

    # Chi-square distribution with 2 df
    p_value = 1 - cdf(Chisq(2), jb)

    is_normal = p_value > alpha

    detail = "JB=$(round(jb, digits=4)), p=$(round(p_value, digits=4))"
    if is_normal
        detail *= " (normality assumption met)"
    else
        detail *= " (normality assumption violated)"
    end

    return (is_normal, detail)
end

"""
Test homogeneity of variance using Levene's test (Brown-Forsythe variant).
"""
function test_homogeneity(groups::Vector{<:AbstractVector{<:Real}}; alpha::Float64=0.05)::Tuple{Bool,String}
    valid_groups = filter(g -> length(g) >= 2, groups)

    if length(valid_groups) < 2
        return (false, "Insufficient groups for Levene's test")
    end

    k = length(valid_groups)

    # Brown-Forsythe: use median instead of mean
    medians = [median(g) for g in valid_groups]

    # Absolute deviations from median
    deviations = [abs.(g .- medians[i]) for (i, g) in enumerate(valid_groups)]

    # ANOVA on deviations
    all_devs = vcat(deviations...)
    grand_mean = mean(all_devs)
    group_means = [mean(d) for d in deviations]
    group_sizes = [length(d) for d in deviations]
    n_total = sum(group_sizes)

    # Between-group SS
    ss_between = sum(group_sizes[i] * (group_means[i] - grand_mean)^2 for i in 1:k)

    # Within-group SS
    ss_within = sum(sum((d .- group_means[i]).^2) for (i, d) in enumerate(deviations))

    df_between = k - 1
    df_within = n_total - k

    if df_within <= 0 || ss_within ≈ 0
        return (false, "Cannot compute Levene's test")
    end

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    w_stat = ms_between / ms_within
    p_value = 1 - cdf(FDist(df_between, df_within), w_stat)

    is_homogeneous = p_value > alpha

    detail = "Levene's W=$(round(w_stat, digits=4)), p=$(round(p_value, digits=4))"
    if is_homogeneous
        detail *= " (homogeneity assumption met)"
    else
        detail *= " (homogeneity assumption violated)"
    end

    return (is_homogeneous, detail)
end

# ============================================================================
# KRUSKAL-WALLIS TEST
# ============================================================================

"""
Perform Kruskal-Wallis H test for k independent groups.

Non-parametric alternative to one-way ANOVA.
"""
function kruskal_wallis_test(groups::Vector{<:AbstractVector{<:Real}})::Tuple{Float64,Float64}
    k = length(groups)

    if k < 2
        return (NaN, NaN)
    end

    # Combine all data with group labels
    all_data = Float64[]
    group_labels = Int[]
    for (i, g) in enumerate(groups)
        append!(all_data, g)
        append!(group_labels, fill(i, length(g)))
    end

    n_total = length(all_data)

    if n_total < k + 1
        return (NaN, NaN)
    end

    # Rank all data
    sorted_indices = sortperm(all_data)
    ranks = zeros(Float64, n_total)

    i = 1
    while i <= n_total
        # Handle ties
        j = i
        while j <= n_total && all_data[sorted_indices[j]] ≈ all_data[sorted_indices[i]]
            j += 1
        end

        # Assign average rank to tied values
        avg_rank = (i + j - 1) / 2
        for idx in i:(j-1)
            ranks[sorted_indices[idx]] = avg_rank
        end
        i = j
    end

    # Sum of ranks for each group
    rank_sums = zeros(Float64, k)
    group_sizes = zeros(Int, k)

    for (idx, label) in enumerate(group_labels)
        rank_sums[label] += ranks[idx]
        group_sizes[label] += 1
    end

    # H statistic
    h = 12 / (n_total * (n_total + 1)) * sum(rank_sums[i]^2 / group_sizes[i] for i in 1:k) - 3 * (n_total + 1)

    # Tie correction
    tie_correction = 1.0
    values_sorted = sort(all_data)
    i = 1
    tie_sum = 0.0
    while i <= n_total
        j = i
        while j <= n_total && values_sorted[j] ≈ values_sorted[i]
            j += 1
        end
        t = j - i
        if t > 1
            tie_sum += t^3 - t
        end
        i = j
    end

    if n_total > 1
        tie_correction = 1 - tie_sum / (n_total^3 - n_total)
    end

    if tie_correction > 0
        h /= tie_correction
    end

    # p-value from chi-square distribution with k-1 df
    df = k - 1
    p_value = 1 - cdf(Chisq(df), h)

    return (h, p_value)
end

# ============================================================================
# MANN-WHITNEY U TEST
# ============================================================================

"""
Perform Mann-Whitney U test for two independent groups.

Non-parametric alternative to independent samples t-test.
"""
function mann_whitney_u_test(
    group1::AbstractVector{<:Real},
    group2::AbstractVector{<:Real};
    alternative::String="two-sided"
)::Tuple{Float64,Float64}
    n1, n2 = length(group1), length(group2)

    if n1 < 1 || n2 < 1
        return (NaN, NaN)
    end

    # Combine and rank
    combined = vcat(collect(group1), collect(group2))
    n_total = n1 + n2

    sorted_indices = sortperm(combined)
    ranks = zeros(Float64, n_total)

    i = 1
    while i <= n_total
        j = i
        while j <= n_total && combined[sorted_indices[j]] ≈ combined[sorted_indices[i]]
            j += 1
        end
        avg_rank = (i + j - 1) / 2
        for idx in i:(j-1)
            ranks[sorted_indices[idx]] = avg_rank
        end
        i = j
    end

    # Sum of ranks for group 1
    r1 = sum(ranks[1:n1])

    # U statistics
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1

    # Use smaller U for test
    u_stat = min(u1, u2)

    # Normal approximation for p-value
    mu_u = n1 * n2 / 2
    sigma_u = sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    # Tie correction
    tie_correction = 0.0
    values_sorted = sort(combined)
    i = 1
    while i <= n_total
        j = i
        while j <= n_total && values_sorted[j] ≈ values_sorted[i]
            j += 1
        end
        t = j - i
        if t > 1
            tie_correction += t^3 - t
        end
        i = j
    end

    if n_total > 1 && tie_correction > 0
        sigma_u = sqrt(n1 * n2 / 12 * ((n_total + 1) - tie_correction / (n_total * (n_total - 1))))
    end

    if sigma_u ≈ 0
        return (u_stat, 1.0)
    end

    z = (u_stat - mu_u) / sigma_u

    if alternative == "two-sided"
        p_value = 2 * (1 - cdf(Normal(), abs(z)))
    elseif alternative == "greater"
        p_value = cdf(Normal(), z)
    else  # less
        p_value = 1 - cdf(Normal(), z)
    end

    return (u1, p_value)
end

# ============================================================================
# CHI-SQUARE TEST
# ============================================================================

"""
Perform chi-square test of independence on a contingency table.
"""
function chi_square_test(contingency::Matrix{<:Real})::Tuple{Float64,Float64,Int,Matrix{Float64}}
    nrow, ncol = size(contingency)

    if nrow < 2 || ncol < 2
        return (NaN, NaN, 0, zeros(0, 0))
    end

    row_sums = vec(sum(contingency, dims=2))
    col_sums = vec(sum(contingency, dims=1))
    total = sum(contingency)

    if total ≈ 0
        return (NaN, NaN, 0, zeros(0, 0))
    end

    # Expected frequencies
    expected = (row_sums * col_sums') / total

    # Chi-square statistic
    chi2 = sum((contingency[i, j] - expected[i, j])^2 / expected[i, j]
               for i in 1:nrow, j in 1:ncol
               if expected[i, j] > 0)

    # Degrees of freedom
    df = (nrow - 1) * (ncol - 1)

    # p-value
    p_value = 1 - cdf(Chisq(df), chi2)

    return (chi2, p_value, df, expected)
end

# ============================================================================
# MULTIPLE COMPARISON CORRECTION
# ============================================================================

"""
Apply Benjamini-Hochberg FDR correction to p-values.
"""
function benjamini_hochberg(p_values::Vector{Float64}; alpha::Float64=0.05)::Vector{Float64}
    n = length(p_values)

    if n == 0
        return Float64[]
    end

    # Sort p-values and track original indices
    sorted_indices = sortperm(p_values)
    sorted_p = p_values[sorted_indices]

    # Compute adjusted p-values
    adjusted = zeros(Float64, n)

    # Start from largest p-value
    prev_adj = 1.0
    for i in n:-1:1
        adj = min(prev_adj, sorted_p[i] * n / i)
        adjusted[i] = adj
        prev_adj = adj
    end

    # Restore original order
    result = zeros(Float64, n)
    for i in 1:n
        result[sorted_indices[i]] = adjusted[i]
    end

    return result
end

"""
Apply Holm-Bonferroni correction to p-values.
"""
function holm_bonferroni(p_values::Vector{Float64}; alpha::Float64=0.05)::Vector{Float64}
    n = length(p_values)

    if n == 0
        return Float64[]
    end

    sorted_indices = sortperm(p_values)
    sorted_p = p_values[sorted_indices]

    adjusted = zeros(Float64, n)

    for i in 1:n
        adjusted[i] = min(1.0, sorted_p[i] * (n - i + 1))
    end

    # Enforce monotonicity
    for i in 2:n
        adjusted[i] = max(adjusted[i], adjusted[i-1])
    end

    result = zeros(Float64, n)
    for i in 1:n
        result[sorted_indices[i]] = adjusted[i]
    end

    return result
end

# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

"""
Compute bootstrap confidence interval for a statistic.
"""
function bootstrap_ci_stat(
    data::AbstractVector{<:Real},
    statistic::Function;
    n_bootstrap::Int=10000,
    confidence::Float64=0.95,
    rng::AbstractRNG=MersenneTwister(42)
)::Tuple{Float64,Float64,Float64}
    n = length(data)

    if n < 2
        return (NaN, NaN, NaN)
    end

    observed = statistic(data)
    n_iter = get_bootstrap_iterations(n_bootstrap)

    boot_stats = Float64[]
    for _ in 1:n_iter
        boot_sample = data[rand(rng, 1:n, n)]
        push!(boot_stats, statistic(boot_sample))
    end

    alpha = 1 - confidence
    ci_lower = quantile(boot_stats, alpha / 2)
    ci_upper = quantile(boot_stats, 1 - alpha / 2)

    return (observed, ci_lower, ci_upper)
end

# ============================================================================
# AVERAGE TREATMENT EFFECT ESTIMATION
# ============================================================================

"""
Compute Average Treatment Effect (ATE) with bootstrap confidence intervals.
"""
function compute_ate_bootstrap(
    treatment_values::AbstractVector{<:Real},
    control_values::AbstractVector{<:Real};
    n_bootstrap::Int=5000,
    confidence::Float64=0.95
)::Tuple{Float64,Float64,Float64,Float64}
    # Filter finite values
    treatment_values = filter(isfinite, treatment_values)
    control_values = filter(isfinite, control_values)

    if length(treatment_values) < 2 || length(control_values) < 2
        return (NaN, NaN, NaN, NaN)
    end

    ate = mean(treatment_values) - mean(control_values)

    # Bootstrap for CI
    rng = MersenneTwister(42)
    bootstrap_ates = Float64[]
    n_iter = get_bootstrap_iterations(n_bootstrap)
    n_t, n_c = length(treatment_values), length(control_values)

    for _ in 1:n_iter
        t_sample = treatment_values[rand(rng, 1:n_t, n_t)]
        c_sample = control_values[rand(rng, 1:n_c, n_c)]
        push!(bootstrap_ates, mean(t_sample) - mean(c_sample))
    end

    se = std(bootstrap_ates)
    alpha = 1 - confidence
    ci_lower = quantile(bootstrap_ates, alpha / 2)
    ci_upper = quantile(bootstrap_ates, 1 - alpha / 2)

    return (ate, se, ci_lower, ci_upper)
end

# ============================================================================
# SPEARMAN CORRELATION
# ============================================================================

"""
Compute Spearman rank correlation with p-value.
"""
function spearman_correlation(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real}
)::Tuple{Float64,Float64}
    n = min(length(x), length(y))

    if n < 3
        return (NaN, NaN)
    end

    x = x[1:n]
    y = y[1:n]

    # Compute ranks
    rank_x = zeros(Float64, n)
    rank_y = zeros(Float64, n)

    for (vals, ranks) in [(x, rank_x), (y, rank_y)]
        sorted_indices = sortperm(vals)
        i = 1
        while i <= n
            j = i
            while j <= n && vals[sorted_indices[j]] ≈ vals[sorted_indices[i]]
                j += 1
            end
            avg_rank = (i + j - 1) / 2
            for idx in i:(j-1)
                ranks[sorted_indices[idx]] = avg_rank
            end
            i = j
        end
    end

    # Pearson correlation on ranks
    mean_rx = mean(rank_x)
    mean_ry = mean(rank_y)

    numerator = sum((rank_x[i] - mean_rx) * (rank_y[i] - mean_ry) for i in 1:n)
    denom_x = sqrt(sum((rank_x[i] - mean_rx)^2 for i in 1:n))
    denom_y = sqrt(sum((rank_y[i] - mean_ry)^2 for i in 1:n))

    if denom_x ≈ 0 || denom_y ≈ 0
        return (0.0, 1.0)
    end

    rho = numerator / (denom_x * denom_y)

    # t-statistic for p-value
    t_stat = rho * sqrt((n - 2) / (1 - rho^2 + 1e-10))
    p_value = 2 * (1 - cdf(TDist(n - 2), abs(t_stat)))

    return (rho, p_value)
end

# ============================================================================
# WELCH'S T-TEST
# ============================================================================

"""
Perform Welch's t-test (unequal variances t-test).
"""
function welch_ttest(
    group1::AbstractVector{<:Real},
    group2::AbstractVector{<:Real};
    alternative::String="two-sided"
)::Tuple{Float64,Float64}
    n1, n2 = length(group1), length(group2)

    if n1 < 2 || n2 < 2
        return (NaN, NaN)
    end

    m1, m2 = mean(group1), mean(group2)
    v1, v2 = var(group1), var(group2)

    # Welch's t-statistic
    se = sqrt(v1 / n1 + v2 / n2)

    if se ≈ 0
        return (0.0, 1.0)
    end

    t_stat = (m1 - m2) / se

    # Welch-Satterthwaite degrees of freedom
    df = (v1 / n1 + v2 / n2)^2 /
         ((v1 / n1)^2 / (n1 - 1) + (v2 / n2)^2 / (n2 - 1))

    # p-value
    if alternative == "two-sided"
        p_value = 2 * (1 - cdf(TDist(df), abs(t_stat)))
    elseif alternative == "greater"
        p_value = 1 - cdf(TDist(df), t_stat)
    else
        p_value = cdf(TDist(df), t_stat)
    end

    return (t_stat, p_value)
end

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

"""
Compute comprehensive descriptive statistics for a vector.
"""
function descriptive_stats(data::AbstractVector{<:Real})::Dict{String,Float64}
    clean_data = filter(isfinite, data)
    n = length(clean_data)

    if n == 0
        return Dict{String,Float64}(
            "n" => 0.0,
            "mean" => NaN,
            "std" => NaN,
            "min" => NaN,
            "max" => NaN,
            "median" => NaN,
            "q1" => NaN,
            "q3" => NaN,
            "skewness" => NaN,
            "kurtosis" => NaN
        )
    end

    m = mean(clean_data)
    s = std(clean_data)

    # Skewness and kurtosis
    if s > 0 && n >= 3
        z = (clean_data .- m) ./ s
        skew = mean(z.^3)
        kurt = mean(z.^4) - 3  # Excess kurtosis
    else
        skew = NaN
        kurt = NaN
    end

    sorted_data = sort(clean_data)

    return Dict{String,Float64}(
        "n" => Float64(n),
        "mean" => m,
        "std" => s,
        "min" => minimum(clean_data),
        "max" => maximum(clean_data),
        "median" => median(clean_data),
        "q1" => quantile(sorted_data, 0.25),
        "q3" => quantile(sorted_data, 0.75),
        "skewness" => skew,
        "kurtosis" => kurt
    )
end

# ============================================================================
# SIGNIFICANCE FORMATTING
# ============================================================================

"""
Get significance stars for p-value.
"""
function significance_stars(p_value::Float64)::String
    if p_value < 0.001
        return "***"
    elseif p_value < 0.01
        return "**"
    elseif p_value < 0.05
        return "*"
    elseif p_value < 0.10
        return "†"
    else
        return ""
    end
end

"""
Format p-value for display.
"""
function format_p_value(p_value::Float64)::String
    if p_value < 0.0001
        return "<0.0001"
    else
        return string(round(p_value, digits=4))
    end
end

# Note: cliffs_delta is defined in causal.jl and returns EffectSizeResult
# Use result.value for the delta value and result.interpretation for text

# ============================================================================
# RIGOROUS STATISTICAL ANALYSIS CLASS
# ============================================================================

"""
Publication-quality statistical analysis suite for ABM results.

All analyses include:
- Effect sizes with 95% confidence intervals
- Assumption testing
- Multiple comparison correction (Benjamini-Hochberg FDR)
- Plain-language interpretations
"""
mutable struct RigorousStatisticalAnalysis
    agent_df::DataFrame
    decision_df::DataFrame
    matured_df::DataFrame
    uncertainty_detail_df::DataFrame
    alpha::Float64
    results::Vector{StatisticalTestResult}
end

function RigorousStatisticalAnalysis(;
    agent_df::DataFrame,
    decision_df::DataFrame,
    matured_df::DataFrame=DataFrame(),
    uncertainty_detail_df::DataFrame=DataFrame(),
    alpha::Float64=0.05
)
    RigorousStatisticalAnalysis(
        agent_df, decision_df, matured_df, uncertainty_detail_df,
        alpha, StatisticalTestResult[]
    )
end

"""
Run complete statistical analysis suite.
"""
function run_all_analyses!(analysis::RigorousStatisticalAnalysis)::DataFrame
    println("\n" * "="^70)
    println("RIGOROUS STATISTICAL ANALYSIS FOR AMJ SUBMISSION")
    println("="^70)

    # Hypothesis 1: AI tier effects on performance
    _test_ai_performance_effects!(analysis)

    # Hypothesis 2: AI effects on survival
    _test_ai_survival_effects!(analysis)

    # Hypothesis 3: AI effects on uncertainty dimensions
    _test_ai_uncertainty_effects!(analysis)

    # Hypothesis 4: AI effects on investment outcomes
    _test_ai_investment_outcomes!(analysis)

    # Hypothesis 5: Paradox of future knowledge
    _test_paradox_of_knowledge!(analysis)

    # Apply multiple comparison correction
    _apply_fdr_correction!(analysis)

    # Generate results table
    results_df = _generate_results_table(analysis)

    println("\n" * "="^70)
    println("Completed $(length(analysis.results)) statistical tests")
    println("="^70)

    return results_df
end

"""
Test H1: AI augmentation affects entrepreneurial performance.
"""
function _test_ai_performance_effects!(analysis::RigorousStatisticalAnalysis)
    println("\n📊 Testing H1: AI Effects on Performance...")

    if isempty(analysis.agent_df) || !("capital_growth" in names(analysis.agent_df))
        println("   ⚠️ Insufficient data for performance analysis")
        return
    end

    # Get AI level column
    ai_col = "primary_ai_canonical" in names(analysis.agent_df) ? "primary_ai_canonical" :
             "ai_level" in names(analysis.agent_df) ? "ai_level" : nothing

    if isnothing(ai_col)
        println("   ⚠️ No AI level column found")
        return
    end

    # Prepare groups
    ai_levels = ["none", "basic", "advanced", "premium"]
    groups = Vector{Float64}[]
    sample_sizes = Dict{String,Int}()

    for level in ai_levels
        mask = coalesce.(analysis.agent_df[!, ai_col], "none") .== level
        data = filter(!isnan, analysis.agent_df[mask, :capital_growth])
        if length(data) > 0
            push!(groups, data)
            sample_sizes[level] = length(data)
        end
    end

    if length(groups) < 2
        println("   ⚠️ Need at least 2 AI groups with data")
        return
    end

    # Assumption tests
    assumptions_met = Dict{String,Bool}()
    assumptions_details = Dict{String,String}()

    available_levels = [l for l in ai_levels if haskey(sample_sizes, l)]
    for (i, level) in enumerate(available_levels)
        is_normal, detail = test_normality(groups[i])
        assumptions_met["normality_$level"] = is_normal
        assumptions_details["normality_$level"] = detail
    end

    is_homogeneous, detail = test_homogeneity(groups)
    assumptions_met["homogeneity"] = is_homogeneous
    assumptions_details["homogeneity"] = detail

    # Kruskal-Wallis test
    h_stat, p_value = kruskal_wallis_test(groups)

    # Effect size: epsilon-squared
    n_total = sum(values(sample_sizes))
    effect_size, effect_interp = epsilon_squared(h_stat, n_total, length(groups))

    # Conclusion
    conclusion = if p_value < analysis.alpha
        "AI tier significantly affects capital growth (H=$(round(h_stat, digits=2)), p=$(round(p_value, digits=4)), ε²=$(round(effect_size, digits=3)) [$effect_interp]). This supports H1."
    else
        "No significant difference in capital growth across AI tiers (H=$(round(h_stat, digits=2)), p=$(round(p_value, digits=4)), ε²=$(round(effect_size, digits=3))). H1 not supported."
    end

    result = StatisticalTestResult(
        test_name="H1: AI Tier → Capital Growth (Kruskal-Wallis)",
        test_statistic=h_stat,
        p_value=p_value,
        effect_size=effect_size,
        effect_size_type="epsilon-squared (ε²)",
        effect_interpretation=effect_interp,
        sample_sizes=sample_sizes,
        assumptions_met=assumptions_met,
        assumptions_details=assumptions_details,
        conclusion=conclusion
    )
    push!(analysis.results, result)
    println("   ✓ $conclusion")

    # Pairwise comparisons
    _pairwise_ai_comparisons!(analysis, groups, available_levels, sample_sizes, "capital_growth")
end

"""
Run pairwise Mann-Whitney U tests with Cliff's delta effect sizes.
"""
function _pairwise_ai_comparisons!(
    analysis::RigorousStatisticalAnalysis,
    groups::Vector{Vector{Float64}},
    levels::Vector{String},
    sample_sizes::Dict{String,Int},
    metric_name::String
)
    for i in 1:length(levels)
        for j in (i+1):length(levels)
            level1, level2 = levels[i], levels[j]
            g1, g2 = groups[i], groups[j]

            if length(g1) < 2 || length(g2) < 2
                continue
            end

            # Mann-Whitney U test
            u_stat, p_value = mann_whitney_u_test(g1, g2)

            # Cliff's delta (returns EffectSizeResult)
            cliff_result = cliffs_delta(g1, g2)
            delta = cliff_result.value
            interp = cliff_result.interpretation

            result = StatisticalTestResult(
                test_name="Pairwise: $level1 vs $level2 ($metric_name)",
                test_statistic=u_stat,
                p_value=p_value,
                effect_size=delta,
                effect_size_type="Cliff's delta (δ)",
                effect_interpretation=interp,
                sample_sizes=Dict(level1 => length(g1), level2 => length(g2)),
                assumptions_met=Dict{String,Bool}(),
                assumptions_details=Dict{String,String}(),
                conclusion="$(p_value < analysis.alpha ? "Significant" : "Non-significant") difference (δ=$(round(delta, digits=3)), $interp)"
            )
            push!(analysis.results, result)
        end
    end
end

"""
Test H2: AI augmentation affects entrepreneurial survival.
"""
function _test_ai_survival_effects!(analysis::RigorousStatisticalAnalysis)
    println("\n📊 Testing H2: AI Effects on Survival...")

    if isempty(analysis.agent_df) || !("survived" in names(analysis.agent_df))
        println("   ⚠️ Insufficient data for survival analysis")
        return
    end

    ai_col = "primary_ai_canonical" in names(analysis.agent_df) ? "primary_ai_canonical" :
             "ai_level" in names(analysis.agent_df) ? "ai_level" : nothing

    if isnothing(ai_col)
        println("   ⚠️ No AI level column found")
        return
    end

    # Create contingency table
    ai_levels = ["none", "basic", "advanced", "premium"]
    contingency = zeros(Int, 4, 2)  # 4 AI levels × 2 survival outcomes
    sample_sizes = Dict{String,Int}()

    for (i, level) in enumerate(ai_levels)
        mask = coalesce.(analysis.agent_df[!, ai_col], "none") .== level
        subset = analysis.agent_df[mask, :]

        if nrow(subset) == 0
            continue
        end

        survived = sum(coalesce.(subset.survived, false))
        failed = nrow(subset) - survived

        contingency[i, 1] = failed
        contingency[i, 2] = survived
        sample_sizes[level] = nrow(subset)
    end

    # Remove empty rows
    non_empty = [i for i in 1:4 if sum(contingency[i, :]) > 0]
    if length(non_empty) < 2
        println("   ⚠️ Insufficient variation for chi-square test")
        return
    end

    contingency_clean = contingency[non_empty, :]

    # Chi-square test
    chi2, p_value, df, expected = chi_square_test(contingency_clean)

    # Cramér's V
    n = sum(contingency_clean)
    min_dim = min(size(contingency_clean, 1) - 1, size(contingency_clean, 2) - 1)
    v, interp = cramers_v(chi2, n, min_dim)

    conclusion = if p_value < analysis.alpha
        "AI tier significantly affects survival rates (χ²=$(round(chi2, digits=2)), df=$df, p=$(round(p_value, digits=4)), V=$(round(v, digits=3)) [$interp])."
    else
        "No significant association between AI tier and survival (χ²=$(round(chi2, digits=2)), df=$df, p=$(round(p_value, digits=4)), V=$(round(v, digits=3)))."
    end

    result = StatisticalTestResult(
        test_name="H2: AI Tier → Survival (Chi-Square)",
        test_statistic=chi2,
        p_value=p_value,
        effect_size=v,
        effect_size_type="Cramér's V",
        effect_interpretation=interp,
        sample_sizes=sample_sizes,
        assumptions_met=Dict("expected_freq_>5" => all(expected .>= 5)),
        assumptions_details=Dict("min_expected" => "$(round(minimum(expected), digits=1))"),
        conclusion=conclusion
    )
    push!(analysis.results, result)
    println("   ✓ $conclusion")
end

"""
Test H3: AI effects on uncertainty dimensions.
"""
function _test_ai_uncertainty_effects!(analysis::RigorousStatisticalAnalysis)
    println("\n📊 Testing H3: AI Effects on Uncertainty Dimensions...")

    if isempty(analysis.decision_df)
        println("   ⚠️ Insufficient decision data")
        return
    end

    uncertainty_cols = Dict(
        "actor_ignorance" => ["perc_actor_ignorance_level", "perc_actor_ignorance", "actor_ignorance_level"],
        "practical_indeterminism" => ["perc_practical_indeterminism_level", "perc_practical_indeterminism", "indeterminism_level"],
        "agentic_novelty" => ["perc_agentic_novelty_potential", "perc_agentic_novelty", "novelty_potential"],
        "competitive_recursion" => ["perc_competitive_recursion_level", "perc_competitive_recursion", "recursion_level"]
    )

    ai_col = "ai_level_used" in names(analysis.decision_df) ? "ai_level_used" :
             "ai_level" in names(analysis.decision_df) ? "ai_level" : nothing

    if isnothing(ai_col)
        println("   ⚠️ No AI level column found")
        return
    end

    for (dimension, possible_cols) in uncertainty_cols
        col = nothing
        for c in possible_cols
            if c in names(analysis.decision_df)
                col = c
                break
            end
        end

        if isnothing(col)
            continue
        end

        # Prepare groups
        ai_levels = ["none", "basic", "advanced", "premium"]
        groups = Vector{Float64}[]
        sample_sizes = Dict{String,Int}()

        for level in ai_levels
            mask = coalesce.(analysis.decision_df[!, ai_col], "none") .== level
            data = filter(x -> !isnan(x) && !ismissing(x), analysis.decision_df[mask, col])
            if length(data) > 0
                push!(groups, Float64.(data))
                sample_sizes[level] = length(data)
            end
        end

        if length(groups) < 2
            continue
        end

        # Kruskal-Wallis test
        h_stat, p_value = kruskal_wallis_test(groups)

        # Effect size
        n_total = sum(values(sample_sizes))
        effect_size, effect_interp = epsilon_squared(h_stat, n_total, length(groups))

        result = StatisticalTestResult(
            test_name="H3a: AI Tier → $(replace(dimension, "_" => " ") |> titlecase)",
            test_statistic=h_stat,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_type="epsilon-squared (ε²)",
            effect_interpretation=effect_interp,
            sample_sizes=sample_sizes,
            assumptions_met=Dict{String,Bool}(),
            assumptions_details=Dict{String,String}(),
            conclusion="AI affects $(replace(dimension, "_" => " ")) (H=$(round(h_stat, digits=2)), p=$(round(p_value, digits=4)), ε²=$(round(effect_size, digits=3)))"
        )
        push!(analysis.results, result)
        println("   ✓ $dimension: H=$(round(h_stat, digits=2)), p=$(round(p_value, digits=4)), ε²=$(round(effect_size, digits=3))")
    end
end

"""
Test H4: AI effects on investment outcomes.
"""
function _test_ai_investment_outcomes!(analysis::RigorousStatisticalAnalysis)
    println("\n📊 Testing H4: AI Effects on Investment Outcomes...")

    if isempty(analysis.matured_df)
        println("   ⚠️ No matured investment data available")
        return
    end

    # Find ROI column
    roi_col = nothing
    for col in ["realized_roi", "return_multiple", "roi", "realized_return"]
        if col in names(analysis.matured_df)
            roi_col = col
            break
        end
    end

    if isnothing(roi_col)
        println("   ⚠️ No ROI column found")
        return
    end

    # Find AI level column
    ai_col = nothing
    for col in ["ai_level_used", "ai_level", "ai_tier"]
        if col in names(analysis.matured_df)
            ai_col = col
            break
        end
    end

    if isnothing(ai_col)
        println("   ⚠️ No AI level column found")
        return
    end

    # Prepare groups
    ai_levels = ["none", "basic", "advanced", "premium"]
    groups = Vector{Float64}[]
    sample_sizes = Dict{String,Int}()

    for level in ai_levels
        mask = coalesce.(analysis.matured_df[!, ai_col], "none") .== level
        data = filter(x -> !isnan(x) && !ismissing(x), analysis.matured_df[mask, roi_col])
        if length(data) > 0
            push!(groups, Float64.(data))
            sample_sizes[level] = length(data)
        end
    end

    if length(groups) < 2
        println("   ⚠️ Need at least 2 AI groups with matured investments")
        return
    end

    # Kruskal-Wallis test
    h_stat, p_value = kruskal_wallis_test(groups)

    # Effect size
    n_total = sum(values(sample_sizes))
    effect_size, effect_interp = epsilon_squared(h_stat, n_total, length(groups))

    conclusion = if p_value < analysis.alpha
        "AI tier significantly affects investment ROI (H=$(round(h_stat, digits=2)), p=$(round(p_value, digits=4)))"
    else
        "AI tier does not significantly affect investment ROI (H=$(round(h_stat, digits=2)), p=$(round(p_value, digits=4)))"
    end

    result = StatisticalTestResult(
        test_name="H4: AI Tier → Investment ROI (Kruskal-Wallis)",
        test_statistic=h_stat,
        p_value=p_value,
        effect_size=effect_size,
        effect_size_type="epsilon-squared (ε²)",
        effect_interpretation=effect_interp,
        sample_sizes=sample_sizes,
        assumptions_met=Dict{String,Bool}(),
        assumptions_details=Dict{String,String}(),
        conclusion=conclusion
    )
    push!(analysis.results, result)
    println("   ✓ $conclusion")
end

"""
Test H5: Paradox of future knowledge.
"""
function _test_paradox_of_knowledge!(analysis::RigorousStatisticalAnalysis)
    println("\n📊 Testing H5: Paradox of Future Knowledge...")

    if isempty(analysis.decision_df)
        println("   ⚠️ Insufficient data")
        return
    end

    ai_col = "ai_level_used" in names(analysis.decision_df) ? "ai_level_used" :
             "ai_level" in names(analysis.decision_df) ? "ai_level" : nothing

    if isnothing(ai_col)
        println("   ⚠️ No AI level column found")
        return
    end

    # Find ignorance column
    ignorance_col = nothing
    for col in names(analysis.decision_df)
        if occursin("ignorance", lowercase(col))
            ignorance_col = col
            break
        end
    end

    if isnothing(ignorance_col)
        println("   ⚠️ Missing uncertainty dimension columns")
        return
    end

    # Compare AI users vs non-users
    ai_users_mask = coalesce.(analysis.decision_df[!, ai_col], "none") .!= "none"
    non_users_mask = coalesce.(analysis.decision_df[!, ai_col], "none") .== "none"

    ai_users = filter(x -> !isnan(x) && !ismissing(x), analysis.decision_df[ai_users_mask, ignorance_col])
    non_users = filter(x -> !isnan(x) && !ismissing(x), analysis.decision_df[non_users_mask, ignorance_col])

    if length(ai_users) < 10 || length(non_users) < 10
        println("   ⚠️ Insufficient sample sizes")
        return
    end

    # Test: AI reduces actor ignorance
    u_stat, p_val = mann_whitney_u_test(Float64.(non_users), Float64.(ai_users))
    cliff_result = cliffs_delta(Float64.(non_users), Float64.(ai_users))
    delta = cliff_result.value
    interp = cliff_result.interpretation

    conclusion = if p_val < analysis.alpha
        "AI users have significantly lower actor ignorance (δ=$(round(delta, digits=3)))"
    else
        "AI users do not have significantly lower actor ignorance (δ=$(round(delta, digits=3)))"
    end

    result = StatisticalTestResult(
        test_name="H5a: AI Users Have Lower Actor Ignorance (Mann-Whitney U)",
        test_statistic=u_stat,
        p_value=p_val,
        effect_size=delta,
        effect_size_type="Cliff's delta (δ)",
        effect_interpretation=interp,
        sample_sizes=Dict("non_users" => length(non_users), "ai_users" => length(ai_users)),
        assumptions_met=Dict{String,Bool}(),
        assumptions_details=Dict("mean_non_users" => string(round(mean(non_users), digits=4)),
                                "mean_ai_users" => string(round(mean(ai_users), digits=4))),
        conclusion=conclusion
    )
    push!(analysis.results, result)
    println("   ✓ Actor ignorance: δ=$(round(delta, digits=3)), p=$(round(p_val, digits=4))")
end

"""
Apply Benjamini-Hochberg FDR correction to all p-values.
"""
function _apply_fdr_correction!(analysis::RigorousStatisticalAnalysis)
    if isempty(analysis.results)
        return
    end

    println("\n📊 Applying Benjamini-Hochberg FDR Correction...")

    p_values = [r.p_value for r in analysis.results]
    p_adjusted = benjamini_hochberg(p_values; alpha=analysis.alpha)

    # Update results with adjusted p-values
    new_results = StatisticalTestResult[]
    for (i, result) in enumerate(analysis.results)
        push!(new_results, StatisticalTestResult(
            test_name=result.test_name,
            test_statistic=result.test_statistic,
            p_value=result.p_value,
            p_value_adjusted=p_adjusted[i],
            effect_size=result.effect_size,
            effect_size_type=result.effect_size_type,
            effect_size_ci=result.effect_size_ci,
            effect_interpretation=result.effect_interpretation,
            sample_sizes=result.sample_sizes,
            assumptions_met=result.assumptions_met,
            assumptions_details=result.assumptions_details,
            conclusion=result.conclusion
        ))
    end
    analysis.results = new_results

    n_significant_raw = count(p -> p < analysis.alpha, p_values)
    n_significant_adj = count(p -> p < analysis.alpha, p_adjusted)

    println("   ✓ $n_significant_raw tests significant at α=$(analysis.alpha) (unadjusted)")
    println("   ✓ $n_significant_adj tests significant after FDR correction")
end

"""
Generate publication-ready results table.
"""
function _generate_results_table(analysis::RigorousStatisticalAnalysis)::DataFrame
    if isempty(analysis.results)
        return DataFrame()
    end

    rows = [to_dict(r) for r in analysis.results]
    df = DataFrame(rows)

    return df
end

"""
Generate comprehensive descriptive statistics table.
"""
function generate_descriptive_statistics(analysis::RigorousStatisticalAnalysis)::DataFrame
    if isempty(analysis.agent_df)
        return DataFrame()
    end

    ai_col = "primary_ai_canonical" in names(analysis.agent_df) ? "primary_ai_canonical" :
             "ai_level" in names(analysis.agent_df) ? "ai_level" : nothing

    if isnothing(ai_col)
        return DataFrame()
    end

    numeric_vars = ["final_capital", "capital_growth", "innovations", "portfolio_diversity", "survived"]
    available_vars = [v for v in numeric_vars if v in names(analysis.agent_df)]

    stats_list = Dict{String,Any}[]

    for level in ["none", "basic", "advanced", "premium"]
        mask = coalesce.(analysis.agent_df[!, ai_col], "none") .== level
        subset = analysis.agent_df[mask, :]

        if nrow(subset) == 0
            continue
        end

        row = Dict{String,Any}("AI_Tier" => titlecase(level), "n" => nrow(subset))

        for var in available_vars
            data = filter(x -> !ismissing(x) && !isnan(x), subset[!, var])
            if length(data) > 0
                row["$(var)_mean"] = mean(data)
                row["$(var)_sd"] = std(data)
                row["$(var)_median"] = median(data)
            end
        end

        push!(stats_list, row)
    end

    return DataFrame(stats_list)
end

# ============================================================================
# CAUSAL IDENTIFICATION ANALYSIS
# ============================================================================

"""
Causal identification analysis for AI tier effects.

Generates publication-ready tables distinguishing between causal estimates
(from fixed-tier designs) and associational estimates (from emergent selection).
"""
mutable struct CausalIdentificationAnalysis
    agent_df::DataFrame
    matured_df::DataFrame
    decision_df::DataFrame
    uncertainty_detail_df::DataFrame
    is_fixed_tier::Bool
    results::Vector{CausalEffectEstimate}
end

function CausalIdentificationAnalysis(;
    agent_df::DataFrame,
    matured_df::DataFrame=DataFrame(),
    decision_df::DataFrame=DataFrame(),
    uncertainty_detail_df::DataFrame=DataFrame(),
    is_fixed_tier::Bool=false
)
    CausalIdentificationAnalysis(
        agent_df, matured_df, decision_df, uncertainty_detail_df,
        is_fixed_tier, CausalEffectEstimate[]
    )
end

"""
Estimate causal effects of AI tier on survival.
"""
function estimate_survival_effects!(analysis::CausalIdentificationAnalysis)
    ai_col = "primary_ai_canonical" in names(analysis.agent_df) ? "primary_ai_canonical" :
             "ai_level" in names(analysis.agent_df) ? "ai_level" : nothing

    if isnothing(ai_col)
        return
    end

    # Determine survival column
    if "survived" in names(analysis.agent_df)
        analysis.agent_df[!, :survived_float] = Float64.(coalesce.(analysis.agent_df.survived, false))
    elseif "final_status" in names(analysis.agent_df)
        analysis.agent_df[!, :survived_float] = Float64.(analysis.agent_df.final_status .== "active")
    elseif "alive" in names(analysis.agent_df)
        analysis.agent_df[!, :survived_float] = Float64.(coalesce.(analysis.agent_df.alive, false))
    else
        return
    end

    analysis.agent_df[!, :ai_tier] = coalesce.(analysis.agent_df[!, ai_col], "none")

    control = Float64.(analysis.agent_df[analysis.agent_df.ai_tier .== "none", :survived_float])

    for tier in ["basic", "advanced", "premium"]
        treatment = Float64.(analysis.agent_df[analysis.agent_df.ai_tier .== tier, :survived_float])

        if length(treatment) < 5 || length(control) < 5
            continue
        end

        ate, se, ci_lower, ci_upper = compute_ate_bootstrap(treatment, control)
        d, interp, (d_lower, d_upper) = cohens_d_with_ci(treatment, control)

        # Two-proportion z-test for p-value
        p1, p2 = mean(treatment), mean(control)
        n1, n2 = length(treatment), length(control)
        pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se_diff = sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
        z = se_diff > 0 ? (p1 - p2) / se_diff : 0.0
        p_value = 2 * (1 - cdf(Normal(), abs(z)))

        identification = analysis.is_fixed_tier ? "fixed-tier (exogenous)" : "emergent-selection (endogenous)"
        robustness = analysis.is_fixed_tier ? "Primary specification" : "Subject to selection bias"

        push!(analysis.results, CausalEffectEstimate(
            "$(tier)_vs_none",
            "survival_rate",
            ate, se, ci_lower, ci_upper,
            d, d_lower, d_upper,
            length(treatment), length(control),
            p_value, identification, robustness
        ))
    end
end

"""
Estimate causal effects of AI tier on investment ROI.
"""
function estimate_roi_effects!(analysis::CausalIdentificationAnalysis)
    if isempty(analysis.matured_df)
        return
    end

    ai_col = nothing
    for col in ["ai_level_used", "ai_level", "ai_tier"]
        if col in names(analysis.matured_df)
            ai_col = col
            break
        end
    end

    if isnothing(ai_col)
        return
    end

    roi_col = nothing
    for col in ["realized_roi", "realized_multiplier", "roi"]
        if col in names(analysis.matured_df)
            roi_col = col
            break
        end
    end

    if isnothing(roi_col)
        return
    end

    analysis.matured_df[!, :ai_tier] = lowercase.(string.(coalesce.(analysis.matured_df[!, ai_col], "none")))

    control_mask = analysis.matured_df.ai_tier .== "none"
    control_data = filter(isfinite, Float64.(analysis.matured_df[control_mask, roi_col]))

    for tier in ["basic", "advanced", "premium"]
        treatment_mask = analysis.matured_df.ai_tier .== tier
        treatment_data = filter(isfinite, Float64.(analysis.matured_df[treatment_mask, roi_col]))

        if length(treatment_data) < 5 || length(control_data) < 5
            continue
        end

        ate, se, ci_lower, ci_upper = compute_ate_bootstrap(treatment_data, control_data)
        d, interp, (d_lower, d_upper) = cohens_d_with_ci(treatment_data, control_data)

        # Welch's t-test for p-value
        _, p_value = welch_ttest(treatment_data, control_data)

        identification = analysis.is_fixed_tier ? "fixed-tier (exogenous)" : "emergent-selection (endogenous)"
        robustness = analysis.is_fixed_tier ? "Primary specification" : "Subject to selection bias"

        push!(analysis.results, CausalEffectEstimate(
            "$(tier)_vs_none",
            "investment_roi",
            ate, se, ci_lower, ci_upper,
            d, d_lower, d_upper,
            length(treatment_data), length(control_data),
            p_value, identification, robustness
        ))
    end
end

"""
Run all causal effect estimations.
"""
function run_all_estimates!(analysis::CausalIdentificationAnalysis)
    estimate_survival_effects!(analysis)
    estimate_roi_effects!(analysis)
end

"""
Generate publication-ready table of causal effects.
"""
function generate_causal_effects_table(analysis::CausalIdentificationAnalysis)::DataFrame
    if isempty(analysis.results)
        return DataFrame()
    end

    rows = Dict{String,Any}[]
    for r in analysis.results
        push!(rows, Dict{String,Any}(
            "Treatment" => r.treatment,
            "Outcome" => r.outcome,
            "ATE" => round(r.ate, digits=4),
            "ATE_SE" => round(r.ate_se, digits=4),
            "ATE_95_CI" => "[$(round(r.ate_ci_lower, digits=4)), $(round(r.ate_ci_upper, digits=4))]",
            "Cohens_d" => round(r.cohens_d, digits=3),
            "d_95_CI" => "[$(round(r.cohens_d_ci_lower, digits=3)), $(round(r.cohens_d_ci_upper, digits=3))]",
            "p_value" => r.p_value >= 0.0001 ? round(r.p_value, digits=4) : "<0.0001",
            "N_Treatment" => r.n_treatment,
            "N_Control" => r.n_control,
            "Identification" => r.identification,
            "Robustness" => r.robustness_check
        ))
    end

    return DataFrame(rows)
end

"""
Interpret Cohen's d effect size.
"""
function _interpret_cohens_d(d::Float64)::String
    d_abs = abs(d)
    if d_abs < 0.2
        return "negligible"
    elseif d_abs < 0.5
        return "small"
    elseif d_abs < 0.8
        return "medium"
    else
        return "large"
    end
end

# ============================================================================
# COX SURVIVAL ANALYSIS (SIMPLIFIED)
# ============================================================================

"""
Results from Cox proportional hazards-style analysis.
"""
struct CoxRegressionResult
    model_name::String
    n_observations::Int
    n_events::Int
    concordance_index::Float64
    coefficients::Dict{String,Dict{String,Float64}}
    interpretation::String
end

"""
Cox-style survival analysis for agent failure.

Note: This is a simplified implementation. For full Cox regression,
use R or Python's lifelines package.
"""
mutable struct CoxSurvivalAnalysis
    agent_df::DataFrame
    max_time::Union{Int,Nothing}
    results::Vector{CoxRegressionResult}
end

function CoxSurvivalAnalysis(agent_df::DataFrame; max_time::Union{Int,Nothing}=nothing)
    CoxSurvivalAnalysis(copy(agent_df), max_time, CoxRegressionResult[])
end

"""
Prepare data for survival analysis.
"""
function prepare_survival_data(analysis::CoxSurvivalAnalysis)::DataFrame
    df = copy(analysis.agent_df)

    # Determine failure time
    if "failure_step" in names(df)
        df[!, :duration] = coalesce.(df.failure_step, analysis.max_time !== nothing ? analysis.max_time : 100)
    elseif "final_step" in names(df)
        df[!, :duration] = df.final_step
    else
        df[!, :duration] = fill(analysis.max_time !== nothing ? analysis.max_time : 100, nrow(df))
    end

    # Determine event indicator
    if "final_status" in names(df)
        df[!, :event] = Int.(df.final_status .== "failed")
    elseif "survived" in names(df)
        df[!, :event] = Int.(.!coalesce.(df.survived, true))
    elseif "alive" in names(df)
        df[!, :event] = Int.(.!coalesce.(df.alive, true))
    else
        println("   ⚠️ Cannot determine event indicator")
        return DataFrame()
    end

    # Create AI tier dummies
    ai_col = "primary_ai_canonical" in names(df) ? "primary_ai_canonical" :
             "ai_level" in names(df) ? "ai_level" : nothing

    if isnothing(ai_col)
        println("   ⚠️ No AI tier column found")
        return DataFrame()
    end

    df[!, :ai_tier] = coalesce.(df[!, ai_col], "none")
    df[!, :ai_basic] = Int.(df.ai_tier .== "basic")
    df[!, :ai_advanced] = Int.(df.ai_tier .== "advanced")
    df[!, :ai_premium] = Int.(df.ai_tier .== "premium")

    # Ensure duration is positive
    df[!, :duration] = max.(df.duration, 1)

    return df
end

"""
Compute log-rank test statistics for survival comparison.
"""
function log_rank_tests(analysis::CoxSurvivalAnalysis)::DataFrame
    df = prepare_survival_data(analysis)
    if isempty(df)
        return DataFrame()
    end

    tiers = ["none", "basic", "advanced", "premium"]
    results = Dict{String,Any}[]

    for i in 1:length(tiers)
        for j in (i+1):length(tiers)
            tier1, tier2 = tiers[i], tiers[j]

            g1 = df[df.ai_tier .== tier1, :]
            g2 = df[df.ai_tier .== tier2, :]

            if nrow(g1) < 5 || nrow(g2) < 5
                continue
            end

            # Simplified log-rank using Mann-Whitney on survival times
            # (Real log-rank would handle censoring properly)
            u_stat, p_value = mann_whitney_u_test(Float64.(g1.duration), Float64.(g2.duration))

            push!(results, Dict{String,Any}(
                "comparison" => "$tier1 vs $tier2",
                "test_statistic" => u_stat,
                "p_value" => p_value,
                "n_tier1" => nrow(g1),
                "n_tier2" => nrow(g2),
                "events_tier1" => sum(g1.event),
                "events_tier2" => sum(g2.event)
            ))
        end
    end

    return DataFrame(results)
end

"""
Kaplan-Meier survival estimates by tier (simplified).
"""
function kaplan_meier_by_tier(analysis::CoxSurvivalAnalysis)::Dict{String,Vector{Float64}}
    df = prepare_survival_data(analysis)
    if isempty(df)
        return Dict{String,Vector{Float64}}()
    end

    km_curves = Dict{String,Vector{Float64}}()

    for tier in ["none", "basic", "advanced", "premium"]
        subset = df[df.ai_tier .== tier, :]
        if nrow(subset) < 5
            continue
        end

        # Simplified KM: proportion surviving at each time point
        max_t = maximum(subset.duration)
        survival = Float64[]
        n = nrow(subset)

        for t in 1:max_t
            alive_at_t = sum(subset.duration .>= t)
            push!(survival, alive_at_t / n)
        end

        km_curves[tier] = survival
    end

    return km_curves
end

# ============================================================================
# DIFFERENCE-IN-DIFFERENCES ANALYSIS (SIMPLIFIED)
# ============================================================================

"""
Results from Difference-in-Differences analysis.
"""
struct DiDResult
    outcome::String
    att::Float64
    att_se::Float64
    att_ci_lower::Float64
    att_ci_upper::Float64
    p_value::Float64
    n_treated::Int
    n_control::Int
    n_periods_pre::Int
    n_periods_post::Int
    interpretation::String
end

"""
Difference-in-Differences analysis for AI adoption effects.

Compares changes over time between AI adopters and non-adopters.
"""
mutable struct DifferenceInDifferencesAnalysis
    panel_df::DataFrame
    agent_df::DataFrame
    results::Vector{DiDResult}
end

function DifferenceInDifferencesAnalysis(;
    panel_df::DataFrame,
    agent_df::DataFrame=DataFrame()
)
    DifferenceInDifferencesAnalysis(copy(panel_df), copy(agent_df), DiDResult[])
end

"""
Estimate DiD treatment effect (simplified two-period design).
"""
function estimate_did_effect!(
    analysis::DifferenceInDifferencesAnalysis,
    outcome_col::String;
    treatment_col::String="uses_ai",
    time_col::String="step",
    agent_id_col::String="agent_id"
)
    df = analysis.panel_df

    if !(outcome_col in names(df) && treatment_col in names(df) && time_col in names(df))
        println("   ⚠️ Missing required columns")
        return
    end

    # Identify treated and control groups
    # Treated: agents who ever use AI
    # Control: agents who never use AI

    treated_agents = unique(df[df[!, treatment_col] .== 1, agent_id_col])
    control_agents = setdiff(unique(df[!, agent_id_col]), treated_agents)

    if length(treated_agents) < 10 || length(control_agents) < 10
        println("   ⚠️ Insufficient sample sizes")
        return
    end

    # Get median time as cutoff
    median_time = median(df[!, time_col])

    # Pre-period means
    pre_treated = mean(filter(isfinite, Float64.(df[(df[!, agent_id_col] .∈ Ref(treated_agents)) .& (df[!, time_col] .< median_time), outcome_col])))
    pre_control = mean(filter(isfinite, Float64.(df[(df[!, agent_id_col] .∈ Ref(control_agents)) .& (df[!, time_col] .< median_time), outcome_col])))

    # Post-period means
    post_treated = mean(filter(isfinite, Float64.(df[(df[!, agent_id_col] .∈ Ref(treated_agents)) .& (df[!, time_col] .>= median_time), outcome_col])))
    post_control = mean(filter(isfinite, Float64.(df[(df[!, agent_id_col] .∈ Ref(control_agents)) .& (df[!, time_col] .>= median_time), outcome_col])))

    # DiD estimate
    att = (post_treated - pre_treated) - (post_control - pre_control)

    # Bootstrap for SE and CI
    rng = MersenneTwister(42)
    bootstrap_atts = Float64[]
    n_iter = get_bootstrap_iterations(1000)

    for _ in 1:n_iter
        boot_treated = treated_agents[rand(rng, 1:length(treated_agents), length(treated_agents))]
        boot_control = control_agents[rand(rng, 1:length(control_agents), length(control_agents))]

        pre_t = mean(filter(isfinite, Float64.(df[(df[!, agent_id_col] .∈ Ref(boot_treated)) .& (df[!, time_col] .< median_time), outcome_col])))
        pre_c = mean(filter(isfinite, Float64.(df[(df[!, agent_id_col] .∈ Ref(boot_control)) .& (df[!, time_col] .< median_time), outcome_col])))
        post_t = mean(filter(isfinite, Float64.(df[(df[!, agent_id_col] .∈ Ref(boot_treated)) .& (df[!, time_col] .>= median_time), outcome_col])))
        post_c = mean(filter(isfinite, Float64.(df[(df[!, agent_id_col] .∈ Ref(boot_control)) .& (df[!, time_col] .>= median_time), outcome_col])))

        push!(bootstrap_atts, (post_t - pre_t) - (post_c - pre_c))
    end

    se = std(bootstrap_atts)
    ci_lower = quantile(bootstrap_atts, 0.025)
    ci_upper = quantile(bootstrap_atts, 0.975)

    # p-value from z-test
    z = se > 0 ? att / se : 0.0
    p_value = 2 * (1 - cdf(Normal(), abs(z)))

    interpretation = if p_value < 0.05
        direction = att > 0 ? "increases" : "decreases"
        "AI adoption $direction $outcome_col by $(round(abs(att), digits=4)) (p=$(round(p_value, digits=4)))"
    else
        "No significant DiD effect of AI adoption on $outcome_col (ATT=$(round(att, digits=4)), p=$(round(p_value, digits=4)))"
    end

    # Count pre and post periods
    unique_times = unique(df[!, time_col])
    n_pre = count(t -> t < median_time, unique_times)
    n_post = count(t -> t >= median_time, unique_times)

    push!(analysis.results, DiDResult(
        outcome_col, att, se, ci_lower, ci_upper, p_value,
        length(treated_agents), length(control_agents), n_pre, n_post, interpretation
    ))

    println("   ✓ $interpretation")
end

"""
Generate DiD results table.
"""
function generate_did_table(analysis::DifferenceInDifferencesAnalysis)::DataFrame
    if isempty(analysis.results)
        return DataFrame()
    end

    rows = Dict{String,Any}[]
    for r in analysis.results
        push!(rows, Dict{String,Any}(
            "Outcome" => r.outcome,
            "ATT" => round(r.att, digits=4),
            "SE" => round(r.att_se, digits=4),
            "95_CI" => "[$(round(r.att_ci_lower, digits=4)), $(round(r.att_ci_upper, digits=4))]",
            "p_value" => round(r.p_value, digits=4),
            "N_Treated" => r.n_treated,
            "N_Control" => r.n_control,
            "N_Pre_Periods" => r.n_periods_pre,
            "N_Post_Periods" => r.n_periods_post,
            "Interpretation" => r.interpretation
        ))
    end

    return DataFrame(rows)
end

# ============================================================================
# REGRESSION DISCONTINUITY ANALYSIS (SIMPLIFIED)
# ============================================================================

"""
Results from Regression Discontinuity analysis.
"""
struct RDResult
    running_variable::String
    cutoff::Float64
    bandwidth::Float64
    treatment_effect::Float64
    effect_se::Float64
    effect_ci_lower::Float64
    effect_ci_upper::Float64
    p_value::Float64
    n_left::Int
    n_right::Int
    polynomial_order::Int
    interpretation::String
end

"""
Regression Discontinuity analysis for AI adoption effects.
"""
mutable struct RegressionDiscontinuityAnalysis
    agent_df::DataFrame
    decision_df::DataFrame
    results::Vector{RDResult}
end

function RegressionDiscontinuityAnalysis(;
    agent_df::DataFrame,
    decision_df::DataFrame=DataFrame()
)
    RegressionDiscontinuityAnalysis(copy(agent_df), copy(decision_df), RDResult[])
end

"""
Estimate RD treatment effect using local linear regression.
"""
function estimate_rd_effect!(
    analysis::RegressionDiscontinuityAnalysis,
    running_var::String,
    outcome_var::String,
    cutoff::Float64;
    bandwidth::Union{Float64,Nothing}=nothing,
    polynomial_order::Int=1
)
    df = isempty(analysis.decision_df) ? analysis.agent_df : analysis.decision_df

    if !(running_var in names(df) && outcome_var in names(df))
        println("   ⚠️ Missing columns: $running_var or $outcome_var")
        return
    end

    data = df[:, [running_var, outcome_var]]
    data = dropmissing(data)

    if nrow(data) < 50
        println("   ⚠️ Insufficient observations")
        return
    end

    # Center running variable at cutoff
    data[!, :X_centered] = data[!, running_var] .- cutoff
    data[!, :treated] = Int.(data[!, running_var] .>= cutoff)

    # Compute bandwidth if not specified (Silverman rule of thumb)
    if isnothing(bandwidth)
        std_x = std(data.X_centered)
        n = nrow(data)
        bandwidth = 1.06 * std_x * (n ^ (-1/5)) * 2
    end

    # Restrict to bandwidth window
    data_bw = data[abs.(data.X_centered) .<= bandwidth, :]

    n_left = sum(data_bw.X_centered .< 0)
    n_right = sum(data_bw.X_centered .>= 0)

    if n_left < 20 || n_right < 20
        println("   ⚠️ Insufficient observations in bandwidth window (left=$n_left, right=$n_right)")
        return
    end

    # Triangular kernel weights
    data_bw[!, :weight] = max.(1 .- abs.(data_bw.X_centered) ./ bandwidth, 0)

    # Local linear regression: Y = α + τ*D + β*X + γ*D*X + ε
    y = Float64.(data_bw[!, outcome_var])
    D = Float64.(data_bw.treated)
    X = Float64.(data_bw.X_centered)
    w = Float64.(data_bw.weight)

    # Design matrix: [1, D, X, D*X]
    design = hcat(ones(length(y)), D, X, D .* X)

    # Weighted least squares
    W = Diagonal(w)
    XtWX = design' * W * design
    XtWy = design' * W * y

    beta = XtWX \ XtWy

    # Treatment effect is coefficient on D (index 2)
    tau = beta[2]

    # Standard error
    residuals = y .- design * beta
    sigma2 = sum(w .* residuals.^2) / (sum(w) - length(beta))
    var_beta = sigma2 * inv(XtWX)
    se_tau = sqrt(var_beta[2, 2])

    # CI and p-value
    ci_lower = tau - 1.96 * se_tau
    ci_upper = tau + 1.96 * se_tau
    z_stat = tau / se_tau
    p_value = 2 * (1 - cdf(Normal(), abs(z_stat)))

    interpretation = if p_value < 0.05
        direction = tau > 0 ? "increases" : "decreases"
        "Crossing the $running_var threshold of $(round(cutoff, digits=2)) $direction $outcome_var by $(round(abs(tau), digits=4)) (p=$(round(p_value, digits=4)))"
    else
        "No significant discontinuity in $outcome_var at $running_var=$(round(cutoff, digits=2)) (τ=$(round(tau, digits=4)), p=$(round(p_value, digits=4)))"
    end

    push!(analysis.results, RDResult(
        running_var, cutoff, bandwidth, tau, se_tau,
        ci_lower, ci_upper, p_value, n_left, n_right,
        polynomial_order, interpretation
    ))

    println("   ✓ RD estimate: τ = $(round(tau, digits=4)) (SE = $(round(se_tau, digits=4)))")
    println("   ✓ $interpretation")
end

"""
Generate RD results table.
"""
function generate_rd_table(analysis::RegressionDiscontinuityAnalysis)::DataFrame
    if isempty(analysis.results)
        return DataFrame()
    end

    rows = Dict{String,Any}[]
    for r in analysis.results
        push!(rows, Dict{String,Any}(
            "Running_Variable" => r.running_variable,
            "Cutoff" => round(r.cutoff, digits=2),
            "Treatment_Effect" => round(r.treatment_effect, digits=4),
            "Std_Error" => round(r.effect_se, digits=4),
            "95_CI" => "[$(round(r.effect_ci_lower, digits=4)), $(round(r.effect_ci_upper, digits=4))]",
            "p_value" => round(r.p_value, digits=4),
            "Bandwidth" => round(r.bandwidth, digits=4),
            "N_left" => r.n_left,
            "N_right" => r.n_right,
            "Polynomial" => r.polynomial_order
        ))
    end

    return DataFrame(rows)
end

# ============================================================================
# CONVENIENCE FUNCTION FOR FULL STATISTICAL ANALYSIS
# ============================================================================

"""
Run complete statistical analysis suite on simulation results.

Returns named tuple with all analysis results.
"""
function run_statistical_analysis(;
    agent_df::DataFrame,
    decision_df::DataFrame=DataFrame(),
    matured_df::DataFrame=DataFrame(),
    uncertainty_detail_df::DataFrame=DataFrame(),
    is_fixed_tier::Bool=false
)
    println("\n" * "="^70)
    println("COMPREHENSIVE STATISTICAL ANALYSIS SUITE")
    println("="^70)

    # Run rigorous hypothesis testing
    rig_analysis = RigorousStatisticalAnalysis(
        agent_df=agent_df,
        decision_df=decision_df,
        matured_df=matured_df,
        uncertainty_detail_df=uncertainty_detail_df
    )
    hypothesis_tests = run_all_analyses!(rig_analysis)
    descriptive_stats = generate_descriptive_statistics(rig_analysis)

    # Run causal identification analysis
    causal_analysis = CausalIdentificationAnalysis(
        agent_df=agent_df,
        matured_df=matured_df,
        decision_df=decision_df,
        uncertainty_detail_df=uncertainty_detail_df,
        is_fixed_tier=is_fixed_tier
    )
    run_all_estimates!(causal_analysis)
    causal_effects = generate_causal_effects_table(causal_analysis)

    # Survival analysis
    survival_analysis = CoxSurvivalAnalysis(agent_df)
    log_rank = log_rank_tests(survival_analysis)

    println("\n" * "="^70)
    println("ANALYSIS COMPLETE")
    println("="^70)

    return (
        hypothesis_tests=hypothesis_tests,
        descriptive_stats=descriptive_stats,
        causal_effects=causal_effects,
        log_rank_tests=log_rank
    )
end

# ============================================================================
# MIXED EFFECTS ANALYSIS (SIMPLIFIED WITHOUT EXTERNAL ML PACKAGES)
# ============================================================================

"""
Mixed-effects (multilevel) models for nested ABM data.

Agent-based simulation data has a natural hierarchical structure:
- Level 1: Decisions/observations within agents
- Level 2: Agents within simulation runs
- Level 3: Simulation runs (for multi-run designs)

This simplified implementation provides ICC calculations and variance
decomposition without requiring external mixed-model packages.
For full mixed-effects modeling, consider using MixedModels.jl.

Theoretical Justification
-------------------------
Standard OLS assumes independence of observations, which is violated
when decisions are nested within agents and agents within runs. This
implementation partitions variance into within-group and between-group
components, providing the Intraclass Correlation Coefficient (ICC).
"""
mutable struct MixedEffectsAnalysis
    agent_df::DataFrame
    decision_df::DataFrame
    matured_df::DataFrame
    results::Vector{MixedEffectsResult}
end

function MixedEffectsAnalysis(;
    agent_df::DataFrame,
    decision_df::DataFrame=DataFrame(),
    matured_df::DataFrame=DataFrame()
)
    MixedEffectsAnalysis(agent_df, decision_df, matured_df, MixedEffectsResult[])
end

"""
Run all mixed-effects models.
"""
function run_all_models!(analysis::MixedEffectsAnalysis)::Vector{MixedEffectsResult}
    println("\n" * "="^70)
    println("MIXED-EFFECTS ANALYSIS FOR NESTED DATA STRUCTURE")
    println("="^70)

    # Model 1: AI effects on capital growth
    _fit_capital_growth_model!(analysis)

    # Model 2: AI effects on decision outcomes
    _fit_decision_outcome_model!(analysis)

    # Model 3: AI effects on uncertainty perception
    _fit_uncertainty_perception_model!(analysis)

    # Model 4: AI effects on investment returns
    _fit_investment_returns_model!(analysis)

    return analysis.results
end

"""
Calculate Intraclass Correlation Coefficient (ICC).

ICC = σ²_between / (σ²_between + σ²_within)
"""
function calculate_icc(data::AbstractVector{<:Real}, groups::AbstractVector)::Tuple{Float64,Float64,Float64}
    clean_mask = .!isnan.(data) .& .!ismissing.(data)
    clean_data = data[clean_mask]
    clean_groups = groups[clean_mask]

    if length(clean_data) < 10
        return (NaN, NaN, NaN)
    end

    unique_groups = unique(clean_groups)
    k = length(unique_groups)

    if k < 2
        return (0.0, var(clean_data), 0.0)
    end

    # Calculate group means
    group_means = Dict{eltype(unique_groups),Float64}()
    group_sizes = Dict{eltype(unique_groups),Int}()

    for g in unique_groups
        mask = clean_groups .== g
        group_data = clean_data[mask]
        group_means[g] = mean(group_data)
        group_sizes[g] = length(group_data)
    end

    grand_mean = mean(clean_data)
    n_total = length(clean_data)

    # Between-group sum of squares
    ss_between = sum(group_sizes[g] * (group_means[g] - grand_mean)^2 for g in unique_groups)

    # Within-group sum of squares
    ss_within = 0.0
    for g in unique_groups
        mask = clean_groups .== g
        group_data = clean_data[mask]
        ss_within += sum((x - group_means[g])^2 for x in group_data)
    end

    # Mean squares
    df_between = k - 1
    df_within = n_total - k

    if df_within <= 0 || df_between <= 0
        return (NaN, NaN, NaN)
    end

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # Estimate variance components
    n_avg = n_total / k  # Average group size
    var_between = max(0.0, (ms_between - ms_within) / n_avg)
    var_within = ms_within

    # ICC
    total_var = var_between + var_within
    icc = total_var > 0 ? var_between / total_var : 0.0

    return (icc, var_between, var_within)
end

"""
Fit simplified mixed model using variance decomposition.

Returns coefficients estimated via OLS with clustered variance estimation.
"""
function fit_simplified_mixed_model(
    y::AbstractVector{<:Real},
    X::Matrix{<:Real},
    groups::AbstractVector;
    var_names::Vector{String}=String[]
)::Dict{String,Any}
    # Filter valid observations
    valid_mask = .!isnan.(y) .& vec(all(.!isnan.(X), dims=2))
    y_clean = y[valid_mask]
    X_clean = X[valid_mask, :]
    groups_clean = groups[valid_mask]

    n = length(y_clean)
    p = size(X_clean, 2)

    if n < p + 10
        return Dict{String,Any}("converged" => false, "error" => "Insufficient observations")
    end

    # OLS estimation
    XtX = X_clean' * X_clean
    Xty = X_clean' * y_clean

    # Check for singularity
    if det(XtX) ≈ 0
        return Dict{String,Any}("converged" => false, "error" => "Singular matrix")
    end

    beta = XtX \ Xty
    residuals = y_clean .- X_clean * beta

    # Standard OLS variance
    sigma2 = sum(residuals.^2) / (n - p)

    # Clustered standard errors (robust to within-group correlation)
    unique_groups = unique(groups_clean)
    n_groups = length(unique_groups)

    # Compute meat matrix for cluster-robust variance
    meat = zeros(p, p)
    for g in unique_groups
        mask = groups_clean .== g
        X_g = X_clean[mask, :]
        e_g = residuals[mask]
        u_g = X_g' * e_g
        meat .+= u_g * u_g'
    end

    # Sandwich estimator
    bread = inv(XtX)
    cluster_var = bread * meat * bread

    # Scale factor for small sample
    scale = n_groups / (n_groups - 1) * (n - 1) / (n - p)
    cluster_var .*= scale

    # Standard errors
    se_clustered = sqrt.(diag(cluster_var))

    # Calculate ICC
    icc, var_between, var_within = calculate_icc(y_clean, groups_clean)

    # Build results
    coefficients = Dict{String,Dict{String,Float64}}()
    for (i, name) in enumerate(var_names)
        if i <= length(beta)
            z = se_clustered[i] > 0 ? beta[i] / se_clustered[i] : 0.0
            p_val = 2 * (1 - cdf(Normal(), abs(z)))
            coefficients[name] = Dict{String,Float64}(
                "coef" => beta[i],
                "se" => se_clustered[i],
                "z" => z,
                "p" => p_val,
                "ci_lower" => beta[i] - 1.96 * se_clustered[i],
                "ci_upper" => beta[i] + 1.96 * se_clustered[i]
            )
        end
    end

    return Dict{String,Any}(
        "converged" => true,
        "coefficients" => coefficients,
        "icc" => icc,
        "var_between" => var_between,
        "var_within" => var_within,
        "n_observations" => n,
        "n_groups" => n_groups,
        "sigma2" => sigma2
    )
end

"""
Model 1: AI Tier → Capital Growth with random intercepts for runs.
"""
function _fit_capital_growth_model!(analysis::MixedEffectsAnalysis)
    println("\n📊 Model 1: AI Tier → Capital Growth (agents nested in runs)")

    if isempty(analysis.agent_df) || !("capital_growth" in names(analysis.agent_df))
        println("   ⚠️ Insufficient data")
        return
    end

    df = analysis.agent_df

    # Check for run_id column
    if !("run_id" in names(df))
        println("   ⚠️ No run_id column; cannot fit mixed model")
        return
    end

    # Get AI level column
    ai_col = "primary_ai_canonical" in names(df) ? "primary_ai_canonical" :
             "ai_level" in names(df) ? "ai_level" : nothing

    if isnothing(ai_col)
        println("   ⚠️ No AI level column found")
        return
    end

    # Create dummy variables
    ai_levels_vec = lowercase.(string.(coalesce.(df[!, ai_col], "none")))
    valid_mask = ai_levels_vec .∈ Ref(["none", "basic", "advanced", "premium"])
    df_clean = df[valid_mask, :]
    ai_levels_clean = ai_levels_vec[valid_mask]

    if nrow(df_clean) < 50
        println("   ⚠️ Insufficient observations (n < 50)")
        return
    end

    # Create design matrix
    y = Float64.(df_clean.capital_growth)
    intercept = ones(length(y))
    ai_basic = Float64.(ai_levels_clean .== "basic")
    ai_advanced = Float64.(ai_levels_clean .== "advanced")
    ai_premium = Float64.(ai_levels_clean .== "premium")

    X = hcat(intercept, ai_basic, ai_advanced, ai_premium)
    groups = df_clean.run_id

    # Fit model
    result = fit_simplified_mixed_model(y, X, groups,
        var_names=["Intercept", "ai_basic", "ai_advanced", "ai_premium"])

    if !result["converged"]
        println("   ⚠️ Model fitting failed: $(get(result, "error", "unknown"))")
        return
    end

    # Build interpretation
    coeffs = result["coefficients"]
    significant_effects = String[]
    for (var, stats) in coeffs
        if var != "Intercept" && stats["p"] < 0.05
            direction = stats["coef"] > 0 ? "higher" : "lower"
            tier = replace(var, "ai_" => "") |> titlecase
            push!(significant_effects, "$tier AI has $direction capital growth")
        end
    end

    icc = result["icc"]
    interp = if !isempty(significant_effects)
        join(significant_effects, "; ") * ". ICC = $(round(icc, digits=3)) ($(round(icc*100, digits=1))% of variance between runs)."
    else
        "No significant AI tier effects on capital growth. ICC = $(round(icc, digits=3))."
    end

    model_result = MixedEffectsResult(
        "Capital Growth ~ AI Tier (Random Intercept: Run)",
        "capital_growth",
        coeffs,
        Dict("run_intercept" => result["var_between"], "residual" => result["var_within"]),
        Dict("ICC" => icc),
        result["n_observations"],
        Dict("runs" => result["n_groups"]),
        true,
        interp
    )
    push!(analysis.results, model_result)

    println("   ✓ Fitted model: n=$(result["n_observations"]), runs=$(result["n_groups"]), ICC=$(round(icc, digits=3))")
    for (var, stats) in coeffs
        sig = stats["p"] < 0.001 ? "***" : stats["p"] < 0.01 ? "**" : stats["p"] < 0.05 ? "*" : ""
        println("      $var: β=$(round(stats["coef"], digits=4)), SE=$(round(stats["se"], digits=4)), p=$(round(stats["p"], digits=4))$sig")
    end
end

"""
Model 2: AI Tier → Decision Success (decisions nested in agents).
"""
function _fit_decision_outcome_model!(analysis::MixedEffectsAnalysis)
    println("\n📊 Model 2: AI Tier → Decision Success (decisions nested in agents)")

    if isempty(analysis.decision_df) || !("success" in names(analysis.decision_df))
        println("   ⚠️ Insufficient decision data")
        return
    end

    df = analysis.decision_df

    if !("agent_id" in names(df))
        println("   ⚠️ No agent_id column")
        return
    end

    ai_col = "ai_level_used" in names(df) ? "ai_level_used" :
             "ai_level" in names(df) ? "ai_level" : nothing

    if isnothing(ai_col)
        println("   ⚠️ No AI level column found")
        return
    end

    # Prepare data
    ai_levels_vec = lowercase.(string.(coalesce.(df[!, ai_col], "none")))
    valid_mask = ai_levels_vec .∈ Ref(["none", "basic", "advanced", "premium"])
    df_clean = df[valid_mask, :]
    ai_levels_clean = ai_levels_vec[valid_mask]

    if nrow(df_clean) < 100
        println("   ⚠️ Insufficient observations")
        return
    end

    # Create design matrix
    success_col = df_clean.success
    y = Float64.([ismissing(x) ? NaN : Float64(x) for x in success_col])
    intercept = ones(length(y))
    ai_basic = Float64.(ai_levels_clean .== "basic")
    ai_advanced = Float64.(ai_levels_clean .== "advanced")
    ai_premium = Float64.(ai_levels_clean .== "premium")

    X = hcat(intercept, ai_basic, ai_advanced, ai_premium)
    groups = df_clean.agent_id

    # Fit model
    result = fit_simplified_mixed_model(y, X, groups,
        var_names=["Intercept", "ai_basic", "ai_advanced", "ai_premium"])

    if !result["converged"]
        println("   ⚠️ Model fitting failed")
        return
    end

    icc = result["icc"]
    interp = "ICC = $(round(icc, digits=3)) ($(round(icc*100, digits=1))% of success variance is between agents)"

    model_result = MixedEffectsResult(
        "Decision Success ~ AI Tier (Random Intercept: Agent)",
        "success",
        result["coefficients"],
        Dict("agent_intercept" => result["var_between"], "residual" => result["var_within"]),
        Dict("ICC" => icc),
        result["n_observations"],
        Dict("agents" => result["n_groups"]),
        true,
        interp
    )
    push!(analysis.results, model_result)

    println("   ✓ Fitted model: n=$(result["n_observations"]), agents=$(result["n_groups"]), ICC=$(round(icc, digits=3))")
end

"""
Model 3: AI Tier → Actor Ignorance (testing paradox hypothesis).
"""
function _fit_uncertainty_perception_model!(analysis::MixedEffectsAnalysis)
    println("\n📊 Model 3: AI Tier → Actor Ignorance (testing paradox hypothesis)")

    if isempty(analysis.decision_df)
        println("   ⚠️ Insufficient data")
        return
    end

    df = analysis.decision_df

    # Find ignorance column
    ignorance_col = nothing
    for col in ["perc_actor_ignorance", "actor_ignorance_level", "ignorance_level"]
        if col in names(df)
            ignorance_col = col
            break
        end
    end

    if isnothing(ignorance_col)
        println("   ⚠️ No actor ignorance column found")
        return
    end

    if !("agent_id" in names(df))
        println("   ⚠️ No agent_id column")
        return
    end

    ai_col = "ai_level_used" in names(df) ? "ai_level_used" :
             "ai_level" in names(df) ? "ai_level" : nothing

    if isnothing(ai_col)
        return
    end

    # Prepare data
    ai_levels_vec = lowercase.(string.(coalesce.(df[!, ai_col], "none")))
    valid_mask = ai_levels_vec .∈ Ref(["none", "basic", "advanced", "premium"])
    df_clean = df[valid_mask, :]
    ai_levels_clean = ai_levels_vec[valid_mask]

    if nrow(df_clean) < 100
        println("   ⚠️ Insufficient observations")
        return
    end

    # Create design matrix
    ignorance_data = df_clean[!, ignorance_col]
    y = Float64.([ismissing(x) || isnan(x) ? NaN : Float64(x) for x in ignorance_data])
    intercept = ones(length(y))
    ai_basic = Float64.(ai_levels_clean .== "basic")
    ai_advanced = Float64.(ai_levels_clean .== "advanced")
    ai_premium = Float64.(ai_levels_clean .== "premium")

    X = hcat(intercept, ai_basic, ai_advanced, ai_premium)
    groups = df_clean.agent_id

    # Fit model
    result = fit_simplified_mixed_model(y, X, groups,
        var_names=["Intercept", "ai_basic", "ai_advanced", "ai_premium"])

    if !result["converged"]
        println("   ⚠️ Model fitting failed")
        return
    end

    # Check if AI reduces ignorance (negative coefficients)
    coeffs = result["coefficients"]
    reduces_ignorance = all(
        get(get(coeffs, "ai_$tier", Dict()), "coef", 0.0) < 0
        for tier in ["basic", "advanced", "premium"]
        if haskey(coeffs, "ai_$tier")
    )

    icc = result["icc"]
    interp = "AI $(reduces_ignorance ? "reduces" : "does not consistently reduce") " *
             "perceived actor ignorance. This $(reduces_ignorance ? "supports" : "does not support") " *
             "the paradox hypothesis (Townsend et al., 2025). ICC = $(round(icc, digits=3))."

    model_result = MixedEffectsResult(
        "Actor Ignorance ~ AI Tier (Random Intercept: Agent)",
        ignorance_col,
        coeffs,
        Dict("agent_intercept" => result["var_between"], "residual" => result["var_within"]),
        Dict("ICC" => icc),
        result["n_observations"],
        Dict("agents" => result["n_groups"]),
        true,
        interp
    )
    push!(analysis.results, model_result)

    println("   ✓ $interp")
end

"""
Model 4: AI Tier → Investment Returns for matured investments.
"""
function _fit_investment_returns_model!(analysis::MixedEffectsAnalysis)
    println("\n📊 Model 4: AI Tier → Investment Returns")

    if isempty(analysis.matured_df)
        println("   ⚠️ No matured investment data")
        return
    end

    df = analysis.matured_df

    # Find return column
    return_col = nothing
    for col in ["realized_roi", "return_multiple", "roi"]
        if col in names(df)
            return_col = col
            break
        end
    end

    if isnothing(return_col)
        println("   ⚠️ No return column found")
        return
    end

    # Find AI and grouping columns
    ai_col = nothing
    for col in ["ai_level_used", "ai_level", "ai_tier"]
        if col in names(df)
            ai_col = col
            break
        end
    end

    group_col = nothing
    for col in ["agent_id", "run_id"]
        if col in names(df)
            group_col = col
            break
        end
    end

    if isnothing(ai_col) || isnothing(group_col)
        println("   ⚠️ Missing required columns")
        return
    end

    # Prepare data
    ai_levels_vec = lowercase.(string.(coalesce.(df[!, ai_col], "none")))
    valid_mask = ai_levels_vec .∈ Ref(["none", "basic", "advanced", "premium"])
    df_clean = df[valid_mask, :]
    ai_levels_clean = ai_levels_vec[valid_mask]

    if nrow(df_clean) < 50
        println("   ⚠️ Insufficient observations")
        return
    end

    # Create design matrix
    return_data = df_clean[!, return_col]
    y = Float64.([ismissing(x) || !isfinite(x) ? NaN : Float64(x) for x in return_data])
    intercept = ones(length(y))
    ai_basic = Float64.(ai_levels_clean .== "basic")
    ai_advanced = Float64.(ai_levels_clean .== "advanced")
    ai_premium = Float64.(ai_levels_clean .== "premium")

    X = hcat(intercept, ai_basic, ai_advanced, ai_premium)
    groups = df_clean[!, group_col]

    # Fit model
    result = fit_simplified_mixed_model(y, X, groups,
        var_names=["Intercept", "ai_basic", "ai_advanced", "ai_premium"])

    if !result["converged"]
        println("   ⚠️ Model fitting failed")
        return
    end

    icc = result["icc"]

    model_result = MixedEffectsResult(
        "Investment ROI ~ AI Tier (Random Intercept: $(titlecase(string(group_col))))",
        return_col,
        result["coefficients"],
        Dict("group_intercept" => result["var_between"], "residual" => result["var_within"]),
        Dict("ICC" => icc),
        result["n_observations"],
        Dict(string(group_col) => result["n_groups"]),
        true,
        "Model fitted successfully. ICC = $(round(icc, digits=3))."
    )
    push!(analysis.results, model_result)

    println("   ✓ Fitted model: n=$(result["n_observations"]), $(group_col)=$(result["n_groups"]), ICC=$(round(icc, digits=3))")
end

"""
Generate results table from MixedEffectsAnalysis.
"""
function generate_mixed_effects_table(analysis::MixedEffectsAnalysis)::DataFrame
    if isempty(analysis.results)
        return DataFrame()
    end

    rows = Dict{String,Any}[]
    for r in analysis.results
        for (var, stats) in r.fixed_effects
            push!(rows, Dict{String,Any}(
                "Model" => r.model_name,
                "DV" => r.dependent_variable,
                "Variable" => var,
                "Coefficient" => round(stats["coef"], digits=4),
                "Std_Error" => round(stats["se"], digits=4),
                "z_value" => round(stats["z"], digits=3),
                "p_value" => stats["p"] >= 0.0001 ? round(stats["p"], digits=4) : "<0.0001",
                "CI_95_Lower" => round(stats["ci_lower"], digits=4),
                "CI_95_Upper" => round(stats["ci_upper"], digits=4),
                "ICC" => round(get(r.model_fit, "ICC", NaN), digits=3),
                "N" => r.n_observations
            ))
        end
    end

    return DataFrame(rows)
end

# ============================================================================
# PROPENSITY SCORE ANALYSIS
# ============================================================================

"""
Results from propensity score analysis.
"""
struct PropensityScoreResult
    treatment::String
    outcome::String
    method::String  # "matching", "weighting", "stratification"
    ate::Float64
    ate_se::Float64
    ate_ci_lower::Float64
    ate_ci_upper::Float64
    att::Float64
    att_se::Float64
    n_treated::Int
    n_control::Int
    n_matched::Int
    balance_improvement::Float64
    interpretation::String
end

"""
Propensity Score Analysis for observational causal inference.

Uses propensity score methods to estimate causal effects when
treatment assignment is non-random (endogenous AI tier selection).
"""
mutable struct PropensityScoreAnalysis
    agent_df::DataFrame
    decision_df::DataFrame
    treatment_col::String
    covariates::Vector{String}
    results::Vector{PropensityScoreResult}
end

function PropensityScoreAnalysis(;
    agent_df::DataFrame,
    decision_df::DataFrame=DataFrame(),
    treatment_col::String="ai_level",
    covariates::Vector{String}=String[]
)
    PropensityScoreAnalysis(agent_df, decision_df, treatment_col, covariates, PropensityScoreResult[])
end

"""
Estimate propensity scores using logistic regression-style approach.
"""
function estimate_propensity_scores(
    treatment::AbstractVector{<:Real},
    covariates::Matrix{<:Real}
)::Vector{Float64}
    n = length(treatment)
    p = size(covariates, 2)

    if n < p + 10
        return fill(0.5, n)
    end

    # Add intercept
    X = hcat(ones(n), covariates)

    # Use iteratively reweighted least squares for logistic regression
    beta = zeros(p + 1)
    max_iter = 50

    for iter in 1:max_iter
        # Predicted probabilities
        eta = X * beta
        # Clip to prevent overflow
        eta = clamp.(eta, -10, 10)
        prob = 1.0 ./ (1.0 .+ exp.(-eta))
        prob = clamp.(prob, 0.001, 0.999)

        # Weights
        W = Diagonal(prob .* (1 .- prob))

        # Score and Hessian
        score = X' * (treatment .- prob)
        hessian = -X' * W * X

        # Check for singularity
        if abs(det(hessian)) < 1e-10
            break
        end

        # Newton-Raphson update
        delta = -hessian \ score

        # Update with damping
        beta_new = beta + 0.5 * delta

        # Check convergence
        if maximum(abs.(delta)) < 1e-6
            beta = beta_new
            break
        end
        beta = beta_new
    end

    # Final propensity scores
    eta = X * beta
    eta = clamp.(eta, -10, 10)
    prob = 1.0 ./ (1.0 .+ exp.(-eta))

    return clamp.(prob, 0.001, 0.999)
end

"""
Perform nearest-neighbor propensity score matching.
"""
function propensity_score_matching(
    ps_treated::Vector{Float64},
    ps_control::Vector{Float64};
    caliper::Float64=0.1
)::Vector{Tuple{Int,Int}}
    matches = Tuple{Int,Int}[]
    used_controls = Set{Int}()

    for (i, ps_t) in enumerate(ps_treated)
        best_match = -1
        best_dist = caliper

        for (j, ps_c) in enumerate(ps_control)
            if j ∈ used_controls
                continue
            end

            dist = abs(ps_t - ps_c)
            if dist < best_dist
                best_dist = dist
                best_match = j
            end
        end

        if best_match > 0
            push!(matches, (i, best_match))
            push!(used_controls, best_match)
        end
    end

    return matches
end

"""
Calculate standardized mean difference for balance assessment.
"""
function standardized_mean_difference(
    treated::AbstractVector{<:Real},
    control::AbstractVector{<:Real}
)::Float64
    m1, m2 = mean(treated), mean(control)
    s1, s2 = std(treated), std(control)

    pooled_sd = sqrt((s1^2 + s2^2) / 2)

    return pooled_sd > 0 ? abs(m1 - m2) / pooled_sd : 0.0
end

"""
Run propensity score analysis for AI tier effects.
"""
function run_propensity_analysis!(
    analysis::PropensityScoreAnalysis;
    outcome_col::String="survived"
)
    println("\n" * "="^70)
    println("PROPENSITY SCORE ANALYSIS")
    println("="^70)

    df = analysis.agent_df

    if isempty(df)
        println("   ⚠️ No agent data")
        return
    end

    # Find AI level column
    ai_col = "primary_ai_canonical" in names(df) ? "primary_ai_canonical" :
             "ai_level" in names(df) ? "ai_level" : nothing

    if isnothing(ai_col) || !(outcome_col in names(df))
        println("   ⚠️ Missing required columns")
        return
    end

    # Create treatment indicator (AI users vs non-users)
    ai_levels = lowercase.(string.(coalesce.(df[!, ai_col], "none")))
    treatment = Int.(ai_levels .!= "none")

    # Get outcome
    outcome_data = df[!, outcome_col]
    outcome = Float64.([ismissing(x) ? NaN : Float64(x) for x in outcome_data])

    # Find covariates for propensity model
    possible_covariates = ["initial_capital", "risk_tolerance", "adaptability",
                          "sector_experience", "innovation_propensity"]
    available_covariates = [c for c in possible_covariates if c in names(df)]

    if isempty(available_covariates)
        println("   ⚠️ No covariates available for propensity model")
        # Use constant propensity
        ps = fill(mean(treatment), length(treatment))
    else
        # Build covariate matrix
        cov_matrix = zeros(nrow(df), length(available_covariates))
        for (i, col) in enumerate(available_covariates)
            col_data = df[!, col]
            cov_matrix[:, i] = Float64.([ismissing(x) || !isfinite(x) ? 0.0 : Float64(x) for x in col_data])
        end

        # Estimate propensity scores
        ps = estimate_propensity_scores(Float64.(treatment), cov_matrix)
        println("   ✓ Estimated propensity scores using: $(join(available_covariates, ", "))")
    end

    # Separate treated and control
    treated_idx = findall(treatment .== 1)
    control_idx = findall(treatment .== 0)

    ps_treated = ps[treated_idx]
    ps_control = ps[control_idx]
    outcome_treated = outcome[treated_idx]
    outcome_control = outcome[control_idx]

    # Perform matching
    matches = propensity_score_matching(ps_treated, ps_control; caliper=0.1)
    n_matched = length(matches)

    if n_matched < 10
        println("   ⚠️ Insufficient matches (n=$n_matched)")
        return
    end

    # Calculate ATT (Average Treatment Effect on Treated)
    matched_treated_outcomes = [outcome_treated[m[1]] for m in matches if !isnan(outcome_treated[m[1]])]
    matched_control_outcomes = [outcome_control[m[2]] for m in matches if !isnan(outcome_control[m[2]])]

    att = mean(matched_treated_outcomes) - mean(matched_control_outcomes)

    # Bootstrap for SE
    rng = MersenneTwister(42)
    n_boot = get_bootstrap_iterations(1000)
    boot_atts = Float64[]

    for _ in 1:n_boot
        boot_matches = matches[rand(rng, 1:n_matched, n_matched)]
        boot_treated = [outcome_treated[m[1]] for m in boot_matches if !isnan(outcome_treated[m[1]])]
        boot_control = [outcome_control[m[2]] for m in boot_matches if !isnan(outcome_control[m[2]])]

        if !isempty(boot_treated) && !isempty(boot_control)
            push!(boot_atts, mean(boot_treated) - mean(boot_control))
        end
    end

    att_se = std(boot_atts)
    att_ci_lower = quantile(boot_atts, 0.025)
    att_ci_upper = quantile(boot_atts, 0.975)

    # Calculate balance improvement
    if !isempty(available_covariates)
        pre_smd = 0.0
        post_smd = 0.0

        for (i, col) in enumerate(available_covariates)
            col_data = df[!, col]
            cov_vals = Float64.([ismissing(x) || !isfinite(x) ? 0.0 : Float64(x) for x in col_data])

            pre_smd += standardized_mean_difference(cov_vals[treated_idx], cov_vals[control_idx])

            matched_treated_cov = [cov_vals[treated_idx[m[1]]] for m in matches]
            matched_control_cov = [cov_vals[control_idx[m[2]]] for m in matches]
            post_smd += standardized_mean_difference(matched_treated_cov, matched_control_cov)
        end

        pre_smd /= length(available_covariates)
        post_smd /= length(available_covariates)
        balance_improvement = pre_smd > 0 ? (pre_smd - post_smd) / pre_smd : 0.0
    else
        balance_improvement = 0.0
    end

    # Interpretation
    p_value = att_se > 0 ? 2 * (1 - cdf(Normal(), abs(att / att_se))) : 1.0
    sig = p_value < 0.05 ? "significant" : "non-significant"
    direction = att > 0 ? "higher" : "lower"

    interp = "Propensity score matching shows AI users have $direction $outcome_col " *
             "(ATT=$(round(att, digits=4)), SE=$(round(att_se, digits=4)), p=$(round(p_value, digits=4)), $sig). " *
             "$n_matched matches made. Balance improvement: $(round(balance_improvement*100, digits=1))%."

    push!(analysis.results, PropensityScoreResult(
        "AI_any_vs_none", outcome_col, "matching",
        att, att_se, att_ci_lower, att_ci_upper,  # ATE ≈ ATT for this design
        att, att_se,
        length(treated_idx), length(control_idx), n_matched,
        balance_improvement, interp
    ))

    println("   ✓ $interp")
end

"""
Generate propensity score results table.
"""
function generate_propensity_table(analysis::PropensityScoreAnalysis)::DataFrame
    if isempty(analysis.results)
        return DataFrame()
    end

    rows = Dict{String,Any}[]
    for r in analysis.results
        push!(rows, Dict{String,Any}(
            "Treatment" => r.treatment,
            "Outcome" => r.outcome,
            "Method" => r.method,
            "ATT" => round(r.att, digits=4),
            "ATT_SE" => round(r.att_se, digits=4),
            "ATT_95_CI" => "[$(round(r.ate_ci_lower, digits=4)), $(round(r.ate_ci_upper, digits=4))]",
            "N_Treated" => r.n_treated,
            "N_Control" => r.n_control,
            "N_Matched" => r.n_matched,
            "Balance_Improvement" => "$(round(r.balance_improvement*100, digits=1))%",
            "Interpretation" => r.interpretation
        ))
    end

    return DataFrame(rows)
end

# ============================================================================
# COMPLETE CAUSAL ANALYSIS PIPELINE
# ============================================================================

"""
Run complete causal analysis pipeline.

Combines all causal inference methods:
1. Fixed-tier causal identification
2. Propensity score analysis
3. Difference-in-differences (if panel data available)
4. Regression discontinuity (if threshold data available)
5. Mixed effects models
"""
function run_complete_causal_analysis(;
    agent_df::DataFrame,
    decision_df::DataFrame=DataFrame(),
    matured_df::DataFrame=DataFrame(),
    panel_df::DataFrame=DataFrame(),
    uncertainty_detail_df::DataFrame=DataFrame(),
    is_fixed_tier::Bool=false
)
    println("\n" * "="^80)
    println("COMPREHENSIVE CAUSAL ANALYSIS SUITE")
    println("="^80)

    results = Dict{String,Any}()

    # 1. Basic causal identification
    println("\n[1/5] Causal Identification Analysis...")
    causal_analysis = CausalIdentificationAnalysis(
        agent_df=agent_df,
        matured_df=matured_df,
        decision_df=decision_df,
        uncertainty_detail_df=uncertainty_detail_df,
        is_fixed_tier=is_fixed_tier
    )
    run_all_estimates!(causal_analysis)
    results["causal_effects"] = generate_causal_effects_table(causal_analysis)

    # 2. Mixed effects models
    println("\n[2/5] Mixed Effects Models...")
    mixed_analysis = MixedEffectsAnalysis(
        agent_df=agent_df,
        decision_df=decision_df,
        matured_df=matured_df
    )
    run_all_models!(mixed_analysis)
    results["mixed_effects"] = generate_mixed_effects_table(mixed_analysis)

    # 3. Propensity score analysis (for observational data)
    if !is_fixed_tier
        println("\n[3/5] Propensity Score Analysis...")
        ps_analysis = PropensityScoreAnalysis(
            agent_df=agent_df,
            decision_df=decision_df
        )
        run_propensity_analysis!(ps_analysis; outcome_col="survived")
        results["propensity_score"] = generate_propensity_table(ps_analysis)
    else
        println("\n[3/5] Propensity Score Analysis... (skipped - fixed tier design)")
        results["propensity_score"] = DataFrame()
    end

    # 4. Difference-in-differences (if panel data available)
    if !isempty(panel_df)
        println("\n[4/5] Difference-in-Differences Analysis...")
        did_analysis = DifferenceInDifferencesAnalysis(
            panel_df=panel_df,
            agent_df=agent_df
        )
        # Try to estimate DiD for capital if available
        if "capital" in names(panel_df) && "uses_ai" in names(panel_df)
            estimate_did_effect!(did_analysis, "capital"; treatment_col="uses_ai")
        end
        results["did"] = generate_did_table(did_analysis)
    else
        println("\n[4/5] Difference-in-Differences Analysis... (skipped - no panel data)")
        results["did"] = DataFrame()
    end

    # 5. Survival analysis
    println("\n[5/5] Survival Analysis...")
    survival_analysis = CoxSurvivalAnalysis(agent_df)
    results["log_rank"] = log_rank_tests(survival_analysis)
    results["km_curves"] = kaplan_meier_by_tier(survival_analysis)

    println("\n" * "="^80)
    println("CAUSAL ANALYSIS COMPLETE")
    println("="^80)

    return results
end
