"""
Causal Inference Methods for GlimpseABM.jl

This module provides statistical methods for causal inference in the ABM:
- Effect size measures (Cohen's d, Cliff's delta, Glass's delta)
- ANOVA and Kruskal-Wallis tests
- Bootstrap confidence intervals
- Mann-Whitney U tests
- Survival analysis utilities

Port of: glimpse_abm/causal/
"""

module Causal

using Statistics
using Random
using DataFrames
using Distributions

export cohens_d, cliffs_delta, glass_delta
export anova_oneway, kruskal_wallis
export bootstrap_ci, permutation_test
export mann_whitney_u, survival_analysis
export EffectSizeResult, ANOVAResult, SurvivalResult

# Advanced causal methods
export CoxRegressionResult, PropensityScoreResult, DiDResult, RDResult
export kaplan_meier_curves, log_rank_test
export estimate_propensity_scores, propensity_score_matching, inverse_probability_weighting
export difference_in_differences, event_study
export regression_discontinuity

# Statistical tests exports
export StatisticalTestResult, MixedEffectsResult, CausalEffectEstimate
export set_fast_stats_mode, get_bootstrap_iterations
export cohens_d_with_ci, eta_squared, epsilon_squared, cramers_v
export test_normality, test_homogeneity
export kruskal_wallis_test, mann_whitney_u_test, chi_square_test
export welch_ttest, spearman_correlation
export benjamini_hochberg, holm_bonferroni
export bootstrap_ci_stat, compute_ate_bootstrap
export descriptive_stats, significance_stars, format_p_value

# Advanced statistical analysis classes
export RigorousStatisticalAnalysis, run_all_analyses!, generate_descriptive_statistics
export CausalIdentificationAnalysis, run_all_estimates!, estimate_survival_effects!, estimate_roi_effects!
export generate_causal_effects_table
export CoxSurvivalAnalysis, CoxRegressionResult, prepare_survival_data, log_rank_tests, kaplan_meier_by_tier
export DifferenceInDifferencesAnalysis, estimate_did_effect!, generate_did_table
export RegressionDiscontinuityAnalysis, estimate_rd_effect!, generate_rd_table
export run_statistical_analysis

# Include statistical tests module
include("statistical_tests.jl")

# ============================================================================
# EFFECT SIZE RESULT TYPES
# ============================================================================

"""
Result container for effect size calculations.
"""
struct EffectSizeResult
    measure::String
    value::Float64
    ci_lower::Float64
    ci_upper::Float64
    interpretation::String
    n_group1::Int
    n_group2::Int
end

"""
Result container for ANOVA tests.
"""
struct ANOVAResult
    test_name::String
    statistic::Float64
    p_value::Float64
    df_between::Int
    df_within::Int
    effect_size::Float64  # eta-squared
    interpretation::String
end

"""
Result container for survival analysis.
"""
struct SurvivalResult
    metric::String
    survival_rates::Dict{String,Float64}
    hazard_ratios::Dict{String,Float64}
    log_rank_p::Float64
    interpretation::String
end

# ============================================================================
# COHEN'S D (STANDARDIZED MEAN DIFFERENCE)
# ============================================================================

"""
Compute Cohen's d effect size.

Cohen's d = (M1 - M2) / pooled_std

Interpretation:
- |d| < 0.2: negligible
- 0.2 <= |d| < 0.5: small
- 0.5 <= |d| < 0.8: medium
- |d| >= 0.8: large

Parameters
----------
group1 : Vector - First group data
group2 : Vector - Second group data
hedges_correction : Bool - Apply Hedges' g correction for small samples

Returns
-------
EffectSizeResult
"""
function cohens_d(
    group1::AbstractVector{<:Real},
    group2::AbstractVector{<:Real};
    hedges_correction::Bool=true
)::EffectSizeResult
    n1, n2 = length(group1), length(group2)

    if n1 < 2 || n2 < 2
        return EffectSizeResult("Cohen's d", NaN, NaN, NaN, "Insufficient data", n1, n2)
    end

    m1, m2 = mean(group1), mean(group2)
    v1, v2 = var(group1), var(group2)

    # Pooled standard deviation
    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    pooled_std = sqrt(pooled_var)

    if pooled_std ≈ 0
        return EffectSizeResult("Cohen's d", 0.0, 0.0, 0.0, "No variance", n1, n2)
    end

    d = (m1 - m2) / pooled_std

    # Apply Hedges' correction for small sample bias
    if hedges_correction
        df = n1 + n2 - 2
        correction = 1 - (3 / (4 * df - 1))
        d *= correction
    end

    # Standard error of d (approximation)
    se_d = sqrt((n1 + n2) / (n1 * n2) + d^2 / (2 * (n1 + n2)))

    # 95% CI
    z = 1.96
    ci_lower = d - z * se_d
    ci_upper = d + z * se_d

    # Interpretation
    abs_d = abs(d)
    interpretation = if abs_d < 0.2
        "negligible effect"
    elseif abs_d < 0.5
        "small effect"
    elseif abs_d < 0.8
        "medium effect"
    else
        "large effect"
    end

    measure = hedges_correction ? "Hedges' g" : "Cohen's d"
    return EffectSizeResult(measure, d, ci_lower, ci_upper, interpretation, n1, n2)
end

# ============================================================================
# CLIFF'S DELTA (NON-PARAMETRIC EFFECT SIZE)
# ============================================================================

"""
Compute Cliff's delta effect size (O(n log n) algorithm).

Cliff's delta measures the probability that a randomly selected value
from one group is greater than a randomly selected value from another.

Interpretation:
- |δ| < 0.147: negligible
- 0.147 <= |δ| < 0.33: small
- 0.33 <= |δ| < 0.474: medium
- |δ| >= 0.474: large

Parameters
----------
group1 : Vector - First group data
group2 : Vector - Second group data

Returns
-------
EffectSizeResult
"""
function cliffs_delta(
    group1::AbstractVector{<:Real},
    group2::AbstractVector{<:Real}
)::EffectSizeResult
    n1, n2 = length(group1), length(group2)

    if n1 == 0 || n2 == 0
        return EffectSizeResult("Cliff's delta", NaN, NaN, NaN, "Empty group", n1, n2)
    end

    # O(n log n) algorithm using sorting and merge-counting
    # Create combined array with group labels
    combined = vcat(
        [(v, 1) for v in group1],
        [(v, 2) for v in group2]
    )
    sort!(combined, by=x -> x[1])

    # Count dominance relationships
    n_greater = 0  # group1 > group2
    n_less = 0     # group1 < group2

    # Running count of group2 elements seen
    count_g2_below = 0

    for (val, group) in combined
        if group == 2
            count_g2_below += 1
        else  # group == 1
            # This group1 element is greater than count_g2_below group2 elements
            n_greater += count_g2_below
            # And less than (n2 - count_g2_below) group2 elements
            n_less += (n2 - count_g2_below)
        end
    end

    # Handle ties by scanning for equal values
    # (simplified - exact tie handling would need more work)
    delta = (n_greater - n_less) / (n1 * n2)

    # Bootstrap CI (simplified)
    se_delta = sqrt((1 - delta^2) / (n1 + n2 - 1)) * sqrt(2)
    z = 1.96
    ci_lower = clamp(delta - z * se_delta, -1.0, 1.0)
    ci_upper = clamp(delta + z * se_delta, -1.0, 1.0)

    # Interpretation
    abs_delta = abs(delta)
    interpretation = if abs_delta < 0.147
        "negligible effect"
    elseif abs_delta < 0.33
        "small effect"
    elseif abs_delta < 0.474
        "medium effect"
    else
        "large effect"
    end

    return EffectSizeResult("Cliff's delta", delta, ci_lower, ci_upper, interpretation, n1, n2)
end

# ============================================================================
# GLASS'S DELTA (CONTROL GROUP STANDARDIZATION)
# ============================================================================

"""
Compute Glass's delta effect size.

Like Cohen's d but uses only the control group's standard deviation,
appropriate when groups have different variances.

Parameters
----------
treatment : Vector - Treatment group data
control : Vector - Control group data

Returns
-------
EffectSizeResult
"""
function glass_delta(
    treatment::AbstractVector{<:Real},
    control::AbstractVector{<:Real}
)::EffectSizeResult
    n_t, n_c = length(treatment), length(control)

    if n_t < 2 || n_c < 2
        return EffectSizeResult("Glass's delta", NaN, NaN, NaN, "Insufficient data", n_t, n_c)
    end

    m_t, m_c = mean(treatment), mean(control)
    s_c = std(control)

    if s_c ≈ 0
        return EffectSizeResult("Glass's delta", 0.0, 0.0, 0.0, "No control variance", n_t, n_c)
    end

    delta = (m_t - m_c) / s_c

    # Standard error approximation
    se = sqrt(1/n_t + 1/n_c + delta^2 / (2 * n_c))
    z = 1.96
    ci_lower = delta - z * se
    ci_upper = delta + z * se

    abs_d = abs(delta)
    interpretation = if abs_d < 0.2
        "negligible effect"
    elseif abs_d < 0.5
        "small effect"
    elseif abs_d < 0.8
        "medium effect"
    else
        "large effect"
    end

    return EffectSizeResult("Glass's delta", delta, ci_lower, ci_upper, interpretation, n_t, n_c)
end

# ============================================================================
# ANOVA (ANALYSIS OF VARIANCE)
# ============================================================================

"""
Perform one-way ANOVA.

Parameters
----------
groups : Vector{Vector} - List of group data vectors

Returns
-------
ANOVAResult
"""
function anova_oneway(groups::Vector{<:AbstractVector{<:Real}})::ANOVAResult
    k = length(groups)

    if k < 2
        return ANOVAResult("One-way ANOVA", NaN, NaN, 0, 0, NaN, "Need at least 2 groups")
    end

    # Filter empty groups
    groups = filter(g -> length(g) > 0, groups)
    k = length(groups)

    if k < 2
        return ANOVAResult("One-way ANOVA", NaN, NaN, 0, 0, NaN, "Need at least 2 non-empty groups")
    end

    ns = [length(g) for g in groups]
    N = sum(ns)

    if N <= k
        return ANOVAResult("One-way ANOVA", NaN, NaN, k-1, N-k, NaN, "Insufficient total observations")
    end

    # Group means
    means = [mean(g) for g in groups]
    grand_mean = sum(sum(g) for g in groups) / N

    # Sum of squares
    ss_between = sum(n * (m - grand_mean)^2 for (n, m) in zip(ns, means))
    ss_within = sum(sum((x - m)^2 for x in g) for (g, m) in zip(groups, means))
    ss_total = ss_between + ss_within

    # Degrees of freedom
    df_between = k - 1
    df_within = N - k

    # Mean squares
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # F-statistic
    F = ms_between / ms_within

    # P-value from F distribution
    f_dist = FDist(df_between, df_within)
    p_value = 1 - cdf(f_dist, F)

    # Effect size (eta-squared)
    eta_sq = ss_between / ss_total

    # Interpretation
    interpretation = if p_value < 0.001
        "highly significant difference (p < 0.001)"
    elseif p_value < 0.01
        "very significant difference (p < 0.01)"
    elseif p_value < 0.05
        "significant difference (p < 0.05)"
    else
        "no significant difference (p >= 0.05)"
    end

    if eta_sq >= 0.14
        interpretation *= ", large effect size"
    elseif eta_sq >= 0.06
        interpretation *= ", medium effect size"
    elseif eta_sq >= 0.01
        interpretation *= ", small effect size"
    end

    return ANOVAResult("One-way ANOVA", F, p_value, df_between, df_within, eta_sq, interpretation)
end

# ============================================================================
# KRUSKAL-WALLIS TEST (NON-PARAMETRIC ANOVA)
# ============================================================================

"""
Perform Kruskal-Wallis H-test (non-parametric ANOVA).

Parameters
----------
groups : Vector{Vector} - List of group data vectors

Returns
-------
ANOVAResult (with H statistic instead of F)
"""
function kruskal_wallis(groups::Vector{<:AbstractVector{<:Real}})::ANOVAResult
    k = length(groups)

    if k < 2
        return ANOVAResult("Kruskal-Wallis H", NaN, NaN, 0, 0, NaN, "Need at least 2 groups")
    end

    # Filter empty groups
    groups = filter(g -> length(g) > 0, groups)
    k = length(groups)
    ns = [length(g) for g in groups]
    N = sum(ns)

    if N < k + 1
        return ANOVAResult("Kruskal-Wallis H", NaN, NaN, k-1, 0, NaN, "Insufficient observations")
    end

    # Combine and rank all data
    all_data = vcat([(v, i) for (i, g) in enumerate(groups) for v in g]...)
    sort!(all_data, by=x -> x[1])

    # Assign ranks (average for ties)
    ranks = zeros(Float64, N)
    i = 1
    while i <= N
        j = i
        while j <= N && all_data[j][1] == all_data[i][1]
            j += 1
        end
        avg_rank = (i + j - 1) / 2
        for idx in i:(j-1)
            ranks[idx] = avg_rank
        end
        i = j
    end

    # Compute sum of ranks for each group
    rank_sums = zeros(Float64, k)
    idx = 1
    for (i, (val, group_idx)) in enumerate(all_data)
        rank_sums[group_idx] += ranks[i]
    end

    # H statistic
    H = (12 / (N * (N + 1))) * sum(R^2 / n for (R, n) in zip(rank_sums, ns)) - 3 * (N + 1)

    # Tie correction (simplified)
    # For exact correction, need to count tied groups

    # P-value from chi-squared distribution
    df = k - 1
    chi_dist = Chisq(df)
    p_value = 1 - cdf(chi_dist, H)

    # Effect size (epsilon-squared)
    epsilon_sq = H / (N - 1)

    interpretation = if p_value < 0.001
        "highly significant difference (p < 0.001)"
    elseif p_value < 0.01
        "very significant difference (p < 0.01)"
    elseif p_value < 0.05
        "significant difference (p < 0.05)"
    else
        "no significant difference (p >= 0.05)"
    end

    return ANOVAResult("Kruskal-Wallis H", H, p_value, df, 0, epsilon_sq, interpretation)
end

# ============================================================================
# MANN-WHITNEY U TEST
# ============================================================================

"""
Perform Mann-Whitney U test (Wilcoxon rank-sum test).

Parameters
----------
group1 : Vector - First group data
group2 : Vector - Second group data

Returns
-------
NamedTuple with U statistic, p_value, and interpretation
"""
function mann_whitney_u(
    group1::AbstractVector{<:Real},
    group2::AbstractVector{<:Real}
)
    n1, n2 = length(group1), length(group2)

    if n1 == 0 || n2 == 0
        return (U=NaN, p_value=NaN, interpretation="Empty group")
    end

    # Combine and rank
    combined = vcat(
        [(v, 1) for v in group1],
        [(v, 2) for v in group2]
    )
    sort!(combined, by=x -> x[1])
    N = n1 + n2

    # Assign ranks
    ranks = zeros(Float64, N)
    i = 1
    while i <= N
        j = i
        while j <= N && combined[j][1] == combined[i][1]
            j += 1
        end
        avg_rank = (i + j - 1) / 2
        for idx in i:(j-1)
            ranks[idx] = avg_rank
        end
        i = j
    end

    # Sum of ranks for group 1
    R1 = sum(ranks[i] for i in 1:N if combined[i][2] == 1)

    # U statistic
    U1 = R1 - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1
    U = min(U1, U2)

    # Normal approximation for p-value
    mu = n1 * n2 / 2
    sigma = sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    z = (U - mu) / sigma
    p_value = 2 * (1 - cdf(Normal(), abs(z)))

    interpretation = if p_value < 0.001
        "highly significant difference (p < 0.001)"
    elseif p_value < 0.01
        "very significant difference (p < 0.01)"
    elseif p_value < 0.05
        "significant difference (p < 0.05)"
    else
        "no significant difference (p >= 0.05)"
    end

    return (U=U, p_value=p_value, z=z, interpretation=interpretation)
end

# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

"""
Compute bootstrap confidence interval for a statistic.

Parameters
----------
data : Vector - Sample data
statistic : Function - Statistic to compute (e.g., mean, median)
n_bootstrap : Int - Number of bootstrap iterations
ci_level : Float64 - Confidence level (default 0.95)
rng : AbstractRNG - Random number generator

Returns
-------
NamedTuple with estimate, ci_lower, ci_upper, se
"""
function bootstrap_ci(
    data::AbstractVector{<:Real};
    statistic::Function=mean,
    n_bootstrap::Int=1000,
    ci_level::Float64=0.95,
    rng::AbstractRNG=Random.default_rng()
)
    n = length(data)

    if n == 0
        return (estimate=NaN, ci_lower=NaN, ci_upper=NaN, se=NaN)
    end

    point_estimate = statistic(data)

    # Generate bootstrap samples
    boot_stats = zeros(n_bootstrap)
    for b in 1:n_bootstrap
        boot_sample = data[rand(rng, 1:n, n)]
        boot_stats[b] = statistic(boot_sample)
    end

    # Percentile method
    alpha = 1 - ci_level
    ci_lower = quantile(boot_stats, alpha / 2)
    ci_upper = quantile(boot_stats, 1 - alpha / 2)

    se = std(boot_stats)

    return (estimate=point_estimate, ci_lower=ci_lower, ci_upper=ci_upper, se=se)
end

# ============================================================================
# PERMUTATION TEST
# ============================================================================

"""
Perform permutation test for difference between two groups.

Parameters
----------
group1 : Vector - First group data
group2 : Vector - Second group data
statistic : Function - Test statistic (default: difference in means)
n_permutations : Int - Number of permutations
rng : AbstractRNG - Random number generator

Returns
-------
NamedTuple with observed statistic, p_value, and interpretation
"""
function permutation_test(
    group1::AbstractVector{<:Real},
    group2::AbstractVector{<:Real};
    statistic::Function=((g1, g2) -> mean(g1) - mean(g2)),
    n_permutations::Int=5000,
    rng::AbstractRNG=Random.default_rng()
)
    n1, n2 = length(group1), length(group2)
    N = n1 + n2

    if n1 == 0 || n2 == 0
        return (observed=NaN, p_value=NaN, interpretation="Empty group")
    end

    combined = vcat(group1, group2)
    observed = statistic(group1, group2)

    # Permutation distribution
    n_extreme = 0
    for _ in 1:n_permutations
        perm = shuffle(rng, combined)
        perm_g1 = perm[1:n1]
        perm_g2 = perm[(n1+1):end]
        perm_stat = statistic(perm_g1, perm_g2)

        if abs(perm_stat) >= abs(observed)
            n_extreme += 1
        end
    end

    p_value = (n_extreme + 1) / (n_permutations + 1)

    interpretation = if p_value < 0.001
        "highly significant (p < 0.001)"
    elseif p_value < 0.01
        "very significant (p < 0.01)"
    elseif p_value < 0.05
        "significant (p < 0.05)"
    else
        "not significant (p >= 0.05)"
    end

    return (observed=observed, p_value=p_value, n_permutations=n_permutations, interpretation=interpretation)
end

# ============================================================================
# SURVIVAL ANALYSIS UTILITIES
# ============================================================================

"""
Compute survival rates by group.

Parameters
----------
df : DataFrame - Agent data with 'survived' and grouping column
group_col : Symbol - Column name for grouping (e.g., :ai_level)

Returns
-------
Dict mapping group -> survival rate
"""
function survival_rates_by_group(df::DataFrame, group_col::Symbol)::Dict{String,Float64}
    if !hasproperty(df, group_col) || !hasproperty(df, :survived)
        return Dict{String,Float64}()
    end

    rates = Dict{String,Float64}()

    for group in unique(df[!, group_col])
        group_data = filter(row -> row[group_col] == group, df)
        if nrow(group_data) > 0
            rate = mean(group_data.survived)
            rates[string(group)] = rate
        end
    end

    return rates
end

"""
Simple survival analysis comparing AI tiers.

Parameters
----------
df : DataFrame - Agent data with survival and AI tier info
tier_col : Symbol - Column with AI tier

Returns
-------
SurvivalResult
"""
function survival_analysis(df::DataFrame; tier_col::Symbol=:primary_ai_level)::SurvivalResult
    if !hasproperty(df, tier_col) || !hasproperty(df, :survived)
        return SurvivalResult(
            "Survival by AI Tier",
            Dict{String,Float64}(),
            Dict{String,Float64}(),
            NaN,
            "Missing required columns"
        )
    end

    # Compute survival rates
    rates = survival_rates_by_group(df, tier_col)

    if isempty(rates)
        return SurvivalResult(
            "Survival by AI Tier",
            Dict{String,Float64}(),
            Dict{String,Float64}(),
            NaN,
            "No groups found"
        )
    end

    # Compute hazard ratios relative to "none" baseline
    baseline_rate = get(rates, "none", get(rates, "None", first(values(rates))))
    baseline_hazard = 1 - baseline_rate

    hazard_ratios = Dict{String,Float64}()
    for (tier, rate) in rates
        tier_hazard = 1 - rate
        if baseline_hazard > 0
            hazard_ratios[tier] = tier_hazard / baseline_hazard
        else
            hazard_ratios[tier] = 0.0
        end
    end

    # Log-rank test (simplified - compare groups)
    groups = [filter(row -> row[tier_col] == tier, df).survived for tier in unique(df[!, tier_col])]
    groups = filter(g -> length(g) > 0, groups)

    if length(groups) >= 2
        kw_result = kruskal_wallis(groups)
        log_rank_p = kw_result.p_value
    else
        log_rank_p = NaN
    end

    # Interpretation
    sorted_rates = sort(collect(rates), by=x->x[2], rev=true)
    best_tier = sorted_rates[1][1]
    best_rate = sorted_rates[1][2]
    worst_tier = sorted_rates[end][1]
    worst_rate = sorted_rates[end][2]

    interpretation = "Survival rates range from $(round(worst_rate*100, digits=1))% ($worst_tier) " *
                     "to $(round(best_rate*100, digits=1))% ($best_tier)"

    if !isnan(log_rank_p)
        if log_rank_p < 0.05
            interpretation *= ". Significant difference across tiers (p=$(round(log_rank_p, digits=4)))"
        else
            interpretation *= ". No significant difference (p=$(round(log_rank_p, digits=4)))"
        end
    end

    return SurvivalResult(
        "Survival by AI Tier",
        rates,
        hazard_ratios,
        log_rank_p,
        interpretation
    )
end

# ============================================================================
# BATCH EFFECT SIZE ANALYSIS
# ============================================================================

"""
Compute all effect sizes comparing AI tiers on a given metric.

Parameters
----------
df : DataFrame - Data with AI tier and outcome columns
outcome_col : Symbol - Outcome variable column
tier_col : Symbol - AI tier column
baseline_tier : String - Reference tier for comparisons

Returns
-------
Dict mapping tier -> EffectSizeResult
"""
function compute_tier_effects(
    df::DataFrame;
    outcome_col::Symbol,
    tier_col::Symbol=:primary_ai_level,
    baseline_tier::String="none"
)::Dict{String,EffectSizeResult}
    results = Dict{String,EffectSizeResult}()

    if !hasproperty(df, outcome_col) || !hasproperty(df, tier_col)
        return results
    end

    # Get baseline data
    baseline_df = filter(row -> lowercase(string(row[tier_col])) == lowercase(baseline_tier), df)
    if nrow(baseline_df) == 0
        return results
    end

    baseline_data = collect(skipmissing(baseline_df[!, outcome_col]))

    # Compare each tier to baseline
    for tier in unique(df[!, tier_col])
        tier_str = lowercase(string(tier))
        if tier_str == lowercase(baseline_tier)
            continue
        end

        tier_df = filter(row -> row[tier_col] == tier, df)
        tier_data = collect(skipmissing(tier_df[!, outcome_col]))

        if length(tier_data) >= 2 && length(baseline_data) >= 2
            results[string(tier)] = cohens_d(tier_data, baseline_data)
        end
    end

    return results
end

# ============================================================================
# ADVANCED RESULT TYPES
# ============================================================================

# Note: CoxRegressionResult, DiDResult, and RDResult are defined in statistical_tests.jl

"""
Result container for Propensity Score analysis.
"""
struct PropensityScoreResult
    method::String
    outcome::String
    ate::Float64
    att::Float64
    ate_se::Float64
    att_se::Float64
    ate_ci::Tuple{Float64,Float64}
    att_ci::Tuple{Float64,Float64}
    p_value_ate::Float64
    p_value_att::Float64
    n_treated::Int
    n_control::Int
    interpretation::String
end


# ============================================================================
# KAPLAN-MEIER SURVIVAL CURVES
# ============================================================================

"""
Compute Kaplan-Meier survival curves by group.

Parameters
----------
df : DataFrame - Data with duration, event, and group columns
duration_col : Symbol - Time-to-event column
event_col : Symbol - Event indicator (1=event, 0=censored)
group_col : Symbol - Grouping column

Returns
-------
Dict mapping group -> DataFrame with time and survival probability
"""
function kaplan_meier_curves(
    df::DataFrame;
    duration_col::Symbol=:duration,
    event_col::Symbol=:event,
    group_col::Symbol=:ai_tier
)::Dict{String,DataFrame}
    if !hasproperty(df, duration_col) || !hasproperty(df, event_col) || !hasproperty(df, group_col)
        return Dict{String,DataFrame}()
    end

    curves = Dict{String,DataFrame}()

    for group in unique(skipmissing(df[!, group_col]))
        group_df = filter(row -> !ismissing(row[group_col]) && row[group_col] == group, df)

        if nrow(group_df) < 5
            continue
        end

        durations = collect(skipmissing(group_df[!, duration_col]))
        events = collect(skipmissing(group_df[!, event_col]))

        # Sort by duration
        order = sortperm(durations)
        durations = durations[order]
        events = events[order]

        n = length(durations)
        times = Float64[0.0]
        survival = Float64[1.0]
        at_risk = n

        current_surv = 1.0
        i = 1
        while i <= n
            t = durations[i]
            d = 0  # deaths at time t
            c = 0  # censored at time t

            while i <= n && durations[i] == t
                if events[i] == 1
                    d += 1
                else
                    c += 1
                end
                i += 1
            end

            if d > 0
                current_surv *= (at_risk - d) / at_risk
                push!(times, t)
                push!(survival, current_surv)
            end

            at_risk -= (d + c)
        end

        curves[string(group)] = DataFrame(time=times, survival=survival)
    end

    return curves
end

# ============================================================================
# LOG-RANK TEST
# ============================================================================

"""
Perform log-rank test comparing survival between two groups.

Parameters
----------
df : DataFrame - Data with duration, event, and group columns
duration_col : Symbol - Time-to-event column
event_col : Symbol - Event indicator
group1_name : String - First group name
group2_name : String - Second group name
group_col : Symbol - Grouping column

Returns
-------
NamedTuple with test statistic, p-value, and interpretation
"""
function log_rank_test(
    df::DataFrame;
    duration_col::Symbol=:duration,
    event_col::Symbol=:event,
    group1_name::String="none",
    group2_name::String="premium",
    group_col::Symbol=:ai_tier
)
    if !hasproperty(df, duration_col) || !hasproperty(df, event_col) || !hasproperty(df, group_col)
        return (statistic=NaN, p_value=NaN, interpretation="Missing columns")
    end

    g1 = filter(row -> !ismissing(row[group_col]) && string(row[group_col]) == group1_name, df)
    g2 = filter(row -> !ismissing(row[group_col]) && string(row[group_col]) == group2_name, df)

    if nrow(g1) < 5 || nrow(g2) < 5
        return (statistic=NaN, p_value=NaN, interpretation="Insufficient data")
    end

    # Combine and get unique event times
    all_times = unique(vcat(
        collect(skipmissing(g1[!, duration_col])),
        collect(skipmissing(g2[!, duration_col]))
    ))
    sort!(all_times)

    # Compute log-rank statistic
    O1 = 0.0  # Observed events in group 1
    E1 = 0.0  # Expected events in group 1
    V = 0.0   # Variance

    for t in all_times
        # At-risk counts
        n1 = count(row -> !ismissing(row[duration_col]) && row[duration_col] >= t, eachrow(g1))
        n2 = count(row -> !ismissing(row[duration_col]) && row[duration_col] >= t, eachrow(g2))
        n = n1 + n2

        if n == 0
            continue
        end

        # Events at time t
        d1 = count(row -> !ismissing(row[duration_col]) && row[duration_col] == t &&
                         !ismissing(row[event_col]) && row[event_col] == 1, eachrow(g1))
        d2 = count(row -> !ismissing(row[duration_col]) && row[duration_col] == t &&
                         !ismissing(row[event_col]) && row[event_col] == 1, eachrow(g2))
        d = d1 + d2

        if n > 1 && d > 0
            O1 += d1
            E1 += n1 * d / n
            V += n1 * n2 * d * (n - d) / (n^2 * (n - 1))
        end
    end

    # Test statistic
    if V > 0
        chi_sq = (O1 - E1)^2 / V
        p_value = 1 - cdf(Chisq(1), chi_sq)
    else
        chi_sq = NaN
        p_value = NaN
    end

    interpretation = if isnan(p_value)
        "Unable to compute"
    elseif p_value < 0.001
        "Highly significant difference in survival (p < 0.001)"
    elseif p_value < 0.01
        "Very significant difference in survival (p < 0.01)"
    elseif p_value < 0.05
        "Significant difference in survival (p < 0.05)"
    else
        "No significant difference in survival (p >= 0.05)"
    end

    return (statistic=chi_sq, p_value=p_value, observed=O1, expected=E1, interpretation=interpretation)
end

# ============================================================================
# PROPENSITY SCORE METHODS
# ============================================================================

"""
Estimate propensity scores using logistic regression.

Parameters
----------
df : DataFrame - Data with treatment and covariate columns
treatment_col : Symbol - Binary treatment indicator
covariate_cols : Vector{Symbol} - Covariates for propensity model

Returns
-------
Vector of propensity scores
"""
function estimate_propensity_scores(
    df::DataFrame;
    treatment_col::Symbol=:uses_ai,
    covariate_cols::Vector{Symbol}=[:initial_capital]
)::Vector{Float64}
    if !hasproperty(df, treatment_col)
        return Float64[]
    end

    # Filter to valid covariates
    valid_covs = filter(c -> hasproperty(df, c), covariate_cols)
    if isempty(valid_covs)
        # Return uniform propensity if no covariates
        return fill(0.5, nrow(df))
    end

    # Build design matrix
    n = nrow(df)
    T = collect(skipmissing(df[!, treatment_col]))

    X = zeros(n, length(valid_covs) + 1)
    X[:, 1] .= 1.0  # Intercept

    for (i, col) in enumerate(valid_covs)
        col_data = df[!, col]
        for j in 1:n
            X[j, i+1] = ismissing(col_data[j]) ? 0.0 : Float64(col_data[j])
        end
    end

    # Simple logistic regression via IRLS
    beta = zeros(size(X, 2))

    for iter in 1:25
        p = 1.0 ./ (1.0 .+ exp.(-X * beta))
        p = clamp.(p, 1e-10, 1 - 1e-10)

        W = Diagonal(p .* (1 .- p))
        z = X * beta .+ (T .- p) ./ (p .* (1 .- p) .+ 1e-10)

        try
            XtWX = X' * W * X
            XtWz = X' * W * z
            beta_new = XtWX \ XtWz

            if maximum(abs.(beta_new .- beta)) < 1e-6
                beta = beta_new
                break
            end
            beta = beta_new
        catch
            break
        end
    end

    propensity_scores = 1.0 ./ (1.0 .+ exp.(-X * beta))
    return propensity_scores
end

"""
Propensity Score Matching using nearest neighbor.

Parameters
----------
df : DataFrame - Data with treatment, outcome, and propensity scores
outcome_col : Symbol - Outcome variable
treatment_col : Symbol - Treatment indicator
ps_col : Symbol - Propensity score column
caliper : Float64 - Maximum distance for matching (in SD)

Returns
-------
PropensityScoreResult
"""
function propensity_score_matching(
    df::DataFrame;
    outcome_col::Symbol=:survived,
    treatment_col::Symbol=:uses_ai,
    ps_col::Symbol=:propensity_score,
    caliper::Float64=0.2,
    rng::AbstractRNG=Random.default_rng()
)::PropensityScoreResult
    if !hasproperty(df, outcome_col) || !hasproperty(df, treatment_col)
        return PropensityScoreResult("PSM", string(outcome_col), NaN, NaN, NaN, NaN,
                                     (NaN, NaN), (NaN, NaN), NaN, NaN, 0, 0, "Missing columns")
    end

    # Estimate propensity scores if not provided
    if !hasproperty(df, ps_col)
        ps = estimate_propensity_scores(df; treatment_col=treatment_col)
        df = copy(df)
        df[!, ps_col] = ps
    end

    T = collect(skipmissing(df[!, treatment_col]))
    Y = collect(skipmissing(df[!, outcome_col]))
    ps = collect(df[!, ps_col])

    treated_idx = findall(x -> x == 1, T)
    control_idx = findall(x -> x == 0, T)

    if length(treated_idx) < 5 || length(control_idx) < 5
        return PropensityScoreResult("PSM", string(outcome_col), NaN, NaN, NaN, NaN,
                                     (NaN, NaN), (NaN, NaN), NaN, NaN,
                                     length(treated_idx), length(control_idx), "Insufficient data")
    end

    caliper_abs = caliper * std(ps)

    # Match each treated to nearest control
    matched_treated = Int[]
    matched_control = Int[]
    control_used = Set{Int}()

    for t_idx in treated_idx
        ps_t = ps[t_idx]
        best_dist = Inf
        best_c = -1

        for c_idx in control_idx
            if c_idx in control_used
                continue
            end
            dist = abs(ps[c_idx] - ps_t)
            if dist < best_dist && dist <= caliper_abs
                best_dist = dist
                best_c = c_idx
            end
        end

        if best_c > 0
            push!(matched_treated, t_idx)
            push!(matched_control, best_c)
            push!(control_used, best_c)
        end
    end

    if length(matched_treated) < 10
        return PropensityScoreResult("PSM", string(outcome_col), NaN, NaN, NaN, NaN,
                                     (NaN, NaN), (NaN, NaN), NaN, NaN,
                                     length(treated_idx), length(control_idx),
                                     "Insufficient matches")
    end

    # Compute ATT
    Y_treated = [Y[i] for i in matched_treated]
    Y_control = [Y[i] for i in matched_control]
    att = mean(Y_treated) - mean(Y_control)

    # Bootstrap SE
    n_boot = 500
    boot_atts = Float64[]
    n_matched = length(matched_treated)

    for _ in 1:n_boot
        boot_idx = rand(rng, 1:n_matched, n_matched)
        boot_att = mean(Y_treated[boot_idx]) - mean(Y_control[boot_idx])
        push!(boot_atts, boot_att)
    end

    att_se = std(boot_atts)
    att_ci = (quantile(boot_atts, 0.025), quantile(boot_atts, 0.975))

    t_stat = att_se > 0 ? att / att_se : 0.0
    p_value = 2 * (1 - cdf(Normal(), abs(t_stat)))

    interpretation = if p_value < 0.05
        direction = att > 0 ? "increases" : "decreases"
        "Treatment $direction $(string(outcome_col)) by $(round(abs(att), digits=4)) (p=$(round(p_value, digits=4)))"
    else
        "No significant effect on $(string(outcome_col)) (p=$(round(p_value, digits=4)))"
    end

    return PropensityScoreResult(
        "nearest_neighbor_matching", string(outcome_col),
        att, att, att_se, att_se, att_ci, att_ci, p_value, p_value,
        length(unique(matched_treated)), length(unique(matched_control)),
        interpretation
    )
end

"""
Inverse Probability Weighting estimator.

Parameters
----------
df : DataFrame - Data with treatment, outcome, and propensity scores
outcome_col : Symbol - Outcome variable
treatment_col : Symbol - Treatment indicator

Returns
-------
PropensityScoreResult
"""
function inverse_probability_weighting(
    df::DataFrame;
    outcome_col::Symbol=:survived,
    treatment_col::Symbol=:uses_ai,
    rng::AbstractRNG=Random.default_rng()
)::PropensityScoreResult
    if !hasproperty(df, outcome_col) || !hasproperty(df, treatment_col)
        return PropensityScoreResult("IPW", string(outcome_col), NaN, NaN, NaN, NaN,
                                     (NaN, NaN), (NaN, NaN), NaN, NaN, 0, 0, "Missing columns")
    end

    # Estimate propensity scores
    ps = estimate_propensity_scores(df; treatment_col=treatment_col)

    T = collect(skipmissing(df[!, treatment_col]))
    Y = collect(skipmissing(df[!, outcome_col]))

    # Trim extreme propensities
    valid = (ps .> 0.01) .& (ps .< 0.99)
    T = T[valid]
    ps = ps[valid]
    Y = Y[valid]

    if length(T) < 50
        return PropensityScoreResult("IPW", string(outcome_col), NaN, NaN, NaN, NaN,
                                     (NaN, NaN), (NaN, NaN), NaN, NaN, 0, 0, "Insufficient data")
    end

    # IPW weights
    weights_ate = T ./ ps .+ (1 .- T) ./ (1 .- ps)

    # Trim extreme weights
    w_threshold = quantile(weights_ate, 0.99)
    weights_ate = min.(weights_ate, w_threshold)

    # Normalize
    weights_ate = weights_ate ./ sum(weights_ate) .* length(weights_ate)

    # ATE
    ate = sum(weights_ate .* Y .* T) / sum(weights_ate .* T) -
          sum(weights_ate .* Y .* (1 .- T)) / sum(weights_ate .* (1 .- T))

    # Bootstrap SE
    n_boot = 500
    boot_ates = Float64[]
    n = length(T)

    for _ in 1:n_boot
        boot_idx = rand(rng, 1:n, n)
        T_b = T[boot_idx]
        ps_b = ps[boot_idx]
        Y_b = Y[boot_idx]

        w_b = T_b ./ ps_b .+ (1 .- T_b) ./ (1 .- ps_b)
        w_b = min.(w_b, w_threshold)
        w_b = w_b ./ sum(w_b) .* length(w_b)

        ate_b = sum(w_b .* Y_b .* T_b) / sum(w_b .* T_b) -
                sum(w_b .* Y_b .* (1 .- T_b)) / sum(w_b .* (1 .- T_b))
        push!(boot_ates, ate_b)
    end

    ate_se = std(boot_ates)
    ate_ci = (quantile(boot_ates, 0.025), quantile(boot_ates, 0.975))

    p_value = 2 * (1 - cdf(Normal(), abs(ate / ate_se)))

    n_treated = sum(T .== 1)
    n_control = sum(T .== 0)

    interpretation = if p_value < 0.05
        direction = ate > 0 ? "increases" : "decreases"
        "Treatment $direction $(string(outcome_col)) by $(round(abs(ate), digits=4)) (p=$(round(p_value, digits=4)))"
    else
        "No significant effect on $(string(outcome_col)) (p=$(round(p_value, digits=4)))"
    end

    return PropensityScoreResult(
        "inverse_probability_weighting", string(outcome_col),
        ate, ate, ate_se, ate_se, ate_ci, ate_ci, p_value, p_value,
        n_treated, n_control, interpretation
    )
end

# ============================================================================
# DIFFERENCE-IN-DIFFERENCES
# ============================================================================

"""
Estimate Difference-in-Differences treatment effect using Two-Way Fixed Effects.

Parameters
----------
df : DataFrame - Panel data (agent × time)
outcome_col : Symbol - Outcome variable
agent_id_col : Symbol - Agent identifier
time_col : Symbol - Time period
treatment_col : Symbol - Treatment indicator (post × treated)

Returns
-------
DiDResult
"""
function difference_in_differences(
    df::DataFrame;
    outcome_col::Symbol=:capital,
    agent_id_col::Symbol=:agent_id,
    time_col::Symbol=:step,
    treatment_col::Symbol=:uses_ai
)::DiDResult
    if !hasproperty(df, outcome_col) || !hasproperty(df, agent_id_col) || !hasproperty(df, time_col)
        return DiDResult(string(outcome_col), NaN, NaN, NaN, NaN, NaN, 0, 0, 0, 0, "Missing columns")
    end

    # Identify treatment timing
    if !hasproperty(df, :did_term)
        # Create DiD terms
        df = copy(df)

        # Find first treatment time per agent
        treated_obs = filter(row -> !ismissing(row[treatment_col]) && row[treatment_col] == 1, df)
        if nrow(treated_obs) > 0
            first_treat = combine(groupby(treated_obs, agent_id_col), time_col => minimum => :first_ai_step)
            df = leftjoin(df, first_treat, on=agent_id_col)
        else
            df[!, :first_ai_step] = fill(missing, nrow(df))
        end

        df[!, :treated] = .!ismissing.(df[!, :first_ai_step])
        df[!, :post] = [ismissing(row.first_ai_step) ? false : row[time_col] >= row.first_ai_step for row in eachrow(df)]
        df[!, :did_term] = df[!, :treated] .& df[!, :post]
    end

    model_df = dropmissing(df, [outcome_col, agent_id_col, time_col, :did_term])

    if nrow(model_df) < 100
        return DiDResult(string(outcome_col), NaN, NaN, NaN, NaN, NaN, 0, 0, 0, 0, "Insufficient data")
    end

    # Within transformation (demean by agent and time)
    Y = collect(model_df[!, outcome_col])
    D = Float64.(collect(model_df[!, :did_term]))

    agent_ids = model_df[!, agent_id_col]
    time_ids = model_df[!, time_col]

    # Compute means
    agent_means = Dict{eltype(agent_ids), Float64}()
    for (aid, y) in zip(agent_ids, Y)
        agent_means[aid] = get(agent_means, aid, 0.0) + y
    end
    agent_counts = Dict{eltype(agent_ids), Int}()
    for aid in agent_ids
        agent_counts[aid] = get(agent_counts, aid, 0) + 1
    end
    for aid in keys(agent_means)
        agent_means[aid] /= agent_counts[aid]
    end

    time_means = Dict{eltype(time_ids), Float64}()
    for (tid, y) in zip(time_ids, Y)
        time_means[tid] = get(time_means, tid, 0.0) + y
    end
    time_counts = Dict{eltype(time_ids), Int}()
    for tid in time_ids
        time_counts[tid] = get(time_counts, tid, 0) + 1
    end
    for tid in keys(time_means)
        time_means[tid] /= time_counts[tid]
    end

    grand_mean = mean(Y)

    # Within transformation
    Y_within = [Y[i] - agent_means[agent_ids[i]] - time_means[time_ids[i]] + grand_mean for i in 1:length(Y)]

    # Similarly for D
    agent_means_d = Dict{eltype(agent_ids), Float64}()
    for (aid, d) in zip(agent_ids, D)
        agent_means_d[aid] = get(agent_means_d, aid, 0.0) + d
    end
    for aid in keys(agent_means_d)
        agent_means_d[aid] /= agent_counts[aid]
    end

    time_means_d = Dict{eltype(time_ids), Float64}()
    for (tid, d) in zip(time_ids, D)
        time_means_d[tid] = get(time_means_d, tid, 0.0) + d
    end
    for tid in keys(time_means_d)
        time_means_d[tid] /= time_counts[tid]
    end

    grand_mean_d = mean(D)

    D_within = [D[i] - agent_means_d[agent_ids[i]] - time_means_d[time_ids[i]] + grand_mean_d for i in 1:length(D)]

    # OLS on within-transformed data
    X = hcat(ones(length(D_within)), D_within)
    beta = X \ Y_within
    att = beta[2]

    # Cluster-robust standard errors (by agent)
    residuals = Y_within .- X * beta
    unique_agents = unique(agent_ids)
    n_clusters = length(unique_agents)

    meat = zeros(2, 2)
    for aid in unique_agents
        mask = agent_ids .== aid
        Xc = X[mask, :]
        rc = residuals[mask]
        meat += Xc' * (rc * rc') * Xc
    end

    bread = inv(X' * X)
    var_robust = bread * meat * bread * (n_clusters / (n_clusters - 1))
    se_att = sqrt(var_robust[2, 2])

    # CI and p-value
    ci_lower = att - 1.96 * se_att
    ci_upper = att + 1.96 * se_att
    t_stat = att / se_att
    p_value = 2 * (1 - cdf(TDist(n_clusters - 1), abs(t_stat)))

    # Count units and periods
    n_treated = length(unique(agent_ids[D .== 1]))
    n_control = length(unique(agent_ids)) - n_treated
    n_pre = length(unique(time_ids[model_df[!, :post] .== false]))
    n_post = length(unique(time_ids[model_df[!, :post] .== true]))

    interpretation = if p_value < 0.05
        direction = att > 0 ? "increases" : "decreases"
        "Treatment $direction $(string(outcome_col)) by $(round(abs(att), digits=4)) (p=$(round(p_value, digits=4)))"
    else
        "No significant effect on $(string(outcome_col)) (p=$(round(p_value, digits=4)))"
    end

    return DiDResult(
        string(outcome_col), att, se_att, ci_lower, ci_upper, p_value,
        n_treated, n_control, n_pre, n_post, interpretation
    )
end

"""
Estimate event study (dynamic DiD) specification.

Parameters
----------
df : DataFrame - Panel data with event_time column
outcome_col : Symbol - Outcome variable
event_time_col : Symbol - Event time (relative to treatment)
event_window : Tuple{Int,Int} - (min_k, max_k) event window

Returns
-------
DataFrame with coefficients by event time
"""
function event_study(
    df::DataFrame;
    outcome_col::Symbol=:capital,
    event_time_col::Symbol=:event_time,
    event_window::Tuple{Int,Int}=(-5, 10)
)::DataFrame
    if !hasproperty(df, outcome_col) || !hasproperty(df, event_time_col)
        return DataFrame()
    end

    min_k, max_k = event_window
    ref_period = -1

    results = DataFrame(
        event_time=Int[],
        coefficient=Float64[],
        se=Float64[],
        ci_lower=Float64[],
        ci_upper=Float64[],
        n=Int[]
    )

    df_es = filter(row -> !ismissing(row[event_time_col]) &&
                          row[event_time_col] >= min_k &&
                          row[event_time_col] <= max_k, df)

    if nrow(df_es) < 50
        return results
    end

    ref_df = filter(row -> row[event_time_col] == ref_period, df_es)
    if nrow(ref_df) < 10
        return results
    end

    ref_mean = mean(skipmissing(ref_df[!, outcome_col]))
    ref_var = var(skipmissing(ref_df[!, outcome_col]))

    event_times = sort(unique(skipmissing(df_es[!, event_time_col])))

    for k in event_times
        if k == ref_period
            continue
        end

        at_k = filter(row -> row[event_time_col] == k, df_es)
        if nrow(at_k) < 10
            continue
        end

        k_mean = mean(skipmissing(at_k[!, outcome_col]))
        k_var = var(skipmissing(at_k[!, outcome_col]))

        coef = k_mean - ref_mean
        se = sqrt(ref_var / nrow(ref_df) + k_var / nrow(at_k))

        push!(results, (
            event_time=Int(k),
            coefficient=coef,
            se=se,
            ci_lower=coef - 1.96 * se,
            ci_upper=coef + 1.96 * se,
            n=nrow(at_k)
        ))
    end

    return results
end

# ============================================================================
# REGRESSION DISCONTINUITY
# ============================================================================

"""
Estimate Regression Discontinuity treatment effect using local polynomial regression.

Parameters
----------
df : DataFrame - Data with running variable and outcome
running_var : Symbol - Running variable (determines treatment)
outcome_var : Symbol - Outcome variable
cutoff : Float64 - Threshold value
bandwidth : Float64 - Window width around cutoff
polynomial_order : Int - Degree of polynomial (1=local linear)
kernel : String - Kernel function ("triangular", "uniform", "epanechnikov")

Returns
-------
RDResult
"""
function regression_discontinuity(
    df::DataFrame;
    running_var::Symbol=:capital_at_decision,
    outcome_var::Symbol=:survived,
    cutoff::Float64=NaN,
    bandwidth::Float64=NaN,
    polynomial_order::Int=1,
    kernel::String="triangular"
)::RDResult
    if !hasproperty(df, running_var) || !hasproperty(df, outcome_var)
        return RDResult(string(running_var), NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0, 0, 1, "Missing columns")
    end

    data = dropmissing(df, [running_var, outcome_var])

    if nrow(data) < 50
        return RDResult(string(running_var), NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0, 0, 1, "Insufficient data")
    end

    X = collect(data[!, running_var])
    Y = collect(data[!, outcome_var])

    # Determine cutoff if not specified (use median)
    if isnan(cutoff)
        cutoff = median(X)
    end

    # Center at cutoff
    X_centered = X .- cutoff

    # Determine bandwidth if not specified (rule-of-thumb)
    if isnan(bandwidth)
        bandwidth = 1.06 * std(X_centered) * length(X)^(-1/5) * 2
    end

    # Restrict to bandwidth window
    in_window = abs.(X_centered) .<= bandwidth
    X_bw = X_centered[in_window]
    Y_bw = Y[in_window]

    n_left = sum(X_bw .< 0)
    n_right = sum(X_bw .>= 0)

    if n_left < 20 || n_right < 20
        return RDResult(string(running_var), cutoff, bandwidth, NaN, NaN, NaN, NaN, NaN,
                       n_left, n_right, polynomial_order, "Insufficient observations in window")
    end

    # Compute kernel weights
    weights = if kernel == "triangular"
        max.(1 .- abs.(X_bw) ./ bandwidth, 0)
    elseif kernel == "uniform"
        ones(length(X_bw))
    elseif kernel == "epanechnikov"
        u = X_bw ./ bandwidth
        max.(0.75 .* (1 .- u.^2), 0)
    else
        ones(length(X_bw))
    end

    # Treatment indicator
    D = Float64.(X_bw .>= 0)

    # Build design matrix: [1, D, X, D*X, X^2, D*X^2, ...]
    n_bw = length(X_bw)
    X_design = ones(n_bw, 1 + 1 + 2*polynomial_order)  # intercept + D + polynomial terms
    X_design[:, 2] = D

    col = 3
    for p in 1:polynomial_order
        X_design[:, col] = X_bw.^p
        X_design[:, col+1] = X_bw.^p .* D
        col += 2
    end

    # Weighted least squares
    W = Diagonal(weights)

    try
        XtWX = X_design' * W * X_design
        XtWy = X_design' * W * Y_bw
        beta = XtWX \ XtWy

        # Treatment effect is coefficient on D (index 2)
        tau = beta[2]

        # Standard error
        residuals = Y_bw .- X_design * beta
        sigma2 = sum(weights .* residuals.^2) / (sum(weights) - size(X_design, 2))
        var_beta = sigma2 * inv(XtWX)
        se_tau = sqrt(var_beta[2, 2])

        # CI and p-value
        ci_lower = tau - 1.96 * se_tau
        ci_upper = tau + 1.96 * se_tau
        z_stat = tau / se_tau
        p_value = 2 * (1 - cdf(Normal(), abs(z_stat)))

        interpretation = if p_value < 0.05
            direction = tau > 0 ? "increases" : "decreases"
            "Crossing $(string(running_var))=$(round(cutoff, digits=2)) $direction $(string(outcome_var)) by $(round(abs(tau), digits=4)) (p=$(round(p_value, digits=4)))"
        else
            "No significant discontinuity at $(string(running_var))=$(round(cutoff, digits=2)) (p=$(round(p_value, digits=4)))"
        end

        return RDResult(
            string(running_var), cutoff, bandwidth, tau, se_tau, ci_lower, ci_upper,
            p_value, n_left, n_right, polynomial_order, interpretation
        )
    catch e
        return RDResult(string(running_var), cutoff, bandwidth, NaN, NaN, NaN, NaN, NaN,
                       n_left, n_right, polynomial_order, "Estimation failed: $e")
    end
end

end # module Causal
