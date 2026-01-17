"""
GlimpseABM.jl - Julia port of the GLIMPSE Agent-Based Model

A high-performance Julia implementation of the GLIMPSE ABM for studying
AI adoption and Knightian uncertainty in entrepreneurial ecosystems.

Theoretical Foundation
----------------------
This model operationalizes concepts from:

    Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).
    Are the futures computable? Knightian uncertainty & artificial intelligence.
    Academy of Management Review, 50(2), 415-440.

The four dimensions of Knightian uncertainty modeled are:
1. Actor Ignorance - Information gaps about current states
2. Practical Indeterminism - Unpredictable execution outcomes
3. Agentic Novelty - Genuinely new possibilities from creative action
4. Competitive Recursion - Strategic interdependence effects

Author: David Townsend
License: MIT
"""
module GlimpseABM

using Random
using Statistics
using LinearAlgebra
using Distributions
using DataFrames
using Dates

# Core configuration and models
include("config.jl")
include("models.jl")

# Knowledge and information systems
include("knowledge.jl")
include("information.jl")

# Innovation system (before agents.jl - uses Any for agent type to avoid circular dep)
include("innovation.jl")

# Simulation components
include("market.jl")
include("uncertainty.jl")
include("agents.jl")
include("simulation.jl")

# Utilities
include("utils.jl")
include("io.jl")

# Causal inference module
include("../causal/causal.jl")
using .Causal

# Analysis module
include("../analysis/analysis.jl")
using .Analysis

# Visualization module (loaded conditionally)
include("../analysis/visualization.jl")
using .Visualization

# CLI module
include("../cli/cli.jl")
using .CLI

# Exports - Configuration
export EmergentConfig
export CalibrationProfile
export apply_calibration_profile
export get_calibration_profile
export load_calibration_profile
export CALIBRATION_LIBRARY

# Exports - Models
export Opportunity
export Information
export Innovation
export Knowledge
export AIAnalysis
export AILearningProfile

# Exports - Knowledge and Information Systems
export KnowledgeBase
export InformationSystem
export InnovationEngine
export CombinationTracker
export get_accessible_knowledge
export get_information
export attempt_innovation!
export get_component_scarcity_metric
export learn_from_success!, learn_from_failure!
export reinforce_agent_resources!

# Exports - Simulation
export EmergentSimulation
export EmergentAgent
export VectorizedAgentState
export MarketEnvironment
export KnightianUncertaintyEnvironment

# Exports - Functions
export run!
export step!
export initialize_agents
export save_results
export load_results

# Exports - Utility functions
export stable_sigmoid, safe_exp, safe_mean, fast_mean
export normalize_ai_label, compute_hhi, compute_gini
export perceive_uncertainty, measure_uncertainty_state!

# Exports - Causal inference
export cohens_d, cliffs_delta, glass_delta
export anova_oneway, kruskal_wallis
export bootstrap_ci, permutation_test
export mann_whitney_u, survival_analysis
export EffectSizeResult, ANOVAResult, SurvivalResult
export CoxRegressionResult, PropensityScoreResult, DiDResult, RDResult
export kaplan_meier_curves, log_rank_test
export estimate_propensity_scores, propensity_score_matching, inverse_probability_weighting
export difference_in_differences, event_study
export regression_discontinuity

# Exports - Statistical testing (rigorous analysis)
export StatisticalTestResult, MixedEffectsResult, CausalEffectEstimate
export set_fast_stats_mode, get_bootstrap_iterations
export cohens_d_with_ci, eta_squared, epsilon_squared, cramers_v
export test_normality, test_homogeneity
export kruskal_wallis_test, mann_whitney_u_test, chi_square_test
export welch_ttest, spearman_correlation
export benjamini_hochberg, holm_bonferroni
export bootstrap_ci_stat, compute_ate_bootstrap
export descriptive_stats, significance_stars, format_p_value

# Exports - Analysis
export AnalysisFramework, run_full_analysis
export compute_survival_summary, compute_capital_summary
export compute_ai_tier_comparison, aggregate_sweep_results

# Exports - Visualization
export plot_survival_by_tier, plot_capital_distribution
export plot_ai_adoption_over_time, plot_uncertainty_dynamics
export plot_effect_sizes, plot_survival_curves
export create_summary_dashboard, save_all_figures

# Exports - CLI
export run_cli, parse_args
export run_master_launcher, run_fixed_level_sweep

end # module
