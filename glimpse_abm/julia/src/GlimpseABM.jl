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
using Random: AbstractRNG  # Explicitly import to avoid ambiguity with RandomNumbers
using Statistics
using LinearAlgebra
using Distributions
using DataFrames
using Dates

# Action-dict field name constants (centralizes producer/consumer keys
# that have caused silent dataflow bugs — see action_keys.jl docstring)
include("action_keys.jl")

# Core configuration and models
include("config.jl")
# MarketConditions typed payload (v3.0) — loaded before models.jl because
# realized_return consumes it. See market_conditions.jl for rationale.
include("market_conditions.jl")
include("models.jl")

# Utilities (loaded early for stable_sigmoid and other helpers)
include("utils.jl")

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

# I/O utilities
include("io.jl")

# NumPy-compatible RNG for cross-language reproducibility
include("numpy_rng.jl")

# Exports - Configuration
export EmergentConfig
export MarketConditions
export CalibrationProfile
export apply_calibration_profile
export get_calibration_profile
export load_calibration_profile
export CALIBRATION_LIBRARY
export initialize!, get_scaled_opportunities

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
export evaluate_innovation_success!
export get_component_scarcity_metric
export learn_from_success!, learn_from_failure!
export reinforce_agent_resources!

# Exports - Simulation
export EmergentSimulation
# EnhancedSimulation removed in v2.1: it was a stale dual-path that crashed
# at runtime (undefined update_burn_history!) and lacked v2/v2.1 wirings.
# Use EmergentSimulation for all production runs.
export EmergentAgent
export VectorizedAgentState
export MarketEnvironment
export KnightianUncertaintyEnvironment

# Exports - Functions
export run!
export step!
export initialize_agents!  # v2.7: removed `initialize_agents` (non-bang form never existed)
export save_results
export load_results

# Exports - Agent distress tracking
# v2.7: removed evaluate_failure_conditions! (Python-only) and update_burn_history!
# (deleted with EnhancedSimulation in v2.2). Neither existed in Julia.
export check_survival!
export get_capital, set_capital!, get_ai_level

# Exports - Emergent uncertainty (agent-level metrics)
export AgentUncertaintyMetrics
export get_emergent_uncertainty, compute_emergent_uncertainty
export aggregate_emergent_uncertainty_by_tier
export record_investment_outcome!, record_creative_action!

# Exports - AI subscription charging
export ensure_subscription_schedule!, start_subscription_schedule!
export charge_subscription_installment!, apply_subscription_carry!

# Exports - Utility functions
export stable_sigmoid, safe_exp, safe_mean, fast_mean
export normalize_ai_label, compute_hhi, compute_gini
export perceive_uncertainty, measure_uncertainty_state!, get_uncertainty_state

# Exports - NumPy-compatible RNG
export NumpyRNG, numpy_rand, numpy_randn, numpy_randint, numpy_seed!
export numpy_gamma, numpy_beta, numpy_uniform, numpy_exponential

end # module
