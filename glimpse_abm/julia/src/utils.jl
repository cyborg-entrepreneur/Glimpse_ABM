"""
Utility functions for GlimpseABM.jl

Port of: glimpse_abm/utils.py
"""

using Random
using Statistics
using Distributions

# ============================================================================
# TRAIT SAMPLING
# ============================================================================

"""
Sample a trait value from a distribution specification.
"""
function sample_trait(dist_spec::TraitDistribution; rng::Random.AbstractRNG = Random.default_rng())::Float64
    dist_type = dist_spec.dist
    params = dist_spec.params

    if dist_type == "beta"
        a = get(params, "a", 1.0)
        b = get(params, "b", 1.0)
        return rand(rng, Beta(a, b))
    elseif dist_type == "uniform"
        low = get(params, "low", 0.0)
        high = get(params, "high", 1.0)
        return rand(rng, Uniform(low, high))
    elseif dist_type == "lognormal"
        mu = get(params, "mean", 0.0)
        sigma = get(params, "sigma", 1.0)
        # Clip to [0, 1] to match Python behavior
        return clamp(rand(rng, LogNormal(mu, sigma)), 0.0, 1.0)
    elseif dist_type == "normal_clipped"
        mean_val = get(params, "mean", 0.5)
        std_val = get(params, "std", 0.2)
        val = mean_val + std_val * randn(rng)
        return clamp(val, 0.0, 1.0)
    elseif dist_type == "normal"
        mean_val = get(params, "mean", 0.0)
        std_val = get(params, "std", 1.0)
        # Clip to [0, 1] to match Python behavior
        return clamp(mean_val + std_val * randn(rng), 0.0, 1.0)
    else
        # Default to uniform [0, 1]
        return rand(rng)
    end
end

"""
Sample all traits for an agent based on configuration.
"""
function sample_all_traits(config::EmergentConfig; rng::Random.AbstractRNG = Random.default_rng())::Dict{String,Float64}
    traits = Dict{String,Float64}()
    for (trait_name, dist_spec) in config.TRAIT_DISTRIBUTIONS
        traits[trait_name] = sample_trait(dist_spec; rng=rng)
    end
    return traits
end

# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

"""
Compute exponential moving average.
"""
function ema(values::Vector{Float64}, alpha::Float64)::Vector{Float64}
    if isempty(values)
        return Float64[]
    end
    result = similar(values)
    result[1] = values[1]
    for i in 2:length(values)
        result[i] = alpha * values[i] + (1.0 - alpha) * result[i-1]
    end
    return result
end

"""
Compute rolling mean with a window size.
"""
function rolling_mean(values::Vector{Float64}, window::Int)::Vector{Float64}
    n = length(values)
    if n == 0
        return Float64[]
    end
    result = similar(values)
    for i in 1:n
        start_idx = max(1, i - window + 1)
        result[i] = mean(@view values[start_idx:i])
    end
    return result
end

"""
Compute rolling standard deviation with a window size.
"""
function rolling_std(values::Vector{Float64}, window::Int)::Vector{Float64}
    n = length(values)
    if n == 0
        return Float64[]
    end
    result = similar(values)
    for i in 1:n
        start_idx = max(1, i - window + 1)
        window_vals = @view values[start_idx:i]
        result[i] = length(window_vals) > 1 ? std(window_vals) : 0.0
    end
    return result
end

# ============================================================================
# CLIPPING / BOUNDING
# ============================================================================

"""
Soft clip a value with smooth transition at boundaries.
"""
function soft_clip(x::Float64, low::Float64, high::Float64; sharpness::Float64 = 10.0)::Float64
    mid = (low + high) / 2.0
    span = (high - low) / 2.0
    return mid + span * tanh(sharpness * (x - mid) / span)
end

"""
Apply logistic sigmoid transformation.
"""
function logistic(x::Float64; k::Float64 = 1.0, x0::Float64 = 0.0)::Float64
    return 1.0 / (1.0 + exp(-k * (x - x0)))
end

"""
Numerically stable sigmoid function.
Matches Python's stable_sigmoid for uncertainty calculations.
"""
function stable_sigmoid(x::Float64)::Float64
    if !isfinite(x)
        return x > 0 ? 1.0 : 0.0
    end
    if x >= 0.0
        z = exp(-x)
        return 1.0 / (1.0 + z)
    else
        z = exp(x)
        return z / (1.0 + z)
    end
end

"""
Safe exponential that handles overflow.
"""
function safe_exp(x::Float64)::Float64
    if !isfinite(x)
        return x > 0 ? Inf : 0.0
    end
    # Clamp to prevent overflow
    x_clamped = clamp(x, -700.0, 700.0)
    return exp(x_clamped)
end

"""
Safe mean that handles empty collections and NaN values.
"""
function safe_mean(values)::Float64
    if isempty(values)
        return 0.0
    end
    # Filter out non-finite values
    finite_vals = filter(isfinite, values)
    if isempty(finite_vals)
        return 0.0
    end
    return mean(finite_vals)
end

"""
Fast mean optimized for collections (matches Python's fast_mean).
"""
function fast_mean(values)::Float64
    if isempty(values)
        return 0.0
    end
    s = 0.0
    n = 0
    for v in values
        if isfinite(v)
            s += v
            n += 1
        end
    end
    return n > 0 ? s / n : 0.0
end

"""
Helper to get field with default value.
"""
function getfield_default(obj, field::Symbol, default)
    return hasfield(typeof(obj), field) ? getfield(obj, field) : default
end

# ============================================================================
# OPPORTUNITY HELPERS
# ============================================================================

"""
Generate a unique opportunity ID.
"""
function generate_opportunity_id(round::Int, index::Int; prefix::String = "OPP")::String
    return "$(prefix)_R$(round)_$(index)"
end

"""
Generate a unique innovation ID.
"""
function generate_innovation_id(round::Int, creator_id::Int, index::Int)::String
    return "INN_R$(round)_A$(creator_id)_$(index)"
end

# ============================================================================
# ARRAY UTILITIES
# ============================================================================

"""
Safe division that returns 0 when denominator is 0.
"""
function safe_div(num::Float64, denom::Float64; default::Float64 = 0.0)::Float64
    return denom == 0.0 ? default : num / denom
end

"""
Normalize a vector to sum to 1 (probability distribution).
"""
function normalize_probs(probs::Vector{Float64})::Vector{Float64}
    total = sum(probs)
    if total <= 0.0
        n = length(probs)
        return n > 0 ? fill(1.0 / n, n) : Float64[]
    end
    return probs ./ total
end

"""
Weighted random choice from an array.
"""
function weighted_choice(items::Vector{T}, weights::Vector{Float64}; rng::Random.AbstractRNG = Random.default_rng()) where T
    if isempty(items)
        error("Cannot choose from empty array")
    end
    probs = normalize_probs(weights)
    cumsum_probs = cumsum(probs)
    r = rand(rng)
    idx = searchsortedfirst(cumsum_probs, r)
    return items[min(idx, length(items))]
end

# ============================================================================
# REGIME TRANSITIONS
# ============================================================================

"""
Sample next regime based on transition probabilities.
"""
function sample_next_regime(
    current_regime::String,
    transitions::Dict{String,Dict{String,Float64}};
    rng::Random.AbstractRNG = Random.default_rng()
)::String
    if !haskey(transitions, current_regime)
        return current_regime
    end
    probs = transitions[current_regime]
    regimes = collect(keys(probs))
    weights = [probs[r] for r in regimes]
    return weighted_choice(regimes, weights; rng=rng)
end

# ============================================================================
# HERFINDAHL-HIRSCHMAN INDEX
# ============================================================================

"""
Compute Herfindahl-Hirschman Index for concentration measurement.
"""
function compute_hhi(shares::Vector{Float64})::Float64
    if isempty(shares)
        return 0.0
    end
    total = sum(shares)
    if total <= 0.0
        return 0.0
    end
    normalized = shares ./ total
    return sum(normalized .^ 2)
end

# ============================================================================
# GINI COEFFICIENT
# ============================================================================

"""
Compute Gini coefficient for inequality measurement.
"""
function compute_gini(values::Vector{Float64})::Float64
    n = length(values)
    if n <= 1
        return 0.0
    end
    sorted_vals = sort(values)
    cumsum_vals = cumsum(sorted_vals)
    total = cumsum_vals[end]
    if total <= 0.0
        return 0.0
    end
    # Gini = 1 - 2 * (area under Lorenz curve)
    # Area under Lorenz = sum(cumsum) / (n * total)
    lorenz_area = sum(cumsum_vals) / (n * total)
    return 1.0 - 2.0 * lorenz_area + 1.0 / n
end

# ============================================================================
# PROFILING UTILITIES
# ============================================================================

"""
Profile a simulation run using Julia's built-in profiler.

Example usage:
```julia
using Profile
config = EmergentConfig()
config.N_AGENTS = 500
config.N_ROUNDS = 100

# Profile the simulation
profile_simulation(config; output_dir="./profiled_run")
```
"""
function profile_simulation(
    config::EmergentConfig;
    output_dir::String="./profiled_run",
    run_id::String="profiled_run",
    seed::Int=42
)
    mkpath(output_dir)
    println("Profiling simulation with $(config.N_AGENTS) agents, $(config.N_ROUNDS) rounds...")

    # Clear any previous profile data
    Profile = Base.require(Base.PkgId(Base.UUID("9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"), "Profile"))

    try
        Profile.clear()
    catch
        # Profile module might not be loaded
        println("Note: For detailed profiling, use `using Profile` and `@profile`")
    end

    # Create and run simulation
    sim = EmergentSimulation(
        config=config,
        output_dir=output_dir,
        run_id=run_id,
        seed=seed
    )

    start_time = time()
    run!(sim)
    elapsed = time() - start_time

    println("Simulation completed in $(round(elapsed, digits=2)) seconds")
    println("Results saved to: $output_dir")

    return Dict{String,Any}(
        "elapsed_seconds" => elapsed,
        "output_dir" => output_dir,
        "n_agents" => config.N_AGENTS,
        "n_rounds" => config.N_ROUNDS
    )
end

# ============================================================================
# AI LEVEL NORMALIZATION
# ============================================================================

"""
Canonical AI level labels.
"""
const CANONICAL_AI_LEVELS = ["none", "basic", "advanced", "premium"]

"""
Mapping from various AI level representations to canonical form.
"""
const AI_LABEL_MAP = Dict{String,String}(
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
Normalize AI level label to canonical form.

Handles various input formats:
- "human", "human_only", "no_ai", "none" → "none"
- "basic", "basic_ai", "standard" → "basic"
- "advanced", "advanced_ai" → "advanced"
- "premium", "premium_ai", "full" → "premium"
"""
function normalize_ai_label(label::Union{String,Missing,Nothing})::String
    if ismissing(label) || isnothing(label)
        return "none"
    end
    label_lower = lowercase(strip(string(label)))
    return get(AI_LABEL_MAP, label_lower, "none")
end
