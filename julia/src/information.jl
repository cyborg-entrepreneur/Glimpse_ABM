"""
Information system components for GlimpseABM.jl

Provides AI-augmented information generation about opportunities,
including hallucination modeling and domain-specific accuracy.

Port of: glimpse_abm/information.py
"""

using Random
using Statistics

# ============================================================================
# INFORMATION SYSTEM
# ============================================================================

"""
System for generating information about opportunities.
"""
mutable struct InformationSystem
    config::EmergentConfig
    information_cache::Dict{Tuple{String,String},Information}
    discovered_opportunities::Set{String}
    cache_hits::Int
    cache_misses::Int
    domain_performance::Dict{String,Vector{Float64}}
end

"""
Create a new InformationSystem.
"""
function InformationSystem(config::EmergentConfig)
    return InformationSystem(
        config,
        Dict{Tuple{String,String},Information}(),
        Set{String}(),
        0,
        0,
        Dict{String,Vector{Float64}}(
            "market_analysis" => Float64[],
            "technical_assessment" => Float64[],
            "uncertainty_evaluation" => Float64[],
            "innovation_potential" => Float64[]
        )
    )
end

"""
Determine the analysis domain for an opportunity.
"""
function determine_domain(opp::Opportunity)::String
    if !isnothing(opp.created_by)
        return "innovation_potential"
    end
    if opp.complexity > 0.7
        return "technical_assessment"
    end
    if opp.sector in ["tech", "manufacturing"]
        return "technical_assessment"
    end
    if opp.competition > 0.5
        return "market_analysis"
    end
    return "uncertainty_evaluation"
end

"""
Generate insights based on information breadth.
"""
function generate_insights(
    opp::Opportunity,
    breadth::Float64,
    domain::Union{String,Nothing}=nothing
)::Vector{String}
    insights = String[]

    if breadth > 0.2
        push!(insights, "Market lifecycle: $(opp.lifecycle_stage)")
    end

    if breadth > 0.4
        if opp.competition > 0.5
            push!(insights, "High competitive pressure detected")
        else
            push!(insights, "Limited competition currently")
        end
    end

    if breadth > 0.6
        if opp.path_dependency > 0.5
            push!(insights, "Strong first-mover advantages")
        end
        if opp.complexity > 0.7
            push!(insights, "Requires specialized capabilities")
        end
        push!(insights, "Sector: $(opp.sector)")
    end

    if breadth > 0.8
        push!(insights, "Hidden uncertainty factors: ~$(round((1-breadth)*opp.complexity*100, digits=1))%")
        if !isnothing(opp.created_by)
            push!(insights, "Novel opportunity (agent-created)")
        end
        if !isnothing(domain)
            push!(insights, "Analysis domain: $domain")
        end
    end

    return insights
end

"""
Generate false insights for hallucination modeling.
"""
function generate_false_insights(domain::String; rng::AbstractRNG=Random.default_rng())::Vector{String}
    false_insight_templates = Dict(
        "market_analysis" => [
            "Untapped customer segment identified in emerging markets",
            "Significant demand spike detected through alternative data",
            "Competitor withdrawal likely within 3 quarters"
        ],
        "technical_assessment" => [
            "Breakthrough resolves key scalability challenges",
            "Prototype efficiency exceeds market benchmarks",
            "Regulatory path streamlined due to new standards"
        ],
        "uncertainty_evaluation" => [
            "Black swan likelihood reduced by recent policy changes",
            "Risk contagion limited to adjacent sectors",
            "Dominant uncertainty source neutralized by supplier agreement"
        ],
        "innovation_potential" => [
            "High cross-domain synergy with existing portfolio",
            "Productized knowledge base accelerates time-to-market",
            "Community adoption hurdle lower than industry peers"
        ]
    )

    domain_insights = get(false_insight_templates, domain, String[])
    n_false = min(2, length(domain_insights))

    if n_false > 0 && !isempty(domain_insights)
        return domain_insights[randperm(rng, length(domain_insights))[1:n_false]]
    end
    return String[]
end

"""
Get information about an opportunity for a given AI level.
"""
function get_information(
    sys::InformationSystem,
    opp::Opportunity,
    ai_level::String;
    agent_id::Union{Int,Nothing}=nothing,
    rng::AbstractRNG=Random.default_rng()
)::Information
    cache_key = (opp.id, ai_level)

    # Check cache
    if haskey(sys.information_cache, cache_key)
        sys.cache_hits += 1
        return sys.information_cache[cache_key]
    end

    sys.cache_misses += 1

    # Get AI configuration
    ai_config = get(sys.config.AI_LEVELS, ai_level, sys.config.AI_LEVELS["none"])
    domain = determine_domain(opp)
    domain_cap = get_ai_domain_capability(sys.config, ai_level, domain)

    # Compute accuracy with noise
    base_accuracy = domain_cap["accuracy"]
    actual_accuracy = clamp(randn(rng) * 0.1 + base_accuracy, 0.0, 1.0)

    # Scale hallucination rate by intensity parameter (for robustness testing)
    hallucination_intensity = sys.config.HALLUCINATION_INTENSITY
    hallucination_rate = domain_cap["hallucination_rate"] * hallucination_intensity
    bias = domain_cap["bias"]

    # Estimate return with noise
    return_noise = randn(rng) * (1 - actual_accuracy) * 0.5
    estimated_return = opp.latent_return_potential + return_noise + bias * 0.3
    estimated_return = clamp(estimated_return,
        sys.config.OPPORTUNITY_RETURN_RANGE[1],
        sys.config.OPPORTUNITY_RETURN_RANGE[2])

    # Estimate uncertainty with noise
    uncertainty_noise = randn(rng) * (1 - actual_accuracy) * 0.3
    estimated_uncertainty = opp.latent_failure_potential + uncertainty_noise - bias * 0.2
    estimated_uncertainty = clamp(estimated_uncertainty,
        sys.config.OPPORTUNITY_UNCERTAINTY_RANGE[1],
        sys.config.OPPORTUNITY_UNCERTAINTY_RANGE[2])

    # Check for hallucination
    contains_hallucination = rand(rng) < hallucination_rate
    if contains_hallucination
        if rand(rng) < 0.5
            estimated_return = rand(rng) * (sys.config.OPPORTUNITY_RETURN_RANGE[2] -
                sys.config.OPPORTUNITY_RETURN_RANGE[1]) + sys.config.OPPORTUNITY_RETURN_RANGE[1]
        else
            estimated_uncertainty = rand(rng) * (sys.config.OPPORTUNITY_UNCERTAINTY_RANGE[2] -
                sys.config.OPPORTUNITY_UNCERTAINTY_RANGE[1]) + sys.config.OPPORTUNITY_UNCERTAINTY_RANGE[1]
        end
    end

    # Compute confidence
    info_quality = get(ai_config, "info_quality", 0.0)
    true_confidence = actual_accuracy * (1 - opp.complexity * (1 - actual_accuracy))
    # Scale overconfidence by intensity parameter (for robustness testing)
    overconfidence_intensity = sys.config.OVERCONFIDENCE_INTENSITY
    base_overconfidence = info_quality < 0.5 ? (0.5 - info_quality) * 0.5 : 0.0
    overconfidence_factor = 1.0 + base_overconfidence * overconfidence_intensity
    stated_confidence = clamp(true_confidence * overconfidence_factor, 0.1, 0.95)

    # Generate insights
    info_breadth = get(ai_config, "info_breadth", 0.0)
    insights = generate_insights(opp, info_breadth, domain)

    # Add false insights if hallucinating
    if contains_hallucination
        append!(insights, generate_false_insights(domain; rng=rng))
    end

    # Create information object
    info = Information(
        estimated_return=estimated_return,
        estimated_uncertainty=estimated_uncertainty,
        confidence=stated_confidence,
        insights=insights,
        hidden_factors=Dict{String,Any}(
            "bias" => bias,
            "unknown_uncertainty" => (1 - info_breadth) * opp.complexity,
            "market_shift_sensitivity" => 1 - actual_accuracy,
            "hallucination_uncertainty" => Float64(contains_hallucination)
        ),
        domain=domain,
        actual_accuracy=actual_accuracy,
        contains_hallucination=contains_hallucination,
        bias_applied=bias,
        overconfidence_factor=overconfidence_factor
    )

    sys.information_cache[cache_key] = info
    return info
end

"""
Get human-only information (no AI assistance).
"""
function get_human_information(
    sys::InformationSystem,
    opp::Opportunity,
    agent_traits::Dict{String,Float64};
    rng::AbstractRNG=Random.default_rng()
)::Information
    # Human quality based on traits
    quality = (
        get(agent_traits, "analytical_ability", 0.5) * 0.4 +
        get(agent_traits, "competence", 0.5) * 0.4 +
        get(agent_traits, "market_awareness", 0.5) * 0.2
    ) * 0.30

    breadth = (
        get(agent_traits, "exploration_tendency", 0.5) * 0.5 +
        get(agent_traits, "market_awareness", 0.5) * 0.5
    ) * 0.25

    # Estimate with higher noise for humans
    return_noise = randn(rng) * (1 - quality) * 0.7
    estimated_return = opp.latent_return_potential + return_noise

    uncertainty_noise = randn(rng) * (1 - quality) * 0.5
    estimated_uncertainty = opp.latent_failure_potential + uncertainty_noise

    confidence = clamp(get(agent_traits, "competence", 0.5) * 0.4, 0.05, 0.45)

    insights = generate_insights(opp, breadth, nothing)

    return Information(
        estimated_return=clamp(estimated_return,
            sys.config.OPPORTUNITY_RETURN_RANGE[1],
            sys.config.OPPORTUNITY_RETURN_RANGE[2]),
        estimated_uncertainty=clamp(estimated_uncertainty,
            sys.config.OPPORTUNITY_UNCERTAINTY_RANGE[1],
            sys.config.OPPORTUNITY_UNCERTAINTY_RANGE[2]),
        confidence=confidence,
        insights=insights,
        hidden_factors=Dict{String,Any}(
            "human_bias" => 0.5 - get(agent_traits, "analytical_ability", 0.5),
            "unknowns" => 1 - quality
        ),
        domain=nothing,
        actual_accuracy=quality,
        contains_hallucination=false,
        bias_applied=0.0,
        overconfidence_factor=1.0
    )
end

"""
Clear the information cache.
"""
function clear_cache!(sys::InformationSystem)
    empty!(sys.information_cache)
end

# ============================================================================
# ENHANCED AI ANALYSIS (Numba-equivalent in Julia)
# ============================================================================

"""
Get enhanced AI analysis (vectorized computation).
"""
function get_enhanced_ai_analysis(
    latent_return_potential::Float64,
    latent_failure_potential::Float64,
    complexity::Float64,
    base_accuracy::Float64,
    base_quality::Float64,
    hallucination_rate::Float64,
    bias::Float64,
    return_range::Tuple{Float64,Float64},
    uncertainty_range::Tuple{Float64,Float64};
    rng::AbstractRNG=Random.default_rng()
)
    actual_accuracy = clamp(randn(rng) * 0.1 + base_accuracy, 0.0, 1.0)

    return_noise = randn(rng) * (1 - actual_accuracy) * 0.5
    estimated_return = latent_return_potential + return_noise + bias * 0.3

    uncertainty_noise = randn(rng) * (1 - actual_accuracy) * 0.3
    estimated_uncertainty = latent_failure_potential + uncertainty_noise - bias * 0.2

    contains_hallucination = rand(rng) < hallucination_rate
    if contains_hallucination
        if rand(rng) < 0.5
            estimated_return = rand(rng) * (return_range[2] - return_range[1]) + return_range[1]
        else
            estimated_uncertainty = rand(rng) * (uncertainty_range[2] - uncertainty_range[1]) + uncertainty_range[1]
        end
    end

    estimated_return = clamp(estimated_return, return_range[1], return_range[2])
    estimated_uncertainty = clamp(estimated_uncertainty, uncertainty_range[1], uncertainty_range[2])

    true_confidence = actual_accuracy * (1 - complexity * (1 - actual_accuracy))
    overconfidence_factor = base_quality < 0.5 ? 1.0 + (0.5 - base_quality) * 0.5 : 1.0
    stated_confidence = clamp(true_confidence * overconfidence_factor, 0.1, 0.95)

    return (estimated_return, estimated_uncertainty, stated_confidence, contains_hallucination)
end

"""
Get stochastic hallucination rate with domain-specific adjustments.
"""
function get_stochastic_hallucination_rate(
    base_rate::Float64,
    domain::String;
    rng::AbstractRNG=Random.default_rng()
)::Float64
    # Beta distribution parameters based on base rate
    if base_rate < 0.1
        alpha, beta = 2, 20
    elseif base_rate < 0.2
        alpha, beta = 3, 12
    else
        alpha, beta = 4, 8
    end

    stochastic_factor = rand(rng)
    stochastic_rate = base_rate * (0.5 + stochastic_factor)

    # Random streak effect
    if rand(rng) < 0.1
        streak = rand(rng) * 0.4 + 0.8  # 0.8 to 1.2
        stochastic_rate *= streak
    end

    return clamp(stochastic_rate, 0.0, 0.5)
end
