#!/usr/bin/env julia
# Phase 5: Information System Validation

using Pkg
Pkg.activate(dirname(@__DIR__))

using GlimpseABM
using Statistics

println("=" ^ 70)
println("PHASE 5: INFORMATION SYSTEM VALIDATION")
println("=" ^ 70)

config = EmergentConfig()

# 5.1 AI Information Generation
println("\n5.1 AI INFORMATION GENERATION")
println("-" ^ 50)

println("\n[Domain-Specific Accuracy by Tier]")
domains = ["market_analysis", "technical_assessment", "uncertainty_evaluation", "innovation_potential"]
for tier in ["none", "basic", "advanced", "premium"]
    println("\n  $tier tier:")
    caps = config.AI_DOMAIN_CAPABILITIES[tier]
    for domain in domains
        cap = caps[domain]
        println("    $domain: accuracy=$(cap.accuracy), halluc=$(cap.hallucination_rate), bias=$(cap.bias)")
    end
end

println("\n[Hallucination Rate Calculation]")
println("  3-step chain (matches Python Enhanced):")
println("    Step 1: Base rate from domain capability")
println("    Step 2: Stochastic modification via get_stochastic_hallucination_rate")
println("    Step 3: Lognormal variance (sigma = 0.25 * tier_noise)")
println("    Step 4: Intensity scaling by HALLUCINATION_INTENSITY = $(config.HALLUCINATION_INTENSITY)")

println("\n[Tier Noise Factor]")
println("  tier_noise = max(0.1, 1.2 - info_quality)")
for tier in ["none", "basic", "advanced", "premium"]
    info_quality = config.AI_LEVELS[tier].info_quality
    tier_noise = max(0.1, 1.2 - info_quality)
    println("    $tier: info_quality=$info_quality, tier_noise=$(round(tier_noise, digits=2))")
end

println("\n[Overconfidence Factor]")
println("  Scaled by OVERCONFIDENCE_INTENSITY = $(config.OVERCONFIDENCE_INTENSITY)")
println("  base_overconfidence = info_quality < 0.5 ? (0.5 - info_quality) * 0.5 : 0.0")

# 5.2 Human Information
println("\n5.2 HUMAN INFORMATION")
println("-" ^ 50)

println("\n[Trait-Based Quality Calculation]")
println("  quality = (analytical_ability * 0.4 + competence * 0.4 + market_awareness * 0.2) * 0.30")
println("  breadth = (exploration_tendency * 0.5 + market_awareness * 0.5) * 0.25")

println("\n[Noise Levels]")
println("  Return noise: randn() * (1 - quality) * 0.7")
println("  Uncertainty noise: randn() * (1 - quality) * 0.5")

println("\n[Confidence Caps]")
println("  Human confidence: clamp(competence * 0.4, 0.05, 0.45)")

# Test information generation
println("\n5.3 TESTING INFORMATION GENERATION")
println("-" ^ 50)

# Create an information system
info_sys = InformationSystem(config)

# Create a test opportunity
using Random
rng = MersenneTwister(42)

# Test opportunity
opp = Opportunity(
    id="test_opp_1",
    sector="tech",
    latent_return_potential=1.5,
    latent_failure_potential=0.3,
    lifecycle_stage="growth",
    complexity=0.6,
    competition=0.4,
    path_dependency=0.5,
    round_created=1,
    source="market",
    created_by=nothing,
    required_knowledge=String[],
    capital_requirement=100000.0,
    time_to_maturity=12
)

println("\nTest Opportunity:")
println("  Sector: $(opp.sector)")
println("  Latent return: $(opp.latent_return_potential)")
println("  Latent failure: $(opp.latent_failure_potential)")
println("  Complexity: $(opp.complexity)")

println("\nInformation Quality by Tier:")
for tier in ["none", "basic", "advanced", "premium"]
    info = get_information(info_sys, opp, tier; rng=MersenneTwister(42))
    println("  $tier:")
    println("    Estimated return: $(round(info.estimated_return, digits=3))")
    println("    Estimated uncertainty: $(round(info.estimated_uncertainty, digits=3))")
    println("    Confidence: $(round(info.confidence, digits=3))")
    println("    Actual accuracy: $(round(info.actual_accuracy, digits=3))")
    println("    Contains hallucination: $(info.contains_hallucination)")
end

println("\n" * "=" ^ 70)
println("PHASE 5 VALIDATION COMPLETE")
println("=" ^ 70)
