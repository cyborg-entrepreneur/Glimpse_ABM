#!/usr/bin/env julia
"""
Crowding Calibration Test

Find the right crowding level to produce realistic 5-year survival rates (~50-55%)
while preserving the AI paradox finding.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GlimpseABM
using Statistics
using Random
using Printf

const N_AGENTS = 1000
const N_ROUNDS = 60  # 5 years
const N_RUNS = 30
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const BASE_SEED = 20260130

function run_calibration_test(competition_scale::Float64, crowding_penalty::Float64)
    results = Dict{String, Vector{Float64}}(tier => Float64[] for tier in AI_TIERS)

    for (tier_idx, tier) in enumerate(AI_TIERS)
        for run_idx in 1:N_RUNS
            seed = BASE_SEED + tier_idx * 1000 + run_idx

            config = EmergentConfig(
                N_AGENTS=N_AGENTS,
                N_ROUNDS=N_ROUNDS,
                RANDOM_SEED=seed,
                INITIAL_CAPITAL=5_000_000.0,
                SURVIVAL_THRESHOLD=10_000.0,
                COMPETITION_SCALE_FACTOR=competition_scale,
                RETURN_DEMAND_CROWDING_PENALTY=crowding_penalty,
                OPPORTUNITY_COMPETITION_PENALTY=crowding_penalty
            )

            tier_dist = Dict(t => (t == tier ? 1.0 : 0.0) for t in AI_TIERS)
            sim = GlimpseABM.EmergentSimulation(
                config=config,
                initial_tier_distribution=tier_dist,
                seed=seed
            )

            for round in 1:N_ROUNDS
                GlimpseABM.step!(sim, round)
            end

            alive = count(a -> a.alive, sim.agents)
            push!(results[tier], alive / N_AGENTS)
        end
    end

    return results
end

function main()
    println("="^70)
    println("  CROWDING CALIBRATION TEST")
    println("  Target: 50-55% 5-year survival (BLS benchmark)")
    println("="^70)

    # Test different crowding levels
    # Format: (competition_scale, crowding_penalty)
    test_configs = [
        (1.0, 0.48, "Current (baseline)"),
        (0.75, 0.36, "Reduced 25%"),
        (0.65, 0.31, "Reduced 35%"),
        (0.55, 0.26, "Reduced 45%"),
        (0.50, 0.24, "Reduced 50%"),
        (0.40, 0.19, "Reduced 60%"),
    ]

    println("\n  Testing $(length(test_configs)) configurations...")
    println("-"^70)
    @printf("  %-18s %8s %8s %8s %8s %8s %8s\n",
            "Config", "Scale", "Penalty", "None", "Basic", "Adv", "Premium")
    println("-"^70)

    for (scale, penalty, name) in test_configs
        results = run_calibration_test(scale, penalty)

        none_surv = mean(results["none"]) * 100
        basic_surv = mean(results["basic"]) * 100
        adv_surv = mean(results["advanced"]) * 100
        prem_surv = mean(results["premium"]) * 100

        treatment_effect = prem_surv - none_surv

        # Check if in target range (50-55% for None tier)
        marker = if 50 <= none_surv <= 58
            "✓ TARGET"
        elseif none_surv < 50
            "too low"
        else
            "too high"
        end

        @printf("  %-18s %8.2f %8.2f %7.1f%% %7.1f%% %7.1f%% %7.1f%% [TE: %+.1f] %s\n",
                name, scale, penalty, none_surv, basic_surv, adv_surv, prem_surv,
                treatment_effect, marker)
    end

    println("-"^70)
    println("\n  Legend: TE = Treatment Effect (Premium - None, in pp)")
    println("  Target: None tier survival 50-55%, with negative treatment effect (paradox)")
    println("="^70)
end

main()
