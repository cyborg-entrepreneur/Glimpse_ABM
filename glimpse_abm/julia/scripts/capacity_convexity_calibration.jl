#!/usr/bin/env julia
"""
Capacity-Convexity Crowding Model Calibration

Calibrate K (capacity), γ (convexity), λ (strength) to achieve:
1. ~50-55% 5-year survival for None tier (BLS benchmark)
2. Negative treatment effect (preserve AI paradox)

Formula: penalty = λ · max(0, C/K - 1)^γ
         net_return = base_return · exp(-penalty)

Usage:
    julia --threads=auto --project=. scripts/capacity_convexity_calibration.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GlimpseABM
using Statistics
using Random
using Printf
using Dates

const N_AGENTS = 1000
const N_ROUNDS = 60  # 5 years
const N_RUNS = 25    # Fewer runs for faster iteration
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const BASE_SEED = 20260131

const OUTPUT_DIR = joinpath(@__DIR__, "..", "results", "crowding_calibration_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")

struct CrowdingConfig
    K::Float64      # Capacity
    gamma::Float64  # Convexity
    lambda::Float64 # Strength
    name::String
end

function run_calibration_test(cfg::CrowdingConfig)
    results = Dict{String, Vector{Float64}}(tier => Float64[] for tier in AI_TIERS)

    Threads.@threads for idx in 1:(N_RUNS * length(AI_TIERS))
        tier_idx = ((idx - 1) ÷ N_RUNS) + 1
        run_idx = ((idx - 1) % N_RUNS) + 1
        tier = AI_TIERS[tier_idx]
        seed = BASE_SEED + tier_idx * 1000 + run_idx

        config = EmergentConfig(
            N_AGENTS=N_AGENTS,
            N_ROUNDS=N_ROUNDS,
            RANDOM_SEED=seed,
            INITIAL_CAPITAL=5_000_000.0,
            SURVIVAL_THRESHOLD=10_000.0,
            USE_CAPACITY_CONVEXITY_CROWDING=true,
            CROWDING_CAPACITY_K=cfg.K,
            CROWDING_CONVEXITY_GAMMA=cfg.gamma,
            CROWDING_STRENGTH_LAMBDA=cfg.lambda
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

    return results
end

function main()
    println("="^80)
    println("  CAPACITY-CONVEXITY CROWDING MODEL CALIBRATION")
    println("="^80)
    println("\n  Formula: penalty = λ · max(0, C/K - 1)^γ")
    println("           net_return = base_return · exp(-penalty)")
    println("\n  Target: 50-55% 5-year survival, negative treatment effect")
    println("="^80)

    mkpath(OUTPUT_DIR)

    # Test configurations
    # Based on calibration results:
    # - Old "Reduced 35%" gave 56.6% survival with -19.3 pp TE
    # - We want similar results with the new model
    #
    # Strategy:
    # - K controls when penalty starts (higher K = more tolerance for crowding)
    # - γ controls sharpness (γ=2 is good default)
    # - λ controls severity (higher λ = stronger penalty at same crowding level)

    test_configs = [
        # Explore K (capacity) at fixed γ=2, λ=0.5
        CrowdingConfig(1.0, 2.0, 0.50, "K=1.0, γ=2, λ=0.50"),
        CrowdingConfig(1.5, 2.0, 0.50, "K=1.5, γ=2, λ=0.50"),
        CrowdingConfig(2.0, 2.0, 0.50, "K=2.0, γ=2, λ=0.50"),
        CrowdingConfig(2.5, 2.0, 0.50, "K=2.5, γ=2, λ=0.50"),

        # Explore λ (strength) at K=1.5, γ=2
        CrowdingConfig(1.5, 2.0, 0.35, "K=1.5, γ=2, λ=0.35"),
        CrowdingConfig(1.5, 2.0, 0.70, "K=1.5, γ=2, λ=0.70"),
        CrowdingConfig(1.5, 2.0, 1.00, "K=1.5, γ=2, λ=1.00"),

        # Explore γ (convexity) at K=1.5, λ=0.5
        CrowdingConfig(1.5, 1.5, 0.50, "K=1.5, γ=1.5, λ=0.50"),
        CrowdingConfig(1.5, 2.5, 0.50, "K=1.5, γ=2.5, λ=0.50"),
        CrowdingConfig(1.5, 3.0, 0.50, "K=1.5, γ=3.0, λ=0.50"),

        # Fine-tuning based on expected sweet spots
        CrowdingConfig(1.2, 2.0, 0.60, "K=1.2, γ=2.0, λ=0.60"),
        CrowdingConfig(1.3, 2.0, 0.55, "K=1.3, γ=2.0, λ=0.55"),
        CrowdingConfig(1.4, 2.0, 0.45, "K=1.4, γ=2.0, λ=0.45"),
    ]

    println("\n  Testing $(length(test_configs)) configurations...")
    println("-"^80)
    @printf("  %-25s %8s %8s %8s %8s %8s\n",
            "Config", "None", "Basic", "Adv", "Premium", "TE (pp)")
    println("-"^80)

    best_config = nothing
    best_score = Inf

    for cfg in test_configs
        results = run_calibration_test(cfg)

        none_surv = mean(results["none"]) * 100
        basic_surv = mean(results["basic"]) * 100
        adv_surv = mean(results["advanced"]) * 100
        prem_surv = mean(results["premium"]) * 100
        treatment_effect = prem_surv - none_surv

        # Score: how close to target (50-55% survival, TE < 0)
        target_survival = 52.5  # Middle of target range
        survival_error = abs(none_surv - target_survival)
        te_penalty = treatment_effect > 0 ? 100.0 : 0.0  # Heavy penalty if paradox reversed
        score = survival_error + te_penalty

        # Check if in target range
        marker = if 50 <= none_surv <= 58 && treatment_effect < 0
            "✓ TARGET"
        elseif none_surv < 50
            "too low"
        elseif none_surv > 58
            "too high"
        elseif treatment_effect > 0
            "⚠ reversed"
        else
            ""
        end

        if score < best_score
            best_score = score
            best_config = (cfg, none_surv, treatment_effect)
        end

        @printf("  %-25s %7.1f%% %7.1f%% %7.1f%% %7.1f%% %+7.1f %s\n",
                cfg.name, none_surv, basic_surv, adv_surv, prem_surv,
                treatment_effect, marker)
    end

    println("-"^80)

    if !isnothing(best_config)
        cfg, surv, te = best_config
        println("\n  BEST CONFIGURATION:")
        println("    K (capacity):    $(cfg.K)")
        println("    γ (convexity):   $(cfg.gamma)")
        println("    λ (strength):    $(cfg.lambda)")
        println("    None survival:   $(round(surv, digits=1))%")
        println("    Treatment effect: $(round(te, digits=1)) pp")
    end

    println("\n  INTERPRETATION:")
    println("    K = Capacity: crowding level where penalties START")
    println("        Higher K = more tolerance for competition")
    println("    γ = Convexity: how sharply penalties increase")
    println("        γ=2 means 'a little crowded is OK, very crowded is brutal'")
    println("    λ = Strength: severity of penalty at given crowding")
    println("        At 2× capacity: return multiplier = exp(-λ)")
    println("="^80)
end

main()
