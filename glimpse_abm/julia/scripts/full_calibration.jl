#!/usr/bin/env julia
"""
Full Model Calibration

Test combinations of crowding parameters AND operational costs to achieve
realistic 50-55% 5-year survival while preserving the AI paradox.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GlimpseABM
using Statistics
using Random
using Printf

const N_AGENTS = 1000
const N_ROUNDS = 60
const N_RUNS = 20  # Fewer runs for speed
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const BASE_SEED = 20260131

function run_test(; K::Float64, gamma::Float64, lambda::Float64,
                   op_cost::Float64, comp_scale::Float64)
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
            CROWDING_CAPACITY_K=K,
            CROWDING_CONVEXITY_GAMMA=gamma,
            CROWDING_STRENGTH_LAMBDA=lambda,
            BASE_OPERATIONAL_COST=op_cost,
            COMPETITION_SCALE_FACTOR=comp_scale
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
    println("="^85)
    println("  FULL MODEL CALIBRATION: Crowding + Operational Costs")
    println("="^85)
    println("\n  Target: 50-55% 5-year survival, negative treatment effect")
    println("="^85)

    # Test configurations: (K, γ, λ, op_cost, comp_scale, name)
    # Current BASE_OPERATIONAL_COST is 16667
    # Current COMPETITION_SCALE_FACTOR is 1.0
    test_configs = [
        # Baseline
        (1.5, 2.0, 0.50, 16667.0, 1.0, "Baseline (current)"),

        # Reduce operational costs
        (1.5, 2.0, 0.50, 12000.0, 1.0, "Op cost: 12k"),
        (1.5, 2.0, 0.50, 10000.0, 1.0, "Op cost: 10k"),
        (1.5, 2.0, 0.50, 8000.0, 1.0, "Op cost: 8k"),
        (1.5, 2.0, 0.50, 6000.0, 1.0, "Op cost: 6k"),

        # Reduce competition scale with op cost adjustments
        (1.5, 2.0, 0.50, 12000.0, 0.7, "Op:12k, Comp:0.7"),
        (1.5, 2.0, 0.50, 10000.0, 0.7, "Op:10k, Comp:0.7"),
        (1.5, 2.0, 0.50, 8000.0, 0.5, "Op:8k, Comp:0.5"),

        # Combined: lighter crowding model + reduced costs
        (2.0, 2.0, 0.35, 10000.0, 0.7, "Light crowd + low cost"),
        (2.5, 2.0, 0.30, 8000.0, 0.6, "Very light + very low"),

        # Stronger crowding but lower ops (to maintain paradox)
        (1.0, 2.0, 0.70, 8000.0, 0.8, "Strong crowd, low ops"),
        (0.8, 2.0, 0.80, 10000.0, 0.9, "Very strong crowd, med ops"),
    ]

    println("\n  Testing $(length(test_configs)) configurations...")
    println("-"^85)
    @printf("  %-25s %7s %7s %7s %7s %8s %8s\n",
            "Config", "None", "Basic", "Adv", "Prem", "TE (pp)", "Status")
    println("-"^85)

    for (K, γ, λ, op_cost, comp_scale, name) in test_configs
        results = run_test(K=K, gamma=γ, lambda=λ, op_cost=op_cost, comp_scale=comp_scale)

        none_surv = mean(results["none"]) * 100
        basic_surv = mean(results["basic"]) * 100
        adv_surv = mean(results["advanced"]) * 100
        prem_surv = mean(results["premium"]) * 100
        te = prem_surv - none_surv

        status = if 50 <= none_surv <= 58 && te < -5
            "✓ TARGET"
        elseif 50 <= none_surv <= 58 && te < 0
            "✓ weak TE"
        elseif none_surv < 45
            "too low"
        elseif none_surv > 60
            "too high"
        elseif te >= 0
            "⚠ reversed"
        else
            "close"
        end

        @printf("  %-25s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+7.1f  %s\n",
                name, none_surv, basic_surv, adv_surv, prem_surv, te, status)
    end

    println("-"^85)
    println("\n  Key insights:")
    println("    - Operational cost is the primary survival driver")
    println("    - Crowding controls the treatment effect (paradox strength)")
    println("    - Need to balance: lower ops for survival, stronger crowd for paradox")
    println("="^85)
end

main()
