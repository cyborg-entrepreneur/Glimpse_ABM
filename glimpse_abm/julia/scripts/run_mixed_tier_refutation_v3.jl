#!/usr/bin/env julia
# Mixed-Tier Refutation Test Suite v3
#
# Ports refutation_test_suite_v3.jl from single-population to mixed-population
# fixed-tier design (1000 agents, 250/tier, all 4 tiers competing in same market).
#
# Each of the 31 conditions is run as a single mixed simulation per seed. Per-tier
# survival is extracted within the run. Treatment effects are computed within-seed
# (premium - none), which is more powerful than the single-pop between-condition
# comparison because random shocks cancel.
#
# Calibration: BLS defaults (SURVIVAL_THRESHOLD=$2M, INITIAL_CAPITAL_RANGE
# $2.5M-$10M, heterogeneous). Matches the canonical mixed-tier baseline (job
# 5143812, mean survival 0.49).

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GlimpseABM
using Statistics
using Random
using DataFrames
using CSV
using Dates
using Printf

include(joinpath(@__DIR__, "_safe_stats.jl"))

const N_AGENTS = 2000
const N_ROUNDS = 60
const N_RUNS = 50
const AGENTS_PER_TIER = N_AGENTS ÷ 4
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const BASE_SEED = 20260425

const OUTPUT_DIR = joinpath(@__DIR__, "..", "results",
    "mixed_tier_refutation_v3_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")

struct RefutationTest
    name::String
    description::String
    category::String
    execution_multipliers::Dict{String,Float64}
    quality_boosts::Dict{String,Float64}
    config_overrides::Dict{String,Any}
end

function RefutationTest(name, description, category;
        execution_multipliers=Dict("none"=>1.0,"basic"=>1.0,"advanced"=>1.0,"premium"=>1.0),
        quality_boosts=Dict("none"=>0.0,"basic"=>0.05,"advanced"=>0.05,"premium"=>0.05),
        config_overrides=Dict{String,Any}())
    RefutationTest(name, description, category, execution_multipliers, quality_boosts, config_overrides)
end

function create_tier_assignments(n_agents::Int, rng::AbstractRNG)
    assignments = String[]
    per_tier = n_agents ÷ length(AI_TIERS)
    for tier in AI_TIERS
        append!(assignments, fill(tier, per_tier))
    end
    remainder = n_agents - length(assignments)
    for i in 1:remainder
        push!(assignments, AI_TIERS[mod1(i, length(AI_TIERS))])
    end
    shuffle!(rng, assignments)
    return assignments
end

function build_config(test::RefutationTest, seed::Int)
    config = EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        AGENT_AI_MODE="fixed",
    )
    for (t, mult) in test.execution_multipliers
        config.AI_EXECUTION_SUCCESS_MULTIPLIERS[t] = mult
    end
    for (t, boost) in test.quality_boosts
        config.AI_QUALITY_BOOST[t] = boost
    end
    for (key, value) in test.config_overrides
        key_sym = Symbol(key)
        # v3.5.13: error on unknown keys instead of silently dropping them.
        # The previous `if hasproperty` check made future refutations no-op
        # without warning if a config field was renamed or deleted upstream.
        if !hasproperty(config, key_sym)
            error("Refutation override targets unknown config field '$key' " *
                  "in test '$(test.name)'. Either the field was renamed/removed " *
                  "or this is a typo. Production overrides must reference live fields.")
        end
        setfield!(config, key_sym, value)
    end
    return config
end

function run_single_mixed_sim(test::RefutationTest, run_idx::Int, seed::Int)
    rng = Random.MersenneTwister(seed)
    config = build_config(test, seed)
    tier_assignments = create_tier_assignments(N_AGENTS, rng)

    initial_dist = Dict(t => 0.25 for t in AI_TIERS)
    sim = GlimpseABM.EmergentSimulation(
        config=config,
        initial_tier_distribution=initial_dist,
        seed=seed
    )

    for (i, agent) in enumerate(sim.agents)
        new_tier = tier_assignments[i]
        agent.fixed_ai_level = new_tier
        agent.current_ai_level = new_tier
        for tier in collect(keys(agent.subscription_accounts))
            GlimpseABM.cancel_subscription_schedule!(agent, tier)
        end
        if new_tier != "none"
            GlimpseABM.ensure_subscription_schedule!(agent, new_tier)
        end
    end

    for round in 1:N_ROUNDS
        GlimpseABM.step!(sim, round)
    end

    tier_stats = Dict{String, Dict{String, Float64}}()
    for tier in AI_TIERS
        tier_agents = filter(a -> a.fixed_ai_level == tier, sim.agents)
        alive = filter(a -> a.alive, tier_agents)
        capitals_alive = [a.resources.capital for a in alive]
        tier_stats[tier] = Dict(
            "survival_rate" => length(alive) / length(tier_agents),
            "mean_capital_alive" => isempty(capitals_alive) ? 0.0 : mean(capitals_alive),
            "innovations_per_agent" => sum(a.innovation_count for a in tier_agents) / length(tier_agents),
        )
    end

    return Dict(
        "test" => test.name,
        "run_idx" => run_idx,
        "seed" => seed,
        "tier_stats" => tier_stats,
    )
end

function run_condition(test::RefutationTest)
    @printf("\n[%s] %s — %s\n", test.category, test.name, test.description)
    results = Vector{Dict}(undef, N_RUNS)
    completed = Threads.Atomic{Int}(0)
    t0 = time()

    Threads.@threads for run_idx in 1:N_RUNS
        seed = BASE_SEED + run_idx
        try
            results[run_idx] = run_single_mixed_sim(test, run_idx, seed)
        catch e
            @warn "Refutation run failed" test=test.name run_idx=run_idx error=e
            results[run_idx] = Dict("test"=>test.name, "run_idx"=>run_idx,
                                     "seed"=>seed, "tier_stats"=>nothing)
        end
        c = Threads.atomic_add!(completed, 1)
        if c % 10 == 9 || c == N_RUNS - 1
            @printf("    %d/%d (%.1fs)\n", c+1, N_RUNS, time()-t0)
        end
    end
    return results
end

function aggregate_condition(test::RefutationTest, results::Vector{Dict})
    tier_survivals = Dict(t => Float64[] for t in AI_TIERS)
    tier_capital = Dict(t => Float64[] for t in AI_TIERS)
    tier_innov = Dict(t => Float64[] for t in AI_TIERS)
    within_run_te_premium = Float64[]
    within_run_te_basic = Float64[]
    within_run_te_advanced = Float64[]

    for r in results
        ts = r["tier_stats"]
        ts === nothing && continue
        none_s = ts["none"]["survival_rate"]
        for tier in AI_TIERS
            push!(tier_survivals[tier], ts[tier]["survival_rate"])
            push!(tier_capital[tier], ts[tier]["mean_capital_alive"])
            push!(tier_innov[tier], ts[tier]["innovations_per_agent"])
        end
        push!(within_run_te_basic, ts["basic"]["survival_rate"] - none_s)
        push!(within_run_te_advanced, ts["advanced"]["survival_rate"] - none_s)
        push!(within_run_te_premium, ts["premium"]["survival_rate"] - none_s)
    end

    summary = Dict(
        "test" => test.name,
        "category" => test.category,
        "description" => test.description,
        "n_runs" => length(within_run_te_premium),
    )
    for tier in AI_TIERS
        summary["$(tier)_survival_mean"] = safe_mean(tier_survivals[tier])
        summary["$(tier)_survival_std"] = safe_std(tier_survivals[tier])
        summary["$(tier)_capital_alive_mean"] = safe_mean(tier_capital[tier])
        summary["$(tier)_innov_per_agent"] = safe_mean(tier_innov[tier])
    end
    summary["te_basic_pp"] = safe_mean(within_run_te_basic) * 100
    summary["te_basic_pp_std"] = safe_std(within_run_te_basic) * 100
    summary["te_advanced_pp"] = safe_mean(within_run_te_advanced) * 100
    summary["te_advanced_pp_std"] = safe_std(within_run_te_advanced) * 100
    summary["te_premium_pp"] = safe_mean(within_run_te_premium) * 100
    summary["te_premium_pp_std"] = safe_std(within_run_te_premium) * 100
    return summary
end

function get_test_suite()
    tests = RefutationTest[]
    push!(tests, RefutationTest("BASELINE", "Standard model (no modifications)", "BASELINE"))

    for mult in [2.0, 3.0, 5.0, 7.0, 10.0]
        push!(tests, RefutationTest(
            "EXEC_$(Int(mult))X",
            "Premium AI gets $(Int(mult))x execution success",
            "EXECUTION";
            execution_multipliers=Dict(
                "none" => 1.0,
                "basic" => 1.0 + (mult-1)*0.1,
                "advanced" => 1.0 + (mult-1)*0.25,
                "premium" => mult)))
    end

    for boost in [0.10, 0.20, 0.30, 0.40, 0.50]
        push!(tests, RefutationTest(
            "QUALITY_+$(Int(boost*100))",
            "Premium AI gets +$(Int(boost*100))% quality boost",
            "QUALITY";
            quality_boosts=Dict(
                "none" => 0.0,
                "basic" => 0.05,
                "advanced" => 0.08,
                "premium" => boost)))
    end

    push!(tests, RefutationTest(
        "COMBINED_3X_+20", "Premium: 3x execution + 20% quality", "COMBINED";
        execution_multipliers=Dict("none"=>1.0,"basic"=>1.15,"advanced"=>1.4,"premium"=>3.0),
        quality_boosts=Dict("none"=>0.0,"basic"=>0.05,"advanced"=>0.10,"premium"=>0.20)))
    push!(tests, RefutationTest(
        "COMBINED_5X_+30", "Premium: 5x execution + 30% quality", "COMBINED";
        execution_multipliers=Dict("none"=>1.0,"basic"=>1.2,"advanced"=>1.6,"premium"=>5.0),
        quality_boosts=Dict("none"=>0.0,"basic"=>0.08,"advanced"=>0.15,"premium"=>0.30)))
    push!(tests, RefutationTest(
        "EXTREME_10X_+50", "Premium: 10x execution + 50% quality (extreme)", "COMBINED";
        execution_multipliers=Dict("none"=>1.0,"basic"=>1.5,"advanced"=>2.5,"premium"=>10.0),
        quality_boosts=Dict("none"=>0.0,"basic"=>0.15,"advanced"=>0.30,"premium"=>0.50)))

    # CROWDING tests must override BOTH the sector-level COMPETITION_SCALE_FACTOR
    # AND the capital-saturation CROWDING_STRENGTH_LAMBDA (default 1.5). The
    # latter is the actual capital-saturation knob in models.jl:302; previously
    # only COMPETITION_SCALE_FACTOR was scaled, so "CROWDING_OFF" left
    # capacity-saturation crowding fully active and the test was uninformative.
    push!(tests, RefutationTest(
        "CROWDING_OFF", "Disable all competition/crowding dynamics", "CROWDING";
        config_overrides=Dict{String,Any}(
            "DISABLE_COMPETITION_DYNAMICS" => true,
            "COMPETITION_SCALE_FACTOR" => 0.0,
            "CROWDING_STRENGTH_LAMBDA" => 0.0)))
    push!(tests, RefutationTest("CROWDING_25%", "Crowding penalties reduced by 75%", "CROWDING";
        config_overrides=Dict{String,Any}(
            "COMPETITION_SCALE_FACTOR" => 0.25,
            "CROWDING_STRENGTH_LAMBDA" => 0.375)))
    push!(tests, RefutationTest("CROWDING_50%", "Crowding penalties reduced by 50%", "CROWDING";
        config_overrides=Dict{String,Any}(
            "COMPETITION_SCALE_FACTOR" => 0.5,
            "CROWDING_STRENGTH_LAMBDA" => 0.75)))
    push!(tests, RefutationTest("CROWDING_75%", "Crowding penalties reduced by 25%", "CROWDING";
        config_overrides=Dict{String,Any}(
            "COMPETITION_SCALE_FACTOR" => 0.75,
            "CROWDING_STRENGTH_LAMBDA" => 1.125)))

    push!(tests, RefutationTest("COST_0%", "AI is completely free (no costs)", "COST";
        config_overrides=Dict{String,Any}("AI_COST_INTENSITY" => 0.0)))
    push!(tests, RefutationTest("COST_25%", "AI costs reduced to 25%", "COST";
        config_overrides=Dict{String,Any}("AI_COST_INTENSITY" => 0.25)))
    push!(tests, RefutationTest("COST_50%", "AI costs reduced to 50%", "COST";
        config_overrides=Dict{String,Any}("AI_COST_INTENSITY" => 0.5)))
    push!(tests, RefutationTest("COST_75%", "AI costs reduced to 75%", "COST";
        config_overrides=Dict{String,Any}("AI_COST_INTENSITY" => 0.75)))

    push!(tests, RefutationTest("HERDING_OFF", "AI herding completely disabled", "HERDING";
        config_overrides=Dict{String,Any}("RECURSION_WEIGHTS" => Dict(
            "crowd_weight" => 0.35, "volatility_weight" => 0.30,
            "ai_herd_weight" => 0.0, "premium_reuse_weight" => 0.20))))
    push!(tests, RefutationTest("HERDING_25%", "AI herding reduced to 25%", "HERDING";
        config_overrides=Dict{String,Any}("RECURSION_WEIGHTS" => Dict(
            "crowd_weight" => 0.35, "volatility_weight" => 0.30,
            "ai_herd_weight" => 0.10, "premium_reuse_weight" => 0.20))))
    push!(tests, RefutationTest("HERDING_50%", "AI herding reduced to 50%", "HERDING";
        config_overrides=Dict{String,Any}("RECURSION_WEIGHTS" => Dict(
            "crowd_weight" => 0.35, "volatility_weight" => 0.30,
            "ai_herd_weight" => 0.20, "premium_reuse_weight" => 0.20))))

    # Operational cost reductions via OPS_COST_INTENSITY scalar (default 1.0).
    # v3.5.9: earlier overrode BASE_OPERATIONAL_COST, but production charges
    # use agent.operating_cost_estimate (sector-derived) — those overrides
    # were no-ops on actual burn. Now uses the OPS_COST_INTENSITY knob,
    # which estimate_operational_costs reads.
    push!(tests, RefutationTest("OPS_COST_50%", "Operational costs halved (intensity=0.5)", "OPERATIONS";
        config_overrides=Dict{String,Any}("OPS_COST_INTENSITY" => 0.5)))
    push!(tests, RefutationTest("OPS_COST_25%", "Operational costs at 25% (intensity=0.25)", "OPERATIONS";
        config_overrides=Dict{String,Any}("OPS_COST_INTENSITY" => 0.25)))

    push!(tests, RefutationTest("NO_CROWD_FREE_AI", "No crowding + Free AI", "COMBINED_FAV";
        config_overrides=Dict{String,Any}(
            "DISABLE_COMPETITION_DYNAMICS" => true,
            "COMPETITION_SCALE_FACTOR" => 0.0,
            "CROWDING_STRENGTH_LAMBDA" => 0.0,
            "AI_COST_INTENSITY" => 0.0)))
    push!(tests, RefutationTest("NO_CROWD_5X_EXEC", "No crowding + 5x execution", "COMBINED_FAV";
        execution_multipliers=Dict("none"=>1.0,"basic"=>1.2,"advanced"=>1.6,"premium"=>5.0),
        config_overrides=Dict{String,Any}(
            "DISABLE_COMPETITION_DYNAMICS" => true,
            "COMPETITION_SCALE_FACTOR" => 0.0,
            "CROWDING_STRENGTH_LAMBDA" => 0.0)))
    push!(tests, RefutationTest("CROWD_50_FREE_AI", "50% crowding + Free AI", "COMBINED_FAV";
        config_overrides=Dict{String,Any}(
            "COMPETITION_SCALE_FACTOR" => 0.5,
            "CROWDING_STRENGTH_LAMBDA" => 0.75,
            "AI_COST_INTENSITY" => 0.0)))
    push!(tests, RefutationTest("ALL_FAVORABLE",
        "All favorable: no crowd, free AI, 10x exec, +50% quality", "COMBINED_FAV";
        execution_multipliers=Dict("none"=>1.0,"basic"=>1.5,"advanced"=>2.5,"premium"=>10.0),
        quality_boosts=Dict("none"=>0.0,"basic"=>0.15,"advanced"=>0.30,"premium"=>0.50),
        config_overrides=Dict{String,Any}(
            "DISABLE_COMPETITION_DYNAMICS" => true,
            "COMPETITION_SCALE_FACTOR" => 0.0,
            "CROWDING_STRENGTH_LAMBDA" => 0.0,
            "AI_COST_INTENSITY" => 0.0,
            "RECURSION_WEIGHTS" => Dict(
                "crowd_weight" => 0.0, "volatility_weight" => 0.30,
                "ai_herd_weight" => 0.0, "premium_reuse_weight" => 0.0))))
    return tests
end

function classify_status(te_premium_pp::Float64, baseline_te_pp::Float64)
    if te_premium_pp > 1.0
        return "REVERSED"
    elseif te_premium_pp > -1.0
        return "NEUTRAL"
    elseif te_premium_pp > baseline_te_pp * 0.6
        return "REDUCED"
    else
        return "PERSISTS"
    end
end

function save_summary(summaries::Vector{Dict}, output_dir::String)
    mkpath(output_dir)
    df = DataFrame(summaries)
    CSV.write(joinpath(output_dir, "refutation_v3_mixed_summary.csv"), df)
    println("\nSaved: ", joinpath(output_dir, "refutation_v3_mixed_summary.csv"))
end

function main()
    println("="^80)
    println("  MIXED-TIER REFUTATION TEST SUITE V3")
    println("="^80)
    println("  N_AGENTS=$N_AGENTS  N_ROUNDS=$N_ROUNDS  N_RUNS=$N_RUNS")
    println("  BLS calibration (defaults inherited)")
    println("  Threads: $(Threads.nthreads())")
    println("="^80)
    mkpath(OUTPUT_DIR)

    tests = get_test_suite()
    @printf("\n  %d refutation conditions queued\n", length(tests))
    summaries = Vector{Dict}()
    t_start = time()

    for (i, test) in enumerate(tests)
        @printf("\n[%d/%d]\n", i, length(tests))
        results = run_condition(test)
        push!(summaries, aggregate_condition(test, results))
    end

    println("\n" * "="^80)
    println("  REFUTATION RESULTS (within-run treatment effects)")
    println("="^80)

    baseline_te = first(s for s in summaries if s["test"] == "BASELINE")["te_premium_pp"]
    @printf("\n  Baseline premium TE: %+.2f pp\n\n", baseline_te)

    @printf("  %-22s %-12s %10s %10s %10s\n", "Test", "Category", "TE(prem)", "TE(adv)", "Status")
    println("  " * "-"^70)
    for s in summaries
        status = classify_status(s["te_premium_pp"], baseline_te)
        @printf("  %-22s %-12s %+9.2f %+9.2f %12s\n",
            s["test"], s["category"], s["te_premium_pp"], s["te_advanced_pp"], status)
    end

    save_summary(summaries, OUTPUT_DIR)
    @printf("\nTotal time: %.1f minutes\n", (time()-t_start)/60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
