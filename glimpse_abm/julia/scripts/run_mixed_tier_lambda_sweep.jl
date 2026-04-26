#!/usr/bin/env julia
# Mixed-tier λ (CROWDING_STRENGTH_LAMBDA) sweep
#
# Sweeps the convex-crowding strength parameter at K_sat=1.5, γ=1.5 fixed.
# For each λ value, runs 30 mixed-tier sims (N=1000, 60 rounds), and emits:
#   - per-tier survival rates and treatment effects vs none
#   - saturation distribution (max, p95, frac above K_sat)
#
# Goal: identify a λ that produces (a) BLS 50-55% mean survival, (b) active
# but not crushing crowding (frac above K_sat ~5-15%, p95 saturation ~1.5-2.5).
# Whatever per-tier pattern emerges at that λ is an empirical finding, not a
# calibration target — calibration tunes only to observable real-world
# benchmarks, never to produce a particular qualitative tier ordering.

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
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const BASE_SEED = 20260425
const LAMBDA_VALUES = [2.0, 4.0, 6.0, 8.0, 12.0]
const K_SAT_FIXED = 1.5
const GAMMA_FIXED = 1.5

const OUTPUT_DIR = joinpath(@__DIR__, "..", "results",
    "mixed_tier_lambda_sweep_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")

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

function build_mixed_sim(seed::Int, lambda::Float64)
    rng = Random.MersenneTwister(seed)
    config = EmergentConfig(
        N_AGENTS=N_AGENTS, N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed, AGENT_AI_MODE="fixed",
    )
    config.CROWDING_STRENGTH_LAMBDA = lambda
    config.CROWDING_CAPACITY_RATIO_K = K_SAT_FIXED
    config.CROWDING_CONVEXITY_GAMMA = GAMMA_FIXED
    tier_assignments = create_tier_assignments(N_AGENTS, rng)
    initial_dist = Dict(t => 0.25 for t in AI_TIERS)
    sim = GlimpseABM.EmergentSimulation(
        config=config, initial_tier_distribution=initial_dist, seed=seed)
    for (i, agent) in enumerate(sim.agents)
        nt = tier_assignments[i]
        agent.fixed_ai_level = nt
        agent.current_ai_level = nt
        for t in collect(keys(agent.subscription_accounts))
            GlimpseABM.cancel_subscription_schedule!(agent, t)
        end
        if nt != "none"
            GlimpseABM.ensure_subscription_schedule!(agent, nt)
        end
    end
    return sim
end

function run_one(seed::Int, lambda::Float64)
    sim = build_mixed_sim(seed, lambda)
    # v3.5.10: time-resolved saturation. End-of-run snapshot misses peak
    # crowding because investments release on maturity. Sample each round.
    all_sats = Float64[]
    round_maxes = Float64[]
    round_fracs_K = Float64[]
    for r in 1:N_ROUNDS
        GlimpseABM.step!(sim, r)
        round_sats = Float64[]
        for o in sim.market.opportunities
            if o.capacity > 0 && o.total_invested > 0
                push!(round_sats, o.total_invested / o.capacity)
            end
        end
        append!(all_sats, round_sats)
        push!(round_maxes, isempty(round_sats) ? 0.0 : maximum(round_sats))
        push!(round_fracs_K, isempty(round_sats) ? 0.0 :
              count(s -> s > K_SAT_FIXED, round_sats) / length(round_sats))
    end
    out = Dict{String, Any}("seed" => seed, "lambda" => lambda)
    for tier in AI_TIERS
        ta = filter(a -> a.fixed_ai_level == tier, sim.agents)
        alive = filter(a -> a.alive, ta)
        out["$(tier)_survival"] = length(alive) / length(ta)
        out["$(tier)_capital_alive_mean"] = isempty(alive) ? 0.0 :
            mean(a.resources.capital for a in alive)
        out["$(tier)_innov_per_agent"] = sum(a.innovation_count for a in ta) / length(ta)
    end
    # Time-resolved saturation diagnostics
    if isempty(all_sats)
        out["sat_max_overall"] = 0.0
        out["sat_p95"] = 0.0
        out["sat_p75"] = 0.0
        out["frac_above_K_overall"] = 0.0
        out["max_round_peak"] = 0.0
        out["mean_round_peak"] = 0.0
        out["max_frac_above_K"] = 0.0
        out["mean_frac_above_K_per_round"] = 0.0
        out["n_obs"] = 0
    else
        out["sat_max_overall"] = maximum(all_sats)
        out["sat_p95"] = quantile(all_sats, 0.95)
        out["sat_p75"] = quantile(all_sats, 0.75)
        out["frac_above_K_overall"] = count(s -> s > K_SAT_FIXED, all_sats) / length(all_sats)
        out["max_round_peak"] = maximum(round_maxes)
        out["mean_round_peak"] = safe_mean(round_maxes)
        out["max_frac_above_K"] = maximum(round_fracs_K)
        out["mean_frac_above_K_per_round"] = safe_mean(round_fracs_K)
        out["n_obs"] = length(all_sats)
    end
    return out
end

function summarize(results::Vector{Dict}, lambda::Float64)
    rows = filter(r -> r["lambda"] == lambda, results)
    summary = Dict("lambda" => lambda, "n_runs" => length(rows))
    none_survs = [r["none_survival"] for r in rows]
    for tier in AI_TIERS
        sv = [r["$(tier)_survival"] for r in rows]
        summary["$(tier)_survival_mean"] = safe_mean(sv)
        summary["$(tier)_survival_std"] = safe_std(sv)
        summary["$(tier)_te_pp"] = (safe_mean(sv) - safe_mean(none_survs)) * 100
    end
    summary["mean_survival_overall"] = safe_mean(
        [safe_mean([r["$(t)_survival"] for r in rows]) for t in AI_TIERS])
    summary["sat_max_overall_mean"] = safe_mean([r["sat_max_overall"] for r in rows])
    summary["sat_p95_mean"] = safe_mean([r["sat_p95"] for r in rows])
    summary["sat_p75_mean"] = safe_mean([r["sat_p75"] for r in rows])
    summary["frac_above_K_overall_mean"] = safe_mean([r["frac_above_K_overall"] for r in rows])
    summary["max_round_peak_mean"] = safe_mean([r["max_round_peak"] for r in rows])
    summary["mean_round_peak_mean"] = safe_mean([r["mean_round_peak"] for r in rows])
    summary["max_frac_above_K_mean"] = safe_mean([r["max_frac_above_K"] for r in rows])
    summary["mean_frac_above_K_per_round_mean"] = safe_mean([r["mean_frac_above_K_per_round"] for r in rows])
    summary["n_obs_mean"] = safe_mean([r["n_obs"] for r in rows])
    return summary
end

function main()
    println("="^80)
    println("  MIXED-TIER LAMBDA SWEEP")
    println("="^80)
    println("  N_AGENTS=$N_AGENTS  N_ROUNDS=$N_ROUNDS  N_RUNS=$N_RUNS per λ")
    println("  Sweeping λ ∈ $LAMBDA_VALUES at K=$K_SAT_FIXED, γ=$GAMMA_FIXED")
    println("  Threads: $(Threads.nthreads())")
    println("="^80)
    mkpath(OUTPUT_DIR)

    total_runs = length(LAMBDA_VALUES) * N_RUNS
    all_results = Vector{Dict}(undef, total_runs)
    completed = Threads.Atomic{Int}(0)
    t0 = time()

    Threads.@threads for idx in 1:total_runs
        lam_idx = ((idx - 1) ÷ N_RUNS) + 1
        run_idx = ((idx - 1) % N_RUNS) + 1
        lambda = LAMBDA_VALUES[lam_idx]
        seed = BASE_SEED + lam_idx * 1000 + run_idx
        try
            all_results[idx] = run_one(seed, lambda)
        catch e
            @warn "Sweep run failed" lambda=lambda run_idx=run_idx error=e
            rethrow(e)
        end
        c = Threads.atomic_add!(completed, 1)
        if c % 25 == 24 || c == total_runs - 1
            @printf "  %d/%d (%.1fs)\n" c+1 total_runs time()-t0
        end
    end

    @printf "\nAll runs complete in %.1fs\n" time()-t0

    summaries = Vector{Dict}()
    for lambda in LAMBDA_VALUES
        push!(summaries, summarize(all_results, lambda))
    end

    println("\n" * "="^80)
    println("  LAMBDA SWEEP RESULTS")
    println("="^80)
    @printf "  %5s %5s %5s %5s %5s %6s %6s %6s %6s %6s %7s %6s %7s\n" "λ" "none" "basic" "adv" "prem" "mean" "TE_b" "TE_a" "TE_p" "p95sat" "rndPk" "%>K_o" "%>K_pk"
    println("  " * "-"^85)
    for s in summaries
        @printf "  %5.2f %5.2f %5.2f %5.2f %5.2f %6.3f %+6.2f %+6.2f %+6.2f %6.2f %7.2f %5.1f%% %6.1f%%\n" (
            s["lambda"], s["none_survival_mean"], s["basic_survival_mean"],
            s["advanced_survival_mean"], s["premium_survival_mean"],
            s["mean_survival_overall"], s["basic_te_pp"], s["advanced_te_pp"],
            s["premium_te_pp"],
            s["sat_p95_mean"], s["max_round_peak_mean"],
            100*s["frac_above_K_overall_mean"],
            100*s["max_frac_above_K_mean"])...
    end

    # Save CSVs
    summary_df = DataFrame(summaries)
    CSV.write(joinpath(OUTPUT_DIR, "lambda_sweep_summary.csv"), summary_df)

    per_run_df = DataFrame(all_results)
    CSV.write(joinpath(OUTPUT_DIR, "lambda_sweep_per_run.csv"), per_run_df)

    println("\nCSVs saved to: $OUTPUT_DIR")
    println("="^80)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
