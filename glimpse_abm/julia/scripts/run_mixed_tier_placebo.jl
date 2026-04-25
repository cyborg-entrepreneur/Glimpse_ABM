#!/usr/bin/env julia
# Mixed-Tier Placebo Test Suite
#
# Four placebo tests for the mixed-tier paradox:
#  1. Permutation test — shuffle tier labels across runs, compute null TE distribution
#  2. Placebo tier test — TE between similar tiers (basic vs advanced) should be small
#  3. Early-period test — does the effect grow monotonically over rounds?
#  4. Dose-response monotonicity — survival should rank monotonically by tier
#
# Self-contained: runs N=50 mixed-tier sims with per-round per-tier trajectory
# tracking, then computes all four placebos against the run-level data.
# Outputs CSVs only — figure generation runs locally.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GlimpseABM
using Statistics
using Random
using DataFrames
using CSV
using Dates
using Printf

const N_AGENTS = 1000
const N_ROUNDS = 60
const N_RUNS = 50
const N_PLACEBO_ITERATIONS = 1000
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const BASE_SEED = 20260425

const OUTPUT_DIR = joinpath(@__DIR__, "..", "results",
    "mixed_tier_placebo_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")

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

function build_mixed_sim(seed::Int)
    rng = Random.MersenneTwister(seed)
    config = EmergentConfig(
        N_AGENTS=N_AGENTS, N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed, AGENT_AI_MODE="fixed")
    tier_assignments = create_tier_assignments(N_AGENTS, rng)
    initial_dist = Dict(t => 0.25 for t in AI_TIERS)
    sim = GlimpseABM.EmergentSimulation(
        config=config, initial_tier_distribution=initial_dist, seed=seed)
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
    return sim
end

function run_with_trajectory(run_idx::Int, seed::Int)
    sim = build_mixed_sim(seed)
    survival_traj = Dict(t => Float64[] for t in AI_TIERS)
    for r in 1:N_ROUNDS
        GlimpseABM.step!(sim, r)
        for tier in AI_TIERS
            tier_agents = filter(a -> a.fixed_ai_level == tier, sim.agents)
            alive = count(a -> a.alive, tier_agents)
            push!(survival_traj[tier], alive / length(tier_agents))
        end
    end
    final = Dict(tier => last(survival_traj[tier]) for tier in AI_TIERS)
    return Dict("run_idx" => run_idx, "seed" => seed,
                "final" => final, "trajectory" => survival_traj)
end

function permutation_test(per_run_per_tier::Dict{String, Vector{Float64}};
                          n_iter=N_PLACEBO_ITERATIONS, target_pair=("none","premium"))
    pool = vcat(per_run_per_tier[target_pair[1]], per_run_per_tier[target_pair[2]])
    n_a = length(per_run_per_tier[target_pair[1]])
    rng = MersenneTwister(12345)
    nulls = Float64[]
    for _ in 1:n_iter
        shuffled = shuffle(rng, pool)
        fake_a = shuffled[1:n_a]
        fake_b = shuffled[n_a+1:end]
        push!(nulls, mean(fake_b) - mean(fake_a))
    end
    actual = mean(per_run_per_tier[target_pair[2]]) - mean(per_run_per_tier[target_pair[1]])
    sorted = sort(nulls)
    ci_lo = sorted[max(1, Int(floor(0.025 * n_iter)))]
    ci_hi = sorted[min(n_iter, Int(ceil(0.975 * n_iter)))]
    p_two_sided = mean(abs.(nulls) .>= abs(actual))
    return (
        actual=actual, null_mean=mean(nulls), null_std=std(nulls),
        ci_lo=ci_lo, ci_hi=ci_hi, p_value=p_two_sided,
        significant=(actual < ci_lo || actual > ci_hi),
        nulls=nulls)
end

function early_period_test(trajs::Vector{Dict})
    time_points = [6, 12, 24, 36, 48, 60]
    rows = []
    for t in time_points
        none_at_t = [tr["trajectory"]["none"][t] for tr in trajs]
        prem_at_t = [tr["trajectory"]["premium"][t] for tr in trajs]
        ate = mean(prem_at_t) - mean(none_at_t)
        push!(rows, (
            round=t,
            none_survival=mean(none_at_t),
            premium_survival=mean(prem_at_t),
            ate_pp=ate*100,
        ))
    end
    df = DataFrame(rows)
    early = mean(df[1:2, :ate_pp])
    late = mean(df[5:6, :ate_pp])
    growth_ratio = abs(early) > 1e-6 ? late / early : NaN
    return (df=df, early_ate=early, late_ate=late,
            growth_ratio=growth_ratio,
            strengthens=abs(late) > abs(early))
end

function dose_response_test(per_run_per_tier::Dict{String, Vector{Float64}})
    means = [mean(per_run_per_tier[t]) for t in AI_TIERS]
    monotonic = all(means[i] >= means[i+1] for i in 1:3)
    pairwise_te = Dict{String, Float64}()
    for tier in ["basic", "advanced", "premium"]
        pairwise_te[tier] = mean(per_run_per_tier[tier]) - mean(per_run_per_tier["none"])
    end
    return (means=means, monotonic_decreasing=monotonic, pairwise_te=pairwise_te)
end

function main()
    println("="^80)
    println("  MIXED-TIER PLACEBO TEST SUITE")
    println("="^80)
    println("  N_AGENTS=$N_AGENTS  N_ROUNDS=$N_ROUNDS  N_RUNS=$N_RUNS")
    println("  Permutation iterations: $N_PLACEBO_ITERATIONS")
    println("  Threads: $(Threads.nthreads())")
    println("="^80)
    mkpath(OUTPUT_DIR)

    println("\nStep 1: running $N_RUNS mixed-tier sims with trajectory tracking...")
    trajs = Vector{Dict}(undef, N_RUNS)
    completed = Threads.Atomic{Int}(0)
    t0 = time()
    Threads.@threads for run_idx in 1:N_RUNS
        seed = BASE_SEED + run_idx
        trajs[run_idx] = run_with_trajectory(run_idx, seed)
        c = Threads.atomic_add!(completed, 1)
        if c % 10 == 9 || c == N_RUNS - 1
            @printf("  %d/%d runs (%.1fs)\n", c+1, N_RUNS, time()-t0)
        end
    end

    per_run_per_tier = Dict(t => [tr["final"][t] for tr in trajs] for t in AI_TIERS)

    # 1. Permutation test: none vs premium
    println("\nStep 2: permutation test (premium vs none)...")
    perm = permutation_test(per_run_per_tier; target_pair=("none","premium"))
    @printf("  Actual ATE:    %+.2f pp\n", perm.actual*100)
    @printf("  Null mean:     %+.2f pp\n", perm.null_mean*100)
    @printf("  Null 95%% CI:   [%+.2f, %+.2f] pp\n", perm.ci_lo*100, perm.ci_hi*100)
    @printf("  p-value:       %.4f\n", perm.p_value)
    @printf("  Significant:   %s\n", perm.significant)

    # 2. Placebo tier test: basic vs advanced (similar tiers)
    println("\nStep 3: placebo tier test (basic vs advanced)...")
    placebo_pair = permutation_test(per_run_per_tier; target_pair=("basic","advanced"))
    @printf("  Basic→Advanced ATE: %+.2f pp\n", placebo_pair.actual*100)
    @printf("  Premium→None ATE:   %+.2f pp\n", perm.actual*100)
    ratio = abs(perm.actual) / max(abs(placebo_pair.actual), 1e-4)
    @printf("  Ratio:              %.1fx\n", ratio)

    # 3. Early-period test
    println("\nStep 4: early-period test...")
    ep = early_period_test(trajs)
    println(ep.df)
    @printf("  Early ATE (mo 6-12):   %+.2f pp\n", ep.early_ate)
    @printf("  Late ATE (mo 48-60):   %+.2f pp\n", ep.late_ate)
    @printf("  Effect strengthens:    %s\n", ep.strengthens)

    # 4. Dose-response monotonicity
    println("\nStep 5: dose-response monotonicity...")
    dr = dose_response_test(per_run_per_tier)
    for (i, tier) in enumerate(AI_TIERS)
        @printf("  %-10s: %.3f\n", tier, dr.means[i])
    end
    @printf("  Monotonically decreasing (none>basic>advanced>premium): %s\n", dr.monotonic_decreasing)
    for tier in ["basic","advanced","premium"]
        @printf("    %-10s vs none: %+.2f pp\n", tier, dr.pairwise_te[tier]*100)
    end

    # Save CSVs
    println("\nStep 6: saving CSVs...")
    CSV.write(joinpath(OUTPUT_DIR, "permutation_null_distribution.csv"),
        DataFrame(null_te_pp=perm.nulls .* 100))
    CSV.write(joinpath(OUTPUT_DIR, "early_period_ate.csv"), ep.df)

    placebo_summary = DataFrame([
        (test="permutation_premium_vs_none", actual_pp=perm.actual*100,
         null_mean_pp=perm.null_mean*100, ci_lo_pp=perm.ci_lo*100,
         ci_hi_pp=perm.ci_hi*100, p_value=perm.p_value, significant=perm.significant),
        (test="placebo_basic_vs_advanced", actual_pp=placebo_pair.actual*100,
         null_mean_pp=placebo_pair.null_mean*100, ci_lo_pp=placebo_pair.ci_lo*100,
         ci_hi_pp=placebo_pair.ci_hi*100, p_value=placebo_pair.p_value,
         significant=placebo_pair.significant),
    ])
    CSV.write(joinpath(OUTPUT_DIR, "placebo_summary.csv"), placebo_summary)

    dr_rows = []
    for (i, tier) in enumerate(AI_TIERS)
        push!(dr_rows, (tier=tier, mean_survival=dr.means[i],
            te_vs_none_pp=(dr.means[i]-dr.means[1])*100))
    end
    CSV.write(joinpath(OUTPUT_DIR, "dose_response.csv"), DataFrame(dr_rows))

    per_run_rows = []
    for tr in trajs
        for tier in AI_TIERS
            push!(per_run_rows, (run_idx=tr["run_idx"], seed=tr["seed"],
                tier=tier, final_survival=tr["final"][tier]))
        end
    end
    CSV.write(joinpath(OUTPUT_DIR, "per_run_final_survival.csv"), DataFrame(per_run_rows))

    println("\nSaved CSVs to: $OUTPUT_DIR")
    @printf("Total time: %.1f minutes\n", (time()-t0)/60)
    println("="^80)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
