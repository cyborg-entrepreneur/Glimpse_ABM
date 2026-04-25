#!/usr/bin/env julia
# Mixed-Tier Mechanism Analysis
#
# Ports run_mechanism_analysis.jl from single-population to mixed-population
# fixed-tier design. Tracks per-tier behavioral, competition, innovation, and
# financial mechanisms within each mixed-tier simulation, then runs mediation
# analysis on the run-level cross-tier panel.
#
# Outputs CSVs only — PDF generation runs locally where CairoMakie is stable.

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
const AGENTS_PER_TIER = N_AGENTS ÷ 4
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const TIER_NUM = Dict("none"=>1, "basic"=>2, "advanced"=>3, "premium"=>4)
const BASE_SEED = 20260425

const OUTPUT_DIR = joinpath(@__DIR__, "..", "results",
    "mixed_tier_mechanism_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")

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
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        AGENT_AI_MODE="fixed",
    )
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

function run_mechanism_simulation(run_idx::Int, seed::Int)
    sim = build_mixed_sim(seed)

    survival_traj = Dict(t => Float64[] for t in AI_TIERS)
    innovate_share_traj = Dict(t => Float64[] for t in AI_TIERS)
    explore_share_traj = Dict(t => Float64[] for t in AI_TIERS)
    competition_traj = Dict(t => Float64[] for t in AI_TIERS)
    capital_traj = Dict(t => Float64[] for t in AI_TIERS)

    for r in 1:N_ROUNDS
        GlimpseABM.step!(sim, r)

        for tier in AI_TIERS
            tier_agents = filter(a -> a.fixed_ai_level == tier, sim.agents)
            alive = filter(a -> a.alive, tier_agents)
            push!(survival_traj[tier], length(alive) / length(tier_agents))

            action_counts = Dict("invest"=>0, "innovate"=>0, "explore"=>0, "maintain"=>0)
            for agent in tier_agents
                if length(agent.action_history) >= r
                    act = agent.action_history[r]
                    haskey(action_counts, act) && (action_counts[act] += 1)
                end
            end
            total = sum(values(action_counts))
            push!(innovate_share_traj[tier],
                  total > 0 ? action_counts["innovate"] / total : 0.0)
            push!(explore_share_traj[tier],
                  total > 0 ? action_counts["explore"] / total : 0.0)

            cr_vals = Float64[]
            for agent in tier_agents
                if !isempty(agent.uncertainty_metrics.competition_levels)
                    push!(cr_vals, last(agent.uncertainty_metrics.competition_levels))
                end
            end
            push!(competition_traj[tier], isempty(cr_vals) ? 0.0 : mean(cr_vals))
            push!(capital_traj[tier],
                  sum(GlimpseABM.get_capital(a) for a in tier_agents))
        end
    end

    out = Dict{String, Any}("run_idx" => run_idx, "seed" => seed)
    for tier in AI_TIERS
        tier_agents = filter(a -> a.fixed_ai_level == tier, sim.agents)
        alive_agents = filter(a -> a.alive, tier_agents)

        action_counts = Dict("invest"=>0, "innovate"=>0, "explore"=>0, "maintain"=>0)
        for agent in tier_agents
            for act in agent.action_history
                haskey(action_counts, act) && (action_counts[act] += 1)
            end
        end
        total_actions = sum(values(action_counts))

        all_competition = Float64[]
        for agent in tier_agents
            append!(all_competition, agent.uncertainty_metrics.competition_levels)
        end
        innovations = [a.innovation_count for a in tier_agents]
        successes = [a.innovation_success_count for a in tier_agents]

        total_invested = sum(a.total_invested for a in tier_agents)
        total_returned = sum(a.total_returned for a in tier_agents)
        final_capitals = [GlimpseABM.get_capital(a) for a in tier_agents]
        final_niches = sum(a.uncertainty_metrics.niches_discovered for a in tier_agents)
        final_combos = sum(a.uncertainty_metrics.new_combinations_created for a in tier_agents)

        out["$(tier)_survival_rate"] = length(alive_agents) / length(tier_agents)
        out["$(tier)_innovate_share"] = total_actions > 0 ? action_counts["innovate"]/total_actions : 0.0
        out["$(tier)_explore_share"] = total_actions > 0 ? action_counts["explore"]/total_actions : 0.0
        out["$(tier)_invest_share"] = total_actions > 0 ? action_counts["invest"]/total_actions : 0.0
        out["$(tier)_maintain_share"] = total_actions > 0 ? action_counts["maintain"]/total_actions : 0.0
        out["$(tier)_mean_competition"] = isempty(all_competition) ? 0.0 : mean(all_competition)
        out["$(tier)_max_competition"] = isempty(all_competition) ? 0.0 : maximum(all_competition)
        out["$(tier)_innovations_per_agent"] = mean(innovations)
        out["$(tier)_innovation_success_rate"] = sum(innovations) > 0 ? sum(successes)/sum(innovations) : 0.0
        out["$(tier)_total_niches"] = final_niches
        out["$(tier)_combinations"] = final_combos
        out["$(tier)_roi"] = total_invested > 0 ? (total_returned - total_invested) / total_invested : 0.0
        out["$(tier)_mean_final_capital"] = mean(final_capitals)
        out["$(tier)_survival_traj"] = survival_traj[tier]
        out["$(tier)_innovate_share_traj"] = innovate_share_traj[tier]
        out["$(tier)_explore_share_traj"] = explore_share_traj[tier]
        out["$(tier)_competition_traj"] = competition_traj[tier]
        out["$(tier)_capital_traj"] = capital_traj[tier]
    end
    return out
end

function flatten_to_run_panel(all_results::Vector{Dict})
    rows = []
    for r in all_results
        for tier in AI_TIERS
            push!(rows, (
                run_idx = r["run_idx"],
                seed = r["seed"],
                tier = tier,
                tier_numeric = TIER_NUM[tier],
                survival_rate = r["$(tier)_survival_rate"],
                innovate_share = r["$(tier)_innovate_share"],
                explore_share = r["$(tier)_explore_share"],
                invest_share = r["$(tier)_invest_share"],
                maintain_share = r["$(tier)_maintain_share"],
                mean_competition = r["$(tier)_mean_competition"],
                max_competition = r["$(tier)_max_competition"],
                innovations_per_agent = r["$(tier)_innovations_per_agent"],
                innovation_success_rate = r["$(tier)_innovation_success_rate"],
                total_niches = r["$(tier)_total_niches"],
                combinations = r["$(tier)_combinations"],
                roi = r["$(tier)_roi"],
                mean_final_capital = r["$(tier)_mean_final_capital"],
            ))
        end
    end
    return DataFrame(rows)
end

function compute_mediation(panel::DataFrame)
    tier_num = panel.tier_numeric
    surv = panel.survival_rate
    innov = panel.innovate_share
    comp = panel.mean_competition
    niches = Float64.(panel.total_niches)
    explr = panel.explore_share

    paths = Dict(
        "tier_to_survival" => cor(tier_num, surv),
        "tier_to_innovate" => cor(tier_num, innov),
        "tier_to_explore" => cor(tier_num, explr),
        "tier_to_competition" => cor(tier_num, comp),
        "tier_to_niches" => cor(tier_num, niches),
        "innovate_to_survival" => cor(innov, surv),
        "explore_to_survival" => cor(explr, surv),
        "competition_to_survival" => cor(comp, surv),
        "niches_to_survival" => cor(niches, surv),
    )
    paths["indirect_via_innovate"] = paths["tier_to_innovate"] * paths["innovate_to_survival"]
    paths["indirect_via_competition"] = paths["tier_to_competition"] * paths["competition_to_survival"]
    paths["indirect_via_niches"] = paths["tier_to_niches"] * paths["niches_to_survival"]
    return paths
end

function aggregate_per_tier_summary(panel::DataFrame)
    rows = []
    for tier in AI_TIERS
        sub = filter(:tier => ==(tier), panel)
        push!(rows, (
            tier = tier,
            n_runs = nrow(sub),
            mean_survival = mean(sub.survival_rate),
            std_survival = std(sub.survival_rate),
            mean_innovate_share = mean(sub.innovate_share),
            mean_explore_share = mean(sub.explore_share),
            mean_invest_share = mean(sub.invest_share),
            mean_maintain_share = mean(sub.maintain_share),
            mean_competition = mean(sub.mean_competition),
            mean_innovations_per_agent = mean(sub.innovations_per_agent),
            mean_innovation_success_rate = mean(sub.innovation_success_rate),
            mean_total_niches = mean(sub.total_niches),
            mean_combinations = mean(sub.combinations),
            mean_roi = mean(sub.roi),
            mean_final_capital = mean(sub.mean_final_capital),
        ))
    end
    return DataFrame(rows)
end

function save_trajectories(all_results::Vector{Dict}, output_dir::String)
    rows = []
    for tier in AI_TIERS
        for r in 1:N_ROUNDS
            survs = [res["$(tier)_survival_traj"][r] for res in all_results]
            innov = [res["$(tier)_innovate_share_traj"][r] for res in all_results]
            explr = [res["$(tier)_explore_share_traj"][r] for res in all_results]
            comp = [res["$(tier)_competition_traj"][r] for res in all_results]
            cap = [res["$(tier)_capital_traj"][r] for res in all_results]
            push!(rows, (
                tier = tier, round = r,
                mean_survival = mean(survs),
                std_survival = std(survs),
                mean_innovate_share = mean(innov),
                mean_explore_share = mean(explr),
                mean_competition = mean(comp),
                mean_total_capital = mean(cap),
            ))
        end
    end
    CSV.write(joinpath(output_dir, "trajectories.csv"), DataFrame(rows))
end

function main()
    println("="^80)
    println("  MIXED-TIER MECHANISM ANALYSIS")
    println("="^80)
    println("  N_AGENTS=$N_AGENTS  N_ROUNDS=$N_ROUNDS  N_RUNS=$N_RUNS")
    println("  Threads: $(Threads.nthreads())")
    println("="^80)
    mkpath(OUTPUT_DIR)

    all_results = Vector{Dict}(undef, N_RUNS)
    completed = Threads.Atomic{Int}(0)
    t0 = time()

    Threads.@threads for run_idx in 1:N_RUNS
        seed = BASE_SEED + run_idx
        try
            all_results[run_idx] = run_mechanism_simulation(run_idx, seed)
        catch e
            @warn "Mechanism run failed" run_idx=run_idx error=e
            rethrow(e)
        end
        c = Threads.atomic_add!(completed, 1)
        if c % 5 == 4 || c == N_RUNS - 1
            @printf("  %d/%d (%.1fs)\n", c+1, N_RUNS, time()-t0)
        end
    end
    @printf("\nAll runs done in %.1fs\n", time()-t0)

    panel = flatten_to_run_panel(all_results)
    summary = aggregate_per_tier_summary(panel)
    mediation = compute_mediation(panel)

    CSV.write(joinpath(OUTPUT_DIR, "run_level_panel.csv"), panel)
    CSV.write(joinpath(OUTPUT_DIR, "mechanism_summary.csv"), summary)
    save_trajectories(all_results, OUTPUT_DIR)

    mediation_rows = [(path=k, value=v) for (k, v) in mediation]
    CSV.write(joinpath(OUTPUT_DIR, "mediation_analysis.csv"), DataFrame(mediation_rows))

    println("\n--- MECHANISM SUMMARY ---")
    @printf("%-10s %10s %12s %12s %10s %10s %10s\n",
        "Tier", "Survival", "Innov%", "Explore%", "Compet.", "Niches", "ROI%")
    println("-"^80)
    for tier in AI_TIERS
        row = first(filter(r -> r.tier == tier, eachrow(summary)))
        @printf("%-10s %9.1f%% %11.1f%% %11.1f%% %9.3f %9.1f %9.1f%%\n",
            tier,
            row.mean_survival * 100,
            row.mean_innovate_share * 100,
            row.mean_explore_share * 100,
            row.mean_competition,
            row.mean_total_niches,
            row.mean_roi * 100)
    end
    println("\n--- MEDIATION ANALYSIS (within-run cross-tier panel, N=$(nrow(panel)) rows) ---")
    @printf("  Tier → Survival (total):    r = %+.3f\n", mediation["tier_to_survival"])
    @printf("  Tier → Innovate share:      r = %+.3f\n", mediation["tier_to_innovate"])
    @printf("  Tier → Competition:         r = %+.3f\n", mediation["tier_to_competition"])
    @printf("  Tier → Niches:              r = %+.3f\n", mediation["tier_to_niches"])
    @printf("  Innovate → Survival:        r = %+.3f\n", mediation["innovate_to_survival"])
    @printf("  Competition → Survival:     r = %+.3f\n", mediation["competition_to_survival"])
    @printf("  Niches → Survival:          r = %+.3f\n", mediation["niches_to_survival"])
    @printf("  Indirect via innovate:      %+.3f (%.0f%% of total)\n",
        mediation["indirect_via_innovate"],
        100*abs(mediation["indirect_via_innovate"])/max(abs(mediation["tier_to_survival"]), 1e-6))
    @printf("  Indirect via competition:   %+.3f\n", mediation["indirect_via_competition"])
    @printf("  Indirect via niches:        %+.3f\n", mediation["indirect_via_niches"])

    println("\nSaved CSVs to: $OUTPUT_DIR")
    println("="^80)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
