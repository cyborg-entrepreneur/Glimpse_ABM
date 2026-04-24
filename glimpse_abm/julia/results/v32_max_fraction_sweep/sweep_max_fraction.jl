# v3.2 sensitivity sweep: max_fraction × tier × seed.
# Characterizes how the equilibrium-trap vs capital-preservation balance
# shifts as agents are allowed larger Kelly-style bets on high-conviction
# opportunities.

push!(LOAD_PATH, joinpath(pwd(), "src"))
using GlimpseABM, Random, Statistics, Printf

const MAX_FRACTIONS = [0.037, 0.07, 0.10, 0.15]
const TARGET_FRACTION = 0.033  # held constant
const TIERS = ["none", "basic", "advanced", "premium"]
const SEEDS = [42, 43, 44]
const N_AGENTS = 1000
const N_ROUNDS = 60

# Collect: Dict{max_fraction => Dict{tier => (mean_surv, std_surv, mean_satmax)}}
results = Dict{Float64, Dict{String, Tuple{Float64,Float64,Float64}}}()

t0 = time()
for mf in MAX_FRACTIONS
    println("\n=== max_fraction = $mf ===")
    tier_stats = Dict{String, Tuple{Float64,Float64,Float64}}()
    for tier in TIERS
        survivals = Float64[]
        satmaxes = Float64[]
        for seed in SEEDS
            cfg = EmergentConfig(N_AGENTS=N_AGENTS, N_ROUNDS=N_ROUNDS,
                                 RANDOM_SEED=seed, AGENT_AI_MODE="fixed",
                                 MAX_INVESTMENT_FRACTION=mf,
                                 TARGET_INVESTMENT_FRACTION=TARGET_FRACTION)
            sim = EmergentSimulation(config=cfg, seed=seed,
                                      initial_tier_distribution=Dict(tier=>1.0))
            GlimpseABM.run!(sim)
            alive = count(a->a.alive, sim.agents) / length(sim.agents)
            push!(survivals, alive)
            # Saturation max across surviving opps
            live_sats = [o.total_invested / o.capacity
                         for o in sim.market.opportunities
                         if o.capacity > 0.0 && o.total_invested > 0.0]
            push!(satmaxes, isempty(live_sats) ? 0.0 : maximum(live_sats))
        end
        m, s = mean(survivals), std(survivals)
        sm = mean(satmaxes)
        tier_stats[tier] = (m, s, sm)
        @printf "  %-10s surv=%.3f±%.3f  sat_max=%.2f\n" tier m s sm
    end
    results[mf] = tier_stats
end

println("\n" * "="^70)
println("SUMMARY — survival (mean ± std across $(length(SEEDS)) seeds)")
println("="^70)
@printf "%-8s" "max_f"
for t in TIERS; @printf " %-15s" t; end
@printf " %-10s\n" "mean"
println(repeat("-", 85))
for mf in MAX_FRACTIONS
    @printf "%-8.3f" mf
    tier_means = Float64[]
    for t in TIERS
        m, s, _ = results[mf][t]
        @printf " %6.3f±%.3f   " m s
        push!(tier_means, m)
    end
    @printf " %-10.3f\n" mean(tier_means)
end

println("\nSAT_MAX (mean across seeds) — signal for trap activity")
println(repeat("-", 85))
@printf "%-8s" "max_f"
for t in TIERS; @printf " %-10s" t; end
println()
for mf in MAX_FRACTIONS
    @printf "%-8.3f" mf
    for t in TIERS
        _, _, sm = results[mf][t]
        @printf " %-10.2f" sm
    end
    println()
end

println("\nELAPSED: $(round(time() - t0, digits=1))s")
