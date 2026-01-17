#!/usr/bin/env julia

using Pkg
Pkg.activate(dirname(@__DIR__))

using GlimpseABM
using Statistics

println("=" ^ 70)
println("COMPREHENSIVE AI PARADOX VALIDATION TEST")
println("=" ^ 70)

println("\nRunning fixed-tier comparison (3 seeds each)...")

config = EmergentConfig()
config.N_ROUNDS = 100
config.N_AGENTS = 30

tiers = ["none", "basic", "advanced", "premium", "emergent"]
results = Dict{String, Dict{String, Vector{Float64}}}()

for tier in tiers
    results[tier] = Dict(
        "capital" => Float64[],
        "innovations" => Float64[],
        "survival" => Float64[]
    )
end

for seed in [42, 123, 456]
    for tier in tiers
        config_copy = deepcopy(config)
        sim = EmergentSimulation(config=config_copy, seed=seed)

        if tier != "emergent"
            for agent in sim.agents
                agent.fixed_ai_level = tier
                agent.current_ai_level = tier
            end
        end

        run!(sim)

        alive_agents = filter(a -> a.alive, sim.agents)
        capitals = [GlimpseABM.get_capital(a) for a in alive_agents]
        mean_capital = isempty(capitals) ? 0.0 : mean(capitals)
        total_innovations = sum(a.innovation_count for a in sim.agents)
        survival_rate = length(alive_agents) / length(sim.agents)

        push!(results[tier]["capital"], mean_capital)
        push!(results[tier]["innovations"], Float64(total_innovations))
        push!(results[tier]["survival"], survival_rate)
    end
    print(".")
end
println(" done")

println("\n" * "-" ^ 70)
println("RESULTS SUMMARY (3-seed average)")
println("-" ^ 70)
println("Config          | Survival% | Mean Capital   | Innovations | Capital/Init")
println("-" ^ 70)

initial_capital = config.INITIAL_CAPITAL

for tier in tiers
    surv = mean(results[tier]["survival"]) * 100
    cap = mean(results[tier]["capital"])
    innov = mean(results[tier]["innovations"])
    ratio = cap / initial_capital

    tier_padded = rpad(tier, 15)
    cap_str = lpad(string(round(Int, cap)), 12)
    innov_str = lpad(string(round(innov, digits=1)), 11)
    ratio_str = lpad(string(round(ratio, digits=2)), 6)
    surv_str = lpad(string(round(surv, digits=1)), 7)

    println(tier_padded, " | ", surv_str, "% | \$", cap_str, " | ", innov_str, " | ", ratio_str, "x")
end

println("-" ^ 70)

adv_capital = mean(results["advanced"]["capital"])
prem_capital = mean(results["premium"]["capital"])
adv_innov = mean(results["advanced"]["innovations"])
prem_innov = mean(results["premium"]["innovations"])

println("\nAI PARADOX CHECK:")
if prem_innov > adv_innov && adv_capital > prem_capital
    println("  PARADOX VISIBLE: Premium has more innovations but less capital than Advanced")
elseif adv_innov > prem_innov
    println("  PARADOX VISIBLE: Advanced has more innovations than Premium")
end
println("  Advanced: capital=", round(Int, adv_capital), ", innovations=", round(adv_innov, digits=1))
println("  Premium: capital=", round(Int, prem_capital), ", innovations=", round(prem_innov, digits=1))

emerg_capital = mean(results["emergent"]["capital"])
none_capital = mean(results["none"]["capital"])
basic_capital = mean(results["basic"]["capital"])
best_fixed = max(none_capital, basic_capital, adv_capital, prem_capital)

println("\nEMERGENT TIER ADVANTAGE:")
println("  Emergent capital: \$", round(Int, emerg_capital))
println("  Best fixed tier: \$", round(Int, best_fixed))
if emerg_capital > best_fixed
    advantage = (emerg_capital / best_fixed - 1) * 100
    println("  Emergent advantage: +", round(advantage, digits=1), "%")
end

println("\n" * "=" ^ 70)
println("COMPREHENSIVE VALIDATION COMPLETE")
println("=" ^ 70)
