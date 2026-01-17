#!/usr/bin/env julia
# Phase 4: Market Dynamics Validation

using Pkg
Pkg.activate(dirname(@__DIR__))

using GlimpseABM
using Statistics

println("=" ^ 70)
println("PHASE 4: MARKET DYNAMICS VALIDATION")
println("=" ^ 70)

config = EmergentConfig()

# 4.1 Opportunity Generation
println("\n4.1 OPPORTUNITY GENERATION")
println("-" ^ 50)

println("\n[Sector Distribution]")
sectors = config.SECTORS
println("  Available sectors: $(join(sectors, ", "))")
for sector in sectors
    profile = config.SECTOR_PROFILES[sector]
    println("  $sector:")
    println("    Return range: $(profile.return_range)")
    println("    Base failure: $(profile.base_failure_rate)")
    println("    Complexity: $(profile.complexity_range)")
end

println("\n[Return Sampling]")
println("  Distribution: Lognormal with sector-specific parameters")
println("  Opportunity return range: $(config.OPPORTUNITY_RETURN_RANGE)")

println("\n[Failure Probability]")
println("  Opportunity uncertainty range: $(config.OPPORTUNITY_UNCERTAINTY_RANGE)")

# 4.2 Regime Transitions
println("\n4.2 REGIME TRANSITIONS")
println("-" ^ 50)

println("\n[Regime States]")
regimes = config.MACRO_REGIME_STATES
println("  States: $(join(regimes, " -> "))")

println("\n[Transition Matrix]")
for from_regime in ["normal", "growth", "boom", "crisis", "recession"]
    transitions = config.MACRO_REGIME_TRANSITIONS[from_regime]
    probs = join(["$k=$(round(v, digits=2))" for (k,v) in sort(collect(transitions), by=x->x[1])], ", ")
    println("  $from_regime -> $probs")
end

println("\n[Regime Multipliers]")
println("  Return modifiers:")
for regime in ["crisis", "recession", "normal", "growth", "boom"]
    mult = config.MACRO_REGIME_RETURN_MODIFIERS[regime]
    println("    $regime: $(mult)x returns")
end

println("\n  Failure modifiers:")
for regime in ["crisis", "recession", "normal", "growth", "boom"]
    mult = config.MACRO_REGIME_FAILURE_MODIFIERS[regime]
    println("    $regime: $(mult)x failure rate")
end

# 4.3 Crowding Effects
println("\n4.3 CROWDING EFFECTS")
println("-" ^ 50)

println("\n[Sector Crowding]")
println("  Crowding threshold: $(config.RETURN_DEMAND_CROWDING_THRESHOLD)")
println("  Crowding penalty: $(config.RETURN_DEMAND_CROWDING_PENALTY)")

println("\n[Return Compression]")
println("  Oversupply penalty: $(config.RETURN_OVERSUPPLY_PENALTY)")
println("  Undersupply bonus: $(config.RETURN_UNDERSUPPLY_BONUS)")

println("\n[AI Tier Concentration]")
println("  Tracked via tier_capital_flow Dict")
println("  Affects competitive_recursion uncertainty dimension")

# Run simulation to verify market dynamics
println("\n4.4 TESTING MARKET DYNAMICS")
println("-" ^ 50)

config.N_ROUNDS = 50
config.N_AGENTS = 30
sim = EmergentSimulation(config=config, seed=42)
run!(sim)

# Check regime changes
println("\nMarket Environment State:")
println("  Current regime: $(sim.market.market_regime)")
println("  Available opportunities: $(length(sim.market.opportunities))")

# Get sector distribution of opportunities
sector_counts = Dict{String, Int}()
for opp in values(sim.market.opportunities)
    sector_counts[opp.sector] = get(sector_counts, opp.sector, 0) + 1
end
println("\n  Opportunity distribution by sector:")
for (sector, count) in sort(collect(sector_counts), by=x->-x[2])
    println("    $sector: $count")
end

# Check crowding metrics
println("\n  Sector pressure (crowding indicator):")
for (sector, pressure) in sim.market.sector_pressure
    println("    $sector: $(round(pressure, digits=3))")
end

println("\n" * "=" ^ 70)
println("PHASE 4 VALIDATION COMPLETE")
println("=" ^ 70)
