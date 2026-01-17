#!/usr/bin/env julia
# Phase 7: Knowledge System Validation

using Pkg
Pkg.activate(dirname(@__DIR__))

using GlimpseABM
using Statistics

println("=" ^ 70)
println("PHASE 7: KNOWLEDGE SYSTEM VALIDATION")
println("=" ^ 70)

config = EmergentConfig()

# 7.1 Knowledge Discovery
println("\n7.1 KNOWLEDGE DISCOVERY")
println("-" ^ 50)

println("\n[Discovery Probability Formula]")
println("  Base components:")
println("    - exploration_tendency trait")
println("    - AI tier discovery bonus")
println("    - network_strength effect (* 0.6)")

println("\n[AI Tier Discovery Bonuses]")
discovery_bonuses = Dict(
    "none" => 0.02,
    "basic" => 0.12,
    "advanced" => 0.28,
    "premium" => 0.42
)
for tier in ["none", "basic", "advanced", "premium"]
    println("  $tier: +$(discovery_bonuses[tier])")
end

println("\n[Info Quality/Breadth Effects]")
println("  bonus_factor = info_quality * 0.55 + info_breadth * 0.45")
for tier in ["none", "basic", "advanced", "premium"]
    ai_cfg = config.AI_LEVELS[tier]
    bonus_factor = ai_cfg.info_quality * 0.55 + ai_cfg.info_breadth * 0.45
    println("  $tier: info_quality=$(ai_cfg.info_quality), info_breadth=$(ai_cfg.info_breadth), factor=$(round(bonus_factor, digits=3))")
end

# 7.2 Learning from Outcomes
println("\n7.2 LEARNING FROM OUTCOMES")
println("-" ^ 50)

println("\n[Success Learning]")
println("  On success: sector_knowledge boost (+0.1)")
println("  Resource reinforcement via reinforce_agent_resources!()")

println("\n[Failure Learning]")
println("  30% chance to learn from failure")
println("  uncertainty_management boost (+0.02)")

println("\n[Decay Rates by Tier]")
decay_rates = Dict(
    "none" => 0.08,
    "basic" => 0.05,
    "advanced" => 0.025,
    "premium" => 0.0
)
for tier in ["none", "basic", "advanced", "premium"]
    println("  $tier: $(decay_rates[tier] * 100)% per round")
end

println("\n[Retention Modifier]")
println("  retention = 1.0 - 0.45 * info_quality")
for tier in ["none", "basic", "advanced", "premium"]
    ai_cfg = config.AI_LEVELS[tier]
    retention = 1.0 - 0.45 * ai_cfg.info_quality
    println("  $tier: retention=$(round(retention, digits=3))")
end

# Test knowledge system in simulation
println("\n7.3 TESTING KNOWLEDGE DYNAMICS")
println("-" ^ 50)

config.N_ROUNDS = 50
config.N_AGENTS = 20
sim = EmergentSimulation(config=config, seed=42)
run!(sim)

# Check knowledge base state
kb = sim.knowledge_base
println("\nKnowledge Base State:")
println("  Total knowledge components: $(length(kb.knowledge_components))")
println("  Agent domain beliefs tracked: $(length(kb.agent_domain_beliefs))")

# Check agent knowledge levels
println("\nSample Agent Knowledge (first 5 agents):")
for i in 1:min(5, length(sim.agents))
    agent = sim.agents[i]
    sector_knowledge = get(agent.resources.sector_knowledge, "tech", 0.0)
    uncertainty_mgmt = get(agent.resources.functional_capabilities, "uncertainty_management", 0.0)
    println("  Agent $i: tech_knowledge=$(round(sector_knowledge, digits=3)), uncertainty_mgmt=$(round(uncertainty_mgmt, digits=3))")
end

# Check component scarcity
scarcity = get_component_scarcity_metric(kb)
println("\nComponent Scarcity Metric: $(round(scarcity, digits=3))")

println("\n" * "=" ^ 70)
println("PHASE 7 VALIDATION COMPLETE")
println("=" ^ 70)
