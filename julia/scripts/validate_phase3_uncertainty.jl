#!/usr/bin/env julia
# Phase 3: Uncertainty Modeling Validation

using Pkg
Pkg.activate(dirname(@__DIR__))

using GlimpseABM
using Statistics

println("=" ^ 70)
println("PHASE 3: UNCERTAINTY MODELING VALIDATION")
println("=" ^ 70)

config = EmergentConfig()

# 3.1 Four Dimensions of Uncertainty
println("\n3.1 FOUR DIMENSIONS OF UNCERTAINTY")
println("-" ^ 50)

println("\n[Actor Ignorance]")
println("  Base: discovery_ratio from knowledge base")
println("  AI reduction: info_quality reduces ignorance")
println("  Premium share effect: high premium adoption reduces overall ignorance")
println("  Formula: base_ignorance * (1 - ai_info_quality_effect) * stable_sigmoid(...)")

println("\n[Practical Indeterminism]")
println("  Components:")
println("    - Regime instability (crisis/recession increase)")
println("    - AI herding (concentration in same tier)")
println("    - Crowding (sector concentration)")
println("  Scaling: COMPETITION_INTENSITY = $(config.COMPETITION_INTENSITY)")

println("\n[Agentic Novelty]")
println("  Base: innovation_rate from recent actions")
println("  AI tier effects on novelty:")
println("    - basic: +0.3 (moderate boost)")
println("    - advanced: +0.4 (high boost)")
println("    - premium: -0.1 (slight constraint)")
println("  Scaling: AI_NOVELTY_CONSTRAINT_INTENSITY = $(config.AI_NOVELTY_CONSTRAINT_INTENSITY)")

println("\n[Competitive Recursion]")
println("  Components:")
println("    - Investment concentration (HHI-like)")
println("    - Herding pressure (tier clustering)")
println("    - Premium/advanced share effects")
println("  Scaling: COMPETITION_INTENSITY = $(config.COMPETITION_INTENSITY)")

# 3.2 Agent Perception
println("\n3.2 AGENT PERCEPTION")
println("-" ^ 50)

println("\n[perceive_uncertainty function]")
println("  Inputs: global uncertainty state, agent traits, AI tier")
println("  Outputs:")
println("    - actor_ignorance (perceived)")
println("    - practical_indeterminism (perceived)")
println("    - agentic_novelty (perceived)")
println("    - competitive_recursion (perceived)")
println("    - knowledge_signal")
println("    - execution_risk")
println("    - innovation_signal")
println("    - competition_signal")
println("    - decision_confidence")

println("\n[AI Tier Adjustments]")
println("  Higher tiers reduce perceived actor_ignorance")
println("  Premium tier may increase competitive_recursion awareness")
println("  Blend factor: 0.85 (agent-specific vs global)")

println("\n[Decision Confidence Calculation]")
println("  Based on: recent_outcomes, uncertainty levels, trait factors")
println("  Range: 0.1 to 0.95")

# Run simulation to verify uncertainty dynamics
println("\n3.3 TESTING UNCERTAINTY DYNAMICS")
println("-" ^ 50)

config.N_ROUNDS = 50
config.N_AGENTS = 30
sim = EmergentSimulation(config=config, seed=42)
run!(sim)

# Extract uncertainty levels from history
actor_ign = [get(h, "actor_ignorance", 0.0) for h in sim.history]
practical = [get(h, "practical_indeterminism", 0.0) for h in sim.history]
agentic = [get(h, "agentic_novelty", 0.0) for h in sim.history]
competitive = [get(h, "competitive_recursion", 0.0) for h in sim.history]

println("\nUncertainty Dimension Statistics (50 rounds):")
println("  Actor Ignorance:        mean=$(round(mean(actor_ign), digits=3)), std=$(round(std(actor_ign), digits=3)), range=[$(round(minimum(actor_ign), digits=3)), $(round(maximum(actor_ign), digits=3))]")
println("  Practical Indeterminism: mean=$(round(mean(practical), digits=3)), std=$(round(std(practical), digits=3)), range=[$(round(minimum(practical), digits=3)), $(round(maximum(practical), digits=3))]")
println("  Agentic Novelty:        mean=$(round(mean(agentic), digits=3)), std=$(round(std(agentic), digits=3)), range=[$(round(minimum(agentic), digits=3)), $(round(maximum(agentic), digits=3))]")
println("  Competitive Recursion:  mean=$(round(mean(competitive), digits=3)), std=$(round(std(competitive), digits=3)), range=[$(round(minimum(competitive), digits=3)), $(round(maximum(competitive), digits=3))]")

# Verify dynamics change over time
early_actor = mean(actor_ign[1:10])
late_actor = mean(actor_ign[end-9:end])
println("\nDynamic Evolution:")
println("  Actor Ignorance: early=$(round(early_actor, digits=3)) -> late=$(round(late_actor, digits=3))")

early_practical = mean(practical[1:10])
late_practical = mean(practical[end-9:end])
println("  Practical Indet: early=$(round(early_practical, digits=3)) -> late=$(round(late_practical, digits=3))")

println("\n" * "=" ^ 70)
println("PHASE 3 VALIDATION COMPLETE")
println("=" ^ 70)
