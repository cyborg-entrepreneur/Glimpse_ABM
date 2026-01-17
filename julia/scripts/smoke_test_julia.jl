#!/usr/bin/env julia
# Smoke test: Julia fixed tier simulations

using Pkg
Pkg.activate(dirname(@__DIR__))

using GlimpseABM
using Random
using Statistics
using Printf

const SEED = 42
const N_AGENTS = 100
const N_ROUNDS = 200
const N_RUNS = 5

println("="^70)
println("JULIA SMOKE TEST: Fixed Tier Comparison")
println("="^70)
println("Seed: $SEED, Agents: $N_AGENTS, Rounds: $N_ROUNDS, Runs: $N_RUNS")
println()

results = Dict{String,Dict{String,Any}}()

for tier in ["none", "basic", "advanced", "premium"]
    println("\n--- Running $tier tier ---")
    
    tier_results = Dict{String,Vector{Float64}}(
        "survival_rate" => Float64[],
        "mean_capital" => Float64[],
        "median_capital" => Float64[],
        "std_capital" => Float64[],
        "total_innovations" => Float64[],
        "invest_share" => Float64[],
        "innovate_share" => Float64[],
        "explore_share" => Float64[],
        "maintain_share" => Float64[],
        "mean_roic_invest" => Float64[],
        "mean_roic_innovate" => Float64[],
    )
    
    for run_idx in 1:N_RUNS
        # Use deterministic seed mixing like Python
        run_seed = SEED + (findfirst(==(tier), ["none", "basic", "advanced", "premium"]) - 1) * N_RUNS + run_idx
        
        config = EmergentConfig(
            N_AGENTS=N_AGENTS,
            N_ROUNDS=N_ROUNDS,
            RANDOM_SEED=run_seed
        )
        
        sim = EmergentSimulation(config=config, seed=run_seed, run_id="$(tier)_run$(run_idx)")
        initialize_agents!(sim; fixed_ai_level=tier)
        run!(sim)
        
        # Collect final round stats
        if !isempty(sim.history)
            final = sim.history[end]
            push!(tier_results["survival_rate"], get(final, "survival_rate", 0.0))
            push!(tier_results["mean_capital"], get(final, "mean_capital", 0.0))
            push!(tier_results["median_capital"], get(final, "median_capital", 0.0))
            push!(tier_results["std_capital"], get(final, "std_capital", 0.0))
            push!(tier_results["mean_roic_invest"], get(final, "mean_roic_invest", 0.0))
            push!(tier_results["mean_roic_innovate"], get(final, "mean_roic_innovate", 0.0))
            
            # Sum innovations across all rounds
            total_innov = sum(get(r, "innovation_successes", 0) for r in sim.history)
            push!(tier_results["total_innovations"], Float64(total_innov))
            
            # Average action shares across rounds
            inv_share = mean(get(r, "action_share_invest", 0.0) for r in sim.history)
            innov_share = mean(get(r, "action_share_innovate", 0.0) for r in sim.history)
            exp_share = mean(get(r, "action_share_explore", 0.0) for r in sim.history)
            maint_share = mean(get(r, "action_share_maintain", 0.0) for r in sim.history)
            
            push!(tier_results["invest_share"], inv_share)
            push!(tier_results["innovate_share"], innov_share)
            push!(tier_results["explore_share"], exp_share)
            push!(tier_results["maintain_share"], maint_share)
        end
        
        print(".")
    end
    println(" done")
    
    results[tier] = tier_results
end

# Print summary
println("\n" * "="^70)
println("JULIA RESULTS SUMMARY")
println("="^70)

for tier in ["none", "basic", "advanced", "premium"]
    r = results[tier]
    println("\n[$tier]")
    @printf("  Survival Rate:     %.1f%% (±%.1f%%)\n", 
            mean(r["survival_rate"])*100, std(r["survival_rate"])*100)
    @printf("  Mean Capital:      \$%.2fM (±\$%.2fM)\n", 
            mean(r["mean_capital"])/1e6, std(r["mean_capital"])/1e6)
    @printf("  Median Capital:    \$%.2fM\n", mean(r["median_capital"])/1e6)
    @printf("  Total Innovations: %.1f (±%.1f)\n", 
            mean(r["total_innovations"]), std(r["total_innovations"]))
    @printf("  Action Shares: invest=%.1f%%, innovate=%.1f%%, explore=%.1f%%, maintain=%.1f%%\n",
            mean(r["invest_share"])*100, mean(r["innovate_share"])*100,
            mean(r["explore_share"])*100, mean(r["maintain_share"])*100)
    @printf("  ROIC: invest=%.2f, innovate=%.2f\n",
            mean(r["mean_roic_invest"]), mean(r["mean_roic_innovate"]))
end

# Save detailed results
using DelimitedFiles

open("/Users/davidtownsend/Downloads/10_Glimpse-ABM-Project/glimpse_abm/julia/smoke_test_comparison/julia_results.txt", "w") do f
    println(f, "JULIA SMOKE TEST RESULTS")
    println(f, "Seed=$SEED, Agents=$N_AGENTS, Rounds=$N_ROUNDS, Runs=$N_RUNS")
    println(f, "")
    for tier in ["none", "basic", "advanced", "premium"]
        r = results[tier]
        println(f, "[$tier]")
        println(f, "survival_rate: $(mean(r["survival_rate"])) ± $(std(r["survival_rate"]))")
        println(f, "mean_capital: $(mean(r["mean_capital"])) ± $(std(r["mean_capital"]))")
        println(f, "median_capital: $(mean(r["median_capital"]))")
        println(f, "std_capital: $(mean(r["std_capital"]))")
        println(f, "total_innovations: $(mean(r["total_innovations"])) ± $(std(r["total_innovations"]))")
        println(f, "invest_share: $(mean(r["invest_share"]))")
        println(f, "innovate_share: $(mean(r["innovate_share"]))")
        println(f, "explore_share: $(mean(r["explore_share"]))")
        println(f, "maintain_share: $(mean(r["maintain_share"]))")
        println(f, "mean_roic_invest: $(mean(r["mean_roic_invest"]))")
        println(f, "mean_roic_innovate: $(mean(r["mean_roic_innovate"]))")
        println(f, "")
    end
end

println("\nResults saved to smoke_test_comparison/julia_results.txt")
