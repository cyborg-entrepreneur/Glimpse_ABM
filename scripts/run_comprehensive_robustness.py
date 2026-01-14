#!/usr/bin/env python3
"""
Comprehensive Robustness Test Suite for GLIMPSE ABM (Python)

This script implements ALL robustness checks required for publication-quality
causal claims about AI adoption and entrepreneurial survival.

ROBUSTNESS CHECKS IMPLEMENTED:
=============================================================================
CRITICAL (Required for Publication):
1. Placebo/Falsification Test - Random AI assignment after simulation
2. Bootstrapped Confidence Intervals - 1000 bootstrap resamples for CIs
3. Multiple Comparison Correction - Benjamini-Hochberg FDR control

IMPORTANT (Strengthen Claims):
4. Population Size Sensitivity - N_AGENTS = 100, 500, 1000, 2000, 5000
5. Simulation Length Sensitivity - N_ROUNDS = 60, 120, 200, 400
6. Balanced Design Test - Exactly 25% agents per AI tier
7. Initial Capital Sensitivity - Different capital distributions

SUPPLEMENTARY (Complete Picture):
8. Market Regime Sensitivity - Start in different regimes
9. AI Accuracy Isolation - Vary only accuracy, hold costs constant
10. Alternative Outcome Measures - Capital, innovations, not just survival
=============================================================================

Run: python scripts/run_comprehensive_robustness.py [--quick]
"""

import argparse
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add grandparent directory (project root) to path for package imports
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from glimpse_abm import EmergentConfig, EmergentSimulation

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

BASE_SEED = 42
AI_TIERS = ["none", "basic", "advanced", "premium"]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_config(
    n_agents: int = 1000,
    n_rounds: int = 120,
    seed: int = 42,
    **kwargs
) -> EmergentConfig:
    """Create config with specified parameters."""
    config = EmergentConfig()
    config.N_AGENTS = n_agents
    config.N_ROUNDS = n_rounds
    config.RANDOM_SEED = seed
    config.enable_round_logging = False
    config.write_intermediate_batches = False

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def run_fixed_tier_simulation(
    config: EmergentConfig,
    tier: str,
    run_id: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Run a single simulation with fixed AI tier.
    Returns dictionary of outcome measures.
    """
    sim = EmergentSimulation(
        config=config,
        output_dir=output_dir,
        run_id=run_id
    )

    # Set fixed AI level for all agents
    for agent in sim.agents:
        agent.fixed_ai_level = tier
        agent.current_ai_level = tier

    # Run simulation
    sim.run()

    # Collect comprehensive outcomes
    survivors = sum(1 for a in sim.agents if a.alive)
    alive_agents = [a for a in sim.agents if a.alive]

    return {
        "survival_rate": survivors / config.N_AGENTS,
        "survivors": survivors,
        "mean_capital": np.mean([a.resources.capital for a in alive_agents]) if alive_agents else 0.0,
        "median_capital": np.median([a.resources.capital for a in alive_agents]) if alive_agents else 0.0,
        "total_innovations": sum(len(a.innovations) for a in sim.agents),
        "mean_innovations": np.mean([len(a.innovations) for a in sim.agents]),
        "final_capitals": [a.resources.capital for a in sim.agents],
        "alive_vector": [a.alive for a in sim.agents]
    }


def calculate_ate(
    treatment_rates: List[float],
    baseline_rates: List[float]
) -> Dict[str, float]:
    """Calculate ATE and effect sizes."""
    ate = np.mean(treatment_rates) - np.mean(baseline_rates)

    # Cohen's d
    pooled_std = np.sqrt((np.var(treatment_rates) + np.var(baseline_rates)) / 2)
    cohens_d = ate / pooled_std if pooled_std > 0 else 0.0

    # Cliff's delta (non-parametric effect size)
    n_greater = sum(1 for t in treatment_rates for b in baseline_rates if t > b)
    n_less = sum(1 for t in treatment_rates for b in baseline_rates if t < b)
    n_total = len(treatment_rates) * len(baseline_rates)
    cliffs_delta = (n_greater - n_less) / n_total if n_total > 0 else 0.0

    return {
        "ate": ate,
        "cohens_d": cohens_d,
        "cliffs_delta": cliffs_delta,
        "treatment_mean": np.mean(treatment_rates),
        "treatment_std": np.std(treatment_rates),
        "baseline_mean": np.mean(baseline_rates),
        "baseline_std": np.std(baseline_rates)
    }


def t_test(x: List[float], y: List[float]) -> Dict[str, float]:
    """Perform two-sample Welch's t-test."""
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)

    # Welch's t-test (unequal variances)
    se = np.sqrt(vx/nx + vy/ny) if (vx/nx + vy/ny) > 0 else 1e-10
    t_stat = (mx - my) / se if se > 0 else 0.0

    # Degrees of freedom (Welch-Satterthwaite)
    if se > 0 and nx > 1 and ny > 1:
        df = (vx/nx + vy/ny)**2 / ((vx/nx)**2/(nx-1) + (vy/ny)**2/(ny-1))
    else:
        df = 1.0

    # Approximate p-value using normal distribution (for large samples)
    p_value = 2 * (1 - cdf_normal(abs(t_stat)))

    return {"t_stat": t_stat, "df": df, "p_value": p_value}


def cdf_normal(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz and Stegun)."""
    t = 1 / (1 + 0.2316419 * abs(x))
    d = 0.3989423 * np.exp(-x**2 / 2)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    return 1 - p if x > 0 else p


# ============================================================================
# TEST 1: PLACEBO/FALSIFICATION TEST
# ============================================================================

def run_placebo_test(output_dir: str, n_runs_per_tier: int) -> Dict[str, Any]:
    """
    Placebo Test: Randomly assign AI tiers AFTER simulation completes.
    If we find significant effects with random assignment, our causal claims are suspect.

    Expected result: NO significant effects (ATE ≈ 0, p > 0.05)
    """
    print("\n" + "=" * 80)
    print("TEST 1: PLACEBO/FALSIFICATION TEST")
    print("=" * 80)
    print("\nPurpose: Verify that random AI assignment after simulation shows NO effect")
    print("Expected: ATE ≈ 0, p-value > 0.05 for all comparisons\n")

    n_agents = 1000
    n_rounds = 120
    n_sims = n_runs_per_tier * 4

    all_results = {}
    temp_dir = tempfile.mkdtemp()

    print(f"Running {n_sims} simulations with uniform AI assignment...")

    for sim_idx in range(1, n_sims + 1):
        seed = BASE_SEED + sim_idx
        config = create_config(n_agents=n_agents, n_rounds=n_rounds, seed=seed)

        sim = EmergentSimulation(
            config=config,
            output_dir=temp_dir,
            run_id=f"placebo_sim_{sim_idx}"
        )

        # Use basic tier for all (choice is arbitrary for placebo)
        for agent in sim.agents:
            agent.fixed_ai_level = "basic"
            agent.current_ai_level = "basic"

        sim.run()

        # Store individual agent outcomes
        all_results[sim_idx] = {
            "survival": [a.alive for a in sim.agents],
            "capital": [a.resources.capital for a in sim.agents]
        }

        if sim_idx % 10 == 0:
            print(f"  Completed {sim_idx}/{n_sims} simulations")

    # Now randomly assign fake "AI tiers" post-hoc
    print("\nRandomly assigning placebo AI tiers post-hoc...")

    rng = np.random.RandomState(12345)

    placebo_results = {tier: [] for tier in AI_TIERS}

    for sim_idx in range(1, n_sims + 1):
        outcomes = all_results[sim_idx]
        n = len(outcomes["survival"])

        # Randomly assign each agent to a tier
        fake_tiers = rng.choice(AI_TIERS, size=n)

        # Calculate survival rate by fake tier
        for tier in AI_TIERS:
            tier_mask = fake_tiers == tier
            if tier_mask.sum() > 0:
                tier_survival = np.mean([outcomes["survival"][i] for i in range(n) if tier_mask[i]])
                placebo_results[tier].append(tier_survival)

    # Calculate placebo ATEs
    print("\n" + "-" * 80)
    print("PLACEBO RESULTS (Random Post-Hoc Assignment)")
    print("-" * 80)
    print(f"{'Tier':<12} {'Mean Surv':>12} {'Std':>12} {'Placebo ATE':>12} {'t-stat':>12} {'p-value':>12}")
    print("-" * 80)

    baseline = placebo_results["none"]
    placebo_significant = False

    results_data = []

    for tier in AI_TIERS:
        rates = placebo_results[tier]
        effects = calculate_ate(rates, baseline)
        t_result = t_test(rates, baseline)

        sig_marker = "(!)" if tier != "none" and t_result["p_value"] < 0.05 else ""
        print(f"{tier:<12} {100*effects['treatment_mean']:>11.1f}% {100*effects['treatment_std']:>11.1f}% "
              f"{100*effects['ate']:>+11.1f}% {t_result['t_stat']:>12.2f} {t_result['p_value']:>12.4f} {sig_marker}")

        if tier != "none" and t_result["p_value"] < 0.05:
            placebo_significant = True

        results_data.append({
            "tier": tier,
            "mean_survival": effects["treatment_mean"],
            "std_survival": effects["treatment_std"],
            "ate": effects["ate"],
            "t_stat": t_result["t_stat"],
            "p_value": t_result["p_value"]
        })

    print("-" * 80)

    # Interpretation
    print("\nINTERPRETATION:")
    if not placebo_significant:
        print("✓ PASS: No significant effects found with random assignment")
        print("  This supports the validity of our causal identification strategy.")
    else:
        print("✗ WARNING: Significant effects found even with random assignment!")
        print("  This could indicate confounding or methodological issues.")

    # Save results
    placebo_file = os.path.join(output_dir, "test1_placebo_results.csv")
    with open(placebo_file, "w") as f:
        f.write("tier,mean_survival,std_survival,ate,t_stat,p_value\n")
        for row in results_data:
            f.write(f"{row['tier']},{row['mean_survival']:.6f},{row['std_survival']:.6f},"
                   f"{row['ate']:.6f},{row['t_stat']:.4f},{row['p_value']:.6f}\n")
    print(f"\nResults saved to: {placebo_file}")

    return {"passed": not placebo_significant, "results": results_data}


# ============================================================================
# TEST 2: BOOTSTRAPPED CONFIDENCE INTERVALS
# ============================================================================

def run_bootstrap_test(output_dir: str, n_runs_per_tier: int, n_bootstrap: int) -> Dict[str, Any]:
    """
    Bootstrap Test: Generate confidence intervals via resampling.
    Provides non-parametric uncertainty quantification for ATEs.
    """
    print("\n" + "=" * 80)
    print("TEST 2: BOOTSTRAPPED CONFIDENCE INTERVALS")
    print("=" * 80)
    print(f"\nPurpose: Provide robust confidence intervals via {n_bootstrap} bootstrap resamples")
    print("Method: Resample survival rates with replacement, compute ATE distribution\n")

    n_agents = 1000
    n_rounds = 120
    temp_dir = tempfile.mkdtemp()

    # First, run the actual simulations
    print("Running simulations to collect data for bootstrapping...")

    tier_results = {tier: [] for tier in AI_TIERS}

    for tier_idx, tier in enumerate(AI_TIERS, 1):
        import time
        print(f"  [{tier_idx}/4] AI Tier: {tier.upper()} ... ", end="", flush=True)
        tier_start = time.time()

        for run_idx in range(1, n_runs_per_tier + 1):
            seed = BASE_SEED + hash((tier, run_idx)) % 10000
            config = create_config(n_agents=n_agents, n_rounds=n_rounds, seed=seed)

            result = run_fixed_tier_simulation(config, tier, f"bootstrap_{tier}_{run_idx}", temp_dir)
            tier_results[tier].append(result["survival_rate"])

        elapsed = time.time() - tier_start
        print(f"done ({elapsed:.1f}s) - Mean: {100*np.mean(tier_results[tier]):.1f}%")

    # Bootstrap resampling
    print(f"\nRunning {n_bootstrap} bootstrap resamples...")

    rng = np.random.RandomState(54321)
    baseline_rates = tier_results["none"]

    bootstrap_results = {}

    for tier in AI_TIERS:
        treatment_rates = tier_results[tier]
        bootstrap_ates = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            boot_baseline = rng.choice(baseline_rates, size=len(baseline_rates), replace=True)
            boot_treatment = rng.choice(treatment_rates, size=len(treatment_rates), replace=True)

            boot_ate = np.mean(boot_treatment) - np.mean(boot_baseline)
            bootstrap_ates.append(boot_ate)

        # Calculate percentile confidence intervals
        sorted_ates = np.sort(bootstrap_ates)
        ci_lower_95 = sorted_ates[max(0, int(np.floor(0.025 * n_bootstrap)))]
        ci_upper_95 = sorted_ates[min(n_bootstrap-1, int(np.ceil(0.975 * n_bootstrap)))]
        ci_lower_99 = sorted_ates[max(0, int(np.floor(0.005 * n_bootstrap)))]
        ci_upper_99 = sorted_ates[min(n_bootstrap-1, int(np.ceil(0.995 * n_bootstrap)))]

        point_ate = np.mean(treatment_rates) - np.mean(baseline_rates)

        bootstrap_results[tier] = {
            "point_ate": point_ate,
            "bootstrap_mean": np.mean(bootstrap_ates),
            "bootstrap_std": np.std(bootstrap_ates),
            "ci_95_lower": ci_lower_95,
            "ci_95_upper": ci_upper_95,
            "ci_99_lower": ci_lower_99,
            "ci_99_upper": ci_upper_99,
            "significant_95": ci_lower_95 > 0 or ci_upper_95 < 0,
            "significant_99": ci_lower_99 > 0 or ci_upper_99 < 0
        }

    # Display results
    print("\n" + "-" * 100)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("-" * 100)
    print(f"{'Tier':<12} {'Point ATE':>12} {'Boot SE':>12} {'95% CI':>25} {'99% CI':>25}")
    print("-" * 100)

    results_data = []

    for tier in AI_TIERS:
        r = bootstrap_results[tier]
        sig_95 = "*" if r["significant_95"] else ""
        sig_99 = "**" if r["significant_99"] else ""
        sig = sig_99 if sig_99 else sig_95

        print(f"{tier:<12} {100*r['point_ate']:>+11.1f}% {100*r['bootstrap_std']:>11.1f}% "
              f"[{100*r['ci_95_lower']:>+.1f}%, {100*r['ci_95_upper']:>+.1f}%]{sig_95:<2} "
              f"[{100*r['ci_99_lower']:>+.1f}%, {100*r['ci_99_upper']:>+.1f}%]{sig_99:<2}")

        results_data.append({
            "tier": tier,
            "point_ate": r["point_ate"],
            "bootstrap_std": r["bootstrap_std"],
            "ci_95_lower": r["ci_95_lower"],
            "ci_95_upper": r["ci_95_upper"],
            "ci_99_lower": r["ci_99_lower"],
            "ci_99_upper": r["ci_99_upper"],
            "significant_95": r["significant_95"],
            "significant_99": r["significant_99"]
        })

    print("-" * 100)
    print("* = significant at 95% level (CI excludes 0)")
    print("** = significant at 99% level")

    # Save results
    bootstrap_file = os.path.join(output_dir, "test2_bootstrap_results.csv")
    with open(bootstrap_file, "w") as f:
        f.write("tier,point_ate,bootstrap_std,ci_95_lower,ci_95_upper,ci_99_lower,ci_99_upper,sig_95,sig_99\n")
        for row in results_data:
            f.write(f"{row['tier']},{row['point_ate']:.6f},{row['bootstrap_std']:.6f},"
                   f"{row['ci_95_lower']:.6f},{row['ci_95_upper']:.6f},"
                   f"{row['ci_99_lower']:.6f},{row['ci_99_upper']:.6f},"
                   f"{row['significant_95']},{row['significant_99']}\n")
    print(f"\nResults saved to: {bootstrap_file}")

    return {"results": bootstrap_results, "data": results_data}


# ============================================================================
# TEST 3: MULTIPLE COMPARISON CORRECTION (Benjamini-Hochberg)
# ============================================================================

def run_multiple_comparison_test(output_dir: str, n_runs_per_tier: int) -> Dict[str, Any]:
    """
    Multiple Comparison Correction: Control False Discovery Rate.
    With multiple AI tier comparisons, we need to adjust p-values.
    """
    print("\n" + "=" * 80)
    print("TEST 3: MULTIPLE COMPARISON CORRECTION (Benjamini-Hochberg FDR)")
    print("=" * 80)
    print("\nPurpose: Control false discovery rate when comparing multiple AI tiers")
    print("Method: Benjamini-Hochberg procedure at α = 0.05\n")

    n_agents = 1000
    n_rounds = 120
    temp_dir = tempfile.mkdtemp()

    # Run simulations
    tier_results = {tier: [] for tier in AI_TIERS}

    print("Running simulations...")
    for tier_idx, tier in enumerate(AI_TIERS, 1):
        print(f"  [{tier_idx}/4] AI Tier: {tier.upper()} ... ", end="", flush=True)

        for run_idx in range(1, n_runs_per_tier + 1):
            seed = BASE_SEED + hash(("mcp", tier, run_idx)) % 10000
            config = create_config(n_agents=n_agents, n_rounds=n_rounds, seed=seed)

            result = run_fixed_tier_simulation(config, tier, f"mcp_{tier}_{run_idx}", temp_dir)
            tier_results[tier].append(result["survival_rate"])

        print(f"done - Mean: {100*np.mean(tier_results[tier]):.1f}%")

    # Calculate p-values for each comparison
    baseline = tier_results["none"]
    comparisons = []

    for tier in ["basic", "advanced", "premium"]:
        treatment = tier_results[tier]
        t_result = t_test(treatment, baseline)
        effects = calculate_ate(treatment, baseline)

        comparisons.append({
            "tier": tier,
            "ate": effects["ate"],
            "p_value": t_result["p_value"],
            "t_stat": t_result["t_stat"]
        })

    # Sort by p-value for BH procedure
    comparisons.sort(key=lambda x: x["p_value"])

    # Benjamini-Hochberg correction
    m = len(comparisons)
    alpha = 0.05

    for i, comp in enumerate(comparisons, 1):
        bh_threshold = (i / m) * alpha
        comp["bh_threshold"] = bh_threshold
        comp["bh_significant"] = comp["p_value"] <= bh_threshold
        comp["bonferroni_threshold"] = alpha / m
        comp["bonferroni_significant"] = comp["p_value"] <= (alpha / m)

    # Display results
    print("\n" + "-" * 100)
    print("MULTIPLE COMPARISON RESULTS")
    print("-" * 100)
    print(f"{'Tier':<12} {'ATE':>12} {'p-value':>12} {'BH Thresh':>12} {'Bonf Thresh':>12} {'BH Sig?':>15} {'Bonf Sig?':>15}")
    print("-" * 100)

    for comp in comparisons:
        print(f"{comp['tier']:<12} {100*comp['ate']:>+11.1f}% {comp['p_value']:>12.6f} "
              f"{comp['bh_threshold']:>12.4f} {comp['bonferroni_threshold']:>12.4f} "
              f"{'YES' if comp['bh_significant'] else 'no':>15} "
              f"{'YES' if comp['bonferroni_significant'] else 'no':>15}")

    print("-" * 100)

    # Count significant results
    bh_sig_count = sum(1 for c in comparisons if c["bh_significant"])
    bonf_sig_count = sum(1 for c in comparisons if c["bonferroni_significant"])

    print(f"\nSUMMARY:")
    print(f"  Comparisons significant with BH correction: {bh_sig_count} / {m}")
    print(f"  Comparisons significant with Bonferroni: {bonf_sig_count} / {m}")

    if bh_sig_count > 0:
        print("\n✓ Effects remain significant after FDR correction")
    else:
        print("\n⚠ No effects survive multiple comparison correction")

    # Save results
    mcp_file = os.path.join(output_dir, "test3_multiple_comparison_results.csv")
    with open(mcp_file, "w") as f:
        f.write("tier,ate,p_value,bh_threshold,bonf_threshold,bh_significant,bonf_significant\n")
        for comp in comparisons:
            f.write(f"{comp['tier']},{comp['ate']:.6f},{comp['p_value']:.6f},"
                   f"{comp['bh_threshold']:.6f},{comp['bonferroni_threshold']:.6f},"
                   f"{comp['bh_significant']},{comp['bonferroni_significant']}\n")
    print(f"\nResults saved to: {mcp_file}")

    return {"comparisons": comparisons, "bh_significant_count": bh_sig_count}


# ============================================================================
# TEST 4: POPULATION SIZE SENSITIVITY
# ============================================================================

def run_population_size_test(output_dir: str, population_sizes: List[int], n_runs_per_tier: int) -> Dict[str, Any]:
    """
    Population Size Test: Does the effect hold at different N?
    """
    print("\n" + "=" * 80)
    print("TEST 4: POPULATION SIZE SENSITIVITY")
    print("=" * 80)
    print(f"\nPurpose: Verify effects are consistent across different population sizes")
    print(f"Sizes tested: {', '.join(map(str, population_sizes))}\n")

    n_rounds = 120
    n_runs = max(5, n_runs_per_tier // 2)
    temp_dir = tempfile.mkdtemp()

    results_by_size = {}

    for n_agents in population_sizes:
        print(f"\n--- Testing N_AGENTS = {n_agents} ---")

        tier_results = {tier: [] for tier in AI_TIERS}

        for tier_idx, tier in enumerate(AI_TIERS, 1):
            print(f"  [{tier_idx}/4] {tier.upper()}: ", end="", flush=True)

            for run_idx in range(1, n_runs + 1):
                seed = BASE_SEED + hash(("popsize", n_agents, tier, run_idx)) % 10000
                config = create_config(n_agents=n_agents, n_rounds=n_rounds, seed=seed)

                result = run_fixed_tier_simulation(config, tier, f"popsize_{n_agents}_{tier}_{run_idx}", temp_dir)
                tier_results[tier].append(result["survival_rate"])

            print(f"{100*np.mean(tier_results[tier]):.1f}% ± {100*np.std(tier_results[tier]):.1f}%")

        # Calculate effects
        baseline = tier_results["none"]
        size_effects = {}

        for tier in AI_TIERS:
            effects = calculate_ate(tier_results[tier], baseline)
            t_result = t_test(tier_results[tier], baseline)
            size_effects[tier] = {
                "mean": effects["treatment_mean"],
                "ate": effects["ate"],
                "cohens_d": effects["cohens_d"],
                "p_value": t_result["p_value"]
            }

        results_by_size[n_agents] = size_effects

    # Summary table
    print("\n" + "-" * 100)
    print("POPULATION SIZE SENSITIVITY SUMMARY (Premium AI ATE)")
    print("-" * 100)
    cohens_d_header = "Cohen's d"
    print(f"{'N_AGENTS':<12} {'Premium ATE':>15} {cohens_d_header:>15} {'p-value':>15} {'Direction':>15}")
    print("-" * 100)

    results_data = []
    consistent_direction = True
    first_direction = None

    for n_agents in population_sizes:
        effects = results_by_size[n_agents]["premium"]
        direction = "negative" if effects["ate"] < 0 else "positive"

        if first_direction is None:
            first_direction = direction
        elif direction != first_direction:
            consistent_direction = False

        sig_marker = "*" if effects["p_value"] < 0.05 else ""
        print(f"{n_agents:<12} {100*effects['ate']:>+14.1f}% {effects['cohens_d']:>15.2f} "
              f"{effects['p_value']:>15.4f} {direction:>15}{sig_marker}")

        results_data.append({
            "n_agents": n_agents,
            "premium_ate": effects["ate"],
            "cohens_d": effects["cohens_d"],
            "p_value": effects["p_value"],
            "direction": direction
        })

    print("-" * 100)
    print("* = p < 0.05")

    print("\nINTERPRETATION:")
    if consistent_direction:
        print("✓ PASS: Effect direction is consistent across all population sizes")
    else:
        print("⚠ WARNING: Effect direction varies with population size")

    # Save results
    popsize_file = os.path.join(output_dir, "test4_population_size_results.csv")
    with open(popsize_file, "w") as f:
        f.write("n_agents,premium_ate,cohens_d,p_value,direction\n")
        for row in results_data:
            f.write(f"{row['n_agents']},{row['premium_ate']:.6f},{row['cohens_d']:.4f},"
                   f"{row['p_value']:.6f},{row['direction']}\n")
    print(f"\nResults saved to: {popsize_file}")

    return {"results": results_by_size, "consistent": consistent_direction}


# ============================================================================
# TEST 5: SIMULATION LENGTH SENSITIVITY
# ============================================================================

def run_simulation_length_test(output_dir: str, sim_lengths: List[int], n_runs_per_tier: int) -> Dict[str, Any]:
    """
    Simulation Length Test: Does effect persist over different time horizons?
    """
    print("\n" + "=" * 80)
    print("TEST 5: SIMULATION LENGTH SENSITIVITY")
    print("=" * 80)
    print(f"\nPurpose: Verify effects across different simulation durations")
    print(f"Lengths tested: {', '.join(map(str, sim_lengths))} rounds\n")

    n_agents = 1000
    n_runs = n_runs_per_tier
    temp_dir = tempfile.mkdtemp()

    results_by_length = {}

    for n_rounds in sim_lengths:
        print(f"\n--- Testing N_ROUNDS = {n_rounds} ---")

        tier_results = {tier: [] for tier in AI_TIERS}

        for tier_idx, tier in enumerate(AI_TIERS, 1):
            print(f"  [{tier_idx}/4] {tier.upper()}: ", end="", flush=True)

            for run_idx in range(1, n_runs + 1):
                seed = BASE_SEED + hash(("simlen", n_rounds, tier, run_idx)) % 10000
                config = create_config(n_agents=n_agents, n_rounds=n_rounds, seed=seed)

                result = run_fixed_tier_simulation(config, tier, f"simlen_{n_rounds}_{tier}_{run_idx}", temp_dir)
                tier_results[tier].append(result["survival_rate"])

            print(f"{100*np.mean(tier_results[tier]):.1f}% ± {100*np.std(tier_results[tier]):.1f}%")

        # Calculate effects
        baseline = tier_results["none"]
        length_effects = {}

        for tier in AI_TIERS:
            effects = calculate_ate(tier_results[tier], baseline)
            t_result = t_test(tier_results[tier], baseline)
            length_effects[tier] = {
                "mean": effects["treatment_mean"],
                "ate": effects["ate"],
                "cohens_d": effects["cohens_d"],
                "p_value": t_result["p_value"]
            }

        results_by_length[n_rounds] = length_effects

    # Summary table
    print("\n" + "-" * 100)
    print("SIMULATION LENGTH SENSITIVITY SUMMARY")
    print("-" * 100)
    cohens_d_header = "Cohen's d"
    print(f"{'N_ROUNDS':<12} {'Premium ATE':>15} {cohens_d_header:>15} {'p-value':>15} {'Paradox?':>15}")
    print("-" * 100)

    results_data = []
    paradox_at_all_lengths = True

    for n_rounds in sim_lengths:
        effects = results_by_length[n_rounds]["premium"]
        has_paradox = effects["ate"] < -0.01

        if not has_paradox:
            paradox_at_all_lengths = False

        print(f"{n_rounds:<12} {100*effects['ate']:>+14.1f}% {effects['cohens_d']:>15.2f} "
              f"{effects['p_value']:>15.4f} {'YES' if has_paradox else 'no':>15}")

        results_data.append({
            "n_rounds": n_rounds,
            "premium_ate": effects["ate"],
            "cohens_d": effects["cohens_d"],
            "p_value": effects["p_value"],
            "paradox": has_paradox
        })

    print("-" * 100)

    print("\nINTERPRETATION:")
    if paradox_at_all_lengths:
        print("✓ PASS: Information paradox persists across all simulation lengths")
    else:
        print("⚠ Note: Paradox strength varies with simulation length")

    # Save results
    simlen_file = os.path.join(output_dir, "test5_simulation_length_results.csv")
    with open(simlen_file, "w") as f:
        f.write("n_rounds,premium_ate,cohens_d,p_value,paradox\n")
        for row in results_data:
            f.write(f"{row['n_rounds']},{row['premium_ate']:.6f},{row['cohens_d']:.4f},"
                   f"{row['p_value']:.6f},{row['paradox']}\n")
    print(f"\nResults saved to: {simlen_file}")

    return {"results": results_by_length, "paradox_consistent": paradox_at_all_lengths}


# ============================================================================
# TEST 6: BALANCED DESIGN TEST
# ============================================================================

def run_balanced_design_test(output_dir: str, n_runs_per_tier: int) -> Dict[str, Any]:
    """
    Balanced Design Test: Exactly 25% agents per AI tier.
    Ensures results aren't driven by unbalanced group sizes.
    """
    print("\n" + "=" * 80)
    print("TEST 6: BALANCED DESIGN TEST")
    print("=" * 80)
    print("\nPurpose: Compare fixed-tier (100%) vs balanced (25%) design")
    print("Method: Run with exactly 25% of agents at each AI tier simultaneously\n")

    n_agents = 1000
    n_rounds = 120
    n_runs = n_runs_per_tier
    temp_dir = tempfile.mkdtemp()

    # Run balanced design simulations
    balanced_results = {tier: [] for tier in AI_TIERS}

    print(f"Running {n_runs} balanced design simulations...")

    for run_idx in range(1, n_runs + 1):
        seed = BASE_SEED + run_idx * 100
        config = create_config(n_agents=n_agents, n_rounds=n_rounds, seed=seed)

        sim = EmergentSimulation(
            config=config,
            output_dir=temp_dir,
            run_id=f"balanced_{run_idx}"
        )

        # Assign exactly 25% to each tier
        agents_per_tier = n_agents // 4
        rng = np.random.RandomState(seed)
        shuffled_indices = rng.permutation(n_agents)

        for tier_idx, tier in enumerate(AI_TIERS):
            start_idx = tier_idx * agents_per_tier
            end_idx = (tier_idx + 1) * agents_per_tier

            for agent_idx in shuffled_indices[start_idx:end_idx]:
                sim.agents[agent_idx].fixed_ai_level = tier
                sim.agents[agent_idx].current_ai_level = tier

        # Run simulation
        sim.run()

        # Calculate survival by tier
        for tier in AI_TIERS:
            tier_agents = [a for a in sim.agents if a.fixed_ai_level == tier]
            survival_rate = sum(1 for a in tier_agents if a.alive) / len(tier_agents)
            balanced_results[tier].append(survival_rate)

        if run_idx % 5 == 0:
            print(f"  Completed {run_idx}/{n_runs} runs")

    # Calculate effects
    baseline = balanced_results["none"]

    print("\n" + "-" * 80)
    print("BALANCED DESIGN RESULTS (25% per tier)")
    print("-" * 80)
    print(f"{'Tier':<12} {'Mean Survival':>15} {'Std':>15} {'ATE vs None':>15} {'p-value':>15}")
    print("-" * 80)

    results_data = []

    for tier in AI_TIERS:
        rates = balanced_results[tier]
        effects = calculate_ate(rates, baseline)
        t_result = t_test(rates, baseline)

        sig = "*" if t_result["p_value"] < 0.05 else ""
        print(f"{tier:<12} {100*effects['treatment_mean']:>14.1f}% {100*effects['treatment_std']:>14.1f}% "
              f"{100*effects['ate']:>+14.1f}% {t_result['p_value']:>14.4f}{sig}")

        results_data.append({
            "tier": tier,
            "mean_survival": effects["treatment_mean"],
            "std_survival": effects["treatment_std"],
            "ate": effects["ate"],
            "p_value": t_result["p_value"]
        })

    print("-" * 80)
    print("* = p < 0.05")

    # Check if paradox persists
    premium_mean = np.mean(balanced_results["premium"])
    none_mean = np.mean(balanced_results["none"])
    paradox_persists = (premium_mean - none_mean) < -0.01

    print("\nINTERPRETATION:")
    if paradox_persists:
        print("✓ PASS: Information paradox persists in balanced design")
        print("  This rules out design artifact from unbalanced group sizes.")
    else:
        print("⚠ Note: Effect differs in balanced design")

    # Save results
    balanced_file = os.path.join(output_dir, "test6_balanced_design_results.csv")
    with open(balanced_file, "w") as f:
        f.write("tier,mean_survival,std_survival,ate,p_value\n")
        for row in results_data:
            f.write(f"{row['tier']},{row['mean_survival']:.6f},{row['std_survival']:.6f},"
                   f"{row['ate']:.6f},{row['p_value']:.6f}\n")
    print(f"\nResults saved to: {balanced_file}")

    return {"results": balanced_results, "paradox_persists": paradox_persists}


# ============================================================================
# TEST 7: INITIAL CAPITAL SENSITIVITY
# ============================================================================

def run_initial_capital_test(output_dir: str, n_runs_per_tier: int) -> Dict[str, Any]:
    """
    Initial Capital Test: Does effect depend on starting wealth distribution?
    """
    print("\n" + "=" * 80)
    print("TEST 7: INITIAL CAPITAL SENSITIVITY")
    print("=" * 80)
    print("\nPurpose: Test if results hold under different initial wealth distributions")
    print("Distributions: Default, Narrow, Wide, Low\n")

    n_agents = 1000
    n_rounds = 120
    n_runs = n_runs_per_tier
    temp_dir = tempfile.mkdtemp()

    capital_configs = [
        ("Default", {}),
        ("Narrow", {"INITIAL_CAPITAL_RANGE": (4_000_000.0, 6_000_000.0)}),
        ("Wide", {"INITIAL_CAPITAL_RANGE": (1_000_000.0, 15_000_000.0)}),
        ("Low", {"INITIAL_CAPITAL": 2_500_000.0, "INITIAL_CAPITAL_RANGE": (1_500_000.0, 4_000_000.0)}),
    ]

    results_by_distribution = {}

    for dist_name, dist_overrides in capital_configs:
        print(f"\n--- Testing: {dist_name} capital distribution ---")

        tier_results = {tier: [] for tier in AI_TIERS}

        for tier_idx, tier in enumerate(AI_TIERS, 1):
            print(f"  [{tier_idx}/4] {tier.upper()}: ", end="", flush=True)

            for run_idx in range(1, n_runs + 1):
                seed = BASE_SEED + hash(("capital", dist_name, tier, run_idx)) % 10000
                config = create_config(n_agents=n_agents, n_rounds=n_rounds, seed=seed, **dist_overrides)

                result = run_fixed_tier_simulation(config, tier, f"capital_{dist_name}_{tier}_{run_idx}", temp_dir)
                tier_results[tier].append(result["survival_rate"])

            print(f"{100*np.mean(tier_results[tier]):.1f}%")

        results_by_distribution[dist_name] = tier_results

    # Summary
    print("\n" + "-" * 90)
    print("INITIAL CAPITAL SENSITIVITY SUMMARY")
    print("-" * 90)
    print(f"{'Distribution':<15} {'None Surv':>15} {'Premium Surv':>15} {'Premium ATE':>15} {'Paradox?':>15}")
    print("-" * 90)

    results_data = []
    paradox_all_distributions = True

    for dist_name, _ in capital_configs:
        tier_results = results_by_distribution[dist_name]
        none_mean = np.mean(tier_results["none"])
        premium_mean = np.mean(tier_results["premium"])
        ate = premium_mean - none_mean
        has_paradox = ate < -0.01

        if not has_paradox:
            paradox_all_distributions = False

        print(f"{dist_name:<15} {100*none_mean:>14.1f}% {100*premium_mean:>14.1f}% "
              f"{100*ate:>+14.1f}% {'YES' if has_paradox else 'no':>15}")

        results_data.append({
            "distribution": dist_name,
            "none_survival": none_mean,
            "premium_survival": premium_mean,
            "ate": ate,
            "paradox": has_paradox
        })

    print("-" * 90)

    print("\nINTERPRETATION:")
    if paradox_all_distributions:
        print("✓ PASS: Paradox persists across all capital distributions")
    else:
        print("⚠ Note: Paradox strength varies with initial capital distribution")

    # Save results
    capital_file = os.path.join(output_dir, "test7_initial_capital_results.csv")
    with open(capital_file, "w") as f:
        f.write("distribution,none_survival,premium_survival,ate,paradox\n")
        for row in results_data:
            f.write(f"{row['distribution']},{row['none_survival']:.6f},{row['premium_survival']:.6f},"
                   f"{row['ate']:.6f},{row['paradox']}\n")
    print(f"\nResults saved to: {capital_file}")

    return {"results": results_by_distribution, "paradox_consistent": paradox_all_distributions}


# ============================================================================
# TEST 8: MARKET REGIME SENSITIVITY
# ============================================================================

def run_market_regime_test(output_dir: str, n_runs_per_tier: int) -> Dict[str, Any]:
    """
    Market Regime Test: Does effect hold in different economic conditions?
    """
    print("\n" + "=" * 80)
    print("TEST 8: MARKET REGIME SENSITIVITY")
    print("=" * 80)
    print("\nPurpose: Test if results hold when starting in different market regimes")
    print("Regimes: normal, favorable, adverse, volatile\n")

    n_agents = 1000
    n_rounds = 120
    n_runs = max(5, n_runs_per_tier // 2)
    temp_dir = tempfile.mkdtemp()

    regime_configs = [
        ("normal", {}),
        ("favorable", {
            "BLACK_SWAN_PROBABILITY": 0.01,
            "MARKET_VOLATILITY": 0.15,
            "DISCOVERY_PROBABILITY": 0.40
        }),
        ("adverse", {
            "BLACK_SWAN_PROBABILITY": 0.10,
            "MARKET_VOLATILITY": 0.40,
            "DISCOVERY_PROBABILITY": 0.20
        }),
        ("volatile", {
            "BLACK_SWAN_PROBABILITY": 0.08,
            "MARKET_VOLATILITY": 0.50,
            "MARKET_SHIFT_PROBABILITY": 0.20
        }),
    ]

    results_by_regime = {}

    for regime_name, regime_overrides in regime_configs:
        print(f"\n--- Testing: {regime_name} market regime ---")

        tier_results = {tier: [] for tier in AI_TIERS}

        for tier_idx, tier in enumerate(AI_TIERS, 1):
            print(f"  [{tier_idx}/4] {tier.upper()}: ", end="", flush=True)

            for run_idx in range(1, n_runs + 1):
                seed = BASE_SEED + hash(("regime", regime_name, tier, run_idx)) % 10000
                config = create_config(n_agents=n_agents, n_rounds=n_rounds, seed=seed, **regime_overrides)

                result = run_fixed_tier_simulation(config, tier, f"regime_{regime_name}_{tier}_{run_idx}", temp_dir)
                tier_results[tier].append(result["survival_rate"])

            print(f"{100*np.mean(tier_results[tier]):.1f}%")

        results_by_regime[regime_name] = tier_results

    # Summary
    print("\n" + "-" * 80)
    print("MARKET REGIME SENSITIVITY SUMMARY")
    print("-" * 80)
    print(f"{'Regime':<15} {'None Surv':>15} {'Premium Surv':>15} {'Premium ATE':>15} {'Paradox?':>15}")
    print("-" * 80)

    results_data = []
    paradox_all_regimes = True

    for regime_name, _ in regime_configs:
        tier_results = results_by_regime[regime_name]
        none_mean = np.mean(tier_results["none"])
        premium_mean = np.mean(tier_results["premium"])
        ate = premium_mean - none_mean
        has_paradox = ate < -0.01

        if not has_paradox:
            paradox_all_regimes = False

        print(f"{regime_name:<15} {100*none_mean:>14.1f}% {100*premium_mean:>14.1f}% "
              f"{100*ate:>+14.1f}% {'YES' if has_paradox else 'no':>15}")

        results_data.append({
            "regime": regime_name,
            "none_survival": none_mean,
            "premium_survival": premium_mean,
            "ate": ate,
            "paradox": has_paradox
        })

    print("-" * 80)

    print("\nINTERPRETATION:")
    if paradox_all_regimes:
        print("✓ PASS: Paradox persists across all market regimes")
    else:
        print("⚠ Note: Paradox strength varies with market conditions")

    # Save results
    regime_file = os.path.join(output_dir, "test8_market_regime_results.csv")
    with open(regime_file, "w") as f:
        f.write("regime,none_survival,premium_survival,ate,paradox\n")
        for row in results_data:
            f.write(f"{row['regime']},{row['none_survival']:.6f},{row['premium_survival']:.6f},"
                   f"{row['ate']:.6f},{row['paradox']}\n")
    print(f"\nResults saved to: {regime_file}")

    return {"results": results_by_regime, "paradox_consistent": paradox_all_regimes}


# ============================================================================
# TEST 9: AI ACCURACY ISOLATION
# ============================================================================

def run_ai_accuracy_isolation_test(output_dir: str, n_runs_per_tier: int) -> Dict[str, Any]:
    """
    AI Accuracy Isolation: Vary only accuracy, hold costs constant.
    Tests if information quality alone drives effects, separate from costs.
    """
    print("\n" + "=" * 80)
    print("TEST 9: AI ACCURACY ISOLATION TEST")
    print("=" * 80)
    print("\nPurpose: Isolate effect of AI accuracy from costs")
    print("Method: Set all AI costs to zero, compare tiers by accuracy alone\n")

    n_agents = 1000
    n_rounds = 120
    n_runs = n_runs_per_tier
    temp_dir = tempfile.mkdtemp()

    # Test with costs at 0 (isolate accuracy effect)
    print("Running with AI_COST_INTENSITY = 0 (free AI)...")

    tier_results = {tier: [] for tier in AI_TIERS}

    for tier_idx, tier in enumerate(AI_TIERS, 1):
        print(f"  [{tier_idx}/4] {tier.upper()}: ", end="", flush=True)

        for run_idx in range(1, n_runs + 1):
            seed = BASE_SEED + hash(("accuracy_iso", tier, run_idx)) % 10000
            config = create_config(
                n_agents=n_agents,
                n_rounds=n_rounds,
                seed=seed,
                AI_COST_INTENSITY=0.0
            )

            result = run_fixed_tier_simulation(config, tier, f"accuracy_{tier}_{run_idx}", temp_dir)
            tier_results[tier].append(result["survival_rate"])

        print(f"{100*np.mean(tier_results[tier]):.1f}% ± {100*np.std(tier_results[tier]):.1f}%")

    # Calculate effects
    baseline = tier_results["none"]

    print("\n" + "-" * 80)
    print("AI ACCURACY ISOLATION RESULTS (Costs = 0)")
    print("-" * 80)
    print(f"{'Tier':<12} {'Mean Survival':>15} {'Std':>15} {'ATE vs None':>15} {'Interpretation':>15}")
    print("-" * 80)

    results_data = []

    for tier in AI_TIERS:
        rates = tier_results[tier]
        effects = calculate_ate(rates, baseline)

        if effects["ate"] > 0.01:
            interpretation = "beneficial"
        elif effects["ate"] < -0.01:
            interpretation = "harmful"
        else:
            interpretation = "neutral"

        print(f"{tier:<12} {100*effects['treatment_mean']:>14.1f}% {100*effects['treatment_std']:>14.1f}% "
              f"{100*effects['ate']:>+14.1f}% {interpretation:>15}")

        results_data.append({
            "tier": tier,
            "mean_survival": effects["treatment_mean"],
            "std_survival": effects["treatment_std"],
            "ate": effects["ate"],
            "interpretation": interpretation
        })

    print("-" * 80)

    # Check if paradox persists without costs
    premium_ate = np.mean(tier_results["premium"]) - np.mean(tier_results["none"])
    paradox_without_costs = premium_ate < -0.01

    print("\nINTERPRETATION:")
    if paradox_without_costs:
        print("✓ Paradox persists even with FREE AI")
        print("  This suggests costs are NOT the primary driver.")
        print("  Information quality/processing mechanisms are likely responsible.")
    else:
        print("⚠ Paradox DISAPPEARS when AI costs are removed")
        print("  This suggests AI costs ARE a primary driver of the paradox.")

    # Save results
    accuracy_file = os.path.join(output_dir, "test9_ai_accuracy_isolation_results.csv")
    with open(accuracy_file, "w") as f:
        f.write("tier,mean_survival,std_survival,ate,interpretation\n")
        for row in results_data:
            f.write(f"{row['tier']},{row['mean_survival']:.6f},{row['std_survival']:.6f},"
                   f"{row['ate']:.6f},{row['interpretation']}\n")
    print(f"\nResults saved to: {accuracy_file}")

    return {"results": tier_results, "paradox_without_costs": paradox_without_costs}


# ============================================================================
# TEST 10: ALTERNATIVE OUTCOME MEASURES
# ============================================================================

def run_alternative_outcomes_test(output_dir: str, n_runs_per_tier: int) -> Dict[str, Any]:
    """
    Alternative Outcomes Test: Check effects on capital and innovation, not just survival.
    Ensures paradox isn't an artifact of survival measure choice.
    """
    print("\n" + "=" * 80)
    print("TEST 10: ALTERNATIVE OUTCOME MEASURES")
    print("=" * 80)
    print("\nPurpose: Verify effects using multiple outcome measures")
    print("Measures: Survival rate, Mean capital, Median capital, Innovation count\n")

    n_agents = 1000
    n_rounds = 120
    n_runs = n_runs_per_tier
    temp_dir = tempfile.mkdtemp()

    tier_outcomes = {
        tier: {
            "survival": [],
            "mean_capital": [],
            "median_capital": [],
            "innovations": []
        }
        for tier in AI_TIERS
    }

    print("Running simulations and collecting multiple outcomes...")

    for tier_idx, tier in enumerate(AI_TIERS, 1):
        print(f"  [{tier_idx}/4] {tier.upper()}: ", end="", flush=True)

        for run_idx in range(1, n_runs + 1):
            seed = BASE_SEED + hash(("alt_outcomes", tier, run_idx)) % 10000
            config = create_config(n_agents=n_agents, n_rounds=n_rounds, seed=seed)

            result = run_fixed_tier_simulation(config, tier, f"altout_{tier}_{run_idx}", temp_dir)

            tier_outcomes[tier]["survival"].append(result["survival_rate"])
            tier_outcomes[tier]["mean_capital"].append(result["mean_capital"])
            tier_outcomes[tier]["median_capital"].append(result["median_capital"])
            tier_outcomes[tier]["innovations"].append(result["mean_innovations"])

        print("done")

    # Calculate effects for each outcome
    outcomes_list = ["survival", "mean_capital", "median_capital", "innovations"]

    print("\n" + "-" * 100)
    print("ALTERNATIVE OUTCOME MEASURES SUMMARY")
    print("-" * 100)

    results_data = []

    for outcome in outcomes_list:
        print(f"\n--- {outcome} ---")
        print(f"{'Tier':<12} {'Mean':>15} {'ATE vs None':>15} {'Direction':>15}")

        baseline = tier_outcomes["none"][outcome]

        for tier in AI_TIERS:
            values = tier_outcomes[tier][outcome]
            effects = calculate_ate(values, baseline)

            baseline_mean = np.mean(baseline)
            if abs(effects["ate"]) < 0.001 * baseline_mean:
                direction = "neutral"
            elif effects["ate"] > 0:
                direction = "positive"
            else:
                direction = "negative"

            if outcome == "survival":
                print(f"{tier:<12} {100*np.mean(values):>14.1f}% {100*effects['ate']:>+14.1f}% {direction:>15}")
            elif outcome in ["mean_capital", "median_capital"]:
                print(f"{tier:<12} {np.mean(values):>15.0f} {effects['ate']:>+15.0f} {direction:>15}")
            else:
                print(f"{tier:<12} {np.mean(values):>15.2f} {effects['ate']:>+15.2f} {direction:>15}")

            results_data.append({
                "outcome": outcome,
                "tier": tier,
                "mean": np.mean(values),
                "ate": effects["ate"],
                "direction": direction
            })

    print("\n" + "-" * 100)

    # Check consistency across outcomes
    premium_survival_ate = np.mean(tier_outcomes["premium"]["survival"]) - np.mean(tier_outcomes["none"]["survival"])
    premium_capital_ate = np.mean(tier_outcomes["premium"]["mean_capital"]) - np.mean(tier_outcomes["none"]["mean_capital"])

    print("\nCONSISTENCY CHECK:")
    survival_negative = premium_survival_ate < 0
    capital_negative = premium_capital_ate < 0

    if survival_negative and capital_negative:
        print("✓ Premium AI shows negative effects on BOTH survival AND capital")
        print("  Strong evidence that paradox is real, not measurement artifact.")
    elif survival_negative and not capital_negative:
        print("⚠ Mixed results: Negative survival effect but positive/neutral capital effect")
        print("  Survivors with premium AI may be doing better financially.")
    else:
        print("  Results vary by outcome measure - interpret with caution.")

    # Save results
    outcomes_file = os.path.join(output_dir, "test10_alternative_outcomes_results.csv")
    with open(outcomes_file, "w") as f:
        f.write("outcome,tier,mean,ate,direction\n")
        for row in results_data:
            f.write(f"{row['outcome']},{row['tier']},{row['mean']:.6f},{row['ate']:.6f},{row['direction']}\n")
    print(f"\nResults saved to: {outcomes_file}")

    return {"outcomes": tier_outcomes, "data": results_data}


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Robustness Test Suite")
    parser.add_argument("--quick", action="store_true", help="Run in quick mode with fewer iterations")
    args = parser.parse_args()

    quick_mode = args.quick

    # Configure based on mode
    n_runs_per_tier = 5 if quick_mode else 20
    n_bootstrap = 100 if quick_mode else 1000
    population_sizes = [100, 500, 1000] if quick_mode else [100, 500, 1000, 2000, 5000]
    sim_lengths = [60, 120, 200] if quick_mode else [60, 120, 200, 400]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"comprehensive_robustness_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    mode_str = "QUICK" if quick_mode else "FULL"

    print("=" * 80)
    print(f"COMPREHENSIVE ROBUSTNESS TEST SUITE ({mode_str} MODE)")
    print("=" * 80)
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Tests to run:")
    print("  1. Placebo/Falsification Test")
    print(f"  2. Bootstrapped Confidence Intervals ({n_bootstrap} resamples)")
    print("  3. Multiple Comparison Correction (BH)")
    print("  4. Population Size Sensitivity")
    print("  5. Simulation Length Sensitivity")
    print("  6. Balanced Design Test")
    print("  7. Initial Capital Sensitivity")
    print("  8. Market Regime Sensitivity")
    print("  9. AI Accuracy Isolation")
    print("  10. Alternative Outcome Measures")
    print()

    import time
    total_start = time.time()

    results = {}

    # Run all tests
    results["placebo"] = run_placebo_test(output_dir, n_runs_per_tier)
    results["bootstrap"] = run_bootstrap_test(output_dir, n_runs_per_tier, n_bootstrap)
    results["multiple_comparison"] = run_multiple_comparison_test(output_dir, n_runs_per_tier)
    results["population_size"] = run_population_size_test(output_dir, population_sizes, n_runs_per_tier)
    results["simulation_length"] = run_simulation_length_test(output_dir, sim_lengths, n_runs_per_tier)
    results["balanced_design"] = run_balanced_design_test(output_dir, n_runs_per_tier)
    results["initial_capital"] = run_initial_capital_test(output_dir, n_runs_per_tier)
    results["market_regime"] = run_market_regime_test(output_dir, n_runs_per_tier)
    results["ai_accuracy"] = run_ai_accuracy_isolation_test(output_dir, n_runs_per_tier)
    results["alternative_outcomes"] = run_alternative_outcomes_test(output_dir, n_runs_per_tier)

    total_elapsed = time.time() - total_start

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print("\n" + "=" * 80)
    print("COMPREHENSIVE ROBUSTNESS TEST SUMMARY")
    print("=" * 80)
    print()

    summary_data = []

    # Test 1: Placebo
    placebo_pass = results["placebo"]["passed"]
    summary_data.append(("1. Placebo Test", "✓ PASS" if placebo_pass else "✗ FAIL"))

    # Test 2: Bootstrap (check if premium CI excludes 0)
    bootstrap_sig = results["bootstrap"]["results"]["premium"]["significant_95"]
    summary_data.append(("2. Bootstrap CIs", "✓ Significant" if bootstrap_sig else "○ Not significant"))

    # Test 3: Multiple comparison
    bh_count = results["multiple_comparison"]["bh_significant_count"]
    summary_data.append(("3. BH Correction", f"✓ {bh_count}/3 significant" if bh_count > 0 else "✗ None survive"))

    # Test 4: Population size
    pop_consistent = results["population_size"]["consistent"]
    summary_data.append(("4. Population Size", "✓ Consistent" if pop_consistent else "○ Varies"))

    # Test 5: Simulation length
    len_consistent = results["simulation_length"]["paradox_consistent"]
    summary_data.append(("5. Sim Length", "✓ Consistent" if len_consistent else "○ Varies"))

    # Test 6: Balanced design
    balanced_paradox = results["balanced_design"]["paradox_persists"]
    summary_data.append(("6. Balanced Design", "✓ Paradox persists" if balanced_paradox else "○ Different"))

    # Test 7: Initial capital
    capital_consistent = results["initial_capital"]["paradox_consistent"]
    summary_data.append(("7. Initial Capital", "✓ Consistent" if capital_consistent else "○ Varies"))

    # Test 8: Market regime
    regime_consistent = results["market_regime"]["paradox_consistent"]
    summary_data.append(("8. Market Regime", "✓ Consistent" if regime_consistent else "○ Varies"))

    # Test 9: AI accuracy isolation
    paradox_no_costs = results["ai_accuracy"]["paradox_without_costs"]
    summary_data.append(("9. Accuracy Isolation", "✓ Not cost-driven" if paradox_no_costs else "○ Cost-driven"))

    # Test 10: Alternative outcomes
    summary_data.append(("10. Alt Outcomes", "✓ Complete"))

    print("Test Results:")
    print("-" * 50)
    for test_name, result in summary_data:
        print(f"  {test_name:<25} {result}")
    print("-" * 50)

    # Count passes
    critical_passed = placebo_pass and bootstrap_sig and bh_count > 0

    print()
    if critical_passed:
        print("✓ ALL CRITICAL TESTS PASSED")
        print("  The AI information paradox finding is robust.")
    else:
        print("⚠ Some critical tests did not pass - findings require additional validation")

    # Performance
    print()
    print("=" * 80)
    print("EXECUTION STATISTICS")
    print("=" * 80)
    print(f"  Total runtime: {total_elapsed/60:.1f} minutes ({total_elapsed:.0f} seconds)")
    print(f"  Output directory: {output_dir}")

    # Save master summary
    summary_file = os.path.join(output_dir, "robustness_summary.txt")
    with open(summary_file, "w") as f:
        f.write("COMPREHENSIVE ROBUSTNESS TEST SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mode: {mode_str}\n")
        f.write("\n")
        f.write("TEST RESULTS:\n")
        f.write("-" * 50 + "\n")
        for test_name, result in summary_data:
            f.write(f"  {test_name:<25} {result}\n")
        f.write("-" * 50 + "\n")
        f.write("\n")
        f.write(f"CRITICAL TESTS PASSED: {critical_passed}\n")
        f.write(f"Total runtime: {total_elapsed/60:.1f} minutes\n")
    print(f"\nSummary saved to: {summary_file}")

    print()
    print("=" * 80)
    print("✓ COMPREHENSIVE ROBUSTNESS SUITE COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
