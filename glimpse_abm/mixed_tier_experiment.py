"""
Mixed-Population Fixed-Tier Experiment

This experiment runs simulations where agents with different AI tiers compete
directly against each other within the same market environment.

Design:
- 1000 agents per run
- 250 agents assigned to each tier (None, Basic, Advanced, Premium)
- Tiers fixed throughout all 60 rounds (no switching)
- 50 independent runs for statistical power

This differs from the standard fixed-tier analysis where all agents in a run
share the same tier. Here, agents compete directly across tiers within each run.
"""

import copy
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .config import EmergentConfig
from .simulation import EmergentSimulation


def run_mixed_tier_simulation(args: tuple) -> Dict[str, Any]:
    """Execute a single mixed-tier simulation run."""
    run_index, base_config, output_dir, tier_assignments = args
    run_id = f"mixed_tier_run_{run_index}"

    run_config = copy.deepcopy(base_config)
    run_config.AGENT_AI_MODE = "fixed"

    try:
        sim = EmergentSimulation(
            config=run_config,
            output_dir=output_dir,
            run_id=run_id
        )

        # Assign tiers to agents based on the tier_assignments list
        for i, agent in enumerate(sim.agents):
            agent.fixed_ai_level = tier_assignments[i]

        sim.run()

        # Collect per-tier survival statistics from this run
        tier_stats = {tier: {"survived": 0, "failed": 0, "total": 0}
                      for tier in ["none", "basic", "advanced", "premium"]}

        for agent in sim.agents:
            tier = agent.fixed_ai_level
            tier_stats[tier]["total"] += 1
            if agent.alive:
                tier_stats[tier]["survived"] += 1
            else:
                tier_stats[tier]["failed"] += 1

        return {
            "run_id": run_id,
            "run_index": run_index,
            "status": "completed",
            "tier_stats": tier_stats
        }

    except Exception as exc:
        print(f"ERROR in mixed-tier run [{run_id}]: {exc}")
        traceback.print_exc()
        return {
            "run_id": run_id,
            "run_index": run_index,
            "status": f"error: {exc}",
            "tier_stats": None
        }


def create_tier_assignments(n_agents: int = 1000) -> List[str]:
    """
    Create balanced tier assignments: 250 agents per tier.
    Shuffle to avoid any ordering effects.
    """
    tiers = ["none", "basic", "advanced", "premium"]
    agents_per_tier = n_agents // len(tiers)

    assignments = []
    for tier in tiers:
        assignments.extend([tier] * agents_per_tier)

    # Handle remainder if n_agents not divisible by 4
    remainder = n_agents - len(assignments)
    for i in range(remainder):
        assignments.append(tiers[i % len(tiers)])

    # Shuffle to randomize agent positions
    np.random.shuffle(assignments)

    return assignments


def run_mixed_tier_experiment(
    n_agents: int = 1000,
    n_runs: int = 50,
    n_rounds: int = 60,
    output_dir: Optional[str] = None,
    parallel_workers: Optional[int] = None,
    base_config: Optional[EmergentConfig] = None,
) -> Dict[str, Any]:
    """
    Execute the mixed-population fixed-tier experiment.

    Parameters:
    -----------
    n_agents : int
        Number of agents per run (default: 1000)
    n_runs : int
        Number of independent runs (default: 50)
    n_rounds : int
        Number of rounds per run (default: 60)
    output_dir : str, optional
        Directory for results (auto-generated if not specified)
    parallel_workers : int, optional
        Number of parallel workers (default: CPU count - 1)
    base_config : EmergentConfig, optional
        Base configuration to use (default: creates new one)

    Returns:
    --------
    Dict with experiment results and statistics
    """

    # Setup configuration
    config = base_config or EmergentConfig()
    config.N_AGENTS = n_agents
    config.N_ROUNDS = n_rounds
    config.N_RUNS = n_runs
    config.AGENT_AI_MODE = "fixed"

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_output_dir = output_dir or f"./mixed_tier_experiment_{timestamp}"
    os.makedirs(exp_output_dir, exist_ok=True)

    # Determine parallel workers
    if parallel_workers is None:
        import multiprocessing
        parallel_workers = max(1, multiprocessing.cpu_count() - 1)

    print("=" * 70)
    print("  MIXED-POPULATION FIXED-TIER EXPERIMENT")
    print("=" * 70)
    print(f"  Agents per run:     {n_agents}")
    print(f"  Agents per tier:    {n_agents // 4}")
    print(f"  Number of runs:     {n_runs}")
    print(f"  Rounds per run:     {n_rounds}")
    print(f"  Parallel workers:   {parallel_workers}")
    print(f"  Output directory:   {exp_output_dir}")
    print("=" * 70)

    # Create jobs - each run gets a fresh shuffled tier assignment
    jobs = []
    for run_idx in range(n_runs):
        tier_assignments = create_tier_assignments(n_agents)
        jobs.append((run_idx, config, exp_output_dir, tier_assignments))

    # Execute runs in parallel
    start_time = time.time()
    all_results = []

    print(f"\nStarting {n_runs} simulation runs...")

    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {executor.submit(run_mixed_tier_simulation, job): job[0]
                   for job in jobs}

        completed = 0
        for future in as_completed(futures):
            run_idx = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                completed += 1

                if completed % 10 == 0 or completed == n_runs:
                    elapsed = time.time() - start_time
                    print(f"  Completed {completed}/{n_runs} runs ({elapsed:.1f}s elapsed)")

            except Exception as exc:
                print(f"  Run {run_idx} generated an exception: {exc}")
                all_results.append({
                    "run_id": f"mixed_tier_run_{run_idx}",
                    "run_index": run_idx,
                    "status": f"exception: {exc}",
                    "tier_stats": None
                })

    elapsed_total = time.time() - start_time
    print(f"\nAll runs complete! Total time: {elapsed_total:.1f} seconds")

    # Aggregate results
    print("\nAggregating results...")
    results_summary = aggregate_mixed_tier_results(all_results, exp_output_dir)

    # Save experiment metadata
    metadata = {
        "experiment_type": "mixed_population_fixed_tier",
        "n_agents": n_agents,
        "agents_per_tier": n_agents // 4,
        "n_runs": n_runs,
        "n_rounds": n_rounds,
        "timestamp": timestamp,
        "elapsed_seconds": elapsed_total,
        "output_directory": exp_output_dir,
    }

    metadata_path = Path(exp_output_dir) / "experiment_metadata.json"
    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nResults saved to: {exp_output_dir}")

    return {
        "metadata": metadata,
        "results_summary": results_summary,
        "raw_results": all_results,
        "output_directory": exp_output_dir,
    }


def aggregate_mixed_tier_results(
    results: List[Dict[str, Any]],
    output_dir: str
) -> Dict[str, Any]:
    """Aggregate results across all runs and compute statistics."""

    tiers = ["none", "basic", "advanced", "premium"]

    # Collect survival rates per tier per run
    tier_survival_rates = {tier: [] for tier in tiers}
    tier_counts = {tier: {"survived": 0, "failed": 0, "total": 0} for tier in tiers}

    successful_runs = 0
    for result in results:
        if result["status"] == "completed" and result["tier_stats"]:
            successful_runs += 1
            for tier in tiers:
                stats = result["tier_stats"][tier]
                if stats["total"] > 0:
                    survival_rate = stats["survived"] / stats["total"]
                    tier_survival_rates[tier].append(survival_rate)
                    tier_counts[tier]["survived"] += stats["survived"]
                    tier_counts[tier]["failed"] += stats["failed"]
                    tier_counts[tier]["total"] += stats["total"]

    print(f"  Successfully completed runs: {successful_runs}/{len(results)}")

    # Compute summary statistics
    summary_stats = {}
    for tier in tiers:
        rates = tier_survival_rates[tier]
        if rates:
            summary_stats[tier] = {
                "mean_survival_rate": np.mean(rates),
                "std_survival_rate": np.std(rates),
                "min_survival_rate": np.min(rates),
                "max_survival_rate": np.max(rates),
                "total_survived": tier_counts[tier]["survived"],
                "total_failed": tier_counts[tier]["failed"],
                "total_agents": tier_counts[tier]["total"],
                "overall_survival_rate": (
                    tier_counts[tier]["survived"] / tier_counts[tier]["total"]
                    if tier_counts[tier]["total"] > 0 else 0
                ),
                "n_runs": len(rates),
            }

    # Print summary table
    print("\n" + "=" * 70)
    print("  MIXED-TIER EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Tier':<12} {'Survival Rate':>15} {'Std Dev':>10} {'95% CI':>20}")
    print("-" * 70)

    for tier in tiers:
        if tier in summary_stats:
            s = summary_stats[tier]
            mean = s["mean_survival_rate"] * 100
            std = s["std_survival_rate"] * 100
            n = s["n_runs"]
            ci_width = 1.96 * std / np.sqrt(n) if n > 1 else 0
            ci_low = mean - ci_width
            ci_high = mean + ci_width
            print(f"  {tier.capitalize():<12} {mean:>14.2f}% {std:>9.2f}% [{ci_low:>7.2f}%, {ci_high:>6.2f}%]")

    print("-" * 70)

    # Compute treatment effects (relative to "none" baseline)
    if "none" in summary_stats:
        baseline = summary_stats["none"]["mean_survival_rate"]
        print(f"\n  Treatment Effects (vs No AI baseline):")
        for tier in ["basic", "advanced", "premium"]:
            if tier in summary_stats:
                effect = (summary_stats[tier]["mean_survival_rate"] - baseline) * 100
                print(f"    {tier.capitalize():<12}: {effect:+.2f} percentage points")

    print("=" * 70)

    # Save to CSV
    df_rows = []
    for tier in tiers:
        if tier in summary_stats:
            s = summary_stats[tier]
            df_rows.append({
                "tier": tier,
                "mean_survival_rate": s["mean_survival_rate"],
                "std_survival_rate": s["std_survival_rate"],
                "min_survival_rate": s["min_survival_rate"],
                "max_survival_rate": s["max_survival_rate"],
                "total_survived": s["total_survived"],
                "total_failed": s["total_failed"],
                "total_agents": s["total_agents"],
                "overall_survival_rate": s["overall_survival_rate"],
                "n_runs": s["n_runs"],
            })

    df = pd.DataFrame(df_rows)
    csv_path = Path(output_dir) / "mixed_tier_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")

    # Save per-run data for detailed analysis
    run_data = []
    for result in results:
        if result["status"] == "completed" and result["tier_stats"]:
            for tier in tiers:
                stats = result["tier_stats"][tier]
                if stats["total"] > 0:
                    run_data.append({
                        "run_index": result["run_index"],
                        "tier": tier,
                        "survived": stats["survived"],
                        "failed": stats["failed"],
                        "total": stats["total"],
                        "survival_rate": stats["survived"] / stats["total"],
                    })

    df_runs = pd.DataFrame(run_data)
    runs_csv_path = Path(output_dir) / "mixed_tier_per_run.csv"
    df_runs.to_csv(runs_csv_path, index=False)
    print(f"Per-run data saved to: {runs_csv_path}")

    return {
        "summary_stats": summary_stats,
        "tier_survival_rates": tier_survival_rates,
        "successful_runs": successful_runs,
    }


def main():
    """Command-line entry point for the mixed-tier experiment."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Mixed-Population Fixed-Tier Experiment"
    )
    parser.add_argument(
        "--agents", type=int, default=1000,
        help="Number of agents per run (default: 1000)"
    )
    parser.add_argument(
        "--runs", type=int, default=50,
        help="Number of independent runs (default: 50)"
    )
    parser.add_argument(
        "--rounds", type=int, default=60,
        help="Number of rounds per run (default: 60)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: CPU count - 1)"
    )

    args = parser.parse_args()

    results = run_mixed_tier_experiment(
        n_agents=args.agents,
        n_runs=args.runs,
        n_rounds=args.rounds,
        output_dir=args.output_dir,
        parallel_workers=args.workers,
    )

    return results


if __name__ == "__main__":
    main()
