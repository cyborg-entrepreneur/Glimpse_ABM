#!/usr/bin/env python3
"""Smoke test: Python fixed tier simulations"""

import sys
import os
import glob

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
from glimpse_abm.simulation import EmergentSimulation
from glimpse_abm.config import EmergentConfig

SEED = 42
N_AGENTS = 100
N_ROUNDS = 200
N_RUNS = 5

print("="*70)
print("PYTHON SMOKE TEST: Fixed Tier Comparison")
print("="*70)
print(f"Seed: {SEED}, Agents: {N_AGENTS}, Rounds: {N_ROUNDS}, Runs: {N_RUNS}")
print()

results = {}
tiers = ["none", "basic", "advanced", "premium"]


def load_summary_data(output_dir, run_id):
    """Load summary data from pickle files after simulation flush."""
    summary_path = os.path.join(output_dir, run_id, "summary")
    if not os.path.exists(summary_path):
        return []

    all_data = []
    for pkl_file in sorted(glob.glob(os.path.join(summary_path, "batch_*.pkl"))):
        df = pd.read_pickle(pkl_file)
        all_data.extend(df.to_dict("records"))
    return all_data


for tier_idx, tier in enumerate(tiers):
    print(f"\n--- Running {tier} tier ---")

    tier_results = {
        "survival_rate": [],
        "mean_capital": [],
        "median_capital": [],
        "std_capital": [],
        "total_innovations": [],
        "invest_share": [],
        "innovate_share": [],
        "explore_share": [],
        "maintain_share": [],
    }

    for run_idx in range(1, N_RUNS + 1):
        run_seed = SEED + tier_idx * N_RUNS + run_idx

        config = EmergentConfig()
        config.N_AGENTS = N_AGENTS
        config.N_ROUNDS = N_ROUNDS
        config.RANDOM_SEED = run_seed

        output_dir = f"/tmp/smoke_test_python/{tier}_run{run_idx}"
        run_id = f"{tier}_run{run_idx}"

        sim = EmergentSimulation(
            config=config,
            output_dir=output_dir,
            run_id=run_id
        )

        for agent in sim.agents:
            agent.fixed_ai_level = tier
            agent.current_ai_level = tier

        sim.run()

        alive_agents = [a for a in sim.agents if a.alive]
        survival_rate = len(alive_agents) / len(sim.agents)

        capitals = [a.resources.capital for a in alive_agents] if alive_agents else [0]
        mean_cap = np.mean(capitals)
        median_cap = np.median(capitals)
        std_cap = np.std(capitals)

        # Load round data from pickle files (buffer is flushed at end of sim)
        round_data = load_summary_data(output_dir, run_id)

        total_innov = 0
        inv_share = innov_share = exp_share = maint_share = 0

        if round_data:
            total_innov = sum(r.get("innovation_successes", 0) for r in round_data)
            inv_share = np.mean([r.get("action_share_invest", 0) for r in round_data])
            innov_share = np.mean([r.get("action_share_innovate", 0) for r in round_data])
            exp_share = np.mean([r.get("action_share_explore", 0) for r in round_data])
            maint_share = np.mean([r.get("action_share_maintain", 0) for r in round_data])

        tier_results["survival_rate"].append(survival_rate)
        tier_results["mean_capital"].append(mean_cap)
        tier_results["median_capital"].append(median_cap)
        tier_results["std_capital"].append(std_cap)
        tier_results["total_innovations"].append(total_innov)
        tier_results["invest_share"].append(inv_share)
        tier_results["innovate_share"].append(innov_share)
        tier_results["explore_share"].append(exp_share)
        tier_results["maintain_share"].append(maint_share)

        print(".", end="", flush=True)

    print(" done")
    results[tier] = tier_results

print("\n" + "="*70)
print("PYTHON RESULTS SUMMARY")
print("="*70)

for tier in tiers:
    r = results[tier]
    print(f"\n[{tier}]")
    surv_mean = np.mean(r["survival_rate"]) * 100
    surv_std = np.std(r["survival_rate"]) * 100
    cap_mean = np.mean(r["mean_capital"]) / 1e6
    cap_std = np.std(r["mean_capital"]) / 1e6
    med_cap = np.mean(r["median_capital"]) / 1e6
    innov_mean = np.mean(r["total_innovations"])
    innov_std = np.std(r["total_innovations"])
    inv = np.mean(r["invest_share"]) * 100
    inno = np.mean(r["innovate_share"]) * 100
    exp = np.mean(r["explore_share"]) * 100
    maint = np.mean(r["maintain_share"]) * 100

    print(f"  Survival Rate:     {surv_mean:.1f}% (±{surv_std:.1f}%)")
    print(f"  Mean Capital:      ${cap_mean:.2f}M (±${cap_std:.2f}M)")
    print(f"  Median Capital:    ${med_cap:.2f}M")
    print(f"  Total Innovations: {innov_mean:.1f} (±{innov_std:.1f})")
    print(f"  Action Shares: invest={inv:.1f}%, innovate={inno:.1f}%, explore={exp:.1f}%, maintain={maint:.1f}%")

# Save results
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "julia/smoke_test_comparison")
os.makedirs(output_path, exist_ok=True)

with open(os.path.join(output_path, "python_results.txt"), "w") as f:
    f.write("PYTHON SMOKE TEST RESULTS\n")
    f.write(f"Seed={SEED}, Agents={N_AGENTS}, Rounds={N_ROUNDS}, Runs={N_RUNS}\n\n")
    for tier in tiers:
        r = results[tier]
        f.write(f"[{tier}]\n")
        f.write(f"survival_rate: {np.mean(r['survival_rate'])} ± {np.std(r['survival_rate'])}\n")
        f.write(f"mean_capital: {np.mean(r['mean_capital'])} ± {np.std(r['mean_capital'])}\n")
        f.write(f"median_capital: {np.mean(r['median_capital'])}\n")
        f.write(f"std_capital: {np.mean(r['std_capital'])}\n")
        f.write(f"total_innovations: {np.mean(r['total_innovations'])} ± {np.std(r['total_innovations'])}\n")
        f.write(f"invest_share: {np.mean(r['invest_share'])}\n")
        f.write(f"innovate_share: {np.mean(r['innovate_share'])}\n")
        f.write(f"explore_share: {np.mean(r['explore_share'])}\n")
        f.write(f"maintain_share: {np.mean(r['maintain_share'])}\n\n")

print("\nResults saved to julia/smoke_test_comparison/python_results.txt")
