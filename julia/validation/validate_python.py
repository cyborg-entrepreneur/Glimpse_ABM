#!/usr/bin/env python3
"""
Run Python GLIMPSE ABM simulation and output validation metrics.
"""

import sys
import json
import os
from pathlib import Path

# Add parent of glimpse_abm to path so we can import it as a package
downloads_dir = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, downloads_dir)
os.chdir(downloads_dir)

try:
    from glimpse_abm.config import EmergentConfig
    from glimpse_abm.simulation import EmergentSimulation
    import numpy as np
    HAS_GLIMPSE = True
except ImportError as e:
    print(f"Error importing glimpse_abm: {e}")
    HAS_GLIMPSE = False
    sys.exit(1)

def run_validation():
    """Run a deterministic simulation and output validation metrics."""

    # Fixed configuration for reproducibility
    config = EmergentConfig()
    config.N_AGENTS = 100
    config.N_ROUNDS = 50
    config.RANDOM_SEED = 42
    config.use_parallel = False

    print("Running Python validation simulation...")
    print(f"  Agents: {config.N_AGENTS}")
    print(f"  Rounds: {config.N_ROUNDS}")
    print(f"  Seed: {config.RANDOM_SEED}")

    # Run simulation
    import tempfile
    tmpdir = tempfile.mkdtemp()
    sim = EmergentSimulation(config=config, output_dir=tmpdir, run_id="validation")
    sim.run()

    # Collect validation metrics
    agents = sim.agents

    # Survival metrics
    alive_agents = [a for a in agents if a.alive]
    dead_agents = [a for a in agents if not a.alive]

    survival_count = len(alive_agents)
    survival_rate = survival_count / len(agents)

    # Capital metrics
    if alive_agents:
        capitals = [a.resources.capital for a in alive_agents]
        avg_capital = np.mean(capitals)
        std_capital = np.std(capitals)
        min_capital = np.min(capitals)
        max_capital = np.max(capitals)
    else:
        avg_capital = std_capital = min_capital = max_capital = 0.0

    # AI tier distribution
    ai_counts = {"none": 0, "basic": 0, "advanced": 0, "premium": 0}
    for a in agents:
        tier = getattr(a, 'primary_ai_level', getattr(a, 'ai_level', 'none'))
        if tier in ai_counts:
            ai_counts[tier] += 1

    # Innovation metrics - Python tracks these differently
    total_innovations = sum(len(getattr(a, 'innovations', [])) for a in agents)

    # Get investment totals from performance tracker
    total_investments = 0.0
    total_deployed = 0.0
    for a in agents:
        perf = a.resources.performance
        deployed = perf.deployed_by_action
        total_deployed += deployed.get('overall', 0.0)
        total_investments += deployed.get('invest', 0.0)

    # Count successes/failures from roi_events
    total_successes = 0
    total_failures = 0
    for a in agents:
        perf = a.resources.performance
        for event in perf.roi_events:
            if event.get('type') == 'return':
                if event.get('amount', 0) > 0:
                    total_successes += 1
                else:
                    total_failures += 1

    results = {
        "language": "python",
        "config": {
            "n_agents": config.N_AGENTS,
            "n_rounds": config.N_ROUNDS,
            "seed": config.RANDOM_SEED
        },
        "survival": {
            "count": survival_count,
            "rate": round(survival_rate, 6)
        },
        "capital": {
            "mean": round(avg_capital, 2),
            "std": round(std_capital, 2),
            "min": round(min_capital, 2),
            "max": round(max_capital, 2)
        },
        "ai_distribution": ai_counts,
        "activity": {
            "total_innovations": total_innovations,
            "total_investments": round(total_investments, 2),
            "total_successes": total_successes,
            "total_failures": total_failures
        }
    }

    # Output as JSON
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS (Python)")
    print("=" * 60)
    print(json.dumps(results, indent=2))

    # Save to file
    output_path = Path(__file__).parent / "python_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results

if __name__ == "__main__":
    run_validation()
