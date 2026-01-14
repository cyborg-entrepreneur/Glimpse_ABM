#!/usr/bin/env python3
"""
Benchmark for Python GLIMPSE ABM implementation.

Measures execution time for equivalent simulation workloads.
"""

import sys
import time
import statistics
from pathlib import Path

# Add glimpse_abm to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "glimpse_abm"))

try:
    from config import EmergentConfig
    from simulation import EmergentSimulation
    HAS_GLIMPSE = True
except ImportError:
    HAS_GLIMPSE = False
    print("Warning: glimpse_abm not found. Using simplified benchmark.")


def benchmark_simulation_full(n_agents: int, n_rounds: int, seed: int = 42) -> int:
    """Run a full simulation benchmark using the actual GLIMPSE ABM."""
    config = EmergentConfig()
    config.N_AGENTS = n_agents
    config.N_ROUNDS = n_rounds
    config.RANDOM_SEED = seed
    config.use_parallel = False  # Ensure fair comparison

    sim = EmergentSimulation(config=config, output_dir=None, run_id="benchmark")
    sim.run()

    return sum(1 for a in sim.agents if a.alive)


def benchmark_simulation_simple(n_agents: int, n_rounds: int, seed: int = 42) -> int:
    """Simplified simulation benchmark for when full package isn't available."""
    import numpy as np

    rng = np.random.default_rng(seed)

    # Initialize agents
    capitals = np.ones(n_agents) * 500_000.0
    alive = np.ones(n_agents, dtype=bool)
    ai_levels = rng.choice(4, size=n_agents)  # 0=none, 1=basic, 2=advanced, 3=premium

    # Simulate rounds
    for round_num in range(n_rounds):
        # Generate opportunities (simplified)
        n_opportunities = 10
        returns = rng.normal(0.05, 0.15, n_opportunities)
        failure_probs = rng.uniform(0.1, 0.4, n_opportunities)

        # Agent decisions
        for i in range(n_agents):
            if not alive[i]:
                continue

            # Evaluate opportunities based on AI level
            ai_bonus = ai_levels[i] * 0.02

            # Pick best opportunity (simplified decision)
            expected_values = returns * (1 - failure_probs + ai_bonus)
            best_opp = np.argmax(expected_values)

            # Investment outcome
            if rng.random() < failure_probs[best_opp]:
                # Failed investment
                capitals[i] *= 0.85
            else:
                # Successful investment
                capitals[i] *= (1 + returns[best_opp])

            # Operational costs
            capitals[i] -= 80_000 / 12

            # Check survival
            if capitals[i] < 200_000:
                alive[i] = False

    return np.sum(alive)


def run_benchmark(n_agents: int, n_rounds: int, n_trials: int = 5, warmup: int = 1, use_full: bool = True):
    """Run multiple trials and compute statistics."""
    benchmark_fn = benchmark_simulation_full if (use_full and HAS_GLIMPSE) else benchmark_simulation_simple

    # Warmup
    for _ in range(warmup):
        benchmark_fn(n_agents, n_rounds)

    # Timed runs
    times = []
    for trial in range(n_trials):
        start = time.perf_counter()
        benchmark_fn(n_agents, n_rounds, seed=42 + trial)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return times


def main():
    print("=" * 70)
    print("GLIMPSE ABM PYTHON BENCHMARK")
    print("=" * 70)
    print()

    use_full = HAS_GLIMPSE
    if use_full:
        print("Using FULL glimpse_abm implementation")
    else:
        print("Using SIMPLIFIED benchmark (glimpse_abm not available)")
    print()

    # Benchmark configurations
    configs = [
        (100, 50),    # Small
        (500, 100),   # Medium
        (1000, 200),  # Standard
    ]

    results = {}

    for n_agents, n_rounds in configs:
        print(f"Benchmarking: {n_agents} agents × {n_rounds} rounds")

        times = run_benchmark(n_agents, n_rounds, n_trials=5, warmup=1, use_full=use_full)

        results[(n_agents, n_rounds)] = {
            "mean": statistics.mean(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times)
        }

        print(f"  Mean: {statistics.mean(times):.3f} s (±{statistics.stdev(times) if len(times) > 1 else 0:.3f} s)")
        print(f"  Range: [{min(times):.3f}, {max(times):.3f}] s")

        # Throughput
        agent_rounds = n_agents * n_rounds
        throughput = agent_rounds / statistics.mean(times)
        print(f"  Throughput: {throughput:.0f} agent-rounds/second")
        print()

    return results


if __name__ == "__main__":
    main()
