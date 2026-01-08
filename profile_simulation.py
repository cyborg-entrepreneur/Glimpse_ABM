"""Utility for profiling Glimpse ABM runs with cProfile."""

from __future__ import annotations

import argparse
import cProfile
import pstats
from pathlib import Path

from .config import EmergentConfig
from .simulation import EmergentSimulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a Glimpse ABM simulation run.")
    parser.add_argument(
        "--agents",
        type=int,
        default=500,
        help="Number of agents (default: 500)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help="Number of rounds (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./profiled_run"),
        help="Output directory for the profiled run (default: ./profiled_run)",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("./profile_stats.prof"),
        help="Path to write cProfile stats (default: ./profile_stats.prof)",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="cumtime",
        help="Sort order for pstats output (default: cumtime)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of functions to display from the profile (default: 30)",
    )
    return parser.parse_args()


def build_simulation(n_agents: int, n_rounds: int, output_dir: Path) -> EmergentSimulation:
    cfg = EmergentConfig()
    cfg.N_AGENTS = n_agents
    cfg.N_ROUNDS = n_rounds
    cfg.use_parallel = False  # deterministic profiling
    output_dir.mkdir(parents=True, exist_ok=True)
    return EmergentSimulation(config=cfg, output_dir=str(output_dir), run_id="profiled_run")


def main() -> None:
    args = parse_args()
    sim = build_simulation(args.agents, args.rounds, args.output)

    profiler = cProfile.Profile()
    profiler.enable()
    sim.run()
    profiler.disable()
    profiler.dump_stats(str(args.stats))

    stats = pstats.Stats(profiler).sort_stats(args.sort)
    stats.print_stats(args.top)


if __name__ == "__main__":
    main()
