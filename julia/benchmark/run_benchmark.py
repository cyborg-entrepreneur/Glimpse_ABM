#!/usr/bin/env python3
"""
Self-contained benchmark for Python ABM performance.

Tests core computational patterns used in GLIMPSE ABM.
"""

import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np


@dataclass
class AgentState:
    id: int
    capital: float = 500_000.0
    alive: bool = True
    ai_level: int = 0  # 0=none, 1=basic, 2=advanced, 3=premium
    risk_tolerance: float = 0.5
    adaptability: float = 0.5
    sector_experience: np.ndarray = field(default_factory=lambda: np.zeros(5))
    beliefs: Dict[str, float] = field(default_factory=lambda: {"market": 0.5, "tech": 0.5, "demand": 0.5})
    portfolio_value: float = 0.0
    innovation_count: int = 0


@dataclass
class Opportunity:
    id: int
    expected_return: float
    failure_prob: float
    capital_required: float
    sector: int
    maturity: int


def create_agent(id: int, rng: np.random.Generator) -> AgentState:
    return AgentState(
        id=id,
        capital=500_000.0,
        alive=True,
        ai_level=rng.integers(0, 4),
        risk_tolerance=rng.random(),
        adaptability=rng.random(),
        sector_experience=rng.random(5),
        beliefs={"market": 0.5, "tech": 0.5, "demand": 0.5},
        portfolio_value=0.0,
        innovation_count=0
    )


def generate_opportunities(rng: np.random.Generator, n: int = 10) -> List[Opportunity]:
    return [
        Opportunity(
            id=i,
            expected_return=rng.normal(0.05, 0.15),
            failure_prob=rng.uniform(0.1, 0.4),
            capital_required=rng.uniform(50_000, 150_000),
            sector=rng.integers(1, 6),
            maturity=rng.integers(3, 13)
        )
        for i in range(n)
    ]


def evaluate_opportunity(agent: AgentState, opp: Opportunity, rng: np.random.Generator) -> float:
    if not agent.alive:
        return float('-inf')

    # Base expected value
    ev = opp.expected_return * (1 - opp.failure_prob)

    # AI accuracy bonus
    ai_bonus = agent.ai_level * 0.015

    # Risk adjustment
    risk_adj = agent.risk_tolerance * (1 - opp.failure_prob)

    # Sector experience
    sector_exp = agent.sector_experience[opp.sector - 1]

    # Belief adjustment
    belief_adj = agent.beliefs.get("market", 0.5) * 0.1

    # Add noise
    noise = rng.normal(0, 0.02)

    return ev + ai_bonus + risk_adj * 0.1 + sector_exp * 0.05 + belief_adj + noise


def make_decision(agent: AgentState, opportunities: List[Opportunity], rng: np.random.Generator) -> None:
    if not agent.alive:
        return

    # Evaluate all opportunities
    scores = [evaluate_opportunity(agent, opp, rng) for opp in opportunities]

    # Find best opportunity
    best_idx = np.argmax(scores)
    best_opp = opportunities[best_idx]

    # Check if we can afford it
    if best_opp.capital_required > agent.capital * 0.3:
        return  # Skip investment

    # Simulate investment outcome
    if rng.random() < best_opp.failure_prob:
        # Failed
        agent.capital -= best_opp.capital_required * 0.5
    else:
        # Success
        agent.capital += best_opp.capital_required * best_opp.expected_return
        agent.innovation_count += 1


def update_agent(agent: AgentState, rng: np.random.Generator) -> None:
    if not agent.alive:
        return

    # Operational costs
    agent.capital -= 80_000 / 12

    # Market returns on remaining capital
    market_return = rng.normal(0, 0.02)
    agent.capital *= (1 + market_return)

    # Update beliefs (Bayesian-like update)
    for key in agent.beliefs:
        agent.beliefs[key] = np.clip(agent.beliefs[key] + rng.normal(0, 0.01), 0.0, 1.0)

    # Check survival
    if agent.capital < 200_000:
        agent.alive = False


def run_simulation(n_agents: int, n_rounds: int, seed: int = 42) -> Tuple[int, float, int]:
    rng = np.random.default_rng(seed)

    # Initialize agents
    agents = [create_agent(i, rng) for i in range(n_agents)]

    # Run simulation
    for round_num in range(n_rounds):
        # Generate opportunities
        opportunities = generate_opportunities(rng)

        # Agent decision loop
        for agent in agents:
            make_decision(agent, opportunities, rng)
            update_agent(agent, rng)

    # Return statistics
    survivors = sum(1 for a in agents if a.alive)
    alive_agents = [a for a in agents if a.alive]
    mean_capital = np.mean([a.capital for a in alive_agents]) if alive_agents else 0
    total_innovations = sum(a.innovation_count for a in agents)

    return survivors, mean_capital, total_innovations


def run_benchmark(n_agents: int, n_rounds: int, n_trials: int = 5, warmup: int = 1) -> List[float]:
    # Warmup
    for _ in range(warmup):
        run_simulation(n_agents, n_rounds)

    # Timed runs
    times = []
    for trial in range(n_trials):
        start = time.perf_counter()
        run_simulation(n_agents, n_rounds, seed=42 + trial)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return times


def main():
    print("=" * 70)
    print("GLIMPSE ABM - PYTHON BENCHMARK")
    print("=" * 70)
    print()
    print("Testing core simulation patterns (agent decisions, market dynamics)")
    print()

    # Benchmark configurations
    configs = [
        (100, 50, "Small"),
        (500, 100, "Medium"),
        (1000, 200, "Standard"),
        (2000, 200, "Large"),
    ]

    results = []

    for n_agents, n_rounds, label in configs:
        agent_rounds = n_agents * n_rounds

        print(f"[{label}] {n_agents} agents × {n_rounds} rounds...", end="", flush=True)
        times = run_benchmark(n_agents, n_rounds, n_trials=5, warmup=1)

        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        throughput = agent_rounds / mean_time

        results.append({
            "label": label,
            "agents": n_agents,
            "rounds": n_rounds,
            "mean": mean_time,
            "std": std_time,
            "throughput": throughput
        })

        print(" done")
        print(f"  Time: {mean_time:.4f} s (±{std_time:.4f} s)")
        print(f"  Throughput: {throughput:,.0f} agent-rounds/sec")
        print()

    # Summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Config':<10} {'Agents':>8} {'Rounds':>8} {'Time (s)':>12} {'Throughput':>18}")
    print("-" * 60)
    for r in results:
        print(f"{r['label']:<10} {r['agents']:>8} {r['rounds']:>8} {r['mean']:>12.4f} {r['throughput']:>15,.0f} /s")

    return results


if __name__ == "__main__":
    main()
