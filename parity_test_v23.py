"""Phase 4: Julia↔Python parity test at N=200, 30 rounds, fixed-tier=premium.

Runs the Python EmergentSimulation natively, then invokes the Julia version
via subprocess + JSON dump. Compares aggregate stats across 5 seeds.

Goal: confirm both codebases are statistically equivalent. They will not be
bit-identical (different RNG implementations) but should agree within ~5-10%
on aggregate survival / capital metrics for each tier.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean, stdev

import numpy as np

REPO = Path("/Users/davidtownsend/projects/glimpse-abm")
GLIMPSE_DIR = REPO / "glimpse_abm"
JULIA_DIR = GLIMPSE_DIR / "julia"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(GLIMPSE_DIR))

from glimpse_abm.config import EmergentConfig
from glimpse_abm.simulation import EmergentSimulation

N_AGENTS = 200
N_ROUNDS = 30
SEEDS = [101, 102, 103, 104, 105]
TIER = "premium"


def run_python(seed: int) -> dict:
    cfg = EmergentConfig()
    cfg.N_AGENTS = N_AGENTS
    cfg.N_ROUNDS = N_ROUNDS
    cfg.N_RUNS = 1
    cfg.use_parallel = False
    cfg.RANDOM_SEED = seed
    cfg.AGENT_AI_MODE = "fixed"

    np.random.seed(seed)
    with tempfile.TemporaryDirectory() as tmp:
        sim = EmergentSimulation(
            cfg, output_dir=tmp, run_id=f"parity_py_{seed}",
        )
        # Force every agent to premium (fixed-tier main-analysis path)
        for agent in sim.agents:
            agent.fixed_ai_level = TIER
            agent.current_ai_level = TIER
            cfg_dict = cfg.AI_LEVELS.get(TIER, {})
            if cfg_dict and hasattr(agent, "_ensure_subscription_schedule"):
                agent._ensure_subscription_schedule(TIER, cfg_dict)
        sim.run()

        alive_caps = [a.resources.capital for a in sim.agents if a.alive]
        all_caps = [a.resources.capital for a in sim.agents]
        return {
            "seed": seed,
            "language": "python",
            "n_alive": int(sum(a.alive for a in sim.agents)),
            "survival_rate": sum(a.alive for a in sim.agents) / N_AGENTS,
            "mean_capital_alive": mean(alive_caps) if alive_caps else 0.0,
            "median_capital_alive": float(np.median(alive_caps)) if alive_caps else 0.0,
            "mean_capital_all": mean(all_caps),
        }


JULIA_DRIVER = """
using GlimpseABM, Statistics, JSON3
seed = parse(Int, ARGS[1])
cfg = EmergentConfig(N_AGENTS=200, N_ROUNDS=30, RANDOM_SEED=seed, AGENT_AI_MODE="fixed")
sim = EmergentSimulation(
    config=cfg, seed=seed,
    initial_tier_distribution=Dict("premium" => 1.0),
    output_dir=mktempdir(),
    run_id="parity_jl_$seed",
)
GlimpseABM.run!(sim)
alive_caps = [GlimpseABM.get_capital(a) for a in sim.agents if a.alive]
all_caps = [GlimpseABM.get_capital(a) for a in sim.agents]
n_alive = count(a -> a.alive, sim.agents)
out = Dict(
    "seed" => seed,
    "language" => "julia",
    "n_alive" => n_alive,
    "survival_rate" => n_alive / 200,
    "mean_capital_alive" => isempty(alive_caps) ? 0.0 : mean(alive_caps),
    "median_capital_alive" => isempty(alive_caps) ? 0.0 : median(alive_caps),
    "mean_capital_all" => mean(all_caps),
)
JSON3.write(stdout, out)
"""


def run_julia(seed: int) -> dict:
    driver_path = Path(tempfile.mkstemp(suffix=".jl")[1])
    driver_path.write_text(JULIA_DRIVER)
    try:
        result = subprocess.run(
            ["julia", "--project=.", str(driver_path), str(seed)],
            cwd=JULIA_DIR,
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
        # Last line of stdout is the JSON
        for line in reversed(result.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        raise RuntimeError(f"No JSON found in Julia stdout:\n{result.stdout}")
    finally:
        driver_path.unlink()


def summarize(results: list[dict], lang: str) -> dict:
    survs = [r["survival_rate"] for r in results]
    means = [r["mean_capital_alive"] for r in results]
    medians = [r["median_capital_alive"] for r in results]
    return {
        "language": lang,
        "n_seeds": len(results),
        "survival_mean": mean(survs),
        "survival_std": stdev(survs) if len(survs) > 1 else 0.0,
        "mean_capital_mean": mean(means),
        "mean_capital_std": stdev(means) if len(means) > 1 else 0.0,
        "median_capital_mean": mean(medians),
    }


def main() -> int:
    print(f"Phase 4 parity test — N={N_AGENTS}, R={N_ROUNDS}, "
          f"tier={TIER}, seeds={SEEDS}")
    print("=" * 70)

    py_results, jl_results = [], []
    for seed in SEEDS:
        print(f"\n--- seed {seed} ---")
        print("  Python ...", end="", flush=True)
        py = run_python(seed)
        py_results.append(py)
        print(f" surv={py['survival_rate']:.3f}  "
              f"cap_alive=${py['mean_capital_alive']/1e6:.2f}M")
        print("  Julia  ...", end="", flush=True)
        jl = run_julia(seed)
        jl_results.append(jl)
        print(f" surv={jl['survival_rate']:.3f}  "
              f"cap_alive=${jl['mean_capital_alive']/1e6:.2f}M")

    py_sum = summarize(py_results, "python")
    jl_sum = summarize(jl_results, "julia")

    print("\n" + "=" * 70)
    print("Aggregate (mean ± std across seeds):")
    print(f"  Python  surv={py_sum['survival_mean']:.3f} "
          f"± {py_sum['survival_std']:.3f}   "
          f"cap_alive=${py_sum['mean_capital_mean']/1e6:.2f}M "
          f"± ${py_sum['mean_capital_std']/1e6:.2f}M")
    print(f"  Julia   surv={jl_sum['survival_mean']:.3f} "
          f"± {jl_sum['survival_std']:.3f}   "
          f"cap_alive=${jl_sum['mean_capital_mean']/1e6:.2f}M "
          f"± ${jl_sum['mean_capital_std']/1e6:.2f}M")

    surv_diff = abs(py_sum["survival_mean"] - jl_sum["survival_mean"])
    cap_ratio = py_sum["mean_capital_mean"] / max(jl_sum["mean_capital_mean"], 1.0)
    print(f"\n  |Δ survival|       = {surv_diff:.3f}  (target ≤ 0.10)")
    print(f"  Python/Julia cap   = {cap_ratio:.3f}  (target 0.85 ≤ ratio ≤ 1.15)")

    payload = {
        "config": {"n_agents": N_AGENTS, "n_rounds": N_ROUNDS,
                   "tier": TIER, "seeds": SEEDS},
        "python": {"per_seed": py_results, "summary": py_sum},
        "julia": {"per_seed": jl_results, "summary": jl_sum},
        "diff": {
            "abs_survival": surv_diff,
            "py_over_jl_capital": cap_ratio,
        },
    }
    out_path = REPO / "parity_v23_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults: {out_path}")

    ok = surv_diff <= 0.10 and 0.85 <= cap_ratio <= 1.15
    print("\n" + ("PASS" if ok else "FAIL — investigate divergence"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
