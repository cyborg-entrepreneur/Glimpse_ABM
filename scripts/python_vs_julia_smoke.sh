#!/usr/bin/env bash
# Smoke test: run 200-round simulations in Python and Julia and compare key outcomes.
#
# Usage:
#   bash scripts/python_vs_julia_smoke.sh [N_AGENTS] [N_ROUNDS] [SEED]
# Defaults: N_AGENTS=200, N_ROUNDS=200, SEED=42

set -euo pipefail

N_AGENTS="${1:-200}"
N_ROUNDS="${2:-200}"
SEED="${3:-42}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_TMP="$(mktemp -d "${TMPDIR:-/tmp}/py_smoke.XXXXXX")"
JL_TMP="$(mktemp -d "${TMPDIR:-/tmp}/jl_smoke.XXXXXX")"

echo "Running Python simulation... (agents=${N_AGENTS}, rounds=${N_ROUNDS}, seed=${SEED})"
PY_JSON="$(PYTHONPATH="${REPO_ROOT}" python - <<'PY' "${N_AGENTS}" "${N_ROUNDS}" "${SEED}" "${PY_TMP}"
import json, sys, pathlib
from glimpse_abm.config import EmergentConfig
from glimpse_abm.simulation import EmergentSimulation

n_agents = int(sys.argv[1]); n_rounds = int(sys.argv[2]); seed = int(sys.argv[3])
outdir = pathlib.Path(sys.argv[4])
cfg = EmergentConfig()
cfg.N_AGENTS = n_agents
cfg.N_ROUNDS = n_rounds
cfg.RANDOM_SEED = seed

sim = EmergentSimulation(config=cfg, output_dir=str(outdir), run_id=f"py_smoke_{seed}")
sim.run()

agents = sim.agents
alive = [a for a in agents if getattr(a, "alive", False)]
survival_rate = len(alive) / len(agents) if agents else 0.0
mean_final_capital = sum(a.resources.capital for a in agents) / len(agents) if agents else 0.0
total_innovations = sum(getattr(a, "innovation_count", 0) for a in agents)

summary = {
    "impl": "python",
    "n_agents": len(agents),
    "n_rounds": n_rounds,
    "seed": seed,
    "survival_rate": survival_rate,
    "mean_final_capital": mean_final_capital,
    "total_innovations": total_innovations,
    "output_dir": str(outdir),
}
print(json.dumps(summary))
PY
)"

echo "Running Julia simulation... (agents=${N_AGENTS}, rounds=${N_ROUNDS}, seed=${SEED})"
JL_JSON="$(julia --project="${REPO_ROOT}/julia" -e '
    using JSON3
    include(joinpath("julia", "src", "GlimpseABM.jl"))
    using .GlimpseABM
    n_agents = parse(Int, ARGS[1]); n_rounds = parse(Int, ARGS[2]); seed = parse(Int, ARGS[3])
    outdir = ARGS[4]
    cfg = EmergentConfig()
    cfg.N_AGENTS = n_agents
    cfg.N_ROUNDS = n_rounds
    cfg.RANDOM_SEED = seed
    sim = EmergentSimulation(config=cfg, output_dir=outdir, run_id="jl_smoke_" * string(seed), seed=seed)
    run!(sim)
    stats = summary_stats(sim)
    # Normalize field names to match Python summary
    summary = Dict(
        "impl" => "julia",
        "n_agents" => length(sim.agents),
        "n_rounds" => cfg.N_ROUNDS,
        "seed" => seed,
        "survival_rate" => stats["final_survival_rate"],
        "mean_final_capital" => stats["mean_final_capital"],
        "total_innovations" => stats["total_innovations"],
        "output_dir" => outdir,
    )
    println(JSON3.write(summary))
' "${N_AGENTS}" "${N_ROUNDS}" "${SEED}" "${JL_TMP}")"

echo
echo "=== Python Summary ==="
echo "${PY_JSON}"
echo
echo "=== Julia Summary ==="
echo "${JL_JSON}"
echo

python - <<'PY' "${PY_JSON}" "${JL_JSON}"
import json, sys
py = json.loads(sys.argv[1]); jl = json.loads(sys.argv[2])
metrics = ["survival_rate", "mean_final_capital", "total_innovations"]
print("=== Comparison (Julia - Python) ===")
for m in metrics:
    diff = jl.get(m, 0) - py.get(m, 0)
    print(f"{m:20s} py={py.get(m):.4f}  jl={jl.get(m):.4f}  diff={diff:.4f}")
PY

echo
echo "Outputs stored in:"
echo "  Python: ${PY_TMP}"
echo "  Julia:  ${JL_TMP}"
