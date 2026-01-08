"""Minimal smoke test to keep the ABM regression-safe."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
for candidate in (PARENT, ROOT):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from glimpse_abm.config import EmergentConfig
from glimpse_abm.simulation import EmergentSimulation


def test_smoke_simulation(tmp_path: Path) -> None:
    cfg = EmergentConfig()
    cfg.N_AGENTS = 5
    cfg.N_ROUNDS = 3
    cfg.N_RUNS = 1
    cfg.use_parallel = False
    out_dir = tmp_path / "smoke_run"
    sim = EmergentSimulation(cfg, output_dir=str(out_dir), run_id="smoke")
    sim.run()
    run_dir = out_dir / "smoke"
    assert run_dir.exists(), "Run directory missing"
    assert (run_dir / "run_log.jsonl").exists(), "Round log missing"
    assert (run_dir / "final_agents.pkl").exists(), "Final agent snapshot missing"
