"""Command-line entry points and experiment runners for Glimpse ABM."""

from __future__ import annotations

import argparse
import atexit
import copy
import json
import multiprocessing as mp
import os
import shutil
import signal
import threading
import time
import traceback
import warnings
from datetime import datetime
import itertools
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set
from concurrent.futures import ProcessPoolExecutor, wait
import weakref

# =============================================================================
# PROCESS POOL CLEANUP INFRASTRUCTURE
# =============================================================================
# Track active executors globally to ensure cleanup on exit/suspend/interrupt.
# This prevents zombie worker processes when the parent is killed or suspended.

_active_executors: Set[ProcessPoolExecutor] = set()
_executor_lock = threading.Lock()
_cleanup_registered = False


def _register_executor(executor: ProcessPoolExecutor) -> None:
    """Register an executor for cleanup tracking."""
    with _executor_lock:
        _active_executors.add(executor)


def _unregister_executor(executor: ProcessPoolExecutor) -> None:
    """Unregister an executor after normal shutdown."""
    with _executor_lock:
        _active_executors.discard(executor)


def _cleanup_all_executors(signum: Optional[int] = None, frame: Any = None) -> None:
    """
    Emergency cleanup of all active executors.
    Called on exit, interrupt (Ctrl+C), or termination signals.
    """
    with _executor_lock:
        executors_to_cleanup = list(_active_executors)
        _active_executors.clear()

    for executor in executors_to_cleanup:
        try:
            # Force immediate shutdown without waiting
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass  # Best effort cleanup

    # If called from signal handler, may need to re-raise or exit
    if signum is not None:
        # For SIGTERM/SIGINT, exit cleanly after cleanup
        if signum in (signal.SIGTERM, signal.SIGINT):
            import sys
            sys.exit(128 + signum)


def _setup_executor_cleanup() -> None:
    """
    Register cleanup handlers once at module load.
    Ensures worker processes are terminated even if parent dies unexpectedly.
    """
    global _cleanup_registered
    if _cleanup_registered:
        return
    _cleanup_registered = True

    # Register atexit handler for normal exit
    atexit.register(_cleanup_all_executors)

    # Register signal handlers for interrupts/termination
    # Note: Only works in main thread, so we guard against RuntimeError
    try:
        signal.signal(signal.SIGTERM, _cleanup_all_executors)
        signal.signal(signal.SIGINT, _cleanup_all_executors)
        # SIGHUP: terminal closed
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, _cleanup_all_executors)
    except (ValueError, RuntimeError):
        # Signal handlers can only be set in main thread
        pass


# Initialize cleanup handlers at module import
_setup_executor_cleanup()

# Force Matplotlib into a non-interactive backend for headless execution.
os.environ.setdefault("MPLBACKEND", "Agg")

try:  # pragma: no cover - optional graphical dependency
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - gracefully degrade when unavailable
    plt = None  # type: ignore[assignment]

try:  # pragma: no cover - optional graphical dependency
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover
    sns = None  # type: ignore[assignment]

import numpy as np
import pandas as pd


def _patch_joblib_for_sandbox() -> None:
    """
    Joblib's loky backend queries ``sysctl`` on macOS to determine physical core
    counts. In sandboxed shells this raises ``Operation not permitted`` and
    aborts the interpreter. We intercept those helpers and provide a safe
    fallback so downstream libraries (e.g., scikit-learn) remain usable.
    """
    try:
        from joblib.externals.loky.backend import context as loky_context  # type: ignore
    except Exception:
        return
    fallback = max(1, os.cpu_count() or 1)

    def _safe_core_tuple() -> tuple[int, None]:
        return fallback, None

    def _safe_core_count() -> int:
        return fallback

    if hasattr(loky_context, "_count_physical_cores"):
        loky_context._count_physical_cores = _safe_core_tuple  # type: ignore[attr-defined]
    if hasattr(loky_context, "_count_physical_cores_darwin"):
        loky_context._count_physical_cores_darwin = _safe_core_count  # type: ignore[attr-defined]


_patch_joblib_for_sandbox()

# Progress tracker is optional - moved to .archive for cleanup
def collect_progress(*args, **kwargs):
    """Stub for removed progress_tracker module."""
    return {}

def render_report(*args, **kwargs):
    """Stub for removed progress_tracker module."""
    return "Progress tracking not available (module archived)"

try:
    from tools.progress_tracker import collect_progress, render_report
except (ModuleNotFoundError, ImportError):
    pass  # Use stub functions defined above

from .analysis import (
    ANALYSIS_VERSION,
    ComprehensiveAnalysisFramework,
    ComprehensiveVisualizationSuite,
    StatisticalAnalysisSuite,
)

# PublicationPipeline is optional - moved to .archive for cleanup
PublicationPipeline = None
try:
    from .publication import PublicationPipeline
except (ModuleNotFoundError, ImportError):
    pass  # PublicationPipeline not available
from .config import (
    CalibrationProfile,
    EmergentConfig,
    apply_calibration_profile,
    get_calibration_profile,
    list_calibration_profiles,
    load_calibration_profile,
)
from .simulation import EmergentSimulation
from .utils import normalize_ai_label

try:  # pragma: no cover - optional dependency
    from IPython.display import display  # type: ignore
except ImportError:  # pragma: no cover
    def display(*_args: Any, **_kwargs: Any) -> None:  # type: ignore
        return None


def suppress_runtime_warnings() -> None:
    """Silence noisy runtime warnings that clutter experiment output."""
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")


suppress_runtime_warnings()


PLOTTING_LIBRARIES_AVAILABLE = plt is not None and sns is not None


DEFAULT_SENSITIVITY_GRID: Dict[str, List[Any]] = {
    "BASE_OPERATIONAL_COST": [60_000.0, 70_000.0, 80_000.0],
    "SURVIVAL_CAPITAL_RATIO": [0.45, 0.52, 0.58],
    "RETURN_OVERSUPPLY_PENALTY": [0.4, 0.55, 0.7],
}
DEFAULT_LHS_RANGES: Dict[str, Tuple[float, float]] = {
    # Economics/viability
    "BASE_OPERATIONAL_COST": (60_000.0, 80_000.0),
    "SURVIVAL_CAPITAL_RATIO": (0.45, 0.60),
    # Demand/returns shape
    "RETURN_OVERSUPPLY_PENALTY": (0.45, 0.70),
    "RETURN_UNDERSUPPLY_BONUS": (0.20, 0.45),
    "RETURN_DEMAND_CROWDING_THRESHOLD": (0.35, 0.50),
    "RETURN_DEMAND_CROWDING_PENALTY": (0.35, 0.60),
    "FAILURE_DEMAND_PRESSURE": (0.15, 0.30),
    "MARKET_SHIFT_PROBABILITY": (0.03, 0.10),
    # Recursion/uncertainty levers
    "RECURSION_WEIGHTS.crowd_weight": (0.35, 0.55),
    "RECURSION_WEIGHTS.volatility_weight": (0.15, 0.25),
    "RECURSION_WEIGHTS.ai_herd_weight": (0.35, 0.60),
    "RECURSION_WEIGHTS.premium_reuse_weight": (0.15, 0.35),
    "AI_NOVELTY_UPLIFT": (0.04, 0.06),
    "DOWNSIDE_OVERSUPPLY_WEIGHT": (0.55, 0.90),
    "RETURN_LOWER_BOUND": (-1.2, -1.0),
}


def _print_calibration_catalog() -> None:
    """Display the registered calibration profiles."""
    catalog: List[CalibrationProfile] = sorted(
        list_calibration_profiles(), key=lambda profile: profile.name.lower()
    )
    if not catalog:
        print("No built-in calibration profiles are registered.")
        return
    print("Available calibration profiles:")
    for profile in catalog:
        print(f"  - {profile.name}: {profile.description}")


def _write_config_dump(config: EmergentConfig, destination: str) -> Path:
    """Persist the resolved configuration to ``destination``."""
    target = Path(destination).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(config.snapshot(), handle, indent=2, sort_keys=True)
    print(f"[CLI] Wrote configuration snapshot to {target}")
    return target


def _persist_config_snapshot(
    results_directory: str,
    config: EmergentConfig,
    cli_args: Optional[Dict[str, Any]],
    calibration_metadata: Optional[List[Dict[str, Any]]],
) -> Path:
    """Store configuration + calibration metadata alongside simulation results."""
    payload = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "cli_args": cli_args or {},
        "calibration_profiles": calibration_metadata or [],
        "config": config.snapshot(),
    }
    snapshot_path = Path(results_directory) / "config_snapshot.json"
    with snapshot_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return snapshot_path


def _persist_analysis_metadata(
    results_directory: str,
    table_paths: Optional[Dict[str, str]],
) -> Path:
    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "tables": table_paths or {},
        "generated_at": datetime.utcnow().isoformat(),
    }
    path = Path(results_directory) / "analysis_metadata.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def _plot_sensitivity_effects(effects_csv: Path, output_dir: Path) -> Optional[Path]:
    if plt is None:
        print("[CLI] Matplotlib is unavailable; skipping sensitivity effects plot.")
        return None
    try:
        df = pd.read_csv(effects_csv)
    except FileNotFoundError:
        return None
    if df.empty or 'metric' not in df.columns:
        return None
    metrics = sorted(df['metric'].dropna().unique())
    if not metrics:
        return None
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        subset = df[df['metric'] == metric]
        subset = subset.sort_values('normalized_effect', ascending=False)
        ax.bar(subset['parameter'], subset['normalized_effect'], color="#3498db")
        ax.set_title(f"{metric} sensitivity")
        ax.set_ylabel("Normalized effect")
        ax.set_ylim(0, 1)
    axes[-1].set_xlabel("Parameter")
    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "sensitivity_effects.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _coerce_value(value: str) -> Any:
    for cast in (int, float):
        try:
            return cast(value)
        except (ValueError, TypeError):
            continue
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return value


def _parse_sensitivity_params(param_args: Optional[List[str]]) -> Dict[str, List[Any]]:
    if not param_args:
        return {}
    grid: Dict[str, List[Any]] = {}
    for arg in param_args:
        if "=" not in arg:
            continue
        name, raw_values = arg.split("=", 1)
        name = name.strip()
        values = [val.strip() for val in raw_values.split(",") if val.strip()]
        if not values:
            continue
        grid[name] = [_coerce_value(v) for v in values]
    return grid


def _parse_lhs_ranges(range_args: Optional[List[str]]) -> Dict[str, Tuple[float, float]]:
    if not range_args:
        return {}
    ranges: Dict[str, Tuple[float, float]] = {}
    for arg in range_args:
        if "=" not in arg or ":" not in arg:
            continue
        name, raw_range = arg.split("=", 1)
        low_str, high_str = raw_range.split(":", 1)
        try:
            low = float(low_str.strip())
            high = float(high_str.strip())
        except ValueError:
            continue
        if high < low:
            low, high = high, low
        ranges[name.strip()] = (low, high)
    return ranges


def _latin_hypercube_samples(
    bounds: Dict[str, Tuple[float, float]],
    n_samples: int,
    seed: Optional[int] = None,
) -> List[Dict[str, float]]:
    if n_samples <= 0 or not bounds:
        return []
    rng = np.random.default_rng(seed)
    param_names = list(bounds.keys())
    intervals = np.linspace(0.0, 1.0, n_samples + 1)
    points = rng.random((len(param_names), n_samples))
    spans = intervals[1:] - intervals[:-1]
    lhs = intervals[:-1] + points * spans
    for dim in range(len(param_names)):
        rng.shuffle(lhs[dim])
    samples: List[Dict[str, float]] = []
    for i in range(n_samples):
        sample: Dict[str, float] = {}
        for dim, name in enumerate(param_names):
            low, high = bounds[name]
            sample[name] = float(low + lhs[dim, i] * (high - low))
        samples.append(sample)
    return samples


def _read_last_round(log_path: Path) -> Optional[int]:
    if not log_path.exists():
        return None
    last_line: Optional[str] = None
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                last_line = stripped
    if not last_line:
        return None
    try:
        record = json.loads(last_line)
    except json.JSONDecodeError:
        return None
    round_value = record.get("round")
    if round_value is None:
        return None
    try:
        return int(round_value)
    except (TypeError, ValueError):
        return None


def _run_is_complete(results_dir: Path, run_id: str, expected_rounds: int) -> bool:
    run_dir = Path(results_dir) / run_id
    log_path = run_dir / "run_log.jsonl"
    last_round = _read_last_round(log_path)
    if last_round is None:
        def _latest_pickle_round(subdir: str) -> Optional[int]:
            """Inspect newest pickle in subdir for 'round' column."""
            dir_path = run_dir / subdir
            if not dir_path.exists():
                return None
            files = sorted(dir_path.glob("*.pkl"))
            for path in reversed(files):
                try:
                    df = pd.read_pickle(path)
                except (OSError, ValueError, EOFError, pd.errors.EmptyDataError):
                    continue
                if "round" not in df.columns:
                    continue
                rounds = pd.to_numeric(df["round"], errors="coerce").dropna()
                if not rounds.empty:
                    return int(rounds.max())
            return None

        candidates = [
            _latest_pickle_round("summary"),
            _latest_pickle_round("decisions"),
            _latest_pickle_round("market"),
        ]
        last_round = max((c for c in candidates if c is not None), default=None)

    if last_round is None:
        return False
    return last_round >= max(0, expected_rounds - 1)


def _reset_run_directory(results_dir: Path, run_id: str) -> None:
    run_dir = Path(results_dir) / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)


def _build_rescue_config(base_config: EmergentConfig) -> EmergentConfig:
    rescue = copy.deepcopy(base_config)
    rescue.use_parallel = False
    rescue.max_workers = 1
    rescue.parallel_threshold = max(1, getattr(base_config, "parallel_threshold", 1))
    return rescue


def _configure_worker_process() -> None:
    """
    Constrain BLAS thread fan-out inside each worker to avoid overwhelming the
    host when many simulation processes run concurrently.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def _compute_parallel_plan(config: EmergentConfig) -> tuple[int, float, float]:
    """
    Derive process count and timeout based on PARALLEL_MODE and workload size.

    PARALLEL_MODE options:
        "max": Use CPUs-1 without workload-based caps (maximum throughput)
        "safe": Apply conservative workload-based caps (prevents memory pressure)

    Returns:
        max_jobs: upper bound on concurrent processes
        timeout: seconds to wait before treating a worker as stalled
        workload: dimensionless load proxy for logging/diagnostics
    """
    cpu_count = os.cpu_count() or 1
    max_jobs = max(1, cpu_count - 1)

    # Respect explicit cap if set
    run_job_cap = int(getattr(config, "MAX_PARALLEL_RUNS", 0) or 0)
    if run_job_cap > 0:
        max_jobs = min(max_jobs, max(1, run_job_cap))

    workload = (float(config.N_AGENTS) / 500.0) * (float(config.N_ROUNDS) / 100.0)

    # Apply workload-based caps only in "safe" mode
    parallel_mode = getattr(config, "PARALLEL_MODE", "max").lower()
    if parallel_mode == "safe":
        if workload >= 5.0:
            max_jobs = min(max_jobs, 3)
        elif workload >= 3.0:
            max_jobs = min(max_jobs, 4)
        elif workload >= 2.0:
            max_jobs = min(max_jobs, 5)

    timeout = max(900.0, workload * 480.0)
    return max_jobs, timeout, workload


def _tune_buffering_for_parallel(config: EmergentConfig, workload: float, workers: int) -> None:
    """
    Loosen disk write cadence when running large jobs in parallel to reduce I/O contention.
    """
    if workers > 1 and workload >= 3.0:
        if getattr(config, "buffer_flush_interval", 5) < 10:
            config.buffer_flush_interval = 10
        config.write_intermediate_batches = True


def _execute_parallel_tasks(
    tasks: Iterable[Any],
    worker: Callable[[Any], Any],
    n_jobs: int,
    desc: str = "tasks",
    timeout: Optional[float] = None,
):
    """Execute ``worker`` across ``tasks`` using a guarded process pool.

    Workers are registered with the global cleanup system to ensure they
    are terminated even if the parent process is killed or suspended.
    """
    tasks = list(tasks)
    if not tasks:
        return []
    n_jobs = max(1, int(n_jobs))
    if n_jobs == 1 or len(tasks) == 1:
        return [worker(task) for task in tasks]
    print(f"[Parallel] Executing {desc} across {n_jobs} processes...")
    executor: Optional[ProcessPoolExecutor] = None
    try:
        executor = ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=mp.get_context("spawn"),
            initializer=_configure_worker_process,
        )
        # Register executor for emergency cleanup on exit/interrupt
        _register_executor(executor)
        results: List[Any] = [None] * len(tasks)
        future_map: Dict[Any, int] = {}
        next_idx = 0
        # Prime submissions
        while next_idx < len(tasks) and len(future_map) < n_jobs:
            future_map[executor.submit(worker, tasks[next_idx])] = next_idx
            next_idx += 1

        import concurrent.futures
        import time as _time

        deadline = None
        if timeout is not None and timeout > 0:
            deadline = _time.time() + timeout

        def _record_result(idx: int, fut: Any) -> None:
            try:
                res = fut.result()
            except Exception as exc:
                res = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            results[idx] = res

        while future_map:
            pending_futs = set(future_map.keys())
            wait_timeout = None
            if deadline is not None:
                wait_timeout = max(0.0, deadline - _time.time())
                if wait_timeout == 0 and pending_futs:
                    done = set()
                else:
                    done, _ = concurrent.futures.wait(
                        pending_futs,
                        timeout=wait_timeout,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
            else:
                done, _ = concurrent.futures.wait(
                    pending_futs,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

            if not done:
                stalled_indices = list(future_map.values()) + list(range(next_idx, len(tasks)))
                for fut in pending_futs:
                    fut.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                _unregister_executor(executor)
                executor = None
                print(
                    f"[Parallel] Timeout after {timeout:.0f}s during {desc}; "
                    f"rerunning {len(stalled_indices)} remaining tasks sequentially."
                )
                for idx in stalled_indices:
                    try:
                        results[idx] = worker(tasks[idx])
                    except Exception as exc:
                        results[idx] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                break

            for fut in done:
                idx = future_map.pop(fut)
                _record_result(idx, fut)
                # Refill the pool
                if next_idx < len(tasks):
                    future_map[executor.submit(worker, tasks[next_idx])] = next_idx
                    next_idx += 1

        # Catch any remaining submissions if executor survived
        if executor is not None:
            executor.shutdown(wait=True)
            _unregister_executor(executor)
        return results
    except (PermissionError, OSError) as exc:
        print(f"[Parallel] Falling back to sequential execution ({exc}).")
        return [worker(task) for task in tasks]
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
            _unregister_executor(executor)


class ProgressMonitor:
    """Background monitor that surfaces run progress at a fixed cadence."""

    def __init__(self, results_dir: Path, interval: float = 30.0, limit: Optional[int] = None):
        self.results_dir = Path(results_dir)
        self.interval = max(1.0, float(interval))
        self.limit = limit
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="glimpse-progress-monitor",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=self.interval + 1.0)

    def _run(self) -> None:
        last_report: Optional[str] = None
        while not self._stop_event.is_set():
            if not self.results_dir.exists():
                report = f"[monitor] Waiting for results directory: {self.results_dir}"
            else:
                try:
                    progress = collect_progress(self.results_dir, limit=self.limit)
                    core = render_report(progress)
                except Exception as exc:  # pragma: no cover - defensive logging
                    core = f"Unable to read progress: {exc}"
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                report = f"[monitor] [{timestamp}]\n{core}"

            if report != last_report:
                print(report, flush=True)
                last_report = report

            if self._stop_event.wait(self.interval):
                break


def _apply_smoke_overrides(config: EmergentConfig) -> EmergentConfig:
    """Clamp configuration values for quick smoke executions."""
    config.N_RUNS = min(int(config.N_RUNS), 2)
    config.N_AGENTS = min(config.N_AGENTS, 50)
    config.N_ROUNDS = min(config.N_ROUNDS, 110)
    config.use_parallel = False
    return config


def _apply_experiment_profile(config: EmergentConfig, profile: str) -> EmergentConfig:
    """Apply predefined experimental overrides (e.g., AGI-tier pricing)."""
    if profile == "agi2027":
        cfg = copy.deepcopy(config)
        cfg.AI_LEVELS = {
            "none": {
                "cost": 0.0,
                "cost_type": "none",
                "info_quality": 0.0,
                "info_breadth": 0.0,
                "per_use_cost": 0.0,
            },
            "basic": {
                "cost": 50.0,
                "cost_type": "subscription",
                "info_quality": 0.55,
                "info_breadth": 0.45,
                "per_use_cost": 4.0,
            },
            "advanced": {
                "cost": 1250.0,
                "cost_type": "subscription",
                "info_quality": 0.82,
                "info_breadth": 0.70,
                "per_use_cost": 5.0,
            },
            "premium": {
                "cost": 18000.0,
                "cost_type": "subscription",
                "info_quality": 0.96,
                "info_breadth": 0.92,
                "per_use_cost": 25.0,
            },
        }
        cfg.AI_DOMAIN_CAPABILITIES = {
            "basic": {
                "market_analysis": {"accuracy": 0.64, "hallucination_rate": 0.20, "bias": 0.03},
                "technical_assessment": {"accuracy": 0.65, "hallucination_rate": 0.19, "bias": -0.02},
                "uncertainty_evaluation": {"accuracy": 0.62, "hallucination_rate": 0.18, "bias": -0.03},
                "innovation_potential": {"accuracy": 0.60, "hallucination_rate": 0.22, "bias": 0.04},
            },
            "advanced": {
                "market_analysis": {"accuracy": 0.83, "hallucination_rate": 0.06, "bias": 0.015},
                "technical_assessment": {"accuracy": 0.85, "hallucination_rate": 0.05, "bias": -0.01},
                "uncertainty_evaluation": {"accuracy": 0.82, "hallucination_rate": 0.07, "bias": -0.015},
                "innovation_potential": {"accuracy": 0.80, "hallucination_rate": 0.08, "bias": 0.012},
            },
            "premium": {
                "market_analysis": {"accuracy": 0.95, "hallucination_rate": 0.012, "bias": 0.005},
                "technical_assessment": {"accuracy": 0.96, "hallucination_rate": 0.010, "bias": -0.003},
                "uncertainty_evaluation": {"accuracy": 0.94, "hallucination_rate": 0.014, "bias": -0.004},
                "innovation_potential": {"accuracy": 0.93, "hallucination_rate": 0.018, "bias": 0.006},
            },
        }
        return cfg
    return config


def run_emergent_simulation(args: tuple[int, EmergentConfig, str]) -> Dict[str, Any]:
    """
    Execute a single emergent simulation where agents dynamically choose their AI level.
    The tuple signature keeps compatibility with the generic parallel executor.
    """
    run_index, base_config, base_output_dir = args
    run_id = f"emergent_run_{run_index}"

    run_config = copy.deepcopy(base_config)
    run_config.AGENT_AI_MODE = "emergent"

    try:
        sim = EmergentSimulation(config=run_config, output_dir=base_output_dir, run_id=run_id)
        sim.run()
        return {"run_id": run_id, "status": "completed"}
    except Exception as exc:  # pragma: no cover - surfaced to console for manual follow-up
        print(f"‼️ ERROR in emergent run [{run_id}] ‼️")
        traceback.print_exc()
        return {"run_id": run_id, "status": f"error: {exc}"}


def run_fixed_level_simulation(args: tuple[int, str, EmergentConfig, str]) -> Dict[str, Any]:
    """Execute a single simulation with all agents locked to ``ai_level``."""
    run_index, ai_level, base_config, base_output_dir = args
    run_id = f"fixed_{ai_level}_run_{run_index}"

    run_config = copy.deepcopy(base_config)
    run_config.AGENT_AI_MODE = "fixed"

    try:
        sim = EmergentSimulation(config=run_config, output_dir=base_output_dir, run_id=run_id)
        for agent in sim.agents:
            agent.fixed_ai_level = ai_level
        sim.run()
        return {"run_id": run_id, "status": "completed"}
    except Exception as exc:  # pragma: no cover - surfaced to console for manual follow-up
        print(f"‼️ ERROR in fixed run [{run_id}] for level {ai_level} ‼️")
        traceback.print_exc()
        return {"run_id": run_id, "status": f"error: {exc}"}


def run_master_launcher(
    base_config: EmergentConfig,
    run_emergent: bool = True,
    run_fixed_levels: bool = False,
    results_dir: Optional[str] = None,
    ai_levels_to_test: Optional[List[str]] = None,
    monitor_progress: bool = False,
    monitor_interval: float = 30.0,
    monitor_limit: Optional[int] = None,
    calibration_metadata: Optional[List[Dict[str, Any]]] = None,
    cli_args: Optional[Dict[str, Any]] = None,
    skip_visualizations: bool = False,
) -> Dict[str, Any]:
    """
    Execute the primary experiment pipeline and return a summary dictionary containing:

    - ``results_directory`` path with raw simulation artefacts
    - ``run_summaries`` with per-job completion metadata
    - ``analysis_results`` from :class:`ComprehensiveAnalysisFramework`
    - ``statistical_summary`` from :class:`StatisticalAnalysisSuite`
    - ``config_snapshot`` path to the saved configuration metadata
    """
    ai_levels = ai_levels_to_test or ["none", "basic", "advanced", "premium"]
    n_runs = int(base_config.N_RUNS)
    max_jobs, task_timeout, workload = _compute_parallel_plan(base_config)
    # Master runs are long; avoid cancelling healthy workers by disabling the timeout.
    task_timeout = None
    _tune_buffering_for_parallel(base_config, workload, max_jobs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_prefix = "./simulation_results_"
    results_directory = results_dir or f"{root_prefix}{timestamp}"
    results_path = Path(results_directory)

    if os.path.exists(results_directory):
        print(f"🧹 Deleting old directory: {results_directory}")
        shutil.rmtree(results_directory)
    os.makedirs(results_directory, exist_ok=True)

    print("╔═══════════════════════════════════════════════════════════════════════╗")
    print("║  🚀 KNIGHTIAN UNCERTAINTY & AI AUGMENTATION SIMULATION (UPGRADED)     ║")
    print("╚═══════════════════════════════════════════════════════════════════════╝")
    print(f"   - Results will be saved to: {results_directory}")
    parallel_mode = getattr(base_config, "PARALLEL_MODE", "max")
    mode_label = f"mode={parallel_mode}"
    print(f"   - Parallel plan: {max_jobs} workers ({mode_label}, load≈{workload:.1f})")
    if getattr(base_config, "buffer_flush_interval", 0) >= 10:
        print(f"   - Buffer flush interval raised to {base_config.buffer_flush_interval} rounds to ease disk I/O")
    snapshot_path = _persist_config_snapshot(
        results_directory,
        base_config,
        cli_args=cli_args,
        calibration_metadata=calibration_metadata,
    )
    print(f"   - Configuration snapshot stored in: {snapshot_path}")

    run_summaries: Dict[str, Any] = {}
    monitor: Optional[ProgressMonitor] = None
    analysis_results: Dict[str, Any] = {}
    statistical_summary: Dict[str, Any] = {}

    try:
        if monitor_progress:
            monitor = ProgressMonitor(Path(results_directory), interval=monitor_interval, limit=monitor_limit)
            monitor.start()

        if run_emergent and n_runs > 0:
            print("\n🔬 Starting EMERGENT Experiment...")
            emergent_jobs = [(i, base_config, results_directory) for i in range(n_runs)]
            run_summaries["emergent"] = []
            for start in range(0, len(emergent_jobs), max_jobs):
                chunk = emergent_jobs[start : start + max_jobs]
                if not chunk:
                    continue
                n_chunk_jobs = min(max_jobs, len(chunk))
                chunk_results = _execute_parallel_tasks(
                    chunk,
                    run_emergent_simulation,
                    n_chunk_jobs or 1,
                    desc="emergent run chunk",
                    timeout=task_timeout,
                )
                for job, result in zip(chunk, chunk_results):
                    run_index = job[0]
                    run_id = f"emergent_run_{run_index}"
                    status = result.get("status")
                    completed = (
                        status == "completed"
                        and _run_is_complete(results_path, run_id, int(base_config.N_ROUNDS))
                    )
                    final_result = result
                    if not completed:
                        print(
                            f"⚠️  {run_id} did not finish cleanly; retrying in rescue mode...",
                            flush=True,
                        )
                        _reset_run_directory(results_path, run_id)
                        rescue_config = _build_rescue_config(base_config)
                        rescue_job = (run_index, rescue_config, results_directory)
                        final_result = run_emergent_simulation(rescue_job)
                        rescue_complete = (
                            final_result.get("status") == "completed"
                            and _run_is_complete(results_path, run_id, int(base_config.N_ROUNDS))
                        )
                        if not rescue_complete:
                            print(
                                f"❌ Rescue attempt for {run_id} failed; marking run as failed and continuing.",
                                flush=True,
                            )
                            final_result["status"] = "failed"
                run_summaries["emergent"].append(final_result)

            # Post-pass: ensure all expected runs exist and are complete; rerun sequentially if needed.
            expected_runs = {f"emergent_run_{i}" for i in range(n_runs)}
            existing_runs = {res.get("run_id"): res for res in run_summaries["emergent"]}
            incomplete: List[str] = []
            for run_id in expected_runs:
                run_dir = results_path / run_id
                is_complete = run_dir.exists() and _run_is_complete(results_path, run_id, int(base_config.N_ROUNDS))
                if not is_complete:
                    incomplete.append(run_id)
            if incomplete:
                print(f"[EMERGENT] Rerunning {len(incomplete)} incomplete/missing runs sequentially...")
            for run_id in incomplete:
                run_index = int(run_id.rsplit("_", 1)[-1])
                # Try a fresh run, then rescue config.
                rerun_result: Dict[str, Any] = {"run_id": run_id, "status": "failed"}
                for attempt, cfg in enumerate((base_config, _build_rescue_config(base_config)), start=1):
                    _reset_run_directory(results_path, run_id)
                    res = run_emergent_simulation((run_index, cfg, results_directory))
                    if (
                        res.get("status") == "completed"
                        and _run_is_complete(results_path, run_id, int(base_config.N_ROUNDS))
                    ):
                        rerun_result = res
                        break
                    else:
                        rerun_result = res
                        rerun_result["status"] = res.get("status", "failed")
                        print(f"[EMERGENT] Rerun attempt {attempt} for {run_id} did not complete.")
                existing_runs[run_id] = rerun_result
            if incomplete:
                # Refresh the summary list with updated statuses.
                run_summaries["emergent"] = list(existing_runs.values())

        if run_fixed_levels and n_runs > 0:
            print("\n🔬 Starting FIXED-LEVEL Experiment...")
            fixed_level_jobs = [
                (i, level, base_config, results_directory)
                for level in ai_levels
                for i in range(n_runs)
            ]
            n_jobs = min(max_jobs, len(fixed_level_jobs))
            run_summaries["fixed_levels"] = _execute_parallel_tasks(
                fixed_level_jobs,
                run_fixed_level_simulation,
                n_jobs or 1,
                desc="fixed-level runs",
                timeout=task_timeout,
            )

        framework = ComprehensiveAnalysisFramework(results_directory, base_config)
        analysis_results = framework.run_full_analysis()
        analysis_metadata_path: Optional[Path] = None

        if analysis_results:
            stat_suite = StatisticalAnalysisSuite(framework)
            statistical_summary = stat_suite.run_comprehensive_analysis()
            analysis_results["statistical_summary"] = statistical_summary

            if skip_visualizations:
                print("[CLI] Visualization suite skipped by user request.")
            elif PLOTTING_LIBRARIES_AVAILABLE:
                viz_suite = ComprehensiveVisualizationSuite(framework)
                viz_suite.create_all_visualizations()
            else:
                print("[CLI] Plotting libraries unavailable; skipping visualization suite.")

            tables_dir = os.path.join(results_directory, "tables")
            table_paths = framework.export_research_tables(tables_dir)
            if table_paths:
                print("\n📊 Research tables exported:")
                for name, path in table_paths.items():
                    print(f"   - {name}: {path}")
            analysis_metadata_path = _persist_analysis_metadata(results_directory, table_paths)
    finally:
        if monitor is not None:
            monitor.stop()

    print("\n" + "─" * 73)
    if analysis_results:
        print(
            "✅ Experiment pipeline complete — simulations finished, "
            "framework assembled, statistical suite executed, and visualization dashboards generated."
        )
    else:
        print("⚠️ Experiment pipeline finished with no analysable data.")
    print(f"   Full results located in: {results_directory}")
    print("─" * 73 + "\n")

    return {
        "results_directory": results_directory,
        "run_summaries": run_summaries,
        "analysis_results": analysis_results,
        "statistical_summary": statistical_summary,
        "config_snapshot": str(snapshot_path),
        "analysis_metadata": str(analysis_metadata_path) if analysis_metadata_path else None,
    }


def run_fixed_level_uncertainty_experiment(
    args: tuple[str, EmergentConfig, int, str]
) -> Dict[str, Any]:
    """
    Worker function for fixed-level experiments. The tuple signature keeps
    joblib invocation simple.
    """
    ai_level, base_config, run_num, output_dir = args
    run_id = f"Fixed_AI_Level_{ai_level}_run_{run_num}"

    run_config = copy.deepcopy(base_config)
    run_config.AGENT_AI_MODE = "fixed"

    try:
        sim = EmergentSimulation(config=run_config, output_dir=output_dir, run_id=run_id)
        for agent in sim.agents:
            agent.fixed_ai_level = ai_level
        sim.run()
        return {"run_id": run_id, "status": "completed", "error": None}
    except Exception as exc:  # pragma: no cover - surfaced to console for manual follow-up
        print(f"Experiment for AI level {ai_level}, run {run_num} failed: {exc}")
        return {"run_id": run_id, "status": "failed", "error": str(exc)}


def run_fixed_level_uncertainty_batch(
    base_config: EmergentConfig,
    ai_levels: Optional[List[str]] = None,
    runs_per_level: Optional[int] = None,
    output_dir: Optional[str] = None,
    calibration_metadata: Optional[List[Dict[str, Any]]] = None,
    cli_args: Optional[Dict[str, Any]] = None,
    skip_visualizations: bool = False,
) -> Dict[str, Any]:
    """Execute the fixed-level uncertainty experiment sweep and return artefacts."""
    ai_levels = ai_levels or ["none", "basic", "advanced", "premium"]
    runs_per_level = runs_per_level if runs_per_level is not None else base_config.N_RUNS

    parallel_workers, task_timeout, workload = _compute_parallel_plan(base_config)
    task_timeout = None  # Long runs; avoid cancelling healthy workers.
    _tune_buffering_for_parallel(base_config, workload, parallel_workers)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_output_dir = output_dir or f"./fixed_level_experiment_results_{timestamp}"
    os.makedirs(exp_output_dir, exist_ok=True)
    snapshot_path = _persist_config_snapshot(
        exp_output_dir,
        base_config,
        cli_args=cli_args,
        calibration_metadata=calibration_metadata,
    )

    print("🔬 Starting Fixed-Level Knightian Uncertainty Experiment...")
    print(f"   AI Levels: {ai_levels}")
    print(f"   Runs per level: {runs_per_level}")
    print(f"   Configuration snapshot stored in: {snapshot_path}")

    experiment_config = copy.deepcopy(base_config)
    start_time = time.time()

    experiment_jobs = [
        (level, experiment_config, i, exp_output_dir)
        for level in ai_levels
        for i in range(runs_per_level)
    ]

    if experiment_jobs:
        num_workers = min(parallel_workers, len(experiment_jobs))
        parallel_mode = getattr(base_config, "PARALLEL_MODE", "max")
        print(f"   - Parallel plan: {num_workers} workers (mode={parallel_mode}, load≈{workload:.1f})")
        all_results = _execute_parallel_tasks(
            experiment_jobs,
            run_fixed_level_uncertainty_experiment,
            num_workers,
            desc="fixed-level uncertainty runs",
            timeout=task_timeout,
        )
    else:
        num_workers = 0
        all_results: List[Dict[str, Any]] = []

    elapsed = time.time() - start_time
    print(f"\n✅ Fixed-Level Experiments Complete! Total time: {elapsed:.2f} seconds")

    print("\n📊 Analyzing results from fixed-level experiments...")
    framework = ComprehensiveAnalysisFramework(exp_output_dir, experiment_config)
    analysis_results = framework.run_full_analysis()

    # Run complete causal analysis with diagnostic plots
    print("\n📊 Running causal inference analysis...")
    try:
        from .statistical_tests import run_complete_causal_analysis
        causal_results = run_complete_causal_analysis(
            results_dir=exp_output_dir,
            is_fixed_tier=(experiment_config.AGENT_AI_MODE == "fixed")
        )
        analysis_results['causal_analysis'] = causal_results
        print(f"   ✓ Causal analysis complete: {len(causal_results)} tables generated")
    except ImportError as e:
        print(f"   ⚠️ Could not import causal analysis: {e}")
    except Exception as e:
        print(f"   ⚠️ Causal analysis failed: {e}")

    analysis_metadata_path: Optional[Path] = None
    if not framework.agent_df.empty:
        if skip_visualizations:
            print("   [CLI] Visualization suite skipped by user request.")
        elif PLOTTING_LIBRARIES_AVAILABLE:
            viz_suite = ComprehensiveVisualizationSuite(framework)
            viz_suite.create_performance_dashboard()
            viz_suite.create_uncertainty_and_market_dashboard()

        tables_dir = os.path.join(exp_output_dir, "tables")
        table_paths = framework.export_research_tables(tables_dir)
        if table_paths:
            print("\n📊 Research tables exported:")
            for name, path in table_paths.items():
                print(f"   - {name}: {path}")
        analysis_metadata_path = _persist_analysis_metadata(exp_output_dir, table_paths)
    else:
        print("\nNo successful experiment runs to analyze.")

    return {
        "results": all_results,
        "results_dir": exp_output_dir,
        "analysis_results": analysis_results,
        "config_snapshot": str(snapshot_path),
        "analysis_metadata": str(analysis_metadata_path) if analysis_metadata_path else None,
    }


def run_emergent_scenario(args: tuple[str, EmergentConfig, int, str]) -> Dict[str, Any]:
    """Run a single EMERGENT simulation for a configured scenario."""
    scenario_name, base_config, run_num, output_dir = args
    run_id = f"Emergent_{scenario_name}_run_{run_num}"

    run_config = copy.deepcopy(base_config)
    run_config.AGENT_AI_MODE = "emergent"
    try:
        sim = EmergentSimulation(config=run_config, output_dir=output_dir, run_id=run_id)
        sim.run()
        return {"run_id": run_id, "status": "completed", "error": None}
    except Exception as exc:  # pragma: no cover - surfaced to console for manual follow-up
        print(f"Scenario {scenario_name} run {run_num} failed: {exc}")
        return {"run_id": run_id, "status": "failed", "error": str(exc)}


def run_uncertainty_scenario_sweep(
    base_config: EmergentConfig,
    scenario_runs: Optional[int] = None,
    scenario_order: Optional[List[str]] = None,
    root_dir: Optional[str] = None,
    calibration_metadata: Optional[List[Dict[str, Any]]] = None,
    cli_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute the uncertainty scenario sweep and produce dashboards."""
    scenario_runs = scenario_runs if scenario_runs is not None else base_config.N_RUNS
    scenario_order = scenario_order or [
        "Baseline",
        "High_Ignorance",
        "High_Indeterminism",
        "High_AgenticNovelty",
        "High_CompetitiveRecursion",
    ]

    parallel_workers, task_timeout, workload = _compute_parallel_plan(base_config)
    task_timeout = None  # Avoid cancelling long scenario runs.
    _tune_buffering_for_parallel(base_config, workload, parallel_workers)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = root_dir or f"./uncertainty_scenario_results_{timestamp}"
    os.makedirs(root_dir, exist_ok=True)
    snapshot_path = _persist_config_snapshot(
        root_dir,
        base_config,
        cli_args=cli_args,
        calibration_metadata=calibration_metadata,
    )
    print(f"[Scenarios] Configuration snapshot stored in: {snapshot_path}")
    figures_dir = _ensure_figures_dir(root_dir)

    scenarios = _build_scenarios(base_config)

    jobs = []
    for scenario_name, scenario_cfg in scenarios.items():
        scenario_dir = os.path.join(root_dir, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)
        for run_idx in range(scenario_runs):
            jobs.append((scenario_name, scenario_cfg, run_idx, scenario_dir))

    if jobs:
        n_jobs = max(1, min(parallel_workers, len(jobs)))
        start_time = time.time()
        print(f"Running {len(jobs)} scenario simulations across {n_jobs} workers...")
        _execute_parallel_tasks(
            jobs,
            run_emergent_scenario,
            n_jobs,
            desc="uncertainty scenario runs",
            timeout=task_timeout,
        )
        elapsed = time.time() - start_time
        print(f"Scenario sweep complete in {elapsed:.2f}s")
    else:
        print("No scenario jobs generated; skipping simulation phase.")

    framework_summaries: Dict[str, Dict[str, Any]] = {}
    for scenario_name in scenario_order:
        scenario_dir = os.path.join(root_dir, scenario_name)
        framework, summary = _summarize_scenario_results(scenario_dir, base_config)
        if framework:
            framework_summaries[scenario_name] = {"framework": framework, "summary": summary}

    scenario_metadata: Dict[str, str] = {}
    if framework_summaries:
        if PLOTTING_LIBRARIES_AVAILABLE:
            _plot_adoption_by_uncertainty(framework_summaries, scenario_order, figures_dir)
            _plot_ai_adoption_over_time(framework_summaries, scenario_order, figures_dir)
            _plot_uncertainty_profiles(framework_summaries, scenario_order, figures_dir)
            _plot_performance_vs_uncertainty(framework_summaries, scenario_order, figures_dir)
        else:
            print("[CLI] Plotting libraries unavailable; skipping scenario visualizations.")

        tables_dir = os.path.join(root_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)
        for scenario, info in framework_summaries.items():
            franchise_dir = os.path.join(tables_dir, scenario)
            os.makedirs(franchise_dir, exist_ok=True)
            framework = info.get('framework')
            if framework:
                table_paths = framework.export_research_tables(franchise_dir)
                if table_paths:
                    print(f"\n📊 Research tables exported for {scenario}:")
                    for name, path in table_paths.items():
                        print(f"   - {name}: {path}")
                    meta_path = _persist_analysis_metadata(franchise_dir, table_paths)
                    scenario_metadata[scenario] = str(meta_path)
    else:
        print("No framework summaries available for visualization.")

    print("Uncertainty scenario analysis complete. Results stored in", root_dir)
    return {
        "root_dir": root_dir,
        "figures_dir": figures_dir,
        "framework_summaries": framework_summaries,
        "config_snapshot": str(snapshot_path),
        "analysis_metadata": scenario_metadata,
    }


def run_sensitivity_sweep(
    base_config: EmergentConfig,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    rounds: int = 80,
    agents: int = 250,
    runs_per_combo: int = 1,
    output_dir: Optional[str] = None,
    calibration_metadata: Optional[List[Dict[str, Any]]] = None,
    cli_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    grid = param_grid or DEFAULT_SENSITIVITY_GRID
    if not grid:
        grid = DEFAULT_SENSITIVITY_GRID
    param_names = list(grid.keys())
    combos = list(itertools.product(*(grid[name] for name in param_names)))
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    root_dir = output_dir or f"./sensitivity_results_{timestamp}"
    os.makedirs(root_dir, exist_ok=True)
    snapshot_path = _persist_config_snapshot(root_dir, base_config, cli_args=cli_args, calibration_metadata=calibration_metadata)
    print(f"[Sensitivity] Configuration snapshot stored in: {snapshot_path}")
    summary_records: List[Dict[str, Any]] = []
    for idx, values in enumerate(combos):
        overrides = dict(zip(param_names, values))
        combo_cfg = base_config.copy_with_overrides(overrides)
        combo_cfg.N_RUNS = runs_per_combo
        combo_cfg.N_ROUNDS = min(rounds, combo_cfg.N_ROUNDS)
        combo_cfg.N_AGENTS = min(agents, combo_cfg.N_AGENTS)
        combo_cfg.use_parallel = False
        combo_dir = os.path.join(root_dir, f"combo_{idx}")
        os.makedirs(combo_dir, exist_ok=True)
        sim = EmergentSimulation(config=combo_cfg, output_dir=combo_dir, run_id=f"sensitivity_run_{idx}")
        print(f"[Sensitivity] Running combo {idx+1}/{len(combos)} with overrides {overrides}")
        sim.run()
        framework = ComprehensiveAnalysisFramework(combo_dir, combo_cfg)
        framework.run_full_analysis()
        agent_df = framework.agent_df
        summary_df = framework.summary_df
        metrics = {
            "combo_index": idx,
            "survival_rate": float(agent_df['survived'].mean()) if not agent_df.empty and 'survived' in agent_df.columns else float('nan'),
            "mean_capital": float(agent_df['final_capital'].mean()) if not agent_df.empty and 'final_capital' in agent_df.columns else float('nan'),
            "median_capital": float(agent_df['final_capital'].median()) if not agent_df.empty and 'final_capital' in agent_df.columns else float('nan'),
            "mean_roic_invest": float(summary_df['mean_roic_invest'].dropna().mean()) if not summary_df.empty and 'mean_roic_invest' in summary_df.columns else float('nan'),
            "mean_roic_innovate": float(summary_df['mean_roic_innovate'].dropna().mean()) if not summary_df.empty and 'mean_roic_innovate' in summary_df.columns else float('nan'),
            "mean_action_share_invest": float(summary_df['share_invest'].dropna().mean()) if not summary_df.empty and 'share_invest' in summary_df.columns else float('nan'),
        }
        metrics.update(overrides)
        summary_records.append(metrics)
    summary_df = pd.DataFrame(summary_records)
    summary_path = Path(root_dir) / "sensitivity_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    effect_rows: List[Dict[str, Any]] = []
    metric_cols = ["survival_rate", "mean_capital", "mean_roic_invest"]
    for metric in metric_cols:
        if metric not in summary_df.columns:
            continue
        total_range = 0.0
        spreads: Dict[str, float] = {}
        for name in param_names:
            grouped = summary_df.groupby(name)[metric].mean()
            if grouped.empty:
                spread = 0.0
            else:
                spread = float(grouped.max() - grouped.min())
            spreads[name] = spread
            total_range += spread
        for name in param_names:
            contribution = 0.0
            if total_range > 0:
                contribution = spreads[name] / total_range
            effect_rows.append({
                "metric": metric,
                "parameter": name,
                "normalized_effect": contribution,
                "range": spreads[name],
            })
    effects_df = pd.DataFrame(effect_rows)
    effects_path = Path(root_dir) / "sensitivity_effects.csv"
    effects_df.to_csv(effects_path, index=False)
    _plot_sensitivity_effects(effects_path, Path(root_dir))
    return {
        "results_directory": root_dir,
        "summary_csv": str(summary_path),
        "effects_csv": str(effects_path),
        "config_snapshot": str(snapshot_path),
    }


def run_lhs_sweep(
    base_config: EmergentConfig,
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    n_samples: int = 12,
    rounds: int = 80,
    agents: int = 250,
    runs_per_sample: int = 1,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    calibration_metadata: Optional[List[Dict[str, Any]]] = None,
    cli_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    bounds = ranges or DEFAULT_LHS_RANGES
    samples = _latin_hypercube_samples(bounds, n_samples, seed=seed)
    if not samples:
        return {"status": "No parameter ranges provided for LHS sweep."}
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    root_dir = output_dir or f"./lhs_results_{timestamp}"
    os.makedirs(root_dir, exist_ok=True)
    snapshot_path = _persist_config_snapshot(root_dir, base_config, cli_args=cli_args, calibration_metadata=calibration_metadata)
    print(f"[LHS] Configuration snapshot stored in: {snapshot_path}")
    summary_records: List[Dict[str, Any]] = []
    for idx, overrides in enumerate(samples):
        combo_cfg = base_config.copy_with_overrides(overrides)
        combo_cfg.N_RUNS = runs_per_sample
        combo_cfg.N_ROUNDS = min(rounds, combo_cfg.N_ROUNDS)
        combo_cfg.N_AGENTS = min(agents, combo_cfg.N_AGENTS)
        combo_cfg.use_parallel = False
        combo_dir = os.path.join(root_dir, f"lhs_{idx}")
        os.makedirs(combo_dir, exist_ok=True)
        sim = EmergentSimulation(config=combo_cfg, output_dir=combo_dir, run_id=f"lhs_run_{idx}")
        print(f"[LHS] Running sample {idx+1}/{len(samples)} with overrides {overrides}")
        sim.run()
        framework = ComprehensiveAnalysisFramework(combo_dir, combo_cfg)
        framework.run_full_analysis()
        agent_df = framework.agent_df
        summary_df = framework.summary_df
        metrics = {
            "sample_index": idx,
            "survival_rate": float(agent_df['survived'].mean()) if not agent_df.empty and 'survived' in agent_df.columns else float('nan'),
            "mean_capital": float(agent_df['final_capital'].mean()) if not agent_df.empty and 'final_capital' in agent_df.columns else float('nan'),
            "mean_roic_invest": float(summary_df['mean_roic_invest'].dropna().mean()) if not summary_df.empty and 'mean_roic_invest' in summary_df.columns else float('nan'),
        }
        metrics.update(overrides)
        summary_records.append(metrics)
    summary_df = pd.DataFrame(summary_records)
    summary_path = Path(root_dir) / "lhs_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return {
        "results_directory": root_dir,
        "summary_csv": str(summary_path),
        "config_snapshot": str(snapshot_path),
    }


def _build_scenarios(base_config: EmergentConfig) -> Dict[str, EmergentConfig]:
    scenarios: Dict[str, EmergentConfig] = {}
    baseline = copy.deepcopy(base_config)
    baseline.AGENT_AI_MODE = "emergent"
    scenarios["Baseline"] = baseline

    ignorance = copy.deepcopy(baseline)
    ignorance.OPPORTUNITY_COMPLEXITY_RANGE = (0.7, 2.0)
    ignorance.DISCOVERY_PROBABILITY = 0.15
    scenarios["High_Ignorance"] = ignorance

    indeterminism = copy.deepcopy(baseline)
    indeterminism.MARKET_VOLATILITY = 0.45
    indeterminism.BLACK_SWAN_PROBABILITY = 0.03
    scenarios["High_Indeterminism"] = indeterminism

    novelty = copy.deepcopy(baseline)
    novelty.INNOVATION_PROBABILITY = min(1.0, base_config.INNOVATION_PROBABILITY * 1.25)
    novelty.EXPLORATION_DECAY = max(0.90, base_config.EXPLORATION_DECAY * 0.95)
    scenarios["High_AgenticNovelty"] = novelty

    recursion = copy.deepcopy(baseline)
    recursion.COMPETITION_COST_MULTIPLIER = base_config.COMPETITION_COST_MULTIPLIER * 1.35
    recursion.COMPETITION_SCALE_FACTOR = min(2.0, base_config.COMPETITION_SCALE_FACTOR * 1.5)
    recursion.BLACK_SWAN_PROBABILITY = max(0.01, base_config.BLACK_SWAN_PROBABILITY * 1.2)
    scenarios["High_CompetitiveRecursion"] = recursion
    return scenarios


def _summarize_scenario_results(
    results_dir: str,
    base_config: EmergentConfig,
) -> tuple[Optional[ComprehensiveAnalysisFramework], Optional[Dict[str, Any]]]:
    try:
        # Fresh config copy prevents cross-scenario mutation.
        framework = ComprehensiveAnalysisFramework(results_dir, copy.deepcopy(base_config))
        summary = framework.run_full_analysis()
        return framework, summary
    except Exception as exc:  # pragma: no cover - surfaced to console for manual follow-up
        print(f"Analysis failed for {results_dir}: {exc}")
        return None, None


def _collect_decision_data(framework_summaries: Dict[str, Dict[str, Any]], scenario_order: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for scenario in scenario_order:
        info = framework_summaries.get(scenario)
        if not info:
            continue
        df = info["framework"].decision_df.copy()
        if df.empty:
            continue
        df["scenario"] = scenario
        uncertainty_df = info["framework"].uncertainty_df
        if not uncertainty_df.empty:
            merge_cols = ["round"]
            if "agent_id" in df.columns and "agent_id" in uncertainty_df.columns:
                merge_cols.append("agent_id")
            elif "agent" in df.columns and "agent" in uncertainty_df.columns:
                df = df.rename(columns={"agent": "agent_id"})
                uncertainty_df = uncertainty_df.rename(columns={"agent": "agent_id"})
                merge_cols.append("agent_id")
            elif "run_id" in df.columns and "run_id" in uncertainty_df.columns:
                merge_cols.append("run_id")
            shared_cols = [col for col in merge_cols if col in df.columns and col in uncertainty_df.columns]
            if len(shared_cols) == 1 and shared_cols[0] == "round":
                merged = (
                    uncertainty_df.groupby("round")["competitive_recursion_level"]
                    .mean()
                    .reset_index()
                    .rename(columns={"competitive_recursion_level": "recursion_level"})
                )
                df = df.merge(merged, on="round", how="left")
            elif shared_cols:
                cols_needed = shared_cols + ["competitive_recursion_level"]
                available = [c for c in cols_needed if c in uncertainty_df.columns]
                merged = uncertainty_df[available].drop_duplicates()
                if "competitive_recursion_level" in merged.columns:
                    merged = merged.rename(columns={"competitive_recursion_level": "recursion_level"})
                df = df.merge(merged, on=shared_cols, how="left")
        frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def _collect_summary_data(framework_summaries: Dict[str, Dict[str, Any]], scenario_order: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for scenario in scenario_order:
        info = framework_summaries.get(scenario)
        if not info:
            continue
        summary_df = getattr(info["framework"], "summary_df", pd.DataFrame()).copy()
        if summary_df.empty:
            continue
        summary_df["scenario"] = scenario
        frames.append(summary_df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def _collect_uncertainty_data(framework_summaries: Dict[str, Dict[str, Any]], scenario_order: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for scenario in scenario_order:
        info = framework_summaries.get(scenario)
        if not info:
            continue
        unc_df = info["framework"].uncertainty_df.copy()
        if unc_df.empty:
            continue
        unc_df["scenario"] = scenario
        frames.append(unc_df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def _ensure_figures_dir(root_dir: str) -> str:
    figures_dir = os.path.join(root_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def _save_and_display(fig: plt.Figure, figures_dir: str, filename: str) -> str:
    filepath = os.path.join(figures_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return filepath


def _plot_adoption_by_uncertainty(
    framework_summaries: Dict[str, Dict[str, Any]],
    scenario_order: List[str],
    figures_dir: str,
) -> None:
    decisions = _collect_decision_data(framework_summaries, scenario_order)
    if decisions.empty or "recursion_level" not in decisions.columns:
        print("Insufficient data for adoption vs. recursion visualization.")
        return
    available = [s for s in scenario_order if s in decisions["scenario"].unique()]
    if not available:
        print("No scenarios with adoption data available.")
        return
    fig, axes = plt.subplots(len(available), 1, figsize=(12, 4 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]
    palette = {"none": "#95a5a6", "basic": "#3498db", "advanced": "#f39c12", "premium": "#e74c3c"}
    for ax, scenario in zip(axes, available):
        df = decisions[decisions["scenario"] == scenario].copy()
        if df.empty or df["recursion_level"].dropna().empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        recursion_series = df["recursion_level"].fillna(df["recursion_level"].median())
        if recursion_series.nunique() <= 1:
            ax.text(0.5, 0.5, "Insufficient variation", ha="center", va="center")
            continue
        bins = np.linspace(recursion_series.min(), recursion_series.max(), 12)
        df["recursion_bin"] = pd.cut(recursion_series, bins=bins, include_lowest=True)
        adoption = df.groupby(["recursion_bin", "ai_level_used"]).size().reset_index(name="count")
        if adoption.empty:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            continue
        adoption["recursion_mid"] = adoption["recursion_bin"].apply(
            lambda x: x.left + (x.width if hasattr(x, "width") else (x.right - x.left)) / 2
        )
        sns.lineplot(
            data=adoption,
            x="recursion_mid",
            y="count",
            hue="ai_level_used",
            hue_order=["none", "basic", "advanced", "premium"],
            palette=palette,
            ax=ax,
        )
        ax.set_title(scenario)
        ax.set_ylabel("Decision count")
        ax.legend(title="AI level")
    axes[-1].set_xlabel("Competitive recursion intensity (binned midpoint)")
    plt.tight_layout()
    path = _save_and_display(fig, figures_dir, "adoption_vs_competitive_recursion.png")
    print("Saved adoption vs. recursion figure to", path)


def _plot_ai_adoption_over_time(
    framework_summaries: Dict[str, Dict[str, Any]],
    scenario_order: List[str],
    figures_dir: str,
) -> None:
    summaries = _collect_summary_data(framework_summaries, scenario_order)
    if summaries.empty:
        print("No summary data available for AI adoption visualization.")
        return
    fig, axes = plt.subplots(len(scenario_order), 1, figsize=(12, 4 * len(scenario_order)), sharex=True)
    if len(scenario_order) == 1:
        axes = [axes]
    palette = {"none": "#95a5a6", "basic": "#3498db", "advanced": "#f39c12", "premium": "#e74c3c"}
    for ax, scenario in zip(axes, scenario_order):
        df = summaries[summaries["scenario"] == scenario].copy()
        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        adoption_cols = ["ai_share_none", "ai_share_basic", "ai_share_advanced", "ai_share_premium"]
        if any(col not in df.columns for col in adoption_cols):
            ax.text(0.5, 0.5, "Missing adoption metrics", ha="center", va="center")
            continue
        melted = df[["round"] + adoption_cols].melt(id_vars="round", var_name="ai_level", value_name="share")
        label_map = {
            "ai_share_none": "none",
            "ai_share_basic": "basic",
            "ai_share_advanced": "advanced",
            "ai_share_premium": "premium",
        }
        melted["ai_level"] = melted["ai_level"].map(label_map)
        sns.lineplot(
            data=melted,
            x="round",
            y="share",
            hue="ai_level",
            palette=palette,
            hue_order=["none", "basic", "advanced", "premium"],
            ax=ax,
        )
        ax.set_title(f"AI adoption share – {scenario}")
        ax.set_ylabel("Share of active agents")
        ax.set_ylim(0, 1)
        ax.legend(title="AI level")
    axes[-1].set_xlabel("Simulation round")
    plt.tight_layout()
    path = _save_and_display(fig, figures_dir, "ai_adoption_share_by_scenario.png")
    print("Saved AI adoption figure to", path)


def _plot_uncertainty_profiles(
    framework_summaries: Dict[str, Dict[str, Any]],
    scenario_order: List[str],
    figures_dir: str,
) -> None:
    unc = _collect_uncertainty_data(framework_summaries, scenario_order)
    if unc.empty:
        print("No uncertainty data available for visualization.")
        return
    plot_cols = {
        "actor_ignorance_level": "Actor Ignorance",
        "practical_indeterminism_level": "Practical Indeterminism",
        "agentic_novelty_level": "Agentic Novelty",
        "competitive_recursion_level": "Competitive Recursion",
    }
    available_cols = [col for col in plot_cols if col in unc.columns]
    if not available_cols:
        print("Uncertainty columns missing; skipping profile visualization.")
        return
    fig, axes = plt.subplots(len(available_cols), 1, figsize=(12, 3.5 * len(available_cols)), sharex=True)
    if len(available_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, available_cols):
        sns.lineplot(data=unc, x="round", y=col, hue="scenario", hue_order=scenario_order, ax=ax)
        ax.set_ylabel("Average level")
        ax.set_title(plot_cols[col])
        ax.legend(title="Scenario")
    axes[-1].set_xlabel("Simulation round")
    plt.tight_layout()
    path = _save_and_display(fig, figures_dir, "uncertainty_profiles_by_scenario.png")
    print("Saved uncertainty profile figure to", path)


def _plot_performance_vs_uncertainty(
    framework_summaries: Dict[str, Dict[str, Any]],
    scenario_order: List[str],
    figures_dir: str,
) -> None:
    summaries = _collect_summary_data(framework_summaries, scenario_order)
    if summaries.empty or "innovation_success_rate" not in summaries.columns:
        print("No performance summary data available.")
        return
    melted = summaries[["scenario", "round", "innovation_success_rate", "mean_capital"]].melt(
        id_vars=["scenario", "round"],
        value_vars=["innovation_success_rate", "mean_capital"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=melted,
        x="round",
        y="value",
        hue="scenario",
        style="metric",
        hue_order=scenario_order,
        ax=ax,
    )
    ax.set_title("Innovation success rate and capital vs. round")
    ax.set_ylabel("Value")
    ax.legend(title="Scenario / Metric")
    plt.tight_layout()
    path = _save_and_display(fig, figures_dir, "performance_vs_round.png")
    print("Saved performance vs. round figure to", path)


AI_LEVEL_ORDER = {"none": 0, "basic": 1, "advanced": 2, "premium": 3}
AI_LEVEL_CANONICAL = list(AI_LEVEL_ORDER.keys())


def _normalize_ai_level_label(value: Any) -> Optional[str]:
    """Normalize AI level labels, returning None for missing/invalid values."""
    if value is None:
        return None
    if isinstance(value, (float, int)) and pd.isna(value):
        return None
    # Use canonical normalize_ai_label from utils
    normalized = normalize_ai_label(value, default="")
    if not normalized or normalized not in AI_LEVEL_ORDER:
        return None
    return normalized


def _encode_ai_level(df: pd.DataFrame, col: str = "primary_ai_level") -> pd.DataFrame:
    df = df.copy()
    normalized = df[col].apply(_normalize_ai_level_label)
    df[col] = normalized
    df["ai_level_ord"] = normalized.map(AI_LEVEL_ORDER)
    for lvl in AI_LEVEL_CANONICAL:
        df[f"ai_{lvl}"] = (normalized == lvl).astype(int)
    return df


def _run_scaling_simulation(
    agent_count: int,
    replicate: int,
    base_config: EmergentConfig,
    scaling_root: Path,
) -> Dict[str, Any]:
    run_config = copy.deepcopy(base_config)
    run_config.N_AGENTS = agent_count
    run_config.AGENT_AI_MODE = "emergent"
    run_id = f"scaling_{agent_count}_rep{replicate:02d}"
    run_dir = scaling_root / f"agents_{agent_count}" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    sim = EmergentSimulation(config=run_config, output_dir=str(run_dir), run_id=run_id)
    try:
        start = time.time()
        sim.run()
        duration = time.time() - start
        return {"status": "completed", "duration_sec": duration, "output_dir": str(run_dir)}
    except Exception as exc:  # pragma: no cover - surfaced to console for manual follow-up
        return {"status": "failed", "error": str(exc), "output_dir": str(run_dir)}


def _summarize_scaling_runs(scaling_root: Path, counts: List[int], replicates: int) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for n_agents in counts:
        for rep in range(replicates):
            run_id = f"scaling_{n_agents}_rep{rep:02d}"
            run_dir = scaling_root / f"agents_{n_agents}" / run_id
            final_agents_path = run_dir / "final_agents.pkl"
            if not final_agents_path.exists():
                records.append({"n_agents": n_agents, "replicate": rep, "status": "missing"})
                continue
            agent_df = pd.read_pickle(final_agents_path)
            record: Dict[str, Any] = {
                "n_agents": n_agents,
                "replicate": rep,
                "status": "completed",
                "survival_rate": agent_df.get("survived", pd.Series(dtype=float)).mean(),
                "innovation_mean": agent_df.get("innovations", pd.Series(dtype=float)).mean(),
                "capital_growth_mean": agent_df.get("capital_growth", pd.Series(dtype=float)).mean(),
            }
            if "final_capital" in agent_df.columns:
                record["capital_gini"] = agent_df["final_capital"].rank(pct=True).std()
            records.append(record)
    return pd.DataFrame(records)


def run_agent_scaling_analysis(
    agent_counts: Optional[Iterable[int]] = None,
    replicates: int = 3,
    n_jobs: Optional[int] = None,
    base_config: Optional[EmergentConfig] = None,
) -> pd.DataFrame:
    """
    Perform an agent-scaling sweep to understand how population size impacts stability.
    Returns the aggregated summary dataframe.
    """
    agent_counts = list(agent_counts) if agent_counts is not None else [200, 400, 600, 800, 1000, 1200, 1500]
    base_cfg = copy.deepcopy(base_config or EmergentConfig())
    workload_cfg = copy.deepcopy(base_cfg)
    workload_cfg.N_AGENTS = max(agent_counts) if agent_counts else workload_cfg.N_AGENTS
    auto_jobs, task_timeout, workload = _compute_parallel_plan(workload_cfg)
    if n_jobs is None:
        n_jobs = auto_jobs
    else:
        n_jobs = max(1, min(int(n_jobs), auto_jobs))
    _tune_buffering_for_parallel(base_cfg, workload, n_jobs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scaling_root = Path(f"./agent_scaling_results_{timestamp}")
    scaling_root.mkdir(parents=True, exist_ok=True)
    print(
        f"Running agent scaling sweep: counts={agent_counts}, "
        f"replicates={replicates}, n_jobs={n_jobs}, load≈{workload:.1f}"
    )
    jobs = [(count, rep) for count in agent_counts for rep in range(replicates)]

    def _scaling_job(job: tuple[int, int]) -> Dict[str, Any]:
        count, rep = job
        return _run_scaling_simulation(count, rep, base_cfg, scaling_root)

    _execute_parallel_tasks(
        jobs,
        _scaling_job,
        n_jobs,
        desc="agent scaling runs",
        timeout=task_timeout,
    )
    summary_df = _summarize_scaling_runs(scaling_root, agent_counts, replicates)
    summary_path = scaling_root / "agent_scaling_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("Agent scaling summary saved to", summary_path)
    if not summary_df.empty:
        pivot = (
            summary_df[summary_df["status"] == "completed"]
            .pivot_table(
                index="n_agents",
                values=["survival_rate", "innovation_mean", "capital_growth_mean"],
                aggfunc="mean",
            )
            .reset_index()
        )
        display(pivot)
    else:
        print("No successful scaling runs to summarize.")
    return summary_df


def _parse_cli_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Glimpse ABM experiment and analysis launcher")
    parser.add_argument(
        "--task",
        choices=["master", "fixed", "scenarios", "sensitivity", "lhs"],
        help="Select which experiment workflow to run.",
    )
    parser.add_argument(
        "--results-dir",
        help="Override the default output directory for generated artefacts.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        help="Override the number of runs for the selected workflow.",
    )
    parser.add_argument(
        "--ai-levels",
        nargs="+",
        help="Custom list of AI levels to evaluate for fixed-level experiments.",
    )
    parser.add_argument(
        "--include-fixed",
        action="store_true",
        help="When running the master workflow, also execute the fixed-level comparison batch.",
    )
    parser.add_argument(
        "--skip-emergent",
        action="store_true",
        help="When running the master workflow, skip the emergent AI experiment.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Enable smoke-mode overrides (small agent/round counts, no parallelism).",
    )
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip visualization rendering after analysis (useful for large batches or profiling).",
    )
    parser.add_argument(
        "--fast-stats",
        action="store_true",
        help="Use reduced bootstrap iterations (500 vs 5000) for faster statistical tests. Useful for robustness sweeps.",
    )
    parser.add_argument(
        "--experiment-profile",
        choices=["baseline", "agi2027"],
        help="Apply predefined experimental overrides (e.g., AGI-tier pricing/capabilities).",
    )
    parser.add_argument(
        "--parallel-mode",
        choices=["max", "safe"],
        help="Parallelism strategy: 'max' uses CPUs-1 for maximum throughput (default); 'safe' applies conservative workload-based caps.",
    )
    parser.add_argument(
        "--max-parallel-runs",
        type=int,
        help="Explicit cap on concurrent simulation processes (overrides --parallel-mode calculation).",
    )
    parser.add_argument(
        "--agent-workers",
        type=int,
        help="Thread workers to use inside each simulation for agent-level decisions. Set to 1 to force serial agent loops.",
    )
    parser.add_argument(
        "--calibration-profile",
        help="Apply a named calibration profile to align the run with empirical targets.",
    )
    parser.add_argument(
        "--calibration-file",
        help="Path to a JSON calibration definition with overrides/targets to apply on top of the selected profile.",
    )
    parser.add_argument(
        "--list-calibrations",
        action="store_true",
        help="List available calibration profiles and exit (unless a task is also requested).",
    )
    parser.add_argument(
        "--monitor-progress",
        action="store_true",
        help="Print periodic progress snapshots while simulations are running.",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=30.0,
        help="Seconds between progress snapshots when monitoring is enabled.",
    )
    parser.add_argument(
        "--monitor-limit",
        type=int,
        help="Only include the first N run directories when monitoring progress.",
    )
    parser.add_argument(
        "--human-only-baseline",
        action="store_true",
        help="Run a human-only baseline (agents locked to no AI) instead of emergent selection.",
    )
    parser.add_argument(
        "--fixed-tier-sweep",
        action="store_true",
        help="Force a fixed-tier sweep (none/basic/advanced/premium) alongside the main run.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Override the base RNG seed used to initialize each run.",
    )
    parser.add_argument(
        "--dump-config",
        help="Optional path to write the resolved configuration JSON before execution.",
    )
    parser.add_argument(
        "--sensitivity-param",
        action="append",
        help="Parameter sweep entry in the form name=value1,value2 (repeatable).",
    )
    parser.add_argument(
        "--sensitivity-rounds",
        type=int,
        default=80,
        help="Rounds per sensitivity simulation (default: 80).",
    )
    parser.add_argument(
        "--sensitivity-agents",
        type=int,
        default=250,
        help="Agents per sensitivity simulation (default: 250).",
    )
    parser.add_argument(
        "--lhs-param-range",
        action="append",
        help="Range definition for LHS sweeps in the form name=low:high (repeatable).",
    )
    parser.add_argument(
        "--lhs-samples",
        type=int,
        default=12,
        help="Number of Latin hypercube samples (default: 12).",
    )
    parser.add_argument(
        "--lhs-seed",
        type=int,
        default=42,
        help="Seed for LHS sampling.",
    )
    # Ablation flags for causal identification
    parser.add_argument(
        "--no-knowledge-decay",
        action="store_true",
        help="Ablation: disable knowledge decay to isolate AI information effects.",
    )
    parser.add_argument(
        "--no-market-regimes",
        action="store_true",
        help="Ablation: fix market regime to 'normal' to remove regime confounds.",
    )
    parser.add_argument(
        "--no-network-effects",
        action="store_true",
        help="Ablation: disable network effects to remove social contagion confounds.",
    )
    parser.add_argument(
        "--no-innovation",
        action="store_true",
        help="Ablation: disable innovation engine to isolate investment dynamics.",
    )
    parser.add_argument(
        "--reduced-noise",
        action="store_true",
        help="Ablation: reduce stochastic noise for cleaner signal detection.",
    )
    parser.add_argument(
        "--ablation-report",
        action="store_true",
        help="Run full ablation study: compare baseline with each mechanism disabled.",
    )
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Generate publication-ready outputs: LaTeX tables, PDF figures, and a comprehensive report.",
    )
    parser.add_argument(
        "--publication-author",
        type=str,
        default="",
        help="Author name(s) for the publication report.",
    )
    return parser.parse_args(args=list(argv) if argv is not None else None)


def run_cli(
    base_config: Optional[EmergentConfig] = None,
    argv: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Parse CLI arguments and dispatch the requested workflow.
    Returns the workflow result dictionary (if any), allowing programmatic reuse.
    """
    args = _parse_cli_args(argv)
    if args.list_calibrations:
        _print_calibration_catalog()
        if not args.task:
            return None
    if not args.task:
        print("No task selected. Use --task with one of 'master', 'fixed', or 'scenarios'.")
        print("Run with --help for details.")
        return None

    base_cfg = copy.deepcopy(base_config or EmergentConfig())
    if args.runs is not None:
        base_cfg.N_RUNS = args.runs
    if args.parallel_mode is not None:
        base_cfg.PARALLEL_MODE = args.parallel_mode
    if args.max_parallel_runs is not None:
        base_cfg.MAX_PARALLEL_RUNS = max(1, args.max_parallel_runs)
    if args.smoke:
        os.environ.setdefault("GLIMPSE_ABM_SMOKE_TEST", "1")
        base_cfg = _apply_smoke_overrides(base_cfg)
    if args.experiment_profile:
        base_cfg = _apply_experiment_profile(base_cfg, args.experiment_profile)
    calibration_metadata: List[Dict[str, Any]] = []
    try:
        if args.calibration_profile:
            builtin_profile = get_calibration_profile(args.calibration_profile)
            base_cfg = apply_calibration_profile(base_cfg, builtin_profile)
            calibration_metadata.append(builtin_profile.to_metadata())
        if args.calibration_file:
            file_profile = load_calibration_profile(args.calibration_file)
            base_cfg = apply_calibration_profile(base_cfg, file_profile)
            calibration_metadata.append(file_profile.to_metadata())
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[CLI] ❌ Calibration error: {exc}")
        return None
    if args.random_seed is not None:
        base_cfg.RANDOM_SEED = args.random_seed

    # Apply ablation overrides for causal identification
    ablation_flags_active = []
    if getattr(args, 'no_knowledge_decay', False):
        base_cfg.KNOWLEDGE_DECAY_RATE = 0.0
        ablation_flags_active.append("no-knowledge-decay")
    if getattr(args, 'no_market_regimes', False):
        # Fix all regime transitions to stay in 'normal'
        base_cfg.MACRO_REGIME_TRANSITIONS = {
            regime: {"crisis": 0.0, "recession": 0.0, "normal": 1.0, "growth": 0.0, "boom": 0.0}
            for regime in base_cfg.MACRO_REGIME_STATES
        }
        ablation_flags_active.append("no-market-regimes")
    if getattr(args, 'no_network_effects', False):
        base_cfg.USE_NETWORK_EFFECTS = False
        ablation_flags_active.append("no-network-effects")
    if getattr(args, 'no_innovation', False):
        base_cfg.INNOVATION_PROBABILITY = 0.0
        base_cfg.INNOVATION_REUSE_PROBABILITY = 0.0
        ablation_flags_active.append("no-innovation")
    if getattr(args, 'reduced_noise', False):
        base_cfg.RETURN_NOISE_SCALE = 0.15
        base_cfg.BLACK_SWAN_PROBABILITY = 0.0
        base_cfg.MARKET_SHIFT_PROBABILITY = 0.0
        base_cfg.MARKET_VOLATILITY = 0.10
        ablation_flags_active.append("reduced-noise")

    print("[CLI] Glimpse ABM launcher starting")
    print(f"[CLI] Task: {args.task}")
    print(f"[CLI] Smoke mode: {'on' if args.smoke else 'off'}")
    if args.results_dir:
        print(f"[CLI] Results directory override: {args.results_dir}")
    if args.skip_visualizations:
        print("[CLI] Visualization suite disabled (--skip-visualizations).")
    if args.fast_stats:
        from .statistical_tests import set_fast_stats_mode
        set_fast_stats_mode(True)
    if args.ai_levels:
        print(f"[CLI] AI levels requested: {args.ai_levels}")
    if args.monitor_progress:
        print(
            f"[CLI] Progress monitor enabled "
            f"(interval={args.monitor_interval}s"
            + (f", limit={args.monitor_limit}" if args.monitor_limit is not None else "")
            + ")"
        )
    if calibration_metadata:
        applied = ", ".join(meta["name"] for meta in calibration_metadata)
        print(f"[CLI] Calibration profiles applied: {applied}")
    if args.random_seed is not None:
        print(f"[CLI] Random seed override: {args.random_seed}")
    if ablation_flags_active:
        print(f"[CLI] Ablation flags active: {', '.join(ablation_flags_active)}")

    if args.dump_config:
        _write_config_dump(base_cfg, args.dump_config)

    result: Optional[Dict[str, Any]] = None
    if args.task == "master":
        try:
            max_jobs, _, workload = _compute_parallel_plan(base_cfg)
            # If the user asks for agent-level threading, still disable it when processes >1 to avoid nested pools.
            if args.agent_workers is not None:
                requested_workers = max(1, args.agent_workers)
                if max_jobs > 1:
                    print(
                        f"[CLI] Agent-level threading request ({requested_workers}) ignored "
                        f"because {max_jobs} process workers are active."
                    )
                else:
                    base_cfg.max_workers = requested_workers
                    base_cfg.use_parallel = requested_workers > 1
            if (
                max_jobs > 1
                and base_cfg.use_parallel
                and base_cfg.max_workers > 1
            ):
                base_cfg.use_parallel = False
                base_cfg.max_workers = 1
                note = f"[CLI] Agent-level threading disabled to prioritize {max_jobs} process workers."
                if workload >= 2.0:
                    note += f" (load≈{workload:.1f})"
                print(note)

            run_emergent_flag = not args.skip_emergent and not args.human_only_baseline
            fixed_levels_flag = args.include_fixed or args.fixed_tier_sweep or args.human_only_baseline
            ai_levels_override = args.ai_levels
            if args.human_only_baseline:
                ai_levels_override = ["none"]
            elif args.fixed_tier_sweep and not ai_levels_override:
                ai_levels_override = ["none", "basic", "advanced", "premium"]
            result = run_master_launcher(
                base_cfg,
                run_emergent=run_emergent_flag,
                run_fixed_levels=fixed_levels_flag,
                results_dir=args.results_dir,
                ai_levels_to_test=ai_levels_override,
                monitor_progress=args.monitor_progress,
                monitor_interval=args.monitor_interval,
                monitor_limit=args.monitor_limit,
                calibration_metadata=calibration_metadata,
                cli_args=vars(args),
                skip_visualizations=args.skip_visualizations,
            )
        except Exception as exc:
            print(f"[CLI] ❌ Master task failed: {exc}")
            raise
    elif args.task == "fixed":
        try:
            result = run_fixed_level_uncertainty_batch(
                base_cfg,
                ai_levels=args.ai_levels,
                runs_per_level=args.runs,
                output_dir=args.results_dir,
                calibration_metadata=calibration_metadata,
                cli_args=vars(args),
                skip_visualizations=args.skip_visualizations,
            )
        except Exception as exc:
            print(f"[CLI] ❌ Fixed-level task failed: {exc}")
            raise
    elif args.task == "scenarios":
        try:
            result = run_uncertainty_scenario_sweep(
                base_cfg,
                scenario_runs=args.runs,
                root_dir=args.results_dir,
                calibration_metadata=calibration_metadata,
                cli_args=vars(args),
            )
        except Exception as exc:
            print(f"[CLI] ❌ Scenario task failed: {exc}")
            raise
    elif args.task == "sensitivity":
        try:
            sweep_params = _parse_sensitivity_params(args.sensitivity_param)
            result = run_sensitivity_sweep(
                base_cfg,
                param_grid=sweep_params if sweep_params else None,
                rounds=args.sensitivity_rounds,
                agents=args.sensitivity_agents,
                output_dir=args.results_dir,
                calibration_metadata=calibration_metadata,
                cli_args=vars(args),
            )
        except Exception as exc:
            print(f"[CLI] ❌ Sensitivity task failed: {exc}")
            raise
    elif args.task == "lhs":
        try:
            range_overrides = _parse_lhs_ranges(args.lhs_param_range)
            result = run_lhs_sweep(
                base_cfg,
                ranges=range_overrides if range_overrides else None,
                n_samples=args.lhs_samples,
                rounds=args.sensitivity_rounds,
                agents=args.sensitivity_agents,
                seed=args.lhs_seed,
                output_dir=args.results_dir,
                calibration_metadata=calibration_metadata,
                cli_args=vars(args),
            )
        except Exception as exc:
            print(f"[CLI] ❌ LHS task failed: {exc}")
            raise

    if isinstance(result, dict):
        if "results_directory" in result:
            print(f"[CLI] Results directory: {result['results_directory']}")
        if "results_dir" in result:
            print(f"[CLI] Results directory: {result['results_dir']}")
        if "config_snapshot" in result:
            print(f"[CLI] Config snapshot: {result['config_snapshot']}")
        if result.get("analysis_metadata"):
            print(f"[CLI] Analysis metadata: {result['analysis_metadata']}")
        if "summary_csv" in result:
            print(f"[CLI] Sensitivity summary: {result['summary_csv']}")
        if "effects_csv" in result:
            print(f"[CLI] Sensitivity effects: {result['effects_csv']}")

        # Generate publication outputs if requested
        if getattr(args, 'publication', False):
            if PublicationPipeline is None:
                print("[CLI] Publication pipeline not available (module archived)")
            else:
                results_dir = result.get('results_directory') or result.get('results_dir')
                if results_dir:
                    print("\n[CLI] Generating publication outputs...")
                    try:
                        pub_output_dir = os.path.join(results_dir, 'publication')
                        pipeline = PublicationPipeline(results_dir, pub_output_dir)
                        pub_results = pipeline.run_full_pipeline(
                            author=getattr(args, 'publication_author', '')
                        )
                        result['publication_outputs'] = pub_results
                        print(f"[CLI] Publication outputs: {pub_output_dir}")
                    except Exception as exc:
                        print(f"[CLI] Publication generation failed: {exc}")

    print("[CLI] ✅ Task completed.")
    return result


def main(argv: Optional[Iterable[str]] = None) -> None:  # pragma: no cover - thin wrapper
    run_cli(argv=argv)


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "run_cli",
    "run_master_launcher",
    "run_fixed_level_uncertainty_batch",
    "run_uncertainty_scenario_sweep",
    "run_agent_scaling_analysis",
    "suppress_runtime_warnings",
]
