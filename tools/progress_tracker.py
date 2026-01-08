#!/usr/bin/env python3
"""Lightweight, read-only progress monitor for active Glimpse ABM runs."""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

RUN_DIR_PATTERN = re.compile(r"emergent_run_(\d+)")


@dataclass(frozen=True)
class RunProgress:
    name: str
    rounds_completed: int

    @property
    def round_label(self) -> str:
        return f"{self.rounds_completed:>5d}"


def _sorted_run_dirs(results_dir: Path) -> Iterable[Path]:
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        match = RUN_DIR_PATTERN.fullmatch(child.name)
        if match:
            yield child


def _load_latest_round_from_subdir(subdir: Path) -> Optional[int]:
    """Return zero-based round index from the newest .pkl, if available."""
    if not subdir.exists():
        return None

    files = sorted(subdir.glob("*.pkl"))
    for path in reversed(files):
        try:
            df = pd.read_pickle(path)
        except (FileNotFoundError, pd.errors.EmptyDataError, EOFError, ValueError):
            continue
        if "round" not in df.columns:
            continue
        rounds = (
            pd.to_numeric(df["round"], errors="coerce")
            .dropna()
            .astype(int)
        )
        if not rounds.empty:
            return int(rounds.max())
    return None


def _load_latest_round_from_log(log_path: Path) -> Optional[int]:
    """Return zero-based round index by counting entries in run_log.jsonl."""
    if not log_path.exists():
        return None
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        return count - 1 if count > 0 else None
    except OSError:
        return None


def _detect_run_progress(run_dir: Path) -> RunProgress:
    candidates = [
        _load_latest_round_from_log(run_dir / "run_log.jsonl"),
        _load_latest_round_from_subdir(run_dir / "summary"),
        _load_latest_round_from_subdir(run_dir / "decisions"),
        _load_latest_round_from_subdir(run_dir / "market"),
    ]
    max_round = max((c for c in candidates if c is not None), default=-1)
    rounds_completed = max_round + 1 if max_round >= 0 else 0
    return RunProgress(name=run_dir.name, rounds_completed=rounds_completed)


def collect_progress(results_dir: Path, limit: Optional[int] = None) -> Dict[str, RunProgress]:
    progress: Dict[str, RunProgress] = {}
    for idx, run_dir in enumerate(_sorted_run_dirs(results_dir)):
        if limit is not None and idx >= limit:
            break
        progress[run_dir.name] = _detect_run_progress(run_dir)
    return progress


def render_report(progress: Dict[str, RunProgress]) -> str:
    if not progress:
        return "No run directories discovered yet."

    ordered = sorted(progress.values(), key=lambda rp: int(RUN_DIR_PATTERN.match(rp.name).group(1)))
    total_rounds = sum(rp.rounds_completed for rp in ordered)
    max_rounds = max((rp.rounds_completed for rp in ordered), default=0)
    active_runs = sum(1 for rp in ordered if rp.rounds_completed > 0)

    lines = [
        f"Runs tracked: {len(ordered)}  |  Active: {active_runs}  |  Total rounds: {total_rounds}  |  Max rounds: {max_rounds}",
        "Recent runs:",
    ]

    preview = ordered[-5:] if len(ordered) > 5 else ordered
    for rp in preview:
        lines.append(f"  {rp.name:<16} -> rounds: {rp.round_label}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor Glimpse ABM run progress without touching active writers.")
    parser.add_argument("results_dir", type=Path, help="Path to the simulation results directory.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only consider the first N run directories (sorted by index).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=15.0,
        help="Polling interval in seconds for continuous monitoring.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Collect a single snapshot and exit.",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Results directory not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    while True:
        progress = collect_progress(args.results_dir, limit=args.limit)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}]")
        print(render_report(progress))
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
