#!/usr/bin/env python3
"""Watch progress by counting rounds in a run_log.jsonl file."""

import argparse
import json
import time
from pathlib import Path


def count_rounds(path: Path) -> int:
    try:
        with path.open('r') as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch simulation progress")
    parser.add_argument("log_path", type=Path, help="Path to run_log.jsonl")
    parser.add_argument("--interval", type=float, default=10.0, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Just print once and exit")
    args = parser.parse_args()

    while True:
        rounds = count_rounds(args.log_path)
        if rounds >= 0:
            print(f"[{time.strftime('%H:%M:%S')}] rounds logged = {rounds}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] file not found")
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
