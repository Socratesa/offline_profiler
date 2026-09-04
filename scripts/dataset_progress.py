#!/usr/bin/env python3
"""Show current inference, scoring, MoE-stat, and process status for one run tag."""

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path("/workspace/fd_afd_dev")
EXPECTED = {
    "gsm8k": 1319,
    "math500": 500,
    "arc_challenge": 1172,
    "financebench": 150,
    "humaneval": 164,
    "livecodebench": 1055,
    "asdiv": 2305,
    "aime": 120,
    "mbpp": 374,
    "openbookqa": 4957,
    "finqa": 6251,
}


def load_best(path):
    best = {}
    if not path.exists():
        return best
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            idx = row.get("idx")
            if idx is None:
                continue
            rank = (row.get("error") is None, row.get("finish_reason", "stop") == "stop")
            if idx not in best or rank >= best[idx][0]:
                best[idx] = (rank, row)
    return {idx: value[1] for idx, value in best.items()}


def scored_counts(path):
    if not path.exists():
        return 0, 0
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
    return sum(row.get("correct") is not None for row in rows), len(rows)


def process_state():
    result = subprocess.run(
        ["pgrep", "-af", "run_all_glm_datasets|run_infer.py|api_server.py|worker_process.py"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if "dataset_progress.py" not in line]
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="glm_think_32k_c512_20260812")
    args = parser.parse_args()
    run_dir = ROOT / "exp" / "runs" / args.tag
    log_dir = run_dir / "logs"

    processes = process_state()
    print(f"tag: {args.tag}")
    print("running: " + ("yes" if processes else "no"))
    for line in processes:
        print(f"  {line}")
    print()
    print(f"{'dataset':<16} {'valid':>11} {'error':>7} {'trunc':>7} {'scored':>13} {'stats':>8}")
    for name, expected in EXPECTED.items():
        rows = load_best(run_dir / "output" / f"{name}.jsonl")
        valid = sum(r.get("error") is None and r.get("finish_reason", "stop") == "stop"
                    for r in rows.values())
        errors = sum(r.get("error") is not None for r in rows.values())
        trunc = sum(r.get("error") is None and r.get("finish_reason") == "length"
                    for r in rows.values())
        judged, scored = scored_counts(run_dir / "score" / f"{name}.scored.jsonl")
        stats_path = run_dir / "stats" / f"{name}.npz"
        stats = "yes" if stats_path.exists() and stats_path.stat().st_size > 20000 else "no"
        print(f"{name:<16} {valid:>5}/{expected:<5} {errors:>7} {trunc:>7} "
              f"{judged:>5}/{scored:<5} {stats:>8}")
    status = log_dir / "run_all_status.md"
    print(f"\nstatus: {status}")


if __name__ == "__main__":
    main()
