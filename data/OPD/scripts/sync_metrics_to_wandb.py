#!/usr/bin/env python3
"""Continuously mirror veRL's JSONL metrics into a resumable W&B run."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-file", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser.parse_args()


def load_state(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"line": 0, "step": 0}
    try:
        state = json.loads(path.read_text())
        return {"line": int(state.get("line", 0)), "step": int(state.get("step", 0))}
    except (OSError, ValueError, TypeError):
        return {"line": 0, "step": 0}


def save_state(path: Path, line_number: int, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"line": line_number, "step": step}) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if not os.getenv("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY is not set")

    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    state = load_state(args.state_file)
    print(f"Watching {args.metrics_file}", flush=True)
    while not stopping:
        if not args.metrics_file.exists():
            time.sleep(args.poll_seconds)
            continue

        lines = args.metrics_file.read_bytes().splitlines()
        pending: list[tuple[int, int, dict]] = []
        for line_number, raw_line in enumerate(lines, start=1):
            if line_number <= state["line"] or not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
                pending.append((line_number, int(row["step"]), row["data"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # A final partial line can be observed while the trainer is writing.
                if line_number == len(lines):
                    break
                raise

        if pending:
            run = wandb.init(
                project=args.project,
                name=args.run_name,
                id=args.run_id,
                resume="allow",
                config={
                    "source": "verl_jsonl_mirror",
                    "source_metrics_file": str(args.metrics_file),
                },
            )
            print(f"Uploading {len(pending)} row(s) through {run.url}", flush=True)
            last_line = state["line"]
            last_step = state["step"]
            for line_number, step, data in pending:
                if step > last_step:
                    wandb.log(data, step=step, commit=True)
                    last_step = step
                last_line = line_number
            # Self-hosted W&B may not expose history until the SDK run is flushed.
            wandb.finish()
            state = {"line": last_line, "step": last_step}
            save_state(args.state_file, state["line"], state["step"])

        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
