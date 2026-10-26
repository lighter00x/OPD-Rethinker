#!/usr/bin/env python3
"""Verify that every GPU advertised by Ray can initialize CUDA."""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter, defaultdict

import ray


def parse_expected_layout(value: str | None) -> dict[str, int] | None:
    if value is None:
        return None
    layout: dict[str, int] = {}
    for item in value.split(","):
        node_ip, separator, gpu_count = item.strip().partition("=")
        if not separator or not node_ip:
            raise argparse.ArgumentTypeError(f"Invalid layout entry {item!r}; expected IP=GPU_COUNT")
        if node_ip in layout:
            raise argparse.ArgumentTypeError(f"Duplicate node IP in layout: {node_ip}")
        try:
            layout[node_ip] = int(gpu_count)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid GPU count in layout entry {item!r}") from exc
        if layout[node_ip] < 0:
            raise argparse.ArgumentTypeError(f"GPU count must be non-negative in layout entry {item!r}")
    return layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpus", type=int, required=True)
    parser.add_argument("--expected-nodes", type=int, required=True)
    parser.add_argument(
        "--expected-layout",
        type=parse_expected_layout,
        help="Exact per-node Ray GPU layout, for example 10.0.0.1=2,10.0.0.2=6,10.0.0.3=4",
    )
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ray.init(address="auto", logging_level="ERROR")
    resources = ray.cluster_resources()
    gpu_count = int(resources.get("GPU", 0))
    alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
    alive_layout = {
        str(node["NodeManagerAddress"]): int(node["Resources"].get("GPU", 0)) for node in alive_nodes
    }
    if gpu_count != args.expected_gpus or len(alive_nodes) != args.expected_nodes:
        print(
            f"Ray resources not ready: nodes={len(alive_nodes)}/{args.expected_nodes}, "
            f"gpus={gpu_count}/{args.expected_gpus}",
            flush=True,
        )
        return 1
    if args.expected_layout is not None:
        if sum(args.expected_layout.values()) != args.expected_gpus or len(args.expected_layout) != args.expected_nodes:
            print("Expected layout is inconsistent with --expected-gpus/--expected-nodes", flush=True)
            return 1
        if alive_layout != args.expected_layout:
            print(f"Ray node GPU layout mismatch: actual={alive_layout}, expected={args.expected_layout}", flush=True)
            return 1

    @ray.remote(num_gpus=1)
    def probe(hold_seconds: float) -> dict[str, object]:
        import torch
        from ray.util import get_node_ip_address

        time.sleep(hold_seconds)
        cuda_ok = torch.cuda.is_available() and torch.cuda.device_count() == 1
        return {
            "node": get_node_ip_address(),
            "visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "cuda_ok": cuda_ok,
            "device_count": torch.cuda.device_count(),
        }

    refs = [probe.remote(args.hold_seconds) for _ in range(args.expected_gpus)]
    results = ray.get(refs, timeout=120)
    counts = Counter(str(result["node"]) for result in results)
    visible_by_node: dict[str, set[str]] = defaultdict(set)
    for result in results:
        visible_by_node[str(result["node"])].add(str(result["visible"]))

    failed = [result for result in results if not result["cuda_ok"]]
    expected_probe_counts = args.expected_layout
    if expected_probe_counts is None:
        expected_per_node = args.expected_gpus // args.expected_nodes
        expected_probe_counts = {node_ip: expected_per_node for node_ip in alive_layout}
    print(f"Ray GPU probe counts: {dict(sorted(counts.items()))}", flush=True)
    print(
        "Ray GPU visibility: "
        + str({node: sorted(devices) for node, devices in sorted(visible_by_node.items())}),
        flush=True,
    )
    if (
        failed
        or dict(counts) != expected_probe_counts
        or any(len(visible_by_node[node_ip]) != expected_count for node_ip, expected_count in expected_probe_counts.items())
    ):
        print(
            f"Ray CUDA probe failed: failures={failed}, counts={dict(counts)}, expected={expected_probe_counts}",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
