#!/usr/bin/env python3

import argparse
import importlib.util
import os
import subprocess
import sys

DEFAULT_PACKAGES = ("latex2sympy2_extended", "math_verify")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install GRPO reward dependencies on every live Ray node.")
    parser.add_argument("--expected-nodes", type=int, default=3)
    parser.add_argument("--packages", nargs="+", default=list(DEFAULT_PACKAGES))
    return parser.parse_args()


def install_on_node(packages: list[str]) -> dict:
    import socket

    missing = [package for package in packages if importlib.util.find_spec(package) is None]
    if missing:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *missing],
            check=True,
        )
    unresolved = [package for package in packages if importlib.util.find_spec(package) is None]
    return {"host": socket.gethostname(), "installed": missing, "unresolved": unresolved}


def main() -> None:
    args = parse_args()
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), logging_level="ERROR")
    nodes = [node for node in ray.nodes() if node["Alive"]]
    if len(nodes) != args.expected_nodes:
        raise RuntimeError(f"Expected {args.expected_nodes} Ray nodes, found {len(nodes)}")

    remote_install = ray.remote(num_cpus=1)(install_on_node)
    futures = [
        remote_install.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
        ).remote(args.packages)
        for node in nodes
    ]
    failures = []
    for result in ray.get(futures):
        print(f"{result['host']}: installed={result['installed']} unresolved={result['unresolved']}")
        if result["unresolved"]:
            failures.append(f"{result['host']}: {result['unresolved']}")
    if failures:
        raise RuntimeError("Dependency installation failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
