#!/usr/bin/env python3

import argparse
import importlib
import os
import subprocess
import urllib.request
from pathlib import Path

DEFAULT_SHARED_ROOT = Path("/workspace/data")
DEFAULT_WANDB_BASE_URL = "http://10.154.24.41:8082"
REQUIRED_MODULES = ("torch", "ray", "verl", "vllm", "wandb", "latex2sympy2_extended", "math_verify")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate all Ray nodes before OPD 4B GRPO.")
    parser.add_argument("--expected-nodes", type=int, default=3)
    parser.add_argument("--expected-gpus", type=int, default=18)
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED_ROOT)
    parser.add_argument("--wandb-base-url", default=os.environ.get("WANDB_BASE_URL", DEFAULT_WANDB_BASE_URL))
    parser.add_argument("--login-wandb", action="store_true")
    return parser.parse_args()


def check_node(required_paths: list[str], wandb_base_url: str, wandb_api_key: str | None) -> dict:
    import socket

    versions = {}
    missing_modules = []
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = getattr(module, "__version__", "installed")
        except Exception as exc:
            missing_modules.append(f"{module_name}: {type(exc).__name__}: {exc}")

    rollout_import = "ok"
    try:
        from verl.workers.rollout.base import get_rollout_class

        get_rollout_class("vllm", "async")
    except Exception as exc:
        rollout_import = f"{type(exc).__name__}: {exc}"

    try:
        with urllib.request.urlopen(wandb_base_url, timeout=10) as response:
            endpoint_status = response.status
    except Exception as exc:
        endpoint_status = f"{type(exc).__name__}: {exc}"

    wandb_login = "not-requested"
    if wandb_api_key and "wandb" in versions:
        try:
            import wandb

            wandb.login(key=wandb_api_key, host=wandb_base_url, relogin=True)
            wandb_login = "ok"
        except Exception as exc:
            wandb_login = f"{type(exc).__name__}: {exc}"

    cuda = {"available": False, "device_count": 0, "device_name": None}
    if "torch" in versions:
        import torch

        cuda["available"] = torch.cuda.is_available()
        cuda["device_count"] = torch.cuda.device_count()
        if cuda["available"] and cuda["device_count"]:
            cuda["device_name"] = torch.cuda.get_device_name(0)

    nvidia_smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )

    return {
        "host": socket.gethostname(),
        "versions": versions,
        "missing_modules": missing_modules,
        "rollout_import": rollout_import,
        "wandb_endpoint": endpoint_status,
        "wandb_login": wandb_login,
        "node_ip": os.environ.get("RAY_NODE_IP_ADDRESS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "cuda": cuda,
        "nvidia_smi": nvidia_smi.stdout.strip() if nvidia_smi.returncode == 0 else nvidia_smi.stderr.strip(),
        "nvidia_smi_returncode": nvidia_smi.returncode,
        "missing_paths": [path for path in required_paths if not Path(path).is_file()],
    }


def main() -> None:
    args = parse_args()
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), logging_level="ERROR")
    nodes = [node for node in ray.nodes() if node["Alive"]]
    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    if len(nodes) != args.expected_nodes:
        raise RuntimeError(f"Expected {args.expected_nodes} Ray nodes, found {len(nodes)}")
    if total_gpus != args.expected_gpus:
        raise RuntimeError(f"Expected {args.expected_gpus} GPUs, found {total_gpus}")

    required_paths = [
        args.shared_root / "models" / "Qwen3-4B-Base" / "config.json",
        args.shared_root / "datasets" / "OPD" / "dapo-math-17k-processed.parquet",
        args.shared_root / "datasets" / "OPD" / "dapo-math-17k-processed-smoke-32.parquet",
        args.shared_root / "datasets" / "OPD" / "test_data" / "AIME24" / "test.parquet",
        args.shared_root / "datasets" / "OPD" / "test_data" / "AIME25" / "test.parquet",
        args.shared_root / "datasets" / "OPD" / "test_data" / "AMC23" / "test.parquet",
        args.shared_root / "manifests" / "grpo_assets.json",
        args.shared_root / "code" / "OPD" / "ttrl_math" / "__init__.py",
        args.shared_root / "code" / "OPD" / "ttrl_math" / "math_utils.py",
    ]
    weight_files = sorted((args.shared_root / "models" / "Qwen3-4B-Base").glob("*.safetensors"))
    if not weight_files:
        raise RuntimeError("No Qwen3-4B-Base safetensors files are visible on the head node")
    required_paths.extend(weight_files)

    wandb_api_key = os.environ.get("WANDB_API_KEY") if args.login_wandb else None
    if args.login_wandb and not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY must be set when --login-wandb is used")

    # A declared Ray GPU is not enough: request one GPU per node so CUDA/NVML mapping is exercised.
    remote_check = ray.remote(num_cpus=1, num_gpus=1)(check_node)
    futures = [
        remote_check.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
        ).remote([str(path) for path in required_paths], args.wandb_base_url, wandb_api_key)
        for node in nodes
    ]
    failures = []
    for result in ray.get(futures):
        print(
            f"{result['host']}({result['node_ip']}): versions={result['versions']} "
            f"rollout_import={result['rollout_import']} "
            f"cuda={result['cuda']} CUDA_VISIBLE_DEVICES={result['cuda_visible_devices']} "
            f"NVIDIA_VISIBLE_DEVICES={result['nvidia_visible_devices']} endpoint={result['wandb_endpoint']} "
            f"wandb_login={result['wandb_login']} missing_paths={result['missing_paths']}"
        )
        if result["missing_modules"]:
            failures.append(f"Missing modules on {result['host']}: {result['missing_modules']}")
        if result["rollout_import"] != "ok":
            failures.append(f"vLLM rollout import failed on {result['host']}: {result['rollout_import']}")
        if not result["cuda"]["available"] or result["cuda"]["device_count"] != 1:
            failures.append(
                f"Ray GPU mapping failed on {result['host']}({result['node_ip']}): "
                f"cuda={result['cuda']} CUDA_VISIBLE_DEVICES={result['cuda_visible_devices']} "
                f"NVIDIA_VISIBLE_DEVICES={result['nvidia_visible_devices']} "
                f"nvidia-smi={result['nvidia_smi']!r}"
            )
        if result["nvidia_smi_returncode"] != 0:
            failures.append(f"nvidia-smi failed on {result['host']}({result['node_ip']}): {result['nvidia_smi']}")
        if result["wandb_endpoint"] != 200:
            failures.append(f"W&B is unreachable from {result['host']}: {result['wandb_endpoint']}")
        if args.login_wandb and result["wandb_login"] != "ok":
            failures.append(f"W&B login failed on {result['host']}: {result['wandb_login']}")
        if result["missing_paths"]:
            failures.append(f"Assets missing on {result['host']}: {result['missing_paths']}")

    if failures:
        raise RuntimeError("\n".join(failures))
    print(f"GRPO environment ready: {len(nodes)} nodes, {total_gpus} GPUs")


if __name__ == "__main__":
    main()
