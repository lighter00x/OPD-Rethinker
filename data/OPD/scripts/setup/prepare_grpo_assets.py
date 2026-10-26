#!/usr/bin/env python3

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL_REPO = "Qwen/Qwen3-4B-Base"
DEFAULT_SHARED_ROOT = Path("/workspace/data")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare shared model and dataset assets for 4B GRPO.")
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED_ROOT)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--smoke-samples", type=int, default=32)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Required file is missing or empty: {path}")


def copy_if_changed(source: Path, destination: Path) -> None:
    require_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and source.stat().st_size == destination.stat().st_size:
        if sha256(source) == sha256(destination):
            print(f"Already current: {destination}")
            return
    shutil.copy2(source, destination)
    print(f"Copied: {source} -> {destination}")


def write_smoke_dataset(source: Path, destination: Path, samples: int) -> None:
    import pyarrow.parquet as parquet

    table = parquet.read_table(source)
    if table.num_rows < samples:
        raise RuntimeError(f"Smoke dataset requests {samples} rows, but {source} contains {table.num_rows}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(table.slice(0, samples), destination)
    print(f"Wrote {samples}-row smoke dataset: {destination}")


def main() -> None:
    args = parse_args()
    from huggingface_hub import HfApi, snapshot_download

    model_dir = args.shared_root / "models" / "Qwen3-4B-Base"
    print(f"Downloading {args.model_repo}@{args.model_revision} to {model_dir}", flush=True)
    snapshot_download(
        repo_id=args.model_repo,
        revision=args.model_revision,
        local_dir=model_dir,
        max_workers=args.download_workers,
    )

    model_info = HfApi().model_info(args.model_repo, revision=args.model_revision)
    require_file(model_dir / "config.json")
    weight_files = sorted(model_dir.glob("*.safetensors"))
    if not weight_files:
        raise RuntimeError(f"No safetensors weights found in {model_dir}")

    dataset_root = args.shared_root / "datasets" / "OPD"
    project_datasets = args.project_root / "datasets"
    asset_sources = {
        "dapo-math-17k-processed.parquet": project_datasets / "dapo-math-17k-processed.parquet",
        "test_data/AIME24/test.parquet": project_datasets / "test_data" / "AIME24" / "test.parquet",
        "test_data/AIME25/test.parquet": project_datasets / "test_data" / "AIME25" / "test.parquet",
        "test_data/AMC23/test.parquet": project_datasets / "test_data" / "AMC23" / "test.parquet",
    }
    for relative_path, source in asset_sources.items():
        copy_if_changed(source, dataset_root / relative_path)

    reward_source = args.project_root / "verl" / "verl" / "utils" / "reward_score" / "ttrl_math"
    reward_root = args.shared_root / "code" / "OPD" / "ttrl_math"
    reward_files = sorted(reward_source.glob("*.py"))
    if not reward_files:
        raise RuntimeError(f"No ttrl_math reward source files found in {reward_source}")
    for source in reward_files:
        copy_if_changed(source, reward_root / source.name)

    smoke_path = dataset_root / f"dapo-math-17k-processed-smoke-{args.smoke_samples}.parquet"
    write_smoke_dataset(dataset_root / "dapo-math-17k-processed.parquet", smoke_path, args.smoke_samples)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "repo_id": args.model_repo,
            "requested_revision": args.model_revision,
            "resolved_revision": model_info.sha,
            "path": str(model_dir),
            "weight_files": [
                {"name": path.name, "size": path.stat().st_size, "sha256": sha256(path)} for path in weight_files
            ],
        },
        "datasets": {
            relative_path: {
                "path": str(dataset_root / relative_path),
                "size": (dataset_root / relative_path).stat().st_size,
                "sha256": sha256(dataset_root / relative_path),
            }
            for relative_path in asset_sources
        },
        "smoke_dataset": {
            "path": str(smoke_path),
            "rows": args.smoke_samples,
            "sha256": sha256(smoke_path),
        },
        "reward_code": {
            path.name: {
                "path": str(reward_root / path.name),
                "sha256": sha256(reward_root / path.name),
            }
            for path in reward_files
        },
    }
    manifest_path = args.shared_root / "manifests" / "grpo_assets.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, ensure_ascii=True)
        output.write("\n")

    print(f"Model ready: {model_dir}")
    print(f"Dataset ready: {dataset_root / 'dapo-math-17k-processed.parquet'}")
    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
