#!/usr/bin/env python3
"""Download and verify FutureHouse benchmark datasets."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
from huggingface_hub import get_token, snapshot_download


@dataclass(frozen=True)
class FutureHouseDataset:
    name: str
    repo_id: str
    gated: bool = False


DATASETS = (
    FutureHouseDataset("hle_bio_chem_gold", "futurehouse/hle-gold-bio-chem", gated=True),
    FutureHouseDataset("lab_bench", "futurehouse/lab-bench"),
    FutureHouseDataset("bixbench", "futurehouse/BixBench"),
    FutureHouseDataset("ether0", "futurehouse/ether0-benchmark"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".data/huggingface"),
        help="Datasets cache directory.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path(".data/hf_snapshots"),
        help="Directory for full Hugging Face dataset snapshots.",
    )
    parser.add_argument(
        "--skip-snapshots",
        action="store_true",
        help="Only verify datasets through the datasets library; do not download full repo snapshots.",
    )
    return parser.parse_args()


def verify_dataset(dataset: FutureHouseDataset, cache_dir: Path) -> None:
    configs = get_dataset_config_names(dataset.repo_id, cache_dir=str(cache_dir))
    print(f"{dataset.name}: configs={configs}", flush=True)
    for config in configs:
        splits = get_dataset_split_names(dataset.repo_id, config, cache_dir=str(cache_dir))
        print(f"  {config}: splits={splits}", flush=True)
        for split in splits:
            loaded = load_dataset(dataset.repo_id, config, split=split, cache_dir=str(cache_dir))
            print(
                f"    {config}/{split}: rows={len(loaded)} columns={loaded.column_names}",
                flush=True,
            )


def download_snapshot(dataset: FutureHouseDataset, snapshot_dir: Path) -> None:
    local_dir = snapshot_dir / dataset.repo_id.replace("/", "__")
    path = snapshot_download(
        repo_id=dataset.repo_id,
        repo_type="dataset",
        local_dir=local_dir,
    )
    print(f"  snapshot: {path}", flush=True)


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir.resolve()
    snapshot_dir = args.snapshot_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    token_present = bool(get_token())
    failures = []

    for dataset in DATASETS:
        print(f"=== {dataset.name} ({dataset.repo_id}) ===", flush=True)
        if dataset.gated and not token_present:
            failures.append((dataset.name, "missing Hugging Face token for gated dataset"))
            print("FAILED: missing Hugging Face token for gated dataset", flush=True)
            continue
        try:
            verify_dataset(dataset, cache_dir)
            if not args.skip_snapshots:
                download_snapshot(dataset, snapshot_dir)
        except Exception as exc:  # noqa: BLE001 - preparation script should report every dataset.
            failures.append((dataset.name, f"{type(exc).__name__}: {exc}"))
            print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)

    if failures:
        print("\nFailures:", flush=True)
        for name, reason in failures:
            print(f"  {name}: {reason}", flush=True)
        return 1

    print("\nAll FutureHouse datasets verified.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
