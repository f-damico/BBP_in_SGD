#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np


def npz_to_dict(npz_path: Path) -> Dict[str, Any]:
    """
    Load all arrays from an .npz into a Python dict.
    """
    out: Dict[str, Any] = {}
    with np.load(npz_path, allow_pickle=True) as data:
        for key in data.files:
            value = data[key]

            # Convert 0-d arrays to Python scalars
            if isinstance(value, np.ndarray) and value.shape == ():
                out[key] = value.item()
            else:
                out[key] = value
    return out


def collect_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    all_results: Dict[str, Dict[str, Any]] = {}

    run_spec_paths = sorted(results_dir.rglob("run_spec.json"))

    for run_spec_path in run_spec_paths:
        run_dir = run_spec_path.parent
        run_id = run_dir.name
        out_npz = run_dir / "output.npz"

        # Skip incomplete runs
        if not out_npz.exists():
            continue

        with open(run_spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)

        npz_data = npz_to_dict(out_npz)

        entry = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "seed": spec.get("seed"),
            "architecture": spec.get("architecture", {}),
            "dataset": spec.get("dataset", {}),
            "train": spec.get("train", {}),
            "source_config": spec.get("source_config"),
            "npz": npz_data,
        }

        all_results[run_id] = entry

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory of one experiment batch, e.g. results/phase_diagram/paper_exact_repro_v1",
    )
    parser.add_argument(
        "--output_pickle",
        type=str,
        default=None,
        help="Optional path to save the collected dict as a pickle file.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    all_results = collect_results(results_dir)

    print(f"[INFO] collected {len(all_results)} completed runs from {results_dir}")

    if args.output_pickle is not None:
        out_path = Path(args.output_pickle).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(all_results, f)
        print(f"[INFO] saved pickle to {out_path}")


if __name__ == "__main__":
    main()