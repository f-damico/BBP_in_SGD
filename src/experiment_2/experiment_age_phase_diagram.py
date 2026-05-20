#!/usr/bin/env python3
"""
Experiment-2 driver: one call == one hyperparameter combination == one GPU.

Key behavior:
- A single invocation selects exactly one hyperparameter combination.
- Within that selected combination, all seeds are run sequentially on the same GPU.
- Each seed creates a run folder containing:
    - run_spec.json
    - train_stdout_stderr.txt
    - metrics_log.jsonl
    - output.npz
    - svd_diagnostics.pt (optional, controlled by flag)
- A master launch log is appended immediately at launch time.
- Supports resume and dry-run.

Typical usage from PBS:
    python -u src/phase_diagram/experiment_2/experiment_age_phase_diagram.py \
        --config src/phase_diagram/experiment_2/utkface_age_phase_diagram.json \
        --output_dir results/phase_diagram/experiment_2 \
        --train_script src/training/train_age_regression.py \
        --combo_index ${PBS_ARRAY_INDEX}
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_config(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    if path.suffix.lower() in [".yml", ".yaml"]:
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Either install pyyaml or use JSON."
            ) from e
        return yaml.safe_load(path.read_text())
    raise ValueError(f"Unsupported config extension: {path.suffix}")


def _as_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    return [x]


def _cartesian_product(grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    keys = sorted(grid.keys())
    values_lists = [_as_list(grid[k]) for k in keys]
    combos: List[Dict[str, Any]] = []
    for values in itertools.product(*values_lists):
        combos.append({k: v for k, v in zip(keys, values)})
    return combos


def _build_hyperparameter_combos(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Two valid config styles.

    Style 1: full Cartesian grid, as before
        "grid": {
            "lr": [...],
            "inv_sigma_w": [...],
            "batch_size": [128],
            ...
        }

    Style 2: explicit coupled points
        "grid": {
            "batch_size": [128],
            "epochs": [300],
            ...
        },
        "coupled_grid": [
            {"lr": lr1, "inv_sigma_w": inv1},
            {"lr": lr2, "inv_sigma_w": inv2}
        ]

    In style 2, the final combos are:
        CartesianProduct(grid) x coupled_grid
    """

    grid = cfg.get("grid", {})
    if not grid:
        raise ValueError("Config must contain non-empty 'grid' dict with hyperparameter lists/scalars.")

    base_combos = _cartesian_product(grid)

    coupled_grid = cfg.get("coupled_grid", None)

    # Old behavior: no coupled_grid -> full Cartesian grid
    if coupled_grid is None:
        return base_combos

    # New behavior: exact coupled points
    if not isinstance(coupled_grid, list) or len(coupled_grid) == 0:
        raise ValueError("'coupled_grid' must be a non-empty list of dictionaries.")

    for i, point in enumerate(coupled_grid):
        if not isinstance(point, dict):
            raise ValueError(f"coupled_grid[{i}] must be a dictionary.")
        if len(point) == 0:
            raise ValueError(f"coupled_grid[{i}] cannot be empty.")

    coupled_keys = set()
    for point in coupled_grid:
        coupled_keys.update(point.keys())

    overlap = coupled_keys.intersection(set(grid.keys()))
    if overlap:
        raise ValueError(
            "A parameter cannot appear both in 'grid' and in 'coupled_grid'. "
            f"Move these keys only to one place: {sorted(overlap)}"
        )

    combos: List[Dict[str, Any]] = []
    for base in base_combos:
        for point in coupled_grid:
            merged = dict(base)
            merged.update(dict(point))
            combos.append(merged)

    return combos


def _stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _now_str() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_launch_line(f, line: str) -> None:
    f.write(line + "\n")
    f.flush()
    os.fsync(f.fileno())


def _select_combo(combos: List[Dict[str, Any]], combo_index: int | None) -> tuple[int, Dict[str, Any]]:
    if combo_index is None:
        if len(combos) == 1:
            return 0, combos[0]
        raise ValueError(
            "Config contains multiple hyperparameter combinations. Pass --combo_index <PBS_ARRAY_INDEX>."
        )
    if combo_index < 0 or combo_index >= len(combos):
        raise IndexError(
            f"combo_index={combo_index} out of range for {len(combos)} hyperparameter combinations."
        )
    return combo_index, combos[combo_index]


def _select_combo_and_seed(
    combos: List[Dict[str, Any]],
    seeds: List[int],
    array_index: int,
) -> tuple[int, Dict[str, Any], int, int]:
    n_combos = len(combos)
    n_seeds = len(seeds)
    n_total = n_combos * n_seeds

    if array_index < 0 or array_index >= n_total:
        raise IndexError(
            f"array_index={array_index} out of range for "
            f"n_combos={n_combos}, n_seeds={n_seeds}, total={n_total}."
        )

    combo_index = array_index // n_seeds
    seed_index = array_index % n_seeds
    seed = int(seeds[seed_index])

    return combo_index, combos[combo_index], seed_index, seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to config (json/yaml).")
    p.add_argument(
        "--train_script",
        type=str,
        default="src/training/train_age_regression.py",
        help="Path to the training script to launch.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base output directory where run folders will be created.",
    )
    p.add_argument(
        "--master_log",
        type=str,
        default=None,
        help="Master launch log txt path. Default: <output_dir>/launch_log.txt",
    )
    p.add_argument(
        "--combo_index",
        type=int,
        default=None,
        help="Index inside the cartesian product of config['grid']; intended for PBS_ARRAY_INDEX.",
    )
    p.add_argument(
        "--array_index",
        type=int,
        default=None,
        help=(
            "Flattened PBS array index over (hyperparameter combo, seed). "
            "Use this when PBS -J has length n_combos * n_seeds."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="If set, skip runs whose output.npz already exists.",
    )
    p.add_argument("--dry_run", action="store_true", help="Print planned runs but do not execute.")
    p.add_argument(
        "--save_svd_diagnostics",
        action="store_true",
        help="If set, enable SVD diagnostics at eval epochs for each run.",
    )
    p.add_argument(
        "--svd_diag_filename",
        type=str,
        default="svd_diagnostics.pt",
        help="Filename used inside each run directory for the saved SVD diagnostics payload.",
    )
    p.add_argument(
        "--save_weight_checkpoints",
        action="store_true",
        help="If set, save model weights at every validation epoch for each run.",
    )
    p.add_argument(
        "--weight_checkpoint_filename",
        type=str,
        default="weight_checkpoints.pt",
        help="Filename inside each run directory for validation-time model weights.",
    )
    p.add_argument(
        "--weight_checkpoint_dtype",
        type=str,
        default="float32",
        help="Dtype for saved floating tensors: original, float32, float16, bfloat16.",
    )
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = _load_config(cfg_path)

    out_base = Path(args.output_dir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    master_log_path = Path(args.master_log).resolve() if args.master_log else (out_base / "launch_log.txt")

    architecture = cfg.get("architecture", {})
    dataset = cfg.get("dataset", {})
    seeds = [int(s) for s in _as_list(cfg.get("seeds", []))]
    if not seeds:
        raise ValueError("Config must contain non-empty 'seeds' list.")

    combos = _build_hyperparameter_combos(cfg)

    if args.array_index is not None:
        if args.combo_index is not None:
            raise ValueError("Pass only one of --array_index or --combo_index, not both.")

        selected_combo_index, selected_combo, selected_seed_index, selected_seed = _select_combo_and_seed(
            combos=combos,
            seeds=seeds,
            array_index=int(args.array_index),
        )
        selected_seeds = [selected_seed]
    else:
        selected_combo_index, selected_combo = _select_combo(combos, args.combo_index)
        selected_seed_index = None
        selected_seeds = list(seeds)

    selected_combo = dict(selected_combo)

    if args.save_svd_diagnostics:
        selected_combo["save_svd_diagnostics"] = True
        selected_combo["svd_diag_filename"] = str(args.svd_diag_filename)
    elif "save_svd_diagnostics" in selected_combo and bool(selected_combo["save_svd_diagnostics"]):
        selected_combo["svd_diag_filename"] = str(selected_combo.get("svd_diag_filename", args.svd_diag_filename))

    if args.save_weight_checkpoints:
        selected_combo["save_weight_checkpoints"] = True
        selected_combo["weight_checkpoint_filename"] = str(args.weight_checkpoint_filename)
        selected_combo["weight_checkpoint_dtype"] = str(args.weight_checkpoint_dtype)
    elif "save_weight_checkpoints" in selected_combo and bool(selected_combo["save_weight_checkpoints"]):
        selected_combo["weight_checkpoint_filename"] = str(
            selected_combo.get("weight_checkpoint_filename", args.weight_checkpoint_filename)
        )
        selected_combo["weight_checkpoint_dtype"] = str(
            selected_combo.get("weight_checkpoint_dtype", args.weight_checkpoint_dtype)
        )

    if "inv_sigma_w" in selected_combo:
        if "sigma2_w" in selected_combo:
            raise ValueError("Use only one of 'inv_sigma_w' or 'sigma2_w' in the config, not both.")
        inv_sigma_w = float(selected_combo["inv_sigma_w"])
        if inv_sigma_w <= 0.0:
            raise ValueError("inv_sigma_w must be strictly positive.")
        selected_combo["sigma2_w"] = 1.0 / (inv_sigma_w ** 2)

    combo_hash = _stable_hash(selected_combo)

    combo_dir = out_base / f"combo_{selected_combo_index:04d}_{combo_hash}"
    combo_dir.mkdir(parents=True, exist_ok=True)

    train_script = Path(args.train_script).resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    with open(master_log_path, "a", buffering=1, encoding="utf-8") as mlog:
        header = (
            f"# --- DRIVER START {_now_str()} "
            f"config={cfg_path} combo_index={selected_combo_index} combo={selected_combo} "
            f"train_script={train_script} ---"
        )
        _write_launch_line(mlog, header)

        if args.dry_run:
            _write_launch_line(
                mlog,
                f"# DRY RUN: would execute {len(seeds)} seeds for combo_index={selected_combo_index} combo={selected_combo}.",
            )
            for seed in selected_seeds:
                spec = {
                    "seed": seed,
                    "architecture": architecture,
                    "dataset": dataset,
                    "train": selected_combo,
                    "source_config": str(cfg_path),
                    "combo_index": selected_combo_index,
                    "combo_hash": combo_hash,
                }
                run_hash = _stable_hash(spec)
                run_id = f"combo{selected_combo_index:04d}_seed{seed:04d}_{run_hash}"
                _write_launch_line(mlog, f"[DRY] launch run_id={run_id} seed={seed} train={selected_combo}")
            return

        _write_launch_line(
            mlog,
            f"# Will execute {len(selected_seeds)} seed(s) for combo_index={selected_combo_index} combo={selected_combo}."
        )

        for seed in selected_seeds:
            spec = {
                "seed": seed,
                "architecture": architecture,
                "dataset": dataset,
                "train": selected_combo,
                "source_config": str(cfg_path),
                "combo_index": selected_combo_index,
                "combo_hash": combo_hash,
            }

            run_hash = _stable_hash(spec)
            run_id = f"combo{selected_combo_index:04d}_seed{seed:04d}_{run_hash}"
            run_dir = combo_dir / f"seed_{seed:04d}_{run_hash}"
            run_dir.mkdir(parents=True, exist_ok=True)

            run_spec_path = run_dir / "run_spec.json"
            run_spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")

            out_npz = run_dir / "output.npz"
            train_log = run_dir / "train_stdout_stderr.txt"

            if args.resume and out_npz.exists():
                _write_launch_line(
                    mlog,
                    f"[SKIP] {run_id} (output exists) seed={seed} train={selected_combo}",
                )
                continue

            _write_launch_line(
                mlog,
                f"[LAUNCH] {run_id} seed={seed} train={selected_combo} run_spec={run_spec_path}",
            )

            cmd = [
                sys.executable,
                "-u",
                str(train_script),
                "--run_spec",
                str(run_spec_path),
                "--output_npz",
                str(out_npz),
            ]

            with open(train_log, "w", buffering=1, encoding="utf-8") as tlog:
                tlog.write(f"# CMD: {' '.join(cmd)}\n")
                tlog.flush()

                proc = subprocess.run(
                    cmd,
                    stdout=tlog,
                    stderr=subprocess.STDOUT,
                    cwd=str(Path.cwd()),
                    env=os.environ.copy(),
                )

            if proc.returncode != 0:
                _write_launch_line(
                    mlog,
                    f"[FAIL] {run_id} returncode={proc.returncode} (see {train_log})",
                )
            else:
                _write_launch_line(
                    mlog,
                    f"[DONE] {run_id} (output={out_npz})",
                )

        _write_launch_line(mlog, f"# --- DRIVER END {_now_str()} combo_index={selected_combo_index} ---")


if __name__ == "__main__":
    main()
