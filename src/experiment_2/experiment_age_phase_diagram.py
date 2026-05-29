#!/usr/bin/env python3
"""
Experiment-2 driver for the OLD/CANONICAL JSON schema.

Canonical input config schema:

{
  "architecture": {...},
  "dataset": {...},
  "seeds": [1, 2, ...],
  "grid": {
    "optimizer": ["sgd"],
    "grad_clip_norm": [10.0],
    "batch_size": [128],
    "epochs": [1000],
    "momentum": [0.0],
    "weight_decay": [0.0],
    "eval_epochs": [[1, 2, 5, ...]]
  },
  "coupled_grid": [
    {"lr": 0.8096, "inv_sigma_w": 8096.0},
    {"lr": 0.0128, "inv_sigma_w": 8.0}
  ]
}

Important behaviour:
- No "model", "training", "initialization", or "output" blocks are required.
- The run hyperparameters are built as CartesianProduct(grid) x coupled_grid.
- If coupled_grid is absent, the run hyperparameters are CartesianProduct(grid).
- A PBS array index is flattened as: array_index = combo_index * n_seeds + seed_index.
- Each array job launches exactly one hyperparameter combo and one seed.
- Each run folder contains run_spec.json, train_stdout_stderr.txt, metrics_log.jsonl, output.npz.
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
from typing import Any, Dict, List, Tuple


def _load_config(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("YAML config requested but PyYAML is not installed. Use JSON or install pyyaml.") from exc
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if loaded is None:
            raise ValueError(f"Empty config file: {path}")
        return dict(loaded)
    raise ValueError(f"Unsupported config extension: {path.suffix}")


def _as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else [x]


def _cartesian_product(grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(grid, dict) or len(grid) == 0:
        raise ValueError("Config must contain a non-empty old-style 'grid' dictionary.")

    keys = sorted(grid.keys())
    value_lists = [_as_list(grid[k]) for k in keys]
    combos: List[Dict[str, Any]] = []
    for values in itertools.product(*value_lists):
        combos.append({k: v for k, v in zip(keys, values)})
    return combos


def _build_hyperparameter_combos(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Old canonical behaviour:
      final_combos = CartesianProduct(cfg['grid'])                         if no coupled_grid
      final_combos = CartesianProduct(cfg['grid']) x cfg['coupled_grid']   otherwise

    Parameters must not appear both in grid and coupled_grid. This prevents silent overwrites.
    """
    if "grid" not in cfg:
        raise ValueError("Old-style config must contain key 'grid'.")
    base_combos = _cartesian_product(dict(cfg["grid"]))

    coupled_grid = cfg.get("coupled_grid", None)
    if coupled_grid is None:
        return base_combos

    if not isinstance(coupled_grid, list) or len(coupled_grid) == 0:
        raise ValueError("If provided, 'coupled_grid' must be a non-empty list of dictionaries.")

    coupled_keys = set()
    for i, point in enumerate(coupled_grid):
        if not isinstance(point, dict):
            raise ValueError(f"coupled_grid[{i}] must be a dictionary.")
        if len(point) == 0:
            raise ValueError(f"coupled_grid[{i}] cannot be empty.")
        coupled_keys.update(point.keys())

    overlap = set(cfg["grid"].keys()).intersection(coupled_keys)
    if overlap:
        raise ValueError(
            "A parameter cannot appear both in old-style 'grid' and 'coupled_grid'. "
            f"Move these keys to only one place: {sorted(overlap)}"
        )

    combos: List[Dict[str, Any]] = []
    for base in base_combos:
        for point in coupled_grid:
            merged = dict(base)
            merged.update(dict(point))
            combos.append(merged)
    return combos


def _stable_hash(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _now_str() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_line(handle, text: str) -> None:
    handle.write(text + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _normalise_selected_combo(combo: Dict[str, Any]) -> Dict[str, Any]:
    """Add sigma2_w from inv_sigma_w while preserving inv_sigma_w in the run spec/output."""
    out = dict(combo)
    if "inv_sigma_w" in out:
        if "sigma2_w" in out:
            raise ValueError("Use only one of 'inv_sigma_w' or 'sigma2_w', not both.")
        inv_sigma_w = float(out["inv_sigma_w"])
        if inv_sigma_w <= 0.0:
            raise ValueError("inv_sigma_w must be strictly positive.")
        out["sigma2_w"] = 1.0 / (inv_sigma_w ** 2)
    elif "sigma2_w" not in out:
        raise ValueError("Each hyperparameter combo must contain either 'inv_sigma_w' or 'sigma2_w'.")
    return out


def _output_npz_is_completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        import numpy as np
        with np.load(path, allow_pickle=True) as data:
            if "completed" not in data.files:
                return True
            return bool(data["completed"].item())
    except Exception:
        # Conservative old behaviour: if the output exists but cannot be read, skip it under --resume.
        return True


def _select_from_flat_array_index(
    *, combos: List[Dict[str, Any]], seeds: List[int], array_index: int
) -> Tuple[int, Dict[str, Any], int, int]:
    n_combos = len(combos)
    n_seeds = len(seeds)
    total = n_combos * n_seeds
    if array_index < 0 or array_index >= total:
        raise IndexError(
            f"array_index={array_index} is out of range. "
            f"n_combos={n_combos}, n_seeds={n_seeds}, total={total}."
        )
    combo_index = array_index // n_seeds
    seed_index = array_index % n_seeds
    return combo_index, combos[combo_index], seed_index, int(seeds[seed_index])


def _select_combo_only(combos: List[Dict[str, Any]], combo_index: int | None) -> Tuple[int, Dict[str, Any]]:
    if combo_index is None:
        if len(combos) == 1:
            return 0, combos[0]
        raise ValueError("Config has multiple combos. Pass --array_index or --combo_index.")
    if combo_index < 0 or combo_index >= len(combos):
        raise IndexError(f"combo_index={combo_index} out of range for {len(combos)} combos.")
    return combo_index, combos[combo_index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Old-schema experiment-2 phase-diagram driver.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_script", type=str, default="src/training/train_age_regression.py")
    parser.add_argument("--master_log", type=str, default=None)
    parser.add_argument("--array_index", type=int, default=None, help="Flattened index over (combo, seed).")
    parser.add_argument("--combo_index", type=int, default=None, help="Run one combo and all seeds; mostly for debugging.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--save_svd_diagnostics", action="store_true")
    parser.add_argument("--svd_diag_filename", type=str, default="svd_diagnostics.pt")
    parser.add_argument("--save_weight_checkpoints", action="store_true")
    parser.add_argument("--weight_checkpoint_filename", type=str, default="weight_checkpoints.pt")
    parser.add_argument("--weight_checkpoint_dtype", type=str, default="float32")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_config(cfg_path)

    if "architecture" not in cfg:
        raise ValueError("Old-style config must contain top-level key 'architecture'.")
    if "dataset" not in cfg:
        raise ValueError("Old-style config must contain top-level key 'dataset'.")
    if "seeds" not in cfg:
        raise ValueError("Old-style config must contain top-level key 'seeds'.")

    architecture = dict(cfg["architecture"])
    dataset = dict(cfg["dataset"])
    seeds = [int(s) for s in _as_list(cfg["seeds"])]
    if not seeds:
        raise ValueError("Config key 'seeds' must be a non-empty list.")

    combos = _build_hyperparameter_combos(cfg)
    if not combos:
        raise ValueError("No hyperparameter combinations were produced from grid/coupled_grid.")

    if args.array_index is not None:
        if args.combo_index is not None:
            raise ValueError("Pass only one of --array_index or --combo_index.")
        combo_index, selected_combo_raw, seed_index, selected_seed = _select_from_flat_array_index(
            combos=combos,
            seeds=seeds,
            array_index=int(args.array_index),
        )
        selected_seeds = [selected_seed]
    else:
        combo_index, selected_combo_raw = _select_combo_only(combos, args.combo_index)
        seed_index = None
        selected_seeds = list(seeds)

    selected_combo = _normalise_selected_combo(selected_combo_raw)

    if args.save_svd_diagnostics:
        selected_combo["save_svd_diagnostics"] = True
        selected_combo["svd_diag_filename"] = str(args.svd_diag_filename)
    elif bool(selected_combo.get("save_svd_diagnostics", False)):
        selected_combo["svd_diag_filename"] = str(selected_combo.get("svd_diag_filename", args.svd_diag_filename))

    if args.save_weight_checkpoints:
        selected_combo["save_weight_checkpoints"] = True
        selected_combo["weight_checkpoint_filename"] = str(args.weight_checkpoint_filename)
        selected_combo["weight_checkpoint_dtype"] = str(args.weight_checkpoint_dtype)
    elif bool(selected_combo.get("save_weight_checkpoints", False)):
        selected_combo["weight_checkpoint_filename"] = str(
            selected_combo.get("weight_checkpoint_filename", args.weight_checkpoint_filename)
        )
        selected_combo["weight_checkpoint_dtype"] = str(
            selected_combo.get("weight_checkpoint_dtype", args.weight_checkpoint_dtype)
        )

    out_base = Path(args.output_dir).expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    master_log_path = Path(args.master_log).expanduser().resolve() if args.master_log else out_base / "launch_log.txt"

    train_script = Path(args.train_script).expanduser().resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    combo_hash = _stable_hash(selected_combo)
    combo_dir = out_base / f"combo_{combo_index:04d}_{combo_hash}"
    combo_dir.mkdir(parents=True, exist_ok=True)

    with master_log_path.open("a", buffering=1, encoding="utf-8") as mlog:
        _write_line(
            mlog,
            f"# --- DRIVER START {_now_str()} config={cfg_path} combo_index={combo_index} "
            f"seed_index={seed_index} combo={selected_combo} train_script={train_script} ---",
        )
        _write_line(
            mlog,
            f"# total_combos={len(combos)} total_seeds={len(seeds)} total_array_jobs={len(combos) * len(seeds)}",
        )
        _write_line(mlog, f"# will execute {len(selected_seeds)} seed(s) for combo_index={combo_index}")

        for seed in selected_seeds:
            run_spec = {
                "seed": int(seed),
                "architecture": architecture,
                "dataset": dataset,
                "train": selected_combo,
                "source_config": str(cfg_path),
                "combo_index": int(combo_index),
                "combo_hash": str(combo_hash),
            }
            run_hash = _stable_hash(run_spec)
            run_id = f"combo{combo_index:04d}_seed{int(seed):04d}_{run_hash}"
            run_dir = combo_dir / f"seed_{int(seed):04d}_{run_hash}"
            run_dir.mkdir(parents=True, exist_ok=True)

            run_spec_path = run_dir / "run_spec.json"
            output_npz = run_dir / "output.npz"
            train_log = run_dir / "train_stdout_stderr.txt"
            run_spec_path.write_text(json.dumps(run_spec, indent=2, sort_keys=True), encoding="utf-8")

            if args.resume and _output_npz_is_completed(output_npz):
                _write_line(mlog, f"[SKIP] {run_id} completed output exists: {output_npz}")
                continue

            cmd = [
                sys.executable,
                "-u",
                str(train_script),
                "--run_spec",
                str(run_spec_path),
                "--output_npz",
                str(output_npz),
            ]
            _write_line(mlog, f"[LAUNCH] {run_id} seed={seed} train={selected_combo} run_spec={run_spec_path}")
            if args.dry_run:
                _write_line(mlog, f"[DRY] {' '.join(cmd)}")
                continue

            with train_log.open("w", buffering=1, encoding="utf-8") as tlog:
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
                _write_line(mlog, f"[FAIL] {run_id} returncode={proc.returncode} see={train_log}")
                raise SystemExit(proc.returncode)
            _write_line(mlog, f"[DONE] {run_id} output={output_npz}")

        _write_line(mlog, f"# --- DRIVER END {_now_str()} combo_index={combo_index} ---")


if __name__ == "__main__":
    main()
