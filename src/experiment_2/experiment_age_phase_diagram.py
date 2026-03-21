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
        "--resume",
        action="store_true",
        help="If set, skip runs whose output.npz already exists.",
    )
    p.add_argument("--dry_run", action="store_true", help="Print planned runs but do not execute.")
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

    grid = cfg.get("grid", {})
    if not grid:
        raise ValueError("Config must contain non-empty 'grid' dict with hyperparameter lists/scalars.")

    combos = _cartesian_product(grid)
    selected_combo_index, selected_combo = _select_combo(combos, args.combo_index)
    selected_combo = dict(selected_combo)
    
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
            for seed in seeds:
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
            f"# Will execute {len(seeds)} seeds for combo_index={selected_combo_index} combo={selected_combo}.",
        )

        for seed in seeds:
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
