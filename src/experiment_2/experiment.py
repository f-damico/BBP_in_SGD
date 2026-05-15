#!/usr/bin/env python3
"""
Experiment driver: one call == one seed == one GPU.

Key behaviors:
- A single invocation selects exactly one seed.
- Within that selected seed, all hyperparameter combinations are run sequentially.
- Each run creates a folder containing:
    - run_spec.json
    - train_stdout_stderr.txt
    - output.npz
- A master launch log is appended immediately at launch time.
- Supports resume and dry-run.

Typical usage from PBS:
    python -u src/phase_diagram/experiment.py \
        --config src/phase_diagram/grid_phase_diagram.json \
        --output_dir results/phase_diagram \
        --train_script src/training/train_teacher_student.py \
        --seed_index ${PBS_ARRAY_INDEX}
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
    """
    grid: dict of param_name -> value OR list of values
    returns list of dicts, one per combination
    """
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


def _select_seed(seeds: List[Any], seed_index: int | None, seed: int | None) -> int:
    seeds_int = [int(s) for s in seeds]

    if seed is not None and seed_index is not None:
        raise ValueError("Pass only one of --seed_index or --seed, not both.")

    if seed is not None:
        seed = int(seed)
        if seed not in seeds_int:
            raise ValueError(f"Requested seed={seed} is not in config seeds={seeds_int}.")
        return seed

    if seed_index is not None:
        if seed_index < 0 or seed_index >= len(seeds_int):
            raise IndexError(
                f"seed_index={seed_index} out of range for seeds list of length {len(seeds_int)}."
            )
        return seeds_int[seed_index]

    if len(seeds_int) == 1:
        return seeds_int[0]

    raise ValueError(
        "Config contains multiple seeds. Pass --seed_index <PBS_ARRAY_INDEX> or --seed <value>."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to config (json/yaml).")
    p.add_argument(
        "--train_script",
        type=str,
        default="src/training/train_teacher_student.py",
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
        "--seed_index",
        type=int,
        default=None,
        help="Index inside config['seeds']; intended for PBS_ARRAY_INDEX.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Explicit seed value; alternative to --seed_index.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="If set, skip runs whose output.npz already exists.",
    )
    p.add_argument(
        "--max_runs",
        type=int,
        default=None,
        help="Optional cap on number of runs executed for this seed.",
    )
    p.add_argument("--dry_run", action="store_true", help="Print planned runs but do not execute.")
    p.add_argument(
        "--save_svd_diagnostics",
        action="store_true",
        help="If set, enable layer SVD/eigenvalue diagnostics at eval steps for each run.",
    )
    p.add_argument(
        "--svd_diag_filename",
        type=str,
        default="svd_diagnostics.pt",
        help="Filename used inside each run directory for the saved SVD diagnostics payload.",
    )
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = _load_config(cfg_path)

    out_base = Path(args.output_dir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    master_log_path = Path(args.master_log).resolve() if args.master_log else (out_base / "launch_log.txt")

    architecture = cfg.get("architecture", {})
    dataset = cfg.get("dataset", {})
    seeds = _as_list(cfg.get("seeds", []))
    if not seeds:
        raise ValueError("Config must contain non-empty 'seeds' list.")

    grid = cfg.get("grid", {})
    if not grid:
        raise ValueError("Config must contain non-empty 'grid' dict with hyperparameter lists/scalars.")

    selected_seed = _select_seed(seeds, args.seed_index, args.seed)

    grid_combos = _cartesian_product(grid)
    if args.max_runs is not None:
        grid_combos = grid_combos[: args.max_runs]

    if args.save_svd_diagnostics:
        for hp in grid_combos:
            hp["save_svd_diagnostics"] = True
            hp["svd_diag_filename"] = str(args.svd_diag_filename)

    seed_dir = out_base / f"seed_{selected_seed:04d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    with open(master_log_path, "a", buffering=1, encoding="utf-8") as mlog:
        header = (
            f"# --- DRIVER START {_now_str()} "
            f"config={cfg_path} seed={selected_seed} "
            f"seed_index={args.seed_index} train_script={args.train_script} ---"
        )
        _write_launch_line(mlog, header)

        if args.dry_run:
            _write_launch_line(mlog, f"# DRY RUN: would execute {len(grid_combos)} runs for seed={selected_seed}.")
            for local_i, hp in enumerate(grid_combos):
                spec = {
                    "seed": selected_seed,
                    "architecture": architecture,
                    "dataset": dataset,
                    "train": hp,
                    "source_config": str(cfg_path),
                }
                rid = f"seed{selected_seed:04d}_i{local_i:06d}_{_stable_hash(spec)}"
                _write_launch_line(
                    mlog,
                    f"[DRY] launch run_id={rid} seed={selected_seed} train={hp}",
                )
            return

        _write_launch_line(mlog, f"# Will execute {len(grid_combos)} runs for seed={selected_seed}.")

        train_script = Path(args.train_script).resolve()
        if not train_script.exists():
            raise FileNotFoundError(f"Training script not found: {train_script}")

        for local_i, hp in enumerate(grid_combos):
            spec = {
                "seed": selected_seed,
                "architecture": architecture,
                "dataset": dataset,
                "train": hp,
                "source_config": str(cfg_path),
            }

            run_hash = _stable_hash(spec)
            run_id = f"seed{selected_seed:04d}_i{local_i:06d}_{run_hash}"
            run_dir = seed_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            run_spec_path = run_dir / "run_spec.json"
            run_spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")

            out_npz = run_dir / "output.npz"
            train_log = run_dir / "train_stdout_stderr.txt"

            if args.resume and out_npz.exists():
                _write_launch_line(
                    mlog,
                    f"[SKIP] {run_id} (output exists) seed={selected_seed} train={hp}",
                )
                continue

            _write_launch_line(
                mlog,
                f"[LAUNCH] {run_id} seed={selected_seed} train={hp} run_spec={run_spec_path}",
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

        _write_launch_line(mlog, f"# --- DRIVER END {_now_str()} seed={selected_seed} ---")


if __name__ == "__main__":
    main()