#!/usr/bin/env python3
"""
Experiment driver: sequentially launch Teacher-Student trainings over (seeds x hyperparam grid).

Key behaviors:
- One call == one GPU occupied (sequential runs).
- Each run creates a unique folder containing:
    - run_spec.json
    - train_stdout_stderr.txt  (captured training output)
    - output.npz               (produced by the training script)
- Master launch log is appended *immediately at launch time* (flushed).
- Supports sharding the run list: shard_id / num_shards.

Training script contract (we will implement next):
    python -u training/train_teacher_student.py --run_spec <path/to/run_spec.json> --output_npz <path/to/output.npz>

Config file: JSON (or YAML if PyYAML installed).
See example at bottom of this message.
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
from typing import Any, Dict, Iterable, List, Tuple


def _load_config(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in [".json"]:
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
    combos = []
    for values in itertools.product(*values_lists):
        combos.append({k: v for k, v in zip(keys, values)})
    return combos


def _stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _now_str() -> str:
    # filesystem-friendly timestamp
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_launch_line(f, line: str) -> None:
    f.write(line + "\n")
    f.flush()
    os.fsync(f.fileno())


def _select_shard(items: List[Any], shard_id: int, num_shards: int) -> List[Tuple[int, Any]]:
    """
    Deterministic modulo-based sharding. Returns list of (global_index, item).
    """
    if num_shards <= 1:
        return list(enumerate(items))
    selected = []
    for i, it in enumerate(items):
        if (i % num_shards) == shard_id:
            selected.append((i, it))
    return selected


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to config (json/yaml).")
    p.add_argument(
        "--train_script",
        type=str,
        default="training/train_teacher_student.py",
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
    p.add_argument("--shard_id", type=int, default=0, help="Shard index (0-based).")
    p.add_argument("--num_shards", type=int, default=1, help="Total number of shards.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="If set, skip runs whose output.npz already exists.",
    )
    p.add_argument(
        "--max_runs",
        type=int,
        default=None,
        help="Optional cap on number of runs executed in this driver.",
    )
    p.add_argument("--dry_run", action="store_true", help="Print planned runs but do not execute.")
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = _load_config(cfg_path)

    out_base = Path(args.output_dir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    master_log_path = Path(args.master_log).resolve() if args.master_log else (out_base / "launch_log.txt")

    # Required config sections
    architecture = cfg.get("architecture", {})
    dataset = cfg.get("dataset", {})
    seeds = _as_list(cfg.get("seeds", []))
    if not seeds:
        raise ValueError("Config must contain non-empty 'seeds' list.")
    grid = cfg.get("grid", {})
    if not grid:
        raise ValueError("Config must contain non-empty 'grid' dict with hyperparameter lists/scalars.")

    # Build runs = seeds x hyperparam grid
    grid_combos = _cartesian_product(grid)
    all_runs: List[Dict[str, Any]] = []
    for seed in seeds:
        for hp in grid_combos:
            run_spec = {
                "seed": int(seed),
                "architecture": architecture,
                "dataset": dataset,
                "train": hp,
                # keep a pointer to the config used
                "source_config": str(cfg_path),
            }
            all_runs.append(run_spec)

    shard_runs = _select_shard(all_runs, args.shard_id, args.num_shards)
    if args.max_runs is not None:
        shard_runs = shard_runs[: args.max_runs]

    # Open master log in line-buffered mode
    with open(master_log_path, "a", buffering=1, encoding="utf-8") as mlog:
        header = (
            f"# --- DRIVER START { _now_str() } "
            f"config={cfg_path} shard={args.shard_id}/{args.num_shards} "
            f"train_script={args.train_script} ---"
        )
        _write_launch_line(mlog, header)

        if args.dry_run:
            _write_launch_line(mlog, f"# DRY RUN: would execute {len(shard_runs)} runs.")
            for global_i, spec in shard_runs:
                rid = f"s{args.shard_id}of{args.num_shards}_i{global_i:06d}_{_stable_hash(spec)}"
                _write_launch_line(
                    mlog,
                    f"[DRY] launch run_id={rid} seed={spec['seed']} train={spec['train']}",
                )
            return

        _write_launch_line(mlog, f"# Will execute {len(shard_runs)} runs in this shard.")

        train_script = Path(args.train_script).resolve()
        if not train_script.exists():
            raise FileNotFoundError(f"Training script not found: {train_script}")

        for global_i, spec in shard_runs:
            run_hash = _stable_hash(spec)
            run_id = f"{_now_str()}_s{args.shard_id}of{args.num_shards}_i{global_i:06d}_{run_hash}"
            run_dir = out_base / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            run_spec_path = run_dir / "run_spec.json"
            run_spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")

            out_npz = run_dir / "output.npz"
            train_log = run_dir / "train_stdout_stderr.txt"

            # Resume behavior
            if args.resume and out_npz.exists():
                _write_launch_line(
                    mlog,
                    f"[SKIP] {run_id} (output exists) seed={spec['seed']} train={spec['train']}",
                )
                continue

            # IMPORTANT: log launch *before* blocking on training completion
            _write_launch_line(
                mlog,
                f"[LAUNCH] {run_id} seed={spec['seed']} train={spec['train']} run_spec={run_spec_path}",
            )

            # Run training as separate process. Use -u for unbuffered output, so logs update live.
            cmd = [
                sys.executable,
                "-u",
                str(train_script),
                "--run_spec",
                str(run_spec_path),
                "--output_npz",
                str(out_npz),
            ]

            with open(train_log, "a", buffering=1, encoding="utf-8") as tlog:
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
                # continue to next run (do not kill the whole shard by default)
            else:
                _write_launch_line(
                    mlog,
                    f"[DONE] {run_id} (output={out_npz})",
                )

        _write_launch_line(mlog, f"# --- DRIVER END { _now_str() } ---")


if __name__ == "__main__":
    main()