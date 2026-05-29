#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _npz_to_dict(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with np.load(path, allow_pickle=True) as data:
        for key in data.files:
            value = data[key]
            if isinstance(value, np.ndarray):
                if value.shape == ():
                    out[key] = value.item()
                else:
                    out[key] = value.tolist()
            else:
                out[key] = value
    return out


def _make_run_id(base_dir: Path, run_dir: Path, npz_dict: Dict[str, Any], run_spec: Optional[Dict[str, Any]]) -> str:
    rel = run_dir.relative_to(base_dir).as_posix()
    seed = None
    if isinstance(run_spec, dict):
        seed = run_spec.get("seed")
        if seed is None and isinstance(run_spec.get("train"), dict):
            seed = run_spec["train"].get("seed")
    if seed is None:
        seed = npz_dict.get("seed")
    if seed is not None:
        return f"{rel}__seed_{seed}"
    return rel


def _find_svd_diagnostics_path(run_dir: Path, npz_dict: Dict[str, Any], run_spec: Optional[Dict[str, Any]]) -> Optional[Path]:
    candidates: List[Path] = []

    npz_path = npz_dict.get("svd_diag_path")
    if isinstance(npz_path, str) and npz_path:
        candidates.append(Path(npz_path))

    filename = npz_dict.get("svd_diag_filename")
    if isinstance(filename, str) and filename:
        candidates.append(run_dir / filename)

    if isinstance(run_spec, dict) and isinstance(run_spec.get("train"), dict):
        train = run_spec["train"]
        filename = train.get("svd_diag_filename", "svd_diagnostics.pt")
        candidates.append(run_dir / str(filename))

    candidates.append(run_dir / "svd_diagnostics.pt")

    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def collect_results(results_dir: Path, verbose: bool = False) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}

    output_files = sorted(results_dir.rglob("output.npz"))
    if verbose:
        print(f"[INFO] searching under: {results_dir}")
        print(f"[INFO] found {len(output_files)} output.npz files")

    for output_npz in output_files:
        run_dir = output_npz.parent
        run_spec_path = run_dir / "run_spec.json"
        metrics_log_path = run_dir / "metrics_log.jsonl"

        try:
            npz_dict = _npz_to_dict(output_npz)
        except Exception as exc:
            if verbose:
                print(f"[WARN] could not read {output_npz}: {exc}")
            continue

        run_spec = _load_json(run_spec_path)
        metrics_log = _load_jsonl(metrics_log_path)

        architecture = None
        dataset = None
        train = None
        seed = None

        if isinstance(run_spec, dict):
            architecture = run_spec.get("architecture")
            dataset = run_spec.get("dataset")
            train = run_spec.get("train")
            seed = run_spec.get("seed")
            if seed is None and isinstance(train, dict):
                seed = train.get("seed")

        if seed is None:
            seed = npz_dict.get("seed")

        run_id = _make_run_id(results_dir, run_dir, npz_dict, run_spec)
        svd_path = _find_svd_diagnostics_path(run_dir, npz_dict, run_spec)
        results[run_id] = {
            "run_dir": str(run_dir),
            "run_spec": run_spec,
            "architecture": architecture,
            "dataset": dataset,
            "train": train,
            "seed": seed,
            "npz": npz_dict,
            "metrics_log": metrics_log,
            "svd_diagnostics_path": None if svd_path is None else str(svd_path),
            "svd_diagnostics_exists": svd_path is not None,
        }

    return results


def _to_float(x: Any, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _to_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def build_aggregated_np_arrays(results: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
    run_ids = sorted(results.keys())

    scalar_fields = {
        "seed": [],
        "lr": [],
        "inv_sigma_w": [],
        "sigma2_w": [],
        "batch_size": [],
        "epochs": [],
        "n_train": [],
        "n_test": [],
        "target_mean": [],
        "target_std": [],
        "target_mean_years": [],
        "target_std_years": [],
        "final_train_loss_std_mse": [],
        "final_test_loss_std_mse": [],
        "final_train_mae": [],
        "final_test_mae": [],
        "final_train_mse": [],
        "final_test_mse": [],
        "final_train_mae_years": [],
        "final_test_mae_years": [],
        "final_train_mse_years": [],
        "final_test_mse_years": [],
        "final_train_grad_norm": [],
    }

    run_dir_arr = []
    dataset_name_arr = []
    dataset_root_arr = []
    target_name_arr = []
    target_units_arr = []
    architecture_json_arr = []
    dataset_json_arr = []
    train_json_arr = []
    svd_diagnostics_path_arr = []
    svd_diagnostics_exists_arr = []

    eval_epochs_arr = []
    train_loss_arr = []
    test_loss_arr = []
    train_mae_arr = []
    test_mae_arr = []
    train_mse_years_arr = []
    test_mse_years_arr = []
    train_grad_norm_arr = []

    for run_id in run_ids:
        entry = results[run_id]
        npz = entry.get("npz", {})

        run_dir_arr.append(entry.get("run_dir"))
        dataset_name_arr.append(npz.get("dataset_name"))
        dataset_root_arr.append(npz.get("dataset_root"))
        target_name_arr.append(npz.get("target_name"))
        target_units_arr.append(npz.get("target_units"))
        architecture_json_arr.append(json.dumps(entry.get("architecture"), sort_keys=True))
        dataset_json_arr.append(json.dumps(entry.get("dataset"), sort_keys=True))
        train_json_arr.append(json.dumps(entry.get("train"), sort_keys=True))
        svd_diagnostics_path_arr.append(entry.get("svd_diagnostics_path"))
        svd_diagnostics_exists_arr.append(bool(entry.get("svd_diagnostics_exists", False)))

        scalar_fields["seed"].append(_to_int(entry.get("seed", npz.get("seed"))))
        scalar_fields["lr"].append(_to_float(npz.get("lr")))
        scalar_fields["inv_sigma_w"].append(_to_float(npz.get("inv_sigma_w")))
        scalar_fields["sigma2_w"].append(_to_float(npz.get("sigma2_w")))
        scalar_fields["batch_size"].append(_to_int(npz.get("batch_size")))
        scalar_fields["epochs"].append(_to_int(npz.get("epochs")))
        scalar_fields["n_train"].append(_to_int(npz.get("n_train")))
        scalar_fields["n_test"].append(_to_int(npz.get("n_test")))
        scalar_fields["target_mean"].append(_to_float(npz.get("target_mean", npz.get("target_mean_years"))))
        scalar_fields["target_std"].append(_to_float(npz.get("target_std", npz.get("target_std_years"))))
        scalar_fields["target_mean_years"].append(_to_float(npz.get("target_mean_years", npz.get("target_mean"))))
        scalar_fields["target_std_years"].append(_to_float(npz.get("target_std_years", npz.get("target_std"))))
        scalar_fields["final_train_loss_std_mse"].append(_to_float(npz.get("final_train_loss_std_mse")))
        scalar_fields["final_test_loss_std_mse"].append(_to_float(npz.get("final_test_loss_std_mse")))
        scalar_fields["final_train_mae"].append(_to_float(npz.get("final_train_mae", npz.get("final_train_mae_years"))))
        scalar_fields["final_test_mae"].append(_to_float(npz.get("final_test_mae", npz.get("final_test_mae_years"))))
        scalar_fields["final_train_mse"].append(_to_float(npz.get("final_train_mse", npz.get("final_train_mse_years"))))
        scalar_fields["final_test_mse"].append(_to_float(npz.get("final_test_mse", npz.get("final_test_mse_years"))))
        scalar_fields["final_train_mae_years"].append(_to_float(npz.get("final_train_mae_years", npz.get("final_train_mae"))))
        scalar_fields["final_test_mae_years"].append(_to_float(npz.get("final_test_mae_years", npz.get("final_test_mae"))))
        scalar_fields["final_train_mse_years"].append(_to_float(npz.get("final_train_mse_years", npz.get("final_train_mse"))))
        scalar_fields["final_test_mse_years"].append(_to_float(npz.get("final_test_mse_years", npz.get("final_test_mse"))))
        scalar_fields["final_train_grad_norm"].append(_to_float(npz.get("final_train_grad_norm")))

        eval_epochs_arr.append(np.array(npz.get("eval_epochs", []), dtype=np.int64))
        train_loss_arr.append(np.array(npz.get("train_loss_std_mse", []), dtype=np.float64))
        test_loss_arr.append(np.array(npz.get("test_loss_std_mse", []), dtype=np.float64))
        train_mae_arr.append(np.array(npz.get("train_mae", npz.get("train_mae_years", [])), dtype=np.float64))
        test_mae_arr.append(np.array(npz.get("test_mae", npz.get("test_mae_years", [])), dtype=np.float64))
        train_mse_years_arr.append(np.array(npz.get("train_mse", npz.get("train_mse_years", [])), dtype=np.float64))
        test_mse_years_arr.append(np.array(npz.get("test_mse", npz.get("test_mse_years", [])), dtype=np.float64))
        train_grad_norm_arr.append(np.array(npz.get("train_grad_norm", []), dtype=np.float64))

    aggregated: Dict[str, np.ndarray] = {
        "run_id": np.array(run_ids, dtype=object),
        "run_dir": np.array(run_dir_arr, dtype=object),
        "dataset_name": np.array(dataset_name_arr, dtype=object),
        "dataset_root": np.array(dataset_root_arr, dtype=object),
        "target_name": np.array(target_name_arr, dtype=object),
        "target_units": np.array(target_units_arr, dtype=object),
        "architecture_json": np.array(architecture_json_arr, dtype=object),
        "dataset_json": np.array(dataset_json_arr, dtype=object),
        "train_json": np.array(train_json_arr, dtype=object),
        "svd_diagnostics_path": np.array(svd_diagnostics_path_arr, dtype=object),
        "svd_diagnostics_exists": np.array(svd_diagnostics_exists_arr, dtype=bool),
        "eval_epochs": np.array(eval_epochs_arr, dtype=object),
        "train_loss_std_mse": np.array(train_loss_arr, dtype=object),
        "test_loss_std_mse": np.array(test_loss_arr, dtype=object),
        "train_mae_years": np.array(train_mae_arr, dtype=object),
        "test_mae_years": np.array(test_mae_arr, dtype=object),
        "train_mse_years": np.array(train_mse_years_arr, dtype=object),
        "test_mse_years": np.array(test_mse_years_arr, dtype=object),
        "train_grad_norm": np.array(train_grad_norm_arr, dtype=object),
    }

    for key, values in scalar_fields.items():
        if key in {"seed", "batch_size", "epochs", "n_train", "n_test"}:
            aggregated[key] = np.array(values, dtype=np.int64)
        else:
            aggregated[key] = np.array(values, dtype=np.float64)

    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect experiment-2 outputs. By default saves results/<run_name>.npy, where run_name is results_dir.name.")
    parser.add_argument("--results_dir", type=str, required=True, help="Root folder containing experiment outputs.")
    parser.add_argument("--save_name", type=str, default=None, help="Base filename without suffix. If omitted, uses results_dir.name.")
    parser.add_argument("--results_root", type=str, default="results", help="Folder used by the default .npy output path.")
    parser.add_argument("--output_pickle", type=str, default=None, help="Exact pickle output path.")
    parser.add_argument("--output_npz", type=str, default=None, help="Exact npz output path.")
    parser.add_argument("--output_npy", type=str, default=None, help="Exact npy output path for a dict.")
    parser.add_argument("--print_keys", action="store_true", help="Print collected run ids.")
    parser.add_argument("--verbose", action="store_true", help="Print diagnostic information.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir does not exist: {results_dir}")

    results = collect_results(results_dir, verbose=args.verbose)

    print(f"[INFO] collected {len(results)} completed runs")
    if args.print_keys:
        for key in results:
            print(key)

    output_pickle: Optional[Path] = None
    output_npz: Optional[Path] = None
    output_npy: Optional[Path] = None

    if args.output_pickle is not None:
        output_pickle = Path(args.output_pickle).expanduser().resolve()
    if args.output_npz is not None:
        output_npz = Path(args.output_npz).expanduser().resolve()
    if args.output_npy is not None:
        output_npy = Path(args.output_npy).expanduser().resolve()

    # Default behaviour requested for the old experiment workflow:
    #   python src/experiment_2/collect_results.py --results_dir data/experiment_2/<RUN_NAME>
    # creates:
    #   results/<RUN_NAME>.npy
    # No need to repeat RUN_NAME twice on the command line.
    if args.save_name is None and output_pickle is None and output_npz is None and output_npy is None:
        output_npy = (Path(args.results_root).expanduser().resolve() / results_dir.name).with_suffix(".npy")

    if args.save_name is not None:
        base = (Path(args.results_root).expanduser().resolve() / args.save_name)
        if output_npy is None and output_pickle is None and output_npz is None:
            output_npy = base.with_suffix(".npy")

    if output_pickle is not None:
        output_pickle.parent.mkdir(parents=True, exist_ok=True)
        with output_pickle.open("wb") as f:
            pickle.dump(results, f)
        print(f"[INFO] saved pickle to: {output_pickle}")

    aggregated = build_aggregated_np_arrays(results)

    if output_npz is not None:
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_npz, **aggregated)
        print(f"[INFO] saved npz to: {output_npz}")

    if output_npy is not None:
        output_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_npy, aggregated, allow_pickle=True)
        print(f"[INFO] saved npy dict to: {output_npy}")

    if len(results) == 0:
        print("[WARN] No output.npz files were found under the supplied results_dir.")
        print("[WARN] Check that you pointed to the correct run folder, for example results/experiment_2/exp2_v1")


if __name__ == "__main__":
    main()