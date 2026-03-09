#!/usr/bin/env python3
"""
Training script for the teacher-student phase-diagram experiment.
Outputs:
- output_npz: compressed NumPy file with loss trajectories + metadata;
- metrics_log.txt next to output_npz: plain-text training log.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.teacher_student_mlp import TeacherStudentMLP, init_linear_normal_scaled  # type: ignore
from phase_diagram.make_teacher_and_data import make_teacher_and_data  # type: ignore


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_or_yaml(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in [".yml", ".yaml"]:
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Install pyyaml or use JSON."
            ) from e
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config extension: {path.suffix}")


def write_log_line(handle, text: str) -> None:
    handle.write(text + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    print(text, flush=True)


def infer_trainable_layer_indices(num_linear_layers: int, train_cfg: Dict[str, Any]) -> List[int]:
    if "trainable_layer_indices" in train_cfg:
        out = [int(i) for i in train_cfg["trainable_layer_indices"]]
        if not out:
            raise ValueError("trainable_layer_indices cannot be empty.")
        return out

    # Default paper case
    if num_linear_layers == 3:
        return [1]

    raise ValueError(
        "trainable_layer_indices not provided and architecture is not the 3-layer paper case. "
        "Please specify train['trainable_layer_indices']."
    )


def get_sigma2_w_trainable(train_cfg: Dict[str, Any]) -> float:
    if "inv_sigma_w" in train_cfg:
        inv_sigma_w = float(train_cfg["inv_sigma_w"])
        if inv_sigma_w <= 0:
            raise ValueError("inv_sigma_w must be positive.")
        return 1.0 / (inv_sigma_w ** 2)

    if "sigma_w" in train_cfg:
        sigma_w = float(train_cfg["sigma_w"])
        if sigma_w <= 0:
            raise ValueError("sigma_w must be positive.")
        return sigma_w ** 2

    if "sigma2_w" in train_cfg:
        sigma2_w = float(train_cfg["sigma2_w"])
        if sigma2_w <= 0:
            raise ValueError("sigma2_w must be positive.")
        return sigma2_w

    raise ValueError("Training config must contain one of: inv_sigma_w, sigma_w, sigma2_w.")


def get_eval_steps(train_cfg: Dict[str, Any], num_updates: int) -> List[int]:
    raw = train_cfg.get("eval_steps", train_cfg.get("log_steps", None))
    if raw is None:
        steps = [0, num_updates]
    else:
        steps = sorted(set(int(s) for s in raw))
        if 0 not in steps:
            steps = [0] + steps
        if num_updates not in steps:
            steps.append(num_updates)
    for s in steps:
        if s < 0 or s > num_updates:
            raise ValueError(f"Invalid eval step {s}; must be in [0, {num_updates}].")
    return steps


def choose_device(train_cfg: Dict[str, Any]) -> torch.device:
    device_str = train_cfg.get("device", None)
    if device_str is not None:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_teacher_data_config(run_spec: Dict[str, Any], run_dir: Path) -> Path:
    """
    1) if a config path is already present in run_spec, use it;
    2) otherwise create a temporary config JSON inside the run folder.
    """
    for key in ["teacher_data_config", "teacher_data_config_path", "model_config_path", "same_as_paper_config"]:
        if key in run_spec:
            return Path(run_spec[key]).resolve()

    config_dict = {
        "architecture": run_spec["architecture"],
        "dataset": run_spec["dataset"],
        "teacher": run_spec.get("teacher", {"sigma2_w": 1.0}),
    }
    out_path = run_dir / "teacher_data_config.json"
    out_path.write_text(json.dumps(config_dict, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def build_student_from_teacher(
    teacher: nn.Module,
    architecture: Dict[str, Any],
    *,
    trainable_layer_indices: Sequence[int],
    sigma2_w_trainable: float,
    seed_student: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TeacherStudentMLP:
    student = TeacherStudentMLP(
        input_dim=int(architecture["input_dim"]),
        hidden_dims=[int(h) for h in architecture["hidden_dims"]],
        output_dim=int(architecture["output_dim"]),
    ).to(device=device, dtype=dtype)

    if len(student.linears) != len(teacher.linears):
        raise ValueError("Teacher and student do not have the same number of linear layers.")

    with torch.no_grad():
        for layer_s, layer_t in zip(student.linears, teacher.linears):
            layer_s.weight.copy_(layer_t.weight.to(device=device, dtype=dtype))

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed_student))
    for idx in trainable_layer_indices:
        init_linear_normal_scaled(student.linears[idx], sigma2_w=sigma2_w_trainable, generator=gen)

    student.set_trainable_layers(trainable_layer_indices)
    return student


@torch.no_grad()
def compute_mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    model.eval()
    pred = model(x.to(device))
    loss = torch.mean((pred - y.to(device)) ** 2)
    return float(loss.detach().cpu().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_spec", type=str, required=True, help="Path to run_spec.json")
    parser.add_argument("--output_npz", type=str, required=True, help="Path to compressed output .npz")
    args = parser.parse_args()

    run_spec_path = Path(args.run_spec).resolve()
    output_npz = Path(args.output_npz).resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    run_dir = output_npz.parent
    metrics_log_path = run_dir / "metrics_log.txt"

    run_spec = load_json_or_yaml(run_spec_path)
    train_cfg = dict(run_spec.get("train", {}))
    architecture = dict(run_spec["architecture"])
    seed = int(run_spec["seed"])

    num_updates = int(train_cfg["num_updates"])
    lr = float(train_cfg["lr"])
    batch_size = int(train_cfg["batch_size"])
    sigma2_w_trainable = get_sigma2_w_trainable(train_cfg)
    if "inv_sigma_w" in train_cfg:
        inv_sigma_w = float(train_cfg["inv_sigma_w"])
    else:
        inv_sigma_w = 1.0 / np.sqrt(sigma2_w_trainable)
    eval_steps = get_eval_steps(train_cfg, num_updates)
    device = choose_device(train_cfg)
    dtype = torch.float64 if bool(train_cfg.get("use_float64", False)) else torch.float32
    trainable_layer_indices = infer_trainable_layer_indices(
        num_linear_layers=len(architecture["hidden_dims"]) + 1,
        train_cfg=train_cfg,
    )

    seed_teacher_data = seed
    seed_student = seed 
    seed_batches = seed 
    set_global_seed(seed)

    teacher_data_config = create_teacher_data_config(run_spec, run_dir)

    with open(metrics_log_path, "a", buffering=1, encoding="utf-8") as logf:
        write_log_line(logf, f"[START] run_spec={run_spec_path}")
        write_log_line(logf, f"[INFO] device={device} dtype={dtype}")
        write_log_line(logf, f"[INFO] seed={seed} lr={lr} batch_size={batch_size} num_updates={num_updates}")
        write_log_line(logf, f"[INFO] inv_sigma_w={inv_sigma_w}")
        write_log_line(logf, f"[INFO] sigma2_w_trainable={sigma2_w_trainable}")
        write_log_line(logf, f"[INFO] trainable_layer_indices={list(trainable_layer_indices)}")
        write_log_line(logf, f"[INFO] eval_steps={eval_steps}")
        write_log_line(logf, f"[INFO] teacher_data_config={teacher_data_config}")

        td = make_teacher_and_data(
            teacher_data_config,
            seed=seed_teacher_data,
            device_teacher=device,
            device_data="cpu",
            dtype=dtype,
        )

        student = build_student_from_teacher(
            td.teacher,
            architecture,
            trainable_layer_indices=trainable_layer_indices,
            sigma2_w_trainable=sigma2_w_trainable,
            seed_student=seed_student,
            device=device,
            dtype=dtype,
        )

        optimizer = torch.optim.SGD(
            (p for p in student.parameters() if p.requires_grad),
            lr=lr,
        )
        criterion = nn.MSELoss()

        X_train_cpu = td.X_train.detach().cpu()
        y_train_cpu = td.y_train.detach().cpu()
        X_test_cpu = td.X_test.detach().cpu()
        y_test_cpu = td.y_test.detach().cpu()

        n_train = X_train_cpu.shape[0]
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size > n_train:
            batch_size = n_train
            write_log_line(logf, f"[INFO] batch_size clipped to n_train={n_train}")

        batch_gen = torch.Generator(device="cpu")
        batch_gen.manual_seed(seed_batches)

        steps_recorded: List[int] = []
        train_losses: List[float] = []
        test_losses: List[float] = []

        def evaluate_and_log(step: int) -> None:
            train_loss = compute_mse(student, X_train_cpu, y_train_cpu, device=device)
            test_loss = compute_mse(student, X_test_cpu, y_test_cpu, device=device)
            steps_recorded.append(int(step))
            train_losses.append(float(train_loss))
            test_losses.append(float(test_loss))
            write_log_line(
                logf,
                f"[EVAL] step={step:8d} train_loss={train_loss:.10e} test_loss={test_loss:.10e}",
            )

        eval_steps_set = set(eval_steps)
        if 0 in eval_steps_set:
            evaluate_and_log(0)

        student.train()
        for step in range(1, num_updates + 1):
            idx = torch.randint(0, n_train, size=(batch_size,), generator=batch_gen)
            xb = X_train_cpu[idx].to(device)
            yb = y_train_cpu[idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = student(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            if step in eval_steps_set:
                evaluate_and_log(step)
                student.train()

        final_train_loss = float(train_losses[-1])
        final_test_loss = float(test_losses[-1])

        metadata = {
            "seed": seed,
            "lr": lr,
            "batch_size": batch_size,
            "num_updates": num_updates,
            "sigma2_w_trainable": sigma2_w_trainable,
            "trainable_layer_indices": list(trainable_layer_indices),
            "device": str(device),
            "dtype": str(dtype),
            "teacher_data_config": str(teacher_data_config),
            "run_spec_path": str(run_spec_path),
            "source_config": run_spec.get("source_config", None),
            "architecture": architecture,
            "dataset": run_spec.get("dataset", None),
            "inv_sigma_w": inv_sigma_w,
        }

        np.savez_compressed(
            str(output_npz),
            steps=np.asarray(steps_recorded, dtype=np.int64),
            train_loss=np.asarray(train_losses, dtype=np.float64),
            test_loss=np.asarray(test_losses, dtype=np.float64),
            final_train_loss=np.asarray(final_train_loss, dtype=np.float64),
            final_test_loss=np.asarray(final_test_loss, dtype=np.float64),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

        write_log_line(logf, f"[SAVE] output_npz={output_npz}")
        write_log_line(logf, f"[DONE] final_train_loss={final_train_loss:.10e} final_test_loss={final_test_loss:.10e}")


if __name__ == "__main__":
    main()
