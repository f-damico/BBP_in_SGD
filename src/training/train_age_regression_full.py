#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.simple_cnn_regressor import SimpleCNNRegressor
from src.models import MODEL_BUILDERS
from src.training.regression_datasets import make_regression_split_datasets

try:
    from svd_diagnostics import (
        append_weight_svd_diagnostics,
        infer_input_shape_from_configs,
        make_initial_svd_payload,
        make_svd_config,
    )
except ImportError:  # when executed as src/training/train_age_regression_full.py
    from src.training.svd_diagnostics import (
        append_weight_svd_diagnostics,
        infer_input_shape_from_configs,
        make_initial_svd_payload,
        make_svd_config,
    )


@dataclass(frozen=True)
class UTKFaceSample:
    image_path: Path
    age: float


class UTKFaceAgeDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        root: Path,
        *,
        image_size: int,
        max_samples: int | None = None,
        shuffle_seed: int = 12345,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(
                f"UTKFace dataset root not found: {self.root}. "
                "Set dataset.root in the config to the directory containing the image files."
            )

        self.image_size = int(image_size)

        files = sorted(
            [
                p
                for p in self.root.rglob("*")
                if p.is_file()
                and not p.name.startswith(".")
                and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )

        samples: List[UTKFaceSample] = []
        for path in files:
            age = self._parse_age_from_filename(path.name)
            if age is None:
                continue
            samples.append(UTKFaceSample(image_path=path, age=float(age)))

        if len(samples) == 0:
            raise RuntimeError(
                f"No UTKFace-like image files with parsable ages found in {self.root}."
            )

        rng = random.Random(int(shuffle_seed))
        rng.shuffle(samples)

        if max_samples is not None:
            max_samples = int(max_samples)
            if max_samples <= 0:
                raise ValueError("max_samples must be positive when provided.")
            samples = samples[:max_samples]

        self.samples = samples

    @staticmethod
    def _parse_age_from_filename(filename: str) -> int | None:
        stem = Path(filename).stem
        parts = stem.split("_")
        if len(parts) < 1:
            return None
        try:
            age = int(parts[0])
        except ValueError:
            return None
        if age < 0 or age > 120:
            return None
        return age

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as img:
            img = img.convert("RGB")
            img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            arr = np.transpose(arr, (2, 0, 1))
            x = torch.from_numpy(arr)
        y = torch.tensor([sample.age], dtype=torch.float32)
        return x, y


class StandardizedTargetDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        base_dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]],
        y_mean: float,
        y_std: float,
    ) -> None:
        self.base_dataset = base_dataset
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.base_dataset[index]
        y_standardized = (y - self.y_mean) / self.y_std
        return x, y_standardized


def set_seed_everywhere(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_determinism(seed: int) -> None:
    set_seed_everywhere(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_model(
    architecture: Dict[str, Any],
    *,
    sigma2_w: float,
    seed: int,
    device: torch.device,
) -> SimpleCNNRegressor:
    model = SimpleCNNRegressor(
        input_channels=int(architecture.get("input_channels", 3)),
        conv_channels=[int(c) for c in architecture["conv_channels"]],
        mlp_hidden_dims=[int(h) for h in architecture.get("mlp_hidden_dims", [])],
        output_dim=int(architecture.get("output_dim", 1)),
        activation=str(architecture.get("activation", "relu")),
        bias=bool(architecture.get("bias", True)),
    )
    model.initialize_all_layers(sigma2_w=float(sigma2_w), seed=int(seed))
    return model.to(device=device)


@torch.no_grad()
def compute_target_stats(dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[float, float]:
    ys: List[float] = []
    for _, y in dataset:
        ys.append(float(y.item()))
    y_array = np.asarray(ys, dtype=np.float64)
    mean = float(y_array.mean())
    std = float(y_array.std())
    if std <= 0.0:
        std = 1.0
    return mean, std


def make_split_datasets(dataset_cfg: Dict[str, Any]) -> Tuple[Dataset, Dataset, float, float]:
    """
    Shared experiment-2 dataset factory.

    This delegates to src.training.regression_datasets so that the old UTKFace
    path and the new Superconductivity path use one common implementation.
    """
    return make_regression_split_datasets(dataset_cfg, repo_root=REPO_ROOT)


def build_loader(
    dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    y_mean: float,
    y_std: float,
) -> Dict[str, float]:
    model.eval()
    mse_std_sum = 0.0
    mae_years_sum = 0.0
    mse_years_sum = 0.0
    n_samples = 0

    for x, y_standardized in loader:
        x = x.to(device, non_blocking=True)
        y_standardized = y_standardized.to(device, non_blocking=True)

        pred_standardized = model(x)

        mse_std_sum += float(torch.sum((pred_standardized - y_standardized) ** 2).item())

        pred_years = pred_standardized * float(y_std) + float(y_mean)
        true_years = y_standardized * float(y_std) + float(y_mean)

        mae_years_sum += float(torch.sum(torch.abs(pred_years - true_years)).item())
        mse_years_sum += float(torch.sum((pred_years - true_years) ** 2).item())
        n_samples += int(x.shape[0])

    denom = max(n_samples, 1)
    return {
        "loss_std_mse": mse_std_sum / denom,
        "mae_years": mae_years_sum / denom,
        "mse_years": mse_years_sum / denom,
    }


def compute_full_train_gradient_norm(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_fn: nn.Module,
) -> float:
    was_training = model.training
    model.train()

    model.zero_grad(set_to_none=True)

    n_samples = len(loader.dataset)
    if n_samples <= 0:
        return float("nan")

    for x, y_standardized in loader:
        x = x.to(device, non_blocking=True)
        y_standardized = y_standardized.to(device, non_blocking=True)

        pred_standardized = model(x)
        batch_loss = loss_fn(pred_standardized, y_standardized)

        weight = float(x.shape[0]) / float(n_samples)
        (batch_loss * weight).backward()

    grad_sq_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            grad_sq_norm += float(torch.sum(g * g).item())

    grad_norm = grad_sq_norm ** 0.5

    model.zero_grad(set_to_none=True)

    if not was_training:
        model.eval()

    return grad_norm


def train_one_epoch_full_batch_accumulation(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip_norm: float | None,
) -> float:
    """
    One optimizer step per epoch over the full training set.

    batch_size from the loader is only the microbatch size used to accumulate
    gradients. The effective optimization batch size is the entire dataset.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    n_samples = len(loader.dataset)
    if n_samples <= 0:
        return float("nan")

    total_loss_sum = 0.0
    total_seen = 0

    for x, y_standardized in loader:
        x = x.to(device, non_blocking=True)
        y_standardized = y_standardized.to(device, non_blocking=True)

        pred_standardized = model(x)
        batch_loss = loss_fn(pred_standardized, y_standardized)

        # Reconstruct gradient of the mean loss over the whole dataset.
        weight = float(x.shape[0]) / float(n_samples)
        (batch_loss * weight).backward()

        total_loss_sum += float(batch_loss.item()) * float(x.shape[0])
        total_seen += int(x.shape[0])

    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return total_loss_sum / max(total_seen, 1)


def maybe_log_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _checkpoint_dtype_from_name(name: str) -> torch.dtype | None:
    name = str(name).strip().lower()
    if name in {"none", "original", "keep"}:
        return None
    if name in {"float32", "fp32", "torch.float32"}:
        return torch.float32
    if name in {"float16", "fp16", "torch.float16"}:
        return torch.float16
    if name in {"bfloat16", "bf16", "torch.bfloat16"}:
        return torch.bfloat16
    raise ValueError(
        "weight_checkpoint_dtype must be one of: original, float32, float16, bfloat16."
    )


def _model_state_dict_cpu(
    model: nn.Module,
    *,
    dtype_name: str = "float32",
) -> Dict[str, torch.Tensor]:
    target_dtype = _checkpoint_dtype_from_name(dtype_name)
    out: Dict[str, torch.Tensor] = {}

    for name, tensor in model.state_dict().items():
        t = tensor.detach().cpu()
        if target_dtype is not None and torch.is_floating_point(t):
            t = t.to(dtype=target_dtype)
        out[name] = t.clone()

    return out


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def make_initial_weight_checkpoint_payload(
    *,
    run_spec: Dict[str, Any],
    run_spec_path: str,
    time_key: str = "epoch",
    dtype_name: str = "float32",
) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "description": (
            "Model weights saved at validation times. "
            "Contains model.state_dict() tensors moved to CPU. "
            "Optimizer state is intentionally not saved."
        ),
        "run_spec_path": str(run_spec_path),
        "architecture": run_spec.get("architecture", None),
        "dataset": run_spec.get("dataset", None),
        "train": run_spec.get("train", None),
        "seed": run_spec.get("seed", None),
        "time_key": str(time_key),
        "weight_checkpoint_dtype": str(dtype_name),
        "times": [],
        "by_time": {},
    }


def append_weight_checkpoint(
    *,
    path: Path,
    payload: Dict[str, Any],
    model: nn.Module,
    time_value: int,
    dtype_name: str,
) -> None:
    time_value = int(time_value)
    key = str(time_value)

    payload.setdefault("times", []).append(time_value)
    payload.setdefault("by_time", {})[key] = {
        "model_state_dict": _model_state_dict_cpu(model, dtype_name=dtype_name),
    }

    _atomic_torch_save(payload, path)


def train_one_run(run_spec: Dict[str, Any], output_npz: Path) -> None:
    seed = int(run_spec["seed"])
    architecture = dict(run_spec["architecture"])
    dataset_cfg = dict(run_spec["dataset"])
    train_cfg = dict(run_spec["train"])

    set_determinism(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = int(dataset_cfg.get("num_workers", 4))

    train_dataset, test_dataset, y_mean, y_std = make_split_datasets(dataset_cfg)

    microbatch_size = int(train_cfg["batch_size"])

    # For exact full-batch accumulation, shuffle is not needed.
    train_loader = build_loader(
        train_dataset,
        batch_size=microbatch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )
    train_eval_loader = build_loader(
        train_dataset,
        batch_size=microbatch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )
    test_loader = build_loader(
        test_dataset,
        batch_size=microbatch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )

    arch = dict(architecture)
    if "sigma2_w" in train_cfg:
        arch["sigma2_w"] = train_cfg["sigma2_w"]

    model_name = str(arch["model_name"]).strip().lower()
    model = MODEL_BUILDERS[model_name](
        arch,
        seed=seed,
        device=device,
    )

    optimizer_name = str(train_cfg.get("optimizer", "sgd")).lower()
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    epochs = int(train_cfg["epochs"])

    grad_clip_norm = train_cfg.get("grad_clip_norm", None)
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)
        if grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be strictly positive when provided.")

    eval_epochs = sorted({int(e) for e in train_cfg.get("eval_epochs", [epochs])})
    if epochs not in eval_epochs:
        eval_epochs.append(epochs)

    save_svd_diagnostics = bool(train_cfg.get("save_svd_diagnostics", False))
    svd_cfg = make_svd_config(train_cfg)
    svd_diag_filename = str(svd_cfg["diag_filename"])
    svd_input_shape = infer_input_shape_from_configs(arch, dataset_cfg)

    save_weight_checkpoints = bool(train_cfg.get("save_weight_checkpoints", False))
    weight_checkpoint_filename = str(
        train_cfg.get("weight_checkpoint_filename", "weight_checkpoints.pt")
    )
    weight_checkpoint_dtype = str(
        train_cfg.get("weight_checkpoint_dtype", "float32")
    )

    if optimizer_name == "sgd":
        momentum = float(train_cfg.get("momentum", 0.0))
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adam":
        beta1 = float(train_cfg.get("beta1", 0.9))
        beta2 = float(train_cfg.get("beta2", 0.999))
        eps = float(train_cfg.get("eps", 1.0e-8))
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}'. Supported: 'sgd', 'adam'."
        )

    loss_fn = nn.MSELoss(reduction="mean")

    run_dir = output_npz.parent
    metrics_log_path = run_dir / "metrics_log.jsonl"
    if metrics_log_path.exists():
        metrics_log_path.unlink()

    svd_diag_path = run_dir / svd_diag_filename
    if save_svd_diagnostics and svd_diag_path.exists():
        svd_diag_path.unlink()

    weight_checkpoint_path = run_dir / weight_checkpoint_filename
    if save_weight_checkpoints and weight_checkpoint_path.exists():
        weight_checkpoint_path.unlink()

    epoch_list: List[int] = []
    train_loss_list: List[float] = []
    test_loss_list: List[float] = []
    train_mae_list: List[float] = []
    test_mae_list: List[float] = []
    train_mse_years_list: List[float] = []
    test_mse_years_list: List[float] = []
    train_grad_norm_list: List[float] = []
    epoch_train_loss_before_step_list: List[float] = []

    svd_payload: Dict[str, Any] | None = None
    if save_svd_diagnostics:
        svd_payload = make_initial_svd_payload(
            run_spec_path=str(run_spec.get("run_spec_path", "")),
            time_key="epoch",
            input_shape=svd_input_shape,
            svd_config=svd_cfg,
        )

    weight_checkpoint_payload: Dict[str, Any] | None = None
    if save_weight_checkpoints:
        weight_checkpoint_payload = make_initial_weight_checkpoint_payload(
            run_spec=run_spec,
            run_spec_path=str(run_spec.get("run_spec_path", "")),
            time_key="epoch",
            dtype_name=weight_checkpoint_dtype,
        )

    for epoch in range(1, epochs + 1):
        train_loss_before_step = train_one_epoch_full_batch_accumulation(
            model,
            train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip_norm=grad_clip_norm,
        )

        if epoch in eval_epochs:
            train_metrics = evaluate(
                model,
                train_eval_loader,
                device=device,
                y_mean=y_mean,
                y_std=y_std,
            )
            test_metrics = evaluate(
                model,
                test_loader,
                device=device,
                y_mean=y_mean,
                y_std=y_std,
            )

            train_grad_norm = compute_full_train_gradient_norm(
                model,
                train_eval_loader,
                device=device,
                loss_fn=loss_fn,
            )

            if save_svd_diagnostics:
                if svd_payload is None:
                    raise RuntimeError("SVD diagnostics were requested but could not be initialised.")
                append_weight_svd_diagnostics(
                    path=svd_diag_path,
                    payload=svd_payload,
                    model=model,
                    time_value=epoch,
                    input_shape=svd_input_shape,
                    svd_config=svd_cfg,
                )

            if save_weight_checkpoints:
                if weight_checkpoint_payload is None:
                    raise RuntimeError("Weight checkpoints were requested but could not be initialised.")
                append_weight_checkpoint(
                    path=weight_checkpoint_path,
                    payload=weight_checkpoint_payload,
                    model=model,
                    time_value=epoch,
                    dtype_name=weight_checkpoint_dtype,
                )

            epoch_list.append(epoch)
            epoch_train_loss_before_step_list.append(train_loss_before_step)
            train_loss_list.append(train_metrics["loss_std_mse"])
            test_loss_list.append(test_metrics["loss_std_mse"])
            train_mae_list.append(train_metrics["mae_years"])
            test_mae_list.append(test_metrics["mae_years"])
            train_mse_years_list.append(train_metrics["mse_years"])
            test_mse_years_list.append(test_metrics["mse_years"])
            train_grad_norm_list.append(train_grad_norm)

            maybe_log_jsonl(
                metrics_log_path,
                {
                    "epoch": epoch,
                    "train_loss_before_step_std_mse": train_loss_before_step,
                    "train_grad_norm": train_grad_norm,
                    "train_loss_std_mse": train_metrics["loss_std_mse"],
                    "test_loss_std_mse": test_metrics["loss_std_mse"],
                    "train_mae_years": train_metrics["mae_years"],
                    "test_mae_years": test_metrics["mae_years"],
                    "train_mse_years": train_metrics["mse_years"],
                    "test_mse_years": test_metrics["mse_years"],
                    "svd_diagnostics_saved": bool(save_svd_diagnostics),
                    "svd_diag_filename": svd_diag_filename if save_svd_diagnostics else None,
                    "svd_diag_path": str(svd_diag_path) if save_svd_diagnostics else None,
                    "weight_checkpoints_saved": bool(save_weight_checkpoints),
                    "weight_checkpoint_filename": weight_checkpoint_filename if save_weight_checkpoints else None,
                    "weight_checkpoint_path": str(weight_checkpoint_path) if save_weight_checkpoints else None,
                    "weight_checkpoint_dtype": weight_checkpoint_dtype if save_weight_checkpoints else None,
                },
            )

            print(
                f"[EVAL] epoch={epoch:03d} "
                f"train_loss_before_step_std_mse={train_loss_before_step:.6f} "
                f"train_loss_std_mse={train_metrics['loss_std_mse']:.6f} "
                f"test_loss_std_mse={test_metrics['loss_std_mse']:.6f} "
                f"train_mae_years={train_metrics['mae_years']:.4f} "
                f"test_mae_years={test_metrics['mae_years']:.4f} "
                f"train_grad_norm={train_grad_norm:.6e} "
                f"svd_diagnostics={'on' if save_svd_diagnostics else 'off'} "
                f"weight_checkpoints={'on' if save_weight_checkpoints else 'off'}",
                flush=True,
            )

    np.savez(
        output_npz,
        seed=np.int64(seed),
        sigma2_w=np.float64(train_cfg["sigma2_w"]),
        inv_sigma_w=np.float64(1.0 / max(float(train_cfg["sigma2_w"]) ** 0.5, 1.0e-12)),
        grad_clip_norm=np.float64(-1.0 if grad_clip_norm is None else grad_clip_norm),
        lr=np.float64(lr),
        batch_size=np.int64(microbatch_size),  # now interpreted as microbatch size
        effective_batch_size=np.int64(len(train_dataset)),  # full batch
        epochs=np.int64(epochs),
        dataset_name=np.array(str(dataset_cfg.get("name", ""))),
        dataset_root=np.array(str(dataset_cfg["root"])),
        target_name=np.array(str(dataset_cfg.get("target", "target"))),
        target_units=np.array(str(dataset_cfg.get("target_units", ""))),
        n_train=np.int64(len(train_dataset)),
        n_test=np.int64(len(test_dataset)),
        target_mean=np.float64(y_mean),
        target_std=np.float64(y_std),
        target_mean_years=np.float64(y_mean),
        target_std_years=np.float64(y_std),
        optimization_mode=np.array("full_batch_accumulation"),
        eval_epochs=np.asarray(epoch_list, dtype=np.int64),
        train_loss_before_step_std_mse=np.asarray(epoch_train_loss_before_step_list, dtype=np.float64),
        train_loss_std_mse=np.asarray(train_loss_list, dtype=np.float64),
        test_loss_std_mse=np.asarray(test_loss_list, dtype=np.float64),
        train_mae=np.asarray(train_mae_list, dtype=np.float64),
        test_mae=np.asarray(test_mae_list, dtype=np.float64),
        train_mse=np.asarray(train_mse_years_list, dtype=np.float64),
        test_mse=np.asarray(test_mse_years_list, dtype=np.float64),
        train_mae_years=np.asarray(train_mae_list, dtype=np.float64),
        test_mae_years=np.asarray(test_mae_list, dtype=np.float64),
        train_mse_years=np.asarray(train_mse_years_list, dtype=np.float64),
        test_mse_years=np.asarray(test_mse_years_list, dtype=np.float64),
        final_train_loss_std_mse=np.float64(train_loss_list[-1]),
        final_test_loss_std_mse=np.float64(test_loss_list[-1]),
        final_train_mae=np.float64(train_mae_list[-1]),
        final_test_mae=np.float64(test_mae_list[-1]),
        final_train_mse=np.float64(train_mse_years_list[-1]),
        final_test_mse=np.float64(test_mse_years_list[-1]),
        final_train_mae_years=np.float64(train_mae_list[-1]),
        final_test_mae_years=np.float64(test_mae_list[-1]),
        final_train_mse_years=np.float64(train_mse_years_list[-1]),
        final_test_mse_years=np.float64(test_mse_years_list[-1]),
        optimizer=np.array(optimizer_name),
        weight_decay=np.float64(weight_decay),
        momentum=np.float64(train_cfg.get("momentum", 0.0)),
        beta1=np.float64(train_cfg.get("beta1", 0.9)),
        beta2=np.float64(train_cfg.get("beta2", 0.999)),
        eps=np.float64(train_cfg.get("eps", 1.0e-8)),
        train_grad_norm=np.asarray(train_grad_norm_list, dtype=np.float64),
        final_train_grad_norm=np.float64(train_grad_norm_list[-1]),
        save_svd_diagnostics=np.bool_(save_svd_diagnostics),
        svd_diag_filename=np.array(svd_diag_filename),
        svd_diag_path=np.array(str(svd_diag_path) if save_svd_diagnostics else ""),
        save_weight_checkpoints=np.bool_(save_weight_checkpoints),
        weight_checkpoint_filename=np.array(weight_checkpoint_filename),
        weight_checkpoint_path=np.array(str(weight_checkpoint_path) if save_weight_checkpoints else ""),
        weight_checkpoint_dtype=np.array(weight_checkpoint_dtype),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_spec", type=str, required=True)
    parser.add_argument("--output_npz", type=str, required=True)
    args = parser.parse_args()

    run_spec_path = Path(args.run_spec).resolve()
    output_npz = Path(args.output_npz).resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    run_spec = load_json(run_spec_path)
    run_spec["run_spec_path"] = str(run_spec_path)
    train_one_run(run_spec, output_npz)


if __name__ == "__main__":
    main()