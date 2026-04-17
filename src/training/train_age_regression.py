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
    def __init__(self, base_dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]], y_mean: float, y_std: float) -> None:
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


def build_model(architecture: Dict[str, Any], *, sigma2_w: float, seed: int, device: torch.device) -> SimpleCNNRegressor:
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
    image_size = int(dataset_cfg.get("image_size", 64))
    root = Path(dataset_cfg["root"])
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()

    max_samples = dataset_cfg.get("max_samples")
    split_seed = int(dataset_cfg.get("split_seed", 12345))
    train_fraction = float(dataset_cfg.get("train_fraction", 0.8))
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("dataset.train_fraction must be in (0, 1).")

    full_dataset = UTKFaceAgeDataset(
        root=root,
        image_size=image_size,
        max_samples=max_samples,
        shuffle_seed=split_seed,
    )

    n_total = len(full_dataset)
    n_train = max(1, int(math.floor(train_fraction * n_total)))
    n_test = n_total - n_train
    if n_test <= 0:
        raise ValueError("The dataset split produced an empty test set.")

    split_generator = torch.Generator().manual_seed(split_seed)
    train_dataset, test_dataset = random_split(full_dataset, [n_train, n_test], generator=split_generator)

    y_mean, y_std = compute_target_stats(train_dataset)
    train_std_dataset = StandardizedTargetDataset(train_dataset, y_mean=y_mean, y_std=y_std)
    test_std_dataset = StandardizedTargetDataset(test_dataset, y_mean=y_mean, y_std=y_std)
    return train_std_dataset, test_std_dataset, y_mean, y_std


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


def _flatten_matrix_for_svd(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.detach()
    if tensor.ndim == 4:
        return tensor.detach().reshape(tensor.shape[0], -1)
    raise ValueError(f"Unsupported tensor shape for SVD diagnostics: {tuple(tensor.shape)}")


def _iter_svd_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    layers: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            layers.append((name, module))
    return layers


def compute_full_train_gradient_info(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_fn: nn.Module,
    capture_matrices: bool = False,
) -> Tuple[float, Dict[str, torch.Tensor] | None]:
    """
    Compute || grad_theta L_train ||_2 for the mean training loss over the whole
    training set, using the same loss definition as training.

    Important:
    - this is done only at eval epochs,
    - it does not change the optimizer state,
    - gradients are cleared before and after the computation.
    """
    was_training = model.training
    model.train()

    model.zero_grad(set_to_none=True)

    n_samples = len(loader.dataset)
    if n_samples <= 0:
        return float("nan"), None

    for x, y_standardized in loader:
        x = x.to(device, non_blocking=True)
        y_standardized = y_standardized.to(device, non_blocking=True)

        pred_standardized = model(x)
        batch_loss = loss_fn(pred_standardized, y_standardized)

        # loss_fn is mean over the batch, so weight it to reconstruct
        # the mean loss over the full dataset.
        weight = float(x.shape[0]) / float(n_samples)
        (batch_loss * weight).backward()

    grad_sq_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            grad_sq_norm += float(torch.sum(g * g).item())

    grad_norm = grad_sq_norm ** 0.5

    grad_matrices: Dict[str, torch.Tensor] | None = None
    if capture_matrices:
        grad_matrices = {}
        for layer_name, module in _iter_svd_layers(model):
            if module.weight.grad is None:
                continue
            grad_matrices[layer_name] = _flatten_matrix_for_svd(module.weight.grad).detach().cpu().clone()

    model.zero_grad(set_to_none=True)

    if not was_training:
        model.eval()

    return grad_norm, grad_matrices


def _tensor_to_serializable_list(x: torch.Tensor) -> List[float]:
    return x.detach().cpu().numpy().astype(np.float64, copy=False).tolist()


def _compute_layer_svd_diagnostics(
    weight_matrix_cpu: torch.Tensor,
    grad_matrix_cpu: torch.Tensor,
    *,
    topk_values: List[int],
) -> Dict[str, Any]:
    weight_matrix_cpu = weight_matrix_cpu.detach().cpu()
    grad_matrix_cpu = grad_matrix_cpu.detach().cpu()

    weight_svals = torch.linalg.svdvals(weight_matrix_cpu)
    grad_u, grad_svals, _ = torch.linalg.svd(grad_matrix_cpu, full_matrices=False)

    overlaps: Dict[str, Any] = {}
    max_rank = int(grad_u.shape[1])
    for requested_k in topk_values:
        effective_k = min(int(requested_k), max_rank)
        if effective_k <= 0:
            q_value = float("nan")
        else:
            k_basis = grad_u[:, :effective_k]
            projected = k_basis.transpose(0, 1).matmul(weight_matrix_cpu)
            q_value = float(torch.sum(projected * projected).item())
        overlaps[str(int(requested_k))] = {
            "effective_k": int(effective_k),
            "q": q_value,
        }

    return {
        "weight_shape": list(weight_matrix_cpu.shape),
        "grad_shape": list(grad_matrix_cpu.shape),
        "weight_singular_values": _tensor_to_serializable_list(weight_svals),
        "grad_singular_values": _tensor_to_serializable_list(grad_svals),
        "grad_topk_overlaps": overlaps,
    }


@torch.no_grad()
def collect_svd_diagnostics(
    model: nn.Module,
    grad_matrices: Dict[str, torch.Tensor],
    *,
    topk_values: List[int],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "layer_order": [],
        "layers": {},
    }

    for layer_name, module in _iter_svd_layers(model):
        if layer_name not in grad_matrices:
            continue
        weight_matrix_cpu = _flatten_matrix_for_svd(module.weight).detach().cpu().clone()
        grad_matrix_cpu = grad_matrices[layer_name]
        payload["layer_order"].append(layer_name)
        payload["layers"][layer_name] = {
            "module_type": type(module).__name__,
            **_compute_layer_svd_diagnostics(
                weight_matrix_cpu,
                grad_matrix_cpu,
                topk_values=topk_values,
            ),
        }
    return payload


def maybe_log_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def save_svd_diagnostics_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_one_run(run_spec: Dict[str, Any], output_npz: Path) -> None:
    seed = int(run_spec["seed"])
    architecture = dict(run_spec["architecture"])
    dataset_cfg = dict(run_spec["dataset"])
    train_cfg = dict(run_spec["train"])

    set_determinism(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = int(dataset_cfg.get("num_workers", 4))

    train_dataset, test_dataset, y_mean, y_std = make_split_datasets(dataset_cfg)

    batch_size = int(train_cfg["batch_size"])
    train_loader = build_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, seed=seed)
    train_eval_loader = build_loader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed)
    test_loader = build_loader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed)

    arch = dict(run_spec["architecture"])

    if "sigma2_w" in train_cfg:
        arch["sigma2_w"] = train_cfg["sigma2_w"]

    model_name = str(arch["model_name"]).strip().lower()
    model = MODEL_BUILDERS[model_name](
        arch,
        seed=int(run_spec["seed"]),
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
    svd_diag_filename = str(train_cfg.get("svd_diag_filename", "svd_diagnostics.pt"))
    svd_topk_values = [int(k) for k in train_cfg.get("svd_topk", [1, 3, 5, 10])]
    if len(svd_topk_values) == 0:
        raise ValueError("svd_topk must contain at least one positive integer when provided.")
    if any(k <= 0 for k in svd_topk_values):
        raise ValueError("All svd_topk values must be strictly positive integers.")
    svd_topk_values = sorted(dict.fromkeys(svd_topk_values))

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
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Supported: 'sgd', 'adam'.")

    loss_fn = nn.MSELoss(reduction="mean")

    run_dir = output_npz.parent
    metrics_log_path = run_dir / "metrics_log.jsonl"
    if metrics_log_path.exists():
        metrics_log_path.unlink()

    svd_diag_path = run_dir / svd_diag_filename
    if save_svd_diagnostics and svd_diag_path.exists():
        svd_diag_path.unlink()

    epoch_list: List[int] = []
    train_loss_list: List[float] = []
    test_loss_list: List[float] = []
    train_mae_list: List[float] = []
    test_mae_list: List[float] = []
    train_mse_years_list: List[float] = []
    test_mse_years_list: List[float] = []
    train_grad_norm_list: List[float] = []

    svd_payload: Dict[str, Any] | None = None
    if save_svd_diagnostics:
        svd_payload = {
            "run_spec_path": str(run_spec.get("run_spec_path", "")),
            "svd_topk": list(svd_topk_values),
            "epochs": [],
            "by_epoch": {},
        }

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y_standardized in train_loader:
            x = x.to(device, non_blocking=True)
            y_standardized = y_standardized.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_standardized = model(x)
            loss = loss_fn(pred_standardized, y_standardized)
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

        if epoch in eval_epochs:
            train_metrics = evaluate(model, train_eval_loader, device=device, y_mean=y_mean, y_std=y_std)
            test_metrics = evaluate(model, test_loader, device=device, y_mean=y_mean, y_std=y_std)

            train_grad_norm, grad_matrices = compute_full_train_gradient_info(
                model,
                train_eval_loader,
                device=device,
                loss_fn=loss_fn,
                capture_matrices=save_svd_diagnostics,
            )

            if save_svd_diagnostics:
                if svd_payload is None or grad_matrices is None:
                    raise RuntimeError("SVD diagnostics were requested but could not be computed.")
                epoch_key = str(int(epoch))
                svd_payload["epochs"].append(int(epoch))
                svd_payload["by_epoch"][epoch_key] = collect_svd_diagnostics(
                    model,
                    grad_matrices,
                    topk_values=svd_topk_values,
                )
                save_svd_diagnostics_file(svd_diag_path, svd_payload)

            epoch_list.append(epoch)
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
                    "train_grad_norm": train_grad_norm,
                    "train_loss_std_mse": train_metrics["loss_std_mse"],
                    "test_loss_std_mse": test_metrics["loss_std_mse"],
                    "train_mae_years": train_metrics["mae_years"],
                    "test_mae_years": test_metrics["mae_years"],
                    "train_mse_years": train_metrics["mse_years"],
                    "test_mse_years": test_metrics["mse_years"],
                    "svd_diagnostics_saved": bool(save_svd_diagnostics),
                    "svd_diag_filename": svd_diag_filename if save_svd_diagnostics else None,
                },
            )

            print(
                f"[EVAL] epoch={epoch:03d} "
                f"train_loss_std_mse={train_metrics['loss_std_mse']:.6f} "
                f"test_loss_std_mse={test_metrics['loss_std_mse']:.6f} "
                f"train_mae_years={train_metrics['mae_years']:.4f} "
                f"test_mae_years={test_metrics['mae_years']:.4f} "
                f"train_grad_norm={train_grad_norm:.6e} "
                f"svd_diagnostics={'on' if save_svd_diagnostics else 'off'}",
                flush=True,
            )

    np.savez(
        output_npz,
        seed=np.int64(seed),
        sigma2_w=np.float64(train_cfg["sigma2_w"]),
        inv_sigma_w=np.float64(1.0 / max(float(train_cfg["sigma2_w"]) ** 0.5, 1.0e-12)),
        grad_clip_norm=np.float64(-1.0 if grad_clip_norm is None else grad_clip_norm),
        lr=np.float64(lr),
        batch_size=np.int64(batch_size),
        epochs=np.int64(epochs),
        dataset_root=np.array(str(dataset_cfg["root"])),
        n_train=np.int64(len(train_dataset)),
        n_test=np.int64(len(test_dataset)),
        target_mean_years=np.float64(y_mean),
        target_std_years=np.float64(y_std),
        eval_epochs=np.asarray(epoch_list, dtype=np.int64),
        train_loss_std_mse=np.asarray(train_loss_list, dtype=np.float64),
        test_loss_std_mse=np.asarray(test_loss_list, dtype=np.float64),
        train_mae_years=np.asarray(train_mae_list, dtype=np.float64),
        test_mae_years=np.asarray(test_mae_list, dtype=np.float64),
        train_mse_years=np.asarray(train_mse_years_list, dtype=np.float64),
        test_mse_years=np.asarray(test_mse_years_list, dtype=np.float64),
        final_train_loss_std_mse=np.float64(train_loss_list[-1]),
        final_test_loss_std_mse=np.float64(test_loss_list[-1]),
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
