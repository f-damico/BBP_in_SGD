#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# Robust path setup for direct execution as:
#   python -u /path/to/repo/src/training/train_age_regression.py ...
# In that case package imports such as src.training.svd_diagnostics can fail on some
# clusters, so we also put the local src/training directory itself on sys.path.
THIS_FILE = Path(__file__).resolve()
TRAINING_DIR = THIS_FILE.parent
SRC_DIR = TRAINING_DIR.parent
REPO_ROOT = SRC_DIR.parent
for _path in (str(TRAINING_DIR), str(SRC_DIR), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from regression_datasets import make_regression_split_datasets as _make_regression_split_datasets  # type: ignore
except Exception:
    try:
        from src.training.regression_datasets import make_regression_split_datasets as _make_regression_split_datasets
    except Exception:
        try:
            from training.regression_datasets import make_regression_split_datasets as _make_regression_split_datasets  # type: ignore
        except Exception:
            _make_regression_split_datasets = None  # type: ignore

try:
    from svd_diagnostics import (  # type: ignore
        append_weight_svd_diagnostics,
        infer_input_shape_from_configs,
        make_initial_svd_payload,
        make_svd_config,
    )
except Exception:
    try:
        from src.training.svd_diagnostics import (
            append_weight_svd_diagnostics,
            infer_input_shape_from_configs,
            make_initial_svd_payload,
            make_svd_config,
        )
    except Exception:
        try:
            from training.svd_diagnostics import (  # type: ignore
                append_weight_svd_diagnostics,
                infer_input_shape_from_configs,
                make_initial_svd_payload,
                make_svd_config,
            )
        except Exception:
            append_weight_svd_diagnostics = None  # type: ignore
            infer_input_shape_from_configs = None  # type: ignore
            make_initial_svd_payload = None  # type: ignore
            make_svd_config = None  # type: ignore

try:
    from src.models import MODEL_BUILDERS as _REPO_MODEL_BUILDERS
except Exception:  # pragma: no cover
    _REPO_MODEL_BUILDERS = {}

try:
    from src.models.dnn_regressor import build_initialized_dnn_regressor
except Exception:  # pragma: no cover
    try:
        from models.dnn_regressor import build_initialized_dnn_regressor  # type: ignore
    except Exception:  # pragma: no cover
        build_initialized_dnn_regressor = None  # type: ignore

try:
    from src.models.simple_cnn_regressor import build_initialized_simple_cnn_regressor
except Exception:  # pragma: no cover
    try:
        from models.simple_cnn_regressor import build_initialized_simple_cnn_regressor  # type: ignore
    except Exception:  # pragma: no cover
        build_initialized_simple_cnn_regressor = None  # type: ignore


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


def _normalise_model_name(name: Any) -> str:
    return str(name).strip().lower()


def _local_model_builders() -> Dict[str, Any]:
    builders: Dict[str, Any] = dict(_REPO_MODEL_BUILDERS or {})

    if build_initialized_dnn_regressor is not None:
        builders.setdefault("dnn", build_initialized_dnn_regressor)
        builders.setdefault("mlp", build_initialized_dnn_regressor)
        builders.setdefault("dnn_regressor", build_initialized_dnn_regressor)

    if build_initialized_simple_cnn_regressor is not None:
        builders.setdefault("cnn", build_initialized_simple_cnn_regressor)
        builders.setdefault("simple_cnn", build_initialized_simple_cnn_regressor)
        builders.setdefault("simple_cnn_regressor", build_initialized_simple_cnn_regressor)

    return builders


def resolve_sigma2_and_inv_sigma(train_cfg: Dict[str, Any]) -> Tuple[float, float]:
    """Accept old hyperparameters inv_sigma_w, sigma_w, or sigma2_w."""
    if "inv_sigma_w" in train_cfg:
        inv_sigma_w = float(train_cfg["inv_sigma_w"])
        if inv_sigma_w <= 0.0:
            raise ValueError("train.inv_sigma_w must be strictly positive.")
        sigma2_w = 1.0 / (inv_sigma_w ** 2)
        return sigma2_w, inv_sigma_w

    if "sigma_w" in train_cfg:
        sigma_w = float(train_cfg["sigma_w"])
        if sigma_w <= 0.0:
            raise ValueError("train.sigma_w must be strictly positive.")
        return sigma_w ** 2, 1.0 / sigma_w

    if "sigma2_w" in train_cfg:
        sigma2_w = float(train_cfg["sigma2_w"])
        if sigma2_w <= 0.0:
            raise ValueError("train.sigma2_w must be strictly positive.")
        return sigma2_w, 1.0 / (sigma2_w ** 0.5)

    raise ValueError("train must contain one of: inv_sigma_w, sigma_w, sigma2_w.")


def choose_device(train_cfg: Mapping[str, Any]) -> torch.device:
    requested = train_cfg.get("device", None)
    if requested is not None:
        device = torch.device(str(requested))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("train.device='cuda' was requested but CUDA is not available.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _NumpyRegressionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        if X.ndim != 2:
            raise ValueError(f"Expected X with shape [n_samples, n_features], got {X.shape}.")
        if y.ndim != 2 or y.shape[1] != 1:
            raise ValueError(f"Expected y with shape [n_samples, 1], got {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have different numbers of samples: {X.shape[0]} vs {y.shape[0]}.")
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


class _StandardizedTargetDataset(Dataset):
    def __init__(self, base_dataset: Dataset, y_mean: float, y_std: float) -> None:
        self.base_dataset = base_dataset
        self.y_mean = float(y_mean)
        self.y_std = float(y_std) if float(y_std) > 0.0 else 1.0

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        x, y = self.base_dataset[index]
        return x, (y - self.y_mean) / self.y_std


class _UTKFaceAgeDataset(Dataset):
    def __init__(self, root: Path, *, image_size: int, max_samples: Any = None, shuffle_seed: int = 12345) -> None:
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:
            raise ImportError("UTKFace loading requires pillow/PIL to be installed.") from exc
        self._Image = Image
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(
                f"UTKFace dataset root not found: {self.root}. "
                "Set dataset.root to the directory containing the image files."
            )
        self.image_size = int(image_size)
        files = sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        samples = []
        for path in files:
            age = self._parse_age_from_filename(path.name)
            if age is not None:
                samples.append((path, float(age)))
        if not samples:
            raise RuntimeError(f"No UTKFace-like image files with parsable ages found in {self.root}.")
        rng = random.Random(int(shuffle_seed))
        rng.shuffle(samples)
        if max_samples is not None:
            max_samples = int(max_samples)
            if max_samples <= 0:
                raise ValueError("dataset.max_samples must be positive when provided.")
            samples = samples[:max_samples]
        self.samples = samples

    @staticmethod
    def _parse_age_from_filename(filename: str) -> int | None:
        try:
            age = int(Path(filename).stem.split("_")[0])
        except Exception:
            return None
        if age < 0 or age > 120:
            return None
        return age

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, age = self.samples[index]
        with self._Image.open(path) as img:
            img = img.convert("RGB")
            img = img.resize((self.image_size, self.image_size), resample=self._Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            arr = np.transpose(arr, (2, 0, 1))
            x = torch.from_numpy(arr)
        y = torch.tensor([age], dtype=torch.float32)
        return x, y


def _compute_target_stats(dataset: Dataset) -> Tuple[float, float]:
    ys: List[float] = []
    for _, y in dataset:
        ys.append(float(torch.as_tensor(y).view(-1)[0].item()))
    y_array = np.asarray(ys, dtype=np.float64)
    mean = float(y_array.mean())
    std = float(y_array.std())
    if std <= 0.0:
        std = 1.0
    return mean, std


def _resolve_path_from_repo(value: Any, repo_root: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def _make_utkface_split_datasets(dataset_cfg: Dict[str, Any], *, repo_root: Path) -> Tuple[Dataset, Dataset, float, float]:
    if "root" not in dataset_cfg:
        raise ValueError("UTKFace dataset requires dataset.root.")
    root = _resolve_path_from_repo(dataset_cfg["root"], repo_root)
    image_size = int(dataset_cfg.get("image_size", 64))
    split_seed = int(dataset_cfg.get("split_seed", 12345))
    train_fraction = float(dataset_cfg.get("train_fraction", 0.8))
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("dataset.train_fraction must be in (0, 1).")
    full_dataset = _UTKFaceAgeDataset(
        root=root,
        image_size=image_size,
        max_samples=dataset_cfg.get("max_samples", None),
        shuffle_seed=split_seed,
    )
    n_total = len(full_dataset)
    n_train = max(1, int(math.floor(train_fraction * n_total)))
    n_test = n_total - n_train
    if n_test <= 0:
        raise ValueError("The dataset split produced an empty test set.")
    gen = torch.Generator().manual_seed(split_seed)
    train_raw, test_raw = random_split(full_dataset, [n_train, n_test], generator=gen)
    y_mean, y_std = _compute_target_stats(train_raw)
    return (
        _StandardizedTargetDataset(train_raw, y_mean, y_std),
        _StandardizedTargetDataset(test_raw, y_mean, y_std),
        y_mean,
        y_std,
    )


def _candidate_npz_paths(dataset_cfg: Dict[str, Any], repo_root: Path) -> List[Path]:
    out: List[Path] = []
    for key in ["processed_path", "npz_path", "data_path"]:
        if dataset_cfg.get(key):
            out.append(_resolve_path_from_repo(dataset_cfg[key], repo_root))
    if dataset_cfg.get("root"):
        root = _resolve_path_from_repo(dataset_cfg["root"], repo_root)
        name = str(dataset_cfg.get("name", "")).strip().lower()
        if "superconduct" in name:
            out.extend([
                root / "processed" / "superconductivity.npz",
                root / "superconductivity_processed.npz",
                root / "superconductivity.npz",
            ])
    # Deduplicate preserving order.
    deduped: List[Path] = []
    seen = set()
    for path in out:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def _load_npz_arrays(dataset_cfg: Dict[str, Any], *, repo_root: Path) -> Tuple[np.ndarray, np.ndarray, Path]:
    candidates = _candidate_npz_paths(dataset_cfg, repo_root)
    npz_path = next((p for p in candidates if p.exists()), None)
    if npz_path is None:
        looked = "\n".join(str(p) for p in candidates) if candidates else "  <no candidate path; set dataset.processed_path>"
        raise FileNotFoundError(
            "Could not find the preprocessed regression dataset .npz. Looked for:\n" + looked
        )
    with np.load(npz_path, allow_pickle=True) as data:
        if "X" in data and "y" in data:
            # Keep X in the stored dtype here. For image datasets this is often uint8,
            # and casting the full array to float32 immediately can waste several GB.
            X = np.asarray(data["X"])
            y = np.asarray(data["y"], dtype=np.float32)
        elif "x" in data and "y" in data:
            X = np.asarray(data["x"])
            y = np.asarray(data["y"], dtype=np.float32)
        else:
            raise KeyError(f"{npz_path} must contain arrays named 'X' and 'y'.")
    if y.ndim == 1:
        y = y[:, None]
    return X, y, npz_path


def _make_npz_regression_split_datasets(dataset_cfg: Dict[str, Any], *, repo_root: Path) -> Tuple[Dataset, Dataset, float, float]:
    X, y, _ = _load_npz_arrays(dataset_cfg, repo_root=repo_root)
    max_samples = dataset_cfg.get("max_samples", None)
    split_seed = int(dataset_cfg.get("split_seed", 12345))
    train_fraction = float(dataset_cfg.get("train_fraction", 0.8))
    normalize_images = bool(dataset_cfg.get("normalize_images", False))
    # For image arrays with normalize_images=true, default to pixel rescaling rather
    # than per-pixel standardisation. You can still force standardize_x=true in JSON.
    standardize_x = bool(dataset_cfg.get("standardize_x", False if normalize_images else True))
    # Backward-compatible aliases: standardize_y and standardize_targets mean the same thing.
    standardize_y = bool(dataset_cfg.get("standardize_y", dataset_cfg.get("standardize_targets", True)))
    x_eps = float(dataset_cfg.get("x_std_eps", 1.0e-12))
    y_eps = float(dataset_cfg.get("y_std_eps", 1.0e-12))
    flatten = bool(dataset_cfg.get("flatten", False))

    if not (0.0 < train_fraction < 1.0):
        raise ValueError("dataset.train_fraction must be in (0, 1).")
    if X.ndim > 2 and flatten:
        X = X.reshape(X.shape[0], -1)
    if X.ndim != 2:
        raise ValueError(
            f"Expected tabular X with shape [n_samples, n_features], got {X.shape}. "
            "For image arrays in npz set dataset.flatten=true if you want a DNN."
        )

    n_total = int(X.shape[0])
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n_total)
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError("dataset.max_samples must be positive when provided.")
        perm = perm[:max_samples]
    X = X[perm]
    y = y[perm]

    n_total = int(X.shape[0])
    n_train = max(1, int(math.floor(train_fraction * n_total)))
    n_test = n_total - n_train
    if n_test <= 0:
        raise ValueError("The dataset split produced an empty test set.")

    X_train = X[:n_train].astype(np.float32, copy=True)
    y_train = y[:n_train].astype(np.float32, copy=True)
    X_test = X[n_train:].astype(np.float32, copy=True)
    y_test = y[n_train:].astype(np.float32, copy=True)

    if normalize_images:
        # Works for Shapes3D-style uint8 images saved either as [N,H,W,C]
        # or already flattened [N,H*W*C].
        image_scale = float(dataset_cfg.get("image_scale", 255.0))
        if image_scale <= 0.0:
            raise ValueError("dataset.image_scale must be positive.")
        # Heuristic: only divide when values look like raw image intensities.
        if X_train.size > 0 and float(np.nanmax(X_train)) > 2.0:
            X_train /= image_scale
            X_test /= image_scale
        image_center = bool(dataset_cfg.get("image_center", False))
        if image_center:
            X_train = 2.0 * X_train - 1.0
            X_test = 2.0 * X_test - 1.0

    if standardize_x:
        x_mean = X_train.mean(axis=0, keepdims=True)
        x_std = X_train.std(axis=0, keepdims=True)
        x_std = np.where(x_std < x_eps, 1.0, x_std)
        X_train = (X_train - x_mean) / x_std
        X_test = (X_test - x_mean) / x_std

    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std < y_eps:
        y_std = 1.0
    if standardize_y:
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std
    else:
        y_mean = 0.0
        y_std = 1.0

    return _NumpyRegressionDataset(X_train, y_train), _NumpyRegressionDataset(X_test, y_test), y_mean, y_std


def make_split_datasets(dataset_cfg: Dict[str, Any]) -> Tuple[Dataset, Dataset, float, float]:
    """
    Self-contained old-schema dataset factory. It does not depend on
    importing src.training.regression_datasets, so the training script works
    even if package imports are fragile on the cluster.
    """
    name = str(dataset_cfg.get("name", "UTKFace")).strip().lower()
    if name in {"utkface", "utkface_age", "utkfaceage", "age"}:
        return _make_utkface_split_datasets(dataset_cfg, repo_root=REPO_ROOT)
    if name in {"superconductivity", "superconductivty", "superconductivity_data", "superconductivty_data"}:
        return _make_npz_regression_split_datasets(dataset_cfg, repo_root=REPO_ROOT)
    if any(dataset_cfg.get(k) for k in ["processed_path", "npz_path", "data_path"]):
        return _make_npz_regression_split_datasets(dataset_cfg, repo_root=REPO_ROOT)
    raise ValueError(
        f"Unsupported dataset.name={dataset_cfg.get('name')!r}. "
        "Supported: 'UTKFace', 'superconductivity', or any .npz dataset with processed_path containing X and y."
    )

def build_loader(
    dataset: Dataset,
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
    mae_orig_sum = 0.0
    mse_orig_sum = 0.0
    n_samples = 0

    for x, y_standardized in loader:
        x = x.to(device, non_blocking=True)
        y_standardized = y_standardized.to(device, non_blocking=True)
        pred_standardized = model(x)

        mse_std_sum += float(torch.sum((pred_standardized - y_standardized) ** 2).item())

        pred_orig = pred_standardized * float(y_std) + float(y_mean)
        true_orig = y_standardized * float(y_std) + float(y_mean)
        mae_orig_sum += float(torch.sum(torch.abs(pred_orig - true_orig)).item())
        mse_orig_sum += float(torch.sum((pred_orig - true_orig) ** 2).item())
        n_samples += int(x.shape[0])

    denom = max(n_samples, 1)
    return {
        "loss_std_mse": mse_std_sum / denom,
        "mae": mae_orig_sum / denom,
        "mse": mse_orig_sum / denom,
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
        pred = model(x)
        batch_loss = loss_fn(pred, y_standardized)
        weight = float(x.shape[0]) / float(n_samples)
        (batch_loss * weight).backward()

    grad_sq_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            grad_sq_norm += float(torch.sum(grad * grad).item())

    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return grad_sq_norm ** 0.5


def maybe_log_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


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
    raise ValueError("weight_checkpoint_dtype must be one of: original, float32, float16, bfloat16.")


def _model_state_dict_cpu(model: nn.Module, *, dtype_name: str = "float32") -> Dict[str, torch.Tensor]:
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
        "description": "Model state_dict tensors saved at validation epochs on CPU. Optimizer state is not saved.",
        "run_spec_path": str(run_spec_path),
        "architecture": run_spec.get("architecture"),
        "dataset": run_spec.get("dataset"),
        "train": run_spec.get("train"),
        "seed": run_spec.get("seed"),
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
    payload.setdefault("times", []).append(time_value)
    payload.setdefault("by_time", {})[str(time_value)] = {
        "model_state_dict": _model_state_dict_cpu(model, dtype_name=dtype_name)
    }
    _atomic_torch_save(payload, path)


def _atomic_np_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp_path, **arrays)
    tmp_path.replace(path)


def _make_output_payload(
    *,
    run_spec: Dict[str, Any],
    train_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    seed: int,
    sigma2_w: float,
    inv_sigma_w: float,
    optimizer_name: str,
    lr: float,
    batch_size: int,
    epochs: int,
    grad_clip_norm: float | None,
    weight_decay: float,
    y_mean: float,
    y_std: float,
    n_train: int,
    n_test: int,
    epoch_list: List[int],
    train_loss_list: List[float],
    test_loss_list: List[float],
    train_mae_list: List[float],
    test_mae_list: List[float],
    train_mse_list: List[float],
    test_mse_list: List[float],
    train_grad_norm_list: List[float],
    save_svd_diagnostics: bool,
    svd_diag_filename: str,
    svd_diag_path: Path,
    save_weight_checkpoints: bool,
    weight_checkpoint_filename: str,
    weight_checkpoint_path: Path,
    weight_checkpoint_dtype: str,
    completed: bool,
) -> Dict[str, Any]:
    have_eval = len(epoch_list) > 0
    target_name = str(dataset_cfg.get("target", dataset_cfg.get("target_name", "target")))
    target_units = str(dataset_cfg.get("target_units", ""))

    payload: Dict[str, Any] = {
        "completed": np.bool_(completed),
        "seed": np.int64(seed),
        "sigma2_w": np.float64(sigma2_w),
        "inv_sigma_w": np.float64(inv_sigma_w),
        "grad_clip_norm": np.float64(-1.0 if grad_clip_norm is None else grad_clip_norm),
        "lr": np.float64(lr),
        "batch_size": np.int64(batch_size),
        "epochs": np.int64(epochs),
        "optimizer": np.array(str(optimizer_name)),
        "weight_decay": np.float64(weight_decay),
        "momentum": np.float64(train_cfg.get("momentum", 0.0)),
        "beta1": np.float64(train_cfg.get("beta1", 0.9)),
        "beta2": np.float64(train_cfg.get("beta2", 0.999)),
        "eps": np.float64(train_cfg.get("eps", 1.0e-8)),
        "dataset_name": np.array(str(dataset_cfg.get("name", ""))),
        "dataset_root": np.array(str(dataset_cfg.get("root", ""))),
        "target_name": np.array(target_name),
        "target_units": np.array(target_units),
        "architecture_json": np.array(json.dumps(run_spec.get("architecture", {}), sort_keys=True)),
        "dataset_json": np.array(json.dumps(dataset_cfg, sort_keys=True)),
        "train_json": np.array(json.dumps(train_cfg, sort_keys=True)),
        "n_train": np.int64(n_train),
        "n_test": np.int64(n_test),
        "target_mean": np.float64(y_mean),
        "target_std": np.float64(y_std),
        # Backwards-compatible aliases used by previous notebooks.
        "target_mean_years": np.float64(y_mean),
        "target_std_years": np.float64(y_std),
        "eval_epochs": np.asarray(epoch_list, dtype=np.int64),
        "train_loss_std_mse": np.asarray(train_loss_list, dtype=np.float64),
        "test_loss_std_mse": np.asarray(test_loss_list, dtype=np.float64),
        "train_mae": np.asarray(train_mae_list, dtype=np.float64),
        "test_mae": np.asarray(test_mae_list, dtype=np.float64),
        "train_mse": np.asarray(train_mse_list, dtype=np.float64),
        "test_mse": np.asarray(test_mse_list, dtype=np.float64),
        "train_mae_years": np.asarray(train_mae_list, dtype=np.float64),
        "test_mae_years": np.asarray(test_mae_list, dtype=np.float64),
        "train_mse_years": np.asarray(train_mse_list, dtype=np.float64),
        "test_mse_years": np.asarray(test_mse_list, dtype=np.float64),
        "train_grad_norm": np.asarray(train_grad_norm_list, dtype=np.float64),
        "save_svd_diagnostics": np.bool_(save_svd_diagnostics),
        "svd_diag_filename": np.array(svd_diag_filename),
        "svd_diag_path": np.array(str(svd_diag_path) if save_svd_diagnostics else ""),
        "save_weight_checkpoints": np.bool_(save_weight_checkpoints),
        "weight_checkpoint_filename": np.array(weight_checkpoint_filename),
        "weight_checkpoint_path": np.array(str(weight_checkpoint_path) if save_weight_checkpoints else ""),
        "weight_checkpoint_dtype": np.array(weight_checkpoint_dtype),
    }

    if have_eval:
        payload.update(
            {
                "final_train_loss_std_mse": np.float64(train_loss_list[-1]),
                "final_test_loss_std_mse": np.float64(test_loss_list[-1]),
                "final_train_mae": np.float64(train_mae_list[-1]),
                "final_test_mae": np.float64(test_mae_list[-1]),
                "final_train_mse": np.float64(train_mse_list[-1]),
                "final_test_mse": np.float64(test_mse_list[-1]),
                "final_train_mae_years": np.float64(train_mae_list[-1]),
                "final_test_mae_years": np.float64(test_mae_list[-1]),
                "final_train_mse_years": np.float64(train_mse_list[-1]),
                "final_test_mse_years": np.float64(test_mse_list[-1]),
                "final_train_grad_norm": np.float64(train_grad_norm_list[-1]),
            }
        )
    else:
        for key in [
            "final_train_loss_std_mse",
            "final_test_loss_std_mse",
            "final_train_mae",
            "final_test_mae",
            "final_train_mse",
            "final_test_mse",
            "final_train_mae_years",
            "final_test_mae_years",
            "final_train_mse_years",
            "final_test_mse_years",
            "final_train_grad_norm",
        ]:
            payload[key] = np.float64(np.nan)

    return payload


def _build_model(architecture: Dict[str, Any], *, sigma2_w: float, seed: int, device: torch.device) -> nn.Module:
    arch = dict(architecture)
    arch["sigma2_w"] = float(sigma2_w)
    model_name = _normalise_model_name(arch.get("model_name", "dnn"))
    builders = _local_model_builders()
    if model_name not in builders:
        raise ValueError(
            f"Unsupported architecture.model_name='{model_name}'. "
            f"Available builders: {sorted(builders.keys())}. "
            "For old configs use model_name='dnn' for fully connected regressors."
        )
    return builders[model_name](arch, seed=int(seed), device=device)


def train_one_run(run_spec: Dict[str, Any], output_npz: Path) -> None:
    for required_key in ["seed", "architecture", "dataset", "train"]:
        if required_key not in run_spec:
            raise ValueError(f"run_spec is missing required old-style key: {required_key!r}")

    seed = int(run_spec["seed"])
    architecture = dict(run_spec["architecture"])
    dataset_cfg = dict(run_spec["dataset"])
    train_cfg = dict(run_spec["train"])

    sigma2_w, inv_sigma_w = resolve_sigma2_and_inv_sigma(train_cfg)
    train_cfg["sigma2_w"] = float(sigma2_w)
    train_cfg["inv_sigma_w"] = float(inv_sigma_w)

    set_determinism(seed)
    device = choose_device(train_cfg)

    epochs = int(train_cfg["epochs"])
    lr = float(train_cfg["lr"])
    batch_size = int(train_cfg["batch_size"])
    if batch_size <= 0:
        raise ValueError("train.batch_size must be positive.")

    test_batch_size = int(train_cfg.get("test_batch_size", batch_size))
    num_workers = int(train_cfg.get("num_workers", dataset_cfg.get("num_workers", 0)))

    train_dataset, test_dataset, y_mean, y_std = make_split_datasets(dataset_cfg)

    train_loader = build_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, seed=seed)
    train_eval_loader = build_loader(train_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, seed=seed)
    test_loader = build_loader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, seed=seed)

    model = _build_model(architecture, sigma2_w=sigma2_w, seed=seed, device=device)

    optimizer_name = str(train_cfg.get("optimizer", "sgd")).strip().lower()
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(train_cfg.get("momentum", 0.0)),
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(float(train_cfg.get("beta1", 0.9)), float(train_cfg.get("beta2", 0.999))),
            eps=float(train_cfg.get("eps", 1.0e-8)),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Supported: sgd, adam.")

    grad_clip_norm = train_cfg.get("grad_clip_norm", None)
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)
        if grad_clip_norm <= 0.0:
            raise ValueError("train.grad_clip_norm must be strictly positive when provided.")

    eval_epochs = sorted({int(e) for e in train_cfg.get("eval_epochs", [epochs])})
    eval_epochs = [e for e in eval_epochs if 1 <= e <= epochs]
    if epochs not in eval_epochs:
        eval_epochs.append(epochs)
    eval_epochs_set = set(eval_epochs)

    save_svd_diagnostics = bool(train_cfg.get("save_svd_diagnostics", False))
    if save_svd_diagnostics and (make_svd_config is None or make_initial_svd_payload is None or append_weight_svd_diagnostics is None):
        raise ImportError("SVD diagnostics requested, but src.training.svd_diagnostics could not be imported.")

    svd_cfg = make_svd_config(train_cfg) if save_svd_diagnostics else {"diag_filename": str(train_cfg.get("svd_diag_filename", "svd_diagnostics.pt"))}
    svd_diag_filename = str(svd_cfg.get("diag_filename", train_cfg.get("svd_diag_filename", "svd_diagnostics.pt")))
    svd_input_shape = infer_input_shape_from_configs(architecture, dataset_cfg) if infer_input_shape_from_configs is not None else None

    save_weight_checkpoints = bool(train_cfg.get("save_weight_checkpoints", False))
    weight_checkpoint_filename = str(train_cfg.get("weight_checkpoint_filename", "weight_checkpoints.pt"))
    weight_checkpoint_dtype = str(train_cfg.get("weight_checkpoint_dtype", "float32"))

    loss_fn = nn.MSELoss(reduction="mean")
    run_dir = output_npz.parent
    metrics_log_path = run_dir / "metrics_log.jsonl"
    svd_diag_path = run_dir / svd_diag_filename
    weight_checkpoint_path = run_dir / weight_checkpoint_filename

    if metrics_log_path.exists():
        metrics_log_path.unlink()
    if save_svd_diagnostics and svd_diag_path.exists():
        svd_diag_path.unlink()
    if save_weight_checkpoints and weight_checkpoint_path.exists():
        weight_checkpoint_path.unlink()

    epoch_list: List[int] = []
    train_loss_list: List[float] = []
    test_loss_list: List[float] = []
    train_mae_list: List[float] = []
    test_mae_list: List[float] = []
    train_mse_list: List[float] = []
    test_mse_list: List[float] = []
    train_grad_norm_list: List[float] = []

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

    def save_current_output(*, completed: bool) -> None:
        payload = _make_output_payload(
            run_spec=run_spec,
            train_cfg=train_cfg,
            dataset_cfg=dataset_cfg,
            seed=seed,
            sigma2_w=sigma2_w,
            inv_sigma_w=inv_sigma_w,
            optimizer_name=optimizer_name,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            grad_clip_norm=grad_clip_norm,
            weight_decay=weight_decay,
            y_mean=y_mean,
            y_std=y_std,
            n_train=len(train_dataset),
            n_test=len(test_dataset),
            epoch_list=epoch_list,
            train_loss_list=train_loss_list,
            test_loss_list=test_loss_list,
            train_mae_list=train_mae_list,
            test_mae_list=test_mae_list,
            train_mse_list=train_mse_list,
            test_mse_list=test_mse_list,
            train_grad_norm_list=train_grad_norm_list,
            save_svd_diagnostics=save_svd_diagnostics,
            svd_diag_filename=svd_diag_filename,
            svd_diag_path=svd_diag_path,
            save_weight_checkpoints=save_weight_checkpoints,
            weight_checkpoint_filename=weight_checkpoint_filename,
            weight_checkpoint_path=weight_checkpoint_path,
            weight_checkpoint_dtype=weight_checkpoint_dtype,
            completed=completed,
        )
        _atomic_np_savez(output_npz, **payload)

    print(
        f"[START] seed={seed} dataset={dataset_cfg.get('name', '')} model={architecture.get('model_name', 'dnn')} "
        f"epochs={epochs} batch_size={batch_size} lr={lr} inv_sigma_w={inv_sigma_w} sigma2_w={sigma2_w} "
        f"device={device} eval_epochs={eval_epochs}",
        flush=True,
    )

    # Save a readable initial file so a killed job still leaves metadata.
    save_current_output(completed=False)

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))

            optimizer.step()

        if epoch in eval_epochs_set:
            train_metrics = evaluate(model, train_eval_loader, device=device, y_mean=y_mean, y_std=y_std)
            test_metrics = evaluate(model, test_loader, device=device, y_mean=y_mean, y_std=y_std)
            train_grad_norm = compute_full_train_gradient_norm(model, train_eval_loader, device=device, loss_fn=loss_fn)

            if save_svd_diagnostics:
                if svd_payload is None or append_weight_svd_diagnostics is None:
                    raise RuntimeError("SVD diagnostics requested but not initialised.")
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
                    raise RuntimeError("Weight checkpoints requested but not initialised.")
                append_weight_checkpoint(
                    path=weight_checkpoint_path,
                    payload=weight_checkpoint_payload,
                    model=model,
                    time_value=epoch,
                    dtype_name=weight_checkpoint_dtype,
                )

            epoch_list.append(int(epoch))
            train_loss_list.append(float(train_metrics["loss_std_mse"]))
            test_loss_list.append(float(test_metrics["loss_std_mse"]))
            train_mae_list.append(float(train_metrics["mae"]))
            test_mae_list.append(float(test_metrics["mae"]))
            train_mse_list.append(float(train_metrics["mse"]))
            test_mse_list.append(float(test_metrics["mse"]))
            train_grad_norm_list.append(float(train_grad_norm))

            maybe_log_jsonl(
                metrics_log_path,
                {
                    "epoch": int(epoch),
                    "seed": int(seed),
                    "lr": float(lr),
                    "inv_sigma_w": float(inv_sigma_w),
                    "sigma2_w": float(sigma2_w),
                    "batch_size": int(batch_size),
                    "optimizer": str(optimizer_name),
                    "train_loss_std_mse": float(train_metrics["loss_std_mse"]),
                    "test_loss_std_mse": float(test_metrics["loss_std_mse"]),
                    "train_mae": float(train_metrics["mae"]),
                    "test_mae": float(test_metrics["mae"]),
                    "train_mse": float(train_metrics["mse"]),
                    "test_mse": float(test_metrics["mse"]),
                    "train_grad_norm": float(train_grad_norm),
                    "svd_diagnostics_saved": bool(save_svd_diagnostics),
                    "svd_diag_filename": svd_diag_filename if save_svd_diagnostics else None,
                    "svd_diag_path": str(svd_diag_path) if save_svd_diagnostics else None,
                    "weight_checkpoints_saved": bool(save_weight_checkpoints),
                    "weight_checkpoint_filename": weight_checkpoint_filename if save_weight_checkpoints else None,
                    "weight_checkpoint_path": str(weight_checkpoint_path) if save_weight_checkpoints else None,
                    "weight_checkpoint_dtype": weight_checkpoint_dtype if save_weight_checkpoints else None,
                    "completed": bool(epoch == epochs),
                },
            )

            save_current_output(completed=(epoch == epochs))

            print(
                f"[EVAL] epoch={epoch:04d} train_loss_std_mse={train_metrics['loss_std_mse']:.8e} "
                f"test_loss_std_mse={test_metrics['loss_std_mse']:.8e} "
                f"train_mae={train_metrics['mae']:.6e} test_mae={test_metrics['mae']:.6e} "
                f"train_grad_norm={train_grad_norm:.6e} "
                f"svd={'on' if save_svd_diagnostics else 'off'} weights={'on' if save_weight_checkpoints else 'off'}",
                flush=True,
            )

    # Ensure a completed output exists even if eval_epochs somehow did not include epochs.
    if not epoch_list or epoch_list[-1] != epochs:
        train_metrics = evaluate(model, train_eval_loader, device=device, y_mean=y_mean, y_std=y_std)
        test_metrics = evaluate(model, test_loader, device=device, y_mean=y_mean, y_std=y_std)
        train_grad_norm = compute_full_train_gradient_norm(model, train_eval_loader, device=device, loss_fn=loss_fn)
        epoch_list.append(int(epochs))
        train_loss_list.append(float(train_metrics["loss_std_mse"]))
        test_loss_list.append(float(test_metrics["loss_std_mse"]))
        train_mae_list.append(float(train_metrics["mae"]))
        test_mae_list.append(float(test_metrics["mae"]))
        train_mse_list.append(float(train_metrics["mse"]))
        test_mse_list.append(float(test_metrics["mse"]))
        train_grad_norm_list.append(float(train_grad_norm))

    save_current_output(completed=True)
    print(f"[DONE] output_npz={output_npz}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one old-schema experiment-2 regression run.")
    parser.add_argument("--run_spec", type=str, required=True)
    parser.add_argument("--output_npz", type=str, required=True)
    args = parser.parse_args()

    run_spec_path = Path(args.run_spec).expanduser().resolve()
    output_npz = Path(args.output_npz).expanduser().resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    run_spec = load_json(run_spec_path)
    run_spec["run_spec_path"] = str(run_spec_path)
    train_one_run(run_spec, output_npz)


if __name__ == "__main__":
    main()
