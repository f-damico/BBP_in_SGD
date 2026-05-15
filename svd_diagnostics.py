#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

import torch
import torch.nn as nn


def _tensor_to_float_list(x: torch.Tensor) -> List[float]:
    return x.detach().cpu().numpy().astype("float64", copy=False).tolist()


def make_svd_config(train_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Normalise the user-facing SVD diagnostic options.

    Current behaviour:
      - compute the full SVD spectrum of every nn.Linear weight matrix;
      - save all singular values;
      - save all eigenvalues of W W^T, i.e. singular_values**2;
      - ignore all convolutional layers completely.

    There is no top-k option: the full available spectrum is always saved.
    """
    dtype_name = str(train_cfg.get("svd_compute_dtype", "float32")).strip().lower()
    if dtype_name in {"float", "fp32", "float32", "torch.float32"}:
        dtype = torch.float32
        dtype_name = "float32"
    elif dtype_name in {"double", "fp64", "float64", "torch.float64"}:
        dtype = torch.float64
        dtype_name = "float64"
    else:
        raise ValueError("svd_compute_dtype must be 'float32' or 'float64'.")

    return {
        "diag_filename": str(train_cfg.get("svd_diag_filename", "svd_diagnostics.pt")),
        "compute_dtype": dtype,
        "compute_dtype_name": dtype_name,
        "included_module_types": ["Linear"],
        "ignored_module_types": ["Conv2d"],
        "spectrum": "full",
        "eigenvalue_definition": "eigenvalues of W @ W.T = singular_values**2",
    }


def infer_input_shape_from_configs(
    architecture: Mapping[str, Any],
    dataset_cfg: Mapping[str, Any] | None = None,
) -> Optional[Tuple[int, int, int]]:
    """
    Kept for backward compatibility with the training scripts.

    Since Conv2d layers are now ignored by SVD diagnostics, no input shape is
    needed to construct any dense convolutional operator.
    """
    return None


def _spectra_from_matrix(matrix: torch.Tensor) -> Dict[str, Any]:
    matrix = matrix.detach().cpu()
    singular_values = torch.linalg.svdvals(matrix)
    gram_eigenvalues = singular_values.square()
    return {
        "matrix_shape": [int(v) for v in matrix.shape],
        "num_singular_values": int(singular_values.numel()),
        "num_gram_eigenvalues": int(gram_eigenvalues.numel()),
        "singular_values": _tensor_to_float_list(singular_values),
        "gram_eigenvalues": _tensor_to_float_list(gram_eigenvalues),
        "spectral_norm": float(singular_values.max().item()) if singular_values.numel() else float("nan"),
        "frobenius_norm": float(torch.linalg.vector_norm(matrix).item()),
    }


def collect_weight_svd_diagnostics(
    model: nn.Module,
    *,
    input_shape: Optional[Tuple[int, int, int]] = None,
    svd_config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Collect the full SVD/eigenvalue spectrum for every Linear weight matrix.

    Conv2d layers are intentionally ignored. We do not flatten convolutional
    filters, and we do not construct the dense convolution-as-linear-operator
    matrix, because that is too expensive for the CNN runs.
    """
    cfg = dict(svd_config or make_svd_config({}))
    compute_dtype = cfg.get("compute_dtype", torch.float32)

    out: Dict[str, Any] = {
        "layer_order": [],
        "layers": {},
        "ignored_layers": {},
    }

    for layer_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_matrix = module.weight.detach().to(device="cpu", dtype=compute_dtype)
            out["layer_order"].append(layer_name)
            out["layers"][layer_name] = {
                "module_type": "Linear",
                "operator_kind": "weight_matrix",
                "weight_shape": [int(v) for v in module.weight.shape],
                **_spectra_from_matrix(weight_matrix),
            }

        elif isinstance(module, nn.Conv2d):
            out["ignored_layers"][layer_name] = {
                "module_type": "Conv2d",
                "operator_kind": "ignored",
                "weight_shape": [int(v) for v in module.weight.shape],
                "reason": "Conv2d layers are ignored by the SVD diagnostics.",
            }

    return out


def make_initial_svd_payload(
    *,
    run_spec_path: str = "",
    time_key: str,
    input_shape: Optional[Tuple[int, int, int]],
    svd_config: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "format_version": 3,
        "description": (
            "Full layer spectra saved at evaluation times. Only Linear layers are included. "
            "Conv2d layers, when present, are listed under ignored_layers and are not used for SVD. "
            "gram_eigenvalues are singular_values**2, i.e. eigenvalues of W @ W.T."
        ),
        "run_spec_path": str(run_spec_path),
        "time_key": str(time_key),
        "input_shape": None,
        "svd_config": {
            k: (str(v) if isinstance(v, torch.dtype) else v)
            for k, v in dict(svd_config).items()
            if k != "compute_dtype"
        },
        "times": [],
        "by_time": {},
    }


def append_weight_svd_diagnostics(
    *,
    path: Path,
    payload: MutableMapping[str, Any],
    model: nn.Module,
    time_value: int,
    input_shape: Optional[Tuple[int, int, int]],
    svd_config: Mapping[str, Any],
) -> None:
    time_value = int(time_value)
    time_key = str(time_value)
    record = collect_weight_svd_diagnostics(
        model,
        input_shape=None,
        svd_config=svd_config,
    )
    payload.setdefault("times", []).append(time_value)
    payload.setdefault("by_time", {})[time_key] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)


def load_svd_diagnostics(path: str | Path) -> Dict[str, Any]:
    return torch.load(Path(path), map_location="cpu")


def available_svd_times(payload: Mapping[str, Any]) -> List[int]:
    return [int(t) for t in payload.get("times", [])]


def available_svd_layers(payload: Mapping[str, Any], time_value: int | None = None) -> List[str]:
    times = available_svd_times(payload)
    if time_value is None:
        if not times:
            return []
        time_value = times[0]
    record = payload.get("by_time", {}).get(str(int(time_value)), {})
    return list(record.get("layer_order", []))


def get_layer_gram_eigenvalues(
    payload: Mapping[str, Any],
    *,
    time_value: int,
    layer_name: str,
) -> List[float]:
    record = payload["by_time"][str(int(time_value))]
    return list(record["layers"][layer_name]["gram_eigenvalues"])


def get_layer_singular_values(
    payload: Mapping[str, Any],
    *,
    time_value: int,
    layer_name: str,
) -> List[float]:
    record = payload["by_time"][str(int(time_value))]
    return list(record["layers"][layer_name]["singular_values"])
