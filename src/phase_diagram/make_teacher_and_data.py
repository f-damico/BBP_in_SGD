#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import json

import torch
import torch.nn as nn


# Load the file config
def load_json_or_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
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


def _make_generator(device: Union[str, torch.device], seed: int) -> torch.Generator:
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


# Teacher model builder 
def _init_linear_normal_scaled(linear: nn.Linear, sigma2_w: float, gen: torch.Generator) -> None:
    """
    Initialize: correct variance, no bias.
    """
    if linear.bias is not None:
        raise ValueError("This experiment assumes no bias. Use bias=False.")
    n_in = linear.in_features
    std = (sigma2_w / float(n_in)) ** 0.5
    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=std, generator=gen)


class _MLPNoBiasTanh(nn.Module):
    """
    Simple MLP with tanh between Linear layers, last layer linear output, No biases   
    """
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
        self.linears = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for li, linear in enumerate(self.linears):
            h = linear(h)
            if li < len(self.linears) - 1:
                h = torch.tanh(h)
        return h


def build_teacher(
    *,
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    sigma2_w_teacher: float,
    seed_teacher: int,
    device: Union[str, torch.device],
    dtype: torch.dtype,
) -> nn.Module:
    
    device = torch.device(device)

    teacher = _MLPNoBiasTanh(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
    ).to(device=device, dtype=dtype)

    gen = _make_generator(device=device, seed=seed_teacher)

    for linear in teacher.linears:
        _init_linear_normal_scaled(
            linear,
            sigma2_w=float(sigma2_w_teacher),
            gen=gen,
        )

    teacher.eval()
    return teacher


# Data generation
@dataclass(frozen=True)
class TeacherData:
    teacher: nn.Module
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor


def make_teacher_and_data(
    config_path: Union[str, Path],
    *,
    seed: int,
    device_teacher: Union[str, torch.device] = "cuda",
    device_data: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float32,
) -> TeacherData:
    """
    Build teacher + dataset using a config .json
    """
    cfg = load_json_or_yaml(config_path)

    arch = cfg.get("architecture", {})
    ds = cfg.get("dataset", {})
    teacher_cfg = cfg.get("teacher", {})

    input_dim = int(arch["input_dim"])
    hidden_dims = [int(x) for x in arch["hidden_dims"]]
    output_dim = int(arch["output_dim"])

    n_train = int(ds["n_train"])
    n_test = int(ds["n_test"])

    sigma2_w_teacher = float(teacher_cfg.get("sigma2_w", 1.0))

    x_mean = float(ds.get("x_mean", 0.0))
    x_std = float(ds.get("x_std", 1.0))

    device_teacher = torch.device(device_teacher)
    device_data = torch.device(device_data)

    teacher = build_teacher(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        sigma2_w_teacher=sigma2_w_teacher,
        seed_teacher=int(seed),
        device=device_teacher,
        dtype=dtype,
    )

    gen_data = _make_generator(device=device_data, seed=int(seed))

    X_train = (
        torch.randn(
            (n_train, input_dim),
            generator=gen_data,
            dtype=dtype,
            device=device_data,
        ) * x_std + x_mean
    )
    X_test = (
        torch.randn(
            (n_test, input_dim),
            generator=gen_data,
            dtype=dtype,
            device=device_data,
        ) * x_std + x_mean
    )

    # Teacher forward for labels
    with torch.no_grad():
        teacher_device = next(teacher.parameters()).device

        X_train_for_teacher = X_train if X_train.device == teacher_device else X_train.to(teacher_device)
        X_test_for_teacher = X_test if X_test.device == teacher_device else X_test.to(teacher_device)

        y_train = teacher(X_train_for_teacher)
        y_test = teacher(X_test_for_teacher)

        if y_train.device != device_data:
            y_train = y_train.to(device_data)
        if y_test.device != device_data:
            y_test = y_test.to(device_data)

    return TeacherData(
        teacher=teacher,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def save_dataset_npz(path: Union[str, Path], data: TeacherData) -> None:
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(path),
        X_train=data.X_train.detach().cpu().numpy(),
        y_train=data.y_train.detach().cpu().numpy(),
        X_test=data.X_test.detach().cpu().numpy(),
        y_test=data.y_test.detach().cpu().numpy(),
    )