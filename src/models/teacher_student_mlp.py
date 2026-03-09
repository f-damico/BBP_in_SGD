#!/usr/bin/env python3
"""
Teacher-student MLP model definition for the phase-diagram experiments.

Design goals:
- simple and close to the paper setup;
- configurable widths through input_dim / hidden_dims / output_dim;
- fixed activations: tanh on all hidden layers, linear output;
- no bias terms;
- weight initialisation: W_ij ~ N(0, sigma2_w / n_in);
- easy construction of:
    1) a TEACHER with all layers sampled,
    2) a STUDENT with first/last layers copied from teacher and selected layers trainable.

For the paper-reproduction experiment, the intended use is:
- same architecture for teacher and student;
- only the middle layer trainable;
- first and last layers fixed to the teacher values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Union
import json

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Initialisation helper
# -----------------------------------------------------------------------------

def init_linear_normal_scaled(
    linear: nn.Linear,
    sigma2_w: float,
    generator=None,
) -> None:
    """
    Initialise a linear layer with
        W_ij ~ N(0, sigma2_w / n_in)
    and no bias.

    The 'generator' argument is kept only for compatibility with the rest
    of the codebase, but it is intentionally ignored to avoid CPU/CUDA
    generator mismatch errors on GPU.
    """
    if linear.bias is not None:
        raise ValueError("This model is defined with bias=False only.")
    if sigma2_w < 0:
        raise ValueError("sigma2_w must be non-negative.")

    n_in = linear.in_features
    std = (float(sigma2_w) / float(n_in)) ** 0.5
    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=std)


def set_seed_everywhere(seed: int) -> None:
    """
    Set PyTorch RNG seed for both CPU and CUDA.
    """
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------

class TeacherStudentMLP(nn.Module):
    """
    MLP with:
    - configurable widths,
    - tanh after every hidden linear layer,
    - linear output,
    - no biases.

    The linear layers are stored in self.linears.
    Layer index convention:
        0, 1, ..., L-1   where L = number of linear layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
    ):
        super().__init__()

        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one hidden layer.")
        if any(h <= 0 for h in hidden_dims):
            raise ValueError("All hidden layer sizes must be positive.")

        self.input_dim = int(input_dim)
        self.hidden_dims = [int(h) for h in hidden_dims]
        self.output_dim = int(output_dim)

        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(len(dims) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for idx, layer in enumerate(self.linears):
            h = layer(h)
            if idx < len(self.linears) - 1:
                h = torch.tanh(h)
        return h

    @property
    def num_linear_layers(self) -> int:
        return len(self.linears)

    def architecture_dict(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "output_dim": self.output_dim,
        }

    def initialize_all_layers(
        self,
        sigma2_w: float,
        seed: int,
    ) -> None:
        """
        Initialise every linear layer with:
            W_ij ~ N(0, sigma2_w / n_in).
        """
        set_seed_everywhere(seed)
        for layer in self.linears:
            init_linear_normal_scaled(layer, sigma2_w=sigma2_w)

    def copy_all_layers_from(self, other: "TeacherStudentMLP") -> None:
        self._check_same_architecture(other)
        with torch.no_grad():
            for self_layer, other_layer in zip(self.linears, other.linears):
                self_layer.weight.copy_(other_layer.weight)

    def copy_layer_from(self, other: "TeacherStudentMLP", layer_idx: int) -> None:
        self._check_same_architecture(other)
        with torch.no_grad():
            self.linears[layer_idx].weight.copy_(other.linears[layer_idx].weight)

    def initialize_selected_layers(
        self,
        layer_indices: Iterable[int],
        sigma2_w: float,
        seed: int,
    ) -> None:
        set_seed_everywhere(seed)
        for idx in layer_indices:
            init_linear_normal_scaled(
                self.linears[idx],
                sigma2_w=sigma2_w,
            )

    def set_trainable_layers(self, layer_indices: Iterable[int]) -> None:
        layer_indices = set(int(i) for i in layer_indices)
        for idx, layer in enumerate(self.linears):
            requires_grad = idx in layer_indices
            layer.weight.requires_grad_(requires_grad)

    def freeze_all_layers(self) -> None:
        self.set_trainable_layers([])

    def _check_same_architecture(self, other: "TeacherStudentMLP") -> None:
        if self.architecture_dict() != other.architecture_dict():
            raise ValueError("Teacher and student must have the same architecture.")


# -----------------------------------------------------------------------------
# Builders used by experiment/training files
# -----------------------------------------------------------------------------

def build_model_from_config(
    config_path: Union[str, Path],
    *,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> TeacherStudentMLP:
    cfg = load_json_or_yaml(config_path)
    arch = cfg["architecture"]
    model = TeacherStudentMLP(
        input_dim=int(arch["input_dim"]),
        hidden_dims=[int(h) for h in arch["hidden_dims"]],
        output_dim=int(arch["output_dim"]),
    )
    return model.to(device=torch.device(device), dtype=dtype)


def build_teacher_from_config(
    config_path: Union[str, Path],
    *,
    seed: int,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> TeacherStudentMLP:
    """
    Build the teacher using the architecture from config and
    teacher variance sigma2_w from config['teacher']['sigma2_w'].
    """
    cfg = load_json_or_yaml(config_path)
    sigma2_w_teacher = float(cfg.get("teacher", {}).get("sigma2_w", 1.0))

    teacher = build_model_from_config(config_path, device=device, dtype=dtype)
    teacher.initialize_all_layers(sigma2_w=sigma2_w_teacher, seed=seed)
    teacher.eval()
    return teacher


def build_student_from_teacher(
    teacher: TeacherStudentMLP,
    *,
    trainable_layer_indices: Sequence[int],
    sigma2_w_trainable: float,
    seed: int,
) -> TeacherStudentMLP:
    """
    Build a student with the same architecture as the teacher.

    Procedure:
    1) copy all teacher layers;
    2) re-initialise only the selected trainable layers using
           W_ij ~ N(0, sigma2_w_trainable / n_in);
    3) freeze all other layers.

    For the paper experiment with 3 linear layers, use:
        trainable_layer_indices = [1]
    so that the first and last layers are fixed to the teacher and only the
    middle one is trained.
    """
    student = TeacherStudentMLP(
        input_dim=teacher.input_dim,
        hidden_dims=teacher.hidden_dims,
        output_dim=teacher.output_dim,
    ).to(device=next(teacher.parameters()).device, dtype=next(teacher.parameters()).dtype)

    student.copy_all_layers_from(teacher)
    student.initialize_selected_layers(
        layer_indices=trainable_layer_indices,
        sigma2_w=sigma2_w_trainable,
        seed=seed,
    )
    student.set_trainable_layers(trainable_layer_indices)
    return student


# -----------------------------------------------------------------------------
# Small helpers for the exact paper setup
# -----------------------------------------------------------------------------

def get_middle_layer_index(model: TeacherStudentMLP) -> int:
    """
    Return the central linear-layer index.

    For the exact paper setup (3 linear layers), this is 1.
    """
    if model.num_linear_layers % 2 == 0:
        raise ValueError(
            "Middle layer is only uniquely defined for an odd number of linear layers."
        )
    return model.num_linear_layers // 2


def build_paper_student_from_teacher(
    teacher: TeacherStudentMLP,
    *,
    sigma2_w_middle: float,
    seed: int,
) -> TeacherStudentMLP:
    """
    Convenience wrapper for the exact experiment in the paper:
    only the middle layer is trainable.
    """
    middle_idx = get_middle_layer_index(teacher)
    return build_student_from_teacher(
        teacher,
        trainable_layer_indices=[middle_idx],
        sigma2_w_trainable=sigma2_w_middle,
        seed=seed,
    )
