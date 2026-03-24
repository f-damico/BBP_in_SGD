#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Union
import json

import torch
import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "linear": nn.Identity,
    "identity": nn.Identity,
    None: nn.Identity,
}

def _canonical_activation_name(name: str | None) -> str | None:
    if name is None:
        return None
    return str(name).strip().lower()

def build_activation(name: str | None) -> nn.Module:
    key = _canonical_activation_name(name)
    if key not in _ACTIVATIONS:
        raise ValueError("Unsupported activation. Supported: relu, tanh, linear")
    return _ACTIVATIONS[key]()



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


def set_seed_everywhere(seed: int) -> None:
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _fan_in_from_module(module: nn.Module) -> int:
    if isinstance(module, nn.Conv2d):
        k_h, k_w = module.kernel_size
        return int(module.in_channels * k_h * k_w)
    if isinstance(module, nn.Linear):
        return int(module.in_features)
    raise TypeError(f"Unsupported module type for custom init: {type(module)!r}")


def init_module_normal_scaled(module: nn.Module, sigma2_w: float) -> None:
    if sigma2_w < 0:
        raise ValueError("sigma2_w must be non-negative.")

    fan_in = _fan_in_from_module(module)
    std = (float(sigma2_w) / float(fan_in)) ** 0.5

    with torch.no_grad():
        module.weight.normal_(mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            module.bias.zero_()


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation_name: str = "relu",
        use_bias: bool = True,
    ) -> None:
        super().__init__()

        if activation_name.lower() != "relu":
            raise ValueError("This simple CNN currently supports activation='relu' only.")

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=use_bias)
        self.activation = build_activation(activation_name)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class SimpleCNNRegressor(nn.Module):
    """
    Very standard small CNN for image-to-scalar regression:
    conv -> relu -> pool repeated a few times,
    then global average pooling,
    then a small MLP head.
    """

    def __init__(
        self,
        input_channels: int,
        conv_channels: Sequence[int],
        mlp_hidden_dims: Sequence[int],
        output_dim: int,
        *,
        activation: str = "relu",
        bias: bool = True,
    ) -> None:
        super().__init__()

        if input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if len(conv_channels) == 0:
            raise ValueError("conv_channels must contain at least one block.")
        if any(c <= 0 for c in conv_channels):
            raise ValueError("All conv channel sizes must be positive.")
        if any(h <= 0 for h in mlp_hidden_dims):
            raise ValueError("All mlp hidden dims must be positive.")

        self.input_channels = int(input_channels)
        self.conv_channels = [int(c) for c in conv_channels]
        self.mlp_hidden_dims = [int(h) for h in mlp_hidden_dims]
        self.output_dim = int(output_dim)
        self.activation = str(activation)
        self.bias = bool(bias)

        blocks: List[nn.Module] = []
        in_ch = self.input_channels
        for out_ch in self.conv_channels:
            blocks.append(
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    activation_name=self.activation,
                    use_bias=self.bias,
                )
            )
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        mlp_dims = [self.conv_channels[-1]] + self.mlp_hidden_dims + [self.output_dim]
        head_layers: List[nn.Module] = []
        for idx in range(len(mlp_dims) - 1):
            head_layers.append(nn.Linear(mlp_dims[idx], mlp_dims[idx + 1], bias=self.bias))
            if idx < len(mlp_dims) - 2:
                head_layers.append(nn.ReLU(inplace=False))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.global_pool(h)
        h = torch.flatten(h, start_dim=1)
        out = self.head(h)
        return out

    def architecture_dict(self) -> Dict[str, Any]:
        return {
            "model_name": "cnn",
            "input_channels": self.input_channels,
            "conv_channels": list(self.conv_channels),
            "mlp_hidden_dims": list(self.mlp_hidden_dims),
            "output_dim": self.output_dim,
            "activation": self.activation_name,
            "bias": self.bias,
        }

    def initialize_all_layers(self, sigma2_w: float, seed: int) -> None:
        set_seed_everywhere(seed)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init_module_normal_scaled(module, sigma2_w=sigma2_w)


def build_simple_cnn_regressor_from_architecture(
    arch: Dict[str, Any],
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SimpleCNNRegressor:
    model = SimpleCNNRegressor(
        input_channels=int(arch["input_channels"]),
        conv_channels=[int(c) for c in arch["conv_channels"]],
        mlp_hidden_dims=[int(h) for h in arch["mlp_hidden_dims"]],
        output_dim=int(arch.get("output_dim", 1)),
        activation=str(arch.get("activation", "relu")),
        bias=bool(arch.get("bias", True)),
    )
    return model.to(device=torch.device(device), dtype=dtype)


def initialize_model_from_architecture(
    model: SimpleCNNRegressor,
    arch: Dict[str, Any],
    *,
    seed: int | None = None,
) -> SimpleCNNRegressor:
    sigma2_w = float(arch.get("sigma2_w", 1.0))
    model.initialize_all_layers(sigma2_w=sigma2_w, seed=seed)
    return model


def build_initialized_simple_cnn_regressor(
    arch: Dict[str, Any],
    *,
    seed: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SimpleCNNRegressor:
    model = build_simple_cnn_regressor_from_architecture(arch, device=device, dtype=dtype)
    initialize_model_from_architecture(model, arch, seed=seed)
    return model
