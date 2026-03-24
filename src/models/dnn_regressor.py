from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn as nn


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "identity": nn.Identity,
    "linear": nn.Identity,
    None: nn.Identity,
}


def set_seed_everywhere(seed: int) -> None:
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



def init_linear_normal_scaled(linear: nn.Linear, sigma2_w: float) -> None:
    """
    Initialize W_ij ~ N(0, sigma2_w / n_in).
    Bias is initialized to zero when present.
    """
    if sigma2_w < 0:
        raise ValueError("sigma2_w must be non-negative.")

    n_in = int(linear.in_features)
    std = (float(sigma2_w) / float(n_in)) ** 0.5
    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=std)
        if linear.bias is not None:
            linear.bias.zero_()



def _canonical_activation_name(name: str | None) -> str | None:
    if name is None:
        return None
    return str(name).strip().lower()



def build_activation(name: str | None) -> nn.Module:
    key = _canonical_activation_name(name)
    if key not in _ACTIVATIONS:
        supported = ", ".join(sorted(k for k in _ACTIVATIONS if isinstance(k, str)))
        raise ValueError(f"Unsupported activation '{name}'. Supported: {supported}")
    return _ACTIVATIONS[key]()



def _resolve_hidden_dims(
    *,
    hidden_dims: Sequence[int] | None,
    hidden_dim: int | None,
    num_hidden_layers: int | None,
    num_layers: int | None,
) -> List[int]:
    """
    Accepted patterns:
    1) hidden_dims = [128, 128, 64]
    2) hidden_dim = 128 and num_hidden_layers = 3
    3) hidden_dim = 128 and num_layers = 4  -> interpreted as total linear layers,
       hence num_hidden_layers = num_layers - 1 for a regression network with one output layer.
    """
    if hidden_dims is not None:
        dims = [int(h) for h in hidden_dims]
        if len(dims) == 0:
            raise ValueError("hidden_dims cannot be empty.")
        if any(h <= 0 for h in dims):
            raise ValueError("All hidden_dims must be positive.")
        return dims

    if hidden_dim is None:
        raise ValueError(
            "You must provide either 'hidden_dims' or the pair ('hidden_dim', 'num_hidden_layers')."
        )

    if num_hidden_layers is None:
        if num_layers is None:
            raise ValueError(
                "When 'hidden_dims' is not provided, you must provide either 'num_hidden_layers' or 'num_layers'."
            )
        num_hidden_layers = int(num_layers) - 1

    hidden_dim = int(hidden_dim)
    num_hidden_layers = int(num_hidden_layers)

    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive.")
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive.")

    return [hidden_dim] * num_hidden_layers


class DNNRegressor(nn.Module):
    """
    Fully-connected regression network with:
    - configurable depth,
    - configurable width,
    - configurable activation,
    - linear output layer,
    - optional bias,
    - controllable Gaussian variance initialization.

    The final layer has no activation, so this is suitable for regression.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dims: Sequence[int] | None = None,
        hidden_dim: int | None = None,
        num_hidden_layers: int | None = None,
        num_layers: int | None = None,
        activation: str = "relu",
        bias: bool = True,
    ) -> None:
        super().__init__()

        input_dim = int(input_dim)
        output_dim = int(output_dim)
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = _resolve_hidden_dims(
            hidden_dims=hidden_dims,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            num_layers=num_layers,
        )
        self.activation_name = _canonical_activation_name(activation) or "identity"
        self.bias = bool(bias)

        dims = [self.input_dim] + list(self.hidden_dims) + [self.output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=self.bias) for i in range(len(dims) - 1)]
        )
        self.activation = build_activation(self.activation_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = torch.flatten(x, start_dim=1)
    
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    @property
    def num_linear_layers(self) -> int:
        return len(self.layers)

    @property
    def num_hidden_layers(self) -> int:
        return len(self.hidden_dims)

    def architecture_dict(self) -> Dict[str, Any]:
        return {
            "model_name": "dnn_regressor",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation_name,
            "bias": self.bias,
        }

    def initialize_all_layers(self, sigma2_w: float, seed: int | None = None) -> None:
        if seed is not None:
            set_seed_everywhere(seed)
        for layer in self.layers:
            init_linear_normal_scaled(layer, sigma2_w=sigma2_w)



def build_dnn_regressor_from_architecture(
    arch: Dict[str, Any],
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> DNNRegressor:
    model = DNNRegressor(
        input_dim=int(arch["input_dim"]),
        output_dim=int(arch.get("output_dim", 1)),
        hidden_dims=arch.get("hidden_dims"),
        hidden_dim=arch.get("hidden_dim"),
        num_hidden_layers=arch.get("num_hidden_layers"),
        num_layers=arch.get("num_layers"),
        activation=str(arch.get("activation", "relu")),
        bias=bool(arch.get("bias", True)),
    )
    return model.to(device=torch.device(device), dtype=dtype)



def initialize_model_from_architecture(
    model: DNNRegressor,
    arch: Dict[str, Any],
    *,
    seed: int | None = None,
) -> DNNRegressor:
    sigma2_w = float(arch.get("sigma2_w", 1.0))
    model.initialize_all_layers(sigma2_w=sigma2_w, seed=seed)
    return model



def build_initialized_dnn_regressor(
    arch: Dict[str, Any],
    *,
    seed: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> DNNRegressor:
    model = build_dnn_regressor_from_architecture(arch, device=device, dtype=dtype)
    initialize_model_from_architecture(model, arch, seed=seed)
    return model
