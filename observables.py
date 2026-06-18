from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# CLI arguments
# -----------------------------------------------------------------------------


def add_observation_args(parent_parser):
    group = parent_parser.add_argument_group("Observation & Logging")
    group.add_argument(
        "--output_filename_prefix", type=str, default="", help="Prefix for output files"
    )
    group.add_argument(
        "--observe_num_saves",
        type=int,
        default=100,
        help=(
            "Number of points at which to compute train/test observables and save "
            "weights. By default these points are completed epochs, approximately "
            "logarithmically spaced from 1 to Nepoch and always including Nepoch. "
            "With --observe_by_step, these points are optimizer steps, approximately "
            "logarithmically spaced from step 1 to the final training step."
        ),
    )
    group.add_argument(
        "--observe_initial",
        action="store_true",
        help="Also compute observables and save a checkpoint at epoch 0 before training.",
    )

    group.add_argument(
        "--observe_by_step",
        action="store_true",
        help=(
            "Use optimizer steps, instead of completed epochs, as the main "
            "observation/checkpoint schedule. Then --observe_num_saves means the "
            "number of log-spaced steps between 1 and the final training step."
        ),
    )
    # Low-level legacy step-grid knobs. They are no longer needed for the main
    # --observe_by_step mode, which uses --observe_num_saves, but are kept so old
    # command lines do not break.
    group.add_argument("--observe_epoch_geom_factor", type=float, default=1.5)
    group.add_argument("--observe_epoch_min", type=int, default=1)
    group.add_argument("--observe_epoch_linear_N", type=int, default=0)
    group.add_argument("--observe_step_geom_factor", type=float, default=1.5)
    group.add_argument("--observe_step_min", type=int, default=100)
    group.add_argument("--observe_step_linear_N", type=int, default=0)
    group.add_argument("--observe_output_path", default="observations.pt", type=str)
    group.add_argument(
        "--observe_N_batches",
        type=int,
        default=1,
        help=(
            "Backward-compatible default number of batches to average over per "
            "observation. Use <= 0 to iterate once over the full dataloader."
        ),
    )
    group.add_argument(
        "--observe_train_N_batches",
        type=int,
        default=1,
        help="Number of train batches per observation. Use <= 0 for the full train dataloader.",
    )
    group.add_argument(
        "--observe_test_N_batches",
        type=int,
        default=-1,
        help="Number of test batches per observation. Use <= 0 for the full test dataloader.",
    )
    group.add_argument(
        "--cluster_margin_eps",
        type=float,
        default=1e-12,
        help="Numerical epsilon for oracle cluster-margin diagnostics.",
    )
    group.add_argument(
        "--disable_weight_diagnostics",
        action="store_true",
        help="Disable L2 and spectral-complexity weight diagnostics during observation.",
    )
    group.add_argument(
        "--spectral_norm_power_iters",
        type=int,
        default=20,
        help=(
            "Power iterations used to estimate spectral norms of weight tensors. "
            "Use 0 for a cheaper Frobenius-only diagnostic without spectral norms."
        ),
    )
    group.add_argument(
        "--spectral_norm_eps",
        type=float,
        default=1e-12,
        help="Numerical epsilon used in spectral-complexity logs.",
    )

    group = parent_parser.add_argument_group("Checkpoint saving")
    group.add_argument("--disable_checkpoint", action="store_true")
    group.add_argument(
        "--checkpoint_by_step",
        action="store_true",
        help="Also save extra checkpoints on the old step-based schedule.",
    )
    group.add_argument("--observe_checkpoint_step_geom_factor", type=float, default=1.5)
    group.add_argument("--observe_checkpoint_step_min", type=int, default=1000)
    group.add_argument("--observe_checkpoint_step_linear_N", type=int, default=0)
    group.add_argument("--observe_checkpoint_folder", default="checkpoints", type=str)
    return parent_parser


# -----------------------------------------------------------------------------
# Schedules
# -----------------------------------------------------------------------------


def logarithmic_integer_grid(max_value: int, num_saves: int) -> list[int]:
    """Return approximately logarithmically spaced positive integers.

    The grid always includes 1 and ``max_value``. If ``num_saves >= max_value``,
    every integer from 1 to ``max_value`` is returned. This is used both for
    epoch-based schedules and for step-based schedules.
    """
    import numpy as np

    max_value = int(max_value)
    num_saves = int(num_saves)

    if max_value <= 0 or num_saves <= 0:
        return []

    target = min(num_saves, max_value)

    if target == max_value:
        return list(range(1, max_value + 1))

    if target == 1:
        return [max_value]

    desired = np.logspace(
        np.log10(1.0),
        np.log10(float(max_value)),
        num=target,
        endpoint=True,
    )

    used: set[int] = set()
    values: list[int] = []

    for x in desired:
        center = int(round(float(x)))
        center = max(1, min(max_value, center))

        if center not in used:
            chosen = center
        else:
            chosen = None
            radius = 1
            while chosen is None:
                candidates: list[int] = []
                lo = center - radius
                hi = center + radius

                if lo >= 1 and lo not in used:
                    candidates.append(lo)
                if hi <= max_value and hi not in used:
                    candidates.append(hi)

                if candidates:
                    chosen = min(
                        candidates,
                        key=lambda e: abs(np.log(float(e)) - np.log(float(x))),
                    )

                radius += 1

        used.add(int(chosen))
        values.append(int(chosen))

    values = sorted(set(values))

    # Be defensive: always include the two endpoints.
    if values[0] != 1:
        if 1 in values:
            values.remove(1)
        values[0] = 1
    if values[-1] != max_value:
        if max_value in values:
            values.remove(max_value)
        values[-1] = max_value

    return sorted(set(values))


def logarithmic_epoch_grid(max_epoch: int, num_saves: int) -> list[int]:
    """Return approximately logarithmically spaced completed epochs."""
    return logarithmic_integer_grid(max_epoch, num_saves)


def logarithmic_step_grid(max_step: int, num_saves: int) -> list[int]:
    """Return approximately logarithmically spaced optimizer steps."""
    return logarithmic_integer_grid(max_step, num_saves)

def geom_and_lin(geom_min, geom_factor, max_val, Nlinear):
    import numpy as np

    if max_val <= 0:
        return []
    geom = []
    if geom_min > 0 and geom_factor > 1:
        max_i = int(np.ceil(np.log(max_val / geom_min) / np.log(geom_factor)))
        geom = [int(np.floor(geom_min * geom_factor**i)) for i in range(max_i + 1)]
    linear = []
    if Nlinear > 2:
        linear = [
            int(s) for s in np.round(np.linspace(0, max_val, num=Nlinear)).astype(int)
        ]
    return sorted(set(x for x in geom + linear if 0 <= x <= max_val))


# -----------------------------------------------------------------------------
# Representation diagnostics
# -----------------------------------------------------------------------------


def _safe_float(x: Any) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def _sparsity(z_hat: torch.Tensor) -> float:
    if z_hat.numel() == 0:
        return 0.0
    norm1 = torch.linalg.norm(z_hat, ord=1, dim=1)
    norm2 = torch.linalg.norm(z_hat, ord=2, dim=1).clamp_min(1e-30)
    n = z_hat.shape[1]
    denom = math.sqrt(float(n)) - 1.0
    if denom <= 0:
        return 0.0
    sparsity = (math.sqrt(float(n)) - norm1 / norm2) / denom
    return float(sparsity.mean().detach().cpu().item())


def _cluster_count(labels: torch.Tensor) -> int:
    if labels.numel() == 0:
        return 0
    return int(torch.unique(labels).numel())


def _misclustering(w: torch.Tensor, w_true: torch.Tensor) -> int:
    misclusterings = 0
    w = w.flatten()
    w_true = w_true.flatten().to(w.device)
    for i in torch.unique(w):
        w_trues = w_true[w == i]
        n = w_trues.shape[0]
        if n > 1:
            total_pairs = n * (n - 1) // 2
            _, counts = torch.unique(w_trues, return_counts=True)
            same_label_pairs = (counts * (counts - 1) // 2).sum()
            misclusterings += (total_pairs - same_label_pairs).item()
    return int(misclusterings)


def _hungarian_true_to_pred(
    pred_flat: torch.Tensor,
    gt_flat: torch.Tensor,
    num_pred_classes: int,
) -> tuple[float, torch.Tensor, list[int]]:
    """Return Hungarian-matched accuracy and true->pred assignment.

    The assignment maps true latent labels to model cluster indices. This is the
    direction needed to compute the oracle cluster-margin diagnostic.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    pred_np = pred_flat.detach().flatten().cpu().numpy().astype(int)
    gt_np = gt_flat.detach().flatten().cpu().numpy().astype(int)
    if pred_np.size == 0:
        return 0.0, torch.empty(0, dtype=torch.long), []

    max_true = int(gt_np.max()) if gt_np.size else 0
    max_pred = int(pred_np.max()) if pred_np.size else 0
    n_true = max_true + 1
    n_pred = max(int(num_pred_classes), max_pred + 1)
    n_square = max(n_true, n_pred)

    conf = np.zeros((n_square, n_square), dtype=np.int64)
    for t, p in zip(gt_np, pred_np):
        if 0 <= t < n_square and 0 <= p < n_square:
            conf[t, p] += 1

    row_ind, col_ind = linear_sum_assignment(-conf)
    true_to_pred_np = -np.ones(n_true, dtype=np.int64)
    for t, p in zip(row_ind, col_ind):
        if t < n_true:
            true_to_pred_np[t] = p if p < n_pred else -1

    assigned = true_to_pred_np[gt_np]
    correct = (assigned == pred_np) & (assigned >= 0) & (assigned < num_pred_classes)
    acc = float(correct.mean()) if correct.size else 0.0
    true_to_pred = torch.from_numpy(true_to_pred_np).long()
    return acc, true_to_pred, true_to_pred_np.tolist()


def _representation_metrics(
    rep_flat: torch.Tensor,
    gt_flat: torch.Tensor,
    *,
    eps: float,
) -> dict[str, Any]:
    """Compute scalar diagnostics for one module on one observed split.

    rep_flat has shape [N_observed_nodes, C] and is assumed to be a non-negative
    soft assignment. It is renormalized defensively before entropy/margin metrics.
    """
    rep_flat = rep_flat.detach().cpu().float()
    gt_flat = gt_flat.detach().cpu().long().flatten()
    if rep_flat.numel() == 0 or gt_flat.numel() == 0:
        return {
            "sparsity": 0.0,
            "cluster_count": 0,
            "misclustering": 0,
            "accuracy": 0.0,
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "max_probability": 0.0,
            "true_probability": 0.0,
            "false_probability": 0.0,
            "cluster_margin": 0.0,
            "cluster_margin_loss": 0.0,
            "nll_true_cluster": 0.0,
            "true_to_pred_assignment": [],
        }

    # Defensive normalization: CtP/PtC outputs should already sum to one, but this
    # keeps the diagnostic meaningful if a future representation is only positive.
    probs = rep_flat.clamp_min(0.0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(eps)

    pred_flat = probs.argmax(dim=1).long()
    num_pred_classes = probs.shape[1]
    accuracy, true_to_pred, assignment_list = _hungarian_true_to_pred(
        pred_flat, gt_flat, num_pred_classes
    )

    row_idx = torch.arange(gt_flat.numel(), dtype=torch.long)
    valid_gt = (gt_flat >= 0) & (gt_flat < true_to_pred.numel())
    target_pred = torch.full_like(gt_flat, fill_value=-1)
    target_pred[valid_gt] = true_to_pred[gt_flat[valid_gt]]
    valid_target = (target_pred >= 0) & (target_pred < num_pred_classes)

    p_true = torch.zeros(gt_flat.numel(), dtype=probs.dtype)
    if valid_target.any():
        p_true[valid_target] = probs[row_idx[valid_target], target_pred[valid_target]]
    p_total = probs.sum(dim=1)
    p_false = (p_total - p_true).clamp_min(0.0)

    margin = torch.log(p_true.clamp_min(eps)) - torch.log(p_false.clamp_min(eps))
    margin_loss = F.softplus(-margin)
    entropy_vec = -(probs.clamp_min(eps) * torch.log(probs.clamp_min(eps))).sum(dim=1)
    norm_entropy = entropy_vec / math.log(num_pred_classes) if num_pred_classes > 1 else entropy_vec * 0

    return {
        "sparsity": _sparsity(probs),
        "cluster_count": _cluster_count(pred_flat),
        "misclustering": _misclustering(pred_flat, gt_flat),
        "accuracy": accuracy,
        "entropy": float(entropy_vec.mean().item()),
        "normalized_entropy": float(norm_entropy.mean().item()),
        "max_probability": float(probs.max(dim=1).values.mean().item()),
        "true_probability": float(p_true.mean().item()),
        "false_probability": float(p_false.mean().item()),
        "cluster_margin": float(margin.mean().item()),
        "cluster_margin_loss": float(margin_loss.mean().item()),
        "nll_true_cluster": float((-torch.log(p_true.clamp_min(eps))).mean().item()),
        "true_to_pred_assignment": assignment_list,
    }


# -----------------------------------------------------------------------------
# Weight diagnostics
# -----------------------------------------------------------------------------


def _matrix_view_for_spectral_norm(param: torch.Tensor) -> torch.Tensor | None:
    if param.ndim < 2:
        return None
    # Linear: [out, in]; Conv1d: [out, in, k]; dictionaries: [out_like, ...].
    return param.detach().cpu().double().reshape(param.shape[0], -1)


def _power_spectral_norm(mat: torch.Tensor, n_iter: int, eps: float) -> float:
    if mat.numel() == 0:
        return 0.0
    if n_iter <= 0:
        return float("nan")
    if torch.count_nonzero(mat).item() == 0:
        return 0.0

    n_cols = mat.shape[1]
    v = torch.ones(n_cols, dtype=mat.dtype)
    v = v / v.norm().clamp_min(eps)
    for _ in range(int(n_iter)):
        u = mat @ v
        u_norm = u.norm()
        if float(u_norm) <= eps:
            return 0.0
        u = u / u_norm
        v = mat.t() @ u
        v_norm = v.norm()
        if float(v_norm) <= eps:
            return 0.0
        v = v / v_norm
    sigma = (mat @ v).norm()
    return float(sigma.item())


def _model_weight_diagnostics(model: torch.nn.Module, *, power_iters: int, eps: float) -> dict[str, Any]:
    total_sq = 0.0
    weight_sq = 0.0
    bias_sq = 0.0
    n_params = 0
    n_trainable = 0
    log_spectral_product = 0.0
    spectral_product = 1.0
    n_spectral_tensors = 0
    per_tensor: dict[str, dict[str, Any]] = {}

    for name, param in model.named_parameters():
        detached = param.detach().cpu().double()
        sq = float(detached.pow(2).sum().item())
        l2 = math.sqrt(max(sq, 0.0))
        total_sq += sq
        n_params += int(detached.numel())
        if param.requires_grad:
            n_trainable += int(detached.numel())

        is_weight_like = detached.ndim >= 2 and not name.endswith(".bias")
        if is_weight_like:
            weight_sq += sq
        else:
            bias_sq += sq

        tensor_info: dict[str, Any] = {
            "shape": list(detached.shape),
            "numel": int(detached.numel()),
            "requires_grad": bool(param.requires_grad),
            "l2": l2,
            "is_weight_like": bool(is_weight_like),
        }

        mat = _matrix_view_for_spectral_norm(detached) if is_weight_like else None
        if mat is not None and power_iters > 0:
            sigma = _power_spectral_norm(mat, power_iters, eps)
            tensor_info["spectral_norm"] = sigma
            if sigma > 0 and math.isfinite(sigma):
                log_spectral_product += math.log(max(sigma, eps))
                spectral_product *= sigma
                n_spectral_tensors += 1
            else:
                tensor_info["spectral_norm_nonpositive"] = True
        elif mat is not None:
            tensor_info["spectral_norm"] = None

        per_tensor[name] = tensor_info

    if not math.isfinite(spectral_product):
        spectral_product_out = float("inf")
    else:
        spectral_product_out = float(spectral_product)

    return {
        "num_parameters": int(n_params),
        "num_trainable_parameters": int(n_trainable),
        "parameter_l2_total": math.sqrt(max(total_sq, 0.0)),
        "weight_l2_total": math.sqrt(max(weight_sq, 0.0)),
        "bias_l2_total": math.sqrt(max(bias_sq, 0.0)),
        "spectral_complexity_norm": spectral_product_out,
        "log_spectral_complexity_norm": float(log_spectral_product),
        "num_spectral_tensors": int(n_spectral_tensors),
        "spectral_norm_power_iters": int(power_iters),
        "per_tensor": per_tensor,
    }


# -----------------------------------------------------------------------------
# Observer
# -----------------------------------------------------------------------------


def init_observer(
    steps_per_epoch: int,
    train_eval_dataloader,
    test_dataloader,
    args,
    observer_state=None,
    rhm_data=None,
):
    obs = observer(
        train_eval_dataloader, test_dataloader, args, steps_per_epoch, rhm_data=rhm_data
    )
    if observer_state is not None:

        def _field(state, name, default):
            if isinstance(state, dict):
                return state.get(name, default)
            return getattr(state, name, default)

        samples = _field(observer_state, "samples", [])
        sampled_epochs = _field(observer_state, "sampled_epochs", [])
        sampled_steps = _field(observer_state, "sampled_steps", [])

        obs.samples = list(samples) if samples is not None else []
        obs.sampled_epochs = list(sampled_epochs) if sampled_epochs is not None else []
        obs.sampled_steps = list(sampled_steps) if sampled_steps is not None else []
    return obs


class observer:
    def __init__(
        self,
        train_eval_dataloader,
        test_dataloader,
        args,
        steps_per_epoch: int,
        rhm_data=None,
    ):
        import os

        self.args = args
        self.observe_output_path = os.path.join(
            args.output_filename_prefix, args.observe_output_path
        )
        self.observe_checkpoint_folder = os.path.join(
            args.output_filename_prefix, args.observe_checkpoint_folder
        )
        self.test_dataloader = test_dataloader
        self.train_dataloader = train_eval_dataloader
        self.rhm_data = rhm_data
        self.observe_N_batches = int(getattr(args, "observe_N_batches", 1))
        self.observe_train_N_batches = int(
            getattr(args, "observe_train_N_batches", self.observe_N_batches)
        )
        self.observe_test_N_batches = int(
            getattr(args, "observe_test_N_batches", self.observe_N_batches)
        )
        self.steps_per_epoch = int(steps_per_epoch)

        max_steps = self.steps_per_epoch * int(args.Nepoch)
        self.max_steps = int(max_steps)
        self.observe_by_step = bool(getattr(args, "observe_by_step", False))
        self.observe_schedule_unit = "step" if self.observe_by_step else "epoch"

        if self.observe_by_step:
            self.sample_epochs = []
            self.sample_steps = logarithmic_step_grid(max_steps, args.observe_num_saves)
        else:
            self.sample_epochs = logarithmic_epoch_grid(args.Nepoch, args.observe_num_saves)
            self.sample_steps = []

        # Optional extra checkpoint-only step schedule. In normal use this is not
        # needed, because the main schedule above already saves checkpoints.
        self.checkpoint_steps = (
            geom_and_lin(
                args.observe_checkpoint_step_min,
                args.observe_checkpoint_step_geom_factor,
                max_steps,
                args.observe_checkpoint_step_linear_N,
            )
            if getattr(args, "checkpoint_by_step", False)
            else []
        )

        self.sampled_epochs: list[int] = []
        self.sampled_steps: list[int] = []
        self.samples: list[dict[str, Any]] = []

    def clear_samples_after_step(self, starting_step: int):
        if starting_step is None:
            return
        starting_step = int(starting_step)
        kept_epochs = []
        kept_steps = []
        kept_samples = []
        for epoch, step, sample in zip(
            self.sampled_epochs, self.sampled_steps, self.samples
        ):
            sample_step = sample.get("step", step)
            if sample_step <= starting_step:
                kept_epochs.append(epoch)
                kept_steps.append(step)
                kept_samples.append(sample)
        self.sampled_epochs = kept_epochs
        self.sampled_steps = kept_steps
        self.samples = kept_samples

    def save(self):
        import os

        os.makedirs(os.path.dirname(self.observe_output_path) or ".", exist_ok=True)
        torch.save(self, self.observe_output_path)

    @staticmethod
    def checkpoint_name(path: str, step: int):
        import os

        return os.path.join(path, f"checkpoint_step_{step}.pt")

    def checkpoint(self, student, teacher, optimizer, train_model_state):
        if self.args.disable_checkpoint:
            return
        state = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict() if teacher is not None else None,
            "optimizer": optimizer.state_dict(),
            "train_model_state": train_model_state,
            "args": vars(self.args),
        }
        step = int(train_model_state["step"])
        fname = observer.checkpoint_name(self.observe_checkpoint_folder, step)
        import os

        os.makedirs(self.observe_checkpoint_folder, exist_ok=True)
        torch.save(state, fname)

    def observe(self, epoch, step, student, teacher):
        sample: dict[str, Any] = {
            "epoch": int(epoch),
            "step": int(step),
            "observe_schedule_unit": getattr(self, "observe_schedule_unit", "epoch"),
        }
        device = next(student.parameters()).device

        import rhm

        def _batch_iter(dataloader, n_batches: int):
            if n_batches <= 0:
                for batch in dataloader:
                    if isinstance(batch, (tuple, list)):
                        batch = batch[0]
                    yield batch
                return

            it = iter(dataloader)
            for _ in range(n_batches):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(dataloader)
                    batch = next(it)
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                yield batch

        def _extract_tokens(batch: torch.Tensor) -> torch.Tensor:
            if batch.dim() == 3:
                return batch.argmax(dim=1)
            return batch

        def _compute_latents(batch_tokens: torch.Tensor):
            if self.rhm_data is None or "inverse_rules" not in self.rhm_data:
                return None
            inv_rules = self.rhm_data["inverse_rules"]
            rhm_params = self.rhm_data.get("rhm_params", {})
            rhm_s = int(rhm_params.get("s"))
            rhm_L = int(rhm_params.get("L"))
            return rhm.build_latents_from_inv_rules(
                batch_tokens, inv_rules, rhm_s, rhm_L
            )

        def _accumulate_losses(acc: dict[str, float], batch_losses: dict[str, torch.Tensor]):
            for key, val in batch_losses.items():
                acc[key] = acc.get(key, 0.0) + _safe_float(val)

        def _average_losses(acc: dict[str, float], n_batches: int):
            n = max(1, int(n_batches))
            return {k: v / n for k, v in acc.items()}

        def _append_rep_chunks(rep_chunks, gt_chunks, reps, latents):
            if latents is None:
                return
            rhm_params = self.rhm_data.get("rhm_params", {}) if self.rhm_data else {}
            rhm_L = int(rhm_params.get("L", 0))
            if not rep_chunks:
                rep_chunks.extend([] for _ in range(len(reps)))
                gt_chunks.extend([] for _ in range(len(reps)))
            for idx, rep in enumerate(reps):
                target_level = rhm_L - 1 - idx
                if target_level not in latents:
                    continue
                gt = latents[target_level]
                rep_flat = rep.detach().permute(0, 2, 1).reshape(-1, rep.shape[1]).cpu()
                gt_flat = gt.reshape(-1).detach().cpu()
                rep_chunks[idx].append(rep_flat)
                gt_chunks[idx].append(gt_flat)

        def _finalize_metrics(rep_chunks, gt_chunks):
            metrics = []
            eps = float(getattr(self.args, "cluster_margin_eps", 1e-12))
            for layer_rep_chunks, layer_gt_chunks in zip(rep_chunks, gt_chunks):
                if not layer_rep_chunks or not layer_gt_chunks:
                    metrics.append({})
                    continue
                rep_flat = torch.cat(layer_rep_chunks, dim=0)
                gt_flat = torch.cat(layer_gt_chunks, dim=0)
                metrics.append(_representation_metrics(rep_flat, gt_flat, eps=eps))
            return metrics

        def _observe_split(dataloader, n_batches: int):
            loss_acc: dict[str, float] = {}
            rep_chunks: list[list[torch.Tensor]] = []
            gt_chunks: list[list[torch.Tensor]] = []
            actual_n_batches = 0
            for batch in _batch_iter(dataloader, n_batches):
                actual_n_batches += 1
                batch = batch.to(device)
                batch_tokens = _extract_tokens(batch).detach().cpu()
                latents = _compute_latents(batch_tokens)
                if hasattr(student, "prepare_batch"):
                    batch_prepared = student.prepare_batch(batch)
                else:
                    batch_prepared = batch

                losses, reps = student.compute_losses(
                    batch_prepared, teacher=teacher, return_reps=True
                )
                _accumulate_losses(loss_acc, losses)
                _append_rep_chunks(rep_chunks, gt_chunks, reps, latents)

            loss_avg = _average_losses(loss_acc, actual_n_batches)
            metrics = _finalize_metrics(rep_chunks, gt_chunks) if rep_chunks else []
            return loss_avg, metrics, actual_n_batches

        was_training = student.training
        teacher_was_training = teacher.training if teacher is not None else False
        student.eval()
        if teacher is not None:
            teacher.eval()
        with torch.no_grad():
            train_losses, train_metrics, train_batches = _observe_split(
                self.train_dataloader, self.observe_train_N_batches
            )
            test_losses, test_metrics, test_batches = _observe_split(
                self.test_dataloader, self.observe_test_N_batches
            )
        if was_training:
            student.train()
        if teacher is not None and teacher_was_training:
            teacher.train()

        sample["train_losses"] = train_losses
        sample["test_losses"] = test_losses
        sample["metrics"] = {"train": train_metrics, "test": test_metrics}
        sample["num_observed_batches"] = {"train": int(train_batches), "test": int(test_batches)}

        if not getattr(self.args, "disable_weight_diagnostics", False):
            power_iters = int(getattr(self.args, "spectral_norm_power_iters", 20))
            eps = float(getattr(self.args, "spectral_norm_eps", 1e-12))
            sample["weight_diagnostics"] = {
                "student": _model_weight_diagnostics(
                    student, power_iters=power_iters, eps=eps
                )
            }
            if teacher is not None:
                sample["weight_diagnostics"]["teacher"] = _model_weight_diagnostics(
                    teacher, power_iters=power_iters, eps=eps
                )

        self.sampled_epochs.append(int(epoch))
        self.sampled_steps.append(int(step))
        self.samples.append(sample)

    def get_observable(self, observable: str):
        return [sample[observable] for sample in self.samples]

    def __getitem__(self, key):
        return self.get_observable(key)

    def __len__(self):
        return len(self.samples)

    def __contains__(self, key):
        return all(key in sample for sample in self.samples)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("test_dataloader", None)
        state.pop("train_dataloader", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.test_dataloader = None
        self.train_dataloader = None
