from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import init


@dataclass(frozen=True)
class OptunaParamSpec:
    name: str
    kind: str
    min_val: float | int | None = None
    max_val: float | int | None = None
    log: bool = False
    choices: list[Any] | None = None


OPTUNA_PARAM_SPECS = [
    OptunaParamSpec("lr", "float", 1e-5, 1e-2, log=True),
    OptunaParamSpec("ema_alpha_bar", "float", 1e-4, 1e-1, log=True),
    OptunaParamSpec("wd", "float", 1e-6, 1e-2, log=True),
    OptunaParamSpec("batch_size", "categorical", choices=[16, 32, 64, 128]),
    OptunaParamSpec("ctp_hidden_dim", "int", 32, 256, log=True),
    OptunaParamSpec("ctp_interlayer_temperature", "float", 0.0, 1.0, log=False),
    OptunaParamSpec("ctp_group_lasso_penalty", "float", 1e-6, 1e-1, log=True),
    OptunaParamSpec("ctp_logit_l2_penalty", "float", 1e-6, 1e-1, log=True),
    OptunaParamSpec("ctp_linear_l2_penalty", "float", 1e-6, 1e-1, log=True),
    OptunaParamSpec("optimizer", "categorical", choices=["Adam", "AdamW", "sgd"]),
    OptunaParamSpec(
        "ctp_activation", "categorical", choices=["relu", "leaky_relu", "gelu"]
    ),
    OptunaParamSpec(
        "ctp_batchnorm_before_activation", "categorical", choices=[False, True]
    ),
    OptunaParamSpec("stop_grad_between_modules", "categorical", choices=[False, True]),
    OptunaParamSpec("use_autojac", "categorical", choices=[False, True]),
]


def add_optuna_args(parser):
    group = parser.add_argument_group("Optuna")
    group.add_argument("--optuna", action="store_true", help="Run Optuna search.")
    group.add_argument("--n_trials", type=int, default=25, help="Number of Optuna trials.")
    group.add_argument(
        "--optuna_timeout",
        type=float,
        default=None,
        help="Optional Optuna timeout (seconds).",
    )
    group.add_argument(
        "--optuna_storage",
        type=str,
        default="",
        help="Optuna storage URL (e.g., sqlite:////path/to.db).",
    )
    group.add_argument(
        "--optuna_study_name", type=str, default="ctp_optuna", help="Study name."
    )
    group.add_argument(
        "--optuna_output_dir",
        type=str,
        default="optuna_runs",
        help="Root directory for trial outputs.",
    )
    group.add_argument(
        "--optuna_pruner",
        type=str,
        default="median",
        choices=["median", "hyperband", "nopruner"],
        help="Pruner selection.",
    )
    group.add_argument(
        "--optuna_n_startup_trials",
        type=int,
        default=5,
        help="Startup trials before pruning.",
    )
    group.add_argument(
        "--optuna_n_warmup_steps",
        type=int,
        default=5,
        help="Warmup steps before pruning.",
    )
    group.add_argument(
        "--optuna_interval_steps",
        type=int,
        default=1,
        help="Pruning check interval (in reported steps).",
    )
    group.add_argument(
        "--optuna_seed", type=int, default=0, help="Sampler seed."
    )
    group.add_argument(
        "--optuna_trial_seed_offset",
        type=int,
        default=0,
        help="Offset added to seed_torch per trial.",
    )
    group.add_argument(
        "--optuna_metric_split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Which split to use for the Optuna metric.",
    )
    group.add_argument(
        "--optuna_score_mode",
        type=str,
        default="last",
        choices=["last", "best"],
        help="Use last or best observed score for the trial objective.",
    )
    group.add_argument(
        "--optuna_miscluster_penalty",
        type=float,
        default=0.0,
        help="Penalty weight for log10(1+misclustering).",
    )
    group.add_argument(
        "--optuna_target_accuracy",
        type=float,
        default=0.95,
        help="Target mean layer accuracy for step-efficiency bonus.",
    )
    group.add_argument(
        "--optuna_step_bonus_weight",
        type=float,
        default=0.02,
        help="Bonus weight for reaching target accuracy earlier.",
    )
    group.add_argument(
        "--optuna_save_observer",
        action="store_true",
        help="Persist observer snapshots during Optuna runs.",
    )
    group.add_argument(
        "--optuna_disable_checkpoint",
        action="store_true",
        help="Disable checkpoints during Optuna runs.",
    )
    group.add_argument(
        "--optuna_all",
        action="store_true",
        help="Enable all optuna_* hyperparameter flags.",
    )

    for spec in OPTUNA_PARAM_SPECS:
        group.add_argument(
            f"--optuna_{spec.name}",
            action="store_true",
            help=f"Optimize {spec.name} with Optuna.",
        )
        if spec.kind in ("float", "int"):
            arg_type = float if spec.kind == "float" else int
            group.add_argument(
                f"--optuna_{spec.name}_min",
                type=arg_type,
                default=spec.min_val,
                help=f"Minimum {spec.name} for Optuna.",
            )
            group.add_argument(
                f"--optuna_{spec.name}_max",
                type=arg_type,
                default=spec.max_val,
                help=f"Maximum {spec.name} for Optuna.",
            )
            group.add_argument(
                f"--optuna_{spec.name}_no_log",
                action="store_true",
                help=f"Disable log scale for {spec.name}.",
            )
    return parser


def build_parser():
    parser = init.init_parser()
    parser = add_optuna_args(parser)
    return parser


def _float_value(val):
    import torch

    if torch.is_tensor(val):
        return float(val.detach().cpu().item())
    return float(val)


def _score_from_sample(sample, split: str, miscluster_penalty: float):
    metrics = sample.get("metrics", {}).get(split, [])
    if not metrics:
        return None
    accs = []
    mis = []
    for layer in metrics:
        if "accuracy" in layer:
            accs.append(_float_value(layer["accuracy"]))
        if "misclustering" in layer:
            mis.append(_float_value(layer["misclustering"]))
    if not accs:
        return None
    mean_acc = sum(accs) / len(accs)
    mean_mis = sum(mis) / len(mis) if mis else 0.0
    penalty = 0.0
    if miscluster_penalty > 0 and mean_mis > 0:
        penalty = miscluster_penalty * math.log10(1.0 + mean_mis)
    return mean_acc - penalty, mean_acc, mean_mis


def _step_bonus(first_reach_step, max_steps: int, weight: float) -> float:
    if weight <= 0 or first_reach_step is None or max_steps <= 0:
        return 0.0
    normalized = 1.0 - (float(first_reach_step) / float(max_steps))
    normalized = max(0.0, min(1.0, normalized))
    return weight * normalized


def _apply_optuna_suggestions(trial, args, optuna_params):
    for spec in optuna_params:
        flag = getattr(args, f"optuna_{spec.name}")
        if not flag:
            continue
        if spec.kind == "float":
            min_val = getattr(args, f"optuna_{spec.name}_min")
            max_val = getattr(args, f"optuna_{spec.name}_max")
            use_log = not getattr(args, f"optuna_{spec.name}_no_log") and spec.log
            value = trial.suggest_float(spec.name, min_val, max_val, log=use_log)
        elif spec.kind == "int":
            min_val = getattr(args, f"optuna_{spec.name}_min")
            max_val = getattr(args, f"optuna_{spec.name}_max")
            use_log = not getattr(args, f"optuna_{spec.name}_no_log") and spec.log
            value = trial.suggest_int(spec.name, min_val, max_val, log=use_log)
        elif spec.kind == "categorical":
            value = trial.suggest_categorical(spec.name, spec.choices)
        else:
            raise ValueError(f"Unsupported optuna param kind: {spec.kind}")
        if spec.name == "ema_alpha_bar":
            setattr(args, "ema_alpha_bar", value)
            setattr(args, "ema_alpha", 1.0 - float(value))
        else:
            setattr(args, spec.name, value)


def _init_pruner(args):
    import optuna

    if args.optuna_pruner == "nopruner":
        return optuna.pruners.NopPruner()
    if args.optuna_pruner == "hyperband":
        return optuna.pruners.HyperbandPruner()
    return optuna.pruners.MedianPruner(
        n_startup_trials=args.optuna_n_startup_trials,
        n_warmup_steps=args.optuna_n_warmup_steps,
        interval_steps=args.optuna_interval_steps,
    )


def _ensure_ctp(args):
    if not getattr(args, "ctp", False):
        args.ctp = True
    if getattr(args, "ptc", False):
        args.ptc = False


def _run_training(args, trial=None):
    import torch
    import model
    import train
    import observables

    _ensure_ctp(args)

    if args.optuna_disable_checkpoint:
        args.disable_checkpoint = True

    torch.manual_seed(args.seed_torch)
    rhm_data = init.init_dataset(args)
    train_dataloader, test_dataloader, train_eval_dataloader = init.init_data_loaders(
        args, rhm_data
    )
    print(
        "Initialized datasets: train size %d, test size %d"
        % (len(rhm_data["train"]), len(rhm_data["test"])),
        flush=True,
    )

    student = model.init_model(args)
    teacher = model.init_teacher(student, args)
    print(
        "Initialized models with device:",
        args.device,
        " student device:",
        next(student.parameters()).device,
        " teacher device:",
        next(teacher.parameters()).device if teacher is not None else None,
        flush=True,
    )

    optimizer = train.init_optimizer(args, student)
    steps_per_epoch = len(train_dataloader)
    max_steps = steps_per_epoch * int(args.Nepoch)
    observe_by_step = bool(getattr(args, "observe_by_step", False))

    observer = observables.init_observer(
        steps_per_epoch,
        train_eval_dataloader,
        test_dataloader,
        args,
        rhm_data=rhm_data,
    )
    print("Initialized observer.", flush=True)
    print(
        "Observation schedule unit:",
        "step" if observe_by_step else "epoch",
        flush=True,
    )
    if observe_by_step:
        print("Observation/save steps:", observer.sample_steps, flush=True)
    else:
        print("Observation/save epochs:", observer.sample_epochs, flush=True)

    def _observe_and_score(epoch: int, step: int, train_model_state):
        observer.observe(epoch, step, student, teacher)
        if args.optuna_save_observer:
            observer.save()
        score_tuple = _score_from_sample(
            observer.samples[-1],
            args.optuna_metric_split,
            args.optuna_miscluster_penalty,
        )
        if score_tuple is None:
            return None
        score, mean_acc, mean_mis = score_tuple
        return score, mean_acc, mean_mis

    # Keep the original Optuna behavior of observing initialization, but do not
    # count it as one of the scheduled save/evaluation points.
    observer.observe(0, 0, student, teacher)
    if args.optuna_save_observer:
        observer.save()

    scores = []
    first_reach_step = None
    observed_steps: set[int] = {0}
    checkpointed_steps: set[int] = set()
    last_train_model_state = None

    training_generator = train.train_model(
        student,
        optimizer,
        train_dataloader,
        args.Nepoch,
        teacher=teacher,
        ema_alpha=args.ema_alpha,
        starting_step=0,
        use_autojac=args.use_autojac,
        rhm_data=rhm_data,
        enable_probes=args.enable_probes,
    )

    for train_model_state in training_generator:
        last_train_model_state = train_model_state
        step = int(train_model_state["step"])
        completed_epoch = step // steps_per_epoch
        end_of_epoch = (step % steps_per_epoch) == 0

        if observe_by_step:
            should_observe = step in observer.sample_steps
        else:
            should_observe = end_of_epoch and completed_epoch in observer.sample_epochs
        should_observe = should_observe and step not in observed_steps

        if should_observe:
            score_tuple = _observe_and_score(completed_epoch, step, train_model_state)
            observed_steps.add(step)
            if score_tuple is not None:
                score, mean_acc, mean_mis = score_tuple
                if mean_acc >= float(args.optuna_target_accuracy) and first_reach_step is None:
                    first_reach_step = step
                bonus = _step_bonus(first_reach_step, max_steps, args.optuna_step_bonus_weight)
                total_score = score + bonus
                scores.append((total_score, score, bonus, mean_acc, mean_mis, step))
                if trial is not None:
                    trial.report(total_score, step)
                    if trial.should_prune():
                        raise _trial_pruned()
                print(
                    f"Step {step} epoch {completed_epoch} score {total_score:.4f} "
                    f"(acc {mean_acc:.4f}, mis {mean_mis:.1f}, bonus {bonus:.4f})",
                    flush=True,
                )

        if (
            getattr(args, "checkpoint_by_step", False)
            and step in observer.checkpoint_steps
            and step not in checkpointed_steps
        ):
            observer.checkpoint(student, teacher, optimizer, train_model_state)
            checkpointed_steps.add(step)

    if last_train_model_state is None:
        step = 0
        completed_epoch = 0
        last_train_model_state = {"step": 0, "epoch": 0}
    else:
        step = int(last_train_model_state["step"])
        completed_epoch = step // max(1, steps_per_epoch)

    # Safety final observation/checkpoint.
    if step not in observed_steps:
        score_tuple = _observe_and_score(completed_epoch, step, last_train_model_state)
        observed_steps.add(step)
        if score_tuple is not None:
            score, mean_acc, mean_mis = score_tuple
            if mean_acc >= float(args.optuna_target_accuracy) and first_reach_step is None:
                first_reach_step = step
            bonus = _step_bonus(first_reach_step, max_steps, args.optuna_step_bonus_weight)
            total_score = score + bonus
            scores.append((total_score, score, bonus, mean_acc, mean_mis, step))

    if step not in checkpointed_steps:
        observer.checkpoint(student, teacher, optimizer, last_train_model_state)
        checkpointed_steps.add(step)

    if not scores:
        return 0.0, observer

    if args.optuna_score_mode == "best":
        score_tuple = max(scores, key=lambda s: s[0])
    else:
        score_tuple = scores[-1]
    final_score, base_score, bonus, final_acc, final_mis, final_step = score_tuple
    print(
        f"Final score {final_score:.4f} (acc {final_acc:.4f}, mis {final_mis:.1f}, "
        f"bonus {bonus:.4f}, step {final_step})",
        flush=True,
    )
    return final_score, observer

def _trial_pruned():
    import optuna

    return optuna.TrialPruned()


def _setup_trial_output(args, trial):
    run_dir = os.path.join(args.optuna_output_dir, f"trial_{trial.number:04d}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _write_trial_args(path, args):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)


def _prepare_optuna_flags(args):
    if args.optuna_all:
        for spec in OPTUNA_PARAM_SPECS:
            setattr(args, f"optuna_{spec.name}", True)


def _check_autojac(args):
    if not args.use_autojac:
        return
    try:
        import torchjd  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "use_autojac requested but torchjd is unavailable."
        ) from exc


def run_optuna(args):
    import optuna

    _prepare_optuna_flags(args)
    _ensure_ctp(args)
    _check_autojac(args)

    storage = args.optuna_storage or None
    sampler = optuna.samplers.TPESampler(seed=args.optuna_seed)
    pruner = _init_pruner(args)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.optuna_study_name,
        storage=storage,
        load_if_exists=bool(storage),
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial):
        trial_args = copy.deepcopy(args)
        _ensure_ctp(trial_args)
        _apply_optuna_suggestions(trial, trial_args, OPTUNA_PARAM_SPECS)
        _check_autojac(trial_args)
        if trial_args.optuna_trial_seed_offset:
            trial_args.seed_torch = (
                int(trial_args.seed_torch)
                + int(trial_args.optuna_trial_seed_offset)
                + int(trial.number)
            )
        run_dir = _setup_trial_output(trial_args, trial)
        trial_args.output_filename_prefix = run_dir
        _write_trial_args(os.path.join(run_dir, "trial_args.json"), trial_args)
        score, observer = _run_training(trial_args, trial=trial)
        if observer.samples:
            last = observer.samples[-1]
            score_tuple = _score_from_sample(
                last, trial_args.optuna_metric_split, trial_args.optuna_miscluster_penalty
            )
            if score_tuple is not None:
                _, mean_acc, mean_mis = score_tuple
                trial.set_user_attr("last_mean_accuracy", mean_acc)
                trial.set_user_attr("last_mean_misclustering", mean_mis)
        if trial_args.optuna_step_bonus_weight > 0:
            trial.set_user_attr("target_accuracy", float(trial_args.optuna_target_accuracy))
        if hasattr(trial_args, "ema_alpha_bar"):
            trial.set_user_attr("ema_alpha_bar", float(trial_args.ema_alpha_bar))
            trial.set_user_attr("ema_alpha", float(trial_args.ema_alpha))
        trial.set_user_attr("output_dir", run_dir)
        return score

    study.optimize(objective, n_trials=args.n_trials, timeout=args.optuna_timeout)
    print("Best trial:", study.best_trial.number, "score:", study.best_value, flush=True)
    print("Best params:", study.best_trial.params, flush=True)
    return study


def main():
    parser = build_parser()
    args = parser.parse_args()
    _ensure_ctp(args)
    _prepare_optuna_flags(args)
    _check_autojac(args)

    if args.optuna:
        run_optuna(args)
        return

    score, _ = _run_training(args, trial=None)
    print("Single-run score:", score, flush=True)


if __name__ == "__main__":
    main()
