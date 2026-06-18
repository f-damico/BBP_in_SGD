import init


def _torch_load_compat(path, *, map_location=None, weights_only=None):
    """
    Compatible torch.load wrapper.

    Newer PyTorch accepts weights_only.
    Older PyTorch does not.
    """
    import torch

    if weights_only is None:
        return torch.load(path, map_location=map_location)

    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def print_observation_summary(observer, args=None, prefix="[OBS]"):
    """
    Print the most important train/test metrics after every saved observation.

    This goes to stdout, therefore to the PBS log, because the PBS runs python -u
    and pipes stdout/stderr through tee.
    """
    import math

    if not hasattr(observer, "samples") or len(observer.samples) == 0:
        return

    sample = observer.samples[-1]
    epoch = sample.get("epoch", None)
    step = sample.get("step", None)

    def _safe_float(x):
        try:
            if hasattr(x, "detach"):
                return float(x.detach().cpu().item())
            return float(x)
        except Exception:
            return float("nan")

    def _total_loss(loss_dict):
        if not loss_dict:
            return float("nan")
        return sum(_safe_float(v) for v in loss_dict.values())

    def _fmt(x):
        x = _safe_float(x)
        if not math.isfinite(x):
            return "nan"
        return f"{x:.6g}"

    train_losses = sample.get("train_losses", {})
    test_losses = sample.get("test_losses", {})

    train_total = _total_loss(train_losses)
    test_total = _total_loss(test_losses)

    print("", flush=True)
    print("=" * 100, flush=True)
    print(
        f"{prefix} epoch={epoch} step={step} "
        f"train_total_loss={_fmt(train_total)} "
        f"test_total_loss={_fmt(test_total)}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Layer-wise losses: pred_loss / clust_loss / probe_loss if present
    # ------------------------------------------------------------------
    for split_name, losses in [("train", train_losses), ("test", test_losses)]:
        if not losses:
            continue

        layer_keys = sorted(
            [k for k in losses.keys() if "_layer" in k],
            key=lambda k: (
                int(k.split("_layer")[1].split("_")[0])
                if "_layer" in k and k.split("_layer")[1].split("_")[0].isdigit()
                else 10**9,
                k,
            ),
        )

        if layer_keys:
            print(f"{prefix} {split_name} layer losses:", flush=True)
            for k in layer_keys:
                print(f"{prefix}   {k}={_fmt(losses[k])}", flush=True)

    # ------------------------------------------------------------------
    # Layer-wise oracle diagnostics
    # ------------------------------------------------------------------
    metrics = sample.get("metrics", {})

    metric_keys_to_print = [
        "accuracy",
        "cluster_margin_loss",
        "cluster_margin",
        "normalized_entropy",
        "entropy",
        "max_probability",
        "true_probability",
        "false_probability",
        "cluster_count",
        "misclustering",
        "sparsity",
    ]

    for split_name in ["train", "test"]:
        split_metrics = metrics.get(split_name, [])
        if not split_metrics:
            continue

        print(f"{prefix} {split_name} layer metrics:", flush=True)

        for layer_idx, layer_metrics in enumerate(split_metrics):
            fields = []
            for key in metric_keys_to_print:
                if key in layer_metrics:
                    fields.append(f"{key}={_fmt(layer_metrics[key])}")

            if fields:
                print(
                    f"{prefix}   module={layer_idx} " + " ".join(fields),
                    flush=True,
                )

    # ------------------------------------------------------------------
    # Weight diagnostics
    # ------------------------------------------------------------------
    wd = sample.get("weight_diagnostics", {}).get("student", {})
    if wd:
        weight_fields = []
        for key in [
            "parameter_l2_total",
            "weight_l2_total",
            "bias_l2_total",
            "log_spectral_complexity_norm",
            "spectral_complexity_norm",
            "num_spectral_tensors",
        ]:
            if key in wd:
                weight_fields.append(f"{key}={_fmt(wd[key])}")

        if weight_fields:
            print(f"{prefix} student weights: " + " ".join(weight_fields), flush=True)

    print("=" * 100, flush=True)
    print("", flush=True)


def _observe_save_checkpoint_and_print(
    *,
    observer,
    student,
    teacher,
    optimizer,
    train_model_state,
    args,
    epoch,
    step,
    do_checkpoint=True,
):
    """
    Common helper used at every validation/saving point.
    """
    observer.observe(epoch, step, student, teacher)
    observer.save()
    print_observation_summary(observer, args)

    if do_checkpoint:
        observer.checkpoint(student, teacher, optimizer, train_model_state)


def main(args):
    import torch

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

    import model

    resumption_state = (
        _torch_load_compat(
            args.resume_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if args.resume
        else None
    )

    student = model.init_model(args, resumption_state=resumption_state)
    teacher = model.init_teacher(student, args, resumption_state=resumption_state)

    print(
        "Initialized models with device:",
        args.device,
        " student device:",
        next(student.parameters()).device,
        " teacher device:",
        next(teacher.parameters()).device if teacher is not None else None,
        flush=True,
    )

    import train

    optimizer = train.init_optimizer(args, student, resumption_state=resumption_state)

    import observables

    observer_state = (
        _torch_load_compat(
            args.resume_observables_path,
            map_location="cpu",
            weights_only=False,
        )
        if args.resume and args.resume_observables_path is not None
        else None
    )

    steps_per_epoch = len(train_dataloader)
    max_steps = steps_per_epoch * int(args.Nepoch)
    observe_by_step = bool(getattr(args, "observe_by_step", False))

    observer = observables.init_observer(
        steps_per_epoch,
        train_eval_dataloader,
        test_dataloader,
        args,
        observer_state=observer_state,
        rhm_data=rhm_data,
    )

    print("Initialized observer.", flush=True)
    print(
        "Observation schedule unit:",
        "step" if observe_by_step else "epoch",
        flush=True,
    )
    print("Steps per epoch:", steps_per_epoch, "max_steps:", max_steps, flush=True)
    if observe_by_step:
        print("Observation/save steps:", observer.sample_steps, flush=True)
    else:
        print("Observation/save epochs:", observer.sample_epochs, flush=True)
    if getattr(args, "checkpoint_by_step", False):
        print("Extra checkpoint-only steps:", observer.checkpoint_steps, flush=True)

    starting_step = 0
    already_observed_steps: set[int] = set()
    already_checkpointed_steps: set[int] = set()

    if args.resume:
        train_model_state_resume = resumption_state["train_model_state"]
        starting_step = int(train_model_state_resume["step"])
        observer.clear_samples_after_step(starting_step)

        for sample in observer.samples:
            if "step" in sample:
                already_observed_steps.add(int(sample["step"]))

        # We do not know all checkpoint files from the observer object, but every
        # observed point also created a checkpoint unless checkpointing was disabled.
        already_checkpointed_steps.update(already_observed_steps)

        print(
            f"Resuming from step={starting_step}, "
            f"kept {len(observer.samples)} previous observation samples.",
            flush=True,
        )
    else:
        # Optional diagnostic before training. This remains step 0 in both modes.
        if getattr(args, "observe_initial", False):
            initial_state = {"step": 0, "epoch": 0}
            _observe_save_checkpoint_and_print(
                observer=observer,
                student=student,
                teacher=teacher,
                optimizer=optimizer,
                train_model_state=initial_state,
                args=args,
                epoch=0,
                step=0,
                do_checkpoint=True,
            )
            already_observed_steps.add(0)
            already_checkpointed_steps.add(0)

    training_generator = train.train_model(
        student,
        optimizer,
        train_dataloader,
        args.Nepoch,
        teacher=teacher,
        ema_alpha=args.ema_alpha,
        starting_step=starting_step,
        use_autojac=args.use_autojac,
        rhm_data=rhm_data,
        enable_probes=args.enable_probes,
    )

    last_train_model_state = None

    for train_model_state in training_generator:
        last_train_model_state = train_model_state

        step = int(train_model_state["step"])
        completed_epoch = step // steps_per_epoch
        end_of_epoch = (step % steps_per_epoch) == 0

        # ------------------------------------------------------------------
        # Main observation/checkpoint schedule.
        #   default: completed epochs in observer.sample_epochs
        #   --observe_by_step: optimizer steps in observer.sample_steps
        # In both modes, the same points are used for metric computation and
        # checkpoint saving.
        # ------------------------------------------------------------------
        if observe_by_step:
            should_observe = step in observer.sample_steps
        else:
            should_observe = end_of_epoch and completed_epoch in observer.sample_epochs

        should_observe = should_observe and step not in already_observed_steps

        if should_observe:
            _observe_save_checkpoint_and_print(
                observer=observer,
                student=student,
                teacher=teacher,
                optimizer=optimizer,
                train_model_state=train_model_state,
                args=args,
                epoch=completed_epoch,
                step=step,
                do_checkpoint=True,
            )
            already_observed_steps.add(step)
            already_checkpointed_steps.add(step)

        # ------------------------------------------------------------------
        # Optional extra checkpoint-only step schedule. This is independent of
        # the main observation/save schedule and is normally unnecessary.
        # ------------------------------------------------------------------
        should_checkpoint_step = (
            getattr(args, "checkpoint_by_step", False)
            and step in observer.checkpoint_steps
            and step not in already_checkpointed_steps
        )

        if should_checkpoint_step:
            observer.checkpoint(student, teacher, optimizer, train_model_state)
            already_checkpointed_steps.add(step)

    # ----------------------------------------------------------------------
    # Safety final save. If the final step was not already observed for any
    # reason, observe it and save a final checkpoint.
    # ----------------------------------------------------------------------
    if last_train_model_state is None:
        # This should happen only for pathological settings, e.g. empty dataloader.
        final_step = starting_step
        final_epoch = starting_step // max(1, steps_per_epoch)
        final_state = {"step": final_step, "epoch": final_epoch}
    else:
        final_step = int(last_train_model_state["step"])
        final_epoch = final_step // max(1, steps_per_epoch)
        final_state = last_train_model_state

    if final_step not in already_observed_steps:
        _observe_save_checkpoint_and_print(
            observer=observer,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            train_model_state=final_state,
            args=args,
            epoch=final_epoch,
            step=final_step,
            do_checkpoint=True,
        )
        already_observed_steps.add(final_step)
        already_checkpointed_steps.add(final_step)

    print(
        "Finished run on step/epoch:",
        final_step,
        "/",
        final_epoch,
        flush=True,
    )

    return observer, student, teacher


if __name__ == "__main__":
    parser = init.init_parser()
    args = parser.parse_args()
    main(args)
