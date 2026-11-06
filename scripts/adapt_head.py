"""Adapt a frozen classifier while anchoring old training logits; calibration only."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from app.evaluation import load_manifest
from app.settings import Settings
from app.trained import (
    LABELS,
    TrainedVision,
    ResidualHead,
    FUSED_ARCHITECTURE,
    RESIDUAL_FUSED_ARCHITECTURE,
)
from scripts.select_checkpoint import calibration_rows
from scripts.stabilize_head import guarded_summary
from scripts.train_views import (
    check_rows,
    check_warm_start_splits,
    extract,
    head_class_weights,
    head_sample_weights,
    verify_parent_dataset,
    effective_head_counts,
)


def recall(metrics, label):
    return next(r["recall"] for r in metrics["per_class"] if r["label"] == label)


def admissible(metrics, baseline):
    return (
        metrics["accuracy"] >= baseline["accuracy"]
        and metrics["core_accuracy"] >= baseline["core_accuracy"]
        and metrics["other_false_positives"] <= baseline["other_false_positives"]
        and all(
            recall(metrics, c) >= recall(baseline, c) for c in ("front", "under_trunk")
        )
    )


def predictions(head, x, rows):
    training = head.training
    try:
        head.eval()
        with torch.no_grad():
            indices = head(x).argmax(1).tolist()
    finally:
        head.train(training)
    return [
        dict(path=r["path"], label=r["label"], candidate=LABELS[i])
        for r, i in zip(rows, indices)
    ]


def fit_anchored(
    teacher,
    x,
    y,
    sample_weights,
    class_weights,
    old_mask,
    cx,
    calibration,
    strengths=(0.1, 0.5),
    epochs=200,
    lr=0.003,
    initial=None,
):
    if not old_mask.any() or old_mask.shape != (len(x),):
        raise ValueError("old training examples required for anchoring")
    baseline_predictions = predictions(teacher, cx, calibration)
    baseline = guarded_summary(baseline_predictions)
    best = dict(
        metrics=baseline, predictions=baseline_predictions, epoch=0, strength=None
    )
    best_state = None
    history = []
    # Detach the teacher: fitting cannot modify its parameters or accumulate gradients.
    with torch.no_grad():
        old_logits = teacher(x[old_mask]).detach().clone()
    for strength in strengths:
        if strength < 0:
            raise ValueError("nonnegative anchor required")
        head = copy.deepcopy(initial if initial is not None else teacher)
        head.requires_grad_(True)
        if isinstance(head, ResidualHead):
            head.base.requires_grad_(False)
        head.train()
        optimizer = torch.optim.AdamW(
            (p for p in head.parameters() if p.requires_grad), lr=lr, weight_decay=0.01
        )
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            logits = head(x)
            losses = nn.functional.cross_entropy(
                logits, y, weight=class_weights, reduction="none"
            )
            ce = (losses * sample_weights).sum() / (
                class_weights[y] * sample_weights
            ).sum()
            anchor = nn.functional.mse_loss(logits[old_mask], old_logits)
            loss = ce + strength * anchor
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                current_predictions = predictions(head, cx, calibration)
                metrics = guarded_summary(current_predictions)
                history.append(
                    dict(
                        epoch=epoch,
                        strength=strength,
                        ce=float(ce.detach()),
                        anchor=float(anchor.detach()),
                        metrics=metrics,
                        eligible=admissible(metrics, baseline),
                    )
                )
                if (
                    admissible(metrics, baseline)
                    and metrics["macro_f1"] > best["metrics"]["macro_f1"]
                ):
                    best = dict(
                        epoch=epoch,
                        strength=strength,
                        metrics=metrics,
                        predictions=current_predictions,
                    )
                    best_state = {
                        k: v.detach().clone() for k, v in head.state_dict().items()
                    }
        print(
            json.dumps(
                dict(
                    strength=strength,
                    epoch=epochs,
                    best_macro_f1=best["metrics"]["macro_f1"],
                    improved=best_state is not None,
                )
            ),
            flush=True,
        )
    return best, best_state, baseline, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--residual", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        parser.error("new immutable paths required")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    started = time.perf_counter()
    rows = load_manifest(args.manifest, LABELS, LABELS[:-1])
    counts = check_rows(rows, False)
    parent_manifest = json.loads(
        (args.parent / "manifest.json").read_text(encoding="utf-8")
    )
    if args.residual and parent_manifest["architecture"] != FUSED_ARCHITECTURE:
        parser.error("residual adaptation requires a linear fused parent")
    previous = load_manifest(Path(parent_manifest["dataset"]), LABELS, LABELS[:-1])
    verify_parent_dataset(previous, parent_manifest)
    check_warm_start_splits(previous, rows)
    cal = calibration_rows(args.manifest, [args.parent])
    train = [r for r in rows if r["split"] == "train"]
    vision = TrainedVision(
        Settings(_env_file=None, cv_backend="trained", trained_model_dir=args.parent)
    )
    print(json.dumps(dict(phase="extract_train", photos=len(train))), flush=True)
    x, y = extract(vision.model, train, args.manifest.parent, True, 16)
    print(json.dumps(dict(phase="extract_calibration", photos=len(cal))), flush=True)
    cx, _ = extract(vision.model, cal, args.manifest.parent, False, 16)
    x, y, cx = x.clone(), y.clone(), cx.clone()
    old_hashes = {r["sha256"] for r in previous if r["split"] == "train"}
    old_mask = torch.tensor(
        [r["sha256"] in old_hashes for r in train for _ in range(2)], dtype=torch.bool
    )
    samples = head_sample_weights(train, True, 0.35)
    classes = head_class_weights(y, samples, 1.0)
    initial = None
    if args.residual:
        initial = ResidualHead()
        initial.base.load_state_dict(vision.model.classifier.state_dict())
    best, state, baseline, history = fit_anchored(
        vision.model.classifier,
        x,
        y,
        samples,
        classes,
        old_mask,
        cx,
        cal,
        initial=initial,
    )
    selected = str(args.parent)
    if state is not None:
        # Replace only the fitted linear head; all active encoder tensors stay unchanged.
        head = (
            copy.deepcopy(initial)
            if args.residual
            else nn.Linear(vision.model.classifier.in_features, len(LABELS))
        )
        head.load_state_dict(state)
        vision.model.classifier = head
        args.output.mkdir()
        target = args.output / "weights.pt"
        torch.save(vision.model.state_dict(), target)
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest = dict(
            parent_manifest,
            sha256=digest,
            dataset=str(args.manifest),
            dataset_id=hashlib.sha256(
                json.dumps(rows, sort_keys=True).encode()
            ).hexdigest(),
            initialized_from=vision.version,
            architecture=RESIDUAL_FUSED_ARCHITECTURE
            if args.residual
            else parent_manifest["architecture"],
            training_method="frozen_residual_head_teacher_anchor_v1"
            if args.residual
            else "frozen_head_teacher_logit_anchor_v1",
            effective_head_training_counts=effective_head_counts(train, True, 0.35),
            independent_groups=counts,
            anchor_strength=best["strength"],
            best_epoch=best["epoch"],
            head_lr=0.003,
            calibration={"validated": False, "classes": {}},
            heldout_approval=False,
        )
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        selected = str(args.output)
    report = dict(
        kind="calibration_only_frozen_head_adaptation",
        residual=args.residual,
        selected_model=selected,
        parent=str(args.parent),
        parent_sha256=parent_manifest["sha256"],
        dataset=str(args.manifest),
        dataset_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        baseline=baseline,
        selected=best,
        history=history,
        test_evaluation_deferred=True,
        production_approved=False,
        elapsed_seconds=time.perf_counter() - started,
        old_teacher_views=int(old_mask.sum()),
        new_training_views=int((~old_mask).sum()),
        strengths=[0.1, 0.5],
        epochs=200,
    )
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(dict(selected_model=selected, metrics=best["metrics"])), flush=True
    )


if __name__ == "__main__":
    main()
