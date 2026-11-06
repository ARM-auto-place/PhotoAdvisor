"""Compare natural-photo head losses on one frozen, integrity-checked feature cache."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filelock import FileLock
import torch
from torch import nn
import transformers

from app.evaluation import load_manifest
from app.settings import Settings
from app.trained import LABELS, TrainedVision
from scripts.adapt_head import admissible
from scripts.feature_cache import digest, read_cache, write_cache
from scripts.select_checkpoint import calibration_rows, evaluate_calibration
from scripts.stabilize_head import guarded_summary
from scripts.train_views import (
    FLIPPED,
    check_rows,
    extract,
    head_class_weights,
    macro_cross_entropy,
    verify_parent_dataset,
)


def loss_weights(rows, weak_weight, derived_weight, flip=True):
    if not 0 <= weak_weight <= 1 or not 0 <= derived_weight <= 1:
        raise ValueError("loss weights must be in [0,1]")
    values = []
    for row in rows:
        if row["split"] != "train":
            raise ValueError("loss weights require training rows only")
        weight = weak_weight if row.get("review_status") == "source_annotated" else 1.0
        if row.get("derived"):
            weight *= derived_weight
        values.extend([weight] * (2 if flip else 1))
    return torch.tensor(values, dtype=torch.float32)


def summarize_natural(predictions, rows):
    result = guarded_summary(predictions)
    by_path = {r["path"]: r for r in rows}
    natural = [p for p in predictions if not by_path[p["path"]].get("derived", False)]
    metrics = guarded_summary(natural)
    result.update(
        natural_accuracy=metrics["accuracy"],
        natural_core_accuracy=metrics["core_accuracy"],
        natural_macro_f1=metrics["macro_f1"],
        natural_count=len(natural),
    )
    return result


def eligible(metrics, baseline):
    return (
        admissible(metrics, baseline)
        and metrics["macro_f1"] >= baseline["macro_f1"]
        and metrics["natural_accuracy"] >= baseline["natural_accuracy"]
        and metrics["natural_core_accuracy"] >= baseline["natural_core_accuracy"]
    )


def objective(metrics):
    return metrics["natural_macro_f1"], metrics["macro_f1"], metrics["accuracy"]


def from_logits(logits, rows, bias=0.0):
    adjusted = logits.detach().clone()
    adjusted[:, LABELS.index("other")] += bias
    return [
        dict(path=r["path"], label=r["label"], candidate=LABELS[i])
        for r, i in zip(rows, adjusted.argmax(1).tolist())
    ]


def cached_features(cache, vision, parent_meta, manifest, rows, train, cal):
    dataset_id = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    metadata = dict(
        version=1,
        parent_sha256=parent_meta["sha256"],
        dataset_id=dataset_id,
        manifest_sha256=digest(manifest),
        architecture=parent_meta["architecture"],
        preprocessing=parent_meta["preprocessing"],
        flip_train=True,
        device="cpu",
        threads=2,
        torch_version=str(torch.__version__),
        transformers_version=transformers.__version__,
        code={
            str(p): digest(p)
            for p in map(
                Path, ["app/trained.py", "app/imaging.py", "scripts/train_views.py"]
            )
        },
    )
    targets = dict(
        train_targets=torch.tensor(
            [LABELS.index(c) for r in train for c in [r["label"], FLIPPED[r["label"]]]],
            dtype=torch.long,
        ),
        calibration_targets=torch.tensor(
            [LABELS.index(r["label"]) for r in cal], dtype=torch.long
        ),
    )
    dimension = vision.model.classifier.in_features
    schema = dict(
        train_features=((len(train) * 2, dimension), torch.float32),
        train_targets=((len(train) * 2,), torch.long),
        calibration_features=((len(cal), dimension), torch.float32),
        calibration_targets=((len(cal),), torch.long),
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache) + ".lock", timeout=0):
        if cache.exists():
            print("Using verified feature cache", flush=True)
        else:
            print(
                json.dumps(dict(phase="extract_train", photos=len(train))), flush=True
            )
            x, y = extract(vision.model, train, manifest.parent, True, 16)
            print(
                json.dumps(dict(phase="extract_calibration", photos=len(cal))),
                flush=True,
            )
            cx, cy = extract(vision.model, cal, manifest.parent, False, 16)
            current = load_manifest(manifest, LABELS, LABELS[:-1])
            if (
                hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
                != dataset_id
            ):
                raise ValueError("dataset changed during feature extraction")
            write_cache(
                cache,
                metadata,
                dict(
                    train_features=x.clone(),
                    train_targets=y.clone(),
                    calibration_features=cx.clone(),
                    calibration_targets=cy.clone(),
                ),
            )
        return read_cache(cache, metadata, schema, targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--warm-anchor", type=float, default=None)
    args = parser.parse_args()
    if args.warm_anchor is not None and args.warm_anchor < 0:
        parser.error("anchor must be nonnegative")
    if args.output.exists() or args.report.exists():
        parser.error("new immutable paths required")
    torch.set_num_threads(2)
    started = time.perf_counter()
    rows = load_manifest(args.manifest, LABELS, LABELS[:-1])
    counts = check_rows(rows, False)
    parent_meta = json.loads(
        (args.parent / "manifest.json").read_text(encoding="utf-8")
    )
    previous = load_manifest(Path(parent_meta["dataset"]), LABELS, LABELS[:-1])
    verify_parent_dataset(previous, parent_meta)
    if rows != previous:
        raise ValueError(
            "natural-head experiment requires the unchanged parent dataset"
        )
    cal = calibration_rows(args.manifest, [args.parent])
    train = [r for r in rows if r["split"] == "train"]
    vision = TrainedVision(
        Settings(_env_file=None, cv_backend="trained", trained_model_dir=args.parent)
    )
    if not isinstance(vision.model.classifier, nn.Linear):
        raise ValueError("linear parent required")
    tensors = cached_features(
        args.cache, vision, parent_meta, args.manifest, rows, train, cal
    )
    x, y, cx, cy = [
        tensors[k]
        for k in [
            "train_features",
            "train_targets",
            "calibration_features",
            "calibration_targets",
        ]
    ]
    natural = torch.tensor([not r.get("derived", False) for r in cal], dtype=torch.bool)
    if set(cy[natural].tolist()) != set(range(len(LABELS))):
        raise ValueError("all classes need natural calibration support")
    with torch.no_grad():
        baseline_predictions = from_logits(vision.model.classifier(cx), cal)
    baseline = summarize_natural(baseline_predictions, cal)
    results = [
        dict(name="baseline", metrics=baseline, predictions=baseline_predictions)
    ]
    best = results[0]
    best_state = None
    training = []
    with torch.no_grad():
        teacher_logits = vision.model.classifier(x).detach().clone()
    for weak, derived in [(0.35, 0.25), (0.1, 0.25), (0.0, 0.0)]:
        torch.manual_seed(42)
        head = nn.Linear(x.shape[1], len(LABELS))
        if args.warm_anchor is not None:
            head.load_state_dict(vision.model.classifier.state_dict())
        samples = loss_weights(train, weak, derived)
        mass = torch.bincount(y, weights=samples, minlength=len(LABELS))
        if not (mass > 0).all():
            raise ValueError("no effective training examples for a class")
        weights = head_class_weights(y, samples, 1.0)
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=0.003 if args.warm_anchor is not None else 0.03,
            weight_decay=0.01 if args.warm_anchor is not None else 0.05,
        )
        with torch.no_grad():
            lowest = (
                float(macro_cross_entropy(head(cx)[natural], cy[natural]))
                if args.warm_anchor is not None
                else float("inf")
            )
        saved = copy.deepcopy(head.state_dict())
        epoch_best = 0
        for epoch in range(1, 301):
            optimizer.zero_grad(set_to_none=True)
            logits = head(x)
            losses = nn.functional.cross_entropy(
                logits, y, weight=weights, reduction="none"
            )
            loss = (losses * samples).sum() / (weights[y] * samples).sum()
            if args.warm_anchor is not None:
                loss = loss + args.warm_anchor * nn.functional.mse_loss(
                    logits, teacher_logits
                )
            if not torch.isfinite(loss):
                raise ValueError("nonfinite head loss")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                value = float(macro_cross_entropy(head(cx)[natural], cy[natural]))
            if value < lowest:
                lowest = value
                epoch_best = epoch
                saved = copy.deepcopy(head.state_dict())
        head.load_state_dict(saved)
        with torch.no_grad():
            logits = head(cx)
        name = f"weak-{weak}-derived-{derived}"
        training.append(
            dict(
                name=name,
                weak_weight=weak,
                derived_weight=derived,
                best_epoch=epoch_best,
                natural_macro_ce=lowest,
                effective_training_mass=mass.tolist(),
            )
        )
        for bias in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            predictions = from_logits(logits, cal, bias)
            metrics = summarize_natural(predictions, cal)
            result = dict(
                name=name,
                weak_weight=weak,
                derived_weight=derived,
                other_logit_bias=bias,
                best_epoch=epoch_best,
                metrics=metrics,
                predictions=predictions,
            )
            results.append(result)
            if eligible(metrics, baseline) and objective(metrics) > objective(
                best["metrics"]
            ):
                best = result
                best_state = {
                    k: v.detach().clone() for k, v in head.state_dict().items()
                }
        print(
            json.dumps(
                dict(
                    recipe=name,
                    best_epoch=epoch_best,
                    best=best["name"],
                    best_natural_f1=best["metrics"]["natural_macro_f1"],
                )
            ),
            flush=True,
        )
    selected = args.parent
    if best_state is not None:
        best_state["bias"][LABELS.index("other")] += best["other_logit_bias"]
        head = nn.Linear(x.shape[1], len(LABELS))
        head.load_state_dict(best_state)
        vision.model.classifier = head
        args.output.mkdir()
        target = args.output / "weights.pt"
        torch.save(vision.model.state_dict(), target)
        metadata = dict(
            parent_meta,
            sha256=digest(target),
            parent_sha256=parent_meta["sha256"],
            initialized_from=vision.version,
            training_method=(
                "anchored_review_weighted_natural_head"
                if args.warm_anchor is not None
                else "review_weighted_natural_calibration_head"
            ),
            head_anchor=args.warm_anchor,
            head_weak_weight=best["weak_weight"],
            head_derived_weight=best["derived_weight"],
            other_logit_bias=best["other_logit_bias"],
            best_epoch=best["best_epoch"],
            effective_head_training_counts={
                LABELS[i]: int(
                    (
                        (y == i)
                        & (
                            loss_weights(
                                train, best["weak_weight"], best["derived_weight"]
                            )
                            > 0
                        )
                    ).sum()
                )
                for i in range(len(LABELS))
            },
            effective_head_training_mass=next(
                t["effective_training_mass"]
                for t in training
                if t["name"] == best["name"]
            ),
            independent_groups=counts,
            calibration={"validated": False, "classes": {}},
            heldout_approval=False,
        )
        (args.output / "manifest.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        selected = args.output
    actual = evaluate_calibration(args.manifest, selected)
    actual["metrics"] = summarize_natural(actual["predictions"], cal)
    if actual["predictions"] != best["predictions"] or not eligible(
        actual["metrics"], baseline
    ):
        raise ValueError("serialized runtime prediction/guard mismatch")
    report = dict(
        kind="natural_calibration_loss_weight_selection",
        selected_model=str(selected),
        parent=str(args.parent),
        manifest=str(args.manifest),
        manifest_sha256=digest(args.manifest),
        cache=str(args.cache),
        cache_sha256=digest(args.cache),
        training=training,
        results=results,
        selected=best,
        runtime_verified=actual,
        test_evaluation_deferred=True,
        production_approved=False,
        warm_anchor=args.warm_anchor,
        elapsed_seconds=time.perf_counter() - started,
    )
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(dict(selected_model=str(selected), metrics=actual["metrics"])),
        flush=True,
    )


if __name__ == "__main__":
    main()
