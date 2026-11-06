import copy

import pytest
import torch

from app.trained import LABELS
from scripts.adapt_head import admissible, fit_anchored, predictions


def test_adaptation_cannot_rewrite_teacher_or_select_a_tied_model():
    teacher = torch.nn.Linear(3, len(LABELS))
    with torch.no_grad():
        teacher.weight.zero_()
        teacher.bias.fill_(-10)
        teacher.weight[LABELS.index("front"), 0] = 20
        teacher.weight[LABELS.index("under_trunk"), 1] = 20
        teacher.weight[LABELS.index("other"), 2] = 20
    before = copy.deepcopy(teacher.state_dict())
    x = torch.eye(3)
    y = torch.tensor([LABELS.index(c) for c in ["front", "under_trunk", "other"]])
    rows = [
        dict(path=str(i), label=c)
        for i, c in enumerate(["front", "under_trunk", "other"])
    ]
    best, state, baseline, history = fit_anchored(
        teacher,
        x,
        y,
        torch.ones(3),
        torch.ones(len(LABELS)),
        torch.ones(3, dtype=torch.bool),
        x,
        rows,
        strengths=(0.1,),
        epochs=10,
    )
    assert state is None and best["epoch"] == 0
    assert best["metrics"] == baseline and len(history) == 1
    assert all(torch.equal(before[k], v) for k, v in teacher.state_dict().items())
    assert all(p.grad is None for p in teacher.parameters())
    with pytest.raises(ValueError, match="old training"):
        fit_anchored(
            teacher,
            x,
            y,
            torch.ones(3),
            torch.ones(len(LABELS)),
            torch.zeros(3, dtype=torch.bool),
            x,
            rows,
            epochs=10,
        )


def test_front_and_underfloor_recall_guard_survives_overall_metric_gain():
    baseline = dict(
        accuracy=0.8,
        core_accuracy=0.8,
        other_false_positives=5,
        per_class=[dict(label=c, recall=1.0) for c in ["front", "under_trunk"]],
    )
    better = dict(baseline, accuracy=0.9, core_accuracy=0.9, other_false_positives=4)
    assert admissible(better, baseline)
    for label in ["front", "under_trunk"]:
        changed = copy.deepcopy(better)
        next(r for r in changed["per_class"] if r["label"] == label)["recall"] = 0.9
        assert not admissible(changed, baseline)


def test_residual_starts_at_baseline_and_evaluation_never_samples_dropout():
    from app.trained import ResidualHead

    head = ResidualHead(features=3)
    x = torch.randn(12, 3)
    assert torch.equal(head(x), head.base(x))
    before = head.base.weight.detach().clone()
    optimizer = torch.optim.AdamW(p for p in head.parameters() if p.requires_grad)
    loss = head(x).square().mean()
    loss.backward()
    optimizer.step()
    assert torch.equal(before, head.base.weight)
    assert head.base.weight.grad is None
    # A nonzero learned branch must still use deterministic eval for selection.
    rows = [dict(path=str(i), label="front") for i in range(len(x))]
    rng = torch.get_rng_state().clone()
    first = predictions(head, x, rows)
    second = predictions(head, x, rows)
    assert first == second and head.training
    assert torch.equal(rng, torch.get_rng_state())


def test_residual_artifact_loads_in_mapped_and_eager_runtime(tmp_path, monkeypatch):
    import hashlib
    import json
    from PIL import Image
    from app.settings import Settings
    from app.trained import (
        FusedAngleNet,
        ResidualFusedAngleNet,
        TrainedVision,
        FUSED_FEATURE_DIM,
        RESIDUAL_FUSED_ARCHITECTURE,
    )

    def tiny_parent(self):
        torch.nn.Module.__init__(self)
        self.features = torch.nn.Linear(1, 1)

    monkeypatch.setattr(FusedAngleNet, "__init__", tiny_parent)
    monkeypatch.setattr(
        ResidualFusedAngleNet,
        "embed",
        lambda self, x: torch.ones(len(x), FUSED_FEATURE_DIM, device=x.device),
    )
    model = ResidualFusedAngleNet().eval()
    with torch.no_grad():
        model.classifier.base.weight.zero_()
        model.classifier.base.bias.fill_(-3)
        model.classifier.base.bias[0] = 3
    target = tmp_path / "weights.pt"
    torch.save(model.state_dict(), target)
    manifest = dict(
        architecture=RESIDUAL_FUSED_ARCHITECTURE,
        labels=list(LABELS),
        input_size=224,
        preprocessing=model.preprocessing,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        dataset_id="synthetic-contract-only",
        calibration={"validated": False, "classes": {}},
        heldout_approval=False,
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    settings = Settings(
        _env_file=None, cv_backend="trained", trained_model_dir=tmp_path
    )
    image = Image.new("RGB", (300, 300))
    try:
        results = [
            TrainedVision(settings, memory_efficient=mapped).predict([image])[0]
            for mapped in (True, False)
        ]
        assert all(
            r["candidate"] == "front" and r["recognition"] == "uncertain"
            for r in results
        )
        assert results[0]["score"] == pytest.approx(results[1]["score"])
    finally:
        image.close()
