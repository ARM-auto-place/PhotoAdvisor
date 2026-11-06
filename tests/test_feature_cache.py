import torch
import pytest

from scripts.feature_cache import read_cache, write_cache
from scripts.tune_natural_head import loss_weights


def example():
    return dict(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        y=torch.tensor([0, 1], dtype=torch.long),
    )


SCHEMA = {"x": ((2, 2), torch.float32), "y": ((2,), torch.long)}


def test_cache_roundtrip_supports_head_gradients(tmp_path):
    path = tmp_path / "features.pt"
    metadata = {"model": "one", "dataset": "two"}
    source = example()
    write_cache(path, metadata, source)
    restored = read_cache(path, metadata, SCHEMA, {"y": source["y"]})
    assert torch.equal(restored["x"], source["x"])
    head = torch.nn.Linear(2, 2)
    head(restored["x"]).sum().backward()
    assert head.weight.grad is not None
    with pytest.raises(ValueError, match="provenance"):
        read_cache(path, dict(metadata, model="different"), SCHEMA)
    with pytest.raises(ValueError, match="target order"):
        read_cache(path, metadata, SCHEMA, {"y": torch.tensor([1, 0])})
    with path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        read_cache(path, metadata, SCHEMA)


@pytest.mark.parametrize("bad", ["shape", "dtype", "nan", "split"])
def test_cache_cannot_hide_invalid_tensors_with_a_valid_checksum(tmp_path, bad):
    source = example()
    path = tmp_path / "features.pt"
    if bad == "shape":
        source["x"] = torch.ones(1, 2)
    elif bad == "dtype":
        source["x"] = source["x"].double()
    elif bad == "nan":
        source["x"][0, 0] = float("nan")
    else:
        source["test_features"] = torch.ones(1)
    if bad == "split":
        with pytest.raises(ValueError, match="test tensors"):
            write_cache(path, {}, source)
    else:
        write_cache(path, {}, source)
        with pytest.raises(ValueError, match="tensor|nonfinite"):
            read_cache(path, {}, SCHEMA)


def test_weak_and_derived_loss_factors_multiply_in_feature_order():
    rows = [
        dict(split="train", review_status="reviewed", derived=False),
        dict(split="train", review_status="source_annotated", derived=False),
        dict(split="train", review_status="reviewed", derived=True),
        dict(split="train", review_status="source_annotated", derived=True),
    ]
    assert loss_weights(rows, 0.35, 0.25).tolist() == pytest.approx(
        [1.0, 1.0, 0.35, 0.35, 0.25, 0.25, 0.0875, 0.0875]
    )
    assert loss_weights(rows, 0.0, 0.0).tolist() == [
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    with pytest.raises(ValueError, match="training rows"):
        loss_weights([dict(rows[0], split="calibration")], 0.35, 0.25)
