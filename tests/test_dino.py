import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from app.trained import (
    DinoAngleNet,
    TripleAngleNet,
    DINO_FEATURE_DIM,
    FUSED_FEATURE_DIM,
    TRIPLE_FEATURE_DIM,
    FEATURE_DIM,
)
from scripts.train_views import verify_dino_source


def test_dino_features_keep_global_and_spatial_information():
    class Encoder(torch.nn.Module):
        def forward(self, pixel_values):
            tokens = (
                torch.arange(257, dtype=torch.float32)
                .view(1, 257, 1)
                .expand(len(pixel_values), 257, 384)
            )
            return SimpleNamespace(last_hidden_state=tokens)

    model = DinoAngleNet(encoder=Encoder())
    features = model.embed(torch.zeros(2, 3, 224, 224))
    assert features.shape == (2, DINO_FEATURE_DIM)
    assert torch.allclose(features.norm(dim=1), torch.ones(2))
    spatial = features[:, 384:].reshape(2, 384, 4)
    assert not torch.equal(spatial[:, :, 0], spatial[:, :, 3])
    with torch.device("meta"):
        unloaded = DinoAngleNet()
    assert all(p.is_meta for p in unloaded.parameters())


def test_three_encoders_have_equal_mass_and_dino_receives_imagenet_input():
    class Fused(torch.nn.Module):
        def embed(self, images):
            values = torch.zeros(len(images), FUSED_FEATURE_DIM)
            values[:, 0] = 2**-0.5
            values[:, FEATURE_DIM] = 2**-0.5
            return values

    class Dino(torch.nn.Module):
        def embed(self, images):
            assert torch.count_nonzero(images) == 0
            values = torch.zeros(len(images), DINO_FEATURE_DIM)
            values[:, 0] = 1
            return values

    model = TripleAngleNet(fused=Fused(), dino=Dino())
    images = torch.stack(
        (torch.zeros(1, 3, 224, 224), torch.ones(1, 3, 224, 224)), dim=1
    )
    features = model.embed(images)
    assert features.shape == (1, TRIPLE_FEATURE_DIM)
    masses = [
        features[:, :FEATURE_DIM].norm(),
        features[:, FEATURE_DIM:FUSED_FEATURE_DIM].norm(),
        features[:, FUSED_FEATURE_DIM:].norm(),
    ]
    assert [float(v) for v in masses] == pytest.approx([3**-0.5] * 3)


def test_dino_pretrained_source_detects_changes(tmp_path):
    names = ("config.json", "model.safetensors", "preprocessor_config.json")
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
    manifest = dict(
        model_id="facebook/dinov2-small",
        revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
        license="Apache-2.0",
        sha256={
            n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest() for n in names
        },
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_dino_source(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_dino_source(tmp_path)
