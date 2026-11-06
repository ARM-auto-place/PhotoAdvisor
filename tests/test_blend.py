import pytest
import torch

from scripts.blend_heads import blend_states


def test_head_blend_equals_average_logits_without_second_encoder():
    generator = torch.Generator().manual_seed(9)
    a = {'features.encoder':torch.ones(3), 'classifier.weight':torch.randn(2,4,generator=generator),
         'classifier.bias':torch.randn(2,generator=generator)}
    b = dict(a, **{'classifier.weight':torch.randn(2,4,generator=generator),
                   'classifier.bias':torch.randn(2,generator=generator)})
    x = torch.randn(5,4,generator=generator)
    merged = blend_states(a,b)
    def predict(s):
        return torch.nn.functional.linear(x,s['classifier.weight'],s['classifier.bias'])
    assert torch.allclose(predict(merged), (predict(a)+predict(b))/2, atol=1e-6)


def test_different_visual_features_cannot_be_silently_blended():
    a = {'features.encoder':torch.ones(3), 'classifier.weight':torch.ones(2,4), 'classifier.bias':torch.zeros(2)}
    b = dict(a, **{'features.encoder':torch.zeros(3)})
    with pytest.raises(ValueError,match='encoder weights differ'):
        blend_states(a,b)
