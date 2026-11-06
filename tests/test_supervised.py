"""Model contract tests use artificial weights; accuracy is measured separately."""
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image
import torch

from app.events import REQUIRED
from app.settings import Settings
from app.trained import (AngleNet, ARCHITECTURE, FEATURE_DIM, LABELS, TrainedVision,
                         CLIP_FEATURE_DIM, FUSED_FEATURE_DIM, ClipAngleNet, FusedAngleNet,
                         SupervisedPolicy, load_policy, prepare_image, restore_runtime_buffers)
from scripts.build_view_dataset import split_groups
from scripts.collect_commons import allowed_license, get
from scripts.train_views import (FLIPPED, balanced_sampling_weights, check_rows,
                                 check_warm_start_splits, effective_head_counts, head_class_weights,
                                 head_sample_weights, macro_cross_entropy)
from scripts.import_hitl import part_crop, proposed_side


def test_frozen_source_groups_never_reenter_training():
    records = [dict(vehicle_id=str(i), label=label)
               for i in range(10) for label in ('front', 'rear')]
    fixed = {'0': 'test', '1': 'calibration', '2': 'train'}
    result = split_groups(records, 42, fixed)
    assert all(result[g] == split for g, split in fixed.items())


def test_full_head_balancing_equalizes_effective_mass_of_rare_classes():
    targets = torch.tensor([0,0,0,1])
    samples = torch.tensor([1.,.35,.35,.35])
    weights = head_class_weights(targets,samples,power=1.)
    mass = torch.bincount(targets,weights=weights[targets]*samples)
    assert torch.allclose(mass[0],mass[1])
    assert torch.isfinite(weights).all()
    assert torch.equal(head_class_weights(targets,samples,power=0.),torch.ones(len(LABELS)))


def test_ignoring_weak_labels_cannot_claim_support_for_weak_only_classes():
    rows = [dict(label=c,review_status='source_annotated') for c in LABELS]
    rows += [dict(label='front',review_status='reviewed')]
    with pytest.raises(ValueError,match='no effective head training'):
        effective_head_counts(rows,False,0.)
    assert effective_head_counts(rows,False,0.,allow_partial=True)=={'front':1}
    assert set(effective_head_counts(rows,False,.35))==set(LABELS)


def test_train_only_weak_expansion_does_not_consume_reviewed_training_data():
    records = [dict(vehicle_id=f'gold-{i}', label='front', review_status='reviewed') for i in range(10)]
    records += [dict(vehicle_id=f'weak-{i}', label='front', review_status='source_annotated') for i in range(100)]
    fixed = {f'weak-{i}': 'train' for i in range(100)}
    result = split_groups(records, 42, fixed)
    assert sum(result[f'gold-{i}'] == 'train' for i in range(10)) == 6


def test_balancing_equalizes_classes_and_reduces_weak_label_influence():
    rows = [dict(label='front', review_status='reviewed'),
            dict(label='front', review_status='source_annotated'),
            dict(label='rear', review_status='reviewed')]
    weights = balanced_sampling_weights(rows)
    assert weights[:2].sum().item() == pytest.approx(weights[2].item())
    assert weights[1].item()/weights[0].item() == pytest.approx(.35)


def test_head_weights_match_original_and_mirrored_feature_order():
    rows = [dict(review_status='source_annotated'), dict(review_status='reviewed')]
    assert head_sample_weights(rows, True, .35).tolist() == pytest.approx([.35,.35,1.,1.])
    assert head_sample_weights(rows, False, .35).tolist() == pytest.approx([.35,1.])


def test_macro_selection_does_not_hide_rare_class_errors():
    logits = torch.tensor([[4.,0.], [4.,0.], [4.,0.]])
    targets = torch.tensor([0,0,1])
    actual = macro_cross_entropy(logits, targets)
    losses = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
    assert actual.item() == pytest.approx(((losses[:2].mean()+losses[2])/2).item())
    assert actual > losses.mean()


def test_clip_encoder_keeps_spatial_features_and_uses_own_normalization():
    from types import SimpleNamespace

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual_projection = torch.nn.Linear(768, 512, bias=False)
            self.vision_model = SimpleNamespace(post_layernorm=torch.nn.Identity())

        def forward(self, pixel_values):
            batch = len(pixel_values)
            return SimpleNamespace(image_embeds=torch.ones(batch, 512),
                last_hidden_state=torch.ones(batch, 50, 768))

    model = ClipAngleNet(encoder=Encoder()).eval()
    with torch.no_grad():
        features = model.embed(torch.zeros(2,3,224,224))
    assert features.shape == (2, CLIP_FEATURE_DIM)
    assert torch.allclose(features.norm(dim=1), torch.ones(2), atol=1e-5)
    image = Image.new('RGB', (400,200), 'red')
    try:
        clip_image = model.prepare_image(image)
        assert clip_image.shape == (3,224,224)
        assert not torch.allclose(clip_image, prepare_image(image))
        assert image.size == (400,200)
    finally:
        image.close()
    fused = FusedAngleNet(mobile=AngleNet(), clip=model).eval()
    with torch.no_grad():
        combined = fused.embed(torch.zeros(2,2,3,224,224))
    assert combined.shape == (2, FUSED_FEATURE_DIM)
    assert torch.allclose(combined.norm(dim=1), torch.ones(2), atol=1e-5)


def test_warm_start_cannot_move_previously_trained_car_to_test():
    with pytest.raises(ValueError, match='contaminated'):
        check_warm_start_splits([dict(vehicle_id='car-1', split='train')],
                                [dict(vehicle_id='car-1', split='test')])


def test_warm_start_cannot_hide_leakage_by_renaming_group():
    old = dict(vehicle_id='old', split='train', source_dataset='source', source_id='image')
    new = dict(old, vehicle_id='new', split='test')
    with pytest.raises(ValueError, match='contaminated'):
        check_warm_start_splits([old], [new])


def test_warm_start_cannot_forget_that_old_images_were_already_used():
    old=dict(vehicle_id='old-car',split='train',sha256='old-image')
    new=dict(vehicle_id='new-car',split='train',sha256='new-image')
    with pytest.raises(ValueError,match='retain parent dataset'):
        check_warm_start_splits([old],[new])


@pytest.mark.parametrize('split', ['calibration', 'test'])
def test_weak_source_labels_are_rejected_in_evaluation(split):
    row = dict(review_status='source_annotated', split=split,
               label_origin='human_part_mask_and_automatic_view_side_hint')
    with pytest.raises(ValueError, match='weak labels are train-only'):
        check_rows([row], True)


def test_ambiguous_part_side_is_not_inferred_from_automatic_area():
    obj = {'tags': [{'name': 'side', 'value': 'left'},
                    {'name': 'needs_review', 'value': 'no'},
                    {'name': 'side_reason', 'value': 'area ratio of paired objects'}]}
    assert proposed_side(obj, 'front-left') is None
    obj['tags'][-1]['value'] = 'single object assigned to visible side'
    assert proposed_side(obj, 'front-left') == 'left'
    assert proposed_side(obj, 'front-right') is None


def test_part_crop_requires_sufficient_visible_context():
    assert part_crop({'points': {'exterior': [[0,0], [5,0], [5,5]]}}, (800,600)) is None
    assert part_crop({'points': {'exterior': [[-1,0], [200,0], [200,200]]}}, (800,600)) is None
    assert part_crop({'points': {'exterior': [[100,100], [300,100], [300,300]]}}, (800,600)) == (30,30,370,370)


@pytest.fixture
def artifact(tmp_path):
    torch.set_num_threads(2)
    model = AngleNet().eval()
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
        model.classifier.bias[0] = 5
    weights = tmp_path/'weights.pt'
    torch.save(model.state_dict(), weights)
    manifest = dict(architecture=ARCHITECTURE, labels=list(LABELS), input_size=224,
        preprocessing='letterbox_rgb_imagenet_v1', sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        dataset_id='test-only', calibration={'validated': False, 'classes': {}}, heldout_approval=False)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return tmp_path


def test_twelve_views_and_one_rejection_class():
    assert LABELS[:-1] == REQUIRED
    assert LABELS[-1] == 'other'
    assert len(LABELS) == 13


@pytest.mark.parametrize('label', LABELS)
def test_flip_is_reversible_and_swaps_only_anatomical_sides(label):
    assert FLIPPED[FLIPPED[label]] == label
    if label in {'front', 'rear', 'engine_bay', 'trunk', 'under_trunk', 'interior', 'other'}:
        assert FLIPPED[label] == label
    else:
        assert FLIPPED[label] != label


def test_letterbox_preserves_whole_image_and_source():
    image = Image.new('RGB', (400, 200), 'red')
    tensor = prepare_image(image)
    assert tensor.shape == (3, 224, 224)
    assert image.size == (400, 200)
    mean, std = torch.tensor([.485,.456,.406]), torch.tensor([.229,.224,.225])
    assert torch.allclose(tensor[:, 0, 0], (torch.tensor([127/255]*3)-mean)/std, atol=1e-5)
    image.close()


def test_spatial_feature_dimension_and_normalization():
    torch.set_num_threads(2)
    model = AngleNet().eval()
    with torch.inference_mode():
        embedding = model.embed(torch.rand(2, 3, 224, 224))
        logits = model.classifier(embedding)
    assert embedding.shape == (2, FEATURE_DIM)
    assert logits.shape == (2, 13)
    assert torch.allclose(embedding.norm(dim=1), torch.ones(2), atol=1e-5)


def test_unvalidated_model_never_confirms_even_high_score(artifact):
    vision = TrainedVision(Settings(_env_file=None, cv_backend='trained', trained_model_dir=artifact))
    image = Image.new('RGB', (500, 300), 'red')
    prediction = vision.predict([image])[0]
    assert prediction['candidate'] == 'front'
    assert prediction['score'] > .9
    assert prediction['confidence'] is None
    assert prediction['recognition'] == 'uncertain'
    assert prediction['score_kind'] == 'softmax'
    assert vision.predict([]) == []
    image.close()


def test_memory_efficient_loading_preserves_real_weights_embeddings_and_scores(artifact):
    settings = Settings(_env_file=None,cv_backend='trained',trained_model_dir=artifact)
    eager = TrainedVision(settings,memory_efficient=False)
    mapped = TrainedVision(settings)
    for key,value in eager.model.state_dict().items():
        assert not mapped.model.state_dict()[key].is_meta
        assert torch.equal(value,mapped.model.state_dict()[key])
    image = Image.new('RGB',(480,320),'orange')
    try:
        first,second = eager.predict([image])[0],mapped.predict([image])[0]
    finally:
        image.close()
    assert first['candidate']==second['candidate']
    assert first['score']==pytest.approx(second['score'],abs=1e-7)
    assert torch.allclose(torch.from_numpy(first['embedding']),torch.from_numpy(second['embedding']))


def test_mapped_clip_restores_nonpersistent_position_ids_and_preserves_features():
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
    config = CLIPVisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=1,
        num_attention_heads=4,projection_dim=512)
    original = ClipAngleNet(encoder=CLIPVisionModelWithProjection(config)).eval()
    with torch.device('meta'):
        mapped = ClipAngleNet(encoder=CLIPVisionModelWithProjection(config))
    mapped.load_state_dict(original.state_dict(),assign=True)
    assert mapped.features.vision_model.embeddings.position_ids.is_meta
    restore_runtime_buffers(mapped)
    mapped.to('cpu').eval()
    assert torch.equal(original.features.vision_model.embeddings.position_ids,
                       mapped.features.vision_model.embeddings.position_ids)
    inputs = torch.rand(1,3,224,224)
    with torch.inference_mode():
        assert torch.allclose(original.embed(inputs),mapped.embed(inputs),atol=1e-7)


def test_checksum_checked_before_loading_checkpoint(artifact):
    (artifact/'weights.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        TrainedVision(Settings(_env_file=None, cv_backend='trained', trained_model_dir=artifact))


def test_changing_preprocessing_is_rejected(artifact):
    manifest = json.loads((artifact/'manifest.json').read_text())
    manifest['input_size'] = 320
    (artifact/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='preprocessing'):
        TrainedVision(Settings(_env_file=None, cv_backend='trained', trained_model_dir=artifact))


def test_flipping_approval_boolean_does_not_enable_model():
    manifest = dict(calibration=dict(validated=True, taxonomy_hash='tax',
        classes={'front': {'min_score': .9, 'min_margin': .2}}), heldout_approval=True)
    with pytest.raises(ValueError, match='held-out approval'):
        load_policy(manifest, 'weights', 'tax')


@pytest.mark.parametrize('score', [-.1, 1.1, float('nan')])
def test_supervised_thresholds_have_valid_ranges(score):
    with pytest.raises(ValueError):
        SupervisedPolicy.model_validate(dict(classes={'front': {'min_score': score, 'min_margin': .1}}))


def test_bound_approval_requires_enough_heldout_evidence():
    policy = SupervisedPolicy.model_validate(dict(validated=True, taxonomy_hash='tax',
        classes={'front': {'min_score': .9, 'min_margin': .2}})).model_dump()
    approval = dict(model_sha256='weights', dataset_id='dataset', test_manifest_sha256='test-hash',
        policy_sha256=hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
        classes={'front': {'tp': 4, 'fp': 0, 'fn': 0, 'tn': 50}})
    manifest = dict(calibration=policy, heldout_approval=approval, dataset_id='dataset')
    with pytest.raises(ValueError, match='insufficient held-out'):
        load_policy(manifest, 'weights', 'tax')
    approval['classes']['front']['tp'] = 100
    assert load_policy(manifest, 'weights', 'tax')['validated']
    policy['classes']['front']['min_score'] = .1
    with pytest.raises(ValueError, match='changed after approval'):
        load_policy(manifest, 'weights', 'tax')


def test_group_split_keeps_all_views_of_source_together():
    rows = [dict(vehicle_id=f'g{i}', label=c) for i in range(20) for c in ('left_side', 'fender_front_left')]
    assignment = split_groups(rows, 42)
    assert len(assignment) == 20
    assert set(assignment.values()) == {'train', 'calibration', 'test'}
    assert assignment == split_groups(rows, 42)


def test_training_requires_visual_review_and_source():
    with pytest.raises(ValueError, match='visual review'):
        check_rows([dict(review_status='pending')], allow_partial=True)
    with pytest.raises(ValueError, match='license'):
        check_rows([dict(review_status='reviewed', license='unknown')], allow_partial=True)


@pytest.mark.parametrize('name,allowed', [('CC BY 2.0',True),('CC BY-SA 4.0',True),
    ('CC0',True),('Public domain',True),('CC BY-NC 4.0',False),('CC BY-ND 4.0',False),('GFDL',False)])
def test_license_allowlist(name, allowed):
    assert allowed_license(name) == allowed


def test_collector_stops_instead_of_bypassing_long_retry_after():
    from unittest.mock import Mock
    client = Mock()
    client.get.return_value.status_code = 429
    client.get.return_value.headers = {'Retry-After': '600'}
    with pytest.raises(RuntimeError, match='rate limited'):
        get(client, 'https://commons.wikimedia.org/w/api.php')
    client.get.assert_called_once()


def test_real_model_artifact_is_not_production_approved():
    path = Path('models/view-classifier-v2/manifest.json')
    if not path.exists():
        pytest.skip('locally trained artifact not distributed with source')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    assert manifest['labels'] == list(LABELS)
    assert manifest['calibration']['validated'] is False
    assert manifest['heldout_approval'] is False


def test_deferred_test_does_not_open_test_images(tmp_path):
    from scripts.train_views import extract_splits
    from PIL import Image

    # A missing test file must never be opened during a calibration-only run.
    for split in ('train', 'calibration'):
        Image.new('RGB', (300, 300)).save(tmp_path / f'{split}.jpg')
    splits = {s: [dict(path=f'{s}.jpg', label='front', split=s)]
              for s in ('train', 'calibration', 'test')}

    class Encoder:
        def prepare_image(self, image):
            return torch.zeros(3, 4, 4)

        def embed(self, images):
            return images.flatten(1)

    result = extract_splits(Encoder(), splits, tmp_path, False, 2, defer_test=True)
    assert set(result) == {'train', 'calibration'}
    with pytest.raises(FileNotFoundError):
        extract_splits(Encoder(), splits, tmp_path, False, 2, defer_test=False)


def test_appearance_features_weights_and_side_labels_stay_aligned(tmp_path):
    from scripts.train_views import extract
    import numpy as np
    rows = [dict(path='original.jpg',label='left_side',split='train',review_status='reviewed'),
            dict(path='weak.jpg',label='front',split='train',review_status='source_annotated'),
            dict(path='crop.jpg',label='rear',split='train',review_status='reviewed',derived=True)]
    for row in rows:
        Image.new('RGB',(300,300),(80,120,180)).save(tmp_path/row['path'])

    class Encoder:
        def prepare_image(self,image):
            return torch.tensor(np.asarray(image).copy(),dtype=torch.float32).permute(2,0,1)

        def embed(self,images):
            return images.mean(dim=(2,3))

    x,y = extract(Encoder(),rows,tmp_path,True,2,appearance_augment=True)
    expected = ['left_side','right_side']*3 + ['front','front','rear','rear']
    assert [LABELS[i] for i in y.tolist()] == expected
    assert len(x)==10
    assert not torch.equal(x[0],x[2]) and not torch.equal(x[0],x[4])
    weights = head_sample_weights(rows,True,.35,appearance_augment=True)
    assert weights.tolist()==pytest.approx([1.]*6+[.35,.35,1.,1.])
    assert effective_head_counts(rows,True,.35,allow_partial=True,appearance_augment=True)=={
        'left_side':3,'right_side':3,'front':2,'rear':2}
    for split in ('calibration','test'):
        heldout = [dict(rows[0],split=split)]
        original,oy = extract(Encoder(),heldout,tmp_path,True,2)
        augmented,ay = extract(Encoder(),heldout,tmp_path,True,2,appearance_augment=True)
        assert torch.equal(original,augmented) and torch.equal(oy,ay)
