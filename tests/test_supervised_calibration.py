import hashlib
import json

from PIL import Image
import pytest

from app.supervised_calibration import fit_gates, group_evidence, load_fresh_holdout
from app.evaluation import load_manifest
from app.events import REQUIRED


def model_descriptor(manifest):
    rows = load_manifest(manifest,(*REQUIRED,'other'),REQUIRED)
    return dict(dataset=str(manifest),dataset_id=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())


def predictions(positive_count=73):
    rows = [dict(vehicle_id=f'positive-{i}', label='front', candidate='front', score=.99,
                 margin=.9, split='calibration') for i in range(positive_count)]
    rows += [dict(vehicle_id=f'negative-{i}', label='other', candidate='other', score=.99,
                  margin=.9, split='calibration') for i in range(100)]
    return rows


def test_one_wrong_frame_is_not_hidden_by_correct_photo_of_same_car():
    rows = [dict(vehicle_id='same-car', label=label, candidate='front', score=.99, margin=.9)
            for label in ('front','other')]
    evidence = group_evidence(rows, 'front', dict(min_score=.5, min_margin=.1))
    assert evidence['tp'] == 0
    assert evidence['fp'] == 1


def test_complete_gallery_provides_both_positive_and_negative_frame_support():
    rows = [dict(vehicle_id='same-car',label='front',candidate='front',score=.99,margin=.9),
            dict(vehicle_id='same-car',label='rear',candidate='rear',score=.99,margin=.9)]
    evidence = group_evidence(rows,'front',dict(min_score=.5,min_margin=.1))
    assert evidence['positive_groups'] == evidence['negative_groups'] == 1
    assert evidence['tp'] == evidence['tn'] == 1


def test_quality_rejected_frames_do_not_inflate_gate_precision():
    rows = [dict(vehicle_id='bad',label='front',candidate='front',score=.99,margin=.9,quality_eligible=False),
            dict(vehicle_id='good',label='other',candidate='front',score=.99,margin=.9,quality_eligible=True)]
    evidence = group_evidence(rows,'front',dict(min_score=.5,min_margin=.1))
    assert evidence['tp'] == 0
    assert evidence['fp'] == 1


def test_small_perfect_sample_does_not_enable_a_gate():
    gates, evidence = fit_gates(predictions(30))
    assert not gates
    assert evidence['front']['reason'] == 'no_gate_meets_precision_bound'


def test_unusable_negative_photos_do_not_qualify_a_gate():
    rows = predictions()
    for row in rows:
        row['quality_eligible'] = row['label'] == 'front'
    gates,evidence = fit_gates(rows)
    assert not gates
    assert evidence['front']['negative_groups'] == 0


def test_duplicate_photos_do_not_inflate_independent_support():
    rows = predictions(1)
    rows += [dict(rows[0]) for _ in range(100)]
    gates, evidence = fit_gates(rows)
    assert not gates
    assert evidence['front']['positive_groups'] == 1


def test_sufficient_perfect_group_evidence_can_propose_a_gate():
    gates, evidence = fit_gates(predictions())
    assert set(gates) == {'front'}
    assert evidence['front']['precision_lower_95'] >= .95


def test_test_data_cannot_choose_its_own_threshold():
    rows = predictions()
    rows[0]['split'] = 'test'
    with pytest.raises(ValueError, match='calibration only'):
        fit_gates(rows)


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -.1, 1.1])
def test_invalid_scores_are_rejected(score):
    rows = predictions()
    rows[0]['score'] = score
    with pytest.raises(ValueError, match='invalid supervised score'):
        fit_gates(rows)


def write_dataset(root, color, split, vehicle='car', **extra):
    root.mkdir()
    image = Image.new('RGB', (200,200), color)
    image.save(root/'image.jpg')
    image.close()
    row = dict(path='image.jpg', label='front', split=split, vehicle_id=vehicle, listing_id=vehicle,
        covers=['front'], review_status='reviewed', reviewer_type='human', vehicle_identity_verified=True,
        sha256=hashlib.sha256((root/'image.jpg').read_bytes()).hexdigest(), derived=False, **extra)
    manifest = root/'manifest.jsonl'
    manifest.write_text(json.dumps(row)+'\n', encoding='utf-8')
    (root/'snapshot.json').write_text(json.dumps(dict(kind='fresh_human_holdout', count=1,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())))
    return manifest


def test_independent_human_frozen_holdout_is_accepted(tmp_path):
    old = write_dataset(tmp_path/'old', 'red', 'train', 'old-car')
    new = write_dataset(tmp_path/'new', 'blue', 'test', 'new-car')
    rows, digest = load_fresh_holdout(new, model_descriptor(old))
    assert len(rows) == 1
    assert digest == hashlib.sha256(new.read_bytes()).hexdigest()


def test_renamed_training_image_cannot_become_blind_holdout(tmp_path):
    old = write_dataset(tmp_path/'old', 'red', 'train', 'old-car')
    new = write_dataset(tmp_path/'new', 'red', 'test', 'new-car')
    with pytest.raises(ValueError, match='overlaps'):
        load_fresh_holdout(new, model_descriptor(old))


def test_renaming_source_group_does_not_hide_training_crop_parent(tmp_path):
    old = write_dataset(tmp_path/'old', 'red', 'train', 'source-car')
    new = write_dataset(tmp_path/'new', 'blue', 'test', 'human-car', origin_vehicle_group='source-car')
    with pytest.raises(ValueError, match='overlaps'):
        load_fresh_holdout(new, model_descriptor(old))


def test_snapshot_or_image_changes_are_detected(tmp_path):
    old = write_dataset(tmp_path/'old', 'red', 'train', 'old-car')
    new = write_dataset(tmp_path/'new', 'blue', 'test', 'new-car')
    with (new.parent/'image.jpg').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='matching checksums'):
        load_fresh_holdout(new, model_descriptor(old))
    new.write_text(new.read_text()+'\n')
    with pytest.raises(ValueError, match='matching frozen'):
        load_fresh_holdout(new, model_descriptor(old))


def test_mutated_parent_dataset_cannot_certify_a_release(tmp_path):
    old = write_dataset(tmp_path/'old','red','train','old-car')
    new = write_dataset(tmp_path/'new','blue','test','new-car')
    descriptor = model_descriptor(old)
    with (old.parent/'image.jpg').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError,match='changed after training'):
        load_fresh_holdout(new,descriptor)


def test_recompressed_training_photo_is_not_an_independent_holdout(tmp_path):
    from PIL import ImageDraw
    old = write_dataset(tmp_path/'old','red','train','old-car')
    new = write_dataset(tmp_path/'new','blue','test','new-car')
    with Image.open(old.parent/'image.jpg') as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((30,40,130,160),fill='black')
        image.save(old.parent/'image.jpg',quality=95)
        image.save(new.parent/'image.jpg',quality=50)
    row = json.loads(new.read_text())
    row['sha256'] = hashlib.sha256((new.parent/'image.jpg').read_bytes()).hexdigest()
    new.write_text(json.dumps(row)+'\n')
    (new.parent/'snapshot.json').write_text(json.dumps(dict(kind='fresh_human_holdout',count=1,
        manifest_sha256=hashlib.sha256(new.read_bytes()).hexdigest())))
    with pytest.raises(ValueError,match='near duplicate'):
        load_fresh_holdout(new,model_descriptor(old))
