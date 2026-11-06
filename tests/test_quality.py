import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from app.evaluation import load_manifest, metrics, threshold_proposal, wilson_lower
from app.pipeline import Pipeline
from app.policy import POLICY_VERSION, validate_calibration
from app.schemas import EvaluateParams
from app.service import Advisor
from app.settings import Settings
from app.state import State
from test_pipeline import advisor, config, event


def test_second_owner_rejected_and_lock_released(tmp_path):
    path = tmp_path / 'state.sqlite3'
    first = State(path)
    with pytest.raises(RuntimeError, match='already_owned'):
        State(path)
    first.close()
    second = State(path)
    second.close()


def test_state_never_rewinds_even_if_caller_skips_needed(tmp_path):
    state = State(tmp_path / 'state.sqlite3')
    state.save(event(5), {'result_id': 'new'})
    state.save(event(2), {'result_id': 'stale'})
    state.rejected('record', {}, event(3), {'result_id': 'stale-error'})
    assert state.db.execute('SELECT revision FROM listings').fetchone()[0] == 5
    assert state.db.execute('SELECT count(*) FROM outbox WHERE kind="result"').fetchone()[0] == 1
    state.close()


def test_outbox_kinds_have_separate_identifiers(tmp_path):
    state = State(tmp_path / 'state.sqlite3')
    with state.db:
        state.enqueue('result', 'identical', 'a', {})
        state.enqueue('dlq', 'identical', 'a', {})
    assert state.db.execute('SELECT count(*) FROM outbox').fetchone()[0] == 2
    state.close()


def test_nonfinite_result_rolls_back_state(tmp_path):
    state = State(tmp_path / 'state.sqlite3')
    with pytest.raises(ValueError):
        state.save(event(), {'result_id': 'new', 'confidence': float('nan')})
    assert state.needed(event())
    assert not state.pending()
    state.close()


@pytest.mark.parametrize('revision', [True, '1', 1.0])
def test_revision_is_strict_integer(revision):
    with pytest.raises(ValidationError):
        event(listing_revision=revision)


def test_quality_rejection_requires_review(tmp_path):
    cv = advisor()
    cv.evaluate.return_value['photo_results'][0].update(
        accepted=False, quality='poor', quality_issues=['blurred'])
    result = Pipeline(config(tmp_path), cv, None).analyze(event())
    assert result['status'] == 'needs_review'
    assert result['uncertain'] == ['p1']
    assert result['photos'][0]['quality_issues'] == ['blurred']
    assert result['recognized'] == 0


def test_corrupt_photo_cannot_be_manually_accepted(tmp_path):
    from test_pipeline import photo
    cv = advisor()
    cv.evaluate.return_value['photo_results'][0].update(
        error='invalid_image', accepted=False, recognition='error')
    result = Pipeline(config(tmp_path), cv, None).analyze(
        event(photos=[photo(manual_category='front')]))
    assert result['recognized'] == 0
    assert result['status'] == 'failed'


@pytest.mark.parametrize('malformed', ['count', 'nan', 'shape', 'missing', 'nonfinite_score'])
def test_model_contract_failure_closes_all_images(tmp_path, monkeypatch, malformed):
    checklist = json.loads(Path('configs/checklist-kafka.json').read_text(encoding='utf-8'))
    images = [Image.new('RGB', (300, 300)), Image.new('RGB', (300, 300))]
    decoded = iter(images)
    monkeypatch.setattr('app.service.decode', lambda *_: (next(decoded), (300, 300)))
    monkeypatch.setattr('app.service.quality', lambda *_: [])
    monkeypatch.setattr('app.service.fingerprint', lambda *_: None)
    vision = Mock(checklist=checklist, version='test', calibration={'validated': True})
    prediction = dict(category='front', candidate='front', recognition='known',
                      quality_issues=[], embedding=np.array([1., 0.]))
    outputs = [prediction.copy(), prediction.copy()]
    if malformed == 'count':
        outputs.pop()
    elif malformed == 'nan':
        outputs[0]['embedding'] = np.array([float('nan'), 0.])
    elif malformed == 'shape':
        outputs[0]['embedding'] = np.array([[1., 0.]])
    elif malformed == 'missing':
        del outputs[0]['embedding']
    else:
        outputs[0]['similarity'] = float('nan')
    vision.predict.return_value = outputs
    storage = Mock()
    storage.fetch.return_value = b'content'
    settings = config(tmp_path).model_copy(update={'download_workers': 1})
    service = Advisor(settings, storage, vision)
    try:
        result = service.evaluate(EvaluateParams(listing_id='a', photos=[
            dict(photo_id=str(i), object_key=f'listings/a/{i}') for i in range(2)]))
        assert result['status'] == 'failed'
        assert all(p['error'] == 'inference_failed' for p in result['photo_results'])
        for image in images:
            with pytest.raises(ValueError):
                image.getpixel((0, 0))
    finally:
        service.close()


def calibration(**kw):
    return dict(validated=True, dataset_id='dataset', model_version='model', taxonomy_hash='taxonomy',
                policy_version=POLICY_VERSION, unknown_similarity=.18,
                classes={'front': {'min_similarity': .3, 'min_margin': .05}}, **kw)


def test_validated_thresholds_require_matching_policy():
    raw = calibration()
    assert validate_calibration(raw, 'model', 'taxonomy', ['front'])['validated']
    del raw['policy_version']
    with pytest.raises(ValueError):
        validate_calibration(raw, 'model', 'taxonomy', ['front'])


@pytest.mark.parametrize('value', [-2, 2, float('nan')])
def test_invalid_threshold_is_not_loaded(value):
    raw = calibration()
    raw['classes']['front']['min_similarity'] = value
    with pytest.raises(ValueError):
        validate_calibration(raw, 'model', 'taxonomy', ['front'])


def test_mismatched_taxonomy_cannot_reuse_thresholds():
    with pytest.raises(ValueError):
        validate_calibration(calibration(), 'model', 'different-taxonomy', ['front'])


def test_metrics_do_not_invent_precision_without_predictions():
    report = metrics(['front'], ['unknown'], ['front', 'rear', 'unknown'])
    assert report['per_class'][0]['precision'] is None
    assert report['per_class'][0]['recall'] == 0
    assert report['per_class'][1]['recall'] is None
    assert report['accuracy'] == 0
    with pytest.raises(ValueError):
        metrics(['front'], [], ['front'])


def manifest(tmp_path, rows):
    path = tmp_path / 'manifest.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    return path


def labelled(path, **kw):
    return dict(path=path, label='front', split='test', vehicle_id='car', listing_id='listing',
                covers=['front'], **kw)


def test_manifest_detects_identical_content_under_different_names(tmp_path):
    (tmp_path / 'a.jpg').write_bytes(b'same actual file bytes')
    (tmp_path / 'b.jpg').write_bytes(b'same actual file bytes')
    rows = [labelled('a.jpg'), dict(labelled('b.jpg'), split='calibration',
                                   vehicle_id='other-car', listing_id='other-listing')]
    with pytest.raises(ValueError, match='duplicate image content'):
        load_manifest(manifest(tmp_path, rows), ['front'], ['front'])


def test_manifest_detects_vehicle_leakage(tmp_path):
    (tmp_path / 'a.jpg').write_bytes(b'file one')
    (tmp_path / 'b.jpg').write_bytes(b'file two')
    with pytest.raises(ValueError, match='leakage'):
        load_manifest(manifest(tmp_path, [labelled('a.jpg'),
            dict(labelled('b.jpg'), split='calibration')]), ['front'], ['front'])


def test_manifest_rejects_external_paths(tmp_path):
    with pytest.raises(ValueError, match='outside dataset'):
        load_manifest(manifest(tmp_path, [labelled('../secret.jpg')]), ['front'], ['front'])


def sample(i, correct=True):
    return dict(split='calibration', vehicle_id=str(i), covers=['front'] if correct else [],
                quality_eligible=True, prediction=dict(candidate='front', similarity=.4, margin=.1))


def test_tiny_perfect_sample_cannot_enable_class():
    thresholds, evidence = threshold_proposal([sample(i) for i in range(4)], ['front'])
    assert thresholds == {}
    assert evidence['front']['reason'] == 'insufficient_independent_evidence'


def test_repeated_vehicle_cannot_manufacture_calibration_support():
    thresholds, _ = threshold_proposal([sample(0) for _ in range(100)], ['front'])
    assert thresholds == {}


def test_thresholds_require_high_precision_on_independent_vehicles():
    thresholds, evidence = threshold_proposal([sample(i) for i in range(100)], ['front'])
    assert thresholds == {'front': {'min_similarity': .4, 'min_margin': .1}}
    assert evidence['front']['precision_lower_95'] >= .95
    assert wilson_lower(100, 100) < 1
    bad = [sample(i, correct=i % 2 == 0) for i in range(100)]
    assert threshold_proposal(bad, ['front'])[0] == {}


def test_test_split_cannot_be_used_for_threshold_selection():
    with pytest.raises(ValueError, match='calibration split'):
        threshold_proposal([dict(sample(1), split='test')], ['front'])


def test_existing_smoke_manifest_has_no_data_leakage():
    checklist = json.loads(Path('configs/checklist-kafka.json').read_text(encoding='utf-8'))
    manifest = Path('datasets/smoke/manifest.jsonl')
    if not manifest.exists():
        pytest.skip('local smoke dataset not distributed with source')
    rows = load_manifest(manifest, checklist['classes'],
                         {c['id'] for c in checklist['categories']})
    assert len(rows) == 4
    assert len({r['sha256'] for r in rows}) == 4


def test_blank_kafka_ca_uses_system_trust():
    assert Settings(_env_file=None, kafka_ssl_cafile='').kafka_ssl_cafile is None


@pytest.mark.parametrize('etag', ['*', '"*"', 'a,b', '"a", "b"', '""', 'a\nb', 'a b', '"broken'])
def test_etag_must_pin_one_object(etag):
    from test_pipeline import photo
    value = photo()
    value['etag'] = etag
    with pytest.raises(ValidationError):
        event(photos=[value])


def test_null_s3_version_requires_exact_etag():
    from test_pipeline import photo
    value = photo(version_id='null')
    value.pop('etag')
    with pytest.raises(ValidationError):
        event(photos=[value])


def test_startup_failure_releases_state_ownership(tmp_path, monkeypatch):
    import asyncio
    from app.consumer import KafkaWorker
    async def scenario():
        settings = config(tmp_path).model_copy(update={'kafka_security_protocol': 'SSL'})
        worker = KafkaWorker(settings, advisor())
        def fail(**_):
            raise ValueError('bad certificate')
        monkeypatch.setattr('app.consumer.ssl.create_default_context', fail)
        with pytest.raises(ValueError, match='bad certificate'):
            await worker.start()
        state = State(settings.state_path)
        state.close()
    asyncio.run(scenario())


def test_corrupt_database_initialization_releases_lock(tmp_path):
    from filelock import FileLock
    path = tmp_path / 'broken.sqlite3'
    path.write_bytes(b'not sqlite')
    with pytest.raises(Exception, match='not a database'):
        State(path)
    lock = FileLock(str(path.resolve()) + '.lock')
    with lock.acquire(timeout=0):
        pass
