from io import BytesIO
import json

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.review import Decision, ReviewWorkspace, create_review_app


def jpeg(color='red'):
    buffer = BytesIO()
    image = Image.new('RGB', (320,240), color)
    image.save(buffer, format='JPEG')
    image.close()
    return buffer.getvalue()


@pytest.fixture
def workspace(tmp_path):
    return ReviewWorkspace(tmp_path/'state', [], None, registry=tmp_path/'reservations.json')


def add_photo(workspace, tmp_path, name='one', color='red', **extra):
    path = tmp_path/(name+'.jpg')
    path.write_bytes(jpeg(color))
    item = dict(author='Photographer', source_url='https://example.org/photo',
                license='CC0', license_url='https://creativecommons.org/publicdomain/zero/1.0/', **extra)
    return workspace.add('test-source', name, path, item)


def decision(**extra):
    return Decision(**dict(dict(status='reviewed', label='front', reviewer='Alice',
        vehicle_group='car-1', vehicle_verified=True, split='test', version=0), **extra))


def test_blind_queue_does_not_disclose_proposals(workspace, tmp_path):
    add_photo(workspace, tmp_path, label_hint='front', source_view_hint='front-left', score=.99)
    row = workspace.next()
    assert not {'label_hint','source_view_hint','score','candidate'} & row.keys()
    assert row['decision'] is None


def test_revision_prevents_silent_overwrite(workspace, tmp_path):
    key = add_photo(workspace, tmp_path)
    assert workspace.save(key, decision()) == 1
    with pytest.raises(RuntimeError, match='annotation_changed'):
        workspace.save(key, decision(label='rear'))


def test_fender_recheck_filters_answers_and_continues_after_correction(workspace, tmp_path):
    keys = [add_photo(workspace,tmp_path,str(i),color) for i,color in enumerate(('red','green','blue'))]
    for key,label in zip(keys,('fender_front_left','front','fender_rear_right')):
        workspace.save(key,decision(label=label))
    assert workspace.stats(fenders_only=True)['reviewed'] == 2
    first = workspace.next(status='reviewed',fenders_only=True)
    assert first['decision']['label'].startswith('fender_')
    workspace.save(first['id'],decision(label='other',version=1))
    following = workspace.next(status='reviewed',after=first['id'],fenders_only=True)
    assert following and following['id'] != first['id']
    assert workspace.stats(fenders_only=True)['reviewed'] == 1
    assert workspace.next(status='pending',fenders_only=True) is None


def test_same_vehicle_cannot_leak_between_splits(workspace, tmp_path):
    first = add_photo(workspace, tmp_path)
    second = add_photo(workspace, tmp_path, 'two', 'blue')
    workspace.save(first, decision())
    with pytest.raises(ValueError, match='same vehicle'):
        workspace.save(second, decision(split='train'))


def test_legacy_group_cannot_be_renamed_into_test(workspace, tmp_path):
    key = add_photo(workspace, tmp_path, fixed_split='train')
    with pytest.raises(ValueError, match='cannot change split'):
        workspace.save(key, decision(vehicle_group='new-name', split='test'))


def test_frozen_holdout_requires_new_verified_originals(workspace, tmp_path):
    key = add_photo(workspace, tmp_path)
    workspace.save(key, decision(vehicle_verified=False))
    with pytest.raises(ValueError, match='no eligible'):
        workspace.export(holdout=True)
    workspace.save(key, decision(version=1, vehicle_verified=True))
    export = workspace.export(holdout=True)
    from pathlib import Path
    root = Path(export['directory'])
    row = json.loads((root/'manifest.jsonl').read_text(encoding='utf-8'))
    assert row['reviewer_type'] == 'human'
    assert row['vehicle_identity_verified'] is True
    assert export['kind'] == 'fresh_human_holdout'
    assert workspace.export(holdout=True)['directory'] != export['directory']
    with pytest.raises(ValueError, match='no eligible'):
        workspace.export()


@pytest.mark.parametrize('flag', ['derived', 'seen_before'])
def test_crops_and_previous_data_cannot_become_fresh_holdout(workspace, tmp_path, flag):
    key = add_photo(workspace, tmp_path, **({'derived': True} if flag=='derived' else {}))
    if flag == 'seen_before':
        with workspace.connect() as db:
            db.execute('UPDATE samples SET seen_before=1 WHERE id=?', (key,))
    workspace.save(key, decision())
    with pytest.raises(ValueError, match='no eligible'):
        workspace.export(holdout=True)


def test_changed_source_is_not_silently_frozen(workspace, tmp_path):
    key = add_photo(workspace, tmp_path)
    workspace.save(key, decision())
    (tmp_path/'one.jpg').write_bytes(jpeg('blue'))
    with pytest.raises(ValueError, match='source photo changed'):
        workspace.export(holdout=True)


def test_actual_export_passes_frozen_integrity_validation_on_windows(workspace, tmp_path):
    import hashlib
    from pathlib import Path
    from app.evaluation import load_manifest
    from app.events import REQUIRED
    from app.supervised_calibration import load_fresh_holdout
    key=add_photo(workspace,tmp_path)
    workspace.save(key,decision())
    exported=Path(workspace.export(holdout=True)['directory'])/'manifest.jsonl'
    (tmp_path/'old.jpg').write_bytes(jpeg('blue'))
    previous=tmp_path/'old.jsonl'
    previous.write_text(json.dumps(dict(path='old.jpg',label='front',split='train',vehicle_id='old-car',
        listing_id='old-car',covers=['front']))+'\n')
    old_rows=load_manifest(previous,(*REQUIRED,'other'),REQUIRED)
    descriptor=dict(dataset=str(previous),dataset_id=hashlib.sha256(json.dumps(old_rows,sort_keys=True).encode()).hexdigest())
    rows,_=load_fresh_holdout(exported,descriptor)
    assert len(rows)==1


def test_assistant_review_cannot_be_exported_as_human_gold(workspace, tmp_path):
    key=add_photo(workspace,tmp_path)
    workspace.save(key,decision(reviewer='Assistant',reviewer_type='assistant'))
    with pytest.raises(ValueError,match='no eligible'):
        workspace.export(holdout=True)


def test_local_origin_and_csrf_guards(workspace):
    with TestClient(create_review_app(workspace)) as client:
        assert client.post('/api/export').status_code == 403
        assert client.get('/api/stats', headers={'sec-fetch-site':'cross-site'}).status_code == 403
        assert client.get('/api/stats', headers={'host':'attacker.example'}).status_code == 400
        assert client.get('/').status_code == 200


def test_upload_requires_permission_and_is_not_a_label(workspace):
    with TestClient(create_review_app(workspace)) as client:
        headers = {'x-review-token':workspace.token}
        assert client.post('/api/upload?vehicle_id=rav4&author=Alice', content=jpeg(), headers=headers).status_code == 422
        result = client.post('/api/upload?vehicle_id=rav4&author=Alice&training_permission=true',
                             content=jpeg(), headers=headers)
        assert result.status_code == 200
        assert workspace.stats()['pending'] == 1
        assert workspace.stats()['reviewed'] == 0
        row = workspace.next()
        assert row['vehicle_group'] == 'own-rav4'
        assert client.get(row['image']).headers['content-type'] == 'image/jpeg'


def test_image_traversal_and_non_images_are_rejected(workspace, tmp_path):
    outside = tmp_path/'outside.jpg'
    outside.write_bytes(jpeg())
    source = tmp_path/'source'
    source.mkdir()
    item = dict(image_id='evil', path='../outside.jpg', author='A', source_url='https://example.org', license='CC0')
    (source/'candidates.json').write_text(json.dumps([item]))
    with pytest.raises(ValueError, match='outside source'):
        workspace.import_source(source)
    with TestClient(create_review_app(workspace)) as client:
        r = client.post('/api/upload?vehicle_id=car&author=Alice&training_permission=true',
                        content=b'<svg onload="alert(1)"></svg>', headers={'x-review-token':workspace.token})
        assert r.status_code == 422


def test_queue_skip_does_not_create_a_fake_annotation(workspace, tmp_path):
    first = add_photo(workspace, tmp_path)
    second = add_photo(workspace, tmp_path, 'two', 'blue')
    # Windows clocks can give several imports the same timestamp.
    with workspace.connect() as db:
        db.execute("UPDATE samples SET created='2026-11-03T00:00:00+00:00'")
    assert workspace.next()['id'] == first
    assert workspace.next(after=first)['id'] == second
    assert workspace.stats()['reviewed'] == 0


def test_reopening_workspace_updates_previously_seen_images(workspace, tmp_path):
    add_photo(workspace, tmp_path)
    assert workspace.next(fresh=True) is not None
    previous = tmp_path/'previous.jsonl'
    previous.write_text(json.dumps(dict(path='one.jpg', label='front', covers=['front'],
        vehicle_id='trained-car', listing_id='trained-car', split='train'))+'\n')
    reopened = ReviewWorkspace(workspace.root, [], previous)
    assert reopened.next(fresh=True) is None


def test_consented_own_export_preserves_training_provenance(workspace, tmp_path, monkeypatch):
    from pathlib import Path
    from scripts.build_view_dataset import main
    with TestClient(create_review_app(workspace)) as client:
        response = client.post('/api/upload?vehicle_id=rav4&author=Alice&training_permission=true',
            content=jpeg(), headers={'x-review-token':workspace.token})
    workspace.save(response.json()['id'], decision(vehicle_group='own-rav4', split='train'))
    export = workspace.export()
    output = tmp_path/'assembled'
    monkeypatch.setattr('sys.argv', ['build_view_dataset', '--source', export['directory'], '--output', str(output)])
    main()
    row = json.loads((Path(output)/'manifest.jsonl').read_text(encoding='utf-8'))
    assert row['training_permission'] is True
    assert row['source_kind'] == 'own'
    assert row['reviewer_type'] == 'human'
    assert row['origin_vehicle_group'] == 'own-rav4'
