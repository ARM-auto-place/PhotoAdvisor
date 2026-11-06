from io import BytesIO
import json
from pathlib import Path
import zipfile

from fastapi import HTTPException
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from app.pilot import NewSession, PilotWorkspace, create_pilot_app
from app.settings import Settings
from scripts.train_views import check_rows
from test_advisor import picture


class Vision:
    checklist = json.loads(Path('configs/checklist-kafka.json').read_text(encoding='utf-8'))
    version = 'pilot-contract-double-not-an-accuracy-test'
    calibration = {'validated': False}

    def predict(self, images):
        return [dict(category='unknown', candidate='front', recognition='uncertain',
                     score=.8, score_kind='softmax', quality_issues=[],
                     embedding=np.array([1., 0.], dtype=np.float32)) for _ in images]


@pytest.fixture
def pilot(tmp_path):
    settings = Settings(_env_file=None, max_photos=3, max_image_bytes=1000000)
    workspace = PilotWorkspace(tmp_path/'pilot', settings)
    app = create_pilot_app(workspace, Vision())
    with TestClient(app) as client:
        client.headers['x-pilot-token'] = workspace.token
        yield client, workspace, app


def create(client):
    response = client.post('/api/sessions', json={'title': 'Toyota RAV4'})
    assert response.status_code == 200
    return response.json()['id']


def upload(client, sid, data=None):
    response = client.post(f'/api/sessions/{sid}/photos', content=picture() if data is None else data,
                           headers={'x-filename': '..%2Fprivate%2Fcar.jpg'})
    assert response.status_code == 200, response.text
    return response.json()['photo_id']


def analyze(client, sid):
    response = client.post(f'/api/sessions/{sid}/analyze')
    assert response.status_code == 200, response.text
    return response.json()


def correct(client, sid, pid, state, label='front'):
    return client.post(f'/api/sessions/{sid}/photos/{pid}/decision',
                       json={'label': label, 'revision': state['revision']})


def test_real_upload_manual_confirmation_and_stale_updates(pilot):
    client, workspace, _ = pilot
    sid = create(client)
    pid = upload(client, sid)
    state = analyze(client, sid)
    assert not state['report']['required']['present']
    assert state['report']['required']['uncertain'] == ['front']
    response = correct(client, sid, pid, state)
    assert response.status_code == 200
    confirmed = response.json()
    assert confirmed['report']['required']['present'] == ['front']
    result = confirmed['report']['photo_results'][0]
    assert result['source'] == 'manual' and result['model_candidate'] == 'front'
    assert result['score'] is None and result['confidence'] is None
    assert correct(client, sid, pid, state, 'rear').status_code == 409
    with workspace.connect() as db:
        assert db.execute('SELECT count(*) FROM history').fetchone()[0] == 1
        # Original model evidence is immutable when a human changes the final view.
        report = json.loads(db.execute('SELECT report FROM sessions').fetchone()[0])
        assert not report['photo_results'][0]['accepted']
    assert len(list(workspace.images.iterdir())) == 1
    assert (workspace.images/f'{pid}.jpg').exists()


def test_feedback_script_is_available_only_with_local_page_protections(pilot):
    client, _, _ = pilot
    response = client.get('/pilot-feedback.js')
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('application/javascript')
    assert response.headers['cache-control'] == 'no-store'
    assert response.headers['x-content-type-options'] == 'nosniff'
    assert "script-src 'self' 'unsafe-inline'" in response.headers['content-security-policy']
    assert client.get('/pilot-feedback.js', headers={'sec-fetch-site':'cross-site'}).status_code == 403
    assert client.get('/pilot-feedback.js', headers={'host':'evil.example'}).status_code == 400


def test_duplicates_and_poor_quality_cannot_close_more_angles(pilot):
    client, _, _ = pilot
    sid = create(client)
    ids = [upload(client, sid), upload(client, sid)]
    state = analyze(client, sid)
    state = correct(client, sid, ids[0], state).json()
    state = correct(client, sid, ids[1], state, 'rear').json()
    assert state['report']['required']['present'] == ['front']
    assert not state['report']['photo_results'][1]['accepted']
    second = create(client)
    pid = upload(client, second, picture(size=(100, 100)))
    state = analyze(client, second)
    state = correct(client, second, pid, state).json()
    assert not state['report']['required']['present']
    assert state['report']['status'] == 'needs_review'


def test_consent_export_keeps_model_assisted_feedback_out_of_training(pilot):
    client, _, _ = pilot
    sid = create(client)
    pid = upload(client, sid)
    state = correct(client, sid, pid, analyze(client, sid)).json()
    route = f'/api/sessions/{sid}/export'
    assert client.get(route).status_code == 403
    state = client.post(f'/api/sessions/{sid}/permission', json={
        'training_permission': True, 'revision': state['revision']}).json()
    response = client.get(route)
    assert response.status_code == 200
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        row = json.loads(archive.read('feedback.jsonl'))
        assert row['training_permission'] is True
        assert row['model_assisted'] is True
        assert row['reviewer_type'] == 'unverified'
        assert row['eligible_for_independent_test'] is False
        assert row['original_prediction']['recognition'] == 'uncertain'
        assert row['path'] in archive.namelist()
        with pytest.raises(ValueError, match='visual review'):
            check_rows([row], allow_partial=True)
    client.post(f'/api/sessions/{sid}/permission', json={
        'training_permission': False, 'revision': state['revision']})
    assert client.get(route).status_code == 403


def test_upload_strips_location_and_delete_removes_history_and_files(pilot):
    client, workspace, _ = pilot
    sid = create(client)
    data = BytesIO()
    image = Image.new('RGB', (400, 300))
    exif = Image.Exif()
    exif[270] = 'private camera metadata'
    image.save(data, format='JPEG', exif=exif)
    pid = upload(client, sid, data.getvalue())
    with Image.open(BytesIO(client.get(f'/media/{sid}/{pid}').content)) as normalized:
        assert not normalized.getexif()
    state = analyze(client, sid)
    correct(client, sid, pid, state)
    other = create(client)
    assert client.get(f'/media/{other}/{pid}').status_code == 404
    assert client.delete(f'/api/sessions/{other}/photos/{pid}').status_code == 404
    assert client.delete(f'/api/sessions/{sid}').status_code == 200
    assert client.get(f'/api/sessions/{sid}').status_code == 404
    assert not (workspace.images/f'{pid}.jpg').exists()
    with workspace.connect() as db:
        assert db.execute('SELECT count(*) FROM history').fetchone()[0] == 0


def test_confirming_later_duplicate_can_replace_unconfirmed_first_copy(pilot):
    client, _, _ = pilot
    sid = create(client)
    first, second = upload(client, sid), upload(client, sid)
    state = correct(client, sid, second, analyze(client, sid)).json()
    assert state['report']['required']['present'] == ['front']
    results = {p['photo_id']: p for p in state['report']['photo_results']}
    assert results[second]['accepted'] and results[second]['duplicate_of'] is None
    assert results[first]['duplicate_of'] == second


def test_mutation_invalidates_analysis_and_old_snapshot_cannot_overwrite(pilot):
    client, workspace, _ = pilot
    sid = create(client)
    pid = upload(client, sid)
    state = correct(client, sid, pid, analyze(client, sid)).json()
    upload(client, sid, picture(seed=2))
    current = client.get(f'/api/sessions/{sid}').json()
    assert current['report'] is None
    assert current['photos'][0]['decision'] == 'front'
    with pytest.raises(HTTPException) as exc:
        workspace.save_report(sid, state['revision'], state['report'])
    assert exc.value.status_code == 409
    assert correct(client, sid, pid, current).status_code == 409
    assert analyze(client, sid)['report']['required']['present'] == ['front']


def test_request_limits_bad_images_invalid_labels_and_origin(pilot):
    client, _, app = pilot
    sid = create(client)
    route = f'/api/sessions/{sid}/photos'
    assert client.post(route, content=b'broken').status_code == 422
    assert client.post(route, content=b'x'*1000001).status_code == 413
    for seed in range(3):
        upload(client, sid, picture(seed=seed))
    assert client.post(route, content=picture()).status_code == 413
    state = analyze(client, sid)
    assert correct(client, sid, state['photos'][0]['id'], state, 'invented').status_code == 422
    app.state.advisor.lock.acquire()
    try:
        assert client.post(f'/api/sessions/{sid}/analyze').status_code == 409
    finally:
        app.state.advisor.lock.release()
    assert client.get('/', headers={'sec-fetch-site': 'cross-site'}).status_code == 403
    assert client.get('/', headers={'host': 'evil.example'}).status_code == 400
    assert client.post('/api/sessions', json={'title': 'bad'}, headers={'x-pilot-token': 'wrong'}).status_code == 403


def test_workspace_persists_and_releases_ownership(tmp_path):
    root = tmp_path/'pilot'
    settings = Settings(_env_file=None)
    first = PilotWorkspace(root, settings)
    sid = first.create(NewSession(title='Saved'))
    with pytest.raises(RuntimeError, match='already owned'):
        PilotWorkspace(root, settings)
    with TestClient(create_pilot_app(first, Vision())):
        pass
    second = PilotWorkspace(root, settings)
    try:
        assert second.snapshot(sid)['title'] == 'Saved'
    finally:
        second.close()


def test_failure_to_load_model_releases_workspace(tmp_path, monkeypatch):
    root = tmp_path/'pilot'
    settings = Settings(_env_file=None)
    first = PilotWorkspace(root, settings)
    def fail(_):
        raise ValueError('missing checkpoint')
    monkeypatch.setattr('app.pilot.load_vision', fail)
    with pytest.raises(ValueError, match='checkpoint'):
        with TestClient(create_pilot_app(first)):
            pass
    second = PilotWorkspace(root, settings)
    second.close()


def test_disk_quota_does_not_leave_a_partial_photo(tmp_path):
    workspace = PilotWorkspace(tmp_path/'pilot', Settings(_env_file=None), max_bytes=1)
    try:
        sid = workspace.create(NewSession(title='Quota'))
        with pytest.raises(HTTPException) as exc:
            workspace.add(sid, picture(), 'test.jpg')
        assert exc.value.status_code == 413
        assert workspace.snapshot(sid)['photos'] == []
        assert list(workspace.images.iterdir()) == []
    finally:
        workspace.close()
