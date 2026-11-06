import hashlib
from io import BytesIO
import json

from PIL import Image
import pytest

from app.review import Decision, ReviewWorkspace
from scripts.evaluate_reviews import freeze_reviews


def prepare(tmp_path):
    workspace = ReviewWorkspace(tmp_path/'live', [], registry=tmp_path/'registry.json')
    ids = []
    for index, color in enumerate(('red','green','blue')):
        data = BytesIO()
        Image.new('RGB',(320,240),color).save(data,format='JPEG')
        path = tmp_path/f'{index}.jpg'
        path.write_bytes(data.getvalue())
        ids.append(workspace.add('open-source',str(index),path,dict(evaluation_cohort='blind_v1',
            vehicle_group=f'source-group-{index}', fixed_split='test', author='Author', license='CC BY 2.0',
            source_url='https://example.org/photo')))
    for key, status in zip(ids, ('reviewed','excluded')):
        workspace.save(key, Decision(status=status,label='front' if status=='reviewed' else None,
            reviewer='Human',vehicle_group='source-group-'+str(ids.index(key)), split='test',version=0))
    return workspace, ids


def test_diagnostic_preserves_unverified_answers_and_never_claims_gold(tmp_path):
    workspace, ids = prepare(tmp_path)
    output = tmp_path/'snapshot'
    snapshot = freeze_reviews(workspace.db_path,output)
    assert snapshot['status_counts'] == dict(reviewed=1,excluded=1,pending=1)
    assert snapshot['verified_count'] == 0
    assert snapshot['kind'] == 'human_photo_diagnostic'
    assert snapshot['production_approved'] is False
    assert snapshot['pending_ids'] == [ids[2]]
    assert snapshot['manifest_sha256'] == hashlib.sha256((output/'manifest.jsonl').read_bytes()).hexdigest()
    frozen = json.loads((output/'manifest.jsonl').read_text(encoding='utf-8'))
    assert frozen['label'] == 'front' and not frozen['vehicle_identity_verified']
    assert len(list((output/'images').iterdir())) == 1
    workspace.save(ids[0],Decision(status='reviewed',label='rear',reviewer='Human',
        vehicle_group='source-group-0',split='test',version=1))
    assert json.loads((output/'manifest.jsonl').read_text(encoding='utf-8'))['label'] == 'front'
    with pytest.raises(FileExistsError):
        freeze_reviews(workspace.db_path,output)


def test_diagnostic_rejects_changed_photo_without_rewriting_live_answers(tmp_path):
    workspace, _ = prepare(tmp_path)
    before = workspace.stats(cohort='blind_v1')
    (tmp_path/'0.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError,match='photo changed'):
        freeze_reviews(workspace.db_path,tmp_path/'bad')
    assert workspace.stats(cohort='blind_v1') == before
    assert not (tmp_path/'bad'/'snapshot.json').exists()


def test_assistant_adjudication_never_becomes_human_release_evidence(tmp_path):
    workspace, ids = prepare(tmp_path)
    workspace.save(ids[0],Decision(status='reviewed',label='other',reviewer='Assistant',reviewer_type='assistant',
        vehicle_group='source-group-0',split='test',version=1))
    snapshot = freeze_reviews(workspace.db_path,tmp_path/'assistant-snapshot')
    assert snapshot['kind'] == 'assistant_adjudicated_photo_diagnostic'
    assert snapshot['reviewer_counts'] == {'assistant':1}
    assert snapshot['production_approved'] is False
    row = json.loads((tmp_path/'assistant-snapshot'/'manifest.jsonl').read_text(encoding='utf-8'))
    assert row['reviewer_type'] == 'assistant'
    with pytest.raises(ValueError,match='no eligible'):
        workspace.export(holdout=True)
