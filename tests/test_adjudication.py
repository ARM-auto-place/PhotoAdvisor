import json
from pathlib import Path

from PIL import Image
import pytest

from app.review import Decision,ReviewWorkspace
from scripts.apply_review_export import apply_reviews


def fixture_export(tmp_path):
    source=tmp_path/'source'
    source.mkdir()
    image=Image.new('RGB',(320,240),'red')
    image.save(source/'one.jpg')
    image.close()
    item=dict(image_id='one',path='one.jpg',vehicle_group='original-car',
        author='A',source_url='https://example.org/image',license='CC0',license_url='https://creativecommons.org/publicdomain/zero/1.0/')
    (source/'candidates.json').write_text(json.dumps([item]))
    (source/'reviewed-labels.json').write_text(json.dumps([dict(image_id='one',status='reviewed',label='front',reviewer='assistant')]))
    previous=tmp_path/'previous.jsonl'
    previous.write_text(json.dumps(dict(path='source/one.jpg',label='front',split='train',
        vehicle_id='original-car',listing_id='original-car',covers=['front']))+'\n')
    workspace=ReviewWorkspace(tmp_path/'state',[source],previous,registry=tmp_path/'reservation.json')
    row=workspace.next()
    workspace.save(row['id'],Decision(status='reviewed',label='rear',reviewer='Alice',
        vehicle_group='verified-car',vehicle_verified=True,split='train',version=0))
    export=Path(workspace.export()['directory'])/'manifest.jsonl'
    return source,previous,export


def test_human_correction_updates_source_with_audit_without_rewriting_benchmark(tmp_path):
    source,previous,export=fixture_export(tmp_path)
    result=apply_reviews(export,source,previous,tmp_path/'audit')
    assert result['updated']==1
    new=json.loads((source/'reviewed-labels.json').read_text())[0]
    assert new['label']=='rear'
    assert new['reviewer_type']=='human'
    assert new['fixed_split']=='train'
    assert json.loads(previous.read_text())['label']=='front'
    assert json.loads(Path(result['audit']).read_text())['changes'][0]['before']['label']=='front'


def test_blind_snapshot_cannot_be_applied_as_training_corrections(tmp_path):
    source,previous,export=fixture_export(tmp_path)
    meta=json.loads(export.with_name('snapshot.json').read_text())
    meta['kind']='fresh_human_holdout'
    export.with_name('snapshot.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='not a blind holdout'):
        apply_reviews(export,source,previous,tmp_path/'audit')
