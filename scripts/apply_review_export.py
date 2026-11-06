"""Apply audited human labels to mutable source annotations, never to archived datasets."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filelock import FileLock

from app.data_guard import is_reserved, load_reservations
from app.evaluation import load_manifest
from app.events import REQUIRED


def apply_reviews(review_manifest, source, previous, audit_root):
    review_manifest, source = Path(review_manifest), Path(source).resolve()
    snapshot_hash = hashlib.sha256(review_manifest.read_bytes()).hexdigest()
    snapshot = json.loads(review_manifest.with_name('snapshot.json').read_text(encoding='utf-8'))
    if snapshot.get('kind') != 'human_reviewed_export' or snapshot.get('manifest_sha256') != snapshot_hash:
        raise ValueError('requires a matching normal human-reviewed export, not a blind holdout')
    declared = [json.loads(line) for line in review_manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = load_manifest(review_manifest,(*REQUIRED,'other'),REQUIRED)
    if snapshot.get('count') != len(rows) or any(a.get('sha256') != b['sha256'] for a,b in zip(declared,rows)):
        raise ValueError('reviewed image or snapshot changed')
    old = load_manifest(previous,(*REQUIRED,'other'),REQUIRED)
    old_splits = {r['sha256']:r['split'] for r in old}
    old_groups = {r['vehicle_id']:r['split'] for r in old}
    reservation = load_reservations()
    candidates = json.loads((source/'candidates.json').read_text(encoding='utf-8'))
    by_id = {str(r.get('image_id',r.get('page_id'))):r for r in candidates}
    reviews_path = source/'reviewed-labels.json'
    changes = []
    with FileLock(str(reviews_path)+'.lock',timeout=5):
        original = json.loads(reviews_path.read_text(encoding='utf-8'))
        decisions = {str(r.get('image_id',r.get('page_id'))):r for r in original}
        for row in rows:
            if row.get('origin_source_dataset') != source.name:
                continue
            if row.get('reviewer_type') != 'human' or row.get('review_status') != 'reviewed':
                raise ValueError('only human reviewed exports may adjudicate source labels')
            if is_reserved(row,reservation):
                raise ValueError('reserved blind photos cannot update training/development annotations')
            source_id = str(row['origin_source_id'])
            item = by_id.get(source_id)
            if not item:
                raise ValueError('original candidate not found')
            path = (source/item['path']).resolve()
            if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('source image differs from reviewed snapshot')
            if row['sha256'] in old_splits and old_splits[row['sha256']] != row['split']:
                raise ValueError('human labels cannot move previously used photos between splits')
            if row.get('origin_vehicle_group') in old_groups and old_groups[row['origin_vehicle_group']] != row['split']:
                raise ValueError('human labels cannot move source groups between splits')
            updated = dict(image_id=source_id,status='reviewed',label=row['label'],
                reviewer=row['reviewer'],reviewer_type='human',label_origin='human_review',
                vehicle_group=row['vehicle_id'],fixed_split=row['split'],
                vehicle_identity_verified=row.get('vehicle_identity_verified',False),
                review_snapshot_sha256=snapshot_hash)
            changes.append(dict(source_id=source_id,before=decisions.get(source_id),after=updated))
            decisions[source_id] = updated
        if not changes:
            raise ValueError('no matching human-reviewed source images')
        audit = Path(audit_root)/f'adjudication-{secrets.token_hex(8)}.json'
        audit.parent.mkdir(parents=True,exist_ok=True)
        with audit.open('x',encoding='utf-8') as stream:
            json.dump(dict(source=str(source),review_snapshot_sha256=snapshot_hash,changes=changes),
                      stream,ensure_ascii=False,indent=2)
        temporary = reviews_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(list(decisions.values()),ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(reviews_path)
    return dict(updated=len(changes),audit=str(audit))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reviews',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--previous',type=Path,default=Path('datasets/view-training-v10/manifest.jsonl'))
    parser.add_argument('--audit-root',type=Path,default=Path('reports/adjudications'))
    args=parser.parse_args()
    print(json.dumps(apply_reviews(args.reviews,args.source,args.previous,args.audit_root)))
