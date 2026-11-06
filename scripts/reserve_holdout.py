"""Reserve unlabelled licensed sources; never generate gold labels or test scores."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filelock import FileLock

from app.data_guard import RESERVATIONS, load_reservations
from app.evaluation import load_manifest
from app.events import REQUIRED
from app.data_sources import author_group, allowed_license


def reserve(source, previous, output, registry=RESERVATIONS):
    source, output, registry = Path(source).resolve(), Path(output), Path(registry)
    if output.exists():
        raise ValueError('choose a new output; reservations are immutable')
    old = load_manifest(previous, (*REQUIRED,'other'), REQUIRED)
    groups, hashes = {r['vehicle_id'] for r in old}, {r['sha256'] for r in old}
    candidates = []
    for item in json.loads((source/'candidates.json').read_text(encoding='utf-8')):
        group = item.get('vehicle_group') or author_group(item)
        if group in groups or item.get('derived') or not allowed_license(item.get('license','')):
            continue
        path = (source/item['path']).resolve()
        if not path.is_relative_to(source):
            raise ValueError('source path outside root')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            continue
        clean = {k:v for k,v in item.items() if k not in {'label','label_hint','source_view_hint','score','candidate'}}
        candidates.append((dict(clean, image_id=digest, path=f'images/{digest}.jpg',
            origin_source_id=str(item.get('image_id',item.get('page_id'))),
            vehicle_group=group, fixed_split='test', evaluation_cohort='blind_v1',
            source_sha256=digest, review_status='pending'), path))
        hashes.add(digest)
    if not candidates:
        raise ValueError('no unused licensed original images')
    registry.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(registry)+'.lock'):
        existing = load_reservations(registry)
        merged = dict(schema_version=1,
            groups=sorted(set(existing.get('groups',[])) | {c['vehicle_group'] for c,_ in candidates}),
            sha256=sorted(set(existing.get('sha256',[])) | {c['source_sha256'] for c,_ in candidates}),
            reason='Reserved unlabelled photos: human review and independent qualification required.')
        temp = registry.with_suffix('.tmp')
        temp.write_text(json.dumps(merged,indent=2),encoding='utf-8')
        temp.replace(registry)
    (output/'images').mkdir(parents=True)
    for item,path in candidates:
        shutil.copyfile(path,output/item['path'])
    records = [r for r,_ in candidates]
    (output/'candidates.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'reviewed-labels.json').write_text(json.dumps([dict(image_id=r['image_id'],status='pending') for r in records],indent=2),encoding='utf-8')
    summary = dict(kind='reserved_unlabelled_holdout', images=len(records),
        source_groups=len({r['vehicle_group'] for r in records}), human_labels=0,
        previous_manifest_sha256=hashlib.sha256(Path(previous).read_bytes()).hexdigest(),
        production_approved=False, note='Source groups are not verified physical vehicle identities. No model predictions generated.')
    (output/'reservation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(reserve(args.source,args.previous,args.output)))
