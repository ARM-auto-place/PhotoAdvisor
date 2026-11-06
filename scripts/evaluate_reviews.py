"""Freeze user answers and evaluate one preselected model without training or gate fitting."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation import load_manifest, metrics
from app.events import REQUIRED

LABELS = (*REQUIRED, 'other')


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def freeze_reviews(database, output, cohort='blind_v1'):
    database, output = Path(database).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # SQLite backup includes committed WAL data and never changes the live answers.
    with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True) as source:
        with sqlite3.connect(output/'reviews.sqlite3') as backup:
            source.backup(backup)
    with sqlite3.connect(output/'reviews.sqlite3') as db:
        db.row_factory = sqlite3.Row
        samples = db.execute('SELECT s.*,r.decision,r.version,r.updated FROM samples s '
                             'LEFT JOIN reviews r USING(id) ORDER BY s.id').fetchall()
    selected = [r for r in samples if json.loads(r['record']).get('evaluation_cohort') == cohort]
    if not selected:
        raise ValueError('no photos in requested cohort')
    (output/'images').mkdir()
    answers, manifest = [], []
    for sample in selected:
        record = json.loads(sample['record'])
        decision = json.loads(sample['decision']) if sample['decision'] else None
        answers.append(dict(id=sample['id'], source=sample['source'], source_id=sample['source_id'],
            sha256=sample['sha256'], decision=decision, version=sample['version'], updated=sample['updated']))
        if not decision or decision['status'] != 'reviewed':
            continue
        if decision['reviewer_type'] not in ('human','assistant') or decision['split'] != 'test' or record.get('derived'):
            raise ValueError('diagnostic evaluation requires reviewed test originals')
        if decision['label'] not in LABELS:
            raise ValueError('invalid user label')
        relative = f"images/{sample['id']}.jpg"
        shutil.copyfile(sample['path'], output/relative)
        digest = hashlib.sha256((output/relative).read_bytes()).hexdigest()
        if digest != sample['sha256']:
            raise ValueError('photo changed after import')
        manifest.append(dict(path=relative, label=decision['label'], split='test',
            vehicle_id=decision['vehicle_group'], listing_id=decision['vehicle_group'],
            covers=[] if decision['label']=='other' else [decision['label']], sha256=digest,
            reviewer=decision['reviewer'], reviewer_type=decision['reviewer_type'], review_status='reviewed',
            vehicle_identity_verified=decision['vehicle_verified'], derived=False,
            origin_vehicle_group=sample['vehicle_group'], source_dataset=sample['source'],
            source_id=sample['source_id'], evaluation_cohort=cohort,
            **{k:record.get(k) for k in ('author','license','license_url','source_url')}))
    if not manifest:
        raise ValueError('no labels to evaluate')
    statuses = Counter(a['decision']['status'] if a['decision'] else 'pending' for a in answers)
    with (output/'manifest.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for row in manifest:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    write_json(output/'answers.json', answers)
    human = all(r['reviewer_type']=='human' for r in manifest)
    snapshot = dict(kind='human_photo_diagnostic' if human else 'assistant_adjudicated_photo_diagnostic',
        reviewer_counts=dict(Counter(r['reviewer_type'] for r in manifest)), cohort=cohort, total=len(selected),
        status_counts=dict(statuses), count=len(manifest),
        verified_count=sum(r['vehicle_identity_verified'] for r in manifest),
        source_groups=len({r['origin_vehicle_group'] for r in manifest}),
        label_counts=dict(Counter(r['label'] for r in manifest)),
        pending_ids=[a['id'] for a in answers if not a['decision']],
        manifest_sha256=hashlib.sha256((output/'manifest.jsonl').read_bytes()).hexdigest(),
        answers_sha256=hashlib.sha256((output/'answers.json').read_bytes()).hexdigest(),
        production_approved=False,
        note='Reviewer provenance is explicit; physical vehicle identity may be unverified. '
             'This is not a release holdout and must never be used for training or threshold fitting.')
    write_json(output/'snapshot.json', snapshot)
    return snapshot


def evaluate(output, model):
    from app.imaging import decode, quality
    from app.settings import Settings
    from app.trained import TrainedVision
    snapshot = json.loads((output/'snapshot.json').read_text(encoding='utf-8'))
    if hashlib.sha256((output/'manifest.jsonl').read_bytes()).hexdigest() != snapshot['manifest_sha256']:
        raise ValueError('diagnostic manifest changed')
    rows = load_manifest(output/'manifest.jsonl', LABELS, REQUIRED)
    model_manifest = json.loads((model/'manifest.json').read_text(encoding='utf-8'))
    previous = load_manifest(Path(model_manifest['dataset']), LABELS, REQUIRED)
    if hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest() != model_manifest['dataset_id']:
        raise ValueError('model training dataset changed')
    old_hashes = {r['sha256'] for r in previous}
    old_groups = {g for r in previous for g in (r['vehicle_id'],r.get('origin_vehicle_group')) if g}
    if any(r['sha256'] in old_hashes or r['origin_vehicle_group'] in old_groups for r in rows):
        raise ValueError('diagnostic overlaps previous model data')
    vision = TrainedVision(Settings(_env_file=None, cv_backend='trained', trained_model_dir=model))
    predictions = []
    for row in rows:
        image, size = decode((output/row['path']).read_bytes(), 24_000_000)
        try:
            prediction = vision.predict([image])[0]
            issues = list(quality(image,size)) + list(prediction.get('quality_issues') or [])
        finally:
            image.close()
        predictions.append(dict(row, candidate=prediction['candidate'], score=prediction['score'],
            margin=prediction['margin'], quality_issues=issues, correct=row['label']==prediction['candidate']))
    summary = metrics([r['label'] for r in predictions], [r['candidate'] for r in predictions], list(LABELS))
    report = dict(model=str(model), model_version=vision.version, snapshot=snapshot,
        **summary, correct=sum(r['correct'] for r in predictions),
        quality_eligible_count=sum(not r['quality_issues'] for r in predictions),
        predictions=predictions, production_approved=False, thresholds_fitted=False,
        note='One preselected frozen model, frozen reviewed labels. Assistant adjudication is not independent human evidence. Missing classes and unverified '
             'vehicle identities prevent a release claim. No training or gate fitting on these photos.')
    write_json(output/'evaluation.json', report)
    print(json.dumps(dict(count=len(rows), correct=report['correct'], accuracy=report['accuracy'],
        label_counts=snapshot['label_counts'], production_approved=False)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('data/review-workspace/review.sqlite3'))
    parser.add_argument('--cohort', default='blind_v1')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    freeze_reviews(args.database,args.output,args.cohort)
    evaluate(args.output,args.model)


if __name__ == '__main__':
    main()
