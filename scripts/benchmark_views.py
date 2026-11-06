"""Compare frozen checkpoints on exactly the same visually reviewed test images."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation import load_manifest, metrics
from app.imaging import decode
from app.settings import Settings
from app.trained import LABELS, TrainedVision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--model', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous-test', type=Path)
    args = parser.parse_args()
    rows = [r for r in load_manifest(args.manifest, LABELS, LABELS[:-1]) if r['split'] == 'test']
    if not rows or any(r.get('review_status') != 'reviewed' for r in rows):
        raise ValueError('nonempty visually reviewed test set required')
    previous = set()
    if args.previous_test:
        previous = {r['path'] for r in load_manifest(args.previous_test, LABELS, LABELS[:-1])
                    if r['split'] == 'test'}
    results = []
    for checkpoint in args.model:
        vision = TrainedVision(Settings(_env_file=None, cv_backend='trained', trained_model_dir=checkpoint))
        predictions = []
        for row in rows:
            image, _ = decode((args.manifest.parent/row['path']).read_bytes(), 24_000_000)
            try:
                pred = vision.predict([image])[0]
                predictions.append(dict(path=row['path'], label=row['label'], candidate=pred['candidate'],
                    score=pred['score'], derived=row.get('derived', False),
                    evaluation_cohort=row.get('evaluation_cohort')))
            finally:
                image.close()
        subsets = {'all': predictions, 'natural': [r for r in predictions if not r['derived']],
                   'core12': [r for r in predictions if r['label'] != 'other']}
        if previous:
            subsets['previous_test'] = [r for r in predictions if r['path'] in previous]
        fresh = [r for r in predictions if r.get('evaluation_cohort') == 'fresh_v2']
        if fresh:
            subsets['fresh_v2'] = fresh
            subsets['fresh_natural'] = [r for r in fresh if not r['derived']]
        for cohort in sorted({r['evaluation_cohort'] for r in predictions
                              if r.get('evaluation_cohort') and r['evaluation_cohort']!='fresh_v2'}):
            subsets['cohort_'+cohort] = [r for r in predictions if r.get('evaluation_cohort')==cohort]
        results.append(dict(model=str(checkpoint), model_version=vision.version,
            subsets={name: dict(count=len(items), **metrics([r['label'] for r in items],
                      [r['candidate'] for r in items], list(LABELS))) for name, items in subsets.items()},
            predictions=predictions))
    report = dict(dataset=str(args.manifest), results=results, production_approved=False,
        note='Preliminary assistant labels; source groups do not establish verified physical vehicle independence.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps([{ 'model': r['model'], 'subsets': {n: {'count': v['count'], 'accuracy': v['accuracy']}
                       for n, v in r['subsets'].items()}} for r in results]))


if __name__ == '__main__':
    main()
