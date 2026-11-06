"""Select a frozen checkpoint on calibration only, with baseline regression guards."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from app.data_guard import is_reserved, load_reservations
from app.evaluation import load_manifest, metrics
from app.events import REQUIRED
from app.imaging import decode
from app.settings import Settings
from app.trained import LABELS, TrainedVision


def summarize(predictions):
    result = metrics([r['label'] for r in predictions],[r['candidate'] for r in predictions],list(LABELS))
    positive = [r for r in predictions if r['candidate'] != 'other']
    result['positive_assignment_precision'] = (
        sum(r['candidate']==r['label'] for r in positive)/len(positive) if positive else None)
    return result


def choose(results, tolerance=.02):
    if not results or not 0 <= tolerance <= 1:
        raise ValueError('baseline and valid tolerance required')
    baseline = results[0]
    if baseline['metrics']['positive_assignment_precision'] is None:
        raise ValueError('baseline must make positive predictions')
    eligible = [baseline]
    for result in results[1:]:
        m,b = result['metrics'],baseline['metrics']
        if (m['positive_assignment_precision'] is not None
                and m['accuracy'] >= b['accuracy']-tolerance
                and m['positive_assignment_precision'] >= b['positive_assignment_precision']-tolerance
                and m['macro_f1'] > b['macro_f1']):
            eligible.append(result)
    # Stable ordering preserves the baseline on a tie.
    return max(eligible,key=lambda r:r['metrics']['macro_f1'])['model']


def calibration_rows(manifest, checkpoints):
    all_rows = load_manifest(manifest,LABELS,REQUIRED)
    registry = load_reservations()
    if any(is_reserved(r,registry) for r in all_rows):
        raise ValueError('reserved diagnostic photos cannot be used for selection')
    rows = [r for r in all_rows if r['split']=='calibration']
    if not rows or any(r.get('review_status')!='reviewed' for r in rows):
        raise ValueError('reviewed calibration photos required')
    for checkpoint in checkpoints:
        model_manifest = json.loads((checkpoint/'manifest.json').read_text(encoding='utf-8'))
        previous = load_manifest(Path(model_manifest['dataset']),LABELS,REQUIRED)
        if hashlib.sha256(json.dumps(previous,sort_keys=True).encode()).hexdigest()!=model_manifest['dataset_id']:
            raise ValueError('model training dataset changed')
        trained = [r for r in previous if r['split']=='train']
        hashes = {r['sha256'] for r in trained}
        groups = {g for r in trained for g in (r['vehicle_id'],r.get('origin_vehicle_group')) if g}
        if any(r['sha256'] in hashes or r['vehicle_id'] in groups or r.get('origin_vehicle_group') in groups for r in rows):
            raise ValueError('calibration overlaps model training')
    return rows


def evaluate_calibration(manifest, checkpoint):
    rows = calibration_rows(manifest,[checkpoint])
    vision = TrainedVision(Settings(_env_file=None,cv_backend='trained',trained_model_dir=checkpoint))
    predictions = []
    for row in rows:
        image,_ = decode((manifest.parent/row['path']).read_bytes(),24_000_000)
        try:
            predicted = vision.predict([image])[0]
            predictions.append(dict(path=row['path'],label=row['label'],candidate=predicted['candidate']))
        finally:
            image.close()
    return dict(model=str(checkpoint),model_version=vision.version,count=len(rows),
                metrics=summarize(predictions),predictions=predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a new report path')
    results = []
    for checkpoint in (args.baseline,*args.candidate):
        result = evaluate_calibration(args.manifest,checkpoint)
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('predictions','metrics')} |
            {k:result['metrics'][k] for k in ('accuracy','macro_f1','positive_assignment_precision')}),flush=True)
    selected = choose(results)
    report = dict(kind='calibration_only_checkpoint_selection',manifest=str(args.manifest),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        selected_model=selected,results=results,regression_tolerance=.02,
        production_approved=False,holdout_used=False,
        criterion='Maximize calibration macro-F1; accuracy and positive assignment precision '
                  'must stay within 2 percentage points of baseline. Baseline wins ties.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
    print('Selected: '+selected)


if __name__=='__main__':
    main()
