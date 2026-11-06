"""Evaluate real, locally labelled images; never uses synthetic model predictions.

Manifest JSONL: {path, label, split, vehicle_id, listing_id, covers: [...]}.
Listing images must all belong to the same split. Labels are taxonomy class IDs.
Use calibration split to select thresholds; test split only for frozen-policy reporting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import psutil

from app.evaluation import load_manifest, metrics, threshold_proposal, wilson_lower
from app.imaging import decode
from app.policy import POLICY_VERSION, validate_calibration
from app.inference import load_vision
from app.schemas import EvaluateParams
from app.service import Advisor
from app.settings import Settings



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--split', choices=['calibration', 'test'], default='test')
    parser.add_argument('--output', type=Path, default=Path('reports/evaluation.json'))
    parser.add_argument('--candidate-policy', type=Path)
    parser.add_argument('--propose-policy', type=Path)
    parser.add_argument('--target-precision', type=float, default=.95)
    parser.add_argument('--min-accepted', type=int, default=30)
    args = parser.parse_args()
    if args.propose_policy and args.split != 'calibration':
        parser.error('--propose-policy requires --split calibration')
    if args.candidate_policy and args.split != 'test':
        parser.error('--candidate-policy requires independent --split test')
    settings = Settings()
    if settings.cv_backend == 'trained' and (args.candidate_policy or args.propose_policy):
        parser.error('CLIP similarity policy tools cannot calibrate supervised softmax scores')
    checklist = json.loads(settings.checklist_path.read_text(encoding='utf-8'))
    all_rows = load_manifest(args.manifest, checklist['classes'],
                             {c['id'] for c in checklist['categories']})
    dataset_id = hashlib.sha256(json.dumps(all_rows, sort_keys=True).encode()).hexdigest()
    rows = [r for r in all_rows if r['split'] == args.split]
    if not rows:
        raise SystemExit('No labelled images for requested split; metrics cannot be measured.')
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def sample():
        while not stop.wait(.05):
            peak[0] = max(peak[0], process.memory_info().rss)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    started = time.perf_counter()
    vision = load_vision(settings)
    load_seconds = time.perf_counter() - started
    if args.candidate_policy:
        proposal = json.loads(args.candidate_policy.read_text(encoding='utf-8'))
        if proposal['dataset_id'] != dataset_id:
            raise ValueError('candidate policy belongs to a different dataset manifest')
        vision.calibration = validate_calibration(dict(proposal, validated=True),
                                                  vision.version, vision.taxonomy_hash, vision.names)
    labels = vision.names + ['unknown']
    truth, top1, selective, outputs, times = [], [], [], [], []
    for row in rows:
        path = (args.manifest.parent / row['path']).resolve()
        image, _ = decode(path.read_bytes(), settings.max_pixels)
        start = time.perf_counter()
        pred = vision.predict([image])[0]
        times.append(time.perf_counter()-start)
        image.close()
        pred.pop('embedding')
        truth.append(row['label'])
        top1.append(pred['candidate'])
        selective.append(pred['category'] if pred['recognition'] == 'known' else 'unknown')
        outputs.append(dict(**row, prediction=pred))
    # Run the full quality + duplicate + checklist path per listing, using local storage.
    index = {str(i): (args.manifest.parent / r['path']).resolve() for i, r in enumerate(rows)}
    class LocalStorage:
        def fetch(self, listing_id, photo):
            return index[photo.photo_id].read_bytes()
        def close(self):
            pass
    advisor = Advisor(settings, LocalStorage(), vision)
    required = [c['id'] for c in vision.checklist['categories'] if c['priority'] == 'required']
    counts = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in required}
    listing_times, listing_results = [], []
    for lid in sorted({r['listing_id'] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r['listing_id'] == lid]
        if len(indices) > settings.max_photos:
            raise ValueError('listing exceeds max_photos')
        expected = set().union(*(set(rows[i]['covers']) for i in indices))
        params = EvaluateParams(listing_id=lid, photos=[{'photo_id': str(i), 'object_key': 'local/image'} for i in indices])
        start = time.perf_counter()
        result = advisor.evaluate(params)
        listing_times.append(time.perf_counter()-start)
        present = set(result['required']['present'])
        listing_results.append(dict(listing_id=lid, expected=sorted(expected), present=sorted(present)))
        for i, photo_result in zip(indices, result['photo_results']):
            outputs[i]['quality_eligible'] = (photo_result['quality'] == 'good'
                                              and not photo_result['duplicate_of']
                                              and photo_result['recognition'] != 'unknown')
            outputs[i]['photo_result'] = photo_result
        for c in required:
            counts[c]['tp'] += int(c in expected and c in present)
            counts[c]['fp'] += int(c not in expected and c in present)
            counts[c]['fn'] += int(c in expected and c not in present)
    advisor.close()
    stop.set()
    monitor.join()
    category_metrics = {c: dict(**v,
        precision=v['tp']/(v['tp']+v['fp']) if v['tp']+v['fp'] else None,
        recall=v['tp']/(v['tp']+v['fn']) if v['tp']+v['fn'] else None,
        precision_lower_95=wilson_lower(v['tp'], v['tp']+v['fp'])) for c, v in counts.items()}
    evidence = None
    if args.propose_policy:
        thresholds, evidence = threshold_proposal(outputs, required,
            args.target_precision, args.min_accepted)
        proposal = dict(validated=False, dataset_id=dataset_id, model_version=vision.version,
                        taxonomy_hash=vision.taxonomy_hash, policy_version=POLICY_VERSION,
                        unknown_similarity=vision.calibration['unknown_similarity'], classes=thresholds,
                        notes='Calibration proposal only. Requires independent held-out test approval.')
        args.propose_policy.parent.mkdir(parents=True, exist_ok=True)
        args.propose_policy.write_text(json.dumps(proposal, indent=2), encoding='utf-8')
    report = dict(dataset=str(args.manifest), dataset_id=dataset_id,
                  purpose='candidate_policy_measurement_only' if args.candidate_policy else 'evaluation',
                  production_accuracy_established=False,
                  missing_positive_classes=[c for c, v in counts.items() if not v['tp']+v['fn']],
                  policy_proposal_evidence=evidence, listings=listing_results, split=args.split, count=len(rows),
                  model_version=vision.version, taxonomy_hash=vision.taxonomy_hash,
                  policy=vision.calibration, candidate_metrics=metrics(truth, top1, labels),
                  accepted_metrics=metrics(truth, selective, labels), required_metrics=category_metrics,
                  false_confirmations=sum(v['fp'] for v in counts.values()),
                  unknown_rate=sum(o['prediction']['recognition']=='unknown' for o in outputs)/len(rows),
                  uncertain_rate=sum(o['prediction']['recognition']=='uncertain' for o in outputs)/len(rows),
                  device=settings.device, cpu_threads=settings.cpu_threads,
                  model_load_seconds=load_seconds, median_inference_seconds=float(np.median(times)),
                  p95_inference_seconds=float(np.percentile(times, 95)),
                  listing_batch_seconds=listing_times, peak_rss_bytes=peak[0], predictions=outputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {args.output}; {len(rows)} real labelled images')


if __name__ == '__main__':
    main()
