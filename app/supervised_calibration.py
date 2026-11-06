"""Select score gates on calibration; qualify only on a separate frozen human holdout."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from app.evaluation import load_manifest, wilson_lower
from app.events import REQUIRED
from app.imaging import decode, fingerprint


def group_evidence(rows, category, gate):
    groups = defaultdict(list)
    for row in rows:
        groups[row['vehicle_id']].append(row)
    tp = fp = fn = tn = positives = negatives = 0
    for values in groups.values():
        positive = any(r['label'] == category for r in values)
        negative = any(r['label'] != category and r.get('quality_eligible', True) for r in values)
        positives += positive
        negatives += negative
        accepted = [r for r in values if r['candidate'] == category
                    and r.get('quality_eligible', True)
                    and r['score'] >= gate['min_score'] and r['margin'] >= gate['min_margin']]
        correct = bool(accepted) and all(r['label'] == category for r in accepted)
        wrong = bool(accepted) and not correct
        # A wrong accepted frame is an error even when this car also has a valid frame.
        tp += correct
        fp += wrong
        fn += positive and not correct
        tn += negative and not wrong
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), positive_groups=int(positives),
                negative_groups=int(negatives), precision_lower_95=wilson_lower(tp, tp+fp),
                recall=tp/positives if positives else None,
                acceptance_rate=(tp+fp)/len(groups) if groups else None)


def fit_gates(rows, target=.95, minimum=30):
    if not .5 < target < 1 or minimum < 1:
        raise ValueError('invalid calibration objective')
    if not rows or any(r['split'] != 'calibration' for r in rows):
        raise ValueError('threshold selection requires calibration only')
    for row in rows:
        if any(not math.isfinite(row[k]) or not 0 <= row[k] <= 1 for k in ('score', 'margin')):
            raise ValueError('invalid supervised score')
    gates, evidence = {}, {}
    for category in REQUIRED:
        base = group_evidence(rows, category, {'min_score': 0., 'min_margin': 0.})
        if base['positive_groups'] < minimum or base['negative_groups'] < minimum:
            evidence[category] = dict(base, reason='insufficient_positive_or_negative_groups')
            continue
        candidates = [r for r in rows if r['candidate'] == category]
        if not candidates:
            evidence[category] = dict(base, reason='no_candidates')
            continue
        scores = np.unique(np.quantile([r['score'] for r in candidates], np.linspace(0,1,min(32,len(candidates))), method='nearest'))
        margins = np.unique(np.quantile([r['margin'] for r in candidates], np.linspace(0,1,min(32,len(candidates))), method='nearest'))
        best = None
        for score in scores:
            for margin in margins:
                gate = dict(min_score=float(score), min_margin=float(margin))
                result = group_evidence(rows, category, gate)
                if result['tp']+result['fp'] < minimum or result['precision_lower_95'] < target:
                    continue
                rank = (result['tp'], -result['fp'], -score, -margin)
                if best is None or rank > best[0]:
                    best = (rank, gate, result)
        if best:
            gates[category], evidence[category] = best[1:]
        else:
            evidence[category] = dict(base, reason='no_gate_meets_precision_bound')
    return gates, evidence


def load_fresh_holdout(manifest, model_manifest):
    manifest = Path(manifest)
    snapshot = json.loads(manifest.with_name('snapshot.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if snapshot.get('kind') != 'fresh_human_holdout' or snapshot.get('manifest_sha256') != digest:
        raise ValueError('holdout is not a matching frozen human snapshot')
    raw = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = load_manifest(manifest, (*REQUIRED, 'other'), REQUIRED)
    if not rows or len(rows) != snapshot.get('count'):
        raise ValueError('empty or incomplete holdout')
    previous = load_manifest(Path(model_manifest['dataset']), (*REQUIRED, 'other'), REQUIRED)
    previous_digest = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
    if model_manifest.get('dataset_id') != previous_digest:
        raise ValueError('parent model dataset changed after training')
    groups = {g for r in previous for g in (r['vehicle_id'], r.get('origin_vehicle_group')) if g}
    hashes = {r['sha256'] for r in previous}
    for declared, row in zip(raw, rows):
        if (row['split'] != 'test' or row.get('reviewer_type') != 'human'
                or row.get('vehicle_identity_verified') is not True or row.get('derived')
                or declared.get('sha256') != row['sha256']):
            raise ValueError('holdout requires verified human original images with matching checksums')
        if row['vehicle_id'] in groups or row.get('origin_vehicle_group') in groups or row['sha256'] in hashes:
            raise ValueError('holdout overlaps previously used model data')
    old_fingerprints = []
    for row in previous:
        data = (Path(model_manifest['dataset']).parent/row['path']).read_bytes()
        image, _ = decode(data, 24_000_000)
        try:
            old_fingerprints.append(fingerprint(data,image))
        finally:
            image.close()
    for row in rows:
        data = (manifest.parent/row['path']).read_bytes()
        image, _ = decode(data,24_000_000)
        try:
            current = fingerprint(data,image)
        finally:
            image.close()
        for old in old_fingerprints:
            if np.count_nonzero(current[1] != old[1]) <= 3 and np.mean((current[2]-old[2])**2) < .004:
                raise ValueError('holdout contains a near duplicate of previously used data')
    return rows, digest
