import hashlib
import json
import math
from pathlib import Path

import numpy as np


def load_manifest(path, labels, categories):
    path = Path(path)
    root = path.parent.resolve()
    rows = []
    owners, images = {}, {}
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k].strip()
                for k in ('path', 'label', 'split', 'vehicle_id', 'listing_id'))
                or row['split'] not in {'train', 'calibration', 'test'}
                or row['label'] not in labels or not isinstance(row.get('covers'), list)
                or any(c not in categories for c in row['covers']) or len(row['covers']) > 1):
            raise ValueError(f'invalid manifest row {number}')
        image = (root / row['path']).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            raise ValueError(f'image outside dataset or missing at row {number}')
        with image.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest in images:
            raise ValueError(f'duplicate image content at rows {images[digest]} and {number}')
        images[digest] = number
        for field in ('vehicle_id', 'listing_id'):
            identity = (field, row[field])
            if identity in owners and owners[identity] != row['split']:
                raise ValueError('vehicle/listing leakage between splits')
            owners[identity] = row['split']
        rows.append(dict(row, sha256=digest))
    return rows


def metrics(y_true, y_pred, labels):
    if len(y_true) != len(y_pred):
        raise ValueError('prediction count mismatch')
    positions = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, predicted in zip(y_true, y_pred):
        matrix[positions[truth], positions[predicted]] += 1
    rows = []
    for i, label in enumerate(labels):
        tp, predicted, support = int(matrix[i, i]), int(matrix[:, i].sum()), int(matrix[i].sum())
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        f1 = 2 * tp / (predicted + support) if predicted + support else None
        rows.append(dict(label=label, precision=precision, recall=recall, f1=f1,
                         support=support, predicted=predicted))
    supported = [r for r in rows if r['support']]
    return dict(accuracy=float(np.trace(matrix)/matrix.sum()) if matrix.sum() else None,
                macro_f1=float(np.mean([r['f1'] for r in supported])) if supported else None,
                per_class=rows, confusion_matrix=matrix.tolist(), labels=labels)


def wilson_lower(tp, total, z=1.96):
    if not total:
        return None
    p = tp / total
    return (p + z*z/(2*total) - z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1 + z*z/total)


def threshold_proposal(predictions, categories, target_precision=.95, min_accepted=30):
    """Candidate thresholds on calibration only, never a validation claim."""
    if not 0 < target_precision < 1 or min_accepted < 1:
        raise ValueError('invalid calibration objective')
    if any(row['split'] != 'calibration' for row in predictions):
        raise ValueError('threshold selection requires calibration split only')
    thresholds, evidence = {}, {}
    for category in categories:
        candidates = [r for r in predictions if r['prediction']['candidate'] == category
                      and r.get('quality_eligible', False)]
        best = None
        scores = sorted({r['prediction']['similarity'] for r in candidates})
        margins = sorted({r['prediction']['margin'] for r in candidates})
        if not candidates:
            evidence[category] = dict(reason='insufficient_independent_evidence')
            continue
        score_grid = np.unique(np.quantile(scores, np.linspace(0, 1, min(32, len(scores))), method='nearest'))
        margin_grid = np.unique(np.quantile(margins, np.linspace(0, 1, min(32, len(margins))), method='nearest'))
        for score in score_grid:
            for margin in margin_grid:
                accepted = [r for r in candidates if r['prediction']['similarity'] >= score
                            and r['prediction']['margin'] >= margin]
                # Count each vehicle once: repeated frames cannot manufacture evidence.
                groups = {}
                for row in accepted:
                    groups.setdefault(row['vehicle_id'], []).append(category in row['covers'])
                tp, total = sum(all(values) for values in groups.values()), len(groups)
                lower = wilson_lower(tp, total)
                if total < min_accepted or lower < target_precision:
                    continue
                value = (tp, -(total-tp), -score, -margin)
                if best is None or value > best[0]:
                    best = (value, score, margin, tp, total, lower)
        if best:
            _, score, margin, tp, total, lower = best
            thresholds[category] = dict(min_similarity=float(score), min_margin=float(margin))
            evidence[category] = dict(accepted_vehicles=total, correct_vehicles=tp,
                                      observed_precision=tp/total, precision_lower_95=lower)
        else:
            evidence[category] = dict(reason='insufficient_independent_evidence')
    return thresholds, evidence
