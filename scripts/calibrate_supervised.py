"""Propose per-class gates; optional release requires an independent frozen human test."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation import load_manifest
from app.data_guard import is_reserved, load_reservations
from app.events import REQUIRED
from app.imaging import decode, quality
from app.settings import Settings
from app.supervised_calibration import fit_gates, group_evidence, load_fresh_holdout
from app.trained import SupervisedPolicy, TrainedVision, load_policy


def predict(vision, rows, root):
    output = []
    for row in rows:
        image, size = decode((root/row['path']).read_bytes(), 24_000_000)
        try:
            result = vision.predict([image])[0]
            eligible = not quality(image,size) and not result.get('quality_issues')
        finally:
            image.close()
        output.append(dict(row, candidate=result['candidate'], score=result['score'], margin=result['margin'],
                           quality_eligible=eligible))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--holdout', type=Path)
    parser.add_argument('--release-to', type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.release_to and args.release_to.exists()):
        parser.error('choose new output paths; reports and releases are immutable')
    if args.release_to and not args.holdout:
        parser.error('release requires a frozen independent human holdout')
    all_rows = load_manifest(args.manifest, (*REQUIRED, 'other'), REQUIRED)
    reservation = load_reservations()
    if any(is_reserved(row,reservation) for row in all_rows):
        raise ValueError('reserved blind photos must remain separate from calibration/development data')
    rows = [r for r in all_rows if r['split'] == 'calibration']
    if not rows:
        parser.error('calibration images required')
    vision = TrainedVision(Settings(_env_file=None, cv_backend='trained', trained_model_dir=args.model))
    model_manifest = json.loads((args.model/'manifest.json').read_text(encoding='utf-8'))
    previous = load_manifest(Path(model_manifest['dataset']), (*REQUIRED, 'other'), REQUIRED)
    trained_groups = {g for r in previous if r['split'] == 'train'
                      for g in (r['vehicle_id'], r.get('origin_vehicle_group')) if g}
    trained_hashes = {r['sha256'] for r in previous if r['split'] == 'train'}
    if any(r['vehicle_id'] in trained_groups or r.get('origin_vehicle_group') in trained_groups
           or r['sha256'] in trained_hashes for r in rows):
        raise ValueError('calibration overlaps training images or vehicle groups')
    gates, evidence = fit_gates(predict(vision, rows, args.manifest.parent))
    human_calibration = all(r.get('reviewer_type') == 'human' and r.get('vehicle_identity_verified')
                            and not r.get('derived') for r in rows)
    report = dict(model=vision.version, calibration_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        proposed_classes=gates, calibration_evidence=evidence, human_calibration=human_calibration,
        release_approved=False, approved_classes=[], disabled_classes=list(REQUIRED),
        reason='frozen_human_holdout_required', note='Scores are not calibrated probabilities. No production settings changed. '
        'Quality matches the runtime filter. Any wrong eligible accepted frame makes its vehicle an error, '
        'even if a duplicate or correct view could hide it in a complete gallery. Positive/negative support may overlap.')
    qualified, heldout_evidence = {}, {}
    test_hash = None
    if args.holdout:
        test, test_hash = load_fresh_holdout(args.holdout, model_manifest)
        calibration_groups = {r['vehicle_id'] for r in rows}
        calibration_hashes = {r['sha256'] for r in rows}
        if any(r['vehicle_id'] in calibration_groups or r['sha256'] in calibration_hashes for r in test):
            raise ValueError('holdout overlaps calibration images or vehicle groups')
        predictions = predict(vision, test, args.holdout.parent)
        for category, gate in gates.items():
            counts = group_evidence(predictions, category, gate)
            heldout_evidence[category] = counts
            if (human_calibration and counts['positive_groups'] >= 30 and counts['negative_groups'] >= 30
                    and counts['precision_lower_95'] is not None and counts['precision_lower_95'] >= .95):
                qualified[category] = gate
        report.update(heldout_evidence=heldout_evidence, test_manifest_sha256=test_hash,
            approved_classes=list(qualified), disabled_classes=[c for c in REQUIRED if c not in qualified],
            release_approved=bool(qualified), reason='qualified_classes_only' if qualified else 'insufficient_heldout_evidence')
    if args.release_to and qualified:
        policy = SupervisedPolicy(validated=True, taxonomy_hash=vision.taxonomy_hash, classes=qualified).model_dump()
        approval = dict(model_sha256=model_manifest['sha256'], dataset_id=model_manifest['dataset_id'],
            test_manifest_sha256=test_hash, policy_sha256=hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
            classes={c: {k: heldout_evidence[c][k] for k in ('tp','fp','fn','tn')} for c in qualified})
        released = dict(model_manifest, calibration=policy, heldout_approval=approval)
        load_policy(released, released['sha256'], vision.taxonomy_hash)
        args.release_to.mkdir(parents=True)
        shutil.copyfile(args.model/'weights.pt', args.release_to/'weights.pt')
        (args.release_to/'manifest.json').write_text(json.dumps(released, indent=2), encoding='utf-8')
        report['release_directory'] = str(args.release_to)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k:report[k] for k in ('release_approved','approved_classes','reason')}))


if __name__ == '__main__':
    main()
