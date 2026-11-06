"""Assemble reviewed licensed images and explicit, traceable fender crops."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from app.events import REQUIRED
from app.data_guard import is_reserved, load_reservations
from app.data_sources import author_group as author_group


def split_groups(records, seed, fixed=None):
    fixed = fixed or {}
    groups = defaultdict(set)
    for row in records:
        groups[row['vehicle_id']].add(row['label'])
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    assignment = dict.fromkeys(order, 'train')
    assignment.update({g: s for g, s in fixed.items() if g in groups})
    # Reserve a train group for every available label before selecting held-out groups.
    for split in ('calibration', 'test'):
        for label in (*REQUIRED, 'other'):
            if any(assignment[g] == split and label in labels for g, labels in groups.items()):
                continue
            train_counts = Counter(c for g, labels in groups.items() if assignment[g] == 'train' for c in labels)
            for group in order:
                if (group not in fixed and assignment[group] == 'train' and label in groups[group]
                        and all(train_counts[c] > 1 for c in groups[group])):
                    assignment[group] = split
                    break
    # Weak train-only expansion must not force all new reviewed photos into holdouts.
    reviewed_groups = {r['vehicle_id'] for r in records if r.get('review_status', 'reviewed') == 'reviewed'}
    target = max(1, round(len(reviewed_groups) * .2))
    for split in ('calibration', 'test'):
        for group in order:
            if sum(s == split for s in assignment.values()) >= target:
                break
            train_counts = Counter(c for g, labels in groups.items() if assignment[g] == 'train' for c in labels)
            if group not in fixed and assignment[group] == 'train' and all(train_counts[c] > 1 for c in groups[group]):
                assignment[group] = split
    return assignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, default=Path('datasets/view-training-v1'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fender-crops', action='store_true')
    parser.add_argument('--allow-weak-train', action='store_true')
    parser.add_argument('--split-from', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a new output directory; assembled datasets are immutable')
    args.output.mkdir(parents=True)
    (args.output/'images').mkdir()
    records, hashes = [], {}
    exclusions = []
    reservation = load_reservations()
    fixed = {}
    if args.split_from:
        for line in args.split_from.read_text(encoding='utf-8').splitlines():
            previous = json.loads(line)
            group, split = previous['vehicle_id'], previous['split']
            if group in fixed and fixed[group] != split:
                raise ValueError('previous manifest splits a vehicle group')
            fixed[group] = split
    for source in args.source:
        candidates = json.loads((source/'candidates.json').read_text(encoding='utf-8'))
        reviews = json.loads((source/'reviewed-labels.json').read_text(encoding='utf-8'))
        decisions = {str(r.get('image_id', r.get('page_id'))): r for r in reviews}
        for item in candidates:
            source_id = str(item.get('image_id', item.get('page_id')))
            review = decisions.get(source_id)
            weak = bool(review and review['status'] == 'source_annotated'
                and args.allow_weak_train and item.get('fixed_split') == 'train'
                and review.get('label_origin') == 'human_part_mask_and_automatic_view_side_hint'
                and review.get('label') != 'other')
            if not review or (review['status'] != 'reviewed' and not weak):
                exclusions.append(dict(source=str(source), source_id=source_id, reason='not reviewed/excluded'))
                continue
            label = review['label']
            if label not in (*REQUIRED, 'other'):
                raise ValueError('label outside thirteen-class training taxonomy')
            path = (source/item['path']).resolve()
            if not path.is_relative_to(source.resolve()):
                raise ValueError('source image outside candidate root')
            group = review.get('vehicle_group') or item.get('vehicle_group') or author_group(item)
            explicit_split = review.get('fixed_split') or item.get('fixed_split')
            if explicit_split:
                if explicit_split not in {'train', 'calibration', 'test'}:
                    raise ValueError('invalid fixed split')
                if group in fixed and fixed[group] != explicit_split:
                    raise ValueError('fixed split conflicts with frozen group assignment')
                fixed[group] = explicit_split
            name = re.sub('[^A-Za-z0-9_-]', '-', f'{source.name}-{source_id}')
            base = dict(label=label, covers=[] if label == 'other' else [label],
                vehicle_id=group, listing_id=group, review_status=review['status'],
                reviewer=review.get('reviewer', 'unspecified'), source_url=item['source_url'],
                reviewer_type=review.get('reviewer_type', 'assistant'),
                vehicle_identity_verified=review.get('vehicle_identity_verified', False),
                source_kind=item.get('source_kind'), training_permission=item.get('training_permission', False),
                origin_vehicle_group=item.get('origin_vehicle_group') or item.get('vehicle_group') or author_group(item),
                author=item['author'], license=item['license'], license_url=item['license_url'],
                source_id=source_id, source_dataset=source.name, derived=item.get('derived', False),
                label_origin=review.get('label_origin', 'assistant_visual_review'),
                evaluation_cohort=review.get('evaluation_cohort'),
                source_revision=item.get('source_revision'), parent_id=item.get('parent_id'),
                crop_box_pixels=item.get('crop_box_pixels'),
                identity_basis=item.get('identity_basis', 'conservative source-author group; not verified physical vehicle identity'))
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            origin = item.get('origin_vehicle_group') or item.get('vehicle_group') or author_group(item)
            if is_reserved(dict(vehicle_id=group, origin_vehicle_group=origin, sha256=digest), reservation):
                exclusions.append(dict(source_id=source_id, reason='reserved_blind_holdout'))
                continue
            if digest in hashes:
                if hashes[digest]['label'] != label:
                    raise ValueError('identical image has conflicting labels')
                exclusions.append(dict(source_id=source_id, reason='exact duplicate'))
                continue
            relative = f'images/{name}.jpg'
            (args.output/relative).write_bytes(data)
            row = dict(base, path=relative)
            records.append(row)
            hashes[digest] = row
            if args.fender_crops and label in {'left_side', 'right_side'} and item.get('box'):
                box = item['box']
                left, top, right, bottom = [float(box[k]) for k in ('XMin', 'YMin', 'XMax', 'YMax')]
                with Image.open(path) as image:
                    # The full side was visually reviewed; boxes define only crop coordinates.
                    for axle in ('front', 'rear'):
                        side = label.split('_')[0]
                        at_left = (side == 'left') == (axle == 'front')
                        x0, x1 = ((0, .46) if at_left else (.54, 1))
                        region = (int((left+x0*(right-left))*image.width),
                                  int((top+.28*(bottom-top))*image.height),
                                  int((left+x1*(right-left))*image.width),
                                  int(bottom*image.height))
                        crop = image.crop(region)
                        if min(crop.size) < 96:
                            crop.close()
                            continue
                        target = f'fender_{axle}_{side}'
                        relative = f'images/{name}-{target}.jpg'
                        crop.save(args.output/relative, quality=93)
                        crop.close()
                        records.append(dict(base, path=relative, label=target, covers=[target],
                            derived=True, parent_path=row['path'], crop_box_pixels=region,
                            derivative_note='Derived crop from reviewed side view; not an independent natural close-up.'))
    assignment = split_groups(records, args.seed, fixed)
    for row in records:
        row['split'] = assignment[row['vehicle_id']]
    manifest = args.output/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    summary = dict(images=len(records), source_images=sum(not r['derived'] for r in records),
        derived_images=sum(r['derived'] for r in records), seed=args.seed,
        review_status_counts=dict(Counter(r['review_status'] for r in records)),
        split_counts={s: dict(Counter(r['label'] for r in records if r['split'] == s))
                      for s in ('train', 'calibration', 'test')},
        natural_test_counts=dict(Counter(r['label'] for r in records
            if r['split'] == 'test' and not r['derived'])),
        exclusions=exclusions, production_dataset_validated=False,
        note='Assistant preliminary labels and conservative source groups. Requires human review and verified vehicle identities.')
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    attribution = ['# Image attributions and transformations\n']
    for row in records:
        attribution.append(f"- {row['path']}: {row['author']}; {row['source_url']}; {row['license']} ({row['license_url']}); "
                           + ('fender crop + JPEG resize' if row['derived'] else 'JPEG resize') + '\n')
    (args.output/'ATTRIBUTION.md').write_text(''.join(attribution), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'exclusions'}, ensure_ascii=True))


if __name__ == '__main__':
    main()
