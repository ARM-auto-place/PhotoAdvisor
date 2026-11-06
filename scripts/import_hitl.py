"""Stream CC0 HITL photos and masks; keep automatic side tags explicitly weak."""
import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re

from PIL import Image


VIEW_LABELS = {'front': 'front', 'back': 'rear', 'left': 'left_side', 'right': 'right_side'}


def split_for_group(group):
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
    return 'test' if bucket < 2 else ('calibration' if bucket < 4 else 'train')


def part_crop(obj, size, padding=.35):
    points = obj.get('points', {}).get('exterior', [])
    if len(points) < 3:
        return None
    width, height = size
    xs, ys = [float(p[0]) for p in points], [float(p[1]) for p in points]
    if not all(math.isfinite(v) for v in xs + ys):
        return None
    if min(xs) < 0 or min(ys) < 0 or max(xs) > width or max(ys) > height:
        return None
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    if (x1-x0)*(y1-y0) < width*height*.015:
        return None
    dx, dy = (x1-x0)*padding, (y1-y0)*padding
    box = (max(0, int(x0-dx)), max(0, int(y0-dy)),
           min(width, int(x1+dx)), min(height, int(y1+dy)))
    return box if min(box[2]-box[0], box[3]-box[1]) >= 96 else None


def proposed_side(obj, view):
    tags = {t['name']: t.get('value') for t in obj.get('tags', [])}
    side = tags.get('side')
    if (side not in {'left', 'right'} or tags.get('needs_review') != 'no'
            or not view.endswith(side) or 'single object' not in tags.get('side_reason', '')):
        return None
    return side


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('datasets/hitl-source'))
    parser.add_argument('--output', type=Path, default=Path('datasets/hitl-candidates'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a new output directory')
    (args.output/'images').mkdir(parents=True)
    (args.output/'annotations').mkdir()
    source = json.loads((args.source/'source.json').read_text(encoding='utf-8'))
    candidates, proposals = [], []
    seen = set()
    def append_record(image, image_id, group, view, label, *, derived=False, parent_id=None, box=None):
        relative = f'images/{image_id}.jpg'
        image.thumbnail((800, 800))
        image.save(args.output/relative, quality=93)
        record = dict(image_id=image_id, path=relative, title=image_id,
            source_url='https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/',
            author='Humans in the Loop / Beetroot Academy', license='CC0',
            license_url='https://creativecommons.org/publicdomain/zero/1.0/',
            source_revision=source['revision'], vehicle_group=group, fixed_split=split_for_group(group),
            source_view_hint=view, label_hint=label, review_status='pending',
            derived=derived, parent_id=parent_id, crop_box_pixels=box,
            identity_basis='filename-family group; physical vehicle identity not independently verified',
            label_origin='human_part_mask_and_automatic_view_side_hint')
        candidates.append(record)
        proposals.append(dict(image_id=image_id, vehicle_group=group, fixed_split=record['fixed_split'],
            label=label, status='source_annotated' if record['fixed_split']=='train' else 'pending',
            reviewer='not_manually_reviewed', label_origin=record['label_origin'],
            note='Weak label from automatic publisher view/side hints. Never use as gold evaluation data.'))
    for path in sorted((args.source/'data').glob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4):
            for row in batch.to_pylist():
                image_id = Path(row['image_rel']).stem.replace(' ', '_')
                family = re.sub(r'_(?:0\d|\d{2})$', '', image_id)
                group = 'hitl-'+family
                data = row['image']['bytes']
                digest = hashlib.sha256(data).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                annotation = json.loads(row['annotation_json'])
                with Image.open(BytesIO(data)) as original:
                    if original.size != (annotation['size']['width'], annotation['size']['height']):
                        raise ValueError('image and annotation dimensions disagree')
                    base = original.convert('RGB')
                try:
                    view = row['view']
                    objects = annotation['objects']
                    titles = {o['classTitle'] for o in objects}
                    # Incomplete views cannot become positive full-car profile examples.
                    global_label = VIEW_LABELS.get(view, 'other')
                    if global_label.endswith('_side') and not {'Front-wheel','Back-wheel'}.issubset(titles):
                        global_label = 'other'
                    if global_label in {'front','rear'} and len(titles) < 6:
                        global_label = 'other'
                    append_record(base.copy(), image_id, group, view, global_label)
                    for index, obj in enumerate(objects):
                        axle = {'Fender': 'front', 'Quarter-panel': 'rear'}.get(obj['classTitle'])
                        side = proposed_side(obj, view)
                        if not axle or not side:
                            continue
                        box = part_crop(obj, base.size)
                        if box is None:
                            continue
                        crop = base.crop(box)
                        try:
                            label = f'fender_{axle}_{side}'
                            append_record(crop, f'{image_id}-part-{index}', group, view, label,
                                          derived=True, parent_id=image_id, box=box)
                        finally:
                            crop.close()
                    (args.output/'annotations'/f'{image_id}.json').write_text(
                        row['annotation_json'], encoding='utf-8')
                finally:
                    base.close()
    (args.output/'candidates.json').write_text(json.dumps(candidates, indent=2), encoding='utf-8')
    (args.output/'suggested-labels.json').write_text(json.dumps(proposals, indent=2), encoding='utf-8')
    (args.output/'reviewed-labels.json').write_text(json.dumps(proposals, indent=2), encoding='utf-8')
    summary = dict(originals=len(seen), samples=len(candidates),
        class_hints=dict(Counter(r['label_hint'] for r in candidates)),
        split_hints=dict(Counter(r['fixed_split'] for r in candidates)),
        warning='Side/view labels are weak. Calibration/test examples must be visually reviewed.')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
