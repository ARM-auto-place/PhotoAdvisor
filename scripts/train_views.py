"""Train a small supervised head on verified images; never auto-promote a model."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import ImageEnhance, ImageOps
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision import transforms

from app.evaluation import load_manifest, metrics
from app.data_guard import is_reserved, load_reservations
from app.imaging import decode
from app.settings import Settings
from app.trained import (AngleNet, ClipAngleNet, FusedAngleNet, LABELS, TrainedVision,
                         DinoAngleNet, TripleAngleNet)
from scripts.collect_commons import allowed_license


FLIPPED = {label: label for label in LABELS}
FLIPPED.update(left_side='right_side', right_side='left_side',
    fender_front_left='fender_front_right', fender_front_right='fender_front_left',
    fender_rear_left='fender_rear_right', fender_rear_right='fender_rear_left')


def check_rows(rows, allow_partial):
    reservation = load_reservations()
    for row in rows:
        if is_reserved(row, reservation):
            raise ValueError('reserved blind holdout cannot enter model development')
        weak_train = (row.get('review_status') == 'source_annotated'
            and row['split'] == 'train'
            and row.get('label_origin') == 'human_part_mask_and_automatic_view_side_hint')
        if row.get('review_status') != 'reviewed' and not weak_train:
            raise ValueError('evaluation requires visual review; weak labels are train-only')
        own = row.get('source_kind') == 'own' and row.get('training_permission') is True
        if not own and not (allowed_license(row.get('license', '')) and row.get('source_url')
                            and row.get('author')):
            raise ValueError('source attribution and usable license required')
    counts = {split: {c: len({r['vehicle_id'] for r in rows
                             if r['split'] == split and r['label'] == c})
                      for c in LABELS} for split in ('train', 'calibration', 'test')}
    if not all(any(r['split'] == split for r in rows) for split in counts):
        raise ValueError('train, calibration and independent test splits are all required')
    missing = [c for c in LABELS if not counts['train'][c]]
    if missing and not allow_partial:
        raise ValueError(f'no reviewed training examples for {missing}; collect data before full training')
    return counts


def appearance_count(row, enabled=False):
    return 3 if (enabled and row['split']=='train' and row.get('review_status')=='reviewed'
                 and not row.get('derived', False)) else 1


def appearance_variants(image, row, enabled=False):
    if appearance_count(row, enabled)==1:
        return [image]
    variants = [image]
    for brightness, contrast, saturation in ((.55, .8, .4), (1.2, 1.3, .6)):
        lit = ImageEnhance.Brightness(image).enhance(brightness)
        contrasted = ImageEnhance.Contrast(lit).enhance(contrast)
        lit.close()
        variants.append(ImageEnhance.Color(contrasted).enhance(saturation))
        contrasted.close()
    return variants


def extract(model, rows, root, flip, batch_size, appearance_augment=False):
    inputs, targets, features = [], [], []
    def flush():
        if inputs:
            with torch.inference_mode():
                features.append(model.embed(torch.stack(inputs)).clone())
            inputs.clear()
    for row in rows:
        image, _ = decode((root / row['path']).read_bytes(), 24_000_000)
        variants = [image]
        try:
            variants = appearance_variants(image, row, appearance_augment)
            for variant in variants:
                inputs.append(model.prepare_image(variant))
                targets.append(LABELS.index(row['label']))
                if len(inputs) >= batch_size:
                    flush()
                if flip and row['split'] == 'train':
                    mirrored = ImageOps.mirror(variant)
                    try:
                        inputs.append(model.prepare_image(mirrored))
                        targets.append(LABELS.index(FLIPPED[row['label']]))
                    finally:
                        mirrored.close()
                    if len(inputs) >= batch_size:
                        flush()
        finally:
            for variant in variants:
                variant.close()
    flush()
    return torch.cat(features), torch.tensor(targets, dtype=torch.long)


def extract_splits(model, split_rows, root, flip, batch_size, defer_test=False, appearance_augment=False):
    # A selection run can leave test images entirely outside model execution.
    extracted = {}
    for split, items in split_rows.items():
        if defer_test and split == 'test':
            continue
        print(json.dumps(dict(phase='extract_features', split=split, photos=len(items))), flush=True)
        extracted[split] = extract(model, items, root, flip, batch_size, appearance_augment)
    return extracted


def balanced_sampling_weights(rows, weak_weight=.35):
    """Equal class mass; weak labels contribute less within their class."""
    raw = torch.tensor([weak_weight if r.get('review_status') == 'source_annotated' else 1.
                        for r in rows], dtype=torch.float64)
    totals = Counter()
    for row, weight in zip(rows, raw.tolist()):
        totals[row['label']] += weight
    return torch.tensor([float(w)/totals[r['label']] for r, w in zip(rows, raw)], dtype=torch.float64)


def macro_cross_entropy(logits, targets):
    losses = nn.functional.cross_entropy(logits, targets, reduction='none')
    return torch.stack([losses[targets == label].mean() for label in targets.unique()]).mean()


def head_sample_weights(rows, flip, weak_weight, appearance_augment=False):
    return torch.tensor([weight for row in rows
        for weight in ([weak_weight if row.get('review_status') == 'source_annotated' else 1.]
                       * (2 if flip and row.get('split','train')=='train' else 1) * appearance_count(row,appearance_augment))])


def effective_head_counts(rows, flip, weak_weight, allow_partial=False, appearance_augment=False):
    counts = Counter(label for row in rows
        if weak_weight>0 or row.get('review_status')!='source_annotated'
        for label in (([row['label'],FLIPPED[row['label']]] if flip and row.get('split','train')=='train'
                       else [row['label']]) * appearance_count(row,appearance_augment)))
    missing = [c for c in LABELS if not counts[c]]
    if missing and not allow_partial:
        raise ValueError(f'no effective head training examples for {missing}')
    return dict(counts)


def head_class_weights(targets, sample_weights, power=.5):
    """Balance effective class mass after downweighting weak source labels."""
    if not 0 <= power <= 1:
        raise ValueError('class weight power must be between zero and one')
    frequency = torch.bincount(targets, weights=sample_weights, minlength=len(LABELS))
    safe_frequency = torch.where(frequency > 0, frequency, torch.ones_like(frequency))
    return (frequency.sum()/safe_frequency).pow(power)


def check_warm_start_splits(previous, current):
    """Previously seen groups cannot become evaluation data after warm starting."""
    assignments = {r['vehicle_id']: r['split'] for r in previous}
    images = {(r.get('source_dataset'), r['source_id']): r['split']
              for r in previous if r.get('source_id') is not None}
    for row in current:
        if row['vehicle_id'] in assignments and assignments[row['vehicle_id']] != row['split']:
            raise ValueError('warm-start group changed split; evaluation would be contaminated')
        identity = (row.get('source_dataset'), row.get('source_id'))
        if identity in images and images[identity] != row['split']:
            raise ValueError('warm-start image changed split; evaluation would be contaminated')
    previous_hashes = {r['sha256'] for r in previous if r.get('sha256')}
    current_hashes = {r['sha256'] for r in current if r.get('sha256')}
    if not previous_hashes.issubset(current_hashes):
        raise ValueError('warm start must retain parent dataset images; start from base weights when removing old data')


def verify_parent_dataset(rows, manifest):
    if hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest() != manifest['dataset_id']:
        raise ValueError('parent dataset changed after training')


def verify_clip_source(root):
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('model_id') != 'openai/clip-vit-base-patch32'
            or manifest.get('revision') != '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'):
        raise ValueError('unexpected pretrained CLIP source')
    for path in root.iterdir():
        if path.name == 'manifest.json' or path.suffix not in {'.json', '.txt', '.bin', '.safetensors'}:
            continue
        relative = path.resolve().relative_to(root.parent.resolve()).as_posix()
        expected = manifest.get('sha256', {}).get(relative)
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if not expected or actual != expected:
            raise ValueError('pretrained CLIP source checksum mismatch')
    return hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()


def verify_dino_source(root):
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('model_id') != 'facebook/dinov2-small'
            or manifest.get('revision') != 'ed25f3a31f01632728cabb09d1542f84ab7b0056'
            or manifest.get('license') != 'Apache-2.0'):
        raise ValueError('unexpected DINO source')
    expected = manifest.get('sha256', {})
    for name in ('config.json', 'model.safetensors', 'preprocessor_config.json'):
        with (root/name).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected.get(name):
            raise ValueError('DINO source checksum mismatch')
    return hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()


def finetune(model, split_rows, root, class_weights, epochs, batch_size, flip,
             balanced=False, backbone_lr=.0001, head_lr=.001):
    for block in list(model.features.children())[-4:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.features.parameters() if p.requires_grad], 'lr': backbone_lr},
        {'params': model.classifier.parameters(), 'lr': head_lr}], weight_decay=.001)
    loss_fn = nn.CrossEntropyLoss(weight=None if balanced else class_weights)
    jitter = transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.1)
    cache = {}
    for split in ('train', 'calibration'):
        rows = split_rows[split]
        for row in rows:
            image, _ = decode((root/row['path']).read_bytes(), 24_000_000)
            try:
                cache[row['path']] = image.copy()
            finally:
                image.close()
    calibration = torch.stack([model.prepare_image(cache[r['path']]) for r in split_rows['calibration']])
    cy = torch.tensor([LABELS.index(r['label']) for r in split_rows['calibration']])
    with torch.no_grad():
        best_loss = macro_cross_entropy(model(calibration), cy).item()
    best, best_epoch, history = copy.deepcopy(model.state_dict()), 0, []
    try:
        rows = split_rows['train']
        sampling = balanced_sampling_weights(rows) if balanced else None
        for epoch in range(1, epochs+1):
            model.train()
            # Small batches must not rewrite pretrained BN running statistics.
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
            order = (torch.multinomial(sampling, len(rows), replacement=True).tolist()
                     if balanced else torch.randperm(len(rows)).tolist())
            total = 0.
            for offset in range(0, len(order), batch_size):
                inputs, targets = [], []
                for index in order[offset:offset+batch_size]:
                    row = rows[index]
                    image = jitter(cache[row['path']])
                    label = row['label']
                    try:
                        if flip and torch.rand(()).item() < .5:
                            mirrored = ImageOps.mirror(image)
                            image.close()
                            image = mirrored
                            label = FLIPPED[label]
                        inputs.append(model.prepare_image(image))
                        targets.append(LABELS.index(label))
                    finally:
                        image.close()
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(torch.stack(inputs)), torch.tensor(targets))
                loss.backward()
                optimizer.step()
                total += loss.item()*len(inputs)
            model.eval()
            with torch.no_grad():
                cal_loss = macro_cross_entropy(model(calibration), cy).item()
            if cal_loss < best_loss:
                best_loss, best_epoch, best = cal_loss, epoch, copy.deepcopy(model.state_dict())
            history.append(dict(phase='finetune', epoch=epoch, loss=total/len(rows), calibration_loss=cal_loss))
            print(json.dumps(history[-1]), flush=True)
        model.load_state_dict(best)
        return history, best_epoch
    finally:
        for image in cache.values():
            image.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, default=Path('models/view-classifier'))
    parser.add_argument('--report', type=Path, default=Path('reports/supervised-training.json'))
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--finetune-epochs', type=int, default=0)
    parser.add_argument('--initialize-from', type=Path)
    parser.add_argument('--balanced-finetune', action='store_true')
    parser.add_argument('--backbone-lr', type=float, default=.0001)
    parser.add_argument('--head-lr', type=float, default=.001)
    parser.add_argument('--backbone', choices=['mobilenet', 'clip', 'fused', 'dino', 'triple'], default='mobilenet')
    parser.add_argument('--dino-dir', type=Path, default=Path('models/dinov2-small-apache'))
    parser.add_argument('--clip-dir', type=Path, default=Path('models/clip'))
    parser.add_argument('--spatial-from', type=Path)
    parser.add_argument('--weak-label-weight', type=float, default=1.)
    parser.add_argument('--macro-head-selection', action='store_true')
    parser.add_argument('--appearance-augment', action='store_true',
                        help='Two fixed lighting/color variants on reviewed natural training photos only')
    parser.add_argument('--defer-test', action='store_true',
                        help='Do not run test inference; evaluate the selected checkpoint separately')
    parser.add_argument('--class-weight-power', type=float, default=.5,
                        help='0: uniform loss, .5: original square-root weighting, 1: equal effective class mass')
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1 or args.batch_size < 1 or args.finetune_epochs < 0:
        parser.error('epochs, threads, batch size must be positive')
    if args.output.exists():
        parser.error('choose a new output directory; trained artifacts are immutable')
    if args.backbone_lr <= 0 or args.head_lr <= 0:
        parser.error('learning rates must be positive')
    if not 0 <= args.weak_label_weight <= 1:
        parser.error('--weak-label-weight must be in [0,1]')
    if not 0 <= args.class_weight_power <= 1:
        parser.error('--class-weight-power must be in [0,1]')
    if args.initialize_from and not args.finetune_epochs:
        parser.error('--initialize-from requires --finetune-epochs')
    if args.appearance_augment and args.finetune_epochs:
        parser.error('--appearance-augment currently supports frozen head training only')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    rows = load_manifest(args.manifest, LABELS, LABELS[:-1])
    counts = check_rows(rows, args.allow_partial)
    effective_counts = effective_head_counts([r for r in rows if r['split']=='train'],
        args.flip,args.weak_label_weight,args.allow_partial,appearance_augment=args.appearance_augment)
    print(json.dumps({'independent_groups': counts}), flush=True)
    started = time.perf_counter()
    torch.hub.set_dir('models/torch-cache')
    initial_version = None
    backbone_manifest_hash = None
    dino_manifest_hash = None
    if args.initialize_from:
        prior_manifest = json.loads((args.initialize_from/'manifest.json').read_text(encoding='utf-8'))
        previous_rows = load_manifest(Path(prior_manifest['dataset']), LABELS, LABELS[:-1])
        verify_parent_dataset(previous_rows,prior_manifest)
        check_warm_start_splits(previous_rows, rows)
        vision = TrainedVision(Settings(_env_file=None, cv_backend='trained',
            trained_model_dir=args.initialize_from, cpu_threads=args.threads), memory_efficient=False)
        model, initial_version = vision.model, vision.version
    else:
        if args.backbone in {'clip', 'fused', 'triple'}:
            backbone_manifest_hash = verify_clip_source(args.clip_dir)
        if args.backbone in {'dino', 'triple'}:
            dino_manifest_hash = verify_dino_source(args.dino_dir)
        if args.backbone in {'fused', 'triple'}:
            if not args.spatial_from:
                parser.error('fused encoder requires --spatial-from')
            prior_manifest = json.loads((args.spatial_from/'manifest.json').read_text(encoding='utf-8'))
            previous_rows = load_manifest(Path(prior_manifest['dataset']), LABELS, LABELS[:-1])
            verify_parent_dataset(previous_rows,prior_manifest)
            check_warm_start_splits(previous_rows, rows)
            vision = TrainedVision(Settings(_env_file=None, cv_backend='trained',
                trained_model_dir=args.spatial_from, cpu_threads=args.threads), memory_efficient=False)
            if not isinstance(vision.model, AngleNet):
                parser.error('--spatial-from must be a MobileNet artifact')
            model = FusedAngleNet(weights=args.clip_dir, mobile=vision.model).eval()
            if args.backbone == 'triple':
                model = TripleAngleNet(fused=model, dino=DinoAngleNet(weights=args.dino_dir)).eval()
            initial_version = vision.version
        elif args.backbone == 'dino':
            model = DinoAngleNet(weights=args.dino_dir).eval()
        else:
            model = (ClipAngleNet(weights=args.clip_dir) if args.backbone == 'clip'
                     else AngleNet(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)).eval()
    if isinstance(model, (ClipAngleNet, FusedAngleNet, DinoAngleNet)) and args.finetune_epochs:
        parser.error('CLIP/fused spatial encoders currently support frozen-feature head training only')
    for parameter in model.features.parameters():
        parameter.requires_grad_(False)
    split_rows = {s: [r for r in rows if r['split'] == s] for s in counts}
    extracted = extract_splits(model, split_rows, args.manifest.parent, args.flip,
                               args.batch_size, args.defer_test, args.appearance_augment)
    # Clones leave inference tensor mode so autograd may save feature inputs.
    extracted = {s: (x.clone(), y) for s, (x, y) in extracted.items()}
    x, y = extracted['train']
    sample_weights = head_sample_weights(split_rows['train'], args.flip, args.weak_label_weight,
                                         args.appearance_augment)
    class_weights = head_class_weights(y, sample_weights, args.class_weight_power)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=.03, weight_decay=.05)
    best_loss, best_epoch, best = float('inf'), 0, None
    history = []
    for epoch in range(1, (0 if args.initialize_from else args.epochs) + 1):
        model.classifier.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model.classifier(x)
        if args.weak_label_weight == 1:
            loss = loss_fn(logits, y)
        else:
            losses = nn.functional.cross_entropy(logits, y, weight=class_weights, reduction='none')
            loss = (losses*sample_weights).sum()/(class_weights[y]*sample_weights).sum()
        loss.backward()
        optimizer.step()
        model.classifier.eval()
        with torch.no_grad():
            cx, cy = extracted['calibration']
            cal_logits = model.classifier(cx)
            cal_loss = (macro_cross_entropy(cal_logits, cy) if args.macro_head_selection
                        else nn.functional.cross_entropy(cal_logits, cy)).item()
        if cal_loss < best_loss:
            best_loss, best_epoch = cal_loss, epoch
            best = copy.deepcopy(model.classifier.state_dict())
        if epoch == 1 or epoch % 25 == 0:
            history.append(dict(epoch=epoch, loss=loss.item(), calibration_loss=cal_loss))
            print(json.dumps(history[-1]), flush=True)
    if best is not None:
        model.classifier.load_state_dict(best)
    finetune_best_epoch = None
    if args.finetune_epochs:
        extra_history, finetune_best_epoch = finetune(model, split_rows, args.manifest.parent,
            class_weights, args.finetune_epochs, args.batch_size, args.flip,
            args.balanced_finetune, args.backbone_lr, args.head_lr)
        history.extend(extra_history)
        extracted = extract_splits(model, split_rows, args.manifest.parent, args.flip,
                                   args.batch_size, args.defer_test)
    outputs = {}
    with torch.no_grad():
        for split, (sx, sy) in extracted.items():
            scores = model.classifier(sx).softmax(1)
            labels = scores.argmax(1).tolist()
            outputs[split] = dict(metrics=metrics([LABELS[int(i)] for i in sy],
                [LABELS[i] for i in labels], list(LABELS)))
            if split != 'train':
                outputs[split]['predictions'] = [dict(path=row['path'], label=row['label'],
                    candidate=LABELS[index], score=float(score[index]), vehicle_id=row['vehicle_id'])
                    for row, index, score in zip(split_rows[split], labels, scores)]
    args.output.mkdir(parents=True, exist_ok=False)
    weight_path = args.output / 'weights.pt'
    torch.save(model.state_dict(), weight_path)
    with weight_path.open('rb') as stream:
        weight_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    dataset_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    manifest = dict(schema_version=1, architecture=model.architecture, labels=list(LABELS),
        appearance_augmentation=('fixed_train_natural_lighting_color_v1' if args.appearance_augment else None),
        effective_head_training_counts=effective_counts,
        backbone_manifest_sha256=backbone_manifest_hash,
        dino_manifest_sha256=dino_manifest_hash,
        initialized_from=initial_version,
        sha256=weight_hash, dataset_id=dataset_hash, dataset=str(args.manifest),
        seed=args.seed, best_epoch=best_epoch, input_size=224, preprocessing=model.preprocessing,
        finetune_best_epoch=finetune_best_epoch,
        backbone=('facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056'
                  if isinstance(model, DinoAngleNet) else
                  'mobilenet-v7+clip-vit-b32+dinov2-small' if isinstance(model, TripleAngleNet) else
                  'openai/clip-vit-base-patch32@3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
                  if isinstance(model, (ClipAngleNet, FusedAngleNet))
                  else 'torchvision/MobileNet_V3_Small_Weights.IMAGENET1K_V1'),
        calibration={'validated': False, 'classes': {}}, heldout_approval=False,
        independent_groups=counts, missing_train_classes=[c for c in LABELS if not counts['train'][c]])
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    report = dict(kind='real_supervised_training', artifact=str(args.output), manifest=manifest,
        test_evaluation_deferred=args.defer_test,
        appearance_augmentation={'enabled':args.appearance_augment,
            'eligible_photos':sum(appearance_count(r,args.appearance_augment)>1 for r in split_rows['train']),
            'variants': [{'brightness':.55,'contrast':.8,'saturation':.4},
                         {'brightness':1.2,'contrast':1.3,'saturation':.6}] if args.appearance_augment else []},
        balanced_finetune=args.balanced_finetune, backbone_lr=args.backbone_lr, head_lr=args.head_lr,
        weak_label_weight=args.weak_label_weight,
        class_weight_power=args.class_weight_power,
        checkpoint_selection='macro_calibration_cross_entropy' if args.finetune_epochs or args.macro_head_selection else 'calibration_cross_entropy',
        review_status_counts=dict(Counter(r.get('review_status') for r in rows)),
        elapsed_seconds=time.perf_counter()-started, weight_bytes=weight_path.stat().st_size,
        feature_extractor_frozen=not bool(finetune_best_epoch), spatial_features=True, flip_augmentation=args.flip,
        history=history, results=outputs, production_approved=False,
        natural_test_support=dict(Counter(r['label'] for r in rows
            if r['split'] == 'test' and not r.get('derived', False))),
        derived_test_count=sum(r.get('derived', False) for r in rows if r['split'] == 'test'),
        warning='Small/provisional data cannot establish reliable accuracy. No automatic acceptance enabled.')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Trained artifact: {args.output}; report: {args.report}; auto-acceptance remains OFF')


if __name__ == '__main__':
    main()
