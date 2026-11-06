"""Combine linear heads only when their active visual encoder weights are identical."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from app.evaluation import load_manifest
from app.trained import FUSED_ARCHITECTURE, LABELS
from scripts.train_views import check_rows, check_warm_start_splits, verify_parent_dataset


def blend_states(first, second, alpha=.5):
    if not 0 < alpha < 1 or set(first) != set(second):
        raise ValueError('matching models and alpha in (0,1) required')
    inactive = ('features.clip.classifier.', 'features.mobile.classifier.')
    for key in first:
        if key.startswith('classifier.') or key.startswith(inactive):
            continue
        if not torch.equal(first[key], second[key]):
            raise ValueError('active visual encoder weights differ; linear-head blending is invalid')
    result = dict(first)
    for key in ('classifier.weight','classifier.bias'):
        result[key] = alpha*first[key]+(1-alpha)*second[key]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first',type=Path,required=True)
    parser.add_argument('--second',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a new immutable output')
    manifests = [json.loads((p/'manifest.json').read_text(encoding='utf-8')) for p in (args.first,args.second)]
    rows = load_manifest(args.dataset,LABELS,LABELS[:-1])
    check_rows(rows,False)
    hashes = {r['sha256'] for r in rows}
    states = []
    for root,manifest in zip((args.first,args.second),manifests):
        if manifest['architecture'] != FUSED_ARCHITECTURE or manifest['labels'] != list(LABELS):
            raise ValueError('requires matching fused classifiers')
        old = load_manifest(Path(manifest['dataset']),LABELS,LABELS[:-1])
        verify_parent_dataset(old,manifest)
        check_warm_start_splits(old,rows)
        if not {r['sha256'] for r in old}.issubset(hashes):
            raise ValueError('dataset must account for all parent training/development images')
        with (root/'weights.pt').open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest() != manifest['sha256']:
                raise ValueError('parent checksum mismatch')
        states.append(torch.load(root/'weights.pt',weights_only=True,map_location='cpu',mmap=True))
    state = blend_states(*states)
    args.output.mkdir(parents=True)
    target = args.output/'weights.pt'
    torch.save(state,target)
    with target.open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    manifest = dict(manifests[0], sha256=digest, dataset=str(args.dataset),
        dataset_id=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        best_epoch=None, training_method='equal_logit_head_blend_shared_encoder',
        parents=[dict(sha256=m['sha256'],weight=.5) for m in manifests],
        calibration={'validated':False,'classes':{}},heldout_approval=False)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Saved shared-encoder head blend: {args.output}; acceptance remains disabled')


if __name__ == '__main__':
    main()
