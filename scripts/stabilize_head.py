"""Select a shared-encoder head blend on calibration, with an explicit other-class guard."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch

from app.evaluation import load_manifest
from app.settings import Settings
from app.trained import FUSED_ARCHITECTURE, LABELS, TrainedVision
from scripts.blend_heads import blend_states
from scripts.select_checkpoint import calibration_rows, summarize
from scripts.train_views import check_rows, check_warm_start_splits, extract, verify_parent_dataset


def guarded_summary(predictions):
    result = summarize(predictions)
    core = [r for r in predictions if r['label']!='other']
    negative = [r for r in predictions if r['label']=='other']
    if not core or not negative:
        raise ValueError('both target-view and other calibration examples required')
    result.update(core_accuracy=sum(r['label']==r['candidate'] for r in core)/len(core),
        other_false_positives=sum(r['candidate']!='other' for r in negative),
        core_count=len(core),other_count=len(negative))
    return result


def select_guarded(results):
    baseline = results[0]['metrics']
    eligible = [r for r in results if r['metrics']['other_false_positives']<=baseline['other_false_positives']
        and r['metrics']['core_accuracy']>=baseline['core_accuracy']
        and r['metrics']['accuracy']>=baseline['accuracy']]
    return max(eligible,key=lambda r:r['metrics']['macro_f1'])


def adjusted_head(first, second, alpha, other_bias):
    if not 0 <= alpha <= 1 or not 0 <= other_bias <= 1:
        raise ValueError('blend and other adjustment must be in [0,1]')
    weight = alpha*first['classifier.weight']+(1-alpha)*second['classifier.weight']
    bias = alpha*first['classifier.bias']+(1-alpha)*second['classifier.bias']
    bias = bias.clone()
    bias[LABELS.index('other')] += other_bias
    return weight,bias


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--first',type=Path,required=True)
    parser.add_argument('--second',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        parser.error('choose new immutable artifact and report paths')
    torch.set_num_threads(4)
    rows = load_manifest(args.manifest,LABELS,LABELS[:-1])
    check_rows(rows,False)
    calibration = calibration_rows(args.manifest,[args.first,args.second])
    manifests,states = [],[]
    for root in (args.first,args.second):
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        if manifest['architecture']!=FUSED_ARCHITECTURE or manifest['labels']!=list(LABELS):
            raise ValueError('matching fused models required')
        previous = load_manifest(Path(manifest['dataset']),LABELS,LABELS[:-1])
        verify_parent_dataset(previous,manifest)
        check_warm_start_splits(previous,rows)
        with (root/'weights.pt').open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=manifest['sha256']:
                raise ValueError('parent weights changed')
        manifests.append(manifest)
        states.append(torch.load(root/'weights.pt',weights_only=True,map_location='cpu',mmap=True))
    # This checks every active encoder tensor before mixing any logits.
    blend_states(*states)
    vision = TrainedVision(Settings(_env_file=None,cv_backend='trained',trained_model_dir=args.first,cpu_threads=4))
    x,_ = extract(vision.model,calibration,args.manifest.parent,False,16)
    results = []
    # Baseline first means ties keep the original artifact.
    grid = [(1.,0.)]+[(a,b) for a in (0.,.25,.5,.75,1.) for b in (0.,.25,.5,1.) if (a,b)!=(1.,0.)]
    for alpha,bias in grid:
        weight,offset = adjusted_head(*states,alpha,bias)
        with torch.inference_mode():
            predicted = torch.nn.functional.linear(x,weight,offset).argmax(1).tolist()
        predictions = [dict(path=row['path'],label=row['label'],candidate=LABELS[index])
                       for row,index in zip(calibration,predicted)]
        result = dict(alpha=alpha,other_bias=bias,metrics=guarded_summary(predictions),predictions=predictions)
        results.append(result)
        print(json.dumps(dict(alpha=alpha,other_bias=bias,**{k:result['metrics'][k]
            for k in ('accuracy','macro_f1','core_accuracy','other_false_positives')})),flush=True)
    selected = select_guarded(results)
    model = str(args.first)
    improved = selected['metrics']['macro_f1']>results[0]['metrics']['macro_f1']
    if improved and selected['alpha']==0. and selected['other_bias']==0.:
        model = str(args.second)
    elif improved:
        state = dict(states[0])
        state['classifier.weight'],state['classifier.bias'] = adjusted_head(*states,selected['alpha'],selected['other_bias'])
        args.output.mkdir(parents=True)
        target = args.output/'weights.pt'
        torch.save(state,target)
        with target.open('rb') as stream:
            digest = hashlib.file_digest(stream,'sha256').hexdigest()
        manifest = dict(manifests[0],sha256=digest,dataset=str(args.manifest),
            dataset_id=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),best_epoch=None,
            training_method='calibration_guarded_shared_encoder_head_blend',other_logit_bias=selected['other_bias'],
            parents=[dict(sha256=manifests[0]['sha256'],weight=selected['alpha']),
                     dict(sha256=manifests[1]['sha256'],weight=1-selected['alpha'])],
            calibration={'validated':False,'classes':{}},heldout_approval=False)
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        model = str(args.output)
    report = dict(kind='calibration_guarded_shared_encoder_head_selection',selected_model=model,
        improved_calibration=improved,selected={k:v for k,v in selected.items() if k!='predictions'},
        manifest=str(args.manifest),manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        results=results,production_approved=False,holdout_used=False,
        note='20 calibration variants; same encoders. Keep other false positives no worse than baseline '
             'and target-view and total accuracy no lower. Multiple-try calibration is not independent release evidence.')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
    print('Selected: '+model)


if __name__=='__main__':
    main()
