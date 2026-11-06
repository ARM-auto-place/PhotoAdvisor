import pytest
import torch

from app.trained import LABELS
from scripts.stabilize_head import adjusted_head, guarded_summary, select_guarded


def test_guard_does_not_trade_away_target_views_for_rejection_of_everything():
    def row(name,f1,core,other,accuracy):
        return dict(name=name,metrics=dict(macro_f1=f1,core_accuracy=core,
                    other_false_positives=other,accuracy=accuracy))
    baseline = row('base',.7,.7,4,.8)
    reject_everything = row('reject',.95,0.,0,.9)
    too_many_wrong_assignments = row('wrong',.96,.9,5,.95)
    balanced = row('balanced',.8,.8,4,.85)
    assert select_guarded([baseline,reject_everything,too_many_wrong_assignments,balanced])['name']=='balanced'
    assert select_guarded([baseline,row('tie',.7,.8,3,.9)])['name']=='base'


def test_other_adjustment_preserves_parent_weights_and_only_shifts_other_logit():
    first = {'classifier.weight':torch.randn(len(LABELS),3),'classifier.bias':torch.randn(len(LABELS))}
    second = {k:v+1 for k,v in first.items()}
    before = first['classifier.bias'].clone()
    weight,bias = adjusted_head(first,second,1.,.5)
    assert torch.equal(weight,first['classifier.weight'])
    expected = before.clone()
    expected[LABELS.index('other')] += .5
    assert torch.equal(bias,expected)
    assert torch.equal(first['classifier.bias'],before)
    with pytest.raises(ValueError):
        adjusted_head(first,second,float('nan'),0.)


def test_negative_guard_counts_wrong_assignments_not_wrong_positive_categories():
    result = guarded_summary([dict(label='other',candidate='front'),dict(label='other',candidate='other'),
        dict(label='front',candidate='rear'),dict(label='rear',candidate='rear')])
    assert result['other_false_positives']==1
    assert result['core_accuracy']==.5
    with pytest.raises(ValueError,match='both target-view'):
        guarded_summary([dict(label='other',candidate='other')])
