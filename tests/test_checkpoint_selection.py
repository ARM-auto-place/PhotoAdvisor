from scripts.select_checkpoint import choose, summarize


def item(name,f1,accuracy=.9,precision=.95):
    return dict(model=name,metrics=dict(macro_f1=f1,accuracy=accuracy,positive_assignment_precision=precision))


def test_calibration_selection_rejects_precision_or_accuracy_regressions():
    baseline = item('baseline',.8)
    assert choose([baseline,item('too many false confirmations',.99,precision=.8),
        item('too many misses',.98,accuracy=.8),item('balanced improvement',.85)]) == 'balanced improvement'
    assert choose([baseline,item('tie',.8),item('lower macro',.79)]) == 'baseline'
    assert choose([baseline,item('all rejected',.99,precision=None)]) == 'baseline'


def test_positive_precision_counts_wrong_view_and_false_acceptance_of_other():
    result = summarize([dict(label='front',candidate='front'),dict(label='rear',candidate='front'),
                        dict(label='other',candidate='rear'),dict(label='front',candidate='other')])
    assert result['positive_assignment_precision'] == 1/3
    assert result['accuracy'] == .25
