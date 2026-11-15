import numpy as np
import pytest
from wastewise.selective import normalize,softmax,head_scores,support_scores,acceptance


def test_portable_linear_head_and_probabilities():
    x=np.array([[2.,4.]])
    arrays=dict(kind=np.array('linear'),mean=np.array([1.,2.]),scale=np.array([1.,2.]),coef=np.eye(2),intercept=np.zeros(2))
    np.testing.assert_allclose(head_scores(x,arrays),[[1.,1.]])
    np.testing.assert_allclose(softmax(head_scores(x,arrays)),[[.5,.5]])


def test_support_requires_neighbors_of_predicted_class():
    bank=normalize(np.eye(4)); labels=np.arange(4)
    x=np.array([[1,0,0,0],[1,0,0,0]])
    np.testing.assert_allclose(support_scores(x,np.array([0,1]),bank,labels),[1.,0.])


def test_rejection_combines_support_confidence_and_margin():
    probs=np.array([[.9,.04,.03,.03],[.9,.04,.03,.03],[.6,.2,.1,.1]])
    result=acceptance(probs,np.array([.9,.3,.9]),[.6]*4)
    assert result.tolist()==[True,False,False]


def test_acceptance_uses_configured_thresholds():
    probs=np.array([[.85,.05,.05,.05]])
    assert not acceptance(probs,np.array([.9]),[.6]*4,confidence_threshold=.9)[0]


def test_invalid_temperature_and_metadata_rejected():
    from wastewise.selective import validate_metadata
    from wastewise.common import LABELS
    with pytest.raises(ValueError):softmax(np.array([[1.,2.]]),0)
    with pytest.raises(ValueError):softmax(np.array([[np.inf,2.]]))
    valid=dict(labels=LABELS,input_size=224,temperature=1,confidence_threshold=.8,margin_threshold=.15,support_thresholds=[.5]*4)
    validate_metadata(valid)
    for update in ({'labels':list(reversed(LABELS))},{'temperature':float('nan')},{'confidence_threshold':1.1},{'support_thresholds':[.5]}):
        with pytest.raises(ValueError):validate_metadata({**valid,**update})


def test_training_rejects_cross_split_duplicate_groups():
    from wastewise.train_selective import validate_split_groups
    rows=[dict(group='same',sha256='a',split='train',label='wet'),dict(group='same',sha256='b',split='organic_holdout',label='wet')]
    with pytest.raises(ValueError,match='crosses splits'):
        validate_split_groups(rows)
    rows[1]['group']='different'
    validate_split_groups(rows)
