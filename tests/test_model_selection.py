"""Deployment must use common validation data, not tempting test scores."""
import pytest
from wastewise.common import LABELS, save_json, read_json, digest
from wastewise import select_live


def metrics(validation, accuracy):
    return dict(selected='test-head', validation={'macro_f1':validation}, test={
        'accuracy':accuracy,'macro_f1':accuracy,'n':4,'selective_coverage':1.,
        'selective_accepted_accuracy':accuracy,
        'selective_per_class':{label:dict(n=1,accepted=1,correct_accepted=1) for label in LABELS}})


def bundle(root, name, values, manifest):
    path=root/'artifacts/runs'/name
    path.mkdir(parents=True)
    for filename in ('head.npz','features.keras'):
        (path/filename).write_bytes(name.encode())
    metadata=dict(selected='test-head',manifest_sha256=manifest,
                  sha256={f:digest(path/f) for f in ('head.npz','features.keras')})
    save_json(path/'metadata.json',metadata)
    save_json(path/'metrics.json',values)
    return path,metadata


@pytest.mark.parametrize('new_validation,new_test,expected',[
    (.94,.99,'selective_efficientnet'), (.96,.80,'selective_organic'),
    (.95,.99,'selective_efficientnet'),
])
def test_promotion_uses_same_validation_not_test(tmp_path,monkeypatch,new_validation,new_test,expected):
    monkeypatch.setattr(select_live,'ROOT',tmp_path)
    (tmp_path/'docs').mkdir()
    bundle(tmp_path,'selective',metrics(.98,.8),'old-manifest')
    _,base_meta=bundle(tmp_path,'selective_efficientnet',metrics(.99,.9),'old-manifest')
    expanded=metrics(new_validation,new_test)
    expanded['frozen_baseline']={'validation':{'macro_f1':.95}}
    path,_=bundle(tmp_path,'selective_organic',expanded,'new-manifest')
    save_json(path/'selection_decision.json',dict(baseline_head_sha256=base_meta['sha256']['head.npz'],
              manifest_sha256='new-manifest',new_validation_macro_f1=new_validation,baseline_validation_macro_f1=.95))
    save_json(tmp_path/'artifacts/baseline_mobilenet.json',{'accuracy':.8,'macro_f1':.7})
    save_json(tmp_path/'artifacts/runs/mobilenet/metrics.json',metrics(.9,.84))
    select_live.main()
    selected=read_json(tmp_path/'artifacts/live_model.json')
    assert selected['run']==f'artifacts/runs/{expected}'
    assert selected['comparison_manifest_sha256']=='new-manifest'


@pytest.mark.parametrize('everyday_validation,expected',[(.93,'selective_everyday'),(.91,'selective_organic'),(.92,'selective_organic')])
def test_everyday_expansion_compares_with_frozen_organic_baseline(tmp_path,monkeypatch,everyday_validation,expected):
    monkeypatch.setattr(select_live,'ROOT',tmp_path)
    (tmp_path/'docs').mkdir()
    bundle(tmp_path,'selective',metrics(.90,.8),'old-manifest')
    _,base_meta=bundle(tmp_path,'selective_efficientnet',metrics(.91,.9),'old-manifest')
    organic=metrics(.96,.95)
    organic['frozen_baseline']={'validation':{'macro_f1':.95}}
    organic_path,organic_meta=bundle(tmp_path,'selective_organic',organic,'organic-manifest')
    save_json(organic_path/'selection_decision.json',dict(baseline_head_sha256=base_meta['sha256']['head.npz'],
              manifest_sha256='organic-manifest',new_validation_macro_f1=.96,baseline_validation_macro_f1=.95))
    held=dict(n=8,accuracy=.75,macro_f1_present_classes=.74,represented_classes=LABELS[:3])
    everyday=metrics(everyday_validation,.80)
    everyday.update(everyday_holdout=held,frozen_baseline=dict(validation={'macro_f1':.92},everyday_holdout=held,test=organic['test']))
    everyday_path,_=bundle(tmp_path,'selective_everyday',everyday,'everyday-manifest')
    save_json(everyday_path/'selection_decision.json',dict(baseline_head_sha256=organic_meta['sha256']['head.npz'],
              manifest_sha256='everyday-manifest',new_validation_macro_f1=everyday_validation,baseline_validation_macro_f1=.92))
    save_json(tmp_path/'artifacts/everyday_sources.json',dict(retained=20,splits=dict(train={'wet':10},validation={'wet':2})))
    save_json(tmp_path/'artifacts/baseline_mobilenet.json',{'accuracy':.8,'macro_f1':.7})
    save_json(tmp_path/'artifacts/runs/mobilenet/metrics.json',metrics(.9,.84))
    select_live.main()
    selected=read_json(tmp_path/'artifacts/live_model.json')
    assert selected['run']==f'artifacts/runs/{expected}'
    assert selected['comparison_manifest_sha256']=='everyday-manifest'
    assert selected['previous_run']=='artifacts/runs/selective_organic'
