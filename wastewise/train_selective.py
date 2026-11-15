"""Compare feature heads on validation, freeze selection, then evaluate test.

The custom integration uses established algorithms: balanced logistic regression,
one-vs-rest RBF SVM and cosine neighbors. Rejection combines calibrated confidence,
class margin and support from similar training images; it is not a certified OOD detector.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import time
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, log_loss
from .common import ROOT, LABELS, read_json, save_json, digest
from .selective import normalize, softmax, head_scores, support_scores, acceptance, validate_metadata, validate_arrays
from .metrics import report


def validate_split_groups(rows):
    seen={}
    for row in rows:
        group=row.get('group',row['sha256'])
        assigned=(row['split'],row['label'])
        if group in seen and seen[group]!=assigned:
            raise ValueError('Duplicate group crosses splits or labels')
        seen[group]=assigned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=ROOT/'artifacts/all-data-manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/runs/retrained')
    parser.add_argument('--backbone',choices=['mobilenet','efficientnet'],default='efficientnet')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--baseline-run',type=Path,help='Frozen same-backbone bundle evaluated on exactly the same expanded validation/evaluation images')
    args = parser.parse_args()
    started = time.perf_counter()
    split_names=('train','validation','test','external_test','organic_holdout','everyday_holdout')
    rows = [r for r in read_json(args.manifest) if r['split'] in split_names]
    validate_split_groups(rows)
    baseline_arrays,baseline_metadata=None,None
    if args.baseline_run:
        if args.output.resolve()==args.baseline_run.resolve():
            raise ValueError('Output must not overwrite the frozen baseline')
        baseline_metadata=read_json(args.baseline_run/'metadata.json')
        validate_metadata(baseline_metadata)
        if baseline_metadata.get('backbone')!=args.backbone:
            raise ValueError('Frozen baseline must use the same backbone for feature reuse')
        for name in ('features.keras','head.npz'):
            if digest(args.baseline_run/name)!=baseline_metadata['sha256'][name]:
                raise ValueError('Frozen baseline failed integrity validation')
        with np.load(args.baseline_run/'head.npz',allow_pickle=False) as archive:
            baseline_arrays={key:archive[key] for key in archive.files}
        validate_arrays(baseline_arrays)
    for row in rows:
        if digest(ROOT/row['path'])!=row['sha256']:
            raise ValueError(f"Image changed since audit: {row['path']}; rerun audit before training")
    cache = ROOT/('artifacts/cache/mobilenet_v2_224_v1' if args.backbone=='mobilenet' else 'artifacts/cache/efficientnet_v2_b0_224_v1')
    cache.mkdir(parents=True,exist_ok=True)
    features=None
    def build_features():
        import tensorflow as tf
        tf.keras.utils.set_random_seed(42)
        if args.baseline_run:
            return tf.keras.models.load_model(args.baseline_run/'features.keras',compile=False,safe_mode=True)
        inputs=tf.keras.Input((224,224,3))
        if args.backbone=='mobilenet':
            base=tf.keras.applications.MobileNetV2(weights='imagenet',include_top=False,pooling='avg',input_shape=(224,224,3))
            return tf.keras.Model(inputs,base(tf.keras.layers.Rescaling(1/127.5,offset=-1)(inputs),training=False))
        base=tf.keras.applications.EfficientNetV2B0(weights='imagenet',include_top=False,include_preprocessing=True,pooling='avg',input_shape=(224,224,3))
        return tf.keras.Model(inputs,base(inputs,training=False))
    def cached(row):
        path=cache/(row['sha256']+'.npy')
        try:
            value=np.load(path,allow_pickle=False)
            return value.shape==(1280,) and np.isfinite(value).all()
        except (OSError,ValueError,EOFError):
            return False
    missing=[r for r in rows if not cached(r)]
    if missing:
        from .train import load_pixels
        import tensorflow as tf
        features=build_features()
        compiled=tf.function(lambda pixels:features(pixels,training=False),input_signature=[tf.TensorSpec((None,224,224,3),tf.float32)])
        for start in range(0,len(missing),args.batch_size):
            batch=missing[start:start+args.batch_size]
            values=compiled(load_pixels([ROOT/r['path'] for r in batch])).numpy()
            for row,value in zip(batch,values):
                np.save(cache/(row['sha256']+'.npy'),value)
            if start%(args.batch_size*10)==0:
                print(f'{args.backbone} features: {min(start+args.batch_size,len(missing))}/{len(missing)}',flush=True)
    x = np.stack([np.load(cache/(r['sha256']+'.npy'),allow_pickle=False) for r in rows])
    y = np.array([LABELS.index(r['label']) for r in rows])
    masks = {name: np.array([r['split']==name for r in rows]) for name in split_names}
    xt, xv = x[masks['train']], x[masks['validation']]
    yt, yv = y[masks['train']], y[masks['validation']]
    if set(yt)!=set(range(4)) or set(yv)!=set(range(4)):
        raise ValueError('Train and validation must each contain all four classes')
    if baseline_metadata:
        baseline_validation_macro_f1=float(f1_score(yv,head_scores(xv,baseline_arrays).argmax(1),average='macro'))
        print({'frozen_baseline_same_validation_macro_f1':baseline_validation_macro_f1},flush=True)
    bank = normalize(xt)
    candidates, best_arrays, best_score = [], None, -1

    def consider(name, arrays, reference=None):
        nonlocal best_arrays,best_score
        scores=head_scores(xv,arrays)
        if reference is not None:
            np.testing.assert_allclose(scores,reference,rtol=1e-4,atol=1e-4)
        score=float(f1_score(yv,scores.argmax(1),average='macro'))
        candidates.append(dict(name=name,validation_macro_f1=score))
        print(candidates[-1],flush=True)
        if score>best_score+1e-12:
            best_arrays,best_score=arrays,score

    scaler=StandardScaler().fit(xt)
    for c in (.01,.1,1):
        clf=LogisticRegression(C=c,class_weight='balanced',max_iter=2000,random_state=42).fit(scaler.transform(xt),yt)
        arrays=dict(kind=np.array('linear'),mean=scaler.mean_,scale=scaler.scale_,coef=clf.coef_,intercept=clf.intercept_)
        consider(f'balanced_logistic_C={c}',arrays,clf.decision_function(scaler.transform(xv)))
    for c in (1,10,100):
        clf=OneVsRestClassifier(SVC(C=c,gamma='scale',class_weight='balanced',kernel='rbf')).fit(bank,yt)
        arrays=dict(kind=np.array('rbf'))
        for i,estimator in enumerate(clf.estimators_):
            arrays.update({f'support_{i}':estimator.support_vectors_,f'dual_{i}':estimator.dual_coef_[0],f'intercept_{i}':estimator.intercept_[0],f'gamma_{i}':estimator._gamma})
        consider(f'balanced_rbf_ovr_C={c}',arrays,clf.decision_function(normalize(xv)))
    for k in (3,7,15):
        consider(f'cosine_neighbors_k={k}',dict(kind=np.array('neighbors'),bank=bank,bank_labels=yt,k=np.array(k)))
    scores=head_scores(xv,best_arrays)
    temperature=float(minimize_scalar(lambda t:log_loss(yv,softmax(scores,t),labels=range(4)),bounds=(.1,10),method='bounded').x)
    val_probs=softmax(scores,temperature)
    val_pred=val_probs.argmax(1)
    val_support=support_scores(xv,val_pred,bank,yt)
    thresholds=[]
    for label in range(4):
        correct=(yv==label)&(val_pred==label)
        if not correct.any():
            raise ValueError('Insufficient correctly classified validation samples for support threshold')
        thresholds.append(float(np.quantile(val_support[correct],.05)))
    best_arrays.update(reference_bank=bank,reference_labels=yt)
    args.output.mkdir(parents=True,exist_ok=True)
    save_json(args.output/'source_integrity.json',dict(images_verified=len(rows),all_manifest_hashes_match=True,duplicate_groups_disjoint=True,manifest_sha256=digest(args.manifest)))
    np.savez_compressed(args.output/'head.npz',**best_arrays)
    selection_decision=dict(new_validation_macro_f1=best_score,manifest_sha256=digest(args.manifest),
                            selection_rule='Compare raw macro F1 on exactly the same expanded validation; no test or holdout scores used.')
    if baseline_metadata:
        selection_decision.update(baseline_validation_macro_f1=baseline_validation_macro_f1,
                                  baseline_run=str(args.baseline_run),baseline_head_sha256=baseline_metadata['sha256']['head.npz'],
                                  recommendation='new' if best_score>baseline_validation_macro_f1+1e-12 else 'retain_baseline')
    else:
        selection_decision['recommendation']='new_candidate_no_frozen_baseline_comparison'
    # Freeze the recommendation before computing any test/holdout result.
    save_json(args.output/'selection_decision.json',selection_decision)
    results=dict(selection=candidates,selected=next(c['name'] for c in candidates if c['validation_macro_f1']==best_score),
                 train_images=len(yt),temperature=temperature,support_thresholds=thresholds,selection_decision=selection_decision,
                 limitations=['Historical test was previously inspected; this is retrospective, not a pristine final benchmark.',
                              'Validation reused for selection, calibration and 5th-percentile class support thresholds.',
                              'Neighbor support and confidence are heuristic unknown-input safeguards; no real-world OOD accuracy claim.',
                              'Selective metrics here use embedding support/confidence/margin only; they exclude foreground, temporal stability and image-detail gates.',
                              'Class crops and foreground boxes need independent live-camera evaluation.'])
    def evaluate_arrays(mask,arrays,calibration,support_thresholds,confidence=.8,margin=.15):
        probs=softmax(head_scores(x[mask],arrays),calibration)
        support=support_scores(x[mask],probs.argmax(1),arrays['reference_bank'],arrays['reference_labels'])
        accepted=acceptance(probs,support,support_thresholds,confidence,margin)
        metrics=report(y[mask],probs,LABELS,threshold=confidence)
        metrics['represented_classes']=[LABELS[i] for i in np.unique(y[mask])]
        metrics['macro_f1_present_classes']=float(f1_score(y[mask],probs.argmax(1),labels=np.unique(y[mask]),average='macro',zero_division=0))
        selected_rows=[r for r,keep in zip(rows,mask) if keep]
        categories=sorted({r['source_category'] for r in selected_rows if 'source_category' in r})
        metrics['by_source_category']={}
        for category in categories:
            subset=np.array([r.get('source_category')==category for r in selected_rows])
            metrics['by_source_category'][category]=dict(n=int(subset.sum()),
                accuracy=float((probs.argmax(1)[subset]==y[mask][subset]).mean()),
                selective_coverage=float(accepted[subset].mean()))
        metrics['selective_coverage']=float(accepted.mean())
        metrics['selective_accepted_accuracy']=float((probs.argmax(1)[accepted]==y[mask][accepted]).mean()) if accepted.any() else None
        metrics['selective_per_class']={label:dict(n=int((y[mask]==i).sum()),accepted=int((accepted&(y[mask]==i)).sum()),correct_accepted=int((accepted&(y[mask]==i)&(probs.argmax(1)==i)).sum())) for i,label in enumerate(LABELS)}
        return metrics,probs,support,accepted
    if baseline_metadata:
        results['frozen_baseline']={}
    for split in split_names[1:]:
        mask=masks[split]
        if not mask.any():continue
        metrics,probs,support,accepted=evaluate_arrays(mask,best_arrays,temperature,thresholds)
        results[split]=metrics
        save_json(args.output/f'{split}_predictions.json',[dict(path=row['path'],truth=row['label'],probabilities=prob.tolist(),support=float(s),accepted=bool(a)) for row,prob,s,a in zip([r for r in rows if r['split']==split],probs,support,accepted)])
        if baseline_metadata:
            bm,bp,bs,ba=evaluate_arrays(mask,baseline_arrays,baseline_metadata['temperature'],baseline_metadata['support_thresholds'],baseline_metadata['confidence_threshold'],baseline_metadata['margin_threshold'])
            results['frozen_baseline'][split]=bm
            save_json(args.output/f'baseline_{split}_predictions.json',[dict(path=row['path'],truth=row['label'],probabilities=prob.tolist(),support=float(s),accepted=bool(a)) for row,prob,s,a in zip([r for r in rows if r['split']==split],bp,bs,ba)])
    if features is None:
        features=build_features()
    features.save(args.output/'features.keras')
    metadata=dict(labels=LABELS,input_size=224,input_range='RGB 0..255',temperature=temperature,backbone=args.backbone,
                  selected=results['selected'],support_thresholds=thresholds,confidence_threshold=.8,margin_threshold=.15,
                  sha256={name:digest(args.output/name) for name in ('head.npz','features.keras')},
                  manifest_sha256=digest(args.manifest),seed=42)
    save_json(args.output/'metadata.json',metadata)
    results['training_seconds']=time.perf_counter()-started
    results['backbone']=args.backbone
    save_json(args.output/'metrics.json',results)
    print({key:results[key] for key in ('selected','selection_decision','test','external_test','organic_holdout','everyday_holdout','training_seconds') if key in results},flush=True)


if __name__=='__main__':
    main()
