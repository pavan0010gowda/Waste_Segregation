"""Portable feature heads and selective camera prediction (no pickle loading)."""
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .common import ROOT, LABELS, read_json, digest


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def softmax(scores, temperature=1.):
    if not np.isfinite(temperature) or temperature<=0 or not np.isfinite(scores).all():
        raise ValueError('Nonfinite scores or invalid temperature')
    z = np.asarray(scores, dtype=np.float64) / temperature
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def head_scores(x, arrays):
    kind = str(arrays['kind'].item())
    if kind == 'linear':
        return ((x - arrays['mean']) / arrays['scale']) @ arrays['coef'].T + arrays['intercept']
    if kind == 'rbf':
        x = normalize(x)
        result = []
        for i in range(len(LABELS)):
            support = arrays[f'support_{i}']
            squared = np.maximum((x*x).sum(1)[:, None] + (support*support).sum(1)[None] - 2*x@support.T, 0)
            kernel = np.exp(-float(arrays[f'gamma_{i}']) * squared)
            result.append(kernel @ arrays[f'dual_{i}'] + arrays[f'intercept_{i}'])
        return np.stack(result, axis=1)
    if kind == 'neighbors':
        similarities = normalize(x) @ arrays['bank'].T
        k = int(arrays['k'])
        nearest = np.argpartition(-similarities, k-1, axis=1)[:, :k]
        distance = np.maximum(1 - np.take_along_axis(similarities, nearest, axis=1), 1e-5)
        weights = 1 / distance
        scores = np.zeros((len(x), len(LABELS)), dtype=np.float64)
        for label in range(len(LABELS)):
            scores[:, label] = (weights * (arrays['bank_labels'][nearest] == label)).sum(1)
        return np.log(np.maximum(scores / scores.sum(1, keepdims=True), 1e-9))
    raise ValueError('Unknown safe feature-head format')


def support_scores(x, predictions, bank, bank_labels):
    similarities = normalize(x) @ bank.T
    scores = np.empty(len(x), dtype=np.float64)
    for label in range(len(LABELS)):
        selected = predictions == label
        if selected.any():
            scores[selected] = similarities[selected][:, bank_labels == label].max(1)
    return scores


def acceptance(probabilities, support, thresholds, confidence_threshold=.8, margin_threshold=.15):
    pred = probabilities.argmax(1)
    ordered = np.sort(probabilities, axis=1)
    return ((ordered[:, -1] >= confidence_threshold) & (ordered[:, -1] - ordered[:, -2] >= margin_threshold)
            & (support >= np.asarray(thresholds)[pred]))


def validate_metadata(metadata):
    if metadata.get('labels')!=LABELS or metadata.get('input_size')!=224:
        raise ValueError('Selective model input or class metadata does not match')
    temperature=metadata.get('temperature',0)
    if not np.isfinite(temperature) or temperature<=0:
        raise ValueError('Invalid calibration temperature')
    for name in ('confidence_threshold','margin_threshold'):
        value=metadata.get(name,-1)
        if not np.isfinite(value) or not 0<=value<=1:
            raise ValueError('Invalid acceptance threshold')
    thresholds=np.asarray(metadata.get('support_thresholds',[]))
    if thresholds.shape!=(4,) or not np.isfinite(thresholds).all() or (np.abs(thresholds)>1.00001).any():
        raise ValueError('Invalid embedding-support thresholds')


def validate_arrays(arrays):
    for name,value in arrays.items():
        if name!='kind' and (not np.issubdtype(value.dtype,np.number) or not np.isfinite(value).all()):
            raise ValueError('Invalid feature-head array')
    bank,labels=arrays['reference_bank'],arrays['reference_labels']
    if bank.ndim!=2 or bank.shape[1]!=1280 or labels.shape!=(len(bank),) or set(labels.tolist())!=set(range(4)):
        raise ValueError('Invalid reference-bank dimensions or labels')
    kind=str(arrays['kind'].item())
    if kind=='linear':
        if arrays['mean'].shape!=(1280,) or arrays['scale'].shape!=(1280,) or (arrays['scale']<=0).any() or arrays['coef'].shape!=(4,1280) or arrays['intercept'].shape!=(4,):
            raise ValueError('Invalid linear head dimensions')
    elif kind=='rbf':
        for i in range(4):
            support=arrays[f'support_{i}']
            if support.ndim!=2 or support.shape[1]!=1280 or arrays[f'dual_{i}'].shape!=(len(support),) or float(arrays[f'gamma_{i}'])<=0:
                raise ValueError('Invalid RBF head dimensions')
    elif kind=='neighbors':
        if arrays['bank'].ndim!=2 or arrays['bank'].shape[1]!=1280 or arrays['bank_labels'].shape!=(len(arrays['bank']),) or not 1<=int(arrays['k'])<=len(arrays['bank']):
            raise ValueError('Invalid neighbor-head dimensions')
    else:
        raise ValueError('Unknown safe feature-head format')


class SelectivePredictor:
    def __init__(self, run=ROOT / 'artifacts/runs/selective'):
        import tensorflow as tf
        run = Path(run)
        self.metadata = read_json(run / 'metadata.json')
        validate_metadata(self.metadata)
        for filename in ('features.keras', 'head.npz'):
            if digest(run / filename) != self.metadata['sha256'][filename]:
                raise ValueError('Selective model integrity check failed')
        with np.load(run / 'head.npz', allow_pickle=False) as archive:
            self.arrays = {name: archive[name] for name in archive.files}
        validate_arrays(self.arrays)
        self.model = tf.keras.models.load_model(run / 'features.keras', compile=False, safe_mode=True)
        if self.model.output_shape[-1]!=1280 or self.model.input_shape[1:]!=(224,224,3):
            raise ValueError('Unexpected feature extractor shape')
        self._infer = tf.function(lambda x: self.model(x, training=False),
                                 input_signature=[tf.TensorSpec((None,224,224,3),tf.float32)])

    def predict(self, image):
        started = time.perf_counter()
        x = np.asarray(image.convert('RGB').resize((224,224), Image.Resampling.BILINEAR), dtype=np.float32)
        features = self._infer(x[None]).numpy()
        probs = softmax(head_scores(features, self.arrays), self.metadata['temperature'])
        predicted = probs.argmax(1)
        support = support_scores(features, predicted, self.arrays['reference_bank'], self.arrays['reference_labels'])
        if not np.isfinite(probs).all() or (probs<0).any() or not np.allclose(probs.sum(1),1,atol=1e-5):
            raise ValueError('Invalid model probabilities')
        accepted = bool(acceptance(probs, support, self.metadata['support_thresholds'],self.metadata['confidence_threshold'],self.metadata['margin_threshold'])[0])
        ordered = np.sort(probs[0])
        reasons = []
        if float(x.std(axis=(0,1)).mean()) < 3.5:
            reasons.append('insufficient_visual_detail')
        if ordered[-1] < self.metadata['confidence_threshold']:
            reasons.append('low_confidence')
        if ordered[-1]-ordered[-2] < self.metadata['margin_threshold']:
            reasons.append('ambiguous_classes')
        if support[0] < self.metadata['support_thresholds'][int(predicted[0])]:
            reasons.append('weak_training_support')
        return dict(label=LABELS[int(predicted[0])], confidence=float(probs.max()),
                    status='accepted' if accepted and not reasons else 'needs_review',
                    probabilities=dict(zip(LABELS, map(float,probs[0]))),
                    support=float(support[0]), rejection_reasons=reasons,
                    latency_ms=(time.perf_counter()-started)*1000,
                    model_sha256=self.metadata['sha256']['head.npz'])
