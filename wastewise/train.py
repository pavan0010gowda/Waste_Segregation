"""CPU-friendly transfer learning. Model selection uses validation only."""
import argparse
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import ROOT, LABELS, read_json, save_json, digest
from .metrics import calibrate, report, temperature_scale


def load_pixels(paths, size=224):
    images = []
    for path in paths:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
            images.append(np.asarray(im, dtype=np.float32))
    return np.stack(images)


def extract(rows, cache_dir):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    cache_dir.mkdir(parents=True, exist_ok=True)
    backbone = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    vectors = []
    for start in range(0, len(rows), 32):
        batch = rows[start:start + 32]
        for row in batch:
            if digest(ROOT / row['path']) != row['sha256']:
                raise ValueError(f"Image changed since audit: {row['path']}; rerun audit before training")
        files = [cache_dir / (r['sha256'] + '.npy') for r in batch]
        missing = [i for i, f in enumerate(files) if not f.exists()]
        if missing:
            x = load_pixels([ROOT / batch[i]['path'] for i in missing])
            values = backbone(tf.keras.applications.mobilenet_v2.preprocess_input(x), training=False).numpy()
            for i, value in zip(missing, values):
                np.save(files[i], value)
        vectors.extend(np.load(f, allow_pickle=False) for f in files)
        if start % 320 == 0:
            print(f'Features: {min(start+32, len(rows))}/{len(rows)}', flush=True)
    return np.stack(vectors), backbone


def train(manifest, output):
    import tensorflow as tf
    started = time.perf_counter()
    rows = [r for r in read_json(manifest) if r['split'] in {'train', 'validation', 'test', 'external_test'}]
    x, backbone = extract(rows, ROOT / 'artifacts/cache/mobilenet_v2_224_v1')
    y = np.array([LABELS.index(r['label']) for r in rows])
    splits = ['train', 'validation', 'test'] + (['external_test'] if any(r['split']=='external_test' for r in rows) else [])
    masks = {s: np.array([r['split'] == s for r in rows]) for s in splits}
    for s, mask in masks.items():
        if s != 'external_test' and set(y[mask]) != set(range(len(LABELS))):
            raise ValueError(f'{s} must contain every class')
    candidates, best, best_score = [], None, -1
    for c in (.001, .01, .1, 1):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight='balanced', max_iter=2000, random_state=42))
        clf.fit(x[masks['train']], y[masks['train']])
        score = f1_score(y[masks['validation']], clf.predict(x[masks['validation']]), average='macro')
        candidates.append(dict(C=c, validation_macro_f1=float(score)))
        print(candidates[-1], flush=True)
        if score > best_score:
            best, best_score = clf, score
    temperature = calibrate(y[masks['validation']], best.predict_proba(x[masks['validation']]))
    # Fold standardization and logistic regression into a portable Keras head.
    scaler, classifier = best.steps[0][1], best.steps[1][1]
    inputs = tf.keras.Input(shape=(224, 224, 3), name='rgb_0_255')
    normalized = tf.keras.layers.Rescaling(1/127.5, offset=-1)(inputs)
    features = backbone(normalized, training=False)
    head = tf.keras.layers.Dense(len(LABELS), activation='softmax', name='calibrated_head')
    model = tf.keras.Model(inputs, head(features))
    weights = classifier.coef_.T / scaler.scale_[:, None] / temperature
    bias = (classifier.intercept_ - (scaler.mean_ / scaler.scale_) @ classifier.coef_.T) / temperature
    head.set_weights([weights.astype('float32'), bias.astype('float32')])
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / 'model.keras')
    # Parity check verifies deployed preprocessing/head matches experiment probabilities.
    sample = rows[:4]
    reference = temperature_scale(best.predict_proba(x[:4]), temperature)
    actual = model(load_pixels([ROOT / r['path'] for r in sample]), training=False).numpy()
    np.testing.assert_allclose(actual, reference, atol=2e-5)
    results = {}
    for split in splits[1:]:
        mask = masks[split]
        probs = temperature_scale(best.predict_proba(x[mask]), temperature)
        results[split] = report(y[mask], probs, LABELS)
        save_json(output / f'{split}_predictions.json', [dict(path=r['path'], truth=r['label'], probabilities=p.tolist()) for r, p in zip([r for r in rows if r['split'] == split], probs)])
    results.update(selection=candidates, temperature=temperature, training_seconds=time.perf_counter()-started,
                   train_images=int(masks['train'].sum()), model_bytes=(output / 'model.keras').stat().st_size,
                   majority_baseline_accuracy=float((y[masks['test']] == np.bincount(y[masks['train']]).argmax()).mean()),
                   limitations=['Legacy test set was previously used for validation; metrics are retrospective, not an untouched final benchmark.',
                                'Validation is reused for hyperparameter selection and temperature calibration.',
                                'Softmax confidence does not detect all out-of-distribution inputs.'])
    save_json(output / 'metrics.json', results)
    save_json(output / 'metadata.json', dict(labels=LABELS, input_size=224, input_range='RGB 0..255',
              threshold=.8, model_sha256=digest(output / 'model.keras'), manifest_sha256=digest(manifest),
              method='ImageNet MobileNetV2 + balanced logistic regression + temperature scaling', seed=42))
    print(results, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, default=ROOT / 'artifacts/manifest.json')
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/runs/mobilenet')
    a = p.parse_args()
    train(a.manifest, a.output)
