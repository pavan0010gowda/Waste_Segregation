"""Optional two-stage MobileNetV2 fine-tuning; validation only selects epochs.

Uses batch 8 by default for a 4 GB GPU. Writes a separate candidate run.
"""
import argparse
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
from pathlib import Path
import numpy as np
import tensorflow as tf
from .common import ROOT, LABELS, read_json, save_json, digest
from .metrics import report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/runs/finetuned')
    p.add_argument('--manifest', type=Path, default=ROOT / 'artifacts/manifest.json')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError('Epochs and batch size must be positive')
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.utils.set_random_seed(42)
    rows = read_json(args.manifest)

    def dataset(split, shuffle=False):
        selected = [r for r in rows if r['split'] == split]
        paths = [str(ROOT / r['path']) for r in selected]
        labels = [LABELS.index(r['label']) for r in selected]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(len(paths), seed=42)

        def decode(path, label):
            # Same Pillow EXIF/RGB/resize preprocessing as deployed inference.
            from .train import load_pixels
            def load(value):
                return load_pixels([value.numpy().decode()])[0]
            image = tf.py_function(load, [path], tf.float32)
            image.set_shape((224, 224, 3))
            return image, label
        return ds.map(decode, num_parallel_calls=2).batch(args.batch_size).prefetch(1), np.array(labels)

    train, y_train = dataset('train', True)
    valid, _ = dataset('validation')
    base = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3))
    base.trainable = False
    inputs = tf.keras.Input((224, 224, 3))
    x = tf.keras.layers.RandomFlip('horizontal', seed=42)(inputs)
    x = tf.keras.layers.RandomRotation(.08, seed=42)(x)
    x = tf.keras.layers.RandomZoom(.1, seed=42)(x)
    x = tf.keras.layers.Rescaling(1/127.5, offset=-1)(x)
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(.3)(x)
    outputs = tf.keras.layers.Dense(len(LABELS), activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    weights = {i: len(y_train)/(len(LABELS)*(y_train == i).sum()) for i in range(len(LABELS))}
    args.output.mkdir(parents=True, exist_ok=True)

    def fit(epochs, rate, name):
        model.compile(tf.keras.optimizers.Adam(rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model.fit(train, validation_data=valid, epochs=epochs, class_weight=weights,
                         callbacks=[tf.keras.callbacks.ModelCheckpoint(str(args.output / f'{name}.keras'), save_best_only=True, monitor='val_loss'),
                                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
    warmup = fit(min(4, args.epochs), 1e-3, 'warmup')
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    fine = fit(args.epochs, 1e-5, 'fine')
    candidates = [('warmup', min(warmup.history['val_loss'])), ('fine', min(fine.history['val_loss']))]
    selected = min(candidates, key=lambda pair: pair[1])[0]
    model = tf.keras.models.load_model(args.output / f'{selected}.keras', compile=False)
    model.save(args.output / 'model.keras')
    results = {}
    for split in ('validation', 'test'):
        ds, y = dataset(split)
        results[split] = report(y, model.predict(ds), LABELS)
    results.update(selected=selected, warmup=warmup.history, fine=fine.history,
                   limitation='Uncalibrated fine-tuned candidate; historical test is not pristine. Do not choose candidate using test scores.')
    save_json(args.output / 'metrics.json', results)
    save_json(args.output / 'metadata.json', dict(labels=LABELS, input_size=224, threshold=.8,
              input_range='RGB 0..255', model_sha256=digest(args.output / 'model.keras'),
              manifest_sha256=digest(args.manifest), seed=42, method='Two-stage MobileNetV2 fine-tuning'))


if __name__ == '__main__':
    main()
