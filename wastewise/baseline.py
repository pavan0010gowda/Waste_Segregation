"""Evaluate the preserved CNN using the corrected alphabetical label mapping."""
import os
import argparse
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import tensorflow as tf
from .common import ROOT, LABELS, read_json, save_json, digest
from .train import load_pixels
from .metrics import report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=ROOT / 'models/waste_classifier.h5')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/baseline.json')
    args = parser.parse_args()
    path = args.model
    model = tf.keras.models.load_model(path, compile=False)
    if model.output_shape[-1] != len(LABELS):
        result = dict(status='not_comparable', output_classes=int(model.output_shape[-1]),
                      expected_classes=len(LABELS), model_sha256=digest(path),
                      reason='Saved legacy model has eight outputs but scripts define four labels. Original class metadata is missing; assigning labels would fabricate a baseline.')
        save_json(args.output, result)
        print(result, flush=True)
        return
    rows = [r for r in read_json(ROOT / 'artifacts/manifest.json') if r['split'] == 'test']
    predictions = []
    for start in range(0, len(rows), 32):
        x = load_pixels([ROOT / r['path'] for r in rows[start:start+32]], int(model.input_shape[1])) / 255.
        predictions.extend(model(x, training=False).numpy())
    results = report([LABELS.index(r['label']) for r in rows], np.array(predictions), LABELS)
    results['model_sha256'] = digest(path)
    results['label_mapping_assumption'] = 'Alphabetical directory order from original training script'
    results['limitation'] = 'Retrospective benchmark: original training used data/test for validation; training provenance unavailable.'
    results['model_path'] = str(path)
    save_json(args.output, results)
    print(results, flush=True)


if __name__ == '__main__':
    main()
