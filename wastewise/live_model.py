"""Shared crop classifier with original and retrained model choices."""
import cv2
import numpy as np
from PIL import Image
from .common import ROOT, LABELS, read_json


class LiveClassifier:
    def __init__(self, model='original'):
        self.name = model
        if model in {'selective', 'previous'}:
            from .selective import SelectivePredictor
            selection = ROOT / 'artifacts/live_model.json'
            run = ROOT / 'artifacts/runs/selective'
            if selection.exists():
                selected = read_json(selection)
                key = 'previous_run' if model == 'previous' and 'previous_run' in selected else 'run'
                run = (ROOT / selected[key]).resolve()
                if not run.is_relative_to((ROOT / 'artifacts/runs').resolve()):
                    raise ValueError('Live model selection must reference an experiment under artifacts/runs')
            self.predictor = SelectivePredictor(run)
            print(f'Using selective camera classifier: {run.name}', flush=True)
            return
        if model == 'improved':
            from .inference import Predictor
            self.predictor = Predictor()
            return
        import tensorflow as tf
        path = ROOT / 'models/waste_classifier_mobilenet.h5'
        if not path.exists():
            path = ROOT.parent / 'models/waste_classifier_mobilenet.h5'
        if not path.exists():
            raise FileNotFoundError('Original MobileNet missing: place waste_classifier_mobilenet.h5 in models/.')
        self.model = tf.keras.models.load_model(path, compile=False)
        if self.model.output_shape[-1] != len(LABELS):
            raise ValueError('This camera model must have four outputs matching the original training labels.')
        self.size = int(self.model.input_shape[1])
        self._infer = tf.function(lambda x: self.model(x, training=False),
                                 input_signature=[tf.TensorSpec((None, self.size, self.size, 3), tf.float32)])
        print(f'Using original camera model: {path}', flush=True)

    def predict_frame(self, frame):
        if self.name in {'improved', 'selective', 'previous'}:
            return self.predictor.predict(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        # Preserve the original trained model's 0..1 input convention.
        rgb = cv2.cvtColor(cv2.resize(frame, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        probabilities = self._infer(rgb.astype(np.float32)[None] / 255.).numpy()[0]
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        return dict(label=LABELS[index], confidence=confidence,
                    status='accepted' if confidence > .7 else 'needs_review',
                    probabilities=dict(zip(LABELS, map(float, probabilities))))
