import io
import time
import warnings
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from .common import ROOT, read_json, digest

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000


def decode_image(data):
    if not data or len(data) > MAX_BYTES:
        raise ValueError('Upload an image smaller than 8 MiB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as im:
                if im.format not in {'JPEG', 'PNG', 'WEBP'}:
                    raise ValueError('Only JPEG, PNG and WebP images are supported.')
                if im.width * im.height > MAX_PIXELS:
                    raise ValueError('Image exceeds 16 megapixels.')
                return ImageOps.exif_transpose(im).convert('RGB').copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise ValueError('Could not decode this image safely.') from e


class Predictor:
    def __init__(self, run=ROOT / 'artifacts/runs/mobilenet'):
        import tensorflow as tf
        run = Path(run)
        self.metadata = read_json(run / 'metadata.json')
        if not 0 <= self.metadata['threshold'] <= 1:
            raise ValueError('Invalid confidence threshold.')
        if digest(run / 'model.keras') != self.metadata['model_sha256']:
            raise ValueError('Model integrity check failed.')
        self.model = tf.keras.models.load_model(run / 'model.keras', compile=False, safe_mode=True)
        if self.model.output_shape[-1] != len(self.metadata['labels']):
            raise ValueError('Model and label metadata do not match.')
        size = self.metadata['input_size']
        self._infer = tf.function(lambda x: self.model(x, training=False),
                                  input_signature=[tf.TensorSpec((None, size, size, 3), tf.float32)])

    def predict(self, image):
        start = time.perf_counter()
        size = self.metadata['input_size']
        x = np.asarray(image.convert('RGB').resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)[None]
        probs = self._infer(x).numpy()[0]
        if not np.isfinite(probs).all() or (probs < 0).any() or not np.isclose(probs.sum(), 1, atol=1e-4):
            raise ValueError('Model produced invalid probabilities.')
        index = int(probs.argmax())
        accepted = float(probs[index]) >= self.metadata['threshold']
        return dict(label=self.metadata['labels'][index], confidence=float(probs[index]),
                    status='accepted' if accepted else 'needs_review',
                    probabilities=dict(zip(self.metadata['labels'], map(float, probs))),
                    latency_ms=(time.perf_counter()-start)*1000,
                    model_sha256=self.metadata['model_sha256'])
