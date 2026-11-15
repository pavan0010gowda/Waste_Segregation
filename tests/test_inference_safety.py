"""Retained local inference/metric checks, independent of the archived web app."""
import io
import numpy as np
import pytest
from PIL import Image
from wastewise.inference import decode_image, MAX_BYTES
from wastewise.metrics import report, temperature_scale


def test_image_limits_and_decode():
    buffer = io.BytesIO()
    Image.new('RGB', (32, 32), 'green').save(buffer, format='PNG')
    assert decode_image(buffer.getvalue()).size == (32, 32)
    for data in (b'not an image', b'', b'x' * (MAX_BYTES + 1)):
        with pytest.raises(ValueError):
            decode_image(data)


def test_metrics_include_missed_classes_and_abstention():
    probs = np.array([[.6, .4], [.6, .4]])
    result = report([0, 1], probs, ['a', 'b'])
    assert result['accuracy'] == .5
    assert result['macro_f1'] == pytest.approx(1 / 3)
    assert result['coverage'] == 0
    assert result['accepted_accuracy'] is None
    np.testing.assert_allclose(temperature_scale(probs, 1), probs)


def test_deployed_model_integrity(tmp_path):
    pytest.importorskip('tensorflow')
    from wastewise.common import save_json
    from wastewise.inference import Predictor
    save_json(tmp_path / 'metadata.json', {'threshold': .8, 'model_sha256': 'invalid'})
    (tmp_path / 'model.keras').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='integrity'):
        Predictor(tmp_path)
