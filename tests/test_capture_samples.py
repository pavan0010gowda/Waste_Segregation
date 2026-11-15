import hashlib
import json
from pathlib import Path
import uuid

import cv2
import numpy as np
import pytest
from PIL import Image

from wastewise.capture_samples import new_session_id, save_sample
from wastewise.common import LABELS


def scene(status='accepted'):
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    frame[25:65, 45:95] = (20, 50, 220)
    proposal = dict(status=status, bbox=(45, 25, 50, 40), roi_bbox=(35, 15, 70, 60))
    return frame, proposal


def test_saved_crop_metadata_and_session_group(tmp_path):
    frame, proposal = scene()
    original = frame.copy()
    session = new_session_id()
    assert uuid.UUID(session).hex == session
    first = save_sample(frame, proposal, 'wet', session, tmp_path)
    second = save_sample(frame, proposal, 'unknown', session, tmp_path)
    first_record = json.loads(first['metadata_path'].read_text())
    second_record = json.loads(second['metadata_path'].read_text())
    assert first_record['group'] == second_record['group'] == session
    assert first_record['session_id'] == session
    assert first['sample_id'] != second['sample_id']
    assert first_record['label'] == 'wet' and second_record['label'] == 'unknown'
    assert first_record['source'] == 'live_camera_crop'
    assert first_record['original_dimensions'] == dict(width=160, height=100)
    assert first_record['crop_dimensions'] == dict(width=70, height=60)
    assert first_record['roi_bbox'] == [35, 15, 70, 60]
    assert first_record['training_status'] == 'pending_curation'
    assert first_record['image_sha256'] == hashlib.sha256(first['image_path'].read_bytes()).hexdigest()
    with Image.open(first['image_path']) as crop:
        assert crop.size == (70, 60)
        assert not crop.getexif()
        red, green, blue = crop.getpixel((30, 30))
        assert red > 200 and blue < 40  # OpenCV input was BGR, JPEG remains correct RGB.
    np.testing.assert_array_equal(frame, original)
    assert not list(tmp_path.rglob('*.part'))


@pytest.mark.parametrize('status', ['candidate', 'checking', 'uncertain', 'accepted', 'object'])
def test_all_foreground_states_allow_deliberate_labeling(tmp_path, status):
    frame, proposal = scene(status)
    saved = save_sample(frame, proposal, LABELS[0], new_session_id(), tmp_path)
    assert saved['image_path'].is_file() and saved['metadata_path'].is_file()


@pytest.mark.parametrize('status', ['no_object', 'calibrating', 'multiple_objects', 'scene_changed', 'garbage'])
def test_empty_or_ambiguous_scene_saves_nothing(tmp_path, status):
    frame, proposal = scene(status)
    target = tmp_path / 'not_created'
    with pytest.raises(ValueError, match='single foreground'):
        save_sample(frame, proposal, 'wet', new_session_id(), target)
    assert not target.exists()


@pytest.mark.parametrize('label', ['../../escape', 'wet/elsewhere', '', 'WET'])
def test_labels_cannot_be_used_as_paths(tmp_path, label):
    frame, proposal = scene()
    with pytest.raises(ValueError):
        save_sample(frame, proposal, label, new_session_id(), tmp_path / 'new')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('session', ['../../escape', '', 'camera-session', 7])
def test_session_is_a_uuid_and_not_a_path(tmp_path, session):
    frame, proposal = scene()
    with pytest.raises(ValueError, match='UUID'):
        save_sample(frame, proposal, 'wet', session, tmp_path / 'new')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('field,box', [
    ('bbox', None), ('roi_bbox', None), ('roi_bbox', (-1, 15, 70, 60)),
    ('roi_bbox', (35, 15, 170, 60)), ('roi_bbox', (35, 15, 0, 60)),
    ('bbox', (45.0, 25, 50, 40)), ('bbox', (True, 25, 50, 40)),
    ('roi_bbox', (45, 25, 30, 30)),
])
def test_invalid_or_unbounded_crop_saves_nothing(tmp_path, field, box):
    frame, proposal = scene()
    proposal[field] = box
    with pytest.raises(ValueError):
        save_sample(frame, proposal, 'wet', new_session_id(), tmp_path / 'new')
    assert not list(tmp_path.iterdir())


def test_invalid_camera_array_saves_nothing(tmp_path):
    _, proposal = scene()
    for frame in (np.zeros((100, 160)), np.zeros((100, 160, 3), dtype=float), np.zeros((0, 160, 3), dtype=np.uint8)):
        with pytest.raises(ValueError):
            save_sample(frame, proposal, 'wet', new_session_id(), tmp_path / 'new')
    assert not list(tmp_path.iterdir())


def test_sidecar_failure_does_not_leave_a_manifest_or_orphan_image(tmp_path, monkeypatch):
    frame, proposal = scene()
    original_rename = Path.rename
    def fail_sidecar(self, target):
        if str(self).endswith('.json.part'):
            raise OSError('simulated filesystem failure')
        return original_rename(self, target)
    monkeypatch.setattr(Path, 'rename', fail_sidecar)
    with pytest.raises(OSError, match='simulated'):
        save_sample(frame, proposal, 'wet', new_session_id(), tmp_path)
    assert not [p for p in tmp_path.rglob('*') if p.is_file()]


def test_identifier_collision_does_not_overwrite_previous_capture(tmp_path, monkeypatch):
    frame, proposal = scene()
    session = new_session_id()
    same_id = uuid.uuid4()
    monkeypatch.setattr('wastewise.capture_samples.uuid.uuid4', lambda: same_id)
    first = save_sample(frame, proposal, 'wet', session, tmp_path)
    image = first['image_path'].read_bytes()
    record = first['metadata_path'].read_bytes()
    with pytest.raises(FileExistsError, match='collision'):
        save_sample(np.zeros_like(frame), proposal, 'wet', session, tmp_path)
    assert first['image_path'].read_bytes() == image
    assert first['metadata_path'].read_bytes() == record
