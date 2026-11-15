"""Explicit, local collection of human-labeled foreground crops.

Nothing is captured on import or in a background task. The camera caller must
invoke ``save_sample`` in response to a deliberate label key. JSON sidecars form
the collection manifest; a sidecar is committed only after its JPEG is complete.
Samples from a camera session share a group for future session-disjoint splits.
"""
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral
from pathlib import Path
import uuid

import cv2
import numpy as np

from .common import ROOT, LABELS


CAPTURE_LABELS = tuple(LABELS) + ('unknown',)
CAPTURE_STATUSES = frozenset({'candidate', 'checking', 'uncertain', 'accepted', 'object'})


def new_session_id():
    """Create a fresh capture-session group; retain it until the camera closes."""
    return uuid.uuid4().hex


def _box(value, frame_width, frame_height, name):
    if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 4:
        raise ValueError(f'{name} must contain x, y, width and height')
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in value):
        raise ValueError(f'{name} coordinates must be integers')
    x, y, width, height = map(int, value)
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x+width > frame_width or y+height > frame_height:
        raise ValueError(f'{name} must lie completely inside the camera frame')
    return x, y, width, height


def save_sample(frame, proposal, label, session_id, directory=ROOT / 'data/camera_samples'):
    """Save one explicitly labeled padded BGR crop and its JSON manifest record.

    ``proposal`` is a ``LivePipeline.process``/``ForegroundDetector.process``
    result and must have a foreground status plus valid ``bbox`` and ``roi_bbox``.
    It is rejected for empty/calibrating/multiple-object/changed scenes. No whole
    webcam frame is retained. This function neither trains nor modifies a model.

    Returns ``{'image_path': Path, 'metadata_path': Path, 'sample_id': str}``.
    Validation errors raise ``ValueError`` before any directory or file is created.
    """
    if not isinstance(label, str) or label not in CAPTURE_LABELS:
        raise ValueError('Choose one of the four waste labels or unknown')
    if not isinstance(session_id, str):
        raise ValueError('Capture session must be a UUID string')
    try:
        session = uuid.UUID(session_id).hex
    except (ValueError, AttributeError) as exc:
        raise ValueError('Capture session must be a UUID string') from exc
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('Expected a uint8 BGR camera frame')
    frame_height, frame_width = frame.shape[:2]
    if not frame_height or not frame_width:
        raise ValueError('Camera frame cannot be empty')
    if (not isinstance(proposal, dict) or not isinstance(proposal.get('status'), str)
            or proposal['status'] not in CAPTURE_STATUSES):
        raise ValueError('A single foreground object must be visible before saving a label')
    bbox = _box(proposal.get('bbox'), frame_width, frame_height, 'bbox')
    roi = _box(proposal.get('roi_bbox'), frame_width, frame_height, 'roi_bbox')
    x, y, width, height = roi
    bx, by, bw, bh = bbox
    if not (x <= bx and y <= by and x+width >= bx+bw and y+height >= by+bh):
        raise ValueError('The padded crop must contain the detected bounding box')

    # Re-encoding a raw array retains no source EXIF or other camera metadata.
    ok, encoded = cv2.imencode('.jpg', frame[y:y+height, x:x+width], [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError('Unable to encode the detected crop')
    image_bytes = encoded.tobytes()
    root = Path(directory).resolve()
    folder = (root / label).resolve()
    if not folder.is_relative_to(root):
        raise ValueError('Sample directory resolves outside the collection folder')
    sample_id = uuid.uuid4().hex
    image_path = folder / f'{sample_id}.jpg'
    metadata_path = folder / f'{sample_id}.json'
    metadata = dict(
        schema_version=1, sample_id=sample_id, label=label,
        label_source='explicit_user_label', source='live_camera_crop',
        session_id=session, group=session,
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        foreground_status=proposal['status'], bbox=list(bbox), roi_bbox=list(roi),
        original_dimensions=dict(width=frame_width, height=frame_height),
        crop_dimensions=dict(width=width, height=height),
        image=image_path.relative_to(root).as_posix(),
        image_sha256=hashlib.sha256(image_bytes).hexdigest(),
        training_status='pending_curation',
    )
    metadata_bytes = json.dumps(metadata, indent=2, allow_nan=False).encode('utf-8')
    folder.mkdir(parents=True, exist_ok=True)
    image_temporary = folder / f'.{sample_id}.jpg.part'
    metadata_temporary = folder / f'.{sample_id}.json.part'
    created = []
    try:
        # Exclusive creation avoids silently replacing an existing file even in
        # the extremely unlikely event of a sample UUID collision.
        if image_path.exists() or metadata_path.exists():
            raise FileExistsError('Sample identifier collision; retry capture')
        for path, content in ((image_temporary, image_bytes), (metadata_temporary, metadata_bytes)):
            with path.open('xb') as stream:
                created.append(path)
                stream.write(content)
                stream.flush()
        image_temporary.rename(image_path)
        created.remove(image_temporary)
        created.append(image_path)
        # Metadata is the commit marker for manifest readers.
        metadata_temporary.rename(metadata_path)
        created.remove(metadata_temporary)
    except BaseException:
        for path in created:
            # If interruption occurs immediately after the sidecar commit, keep
            # its already published image instead of creating a dangling record.
            if path != image_path or not metadata_path.exists():
                path.unlink(missing_ok=True)
        raise
    return dict(image_path=image_path, metadata_path=metadata_path, sample_id=sample_id)
