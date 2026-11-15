import cv2
import numpy as np
import pytest
from wastewise.foreground import ForegroundDetector


def wall(value=220):
    return np.full((240, 320, 3), value, np.uint8)


def ready(background=None):
    detector = ForegroundDetector(calibration_frames=4, confirm_frames=3)
    background = wall() if background is None else background
    for _ in range(4):
        result = detector.process(background)
    assert result['status'] == 'no_object'
    return detector


def object_frame(background=None):
    frame = wall() if background is None else background.copy()
    cv2.rectangle(frame, (100, 65), (185, 185), (25, 85, 170), -1)
    return frame


def test_blank_wall_never_gets_a_box():
    detector = ready()
    for _ in range(50):
        result = detector.process(wall())
        assert result['status'] == 'no_object'
        assert result['bbox'] is None


def test_stationary_object_is_confirmed_and_stays_detected_then_clears():
    detector = ready()
    for expected in ('candidate', 'candidate', 'object'):
        result = detector.process(object_frame())
        assert result['status'] == expected
    x, y, w, h = result['bbox']
    assert abs(x-100) <= 3 and abs(y-65) <= 3
    assert abs(w-86) <= 6 and abs(h-121) <= 6
    rx, ry, rw, rh = result['roi_bbox']
    assert rx < x and ry < y and rw > w and rh > h
    for _ in range(100):
        assert detector.process(object_frame())['status'] == 'object'
    assert detector.process(wall())['status'] == 'no_object'


def test_global_brightness_and_white_balance_do_not_create_an_object():
    detector = ready()
    for color in ((180, 180, 180), (245, 245, 245), (195, 210, 225)):
        frame = np.full((240, 320, 3), color, np.uint8)
        assert detector.process(frame)['status'] == 'no_object'


def test_noise_and_soft_shadow_are_ignored():
    detector = ready()
    rng = np.random.default_rng(42)
    noisy = np.clip(wall().astype(float)+rng.normal(0, 5, (240, 320, 3)), 0, 255).astype(np.uint8)
    assert detector.process(noisy)['status'] == 'no_object'
    shadow = wall()
    cv2.rectangle(shadow, (80, 50), (220, 190), (180, 180, 180), -1)
    assert detector.process(shadow)['status'] == 'no_object'


def test_multiple_objects_do_not_get_classified_as_one():
    detector = ready()
    frame = wall()
    cv2.rectangle(frame, (30, 60), (100, 180), (10, 80, 160), -1)
    cv2.rectangle(frame, (210, 60), (280, 180), (150, 40, 20), -1)
    assert detector.process(frame)['status'] == 'multiple_objects'


def test_camera_displacement_requires_reset():
    rng = np.random.default_rng(8)
    background = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    background = cv2.resize(cv2.resize(background, (40, 30)), (320, 240))
    detector = ready(background)
    shifted = np.roll(background, 15, axis=1)
    assert detector.process(shifted)['status'] == 'scene_changed'
    assert detector.process(background)['status'] == 'scene_changed'
    detector.reset()
    assert detector.process(background)['status'] == 'calibrating'


def test_movement_during_calibration_restarts_stability_count():
    detector = ForegroundDetector(calibration_frames=4)
    detector.process(wall())
    detector.process(wall())
    result = detector.process(object_frame())
    assert result['status'] == 'calibrating'
    assert result['progress'] == .25
    assert not detector.calibrated


def test_frame_resize_is_a_scene_change_and_bad_input_is_rejected():
    detector = ready()
    assert detector.process(np.full((120, 160, 3), 220, np.uint8))['status'] == 'scene_changed'
    with pytest.raises(ValueError):
        detector.process(np.zeros((10, 10)))


def test_object_on_textured_background_is_not_mistaken_for_camera_motion():
    rng = np.random.default_rng(9)
    background = rng.integers(20, 240, (30, 40, 3), dtype=np.uint8)
    background = cv2.resize(background, (320, 240))
    detector = ready(background)
    frame = object_frame(background)
    for _ in range(3):
        result = detector.process(frame)
    assert result['status'] == 'object'
    for _ in range(12):
        assert detector.process(frame)['status'] == 'object'
    assert detector.process(background)['status'] == 'no_object'


def test_large_foreground_plane_fails_closed():
    detector = ready()
    frame = wall()
    cv2.rectangle(frame, (10, 10), (310, 230), (20, 50, 140), -1)
    result = detector.process(frame)
    assert result['status'] == 'scene_changed'
    assert result['bbox'] is None


def test_real_detector_and_temporal_pipeline_hold_and_clear_without_stale_labels():
    from wastewise.common import LABELS
    from wastewise.live_pipeline import LivePipeline

    class FakeClassifier:
        calls = 0
        def predict_frame(self, crop):
            self.calls += 1
            # Different physical objects in the same location have different color
            # here, allowing us to check replacement without involving model accuracy.
            index = 2 if crop[:, :, 2].mean() > crop[:, :, 0].mean() else 3
            probabilities = [.02]*4
            probabilities[index] = .94
            return dict(status='accepted', confidence=.94, label=LABELS[index],
                        probabilities=dict(zip(LABELS, probabilities)))

    classifier = FakeClassifier()
    pipeline = LivePipeline(classifier, ForegroundDetector(calibration_frames=4, confirm_frames=3))
    for _ in range(12):
        result = pipeline.process(wall())
        assert result['label'] is None
    assert classifier.calls == 0
    for _ in range(5):
        result = pipeline.process(object_frame())
    assert result['label'] == 'dry_recyclable'
    for _ in range(12):
        assert pipeline.process(object_frame())['label'] == 'dry_recyclable'
    replacement = wall()
    cv2.rectangle(replacement, (100, 65), (185, 185), (170, 85, 25), -1)
    assert pipeline.process(replacement)['label'] is None
    assert pipeline.process(replacement)['label'] is None
    assert pipeline.process(replacement)['label'] == 'wet'
    count = classifier.calls
    assert pipeline.process(wall())['label'] is None
    assert classifier.calls == count
    assert pipeline.last_result is None
