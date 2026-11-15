import numpy as np
import pytest
from wastewise.common import LABELS
from wastewise.live_pipeline import LivePipeline


class Detector:
    status = 'object'
    box = (20, 20, 80, 80)
    def process(self, frame):
        return {'status': self.status, 'bbox': self.box if self.status=='object' else None,
                'roi_bbox': (15, 15, 90, 90), 'progress': 1}
    def reset(self): self.status = 'calibrating'


class Classifier:
    calls = 0
    label = 2
    status = 'accepted'
    def predict_frame(self, frame):
        self.calls += 1
        assert frame.shape[:2] == (90, 90), 'Must classify the crop, not the wall/full frame'
        probs = [.02]*4
        probs[self.label] = .94
        return dict(label=LABELS[self.label], confidence=.94, status=self.status,
                    probabilities=dict(zip(LABELS, probs)))


def test_empty_scene_never_calls_classifier():
    detector, classifier = Detector(), Classifier()
    pipeline = LivePipeline(classifier, detector)
    for status in ('calibrating','no_object','scene_changed','multiple_objects','candidate'):
        detector.status=status
        result=pipeline.process(np.zeros((240,320,3),dtype=np.uint8))
        assert result['label'] is None
        assert result.get('demo_candidate') is None
    assert classifier.calls == 0


def test_crop_confirmation_removal_and_rejection_clear_labels():
    detector, classifier = Detector(), Classifier()
    pipeline=LivePipeline(classifier, detector, stable_frames=3)
    frame=np.zeros((240,320,3),dtype=np.uint8)
    assert pipeline.process(frame)['status']=='checking'
    assert pipeline.process(frame)['status']=='checking'
    assert pipeline.process(frame)['label']=='dry_recyclable'
    classifier.status='needs_review'
    assert pipeline.process(frame)['label'] is None
    assert pipeline.last_result is None
    classifier.status='accepted'
    for _ in range(3): result=pipeline.process(frame)
    assert result['label']=='dry_recyclable'
    detector.status='no_object'
    assert pipeline.process(frame)['label'] is None
    detector.status='object'
    assert pipeline.process(frame)['status']=='checking'


def test_label_switch_requires_new_evidence():
    classifier=Classifier()
    pipeline=LivePipeline(classifier, Detector())
    frame=np.zeros((240,320,3),dtype=np.uint8)
    for _ in range(3): pipeline.process(frame)
    classifier.label=3
    assert pipeline.process(frame)['status']=='checking'
    pipeline.reset()
    assert pipeline.process(frame)['status']=='calibrating'


def test_current_weak_frame_is_not_hidden_by_previous_confidence():
    classifier=Classifier()
    pipeline=LivePipeline(classifier, Detector())
    frame=np.zeros((240,320,3),dtype=np.uint8)
    for _ in range(3): assert pipeline.process(frame)
    classifier.predict_frame=lambda crop: dict(label='dry_recyclable',confidence=.72,status='accepted',
                                               probabilities=dict(zip(LABELS,[.1,.1,.72,.08])))
    result=pipeline.process(frame)
    assert result['status']=='uncertain'
    assert result['label'] is None and pipeline.last_result is None


def test_support_only_suggestion_stays_unverified_and_clears():
    classifier=Classifier()
    normal=classifier.predict_frame
    def unsupported(crop):
        result=normal(crop)
        result.update(status='needs_review',rejection_reasons=['weak_training_support'])
        return result
    classifier.predict_frame=unsupported
    detector=Detector()
    pipeline=LivePipeline(classifier,detector)
    frame=np.zeros((240,320,3),dtype=np.uint8)
    assert pipeline.process(frame)['suggested_label'] is None
    assert pipeline.process(frame)['suggested_label'] is None
    result=pipeline.process(frame)
    assert result['suggested_label']=='dry_recyclable'
    assert result['status']=='uncertain' and result['label'] is None
    assert pipeline.last_result is None
    detector.status='no_object'
    assert pipeline.process(frame)['label'] is None
    assert not pipeline.suggestions


def test_demo_guess_does_not_loosen_strict_hardware_acceptance():
    classifier = Classifier()
    classifier.predict_frame = lambda crop: dict(status='needs_review', confidence=.61,
        rejection_reasons=['low_confidence', 'weak_training_support'],
        probabilities=dict(zip(LABELS, [.13, .13, .61, .13])))
    pipeline = LivePipeline(classifier, Detector())
    for _ in range(5):
        result = pipeline.process(np.zeros((240, 320, 3), np.uint8))
        assert result['demo_candidate'] == dict(label='dry_recyclable', confidence=.61)
        assert result['label'] is None and result['status'] == 'uncertain'
        assert pipeline.last_result is None


@pytest.mark.parametrize('probabilities,reasons', [
    ([.2, .2, .5, .1], []),
    ([.1, .1, .7, .1], ['insufficient_visual_detail']),
    ([.1, .1, float('nan'), .1], []),
    ([.1, .1, 1.2, .1], []),
])
def test_invalid_low_or_flat_predictions_have_no_demo_candidate(probabilities, reasons):
    classifier = Classifier()
    classifier.predict_frame = lambda crop: dict(status='needs_review', confidence=.7,
        rejection_reasons=reasons, probabilities=dict(zip(LABELS, probabilities)))
    result = LivePipeline(classifier, Detector()).process(np.zeros((240, 320, 3), np.uint8))
    assert result['demo_candidate'] is None
