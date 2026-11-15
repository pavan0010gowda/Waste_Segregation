"""Foreground-first, crop-only classification with temporal confirmation.

These are engineered safeguards around established vision methods, not a claim
of a novel research algorithm or a semantic instance detector.
"""
from collections import deque
import cv2
import numpy as np
from .common import LABELS
from .foreground import ForegroundDetector


def overlap(a, b):
    if a is None or b is None:
        return 0.
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    intersection = max(0, x2-x1)*max(0, y2-y1)
    return intersection / max(1, a[2]*a[3]+b[2]*b[3]-intersection)


class LivePipeline:
    def __init__(self, classifier, detector=None, stable_frames=3, min_confidence=.8, min_margin=.15):
        self.classifier = classifier
        self.detector = detector or ForegroundDetector()
        self.stable_frames = stable_frames
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.history = deque(maxlen=stable_frames)
        self.suggestions = deque(maxlen=stable_frames)
        self.previous_box = None
        self.last_result = None

    def reset(self):
        self.detector.reset()
        self._clear()

    def _clear(self):
        self.history.clear()
        self.suggestions.clear()
        self.previous_box = None
        self.last_result = None

    def process(self, frame):
        proposal = self.detector.process(frame)
        if proposal['status'] != 'object':
            self._clear()
            return {**proposal, 'label': None, 'confidence': None}
        bbox = proposal['bbox']
        if overlap(bbox, self.previous_box) < .25:
            self.history.clear()
            self.suggestions.clear()
        self.previous_box = bbox
        x, y, w, h = proposal.get('roi_bbox') or bbox
        crop = frame[y:y+h, x:x+w]
        result = self.classifier.predict_frame(crop)
        # Expose the raw guess separately for the explicitly enabled SOFTWARE demo.
        # This never changes the strict status/label consumed by Arduino mode.
        probabilities = np.asarray([result['probabilities'][label] for label in LABELS])
        demo_candidate = None
        if (np.isfinite(probabilities).all() and (probabilities >= 0).all()
                and (probabilities <= 1).all() and np.isclose(probabilities.sum(), 1, atol=.001)
                and probabilities.max() > .5
                and 'insufficient_visual_detail' not in result.get('rejection_reasons', [])):
            demo_candidate = {'label': LABELS[int(probabilities.argmax())],
                              'confidence': float(probabilities.max())}
        proposal = {**proposal, 'demo_candidate': demo_candidate}
        # A rejected current frame clears the old label immediately.
        if result['status'] != 'accepted':
            self.history.clear()
            self.last_result = None
            suggestion = None
            ordered = np.sort(probabilities)
            support_only = (result.get('rejection_reasons') == ['weak_training_support']
                            and ordered[-1] >= self.min_confidence
                            and ordered[-1]-ordered[-2] >= self.min_margin)
            if support_only:
                predicted = LABELS[int(probabilities.argmax())]
                if self.suggestions and self.suggestions[-1] != predicted:
                    self.suggestions.clear()
                self.suggestions.append(predicted)
                if len(self.suggestions) == self.stable_frames:
                    suggestion = predicted
            else:
                self.suggestions.clear()
            return {**proposal, 'status': 'uncertain', 'label': None,
                    'confidence': result['confidence'], 'suggested_label': suggestion,
                    'reason': ', '.join(result.get('rejection_reasons', [])) or 'Classifier uncertain'}
        self.suggestions.clear()
        ordered = np.sort(probabilities)
        if ordered[-1] < self.min_confidence or ordered[-1]-ordered[-2] < self.min_margin:
            self.history.clear()
            self.last_result = None
            return {**proposal, 'status': 'uncertain', 'label': None,
                    'confidence': float(ordered[-1]), 'reason': 'Current frame lacks confidence or margin'}
        if self.history and probabilities.argmax() != self.history[-1].argmax():
            self.history.clear()
        self.history.append(probabilities)
        mean = np.mean(self.history, axis=0)
        ranked = np.sort(mean)
        confident = ranked[-1] >= self.min_confidence and ranked[-1]-ranked[-2] >= self.min_margin
        accepted = len(self.history) == self.stable_frames and confident
        output = {**proposal, 'status': 'accepted' if accepted else 'checking',
                  'label': LABELS[int(mean.argmax())] if accepted else None,
                  'confidence': float(mean.max()), 'reason': 'Temporal confirmation'}
        self.last_result = output if accepted else None
        return output


def draw_result(frame, result):
    shown = frame.copy()
    status = result['status']
    messages = {
        'calibrating': f"Learning empty background: {result.get('progress', 0):.0%} - keep items out",
        'no_object': 'No object - show one waste item',
        'candidate': 'Object entering - hold it steady',
        'checking': 'Checking object - hold it steady',
        'uncertain': 'Uncertain / unfamiliar object - no automatic classification',
        'scene_changed': 'Camera/background changed - clear scene and press B',
        'multiple_objects': 'Show only one object at a time',
    }
    text = messages.get(status, 'Waiting for object')
    color = (90, 210, 255)
    if status == 'accepted':
        color = (100, 230, 100)
        text = f"{result['label']}: {result['confidence']:.0%}"
    elif status == 'uncertain' and result.get('suggested_label'):
        text = f"Possible {result['suggested_label']} - UNVERIFIED, check manually"
    bbox = result.get('bbox')
    if bbox is not None and status in {'accepted', 'checking', 'uncertain', 'candidate'}:
        x, y, w, h = bbox
        cv2.rectangle(shown, (x, y), (x+w, y+h), color, 2)
        if status == 'accepted':
            cv2.putText(shown, text, (x, max(18, y-8)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    cv2.rectangle(shown, (0, 0), (shown.shape[1], 65), (25, 25, 25), -1)
    # Scale banner to fit smaller webcam resolutions.
    scale = min(.65, max(.3, (shown.shape[1]-20)/max(1, len(text)*12)))
    cv2.putText(shown, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    cv2.putText(shown, 'Fixed camera | B: reset empty background | Q: quit', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .42, (210, 210, 210), 1)
    return shown
