"""Foreground proposals for a stationary, single-item sorting camera.

This is classical background differencing, not a trained object detector. Calibrate
with an empty scene, keep the camera fixed, and call reset() after moving it. A hand,
person or other foreground object can also produce a proposal; the classifier must
still reject uncertain/unfamiliar items. Low-contrast objects and hard shadows can
be ambiguous. The background is fixed after calibration so a stationary item does
not disappear into an adaptive background model.
"""
import cv2
import numpy as np


class ForegroundDetector:
    def __init__(self, calibration_frames=30, confirm_frames=3,
                 min_area_ratio=.008, max_area_ratio=.60, max_width=480):
        if calibration_frames < 2 or confirm_frames < 1:
            raise ValueError('Use at least two calibration frames and one confirmation frame')
        if not 0 < min_area_ratio < max_area_ratio < 1:
            raise ValueError('Area ratios must satisfy 0 < minimum < maximum < 1')
        self.calibration_frames = calibration_frames
        self.confirm_frames = confirm_frames
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.max_width = max_width
        self.reset()

    def reset(self):
        self._samples = []
        self._previous = None
        self._background = None
        self._threshold = None
        self._gray_background = None
        self._shape = None
        self._confirm = 0
        self._last_box = None
        self._scene_changed = False

    @property
    def calibrated(self):
        return self._background is not None

    def _result(self, status, reason, fraction=0., box=None, scale=(1., 1.), shape=None):
        bbox = roi = None
        if box is not None:
            sx, sy = scale
            x, y, w, h = box
            x, y = round(x*sx), round(y*sy)
            w, h = max(1, round(w*sx)), max(1, round(h*sy))
            frame_h, frame_w = shape[:2]
            w, h = min(w, frame_w-x), min(h, frame_h-y)
            bbox = (x, y, w, h)
            pad = max(8, round(max(w, h)*.12))
            left, top = max(0, x-pad), max(0, y-pad)
            right, bottom = min(frame_w, x+w+pad), min(frame_h, y+h+pad)
            roi = (left, top, right-left, bottom-top)
        progress = 1. if self.calibrated else len(self._samples)/self.calibration_frames
        return dict(status=status, reason=reason, bbox=bbox, roi_bbox=roi,
                    progress=progress, foreground_fraction=float(fraction))

    def process(self, frame):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Expected a uint8 BGR camera frame')
        if self._shape is not None and frame.shape != self._shape:
            self._scene_changed = True
        self._shape = frame.shape
        if self._scene_changed:
            return self._result('scene_changed', 'Scene changed. Clear the scene and press B to recalibrate.')
        h, w = frame.shape[:2]
        factor = min(1., self.max_width/w)
        small = cv2.resize(frame, (max(1, round(w*factor)), max(1, round(h*factor))))
        smooth = cv2.GaussianBlur(small, (5, 5), 0)
        scale = (w/smooth.shape[1], h/smooth.shape[0])

        if not self.calibrated:
            # Require a consecutive stable sequence, rather than averaging a moving
            # hand into the background. A static object cannot be identified here:
            # the user must explicitly keep this calibration view empty.
            if self._previous is not None:
                change = np.max(np.abs(smooth.astype(np.float32)-self._previous), axis=2)
                moving_fraction = float((change > 18).mean())
                if moving_fraction > .01 or float(change.mean()) > 4:
                    self._samples = []
            self._previous = smooth.astype(np.float32)
            self._samples.append(smooth)
            if len(self._samples) < self.calibration_frames:
                return self._result('calibrating', 'Keep the camera still and the sorting area empty.')
            stack = np.stack(self._samples).astype(np.float32)
            self._background = np.median(stack, axis=0)
            noise = np.max(np.median(np.abs(stack-self._background), axis=0), axis=2)
            self._threshold = np.maximum(22., 5*noise+8)
            self._gray_background = cv2.cvtColor(self._background.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            self._samples = []
            return self._result('no_object', 'Ready. Show one item in the sorting area.')

        current = smooth.astype(np.float32)
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Textured backgrounds expose camera displacement even when changed pixels
        # cover less than the global-foreground threshold. Uniform walls cannot
        # provide this evidence and rely on the residual-area check instead.
        if self._gray_background.std() > 12 and gray.std() > 12:
            displacement, response = cv2.phaseCorrelate(self._gray_background, gray)
            if response > .35 and np.hypot(*displacement) > 6:
                self._scene_changed = True
                self._confirm = 0
                return self._result('scene_changed', 'Camera moved. Clear the scene and press B to recalibrate.')

        # Robust per-channel exposure compensation: a minority foreground object
        # cannot change the median wall/background color. This also compensates
        # moderate white-balance drift without continuously learning the object.
        shift = np.median(current-self._background, axis=(0, 1))
        residual = np.max(np.abs(current-self._background-shift), axis=2)
        mask = (residual > self._threshold).astype(np.uint8)*255
        raw_fraction = float((mask > 0).mean())
        if raw_fraction > self.max_area_ratio:
            self._scene_changed = True
            self._confirm = 0
            return self._result('scene_changed', 'Too much of the scene changed. Clear the scene and press B.', raw_fraction)

        # Suppress moderate neutral darkening (typical soft shadows) in low-edge
        # regions. Dark/colored material remains; this is a heuristic, not a
        # semantic shadow classifier, and low-contrast gray objects may be missed.
        compensated = current-shift
        ratio = (compensated.mean(axis=2)+1)/(self._background.mean(axis=2)+1)
        bg_chroma = self._background/(self._background.sum(axis=2, keepdims=True)+1)
        cur_chroma = compensated/(compensated.sum(axis=2, keepdims=True)+1)
        chroma_distance = np.max(np.abs(cur_chroma-bg_chroma), axis=2)
        shadow = (ratio > .72) & (ratio < .99) & (chroma_distance < .025)
        mask[shadow] = 0
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = smooth.shape[0]*smooth.shape[1]
        proposals = sorted([(cv2.contourArea(c), cv2.boundingRect(c)) for c in contours
                            if cv2.contourArea(c) >= area*self.min_area_ratio], reverse=True)
        if not proposals:
            self._confirm = 0
            self._last_box = None
            return self._result('no_object', 'No object. Show one item in the sorting area.', raw_fraction)
        largest, box = proposals[0]
        if len(proposals) > 1 and proposals[1][0] > largest*.30:
            self._confirm = 0
            self._last_box = None
            return self._result('multiple_objects', 'Show one separated item at a time.', raw_fraction)
        x, y, bw, bh = box
        if bw*bh/area > .8:
            self._scene_changed = True
            self._confirm = 0
            return self._result('scene_changed', 'Object or scene fills the view. Clear it and press B.', raw_fraction)
        overlap = 0.
        if self._last_box is not None:
            lx, ly, lw, lh = self._last_box
            intersection = max(0, min(x+bw, lx+lw)-max(x, lx))*max(0, min(y+bh, ly+lh)-max(y, ly))
            overlap = intersection/max(1, bw*bh+lw*lh-intersection)
        self._confirm = self._confirm+1 if overlap > .3 else 1
        self._last_box = box
        status = 'object' if self._confirm >= self.confirm_frames else 'candidate'
        return self._result(status, 'Object detected.' if status == 'object' else 'Hold the item steady.',
                            raw_fraction, box, scale, frame.shape)
