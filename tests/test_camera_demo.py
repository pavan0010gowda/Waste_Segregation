"""Two-window integration without accessing a camera, display or serial port."""
import sys

import numpy as np
import pytest

from wastewise import camera
from wastewise.bin_demo import CAMERA_WINDOW, WINDOW, BinDemo


@pytest.mark.parametrize('use_mouse', [False, True])
def test_camera_and_bin_windows_share_review_controls_without_saving(monkeypatch, use_mouse):
    seen = dict(windows=[], displays=[], saves=[], released=False, closed=False,
                callback=None, i=0, resets=0)
    demo = BinDemo()
    keys = iter([-1, -1, ord('f'), ord('4'), ord('b'), ord('q')])

    class Capture:
        def isOpened(self): return True
        def read(self): return True, np.full((480, 640, 3), 140, np.uint8)
        def release(self): seen['released'] = True

    class Pipeline:
        def __init__(self, classifier): pass
        def reset(self): seen['resets'] += 1
        def process(self, frame):
            return dict(status='uncertain', label=None, confidence=.61, bbox=(100, 100, 100, 100),
                        demo_candidate=dict(label='dry_recyclable', confidence=.61))

    def wait(_):
        key = next(keys)
        seen['i'] += 1
        if seen['i'] == 4:
            assert demo.phase == 'review'
        if seen['i'] == 5:
            assert demo.phase == 'routing' and demo.label == 'wet' and demo.manual
        if use_mouse and key == ord('f'):
            seen['callback'](camera.cv2.EVENT_LBUTTONDOWN, 80, 530, 0, None)
            return -1
        if use_mouse and key == ord('4'):
            seen['callback'](camera.cv2.EVENT_LBUTTONDOWN, 600, 400, 0, None)
            return -1
        return key

    monkeypatch.setattr(camera, 'BinDemo', lambda: demo)
    monkeypatch.setattr(camera, 'LiveClassifier', lambda _: object())
    monkeypatch.setattr(camera, 'LivePipeline', Pipeline)
    monkeypatch.setattr(camera, 'save_sample', lambda *args: seen['saves'].append(args))
    monkeypatch.setattr(camera.time, 'monotonic', lambda: seen['i']*.1)
    monkeypatch.setattr(camera.cv2, 'VideoCapture', lambda _: Capture())
    monkeypatch.setattr(camera.cv2, 'namedWindow', lambda name, mode: seen['windows'].append(name))
    monkeypatch.setattr(camera.cv2, 'resizeWindow', lambda *args: None)
    monkeypatch.setattr(camera.cv2, 'moveWindow', lambda *args: None)
    monkeypatch.setattr(camera.cv2, 'setMouseCallback', lambda name, callback: seen.update(callback=callback))
    monkeypatch.setattr(camera.cv2, 'imshow', lambda name, frame: seen['displays'].append(name))
    monkeypatch.setattr(camera.cv2, 'getWindowProperty', lambda *args: 1.)
    monkeypatch.setattr(camera.cv2, 'waitKey', wait)
    monkeypatch.setattr(camera.cv2, 'destroyAllWindows', lambda: seen.update(closed=True))
    monkeypatch.setattr(sys, 'argv', ['live_detect.py', '--demo', '--collect'])
    camera.main()
    assert set(seen['windows']) == {CAMERA_WINDOW, WINDOW}
    assert set(seen['displays']) == {CAMERA_WINDOW, WINDOW}
    assert seen['saves'] == []
    assert seen['resets'] == 1 and demo.phase == 'waiting'
    assert seen['released'] and seen['closed']
