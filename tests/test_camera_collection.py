"""Exercise explicit capture keys without a webcam, GUI, model or filesystem writes."""
import sys
from pathlib import Path

import numpy as np

from wastewise import camera
from wastewise.common import LABELS


def simulated_camera(monkeypatch, keys, collect):
    frames = [np.full((120, 200, 3), 100+i, np.uint8) for i in range(len(keys))]
    seen = dict(saves=[], notices=[], frames=[], proposals=[], released=False,
                closed=False, session_calls=0, key_index=0, calls_before_keys=[])
    session = '11111111111141118111111111111111'

    class Capture:
        def isOpened(self):
            return True

        def read(self):
            frame = frames[len(seen['frames'])]
            seen['frames'].append(frame)
            return True, frame

        def release(self):
            seen['released'] = True

    class Pipeline:
        def __init__(self, classifier):
            pass

        def process(self, frame):
            # The predicted label deliberately differs from key 3's user label.
            proposal = dict(status='accepted', label='wet', confidence=.95,
                            bbox=(50, 30, 40, 40), roi_bbox=(40, 20, 60, 60))
            seen['proposals'].append(proposal)
            return proposal

        def reset(self):
            pass

    def new_session():
        seen['session_calls'] += 1
        return session

    def draw(frame, proposal):
        # Simulate an annotated display copy distinct from the original pixels.
        shown = frame.copy()
        shown[:] = 255
        return shown

    def save(frame, proposal, label, session_id):
        seen['saves'].append(dict(frame=frame, proposal=proposal, label=label, session_id=session_id))
        return {'image_path': Path('simulated') / f'{len(seen["saves"])}.jpg'}

    def wait(_):
        seen['calls_before_keys'].append(len(seen['saves']))
        key = keys[seen['key_index']]
        seen['key_index'] += 1
        return key

    monkeypatch.setattr(camera, 'LiveClassifier', lambda _: object())
    monkeypatch.setattr(camera, 'LivePipeline', Pipeline)
    monkeypatch.setattr(camera, 'new_session_id', new_session)
    monkeypatch.setattr(camera, 'save_sample', save)
    monkeypatch.setattr(camera, 'draw_result', draw)
    monkeypatch.setattr(camera.time, 'monotonic', lambda: 100.)
    monkeypatch.setattr(camera.cv2, 'VideoCapture', lambda _: Capture())
    monkeypatch.setattr(camera.cv2, 'rectangle', lambda *a: None)
    monkeypatch.setattr(camera.cv2, 'putText', lambda image, text, *a: seen['notices'].append(text))
    monkeypatch.setattr(camera.cv2, 'imshow', lambda *a: None)
    monkeypatch.setattr(camera.cv2, 'waitKey', wait)
    monkeypatch.setattr(camera.cv2, 'destroyAllWindows', lambda: seen.update(closed=True))
    monkeypatch.setattr(sys, 'argv', ['live_detect.py'] + (['--collect'] if collect else []))
    camera.main()
    return seen, frames, session


def test_digits_do_not_save_when_collection_is_disabled(monkeypatch):
    seen, frames, _ = simulated_camera(monkeypatch, [-1, ord('3'), ord('0'), ord('q')], collect=False)
    assert seen['saves'] == []
    assert seen['calls_before_keys'] == [0, 0, 0, 0]
    assert seen['notices'] == []
    assert seen['released'] and seen['closed']
    assert len(seen['frames']) == len(frames)


def test_enabled_collection_saves_only_explicit_raw_crops_and_confirms(monkeypatch, capsys):
    seen, frames, session = simulated_camera(monkeypatch, [-1, ord('3'), -1, ord('0'), ord('q')], collect=True)
    # No implicit capture during either idle frame; saving happens after the key.
    assert seen['calls_before_keys'] == [0, 0, 1, 1, 2]
    assert len(seen['saves']) == 2
    for saved, frame_index in zip(seen['saves'], (1, 3)):
        assert saved['frame'] is frames[frame_index]
        assert saved['proposal'] is seen['proposals'][frame_index]
        assert saved['session_id'] == session
        assert np.all(saved['frame'] == 100+frame_index)  # No on-screen annotation saved.
    assert seen['saves'][0]['label'] == LABELS[2] == 'dry_recyclable'
    assert seen['saves'][1]['label'] == 'unknown'
    assert seen['session_calls'] == 1
    assert any('Saved dry_recyclable crop' in text for text in seen['notices'])
    assert any('Saved unknown crop' in text for text in seen['notices'])
    assert 'model unchanged' in capsys.readouterr().out
    assert seen['released'] and seen['closed']
