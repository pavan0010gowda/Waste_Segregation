"""Use fake serial/camera objects; these tests never touch motors or a webcam."""
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import arduino_live


@pytest.mark.parametrize('statuses,messages,expected', [
    (['calibrating','no_object','accepted','accepted','no_object','no_object','no_object','accepted'],
     [b'WET\n',b'DRY\n',b'WET\n',b'WET\n',b'',b'',b'',b'DRY\n'], [b'MOVE_WET\n',b'MOVE_DRY\n']),
    (['candidate','checking','accepted'], [b'WET\n',b'',b''], [b'MOVE_WET\n']),
    (['candidate','no_object','accepted'], [b'WET\n',b'',b''], []),
    (['candidate','scene_changed','accepted'], [b'WET\n',b'',b''], []),
    (['uncertain','uncertain'], [b'DRY\n',b'DRY\n'], []),
])
def test_no_empty_scene_commands_and_one_route_per_object(monkeypatch,statuses,messages,expected):
    observed = {'index': -1, 'writes': [], 'released': False}
    class Pipeline:
        def __init__(self, classifier): pass
        def process(self, frame):
            observed['index'] += 1
            status=statuses[observed['index']]
            return dict(status=status, bbox=(20,20,50,50) if status in {'accepted','candidate','checking','uncertain'} else None,
                        label='dry_recyclable' if status=='accepted' else None, confidence=.95)
    class Serial:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        @property
        def in_waiting(self): return len(messages[observed['index']])
        def read(self, n): return messages[observed['index']][:n]
        def write(self, data): observed['writes'].append(data)
    class Camera:
        def isOpened(self): return True
        def read(self): return True,np.zeros((240,320,3),np.uint8)
        def release(self): observed['released']=True
    monkeypatch.setitem(sys.modules,'serial',SimpleNamespace(Serial=Serial))
    monkeypatch.setattr(arduino_live,'LiveClassifier',lambda _:None)
    monkeypatch.setattr(arduino_live,'LivePipeline',Pipeline)
    monkeypatch.setattr(arduino_live.cv2,'VideoCapture',lambda _:Camera())
    monkeypatch.setattr(arduino_live,'draw_result',lambda frame,result:frame)
    monkeypatch.setattr(arduino_live.cv2,'imshow',lambda *a:None)
    monkeypatch.setattr(arduino_live.cv2,'destroyAllWindows',lambda:None)
    monkeypatch.setattr(arduino_live.cv2,'waitKey',lambda _:ord('q') if observed['index']==len(statuses)-1 else -1)
    monkeypatch.setattr(sys,'argv',['arduino_live.py','--port','TEST'])
    arduino_live.main()
    assert observed['writes']==expected
    assert observed['released']
