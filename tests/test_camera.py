import sys
import numpy as np
from wastewise import camera
from wastewise.common import LABELS
from scripts.arduino_live import command_for_sensor


def test_live_camera_displays_labels_and_releases(monkeypatch):
    seen = {'labels': [], 'released': False, 'closed': False, 'models': []}
    class FakeModel:
        def __init__(self, model): seen['models'].append(model)
        def predict_frame(self, frame):
            return {'label': 'dry_recyclable', 'confidence': .92, 'status': 'accepted',
                    'probabilities': dict(zip(LABELS, [.03,.02,.92,.03]))}
    class FakePipeline:
        def __init__(self, classifier): self.classifier=classifier
        def process(self, frame): return self.classifier.predict_frame(frame)
        def reset(self): pass
    class FakeCamera:
        def isOpened(self): return True
        def read(self): return True, np.zeros((240, 320, 3), dtype=np.uint8)
        def release(self): seen['released'] = True
    monkeypatch.setattr(camera, 'LiveClassifier', FakeModel)
    monkeypatch.setattr(camera, 'LivePipeline', FakePipeline)
    monkeypatch.setattr(camera.cv2, 'VideoCapture', lambda _: FakeCamera())
    monkeypatch.setattr(camera.cv2, 'putText', lambda image, text, *a: seen['labels'].append(text))
    monkeypatch.setattr(camera.cv2, 'imshow', lambda *a: None)
    monkeypatch.setattr(camera.cv2, 'waitKey', lambda _: ord('q'))
    monkeypatch.setattr(camera.cv2, 'destroyAllWindows', lambda: seen.update(closed=True))
    monkeypatch.setattr(sys, 'argv', ['live_detect.py'])
    camera.main()
    assert seen['models'] == ['selective']
    assert 'dry_recyclable: 92%' in seen['labels']
    assert seen['released'] and seen['closed']


def test_original_sensor_protocol():
    assert command_for_sensor('WET') == 'MOVE_WET'
    assert command_for_sensor('DRY') == 'MOVE_DRY'
    assert command_for_sensor('noise') is None


def test_previous_model_uses_frozen_bundle_without_changing_selection(tmp_path,monkeypatch):
    from wastewise import live_model, selective
    from wastewise.common import save_json, read_json
    selection=dict(run='artifacts/runs/new',previous_run='artifacts/runs/frozen')
    selection_path=tmp_path/'artifacts/live_model.json'
    save_json(selection_path,selection)
    seen=[]
    class Predictor:
        def __init__(self,path): seen.append(path)
        def predict(self,image): return dict(label='wet',confidence=.9)
    monkeypatch.setattr(live_model,'ROOT',tmp_path)
    monkeypatch.setattr(selective,'SelectivePredictor',Predictor)
    model=live_model.LiveClassifier('previous')
    assert seen==[(tmp_path/'artifacts/runs/frozen').resolve()]
    assert model.predict_frame(np.zeros((12,12,3),np.uint8))['label']=='wet'
    assert read_json(selection_path)==selection
