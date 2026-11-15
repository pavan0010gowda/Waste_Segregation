import numpy as np
from PIL import Image
from wastewise.audit import audit
from wastewise.common import ROOT, LABELS


def test_cross_split_duplicates_and_label_conflicts(tmp_path, monkeypatch):
    import wastewise.audit as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    data = tmp_path / 'data'
    rng = np.random.default_rng(5)
    for split in ('train', 'test'):
        for label in LABELS:
            (data / split / label).mkdir(parents=True)
            for i in range(4):
                Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(data / split / label / f'{i}.png')
    image = (data / 'train/wet/0.png').read_bytes()
    (data / 'test/wet/copy.png').write_bytes(image)
    (data / 'train/dry_recyclable/conflict.png').write_bytes((data / 'train/dry_decomposable/0.png').read_bytes())
    rows = audit(data, tmp_path / 'artifacts')
    by_group = {}
    for row in rows:
        if row['split'] in ('train', 'validation', 'test'):
            assert row['group'] not in by_group
            by_group[row['group']] = row['split']
    assert next(r for r in rows if r['path'].endswith('train/wet/0.png'))['split'] == 'duplicate'
    assert next(r for r in rows if 'conflict.png' in r['path'])['split'] == 'quarantine'
