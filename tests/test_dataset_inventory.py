"""Regression checks for format coverage, safe additions and data accounting."""
import numpy as np
from PIL import Image

from wastewise.common import save_json
from wastewise.dataset_inventory import inspect_image, inventory, extend_manifest, classify_file


def sample(root, relative, seed=1):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.default_rng(seed).integers(0, 256, (48, 48, 3), dtype=np.uint8)).save(path)
    return inspect_image(root, path)


def test_avif_included_without_resplitting_existing_data(tmp_path):
    old = sample(tmp_path, 'data/test/wet/held.png')
    row = dict(old, label='wet', group=old['sha256'], split='test')
    added = sample(tmp_path, 'data/train/dry_decomposable/new.avif', seed=9)
    save_json(tmp_path / 'artifacts/base.json', [row])
    result = inventory(tmp_path, tmp_path / 'artifacts/base.json', tmp_path / 'report.json',
                       tmp_path / 'artifacts/extended.json')
    assert result['issues'] == 0
    assert result['counts'] == {'test': 1, 'train': 1}
    assert result['new_manifest_images'][0]['path'] == added['path']
    import json
    assert json.loads((tmp_path / 'artifacts/extended.json').read_text())[0] == row


def test_new_train_copy_of_test_is_rejected():
    held = dict(path='data/test/wet/held.png', sha256='same', phash='0123456789abcdef', split='test')
    new = dict(held, path='data/train/wet/copy.avif', size=[48, 48])
    rows, added, rejected = extend_manifest([held], [new])
    assert rows == [held] and not added
    assert rejected[0]['reason'] == 'unlisted_near_duplicate_or_overlap'


def test_new_related_images_never_cross_splits_or_labels():
    a = dict(path='data/train/wet/a.png', sha256='a', phash='0000000000000000', size=[48, 48])
    b = dict(a, path='data/test/dry_recyclable/b.png', sha256='b', phash='0000000000000001')
    rows, added, rejected = extend_manifest([], [a, b])
    assert not rows and not added and len(rejected) == 2


def test_unknown_crops_and_changed_bytes_are_not_training_data():
    item = dict(path='data/camera_samples/wet/a.jpg', sha256='new')
    assert classify_file(item, {}, {}, set())[0] == 'pending_curation'
    assert classify_file(item, {item['path']: {'sha256': 'old', 'split': 'train'}}, {}, set())[0] == 'changed_since_manifest'
    item['path'] = 'data/external/unmapped/example.jpg'
    assert classify_file(item, {}, {}, set())[0] == 'unaccounted'
    assert classify_file(item, {}, {}, {item['path']})[0] == 'crop_source'


def test_macos_sidecar_is_not_claimed_as_a_corrupt_training_image():
    item = dict(path='data/external/trashnet/__MACOSX/cardboard/._cardboard9.jpg',
                sha256='sidecar', error='Cannot decode')
    assert classify_file(item, {}, {}, set())[0] == 'archive_metadata'


def test_missing_and_corrupt_images_reported(tmp_path):
    corrupt = tmp_path / 'data/train/wet/broken.avif'
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b'not an image')
    save_json(tmp_path / 'artifacts/base.json', [dict(path='data/test/wet/missing.png', split='test', sha256='absent')])
    result = inventory(tmp_path, tmp_path / 'artifacts/base.json', tmp_path / 'report.json')
    assert result['issues'] == 2
    assert result['counts'] == {'invalid': 1}
    assert result['missing_manifest_images'] == ['data/test/wet/missing.png']
