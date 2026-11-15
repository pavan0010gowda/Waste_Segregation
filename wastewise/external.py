"""Download TrashNet with provenance; audit overlap before external evaluation.

External material labels do not establish moisture or local recyclability.
Paper/cardboard/glass/metal/plastic -> dry_recyclable, matching legacy training examples.
Trash is excluded because it has no reliable one-to-one legacy label mapping.
"""
import argparse
from pathlib import Path
import shutil
import urllib.request
import zipfile
import imagehash
import numpy as np
from PIL import Image, ImageOps
from .common import ROOT, LABELS, digest, read_json, save_json

URL = 'https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip'
MAPPING = {name: 'dry_recyclable' for name in ('paper', 'cardboard', 'glass', 'metal', 'plastic')}


def fetch():
    directory = ROOT / 'data/external/trashnet'
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / 'dataset-resized.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.part')
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open('wb') as out:
            shutil.copyfileobj(response, out)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (directory / member.filename).resolve()
            if not target.is_relative_to(directory.resolve()):
                raise ValueError('Unsafe archive member')
            if not member.is_dir() and Path(member.filename).suffix.lower() == '.jpg':
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as source, target.open('wb') as out:
                    shutil.copyfileobj(source, out)
    save_json(directory / 'provenance.json', dict(source=URL, project='https://github.com/garythung/trashnet', authors='Gary Thung and Mindy Yang', archive_sha256=digest(archive),
              license='MIT (upstream repository LICENSE)', mapping=MAPPING, excluded='trash: ambiguous legacy category',
              purpose='External material-transfer evaluation; not automatically merged into training.'))
    print(f'Downloaded TrashNet to {directory}', flush=True)


def evaluate():
    from .inference import Predictor
    from .metrics import report
    from .train import load_pixels
    rows = read_json(ROOT / 'artifacts/manifest.json')
    existing = [int(r['phash'], 16) for r in rows]
    seen, retained, excluded = [], [], []
    for path in sorted((ROOT / 'data/external/trashnet').rglob('*.jpg')):
        if '__MACOSX' in path.parts or path.name.startswith('._'):
            continue
        label = MAPPING.get(path.parent.name)
        if label is None:
            continue
        with Image.open(path) as im:
            phash = int(str(imagehash.phash(ImageOps.exif_transpose(im).convert('RGB'))), 16)
        if any((phash ^ other).bit_count() <= 4 for other in existing + seen):
            excluded.append(path.relative_to(ROOT).as_posix())
        else:
            seen.append(phash)
            retained.append((path, label))
    if not retained:
        raise ValueError('No non-overlapping external images remain')
    predictor = Predictor()
    probabilities, labels = [], []
    for start in range(0, len(retained), 32):
        batch = retained[start:start+32]
        x = load_pixels([path for path, label in batch])
        probabilities.extend(predictor.model(x, training=False).numpy())
        labels.extend(LABELS.index(label) for path, label in batch)
        if start % 320 == 0:
            print(f'External inference: {start}/{len(retained)}', flush=True)
    metrics = report(labels, np.array(probabilities), LABELS)
    metrics.update(excluded_overlap=len(excluded), excluded_paths=excluded,
                   model_sha256=predictor.metadata['model_sha256'],
                   limitation='Only dry_recyclable is represented after mapping. Accuracy here is recyclable-class recall, not four-class accuracy; macro F1 includes absent classes. Mapping was corrected after inspecting original training taxonomy.',
                   mapping=MAPPING)
    save_json(ROOT / 'artifacts/external_trashnet.json', metrics)
    print({k:v for k,v in metrics.items() if k != 'excluded_paths'}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['fetch', 'evaluate'])
    args = p.parse_args()
    fetch() if args.action == 'fetch' else evaluate()
