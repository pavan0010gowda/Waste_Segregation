"""Acquire a modest, attributed Fruits-360 subset without rotating-frame leakage.

Fresh produce is mapped to the inherited ``wet`` label as an organic-material
proxy. This does not establish moisture, spoilage, or disposal suitability. Entire
upstream class folders are assigned to a split before any classifier is fitted.
"""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

import imagehash
import numpy as np
from PIL import Image, ImageOps
from .common import ROOT, read_json, save_json, digest

REPOSITORY = 'fruits-360/fruits-360-100x100'
REVISION = 'bed1c24ee70de1569272de3882fbde88211c08b5'
FOLDER_SPLITS = {
    'Apple Braeburn 1': 'train', 'Banana 1': 'train',
    'Cucumber 1': 'train', 'Orange 1': 'train', 'Pear 1': 'train',
    'Pepper Red 1': 'train', 'Tomato 1': 'train', 'Tomato Yellow 1': 'train',
    'Tomato 2': 'validation', 'Pear 2': 'validation',
    'Peach 1': 'organic_holdout', 'Plum 1': 'organic_holdout',
}
PER_FOLDER = 70


def natural_key(value):
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower())
                 for part in re.split(r'(\d+)', value))


def sample_entries(entries, limit=PER_FOLDER):
    """Evenly sample the natural filename order, rather than adjacent frames."""
    images = sorted((e for e in entries if e.get('type') == 'blob'
                     and Path(e['path']).suffix.lower() in {'.jpg', '.jpeg', '.png'}),
                    key=lambda e: natural_key(e['path']))
    if not images:
        raise ValueError('No supported images in selected folder')
    if limit < 1:
        raise ValueError('Sampling limit must be positive')
    indices = np.linspace(0, len(images)-1, min(limit, len(images)), dtype=int)
    return [images[int(index)] for index in indices]


def read_url(url):
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'WasteWise-dataset-audit'})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def git_blob_sha(content):
    return hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()


def verified_download(entry, target):
    target = Path(target)
    if target.exists():
        content = target.read_bytes()
        if git_blob_sha(content) == entry['sha']:
            return target
    url = f'https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/' + urllib.parse.quote(entry['source_path'])
    content = read_url(url)
    if git_blob_sha(content) != entry['sha']:
        raise ValueError(f"Git blob checksum mismatch: {entry['source_path']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.part')
    temporary.write_bytes(content)
    temporary.replace(target)
    return target


def screen_candidates(candidates, original, distance=4):
    """Exclude overlap with old data; quarantine cross-split duplicate components."""
    retained, excluded = [], []
    previous = [(int(row['phash'], 16), row['label']) for row in original]
    for row in candidates:
        nearby = [label for phash, label in previous
                  if (int(row['phash'], 16) ^ phash).bit_count() <= distance]
        if nearby:
            excluded.append(dict(path=row['path'], reason='legacy_overlap', conflicting_label=any(label != 'wet' for label in nearby)))
        else:
            retained.append(row)
    parent = list(range(len(retained)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hashes = [int(row['phash'], 16) for row in retained]
    for i in range(len(retained)):
        for j in range(i):
            if (hashes[i] ^ hashes[j]).bit_count() <= distance:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, row in enumerate(retained):
        groups[find(i)].append(row)
    unique = []
    for group in groups.values():
        if len({row['split'] for row in group}) > 1:
            excluded.extend(dict(path=row['path'], reason='cross_split_component', conflicting_label=False) for row in group)
            continue
        selected = min(group, key=lambda row: row['path'])
        selected = {**selected, 'duplicate_group': min(row['sha256'] for row in group)}
        unique.append(selected)
        excluded.extend(dict(path=row['path'], reason='same_split_near_duplicate', conflicting_label=False)
                        for row in group if row['path'] != selected['path'])
    return sorted(unique, key=lambda row: row['path']), excluded


def main():
    directory = ROOT / 'data/external/fruits360'
    cache = directory / 'metadata'
    cache.mkdir(parents=True, exist_ok=True)
    original_path = ROOT / 'artifacts/augmented-manifest.json'
    original_sha = digest(original_path)
    original = read_json(original_path)

    def tree(identifier):
        path = cache / f'{identifier}.json'
        if not path.exists():
            response = json.loads(read_url(f'https://api.github.com/repos/{REPOSITORY}/git/trees/{identifier}'))
            if response.get('truncated'):
                raise ValueError('Truncated tree response cannot establish source inventory')
            save_json(path, response)
        return read_json(path)

    root = tree(REVISION)
    top = {entry['path']: entry for entry in root['tree']}
    for name in ('LICENSE', 'README.md'):
        verified_download({**top[name], 'source_path': name}, directory / name)
    folders = {entry['path']: entry for entry in tree(top['Training']['sha'])['tree']}
    selected = []
    inventory = {}
    for folder, split in FOLDER_SPLITS.items():
        if folder not in folders:
            raise ValueError(f'Pinned source lacks planned folder: {folder}')
        entries = tree(folders[folder]['sha'])['tree']
        chosen = sample_entries(entries)
        inventory[folder] = dict(split=split, available=sum(e.get('type') == 'blob' for e in entries), sampled=len(chosen))
        for entry in chosen:
            if not re.fullmatch(r'[A-Za-z0-9_. -]+', entry['path']) or entry['path'] in {'.', '..'}:
                raise ValueError('Unexpected filename in source folder')
            selected.append({**entry, 'folder': folder, 'split': split,
                             'source_path': f"Training/{folder}/{entry['path']}"})
    # Save split plan before reading any images or classifier predictions.
    save_json(directory / 'selection_plan.json', dict(revision=REVISION, folder_splits=FOLDER_SPLITS,
              per_folder=PER_FOLDER, inventory=inventory, rule='Natural filename order, evenly spaced indices; whole folders assigned to a split'))

    def acquire(entry):
        path = directory / 'images' / entry['folder'] / entry['path']
        verified_download(entry, path)
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert('RGB')
            rgb.load()
            phash, size = str(imagehash.phash(rgb)), list(rgb.size)
        return dict(path=path.relative_to(ROOT).as_posix(), label='wet', original_split='external_training',
                    split=entry['split'], sha256=digest(path), phash=phash, size=size,
                    source='Fruits-360', source_revision=REVISION, git_blob_sha1=entry['sha'],
                    source_group=entry['folder'], group='fruits360:' + entry['folder'])

    with ThreadPoolExecutor(max_workers=8) as pool:
        candidates = []
        for row in pool.map(acquire, selected):
            candidates.append(row)
            if len(candidates) % 70 == 0:
                print(f'Fruits-360 downloaded and verified: {len(candidates)}/{len(selected)}', flush=True)
    unique, excluded = screen_candidates(candidates, original)
    for folder in FOLDER_SPLITS:
        inventory[folder]['retained'] = sum(row['source_group'] == folder for row in unique)
    if digest(original_path) != original_sha:
        raise ValueError('Original manifest changed during acquisition; rerun with a stable manifest')
    manifest = original + unique
    save_json(ROOT / 'artifacts/organic-manifest.json', manifest)
    report = dict(source=f'https://github.com/{REPOSITORY}', revision=REVISION,
                  authors='Mihai Oltean; Fruits-360 dataset contributors', license='CC BY-SA 4.0',
                  license_url=f'https://github.com/{REPOSITORY}/blob/{REVISION}/LICENSE',
                  license_file_sha256=digest(directory / 'LICENSE'),
                  source_readme_sha256=digest(directory / 'README.md'),
                  original_manifest_sha256=original_sha,
                  new_manifest_sha256=digest(ROOT / 'artifacts/organic-manifest.json'),
                  requested=len(selected), retained=len(unique), download_bytes=sum((ROOT / row['path']).stat().st_size for row in candidates),
                  added_split_counts=dict(Counter(row['split'] for row in unique)),
                  mapping={'selected fresh fruit/vegetable classes': 'wet'},
                  group_policy='Entire upstream Training class-folder variants stay in one split. Peach/Plum held out; adjacent rotating frames never split across train/validation/holdout.',
                  sampling='Up to70 evenly spaced indices in natural filename order per folder; pHash connected-component screening at Hamming distance<=4.',
                  exclusions=excluded, folder_inventory=inventory,
                  limitations=['Fresh produce is an organic-material proxy, not a moisture measurement or proof an item is waste.',
                               'White studio backgrounds differ from the sorting camera. New holdout is organic-only and does not measure four-class accuracy.',
                               'Class folders serve as conservative capture groups; metadata does not prove every folder represents a unique physical specimen.',
                               'pHash screening is heuristic and does not prove complete independence.',
                               'Dataset expansion follows diagnosis on historical test data; historical scores remain exploratory.'],
                  attribution='Mihai Oltean, Fruits-360 dataset, 2017–. CC BY-SA 4.0. Source files retained; no alterations to downloaded images. Preserve attribution and ShareAlike terms when redistributing adapted dataset material.')
    save_json(ROOT / 'artifacts/organic_sources.json', report)
    save_json(directory / 'provenance.json', report)
    print({key: report[key] for key in ('requested', 'retained', 'download_bytes', 'added_split_counts')}, flush=True)


if __name__ == '__main__':
    main()
