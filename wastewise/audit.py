"""Audit images and make deterministic, duplicate-group-disjoint splits.

Original test images remain test. Groups touching test are excluded from training.
Conflicting labels are quarantined. Near duplicates use perceptual hash distance <= 4.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import random

import imagehash
from PIL import Image, ImageOps

from .common import ROOT, LABELS, IMAGE_EXTENSIONS, digest, save_json


def audit(data_dir, output, seed=42):
    rows, invalid = [], []
    for split in ('train', 'test'):
        for label in LABELS:
            for path in sorted((data_dir / split / label).rglob('*')):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                try:
                    with Image.open(path) as im:
                        im = ImageOps.exif_transpose(im).convert('RGB')
                        im.load()
                        phash = str(imagehash.phash(im))
                        pixels = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
                        size = list(im.size)
                    rows.append(dict(path=path.resolve().relative_to(ROOT).as_posix(), label=label,
                                     original_split=split, sha256=digest(path), pixels=pixels,
                                     phash=phash, size=size))
                    if len(rows) % 250 == 0:
                        print(f'Audited {len(rows)} images', flush=True)
                except (OSError, ValueError, Image.DecompressionBombError) as e:
                    invalid.append(dict(path=str(path), reason=str(e)))
    if not rows:
        raise ValueError('No images found in data/train or data/test')
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hashes = [int(r['phash'], 16) for r in rows]
    for i in range(len(rows)):
        for j in range(i):
            if (hashes[i] ^ hashes[j]).bit_count() <= 4:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    conflicts, overlap, duplicate_count = [], [], 0
    candidates = defaultdict(list)
    for members in groups.values():
        group_id = min(r['sha256'] for r in members)
        duplicate_count += len(members) - 1
        labels = {r['label'] for r in members}
        splits = {r['original_split'] for r in members}
        if len(labels) > 1:
            conflicts.append([r['path'] for r in members])
            assigned = 'quarantine'
        elif 'test' in splits:
            assigned = 'test'
            if 'train' in splits:
                overlap.append([r['path'] for r in members])
        else:
            assigned = 'pending'
            candidates[members[0]['label']].append(members)
        for row in members:
            row.update(group=group_id, split=assigned)
        # One representative per group; never move train copies into test.
        eligible = [r for r in members if r['original_split'] == 'test'] if assigned == 'test' else members
        chosen = eligible[0]
        for row in members:
            if row is not chosen and assigned != 'quarantine':
                row['split'] = 'duplicate'
    rng = random.Random(seed)
    for label, members_list in sorted(candidates.items()):
        rng.shuffle(members_list)
        n_val = max(1, round(len(members_list) * .2))
        for i, members in enumerate(members_list):
            for row in members:
                if row['split'] == 'pending':
                    row['split'] = 'validation' if i < n_val else 'train'
    counts = {s: dict(Counter(r['label'] for r in rows if r['split'] == s))
              for s in ('train', 'validation', 'test', 'duplicate', 'quarantine')}
    report = dict(seed=seed, images=len(rows), invalid=invalid, counts=counts,
                  near_duplicate_extras=duplicate_count, cross_split_groups=overlap,
                  conflicting_groups=conflicts,
                  caveat='pHash is a heuristic; manually inspect conflicts. Original test was used by legacy training, so this is not a pristine holdout.')
    save_json(output / 'manifest.json', rows)
    save_json(output / 'audit.json', report)
    print({k: v for k, v in report.items() if k not in ('invalid', 'cross_split_groups', 'conflicting_groups')}, flush=True)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts')
    args = p.parse_args()
    audit(args.data, args.output)
