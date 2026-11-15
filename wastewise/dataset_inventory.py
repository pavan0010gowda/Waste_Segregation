"""Account for every local image without mixing training and evaluation data.

By default this is an inventory only. --include-unlisted writes a NEW manifest
with previously unlisted, folder-labelled train/test images after screening.
Existing images and split assignments are never modified. Camera feedback still
requires explicit curation; an arbitrary download never becomes a training label.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
from PIL import Image, ImageOps

from .common import ROOT, LABELS, IMAGE_EXTENSIONS, digest, read_json, save_json

ACTIVE_SPLITS = {'train', 'validation', 'test', 'external_test', 'organic_holdout', 'everyday_holdout'}


def normalize_path(value):
    return str(value).replace('\\', '/')


def exclusions(root):
    result = {}
    for filename, key in [('artifacts/organic_sources.json', 'exclusions'),
                          ('artifacts/everyday_sources.json', 'excluded')]:
        path = root / filename
        if path.exists():
            for row in read_json(path).get(key, []):
                result[normalize_path(row['path'])] = row['reason']
    path = root / 'artifacts/dataset_sources.json'
    if path.exists():
        for value in read_json(path).get('realwaste-food', {}).get('excluded', []):
            result[normalize_path(value)] = 'near_duplicate_or_overlap'
    path = root / 'artifacts/external_trashnet.json'
    if path.exists():
        for value in read_json(path).get('excluded_paths', []):
            result[normalize_path(value)] = 'external_benchmark_overlap'
    return result


def inspect_image(root, path):
    result = dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size,
                  sha256=digest(path))
    try:
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert('RGB')
            rgb.load()
            result.update(size=list(rgb.size), phash=str(imagehash.phash(rgb)))
            if getattr(image, 'n_frames', 1) != 1:
                result['error'] = 'Multi-frame image requires explicit frame curation'
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        result['error'] = str(error)
    return result


def classify_file(item, manifest, excluded, source_parents):
    """Prefer manifest membership, then explicit provenance, never folder guesses."""
    path = item['path']
    parts = Path(path).parts
    if path not in manifest and path.startswith('data/external/trashnet/') and (
            '__MACOSX' in parts or Path(path).name.startswith('._')):
        return 'archive_metadata', 'macOS archive sidecar with an image-like filename; not an image'
    if item.get('error'):
        return 'invalid', item['error']
    if path in manifest:
        row = manifest[path]
        if item['sha256'] != row['sha256']:
            return 'changed_since_manifest', 'Re-audit changed bytes before using this image'
        return row['split'], 'Current manifest assignment'
    if path in excluded:
        return 'excluded', excluded[path]
    if path in source_parents:
        return 'crop_source', 'Original source photograph; its annotation crop is screened separately'
    if path.startswith('data/external/trashnet/'):
        if Path(path).parent.name == 'trash':
            return 'unmapped_external', 'Trash category has no unambiguous four-class mapping'
        if Path(path).parent.name in {'paper', 'cardboard', 'glass', 'metal', 'plastic'}:
            return 'external_benchmark', 'Reserved external material benchmark; not current training data'
    if path.startswith(('data/camera_samples/', 'data/review/')):
        return 'pending_curation', 'Human label and capture-session split require review before training'
    return 'unaccounted', 'No manifest assignment or recorded source/exclusion'


def extend_manifest(rows, items):
    """Keep existing assignments; accept only labelled original train/test folders.

    Check against every old row, including quarantined and duplicate rows. Screen
    new candidates against each other first so no cross-label/split chain can leak.
    A new training image joins train; existing validation/test remain frozen.
    """
    old_paths = {normalize_path(row['path']) for row in rows}
    candidates = []
    for item in items:
        parts = Path(item['path']).parts
        if item['path'] in old_paths or item.get('error'):
            continue
        if len(parts) >= 4 and parts[0] == 'data' and parts[1] in {'train', 'test'} and parts[2] in LABELS:
            candidates.append(dict(item, label=parts[2], original_split=parts[1]))
    hashes = [int(row['phash'], 16) for row in rows]
    added, rejected = [], []
    for item in candidates:
        code = int(item['phash'], 16)
        if any(item['sha256'] == row['sha256'] for row in rows) or any((code ^ h).bit_count() <= 4 for h in hashes):
            rejected.append(dict(path=item['path'], reason='unlisted_near_duplicate_or_overlap'))
            continue
        if any(other['path'] != item['path'] and (code ^ int(other['phash'], 16)).bit_count() <= 4
               for other in candidates):
            rejected.append(dict(path=item['path'], reason='unlisted_related_group_requires_curation'))
            continue
        added.append({**{k: item[k] for k in ('path', 'sha256', 'phash', 'size', 'label', 'original_split')},
                      'split': item['original_split'], 'group': item['sha256'],
                      'source': 'Existing locally supplied labelled dataset',
                      'label_source': 'Existing class folder; no model-generated label',
                      'license': 'Inherited source rights not established; image is not published'})
    return rows + added, added, rejected


def inventory(root, manifest_path, output, include_unlisted=None):
    root, manifest_path = Path(root).resolve(), Path(manifest_path).resolve()
    if include_unlisted is not None:
        include_unlisted = Path(include_unlisted).resolve()
    rows = read_json(manifest_path)
    files = sorted(p for p in (root / 'data').rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    with ThreadPoolExecutor(max_workers=4) as pool:
        items = list(pool.map(lambda path: inspect_image(root, path), files))
    added, rejected = [], []
    if include_unlisted is not None:
        if Path(include_unlisted).resolve() == manifest_path:
            raise ValueError('Use a different output manifest to preserve the original')
        rows, added, rejected = extend_manifest(rows, items)
        save_json(include_unlisted, rows)
    row_map = {normalize_path(row['path']): row for row in rows}
    if len(row_map) != len(rows):
        raise ValueError('Duplicate path entries in manifest')
    excluded = exclusions(root)
    excluded.update({r['path']: r['reason'] for r in rejected})
    source_parents = set()
    annotation_path = root / 'data/external/everyday/taco_annotations.json'
    plan_path = root / 'data/external/everyday/selection_plan.json'
    if annotation_path.exists() and plan_path.exists():
        ids = set(read_json(plan_path)['image_ids'])
        for entry in read_json(annotation_path)['images']:
            if entry['id'] in ids:
                source_parents.add(f"data/external/everyday/taco_originals/{entry['id']:05}.jpg")
    for item in items:
        item['role'], item['reason'] = classify_file(item, row_map, excluded, source_parents)
    present = {item['path'] for item in items}
    missing = sorted(set(row_map) - present)
    counts = dict(sorted(Counter(item['role'] for item in items).items()))
    summary = dict(total_images=len(items), total_bytes=sum(item['bytes'] for item in items),
                   decoded_images=sum(not item.get('error') for item in items),
                   counts=counts, manifest=str(Path(include_unlisted or manifest_path).relative_to(root).as_posix()),
                   manifest_sha256=digest(include_unlisted or manifest_path),
                   missing_manifest_images=missing, new_manifest_images=added,
                   exclusions_of_unlisted_images=rejected,
                   issues=sum(counts.get(k, 0) for k in ('unaccounted', 'changed_since_manifest', 'invalid')) + len(missing),
                   policy='Every file is accounted for. Only train rows fit the model; validation selects it; holdouts remain separate. Source photos and rejected duplicates are not extra training examples.')
    save_json(output, dict(summary=summary, images=items))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'artifacts/all-data-manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/local/dataset_inventory.json')
    parser.add_argument('--include-unlisted', type=Path, metavar='NEW_MANIFEST')
    args = parser.parse_args()
    result = inventory(ROOT, args.manifest, args.output, args.include_unlisted)
    save_json(ROOT / 'artifacts/dataset_inventory_summary.json', result)
    print(result, flush=True)
    if result['issues']:
        raise SystemExit('Resolve inventory issues before claiming complete data coverage')


if __name__ == '__main__':
    main()
