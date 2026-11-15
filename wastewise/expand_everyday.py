"""Prepare attributed everyday-object crops without changing old dataset splits.

TACO v1.0 images are read from the authors' Zenodo archive using HTTP byte ranges.
Only selected members are downloaded; each member's ZIP CRC/size is checked.
RealWaste vegetation supplies decomposable plant material. Class mappings are
project material proxies, not a determination of local recyclability/moisture.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import io
from http.client import HTTPException
import json
import math
from pathlib import Path
import random
import struct
import time
import urllib.parse
import urllib.request
import zipfile
import zlib

import imagehash
from PIL import Image, ImageOps

from .common import ROOT, digest, read_json, save_json
from .expand_organic import git_blob_sha, sample_entries
from .expand import COMMIT as REALWASTE_REVISION

TACO_URL = 'https://zenodo.org/api/records/3587843/files/TACO.zip/content'
TACO_SIZE = 2719810615
TACO_MAPPING = {
    **dict.fromkeys(('Clear plastic bottle', 'Other plastic bottle', 'Glass bottle',
                     'Drink can', 'Food Can', 'Normal paper', 'Magazine paper',
                     'Corrugated carton', 'Other carton', 'Egg carton', 'Paper bag',
                     'Metal bottle cap', 'Metal lid'), 'dry_recyclable'),
    **dict.fromkeys(('Other plastic wrapper', 'Crisp packet', 'Plastic film',
                     'Plastic straw', 'Plastic utensils', 'Styrofoam piece',
                     'Foam cup', 'Foam food container', 'Cigarette'), 'dry_non_degradable'),
    'Food waste': 'wet',
}


def fetch(url, start=None, stop=None):
    headers = {'User-Agent': 'WasteWise research dataset preparation'}
    if start is not None:
        headers['Range'] = f'bytes={start}-{stop-1}'
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                if start is not None:
                    expected = f'bytes {start}-{stop-1}/{TACO_SIZE}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError('Archive server did not honor the requested byte range')
                content = response.read()
            if start is not None and len(content) != stop-start:
                raise ValueError('Truncated archive byte range')
            return content
        except (OSError, HTTPException):
            if attempt == 2:
                raise
            time.sleep(1+attempt)


class RemoteArchive(io.RawIOBase):
    """Seekable view used only to read the ZIP central directory."""
    def __init__(self): self.position = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else TACO_SIZE)
        if self.position < 0: raise ValueError('Invalid archive offset')
        return self.position
    def read(self, size=-1):
        end = TACO_SIZE if size < 0 else min(TACO_SIZE, self.position+size)
        if end <= self.position: return b''
        content = fetch(TACO_URL, self.position, end)
        self.position = end
        return content


def archive_member(entry):
    offset, size = entry['offset'], entry['compressed_size']
    content = fetch(TACO_URL, offset, min(TACO_SIZE, offset+size+4096))
    header = struct.unpack('<4s5H3I2H', content[:30])
    if header[0] != b'PK\x03\x04' or header[2] & 1:
        raise ValueError('Unsupported/encrypted ZIP member')
    start = 30 + header[-2] + header[-1]
    if start+size > len(content):
        content = fetch(TACO_URL, offset, offset+start+size)
    compressed = content[start:start+size]
    if header[3] == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    elif header[3] == zipfile.ZIP_STORED:
        data = compressed
    else:
        raise ValueError('Unsupported ZIP compression')
    if len(data) != entry['size'] or zlib.crc32(data) != entry['crc']:
        raise ValueError('ZIP member integrity check failed')
    return data


def crop_box(bbox, width, height, padding=.12):
    if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
        raise ValueError('Invalid annotation bounding box')
    x, y, w, h = bbox
    if x < 0 or y < 0 or w < 24 or h < 24 or x+w > width+1 or y+h > height+1:
        raise ValueError('Tiny or out-of-bounds annotation')
    return (max(0, math.floor(x-w*padding)), max(0, math.floor(y-h*padding)),
            min(width, math.ceil(x+w*(1+padding))), min(height, math.ceil(y+h*(1+padding))))


def clean_crop(annotation, all_annotations, width, height):
    box = crop_box(annotation['bbox'], width, height)
    area = (box[2]-box[0])*(box[3]-box[1])
    for other in all_annotations:
        if other['id'] == annotation['id']: continue
        x, y, w, h = other['bbox']
        intersection = max(0, min(box[2], x+w)-max(box[0], x))*max(0, min(box[3], y+h)-max(box[1], y))
        if intersection > .08*area:
            raise ValueError('Crop contains another substantial labeled object')
    return box


def annotated_pixels(image, width, height):
    rgb = ImageOps.exif_transpose(image).convert('RGB')
    if rgb.size != (width, height):
        raise ValueError('Annotation/image dimension mismatch after EXIF correction')
    return rgb


def assign_splits(rows):
    """Split whole source-image groups, stratified by source category, before training."""
    if len({r['group'] for r in rows}) != len(rows):
        raise ValueError('Deduplicate source-image groups before assigning splits')
    categories = defaultdict(list)
    for row in rows: categories[(row['source'], row['source_category'])].append(row)
    for key, group in sorted(categories.items()):
        group.sort(key=lambda row: row['path'])
        random.Random('everyday-42-' + str(key)).shuffle(group)
        n = len(group)
        n_val = max(1, round(n*.15)) if n >= 5 else 0
        n_test = max(1, round(n*.15)) if n >= 5 else 0
        for index, row in enumerate(group):
            row['split'] = 'everyday_holdout' if index < n_test else 'validation' if index < n_test+n_val else 'train'
    return rows


def deduplicate(candidates, original, distance=4):
    """Quarantine overlap/label conflicts, then retain one example per new component."""
    old = [(row['sha256'], int(row['phash'], 16)) for row in original]
    retained, excluded = [], []
    for row in sorted(candidates, key=lambda row: row['path']):
        if any(row['sha256'] == sha or (int(row['phash'], 16)^phash).bit_count() <= distance for sha, phash in old):
            excluded.append(dict(path=row['path'], reason='overlap_with_existing_data'))
        else:
            retained.append(row)
    parents = list(range(len(retained)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for i, row in enumerate(retained):
        for j in range(i):
            other = retained[j]
            if (row['group'] == other['group'] or row['sha256'] == other['sha256']
                    or (int(row['phash'], 16)^int(other['phash'], 16)).bit_count() <= distance):
                parents[find(i)] = find(j)
    groups = defaultdict(list)
    for index, row in enumerate(retained): groups[find(index)].append(row)
    unique = []
    for group in groups.values():
        if len({row['label'] for row in group}) != 1:
            excluded.extend(dict(path=row['path'], reason='conflicting_near_duplicate_labels') for row in group)
        else:
            unique.append(group[0])
            excluded.extend(dict(path=row['path'], reason='new_near_duplicate') for row in group[1:])
    return unique, excluded


def main():
    directory = ROOT/'data/external/everyday'
    directory.mkdir(parents=True, exist_ok=True)
    source_manifest = ROOT/'artifacts/organic-manifest.json'
    original = read_json(source_manifest)
    index_path = directory/'taco_archive_index.json'
    if not index_path.exists():
        with zipfile.ZipFile(RemoteArchive()) as archive:
            entries = {info.filename: dict(offset=info.header_offset, compressed_size=info.compress_size,
                         size=info.file_size, crc=info.CRC) for info in archive.infolist() if not info.is_dir()}
        save_json(index_path, entries)
    entries = read_json(index_path)
    annotation_name = next(name for name in entries if name.endswith('/annotations.json'))
    annotation_path = directory/'taco_annotations.json'
    if not annotation_path.exists(): annotation_path.write_bytes(archive_member(entries[annotation_name]))
    annotations = read_json(annotation_path)
    categories = {c['id']: c['name'] for c in annotations['categories']}
    images = {im['id']: im for im in annotations['images']}
    per_image = defaultdict(list)
    for annotation in annotations['annotations']: per_image[annotation['image_id']].append(annotation)
    available, rejected = [], []
    for image_id, items in sorted(per_image.items()):
        im = images[image_id]
        # At most one crop per photo avoids training/holdout views of the same scene.
        for annotation in sorted(items, key=lambda a: -a['area']):
            category = categories[annotation['category_id']]
            if category not in TACO_MAPPING: continue
            try: box = clean_crop(annotation, items, im['width'], im['height'])
            except ValueError: continue
            available.append(dict(image=im, annotation=annotation, category=category, box=box))
            break
    random.Random(42).shuffle(available)
    selected, counts = [], Counter()
    for entry in available:
        if counts[entry['category']] >= 80: continue
        counts[entry['category']] += 1
        selected.append(entry)
    save_json(directory/'selection_plan.json', dict(seed=42, per_category_limit=80,
        taco_selected=dict(counts), image_ids=[e['image']['id'] for e in selected],
        mapping=TACO_MAPPING, annotation_sha256=digest(annotation_path),
        rule='Largest eligible isolated annotated object; one crop per source image; no classifier scores used'))
    print('TACO crop plan:', dict(counts), flush=True)

    def acquire_taco(entry):
        im, annotation = entry['image'], entry['annotation']
        target = directory/'taco_crops'/f"{im['id']:05}_{annotation['id']:05}.jpg"
        source = directory/'taco_originals'/f"{im['id']:05}.jpg"
        try:
            suffix = '/'+im['file_name']
            member = next(name for name in entries if name.endswith(suffix))
            expected = entries[member]
            cached = source.read_bytes() if source.exists() else b''
            if len(cached) != expected['size'] or zlib.crc32(cached) != expected['crc']:
                source.parent.mkdir(parents=True, exist_ok=True)
                temporary = source.with_suffix('.part')
                temporary.write_bytes(archive_member(expected))
                temporary.replace(source)
            with Image.open(source) as image:
                # TACO annotations follow the displayed (EXIF-corrected) image.
                rgb = annotated_pixels(image, im['width'], im['height'])
                crop = rgb.crop(entry['box'])
                target.parent.mkdir(parents=True, exist_ok=True)
                crop.save(target, quality=95)
                phash, size = str(imagehash.phash(crop)), list(crop.size)
            return dict(path=target.relative_to(ROOT).as_posix(), label=TACO_MAPPING[entry['category']],
                source='TACO v1.0', source_category=entry['category'], source_image=im['file_name'],
                source_image_sha256=digest(source), annotation_id=annotation['id'], crop_box=entry['box'],
                group='taco-image-'+str(im['id']), sha256=digest(target), phash=phash, size=size,
                attribution='Pedro F. Proenca and Pedro Simoes; TACO v1.0; CC BY 4.0; cropped for classification')
        except (OSError, HTTPException, ValueError, StopIteration, zlib.error) as error:
            return dict(failure=f"TACO image {im['id']}: {error}")

    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, row in enumerate(pool.map(acquire_taco, selected)):
            if 'failure' in row: rejected.append(row)
            else: rows.append(row)
            if (i+1)%40 == 0: print(f'TACO processed {i+1}/{len(selected)}; usable crops: {len(rows)}', flush=True)
    tree_path = ROOT/'artifacts/realwaste-tree.json'
    if not tree_path.exists():
        save_json(tree_path, json.loads(fetch(f'https://api.github.com/repos/sam-single/realwaste/git/trees/{REALWASTE_REVISION}?recursive=1')))
    tree = read_json(tree_path)
    if tree['sha'] != REALWASTE_REVISION or tree.get('truncated'):
        raise ValueError('RealWaste source inventory is incomplete or has a revision mismatch')
    vegetation = sample_entries([e for e in tree['tree'] if e['path'].startswith('RealWaste/Vegetation/')], 180)
    def acquire_plant(entry):
        target = directory/'realwaste_vegetation'/Path(entry['path']).name
        content = target.read_bytes() if target.exists() else b''
        if git_blob_sha(content) != entry['sha']:
            content = fetch(f'https://raw.githubusercontent.com/sam-single/realwaste/{REALWASTE_REVISION}/'+urllib.parse.quote(entry['path']))
            if git_blob_sha(content) != entry['sha']: raise ValueError('RealWaste blob checksum mismatch')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        with Image.open(target) as image:
            rgb = image.convert('RGB')
            return dict(path=target.relative_to(ROOT).as_posix(), label='dry_decomposable',
                source='RealWaste vegetation', source_category='Vegetation', group='realwaste-'+entry['sha'],
                sha256=digest(target), phash=str(imagehash.phash(rgb)), size=list(rgb.size),
                attribution='Sam Single, Saeid Iranmanesh, Raad Raad; CC BY-NC-SA 4.0 (authors repository)')
    with ThreadPoolExecutor(max_workers=6) as pool: rows.extend(pool.map(acquire_plant, vegetation))
    retained, duplicates = deduplicate(rows, original)
    retained = assign_splits(retained)
    save_json(ROOT/'artifacts/everyday-manifest.json', original+retained)
    provenance = dict(taco=dict(source='https://zenodo.org/records/3587843', license='CC BY 4.0',
        archive_url=TACO_URL, archive_size=TACO_SIZE, integrity='Selected ZIP members checked using CRC32 and byte length; per-image SHA-256 recorded. Full archive MD5 not checked because only selected byte ranges were downloaded.',
        authors='Pedro F. Proenca and Pedro Simoes', citation='TACO: Trash Annotations in Context for Litter Detection (2020)',
        mapping=TACO_MAPPING, exclusions='Unlabeled/ambiguous categories, batteries, aerosols, broken glass; tiny/crowded crops'),
        realwaste=dict(source='https://github.com/sam-single/realwaste', revision=REALWASTE_REVISION,
            license='CC BY-NC-SA 4.0 (authors repository)', mapping={'Vegetation':'dry_decomposable'},
            authors='Sam Single, Saeid Iranmanesh, Raad Raad'),
        previous_manifest_sha256=digest(source_manifest), retained=len(retained),
        splits={s:dict(Counter(r['label'] for r in retained if r['split']==s)) for s in ('train','validation','everyday_holdout')},
        categories=dict(Counter(r['source_category'] for r in retained)), excluded=duplicates, failures=rejected,
        limitations=['Material proxies do not determine actual moisture or local recycling eligibility.',
          'Perceptual hashing does not guarantee all related views are found; no live-camera accuracy claim.',
          'Existing test/validation assignments and images are preserved. New source groups are disjoint.'])
    save_json(ROOT/'artifacts/everyday_sources.json', provenance)
    print({k:provenance[k] for k in ('retained','splits','categories')}, flush=True)


if __name__ == '__main__': main()
