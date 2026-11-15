"""Acquire attributed RealWaste food-organic images and build an augmented manifest."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import random
import urllib.parse
import urllib.request
from pathlib import Path
import imagehash
from PIL import Image, ImageOps
from .common import ROOT, read_json, save_json, digest

COMMIT = '156e8e9154eeb163df30fcfd767f55b19633bb06'


def main():
    tree_path = ROOT / 'artifacts/realwaste-tree.json'
    if not tree_path.exists():
        import json
        with urllib.request.urlopen(f'https://api.github.com/repos/sam-single/realwaste/git/trees/{COMMIT}?recursive=1', timeout=60) as response:
            save_json(tree_path, json.load(response))
    tree = read_json(tree_path)
    if tree['sha'] != COMMIT:
        raise ValueError('Unexpected source revision')
    files = [r for r in tree['tree'] if r['path'].startswith('RealWaste/Food Organics/') and r['path'].endswith('.jpg')]
    directory = ROOT / 'data/external/realwaste-food'
    directory.mkdir(parents=True, exist_ok=True)

    def download(entry):
        path = directory / Path(entry['path']).name
        if not path.exists():
            url = f'https://raw.githubusercontent.com/sam-single/realwaste/{COMMIT}/' + urllib.parse.quote(entry['path'])
            with urllib.request.urlopen(url, timeout=90) as response:
                content = response.read()
            if hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest() != entry['sha']:
                raise ValueError('Git blob checksum mismatch')
            path.write_bytes(content)
        return path
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(download, files))
    original = read_json(ROOT / 'artifacts/manifest.json')
    existing = [int(r['phash'], 16) for r in original]
    candidates, excluded = [], []
    for path in sorted(paths):
        with Image.open(path) as im:
            rgb = ImageOps.exif_transpose(im).convert('RGB')
            phash = str(imagehash.phash(rgb))
            size = list(rgb.size)
        if any((int(phash, 16) ^ value).bit_count() <= 4 for value in existing):
            excluded.append(str(path.relative_to(ROOT)))
            continue
        candidates.append(dict(path=path.relative_to(ROOT).as_posix(), label='wet', original_split='external',
                               sha256=digest(path), phash=phash, size=size, source='RealWaste Food Organics'))
    # Group external near duplicates transitively before assignment.
    parent = list(range(len(candidates)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, row in enumerate(candidates):
        for j in range(i):
            if (int(row['phash'], 16) ^ int(candidates[j]['phash'], 16)).bit_count() <= 4:
                parent[find(i)] = find(j)
    groups = {}
    for i, row in enumerate(candidates):
        groups.setdefault(find(i), []).append(row)
    representatives = []
    for group in groups.values():
        row = group[0]
        row['group'] = min(r['sha256'] for r in group)
        representatives.append(row)
        excluded.extend(r['path'] for r in group[1:])
    random.Random(42).shuffle(representatives)
    n = len(representatives)
    for i, row in enumerate(representatives):
        row['split'] = 'external_test' if i < round(n*.15) else ('validation' if i < round(n*.3) else 'train')
    save_json(ROOT / 'artifacts/augmented-manifest.json', original + representatives)
    save_json(directory / 'provenance.json', dict(source='https://github.com/sam-single/realwaste', commit=COMMIT,
              authors='Sam Single, Saeid Iranmanesh, Raad Raad', license='CC BY-NC-SA 4.0 (authors repository)',
              citation='RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning (2023)',
              mapping={'Food Organics': 'wet'}, downloaded=len(paths), retained=n, excluded=excluded,
              scope='Organic material proxy; image classification does not measure moisture.',
              splits={s:sum(r['split']==s for r in representatives) for s in ('train','validation','external_test')}))
    print(f'Downloaded {len(paths)}; retained {n}; splits: ' + str({s:sum(r['split']==s for r in representatives) for s in ('train','validation','external_test')}), flush=True)


if __name__ == '__main__':
    main()
