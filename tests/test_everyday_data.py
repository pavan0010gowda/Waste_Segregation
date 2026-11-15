import io
import zipfile

import pytest
from PIL import Image

from wastewise import expand_everyday as data


def row(name, label='dry_recyclable', phash='0000000000000000', category='Drink can'):
    return dict(path=name, sha256=name, label=label, phash=phash, group=name,
                source='TACO v1.0', source_category=category)


def test_crop_contains_padding_and_rejects_crowded_or_tiny_regions():
    a=dict(id=1,bbox=[20,20,80,80])
    assert data.clean_crop(a,[a],120,120)==(10,10,110,110)
    b=dict(id=2,bbox=[40,40,40,40])
    with pytest.raises(ValueError,match='another substantial'):
        data.clean_crop(a,[a,b],120,120)
    for box in ([0,0,2,2],[-1,0,50,50],[0,0,float('nan'),50],[110,0,80,80]):
        with pytest.raises(ValueError): data.crop_box(box,120,120)


def test_exif_is_applied_before_annotation_coordinates():
    image=Image.new('RGB',(100,60),'blue')
    image.putpixel((0,0),(255,0,0))
    image.getexif()[274]=6
    corrected=data.annotated_pixels(image,60,100)
    assert corrected.size==(60,100)
    assert corrected.getpixel((59,0))==(255,0,0)
    with pytest.raises(ValueError,match='dimension mismatch'):
        data.annotated_pixels(image,100,60)


def test_dedup_excludes_old_overlap_and_quarantines_label_conflicts():
    original=[row('old')]
    candidates=[row('old-copy'),row('a',phash='ffffffffffffffff'),
                row('b',label='wet',phash='fffffffffffffffe'),
                row('unique',phash='ff00ff00ff00ff00')]
    kept,excluded=data.deduplicate(candidates,original)
    assert [r['path'] for r in kept]==['unique']
    assert {r['reason'] for r in excluded}=={'overlap_with_existing_data','conflicting_near_duplicate_labels'}


def test_component_dedup_is_transitive_and_source_groups_are_disjoint():
    candidates=[row('a',phash='0000000000000000'),row('b',phash='000000000000000f'),
                row('c',phash='00000000000000ff')]
    kept,excluded=data.deduplicate(candidates,[])
    assert len(kept)==1 and len(excluded)==2
    rows=[row(str(i)) for i in range(20)]
    split=data.assign_splits(rows)
    assert len({r['group'] for r in split})==20
    assert {s:sum(r['split']==s for r in split) for s in ('train','validation','everyday_holdout')}==dict(train=14,validation=3,everyday_holdout=3)
    first={r['path']:r['split'] for r in data.assign_splits([dict(r) for r in reversed(rows)])}
    second={r['path']:r['split'] for r in data.assign_splits([dict(r) for r in rows])}
    assert first==second
    with pytest.raises(ValueError,match='source-image groups'):
        data.assign_splits([row('same'),row('same')])


def test_remote_archive_member_verifies_integrity_without_extracting_paths(monkeypatch):
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('data/example.txt',b'verified image bytes')
    content=archive.getvalue()
    with zipfile.ZipFile(io.BytesIO(content)) as z: info=z.infolist()[0]
    entry=dict(offset=info.header_offset,compressed_size=info.compress_size,size=info.file_size,crc=info.CRC)
    monkeypatch.setattr(data,'TACO_SIZE',len(content))
    monkeypatch.setattr(data,'fetch',lambda url,start,stop:content[start:stop])
    assert data.archive_member(entry)==b'verified image bytes'
    with pytest.raises(ValueError,match='integrity'):
        data.archive_member({**entry,'crc':0})
