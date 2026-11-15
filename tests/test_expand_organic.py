from wastewise.expand_organic import sample_entries, screen_candidates, FOLDER_SPLITS, git_blob_sha


def row(name, phash, split, folder=None):
    return dict(path=name, phash=phash, sha256=name, label='wet', split=split,
                source_group=folder or name, group=folder or name)


def test_sampling_spans_filename_sequence():
    entries = [dict(path=f'{i}_100.jpg', type='blob') for i in range(100)]
    assert [e['path'] for e in sample_entries(entries, 3)] == ['0_100.jpg', '49_100.jpg', '99_100.jpg']


def test_folder_assignments_disjoint_and_holdout_not_training():
    assert len(FOLDER_SPLITS) == 12
    assert list(FOLDER_SPLITS.values()).count('train') == 8
    assert FOLDER_SPLITS['Peach 1'] == FOLDER_SPLITS['Plum 1'] == 'organic_holdout'


def test_cross_split_transitive_component_is_removed():
    # A~B, B~C, but A not near C. All must be removed across split boundaries.
    candidates = [row('a','0000000000000000','train'),
                  row('b','000000000000000f','train'),
                  row('c','00000000000000ff','organic_holdout')]
    kept, exclusions = screen_candidates(candidates, [])
    assert kept == []
    assert len(exclusions) == 3
    assert all(e['reason'] == 'cross_split_component' for e in exclusions)


def test_legacy_overlap_excluded_and_duplicate_component_collapsed():
    original = [dict(phash='ffffffffffffffff',label='dry_recyclable')]
    candidates = [row('overlap','fffffffffffffffe','train'),
                  row('a','0000000000000000','train','banana'),
                  row('b','000000000000000f','train','banana')]
    kept, exclusions = screen_candidates(candidates, original)
    assert len(kept) == 1 and kept[0]['path'] == 'a'
    assert next(e for e in exclusions if e['path']=='overlap')['conflicting_label']
    assert len(original) == 1  # Original records are never changed.


def test_git_blob_checksum_is_git_not_plain_file_sha():
    assert git_blob_sha(b'test\n') == '9daeafb9864cf43055ae93beb0afd6c7d144bfa4'
