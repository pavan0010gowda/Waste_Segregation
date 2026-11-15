# Dataset layout and reproduction

Git includes this guide and `.gitkeep` placeholders. Images, downloaded archives,
source metadata, review databases and personal camera samples stay local. Existing
data files were not moved into the legacy archive or deleted during cleanup.

```text
data/
  train/{dry_decomposable,dry_non_degradable,dry_recyclable,wet}/
  test/{dry_decomposable,dry_non_degradable,dry_recyclable,wet}/
  external/           Downloaded public sources and derived crops
  camera_samples/     Explicitly saved crops awaiting human curation
  review/             Earlier local review database
```

## The manifest determines each image's role

`artifacts/all-data-manifest.json` records paths, labels, SHA-256, similarity hashes
and splits. Folder names describe original storage; some `data/train` images are
validation examples or excluded duplicates. The trainer uses manifest assignments,
not a recursive folder loader that accidentally includes test data.

The latest candidate fits **3,892 train** images and uses **963 validation** images
for model selection/calibration. Evaluation remains separate: **922 historical
test**, **146 everyday holdout**, **61 RealWaste organic**, **90 produce holdout**.
The inventory accounts for all 10,064 files with image extensions: 10,063 decoded
images and one macOS archive metadata sidecar. This includes source photos,
duplicates and ambiguous external categories; see
[the complete audit](../docs/DATA_INVENTORY.md).

```powershell
.venv\Scripts\python.exe -m wastewise.dataset_inventory
.venv\Scripts\python.exe scripts/train.py --output artifacts/runs/my_candidate --baseline-run artifacts/runs/selective_everyday
```

For a new folder-labelled addition, create a **separate** manifest after checking
its labels. Existing assignments stay fixed; near-duplicates require curation:

```powershell
.venv\Scripts\python.exe -m wastewise.dataset_inventory --manifest artifacts/all-data-manifest.json --include-unlisted artifacts/local/next-manifest.json
```

Only new images under the four original `train`/`test` class directories can be
included this way. New `test` images stay test. Camera samples and arbitrary
downloads never become training images automatically. Related frames from one
camera session must stay in one split. GIF/TIFF with multiple frames require
explicit frame selection. Pillow 12+ provides AVIF support in the environment.

An audit does not retrain or promote a model. Training writes its own bundle and
selection decision. Validation ties retain the deployed baseline. Read the
candidate's `selection_decision.json` and metrics before changing deployment.

## Restore or rebuild on another computer

1. Install the root requirements in an isolated Python environment.
2. Restore the original labelled `train`/`test` dataset from your own copy. The
   original source download and redistribution rights are not established by this
   repository; no downloadable copy is promised. Keep original names and bytes to
   reproduce the published manifest hashes.
3. Restore the public-source folders from your local data copy, or use the pinned
   acquisition modules below. Keep attribution and source licenses. Published
   manifests record the exact selected images; downloading extra images alone
   does not assign them a training role.
4. Run the complete inventory to verify hashes, decodability, coverage and splits.
5. Restore trusted model bundles following `models/README.md`, or train a separate
   candidate. The existing `--baseline-run` command requires that baseline bundle.
   Without it, omit the flag to fit a candidate; this does not establish improvement
   over a missing model and does not update the camera pointer automatically.

Public-data preparation (with the original audited manifest present):

```powershell
.venv\Scripts\python.exe -m wastewise.expand
.venv\Scripts\python.exe -m wastewise.expand_organic
.venv\Scripts\python.exe -m wastewise.expand_everyday
.venv\Scripts\python.exe -m wastewise.external fetch
```

These commands write acquisition manifests and preserve downloaded files. When
working with the original local dataset, keep the checked-in frozen manifests
for reproduction. A new dataset requires a new audit output, for example
`python -m wastewise.audit --output artifacts/local/new_audit`; do not overwrite
the published split records and then compare scores as if the benchmark were
unchanged.

## Sources and labels

| Source | Role and recorded provenance |
| --- | --- |
| Original local four-class data | Audited manifest; two AVIF additions recorded in `artifacts/all_data_additions.json` |
| RealWaste Food Organics | Organic examples; pinned revision and exclusions in `artifacts/dataset_sources.json` |
| Fruits-360 | Group-separated produce examples; `artifacts/organic_sources.json` |
| TACO v1.0 / RealWaste vegetation | Annotation crops and vegetation; `artifacts/everyday_sources.json` |
| TrashNet | External material benchmark, not training; `artifacts/external_trashnet.json` |

TACO source photos contain scene context and sometimes multiple objects; their
derived crops are the classification examples. Adding both as independently
labelled examples would duplicate evidence and can assign incorrect labels.
TrashNet's `trash` category has no reliable mapping to these four labels and is
kept for later curation. Duplicates and conflicting labels stay excluded.

Source URLs, authors, license declarations and revisions are recorded in the
provenance files. RealWaste includes non-commercial/share-alike terms; Fruits-360
and TACO have their own attribution/share-alike conditions. This project does not
relicense the source images. Material classes do not establish actual moisture,
whether an item should be discarded, or local recycling eligibility.
