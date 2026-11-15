# Everyday-item coverage

This page describes the 976-image public-data expansion. A later full-folder
audit also found two previously omitted local AVIF training images; see
[DATA_INVENTORY.md](DATA_INVENTORY.md) for complete file accounting and the
separate retraining comparison. Earlier evaluation splits remain unchanged.

The classifier still returns the project's four material categories. The public
dataset expansion adds more examples of common objects and backgrounds; it does
not turn the program into a universal object recognizer. The exact counts and
measured before/after results are in [LIVE_RESULTS.md](LIVE_RESULTS.md).

This expansion retained **976 images across 24 source item types**: **684 training**,
**146 validation** and **146 everyday holdout**. Of those retained images, 796 are
TACO crops and 180 are RealWaste vegetation photos. Two near-duplicates were
excluded; there were no final download failures. The resulting training set has
3,890 images, with 963 validation images. The old 922-image retrospective test,
61-image RealWaste organic holdout and 90-image produce holdout are preserved.

The new everyday holdout represents **three** categories: 27 decomposable,
59 non-degradable and 60 recyclable images. TACO contributes only three isolated
food-waste crops in this subset, all used for training; the separate earlier
organic holdouts remain the source of organic evaluation evidence.

| Project category | Additional examples |
| --- | --- |
| Dry recyclable | Plastic/glass bottles, drink/food cans, paper, cartons, paper bags, metal caps and lids |
| Dry non-degradable | Wrappers, crisp packets, plastic film, straws, plastic utensils, foam and cigarette litter |
| Dry decomposable | Plant material from RealWaste Vegetation |
| Wet / organic | Existing food-organic and produce collections, plus a small number of TACO food-waste crops |

These are **material proxies** chosen to match the inherited taxonomy. A material
class alone does not establish whether an item is accepted by a local recycler,
and a photo cannot measure moisture. TACO battery, aerosol and unlabeled/ambiguous
categories are excluded because they lack a reliable mapping to these four labels.

## How the expansion is prepared

- [TACO v1.0](https://zenodo.org/records/3587843), by Pedro F. Proenca and Pedro
  Simoes, supplies manually annotated litter images under CC BY 4.0. Its published
  archive is read in byte ranges; selected ZIP members are checked against their
  stored CRC32 and length. The full 2.7 GB archive is not required. Image SHA-256
  values and annotation IDs are recorded for the retained subset.
- EXIF orientation is corrected **before** applying the source bounding boxes.
  Each crop has 12% padding. Tiny boxes and crops containing another substantial
  annotated object are excluded. At most one crop is kept per source photo.
- [RealWaste](https://github.com/sam-single/realwaste), by Sam Single, Saeid
  Iranmanesh and Raad Raad, adds vegetation from a pinned revision. The authors'
  repository specifies CC BY-NC-SA 4.0. Downloaded images are checked against Git
  blob hashes. This is the same source used for the earlier food-organic expansion.
- Overlap with **all** existing data is screened using SHA-256 and perceptual
  hashes. Conflicting near-duplicate labels are quarantined; one representative
  remains per new duplicate component. Existing images and splits stay unchanged.
- New source groups are assigned to training, validation and an everyday holdout
  before training. Groups are disjoint. Perceptual hashing is a heuristic and may
  miss related views; these are not independently collected live-camera sessions.
- The frozen previous model and the candidate use the **same expanded validation
  set**. Validation macro F1 determines promotion; ties keep the previous model.
  Only after that decision is saved are holdout/test results computed.

The source record is `artifacts/everyday_sources.json`; the expanded inventory is
`artifacts/everyday-manifest.json`. Download failures and exclusions are recorded,
not silently counted as successful additions. Data and model binaries remain
local and excluded from Git.

## Reproduce the public-data experiment

After preparing the earlier audited/organic datasets and baseline bundle:

```powershell
.venv\Scripts\python.exe -m wastewise.expand_everyday
.venv\Scripts\python.exe -m wastewise.train_selective --backbone efficientnet --manifest artifacts/everyday-manifest.json --output artifacts/runs/selective_everyday --baseline-run artifacts/runs/selective_organic
.venv\Scripts\python.exe -m wastewise.select_live
```

The CNN feature extractor stays frozen; the balanced classifier head, probability
calibration and training-example support thresholds are fitted on the expanded
data. This is real retraining of the classification stage, not a claim that the
entire CNN was trained from scratch. The previous bundle remains available through
`demo.py --model previous`.

## When your own item is still wrong

Different cameras, hands, lighting, mixed materials and unrepresented objects can
still cause errors. Flag a wrong demo prediction and choose the correct bin.
For deliberate local example collection, run:

```powershell
.venv\Scripts\python.exe live_detect.py --collect
```

Use keys **1–4** for your verified category, or **0** for an unknown/non-waste
example. Collect different views across separate camera sessions. Only an explicit
key press stores a crop. Those crops await curation and retraining; saving them does
not instantly teach the running model. Keep entire sessions separate for future
evaluation, rather than distributing adjacent frames across training and test.
