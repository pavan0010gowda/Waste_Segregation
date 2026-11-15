# Complete local dataset inventory

Audited on 15 November 2026. All **10,064 files with image extensions** were read and hashed: **10,063 decoded images** plus **one macOS metadata sidecar**. Total storage is 2,651,120,863 bytes (about 2.65 GB). No data image was deleted, moved or rewritten.

| Role | Files |
| --- | ---: |
| Training (latest candidate) | 3,892 |
| Validation / calibration | 963 |
| Historical test | 922 |
| Everyday holdout | 146 |
| RealWaste organic holdout | 61 |
| Produce holdout | 90 |
| Original duplicate copies | 316 |
| Conflicting original labels | 2 |
| Recorded duplicate / overlap exclusions | 564 |
| TACO originals used to prepare crops | 798 |
| TrashNet external material benchmark | 2,172 |
| TrashNet trash: mapping requires curation | 137 |
| macOS archive metadata sidecar (not an image) | 1 |

**Unaccounted files: 0. Missing manifest files: 0. Changed manifest images: 0.**

The 564 explicit exclusions comprise 342 Fruits-360 near-duplicate/overlap exclusions, 2 RealWaste Food Organics exclusions, 2 TACO crop exclusions and 218 TrashNet overlap exclusions. The 316 original duplicate rows and 2 conflicting labels are separate counts. All 798 TACO original photographs remain available as crop sources; the 798 crops were separately screened, with 796 retained.

Validation and holdouts are used for selection/evaluation, not fitted as training examples. TrashNet remains an external benchmark; its archived result concerns an earlier model and is not a current-model accuracy claim. Its 137 ambiguous trash images await mapping. Raw source photographs and duplicate copies do not provide independent extra training evidence.

## Two images were genuinely omitted

The old audit only recognized JPEG, PNG, WebP and BMP. Two AVIF photographs of dried plant material were therefore missing. Both decode successfully, match their existing dry-decomposable folder label on inspection, and have no pHash match within distance 4 of the old manifest. AVIF, GIF and TIFF extensions are now included in the audit. Multi-frame images require explicit curation.

The new `artifacts/all-data-manifest.json` preserves all 6,390 previous rows and adds these two training rows. It has 6,074 active examples across training/validation/evaluation, plus 318 duplicate/quarantine rows. The exact additions and hashes are in `artifacts/all_data_additions.json`. Rights for inherited images are not established, so the images remain local.

## Retraining result

A separate EfficientNetV2B0-feature candidate was trained with **3,892 training images**. All 6,074 active image hashes were verified, split groups checked, and nine classifier configurations compared on the same 963 validation images. Cached features were reused; two new features were computed. The measured run took **103.25 seconds on CPU**, including reuse of existing feature caches. This is not a from-scratch CNN or GPU timing.

| Measurement | Existing default | Complete-data candidate |
| --- | ---: | ---: |
| Validation macro F1 | 0.9699379 | 0.9699379 |
| Validation accuracy | 97.82% | 97.82% |
| Everyday holdout accuracy (146 images) | 84.25% | 84.25% |
| Historical accuracy (922 images) | 92.73% | 92.73% |
| Historical macro F1 | 0.8867332 | 0.8859384 |

**The validation score tied, so the existing default was retained.** The decision was recorded before holdout evaluation. The candidate includes every eligible training image; this does not mean the deployed model magically gained accuracy. Existing `selective`, `previous`, `improved` and `original` camera options still refer to their earlier models. The extra candidate is saved locally at `artifacts/runs/selective_all_data/`; its compact public evidence is `artifacts/all_data_comparison.json`.

## Repeat the audit

```powershell
.venv\Scripts\python.exe -m wastewise.dataset_inventory
```

This fully reads and decodes the data again, verifies manifest hashes, and writes `artifacts/dataset_inventory_summary.json` plus the ignored per-file report `artifacts/local/dataset_inventory.json`. New unexplained images, corruption, changed bytes or missing manifest files are reported as issues. Run the separate inclusion/training workflow in [the data guide](../data/README.md) after reviewing new labels.

[Current deployment results](LIVE_RESULTS.md) | [Repository layout](REPOSITORY.md)
