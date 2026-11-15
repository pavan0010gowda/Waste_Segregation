# Repository organization

The root launchers (`live_detect.py`, `demo.py`) use the current `wastewise/`
Python package. It holds the foreground gate, CNN/head inference, temporal
checks, simulator, image collection and reproducible offline ML tools. It is
not a duplicate copy of the project. `scripts/arduino_live.py` is the hardware
entry point and retains its existing protocol and behavior.

## Local archive

Earlier material is preserved under one ignored `legacy/` folder:

| Previous location | Local archive location |
| --- | --- |
| `backups/original_scripts/` | `legacy/original_scripts/` |
| `experiments/web_demo/` | `legacy/web_demo/` |
| Root web requirements and old full environment freeze | `legacy/web_demo/` |
| Web-only API, review database and unused routing helper modules | `legacy/web_demo/wastewise/` |
| Dashboard/API tests and remote-serving security notes | `legacy/web_demo/tests/`, `legacy/web_demo/SECURITY.md` |
| Earlier latency/report-generation tools | `legacy/offline_tools/` |
| Duplicate `hi.py` and `ld.py` launchers | `legacy/compatibility_scripts/` |
| Earlier comprehensive PDF | `legacy/Comprehensive_Waste_Classification_Project_Report.pdf` |

Archived code is a historical snapshot, not an independently installed package.
Its original import paths may need restoring before use. Do not run old training
scripts against the current models without separate output paths. The current
PDF, `Waste_Classification_Project_Explanation.pdf`, remains in the root.

## What Git includes

Source, tests, documentation, small measured-result records, dataset manifests,
attribution and `.gitkeep` data placeholders. Manifests contain relative file
paths, hashes, labels and splits; they are not the image data itself.

## What stays local

`legacy/`, Python environments, dataset images/archives/metadata downloads,
review databases, collected camera images, model weights, generated experiment
bundles, feature caches and detailed local inventory logs. The data guide is
versioned. Dataset files were neither deleted nor relocated during cleanup.
Documentation images such as the bin preview remain visible on GitHub.

The `.gitignore` rules apply to future additions. They do not erase history or
remove already tracked files. No dataset images were tracked in this checkout
at cleanup time. No Git history was rewritten and no commit or push was made.

## Verification commands

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m wastewise.dataset_inventory
git status --short
git check-ignore data/train/wet/example.jpg legacy/example.py
```

The current data inventory reports unaccounted, changed, missing or corrupt
images as issues. It never treats an unlabeled download or unreviewed camera
sample as a verified training example.
