# Experiment records

These small text records explain and reproduce the current ML workflow. They are
kept in Git; images, generated feature caches, run bundles and local logs are not.

| Record | Purpose |
| --- | --- |
| `live_model.json`, `live_metrics.json` | Deployed model pointer and its measured results |
| `all-data-manifest.json` | Complete manifest for the latest training candidate |
| `all_data_additions.json`, `all_data_comparison.json` | Two omitted AVIF additions and the tied-validation retraining result |
| `dataset_inventory_summary.json` | Accounting for every local image-like file |
| `cleanup_verification.json` | Checksums confirming preserved camera, simulator, Arduino and original bundles |
| `manifest.json`, `augmented-manifest.json`, `organic-manifest.json`, `everyday-manifest.json` | Frozen stages of the dataset; retained to reproduce earlier comparisons and source acquisition |
| `audit.json`, `*_sources.json` | Duplicate/conflict evidence, source revisions, mappings and attribution |
| `baseline*.json`, `external_trashnet.json`, `published_metrics.json`, `latency.json` | Earlier measured baselines; not current live-camera accuracy claims |
| `live_camera_smoke.json`, `live_integration_checks.json` | Earlier camera/integration smoke checks, not labelled field benchmarks |

`runs/`, `cache/`, `local/` and `smoke/` are local-only. Run folders include weights,
model metadata, per-image predictions and validation-first decisions. Never mix
metadata from one bundle with another bundle's weights. See
[the data guide](../data/README.md) and [current results](../docs/LIVE_RESULTS.md).
