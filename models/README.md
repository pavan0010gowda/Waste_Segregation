# Local model bundles

Weights stay on the development laptop and are ignored by Git. No public model
download is currently configured. A fresh clone needs a trusted copy of the
required bundle or a reproduced training run before the camera can predict.

| Camera option | Required local files |
| --- | --- |
| `selective` (default) | Run named by `artifacts/live_model.json`: `features.keras`, `head.npz`, `metadata.json` |
| `previous` | Bundle named by that file's `previous_run`, with the same three files |
| `original` | `models/waste_classifier_mobilenet.h5`, four outputs in the documented label order |
| `improved` | `artifacts/runs/mobilenet/model.keras` and `metadata.json` |

Keep each model and its matching metadata together; loaders check hashes and
labels. Do not substitute the preserved eight-output `waste_classifier.h5` for
the four-class MobileNet. A hash checks against trusted metadata; it does not
establish the origin of an unknown downloaded bundle.

Training instructions are in [the data guide](../data/README.md). Original models,
previous bundles and the current deployed bundle remain unchanged by cleanup.
