# Research and design decisions

This records the earlier research and baseline experiments. The current camera
uses EfficientNetV2B0 with a validation-selected RBF SVM; see
[LIVE_DETECTION.md](LIVE_DETECTION.md). The old web review database and API are
archived locally under `legacy/web_demo/`; the live simulator's Flag button
changes the pending route and does not automatically store a training label.

Sources reviewed for this rebuild (14 November 2026). This is a targeted review,
not a claim to have inspected every online project or established a new state of the art.

| Source | What it contributes | Decision |
| --- | --- | --- |
| [TrashNet, Gary Thung and Mindy Yang](https://github.com/garythung/trashnet) | Small six-material classification dataset, controlled backgrounds | Download with provenance; screen against local data before external evaluation |
| [TACO, Proença and Simões](https://github.com/pedropro/TACO) / [paper](https://arxiv.org/abs/2003.06975) | Litter detection and segmentation in context | Useful future multi-object detection benchmark; not interchangeable with image-level classification |
| [RealWaste, UCI](https://archive.ics.uci.edu/dataset/908/realwaste) / [authors' repository](https://github.com/sam-single/realwaste) | 4,752 images, nine materials, real landfill conditions | Added its 411 Food Organics images with pinned revision and duplicate screening; reserve 61 unique examples as an organic-only holdout. Authors' repository states CC BY-NC-SA 4.0 |
| [WasteVision](https://github.com/Hydrosamueldc/WasteVision) | Comparable educational six-class classification application | A demo alone is insufficient differentiation; emphasize audit, review, calibration and reproducible evidence |
| [TensorFlow MobileNetV2 preprocessing](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/preprocess_input) | ImageNet backbone expects pixels scaled to [-1, 1] | Embed preprocessing into saved model, so all clients use the same contract |
| [TensorFlow GPU installation](https://www.tensorflow.org/install/pip) | Modern TensorFlow GPU support on Windows uses WSL2 | Provide an isolated WSL setup for the RTX 3050, batch size 8 and memory growth |

## Why the new pipeline is useful

Existing scripts combine a few common sources of misleading results: hard-coded
class order, train/test preprocessing mismatch, selection using the test set,
and aggregate accuracy on imbalanced classes. We address these directly with
saved metadata, model-integrated preprocessing, duplicate-aware manifests,
validation-only selection, macro F1, confusion matrices and confidence intervals.

Frozen ImageNet embeddings plus balanced logistic regression provide a fast,
reproducible baseline. Four regularization settings are compared on validation
macro F1. Temperature is fitted on validation probabilities. This is an established
method, not a newly invented algorithm. Reusing validation for selection and
calibration is disclosed; a larger dataset should have a separate calibration split.

The most useful differentiator is the entire evidence and review lifecycle:
questionable images are quarantined, uncertain predictions can be reviewed,
corrections are stored without mutating the deployed model, and all reported
results have explicit scope. Neither softmax confidence nor an 80% cutoff proves
an image is in distribution. The cutoff is a configurable policy, not a certified
precision target.

## Label limitations

The inherited categories overlap conceptually: paper can be both decomposable
and recyclable; material appearance does not establish moisture. Existing images
need human label review before stronger claims. Visual inspection of original
training examples shows paper/cardboard in dry_recyclable, and bark, leaves and
coconut husks in dry_decomposable. Thus all five TrashNet material labels map to
dry_recyclable; trash is excluded. An initial mapping of paper/cardboard to
dry_decomposable was incorrect and was corrected after taxonomy inspection.
This test measures recyclable recall only, not four-class field performance or
municipal sorting compliance. It is not an independent material-discrimination benchmark.

Do not merge public datasets blindly. Retain provenance, licenses, scene groups
and source-specific holdouts. Source-grouped splitting is preferable when capture
session metadata exists. pHash does not discover every augmented copy or shared scene.

## Diagnosed organic-data bias

The first experiment scored 99.69% on validation but 80.48% on the historical test
set. A systematic contact-sheet sample revealed pumpkins in organic training and
tomatoes in organic test. This is a domain/category bias, not something to solve
by reporting the better split. RealWaste food-organic augmentation was added after
this diagnosis; subsequent historical-test results are therefore exploratory.
The original experiment remains under `artifacts/runs/original_features/`.
