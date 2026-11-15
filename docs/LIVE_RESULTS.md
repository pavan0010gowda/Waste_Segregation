# Live detector experiment results

Selected experiment: `artifacts/runs/selective_everyday`; head: `balanced_rbf_ovr_C=10`.

Backbone and head selected using validation macro F1 only. The historical test set was previously examined during development; these are retrospective results, not a pristine field benchmark.

| Classifier | Historical accuracy | Macro F1 |
| --- | ---: | ---: |
| Original MobileNet | 80.59% | 0.709 |
| Previous transfer model | 83.95% | 0.760 |
| EfficientNet before produce expansion | 90.02% | 0.860 |
| Produce-expanded previous model | 96.96% | 0.947 |
| Selected classifier, all images | 92.73% | 0.887 |

Historical test images: **922**. The selected classifier's rejection gate accepts **71.91%** of these; accepted accuracy is **99.25%**. Rejections are not removed from the all-image accuracy above.

These selective scores include confidence, class margin and training-neighbor support. They exclude foreground localization, low-detail crop filtering and temporal confirmation. They do not measure live-camera accuracy.

| True class | Total | Accepted | Correct among accepted |
| --- | ---: | ---: | ---: |
| dry_decomposable | 151 | 122 | 119 |
| dry_non_degradable | 16 | 16 | 16 |
| dry_recyclable | 555 | 524 | 523 |
| wet | 200 | 1 | 0 |

Class-specific acceptance matters: a high accepted accuracy can hide a class that is mostly rejected. Thresholds were fitted on whole-image validation; foreground crops introduce a different input distribution.

## Validation selection

| Experiment | Validation macro F1 | Head |
| --- | ---: | --- |
| artifacts/runs/selective_organic | 0.85326 | balanced_logistic_C=0.1 |
| artifacts/runs/selective_everyday | 0.96994 | balanced_rbf_ovr_C=10 |

## Additional organic data and untouched source-folder holdout

Fruits-360: 840 sampled images; 498 retained after duplicate screening, with 306 added to training, 102 to validation and 90 peach/plum images held out by source folder. Attribution and CC BY-SA 4.0 terms are recorded in `artifacts/organic_sources.json`. Produce is an organic-material proxy, not a measurement of moisture or proof that food should be discarded.

New peach/plum holdout: 100.00% raw organic recall across 90 images; selective coverage 68.89%. This is one class, not four-class accuracy.

Existing RealWaste organic holdout: 100.00% raw organic recall across 61 images; selective coverage 80.33%.

The candidate and frozen baseline were compared on exactly the same expanded validation set. The deployment recommendation was written before holdout/test evaluation. Historical results are still exploratory because the data expansion followed historical error analysis.


## Everyday-object expansion

Added 976 audited examples from TACO v1.0 and RealWaste vegetation. Added training images: 684; validation: 146; separate everyday holdout: 146.

Includes bottles, cans, paper/cardboard, packaging film/wrappers, straws, foam and vegetation. TACO uses isolated annotation crops; only one crop per source image is retained. Near-duplicate overlap with existing data is excluded before split assignment. Old images and assignments remain intact.

| Model | New everyday holdout accuracy | Macro F1 |
| --- | ---: | ---: |
| Frozen previous model | 44.52% | 0.345 |
| Retrained everyday model | 84.25% | 0.863 |

Everyday holdout macro F1 averages only its represented classes: dry_decomposable, dry_non_degradable, dry_recyclable. It is a source-image test, not a live-camera benchmark. Promotion uses common validation data, never the holdout scores above.

**Tradeoff:** historical 922-image accuracy changes from 96.96% to 92.73%. The default is chosen for the broader validation distribution. Run `demo.py --model previous` to compare the preserved earlier bundle.

Source licenses, mappings, hashes, exclusions and download failures are recorded in `artifacts/everyday_sources.json`. Material labels do not prove local recycling eligibility. Unrepresented items can still be misclassified; human review remains useful.


## Camera behavior

Fixed-camera background calibration, foreground bounding boxes, crop-only inference, uncertainty rejection and three-frame label confirmation are integrated in the OpenCV program. B recalibrates an empty view; Q exits. Boxes are classical foreground proposals, not trained semantic detections.

The actual webcam acquisition check processed 90 frames, saved none, and reported 58 no-object frames after calibration. This was not a labeled field evaluation. Regression tests cover synthetic empty walls, stationary objects, removal, lighting changes, multiple objects, crop-only classification and simulated Arduino gating.

The original model is unchanged. No images were silently captured for training and no physical motors were activated.

## Evidence-backed resume bullet

- Built an OpenCV waste-segregation pipeline with foreground localization and uncertainty rejection; compared pretrained CNN backbones and classifier configurations, improving retrospective accuracy from 80.59% to 92.73% and macro F1 from 0.709 to 0.887 on 922 audited images.

This describes classifier metrics on the historical image set, not measured localization accuracy, live accuracy or a novel research architecture.
