# Object-aware OpenCV waste detection

## Operation

Run `python live_detect.py`. Hold the camera fixed and keep the view empty during
the first 30 stable frames. The program then shows a bounding box for one separated
foreground item. It clears the box and classification when the item leaves.
Press **B** with an empty view after repositioning the camera; **Q** exits.
No website, browser or hosted service is involved.

For the two-window hardware-free demonstration, run `python demo.py` or
`python live_detect.py --demo`. The second window animates four bins and a sliding
lid, gives five seconds to flag a prediction, and accepts a clicked/numbered
human correction. This mode uses a separately exposed raw guess over 50% after
three consistent foreground frames. It does not change the strict `status` or
`label` used by the hardware path. The camera marks these relaxed predictions as
**DEMO guesses**. See [demo controls and recording walkthrough](DEMO.md).

## Implemented algorithm

1. **Empty-background model:** use the per-pixel median of stable calibration
   frames. Estimate pixel noise with median absolute deviation. Calibration restarts
   when substantial movement is detected. The background is fixed afterwards, so
   an item held still remains foreground.
2. **Foreground extraction:** compensate for a robust global color/exposure offset;
   threshold changes using the measured noise floor; suppress modest neutral
   shadows and tiny speckles; clean the mask with morphological operations.
3. **Object proposal:** find connected contours, reject very small changes, reject
   multiple substantial components, and check for large camera/scene changes.
   Return a bounding box and padded crop only after three spatial confirmations.
4. **CNN features and trained head:** run the chosen classifier on the crop, not the
   full webcam view. Offline experiments compare balanced logistic regression,
   one-vs-rest RBF SVM and cosine neighbors using validation macro F1 only.
5. **Rejection:** use calibrated class confidence, the gap between the first two
   classes, and similarity to actual training examples of the predicted class.
   Flat crops are also rejected. There is no guaranteed unknown-object detection.
6. **Temporal confirmation:** require three consecutive accepted predictions with
   the same class; reject a weak current frame immediately and reset history on
   removal, scene changes, multiple objects, rejection or a class switch.

When training-neighbor support is the **only** rejected criterion, three consistent
high-confidence predictions can display an amber **Possible category — UNVERIFIED**
suggestion. It remains `status=uncertain`, `label=None`; it never authorizes an
Arduino command. Low confidence, low detail, ambiguous classes and empty scenes
do not receive this suggestion. The display is a manual review aid, not a way to
count rejected cases as accepted.

This is a custom implementation and integration of established methods, not a
claim of a new research algorithm. A new algorithm name does not establish accuracy.

## Dataset and evaluation

See [executed classifier comparison](LIVE_RESULTS.md) for the selected backbone,
raw accuracy, rejection coverage and per-class acceptance. The run selection is
recorded in `artifacts/live_model.json`; `artifacts/live_metrics.json` is a compact
copy of the selected run's actual metrics.

The everyday-item experiment uses the audited expanded dataset: 3,890 training
images, 963 validation images, 922 retrospective test images, a 61-image organic
holdout, a 90-image produce holdout and a 146-image everyday-item holdout. It includes
the earlier attributed RealWaste food-organic and Fruits-360 data, plus annotated
TACO crops and RealWaste vegetation. See [coverage and preparation](DATA_COVERAGE.md).
No new captures or automatically guessed labels are silently added to training.
Dataset provenance and licenses remain in `artifacts/dataset_sources.json`.

Keep **raw accuracy**, **accepted accuracy**, and **coverage** separate. Rejecting
hard examples can raise accepted accuracy without improving raw accuracy. Per-class
acceptance counts are required: a system that rejects most organic examples is
not equally useful for all four classes. The old test set has already been examined,
so these results are retrospective. Whole-image accuracy is not a live-crop accuracy
measurement, and synthetic foreground tests are not a real-world detection benchmark.

## Limits

- The boxes mark foreground changes; this is not a trained semantic detector such
  as a waste-specific YOLO model. The existing dataset lacks box annotations.
- Hands or other non-waste foreground objects can create boxes. Place one item in
  a clear sorting area and withdraw your hand. Similarity rejection can still fail.
- An object present during calibration becomes part of the background. Empty the
  scene and press B if this happens.
- Transparent, low-contrast and reflective objects, hard shadows, camera motion,
  changing lights, or clutter may cause missed or incorrect proposals.
- Wet/dry is a material proxy in the vision dataset. Physical moisture still comes
  from the Arduino sensor in the autonomous workflow.

For stronger camera accuracy, collect independently labeled sessions in the actual
camera/tray setup, include empty scenes and non-waste objects, annotate bounding
boxes if training a semantic detector, and hold out entire capture sessions.

The `--collect` camera option now supports deliberate labeled crop capture:
keys 1–4 correspond to the four categories in displayed order; 0 means unknown.
No frames are stored automatically. Each crop has a JSON sidecar, SHA-256 and
session group for later curation and session-disjoint splitting. Capturing images
does not itself improve the running model or certify labels.

## Produce diversity expansion

The additional Fruits-360 experiment samples 840 images from 12 source folders
at pinned revision `bed1c24ee70de1569272de3882fbde88211c08b5`. Duplicate screening
retains 498: 306 training, 102 validation and 90 held-out peach/plum examples.
Whole folders stay in one split. The previous EfficientNet model is frozen and
evaluated on exactly the same expanded validation set before any promotion.
Fresh produce is a proxy for organic material; it does not measure moisture,
spoilage or whether an edible item should be discarded.
Source attribution, exclusion records and CC BY-SA 4.0 terms are in
`artifacts/organic_sources.json` and `data/external/fruits360/`.

## References

- [OpenCV background-subtraction guide](https://docs.opencv.org/4.13.0/d1/dc5/tutorial_background_subtraction.html)
- [Scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [RealWaste authors and licensing](https://github.com/sam-single/realwaste)
