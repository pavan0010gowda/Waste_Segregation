# Live camera and autonomous segregation demo

## Record a hardware-free demonstration

Run `.venv\Scripts\python.exe demo.py` from the project folder. Two local OpenCV
windows open side by side: the live camera and **WasteWise — Simulated sorting
hardware**. The equivalent command is `live_detect.py --demo`. You can drag the
windows onto different monitors. No board, browser or server is needed.

Suggested 30–45 second video sequence:

1. Start with a fixed, empty camera view and let background calibration finish.
   Show both windows in your screen recording; keep the Software simulation label
   visible. The program itself does not record or upload video.
2. Show one separated item and hold it steady. Three consistent foreground guesses
   **over 50%** start the five-second timer. The camera shows the demo guess and
   confidence. Let one timer finish to demonstrate automatic simulated sorting:
   the lid slides over the predicted bin, opens, drops the animated item and returns.
3. Remove the item until the station says it is ready. Show another item, click
   **Flag prediction** or press **F**, then click its correct bin or press **1–4**.
   The countdown pauses while you choose. The animation follows your selected
   route and increments the human-route counter.
4. Remove the object to re-arm. **R** resets just the demo; **B** recalibrates an
   empty camera view and cancels the demo action. **Q** or Escape exits both windows.

The bin order is **1 dry decomposable**, **2 dry non-degradable**,
**3 dry recyclable**, **4 wet / organic**. A pen is not automatically recyclable;
use the classification appropriate to its material and your local sorting rules.
The human correction feature is useful precisely when the model's guess is wrong.

The >50% threshold is a **demo routing policy**, not a measured accuracy claim.
Low-detail crops, empty scenes and multiple separated objects cannot start a
countdown. Removal, a changed guess, a separate box or lost confidence cancels or
restarts it. A flagged snapshot stays paused until a human chooses or resets.
Corrections do not silently save camera images or retrain the model. With
`--demo --collect`, correction keys take priority over crop-saving keys.

Describe the video as **“Live waste classification with a software sorting-hardware
simulator and human-in-the-loop correction.”** The four-bin simulator demonstrates
a proposed mechanism. The existing physical Arduino integration is unchanged and
still uses the original moisture-sensor-controlled two-route wet/dry protocol.
Do not describe this as tested physical hardware.

## Camera-only and physical-hardware walkthrough

1. Run `.venv\Scripts\python.exe live_detect.py` from the project folder.
2. Keep the camera fixed and the sorting area empty until calibration completes.
   Verify that the empty view says No object. Show one separated item, withdraw your
   hand where possible, and hold it still. The program draws a foreground box and
   displays a category only after consistent, supported predictions. Remove the
   item to demonstrate immediate label clearing. Press B to recalibrate if needed.
3. Explain low-confidence results and how lighting, background and unfamiliar
   materials affect predictions. Press Q to release the camera.
4. For the original sensor/Arduino integration, run
   `.venv\Scripts\python.exe scripts/arduino_live.py --port COM3 --dry-run`.
   Show how WET maps to MOVE_WET and DRY maps to MOVE_DRY only after an object is
   confirmed; empty or uncertain scenes hold. Omit `--dry-run` for the actual autonomous
   routing when your hardware is ready. The moisture sensor remains authoritative.
5. Run `live_detect.py --model original` to compare the preserved classifier
   in the same object-aware camera interface. Do not claim that offline accuracy
   establishes real-world camera accuracy.
6. Show LIVE_RESULTS.md to explain the measured dataset comparison and remaining
   weaknesses, including organic-category bias and historical-test limitations.

No hosted website or browser is part of this demonstration.

## Restoration checks

- Original MobileNet copied unchanged into the project; SHA-256 matched the source.
- OpenCV-frame predictions matched the original preprocessing implementation.
- Actual webcam opened and supplied a frame; model inference succeeded. No frame saved.
- Camera overlay and cleanup tested with simulated video input.
- Arduino WET/DRY command mapping tested without sending motor commands.
- Foreground, temporal classification and simulated Arduino regression tests were
  added for the object-aware pipeline. Actual motors/firmware were not exercised.

Exact original scripts are retained locally in `legacy/original_scripts/` (Git-ignored).

## Simulator verification

- Automated checks cover the five-second deadline, confidence boundary, lost
  objects, changed predictions, stale frames, paused review, manual routing and
  one sort per object presence. Both-window wiring and collection-key precedence
  are also tested without a webcam or serial connection.
- Native Windows checks with clearly marked synthetic input confirmed both
  windows, mouse flagging, choosing a different bin, the completed human route,
  and camera release when a window closes. An extended countdown was used for
  the external mouse-automation check; production remains five seconds.
- The selected real classifier successfully supplied a demo candidate and started
  the countdown on a local dataset image. This is an integration check, not an
  additional accuracy measurement. The Arduino script's SHA-256 remained unchanged.
