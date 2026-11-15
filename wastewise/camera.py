"""Live OpenCV waste detection: open camera, show label, Q to quit."""
import argparse
import time
import cv2
from .common import LABELS
from .capture_samples import new_session_id, save_sample
from .live_model import LiveClassifier
from .live_pipeline import LivePipeline, draw_result
from .bin_demo import BinDemo, CAMERA_WINDOW, WINDOW, render, draw_camera_demo


def main(default_demo=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model', choices=['original', 'improved', 'selective', 'previous'], default='selective')
    parser.add_argument('--collect', action='store_true', help='Enable explicit crop labeling: keys 1-4 for waste classes, 0 for unknown; no automatic saving')
    parser.add_argument('--demo', action='store_true', default=default_demo,
                        help='Open a second window with simulated bins, a 5-second review timer and manual correction; no Arduino needed')
    args = parser.parse_args()
    predictor = LiveClassifier(args.model)
    pipeline = LivePipeline(predictor)
    session = new_session_id()
    notice, notice_until = '', 0.
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError('Camera unavailable; check permissions or --camera index')
    print('Keep camera fixed and the view EMPTY during background calibration. Then show ONE waste item. B: recalibrate empty scene. Q: quit.', flush=True)
    if args.collect:
        print('Manual crop collection enabled. ' + ' | '.join(f'{i+1}: {label}' for i,label in enumerate(LABELS)) + ' | 0: unknown/non-waste. Only pressing these keys saves an image.', flush=True)
    demo = BinDemo() if args.demo else None
    clicks = []
    try:
        if demo:
            cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(CAMERA_WINDOW, 600, 450)
            cv2.moveWindow(CAMERA_WINDOW, 10, 60)
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(WINDOW, 620, 60)
            # Queue callbacks; process them on the same clock as keyboard actions.
            cv2.setMouseCallback(WINDOW, lambda event, x, y, flags, param:
                                 clicks.append((x, y)) if event == cv2.EVENT_LBUTTONDOWN else None)
            print('SOFTWARE DEMO: three consistent foreground guesses >50% start a 5-second timer. F: flag; then click a bin or press 1-4. R: reset demo. No Arduino is connected by this mode.', flush=True)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = pipeline.process(frame)
            shown = draw_result(frame, result)
            if demo:
                now = time.monotonic()
                demo.observe(result, now)
                shown = draw_camera_demo(shown, result, demo)
                cv2.imshow(WINDOW, render(demo, now))
            if args.collect:
                text = notice if time.monotonic() < notice_until else 'Save crop: 1 bio-dry | 2 non-degradable | 3 recyclable | 4 wet | 0 unknown'
                if demo and demo.phase == 'review':
                    text = 'Choose bin: 1 bio-dry | 2 non-degradable | 3 recyclable | 4 wet (no image saved)'
                cv2.rectangle(shown, (0, shown.shape[0]-30), (shown.shape[1], shown.shape[0]), (25,25,25), -1)
                cv2.putText(shown, text, (8, shown.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, .38, (240,240,240), 1)
            cv2.imshow(CAMERA_WINDOW, shown)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), ord('Q'), 27):
                break
            if key in (ord('b'), ord('B')):
                pipeline.reset()
                if demo:
                    demo.reset('Learning the empty view - keep items out')
                    clicks.clear()
            consumed = False
            if demo:
                now = time.monotonic()
                # Honor a flag on the last displayed countdown frame before routing.
                for x, y in clicks:
                    demo.click(x, y, now)
                clicks.clear()
                consumed = demo.key(key, now)
                demo.tick(now)
                if (cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
                        or cv2.getWindowProperty(CAMERA_WINDOW, cv2.WND_PROP_VISIBLE) < 1):
                    break
            if args.collect and not consumed and key in map(ord, '01234'):
                label = 'unknown' if key == ord('0') else LABELS[key-ord('1')]
                try:
                    saved = save_sample(frame, result, label, session)
                    notice = f'Saved {label} crop for later review (model unchanged)'
                    print(notice + ': ' + str(saved['image_path']), flush=True)
                except (ValueError, OSError) as error:
                    notice = str(error)
                    print('Not saved: ' + notice, flush=True)
                notice_until = time.monotonic()+3
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
