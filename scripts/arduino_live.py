"""Autonomous sensor-led segregation with a continuous OpenCV camera preview."""
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from wastewise.live_model import LiveClassifier
from wastewise.live_pipeline import LivePipeline, draw_result


def command_for_sensor(message):
    """Preserve the original Arduino protocol: moisture sensor controls wet/dry."""
    return {'WET': 'MOVE_WET', 'DRY': 'MOVE_DRY'}.get(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='COM3')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model', choices=['original', 'improved', 'selective'], default='selective')
    parser.add_argument('--dry-run', action='store_true', help='Preview commands without moving motors')
    args = parser.parse_args()
    import serial
    classifier = LiveClassifier(args.model)
    pipeline = LivePipeline(classifier)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError('Camera unavailable; check --camera and camera permissions')
    last_sensor, buffer = 'Waiting for sensor', b''
    pending, pending_at, routed, absent_frames = None, 0., False, 0
    try:
        with serial.Serial(args.port, 9600, timeout=0) as device:
            print('Keep the camera fixed and sorting area empty during calibration. B: reset. Q: quit.', flush=True)
            print('DRY RUN' if args.dry_run else 'Autonomous segregation: confirmed object + sensor required; WET -> MOVE_WET; DRY -> MOVE_DRY', flush=True)
            while True:
                ok, frame = capture.read()
                if not ok:
                    print('Camera frame unavailable; stopping.')
                    break
                result = pipeline.process(frame)
                if result['status'] == 'no_object':
                    absent_frames += 1
                    if absent_frames >= 3:
                        routed = False
                else:
                    absent_frames = 0
                if result['status'] in {'no_object', 'calibrating', 'scene_changed', 'multiple_objects'}:
                    pending = None
                # Assemble complete serial lines; an idle sensor never freezes the preview.
                buffer += device.read(min(device.in_waiting, 256))
                while b'\n' in buffer:
                    raw, buffer = buffer.split(b'\n', 1)
                    message = raw.decode('ascii', errors='ignore').strip()
                    command = command_for_sensor(message)
                    if command:
                        last_sensor = message
                        if result.get('bbox') is not None and not routed:
                            pending, pending_at = command, time.monotonic()
                        else:
                            print(f'Sensor: {message}; HOLD (no confirmed new object)', flush=True)
                if len(buffer) > 1024:
                    buffer = b''
                if pending is not None and time.monotonic()-pending_at > 2:
                    pending = None
                if pending and result['status'] == 'accepted' and not routed:
                    print(f'Confirmed {result["label"]}; command: {pending}', flush=True)
                    if not args.dry_run:
                        device.write((pending + '\n').encode('ascii'))
                    routed, pending = True, None
                shown = draw_result(frame, result)
                cv2.putText(shown, f'Sensor: {last_sensor}', (10, shown.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,220,120), 1)
                cv2.imshow('Autonomous Waste Segregation', shown)
                key = cv2.waitKey(1) & 0xff
                if key == ord('q'):
                    break
                if key == ord('b'):
                    pipeline.reset()
                    pending, routed, buffer = None, False, b''
                    device.reset_input_buffer()
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
