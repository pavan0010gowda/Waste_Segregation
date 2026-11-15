"""Local, visual bin simulator. No serial connection, motor commands or training.

BinDemo owns the decision/animation clock independently of the OpenCV renderer.
Only a foreground crop with a valid raw guess over 50% can start a countdown.
The stricter model acceptance policy remains separate from this demo policy.
"""
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import LABELS
from .live_pipeline import overlap


WINDOW = 'WasteWise - Simulated sorting hardware'
CAMERA_WINDOW = 'Waste Detection System - Q to quit'
WIDTH, HEIGHT = 720, 600
NAMES = dict(zip(LABELS, ('Dry decomposable', 'Dry non-degradable', 'Dry recyclable', 'Wet / organic')))
# OpenCV colors are BGR.
COLORS = ((136, 198, 96), (175, 149, 166), (234, 174, 74), (86, 181, 244))
WHITE, MUTED = (243, 240, 230), (177, 161, 143)
BIN_RECTS = tuple((36 + i*166, 328, 186 + i*166, 474) for i in range(4))
CENTERS = tuple((rect[0]+rect[2])//2 for rect in BIN_RECTS)
FLAG_RECT, RESET_RECT = (24, 512, 238, 554), (254, 512, 444, 554)


class BinDemo:
    countdown_seconds = 5.
    slide_seconds = 1.1
    drop_seconds = .7
    return_seconds = 1.1
    rearm_seconds = .7
    max_observation_age = 1.0

    def __init__(self, stable_frames=3):
        self.stable_frames = stable_frames
        self.sorted_count = 0
        self.override_count = 0
        self.bin_counts = dict.fromkeys(LABELS, 0)
        self.reset()

    def reset(self, message='Show one item in the camera'):
        """Cancel the pending action; keep this session's completed-sort counters."""
        self.phase = 'waiting'
        self.label = None
        self.confidence = None
        self.predicted_label = None
        self.manual = False
        self.message = message
        self.deadline = None
        self.route_started = None
        self.last_observation = None
        self.previous_box = None
        self.empty_since = None
        self._candidate_label = None
        self._candidate_frames = 0

    @property
    def route_duration(self):
        return self.slide_seconds + self.drop_seconds + self.return_seconds

    def remaining(self, now):
        return max(0., self.deadline-now) if self.deadline is not None else 0.

    def observe(self, result, now):
        """Call exactly once per new processed camera frame, before tick()."""
        # A long inference/camera gap invalidates an automatic decision.
        if (self.phase == 'countdown' and self.last_observation is not None
                and now-self.last_observation > self.max_observation_age):
            self.reset('Camera paused - confirming the item again')
        self.last_observation = now
        status = result.get('status')
        if self.phase in {'routing', 'done'}:
            # Do not re-sort a held item. Require an actual stable empty view.
            if status == 'no_object':
                if self.empty_since is None:
                    self.empty_since = now
                if self.phase == 'done' and now-self.empty_since >= self.rearm_seconds:
                    self.reset('Ready for the next item')
            else:
                self.empty_since = None
            return
        if self.phase == 'review':
            # Flagging freezes the snapshot for deliberate human correction.
            return

        candidate = result.get('demo_candidate')
        box = result.get('bbox')
        confidence = candidate.get('confidence', 0.) if isinstance(candidate, dict) else 0.
        valid = (status in {'accepted', 'checking', 'uncertain'} and box is not None
                 and isinstance(candidate, dict) and candidate.get('label') in LABELS
                 and isinstance(confidence, (int, float)) and math.isfinite(confidence)
                 and .5 < confidence <= 1.)
        if not valid:
            messages = {
                'calibrating': 'Learning the empty view - keep items out',
                'no_object': 'Show one item in the camera',
                'candidate': 'Hold the item steady',
                'scene_changed': 'Clear the view, then press B to recalibrate',
                'multiple_objects': 'Show only one item at a time',
            }
            self.reset(messages.get(status, 'Waiting for a usable guess above 50%'))
            return

        label = candidate['label']
        if (self._candidate_label != label
                or (self.previous_box is not None and overlap(box, self.previous_box) < .25)):
            self.reset('Confirming the new guess - hold steady')
        self.last_observation = now
        self.previous_box = box
        self._candidate_label = label
        self._candidate_frames += 1
        self.confidence = confidence
        if self.phase == 'waiting' and self._candidate_frames >= self.stable_frames:
            self.label = self.predicted_label = label
            self.phase = 'countdown'
            self.deadline = now + self.countdown_seconds
            self.message = 'Flag a wrong prediction before the timer ends'

    def tick(self, now):
        if self.phase == 'countdown':
            if self.last_observation is None or now-self.last_observation > self.max_observation_age:
                self.reset('Camera paused - waiting for a fresh prediction')
            elif now >= self.deadline:
                self._route(self.label, now, manual=False)
        elif self.phase == 'routing' and now-self.route_started >= self.route_duration:
            self.phase = 'done'
            self.sorted_count += 1
            self.bin_counts[self.label] += 1
            if self.manual:
                self.override_count += 1
            self.message = 'Remove the item to prepare the next sort'

    def flag(self):
        if self.phase != 'countdown':
            return False
        self.phase = 'review'
        self.deadline = None
        self.message = 'Choose the correct bin below, or press 1-4'
        return True

    def choose(self, label, now):
        if self.phase != 'review' or label not in LABELS:
            return False
        self._route(label, now, manual=True)
        return True

    def _route(self, label, now, manual):
        self.label = label
        self.manual = manual
        self.phase = 'routing'
        self.deadline = None
        self.route_started = now
        self.empty_since = None
        self.message = 'Human-selected route' if manual else 'Timer completed - simulated sort'

    def key(self, key, now):
        """True consumes a key, preventing review keys from also saving crops."""
        if key in (ord('f'), ord('F')):
            self.flag()
            return True
        if key in (ord('r'), ord('R')):
            self.reset('Demo reset - show one item')
            return True
        if self.phase == 'review' and key in map(ord, '1234'):
            self.choose(LABELS[key-ord('1')], now)
            return True
        return False

    def click(self, x, y, now):
        def inside(rect):
            return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]
        if inside(RESET_RECT):
            self.reset('Demo reset - show one item')
        elif inside(FLAG_RECT):
            self.flag()
        elif self.phase == 'review':
            for label, rect in zip(LABELS, BIN_RECTS):
                if inside(rect):
                    self.choose(label, now)
                    break

    def motion(self, now):
        """Return lid center, falling-item fraction and hatch opening (0..1)."""
        home = WIDTH/2
        if self.phase != 'routing':
            return home, None, 0.
        target = CENTERS[LABELS.index(self.label)]
        elapsed = max(0., now-self.route_started)
        def ease(t):
            t = min(1., max(0., t))
            return t*t*(3-2*t)
        if elapsed < self.slide_seconds:
            return home+(target-home)*ease(elapsed/self.slide_seconds), None, 0.
        elapsed -= self.slide_seconds
        if elapsed < self.drop_seconds:
            drop = elapsed/self.drop_seconds
            return target, drop, math.sin(math.pi*drop)
        elapsed -= self.drop_seconds
        return target+(home-target)*ease(elapsed/self.return_seconds), None, 0.


def _text(canvas, text, x, y, size=.5, color=WHITE, thickness=1):
    cv2.putText(canvas, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, thickness, cv2.LINE_AA)


def render(demo, now):
    """A plain desktop workbench with system typography and a drawn bin mechanism."""
    paper, ink, muted = '#f4f2ed', '#28352f', '#6d756e'
    line, green = '#d9dcd5', '#456653'
    palette = ('#839775', '#a39385', '#7f9bac', '#b9a164')
    canvas = Image.new('RGB', (WIDTH, HEIGHT), paper)
    draw = ImageDraw.Draw(canvas)

    def text(value, x, y, size=14, color=ink, bold=False, centered=False):
        font = _font(size, bold)
        if centered:
            x -= draw.textlength(value, font=font)/2
        draw.text((int(x), int(y)), value, font=font, fill=color, anchor='lt')

    text('Waste sorting', 24, 20, 27, bold=True)
    text('Camera & sorting tray', 25, 54, 13, muted)
    text('Software simulation', 555, 25, 13, muted)
    text('Hardware not connected', 546, 46, 12, muted)
    draw.line((24, 78, 696, 78), fill=line)

    captions = {'waiting': 'Waiting for an item', 'countdown': 'Review this prediction',
                'review': 'Which bin should it go in?', 'routing': 'Moving the lid', 'done': 'Item sorted'}
    text(captions[demo.phase], 24, 92, 13, muted)
    if demo.label:
        text(NAMES[demo.label], 24, 115, 24, bold=True)
        detail = 'Selected by you' if demo.manual else f'Camera estimate  /  {demo.confidence:.0%} confidence'
    else:
        text('Show one item to the camera', 24, 115, 24, bold=True)
        detail = 'Keep the camera fixed and the item steady.'
    text(detail, 25, 149, 13, muted)
    message = demo.message
    if demo.phase == 'countdown': message = 'Wrong bin? Flag it before the timer finishes.'
    elif demo.phase == 'review': message = 'Timer paused. Click a bin below, or press its number.'
    text(message, 25, 174, 13, green if demo.phase == 'review' else muted)
    if demo.phase == 'countdown':
        remaining = demo.remaining(now)
        text(str(math.ceil(remaining)), 650, 95, 44, green, bold=True, centered=True)
        text('seconds', 650, 147, 12, muted, centered=True)
        draw.line((25, 201, 695, 201), fill=line, width=3)
        draw.line((25, 201, 25+int(670*remaining/demo.countdown_seconds), 201), fill=green, width=3)

    # The tray is drawn as a physical assembly, with a rail and four removable bins.
    text('Sorting tray', 39, 223, 12, muted)
    for x in (47, 673):
        draw.rounded_rectangle((x-4, 254, x+4, 307), radius=2, fill='#bdc3bc')
        draw.ellipse((x-2, 257, x+2, 261), fill='#7b867c')
    draw.rounded_rectangle((44, 266, 676, 273), radius=3, fill='#aeb8ae')
    draw.line((49, 267, 671, 267), fill='#dfe3dc')
    draw.line((39, 490, 681, 490), fill='#c3c5bd', width=2)

    names = (('Dry', 'Decomposable'), ('Dry', 'Non-degradable'),
             ('Dry', 'Recyclable'), ('Wet', 'Organic'))
    for index, (label, rect, color, name) in enumerate(zip(LABELS, BIN_RECTS, palette, names)):
        x1, y1, x2, y2 = rect
        center = CENTERS[index]
        selected = demo.label == label and demo.phase in {'countdown', 'routing', 'done'}
        if selected or demo.phase == 'review':
            draw.rounded_rectangle((x1-3, y1-10, x2+3, y2+10), radius=7,
                                   outline=green if selected else '#b1bbad', width=2 if selected else 1)
        draw.ellipse((x1+12, y2-4, x2-7, y2+8), fill='#e3e1d9')
        draw.polygon([(x1+5,y1), (x2-5,y1), (x2-15,y2), (x1+15,y2)], fill='#e3e5de', outline='#acb5a9')
        draw.polygon([(x2-16,y1+4), (x2-5,y1+4), (x2-15,y2), (x2-24,y2)], fill='#d6dbd2')
        draw.rounded_rectangle((x1,y1-4,x2,y1+10), radius=3, fill=color)
        draw.line((x1+7,y1-2,x2-7,y1-2), fill='#f5f5ed')
        draw.rounded_rectangle((center-17,y1+19,center+17,y1+24), radius=2, fill='#b7c0b2')
        text(str(index+1), x1+17, y1+34, 12, muted)
        text(name[0], center, y1+49, 12, muted, centered=True)
        text(name[1], center, y1+69, 15, bold=True, centered=True)
        text('Click to choose' if demo.phase == 'review' else f'{demo.bin_counts[label]} sorted',
             center, y1+108, 12, green if demo.phase == 'review' else muted, centered=True)

    lid_x, drop, opening = demo.motion(now)
    lid_x, gap = round(lid_x), round(opening*23)
    draw.rounded_rectangle((lid_x-61,250,lid_x+61,280), radius=3, fill='#7e8b80', outline='#657669')
    for x1, x2 in ((lid_x-57,lid_x-gap),(lid_x+gap,lid_x+57)):
        draw.rectangle((x1,246,x2,264), fill='#cdd4c9', outline='#8e9b8a')
        draw.line((x1+1,247,x2-1,247), fill='#ffffff')
    for x in (lid_x-46,lid_x+46):
        draw.ellipse((x-4,275,x+4,283), fill='#4c5b4e')
        draw.ellipse((x-1,278,x+1,280), fill='#c9d1c5')
    item_y = None
    if demo.phase in {'countdown','review'} or (demo.phase == 'routing' and now-demo.route_started < demo.slide_seconds):
        item_y = 235
    elif drop is not None:
        item_y = int(259+104*drop*drop)
    if item_y is not None:
        draw.polygon([(lid_x-10,item_y-6),(lid_x-3,item_y-11),(lid_x+10,item_y-5),
                      (lid_x+9,item_y+7),(lid_x-4,item_y+10),(lid_x-11,item_y+3)],
                     fill='#cbbd96', outline='#8b8064')
        draw.line((lid_x-3,item_y-8,lid_x+2,item_y,lid_x-6,item_y+5), fill='#a69878')

    for rect, title, active in ((FLAG_RECT, 'F   Flag prediction', demo.phase == 'countdown'),
                                (RESET_RECT, 'R   Reset', False)):
        draw.rounded_rectangle(rect, radius=4, fill=green if active else '#faf9f5', outline=green if active else '#cbd0c5')
        text(title, (rect[0]+rect[2])/2, rect[1]+12, 14, '#ffffff' if active else ink, centered=True)
    text(f'{demo.sorted_count} sorted', 482, 516, 14, bold=True)
    text(f'{demo.override_count} chosen manually', 482, 539, 12, muted)
    text('1-4  Choose bin after flagging     B  Reset camera     Q  Quit', 24, 577, 12, muted)
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


@lru_cache(maxsize=32)
def _font(size, bold=False):
    """Use the system's ordinary UI font; retain a portable fallback for Linux."""
    candidates = [Path('C:/Windows/Fonts')/('segoeuib.ttf' if bold else 'segoeui.ttf'),
                  'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf']
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_camera_demo(shown, result, demo):
    """Explicitly distinguish a demo guess from the strict classifier decision."""
    candidate = result.get('demo_candidate')
    if not candidate:
        return shown
    text = f"DEMO guess: {NAMES[candidate['label']]} | {candidate['confidence']:.0%}"
    color = COLORS[LABELS.index(candidate['label'])]
    cv2.rectangle(shown, (0, 0), (shown.shape[1], 65), (25, 25, 25), -1)
    scale = min(.60, max(.25, (shown.shape[1]-20)/max(1, len(text)*11)))
    _text(shown, text, 10, 25, scale, color)
    help_text = 'SIMULATION ONLY | F: flag | Q: quit'
    if demo.phase == 'review':
        help_text = 'FLAGGED | Choose bin: 1, 2, 3 or 4 in simulator'
    _text(shown, help_text, 10, 50, min(.41, (shown.shape[1]-20)/(len(help_text)*9)), MUTED)
    return shown
