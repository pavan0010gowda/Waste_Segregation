"""Exercise the review/routing state machine with a deterministic clock."""
import numpy as np
import pytest

from wastewise.bin_demo import BinDemo, CENTERS, HEIGHT, WIDTH, render
from wastewise.common import LABELS


def item(label=LABELS[2], confidence=.61, box=(20, 20, 80, 80)):
    return dict(status='uncertain', label=None, bbox=box,
                demo_candidate=dict(label=label, confidence=confidence))


def start(demo, now=0., proposal=None):
    for index in range(3):
        demo.observe(proposal or item(), now + index*.1)
    assert demo.phase == 'countdown'
    return now + .2


def advance_visible(demo, begin, end, proposal=None):
    for now in np.arange(begin, end, .2):
        demo.observe(proposal or item(), float(now))
        demo.tick(float(now))
    demo.observe(proposal or item(), end)
    demo.tick(end)


def test_five_second_countdown_then_exactly_one_sort_per_presence():
    demo = BinDemo()
    began = start(demo)
    assert demo.remaining(began) == 5.
    advance_visible(demo, began, began+4.99)
    assert demo.phase == 'countdown' and demo.sorted_count == 0
    demo.observe(item(), began+5)
    demo.tick(began+5)
    assert demo.phase == 'routing'
    demo.tick(began+5+demo.route_duration)
    assert demo.phase == 'done'
    assert demo.sorted_count == 1 and demo.bin_counts[LABELS[2]] == 1
    advance_visible(demo, 10, 30)
    assert demo.sorted_count == 1
    demo.observe(dict(status='no_object'), 31)
    demo.observe(dict(status='no_object'), 31.71)
    assert demo.phase == 'waiting'
    start(demo, 32)


@pytest.mark.parametrize('confidence', [.5, .49, 0, 1.1, float('nan'), float('inf')])
def test_only_finite_confidence_strictly_over_half_can_start(confidence):
    demo = BinDemo()
    for index in range(20):
        demo.observe(item(confidence=confidence), index*.1)
        demo.tick(index*.1)
    assert demo.phase == 'waiting' and demo.sorted_count == 0


@pytest.mark.parametrize('status', ['no_object', 'calibrating', 'multiple_objects', 'scene_changed', 'candidate'])
def test_foreground_loss_cancels_even_on_deadline(status):
    demo = BinDemo()
    began = start(demo)
    advance_visible(demo, began, began+4.9)
    demo.observe(dict(status=status), began+5)
    demo.tick(began+5)
    assert demo.phase == 'waiting' and demo.sorted_count == 0


def test_new_label_or_disjoint_box_requires_new_confirmation_and_full_timer():
    demo = BinDemo()
    start(demo)
    demo.observe(item(LABELS[3]), .3)
    assert demo.phase == 'waiting'
    demo.observe(item(LABELS[3]), .4)
    assert demo.phase == 'waiting'
    demo.observe(item(LABELS[3]), .5)
    assert demo.phase == 'countdown' and demo.deadline == 5.5
    demo.observe(item(LABELS[3], box=(200, 200, 80, 80)), .7)
    assert demo.phase == 'waiting'


def test_low_confidence_and_stale_frames_cancel_automatic_route():
    demo = BinDemo()
    start(demo)
    demo.observe(item(confidence=.5), .3)
    assert demo.phase == 'waiting'
    start(demo, 1)
    demo.tick(6.5)
    assert demo.phase == 'waiting'
    start(demo, 7)
    demo.observe(item(), 13)  # A late new frame cannot revive the old deadline.
    demo.tick(13)
    assert demo.phase == 'waiting'


def test_flag_freezes_snapshot_and_human_choice_routes_to_correct_bin():
    demo = BinDemo()
    start(demo)
    assert demo.key(ord('f'), .3)
    assert demo.phase == 'review' and demo.deadline is None
    demo.observe(dict(status='no_object'), 100)
    demo.tick(100)
    assert demo.phase == 'review' and demo.sorted_count == 0
    assert demo.key(ord('4'), 101)
    assert demo.phase == 'routing' and demo.label == LABELS[3]
    assert demo.predicted_label == LABELS[2] and demo.manual
    lid_x, drop, opening = demo.motion(101+demo.slide_seconds+.2)
    assert lid_x == CENTERS[3] and 0 < drop < 1 and opening > 0
    demo.tick(104)
    demo.tick(105)
    assert demo.sorted_count == demo.override_count == demo.bin_counts[LABELS[3]] == 1
    assert demo.bin_counts[LABELS[2]] == 0


def test_mouse_controls_flag_choose_and_reset():
    demo = BinDemo()
    start(demo)
    demo.click(80, 530, .3)
    assert demo.phase == 'review'
    demo.click(CENTERS[0], 400, .4)
    assert demo.label == LABELS[0] and demo.phase == 'routing'
    demo.click(320, 530, .5)
    assert demo.phase == 'waiting' and demo.sorted_count == 0
    assert not demo.choose(LABELS[1], .6)
    assert not demo.flag()
    assert not demo.key(ord('2'), .7)  # Collection can still consume this key.


def test_empty_gap_must_be_continuous_before_rearming():
    demo = BinDemo()
    start(demo)
    demo.flag()
    demo.choose(LABELS[1], 1)
    demo.tick(4)
    demo.observe(dict(status='no_object'), 5)
    demo.observe(item(), 5.6)
    demo.observe(dict(status='no_object'), 5.8)
    demo.observe(dict(status='no_object'), 6.1)
    assert demo.phase == 'done'
    demo.observe(dict(status='no_object'), 6.6)
    assert demo.phase == 'waiting'


def test_renderer_all_phases_and_all_destinations():
    demo = BinDemo()
    states = []
    states.append(render(demo, 0))
    for label in LABELS:
        demo.reset()
        start(demo, 0, item(label))
        states.append(render(demo, 1))
        demo.flag()
        states.append(render(demo, 1))
        demo.choose(label, 2)
        for now in (2, 2.5, 3.3, 4.2):
            states.append(render(demo, now))
        demo.tick(5)
        states.append(render(demo, 5))
    assert all(frame.shape == (HEIGHT, WIDTH, 3) and frame.dtype == np.uint8 for frame in states)
    assert demo.sorted_count == 4 and demo.override_count == 4
