import numpy as np
import pytest
import supervision as sv

from sports.common.ball import BallTracker, KalmanBallTracker
from synthetic import centers, kick_path, lob_path, make_scene, straight_path


def detections(xy_conf):
    """Detections with 10 px boxes from (x, y, confidence) triples."""
    xy_conf = np.asarray(xy_conf, dtype=float).reshape(-1, 3)
    xy = xy_conf[:, :2]
    return sv.Detections(
        xyxy=np.hstack([xy - 5, xy + 5]).astype(np.float32),
        confidence=xy_conf[:, 2].astype(np.float32),
        class_id=np.zeros(len(xy), dtype=int),
    )


def run(tracker, candidates):
    return [tracker.update(d) for d in candidates]


def selected_ball(outputs, scene):
    """Per frame: True if the output is the ball's detection."""
    result = []
    for t, out in enumerate(outputs):
        if len(out) == 0 or scene.ball_index[t] < 0:
            result.append(False)
            continue
        ball = scene.candidates[t].xyxy[scene.ball_index[t]]
        result.append(bool(np.allclose(out.xyxy[0], ball)))
    return np.array(result)


def assert_subset(output, candidates):
    assert len(output) <= 1
    if len(output):
        assert any(np.array_equal(output.xyxy[0], row) for row in candidates.xyxy)
        i = int(np.flatnonzero((candidates.xyxy == output.xyxy[0]).all(axis=1))[0])
        if candidates.confidence is not None:
            assert output.confidence[0] == candidates.confidence[i]
        assert output.class_id[0] == candidates.class_id[i]


def test_follows_ball_past_higher_confidence_stationary_decoy():
    path = straight_path(60, start=(200, 500), velocity=(12, 0))
    # Penalty spot at conf 0.9 appears after the track starts; the ball passes
    # 40 px from it around frame 33.
    scene = make_scene(path, ball_confidence=0.5, noise_std=1.0,
                       decoys=[(600, 540, 0.9)], decoy_frames=(5, 60))
    outputs = run(KalmanBallTracker(), scene.candidates)
    assert selected_ball(outputs, scene).all()

    baseline = run(BallTracker(buffer_size=20), scene.candidates)
    assert not selected_ball(baseline, scene).all()  # the naive tracker is fooled


def test_follows_lob_with_clutter():
    path = lob_path(40, start=(300, 800), velocity=(10, -16), gravity=0.8)
    scene = make_scene(path, seed=3, ball_confidence=0.5, noise_std=1.0,
                       clutter_per_frame=3)
    outputs = run(KalmanBallTracker(), scene.candidates)
    assert selected_ball(outputs, scene).mean() >= 0.95


def test_accepts_weak_detection_in_gate():
    tracker = KalmanBallTracker()
    for t in range(10):
        assert len(tracker.update(detections([(100 + 10 * t, 100, 0.8)]))) == 1
    out = tracker.update(detections([(200, 100, 0.06), (900, 900, 0.5)]))
    assert len(out) == 1
    np.testing.assert_allclose(centers(out)[0], (200, 100))


def test_rejects_weak_detection_out_of_gate_while_searching():
    tracker = KalmanBallTracker()
    for t in range(10):
        tracker.update(detections([(100 + 10 * t, 100, 0.8)]))
    for _ in range(3):  # ball lost, track still alive
        assert len(tracker.update(detections([]))) == 0
    assert tracker.is_tracking
    # Below reacquire_confidence and far outside the gate: rejected, not re-init.
    assert len(tracker.update(detections([(800, 600, 0.5)]))) == 0
    assert tracker.is_tracking


def test_does_not_start_on_weak_detection():
    tracker = KalmanBallTracker(init_confidence=0.3)
    assert len(tracker.update(detections([(100, 100, 0.29)]))) == 0
    assert not tracker.is_tracking
    assert len(tracker.update(detections([(100, 100, 0.2), (300, 300, 0.4)]))) == 1
    assert tracker.is_tracking


def test_reacquires_after_kick():
    path = kick_path(40, start=(500, 500), velocity=(8, 0), kick_frame=20,
                     kick_velocity=(-45, -20))
    scene = make_scene(path, ball_confidence=0.8, noise_std=1.0,
                       clutter_per_frame=1, seed=5)
    outputs = run(KalmanBallTracker(), scene.candidates)
    hits = selected_ball(outputs, scene)
    assert hits[:20].all()
    assert hits[21:].all()  # at most the kick frame itself may be missed


def test_drops_track_after_max_misses():
    tracker = KalmanBallTracker(max_misses=5)
    for t in range(10):
        tracker.update(detections([(100 + 10 * t, 100, 0.8)]))
    for _ in range(4):
        tracker.update(detections([]))
    assert tracker.is_tracking
    tracker.update(detections([]))
    assert not tracker.is_tracking
    # A weak candidate right where the ball would be is no longer accepted.
    assert len(tracker.update(detections([(240, 100, 0.2)]))) == 0
    # A confident one starts a new track.
    assert len(tracker.update(detections([(250, 100, 0.35)]))) == 1
    assert tracker.is_tracking


def test_weak_detection_accepted_before_max_misses():
    tracker = KalmanBallTracker(max_misses=5)
    for t in range(10):
        tracker.update(detections([(100 + 10 * t, 100, 0.8)]))
    for _ in range(4):
        tracker.update(detections([]))
    out = tracker.update(detections([(240, 100, 0.1)]))
    assert len(out) == 1


def test_empty_input():
    tracker = KalmanBallTracker()
    out = tracker.update(sv.Detections.empty())
    assert len(out) == 0
    tracker.update(detections([(10, 10, 0.9)]))
    assert len(tracker.update(sv.Detections.empty())) == 0


def test_confidence_none_treated_as_one():
    path = straight_path(20, start=(100, 100), velocity=(5, 5))
    scene = make_scene(path, confidence_none=True)
    assert scene.candidates[0].confidence is None
    outputs = run(KalmanBallTracker(init_confidence=0.99), scene.candidates)
    assert all(len(o) == 1 for o in outputs)
    assert all(o.confidence is None for o in outputs)


def test_reset():
    tracker = KalmanBallTracker()
    tracker.update(detections([(10, 10, 0.9)]))
    tracker.reset()
    assert not tracker.is_tracking
    assert len(tracker.update(detections([(500, 500, 0.1)]))) == 0


@pytest.mark.parametrize("seed", range(5))
def test_output_is_subset_with_at_most_one_detection(seed):
    path = kick_path(80, start=(300, 300), velocity=(15, 4), kick_frame=40,
                     kick_velocity=(-30, 10))
    scene = make_scene(path, seed=seed, ball_confidence=0.4, noise_std=2.0,
                       dropouts=[(10, 14), (50, 70)], decoys=[(960, 540, 0.95)],
                       false_positive_rate=0.2, clutter_per_frame=4)
    tracker = KalmanBallTracker()
    for candidates in scene.candidates:
        assert_subset(tracker.update(candidates), candidates)
