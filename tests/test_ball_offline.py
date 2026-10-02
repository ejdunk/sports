import time

import numpy as np
import pytest
import supervision as sv

from sports.common.ball_offline import OfflineBallTrack, track_ball_offline
from synthetic import kick_path, lob_path, make_scene, straight_path


def steps(xy):
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return step[~np.isnan(step)]


def test_empty_clip():
    track = track_ball_offline([])
    assert isinstance(track, OfflineBallTrack)
    assert track.xy.shape == (0, 2)
    assert track.interpolated.shape == (0,)
    assert track.source_index.shape == (0,)


def test_frames_without_candidates():
    track = track_ball_offline([sv.Detections.empty()] * 5)
    assert np.isnan(track.xy).all()
    assert (track.source_index == -1).all()
    assert not track.interpolated.any()


@pytest.mark.parametrize("seed", range(3))
def test_recovers_true_path_through_decoys(seed):
    path = lob_path(120, start=(200, 700), velocity=(12, -14), gravity=0.25)
    scene = make_scene(
        path, seed=seed, ball_confidence=0.85, noise_std=1.0,
        decoys=[(1500, 300, 0.95)], decoy_frames=(40, 43),
        false_positive_rate=0.3, false_positive_confidence=0.9,
        clutter_per_frame=5,
    )
    track = track_ball_offline(scene.candidates)
    assert (track.source_index == scene.ball_index).mean() >= 0.98
    assert not track.interpolated.any()


def test_fills_gaps_up_to_max_gap_only():
    path = straight_path(100, start=(100, 500), velocity=(6, 1))
    # 10 missing frames (= max_gap) and 11 missing frames (> max_gap).
    scene = make_scene(path, ball_confidence=0.8, dropouts=[(20, 30), (60, 71)])
    # Bridging k missing frames costs k * gap_penalty plus motion; keep it below
    # segment_penalty so the 10-frame gap is bridged rather than split.
    track = track_ball_offline(scene.candidates, max_gap=10, gap_penalty=0.1)

    assert track.interpolated[20:30].all()
    np.testing.assert_allclose(track.xy[20:30], path[20:30], atol=1e-4)
    assert (track.source_index[20:30] == -1).all()

    assert np.isnan(track.xy[60:71]).all()
    assert not track.interpolated[60:71].any()

    assert track.interpolated.sum() == 10
    detected = ~track.interpolated & ~np.isnan(track.xy).any(axis=1)
    assert (track.source_index[detected] >= 0).all()
    np.testing.assert_allclose(track.xy[detected], path[detected], atol=1e-4)


def test_interpolates_across_kick_gap_only_within_speed():
    path = kick_path(60, start=(400, 400), velocity=(5, 0), kick_frame=30,
                     kick_velocity=(-25, -10))
    scene = make_scene(path, ball_confidence=0.9, dropouts=[(28, 33)])
    track = track_ball_offline(scene.candidates, gap_penalty=0.1)
    assert track.interpolated[28:33].all()
    assert steps(track.xy).max() <= 40.0 + 1e-6


@pytest.mark.parametrize("seed", range(5))
def test_never_exceeds_max_speed(seed):
    path = kick_path(150, start=(300, 500), velocity=(10, 0), kick_frame=70,
                     kick_velocity=(-70, 20))  # faster than max_speed
    scene = make_scene(
        path, seed=seed, ball_confidence=0.7, noise_std=2.0,
        dropouts=[(20, 26), (100, 120)], false_positive_rate=0.3,
        clutter_per_frame=6, clutter_confidence=(0.05, 0.6),
    )
    for max_speed in (20.0, 40.0):
        track = track_ball_offline(scene.candidates, max_speed=max_speed)
        assert (~np.isnan(track.xy[:70, 0])).mean() > 0.8
        assert steps(track.xy).max() <= max_speed + 1e-6


def test_confidence_none_treated_as_one():
    path = straight_path(10, start=(10, 10), velocity=(3, 3))
    scene = make_scene(path, confidence_none=True)
    track = track_ball_offline(scene.candidates)
    assert (track.source_index == 0).all()


def test_runs_fast_on_clip_sized_input():
    path = lob_path(750, start=(0, 900), velocity=(2.5, -8), gravity=0.02)
    scene = make_scene(path, seed=0, clutter_per_frame=19)
    start = time.perf_counter()
    track_ball_offline(scene.candidates)
    assert time.perf_counter() - start < 5.0
