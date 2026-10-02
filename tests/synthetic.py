"""Synthetic ball-candidate scenes for tracker tests.

A scene is a ground-truth ball path plus per-frame `sv.Detections` built from
it: Gaussian position noise, dropout stretches, stationary high-confidence
decoys (e.g. the penalty spot), single-frame false positives and
low-confidence clutter. No video, model or GPU is involved.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import supervision as sv


def straight_path(
    n_frames: int, start: Tuple[float, float], velocity: Tuple[float, float]
) -> np.ndarray:
    """Constant-velocity path.

    Args:
        n_frames (int): Number of frames.
        start (Tuple[float, float]): Position at frame 0, px.
        velocity (Tuple[float, float]): px/frame.

    Returns:
        np.ndarray: (n_frames, 2) positions.
    """
    t = np.arange(n_frames, dtype=float)[:, None]
    return np.asarray(start, dtype=float) + t * np.asarray(velocity, dtype=float)


def lob_path(
    n_frames: int,
    start: Tuple[float, float],
    velocity: Tuple[float, float],
    gravity: float = 0.8,
) -> np.ndarray:
    """Lobbed arc: constant x velocity, image-y velocity growing by `gravity`.

    Args:
        n_frames (int): Number of frames.
        start (Tuple[float, float]): Position at frame 0, px.
        velocity (Tuple[float, float]): Initial px/frame (negative y is up).
        gravity (float): px/frame² downward.

    Returns:
        np.ndarray: (n_frames, 2) positions.
    """
    t = np.arange(n_frames, dtype=float)
    xy = straight_path(n_frames, start, velocity)
    xy[:, 1] += 0.5 * gravity * t ** 2
    return xy


def kick_path(
    n_frames: int,
    start: Tuple[float, float],
    velocity: Tuple[float, float],
    kick_frame: int,
    kick_velocity: Tuple[float, float],
) -> np.ndarray:
    """Straight path whose velocity changes abruptly at `kick_frame`.

    Args:
        n_frames (int): Number of frames.
        start (Tuple[float, float]): Position at frame 0, px.
        velocity (Tuple[float, float]): px/frame before the kick.
        kick_frame (int): First frame moving with `kick_velocity`.
        kick_velocity (Tuple[float, float]): px/frame from the kick on.

    Returns:
        np.ndarray: (n_frames, 2) positions.
    """
    before = straight_path(kick_frame, start, velocity)
    kick_start = np.asarray(start, dtype=float) + kick_frame * np.asarray(velocity)
    after = straight_path(n_frames - kick_frame, kick_start, kick_velocity)
    return np.concatenate([before, after])[:n_frames]


@dataclass
class Scene:
    """Per-frame candidates with the ground truth they were built from.

    Attributes:
        candidates (List[sv.Detections]): One detections object per frame.
        truth_xy (np.ndarray): (T, 2) true ball center; NaN while dropped out.
        ball_index (np.ndarray): (T,) index of the ball's detection in each
            frame, -1 when the ball was not detected.
    """

    candidates: List[sv.Detections]
    truth_xy: np.ndarray
    ball_index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))


def make_scene(
    path: np.ndarray,
    seed: int = 0,
    ball_confidence: float = 0.6,
    noise_std: float = 0.0,
    dropouts: Sequence[Tuple[int, int]] = (),
    decoys: Sequence[Tuple[float, float, float]] = (),
    decoy_frames: Optional[Tuple[int, int]] = None,
    false_positive_rate: float = 0.0,
    false_positive_confidence: float = 0.9,
    clutter_per_frame: float = 0.0,
    clutter_confidence: Tuple[float, float] = (0.05, 0.2),
    frame_wh: Tuple[int, int] = (1920, 1080),
    box_size: float = 10.0,
    confidence_none: bool = False,
) -> Scene:
    """Build per-frame candidates from a ground-truth path.

    Args:
        path (np.ndarray): (T, 2) true ball centers.
        seed (int): RNG seed.
        ball_confidence (float): Confidence of the ball detection.
        noise_std (float): Gaussian noise added to the detected center, px.
        dropouts (Sequence[Tuple[int, int]]): ``[start, stop)`` frame ranges in
            which the ball is not detected (truth stays NaN there).
        decoys (Sequence[Tuple[float, float, float]]): Stationary candidates as
            ``(x, y, confidence)``, e.g. the penalty spot.
        decoy_frames (Optional[Tuple[int, int]]): ``[start, stop)`` frames in
            which decoys appear; all frames when None.
        false_positive_rate (float): Per-frame probability of one candidate at a
            uniformly random position with `false_positive_confidence`.
        false_positive_confidence (float): Confidence of those candidates.
        clutter_per_frame (float): Mean number (Poisson) of random
            low-confidence candidates per frame.
        clutter_confidence (Tuple[float, float]): Uniform range of their
            confidence.
        frame_wh (Tuple[int, int]): Frame size for random positions.
        box_size (float): Side of every candidate box, px.
        confidence_none (bool): Build detections with ``confidence=None``.

    Returns:
        Scene: Candidates (shuffled within each frame) and ground truth.
    """
    rng = np.random.default_rng(seed)
    path = np.asarray(path, dtype=float)
    n_frames = len(path)
    dropped = np.zeros(n_frames, dtype=bool)
    for start, stop in dropouts:
        dropped[start:stop] = True
    w, h = frame_wh

    candidates, ball_index = [], np.full(n_frames, -1, dtype=int)
    for t in range(n_frames):
        centers, confidences, is_ball = [], [], []
        if not dropped[t]:
            centers.append(path[t] + rng.normal(0.0, noise_std, 2) if noise_std else path[t])
            confidences.append(ball_confidence)
            is_ball.append(True)
        if decoy_frames is None or decoy_frames[0] <= t < decoy_frames[1]:
            for x, y, confidence in decoys:
                centers.append(np.array([x, y], dtype=float))
                confidences.append(confidence)
                is_ball.append(False)
        if rng.random() < false_positive_rate:
            centers.append(rng.uniform((0, 0), (w, h)))
            confidences.append(false_positive_confidence)
            is_ball.append(False)
        for _ in range(rng.poisson(clutter_per_frame)):
            centers.append(rng.uniform((0, 0), (w, h)))
            confidences.append(rng.uniform(*clutter_confidence))
            is_ball.append(False)

        order = rng.permutation(len(centers))
        centers = np.array(centers, dtype=float).reshape(-1, 2)[order]
        confidences = np.array(confidences, dtype=np.float32)[order]
        is_ball = np.array(is_ball, dtype=bool)[order]
        if is_ball.any():
            ball_index[t] = int(np.flatnonzero(is_ball)[0])
        half = box_size / 2.0
        candidates.append(sv.Detections(
            xyxy=np.hstack([centers - half, centers + half]).astype(np.float32),
            confidence=None if confidence_none else confidences,
            class_id=np.zeros(len(centers), dtype=int),
        ))

    truth_xy = path.copy()
    truth_xy[dropped] = np.nan
    return Scene(candidates=candidates, truth_xy=truth_xy, ball_index=ball_index)


def centers(detections: sv.Detections) -> np.ndarray:
    """(N, 2) box centers of detections."""
    return detections.get_anchors_coordinates(sv.Position.CENTER)
