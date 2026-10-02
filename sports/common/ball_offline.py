from dataclasses import dataclass
from typing import List

import numpy as np
import supervision as sv


@dataclass
class OfflineBallTrack:
    """
    Ball positions for a whole clip, as returned by `track_ball_offline`.

    Attributes:
        xy (np.ndarray): (T, 2) float ball center per frame, NaN where no ball.
        interpolated (np.ndarray): (T,) bool, True where `xy` was filled by
            linear interpolation inside a gap.
        source_index (np.ndarray): (T,) int, index of the selected candidate in
            that frame's detections, -1 if none (no ball or interpolated).
    """

    xy: np.ndarray
    interpolated: np.ndarray
    source_index: np.ndarray


def _log_odds(confidence: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(confidence, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def track_ball_offline(
    candidates: List[sv.Detections],
    max_speed: float = 40.0,
    max_gap: int = 10,
    gap_penalty: float = 1.0,
    motion_penalty: float = 0.05,
    segment_penalty: float = 5.0,
) -> OfflineBallTrack:
    """
    Selects the ball in every frame of a clip by dynamic programming.

    Nodes are candidates. An edge links a candidate at frame t to one at frame
    t + k when at most `max_gap` frames are skipped in between and the
    displacement is at most `max_speed * k`. A path scores the sum of its
    candidates' confidence log-odds, minus `motion_penalty` times each
    displacement, minus `gap_penalty` per skipped frame. The path may break into
    segments, each costing `segment_penalty`, so long occlusions don't force
    bad links; consecutive segments are separated by at least one frame without
    a ball. Gaps inside a segment are filled by linear interpolation.

    Args:
        candidates (List[sv.Detections]): Ball candidates, one per frame.
            Missing `confidence` is treated as 1.0.
        max_speed (float): Maximum ball speed in px/frame.
        max_gap (int): Maximum number of consecutive frames without a selected
            candidate inside a segment.
        gap_penalty (float): Cost per skipped frame inside a segment.
        motion_penalty (float): Cost per pixel of displacement.
        segment_penalty (float): Cost of starting a segment.

    Returns:
        OfflineBallTrack: Per-frame positions, interpolation flags and the
        selected candidate indices.
    """
    n_frames = len(candidates)
    xy = [d.get_anchors_coordinates(sv.Position.CENTER).astype(float) for d in candidates]
    unary = [
        _log_odds(np.ones(len(d)) if d.confidence is None
                  else np.asarray(d.confidence, dtype=float))
        for d in candidates
    ]

    # best[t][i]: best score of a path ending at candidate i of frame t.
    # prev_t/prev_i: predecessor node (-1 if none); linked: whether the
    # predecessor is in the same segment (else it ends the previous segment).
    best, prev_t, prev_i, linked = [], [], [], []
    frame_best = np.full(n_frames, -np.inf)
    frame_argbest = np.full(n_frames, -1, dtype=int)
    # Best path ending at frame <= t - 2, or the empty path (score 0).
    closed_score, closed_node = 0.0, (-1, -1)

    for t in range(n_frames):
        if t >= 2 and frame_best[t - 2] > closed_score:
            closed_score = frame_best[t - 2]
            closed_node = (t - 2, frame_argbest[t - 2])

        m = len(xy[t])
        score = np.full(m, closed_score - segment_penalty)
        p_t = np.full(m, closed_node[0], dtype=int)
        p_i = np.full(m, closed_node[1], dtype=int)
        link = np.zeros(m, dtype=bool)
        for k in range(1, max_gap + 2):
            s = t - k
            if s < 0 or m == 0:
                break
            if len(xy[s]) == 0:
                continue
            distance = np.linalg.norm(xy[t][:, None, :] - xy[s][None, :, :], axis=2)
            value = best[s][None, :] - motion_penalty * distance - gap_penalty * (k - 1)
            value[distance > max_speed * k] = -np.inf
            j = np.argmax(value, axis=1)
            v = value[np.arange(m), j]
            better = v > score
            score[better] = v[better]
            p_t[better] = s
            p_i[better] = j[better]
            link[better] = True

        score = score + unary[t]
        best.append(score)
        prev_t.append(p_t)
        prev_i.append(p_i)
        linked.append(link)
        if m:
            frame_argbest[t] = int(np.argmax(score))
            frame_best[t] = score[frame_argbest[t]]

    result_xy = np.full((n_frames, 2), np.nan)
    interpolated = np.zeros(n_frames, dtype=bool)
    source_index = np.full(n_frames, -1, dtype=int)

    end = (-1, -1)
    if n_frames and np.isfinite(frame_best).any():
        t_end = int(np.argmax(frame_best))
        if frame_best[t_end] > 0.0:
            end = (t_end, frame_argbest[t_end])

    t, i = end
    while t >= 0:
        result_xy[t] = xy[t][i]
        source_index[t] = i
        s, j = prev_t[t][i], prev_i[t][i]
        if linked[t][i] and t - s > 1:
            fraction = (np.arange(s + 1, t) - s)[:, None] / (t - s)
            result_xy[s + 1:t] = xy[s][j] + fraction * (xy[t][i] - xy[s][j])
            interpolated[s + 1:t] = True
        t, i = s, j

    return OfflineBallTrack(
        xy=result_xy, interpolated=interpolated, source_index=source_index
    )
