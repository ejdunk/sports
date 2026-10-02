"""Scoring of ball tracker outputs against frame labels.

Per labeled frame, with radius r (DESIGN.md, "Evaluation"):

| Ground truth | Tracker output   | Outcome   |
|--------------|------------------|-----------|
| visible      | within r px      | TP        |
| visible      | farther than r   | FP + FN   |
| visible      | nothing          | FN        |
| not visible  | something        | FP        |
| not visible  | nothing          | TN        |
| unsure       | anything         | excluded  |

Positions are (N, 2) float arrays with NaN rows meaning "no ball" (ground
truth) or "no output" (tracker).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

VISIBLE = "visible"
NOT_VISIBLE = "not_visible"
UNSURE = "unsure"
RADII = (5.0, 10.0, 20.0)
PRIMARY_RADIUS = 10.0


@dataclass
class FrameOutcomes:
    """Per-frame outcome indicators.

    Attributes:
        tp (np.ndarray): (N,) int, 1 where the frame is a true positive.
        fp (np.ndarray): (N,) int, 1 where the frame counts a false positive.
        fn (np.ndarray): (N,) int, 1 where the frame counts a false negative.
        tn (np.ndarray): (N,) int, 1 where the frame is a true negative.
        excluded (np.ndarray): (N,) bool, True for ``unsure`` frames.
        error (np.ndarray): (N,) float, localization error on TPs, NaN otherwise.
    """

    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    tn: np.ndarray
    excluded: np.ndarray
    error: np.ndarray


def score_frames(
    status: Sequence[str],
    gt_xy: np.ndarray,
    pred_xy: np.ndarray,
    radius: float,
) -> FrameOutcomes:
    """Classify each labeled frame according to the outcome table.

    Args:
        status (Sequence[str]): (N,) ``visible``, ``not_visible`` or ``unsure``.
        gt_xy (np.ndarray): (N, 2) labeled ball centers; ignored unless visible.
        pred_xy (np.ndarray): (N, 2) tracker output, NaN rows for no output.
        radius (float): Match radius r in pixels (inclusive).

    Returns:
        FrameOutcomes: Per-frame indicators.
    """
    status = np.asarray(status)
    gt_xy = np.asarray(gt_xy, dtype=float).reshape(-1, 2)
    pred_xy = np.asarray(pred_xy, dtype=float).reshape(-1, 2)
    if not (len(status) == len(gt_xy) == len(pred_xy)):
        raise ValueError("status, gt_xy and pred_xy must have the same length")
    unknown = ~np.isin(status, (VISIBLE, NOT_VISIBLE, UNSURE))
    if unknown.any():
        raise ValueError(f"unknown status values: {sorted(set(status[unknown]))}")

    visible = status == VISIBLE
    not_visible = status == NOT_VISIBLE
    has_output = ~np.isnan(pred_xy).any(axis=1)
    distance = np.linalg.norm(pred_xy - gt_xy, axis=1)
    if np.isnan(gt_xy[visible]).any():
        raise ValueError("visible frames need a ground-truth position")

    hit = visible & has_output & (distance <= radius)
    miss = visible & has_output & ~hit
    tp = hit
    fp = miss | (not_visible & has_output)
    fn = miss | (visible & ~has_output)
    tn = not_visible & ~has_output
    error = np.where(hit, distance, np.nan)
    return FrameOutcomes(
        tp=tp.astype(int),
        fp=fp.astype(int),
        fn=fn.astype(int),
        tn=tn.astype(int),
        excluded=status == UNSURE,
        error=error,
    )


def precision_recall_f1(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    """Precision, recall and F1; NaN where a denominator is zero."""
    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else float("nan")
    return precision, recall, f1


def summarize(
    outcomes: FrameOutcomes,
    interpolated: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Aggregate per-frame outcomes.

    Args:
        outcomes (FrameOutcomes): Output of `score_frames`.
        interpolated (Optional[np.ndarray]): (N,) bool, frames whose output was
            interpolated (tracker B). When given, recall is also split into
            ``recall_detected`` and ``recall_interpolated``, which sum to recall.

    Returns:
        Dict[str, float]: ``n_frames`` (excluding unsure), ``tp``, ``fp``, ``fn``,
        ``tn``, ``precision``, ``recall``, ``f1``, ``mean_error`` and, with
        ``interpolated``, the recall split.
    """
    tp, fp, fn, tn = (int(a.sum()) for a in (outcomes.tp, outcomes.fp, outcomes.fn, outcomes.tn))
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    errors = outcomes.error[~np.isnan(outcomes.error)]
    summary = {
        "n_frames": int((~outcomes.excluded).sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_error": float(errors.mean()) if len(errors) else float("nan"),
    }
    if interpolated is not None:
        interpolated = np.asarray(interpolated, dtype=bool)
        positives = tp + fn
        tp_interpolated = int(outcomes.tp[interpolated].sum())
        summary["recall_detected"] = (
            (tp - tp_interpolated) / positives if positives else float("nan")
        )
        summary["recall_interpolated"] = (
            tp_interpolated / positives if positives else float("nan")
        )
    return summary


def run_ids(clips: Sequence[str], frames: Sequence[int]) -> np.ndarray:
    """Assign a run id to each labeled frame.

    A run is a block of consecutive frames within one clip (label sampling keeps
    runs non-adjacent). Ids are integers unique across clips.

    Args:
        clips (Sequence[str]): (N,) clip of each labeled frame.
        frames (Sequence[int]): (N,) frame index of each labeled frame.

    Returns:
        np.ndarray: (N,) int run id, aligned with the input order.
    """
    clips = np.asarray(clips)
    frames = np.asarray(frames, dtype=int)
    order = np.lexsort((frames, clips))
    sorted_clips, sorted_frames = clips[order], frames[order]
    new_run = np.ones(len(order), dtype=bool)
    new_run[1:] = (sorted_clips[1:] != sorted_clips[:-1]) | (
        sorted_frames[1:] != sorted_frames[:-1] + 1
    )
    ids = np.empty(len(order), dtype=int)
    ids[order] = np.cumsum(new_run) - 1
    return ids


def bootstrap_ci(
    outcomes: FrameOutcomes,
    runs: np.ndarray,
    interpolated: Optional[np.ndarray] = None,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    """Percentile bootstrap CIs, resampling whole runs with replacement.

    Frames within a 20-frame run are correlated, so the resampling unit is the
    run, not the frame.

    Args:
        outcomes (FrameOutcomes): Output of `score_frames`.
        runs (np.ndarray): (N,) run id per frame, e.g. from `run_ids`.
        interpolated (Optional[np.ndarray]): As in `summarize`.
        n_boot (int): Bootstrap replicates.
        seed (int): RNG seed.
        level (float): Confidence level.

    Returns:
        Dict[str, Tuple[float, float]]: ``(low, high)`` for ``precision``,
        ``recall``, ``f1``, ``mean_error`` and, with ``interpolated``, the
        recall split. NaN bounds where a statistic is undefined in most
        replicates.
    """
    _, run_index = np.unique(np.asarray(runs), return_inverse=True)
    n_runs = int(run_index.max()) + 1 if len(run_index) else 0
    is_tp = ~np.isnan(outcomes.error)
    columns = {
        "tp": outcomes.tp,
        "fp": outcomes.fp,
        "fn": outcomes.fn,
        "error_sum": np.where(is_tp, outcomes.error, 0.0),
    }
    if interpolated is not None:
        columns["tp_interpolated"] = outcomes.tp * np.asarray(interpolated, dtype=bool)
    per_run = {
        name: np.bincount(run_index, weights=values, minlength=n_runs)
        for name, values in columns.items()
    }

    names = ["precision", "recall", "f1", "mean_error"]
    if interpolated is not None:
        names += ["recall_detected", "recall_interpolated"]
    if n_runs == 0:
        return {name: (float("nan"), float("nan")) for name in names}

    rng = np.random.default_rng(seed)
    sample = rng.integers(0, n_runs, size=(n_boot, n_runs))
    s = {name: values[sample].sum(axis=1) for name, values in per_run.items()}
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = {
            "precision": s["tp"] / (s["tp"] + s["fp"]),
            "recall": s["tp"] / (s["tp"] + s["fn"]),
            "f1": 2 * s["tp"] / (2 * s["tp"] + s["fp"] + s["fn"]),
            "mean_error": s["error_sum"] / s["tp"],
        }
        if interpolated is not None:
            positives = s["tp"] + s["fn"]
            stats["recall_detected"] = (s["tp"] - s["tp_interpolated"]) / positives
            stats["recall_interpolated"] = s["tp_interpolated"] / positives

    alpha = (1.0 - level) / 2.0
    result = {}
    for name in names:
        values = stats[name][np.isfinite(stats[name])]
        if len(values) < n_boot / 2:
            result[name] = (float("nan"), float("nan"))
        else:
            low, high = np.quantile(values, [alpha, 1.0 - alpha])
            result[name] = (float(low), float(high))
    return result


def coverage(pred_xy: np.ndarray) -> float:
    """Fraction of frames with a tracker output (label-free)."""
    pred_xy = np.asarray(pred_xy, dtype=float).reshape(-1, 2)
    if len(pred_xy) == 0:
        return float("nan")
    return float((~np.isnan(pred_xy).any(axis=1)).mean())


def jump_rate(pred_xy: np.ndarray, max_speed: float) -> float:
    """Fraction of consecutive output pairs further apart than ``max_speed``.

    Only pairs of adjacent frames that both have an output are counted
    (label-free).

    Args:
        pred_xy (np.ndarray): (T, 2) tracker output, NaN rows for no output.
        max_speed (float): Plausible ball speed in px/frame.

    Returns:
        float: Jumps / consecutive pairs, NaN without any pair.
    """
    pred_xy = np.asarray(pred_xy, dtype=float).reshape(-1, 2)
    step = np.linalg.norm(np.diff(pred_xy, axis=0), axis=1)
    pairs = ~np.isnan(step)
    if not pairs.any():
        return float("nan")
    return float((step[pairs] > max_speed).mean())
