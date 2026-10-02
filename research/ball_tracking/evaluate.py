"""Run the ball trackers over detection caches and score them against labels.

Usage:
    python evaluate.py --split tune
    python evaluate.py --split tune --params params.json
    python evaluate.py --split test --params params.json   # once frozen

Rows (DESIGN.md, "Evaluation"):

- ``baseline``: `BallTracker` on candidates with confidence >= 0.25
- ``kalman_025``: `KalmanBallTracker` at confidence >= 0.25
- ``kalman_005``: `KalmanBallTracker` at confidence >= 0.05
- ``offline_005``: `track_ball_offline` at confidence >= 0.05

``params.json`` optionally overrides tracker keyword arguments:
``{"baseline": {...}, "kalman": {...}, "offline": {...}}``.

Writes to ``results/<split>/``:

- ``metrics.csv``: per row, clip (and ``pooled``) and radius: counts,
  precision, recall, F1, mean error on TPs, recall split (B), with bootstrap
  95% CIs resampling whole runs.
- ``label_free.csv``: coverage, jump rate and ms/frame on all frames.
- ``summary.md``: the results table at the primary radius.
- ``tracks/<clip>.npz``: per-row positions for `render.py`.
- ``config.json``: parameters, clips and cache provenance.
"""

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import supervision as sv

from cache import DetectionCache, load_detection_cache
from label_tool import Label, read_labels
from metrics import (
    PRIMARY_RADIUS,
    RADII,
    bootstrap_ci,
    coverage,
    jump_rate,
    run_ids,
    score_frames,
    summarize,
)
from sports.common.ball import BallTracker, KalmanBallTracker
from sports.common.ball_offline import track_ball_offline

HERE = os.path.dirname(os.path.abspath(__file__))

SPLITS = {
    "tune": ("08fd33_0", "0bfacc_0"),
    "test": ("121364_0", "2e57b9_0", "573e61_0"),
}
BASELINE_CONFIDENCE = 0.25
LOW_CONFIDENCE = 0.05
ROWS = ("baseline", "kalman_025", "kalman_005", "offline_005")
ROW_TITLES = {
    "baseline": "Baseline (BallTracker, conf >= 0.25)",
    "kalman_025": "A: KalmanBallTracker, conf >= 0.25",
    "kalman_005": "A: KalmanBallTracker, conf >= 0.05",
    "offline_005": "B: track_ball_offline, conf >= 0.05",
}
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "baseline": {"buffer_size": 20},  # as in examples/soccer/main.py
    "kalman": {},
    "offline": {},
}
STATS = ("precision", "recall", "f1", "mean_error", "recall_detected", "recall_interpolated")


@dataclass
class TrackOutput:
    """Output of one tracker on one clip.

    Attributes:
        xy (np.ndarray): (T, 2) ball center per frame, NaN where none.
        interpolated (Optional[np.ndarray]): (T,) bool for tracker B, else None.
        seconds_per_frame (float): Mean tracker time per frame.
    """

    xy: np.ndarray
    interpolated: Optional[np.ndarray]
    seconds_per_frame: float


def run_online(tracker: Any, frames: List[sv.Detections]) -> TrackOutput:
    """Feed frames one by one to a tracker with an ``update`` method."""
    xy = np.full((len(frames), 2), np.nan)
    elapsed = 0.0
    for t, detections in enumerate(frames):
        start = time.perf_counter()
        selected = tracker.update(detections)
        elapsed += time.perf_counter() - start
        if len(selected) > 1:
            raise RuntimeError(f"tracker returned {len(selected)} detections")
        if len(selected) == 1:
            xy[t] = selected.get_anchors_coordinates(sv.Position.CENTER)[0]
    return TrackOutput(xy, None, elapsed / max(len(frames), 1))


def run_offline(frames: List[sv.Detections], **kwargs: Any) -> TrackOutput:
    """Run `track_ball_offline` on a whole clip."""
    start = time.perf_counter()
    track = track_ball_offline(frames, **kwargs)
    elapsed = time.perf_counter() - start
    return TrackOutput(track.xy, track.interpolated, elapsed / max(len(frames), 1))


def run_rows(cache: DetectionCache, params: Dict[str, Dict[str, Any]]) -> Dict[str, TrackOutput]:
    """Run all four result rows on one clip."""
    high = cache.frames(BASELINE_CONFIDENCE)
    low = cache.frames(LOW_CONFIDENCE)
    return {
        "baseline": run_online(BallTracker(**params["baseline"]), high),
        "kalman_025": run_online(KalmanBallTracker(**params["kalman"]), high),
        "kalman_005": run_online(KalmanBallTracker(**params["kalman"]), low),
        "offline_005": run_offline(low, **params["offline"]),
    }


def validate(clip: str, cache: DetectionCache, labels: Dict[int, Label]) -> None:
    """Refuse caches and labels that do not belong together.

    Raises:
        ValueError: On a clip name mismatch, a cache cutoff above 0.05, labels
            past the clip's end, or labeled positions outside the frame.
    """
    meta = cache.meta
    if "clip" in meta and meta["clip"] != clip:
        raise ValueError(f"{clip}: cache meta is for clip {meta['clip']!r}")
    if float(meta["confidence_threshold"]) > LOW_CONFIDENCE:
        raise ValueError(
            f"{clip}: cache cutoff {meta['confidence_threshold']} > {LOW_CONFIDENCE}"
        )
    if not labels:
        raise ValueError(f"{clip}: no labels")
    frame_count, width, height = int(meta["frame_count"]), meta["width"], meta["height"]
    late = [frame for frame in labels if not 0 <= frame < frame_count]
    if late:
        raise ValueError(f"{clip}: labeled frames {late[:5]} outside [0, {frame_count})")
    outside = [
        label.frame for label in labels.values()
        if label.status == "visible"
        and not (0 <= label.x <= width and 0 <= label.y <= height)
    ]
    if outside:
        raise ValueError(
            f"{clip}: labels at frames {outside[:5]} lie outside {width}x{height}; "
            "wrong resolution?"
        )


def _fmt(value: float, digits: int = 3) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def evaluate(
    split: str,
    detections_dir: str,
    labels_dir: str,
    results_dir: str,
    params: Optional[Dict[str, Dict[str, Any]]] = None,
    clips: Optional[Sequence[str]] = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Run every row on the clips of one split and write ``results/<split>/``.

    Args:
        split (str): ``tune`` or ``test``.
        detections_dir (str): Directory with ``<clip>.npz`` caches.
        labels_dir (str): Directory with ``<clip>.csv`` labels.
        results_dir (str): Results root; output goes to ``<results_dir>/<split>``.
        params (Optional[Dict[str, Dict[str, Any]]]): Tracker keyword overrides.
        clips (Optional[Sequence[str]]): Subset of the split's clips.
        n_boot (int): Bootstrap replicates.
        seed (int): Bootstrap seed.

    Returns:
        List[Dict[str, Any]]: The rows written to ``metrics.csv``.

    Raises:
        ValueError: On an unknown split, a clip outside the split, or a
            cache/label mismatch.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLITS)}")
    clips = list(clips) if clips else list(SPLITS[split])
    foreign = [clip for clip in clips if clip not in SPLITS[split]]
    if foreign:
        raise ValueError(f"clips {foreign} are not in the {split} split")
    merged = {key: dict(value) for key, value in DEFAULT_PARAMS.items()}
    for key, value in (params or {}).items():
        if key not in merged:
            raise ValueError(f"unknown params section {key!r}")
        merged[key].update(value)
    max_speed = merged["offline"].get("max_speed", 40.0)

    out_dir = os.path.join(results_dir, split)
    os.makedirs(os.path.join(out_dir, "tracks"), exist_ok=True)

    per_clip: Dict[str, Dict[str, Any]] = {}
    label_free_rows = []
    provenance = {}
    for clip in clips:
        cache = load_detection_cache(os.path.join(detections_dir, f"{clip}.npz"))
        labels = read_labels(os.path.join(labels_dir, f"{clip}.csv"))
        validate(clip, cache, labels)
        provenance[clip] = {
            key: cache.meta.get(key)
            for key in ("model_sha256", "frame_count", "width", "height", "fps",
                        "confidence_threshold", "supervision_version",
                        "ultralytics_version")
        }
        outputs = run_rows(cache, merged)
        np.savez_compressed(
            os.path.join(out_dir, "tracks", f"{clip}.npz"),
            **{f"{row}_xy": out.xy for row, out in outputs.items()},
            **{f"{row}_interpolated": out.interpolated for row, out in outputs.items()
               if out.interpolated is not None},
        )
        frames = np.array(sorted(labels))
        per_clip[clip] = {"frames": frames, "labels": [labels[f] for f in frames],
                          "outputs": outputs}
        for row, out in outputs.items():
            label_free_rows.append({
                "row": row,
                "clip": clip,
                "frames": len(out.xy),
                "coverage": _fmt(coverage(out.xy)),
                "jump_rate": _fmt(jump_rate(out.xy, max_speed)),
                "ms_per_frame": _fmt(out.seconds_per_frame * 1000.0, 4),
            })

    metric_rows = []
    groups = [(clip, [clip]) for clip in clips] + [("pooled", clips)]
    for row in ROWS:
        for group, group_clips in groups:
            status, gt, pred, interp, clip_names, frame_ids = [], [], [], [], [], []
            for clip in group_clips:
                data = per_clip[clip]
                out = data["outputs"][row]
                for frame, label in zip(data["frames"], data["labels"]):
                    status.append(label.status)
                    gt.append((np.nan, np.nan) if label.x is None else (label.x, label.y))
                    pred.append(out.xy[frame])
                    interp.append(False if out.interpolated is None
                                  else out.interpolated[frame])
                    clip_names.append(clip)
                    frame_ids.append(frame)
            interpolated = np.array(interp) if row == "offline_005" else None
            runs = run_ids(clip_names, frame_ids)
            for radius in RADII:
                outcomes = score_frames(status, gt, pred, radius)
                keep = ~outcomes.excluded
                summary = summarize(outcomes, interpolated)
                ci = bootstrap_ci(outcomes, runs, interpolated, n_boot=n_boot, seed=seed)
                record: Dict[str, Any] = {
                    "row": row, "clip": group, "radius": radius,
                    "runs": len(np.unique(runs[keep])),
                }
                for key in ("n_frames", "tp", "fp", "fn", "tn"):
                    record[key] = summary[key]
                for stat in STATS:
                    if stat not in summary:
                        record[stat] = record[f"{stat}_lo"] = record[f"{stat}_hi"] = ""
                        continue
                    record[stat] = _fmt(summary[stat], 4)
                    record[f"{stat}_lo"] = _fmt(ci[stat][0], 4)
                    record[f"{stat}_hi"] = _fmt(ci[stat][1], 4)
                metric_rows.append(record)

    _write_csv(os.path.join(out_dir, "metrics.csv"), metric_rows)
    _write_csv(os.path.join(out_dir, "label_free.csv"), label_free_rows)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(_summary_markdown(split, clips, metric_rows, label_free_rows))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({
            "split": split,
            "clips": clips,
            "params": merged,
            "baseline_confidence": BASELINE_CONFIDENCE,
            "low_confidence": LOW_CONFIDENCE,
            "radii": RADII,
            "primary_radius": PRIMARY_RADIUS,
            "n_boot": n_boot,
            "seed": seed,
            "caches": provenance,
            "versions": {"numpy": np.__version__, "supervision": sv.__version__},
        }, f, indent=2, sort_keys=True)
    return metric_rows


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _with_ci(record: Dict[str, Any], stat: str) -> str:
    if record[stat] == "":
        return "-"
    if record[f"{stat}_lo"] == "":
        return record[stat]
    return f"{record[stat]} [{record[f'{stat}_lo']}, {record[f'{stat}_hi']}]"


def _summary_markdown(
    split: str,
    clips: Sequence[str],
    metric_rows: List[Dict[str, Any]],
    label_free_rows: List[Dict[str, Any]],
) -> str:
    lines = [
        f"# Results: {split} split",
        "",
        f"Clips: {', '.join(clips)}. Radius r = {PRIMARY_RADIUS:g} px. "
        "Brackets: bootstrap 95% CI over 20-frame runs.",
        "",
    ]
    header = ("| Tracker | Frames | Precision | Recall | F1 | Mean error (px) "
              "| Recall detected / interpolated |")
    for group in ["pooled"] + list(clips):
        lines += [f"## {group}", "", header, "|" + "---|" * 7]
        for record in metric_rows:
            if record["clip"] != group or record["radius"] != PRIMARY_RADIUS:
                continue
            split_recall = (
                "-" if record["recall_detected"] == ""
                else f"{record['recall_detected']} / {record['recall_interpolated']}"
            )
            lines.append(
                f"| {ROW_TITLES[record['row']]} | {record['n_frames']} "
                f"| {_with_ci(record, 'precision')} | {_with_ci(record, 'recall')} "
                f"| {_with_ci(record, 'f1')} | {_with_ci(record, 'mean_error')} "
                f"| {split_recall} |"
            )
        lines.append("")
    lines += ["## Label-free checks (all frames)", "",
              "| Tracker | Clip | Coverage | Jump rate | ms/frame |", "|---|---|---|---|---|"]
    for record in label_free_rows:
        lines.append(
            f"| {ROW_TITLES[record['row']]} | {record['clip']} | {record['coverage']} "
            f"| {record['jump_rate']} | {record['ms_per_frame']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate ball trackers.")
    parser.add_argument("--split", required=True, choices=sorted(SPLITS))
    parser.add_argument("--clips", nargs="*", default=None,
                        help="subset of the split's clips (default: all)")
    parser.add_argument("--params", default=None, help="JSON with tracker overrides")
    parser.add_argument("--detections", default=os.path.join(HERE, "detections"))
    parser.add_argument("--labels", default=os.path.join(HERE, "labels"))
    parser.add_argument("--results", default=os.path.join(HERE, "results"))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    params = None
    if args.params:
        with open(args.params) as f:
            params = json.load(f)
    evaluate(
        split=args.split,
        detections_dir=args.detections,
        labels_dir=args.labels,
        results_dir=args.results,
        params=params,
        clips=args.clips,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    print(open(os.path.join(args.results, args.split, "summary.md")).read())


if __name__ == "__main__":
    main()
