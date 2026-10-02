"""Side-by-side comparison videos of ball tracker outputs.

Usage:
    python render.py --clip 121364_0 \\
        --video ../../examples/soccer/data/121364_0.mp4 \\
        --tracks results/test/tracks/121364_0.npz \\
        --rows baseline kalman_005 offline_005 \\
        --output videos/121364_0.mp4

Each panel shows the frame, all raw candidates faintly, and one tracker's
output with a short trail. Interpolated positions (tracker B) are drawn as
hollow circles. Panels are tiled two per row.
"""

import argparse
import math
import os
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

from cache import DetectionCache, load_detection_cache

HERE = os.path.dirname(os.path.abspath(__file__))
ROW_LABELS = {
    "baseline": "Baseline (conf>=0.25)",
    "kalman_025": "A: Kalman (conf>=0.25)",
    "kalman_005": "A: Kalman (conf>=0.05)",
    "offline_005": "B: Offline (conf>=0.05)",
}
ROW_COLORS = {  # BGR
    "baseline": (0, 0, 255),
    "kalman_025": (0, 200, 255),
    "kalman_005": (0, 255, 0),
    "offline_005": (255, 128, 0),
}
CANDIDATE_COLOR = (200, 200, 200)
TRAIL = 10


def load_tracks(path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Load ``tracks/<clip>.npz`` written by evaluate.py.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: Per row, ``xy`` (T, 2) and
        ``interpolated`` (T,) bool (all False for online trackers).
    """
    tracks = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            if key.endswith("_xy"):
                row = key[:-len("_xy")]
                xy = data[key]
                interpolated = (data[f"{row}_interpolated"]
                                if f"{row}_interpolated" in data.files
                                else np.zeros(len(xy), dtype=bool))
                tracks[row] = {"xy": xy, "interpolated": interpolated.astype(bool)}
    return tracks


def draw_panel(
    frame: np.ndarray,
    frame_index: int,
    cache: DetectionCache,
    row: str,
    xy: np.ndarray,
    interpolated: np.ndarray,
    radius: int = 8,
) -> np.ndarray:
    """Draw candidates and one tracker's output on a copy of the frame.

    Args:
        frame (np.ndarray): BGR frame.
        frame_index (int): Index of the frame in the clip.
        cache (DetectionCache): Candidates of the clip.
        row (str): Result row name.
        xy (np.ndarray): (T, 2) tracker output, NaN rows for no output.
        interpolated (np.ndarray): (T,) bool interpolation flags.
        radius (int): Marker radius in native pixels.

    Returns:
        np.ndarray: Annotated frame.
    """
    panel = frame.copy()
    overlay = panel.copy()
    for cx, cy in cache.centers_in_frame(frame_index):
        cv2.circle(overlay, (int(round(cx)), int(round(cy))), radius + 4, CANDIDATE_COLOR, 1)
    panel = cv2.addWeighted(overlay, 0.4, panel, 0.6, 0)

    color = ROW_COLORS.get(row, (255, 0, 255))
    start = max(0, frame_index - TRAIL)
    trail = [p for p in xy[start:frame_index + 1] if not np.isnan(p).any()]
    for a, b in zip(trail, trail[1:]):
        cv2.line(panel, tuple(int(v) for v in np.round(a)),
                 tuple(int(v) for v in np.round(b)), color, 1, cv2.LINE_AA)
    if frame_index < len(xy) and not np.isnan(xy[frame_index]).any():
        center = tuple(int(v) for v in np.round(xy[frame_index]))
        thickness = 2 if interpolated[frame_index] else -1
        cv2.circle(panel, center, radius, color, thickness, cv2.LINE_AA)

    label = ROW_LABELS.get(row, row)
    text = f"{label}  #{frame_index}"
    scale = max(panel.shape[1] / 1600.0, 0.35)
    org = (8, int(30 * scale) + 6)
    cv2.putText(panel, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(panel, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return panel


def tile(panels: Sequence[np.ndarray], columns: int = 2) -> np.ndarray:
    """Tile equally sized panels into a grid, padding with black."""
    columns = min(columns, len(panels))
    rows = math.ceil(len(panels) / columns)
    h, w = panels[0].shape[:2]
    grid = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
    for i, panel in enumerate(panels):
        r, c = divmod(i, columns)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = panel
    return grid


def render_comparison(
    video_path: str,
    cache: DetectionCache,
    tracks: Dict[str, Dict[str, np.ndarray]],
    rows: Sequence[str],
    output_path: str,
    scale: float = 0.5,
    start: int = 0,
    end: Optional[int] = None,
) -> int:
    """Write a tiled comparison video.

    Args:
        video_path (str): Source clip.
        cache (DetectionCache): Candidates of the clip.
        tracks (Dict[str, Dict[str, np.ndarray]]): Output of `load_tracks`.
        rows (Sequence[str]): Rows to show, one panel each.
        output_path (str): Destination video (``.mp4``).
        scale (float): Panel scale relative to native resolution.
        start (int): First frame.
        end (Optional[int]): Stop before this frame; clip end when None.

    Returns:
        int: Number of frames written.

    Raises:
        ValueError: If a row is missing or the video does not match the cache.
    """
    missing = [row for row in rows if row not in tracks]
    if missing:
        raise ValueError(f"rows {missing} not in tracks (have {sorted(tracks)})")
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != (int(cache.meta["width"]), int(cache.meta["height"])):
        raise ValueError(
            f"video is {width}x{height}, cache is {cache.meta['width']}x{cache.meta['height']}"
        )
    end = cache.frame_count if end is None else min(end, cache.frame_count)
    fps = float(cache.meta.get("fps") or capture.get(cv2.CAP_PROP_FPS) or 25.0)

    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = None
    written = 0
    try:
        for frame_index in range(start, end):
            ok, frame = capture.read()
            if not ok:
                break
            panels = [
                draw_panel(frame, frame_index, cache, row,
                           tracks[row]["xy"], tracks[row]["interpolated"])
                for row in rows
            ]
            if scale != 1.0:
                panels = [cv2.resize(p, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA) for p in panels]
            image = tile(panels)
            if writer is None:
                directory = os.path.dirname(os.path.abspath(output_path))
                os.makedirs(directory, exist_ok=True)
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (image.shape[1], image.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"cannot open video writer for {output_path}")
            writer.write(image)
            written += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return written


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Render tracker comparison videos.")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True,
                        help="results/<split>/tracks/<clip>.npz from evaluate.py")
    parser.add_argument("--detections", default=None,
                        help="cache path (default: detections/<clip>.npz)")
    parser.add_argument("--rows", nargs="+", default=["baseline", "kalman_005", "offline_005"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args(argv)

    detections_path = args.detections or os.path.join(HERE, "detections", f"{args.clip}.npz")
    cache = load_detection_cache(detections_path)
    count = render_comparison(args.video, cache, load_tracks(args.tracks), args.rows,
                              args.output, args.scale, args.start, args.end)
    print(f"wrote {count} frames to {args.output}")


if __name__ == "__main__":
    main()
