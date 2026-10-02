"""OpenCV tool for labeling the ball center on sampled frames.

Usage:
    python label_tool.py --clip 08fd33_0 \\
        --video ../../examples/soccer/data/08fd33_0.mp4

Controls:
    click         mark the ball center (snaps to a candidate within 8 px)
    n             ball not visible
    u             unsure
    right / d     next frame
    left / a      previous frame
    q / Esc       quit (labels are already saved)

Per clip, 5 runs of 20 consecutive frames are sampled at seeded random,
non-overlapping offsets. Every label is saved immediately via
write-to-temp-then-rename, and restarting the tool resumes at the first
unlabeled frame. Raw candidates are drawn faintly; no tracker output is shown.
"""

import argparse
import csv
import os
import tempfile
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cache import DetectionCache, load_detection_cache

STATUSES = ("visible", "not_visible", "unsure")
CSV_COLUMNS = ("frame", "x", "y", "status")
RUN_LENGTH = 20
RUNS_PER_CLIP = 5
SNAP_RADIUS = 8.0
DEFAULT_SEED = 0

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class Label:
    """Ground truth for one frame.

    Attributes:
        frame (int): Frame index in the clip.
        status (str): One of ``visible``, ``not_visible``, ``unsure``.
        x (Optional[float]): Ball center x in pixels; only when visible.
        y (Optional[float]): Ball center y in pixels; only when visible.
    """

    frame: int
    status: str
    x: Optional[float] = None
    y: Optional[float] = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"frame {self.frame}: unknown status {self.status!r}")
        has_xy = self.x is not None and self.y is not None
        if (self.status == "visible") != has_xy:
            raise ValueError(f"frame {self.frame}: x, y must be set iff visible")


def clip_seed(clip: str, seed: int = DEFAULT_SEED) -> int:
    """Derive a stable per-clip seed from the clip name and a global seed."""
    return (zlib.crc32(clip.encode("utf-8")) + seed * 1_000_003) % (2 ** 32)


def sample_runs(
    frame_count: int,
    seed: int,
    n_runs: int = RUNS_PER_CLIP,
    run_length: int = RUN_LENGTH,
) -> List[Tuple[int, int]]:
    """Sample non-overlapping runs of consecutive frames.

    Runs are also kept non-adjacent (at least one unlabeled frame between
    them), so a run can be recovered from a label file as a block of
    consecutive frames.

    Args:
        frame_count (int): Number of frames in the clip.
        seed (int): RNG seed.
        n_runs (int): Number of runs.
        run_length (int): Frames per run.

    Returns:
        List[Tuple[int, int]]: Sorted ``(start, stop)`` pairs, stop exclusive.

    Raises:
        ValueError: If the clip is too short to fit the runs.
    """
    # Each run plus its one-frame separator occupies run_length + 1 slots; the
    # remaining free frames are split into n_runs + 1 random gaps.
    free = frame_count - n_runs * run_length - (n_runs - 1)
    if free < 0:
        raise ValueError(
            f"{frame_count} frames cannot fit {n_runs} runs of {run_length}"
        )
    rng = np.random.default_rng(seed)
    cuts = np.sort(rng.integers(0, free + 1, size=n_runs))
    starts = cuts + np.arange(n_runs) * (run_length + 1)
    return [(int(start), int(start) + run_length) for start in starts]


def sampled_frames(runs: Sequence[Tuple[int, int]]) -> List[int]:
    """Flatten runs into a sorted list of frame indices."""
    return [frame for start, stop in runs for frame in range(start, stop)]


def snap_to_candidate(
    click_xy: Tuple[float, float],
    candidate_xy: np.ndarray,
    radius: float = SNAP_RADIUS,
) -> Tuple[Tuple[float, float], bool]:
    """Snap a click to the nearest candidate center within ``radius``.

    Args:
        click_xy (Tuple[float, float]): Clicked point in native pixels.
        candidate_xy (np.ndarray): (M, 2) candidate centers in native pixels.
        radius (float): Snap radius in pixels (inclusive).

    Returns:
        Tuple[Tuple[float, float], bool]: The labeled point and whether it
        snapped. Without a candidate in range, the exact click is kept.
    """
    candidate_xy = np.asarray(candidate_xy, dtype=float).reshape(-1, 2)
    click = np.asarray(click_xy, dtype=float)
    if len(candidate_xy) == 0:
        return (float(click[0]), float(click[1])), False
    distances = np.linalg.norm(candidate_xy - click, axis=1)
    index = int(np.argmin(distances))
    if distances[index] <= radius:
        return (float(candidate_xy[index, 0]), float(candidate_xy[index, 1])), True
    return (float(click[0]), float(click[1])), False


def read_labels(path: str) -> Dict[int, Label]:
    """Read a label CSV. A missing file means no labels yet.

    Args:
        path (str): Path to ``labels/<clip>.csv``.

    Returns:
        Dict[int, Label]: Labels keyed by frame.

    Raises:
        ValueError: On a malformed row or a duplicate frame.
    """
    labels: Dict[int, Label] = {}
    if not os.path.exists(path):
        return labels
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(f"{path}: expected columns {CSV_COLUMNS}")
        for row in reader:
            frame = int(row["frame"])
            if frame in labels:
                raise ValueError(f"{path}: duplicate frame {frame}")
            x = float(row["x"]) if row["x"] else None
            y = float(row["y"]) if row["y"] else None
            labels[frame] = Label(frame=frame, status=row["status"], x=x, y=y)
    return labels


def write_labels(path: str, labels: Dict[int, Label]) -> None:
    """Write labels sorted by frame via write-to-temp-then-rename.

    The temporary file lives in the same directory, so ``os.replace`` is atomic
    and a crash leaves either the old or the new file, never a partial one.

    Args:
        path (str): Destination CSV.
        labels (Dict[int, Label]): Labels keyed by frame.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".labels-", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
            for frame in sorted(labels):
                label = labels[frame]
                writer.writerow([
                    label.frame,
                    "" if label.x is None else f"{label.x:.2f}",
                    "" if label.y is None else f"{label.y:.2f}",
                    label.status,
                ])
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def first_unlabeled(frames: Sequence[int], labels: Dict[int, Label]) -> int:
    """Position in ``frames`` to resume at: the first unlabeled frame, else the last."""
    for position, frame in enumerate(frames):
        if frame not in labels:
            return position
    return max(len(frames) - 1, 0)


# --------------------------------------------------------------------------- GUI

KEYS_NEXT = {ord("d"), 83, 65363, 63235, 2555904}  # 'd', right arrow (Linux/Mac/Win)
KEYS_PREV = {ord("a"), 81, 65361, 63234, 2424832}  # 'a', left arrow
KEYS_QUIT = {ord("q"), 27}
WINDOW = "label_tool"
INSET_RADIUS = 40
INSET_ZOOM = 4


class LabelSession:
    """Interactive labeling state for one clip (OpenCV window)."""

    def __init__(
        self,
        video_path: str,
        labels_path: str,
        frames: List[int],
        candidates: DetectionCache,
        display_scale: float,
    ) -> None:
        import cv2

        self.cv2 = cv2
        self.capture = cv2.VideoCapture(video_path)
        if not self.capture.isOpened():
            raise FileNotFoundError(video_path)
        self.labels_path = labels_path
        self.labels = read_labels(labels_path)
        self.frames = frames
        self.candidates = candidates
        self.scale = display_scale
        self.position = first_unlabeled(frames, self.labels)
        self.cursor = (0, 0)
        self.image: Optional[np.ndarray] = None

    @property
    def frame(self) -> int:
        return self.frames[self.position]

    def load_frame(self) -> None:
        self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, self.frame)
        ok, image = self.capture.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {self.frame}")
        self.image = image

    def set_label(self, label: Label) -> None:
        self.labels[label.frame] = label
        write_labels(self.labels_path, self.labels)
        self.step(1)

    def step(self, delta: int) -> None:
        position = min(max(self.position + delta, 0), len(self.frames) - 1)
        if position != self.position:
            self.position = position
            self.load_frame()

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        self.cursor = (x, y)
        if event == self.cv2.EVENT_LBUTTONDOWN:
            click = (x / self.scale, y / self.scale)
            (bx, by), _ = snap_to_candidate(
                click, self.candidates.centers_in_frame(self.frame)
            )
            self.set_label(Label(frame=self.frame, status="visible", x=bx, y=by))

    def render(self) -> np.ndarray:
        cv2 = self.cv2
        canvas = self.image.copy()
        overlay = canvas.copy()
        for cx, cy in self.candidates.centers_in_frame(self.frame):
            cv2.circle(overlay, (int(round(cx)), int(round(cy))), 10, (255, 255, 0), 1)
        canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
        label = self.labels.get(self.frame)
        if label is not None and label.status == "visible":
            cv2.drawMarker(
                canvas, (int(round(label.x)), int(round(label.y))), (0, 0, 255),
                cv2.MARKER_CROSS, 14, 1,
            )
        if self.scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=self.scale, fy=self.scale)

        # Zoomed inset around the cursor, taken from the full-resolution frame.
        h, w = self.image.shape[:2]
        cx = int(np.clip(self.cursor[0] / self.scale, INSET_RADIUS, w - INSET_RADIUS))
        cy = int(np.clip(self.cursor[1] / self.scale, INSET_RADIUS, h - INSET_RADIUS))
        crop = self.image[
            cy - INSET_RADIUS:cy + INSET_RADIUS, cx - INSET_RADIUS:cx + INSET_RADIUS
        ]
        inset = cv2.resize(crop, None, fx=INSET_ZOOM, fy=INSET_ZOOM,
                           interpolation=cv2.INTER_NEAREST)
        mid = inset.shape[0] // 2
        cv2.drawMarker(inset, (mid, mid), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
        ih, iw = inset.shape[:2]
        if ih <= canvas.shape[0] and iw <= canvas.shape[1]:
            canvas[:ih, -iw:] = inset

        status = label.status if label is not None else "-"
        done = sum(frame in self.labels for frame in self.frames)
        text = (f"frame {self.frame}  [{self.position + 1}/{len(self.frames)}]  "
                f"label: {status}  done: {done}/{len(self.frames)}")
        cv2.putText(canvas, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def run(self) -> None:
        cv2 = self.cv2
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        self.load_frame()
        while True:
            cv2.imshow(WINDOW, self.render())
            key = cv2.waitKeyEx(30)
            if key == -1:
                continue
            if key in KEYS_QUIT:
                break
            if key in KEYS_NEXT:
                self.step(1)
            elif key in KEYS_PREV:
                self.step(-1)
            elif key == ord("n"):
                self.set_label(Label(frame=self.frame, status="not_visible"))
            elif key == ord("u"):
                self.set_label(Label(frame=self.frame, status="unsure"))
        cv2.destroyAllWindows()
        self.capture.release()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clip", required=True, help="clip name, e.g. 08fd33_0")
    parser.add_argument("--video", required=True, help="path to <clip>.mp4")
    parser.add_argument("--detections", default=None,
                        help="cache path (default: detections/<clip>.npz)")
    parser.add_argument("--labels", default=None,
                        help="output CSV (default: labels/<clip>.csv)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--display-scale", type=float, default=0.75,
                        help="window scale; labels are stored in native pixels")
    args = parser.parse_args(argv)

    detections_path = args.detections or os.path.join(HERE, "detections", f"{args.clip}.npz")
    labels_path = args.labels or os.path.join(HERE, "labels", f"{args.clip}.csv")
    cache = load_detection_cache(detections_path)
    runs = sample_runs(cache.frame_count, clip_seed(args.clip, args.seed))
    frames = sampled_frames(runs)
    print(f"{args.clip}: runs {runs}")
    LabelSession(args.video, labels_path, frames, cache, args.display_scale).run()


if __name__ == "__main__":
    main()
