"""Read and write per-clip ball detection caches (`detections/<clip>.npz`).

The format is described in the "Detection cache" section of DESIGN.md:

- ``frame_index`` (N,) int
- ``xyxy`` (N, 4) float32, pixels at native resolution
- ``confidence`` (N,) float32
- ``meta`` JSON string (fps, width, height, frame_count, model SHA-256,
  confidence cutoff, slicer settings, supervision and ultralytics versions)

``detect_ball.ipynb`` writes the same format inline, because it runs on Colab
without this repository installed.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import supervision as sv

REQUIRED_META_KEYS = ("fps", "width", "height", "frame_count", "confidence_threshold")


@dataclass
class DetectionCache:
    """All ball candidates of one clip.

    Attributes:
        frame_index (np.ndarray): (N,) int, frame of each candidate.
        xyxy (np.ndarray): (N, 4) float32 boxes in native pixels.
        confidence (np.ndarray): (N,) float32 detector confidence.
        meta (Dict[str, Any]): Parsed ``meta`` JSON.
    """

    frame_index: np.ndarray
    xyxy: np.ndarray
    confidence: np.ndarray
    meta: Dict[str, Any]

    @property
    def frame_count(self) -> int:
        return int(self.meta["frame_count"])

    def frames(self, min_confidence: float = 0.0) -> List[sv.Detections]:
        """Split the cache into one `sv.Detections` per frame.

        Args:
            min_confidence (float): Keep candidates with confidence >= this.

        Returns:
            List[sv.Detections]: ``frame_count`` items; frames without candidates
            are empty detections.
        """
        keep = self.confidence >= min_confidence
        frame_index = self.frame_index[keep]
        xyxy = self.xyxy[keep]
        confidence = self.confidence[keep]
        order = np.argsort(frame_index, kind="stable")
        frame_index, xyxy, confidence = frame_index[order], xyxy[order], confidence[order]
        bounds = np.searchsorted(frame_index, np.arange(self.frame_count + 1))
        return [
            sv.Detections(
                xyxy=xyxy[start:end],
                confidence=confidence[start:end],
                class_id=np.zeros(end - start, dtype=int),
            )
            for start, end in zip(bounds[:-1], bounds[1:])
        ]

    def centers_in_frame(self, frame: int) -> np.ndarray:
        """Return (M, 2) candidate box centers in one frame."""
        xyxy = self.xyxy[self.frame_index == frame]
        return (xyxy[:, :2] + xyxy[:, 2:]) / 2


def save_detection_cache(
    path: str,
    frame_index: np.ndarray,
    xyxy: np.ndarray,
    confidence: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    """Write a detection cache atomically.

    Args:
        path (str): Destination ``.npz`` path.
        frame_index (np.ndarray): (N,) frame of each candidate.
        xyxy (np.ndarray): (N, 4) boxes in native pixels.
        confidence (np.ndarray): (N,) detector confidence.
        meta (Dict[str, Any]): Clip and pipeline metadata, stored as JSON.
    """
    tmp_path = path + ".tmp.npz"
    np.savez_compressed(
        tmp_path,
        frame_index=np.asarray(frame_index, dtype=np.int64),
        xyxy=np.asarray(xyxy, dtype=np.float32).reshape(-1, 4),
        confidence=np.asarray(confidence, dtype=np.float32),
        meta=np.array(json.dumps(meta, sort_keys=True)),
    )
    os.replace(tmp_path, path)


def load_detection_cache(path: str) -> DetectionCache:
    """Load and sanity-check a detection cache.

    Args:
        path (str): Path to ``<clip>.npz``.

    Returns:
        DetectionCache: The loaded cache.

    Raises:
        ValueError: If arrays are inconsistent with each other or with ``meta``.
    """
    with np.load(path, allow_pickle=False) as data:
        frame_index = data["frame_index"].astype(np.int64)
        xyxy = data["xyxy"].astype(np.float32).reshape(-1, 4)
        confidence = data["confidence"].astype(np.float32)
        meta = json.loads(str(data["meta"]))

    missing = [key for key in REQUIRED_META_KEYS if key not in meta]
    if missing:
        raise ValueError(f"{path}: meta is missing {missing}")
    if not (len(frame_index) == len(xyxy) == len(confidence)):
        raise ValueError(f"{path}: frame_index, xyxy and confidence lengths differ")
    if len(frame_index) and (
        frame_index.min() < 0 or frame_index.max() >= int(meta["frame_count"])
    ):
        raise ValueError(f"{path}: frame_index outside [0, frame_count)")
    return DetectionCache(frame_index, xyxy, confidence, meta)
