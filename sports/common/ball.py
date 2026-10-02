from collections import deque

import cv2
import numpy as np
import supervision as sv


class BallAnnotator:
    """
    A class to annotate frames with circles of varying radii and colors.

    Attributes:
        radius (int): The maximum radius of the circles to be drawn.
        buffer (deque): A deque buffer to store recent coordinates for annotation.
        color_palette (sv.ColorPalette): A color palette for the circles.
        thickness (int): The thickness of the circle borders.
    """

    def __init__(self, radius: int, buffer_size: int = 5, thickness: int = 2):

        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        """
        Interpolates the radius between 1 and the maximum radius based on the index.

        Args:
            i (int): The current index in the buffer.
            max_i (int): The maximum index in the buffer.

        Returns:
            int: The interpolated radius.
        """
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Annotates the frame with circles based on detections.

        Args:
            frame (np.ndarray): The frame to annotate.
            detections (sv.Detections): The detections containing coordinates.

        Returns:
            np.ndarray: The annotated frame.
        """
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)
        self.buffer.append(xy)
        for i, xy in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))
            for center in xy:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=interpolated_radius,
                    color=color.as_bgr(),
                    thickness=self.thickness
                )
        return frame


class BallTracker:
    """
    A class used to track a soccer ball's position across video frames.

    The BallTracker class maintains a buffer of recent ball positions and uses this
    buffer to predict the ball's position in the current frame by selecting the
    detection closest to the average position (centroid) of the recent positions.

    Attributes:
        buffer (collections.deque): A deque buffer to store recent ball positions.
    """
    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates the buffer with new detections and returns the detection closest to the
        centroid of recent positions.

        Args:
            detections (sv.Detections): The current frame's ball detections.

        Returns:
            sv.Detections: The detection closest to the centroid of recent positions.
            If there are no detections, returns the input detections.
        """
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)

        if len(detections) == 0:
            return detections

        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = np.argmin(distances)
        return detections[[index]]


class KalmanBallTracker:
    """
    Online ball tracker that picks one candidate per frame with a Kalman filter.

    The state is `[x, y, vx, vy]` with a constant-velocity model, one step per
    frame; the measurement is the box center (the same anchor as `BallTracker`).
    Each frame the filter predicts the ball position, gates the candidates by
    Mahalanobis distance under the innovation covariance, and updates on the
    nearest candidate in the gate. The gate widens as uncertainty grows during
    misses.

    - While tracking, any candidate in the gate is eligible regardless of
      confidence. Starting a track requires confidence >= `init_confidence`.
    - If nothing is in the gate but a candidate has confidence >=
      `reacquire_confidence` (e.g. after a kick), the track restarts on it.
    - After `max_misses` consecutive frames without an accepted candidate the
      track is dropped until a confident candidate appears.

    The output is always a subset of the input with at most one detection; no
    box is ever synthesized. Missing `confidence` is treated as 1.0.

    Attributes:
        process_noise (float): Acceleration tolerance in px/frame².
        measurement_noise (float): Detection jitter in px.
        gate_threshold (float): Squared Mahalanobis gate (chi-square, 2 dof).
        init_confidence (float): Minimum confidence to start a track.
        reacquire_confidence (float): Minimum confidence to restart an existing
            track on a candidate outside the gate.
        max_misses (int): Consecutive misses after which the track is dropped.
    """

    # Velocity uncertainty of a new track, in px/frame. A new track knows the
    # position but not the motion, so the first gate is wide enough for a
    # moving ball (about 30 px with the default noise settings).
    INITIAL_VELOCITY_STD = 10.0

    def __init__(
        self,
        process_noise: float = 5.0,
        measurement_noise: float = 2.0,
        gate_threshold: float = 9.21,
        init_confidence: float = 0.3,
        reacquire_confidence: float = 0.6,
        max_misses: int = 10,
    ) -> None:
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.gate_threshold = gate_threshold
        self.init_confidence = init_confidence
        self.reacquire_confidence = reacquire_confidence
        self.max_misses = max_misses

        self._F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self._H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        # Discrete white-noise acceleration, dt = 1 frame.
        self._Q = process_noise ** 2 * np.array([
            [0.25, 0.0, 0.5, 0.0],
            [0.0, 0.25, 0.0, 0.5],
            [0.5, 0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0, 1.0],
        ])
        self._R = measurement_noise ** 2 * np.eye(2)
        self.reset()

    @property
    def is_tracking(self) -> bool:
        """Whether a track currently exists."""
        return self._x is not None

    def reset(self) -> None:
        """
        Drops the current track, if any.
        """
        self._x = None
        self._P = None
        self._misses = 0

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Advances the filter by one frame and selects the ball among the candidates.

        Args:
            detections (sv.Detections): The current frame's ball candidates.

        Returns:
            sv.Detections: A subset of `detections` with at most one detection,
            the selected ball. Empty when no candidate is accepted.
        """
        xy = detections.get_anchors_coordinates(sv.Position.CENTER).astype(float)
        if detections.confidence is None:
            confidence = np.ones(len(detections))
        else:
            confidence = np.asarray(detections.confidence, dtype=float)

        if self._x is None:
            return self._start(detections, xy, confidence, self.init_confidence)

        self._predict()
        S = self._H @ self._P @ self._H.T + self._R
        innovation = xy - self._x[:2]
        distance = np.einsum(
            "ni,ij,nj->n", innovation, np.linalg.inv(S), innovation
        )
        in_gate = distance <= self.gate_threshold
        if in_gate.any():
            index = int(np.argmin(np.where(in_gate, distance, np.inf)))
            self._correct(xy[index], S)
            self._misses = 0
            return detections[[index]]

        selected = self._start(detections, xy, confidence, self.reacquire_confidence)
        if len(selected) > 0:
            return selected

        self._misses += 1
        if self._misses >= self.max_misses:
            self.reset()
        return detections[np.zeros(0, dtype=int)]

    def _start(
        self,
        detections: sv.Detections,
        xy: np.ndarray,
        confidence: np.ndarray,
        min_confidence: float,
    ) -> sv.Detections:
        eligible = confidence >= min_confidence
        if not eligible.any():
            return detections[np.zeros(0, dtype=int)]
        index = int(np.argmax(np.where(eligible, confidence, -np.inf)))
        self._x = np.array([xy[index, 0], xy[index, 1], 0.0, 0.0])
        self._P = np.diag([
            self.measurement_noise ** 2,
            self.measurement_noise ** 2,
            self.INITIAL_VELOCITY_STD ** 2,
            self.INITIAL_VELOCITY_STD ** 2,
        ])
        self._misses = 0
        return detections[[index]]

    def _predict(self) -> None:
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

    def _correct(self, z: np.ndarray, S: np.ndarray) -> None:
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ (z - self._x[:2])
        I_KH = np.eye(4) - K @ self._H
        self._P = I_KH @ self._P @ I_KH.T + K @ self._R @ K.T
