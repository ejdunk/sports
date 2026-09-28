# Ball tracking: design

Status: approved design, 2026-09-28. Implementation plan to follow.

## Goal

Replace the naive soccer `BallTracker` with trackers that pick the right ball
candidate more often and recover more frames, and prove it with measured
results.

Two outcomes, both required:

1. **Upstream PR** to `roboflow/sports`: an additive, drop-in online tracker
   (`KalmanBallTracker`) with tests. Small, focused, no new dependencies.
2. **Write-up** in this fork: a results table (baseline vs online vs offline)
   with confidence intervals and before/after videos.

Out of scope for this spec: the RADAR flicker fix (separate design round
after this ships), learned trackers (TrackNet-style), basketball.

## Background

`sports/common/ball.py::BallTracker` keeps a deque of the last 20 frames'
candidate centers and, each frame, returns the candidate nearest their mean.
It has no motion model, returns nothing when the detector misses, and stale
positions pull the mean. `examples/soccer/main.py` runs the ball detector
(YOLOv8x) through `sv.InferenceSlicer` with 640×640 tiles, default confidence
0.25, then `with_nms(threshold=0.1)`.

Related upstream context:

- README lists ball tracking as an open challenge.
- Issue #18 ("Multiple Ball Detections") asks how to pick the correct one.
- PR #66 (Roboflow staff, stacked on `feat/soccer-*` branches) adds
  `select_best_ball()` to `ball.py` and leaves `BallTracker` untouched. Our
  change is additive, so any conflict should be trivial.
- `main` has had no commits since 2025-05; external PRs often go unreviewed.
  Hence issue-first, and the write-up stands on its own regardless of merge.

## Open-source practice

- Open an issue on `roboflow/sports` before the PR, with a baseline failure
  clip and early results. Nothing is posted without the user's approval.
- Fork to `ejdunk/sports`; `origin` = fork, `upstream` = roboflow.
- Research work lives on `research/ball-tracking` under `research/`. It never
  goes upstream.
- The PR branch is cut from `upstream/main` and rebased before opening. It
  contains only `KalmanBallTracker`, its tests, and a one-line opt-in in
  `examples/soccer/main.py`.
- `BallTracker` is left exactly as is. No new dependencies (numpy only).
- Match repo style: Google-style docstrings, type hints, pytest in `tests/`.
- All code is original and contributed under MIT. Match videos and model
  weights stay out of git; we commit only coordinates we produce.
- PR body follows the roboflow org template, includes before/after clips and
  the results table, and discloses AI assistance.
- Send the PR after the maintainers respond to the issue, or after ~2 weeks of
  silence. One polite follow-up after ~2 weeks without review; no repeated
  pings.
- Offline tracker (B) is a separate follow-up PR, only if maintainers want it.

## Architecture

```
Colab (GPU, once)                     Mac (CPU, fast iteration)
─────────────────                     ─────────────────────────────────────────
detect_ball.ipynb                     label_tool.py ──► labels/<clip>.csv
  5 clips → YOLOv8x + slicer                │
  conf ≥ 0.05, NMS 0.1                      ▼
  └──► detections/<clip>.npz ────────► evaluate.py ◄── trackers:
       (committed)                          │            • BallTracker (baseline)
                                            │            • KalmanBallTracker (A)
                                            │            • track_ball_offline (B)
                                            ▼
                                      results/ + render.py → videos
```

The key boundary: trackers never touch video or models. They take
per-frame candidates and return per-frame positions. This makes them
testable with synthetic data and trivially swappable.

The detector runs once, on Colab, because the development machine is an
Intel Mac without a usable GPU (YOLOv8x × 6 tiles/frame ≈ 10 s/frame on CPU).

### Layout

```
sports/common/ball.py            + KalmanBallTracker            (upstream PR)
sports/common/ball_offline.py      track_ball_offline, OfflineBallTrack
tests/synthetic.py                 synthetic scene generator     (upstream PR)
tests/test_ball_tracker.py         tests for A                   (upstream PR)
tests/test_ball_offline.py         tests for B
research/ball_tracking/
  DESIGN.md                        this file
  README.md                        write-up (step 7)
  detect_ball.ipynb                Colab detection pass
  label_tool.py                    OpenCV labeling tool
  metrics.py                       scoring
  evaluate.py                      runs trackers, writes results/
  render.py                        side-by-side videos
  detections/<clip>.npz            cached candidates (committed)
  labels/<clip>.csv                ground truth (committed)
  results/                         tables and plots (committed)
  tests/test_metrics.py            tests for scoring
```

Environment: `.venv` from `/usr/local/bin/python3.11` (system `python3` is
3.7, too old). Colab uses its own runtime.

## Data

Clips (from `examples/soccer/setup.sh`): `08fd33_0`, `0bfacc_0`, `121364_0`,
`2e57b9_0`, `573e61_0`.

**Split, fixed now (sorted order, before any results):**

- Tune: `08fd33_0`, `0bfacc_0`
- Test: `121364_0`, `2e57b9_0`, `573e61_0`

All parameter tuning uses tune clips only. Test numbers are computed once
parameters are frozen.

### Detection cache (`detections/<clip>.npz`)

Produced by `detect_ball.ipynb` with the same pipeline as `main.py`
(`InferenceSlicer`, 640×640 tiles, `OverlapFilter.NONE`, `imgsz=640`, then
`with_nms(threshold=0.1)`), except the confidence cutoff is lowered to 0.05.

Arrays:

- `frame_index` (N,) int
- `xyxy` (N, 4) float32, pixels at native resolution
- `confidence` (N,) float32
- `meta`: JSON string with fps, width, height, frame_count, model file
  SHA-256, confidence cutoff, slicer settings, `supervision` and
  `ultralytics` versions.

The baseline is evaluated on candidates filtered to confidence ≥ 0.25, which
reproduces `main.py` (NMS keeps the higher-confidence box, so filtering after
NMS equals filtering before).

### Labels (`labels/<clip>.csv`)

Columns: `frame, x, y, status` where status ∈ {`visible`, `not_visible`,
`unsure`}. `x, y` are the ball center in pixels, empty unless visible.

Sampling: per clip, 5 runs of 20 consecutive frames at random start offsets
(seeded, non-overlapping). 25 runs, 500 frames total. Runs are random, not
chosen at "interesting" moments, so the comparison is fair to the baseline.

Optional: re-label 50 random frames later to report self-consistency.

## Tracker A: `KalmanBallTracker` (online)

```python
class KalmanBallTracker:
    def __init__(
        self,
        process_noise: float = 5.0,       # acceleration tolerance (px/frame²)
        measurement_noise: float = 2.0,   # detection jitter (px)
        gate_threshold: float = 9.21,     # chi-square, 2 dof, 99%
        init_confidence: float = 0.3,
        reacquire_confidence: float = 0.6,
        max_misses: int = 10,
    ) -> None: ...

    def update(self, detections: sv.Detections) -> sv.Detections: ...
    def reset(self) -> None: ...
```

Defaults above are starting guesses; final defaults come from tuning on the
tune clips.

- State `[x, y, vx, vy]`, constant-velocity model, one step per frame.
  Measurement is the box center (same anchor as `BallTracker`).
- **Tracking:** predict; compute the Mahalanobis distance of each candidate to
  the prediction using the innovation covariance; keep candidates under
  `gate_threshold`; select the nearest; update. The gate widens as
  uncertainty grows during misses.
- **Weak detections count while tracking:** any candidate in the gate is
  eligible regardless of confidence. Starting or restarting a track requires
  confidence ≥ `init_confidence`.
- **Kicks:** if nothing is in the gate but a candidate has confidence ≥
  `reacquire_confidence`, re-initialize on it immediately.
- **Lost:** after `max_misses` consecutive frames with no accepted candidate,
  drop the track and wait for a confident candidate.
- **Output:** a subset of the input with at most one detection, same fields.
  Never a synthesized box. Gap filling is B's job.
- Missing `confidence` (None) is treated as 1.0.

## Tracker B: `track_ball_offline` (whole clip)

```python
@dataclass
class OfflineBallTrack:
    xy: np.ndarray            # (T, 2) float, NaN where no ball
    interpolated: np.ndarray  # (T,) bool
    source_index: np.ndarray  # (T,) int, candidate index in that frame, -1 if none

def track_ball_offline(
    candidates: list[sv.Detections],  # one per frame
    max_speed: float = 40.0,          # px/frame
    max_gap: int = 10,
    gap_penalty: float = 1.0,
    motion_penalty: float = 0.05,
    segment_penalty: float = 5.0,
) -> OfflineBallTrack: ...
```

- Dynamic programming over a layered graph: nodes are candidates; an edge
  links a candidate at frame t to one at t+k (1 ≤ k ≤ `max_gap`) when the
  displacement is ≤ `max_speed · k`.
- Score: sum of candidate log-odds from confidence, minus `motion_penalty` ×
  displacement, minus `gap_penalty` × skipped frames. The path may break into
  segments at `segment_penalty` each, so long occlusions don't force bad links.
- Gaps ≤ `max_gap` inside a segment are filled by linear interpolation and
  flagged in `interpolated`.
- Defaults are starting guesses, tuned on the tune clips like A's.
- Cost ≈ T × C² × `max_gap` (750 × 20² × 10 ≈ 3M), well under a second in
  numpy.

## Evaluation

Per labeled frame, with radius r:

| Ground truth | Tracker output   | Outcome                      |
|--------------|------------------|------------------------------|
| visible      | within r px      | TP                           |
| visible      | farther than r   | FP + FN                      |
| visible      | nothing          | FN                           |
| not visible  | something        | FP                           |
| not visible  | nothing          | TN                           |
| unsure       | anything         | excluded                     |

- **Primary:** precision, recall, F1 at r = 10 px; r = 5 and 20 px as
  sensitivity. Mean localization error on TPs. The primary r may be revised
  after labeling the tune clips (if the ball is much larger than expected),
  but is frozen before any test-clip evaluation.
- **B:** recall split into detected vs interpolated frames.
- **Uncertainty:** bootstrap 95% CIs resampling whole 20-frame runs, since
  frames within a run are correlated.
- **Label-free checks on all frames:** output coverage, jump rate
  (consecutive outputs further apart than `max_speed`), time per frame.
- **Rows in the results table:**
  - Baseline (`BallTracker`, conf ≥ 0.25)
  - A at conf ≥ 0.25 (motion model alone)
  - A at conf ≥ 0.05 (motion model + weak detections)
  - B at conf ≥ 0.05
- Per-clip breakdown alongside the pooled test numbers.

## Labeling tool

OpenCV window, runs locally.

- Shows the frame with a zoomed inset around the cursor. All raw candidates
  are drawn faintly; no tracker's selection is shown, to avoid anchoring.
- Click: mark ball center. Snaps to a candidate center within 8 px, otherwise
  keeps the exact click.
- `n`: not visible. `u`: unsure. Arrow keys: previous/next frame.
- Autosaves after every label via write-to-temp-then-rename. Resumable.
- Expected effort: 45–70 minutes for 500 frames.

## Testing

No test needs video, a model, or a GPU. All run in seconds.

- `tests/synthetic.py` generates scenes: straight runs, lobbed arcs, sudden
  kicks, Gaussian position noise, dropout stretches, a stationary
  high-confidence decoy (penalty spot), and low-confidence clutter.
- **A (upstream):** follows the ball past a higher-confidence stationary
  decoy; accepts a weak in-gate detection; rejects a weak out-of-gate one
  while searching; re-acquires after a kick; drops the track after
  `max_misses`; handles empty input and `confidence=None`; output is always
  a subset of input with ≤ 1 detection.
- **B:** recovers the true path through decoys; fills gaps ≤ `max_gap` and not
  longer ones; flags interpolated frames; never exceeds `max_speed`.
- **Metrics:** hand-built cases covering every row of the outcome table.

## Robustness

- `evaluate.py` checks each detection cache's `meta` against its labels and
  refuses mismatches (frames past the clip's end, wrong resolution).
- Labeling tool writes atomically so a crash cannot corrupt labels.

## Order of work

Each step produces something reviewable.

1. Setup: fork (with approval), venv, run a few frames of the existing
   pipeline on CPU to confirm the install.
2. `detect_ball.ipynb`; user runs it on Colab (~30 min); commit caches.
3. `label_tool.py`; user labels 500 frames; commit labels.
4. `metrics.py` + `evaluate.py` + tests; baseline numbers.
5. `KalmanBallTracker` + tests; numbers. Draft the GitHub issue (baseline
   failure clip + early results) for the user to review and post.
6. `track_ball_offline` + tests; numbers.
7. `render.py`, before/after videos, write-up in `README.md`.
8. Upstream PR (A only) from a clean branch off `upstream/main`, after the
   issue gets a response or ~2 weeks pass.
9. Later: B follow-up PR if wanted; RADAR flicker fix via its own design.

## Risks

- **Kicks break constant velocity.** Mitigated by the gate, the
  `reacquire_confidence` rule, and `max_misses`; measured on real clips.
- **Maintainers don't respond.** The write-up does not depend on a merge.
- **Conflict with PR #66** in `ball.py`. Additive change; rebase if it lands.
- **Ball size vs r.** Checked during tune-clip labeling, frozen before test
  evaluation.
- **Colab availability.** Fallback is an overnight CPU run on the Mac
  (~10–12 h for all clips); the cache format is the same.
