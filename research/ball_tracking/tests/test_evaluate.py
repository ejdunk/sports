"""End-to-end test of evaluate.py on a tiny synthetic cache + labels fixture."""

import csv
import json
import os

import numpy as np
import pytest

from cache import save_detection_cache
from evaluate import ROWS, SPLITS, evaluate, main
from label_tool import Label, clip_seed, sample_runs, sampled_frames, write_labels

N_FRAMES = 120
WIDTH, HEIGHT = 640, 360


def ball_path(n_frames, phase):
    t = np.arange(n_frames)
    x = 320 + 200 * np.sin(2 * np.pi * (t / n_frames) + phase)
    y = 180 + 80 * np.sin(4 * np.pi * (t / n_frames) + phase)
    return np.stack([x, y], axis=1)


def make_fixture(root, clip, seed, meta_overrides=None):
    """Synthetic cache (ball + stationary decoy + clutter) and labels for a clip."""
    rng = np.random.default_rng(seed)
    path = ball_path(N_FRAMES, phase=seed)
    hidden = np.zeros(N_FRAMES, dtype=bool)
    hidden[50:56] = True

    frame_index, centers, confidence = [], [], []
    for t in range(N_FRAMES):
        if not hidden[t]:
            frame_index.append(t)
            centers.append(path[t] + rng.normal(0, 1.0, 2))
            confidence.append(0.7 if t < 10 else rng.uniform(0.15, 0.8))
        if t >= 10:  # stationary high-confidence decoy comes into view
            frame_index.append(t)
            centers.append((100.0, 300.0))
            confidence.append(0.9)
        for _ in range(rng.poisson(2)):
            frame_index.append(t)
            centers.append(rng.uniform((0, 0), (WIDTH, HEIGHT)))
            confidence.append(rng.uniform(0.05, 0.2))
    centers = np.array(centers)
    xyxy = np.hstack([centers - 5, centers + 5])
    meta = {
        "clip": clip, "fps": 25, "width": WIDTH, "height": HEIGHT,
        "frame_count": N_FRAMES, "confidence_threshold": 0.05,
        "model_sha256": "0" * 64, "supervision_version": "test",
        "ultralytics_version": "test",
    }
    meta.update(meta_overrides or {})
    os.makedirs(root / "detections", exist_ok=True)
    save_detection_cache(str(root / "detections" / f"{clip}.npz"),
                         np.array(frame_index), xyxy, np.array(confidence), meta)

    labels = {}
    for i, frame in enumerate(sampled_frames(sample_runs(N_FRAMES, clip_seed(clip)))):
        if hidden[frame]:
            labels[frame] = Label(frame, "not_visible")
        elif i % 17 == 0:
            labels[frame] = Label(frame, "unsure")
        else:
            labels[frame] = Label(frame, "visible", *path[frame])
    write_labels(str(root / "labels" / f"{clip}.csv"), labels)
    return path


@pytest.fixture
def fixture_dir(tmp_path):
    for seed, clip in enumerate(SPLITS["tune"]):
        make_fixture(tmp_path, clip, seed)
    return tmp_path


def run(root, **kwargs):
    return evaluate(
        detections_dir=str(root / "detections"),
        labels_dir=str(root / "labels"),
        results_dir=str(root / "results"),
        n_boot=200,
        **kwargs,
    )


def test_end_to_end_on_tune_split(fixture_dir):
    rows = run(fixture_dir, split="tune")
    out = fixture_dir / "results" / "tune"
    for name in ("metrics.csv", "label_free.csv", "summary.md", "config.json"):
        assert (out / name).exists()

    groups = list(SPLITS["tune"]) + ["pooled"]
    assert len(rows) == len(ROWS) * len(groups) * 3
    with open(out / "metrics.csv") as f:
        written = list(csv.DictReader(f))
    assert len(written) == len(rows)
    for record in written:
        for stat in ("precision", "recall", "f1"):
            if record[stat]:
                assert 0.0 <= float(record[stat]) <= 1.0
                assert float(record[f"{stat}_lo"]) <= float(record[stat]) + 1e-9
                assert float(record[stat]) <= float(record[f"{stat}_hi"]) + 1e-9
        assert record["runs"] == ("10" if record["clip"] == "pooled" else "5")
    offline = [r for r in written if r["row"] == "offline_005"]
    assert all(r["recall_detected"] != "" for r in offline)
    assert all(r["recall_detected"] == "" for r in written if r["row"] != "offline_005")

    # The pooled counts are the sum of the per-clip counts.
    def get(row, clip, radius=10.0):
        return next(r for r in written
                    if r["row"] == row and r["clip"] == clip and float(r["radius"]) == radius)
    for row in ROWS:
        for key in ("tp", "fp", "fn", "tn", "n_frames"):
            assert int(get(row, "pooled")[key]) == sum(
                int(get(row, clip)[key]) for clip in SPLITS["tune"])

    # The baseline locks onto the high-confidence decoy; A follows the ball.
    assert float(get("kalman_005", "pooled")["f1"]) > float(get("baseline", "pooled")["f1"])

    with open(out / "label_free.csv") as f:
        label_free = list(csv.DictReader(f))
    assert len(label_free) == len(ROWS) * 2
    assert all(0.0 <= float(r["coverage"]) <= 1.0 for r in label_free)

    with np.load(out / "tracks" / f"{SPLITS['tune'][0]}.npz") as tracks:
        assert tracks["kalman_005_xy"].shape == (N_FRAMES, 2)
        assert tracks["offline_005_interpolated"].shape == (N_FRAMES,)

    config = json.loads((out / "config.json").read_text())
    assert config["params"]["baseline"] == {"buffer_size": 20}
    summary = (out / "summary.md").read_text()
    assert "Baseline (BallTracker, conf >= 0.25)" in summary


def test_params_override_and_cli(fixture_dir):
    params = fixture_dir / "params.json"
    params.write_text(json.dumps({"kalman": {"max_misses": 3}, "offline": {"max_gap": 4}}))
    main(["--split", "tune", "--clips", SPLITS["tune"][0], "--n-boot", "50",
          "--params", str(params),
          "--detections", str(fixture_dir / "detections"),
          "--labels", str(fixture_dir / "labels"),
          "--results", str(fixture_dir / "results")])
    config = json.loads((fixture_dir / "results" / "tune" / "config.json").read_text())
    assert config["params"]["kalman"] == {"max_misses": 3}
    assert config["params"]["offline"] == {"max_gap": 4}
    assert config["clips"] == [SPLITS["tune"][0]]


def test_refuses_clip_outside_split(fixture_dir):
    with pytest.raises(ValueError, match="not in the tune split"):
        run(fixture_dir, split="tune", clips=[SPLITS["test"][0]])
    with pytest.raises(ValueError, match="unknown split"):
        run(fixture_dir, split="all")


def test_refuses_labels_past_clip_end(fixture_dir):
    clip = SPLITS["tune"][0]
    make_fixture(fixture_dir, clip, 0, meta_overrides={"frame_count": 100})
    with pytest.raises(ValueError):
        run(fixture_dir, split="tune")


def test_refuses_wrong_resolution(fixture_dir):
    clip = SPLITS["tune"][1]
    make_fixture(fixture_dir, clip, 1, meta_overrides={"width": 320, "height": 180})
    with pytest.raises(ValueError, match="wrong resolution"):
        run(fixture_dir, split="tune")


def test_refuses_cache_for_other_clip_or_high_cutoff(fixture_dir):
    clip = SPLITS["tune"][0]
    make_fixture(fixture_dir, clip, 0, meta_overrides={"clip": "573e61_0"})
    with pytest.raises(ValueError, match="meta is for clip"):
        run(fixture_dir, split="tune")
    make_fixture(fixture_dir, clip, 0, meta_overrides={"confidence_threshold": 0.25})
    with pytest.raises(ValueError, match="cutoff"):
        run(fixture_dir, split="tune")


def test_refuses_missing_labels(fixture_dir):
    os.remove(fixture_dir / "labels" / f"{SPLITS['tune'][0]}.csv")
    with pytest.raises(ValueError, match="no labels"):
        run(fixture_dir, split="tune")
