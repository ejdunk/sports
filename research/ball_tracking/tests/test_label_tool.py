import os

import numpy as np
import pytest

from label_tool import (
    Label,
    clip_seed,
    first_unlabeled,
    read_labels,
    sample_runs,
    sampled_frames,
    snap_to_candidate,
    write_labels,
)


@pytest.mark.parametrize("frame_count", [104, 150, 750])
@pytest.mark.parametrize("seed", range(20))
def test_sample_runs_are_disjoint_non_adjacent_and_in_range(frame_count, seed):
    runs = sample_runs(frame_count, seed)
    assert len(runs) == 5
    assert all(stop - start == 20 for start, stop in runs)
    assert runs[0][0] >= 0 and runs[-1][1] <= frame_count
    for (_, prev_stop), (start, _) in zip(runs, runs[1:]):
        assert start >= prev_stop + 1
    assert len(set(sampled_frames(runs))) == 100


def test_sample_runs_is_seeded():
    assert sample_runs(750, 3) == sample_runs(750, 3)
    assert any(sample_runs(750, 3) != sample_runs(750, s) for s in range(4, 10))


def test_sample_runs_rejects_short_clip():
    with pytest.raises(ValueError):
        sample_runs(103, 0)


def test_clip_seed_is_stable_and_clip_specific():
    assert clip_seed("08fd33_0") == clip_seed("08fd33_0")
    assert clip_seed("08fd33_0") != clip_seed("0bfacc_0")
    assert clip_seed("08fd33_0", 1) != clip_seed("08fd33_0", 0)


def test_snap_within_radius():
    candidates = np.array([[100.0, 100.0], [105.0, 100.0], [300.0, 300.0]])
    xy, snapped = snap_to_candidate((104.0, 101.0), candidates)
    assert snapped and xy == (105.0, 100.0)


def test_snap_radius_is_inclusive_and_8px():
    candidates = np.array([[100.0, 100.0]])
    assert snap_to_candidate((108.0, 100.0), candidates)[1]
    xy, snapped = snap_to_candidate((108.5, 100.0), candidates)
    assert not snapped and xy == (108.5, 100.0)


def test_snap_without_candidates_keeps_click():
    xy, snapped = snap_to_candidate((10.0, 20.0), np.zeros((0, 2)))
    assert not snapped and xy == (10.0, 20.0)


def test_label_validation():
    with pytest.raises(ValueError):
        Label(frame=0, status="visible")
    with pytest.raises(ValueError):
        Label(frame=0, status="not_visible", x=1.0, y=2.0)
    with pytest.raises(ValueError):
        Label(frame=0, status="maybe")


def test_csv_round_trip_and_atomic_write(tmp_path):
    path = str(tmp_path / "labels" / "clip.csv")
    assert read_labels(path) == {}
    labels = {
        12: Label(12, "visible", 10.25, 20.5),
        3: Label(3, "not_visible"),
        7: Label(7, "unsure"),
    }
    write_labels(path, labels)
    assert read_labels(path) == labels
    with open(path) as f:
        lines = f.read().splitlines()
    assert lines[0] == "frame,x,y,status"
    assert [line.split(",")[0] for line in lines[1:]] == ["3", "7", "12"]
    assert os.listdir(tmp_path / "labels") == ["clip.csv"]  # no temp left behind


def test_failed_write_keeps_previous_file(tmp_path, monkeypatch):
    path = str(tmp_path / "clip.csv")
    write_labels(path, {1: Label(1, "unsure")})

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        write_labels(path, {1: Label(1, "not_visible")})
    monkeypatch.undo()
    assert read_labels(path) == {1: Label(1, "unsure")}
    assert os.listdir(tmp_path) == ["clip.csv"]


def test_read_rejects_duplicates(tmp_path):
    path = tmp_path / "clip.csv"
    path.write_text("frame,x,y,status\n1,,,unsure\n1,,,unsure\n")
    with pytest.raises(ValueError):
        read_labels(str(path))


def test_resume_position():
    frames = [10, 11, 12, 13]
    assert first_unlabeled(frames, {}) == 0
    labels = {10: Label(10, "unsure"), 11: Label(11, "unsure"), 13: Label(13, "unsure")}
    assert first_unlabeled(frames, labels) == 2
    labels[12] = Label(12, "unsure")
    assert first_unlabeled(frames, labels) == 3
