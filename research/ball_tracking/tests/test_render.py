"""Smoke test of render.py on a tiny synthetic video."""

import cv2
import numpy as np
import pytest

from cache import load_detection_cache, save_detection_cache
from render import load_tracks, main, render_comparison, tile

W, H, N = 96, 64, 12


@pytest.fixture
def clip(tmp_path):
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    assert writer.isOpened()
    path = np.stack([10 + 6 * np.arange(N), np.full(N, 30.0)], axis=1)
    for x, y in path:
        frame = np.full((H, W, 3), (40, 120, 40), dtype=np.uint8)
        cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    centers = np.vstack([path, np.tile([[80.0, 50.0]], (N, 1))])
    frame_index = np.concatenate([np.arange(N), np.arange(N)])
    save_detection_cache(
        str(tmp_path / "clip.npz"), frame_index,
        np.hstack([centers - 3, centers + 3]), np.full(2 * N, 0.5),
        {"fps": 10, "width": W, "height": H, "frame_count": N, "confidence_threshold": 0.05},
    )
    online = path.copy()
    online[5] = np.nan
    offline = path.copy()
    interpolated = np.zeros(N, dtype=bool)
    interpolated[4:6] = True
    np.savez(tmp_path / "tracks.npz", baseline_xy=online, kalman_005_xy=online,
             offline_005_xy=offline, offline_005_interpolated=interpolated)
    return tmp_path


def read_video(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def test_load_tracks(clip):
    tracks = load_tracks(str(clip / "tracks.npz"))
    assert set(tracks) == {"baseline", "kalman_005", "offline_005"}
    assert not tracks["baseline"]["interpolated"].any()
    assert tracks["offline_005"]["interpolated"].sum() == 2


@pytest.mark.parametrize("rows, grid", [
    (["baseline", "offline_005"], (1, 2)),
    (["baseline", "kalman_005", "offline_005"], (2, 2)),
])
def test_render_comparison(clip, rows, grid):
    out = clip / "out" / "compare.mp4"
    count = render_comparison(str(clip / "clip.mp4"), load_detection_cache(str(clip / "clip.npz")),
                              load_tracks(str(clip / "tracks.npz")), rows, str(out), scale=1.0)
    assert count == N
    frames = read_video(out)
    assert len(frames) == N
    assert frames[0].shape[:2] == (grid[0] * H, grid[1] * W)


def test_cli_with_frame_range_and_scale(clip):
    out = clip / "cli.mp4"
    main(["--clip", "clip", "--video", str(clip / "clip.mp4"),
          "--detections", str(clip / "clip.npz"), "--tracks", str(clip / "tracks.npz"),
          "--rows", "baseline", "kalman_005", "--output", str(out),
          "--start", "2", "--end", "8", "--scale", "0.5"])
    frames = read_video(out)
    assert len(frames) == 6
    assert frames[0].shape[:2] == (H // 2, W)


def test_rejects_unknown_row_and_resolution_mismatch(clip):
    cache = load_detection_cache(str(clip / "clip.npz"))
    tracks = load_tracks(str(clip / "tracks.npz"))
    with pytest.raises(ValueError, match="not in tracks"):
        render_comparison(str(clip / "clip.mp4"), cache, tracks, ["kalman_025"],
                          str(clip / "x.mp4"))
    cache.meta["width"] = 1920
    with pytest.raises(ValueError, match="cache is"):
        render_comparison(str(clip / "clip.mp4"), cache, tracks, ["baseline"],
                          str(clip / "x.mp4"))


def test_tile_pads_grid():
    panels = [np.full((2, 3, 3), i, dtype=np.uint8) for i in range(3)]
    grid = tile(panels)
    assert grid.shape == (4, 6, 3)
    assert (grid[2:, 3:] == 0).all()
