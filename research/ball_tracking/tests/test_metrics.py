import numpy as np
import pytest

from metrics import (
    bootstrap_ci,
    coverage,
    jump_rate,
    precision_recall_f1,
    run_ids,
    score_frames,
    summarize,
)

NAN = (np.nan, np.nan)


def outcome(status, gt, pred, radius=10.0):
    o = score_frames([status], [gt], [pred], radius)
    return {
        "tp": int(o.tp[0]),
        "fp": int(o.fp[0]),
        "fn": int(o.fn[0]),
        "tn": int(o.tn[0]),
        "excluded": bool(o.excluded[0]),
        "error": float(o.error[0]),
    }


# One test per row of the outcome table.

def test_visible_within_r_is_tp():
    o = outcome("visible", (100, 100), (106, 108))
    assert (o["tp"], o["fp"], o["fn"], o["tn"]) == (1, 0, 0, 0)
    assert o["error"] == pytest.approx(10.0)  # exactly r counts as within


def test_visible_farther_than_r_is_fp_and_fn():
    o = outcome("visible", (100, 100), (111, 100))
    assert (o["tp"], o["fp"], o["fn"], o["tn"]) == (0, 1, 1, 0)
    assert np.isnan(o["error"])


def test_visible_nothing_is_fn():
    o = outcome("visible", (100, 100), NAN)
    assert (o["tp"], o["fp"], o["fn"], o["tn"]) == (0, 0, 1, 0)


def test_not_visible_something_is_fp():
    o = outcome("not_visible", NAN, (5, 5))
    assert (o["tp"], o["fp"], o["fn"], o["tn"]) == (0, 1, 0, 0)


def test_not_visible_nothing_is_tn():
    o = outcome("not_visible", NAN, NAN)
    assert (o["tp"], o["fp"], o["fn"], o["tn"]) == (0, 0, 0, 1)


@pytest.mark.parametrize("pred", [NAN, (100, 100), (500, 500)])
def test_unsure_is_excluded(pred):
    o = outcome("unsure", NAN, pred)
    assert o["excluded"]
    assert (o["tp"], o["fp"], o["fn"], o["tn"]) == (0, 0, 0, 0)


@pytest.mark.parametrize("radius, expected", [(5.0, "miss"), (10.0, "hit"), (20.0, "hit")])
def test_radius_sensitivity(radius, expected):
    o = outcome("visible", (0, 0), (6, 0), radius)
    assert o["tp"] == (expected == "hit")


def test_score_frames_validates_input():
    with pytest.raises(ValueError):
        score_frames(["seen"], [NAN], [NAN], 10)
    with pytest.raises(ValueError):
        score_frames(["visible"], [NAN], [NAN], 10)
    with pytest.raises(ValueError):
        score_frames(["visible", "unsure"], [(0, 0)], [NAN], 10)


def test_precision_recall_f1():
    assert precision_recall_f1(6, 2, 4) == pytest.approx((0.75, 0.6, 12 / 18))
    p, r, f = precision_recall_f1(0, 0, 0)
    assert np.isnan(p) and np.isnan(r) and np.isnan(f)


def table_case():
    status = ["visible", "visible", "visible", "not_visible", "not_visible", "unsure"]
    gt = [(0, 0), (0, 0), (0, 0), NAN, NAN, NAN]
    pred = [(3, 4), (50, 0), NAN, (1, 1), NAN, (0, 0)]
    return score_frames(status, gt, pred, 10.0)


def test_summarize_pools_all_rows():
    s = summarize(table_case())
    assert s["n_frames"] == 5
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (1, 2, 2, 1)
    assert s["precision"] == pytest.approx(1 / 3)
    assert s["recall"] == pytest.approx(1 / 3)
    assert s["f1"] == pytest.approx(2 / 6)
    assert s["mean_error"] == pytest.approx(5.0)
    assert "recall_detected" not in s


def test_summarize_splits_recall_by_interpolation():
    status = ["visible"] * 4
    gt = [(0, 0)] * 4
    pred = [(0, 0), (1, 0), NAN, (2, 0)]
    interpolated = [False, True, False, True]
    s = summarize(score_frames(status, gt, pred, 10.0), interpolated)
    assert s["recall"] == pytest.approx(0.75)
    assert s["recall_detected"] == pytest.approx(0.25)
    assert s["recall_interpolated"] == pytest.approx(0.5)


def test_run_ids_split_on_gaps_and_clips():
    clips = ["b", "a", "a", "a", "a", "b"]
    frames = [0, 5, 3, 4, 10, 1]
    ids = run_ids(clips, frames)
    assert ids[1] == ids[2] == ids[3]  # a: 3, 4, 5
    assert ids[4] != ids[1]  # a: 10
    assert ids[0] == ids[5]  # b: 0, 1
    assert len(set(ids)) == 3


def test_bootstrap_identical_runs_give_degenerate_ci():
    status = ["visible", "visible", "not_visible"] * 4
    gt = [(0, 0), (0, 0), NAN] * 4
    pred = [(0, 0), NAN, (1, 1)] * 4
    runs = np.repeat(np.arange(4), 3)
    ci = bootstrap_ci(score_frames(status, gt, pred, 10.0), runs, n_boot=200)
    assert ci["precision"] == pytest.approx((0.5, 0.5))
    assert ci["recall"] == pytest.approx((0.5, 0.5))
    assert ci["mean_error"] == pytest.approx((0.0, 0.0))


def test_bootstrap_resamples_runs_not_frames():
    # Run 0 is all hits, run 1 all misses. Resampling whole runs can only give
    # recall in {0, 0.5, 1}; frame-level resampling would give intermediate values.
    status = ["visible"] * 20
    gt = [(0, 0)] * 20
    pred = [(0, 0)] * 10 + [NAN] * 10
    runs = np.repeat([0, 1], 10)
    outcomes = score_frames(status, gt, pred, 10.0)
    ci = bootstrap_ci(outcomes, runs, n_boot=500, seed=1)
    assert ci["recall"][0] == pytest.approx(0.0)
    assert ci["recall"][1] == pytest.approx(1.0)
    assert bootstrap_ci(outcomes, runs, n_boot=500, seed=1) == ci


def test_bootstrap_contains_point_estimate():
    rng = np.random.default_rng(0)
    n = 200
    status = np.where(rng.random(n) < 0.8, "visible", "not_visible")
    gt = np.where((status == "visible")[:, None], 100.0, np.nan) * np.ones((n, 2))
    pred = 100.0 + rng.normal(0, 8, (n, 2))
    pred[rng.random(n) < 0.2] = np.nan
    outcomes = score_frames(status, gt, pred, 10.0)
    runs = np.arange(n) // 20
    point = summarize(outcomes, np.zeros(n, bool))
    ci = bootstrap_ci(outcomes, runs, np.zeros(n, bool), n_boot=500)
    for name in ("precision", "recall", "f1", "mean_error", "recall_detected"):
        low, high = ci[name]
        assert low <= point[name] <= high


def test_coverage_and_jump_rate():
    pred = np.array([[0, 0], [10, 0], [100, 0], NAN, [105, 0], [106, 0]], dtype=float)
    assert coverage(pred) == pytest.approx(5 / 6)
    # Pairs with both outputs: (0,1) 10px, (1,2) 90px, (4,5) 1px.
    assert jump_rate(pred, max_speed=40) == pytest.approx(1 / 3)
    assert np.isnan(jump_rate(np.full((3, 2), np.nan), 40))
    assert np.isnan(coverage(np.zeros((0, 2))))
