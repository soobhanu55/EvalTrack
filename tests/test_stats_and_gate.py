import pytest

from evaltrack.regression import accuracy_evidence, check
from evaltrack.stats import mcnemar_exact, paired_bootstrap_diff, percentile


def rec(acc, lat=100.0, samples=None, **extra):
    r = {"accuracy": acc, "latency_ms": lat, "scorer": "s", "n": len(samples) if samples else 100, **extra}
    if samples is not None:
        r["samples"] = samples
    return r


# --- statistics -------------------------------------------------------------------------------

def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
    assert percentile([1, 2, 3, 4, 5], 100) == 5


def test_percentile_of_empty_list_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_mcnemar_known_value():
    # n = 12 discordant, 2 vs 10: p = 2 * (C(12,0)+C(12,1)+C(12,2)) / 2^12 = 2 * 79 / 4096
    assert mcnemar_exact(10, 2) == pytest.approx(158 / 4096)


def test_mcnemar_symmetric_and_capped_at_one():
    assert mcnemar_exact(3, 9) == mcnemar_exact(9, 3)
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 0) == 1.0


def test_paired_bootstrap_identical_runs_have_zero_diff():
    diff, lo, hi = paired_bootstrap_diff([1, 0, 1, 1], [1, 0, 1, 1])
    assert diff == lo == hi == 0


def test_paired_bootstrap_interval_covers_the_true_difference():
    base = [1.0] * 100
    new = [1.0] * 80 + [0.0] * 20
    diff, lo, hi = paired_bootstrap_diff(base, new)
    assert diff == pytest.approx(-0.2) and lo < -0.2 < hi and hi < 0


def test_paired_bootstrap_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 0], [1])


# --- gate: threshold behaviour kept from v0.1 ------------------------------------------------------

def test_needs_two_runs():
    assert check([rec(0.9)])[0]
    assert check([])[0]


def test_catches_real_accuracy_drop_without_samples():
    ok, msg = check([rec(0.75), rec(0.0)])
    assert not ok and "accuracy dropped" in msg and "threshold only" in msg


def test_catches_latency_spike():
    ok, msg = check([rec(0.75, 100), rec(0.75, 300)])
    assert not ok and "latency increased" in msg


def test_small_noise_and_improvement_pass():
    assert check([rec(0.750, 110), rec(0.748, 112)])[0]
    assert check([rec(0.0), rec(0.75)])[0]


def test_latency_uses_p95_when_present():
    ok, msg = check([rec(0.8, 100, latency_p95_ms=120), rec(0.8, 100, latency_p95_ms=400)])
    assert not ok and "latency increased" in msg


# --- gate: significance and baseline window --------------------------------------------------------

def test_drop_within_noise_only_warns():
    a = [1] * 7 + [0] * 3 + [1] * 70 + [0] * 20  # 0.77
    b = [0] * 7 + [1] * 3 + [1] * 70 + [0] * 20  # 0.73: 7 lost, 3 gained, McNemar p = 0.34
    ok, msg = check([rec(0.77, samples=a), rec(0.73, samples=b)], max_accuracy_drop=0.03)
    assert ok and "within noise" in msg and "McNemar" in msg


def test_large_consistent_drop_fails_with_mcnemar():
    a = [1] * 100
    b = [1] * 80 + [0] * 20
    ok, msg = check([rec(1.0, samples=a), rec(0.8, samples=b)])
    assert not ok and "McNemar p=" in msg


def test_non_binary_scores_use_paired_bootstrap():
    a = [0.9] * 50
    b = [0.7] * 50
    ok, msg = check([rec(0.9, samples=a), rec(0.7, samples=b)])
    assert not ok and "paired bootstrap" in msg


def test_evidence_falls_back_when_sample_lengths_differ():
    real, text = accuracy_evidence({"samples": [1, 0]}, {"samples": [1]}, 0.05)
    assert real and "threshold only" in text


def test_window_median_baseline_resists_one_unlucky_previous_run():
    runs = [rec(0.80), rec(0.80), rec(0.80), rec(0.50), rec(0.60)]
    assert check(runs, window=1)[0]  # better than the unlucky previous run
    ok, msg = check(runs, window=4)  # worse than the median of the last four earlier runs (0.80)
    assert not ok and "accuracy dropped" in msg
