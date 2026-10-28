"""Small statistics helpers (pure Python): exact McNemar test, paired bootstrap, percentiles."""
from __future__ import annotations

import math
import random


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    xs = sorted(values)
    if not xs:
        raise ValueError("percentile of empty list")
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mcnemar_exact(baseline_only: int, latest_only: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pairs: samples only the baseline got right vs only
    the latest run got right. Small p means the two runs differ by more than coin-flip disagreement."""
    n = baseline_only + latest_only
    if n == 0:
        return 1.0
    k = min(baseline_only, latest_only)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_bootstrap_diff(baseline: list[float], latest: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean per-sample score of `latest` minus `baseline`, with a 95% percentile bootstrap interval (resampling samples)."""
    if len(baseline) != len(latest) or not baseline:
        raise ValueError("per-sample scores must be non-empty and the same length")
    diffs = [l - b for b, l in zip(baseline, latest)]
    rng, n = random.Random(seed), len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(diffs) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]
