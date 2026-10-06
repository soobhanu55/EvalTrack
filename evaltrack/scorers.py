"""Helpers for writing a scorer's run(): evaluate any predict function over a labelled dataset and return the
dict the harness records, including per-sample scores (for significance tests) and latency percentiles."""
from __future__ import annotations

import time
from typing import Any, Callable, Iterable

from evaltrack.stats import percentile


def exact_match(prediction: Any, expected: Any) -> float:
    return float(prediction == expected)


def evaluate(predict: Callable[[Any], Any], dataset: Iterable[tuple[Any, Any]],
             metric: Callable[[Any, Any], float] = exact_match) -> dict:
    """dataset: (input, expected) pairs. metric returns a score in [0, 1] per sample (default exact match)."""
    scores, latencies = [], []
    for x, expected in dataset:
        start = time.perf_counter()
        pred = predict(x)
        latencies.append((time.perf_counter() - start) * 1000)
        scores.append(metric(pred, expected))
    if not scores:
        raise ValueError("empty dataset")
    return {
        "accuracy": sum(scores) / len(scores),
        "latency_ms": sum(latencies) / len(latencies),
        "latency_p95_ms": percentile(latencies, 95),
        "n": len(scores),
        "samples": [int(s) if s in (0.0, 1.0) else s for s in scores],
    }
