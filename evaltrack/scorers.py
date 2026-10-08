"""Helpers for writing a scorer's run(): evaluate any predict function over a labelled dataset and return the
dict the harness records, including per-sample scores (for significance tests) and latency percentiles."""
from __future__ import annotations

import time
from typing import Any, Callable, Iterable

from evaltrack.stats import percentile


def exact_match(prediction: Any, expected: Any) -> float:
    return float(prediction == expected)


def evaluate_llm(call: Callable[[Any], tuple[Any, dict]], dataset: Iterable[tuple[Any, Any]],
                 metric: Callable[[Any, Any], float] = exact_match) -> dict:
    """Like evaluate(), for a model call that reports its token usage: call(x) -> (prediction, {"input_tokens": n,
    "output_tokens": m}). The result also carries mean tokens per sample, which the regression gate compares so a prompt
    change that raises accuracy but triples the spend does not ship unnoticed."""
    usage: list[dict] = []

    def predict(x):
        prediction, used = call(x)
        usage.append(used)
        return prediction

    out = evaluate(predict, dataset, metric)
    n = len(usage)
    out["input_tokens_mean"] = sum(u["input_tokens"] for u in usage) / n
    out["output_tokens_mean"] = sum(u["output_tokens"] for u in usage) / n
    out["tokens_mean"] = out["input_tokens_mean"] + out["output_tokens_mean"]
    return out


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
