"""Compares the latest eval run against a baseline and fails (exit 1) on a real regression.

Accuracy: a drop beyond --max-accuracy-drop fails the gate, but when both runs carry per-sample scores the drop must
also be statistically real (exact McNemar test on 0/1 scores, paired-bootstrap interval otherwise); a drop that is
indistinguishable from noise only warns. Latency: p95 (or mean if p95 is absent) against the baseline's, with a
relative allowance. Spend: when the scorer reports `tokens_mean` (see scorers.evaluate_llm) it is compared the same way. The baseline is the median of the last --window earlier runs of the same scorer (window 1 = the
previous run). Only ever compares a scorer with its own history.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

from evaltrack.stats import mcnemar_exact, paired_bootstrap_diff


def load_history(path: Path, scorer: str) -> list[dict]:
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in records if r["scorer"] == scorer]


def _latency(r: dict) -> float:
    return r.get("latency_p95_ms", r["latency_ms"])


def accuracy_evidence(baseline: dict, latest: dict, alpha: float) -> tuple[bool, str]:
    """(is the drop statistically real, description). Without per-sample data the drop is taken at face value."""
    a, b = baseline.get("samples"), latest.get("samples")
    if not a or not b or len(a) != len(b):
        return True, "no per-sample scores, threshold only"
    if all(x in (0, 1) for x in a + b):
        p = mcnemar_exact(sum(x == 1 and y == 0 for x, y in zip(a, b)), sum(x == 0 and y == 1 for x, y in zip(a, b)))
        return p < alpha, f"McNemar p={p:.4f}"
    diff, lo, hi = paired_bootstrap_diff(a, b)
    return hi < 0, f"paired bootstrap diff {diff:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})"


def check(records: list[dict], max_accuracy_drop: float = 0.05, max_latency_increase: float = 0.5,
          window: int = 1, alpha: float = 0.05, max_token_increase: float = 0.5) -> tuple[bool, str]:
    if len(records) < 2:
        return True, "Not enough history to compare yet (need at least 2 runs), passing by default."

    latest, earlier = records[-1], records[:-1][-window:]
    base_acc = statistics.median(r["accuracy"] for r in earlier)
    base_lat = statistics.median(_latency(r) for r in earlier)
    drop = base_acc - latest["accuracy"]
    latency_increase = (_latency(latest) - base_lat) / base_lat if base_lat > 0 else 0.0

    problems, notes = [], []
    if "tokens_mean" in latest and all("tokens_mean" in r for r in earlier):
        base_tok = statistics.median(r["tokens_mean"] for r in earlier)
        token_increase = (latest["tokens_mean"] - base_tok) / base_tok if base_tok > 0 else 0.0
        if token_increase > max_token_increase:
            problems.append(f"tokens per sample rose {token_increase:.1%} ({base_tok:.0f} -> {latest['tokens_mean']:.0f}), "
                            f"exceeds allowed {max_token_increase:.0%}")
        else:
            notes.append(f"tokens/sample {base_tok:.0f} -> {latest['tokens_mean']:.0f}")
    if drop > max_accuracy_drop:
        real, evidence = accuracy_evidence(earlier[-1], latest, alpha)
        if real:
            problems.append(f"accuracy dropped {drop:.4f} (baseline {base_acc:.4f} -> {latest['accuracy']:.4f}), "
                            f"exceeds allowed {max_accuracy_drop} [{evidence}]")
        else:
            notes.append(f"drop of {drop:.4f} is within noise [{evidence}]")
    if latency_increase > max_latency_increase:
        problems.append(f"latency increased {latency_increase:.1%} ({base_lat:.1f}ms -> {_latency(latest):.1f}ms), "
                        f"exceeds allowed {max_latency_increase:.0%}")

    if problems:
        return False, "REGRESSION: " + "; ".join(problems)
    suffix = f" (note: {'; '.join(notes)})" if notes else ""
    return True, f"OK: accuracy {base_acc:.4f} -> {latest['accuracy']:.4f}, latency {base_lat:.1f}ms -> {_latency(latest):.1f}ms{suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scorer_module")
    ap.add_argument("--history", default="history.jsonl")
    ap.add_argument("--max-accuracy-drop", type=float, default=0.05)
    ap.add_argument("--max-latency-increase", type=float, default=0.5)
    ap.add_argument("--max-token-increase", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=1, help="baseline = median of this many earlier runs")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    records = load_history(Path(args.history), args.scorer_module)
    ok, message = check(records, args.max_accuracy_drop, args.max_latency_increase, args.window, args.alpha, args.max_token_increase)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
