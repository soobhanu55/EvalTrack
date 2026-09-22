"""Compares the latest eval run against the previous one and fails (exit 1)
on a real regression: accuracy drop beyond --max-accuracy-drop, or latency
increase beyond --max-latency-increase (fraction, e.g. 0.5 = 50% slower).
Only ever looks at the same scorer's own history, never compares across
different eval sets.
"""
import argparse
import json
import sys
from pathlib import Path


def load_history(path: Path, scorer: str) -> list[dict]:
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in records if r["scorer"] == scorer]


def check(records: list[dict], max_accuracy_drop: float, max_latency_increase: float) -> tuple[bool, str]:
    if len(records) < 2:
        return True, "Not enough history to compare yet (need at least 2 runs), passing by default."

    prev, latest = records[-2], records[-1]
    accuracy_drop = prev["accuracy"] - latest["accuracy"]
    latency_increase = (
        (latest["latency_ms"] - prev["latency_ms"]) / prev["latency_ms"]
        if prev["latency_ms"] > 0 else 0.0
    )

    problems = []
    if accuracy_drop > max_accuracy_drop:
        problems.append(
            f"accuracy dropped {accuracy_drop:.4f} ({prev['accuracy']:.4f} -> {latest['accuracy']:.4f}), "
            f"exceeds allowed {max_accuracy_drop}"
        )
    if latency_increase > max_latency_increase:
        problems.append(
            f"latency increased {latency_increase:.1%} ({prev['latency_ms']:.1f}ms -> {latest['latency_ms']:.1f}ms), "
            f"exceeds allowed {max_latency_increase:.0%}"
        )

    if problems:
        return False, "REGRESSION: " + "; ".join(problems)
    return True, f"OK: accuracy {prev['accuracy']:.4f} -> {latest['accuracy']:.4f}, latency {prev['latency_ms']:.1f}ms -> {latest['latency_ms']:.1f}ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scorer_module")
    ap.add_argument("--history", default="history.jsonl")
    ap.add_argument("--max-accuracy-drop", type=float, default=0.05)
    ap.add_argument("--max-latency-increase", type=float, default=0.5)
    args = ap.parse_args()

    records = load_history(Path(args.history), args.scorer_module)
    ok, message = check(records, args.max_accuracy_drop, args.max_latency_increase)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
