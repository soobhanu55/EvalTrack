"""Markdown summary of a scorer's history (for the GitHub Actions job summary or a PR comment)."""
from __future__ import annotations

import argparse
from pathlib import Path

from evaltrack.regression import check, load_history


def markdown(records: list[dict], last: int = 10) -> str:
    ok, message = check(records) if records else (True, "no runs recorded")
    lines = [f"**{'PASS' if ok else 'FAIL'}**: {message}", "",
             "| Time (UTC) | Commit | Accuracy | Latency ms (p95 if known) | n |", "|---|---|---|---|---|"]
    for r in records[-last:]:
        lines.append(f"| {r['timestamp']} | `{r['git_sha']}` | {r['accuracy']:.4f} | "
                     f"{r.get('latency_p95_ms', r['latency_ms']):.1f} | {r['n']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scorer_module")
    ap.add_argument("--history", default="history.jsonl")
    ap.add_argument("--last", type=int, default=10)
    args = ap.parse_args()
    print(markdown(load_history(Path(args.history), args.scorer_module), args.last))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
