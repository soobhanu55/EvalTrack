"""How often does each gate cry wolf, and how often does it catch a real regression? (writes docs/gate_eval.md)

    python scripts/simulate_gate.py

Setup: an LLM-style scorer is nondeterministic, so each of n eval items is answered correctly with its own
probability p_i (half the items are near-certain, the rest coin-flip-ish). Two runs of the SAME scorer differ by
sampling noise alone; a "regressed" scorer has p_i lowered on a share of items. The fixed-threshold gate fails on any
drop above 5 points; the significance gate also needs McNemar p < 0.05.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaltrack.regression import check  # noqa: E402

RUNS = 2000


def run(probs: list[float], rng: random.Random) -> dict:
    samples = [int(rng.random() < p) for p in probs]
    return {"accuracy": sum(samples) / len(samples), "latency_ms": 100.0, "n": len(samples), "samples": samples}


def rates(n: int, regress_to: float | None, rng: random.Random) -> tuple[float, float]:
    base = [0.97] * (n // 2) + [rng.uniform(0.4, 0.8) for _ in range(n - n // 2)]
    new = base if regress_to is None else [p * regress_to for p in base]
    fixed = signif = 0
    for _ in range(RUNS):
        a, b = run(base, rng), run(new, rng)
        a["scorer"] = b["scorer"] = "s"
        fixed += (a["accuracy"] - b["accuracy"]) > 0.05
        signif += not check([a, b])[0]
    return fixed / RUNS, signif / RUNS


def main() -> None:
    rng = random.Random(0)
    lines = ["# Gate simulation: fixed threshold vs significance test\n",
             f"{RUNS} simulated pairs of runs per row. 'No change' rows compare two runs of the same scorer (any alarm is a false alarm); "
             "'regressed' rows lower every item's success probability to 90% (about a 6-8 point drop).\n",
             "| Eval size | Scenario | Fixed-threshold gate fails | Significance gate fails |", "|---|---|---|---|"]
    for n in (50, 100, 300):
        for label, factor in (("no change", None), ("regressed (x0.90)", 0.90)):
            f, s = rates(n, factor, rng)
            lines.append(f"| {n} | {label} | {f:.1%} | {s:.1%} |")
    lines += ["", "On small evals a fixed 5-point threshold fires on noise alone; requiring the drop to be statistically real removes "
                  "those false alarms, at the cost of missing some real regressions when n is small (power rises with eval size)."]
    out = ROOT / "docs" / "gate_eval.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
