"""Runs a scorer against a fixed eval set and appends a real result to
history.jsonl. A "scorer" is any importable module with a module-level
`run()` function returning {"accuracy": float, "latency_ms": float, "n": int}.
Works the same whether that function calls a local model or an LLM API,
the harness doesn't care what's inside run(), only that it returns those
three numbers honestly.
"""
import argparse
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def run_scorer(scorer_module: str) -> dict:
    mod = importlib.import_module(scorer_module)
    start = time.time()
    result = mod.run()
    elapsed_ms = (time.time() - start) * 1000

    if "latency_ms" not in result:
        result["latency_ms"] = elapsed_ms

    required = {"accuracy", "latency_ms", "n"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"scorer.run() missing required keys: {missing}")

    return result


def append_history(history_path: Path, scorer_module: str, result: dict):
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        "scorer": scorer_module,
        **result,
    }
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scorer_module", help="importable module with a run() function")
    ap.add_argument("--history", default="history.jsonl")
    args = ap.parse_args()

    result = run_scorer(args.scorer_module)
    record = append_history(Path(args.history), args.scorer_module, result)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
