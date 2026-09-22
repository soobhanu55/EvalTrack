# EvalTrack — catches a model regression before it ships

A small, general-purpose eval harness: run a fixed test set against
whatever's currently deployed, record accuracy and latency, and fail CI
the moment either one regresses past a real threshold. Not tied to any
one model type, the interface is a plain `run() -> dict`, so it works the
same for a local vision model or a hosted LLM call.

## How it works

```
scorer.run() → {accuracy, latency_ms, n} → history.jsonl (append-only)
                                                  ↓
                          regression.py compares latest vs previous run
                                                  ↓
                              CI fails on a real drop, not a guess
```

- **`evaltrack/runner.py`**: imports any module with a `run()` function,
  calls it, appends a timestamped, git-sha-tagged record to
  `history.jsonl`.
- **`evaltrack/regression.py`**: compares the two most recent runs of the
  same scorer, fails (exit 1) if accuracy drops more than
  `--max-accuracy-drop` (default 0.05) or latency rises more than
  `--max-latency-increase` (default 50%).
- **`.github/workflows/eval.yml`**: runs the eval on every push, fails
  the job on regression, commits the updated history back to the repo.
- **`dashboard/index.html`**: a small, dependency-free page (plain
  canvas, no charting library) that loads any `history.jsonl` and plots
  accuracy and latency over time. Runs entirely in the browser, no
  server or build step.

## Worked example, with a real caught regression

`examples/inspectai_defect_scorer.py` wraps
[InspectAI](https://github.com/soobhanu55/InspectAI)'s fine-tuned
defect-detection model, free and local, no API key: it runs the model
against the real 180-image NEU-DET held-out test set and reports mAP50 as
accuracy.

Three real runs, in order, recorded in `history.jsonl`:

| Run | Weights | Accuracy | What happened |
|---|---|---|---|
| 1 | fine-tuned | 0.7498 | baseline |
| 2 | stock (untrained) | 0.0000 | simulated the real failure mode of accidentally shipping an untrained checkpoint |
| 3 | fine-tuned | 0.7498 | recovery |

Running `regression.py` after run 2 against run 1:

```
REGRESSION: accuracy dropped 0.7498 (0.7498 -> 0.0000), exceeds allowed 0.05
exit code: 1
```

And after run 3 against run 2:

```
OK: accuracy 0.0000 -> 0.7498, latency 110.5ms -> 109.1ms
exit code: 0
```

This isn't a synthetic pass/fail test, it's the harness catching an
actual, realistic failure (wrong weights file deployed) and then
confirming recovery, the same three-step story a real CI run would show.

## Run it

```bash
pip install -e . -r requirements.txt
cd examples
python -m evaltrack.runner inspectai_defect_scorer --history ../history.jsonl
cd ..
python -m evaltrack.regression inspectai_defect_scorer --history history.jsonl
```

## Limitations, stated honestly

- **Compares only the two most recent runs**, not a rolling average or a
  statistically robust baseline. Good enough to catch an obvious
  regression like the one demonstrated above, not tuned for detecting
  small, noisy drift over many runs.
- **The worked example is a vision model, not an LLM**, chosen so the
  demo runs free and fast with no API key needed. The interface
  (`run() -> {accuracy, latency_ms, n}`) is exactly what an LLM-eval
  scorer would also implement, calling an LLM API inside `run()` instead
  of a local model, nothing else in the harness changes.
- **Dashboard is intentionally minimal.** No charting library, no
  aggregation across multiple scorers on one chart, it plots one
  scorer's history because that's what the worked example needed, not
  because more wasn't possible.

## Cost: €0.00

Local model inference, GitHub Actions free minutes, no paid API calls in
the worked example.
