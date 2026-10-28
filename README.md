# EvalTrack: catches a model regression before it ships

[![Tests](https://github.com/soobhanu55/EvalTrack/actions/workflows/test.yml/badge.svg)](https://github.com/soobhanu55/EvalTrack/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

```bash
pip install evaltrack-gate        # no dependencies; import name: evaltrack_gate
```

Your scorer is any function that returns `{accuracy, latency_ms, n}`. `evaltrack-gate run` appends the result to a `history.jsonl` you
commit, and `evaltrack-gate check` exits 1 when the latest run is worse than the baseline in a way that is statistically real
(accuracy), measurably slower (p95 latency) or much more expensive (tokens per sample).

```python
# my_scorer.py
from evaltrack_gate.scorers import evaluate_llm
def run():
    return evaluate_llm(call_my_model, [("input 1", "label 1"), ("input 2", "label 2")])   # call(x) -> (prediction, {"input_tokens": n, "output_tokens": m})
```
```bash
evaltrack-gate run my_scorer --history history.jsonl
evaltrack-gate check my_scorer --history history.jsonl --window 3 --max-token-increase 0.5     # exit 1 on regression
```

**How it differs from other tools.** [evaltrack](https://pypi.org/project/evaltrack/) (a different project by another author) is a pytest
plugin that turns an existing eval runner's assertions and score bars into a gate, reruns flaky cases and stores runs with a dashboard.
EvalTrack is smaller and sits elsewhere: standard library only, a significance test on the accuracy drop (so noise on a small eval does not
fail the build), a token-spend gate, and a history file that lives in git with no storage backend. Use it when you want a few-line CI gate
with no framework; use a pytest-based tool when your evals already live in a test suite.

## How it works

```
scorer.run() → {accuracy, latency_ms, n} → history.jsonl (append-only)
                                                  ↓
                          regression.py compares latest vs previous run
                                                  ↓
                              CI fails on a real drop, not a guess
```

- **`evaltrack_gate/runner.py`**: imports any module with a `run()` function,
  calls it, appends a timestamped, git-sha-tagged record to
  `history.jsonl`.
- **`evaltrack_gate/regression.py`**: compares the two most recent runs of the
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
pip install -e . -r requirements.txt     # requirements.txt only adds the vision example's model library
cd examples
evaltrack-gate run inspectai_defect_scorer --history ../history.jsonl
cd ..
evaltrack-gate check inspectai_defect_scorer --history history.jsonl
```

## Noise-aware gate (v0.2)

A fixed threshold ("fail on a 5-point drop") fires on noise whenever the scorer is not deterministic (LLM calls,
sampling) or the eval set is small. If a scorer returns per-sample scores (`evaltrack_gate.scorers.evaluate` does this for any
`predict(x)` function and labelled dataset), the gate also requires the drop to be statistically real: an exact McNemar
test on 0/1 scores, a paired bootstrap interval otherwise. The baseline can be the median of the last N runs
(`--window N`) instead of just the previous one, and latency is compared on p95.

Simulation of 2,000 run pairs per row (`scripts/simulate_gate.py`, `docs/gate_eval.md`):

| Eval size | Fixed threshold, no real change | Significance gate, no real change | Fixed, real 6-8 point drop caught | Significance, real drop caught |
|---|---|---|---|---|
| 50 | 24.0% false alarms | 1.0% | 66.2% | 12.9% |
| 100 | 15.2% | 1.6% | 71.5% | 21.5% |
| 300 | 4.2% | 2.1% | 81.5% | 64.3% |

The trade-off is explicit: the significance gate removes most false alarms but misses more real regressions on small
evals, so use `--window` and larger eval sets, or keep the plain threshold for deterministic scorers (the InspectAI example,
whose weights either match or do not).

```python
from evaltrack_gate.scorers import evaluate
def run():
    return evaluate(my_model.predict, [(x1, y1), (x2, y2)])   # accuracy, p95 latency, per-sample scores
```

Quality gates on this repo: 36 tests, 97% line coverage (CI fails below 90%).

## Spend gate for LLM scorers (v0.3)

A prompt change can raise accuracy and still be a bad deploy if it multiplies what every request costs. A scorer built with
`evaltrack_gate.scorers.evaluate_llm` reports tokens per sample, and the gate fails when that rises more than `--max-token-increase`
(default 50%) over the baseline, whatever accuracy did.

`examples/lexpilot_tier_scorer.py` is a real LLM scorer, free and local: Qwen2.5-1.5B-Instruct classifies 60 AI-system
descriptions into EU AI Act risk tiers (LexPilot's evaluation set). Two prompts, real runs on an RTX 4050:

| Run | Prompt | Accuracy | Tokens / sample | p95 latency | Gate |
|---|---|---|---|---|---|
| 1 | terse (one-line instruction) | 35.0% | 67 | 147 ms | baseline |
| 2 | verbose (tier definitions + 4 examples) | 61.7% | 361 | 163 ms | **fails: tokens +441%** |

```
REGRESSION: tokens per sample rose 441.2% (67 -> 361), exceeds allowed 50%
```

The verbose prompt is clearly better (+26.7 points), and the gate still stops it, because at a hosted per-token price the same
traffic would cost about 5x as much. The decision is then explicit: accept it by rerunning with `--max-token-increase 5`, or
shorten the prompt. Tokens are a proxy for cost here (the local model costs nothing); multiply by your provider's price for euros.
The LLM example is not in CI (it needs torch and a model download); the InspectAI example is.

## Limitations, stated honestly

- **The CI example is a vision model; the LLM example runs locally.** The significance gate is validated on simulated noise,
  not on a live LLM; the spend gate is demonstrated on one prompt change with a 1.5B model, not on a hosted API.
- **The significance test is on one eval set.** It cannot tell a model regression from a shifted eval set; keep the
  eval set fixed and versioned.
- **The latency gate is relative.** A scorer that takes under a millisecond can vary by more than 50% run to run; raise `--max-latency-increase` or
  gate on accuracy only for such scorers.
- **Dashboard is intentionally minimal.** It plots one scorer's history from `history.jsonl`.

## Cost: €0.00

Local model inference, GitHub Actions free minutes, no paid API calls in
the worked example.
