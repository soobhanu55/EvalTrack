# Changelog

## 0.3.0
- Token-spend gate: `evaluate_llm` records tokens per sample and `--max-token-increase` fails the build when spend jumps.
- Packaging for PyPI as `evaltrack-gate` (import name `evaltrack_gate`), a single `evaltrack-gate run|check|report` command, no runtime dependencies.

## 0.2.0
- Significance-aware accuracy gate (exact McNemar test on 0/1 scores, paired bootstrap otherwise), rolling-median baseline (`--window`), p95 latency, scorer helpers (`evaluate`), markdown report.

## 0.1.0
- Fixed-threshold accuracy and latency gate over an append-only `history.jsonl`, GitHub Actions workflow, browser dashboard.
