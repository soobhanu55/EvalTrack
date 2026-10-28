# Contributing

```bash
git clone https://github.com/soobhanu55/EvalTrack && cd EvalTrack
pip install -e ".[dev]"
pytest -q --cov=evaltrack_gate --cov-fail-under=90
```

Pull requests are welcome. Keep the library standard-library only (a heavy dependency belongs in an example, not in `evaltrack_gate/`),
add a test with every change, and keep claims in the README tied to something the repo measures.
Good first issues: a scorer helper for another model type, a JUnit/JSON output for `check`, a second example that runs in CI.
