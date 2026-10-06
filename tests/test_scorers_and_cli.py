import json
import sys

import pytest

from evaltrack import report, runner
from evaltrack.regression import load_history
from evaltrack.scorers import evaluate


def test_evaluate_scores_and_records_per_sample_results():
    data = [(1, 2), (2, 4), (3, 7)]
    out = evaluate(lambda x: x * 2, data)
    assert out["accuracy"] == pytest.approx(2 / 3)
    assert out["samples"] == [1, 1, 0] and out["n"] == 3
    assert 0 <= out["latency_ms"] <= out["latency_p95_ms"] * 3 + 1


def test_evaluate_accepts_a_fractional_metric():
    out = evaluate(lambda x: x, [(1, 1), (1, 3)], metric=lambda p, e: 1 - abs(p - e) / 4)
    assert out["accuracy"] == pytest.approx(0.75) and out["samples"] == [1, 0.5]


def test_evaluate_rejects_empty_dataset():
    with pytest.raises(ValueError):
        evaluate(lambda x: x, [])


@pytest.fixture
def scorer_module(tmp_path, monkeypatch):
    (tmp_path / "toy_scorer.py").write_text(
        "from evaltrack.scorers import evaluate\n"
        "def run():\n    return evaluate(lambda x: x % 2, [(i, i % 2) for i in range(20)])\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("toy_scorer", None)
    return "toy_scorer"


def test_runner_appends_a_tagged_record_with_samples(scorer_module, tmp_path):
    hist = tmp_path / "h.jsonl"
    result = runner.run_scorer(scorer_module)
    record = runner.append_history(hist, scorer_module, result)
    stored = json.loads(hist.read_text(encoding="utf-8").splitlines()[0])
    assert stored["scorer"] == scorer_module and stored["accuracy"] == 1.0 and len(stored["samples"]) == 20
    assert stored["timestamp"] == record["timestamp"] and stored["git_sha"]


def test_runner_rejects_inconsistent_sample_count(tmp_path, monkeypatch):
    (tmp_path / "bad_scorer.py").write_text(
        "def run():\n    return {'accuracy': 1.0, 'latency_ms': 1.0, 'n': 3, 'samples': [1, 1]}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError, match="per-sample"):
        runner.run_scorer("bad_scorer")


def test_runner_rejects_missing_keys(tmp_path, monkeypatch):
    (tmp_path / "short_scorer.py").write_text("def run():\n    return {'accuracy': 1.0}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError, match="missing"):
        runner.run_scorer("short_scorer")


def test_load_history_filters_by_scorer_and_handles_missing_file(tmp_path):
    assert load_history(tmp_path / "none.jsonl", "x") == []
    h = tmp_path / "h.jsonl"
    h.write_text(json.dumps({"scorer": "a", "accuracy": 1}) + "\n\n" + json.dumps({"scorer": "b", "accuracy": 0}) + "\n")
    assert [r["accuracy"] for r in load_history(h, "a")] == [1]


def test_report_markdown_shows_status_and_table():
    recs = [{"timestamp": "t1", "git_sha": "abc", "accuracy": 0.8, "latency_ms": 10.0, "n": 5, "scorer": "s"},
            {"timestamp": "t2", "git_sha": "def", "accuracy": 0.1, "latency_ms": 10.0, "n": 5, "scorer": "s"}]
    md = report.markdown(recs)
    assert md.startswith("**FAIL**") and "`def`" in md and "0.1000" in md
    assert report.markdown([]).startswith("**PASS**")


def test_cli_entry_points_run_check_and_report(scorer_module, tmp_path, monkeypatch, capsys):
    from evaltrack import regression

    hist = str(tmp_path / "h.jsonl")
    monkeypatch.setattr(sys, "argv", ["runner", scorer_module, "--history", hist])
    assert runner.main() == 0
    assert runner.main() == 0  # second run so there is something to compare
    monkeypatch.setattr(sys, "argv", ["regression", scorer_module, "--history", hist, "--window", "1"])
    assert regression.main() == 0
    monkeypatch.setattr(sys, "argv", ["report", scorer_module, "--history", hist, "--last", "1"])
    assert report.main() == 0
    assert "**PASS**" in capsys.readouterr().out


def test_cli_exits_nonzero_on_regression(tmp_path, monkeypatch):
    from evaltrack import regression

    hist = tmp_path / "h.jsonl"
    rows = [{"scorer": "s", "accuracy": a, "latency_ms": 10.0, "n": 10, "timestamp": "t", "git_sha": "x"} for a in (0.9, 0.2)]
    hist.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["regression", "s", "--history", str(hist)])
    assert regression.main() == 1
