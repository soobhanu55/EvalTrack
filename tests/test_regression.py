from evaltrack.regression import check


def test_passes_with_less_than_two_runs():
    ok, msg = check([{"accuracy": 0.9, "latency_ms": 100}], 0.05, 0.5)
    assert ok


def test_catches_real_accuracy_drop():
    records = [
        {"accuracy": 0.75, "latency_ms": 110, "scorer": "x"},
        {"accuracy": 0.0, "latency_ms": 110, "scorer": "x"},
    ]
    ok, msg = check(records, max_accuracy_drop=0.05, max_latency_increase=0.5)
    assert not ok
    assert "accuracy dropped" in msg


def test_catches_real_latency_spike():
    records = [
        {"accuracy": 0.75, "latency_ms": 100, "scorer": "x"},
        {"accuracy": 0.75, "latency_ms": 300, "scorer": "x"},
    ]
    ok, msg = check(records, max_accuracy_drop=0.05, max_latency_increase=0.5)
    assert not ok
    assert "latency increased" in msg


def test_passes_on_small_noise():
    records = [
        {"accuracy": 0.750, "latency_ms": 110, "scorer": "x"},
        {"accuracy": 0.748, "latency_ms": 112, "scorer": "x"},
    ]
    ok, msg = check(records, max_accuracy_drop=0.05, max_latency_increase=0.5)
    assert ok


def test_passes_on_improvement():
    records = [
        {"accuracy": 0.0, "latency_ms": 110, "scorer": "x"},
        {"accuracy": 0.75, "latency_ms": 109, "scorer": "x"},
    ]
    ok, msg = check(records, max_accuracy_drop=0.05, max_latency_increase=0.5)
    assert ok


def rec(acc, tokens=None, lat=100.0):
    r = {"accuracy": acc, "latency_ms": lat, "n": 10}
    if tokens is not None:
        r["tokens_mean"] = tokens
    return r


def test_token_spend_increase_fails_the_gate_even_when_accuracy_improves():
    from evaltrack.regression import check

    ok, msg = check([rec(0.60, 200), rec(0.80, 700)])
    assert not ok and "tokens per sample rose 250.0%" in msg


def test_token_spend_within_allowance_passes_and_is_reported():
    from evaltrack.regression import check

    ok, msg = check([rec(0.60, 200), rec(0.62, 260)])
    assert ok and "tokens/sample 200 -> 260" in msg


def test_token_check_is_skipped_when_the_baseline_has_no_token_data():
    from evaltrack.regression import check

    assert check([rec(0.60), rec(0.62, 9999)])[0]


def test_evaluate_llm_records_mean_token_usage():
    from evaltrack.scorers import evaluate_llm

    out = evaluate_llm(lambda x: (x, {"input_tokens": 100 + x, "output_tokens": 10}), [(1, 1), (3, 3)])
    assert out["accuracy"] == 1.0 and out["input_tokens_mean"] == 102 and out["tokens_mean"] == 112
