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
