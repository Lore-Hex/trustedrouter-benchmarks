from __future__ import annotations

from trbench.evals.terminal_bench.run import _aggregate_task_rows


def test_aggregate_task_rows_counts_errors_as_unresolved_tasks():
    rows = [
        {
            "model": "m",
            "task": "passed",
            "accuracy": 100.0,
            "resolved": 1,
            "n_tasks": 1,
            "input_tokens": 10,
            "output_tokens": 2,
            "resolved_ids": ["passed"],
            "unresolved_ids": [],
        },
        {
            "model": "m",
            "task": "failed",
            "accuracy": 0.0,
            "resolved": 0,
            "n_tasks": 1,
            "input_tokens": 20,
            "output_tokens": 3,
            "resolved_ids": [],
            "unresolved_ids": ["failed"],
        },
        {
            "model": "m",
            "task": "timed-out",
            "error": "per-model timeout (1200.0s)",
            "input_tokens": 0,
            "output_tokens": 0,
        },
    ]

    [aggregate] = _aggregate_task_rows(rows)

    assert aggregate["resolved"] == 1
    assert aggregate["n_tasks"] == 3
    assert aggregate["accuracy"] == 33.3
    assert aggregate["resolved_ids"] == ["passed"]
    assert aggregate["unresolved_ids"] == ["failed", "timed-out"]
    assert aggregate["errors"] == [
        {"task": "timed-out", "error": "per-model timeout (1200.0s)"}
    ]
