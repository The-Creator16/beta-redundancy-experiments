from __future__ import annotations

import numpy as np

from redundancy_exp import DigitWindow, make_default_systems
from redundancy_exp.burst import corrupt_burst, run_burst_corruption_benchmark_for_system


def test_burst_corruption_changes_at_least_one_digit() -> None:
    rng = np.random.default_rng(1)
    systems = make_default_systems()
    word = (0, 1, 0, 0, 1, 0, 0, 1)

    corrupted, start, mask, active_count = corrupt_burst(
        systems["phi"],
        word,
        burst_width=3,
        rng=rng,
    )

    assert len(corrupted) == len(word)
    assert 0 <= start <= len(word) - 3
    assert len(mask) == 3
    assert active_count >= 1
    assert corrupted != word


def test_burst_benchmark_runs_for_phi() -> None:
    rng = np.random.default_rng(2)
    systems = make_default_systems()
    window = DigitWindow.balanced(8)

    summary, by_width = run_burst_corruption_benchmark_for_system(
        systems["phi"],
        window,
        trials=25,
        rng=rng,
        burst_widths=(2, 3),
    )

    assert summary.system == "phi"
    assert summary.W == 8
    assert summary.trials == 25
    assert {row.burst_width for row in by_width}.issubset({2, 3})
    assert 0.0 <= summary.structural_detectability_rate <= 1.0
    assert 0.0 <= summary.semantic_survival_rate <= 1.0
