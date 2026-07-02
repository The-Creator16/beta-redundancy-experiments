from __future__ import annotations

import numpy as np

from redundancy_exp import DigitWindow, make_default_systems
from redundancy_exp.corruption import corrupt_single_digit, run_single_digit_corruption_benchmark_for_system


def test_single_digit_corruption_changes_exactly_one_cell():
    rng = np.random.default_rng(123)
    system = make_default_systems()["phi"]
    word = (0, 1, 0, 0, 1, 0)
    corrupted, idx, old, new = corrupt_single_digit(system, word, rng)
    assert old != new
    assert corrupted[idx] == new
    assert sum(a != b for a, b in zip(word, corrupted)) == 1


def test_experiment_3_runs_small_binary():
    rng = np.random.default_rng(123)
    system = make_default_systems()["binary"]
    window = DigitWindow.balanced(8)
    summary, positions = run_single_digit_corruption_benchmark_for_system(system, window, trials=50, rng=rng)
    assert summary.trials == 50
    assert summary.structural_detectability_rate == 0.0
    assert summary.structural_correctness_rate == 1.0
    assert summary.structural_exactness_rate == 1.0
    assert len(positions) > 0


def test_experiment_3_phi_has_some_detectable_trials():
    rng = np.random.default_rng(456)
    system = make_default_systems()["phi"]
    window = DigitWindow.balanced(12)
    summary, _ = run_single_digit_corruption_benchmark_for_system(system, window, trials=500, rng=rng)
    assert 0.0 <= summary.structural_detectability_rate <= 1.0
    assert summary.structural_detectability_rate > 0.0
