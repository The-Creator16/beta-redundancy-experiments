from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from math import log
from statistics import mean
from typing import Iterable

import numpy as np

from .corruption import (
    storage_alphabet,
    repair_single_digit_corruption,
    value_key_to_float,
    system_base_float,
)
from .systems import RepresentationSystem
from .window import DigitWindow


@dataclass(frozen=True)
class BurstTrialResult:
    """One Monte Carlo trial for Experiment 4."""

    burst_width: int
    start_index: int
    start_position: int
    mask: tuple[int, ...]
    active_count: int
    detectable: bool
    structural_correct: bool
    structural_exact: bool
    semantic_survival: bool
    normalization_success: bool
    overflow: bool
    truncation: bool
    normalization_steps: int
    propagation_depth: int
    abs_error: float
    log_error_magnitude: float


@dataclass(frozen=True)
class BurstBenchmarkSummary:
    """Aggregate metrics for Experiment 4: local burst corruption."""

    system: str
    W: int
    L: int
    R: int
    trials: int
    burst_widths: str
    mean_burst_width: float
    mean_active_count: float
    structural_detectability_rate: float
    structural_correctness_rate: float
    structural_exactness_rate: float
    semantic_survival_rate: float
    normalization_success_rate: float
    overflow_rate: float
    truncation_rate: float
    mean_abs_error: float
    worst_abs_error: float
    mean_log_error_magnitude: float
    mean_normalization_steps: float
    max_normalization_steps: int
    p50_normalization_steps: float
    p90_normalization_steps: float
    p99_normalization_steps: float
    mean_propagation_depth: float
    max_propagation_depth: int
    p50_propagation_depth: float
    p90_propagation_depth: float
    p99_propagation_depth: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BurstWidthSummary:
    """Aggregate Experiment 4 metrics conditioned on one burst width."""

    system: str
    W: int
    L: int
    R: int
    burst_width: int
    trials: int
    mean_active_count: float
    structural_detectability_rate: float
    structural_correctness_rate: float
    structural_exactness_rate: float
    semantic_survival_rate: float
    normalization_success_rate: float
    overflow_rate: float
    truncation_rate: float
    mean_abs_error: float
    mean_normalization_steps: float
    mean_propagation_depth: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _mean_bool(values: Iterable[bool]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(int(v) for v in values_list) / len(values_list)


def _quantile(values: list[int] | list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _sample_nonzero_mask(width: int, rng: np.random.Generator) -> tuple[int, ...]:
    r"""Sample uniformly from {0,1}^width \ {0^width}."""
    if width <= 0:
        raise ValueError("burst width must be positive")
    value = int(rng.integers(1, 2**width, endpoint=False))
    # MSB-to-LSB order inside the contiguous burst interval.
    return tuple((value >> shift) & 1 for shift in range(width - 1, -1, -1))


def corrupt_burst(
    system: RepresentationSystem,
    digits: tuple[int, ...],
    burst_width: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], int, tuple[int, ...], int]:
    """Apply one alphabet-aware local burst corruption.

    A contiguous burst window of size `burst_width` is sampled uniformly among
    the windows fully contained in the stored digit register.  A nonzero binary
    mask is then sampled uniformly.  Each active mask position is corrupted by
    replacing the current digit with a uniformly sampled different digit from
    the system's storage alphabet.
    """
    n = len(digits)
    if burst_width <= 0:
        raise ValueError("burst_width must be positive")
    if burst_width > n:
        raise ValueError("burst_width cannot exceed the digit window length")

    start = int(rng.integers(0, n - burst_width + 1, endpoint=False))
    mask = _sample_nonzero_mask(burst_width, rng)

    alphabet = storage_alphabet(system)
    out = list(digits)
    active_count = 0
    for offset, active in enumerate(mask):
        if not active:
            continue
        idx = start + offset
        old = int(out[idx])
        candidates = [d for d in alphabet if d != old]
        if not candidates:
            raise ValueError("burst corruption has no valid replacement digit")
        out[idx] = int(rng.choice(candidates))
        active_count += 1

    return tuple(out), start, mask, active_count


def evaluate_burst_trial(
    system: RepresentationSystem,
    window: DigitWindow,
    burst_widths: tuple[int, ...],
    rng: np.random.Generator,
    max_steps: int,
) -> BurstTrialResult:
    """Run one local burst corruption and bounded-window repair trial."""
    if not burst_widths:
        raise ValueError("burst_widths must be non-empty")

    original = system.sample_word(window, rng)
    burst_width = int(rng.choice(burst_widths))
    corrupted, start_idx, mask, active_count = corrupt_burst(system, original, burst_width, rng)
    repaired = repair_single_digit_corruption(system, corrupted, window, max_steps=max_steps)
    output = repaired.digits

    original_key = system.value_key(original, window)
    corrupted_key = system.value_key(corrupted, window)
    output_key = system.value_key(output, window)

    corrupted_admissible = system.is_admissible(corrupted)
    output_admissible = system.is_admissible(output)

    detectable = not corrupted_admissible
    structural_correct = output_admissible
    structural_exact = output_admissible and output_key == corrupted_key
    semantic_survival = output_admissible and output_key == original_key
    normalization_success = repaired.success and output_admissible

    original_value = value_key_to_float(system, original_key)
    output_value = value_key_to_float(system, output_key)
    abs_error = abs(output_value - original_value)
    beta = system_base_float(system)
    log_error = log(1.0 + abs_error, beta)

    return BurstTrialResult(
        burst_width=burst_width,
        start_index=start_idx,
        start_position=window.positions_high_to_low[start_idx],
        mask=mask,
        active_count=active_count,
        detectable=detectable,
        structural_correct=structural_correct,
        structural_exact=structural_exact,
        semantic_survival=semantic_survival,
        normalization_success=normalization_success,
        overflow=repaired.overflow,
        truncation=repaired.truncation,
        normalization_steps=repaired.steps,
        propagation_depth=repaired.propagation_depth,
        abs_error=abs_error,
        log_error_magnitude=log_error,
    )


def _summarize_burst_width(
    system_name: str,
    window: DigitWindow,
    burst_width: int,
    trials: list[BurstTrialResult],
) -> BurstWidthSummary:
    if not trials:
        raise ValueError("cannot summarize an empty burst-width group")
    return BurstWidthSummary(
        system=system_name,
        W=window.width,
        L=window.L,
        R=window.R,
        burst_width=burst_width,
        trials=len(trials),
        mean_active_count=mean(t.active_count for t in trials),
        structural_detectability_rate=_mean_bool(t.detectable for t in trials),
        structural_correctness_rate=_mean_bool(t.structural_correct for t in trials),
        structural_exactness_rate=_mean_bool(t.structural_exact for t in trials),
        semantic_survival_rate=_mean_bool(t.semantic_survival for t in trials),
        normalization_success_rate=_mean_bool(t.normalization_success for t in trials),
        overflow_rate=_mean_bool(t.overflow for t in trials),
        truncation_rate=_mean_bool(t.truncation for t in trials),
        mean_abs_error=mean(t.abs_error for t in trials),
        mean_normalization_steps=mean(t.normalization_steps for t in trials),
        mean_propagation_depth=mean(t.propagation_depth for t in trials),
    )


def run_burst_corruption_benchmark_for_system(
    system: RepresentationSystem,
    window: DigitWindow,
    trials: int,
    rng: np.random.Generator,
    burst_widths: tuple[int, ...] = (2, 3, 4, 5),
    max_steps_factor: int = 10,
) -> tuple[BurstBenchmarkSummary, list[BurstWidthSummary]]:
    """Run Experiment 4 for one system and one finite window.

    Each trial samples one admissible stored word, injects a local burst fault,
    attempts bounded-window repair, and records the same corruption-oriented
    metrics used in the single-digit benchmark.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    if max_steps_factor <= 0:
        raise ValueError("max_steps_factor must be positive")
    if not burst_widths:
        raise ValueError("burst_widths must be non-empty")
    if any(b <= 0 for b in burst_widths):
        raise ValueError("burst widths must be positive")
    if any(b > window.width for b in burst_widths):
        raise ValueError("burst width cannot exceed the finite-window width")

    max_steps = max_steps_factor * window.width
    trial_results: list[BurstTrialResult] = []
    by_width: dict[int, list[BurstTrialResult]] = defaultdict(list)

    for _ in range(trials):
        result = evaluate_burst_trial(
            system=system,
            window=window,
            burst_widths=burst_widths,
            rng=rng,
            max_steps=max_steps,
        )
        trial_results.append(result)
        by_width[result.burst_width].append(result)

    steps = [t.normalization_steps for t in trial_results]
    depths = [t.propagation_depth for t in trial_results]
    errors = [t.abs_error for t in trial_results]
    log_errors = [t.log_error_magnitude for t in trial_results]

    summary = BurstBenchmarkSummary(
        system=system.name,
        W=window.width,
        L=window.L,
        R=window.R,
        trials=trials,
        burst_widths=",".join(str(b) for b in burst_widths),
        mean_burst_width=mean(t.burst_width for t in trial_results),
        mean_active_count=mean(t.active_count for t in trial_results),
        structural_detectability_rate=_mean_bool(t.detectable for t in trial_results),
        structural_correctness_rate=_mean_bool(t.structural_correct for t in trial_results),
        structural_exactness_rate=_mean_bool(t.structural_exact for t in trial_results),
        semantic_survival_rate=_mean_bool(t.semantic_survival for t in trial_results),
        normalization_success_rate=_mean_bool(t.normalization_success for t in trial_results),
        overflow_rate=_mean_bool(t.overflow for t in trial_results),
        truncation_rate=_mean_bool(t.truncation for t in trial_results),
        mean_abs_error=mean(errors),
        worst_abs_error=max(errors),
        mean_log_error_magnitude=mean(log_errors),
        mean_normalization_steps=mean(steps),
        max_normalization_steps=max(steps),
        p50_normalization_steps=_quantile(steps, 0.50),
        p90_normalization_steps=_quantile(steps, 0.90),
        p99_normalization_steps=_quantile(steps, 0.99),
        mean_propagation_depth=mean(depths),
        max_propagation_depth=max(depths),
        p50_propagation_depth=_quantile(depths, 0.50),
        p90_propagation_depth=_quantile(depths, 0.90),
        p99_propagation_depth=_quantile(depths, 0.99),
    )

    width_summaries = [
        _summarize_burst_width(system.name, window, b, grouped)
        for b, grouped in sorted(by_width.items())
    ]
    return summary, width_summaries
