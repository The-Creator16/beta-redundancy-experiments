from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from fractions import Fraction
from math import log
from statistics import mean
from typing import Iterable

import numpy as np

from .algebraic import multinacci_root_float
from .arithmetic import canonicalize_multinacci_addition, canonicalize_signed2_naf
from .normalization import NormalizationResult, identity_normalization
from .systems import RepresentationSystem, ValueKey
from .window import DigitWindow


@dataclass(frozen=True)
class SingleDigitTrialResult:
    """One Monte Carlo trial for Experiment 3."""

    position_index: int
    position_value: int
    original_digit: int
    corrupted_digit: int
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
class CorruptionBenchmarkSummary:
    """Aggregate metrics for Experiment 3: single-digit corruption."""

    system: str
    W: int
    L: int
    R: int
    trials: int
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
class PositionCorruptionSummary:
    """Per-position diagnostics for Experiment 3."""

    system: str
    W: int
    L: int
    R: int
    position_index: int
    position_value: int
    trials: int
    structural_detectability_rate: float
    structural_correctness_rate: float
    structural_exactness_rate: float
    semantic_survival_rate: float
    normalization_success_rate: float
    mean_abs_error: float
    mean_normalization_steps: float
    mean_propagation_depth: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def storage_alphabet(system: RepresentationSystem) -> tuple[int, ...]:
    """Return the storage alphabet used by the single-digit corruption model."""
    if system.name == "signed2_naf":
        return (-1, 0, 1)
    return (0, 1)


def corrupt_single_digit(
    system: RepresentationSystem,
    digits: tuple[int, ...],
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], int, int, int]:
    """Apply one alphabet-aware single-digit corruption.

    A uniformly random cell is selected.  Its replacement is sampled uniformly
    from the storage alphabet excluding the original digit.  For binary and
    multinacci systems this is exactly a bit flip; for Signed NAF it changes the
    selected digit to one of the two other symbols in {-1,0,1}.
    """
    if not digits:
        raise ValueError("digits must be non-empty")

    idx = int(rng.integers(0, len(digits), endpoint=False))
    original = int(digits[idx])
    candidates = [d for d in storage_alphabet(system) if d != original]
    if not candidates:
        raise ValueError("single-digit corruption has no valid replacement")

    replacement = int(rng.choice(candidates))
    out = list(digits)
    out[idx] = replacement
    return tuple(out), idx, original, replacement


def repair_single_digit_corruption(
    system: RepresentationSystem,
    corrupted: tuple[int, ...],
    window: DigitWindow,
    max_steps: int,
) -> NormalizationResult:
    """Attempt bounded-window repair after a single-digit corruption.

    For binary, every bit string is admissible, so repair is the identity.
    For Signed NAF, repair is deterministic bounded-window NAF recoding.
    For multinacci systems, repair uses the same deterministic local rewrite
    machinery as arithmetic canonicalization, but the input is interpreted as a
    corrupted stored word rather than a raw sum.
    """
    if system.name == "binary":
        return identity_normalization(corrupted)
    if system.name == "signed2_naf":
        return canonicalize_signed2_naf(corrupted, window, max_steps=max_steps)

    degree = getattr(system, "degree", None)
    if isinstance(degree, int):
        return canonicalize_multinacci_addition(corrupted, window, degree=degree, max_steps=max_steps)

    # Fallback for future systems if they define their own normalizer.
    return system.normalize(corrupted, window)


def _key_to_float_for_base2(key: tuple[int, int]) -> float:
    return float(Fraction(int(key[0]), int(key[1])))


def value_key_to_float(system: RepresentationSystem, key: ValueKey) -> float:
    """Convert an exact value key into a float for error magnitudes.

    Equality tests in the experiment use exact keys.  This conversion is used
    only for scalar error summaries such as MAE and worst-case absolute error.
    """
    if system.name in {"binary", "signed2_naf"}:
        return _key_to_float_for_base2(key)  # type: ignore[arg-type]

    degree = getattr(system, "degree", None)
    if not isinstance(degree, int):
        raise TypeError(f"Cannot convert value key for system {system.name!r}")

    beta = multinacci_root_float(degree)
    total = 0.0
    for i, pair in enumerate(key):  # type: ignore[union-attr]
        num, den = pair
        total += float(Fraction(int(num), int(den))) * (beta**i)
    return total


def system_base_float(system: RepresentationSystem) -> float:
    if system.name in {"binary", "signed2_naf"}:
        return 2.0
    degree = getattr(system, "degree", None)
    if isinstance(degree, int):
        return multinacci_root_float(degree)
    raise TypeError(f"Cannot determine base for system {system.name!r}")


def _quantile(values: list[int] | list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), q))


def evaluate_single_digit_trial(
    system: RepresentationSystem,
    window: DigitWindow,
    rng: np.random.Generator,
    max_steps: int,
) -> SingleDigitTrialResult:
    """Run one single-digit corruption and bounded-window repair trial."""
    original = system.sample_word(window, rng)
    corrupted, idx, old_digit, new_digit = corrupt_single_digit(system, original, rng)
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

    return SingleDigitTrialResult(
        position_index=idx,
        position_value=window.positions_high_to_low[idx],
        original_digit=old_digit,
        corrupted_digit=new_digit,
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


def _mean_bool(values: Iterable[bool]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(int(v) for v in values_list) / len(values_list)


def _summarize_position(
    system_name: str,
    window: DigitWindow,
    position_index: int,
    trials: list[SingleDigitTrialResult],
) -> PositionCorruptionSummary:
    if not trials:
        raise ValueError("cannot summarize an empty position group")
    return PositionCorruptionSummary(
        system=system_name,
        W=window.width,
        L=window.L,
        R=window.R,
        position_index=position_index,
        position_value=trials[0].position_value,
        trials=len(trials),
        structural_detectability_rate=_mean_bool(t.detectable for t in trials),
        structural_correctness_rate=_mean_bool(t.structural_correct for t in trials),
        structural_exactness_rate=_mean_bool(t.structural_exact for t in trials),
        semantic_survival_rate=_mean_bool(t.semantic_survival for t in trials),
        normalization_success_rate=_mean_bool(t.normalization_success for t in trials),
        mean_abs_error=mean(t.abs_error for t in trials),
        mean_normalization_steps=mean(t.normalization_steps for t in trials),
        mean_propagation_depth=mean(t.propagation_depth for t in trials),
    )


def run_single_digit_corruption_benchmark_for_system(
    system: RepresentationSystem,
    window: DigitWindow,
    trials: int,
    rng: np.random.Generator,
    max_steps_factor: int = 6,
) -> tuple[CorruptionBenchmarkSummary, list[PositionCorruptionSummary]]:
    """Run Experiment 3 for one system and one finite window.

    Each trial samples one admissible stored word, corrupts one uniformly random
    digit, attempts bounded-window repair, and records structural detectability,
    structural correctness, structural exactness, semantic survival, numerical
    error, and repair cost.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    if max_steps_factor <= 0:
        raise ValueError("max_steps_factor must be positive")

    max_steps = max_steps_factor * window.width
    trial_results: list[SingleDigitTrialResult] = []
    by_position: dict[int, list[SingleDigitTrialResult]] = defaultdict(list)

    for _ in range(trials):
        result = evaluate_single_digit_trial(system, window, rng, max_steps=max_steps)
        trial_results.append(result)
        by_position[result.position_index].append(result)

    steps = [t.normalization_steps for t in trial_results]
    depths = [t.propagation_depth for t in trial_results]
    errors = [t.abs_error for t in trial_results]
    log_errors = [t.log_error_magnitude for t in trial_results]

    summary = CorruptionBenchmarkSummary(
        system=system.name,
        W=window.width,
        L=window.L,
        R=window.R,
        trials=trials,
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

    position_summaries = [
        _summarize_position(system.name, window, idx, grouped)
        for idx, grouped in sorted(by_position.items())
    ]
    return summary, position_summaries
