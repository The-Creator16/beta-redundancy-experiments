from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from statistics import mean

import numpy as np

from .normalization import NormalizationResult
from .systems import RepresentationSystem
from .window import DigitWindow


def digitwise_add(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """Return the raw digit-wise sum of two words of equal length.

    This is the raw arithmetic state of Experiment 2.  For binary and
    multinacci systems it may contain the temporary digit 2.  For signed NAF it
    may contain digits in {-2,-1,0,1,2}.
    """
    if len(a) != len(b):
        raise ValueError("Cannot add words of different lengths.")
    return tuple(int(x) + int(y) for x, y in zip(a, b))


def _span_depth(touched: set[int]) -> int:
    """Return the width of the touched interval inside the stored word."""
    if not touched:
        return 0
    return max(touched) - min(touched) + 1


def canonicalize_binary_addition(
    raw: tuple[int, ...],
    window: DigitWindow,
    max_steps: int | None = None,
) -> NormalizationResult:
    """Canonicalize a raw binary addition state inside a finite window.

    Digits are stored in high-to-low order.  The local carry rule is
        2 at position k -> 0 at k and +1 at position k+1.
    A carry beyond the most significant window position is recorded as overflow
    and discarded, matching the bounded-window convention.
    """
    out = [int(d) for d in raw]
    n = len(out)
    max_steps = max_steps if max_steps is not None else 3 * max(1, n)
    steps = 0
    overflow = False
    touched: set[int] = set()

    while steps < max_steps:
        changed = False
        # Carries propagate from least significant positions toward the MSB.
        for idx in range(n - 1, -1, -1):
            if out[idx] >= 2:
                out[idx] -= 2
                touched.add(idx)
                if idx == 0:
                    overflow = True
                else:
                    out[idx - 1] += 1
                    touched.add(idx - 1)
                steps += 1
                changed = True
                break
            if out[idx] < 0:
                # Negative digits are not expected in binary addition.
                return NormalizationResult(
                    digits=tuple(out),
                    steps=steps,
                    propagation_depth=_span_depth(touched),
                    overflow=overflow,
                    success=False,
                )
        if not changed:
            success = all(d in (0, 1) for d in out)
            return NormalizationResult(
                digits=tuple(out),
                steps=steps,
                propagation_depth=_span_depth(touched),
                overflow=overflow,
                success=success,
            )

    return NormalizationResult(
        digits=tuple(out),
        steps=steps,
        propagation_depth=_span_depth(touched),
        overflow=overflow,
        success=False,
    )


def _scaled_integer_base2(digits: tuple[int, ...], window: DigitWindow) -> int:
    """Return N = sum_i d_i 2^(i+R), an exact integer."""
    total = 0
    for d, pos in zip(digits, window.positions_high_to_low):
        total += int(d) * (1 << (pos + window.R))
    return total


def canonicalize_signed2_naf(
    raw: tuple[int, ...],
    window: DigitWindow,
    max_steps: int | None = None,
) -> NormalizationResult:
    """Canonicalize a signed radix-2 raw word by deterministic NAF recoding.

    The finite-window word is interpreted as the scaled integer
        N = sum_i z_i 2^(i+R).
    We then apply the standard Reitwiesner NAF recoding.  Emitted nonzero
    digits beyond the MSB of the window are recorded as overflow and discarded.

    The `steps` count is the number of Reitwiesner loop iterations.  This is a
    deterministic recoding cost for the signed-digit baseline, not a local
    beta-rewrite count.
    """
    n = len(raw)
    max_steps = max_steps if max_steps is not None else 3 * max(1, n)
    original = tuple(int(d) for d in raw)
    N = _scaled_integer_base2(original, window)
    out_low_to_high: list[int] = []
    steps = 0
    overflow = False

    # N may be negative because signed NAF operands can contain -1.  The same
    # odd/even recurrence works for Python integers with modulo arithmetic.
    while N != 0 and steps < max_steps:
        if N % 2:
            u = 2 - (N % 4)
            # For N mod 4 == 3, u=-1.  For N mod 4 == 1, u=1.
            if u == 2:
                u = -1
            N = (N - u) // 2
            out_low_to_high.append(int(u))
        else:
            N //= 2
            out_low_to_high.append(0)
        steps += 1

    if N != 0:
        return NormalizationResult(
            digits=original,
            steps=steps,
            propagation_depth=n,
            overflow=True,
            success=False,
        )

    # Map low-to-high emitted digits back to high-to-low window storage.
    out = [0 for _ in range(n)]
    for offset, digit in enumerate(out_low_to_high):
        pos = -window.R + offset
        if pos > window.L:
            if digit != 0:
                overflow = True
            continue
        idx = window.index_of_position(pos)
        out[idx] = digit

    changed = {i for i, (a, b) in enumerate(zip(original, out)) if a != b}
    if overflow:
        changed.add(0)
    return NormalizationResult(
        digits=tuple(out),
        steps=steps,
        propagation_depth=_span_depth(changed),
        overflow=overflow,
        success=True,
    )


def _find_first_multinacci_violation(out: list[int], degree: int) -> tuple[str, int] | None:
    """Find the most significant visible violation.

    Returns ("digit", idx) for a digit outside {0,1} with value >=2, or
    ("run", start) for a forbidden run of `degree` consecutive ones.
    The list is stored MSB-to-LSB, so a smaller index is more significant.
    """
    n = len(out)
    for idx in range(n):
        if out[idx] >= 2:
            return ("digit", idx)
        if out[idx] < 0:
            return ("bad_negative", idx)
        if idx <= n - degree and all(out[idx + j] == 1 for j in range(degree)):
            return ("run", idx)
    return None


def canonicalize_multinacci_addition(
    raw: tuple[int, ...],
    window: DigitWindow,
    degree: int,
    max_steps: int | None = None,
) -> NormalizationResult:
    """Canonicalize a raw addition state in an m-step multinacci system.

    The rule for a temporary digit 2 follows from
        2 = beta + beta^{-m},
    hence
        2 beta^k = beta^{k+1} + beta^{k-m}.

    Therefore a local digit-2 defect at position k is rewritten by subtracting
    2 at k and adding 1 at k+1 and k-m.  Boundary writes are recorded as
    overflow/truncation and discarded.  Forbidden runs are resolved by the
    directed multinacci carry rule
        0 1...1 -> 1 0...0,
    with `degree` ones.
    """
    if degree < 2:
        raise ValueError("degree must be at least 2")
    out = [int(d) for d in raw]
    n = len(out)
    max_steps = max_steps if max_steps is not None else 3 * max(1, n)
    steps = 0
    overflow = False
    truncation = False
    touched: set[int] = set()

    while steps < max_steps:
        violation = _find_first_multinacci_violation(out, degree)
        if violation is None:
            success = all(d in (0, 1) for d in out)
            # Final admissibility check: no degree-long run of ones.
            success = success and all(
                not all(out[i + j] == 1 for j in range(degree))
                for i in range(n - degree + 1)
            )
            return NormalizationResult(
                digits=tuple(out),
                steps=steps,
                propagation_depth=_span_depth(touched),
                overflow=overflow,
                truncation=truncation,
                success=success,
            )

        kind, idx = violation
        if kind == "bad_negative":
            return NormalizationResult(
                digits=tuple(out),
                steps=steps,
                propagation_depth=_span_depth(touched),
                overflow=overflow,
                truncation=truncation,
                success=False,
            )

        if kind == "digit":
            # Resolve one temporary 2 at this position.
            out[idx] -= 2
            touched.add(idx)

            hi = idx - 1          # position k+1
            lo = idx + degree     # position k-degree
            if hi < 0:
                overflow = True
            else:
                out[hi] += 1
                touched.add(hi)
            if lo >= n:
                truncation = True
            else:
                out[lo] += 1
                touched.add(lo)
            steps += 1
            continue

        if kind == "run":
            start = idx
            carry_idx = start - 1
            if carry_idx < 0:
                overflow = True
            else:
                out[carry_idx] += 1
                touched.add(carry_idx)
            for j in range(degree):
                out[start + j] = 0
                touched.add(start + j)
            steps += 1
            continue

        raise RuntimeError(f"unknown violation kind: {kind}")

    return NormalizationResult(
        digits=tuple(out),
        steps=steps,
        propagation_depth=_span_depth(touched),
        overflow=overflow,
        truncation=truncation,
        success=False,
    )


def canonicalize_raw_sum(
    system: RepresentationSystem,
    raw: tuple[int, ...],
    window: DigitWindow,
    max_steps: int | None = None,
) -> NormalizationResult:
    """Dispatch arithmetic canonicalization for one representation system."""
    if system.name == "binary":
        return canonicalize_binary_addition(raw, window, max_steps=max_steps)
    if system.name == "signed2_naf":
        return canonicalize_signed2_naf(raw, window, max_steps=max_steps)
    degree = getattr(system, "degree", None)
    if isinstance(degree, int):
        return canonicalize_multinacci_addition(raw, window, degree=degree, max_steps=max_steps)
    raise TypeError(f"No arithmetic canonicalizer registered for system {system.name!r}")


@dataclass(frozen=True)
class ArithmeticBenchmarkSummary:
    system: str
    W: int
    L: int
    R: int
    trials: int
    arithmetic_exactness_rate: float
    canonicalization_success_rate: float
    output_admissible_rate: float
    raw_admissible_rate: float
    raw_defect_rate: float
    overflow_rate: float
    truncation_rate: float
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
    mean_raw_noncanonical_digit_density: float
    mean_output_nonzero_density: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _quantile(values: list[int] | list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), q))


def raw_noncanonical_digit_density(system: RepresentationSystem, raw: tuple[int, ...]) -> float:
    """Fraction of cells that are outside the storage alphabet of the system."""
    if system.name == "signed2_naf":
        allowed = {-1, 0, 1}
    else:
        allowed = {0, 1}
    return sum(1 for d in raw if d not in allowed) / len(raw)


def nonzero_density(digits: tuple[int, ...]) -> float:
    return sum(1 for d in digits if d != 0) / len(digits)


def run_arithmetic_benchmark_for_system(
    system: RepresentationSystem,
    window: DigitWindow,
    trials: int,
    rng: np.random.Generator,
    max_steps_factor: int = 3,
) -> tuple[ArithmeticBenchmarkSummary, Counter[int]]:
    """Run Experiment 2 for one system and one window width.

    Each trial samples two independent admissible words, forms the raw digit-wise
    sum, canonicalizes it under the bounded-window procedure, and checks exact
    arithmetic preservation using the exact value map of the system.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    if max_steps_factor <= 0:
        raise ValueError("max_steps_factor must be positive")

    arithmetic_exact = 0
    norm_success = 0
    output_admissible = 0
    raw_admissible = 0
    overflow_count = 0
    truncation_count = 0
    steps_values: list[int] = []
    depths: list[int] = []
    raw_bad_density: list[float] = []
    output_densities: list[float] = []
    step_hist: Counter[int] = Counter()

    max_steps = max_steps_factor * window.width

    for _ in range(trials):
        a = system.sample_word(window, rng)
        b = system.sample_word(window, rng)
        raw = digitwise_add(a, b)
        raw_adm = system.is_admissible(raw)
        raw_admissible += int(raw_adm)

        result = canonicalize_raw_sum(system, raw, window, max_steps=max_steps)
        out = result.digits
        out_adm = system.is_admissible(out)
        output_admissible += int(out_adm)
        norm_success += int(result.success and out_adm)

        # Arithmetic exactness is stricter than value equality alone: the output
        # must be a successful admissible canonical word and must preserve the
        # value of the raw digit-wise sum.
        raw_key = system.value_key(raw, window)
        out_key = system.value_key(out, window)
        arithmetic_exact += int(result.success and out_adm and out_key == raw_key)

        overflow_count += int(result.overflow)
        truncation_count += int(result.truncation)
        steps_values.append(result.steps)
        depths.append(result.propagation_depth)
        raw_bad_density.append(raw_noncanonical_digit_density(system, raw))
        output_densities.append(nonzero_density(out))
        step_hist[result.steps] += 1

    summary = ArithmeticBenchmarkSummary(
        system=system.name,
        W=window.width,
        L=window.L,
        R=window.R,
        trials=trials,
        arithmetic_exactness_rate=arithmetic_exact / trials,
        canonicalization_success_rate=norm_success / trials,
        output_admissible_rate=output_admissible / trials,
        raw_admissible_rate=raw_admissible / trials,
        raw_defect_rate=1.0 - raw_admissible / trials,
        overflow_rate=overflow_count / trials,
        truncation_rate=truncation_count / trials,
        mean_normalization_steps=mean(steps_values),
        max_normalization_steps=max(steps_values),
        p50_normalization_steps=_quantile(steps_values, 0.50),
        p90_normalization_steps=_quantile(steps_values, 0.90),
        p99_normalization_steps=_quantile(steps_values, 0.99),
        mean_propagation_depth=mean(depths),
        max_propagation_depth=max(depths),
        p50_propagation_depth=_quantile(depths, 0.50),
        p90_propagation_depth=_quantile(depths, 0.90),
        p99_propagation_depth=_quantile(depths, 0.99),
        mean_raw_noncanonical_digit_density=mean(raw_bad_density),
        mean_output_nonzero_density=mean(output_densities),
    )
    return summary, step_hist
