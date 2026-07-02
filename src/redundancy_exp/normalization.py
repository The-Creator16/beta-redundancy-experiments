from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizationResult:
    digits: tuple[int, ...]
    steps: int
    propagation_depth: int
    overflow: bool = False
    truncation: bool = False
    success: bool = True


def identity_normalization(digits: tuple[int, ...]) -> NormalizationResult:
    return NormalizationResult(digits=tuple(digits), steps=0, propagation_depth=0)


def normalize_multinacci_forbidden_runs(
    digits: tuple[int, ...],
    degree: int,
    max_steps: int | None = None,
) -> NormalizationResult:
    """Bounded local re-admissibilization for multinacci-like systems.

    It applies the directed rule
        0 1...1 -> 1 0...0
    where the number of ones is `degree`.

    This function is intentionally conservative: it is sufficient for clean
    Experiment 1 and for simple future forbidden-run repair. Full arithmetic
    canonicalization for temporary digit 2 should be implemented as a separate
    experiment-specific rule set.
    """
    if degree < 2:
        raise ValueError("degree must be at least 2")
    out = list(digits)
    n = len(out)
    max_steps = max_steps if max_steps is not None else 3 * max(1, n)
    steps = 0
    first_touched = None
    last_touched = None

    while steps < max_steps:
        changed = False
        # Find a forbidden run of `degree` ones.
        for start in range(0, n - degree + 1):
            if all(out[start + j] == 1 for j in range(degree)):
                carry_index = start - 1
                if carry_index < 0:
                    return NormalizationResult(
                        digits=tuple(out),
                        steps=steps,
                        propagation_depth=0 if first_touched is None else last_touched - first_touched + 1,
                        overflow=True,
                        success=False,
                    )
                if out[carry_index] != 0:
                    # The simple rule cannot resolve a nonzero carry target.
                    return NormalizationResult(
                        digits=tuple(out),
                        steps=steps,
                        propagation_depth=0 if first_touched is None else last_touched - first_touched + 1,
                        success=False,
                    )
                out[carry_index] = 1
                for j in range(degree):
                    out[start + j] = 0
                touched_left = carry_index
                touched_right = start + degree - 1
                first_touched = touched_left if first_touched is None else min(first_touched, touched_left)
                last_touched = touched_right if last_touched is None else max(last_touched, touched_right)
                steps += 1
                changed = True
                break
        if not changed:
            depth = 0 if first_touched is None else last_touched - first_touched + 1
            return NormalizationResult(digits=tuple(out), steps=steps, propagation_depth=depth)

    depth = 0 if first_touched is None else last_touched - first_touched + 1
    return NormalizationResult(digits=tuple(out), steps=steps, propagation_depth=depth, success=False)
