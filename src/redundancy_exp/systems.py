from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import log2
from typing import Protocol

import numpy as np

from .algebraic import multinacci_value_key
from .normalization import NormalizationResult, identity_normalization, normalize_multinacci_forbidden_runs
from .sampling import NAFDP, RunLengthDP, random_rank
from .window import DigitWindow


ValueKey = tuple[tuple[int, int], ...] | tuple[int, int]


class RepresentationSystem(Protocol):
    name: str
    alphabet_size: int

    def count_codebook(self, width: int) -> int: ...
    def ambient_size(self, width: int) -> int: ...
    def sample_word(self, window: DigitWindow, rng: np.random.Generator) -> tuple[int, ...]: ...
    def is_admissible(self, digits: tuple[int, ...]) -> bool: ...
    def normalize(self, digits: tuple[int, ...], window: DigitWindow) -> NormalizationResult: ...
    def value_key(self, digits: tuple[int, ...], window: DigitWindow) -> ValueKey: ...

    def capacity_bits_per_digit(self, width: int) -> float:
        return log2(self.count_codebook(width)) / width

    def structural_redundancy_per_digit(self, width: int) -> float:
        return log2(self.ambient_size(width)) / width - self.capacity_bits_per_digit(width)


@dataclass(frozen=True)
class BinarySystem:
    name: str = "binary"
    alphabet_size: int = 2

    def count_codebook(self, width: int) -> int:
        return 2**width

    def ambient_size(self, width: int) -> int:
        return 2**width

    def sample_word(self, window: DigitWindow, rng: np.random.Generator) -> tuple[int, ...]:
        return tuple(int(x) for x in rng.integers(0, 2, size=window.width))

    def is_admissible(self, digits: tuple[int, ...]) -> bool:
        return all(d in (0, 1) for d in digits)

    def normalize(self, digits: tuple[int, ...], window: DigitWindow) -> NormalizationResult:
        return identity_normalization(tuple(digits))

    def value_key(self, digits: tuple[int, ...], window: DigitWindow) -> tuple[int, int]:
        value = Fraction(0)
        for d, pos in zip(digits, window.positions_high_to_low):
            if pos >= 0:
                value += int(d) * (2**pos)
            else:
                value += Fraction(int(d), 2 ** (-pos))
        return (value.numerator, value.denominator)


@dataclass(frozen=True)
class SignedRadix2NAFSystem:
    name: str = "signed2_naf"
    alphabet_size: int = 3

    def count_codebook(self, width: int) -> int:
        return NAFDP.build(width).count

    def ambient_size(self, width: int) -> int:
        return 3**width

    def sample_word(self, window: DigitWindow, rng: np.random.Generator) -> tuple[int, ...]:
        dp = NAFDP.build(window.width)
        return dp.unrank(random_rank(dp.count, rng))

    def is_admissible(self, digits: tuple[int, ...]) -> bool:
        if not all(d in (-1, 0, 1) for d in digits):
            return False
        return all(not (digits[i] != 0 and digits[i + 1] != 0) for i in range(len(digits) - 1))

    def normalize(self, digits: tuple[int, ...], window: DigitWindow) -> NormalizationResult:
        # Clean Experiment 1 samples only canonical NAF words; full carry-limited
        # NAF canonicalization belongs to the arithmetic experiment.
        if not self.is_admissible(tuple(digits)):
            return NormalizationResult(digits=tuple(digits), steps=0, propagation_depth=0, success=False)
        return identity_normalization(tuple(digits))

    def value_key(self, digits: tuple[int, ...], window: DigitWindow) -> tuple[int, int]:
        value = Fraction(0)
        for d, pos in zip(digits, window.positions_high_to_low):
            if pos >= 0:
                value += int(d) * (2**pos)
            else:
                value += Fraction(int(d), 2 ** (-pos))
        return (value.numerator, value.denominator)


@dataclass(frozen=True)
class MultinacciSystem:
    """Canonical binary finite-window system for an m-step multinacci base."""

    degree: int
    name: str
    alphabet_size: int = 2

    @property
    def forbidden_run_length(self) -> int:
        return self.degree

    @property
    def max_allowed_ones_run(self) -> int:
        return self.degree - 1

    def count_codebook(self, width: int) -> int:
        return RunLengthDP.build(width, self.max_allowed_ones_run).count

    def ambient_size(self, width: int) -> int:
        return 2**width

    def sample_word(self, window: DigitWindow, rng: np.random.Generator) -> tuple[int, ...]:
        dp = RunLengthDP.build(window.width, self.max_allowed_ones_run)
        return dp.unrank(random_rank(dp.count, rng))

    def is_admissible(self, digits: tuple[int, ...]) -> bool:
        if not all(d in (0, 1) for d in digits):
            return False
        k = self.forbidden_run_length
        return all(not all(digits[i + j] == 1 for j in range(k)) for i in range(len(digits) - k + 1))

    def normalize(self, digits: tuple[int, ...], window: DigitWindow) -> NormalizationResult:
        if self.is_admissible(tuple(digits)):
            return identity_normalization(tuple(digits))
        return normalize_multinacci_forbidden_runs(tuple(digits), degree=self.degree, max_steps=3 * window.width)

    def value_key(self, digits: tuple[int, ...], window: DigitWindow) -> tuple[tuple[int, int], ...]:
        return multinacci_value_key(self.degree, digits, window.positions_high_to_low)


def make_default_systems() -> dict[str, RepresentationSystem]:
    return {
        "binary": BinarySystem(),
        "signed2_naf": SignedRadix2NAFSystem(),
        "phi": MultinacciSystem(degree=2, name="phi"),
        "tribonacci": MultinacciSystem(degree=3, name="tribonacci"),
        "tetranacci": MultinacciSystem(degree=4, name="tetranacci"),
    }
