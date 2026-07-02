from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
from math import sqrt
from typing import Iterable


@dataclass(frozen=True)
class AlgebraicValue:
    """Exact element of Q(beta) for multinacci bases.

    For an m-step multinacci base beta_m satisfying
        beta^m = beta^{m-1} + ... + beta + 1,
    each value is represented as
        c_0 + c_1 beta + ... + c_{m-1} beta^{m-1},
    with rational coefficients.

    This lets the experiment test value preservation exactly, without relying
    on floating-point equality.
    """

    coeffs: tuple[Fraction, ...]

    @classmethod
    def zero(cls, degree: int) -> "AlgebraicValue":
        return cls(tuple(Fraction(0) for _ in range(degree)))

    @classmethod
    def one(cls, degree: int) -> "AlgebraicValue":
        return cls((Fraction(1),) + tuple(Fraction(0) for _ in range(degree - 1)))

    @property
    def degree(self) -> int:
        return len(self.coeffs)

    def __add__(self, other: "AlgebraicValue") -> "AlgebraicValue":
        self._check_degree(other)
        return AlgebraicValue(tuple(a + b for a, b in zip(self.coeffs, other.coeffs)))

    def __sub__(self, other: "AlgebraicValue") -> "AlgebraicValue":
        self._check_degree(other)
        return AlgebraicValue(tuple(a - b for a, b in zip(self.coeffs, other.coeffs)))

    def __mul__(self, scalar: int | Fraction) -> "AlgebraicValue":
        q = Fraction(scalar)
        return AlgebraicValue(tuple(q * c for c in self.coeffs))

    __rmul__ = __mul__

    def _check_degree(self, other: "AlgebraicValue") -> None:
        if self.degree != other.degree:
            raise ValueError("Cannot combine algebraic values from different fields.")

    def key(self) -> tuple[tuple[int, int], ...]:
        """Hashable exact key: ((num, den), ...)."""
        return tuple((c.numerator, c.denominator) for c in self.coeffs)

    def to_float(self) -> float:
        beta = multinacci_root_float(self.degree)
        return sum(float(c) * (beta**i) for i, c in enumerate(self.coeffs))


@lru_cache(maxsize=None)
def multinacci_root_float(degree: int) -> float:
    """Numerically approximate the real multinacci root.

    Used only for human-readable diagnostics. Exact equality in experiments
    is based on AlgebraicValue keys, not on this float.
    """
    if degree == 2:
        return (1.0 + sqrt(5.0)) / 2.0

    # Simple bisection on f(x)=x^m - sum_{j=0}^{m-1} x^j over (1, 2).
    lo, hi = 1.0, 2.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        f_mid = mid**degree - sum(mid**j for j in range(degree))
        if f_mid >= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def mul_beta(value: AlgebraicValue) -> AlgebraicValue: # ???67?????
    """Multiply by beta and reduce using beta^m = beta^{m-1}+...+1."""
    c = value.coeffs
    m = len(c)
    highest = c[-1]
    out = [Fraction(0) for _ in range(m)]
    out[0] = highest
    for j in range(1, m):
        out[j] = c[j - 1] + highest
    return AlgebraicValue(tuple(out))


def mul_beta_inverse(value: AlgebraicValue) -> AlgebraicValue: # ????67????
    """Multiply by beta^{-1} exactly.

    This inverts mul_beta. If x = beta*y, then:
        x_0 = y_{m-1}
        x_j = y_{j-1} + y_{m-1}, j>=1
    hence y_{j-1}=x_j-x_0 and y_{m-1}=x_0.
    """
    x = value.coeffs
    m = len(x)
    out = [Fraction(0) for _ in range(m)]
    for j in range(m - 1):
        out[j] = x[j + 1] - x[0]
    out[m - 1] = x[0]
    return AlgebraicValue(tuple(out))


@lru_cache(maxsize=None)
def beta_power(degree: int, exponent: int) -> AlgebraicValue: # ????67????
    """Return beta^exponent in Q(beta), exactly."""
    value = AlgebraicValue.one(degree)
    if exponent >= 0:
        for _ in range(exponent):
            value = mul_beta(value)
    else:
        for _ in range(-exponent):
            value = mul_beta_inverse(value)
    return value


def multinacci_value_key(degree: int, digits: Iterable[int], positions: Iterable[int]) -> tuple[tuple[int, int], ...]:
    """Exact value key for a finite digit string in a multinacci base."""
    total = AlgebraicValue.zero(degree)
    for d, pos in zip(digits, positions):
        if d:
            total = total + beta_power(degree, pos) * int(d)
    return total.key()
