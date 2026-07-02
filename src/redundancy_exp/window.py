from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DigitWindow:
    """Finite hardware window I_{L,R} = {-R, ..., L}.

    Digits are stored in high-to-low order, corresponding to positions
    L, L-1, ..., -R. The radix point is between positions 0 and -1.
    """

    L: int
    R: int

    @classmethod
    def balanced(cls, width: int) -> "DigitWindow":
        """Use L=floor(W/2), R=W-L-1 as in the experimental protocol."""
        if width <= 0:
            raise ValueError("Window width must be positive.")
        L = width // 2
        R = width - L - 1
        return cls(L=L, R=R)

    @property
    def width(self) -> int:
        return self.L + self.R + 1

    @property
    def positions_high_to_low(self) -> tuple[int, ...]:
        return tuple(range(self.L, -self.R - 1, -1))

    def index_of_position(self, position: int) -> int:
        if position < -self.R or position > self.L:
            raise IndexError(f"Position {position} is outside I_{{L,R}}.")
        return self.L - position
