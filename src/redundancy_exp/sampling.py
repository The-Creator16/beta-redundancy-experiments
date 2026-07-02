from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RunLengthDP:
    """DP table for binary strings with no run longer than max_run."""

    length: int
    max_run: int
    table: tuple[tuple[int, ...], ...]

    @classmethod
    def build(cls, length: int, max_run: int) -> "RunLengthDP":
        if length < 0:
            raise ValueError("length must be non-negative")
        if max_run < 0:
            raise ValueError("max_run must be non-negative")

        # table[remaining][current_run_of_ones]
        rows: list[list[int]] = [[0] * (max_run + 1) for _ in range(length + 1)]
        for run in range(max_run + 1):
            rows[0][run] = 1
        for remaining in range(1, length + 1):
            for run in range(max_run + 1):
                count = rows[remaining - 1][0]  # choose 0
                if run < max_run:
                    count += rows[remaining - 1][run + 1]  # choose 1
                rows[remaining][run] = count
        return cls(length=length, max_run=max_run, table=tuple(tuple(r) for r in rows))

    @property
    def count(self) -> int:
        return self.table[self.length][0]

    def unrank(self, rank: int) -> tuple[int, ...]:
        """Return the rank-th valid word in deterministic lexicographic order."""
        if rank < 0 or rank >= self.count:
            raise ValueError("rank outside codebook")
        out: list[int] = []
        run = 0
        for idx in range(self.length):
            remaining_after = self.length - idx - 1
            for digit in (0, 1):
                if digit == 1 and run >= self.max_run:
                    continue
                next_run = run + 1 if digit == 1 else 0
                count = self.table[remaining_after][next_run]
                if rank < count:
                    out.append(digit)
                    run = next_run
                    break
                rank -= count
            else:
                raise RuntimeError("unranking failed")
        return tuple(out)


@dataclass(frozen=True)
class NAFDP:
    """DP table for signed binary NAF words: no adjacent nonzero digits."""

    length: int
    table: tuple[tuple[int, int], ...]

    @classmethod
    def build(cls, length: int) -> "NAFDP":
        # table[remaining][prev_nonzero]
        rows = [[0, 0] for _ in range(length + 1)]
        rows[0][0] = rows[0][1] = 1
        for remaining in range(1, length + 1):
            # If previous digit is zero, we may choose -1, 0, or 1.
            rows[remaining][0] = rows[remaining - 1][0] + 2 * rows[remaining - 1][1]
            # If previous digit is nonzero, we must choose 0.
            rows[remaining][1] = rows[remaining - 1][0]
        return cls(length=length, table=tuple((a, b) for a, b in rows))

    @property
    def count(self) -> int:
        return self.table[self.length][0]
    def unrank(self, rank: int) -> tuple[int, ...]:
        if rank < 0 or rank >= self.count:
            raise ValueError("rank outside NAF codebook")
        out: list[int] = []
        prev_nonzero = 0
        for idx in range(self.length):
            remaining_after = self.length - idx - 1
            allowed = (0,) if prev_nonzero else (-1, 0, 1)
            for digit in allowed:
                next_prev = 1 if digit != 0 else 0
                count = self.table[remaining_after][next_prev]
                if rank < count:
                    out.append(digit)
                    prev_nonzero = next_prev
                    break
                rank -= count
            else:
                raise RuntimeError("NAF unranking failed")
        return tuple(out)



def random_rank(count: int, rng: np.random.Generator) -> int:
    """Draw a uniform rank from [0, count-1].

    The current experiments use W<=24, so the count fits in int64. The function
    is separated so it can be replaced later for very large codebooks.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    return int(rng.integers(0, count, endpoint=False))
