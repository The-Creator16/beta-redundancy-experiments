from __future__ import annotations

from dataclasses import dataclass, asdict
from math import log2
from statistics import mean, pstdev

import numpy as np

from .systems import RepresentationSystem
from .window import DigitWindow


def nonzero_density(digits: tuple[int, ...]) -> float:
    return sum(1 for d in digits if d != 0) / len(digits)


def active_span(digits: tuple[int, ...]) -> int:
    """Distance between first and last nonzero stored digit, in digit cells."""
    indices = [i for i, d in enumerate(digits) if d != 0]
    if not indices:
        return 0
    return indices[-1] - indices[0] + 1


@dataclass(frozen=True)
class CleanBenchmarkSummary:
    system: str
    W: int
    L: int
    R: int
    trials: int
    codebook_size: int
    ambient_size: int
    codebook_sparsity: float
    information_bits: float
    capacity_bits_per_digit: float
    structural_redundancy_per_digit: float
    overhead_digits_per_information_bit: float
    mean_nonzero_density: float
    std_nonzero_density: float
    mean_active_span: float
    round_trip_exactness_rate: float
    admissible_rate: float
    mean_clean_normalization_steps: float
    max_clean_normalization_steps: int
    mean_clean_propagation_depth: float
    max_clean_propagation_depth: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_clean_benchmark_for_system(
    system: RepresentationSystem,
    window: DigitWindow,
    trials: int,
    rng: np.random.Generator,
) -> CleanBenchmarkSummary:
    """Run Experiment 1 for one representation system and one window width."""
    densities: list[float] = []
    spans: list[int] = []
    roundtrip_ok = 0
    admissible_ok = 0
    norm_steps: list[int] = []
    prop_depths: list[int] = []

    for _ in range(trials):
        word = system.sample_word(window, rng)
        admissible = system.is_admissible(word)
        admissible_ok += int(admissible)

        norm = system.normalize(word, window)

        # In the clean benchmark, sampled words are canonical codebook elements.
        # Therefore a correct clean round-trip must leave the stored word unchanged.
        # Exact algebraic decoders are implemented in systems.py for later
        # experiments, but we avoid calling them in every clean trial because
        # identity of the digit vector already implies exact value preservation.
        roundtrip_ok += int(norm.success and norm.digits == word and system.is_admissible(norm.digits))

        densities.append(nonzero_density(word))
        spans.append(active_span(word))
        norm_steps.append(norm.steps)
        prop_depths.append(norm.propagation_depth)

    codebook_size = system.count_codebook(window.width)
    ambient_size = system.ambient_size(window.width)
    information_bits = log2(codebook_size)
    capacity = information_bits / window.width
    redundancy = log2(ambient_size) / window.width - capacity

    return CleanBenchmarkSummary(
        system=system.name,
        W=window.width,
        L=window.L,
        R=window.R,
        trials=trials,
        codebook_size=codebook_size,
        ambient_size=ambient_size,
        codebook_sparsity=codebook_size / ambient_size,
        information_bits=information_bits,
        capacity_bits_per_digit=capacity,
        structural_redundancy_per_digit=redundancy,
        overhead_digits_per_information_bit=window.width / information_bits,
        mean_nonzero_density=mean(densities),
        std_nonzero_density=pstdev(densities) if len(densities) > 1 else 0.0,
        mean_active_span=mean(spans),
        round_trip_exactness_rate=roundtrip_ok / trials,
        admissible_rate=admissible_ok / trials,
        mean_clean_normalization_steps=mean(norm_steps),
        max_clean_normalization_steps=max(norm_steps),
        mean_clean_propagation_depth=mean(prop_depths),
        max_clean_propagation_depth=max(prop_depths),
    )
