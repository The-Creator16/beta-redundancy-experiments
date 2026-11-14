from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# Allow running this script directly from the repository root without installing.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redundancy_exp import DigitWindow, make_default_systems
from redundancy_exp.sampling import NAFDP, RunLengthDP


SYSTEM_ORDER = ["binary", "signed2_naf", "phi", "tribonacci", "tetranacci"]


@dataclass
class MetricCounts:
    total_corruptions: int = 0
    structural_detectability_count: int = 0
    normalization_success_count: int = 0
    structural_correctness_count: int = 0
    structural_exactness_count: int = 0
    semantic_survival_count: int = 0
    active_digit_total: int = 0

    def update(
        self,
        *,
        system,
        window: DigitWindow,
        original_word: tuple[int, ...],
        corrupted_word: tuple[int, ...],
        active_count: int,
    ) -> None:
        """Evaluate one corrupted word against the original admissible word."""
        self.total_corruptions += 1
        self.active_digit_total += active_count

        original_value = system.value_key(original_word, window)
        corrupted_value = system.value_key(corrupted_word, window)

        if not system.is_admissible(corrupted_word):
            self.structural_detectability_count += 1

        norm_result = system.normalize(corrupted_word, window)
        if not getattr(norm_result, "success", False):
            return

        self.normalization_success_count += 1
        repaired_word = tuple(norm_result.digits)

        if system.is_admissible(repaired_word):
            self.structural_correctness_count += 1

        repaired_value = system.value_key(repaired_word, window)

        if repaired_value == corrupted_value:
            self.structural_exactness_count += 1

        if repaired_value == original_value:
            self.semantic_survival_count += 1

    def rates(self, prefix: str) -> dict[str, Any]:
        n = self.total_corruptions
        denom = n if n else 1
        return {
            f"{prefix}_total_corruptions": n,
            f"{prefix}_structural_detectability_count": self.structural_detectability_count,
            f"{prefix}_structural_detectability_rate": self.structural_detectability_count / denom,
            f"{prefix}_normalization_success_count": self.normalization_success_count,
            f"{prefix}_normalization_success_rate": self.normalization_success_count / denom,
            f"{prefix}_structural_correctness_count": self.structural_correctness_count,
            f"{prefix}_structural_correctness_rate": self.structural_correctness_count / denom,
            f"{prefix}_structural_exactness_count": self.structural_exactness_count,
            f"{prefix}_structural_exactness_rate": self.structural_exactness_count / denom,
            f"{prefix}_semantic_survival_count": self.semantic_survival_count,
            f"{prefix}_semantic_survival_rate": self.semantic_survival_count / denom,
            f"{prefix}_mean_active_count": self.active_digit_total / denom,
        }


def digit_alphabet(system) -> tuple[int, ...]:
    """Return the storage alphabet used by a representation system."""
    if system.name == "signed2_naf":
        return (-1, 0, 1)
    return tuple(range(system.alphabet_size))


def build_unranker(system, W: int):
    """Build the DP object needed for exhaustive unranking, if any."""
    if system.name == "binary":
        return None
    if system.name == "signed2_naf":
        return NAFDP.build(W)
    return RunLengthDP.build(W, system.max_allowed_ones_run)


def unrank_word(system, dp, rank: int, W: int) -> tuple[int, ...]:
    """Return the rank-th admissible word in the system's finite codebook."""
    if system.name == "binary":
        return tuple(int(ch) for ch in format(rank, f"0{W}b"))
    return tuple(dp.unrank(rank))


def iter_single_digit_corruptions(
    word: tuple[int, ...],
    alphabet: tuple[int, ...],
) -> Iterable[tuple[tuple[int, ...], int]]:
    """Yield all single-digit corruptions and their active digit count."""
    for pos_idx, original_digit in enumerate(word):
        for new_digit in alphabet:
            if new_digit == original_digit:
                continue
            corrupted = list(word)
            corrupted[pos_idx] = new_digit
            yield tuple(corrupted), 1


def iter_burst_corruptions(
    word: tuple[int, ...],
    alphabet: tuple[int, ...],
    burst_width: int,
) -> Iterable[tuple[tuple[int, ...], int]]:
    """Yield all contiguous burst corruptions for one burst width.

    For every start position, this enumerates every nonzero binary mask and every
    alphabet-aware replacement at active positions. For binary alphabets each
    active position has one possible replacement; for signed NAF it has two.
    """
    W = len(word)
    if burst_width < 1 or burst_width > W:
        return

    for start in range(W - burst_width + 1):
        offsets = range(burst_width)
        for mask_bits in itertools.product((0, 1), repeat=burst_width):
            if not any(mask_bits):
                continue

            active_positions = [start + offset for offset in offsets if mask_bits[offset]]
            replacement_choices = [
                tuple(digit for digit in alphabet if digit != word[pos])
                for pos in active_positions
            ]

            for replacements in itertools.product(*replacement_choices):
                corrupted = list(word)
                for pos, new_digit in zip(active_positions, replacements):
                    corrupted[pos] = new_digit
                yield tuple(corrupted), len(active_positions)


def run_exhaustive_for_system(
    system,
    window: DigitWindow,
    burst_widths: list[int],
    *,
    include_bursts: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run exhaustive single-digit and burst checks for one system/window."""
    W = window.width
    total_words = system.count_codebook(W)
    alphabet = digit_alphabet(system)
    dp = build_unranker(system, W)

    single_counts = MetricCounts()
    burst_counts = MetricCounts()
    burst_by_width = {b: MetricCounts() for b in burst_widths}

    for rank in range(total_words):
        word = unrank_word(system, dp, rank, W)

        for corrupted_word, active_count in iter_single_digit_corruptions(word, alphabet):
            single_counts.update(
                system=system,
                window=window,
                original_word=word,
                corrupted_word=corrupted_word,
                active_count=active_count,
            )

        if include_bursts:
            for b in burst_widths:
                for corrupted_word, active_count in iter_burst_corruptions(word, alphabet, b):
                    burst_counts.update(
                        system=system,
                        window=window,
                        original_word=word,
                        corrupted_word=corrupted_word,
                        active_count=active_count,
                    )
                    burst_by_width[b].update(
                        system=system,
                        window=window,
                        original_word=word,
                        corrupted_word=corrupted_word,
                        active_count=active_count,
                    )

    summary_row: dict[str, Any] = {
        "system": system.name,
        "W": W,
        "L": window.L,
        "R": window.R,
        "total_words": total_words,
        "burst_widths": ",".join(str(b) for b in burst_widths) if include_bursts else "",
    }
    summary_row.update(single_counts.rates("single"))
    summary_row.update(burst_counts.rates("burst"))

    burst_rows: list[dict[str, Any]] = []
    if include_bursts:
        for b in burst_widths:
            row: dict[str, Any] = {
                "system": system.name,
                "W": W,
                "L": window.L,
                "R": window.R,
                "total_words": total_words,
                "burst_width": b,
            }
            row.update(burst_by_width[b].rates("burst"))
            burst_rows.append(row)

    return summary_row, burst_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exhaustive verification for Experiment 5 according to the paper protocol."
    )
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[4, 6, 8, 10, 12, 14, 16],
        help="Small finite-window widths to enumerate. Protocol default: all even W <= 16.",
    )
    parser.add_argument(
        "--systems",
        type=str,
        nargs="+",
        default=SYSTEM_ORDER,
    )
    parser.add_argument(
        "--burst-widths",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5],
        help="Burst widths to exhaustively enumerate.",
    )
    parser.add_argument(
        "--no-bursts",
        action="store_true",
        help="Debug/smoke-test option: enumerate only single-digit corruptions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "data" / "experiment_05",
    )
    args = parser.parse_args()

    systems = make_default_systems()
    summary_rows: list[dict[str, Any]] = []
    burst_rows: list[dict[str, Any]] = []

    include_bursts = not args.no_bursts

    for W in args.widths:
        window = DigitWindow.balanced(W)
        for system_name in args.systems:
            print(f"Running exhaustive eval for {system_name} at W={W}...")
            summary_row, system_burst_rows = run_exhaustive_for_system(
                systems[system_name],
                window,
                args.burst_widths,
                include_bursts=include_bursts,
            )
            summary_rows.append(summary_row)
            burst_rows.extend(system_burst_rows)

            single_rate = summary_row["single_semantic_survival_rate"]
            burst_rate = summary_row["burst_semantic_survival_rate"] if include_bursts else 0.0
            print(
                "  -> single corruptions: "
                f"{summary_row['single_total_corruptions']}, "
                f"single semantic survival: {single_rate:.6f}, "
                f"burst semantic survival: {burst_rate:.6f}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)
    print(f"[done] Saved {args.output_dir / 'summary.csv'}")

    single_columns = [
        col for col in summary_df.columns
        if col in {"system", "W", "L", "R", "total_words"} or col.startswith("single_")
    ]
    summary_df[single_columns].to_csv(args.output_dir / "single_digit_summary.csv", index=False)
    print(f"[done] Saved {args.output_dir / 'single_digit_summary.csv'}")

    if include_bursts:
        burst_df = pd.DataFrame(burst_rows)
        burst_df.to_csv(args.output_dir / "burst_by_width.csv", index=False)
        print(f"[done] Saved {args.output_dir / 'burst_by_width.csv'}")


if __name__ == "__main__":
    main()