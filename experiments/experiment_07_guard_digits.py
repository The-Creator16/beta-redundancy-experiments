from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redundancy_exp import DigitWindow, make_default_systems
from redundancy_exp.arithmetic import (
    digitwise_add,
    canonicalize_multinacci_addition,
    canonicalize_binary_addition,
    canonicalize_signed2_naf,
)
from redundancy_exp.corruption import value_key_to_float


SYSTEM_ORDER = ["binary", "signed2_naf", "phi", "tribonacci", "tetranacci"]


def result_attr(result: Any, names: tuple[str, ...], default: float = 0.0) -> Any:
    """Read the first available attribute from a canonicalization result."""
    for name in names:
        if hasattr(result, name):
            return getattr(result, name)
    return default


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else 0.0


def canonicalize_for_system(system, raw: tuple[int, ...], window: DigitWindow):
    if system.name == "binary":
        return canonicalize_binary_addition(raw, window)
    if system.name == "signed2_naf":
        return canonicalize_signed2_naf(raw, window)
    return canonicalize_multinacci_addition(raw, window, degree=system.degree)


def run_guard_digit_trial(
    system,
    window_orig: DigitWindow,
    guard_digits: int,
    trials: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    W = window_orig.width
    window_ext = DigitWindow(L=window_orig.L, R=window_orig.R + guard_digits)

    exactness_count = 0
    success_count = 0
    overflow_count = 0
    truncation_count = 0

    errors: list[float] = []
    steps_list: list[float] = []
    propagation_depths: list[float] = []

    for _ in range(trials):
        a = system.sample_word(window_orig, rng)
        b = system.sample_word(window_orig, rng)

        # Add lower-order guard digits on the fractional side.
        a_ext = a + (0,) * guard_digits
        b_ext = b + (0,) * guard_digits

        raw = digitwise_add(a_ext, b_ext)
        result = canonicalize_for_system(system, raw, window_ext)

        success = bool(result_attr(result, ("success",), True))
        overflow = bool(result_attr(result, ("overflow", "overflowed"), False))
        truncation = bool(result_attr(result, ("truncation", "truncated"), False))
        steps = float(result_attr(result, ("steps", "normalization_steps", "kappa"), 0.0))
        propagation_depth = float(
            result_attr(result, ("propagation_depth", "max_propagation_depth", "depth", "pi"), np.nan)
        )

        success_count += int(success)
        overflow_count += int(overflow)
        truncation_count += int(truncation)
        steps_list.append(steps)
        propagation_depths.append(propagation_depth)

        repaired_ext = tuple(result.digits)
        repaired_orig = repaired_ext if guard_digits == 0 else repaired_ext[:-guard_digits]

        val_a = value_key_to_float(system, system.value_key(a, window_orig))
        val_b = value_key_to_float(system, system.value_key(b, window_orig))
        val_repaired = value_key_to_float(system, system.value_key(repaired_orig, window_orig))

        err = abs(val_repaired - (val_a + val_b))
        errors.append(err)

        if err < 1e-9:
            exactness_count += 1

    finite_depths = [x for x in propagation_depths if not np.isnan(x)]

    return {
        "system": system.name,
        "W": W,
        "L": window_orig.L,
        "R": window_orig.R,
        "guard_digits": guard_digits,
        "trials": trials,
        "arithmetic_exactness_rate": exactness_count / trials,
        "canonicalization_success_rate": success_count / trials,
        "overflow_rate": overflow_count / trials,
        "truncation_rate": truncation_count / trials,
        "mean_abs_error": float(np.mean(errors)),
        "worst_abs_error": float(np.max(errors)),
        "mean_normalization_steps": float(np.mean(steps_list)),
        "max_normalization_steps": float(np.max(steps_list)),
        "p50_normalization_steps": quantile(steps_list, 0.50),
        "p90_normalization_steps": quantile(steps_list, 0.90),
        "p99_normalization_steps": quantile(steps_list, 0.99),
        "mean_propagation_depth": float(np.mean(finite_depths)) if finite_depths else np.nan,
        "max_propagation_depth": float(np.max(finite_depths)) if finite_depths else np.nan,
        "p50_propagation_depth": quantile(finite_depths, 0.50) if finite_depths else np.nan,
        "p90_propagation_depth": quantile(finite_depths, 0.90) if finite_depths else np.nan,
        "p99_propagation_depth": quantile(finite_depths, 0.99) if finite_depths else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run guard-digit analysis for Experiment 7 according to the paper protocol."
    )
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[8, 12, 16, 20, 24],
        help="Base window widths W to test. Protocol default: 8 12 16 20 24.",
    )
    parser.add_argument(
        "--guards",
        type=int,
        nargs="+",
        default=[0, 2, 4, 8, 12, 16],
        help="Lower-order guard digit counts.",
    )
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--systems",
        type=str,
        nargs="+",
        default=SYSTEM_ORDER,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "data" / "experiment_07",
    )
    args = parser.parse_args()

    systems = make_default_systems()
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []

    for W in args.widths:
        window_orig = DigitWindow.balanced(W)
        for system_name in args.systems:
            for guard_digits in args.guards:
                print(f"Running Guard Digits for {system_name} (W={W}, g={guard_digits})...")
                row = run_guard_digit_trial(
                    systems[system_name],
                    window_orig,
                    guard_digits,
                    args.trials,
                    rng,
                )
                rows.append(row)
                print(
                    f"  -> Exactness: {row['arithmetic_exactness_rate']:.4f}, "
                    f"MAE: {row['mean_abs_error']:.4g}, "
                    f"Truncation: {row['truncation_rate']:.4f}, "
                    f"p99 kappa: {row['p99_normalization_steps']:.4g}"
                )

    df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "summary.csv", index=False)
    print(f"[done] Saved to {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()