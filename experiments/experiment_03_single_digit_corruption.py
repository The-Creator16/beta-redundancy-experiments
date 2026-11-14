from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running this script directly from the repository root without installing.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redundancy_exp import DigitWindow, make_default_systems
from redundancy_exp.corruption import run_single_digit_corruption_benchmark_for_system


def _format_markdown_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    columns = [str(c) for c in df.columns]
    rows = [
        [_format_markdown_cell(value) for value in row]
        for row in df.itertuples(index=False, name=None)
    ]
    widths = []
    for j, column in enumerate(columns):
        content_width = max((len(row[j]) for row in rows), default=0)
        widths.append(max(len(column), content_width))

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[j]) for j, v in enumerate(values)) + " |"

    header = render_row(columns)
    separator = "| " + " | ".join("-" * w for w in widths) + " |"
    body = [render_row(row) for row in rows]
    path.write_text("\n".join([header, separator, *body]) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 3: single-digit corruption benchmark."
    )
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[8, 12, 16, 20, 24],
        help="Window widths W to evaluate.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10_000,
        help="Number of Monte Carlo corruptions per (system, W).",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["binary", "signed2_naf", "phi", "tribonacci", "tetranacci"],
        help="Systems to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=20260622, help="Random seed.")
    parser.add_argument(
        "--max-steps-factor",
        type=int,
        default=6,
        help="Repair rewrite budget as factor * W.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "data" / "experiment_03",
        help="Directory where outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")

    systems = make_default_systems()
    unknown = sorted(set(args.systems) - set(systems))
    if unknown:
        raise ValueError(f"Unknown systems: {unknown}. Available: {sorted(systems)}")

    rng = np.random.default_rng(args.seed)
    summary_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []

    for W in args.widths:
        window = DigitWindow.balanced(W)
        for system_name in args.systems:
            system = systems[system_name]
            summary, by_position = run_single_digit_corruption_benchmark_for_system(
                system=system,
                window=window,
                trials=args.trials,
                rng=rng,
                max_steps_factor=args.max_steps_factor,
            )
            summary_rows.append(summary.to_dict())
            position_rows.extend(row.to_dict() for row in by_position)

            print(
                f"{summary.system:12s} W={summary.W:2d} "
                f"delta_det={summary.structural_detectability_rate:.4f} "
                f"rho_struct={summary.structural_correctness_rate:.4f} "
                f"rho_exact={summary.structural_exactness_rate:.4f} "
                f"rho_sem={summary.semantic_survival_rate:.4f} "
                f"mae={summary.mean_abs_error:.4g} "
                f"k_mean={summary.mean_normalization_steps:.2f} "
                f"pi_mean={summary.mean_propagation_depth:.2f}"
            )

    df_summary = pd.DataFrame(summary_rows)
    df_positions = pd.DataFrame(position_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = args.output_dir / "summary.csv"
    summary_md = args.output_dir / "summary.md"
    positions_csv = args.output_dir / "position_summary.csv"
    positions_md = args.output_dir / "position_summary.md"

    df_summary.to_csv(summary_csv, index=False)
    df_positions.to_csv(positions_csv, index=False)
    write_markdown_table(df_summary, summary_md)
    write_markdown_table(df_positions, positions_md)

    print(f"\nWrote: {summary_csv}")
    print(f"Wrote: {summary_md}")
    print(f"Wrote: {positions_csv}")
    print(f"Wrote: {positions_md}")


if __name__ == "__main__":
    main()
