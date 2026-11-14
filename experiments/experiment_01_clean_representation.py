from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running the script directly from the repository root without installing.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redundancy_exp import DigitWindow, make_default_systems
from redundancy_exp.metrics import run_clean_benchmark_for_system


def _format_markdown_cell(value: object) -> str:
    """Format one DataFrame cell for a dependency-free Markdown table.

    pandas.DataFrame.to_markdown requires the optional third-party package
    ``tabulate``. We avoid that optional dependency here so that the experiment
    can run in a minimal scientific Python environment containing only numpy
    and pandas.
    """
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value)
    return text.replace("|", "\\|")


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    """Write a compact Markdown table without pandas optional dependencies."""
    columns = [str(c) for c in df.columns]
    rows = [
        [_format_markdown_cell(value) for value in row]
        for row in df.itertuples(index=False, name=None)
    ]

    # Compute column widths for readable plain-text Markdown output.
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
        description="Experiment 1: clean finite-window representation benchmark."
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
        help="Number of Monte Carlo samples per (system, W).",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["binary", "signed2_naf", "phi", "tribonacci", "tetranacci"],
        help="Systems to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=20260620, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "data" / "experiment_01",
        help="Directory where CSV outputs will be written.",
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
    rows: list[dict[str, object]] = []

    for W in args.widths:
        window = DigitWindow.balanced(W)
        for system_name in args.systems:
            system = systems[system_name]
            summary = run_clean_benchmark_for_system(
                system=system,
                window=window,
                trials=args.trials,
                rng=rng,
            )
            rows.append(summary.to_dict())
            print(
                f"{summary.system:12s} W={summary.W:2d} "
                f"sparsity={summary.codebook_sparsity:.6g} "
                f"density={summary.mean_nonzero_density:.4f} "
                f"rho_rt={summary.round_trip_exactness_rate:.4f}"
            )

    df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.csv"
    df.to_csv(summary_path, index=False)

    # A compact Markdown table is convenient for direct paper notes.
    # We deliberately do not use pandas.to_markdown here because it requires
    # the optional ``tabulate`` dependency and can make the experiment fail
    # after all numerical results have already been computed.
    md_path = args.output_dir / "summary.md"
    write_markdown_table(df, md_path)

    print(f"\nWrote: {summary_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
