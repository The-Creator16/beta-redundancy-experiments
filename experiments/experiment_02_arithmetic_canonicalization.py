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
from redundancy_exp.arithmetic import run_arithmetic_benchmark_for_system


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
        description="Experiment 2: arithmetic canonicalization benchmark."
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
        help="Number of Monte Carlo additions per (system, W).",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["binary", "signed2_naf", "phi", "tribonacci", "tetranacci"],
        help="Systems to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=20260621, help="Random seed.")
    parser.add_argument(
        "--max-steps-factor",
        type=int,
        default=3,
        help="Canonicalization rewrite budget as factor * W.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "data" / "experiment_02",
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
    hist_rows: list[dict[str, object]] = []

    for W in args.widths:
        window = DigitWindow.balanced(W)
        for system_name in args.systems:
            system = systems[system_name]
            summary, hist = run_arithmetic_benchmark_for_system(
                system=system,
                window=window,
                trials=args.trials,
                rng=rng,
                max_steps_factor=args.max_steps_factor,
            )
            summary_rows.append(summary.to_dict())
            for steps, count in sorted(hist.items()):
                hist_rows.append(
                    {
                        "system": summary.system,
                        "W": summary.W,
                        "steps": steps,
                        "count": count,
                        "probability": count / summary.trials,
                    }
                )
            print(
                f"{summary.system:12s} W={summary.W:2d} "
                f"rho_arith={summary.arithmetic_exactness_rate:.4f} "
                f"p_norm={summary.canonicalization_success_rate:.4f} "
                f"k_mean={summary.mean_normalization_steps:.2f} "
                f"pi_mean={summary.mean_propagation_depth:.2f} "
                f"overflow={summary.overflow_rate:.4f} "
                f"trunc={summary.truncation_rate:.4f}"
            )

    df_summary = pd.DataFrame(summary_rows)
    df_hist = pd.DataFrame(hist_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = args.output_dir / "summary.csv"
    summary_md = args.output_dir / "summary.md"
    hist_csv = args.output_dir / "cascade_histogram.csv"
    hist_md = args.output_dir / "cascade_histogram.md"

    df_summary.to_csv(summary_csv, index=False)
    df_hist.to_csv(hist_csv, index=False)
    write_markdown_table(df_summary, summary_md)
    write_markdown_table(df_hist, hist_md)

    print(f"\nWrote: {summary_csv}")
    print(f"Wrote: {summary_md}")
    print(f"Wrote: {hist_csv}")
    print(f"Wrote: {hist_md}")


if __name__ == "__main__":
    main()
