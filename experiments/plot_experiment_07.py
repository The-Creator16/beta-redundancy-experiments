from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# Repository root: this script is expected to live in experiments/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# Figure settings
# =============================================================================

PANEL_FIGSIZE = (4.8, 4.8)
LEGEND_FIGSIZE = (7.2, 0.72)
PNG_DPI = 240


# =============================================================================
# Data schema
# =============================================================================

REQUIRED_SUMMARY_COLUMNS = {
    "system",
    "W",
    "guard_digits",
    "arithmetic_exactness_rate",
    "truncation_rate",
    "mean_abs_error",
    "mean_normalization_steps",
    "p99_normalization_steps",
}


# =============================================================================
# System styles
# =============================================================================

SYSTEM_STYLES = {
    "binary": {
        "label": "Binary",
        "short": "B",
        "color": "#333333",
        "marker": "s",
    },
    "signed2_naf": {
        "label": "Signed NAF",
        "short": "S",
        "color": "#0072B2",
        "marker": "D",
    },
    "phi": {
        "label": r"$\Phi$",
        "short": r"$\Phi$",
        "color": "#009E73",
        "marker": "o",
    },
    "tribonacci": {
        "label": r"$T_3$",
        "short": r"$T_3$",
        "color": "#E69F00",
        "marker": "^",
    },
    "tetranacci": {
        "label": r"$T_4$",
        "short": r"$T_4$",
        "color": "#CC79A7",
        "marker": "v",
    },
}

SYSTEM_ORDER = [
    "binary",
    "signed2_naf",
    "phi",
    "tribonacci",
    "tetranacci",
]


# =============================================================================
# Matplotlib style
# =============================================================================

def configure_matplotlib_style() -> None:
    """Configure a clean, paper-friendly Matplotlib style."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "mathtext.fontset": "dejavusans",

            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.titleweight": "normal",
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.3,

            "axes.linewidth": 0.75,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#111111",
            "xtick.color": "#222222",
            "ytick.color": "#222222",

            "grid.color": "#dddddd",
            "grid.linestyle": "-",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.60,

            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",

            # Keep text editable in vector outputs.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# =============================================================================
# Data loading and validation
# =============================================================================

def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    *,
    source_name: str,
) -> None:
    """Raise a clear error if required columns are missing."""

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{source_name} is missing columns: {sorted(missing_columns)}\n"
            f"Available columns: {list(dataframe.columns)}"
        )


def convert_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert all non-system columns to numeric values."""

    converted = dataframe.copy()
    converted["system"] = converted["system"].astype(str)

    for column in converted.columns:
        if column != "system":
            converted[column] = pd.to_numeric(converted[column], errors="raise")

    return converted


def warn_about_unknown_systems(
    dataframe: pd.DataFrame,
    *,
    source_name: str,
) -> None:
    """Print a warning when input files contain systems without plot styles."""

    unknown_systems = sorted(set(dataframe["system"]) - set(SYSTEM_STYLES))

    if unknown_systems:
        print(
            f"[warn] Unknown systems in {source_name} will be skipped: "
            f"{unknown_systems}"
        )


def load_experiment_data(
    summary_csv: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Load, validate, and normalize Experiment 7 input data."""

    if not summary_csv.exists():
        raise FileNotFoundError(f"Could not find summary CSV: {summary_csv}")

    summary = pd.read_csv(summary_csv)

    validate_columns(
        summary,
        REQUIRED_SUMMARY_COLUMNS,
        source_name=str(summary_csv),
    )

    summary = convert_numeric_columns(summary)

    # These fields are discrete parameters in the experiment.
    summary["W"] = summary["W"].astype(int)
    summary["guard_digits"] = summary["guard_digits"].astype(int)

    warn_about_unknown_systems(summary, source_name=str(summary_csv))

    output_dir.mkdir(parents=True, exist_ok=True)

    return summary


# =============================================================================
# Plot helpers
# =============================================================================

def create_square_panel() -> tuple[plt.Figure, plt.Axes]:
    """Create a square panel figure with one axis."""

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE, constrained_layout=True)
    ax.set_box_aspect(1)

    return fig, ax


def style_axis(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    """Apply shared clean axis styling."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)


def save_panel_figure(fig: plt.Figure, output_dir: Path, output_stem: str) -> None:
    """Save a square panel figure as PNG and PDF."""

    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"

    fig.savefig(png_path, dpi=PNG_DPI)
    fig.savefig(pdf_path)

    print(f"[ok] Wrote {png_path}")
    print(f"[ok] Wrote {pdf_path}")


def save_legend_figure(
    fig: plt.Figure,
    output_dir: Path,
    output_stem: str,
    *,
    extra_artists: tuple[Any, ...] = (),
) -> None:
    """Save the separate legend strip with a tight bounding box."""

    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"

    fig.savefig(
        png_path,
        dpi=PNG_DPI,
        bbox_inches="tight",
        pad_inches=0.02,
        bbox_extra_artists=extra_artists,
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.02,
        bbox_extra_artists=extra_artists,
    )

    print(f"[ok] Wrote {png_path}")
    print(f"[ok] Wrote {pdf_path}")


def get_system_subset(dataframe: pd.DataFrame, system: str) -> pd.DataFrame:
    """Return rows for one known system, sorted by guard digits."""
    return dataframe[dataframe["system"] == system].sort_values("guard_digits")


# =============================================================================
# Line metric panels
# =============================================================================

def plot_guard_metric_panel(
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    target_w: int,
    metric_column: str,
    title: str,
    ylabel: str,
    output_stem: str,
    is_log_scale: bool = False,
) -> None:
    """Plot one metric as a function of guard digits."""

    if metric_column not in summary.columns:
        print(f"[skip] Column not found: {metric_column}")
        return

    subset_w = summary[summary["W"] == target_w]
    if subset_w.empty:
        print(f"[skip] No data found for W={target_w}")
        return

    guard_digits_list = sorted(int(value) for value in subset_w["guard_digits"].unique())
    fig, ax = create_square_panel()

    for system in SYSTEM_ORDER:
        style = SYSTEM_STYLES[system]
        subset = get_system_subset(subset_w, system)

        if subset.empty:
            continue

        x = subset["guard_digits"].to_numpy(dtype=float)
        y = subset[metric_column].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            markersize=5.8,
            markeredgecolor="white",
            markeredgewidth=0.85,
            linewidth=1.95,
            alpha=0.94,
        )

    ax.set_title(rf"{title} at $W={target_w}$", pad=7)
    ax.set_xlabel(r"Guard digits ($g$)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(guard_digits_list)

    if is_log_scale:
        ax.set_yscale("log")

    style_axis(ax, grid_axis="both")

    save_panel_figure(fig, output_dir, output_stem)
    plt.close(fig)


def plot_truncation_rate(summary: pd.DataFrame, output_dir: Path, target_w: int) -> None:
    """Plot truncation rate by guard digits."""

    plot_guard_metric_panel(
        summary,
        output_dir,
        target_w=target_w,
        metric_column="truncation_rate",
        title="Truncation rate",
        ylabel=r"Truncation rate ($\rho_{\mathrm{trunc}}$)",
        output_stem="fig_exp7_truncation_vs_guard",
        is_log_scale=False,
    )


def plot_mean_abs_error(summary: pd.DataFrame, output_dir: Path, target_w: int) -> None:
    """Plot mean absolute error by guard digits (log scale)."""

    plot_guard_metric_panel(
        summary,
        output_dir,
        target_w=target_w,
        metric_column="mean_abs_error",
        title="Mean Absolute Error",
        ylabel=r"Mean Absolute Error (MAE)",
        output_stem="fig_exp7_mae_vs_guard",
        is_log_scale=True, # MAE is typically plotted on log scale
    )


# =============================================================================
# Separate legend strip
# =============================================================================

def build_legend_handles() -> tuple[list[Line2D], list[str]]:
    """Build legend handles and labels for all known systems."""

    handles: list[Line2D] = []
    labels: list[str] = []

    for system in SYSTEM_ORDER:
        style = SYSTEM_STYLES[system]

        handle = Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=1.95,
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.65,
        )

        handles.append(handle)
        labels.append(style["label"])

    return handles, labels


def plot_legend_strip(output_dir: Path) -> None:
    """Export the separate legend strip used below the panel block."""

    fig, ax = plt.subplots(figsize=LEGEND_FIGSIZE)
    ax.axis("off")

    handles, labels = build_legend_handles()

    legend = fig.legend(
        handles,
        labels,
        loc="center",
        ncol=5,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.3,
        handletextpad=0.55,
        borderaxespad=0.0,
    )

    save_legend_figure(
        fig,
        output_dir,
        "fig_exp7_legend_strip",
        extra_artists=(legend,),
    )
    plt.close(fig)


# =============================================================================
# Main entry point
# =============================================================================

def generate_all_figures(
    summary_csv: Path,
    output_dir: Path,
    target_w: int,
) -> None:
    """Generate all Experiment 7 paper figures."""

    configure_matplotlib_style()

    summary = load_experiment_data(
        summary_csv=summary_csv,
        output_dir=output_dir,
    )

    print(f"[info] Loaded summary: {summary_csv}")
    print(f"[info] Output directory: {output_dir}")
    print(f"[info] Generating Experiment 7 figures for W={target_w}...")

    plot_truncation_rate(summary, output_dir, target_w)
    plot_mean_abs_error(summary, output_dir, target_w)
    plot_legend_strip(output_dir)

    print("[done] Experiment 7 paper figures generated.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create clean paper-style figures for Experiment 7."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiment_07" / "summary.csv",
        help="Path to Experiment 7 summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiment_07" / "paper_figures",
        help="Directory where paper figures will be saved.",
    )
    parser.add_argument(
        "--target-w",
        type=int,
        default=16,
        help="Window width used for the plots (default from your CSV is 16)."
    )

    args = parser.parse_args()

    generate_all_figures(
        summary_csv=args.summary_csv,
        output_dir=args.output_dir,
        target_w=args.target_w,
    )


if __name__ == "__main__":
    main()