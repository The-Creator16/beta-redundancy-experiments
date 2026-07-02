from __future__ import annotations

import argparse
import math
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
    "codebook_sparsity",
    "mean_nonzero_density",
    "capacity_bits_per_digit",
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
    """Load, validate, and normalize Experiment 1 input data."""

    if not summary_csv.exists():
        raise FileNotFoundError(f"Could not find summary CSV: {summary_csv}")

    summary = pd.read_csv(summary_csv)

    validate_columns(
        summary,
        REQUIRED_SUMMARY_COLUMNS,
        source_name=str(summary_csv),
    )

    summary = convert_numeric_columns(summary)

    # Window width is a discrete experiment parameter.
    summary["W"] = summary["W"].astype(int)

    warn_about_unknown_systems(summary, source_name=str(summary_csv))

    output_dir.mkdir(parents=True, exist_ok=True)

    return summary


# =============================================================================
# Plot helpers
# =============================================================================

def get_window_widths(dataframe: pd.DataFrame) -> list[int]:
    """Return sorted window widths from a dataframe."""

    return sorted(int(value) for value in dataframe["W"].unique())


def create_square_panel() -> tuple[plt.Figure, plt.Axes]:
    """Create a square panel figure with one axis."""

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE, constrained_layout=True)
    ax.set_box_aspect(1)

    return fig, ax


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
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
    """Return rows for one known system, sorted by window width."""

    return dataframe[dataframe["system"] == system].sort_values("W")


# =============================================================================
# Endpoint labels
# =============================================================================

def create_endpoint_label_layout(
    ax: plt.Axes,
    curves: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create non-overlapping right-side label positions."""

    if not curves:
        return []

    y_min, y_max = ax.get_ylim()

    if ax.get_yscale() == "log":
        y_min_log = math.log10(max(y_min, 1e-300))
        y_max_log = math.log10(max(y_max, 1e-300))
        y_range_log = max(y_max_log - y_min_log, 1.0)

        min_gap_log = 0.060 * y_range_log
        bottom_limit_log = y_min_log + 0.045 * y_range_log
        top_limit_log = y_max_log - 0.045 * y_range_log

        items: list[dict[str, Any]] = []

        for curve in curves:
            x = np.asarray(curve["x"], dtype=float)
            y = np.asarray(curve["y"], dtype=float)
            y_end_log = math.log10(max(float(y[-1]), 1e-300))

            items.append(
                {
                    "system": str(curve["system"]),
                    "x_end": float(x[-1]),
                    "y_end_log": y_end_log,
                    "y_label_log": y_end_log,
                }
            )

        items.sort(key=lambda item: item["y_end_log"])

        previous_y_log = -np.inf

        for item in items:
            item["y_label_log"] = max(
                float(item["y_label_log"]),
                previous_y_log + min_gap_log,
            )
            previous_y_log = float(item["y_label_log"])

        if items:
            overflow = float(items[-1]["y_label_log"]) - top_limit_log

            if overflow > 0:
                for item in items:
                    item["y_label_log"] = float(item["y_label_log"]) - overflow

            underflow = bottom_limit_log - float(items[0]["y_label_log"])

            if underflow > 0:
                for item in items:
                    item["y_label_log"] = float(item["y_label_log"]) + underflow

        for item in items:
            item["y_label"] = 10.0 ** float(item["y_label_log"])

        return items

    y_range = max(y_max - y_min, 1.0)
    min_gap = 0.060 * y_range
    bottom_limit = y_min + 0.045 * y_range
    top_limit = y_max - 0.045 * y_range

    items = []

    for curve in curves:
        x = np.asarray(curve["x"], dtype=float)
        y = np.asarray(curve["y"], dtype=float)

        items.append(
            {
                "system": str(curve["system"]),
                "x_end": float(x[-1]),
                "y_end": float(y[-1]),
                "y_label": float(y[-1]),
            }
        )

    # Push labels apart from bottom to top.
    items.sort(key=lambda item: item["y_end"])

    previous_y = -np.inf

    for item in items:
        item["y_label"] = max(float(item["y_label"]), previous_y + min_gap)
        previous_y = float(item["y_label"])

    # Keep labels inside the current y-axis range.
    if items:
        overflow = float(items[-1]["y_label"]) - top_limit

        if overflow > 0:
            for item in items:
                item["y_label"] = float(item["y_label"]) - overflow

        underflow = bottom_limit - float(items[0]["y_label"])

        if underflow > 0:
            for item in items:
                item["y_label"] = float(item["y_label"]) + underflow

    return items


def annotate_curve_endpoints(
    ax: plt.Axes,
    curves: list[dict[str, Any]],
) -> None:
    """Add compact labels near the right endpoint of each curve."""

    label_layout = create_endpoint_label_layout(ax, curves)

    if not label_layout:
        return

    x_anchor = max(float(item["x_end"]) for item in label_layout)
    x_label = x_anchor + 0.42

    for item in label_layout:
        system = str(item["system"])
        style = SYSTEM_STYLES[system]

        ax.text(
            x_label,
            float(item["y_label"]),
            style["short"],
            color=style["color"],
            fontsize=9.0,
            ha="left",
            va="center",
            clip_on=False,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
            zorder=10,
        )


# =============================================================================
# Reference lines
# =============================================================================

def add_nonzero_density_references(ax: plt.Axes, max_window: int) -> None:
    """Add standard asymptotic density references."""

    phi_value = (1.0 + math.sqrt(5.0)) / 2.0
    phi_density = 2.0 - phi_value

    references = [
        ("binary", 0.50, r"0.50"),
        ("signed2_naf", 1.0 / 3.0, r"0.33"),
        ("phi", phi_density, r"$2-\Phi$"),
    ]

    # Put asymptote labels farther right than curve endpoint labels.
    reference_label_x = max_window + 1.20

    for system, y_value, label in references:
        style = SYSTEM_STYLES[system]

        ax.axhline(
            y_value,
            color=style["color"],
            linestyle=":",
            linewidth=1.20,
            alpha=0.45,
            zorder=1,
        )

        ax.text(
            reference_label_x,
            y_value,
            label,
            color=style["color"],
            fontsize=8.0,
            ha="left",
            va="center",
            clip_on=False,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.90,
            },
            zorder=9,
        )

def add_capacity_references(ax: plt.Axes, max_window: int) -> None:
    """Add the Fibonacci-shift information capacity reference."""

    phi_value = (1.0 + math.sqrt(5.0)) / 2.0
    shannon_phi = math.log2(phi_value)

    reference_label_x = max_window + 1.20

    ax.axhline(
        shannon_phi,
        color=SYSTEM_STYLES["phi"]["color"],
        linestyle=":",
        linewidth=1.20,
        alpha=0.45,
        zorder=1,
    )

    ax.text(
        reference_label_x,
        shannon_phi,
        r"$\log_2 \Phi$",
        color=SYSTEM_STYLES["phi"]["color"],
        fontsize=8.0,
        ha="left",
        va="center",
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
        },
        zorder=9,
    )


# =============================================================================
# Line metric panels
# =============================================================================

def plot_window_metric_panel(
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    metric_column: str,
    title: str,
    ylabel: str,
    output_stem: str,
    log_y: bool = False,
    y_limits: tuple[float, float] | None = None,
    x_padding: float = 0.95,
    endpoint_labels: bool = True,
    add_references: bool = False,
) -> None:
    """Plot one Experiment 1 metric as a function of window width."""

    if metric_column not in summary.columns:
        print(f"[skip] Column not found: {metric_column}")
        return

    window_widths = get_window_widths(summary)
    min_window = min(window_widths)
    max_window = max(window_widths)

    fig, ax = create_square_panel()

    curves: list[dict[str, Any]] = []

    for system in SYSTEM_ORDER:
        style = SYSTEM_STYLES[system]
        subset = get_system_subset(summary, system)

        if subset.empty:
            continue

        x = subset["W"].to_numpy(dtype=float)
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

        curves.append(
            {
                "system": system,
                "x": x,
                "y": y,
            }
        )

    if log_y:
        ax.set_yscale("log")

    ax.set_title(title, pad=7)
    ax.set_xlabel(r"Window width ($W$)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(window_widths)

    # Extra right-side room is used for compact curve labels and references.
    ax.set_xlim(min_window, max_window + x_padding)

    if y_limits is not None:
        ax.set_ylim(*y_limits)

    if add_references:
        if metric_column == "mean_nonzero_density":
            add_nonzero_density_references(ax, max_window)
        elif metric_column == "capacity_bits_per_digit":
            add_capacity_references(ax, max_window)

    style_axis(ax, grid_axis="both" if log_y else "y")

    if endpoint_labels:
        annotate_curve_endpoints(ax, curves)

    save_panel_figure(fig, output_dir, output_stem)
    plt.close(fig)


def plot_codebook_sparsity(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot codebook sparsity by window width."""

    plot_window_metric_panel(
        summary,
        output_dir,
        metric_column="codebook_sparsity",
        title="Representational codebook sparsity",
        ylabel=r"Sparsity ratio ($|C_X^{L,R}| \,/\, |A_X|^W$)",
        output_stem="fig_exp1_codebook_sparsity_vs_w",
        log_y=True,
        x_padding=0.95,
        endpoint_labels=True,
    )


def plot_nonzero_density(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot average nonzero digit density by window width."""

    plot_window_metric_panel(
        summary,
        output_dir,
        metric_column="mean_nonzero_density",
        title="Average nonzero digit density",
        ylabel=r"Average nonzero density ($\omega_X$)",
        output_stem="fig_exp1_nonzero_density_vs_w",
        y_limits=(0.20, 0.55),
        x_padding=2.35,
        endpoint_labels=True,
        add_references=True,
    )


def plot_information_capacity(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot information capacity per digit by window width."""

    plot_window_metric_panel(
        summary,
        output_dir,
        metric_column="capacity_bits_per_digit",
        title="Information capacity",
        ylabel="Capacity (bits per digit)",
        output_stem="fig_exp1_capacity_bits_per_digit_vs_w",
        y_limits=(0.65, 1.10),
        x_padding=2.35,
        endpoint_labels=True,
        add_references=True,
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
    """Export a separate legend strip matching the Experiment 2 style."""

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
        "fig_exp1_legend_strip",
        extra_artists=(legend,),
    )
    plt.close(fig)


# =============================================================================
# Main entry point
# =============================================================================

def generate_all_figures(
    summary_csv: Path,
    output_dir: Path,
) -> None:
    """Generate all Experiment 1 paper figures."""

    configure_matplotlib_style()

    summary = load_experiment_data(
        summary_csv=summary_csv,
        output_dir=output_dir,
    )

    print(f"[info] Loaded summary: {summary_csv}")
    print(f"[info] Output directory: {output_dir}")
    print("[info] Generating Experiment 1 figures...")

    plot_codebook_sparsity(summary, output_dir)
    plot_nonzero_density(summary, output_dir)
    plot_information_capacity(summary, output_dir)
    plot_legend_strip(output_dir)

    print("[done] Experiment 1 paper figures generated.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create clean paper-style figures for Experiment 1."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiment_01" / "summary.csv",
        help="Path to Experiment 1 summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiment_01" / "paper_figures",
        help="Directory where paper figures will be saved.",
    )

    args = parser.parse_args()

    generate_all_figures(
        summary_csv=args.summary_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()