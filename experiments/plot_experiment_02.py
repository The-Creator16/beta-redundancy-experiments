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
    "arithmetic_exactness_rate",
    "truncation_rate",
    "mean_normalization_steps",
    "p99_normalization_steps",
}

REQUIRED_CASCADE_COLUMNS = {
    "system",
    "W",
    "steps",
    "probability",
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
    cascade_csv: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, validate, and normalize Experiment 2 input data."""

    if not summary_csv.exists():
        raise FileNotFoundError(f"Could not find summary CSV: {summary_csv}")

    if not cascade_csv.exists():
        raise FileNotFoundError(f"Could not find cascade histogram CSV: {cascade_csv}")

    summary = pd.read_csv(summary_csv)
    cascade = pd.read_csv(cascade_csv)

    validate_columns(
        summary,
        REQUIRED_SUMMARY_COLUMNS,
        source_name=str(summary_csv),
    )
    validate_columns(
        cascade,
        REQUIRED_CASCADE_COLUMNS,
        source_name=str(cascade_csv),
    )

    summary = convert_numeric_columns(summary)
    cascade = convert_numeric_columns(cascade)

    # These fields are discrete parameters in the experiment.
    summary["W"] = summary["W"].astype(int)
    cascade["W"] = cascade["W"].astype(int)
    cascade["steps"] = cascade["steps"].astype(int)

    warn_about_unknown_systems(summary, source_name=str(summary_csv))
    warn_about_unknown_systems(cascade, source_name=str(cascade_csv))

    output_dir.mkdir(parents=True, exist_ok=True)

    return summary, cascade


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
# Line metric panels
# =============================================================================

def create_endpoint_label_layout(
    ax: plt.Axes,
    curves: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create non-overlapping right-side label positions."""

    if not curves:
        return []

    y_min, y_max = ax.get_ylim()
    y_range = max(y_max - y_min, 1.0)

    min_gap = 0.060 * y_range
    bottom_limit = y_min + 0.045 * y_range
    top_limit = y_max - 0.045 * y_range

    items: list[dict[str, Any]] = []

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


def plot_window_metric_panel(
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    metric_column: str,
    title: str,
    ylabel: str,
    output_stem: str,
    endpoint_labels: bool = False,
) -> None:
    """Plot one metric as a function of window width."""

    if metric_column not in summary.columns:
        print(f"[skip] Column not found: {metric_column}")
        return

    window_widths = get_window_widths(summary)
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

    ax.set_title(title, pad=7)
    ax.set_xlabel(r"Window width ($W$)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(window_widths)

    if endpoint_labels:
        ax.set_xlim(min(window_widths), max(window_widths) + 0.95)
        annotate_curve_endpoints(ax, curves)

    style_axis(ax, grid_axis="y")

    save_panel_figure(fig, output_dir, output_stem)
    plt.close(fig)


def plot_arithmetic_exactness_rate(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot arithmetic exactness rate by window width."""

    plot_window_metric_panel(
        summary,
        output_dir,
        metric_column="arithmetic_exactness_rate",
        title="Arithmetic exactness rate",
        ylabel=r"Arithmetic exactness rate ($\rho_{\mathrm{arith}}$)",
        output_stem="fig_exp2_arithmetic_exactness_rate_vs_w",
        endpoint_labels=True,
    )


def plot_truncation_rate(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot truncation rate by window width."""

    plot_window_metric_panel(
        summary,
        output_dir,
        metric_column="truncation_rate",
        title="Truncation rate",
        ylabel=r"Truncation rate ($\rho_{\mathrm{trunc}}$)",
        output_stem="fig_exp2_truncation_rate_vs_w",
        endpoint_labels=True,
    )


def plot_mean_rewrite_cost(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot mean canonicalization cost by window width."""

    plot_window_metric_panel(
        summary,
        output_dir,
        metric_column="mean_normalization_steps",
        title="Mean canonicalization cost",
        ylabel=r"Mean rewrite operations ($\overline{\kappa}$)",
        output_stem="fig_exp2_mean_rewrite_cost_clean",
        endpoint_labels=True,
    )


def plot_p99_rewrite_cost(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot p99 canonicalization cost by window width."""

    plot_window_metric_panel(
        summary,
        output_dir,
        metric_column="p99_normalization_steps",
        title="Tail canonicalization cost",
        ylabel=r"99th percentile rewrite operations ($p_{99}(\kappa)$)",
        output_stem="fig_exp2_p99_rewrite_cost_clean",
        endpoint_labels=True,
    )


# =============================================================================
# Cascade tail panel
# =============================================================================

def compute_tail_distribution(subset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute Pr[kappa >= s] from a cascade histogram subset."""

    ordered = subset.sort_values("steps")

    steps = ordered["steps"].to_numpy(dtype=int)
    probabilities = ordered["probability"].to_numpy(dtype=float)

    # Tail probability: P[kappa >= s].
    tail_probabilities = probabilities[::-1].cumsum()[::-1]
    nonzero_mask = tail_probabilities > 0

    return steps[nonzero_mask], tail_probabilities[nonzero_mask]


def plot_cascade_tail_distribution(cascade: pd.DataFrame, output_dir: Path) -> None:
    """Plot cascade tail distributions at the maximum observed window width."""

    max_window = int(cascade["W"].max())
    cascade_at_max_window = cascade[cascade["W"] == max_window].copy()

    if cascade_at_max_window.empty:
        print("[skip] No cascade histogram rows found.")
        return

    fig, ax = create_square_panel()

    max_step = int(cascade_at_max_window["steps"].max())

    for system in SYSTEM_ORDER:
        style = SYSTEM_STYLES[system]
        subset = cascade_at_max_window[cascade_at_max_window["system"] == system]

        if subset.empty:
            continue

        steps, tail_probabilities = compute_tail_distribution(subset)

        if len(steps) == 0:
            continue

        ax.step(
            steps,
            tail_probabilities,
            where="post",
            color=style["color"],
            linewidth=1.95,
        )

        # Sparse markers make the step curve readable without clutter.
        marker_stride = max(1, len(steps) // 7)

        ax.plot(
            steps[::marker_stride],
            tail_probabilities[::marker_stride],
            linestyle="None",
            marker=style["marker"],
            markersize=4.5,
            color=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.55,
        )

    ax.set_yscale("log")
    ax.set_xlim(0, max_step + 2)
    ax.set_ylim(5e-5, 1.2)

    ax.set_title(rf"Cascade tail distribution at $W={max_window}$", pad=7)
    ax.set_xlabel(r"Rewrite operation threshold ($s$)")
    ax.set_ylabel(r"Tail probability $\Pr[\kappa \geq s]$")

    style_axis(ax, grid_axis="both")

    save_panel_figure(fig, output_dir, "fig_exp2_cascade_tail_w_max_clean")
    plt.close(fig)


# =============================================================================
# Exactness-cost trade-off panel
# =============================================================================

def get_max_window_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Return summary rows at maximum W, ordered by known system order."""

    max_window = int(summary["W"].max())
    subset = summary[summary["W"] == max_window].copy()

    order_map = {system: index for index, system in enumerate(SYSTEM_ORDER)}
    subset["system_order"] = subset["system"].map(order_map)

    return subset.sort_values("system_order")


def plot_exactness_cost_tradeoff(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot exactness versus mean rewrite cost at the maximum window width."""

    max_window = int(summary["W"].max())
    subset = get_max_window_summary(summary)

    if subset.empty:
        print("[skip] No summary rows found for maximum W.")
        return

    fig, ax = create_square_panel()

    for _, row in subset.iterrows():
        system = str(row["system"])

        if system not in SYSTEM_STYLES:
            continue

        style = SYSTEM_STYLES[system]

        x = float(row["mean_normalization_steps"])
        y = float(row["arithmetic_exactness_rate"])

        # Soft halo plus solid marker improves visibility in print.
        ax.scatter(
            x,
            y,
            s=230,
            color=style["color"],
            alpha=0.18,
            edgecolor="none",
            zorder=4,
        )

        ax.scatter(
            x,
            y,
            s=125,
            color=style["color"],
            edgecolor="white",
            linewidth=1.15,
            zorder=6,
        )

        label_offset = 0.024

        if system == "tetranacci":
            label_offset = 0.032

        ax.text(
            x,
            y + label_offset,
            style["short"],
            ha="center",
            va="bottom",
            fontsize=9.0,
            color=style["color"],
            zorder=7,
        )

    ax.set_title(rf"Exactness--cost trade-off at $W={max_window}$", pad=7)
    ax.set_xlabel(r"Mean rewrite operations ($\overline{\kappa}$)")
    ax.set_ylabel(r"Arithmetic exactness rate ($\rho_{\mathrm{arith}}$)")

    ax.set_ylim(-0.03, 1.05)

    x_min = float(subset["mean_normalization_steps"].min())
    x_max = float(subset["mean_normalization_steps"].max())
    x_padding = max(0.8, 0.10 * (x_max - x_min))

    ax.set_xlim(x_min - x_padding, x_max + x_padding)

    style_axis(ax, grid_axis="both")

    save_panel_figure(fig, output_dir, "fig_exp2_exactness_cost_tradeoff_w_max_clean")
    plt.close(fig)


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
        "fig_exp2_legend_strip",
        extra_artists=(legend,),
    )
    plt.close(fig)


# =============================================================================
# Main entry point
# =============================================================================

def generate_all_figures(
    summary_csv: Path,
    cascade_csv: Path,
    output_dir: Path,
) -> None:
    """Generate all Experiment 2 paper figures."""

    configure_matplotlib_style()

    summary, cascade = load_experiment_data(
        summary_csv=summary_csv,
        cascade_csv=cascade_csv,
        output_dir=output_dir,
    )

    print(f"[info] Loaded summary: {summary_csv}")
    print(f"[info] Loaded cascade histogram: {cascade_csv}")
    print(f"[info] Output directory: {output_dir}")
    print("[info] Generating Experiment 2 figures...")

    plot_arithmetic_exactness_rate(summary, output_dir)
    plot_truncation_rate(summary, output_dir)
    plot_mean_rewrite_cost(summary, output_dir)
    plot_p99_rewrite_cost(summary, output_dir)
    plot_cascade_tail_distribution(cascade, output_dir)
    plot_exactness_cost_tradeoff(summary, output_dir)
    plot_legend_strip(output_dir)

    print("[done] Experiment 2 paper figures generated.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create clean paper-style figures for Experiment 2."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiment_02" / "summary.csv",
        help="Path to Experiment 2 summary.csv.",
    )
    parser.add_argument(
        "--cascade-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiment_02" / "cascade_histogram.csv",
        help="Path to Experiment 2 cascade_histogram.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiment_02" / "paper_figures",
        help="Directory where paper figures will be saved.",
    )

    args = parser.parse_args()

    generate_all_figures(
        summary_csv=args.summary_csv,
        cascade_csv=args.cascade_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()