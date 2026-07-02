from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation


# Repository root: this script is expected to live in experiments/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# Figure settings
# =============================================================================

PANEL_FIGSIZE = (4.8, 4.8)
POSITION_FIGSIZE = (7.4, 3.9)
LEGEND_FIGSIZE = (7.2, 0.72)
VERTICAL_LEGEND_FIGSIZE = (1.85, 3.05)
PNG_DPI = 240


# =============================================================================
# Data schema
# =============================================================================

REQUIRED_SUMMARY_COLUMNS = {
    "system",
    "W",
    "structural_detectability_rate",
    "structural_exactness_rate",
    "semantic_survival_rate",
    "mean_abs_error",
    "mean_normalization_steps",
}

REQUIRED_POSITION_COLUMNS = {
    "system",
    "W",
    "position_value",
    "structural_detectability_rate",
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


def load_summary_data(summary_csv: Path, output_dir: Path) -> pd.DataFrame:
    """Load, validate, and normalize Experiment 3 summary data."""

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


def load_position_data(position_csv: Path) -> pd.DataFrame:
    """Load, validate, and normalize per-position Experiment 3 data."""

    if not position_csv.exists():
        raise FileNotFoundError(f"Could not find position CSV: {position_csv}")

    position = pd.read_csv(position_csv)

    validate_columns(
        position,
        REQUIRED_POSITION_COLUMNS,
        source_name=str(position_csv),
    )

    position = convert_numeric_columns(position)

    # Position-level plots use discrete W and digit position values.
    position["W"] = position["W"].astype(int)
    position["position_value"] = position["position_value"].astype(int)

    warn_about_unknown_systems(position, source_name=str(position_csv))

    return position


# =============================================================================
# Plot helpers
# =============================================================================

def get_window_widths(dataframe: pd.DataFrame) -> list[int]:
    """Return sorted window widths from a dataframe."""

    return sorted(int(value) for value in dataframe["W"].unique())


def choose_signed_position_ticks(values: list[int]) -> list[int]:
    """Return readable ticks for signed digit positions."""

    min_value = min(values)
    max_value = max(values)

    ticks = list(range(-8, max_value + 1, 4))

    if min_value not in ticks:
        ticks.insert(0, min_value)

    if max_value not in ticks:
        ticks.append(max_value)

    return sorted(set(ticks))


def create_square_panel() -> tuple[plt.Figure, plt.Axes]:
    """Create a square panel figure with one axis."""

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE, constrained_layout=True)
    ax.set_box_aspect(1)

    return fig, ax


def create_position_panel() -> tuple[plt.Figure, plt.Axes]:
    """Create a wide rectangular panel for signed-position curves."""

    fig, ax = plt.subplots(figsize=POSITION_FIGSIZE, constrained_layout=True)

    return fig, ax


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    """Apply shared clean axis styling."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)


def save_panel_figure(fig: plt.Figure, output_dir: Path, output_stem: str) -> None:
    """Save a figure as PNG and PDF."""

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


def padded_limits(
    values: pd.Series | np.ndarray,
    *,
    lower_floor: float | None = None,
    upper_ceiling: float | None = None,
    relative_padding: float = 0.10,
    absolute_padding: float = 0.02,
) -> tuple[float, float]:
    """Return compact y-limits around the observed values."""

    array = np.asarray(values, dtype=float)

    y_min = float(np.nanmin(array))
    y_max = float(np.nanmax(array))

    if y_min == y_max:
        padding = max(abs(y_max) * relative_padding, absolute_padding)
    else:
        padding = max((y_max - y_min) * relative_padding, absolute_padding)

    lower = y_min - padding
    upper = y_max + padding

    if lower_floor is not None:
        lower = max(lower, lower_floor)

    if upper_ceiling is not None:
        upper = min(upper, upper_ceiling)

    if lower >= upper:
        upper = lower + max(absolute_padding, 0.01)

    return lower, upper


def add_scale_note(ax: plt.Axes, text: str) -> None:
    """Add a small unobtrusive note about a non-linear axis scale."""

    ax.text(
        0.03,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#555555",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#dddddd",
            "alpha": 0.92,
        },
        zorder=20,
    )


# =============================================================================
# Endpoint labels
# =============================================================================

def annotate_curve_endpoints(
    ax: plt.Axes,
    curves: list[dict[str, Any]],
    *,
    x_offset: float = 0.42,
    x_offsets: dict[str, float] | None = None,
    y_offsets_points: dict[str, float] | None = None,
) -> None:
    """Add compact labels at the actual right endpoint of each curve.

    Labels are anchored at the real endpoint y-value. Small point offsets are
    used only when several labels are very close.
    """

    if not curves:
        return

    x_offsets = x_offsets or {}
    y_offsets_points = y_offsets_points or {}

    x_anchor = max(float(np.asarray(curve["x"], dtype=float)[-1]) for curve in curves)

    for curve in curves:
        system = str(curve["system"])
        style = SYSTEM_STYLES[system]

        y = np.asarray(curve["y"], dtype=float)
        y_end = float(y[-1])

        label_x = x_anchor + x_offset + float(x_offsets.get(system, 0.0))
        label_y = y_end

        point_offset = float(y_offsets_points.get(system, 0.0))
        transform = ax.transData

        if point_offset != 0.0:
            transform = (
                ax.transData
                + ScaledTranslation(
                    0.0,
                    point_offset / 72.0,
                    ax.figure.dpi_scale_trans,
                )
            )

        ax.text(
            label_x,
            label_y,
            style["short"],
            transform=transform,
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
    y_limits: tuple[float, float] | None = None,
    yscale: str = "linear",
    endpoint_labels: bool = True,
    label_y_offsets_points: dict[str, float] | None = None,
) -> None:
    """Plot one Experiment 3 metric as a function of window width."""

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

    if yscale == "symlog":
        # Useful when one system is much larger but zeros must remain visible.
        ax.set_yscale("symlog", linthresh=0.25, linscale=0.9)
    elif yscale != "linear":
        ax.set_yscale(yscale)

    ax.set_title(title, pad=7)
    ax.set_xlabel(r"Window width ($W$)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(window_widths)

    if endpoint_labels:
        ax.set_xlim(min(window_widths), max(window_widths) + 0.95)
    else:
        # Small padding keeps the rightmost markers from touching the border.
        ax.set_xlim(min(window_widths), max(window_widths) + 0.25)

    if y_limits is not None:
        ax.set_ylim(*y_limits)

    style_axis(ax, grid_axis="y")

    if yscale == "symlog":
        add_scale_note(ax, "symlog y-scale")
    elif yscale == "log":
        add_scale_note(ax, "log y-scale")

    if endpoint_labels:
        annotate_curve_endpoints(
            ax,
            curves,
            y_offsets_points=label_y_offsets_points,
        )

    save_panel_figure(fig, output_dir, output_stem)
    plt.close(fig)


def plot_structural_detectability(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot structural detectability by window width with compact y-limits."""

    y_limits = padded_limits(
        summary["structural_detectability_rate"],
        lower_floor=-0.03,
        upper_ceiling=1.05,
        relative_padding=0.12,
        absolute_padding=0.035,
    )

    # Keep the zero baseline visible when binary is exactly zero.
    y_limits = (min(y_limits[0], -0.03), y_limits[1])

    plot_window_metric_panel(
        summary,
        output_dir,
        metric_column="structural_detectability_rate",
        title="Structural detectability",
        ylabel=r"Structural detectability rate",
        output_stem="fig_exp3_structural_detectability_vs_w",
        y_limits=y_limits,
        endpoint_labels=True,
    )


def plot_structural_exactness_gap(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot structural exactness as a failure gap: 1 - exactness."""

    transformed = summary.copy()
    transformed["structural_exactness_gap"] = (
        1.0 - transformed["structural_exactness_rate"]
    ).clip(lower=0.0)

    gap_max = float(transformed["structural_exactness_gap"].max())
    upper = max(gap_max * 1.30, 0.015)

    plot_window_metric_panel(
        transformed,
        output_dir,
        metric_column="structural_exactness_gap",
        title="Structural exactness gap",
        ylabel=r"Failure rate ($1-\rho_{\mathrm{exact}}$)",
        output_stem="fig_exp3_structural_exactness_gap_vs_w",
        y_limits=(-0.002, upper),
        endpoint_labels=False,
    )


def plot_semantic_survival(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot semantic survival, using a special panel when all values are zero."""

    values = summary["semantic_survival_rate"].to_numpy(dtype=float)

    if np.allclose(values, 0.0):
        plot_zero_semantic_survival(summary, output_dir)
        return

    y_limits = padded_limits(
        values,
        lower_floor=-0.03,
        upper_ceiling=1.05,
        relative_padding=0.12,
        absolute_padding=0.035,
    )

    plot_window_metric_panel(
        summary,
        output_dir,
        metric_column="semantic_survival_rate",
        title="Semantic survival",
        ylabel=r"Semantic survival rate",
        output_stem="fig_exp3_semantic_survival_vs_w",
        y_limits=y_limits,
        endpoint_labels=True,
    )


def plot_zero_semantic_survival(summary: pd.DataFrame, output_dir: Path) -> None:
    """Show the all-zero semantic survival case without wasting the panel."""

    window_widths = get_window_widths(summary)

    fig, ax = create_square_panel()

    ax.axhline(
        0.0,
        color="#333333",
        linewidth=1.45,
        alpha=0.75,
        zorder=1,
    )

    # Small horizontal offsets reveal that all systems are present at y=0.
    offsets = np.linspace(-0.12, 0.12, len(SYSTEM_ORDER))

    for offset, system in zip(offsets, SYSTEM_ORDER):
        style = SYSTEM_STYLES[system]
        subset = get_system_subset(summary, system)

        if subset.empty:
            continue

        x = subset["W"].to_numpy(dtype=float) + offset
        y = subset["semantic_survival_rate"].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            linestyle="None",
            color=style["color"],
            marker=style["marker"],
            markersize=5.8,
            markeredgecolor="white",
            markeredgewidth=0.85,
            alpha=0.95,
        )

    ax.text(
        0.5,
        0.70,
        "All systems have zero semantic survival",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.0,
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.26",
            "facecolor": "white",
            "edgecolor": "#dddddd",
            "alpha": 0.96,
        },
    )

    ax.text(
        0.5,
        0.59,
        "Markers are horizontally offset only to show overlap.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.6,
        color="#555555",
    )

    ax.set_title("Semantic survival collapse", pad=7)
    ax.set_xlabel(r"Window width ($W$)")
    ax.set_ylabel(r"Semantic survival rate")
    ax.set_xticks(window_widths)
    ax.set_xlim(min(window_widths) - 0.45, max(window_widths) + 0.45)
    ax.set_ylim(-0.012, 0.075)

    style_axis(ax, grid_axis="y")

    save_panel_figure(fig, output_dir, "fig_exp3_semantic_survival_vs_w")
    plt.close(fig)


def plot_mean_abs_error(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot mean absolute error by window width."""

    values = summary["mean_abs_error"].to_numpy(dtype=float)
    y_limits = padded_limits(
        values,
        lower_floor=0.0,
        relative_padding=0.12,
        absolute_padding=0.20,
    )

    plot_window_metric_panel(
        summary,
        output_dir,
        metric_column="mean_abs_error",
        title="Mean absolute error",
        ylabel=r"Mean absolute error",
        output_stem="fig_exp3_mean_abs_error_vs_w",
        y_limits=y_limits,
        endpoint_labels=True,
    )


def plot_mean_repair_steps(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot mean repair cost using a symlog scale."""

    values = summary["mean_normalization_steps"].to_numpy(dtype=float)
    y_max = float(np.nanmax(values))
    y_limits = (-0.03, max(y_max * 1.18, 1.0))

    plot_window_metric_panel(
        summary,
        output_dir,
        metric_column="mean_normalization_steps",
        title="Mean repair cost",
        ylabel=r"Mean repair operations",
        output_stem="fig_exp3_mean_repair_steps_vs_w",
        y_limits=y_limits,
        yscale="symlog",
        endpoint_labels=True,
        label_y_offsets_points={
            "binary": 5.0,
        },
    )


# =============================================================================
# Position panel
# =============================================================================

def choose_position_window(position: pd.DataFrame, requested_window: int | None) -> int:
    """Resolve the W value used for the per-position plot."""

    available_windows = get_window_widths(position)

    if requested_window is None:
        return max(available_windows)

    if requested_window not in available_windows:
        print(
            f"[warn] Requested W={requested_window} is not available in "
            f"position_summary.csv. Using W={max(available_windows)} instead."
        )
        return max(available_windows)

    return requested_window


def plot_position_detectability(
    position: pd.DataFrame,
    output_dir: Path,
    *,
    target_window: int,
) -> None:
    """Plot structural detectability by signed digit position."""

    position_at_window = position[position["W"] == target_window].copy()

    if position_at_window.empty:
        print(f"[skip] No position rows found for W={target_window}.")
        return

    position_values = sorted(
        int(value) for value in position_at_window["position_value"].unique()
    )

    y_limits = padded_limits(
        position_at_window["structural_detectability_rate"],
        lower_floor=-0.03,
        upper_ceiling=1.05,
        relative_padding=0.10,
        absolute_padding=0.04,
    )

    # Keep the binary zero baseline visible.
    y_limits = (min(y_limits[0], -0.03), y_limits[1])

    fig, ax = create_position_panel()

    for system in SYSTEM_ORDER:
        style = SYSTEM_STYLES[system]
        subset = position_at_window[
            position_at_window["system"] == system
        ].sort_values("position_value")

        if subset.empty:
            continue

        x = subset["position_value"].to_numpy(dtype=float)
        y = subset["structural_detectability_rate"].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            markevery=2,
            markersize=4.7,
            markeredgecolor="white",
            markeredgewidth=0.65,
            linewidth=1.75,
            alpha=0.94,
        )

    ax.set_title(rf"Position-wise detectability at $W={target_window}$", pad=7)
    ax.set_xlabel(r"Digit position ($i$)")
    ax.set_ylabel(r"Structural detectability rate")

    ax.set_xticks(choose_signed_position_ticks(position_values))
    ax.set_xlim(min(position_values), max(position_values) + 0.35)
    ax.set_ylim(*y_limits)

    style_axis(ax, grid_axis="y")

    save_panel_figure(
        fig,
        output_dir,
        f"fig_exp3_structural_detectability_by_position_w_{target_window}",
    )
    plt.close(fig)


# =============================================================================
# Separate legend strips
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
    """Export a horizontal legend strip matching the other experiments."""

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
        "fig_exp3_legend_strip",
        extra_artists=(legend,),
    )
    plt.close(fig)


def plot_vertical_legend_strip(output_dir: Path) -> None:
    """Export a vertical legend strip for side placement."""

    fig, ax = plt.subplots(figsize=VERTICAL_LEGEND_FIGSIZE)
    ax.axis("off")

    handles, labels = build_legend_handles()

    legend = fig.legend(
        handles,
        labels,
        loc="center",
        ncol=1,
        frameon=False,
        handlelength=2.0,
        handletextpad=0.65,
        labelspacing=1.05,
        borderaxespad=0.0,
    )

    save_legend_figure(
        fig,
        output_dir,
        "fig_exp3_legend_strip_vertical",
        extra_artists=(legend,),
    )
    plt.close(fig)


# =============================================================================
# Main entry point
# =============================================================================

def generate_all_figures(
    summary_csv: Path,
    position_csv: Path,
    output_dir: Path,
    position_window: int | None,
) -> None:
    """Generate all Experiment 3 paper figures."""

    configure_matplotlib_style()

    summary = load_summary_data(
        summary_csv=summary_csv,
        output_dir=output_dir,
    )

    print(f"[info] Loaded summary: {summary_csv}")
    print(f"[info] Output directory: {output_dir}")
    print("[info] Generating Experiment 3 figures...")

    plot_structural_detectability(summary, output_dir)
    plot_structural_exactness_gap(summary, output_dir)
    plot_semantic_survival(summary, output_dir)
    plot_mean_abs_error(summary, output_dir)
    plot_mean_repair_steps(summary, output_dir)

    if position_csv.exists():
        position = load_position_data(position_csv)
        target_window = choose_position_window(position, position_window)

        print(f"[info] Loaded position summary: {position_csv}")
        print(f"[info] Position plot window: W={target_window}")

        plot_position_detectability(
            position,
            output_dir,
            target_window=target_window,
        )
    else:
        print(f"[skip] Position summary not found: {position_csv}")

    plot_legend_strip(output_dir)
    plot_vertical_legend_strip(output_dir)

    print("[done] Experiment 3 paper figures generated.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create clean paper-style figures for Experiment 3."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiment_03" / "summary.csv",
        help="Path to Experiment 3 summary.csv.",
    )
    parser.add_argument(
        "--position-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiment_03" / "position_summary.csv",
        help="Path to Experiment 3 position_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiment_03" / "paper_figures",
        help="Directory where paper figures will be saved.",
    )
    parser.add_argument(
        "--position-w",
        type=int,
        default=None,
        help=(
            "Window width for the per-position plot. "
            "If omitted, the maximum W in position_summary.csv is used."
        ),
    )

    args = parser.parse_args()

    generate_all_figures(
        summary_csv=args.summary_csv,
        position_csv=args.position_csv,
        output_dir=args.output_dir,
        position_window=args.position_w,
    )


if __name__ == "__main__":
    main()