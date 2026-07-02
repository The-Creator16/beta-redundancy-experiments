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
WIDE_PANEL_FIGSIZE = (5.8, 3.9)
LEGEND_FIGSIZE = (7.2, 0.72)
VERTICAL_LEGEND_FIGSIZE = (1.85, 3.05)
PNG_DPI = 240


# =============================================================================
# Data schema
# =============================================================================

REQUIRED_SUMMARY_COLUMNS = {
    "system",
    "W",
}

REQUIRED_BURST_WIDTH_COLUMNS = {
    "system",
    "W",
    "burst_width",
}

NON_NUMERIC_COLUMNS = {
    "system",
    "burst_widths",
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
# Loading and validation
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
    """Convert numeric columns while preserving known string columns."""

    converted = dataframe.copy()

    if "system" in converted.columns:
        converted["system"] = converted["system"].astype(str)

    for column in converted.columns:
        if column in NON_NUMERIC_COLUMNS:
            continue

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
    """Load, validate, and normalize Experiment 4 summary data."""

    if not summary_csv.exists():
        raise FileNotFoundError(f"Could not find summary CSV: {summary_csv}")

    summary = pd.read_csv(summary_csv)

    validate_columns(
        summary,
        REQUIRED_SUMMARY_COLUMNS,
        source_name=str(summary_csv),
    )

    summary = convert_numeric_columns(summary)
    summary["W"] = summary["W"].astype(int)
    summary = summary.sort_values(["system", "W"]).reset_index(drop=True)

    warn_about_unknown_systems(summary, source_name=str(summary_csv))

    output_dir.mkdir(parents=True, exist_ok=True)

    return summary


def load_burst_width_data(burst_width_csv: Path) -> pd.DataFrame:
    """Load, validate, and normalize burst-width Experiment 4 data."""

    if not burst_width_csv.exists():
        raise FileNotFoundError(f"Could not find burst-width CSV: {burst_width_csv}")

    burst_width = pd.read_csv(burst_width_csv)

    validate_columns(
        burst_width,
        REQUIRED_BURST_WIDTH_COLUMNS,
        source_name=str(burst_width_csv),
    )

    burst_width = convert_numeric_columns(burst_width)
    burst_width["W"] = burst_width["W"].astype(int)
    burst_width["burst_width"] = burst_width["burst_width"].astype(int)
    burst_width = burst_width.sort_values(
        ["system", "W", "burst_width"]
    ).reset_index(drop=True)

    warn_about_unknown_systems(burst_width, source_name=str(burst_width_csv))

    return burst_width


# =============================================================================
# Plot helpers
# =============================================================================

def get_window_widths(dataframe: pd.DataFrame) -> list[int]:
    """Return sorted window widths."""

    return sorted(int(value) for value in dataframe["W"].unique())


def get_burst_widths(dataframe: pd.DataFrame) -> list[int]:
    """Return sorted burst widths."""

    return sorted(int(value) for value in dataframe["burst_width"].unique())


def create_square_panel() -> tuple[plt.Figure, plt.Axes]:
    """Create a square panel."""

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE, constrained_layout=True)
    ax.set_box_aspect(1)

    return fig, ax


def create_wide_panel() -> tuple[plt.Figure, plt.Axes]:
    """Create a square data panel.

    Kept for backward compatibility with older calls that passed wide=True.
    All non-legend Experiment 4 panels are intentionally square.
    """

    return create_square_panel()


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    """Apply shared clean axis styling."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)


def save_panel_figure(fig: plt.Figure, output_dir: Path, output_stem: str) -> None:
    """Save figure as PNG and PDF."""

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
    """Save legend strip as PNG and PDF."""

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


def padded_limits(
    values: pd.Series | np.ndarray,
    *,
    lower_floor: float | None = None,
    upper_ceiling: float | None = None,
    relative_padding: float = 0.10,
    absolute_padding: float = 0.02,
) -> tuple[float, float]:
    """Return compact y-limits around observed values."""

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
    """Add a small unobtrusive note about non-linear scale."""

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


def get_system_subset(dataframe: pd.DataFrame, system: str, sort_column: str) -> pd.DataFrame:
    """Return rows for one known system."""

    return dataframe[dataframe["system"] == system].sort_values(sort_column)


# =============================================================================
# Generic panels
# =============================================================================

def plot_lines(
    dataframe: pd.DataFrame,
    output_dir: Path,
    *,
    x_column: str,
    metric_column: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_stem: str,
    y_limits: tuple[float, float] | None = None,
    yscale: str = "linear",
    wide: bool = False,
    x_padding: float = 0.25,
    zero_line: bool = False,
) -> None:
    """Plot one metric as a square line panel.

    The ``wide`` argument is kept only to avoid touching all older call sites.
    It no longer changes the panel geometry: every data figure is square.
    Legend strips still use their own non-square figure sizes.
    """

    if metric_column not in dataframe.columns:
        print(f"[skip] Column not found: {metric_column}")
        return

    fig, ax = create_square_panel()

    x_values = sorted(int(value) for value in dataframe[x_column].unique())

    for system in SYSTEM_ORDER:
        style = SYSTEM_STYLES[system]
        subset = get_system_subset(dataframe, system, x_column)

        if subset.empty:
            continue

        ax.plot(
            subset[x_column],
            subset[metric_column],
            color=style["color"],
            marker=style["marker"],
            markersize=5.6 if wide else 5.8,
            markeredgecolor="white",
            markeredgewidth=0.80,
            linewidth=1.90,
            alpha=0.94,
        )

    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=0.25, linscale=0.9)
    elif yscale == "log":
        ax.set_yscale("log")
    elif yscale != "linear":
        ax.set_yscale(yscale)

    ax.set_title(title, pad=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_values)
    ax.set_xlim(min(x_values), max(x_values) + x_padding)

    if y_limits is not None:
        ax.set_ylim(*y_limits)

    if zero_line:
        ax.axhline(
            0.0,
            color="#777777",
            linewidth=0.75,
            linestyle="--",
            alpha=0.70,
            zorder=0,
        )

    style_axis(ax, grid_axis="y")

    if yscale == "symlog":
        add_scale_note(ax, "symlog y-scale")
    elif yscale == "log":
        add_scale_note(ax, "log y-scale")

    save_panel_figure(fig, output_dir, output_stem)
    plt.close(fig)


# =============================================================================
# Derived data
# =============================================================================

def make_exactness_gap(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add structural exactness gap."""

    if "structural_exactness_rate" not in dataframe.columns:
        raise ValueError("Column not found: structural_exactness_rate")

    transformed = dataframe.copy()
    transformed["structural_exactness_gap"] = (
        1.0 - transformed["structural_exactness_rate"]
    ).clip(lower=0.0)

    return transformed


def make_metric_delta_against_baseline(
    dataframe: pd.DataFrame,
    *,
    metric_column: str,
    output_column: str,
    baseline_system: str = "binary",
) -> pd.DataFrame:
    """Add metric difference against a baseline system at the same W."""

    if metric_column not in dataframe.columns:
        raise ValueError(f"Column not found: {metric_column}")

    baseline = dataframe[dataframe["system"] == baseline_system][["W", metric_column]]

    if baseline.empty:
        raise ValueError(f"Baseline system not found: {baseline_system}")

    baseline = baseline.rename(columns={metric_column: "baseline_value"})

    transformed = dataframe.merge(
        baseline,
        on="W",
        how="left",
        validate="many_to_one",
    )

    if transformed["baseline_value"].isna().any():
        missing_windows = sorted(
            int(value)
            for value in transformed.loc[
                transformed["baseline_value"].isna(),
                "W",
            ].unique()
        )
        raise ValueError(
            f"Missing {baseline_system} baseline values for W={missing_windows}"
        )

    transformed[output_column] = transformed[metric_column] - transformed["baseline_value"]
    transformed = transformed.drop(columns=["baseline_value"])

    return transformed


def make_log1p_metric(
    dataframe: pd.DataFrame,
    *,
    metric_column: str,
    output_column: str,
) -> pd.DataFrame:
    """Add log1p(metric) for zero-safe logarithmic plotting."""

    if metric_column not in dataframe.columns:
        raise ValueError(f"Column not found: {metric_column}")

    if (dataframe[metric_column] < 0).any():
        raise ValueError(f"log1p transform requires non-negative values: {metric_column}")

    transformed = dataframe.copy()
    transformed[output_column] = np.log1p(transformed[metric_column].astype(float))

    return transformed


def set_log1p_operation_ticks(
    ax: plt.Axes,
    *,
    max_value: float,
    candidate_ticks: tuple[int, ...] = (0, 1, 2, 3, 5, 10, 20, 30, 50, 100),
) -> None:
    """Use original operation counts as labels on a log1p-transformed axis."""

    ticks = [tick for tick in candidate_ticks if tick <= max_value]

    if not ticks or ticks[-1] < max_value:
        ticks.append(int(np.ceil(max_value)))

    ax.set_yticks(np.log1p(ticks))
    ax.set_yticklabels([str(tick) for tick in ticks])


def make_burst_gain_dataframe(
    burst_width: pd.DataFrame,
    *,
    metric_column: str,
    output_column: str,
    low_burst: int = 2,
    high_burst: int = 5,
) -> pd.DataFrame:
    """Compute metric gain between two burst widths for every system and W."""

    rows: list[dict[str, Any]] = []

    for system in SYSTEM_ORDER:
        system_rows = burst_width[burst_width["system"] == system]

        for window_width in sorted(system_rows["W"].unique()):
            at_window = system_rows[system_rows["W"] == window_width]

            low_rows = at_window[at_window["burst_width"] == low_burst]
            high_rows = at_window[at_window["burst_width"] == high_burst]

            if low_rows.empty or high_rows.empty:
                continue

            low_value = float(low_rows[metric_column].iloc[0])
            high_value = float(high_rows[metric_column].iloc[0])

            rows.append(
                {
                    "system": system,
                    "W": int(window_width),
                    output_column: high_value - low_value,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Burst-width conditioned figures at W = target_window
# =============================================================================

def filter_target_window(
    burst_width: pd.DataFrame,
    *,
    target_window: int,
) -> pd.DataFrame:
    """Return burst-width rows for one W."""

    filtered = burst_width[burst_width["W"] == target_window].copy()

    if filtered.empty:
        raise ValueError(f"No burst-width rows found for W={target_window}")

    return filtered


def plot_detectability_by_burst_width(
    burst_width: pd.DataFrame,
    output_dir: Path,
    *,
    target_window: int,
) -> None:
    """Plot structural detectability by burst width."""

    data = filter_target_window(burst_width, target_window=target_window)

    y_limits = padded_limits(
        data["structural_detectability_rate"],
        lower_floor=-0.03,
        upper_ceiling=1.05,
        relative_padding=0.12,
        absolute_padding=0.04,
    )

    plot_lines(
        data,
        output_dir,
        x_column="burst_width",
        metric_column="structural_detectability_rate",
        title=rf"Detectability by burst width at $W={target_window}$",
        xlabel=r"Burst width ($b$)",
        ylabel=r"Structural detectability rate",
        output_stem=f"fig_exp4_detectability_by_burst_width_w_{target_window}",
        y_limits=y_limits,
        wide=False,
        x_padding=0.35,
    )


def plot_exactness_gap_by_burst_width(
    burst_width: pd.DataFrame,
    output_dir: Path,
    *,
    target_window: int,
) -> None:
    """Plot exactness failure gap by burst width."""

    data = filter_target_window(burst_width, target_window=target_window)
    data = make_exactness_gap(data)

    y_limits = padded_limits(
        data["structural_exactness_gap"],
        lower_floor=-0.004,
        upper_ceiling=0.13,
        relative_padding=0.14,
        absolute_padding=0.006,
    )

    plot_lines(
        data,
        output_dir,
        x_column="burst_width",
        metric_column="structural_exactness_gap",
        title=rf"Structural exactness gap by burst width at $W={target_window}$",
        xlabel=r"Burst width ($b$)",
        ylabel=r"Failure rate ($1-\rho_{\mathrm{exact}}$)",
        output_stem=f"fig_exp4_exactness_gap_by_burst_width_w_{target_window}",
        y_limits=y_limits,
        wide=False,
        x_padding=0.35,
    )


def plot_semantic_survival_by_burst_width(
    burst_width: pd.DataFrame,
    output_dir: Path,
    *,
    target_window: int,
) -> None:
    """Plot semantic survival by burst width with a zoomed y-axis."""

    data = filter_target_window(burst_width, target_window=target_window)

    y_limits = padded_limits(
        data["semantic_survival_rate"],
        lower_floor=-0.002,
        upper_ceiling=0.04,
        relative_padding=0.16,
        absolute_padding=0.003,
    )

    plot_lines(
        data,
        output_dir,
        x_column="burst_width",
        metric_column="semantic_survival_rate",
        title=rf"Semantic survival by burst width at $W={target_window}$",
        xlabel=r"Burst width ($b$)",
        ylabel=r"Semantic survival rate",
        output_stem=f"fig_exp4_semantic_survival_by_burst_width_w_{target_window}",
        y_limits=y_limits,
        wide=False,
        x_padding=0.35,
    )


def plot_propagation_depth_by_burst_width(
    burst_width: pd.DataFrame,
    output_dir: Path,
    *,
    target_window: int,
) -> None:
    """Plot mean propagation depth by burst width."""

    data = filter_target_window(burst_width, target_window=target_window)

    y_limits = padded_limits(
        data["mean_propagation_depth"],
        lower_floor=-0.08,
        relative_padding=0.14,
        absolute_padding=0.12,
    )

    plot_lines(
        data,
        output_dir,
        x_column="burst_width",
        metric_column="mean_propagation_depth",
        title=rf"Propagation depth by burst width at $W={target_window}$",
        xlabel=r"Burst width ($b$)",
        ylabel=r"Mean propagation depth",
        output_stem=f"fig_exp4_propagation_depth_by_burst_width_w_{target_window}",
        y_limits=y_limits,
        wide=False,
        x_padding=0.35,
    )


def plot_mean_abs_error_by_burst_width(
    burst_width: pd.DataFrame,
    output_dir: Path,
    *,
    target_window: int,
) -> None:
    """Plot mean absolute error by burst width with log y-scale."""

    data = filter_target_window(burst_width, target_window=target_window)

    positive_values = data["mean_abs_error"][data["mean_abs_error"] > 0]
    lower = max(float(positive_values.min()) * 0.75, 1e-6)
    upper = float(positive_values.max()) * 1.35

    plot_lines(
        data,
        output_dir,
        x_column="burst_width",
        metric_column="mean_abs_error",
        title=rf"Mean absolute error by burst width at $W={target_window}$",
        xlabel=r"Burst width ($b$)",
        ylabel=r"Mean absolute error",
        output_stem=f"fig_exp4_mean_abs_error_by_burst_width_w_{target_window}",
        y_limits=(lower, upper),
        yscale="log",
        wide=False,
        x_padding=0.35,
    )


def plot_repair_cost_by_burst_width(
    burst_width: pd.DataFrame,
    output_dir: Path,
    *,
    target_window: int,
) -> None:
    """Plot mean repair cost by burst width with symlog y-scale."""

    data = filter_target_window(burst_width, target_window=target_window)
    y_max = float(np.nanmax(data["mean_normalization_steps"]))

    plot_lines(
        data,
        output_dir,
        x_column="burst_width",
        metric_column="mean_normalization_steps",
        title=rf"Repair cost by burst width at $W={target_window}$",
        xlabel=r"Burst width ($b$)",
        ylabel=r"Mean repair operations",
        output_stem=f"fig_exp4_repair_cost_by_burst_width_w_{target_window}",
        y_limits=(-0.03, max(y_max * 1.18, 1.0)),
        yscale="symlog",
        wide=False,
        x_padding=0.35,
    )


# =============================================================================
# Burst sensitivity figures across W
# =============================================================================

def plot_detectability_gain(
    burst_width: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot detectability gain from burst width 2 to burst width 5."""

    gain = make_burst_gain_dataframe(
        burst_width,
        metric_column="structural_detectability_rate",
        output_column="detectability_gain",
        low_burst=2,
        high_burst=5,
    )

    y_limits = padded_limits(
        gain["detectability_gain"],
        lower_floor=-0.02,
        upper_ceiling=0.25,
        relative_padding=0.12,
        absolute_padding=0.02,
    )

    plot_lines(
        gain,
        output_dir,
        x_column="W",
        metric_column="detectability_gain",
        title=r"Detectability gain from $b=2$ to $b=5$",
        xlabel=r"Window width ($W$)",
        ylabel=r"$\Delta$ detectability",
        output_stem="fig_exp4_detectability_gain_b5_minus_b2_vs_w",
        y_limits=y_limits,
        wide=False,
        x_padding=0.25,
    )


def plot_propagation_gain(
    burst_width: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot propagation-depth gain from burst width 2 to burst width 5."""

    gain = make_burst_gain_dataframe(
        burst_width,
        metric_column="mean_propagation_depth",
        output_column="propagation_gain",
        low_burst=2,
        high_burst=5,
    )

    y_limits = padded_limits(
        gain["propagation_gain"],
        lower_floor=-0.08,
        relative_padding=0.12,
        absolute_padding=0.10,
    )

    plot_lines(
        gain,
        output_dir,
        x_column="W",
        metric_column="propagation_gain",
        title=r"Propagation-depth gain from $b=2$ to $b=5$",
        xlabel=r"Window width ($W$)",
        ylabel=r"$\Delta$ mean propagation depth",
        output_stem="fig_exp4_propagation_gain_b5_minus_b2_vs_w",
        y_limits=y_limits,
        wide=False,
        x_padding=0.25,
    )


# =============================================================================
# Summary-level severity and tail figures
# =============================================================================

def plot_mean_log_error_magnitude(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot mean log error magnitude as excess over Binary.

    The raw curves share the same increasing W trend and overlap visually.
    Subtracting Binary at each W keeps the comparison in the original metric
    units while making the between-system differences readable.
    """

    if "mean_log_error_magnitude" not in summary.columns:
        print("[skip] Column not found: mean_log_error_magnitude")
        return

    delta_column = "mean_log_error_magnitude_delta_vs_binary"
    data = make_metric_delta_against_baseline(
        summary,
        metric_column="mean_log_error_magnitude",
        output_column=delta_column,
        baseline_system="binary",
    )

    y_limits = padded_limits(
        data[delta_column],
        relative_padding=0.16,
        absolute_padding=0.035,
    )

    plot_lines(
        data,
        output_dir,
        x_column="W",
        metric_column=delta_column,
        title=r"Mean log error magnitude relative to Binary",
        xlabel=r"Window width ($W$)",
        ylabel=r"Excess mean log error magnitude",
        output_stem="fig_exp4_mean_log_error_magnitude_vs_w",
        y_limits=y_limits,
        wide=False,
        x_padding=0.25,
        zero_line=True,
    )


def plot_p99_propagation_depth(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot p99 propagation depth by window width."""

    if "p99_propagation_depth" not in summary.columns:
        print("[skip] Column not found: p99_propagation_depth")
        return

    y_limits = padded_limits(
        summary["p99_propagation_depth"],
        lower_floor=-0.5,
        relative_padding=0.12,
        absolute_padding=0.50,
    )

    plot_lines(
        summary,
        output_dir,
        x_column="W",
        metric_column="p99_propagation_depth",
        title=r"Tail propagation depth",
        xlabel=r"Window width ($W$)",
        ylabel=r"$p_{99}$ propagation depth",
        output_stem="fig_exp4_p99_propagation_depth_vs_w",
        y_limits=y_limits,
        wide=False,
        x_padding=0.25,
    )


def plot_p99_repair_steps(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot p99 repair steps by window width on a zero-safe log scale.

    Values contain zeros for Binary, small integer tails for several systems,
    and a much larger Signed NAF tail. A direct symlog axis leaves too much
    unused space. Plotting log1p(value) but labelling ticks in original repair
    operations keeps zero visible and uses the panel area more evenly.
    """

    if "p99_normalization_steps" not in summary.columns:
        print("[skip] Column not found: p99_normalization_steps")
        return

    transformed_column = "log1p_p99_normalization_steps"
    data = make_log1p_metric(
        summary,
        metric_column="p99_normalization_steps",
        output_column=transformed_column,
    )

    y_max = float(np.nanmax(summary["p99_normalization_steps"]))
    transformed_y_max = float(np.log1p(y_max))

    fig, ax = create_square_panel()
    x_values = get_window_widths(data)

    for system in SYSTEM_ORDER:
        style = SYSTEM_STYLES[system]
        subset = get_system_subset(data, system, "W")

        if subset.empty:
            continue

        ax.plot(
            subset["W"],
            subset[transformed_column],
            color=style["color"],
            marker=style["marker"],
            markersize=5.8,
            markeredgecolor="white",
            markeredgewidth=0.80,
            linewidth=1.90,
            alpha=0.94,
        )

    ax.set_title(r"Tail repair cost", pad=7)
    ax.set_xlabel(r"Window width ($W$)")
    ax.set_ylabel(r"$p_{99}$ repair operations")
    ax.set_xticks(x_values)
    ax.set_xlim(min(x_values), max(x_values) + 0.25)
    ax.set_ylim(-0.04, transformed_y_max + 0.18)

    set_log1p_operation_ticks(ax, max_value=y_max)
    add_scale_note(ax, r"log1p y-scale")
    style_axis(ax, grid_axis="y")

    save_panel_figure(fig, output_dir, "fig_exp4_p99_repair_steps_vs_w")
    plt.close(fig)


# =============================================================================
# Legend strips
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
    """Export a horizontal legend strip."""

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
        "fig_exp4_legend_strip",
        extra_artists=(legend,),
    )
    plt.close(fig)


def plot_vertical_legend_strip(output_dir: Path) -> None:
    """Export a vertical legend strip."""

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
        "fig_exp4_legend_strip_vertical",
        extra_artists=(legend,),
    )
    plt.close(fig)


# =============================================================================
# Main entry point
# =============================================================================

def generate_all_figures(
    summary_csv: Path,
    burst_width_csv: Path,
    output_dir: Path,
    target_window: int,
) -> None:
    """Generate all Experiment 4 paper figures."""

    configure_matplotlib_style()

    summary = load_summary_data(
        summary_csv=summary_csv,
        output_dir=output_dir,
    )
    burst_width = load_burst_width_data(burst_width_csv)

    print(f"[info] Loaded summary: {summary_csv}")
    print(f"[info] Loaded burst-width summary: {burst_width_csv}")
    print(f"[info] Output directory: {output_dir}")
    print(f"[info] Burst-width plot window: W={target_window}")
    print("[info] Generating Experiment 4 figures...")

    # Main burst-width response at W = target_window.
    plot_detectability_by_burst_width(
        burst_width,
        output_dir,
        target_window=target_window,
    )
    plot_exactness_gap_by_burst_width(
        burst_width,
        output_dir,
        target_window=target_window,
    )
    plot_semantic_survival_by_burst_width(
        burst_width,
        output_dir,
        target_window=target_window,
    )
    plot_propagation_depth_by_burst_width(
        burst_width,
        output_dir,
        target_window=target_window,
    )
    plot_mean_abs_error_by_burst_width(
        burst_width,
        output_dir,
        target_window=target_window,
    )
    plot_repair_cost_by_burst_width(
        burst_width,
        output_dir,
        target_window=target_window,
    )

    # Derived burst-sensitivity summaries across W.
    plot_detectability_gain(burst_width, output_dir)
    plot_propagation_gain(burst_width, output_dir)

    # Summary-level severity and tail behavior.
    plot_mean_log_error_magnitude(summary, output_dir)
    plot_p99_propagation_depth(summary, output_dir)
    plot_p99_repair_steps(summary, output_dir)

    # Shared legends.
    plot_legend_strip(output_dir)
    plot_vertical_legend_strip(output_dir)

    print("[done] Experiment 4 figures generated.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create clean paper-style figures for Experiment 4."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiment_04" / "summary.csv",
        help="Path to Experiment 4 summary.csv.",
    )
    parser.add_argument(
        "--burst-width-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiment_04" / "burst_width_summary.csv",
        help="Path to Experiment 4 burst_width_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiment_04" / "paper_figures",
        help="Directory where paper figures will be saved.",
    )
    parser.add_argument(
        "--target-w",
        type=int,
        default=24,
        help="Window width used for burst-width conditioned figures.",
    )

    args = parser.parse_args()

    generate_all_figures(
        summary_csv=args.summary_csv,
        burst_width_csv=args.burst_width_csv,
        output_dir=args.output_dir,
        target_window=args.target_w,
    )


if __name__ == "__main__":
    main()