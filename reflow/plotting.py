"""
Figures for a run: the variance-ratio curve and the three-stage accuracy bars.

Importing this module selects matplotlib's non-interactive Agg backend, so the
CLI renders to PNG on a headless machine. Import it before pyplot elsewhere.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)
from matplotlib.ticker import MaxNLocator  # noqa: E402

# Categorical slots, assigned in fixed order and never cycled.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")   # blue, orange, aqua
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#d9d8d4"


def _style_axes(ax):
    """Recessive grid and axes, so the data carries the figure."""
    ax.grid(True, color=GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def plot_variance_ratios(ratios_by_stage, save_path, title=None, key="output"):
    """
    Per-layer variance ratio, one line per stage.

    ``ratios_by_stage`` maps a stage label ("pruned", "pruned + reflow") to the
    dict :func:`reflow.signal.variance_ratios` returns. ``key`` picks which
    reduction to draw -- "output" is eta_l, the post-normalization ratio.

    The dashed line at 1.0 is the dense model: a curve sitting on it preserves
    the signal, one decaying toward zero with depth is signal collapse.
    """
    stages = list(ratios_by_stage)
    n_layers = len(ratios_by_stage[stages[0]][key])
    x = range(1, n_layers + 1)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axhline(1.0, ls="--", lw=1.2, color=INK_SECONDARY, alpha=0.55, zorder=1)
    ax.text(n_layers, 1.0, " dense", va="center", ha="left",
            fontsize=8, color=INK_SECONDARY)

    # Markers stop helping once layers get dense; past ~30 the line alone reads better.
    marker = "o" if n_layers <= 30 else None
    for i, stage in enumerate(stages):
        ax.plot(x, ratios_by_stage[stage][key], lw=2, marker=marker, ms=4,
                color=SERIES[i % len(SERIES)], label=stage, zorder=2 + i)

    ax.set_xlabel("Normalization layer (forward order)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(r"variance ratio  $\eta_\ell$", color=INK_SECONDARY, fontsize=10)
    if title:
        ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
    ax.set_xlim(1, n_layers)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # layer indices are whole numbers
    _style_axes(ax)
    legend = ax.legend(frameon=False, fontsize=9, loc="best")
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path


def plot_accuracy(accuracies, save_path, title=None, ylabel="Top-1 accuracy (%)"):
    """
    Bar chart of the run's three measurements, each labeled with its value.

    ``accuracies`` maps a stage label to a percentage.
    """
    labels = list(accuracies)
    values = [accuracies[k] for k in labels]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(range(len(labels)), values, width=0.6,
                  color=[SERIES[i % len(SERIES)] for i in range(len(labels))],
                  zorder=2)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color=INK_PRIMARY, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    if title:
        ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
    ax.set_ylim(0, max(100.0, max(values) * 1.15))
    _style_axes(ax)
    ax.grid(axis="x", visible=False)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.2f}",
                ha="center", va="bottom", fontsize=10, color=INK_PRIMARY)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path
