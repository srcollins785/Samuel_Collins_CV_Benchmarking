"""Training curves and the combined comparison plots.

Section 16 wants per-model convergence curves; section 24 wants eight plots
that show the traditional ML baselines and the CNN architectures *together*.
The design decisions worth recording:

*Color carries three groups, not eleven families.* Section 23 gives every
model a family - Traditional ML, Neural Baseline, Residual CNN, Efficient CNN
and so on - which is useful in a table and unreadable as color. Eleven
categorical hues cannot be told apart, and the scatter plots in section 24
are an all-pairs form where even the validated palette caps at three slots.
So color encodes the three groups the assignment's central question actually
contrasts: traditional ML, the neural baselines, and the deep CNNs. The finer
family stays on the axis label and in the table, where there is room to read
it. The three hues are slots 1-3 of the project's validated palette, checked
all-pairs: worst CVD Delta E 9.2, worst normal-vision Delta E 24.0.

*Aqua sits below 3:1 on this surface, so the relief rule applies.* Every bar
is named on its axis and every scatter point is directly labeled, and the
master CSV is the table view, so identity never rests on color alone.

*Missing is not zero.* A Logistic Regression has no parameter count, no
checkpoint size and no GPU memory figure in any sense comparable to a CNN's.
Section 23 says to write N/A and explain rather than invent a value, so the
plots for those metrics simply omit the models that lack them and say so in
the subtitle. Drawing a zero-height bar would be a fabricated measurement
that looks like a real one.

*Log scales where the range demands it, never a second axis.* Parameters run
from about four million to a hundred and thirty-four million and training time
spans two orders of magnitude, so those axes are logarithmic - which is one
axis, honestly labeled. A dual-axis chart would put two scales on one frame
and let their crossing point imply a relationship that is an artifact of the
scaling.
"""

from pathlib import Path

import numpy as np

from .visualization import (
    AXIS,
    FONT,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SERIES,
    SURFACE,
    _figure,
    _style_axes,
    plt,
)

# Slots 1-3 of the validated palette. Validated all-pairs in light mode, which
# is what the scatter plots require.
GROUP_COLORS = {
    "Traditional ML": "#2a78d6",
    "Baseline neural": "#eb6834",
    "Deep CNN": "#1baf7a",
}
GROUP_ORDER = ("Traditional ML", "Baseline neural", "Deep CNN")

TRADITIONAL_KEYS = {"logistic_regression", "decision_tree", "random_forest", "svm"}
BASELINE_KEYS = {"neural_network", "simple_cnn"}


def group_for(key: str) -> str:
    """Which of the three color groups a model key belongs to."""
    if key in TRADITIONAL_KEYS:
        return "Traditional ML"
    if key in BASELINE_KEYS:
        return "Baseline neural"
    return "Deep CNN"


def _annotate(axes, subtitle: str) -> None:
    """A caption under the axes, for the 'what was omitted' note."""
    if subtitle:
        axes.set_xlabel(axes.get_xlabel() + f"\n{subtitle}", fontsize=8,
                        color=INK_MUTED, labelpad=8)


def _legend(axes, groups) -> None:
    """A legend is always present once two groups are on screen."""
    from matplotlib.patches import Patch

    present = [g for g in GROUP_ORDER if g in groups]
    if len(present) < 2:
        return
    axes.legend(
        handles=[Patch(facecolor=GROUP_COLORS[g], label=g) for g in present],
        loc="lower right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)


# -- section 16: convergence -------------------------------------------------

def plot_training_curves(result, path: Path) -> Path:
    """Loss and accuracy against epoch, training against validation.

    Two panels rather than one: loss and accuracy are different measures on
    different scales, and putting them on one frame would need a second y-axis.
    The selected epoch is marked, because section 12 evaluates that epoch's
    weights and a reader should see which point in the curve the test metrics
    actually describe.
    """
    plt.rcParams["font.family"] = FONT
    records = (result.history or {}).get("per_epoch") or []
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        figure, axes = _figure(7.4, 3.2)
        axes.text(0.5, 0.5, "no training history", ha="center", va="center",
                  color=INK_MUTED, fontsize=10, transform=axes.transAxes)
        axes.axis("off")
        figure.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        plt.close(figure)
        return path

    epochs = [record["epoch"] for record in records]
    best_epoch = (result.history or {}).get("best_epoch")

    figure, panels = plt.subplots(1, 2, figsize=(10.2, 3.9))
    figure.patch.set_facecolor(SURFACE)

    series = [
        ("Loss", "train_loss", "validation_loss"),
        ("Accuracy", "train_accuracy", "validation_accuracy"),
    ]
    for axes, (label, train_key, validation_key) in zip(panels, series):
        _style_axes(axes, grid_axis="both")

        # A series the trainer did not record is omitted rather than drawn as
        # zeros. ultralytics logs no training accuracy for classification, so
        # YOLO's accuracy panel carries the validation curve alone - which is
        # a gap in the data, not a model that scored zero.
        if all(r.get(train_key) is not None for r in records):
            axes.plot(epochs, [r[train_key] for r in records], color=SERIES[0],
                      linewidth=2.0, label="Training", zorder=3)
        else:
            axes.plot([], [], color=SERIES[0], linewidth=2.0,
                      label="Training (not recorded)", alpha=0.35)
        axes.plot(epochs, [r.get(validation_key) for r in records],
                  color=SERIES[1], linewidth=2.0, label="Validation", zorder=3)

        if best_epoch:
            axes.axvline(best_epoch, color=INK_MUTED, linewidth=1.0,
                         linestyle=(0, (4, 3)), zorder=1)
            axes.annotate(
                f"selected\nepoch {best_epoch}", xy=(best_epoch, 0.02),
                xycoords=("data", "axes fraction"), fontsize=7.5,
                color=INK_MUTED, ha="left" if best_epoch < max(epochs) * 0.8 else "right",
                xytext=(4 if best_epoch < max(epochs) * 0.8 else -4, 0),
                textcoords="offset points")

        axes.set_xlabel("Epoch", fontsize=9, color=INK_SECONDARY)
        axes.set_ylabel(label, fontsize=9, color=INK_SECONDARY)
        axes.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    gap = (result.overfitting or {}).get("generalization_gap")
    detail = f"   generalization gap {gap:+.3f}" if gap is not None else ""
    figure.suptitle(f"{result.name} - convergence{detail}", fontsize=11,
                    color=INK, x=0.02, ha="left", y=1.02)
    figure.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return path


# -- section 24: the eight comparison plots ---------------------------------

def _bar_plot(rows, value_key, title, axis_label, path, subtitle="",
              log=False):
    """One horizontal bar per model, sorted, colored by group.

    Horizontal because sixteen model names do not fit as vertical tick labels
    without rotating them, and rotated labels are slower to read.

    Always sorted with the largest value at the top. For accuracy, F1 and
    throughput that puts the best model first, which is what a reader looks
    for; for parameters, size and training time it is a plain magnitude
    ordering with the longest bar at the top. One rule rather than a
    per-metric flag, because a flag here was already wrong once - it sorted
    ascending and then inverted the axis, which quietly put the *worst* model
    at the top of the accuracy chart.
    """
    plt.rcParams["font.family"] = FONT
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    present = [row for row in rows if row.get(value_key) is not None]
    if not present:
        return None
    # Descending, then invert_yaxis below places index 0 at the top.
    present.sort(key=lambda row: row[value_key], reverse=True)

    names = [row["name"] for row in present]
    values = [row[value_key] for row in present]
    colors = [GROUP_COLORS[row["group"]] for row in present]

    height = max(3.0, 0.34 * len(present) + 1.5)
    figure, axes = _figure(8.4, height)
    _style_axes(axes, grid_axis="x")

    positions = np.arange(len(present))
    axes.barh(positions, values, color=colors, height=0.62, zorder=3,
              edgecolor=SURFACE, linewidth=1.5)
    axes.set_yticks(positions)
    axes.set_yticklabels(names, fontsize=8.5, color=INK_SECONDARY)
    axes.invert_yaxis()

    if log:
        axes.set_xscale("log")

    # Direct labels on the bars: the relief rule for the aqua slot, and it
    # spares the reader reading values off a log axis.
    span = max(values) if values else 1
    for position, value in zip(positions, values):
        axes.annotate(
            _format_value(value), xy=(value, position), xytext=(4, 0),
            textcoords="offset points", va="center", fontsize=7.5,
            color=INK_SECONDARY)
    if not log:
        axes.set_xlim(0, span * 1.18)

    axes.set_xlabel(axis_label, fontsize=9, color=INK_SECONDARY)
    _annotate(axes, subtitle)
    axes.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
    _legend(axes, {row["group"] for row in present})

    figure.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return path


def _format_value(value) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _scatter_plot(rows, x_key, y_key, title, x_label, y_label, path,
                  subtitle="", log_x=True):
    """Accuracy against cost - the two plots that carry the trade-off.

    Every point is directly labeled. Sixteen points in three colors with
    names attached is readable; the same points relying on a legend to be
    identified would not be, and these two figures are the ones the
    deployment discussion actually argues from.
    """
    plt.rcParams["font.family"] = FONT
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    present = [row for row in rows
               if row.get(x_key) is not None and row.get(y_key) is not None]
    if not present:
        return None

    figure, axes = _figure(8.6, 5.8)
    _style_axes(axes, grid_axis="both")

    for row in present:
        axes.scatter(row[x_key], row[y_key], s=78,
                     color=GROUP_COLORS[row["group"]], zorder=3,
                     edgecolor=SURFACE, linewidth=1.6)

    if log_x:
        axes.set_xscale("log")

    # Nudge labels alternately above and below so neighbouring points do not
    # overwrite each other's text.
    ordered = sorted(present, key=lambda row: row[x_key])
    for index, row in enumerate(ordered):
        offset = 9 if index % 2 == 0 else -15
        axes.annotate(
            row["name"], xy=(row[x_key], row[y_key]),
            xytext=(0, offset), textcoords="offset points",
            fontsize=7.5, color=INK_SECONDARY, ha="center")

    axes.set_xlabel(x_label, fontsize=9, color=INK_SECONDARY)
    axes.set_ylabel(y_label, fontsize=9, color=INK_SECONDARY)
    _annotate(axes, subtitle)
    axes.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
    _legend(axes, {row["group"] for row in present})

    figure.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return path


def _omitted_note(rows, key: str) -> str:
    """Name the models a metric does not apply to, rather than zeroing them."""
    missing = [row["name"] for row in rows if row.get(key) is None]
    if not missing:
        return ""
    shown = ", ".join(missing[:6])
    more = f" and {len(missing) - 6} more" if len(missing) > 6 else ""
    return f"Not applicable for {shown}{more}; reported as N/A rather than zero."


def plot_all_comparisons(rows, output_dir) -> dict:
    """Every figure section 24 requires, from the combined row list.

    Parameters
    ----------
    rows
        Combined Part 1 and Part 2 rows, each carrying ``name``, ``group``
        and whichever metrics apply to it.
    output_dir
        The ``plots/`` directory.

    Returns
    -------
    dict
        Figure name to the path written, omitting any figure whose metric no
        model reported.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    specifications = [
        ("accuracy_comparison", _bar_plot, dict(
            value_key="accuracy", title="Test accuracy",
            axis_label="Accuracy on the shared test half")),
        ("f1_comparison", _bar_plot, dict(
            value_key="macro_f1", title="Macro F1",
            axis_label="Macro F1 (every class weighted equally)")),
        ("parameter_comparison", _bar_plot, dict(
            value_key="total_parameters", title="Model parameters",
            axis_label="Total parameters (log scale)", log=True,
            subtitle=None)),
        ("model_size", _bar_plot, dict(
            value_key="checkpoint_mb", title="Checkpoint size on disk",
            axis_label="Megabytes (log scale)", log=True, subtitle=None)),
        ("training_time", _bar_plot, dict(
            value_key="training_seconds", title="Training time",
            axis_label="Seconds (log scale)", log=True)),
        ("inference_speed", _bar_plot, dict(
            value_key="throughput", title="Inference throughput",
            axis_label="Images per second (batched)")),
    ]

    for name, function, options in specifications:
        options = dict(options)
        if options.get("subtitle") is None:
            options["subtitle"] = _omitted_note(rows, options["value_key"])
        path = function(rows, path=output_dir / f"{name}.png", **options)
        if path:
            written[name] = str(path)

    scatters = [
        ("accuracy_vs_parameters", dict(
            x_key="total_parameters", y_key="accuracy",
            title="Accuracy against parameter count",
            x_label="Total parameters (log scale)", y_label="Test accuracy")),
        ("accuracy_vs_latency", dict(
            x_key="latency_ms", y_key="accuracy",
            title="Accuracy against inference latency",
            x_label="Latency in milliseconds per image (log scale)",
            y_label="Test accuracy")),
    ]
    for name, options in scatters:
        options = dict(options)
        options["subtitle"] = _omitted_note(rows, options["x_key"])
        path = _scatter_plot(rows, path=output_dir / f"{name}.png", **options)
        if path:
            written[name] = str(path)

    return written
