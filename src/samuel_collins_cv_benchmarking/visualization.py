"""Comparison plots, confusion matrices and class distributions.

Three figures, and each one's form follows from the job its data does.

*Class distribution* is a magnitude comparison across classes that also has to
show how each class divides between training and testing. Stacked bars carry
both: the bar's length is the class total, the split within it is the two
halves. Two series, so a legend is always present.

*Model comparison* has to show accuracy, macro F1, training time and inference
time together. Those are four measures on wildly different scales, and putting
two of them on one pair of axes would be a dual-axis chart - the single worst
thing you can do to a reader, because the crossing point of the two series is
an artefact of the scales rather than a fact about the data. Four small
multiples instead, each with its own axis, models in the same order in every
panel so the eye can track one model across all four.

*Confusion matrices* encode magnitude, so they get one hue running light to
dark rather than a rainbow. Cells are coloured by the share of their true
class and annotated with the raw count: on an imbalanced set, colouring by
raw count would leave the whole row of a small class pale and unreadable
regardless of how well the model did on it.

Colours come from a validated palette. The categorical pair used here was
checked for colourblind separation rather than eyeballed - worst adjacent
CVD Delta E 24.7, normal-vision 33.6, both clear of their floors.

These render to PNG for a printed report, so they are drawn for the light
surface only.
"""

from pathlib import Path

import matplotlib

# Chosen before pyplot is imported: the benchmark runs headless, and the
# default interactive backend would fail or block without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# --- palette -------------------------------------------------------------
# Categorical slots 1 and 2, for the two halves of the split.
SERIES = ["#2a78d6", "#eb6834"]
# One hue, light to dark, for magnitude.
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
              "#0d366b"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUES = LinearSegmentedColormap.from_list("benchmark_blues", SEQUENTIAL)

# Real font family names only. "system-ui" and "-apple-system" are CSS
# keywords rather than families: matplotlib cannot resolve them, falls back
# silently, and warns once per text object - thousands of lines across a full
# run, which would bury anything worth reading.
FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

# A hairline of surface colour drawn around each bar, so stacked segments and
# neighbouring bars are separated by a visible gap rather than touching.
SEGMENT_GAP = 1.5


def _style_axes(axes, grid_axis="y"):
    """Recessive chrome: the data should be the darkest thing in the figure."""
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
        axes.spines[side].set_linewidth(0.8)
    axes.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=0.8)
    axes.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)


def _figure(width, height):
    figure, axes = plt.subplots(figsize=(width, height))
    figure.patch.set_facecolor(SURFACE)
    return figure, axes


def plot_class_distribution(split, path: Path) -> Path:
    """Images per class, split into the training and testing halves.

    The bar length is the class total, so the figure answers both "how big is
    each class" and "did the split preserve the proportions" at once.
    """
    plt.rcParams["font.family"] = FONT
    distributions = split.distributions()
    names = list(split.dataset.class_names)
    training = [distributions["training"][name] for name in names]
    testing = [distributions["testing"][name] for name in names]

    height = max(3.0, 0.45 * len(names) + 1.8)
    figure, axes = _figure(8.0, height)
    positions = np.arange(len(names))

    # Thin bars with room between them; a saturated fill belongs on a small
    # mark, not a heavy block.
    axes.barh(positions, training, height=0.55, color=SERIES[0], label="Training",
              edgecolor=SURFACE, linewidth=SEGMENT_GAP, zorder=2)
    axes.barh(positions, testing, left=training, height=0.55, color=SERIES[1],
              label="Testing", edgecolor=SURFACE, linewidth=SEGMENT_GAP, zorder=2)

    for position, (train_count, test_count) in enumerate(zip(training, testing)):
        total = train_count + test_count
        axes.text(total + max(training + testing) * 0.015, position, str(total),
                  va="center", fontsize=9, color=INK_SECONDARY)

    axes.set_yticks(positions)
    axes.set_yticklabels(names, fontsize=10, color=INK)
    axes.invert_yaxis()
    axes.set_xlabel("Images", fontsize=10, color=INK_SECONDARY)
    axes.set_title("Class distribution", fontsize=13, color=INK,
                   loc="left", pad=14, fontweight="medium")
    _style_axes(axes, grid_axis="x")
    # Above the plot rather than inside it - with ten classes any in-plot
    # corner is somewhere a bar might reach - and anchored right so it shares
    # that band with the left-aligned title instead of sitting under it.
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
                loc="lower right", bbox_to_anchor=(1, 1.01), ncols=2)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return path


# The four measures section 9 requires the comparison figure to cover.
# `lower_is_better` only decides the caption; every panel keeps its own axis.
PANELS = [
    ("accuracy", "Accuracy", False, False),
    ("macro_f1", "Macro F1", False, False),
    ("training_time_seconds", "Training time (s)", True, True),
    ("inference_time_ms_per_image", "Inference time (ms/image)", True, True),
]


def plot_model_comparison(results, path: Path) -> Path:
    """Four small multiples: accuracy, macro F1, training time, inference time.

    Deliberately four panels rather than one chart with two y-axes. A
    dual-axis chart invites the reader to compare the height of a bar against
    a line drawn on a different scale, and any crossing they see is a
    consequence of how the two axes were chosen.
    """
    plt.rcParams["font.family"] = FONT
    successful = [result for result in results if result.succeeded]
    if not successful:
        raise ValueError("No successful models to plot.")

    # Same order in every panel, so a reader can follow one model across all
    # four without re-reading the labels.
    names = [result.name for result in successful]
    positions = np.arange(len(names))

    figure, axes_grid = plt.subplots(2, 2, figsize=(12.0, 2.2 + 0.55 * len(names) * 2))
    figure.patch.set_facecolor(SURFACE)

    for axes, (field, title, lower_better, may_log) in zip(axes_grid.ravel(), PANELS):
        values = [getattr(result, field) or 0.0 for result in successful]

        # Timings routinely span three orders of magnitude - an RBF SVM can
        # take minutes where a Decision Tree takes milliseconds. On a linear
        # axis every model but the slowest becomes an invisible sliver, so a
        # wide spread switches to a log axis and says so in the label.
        spread = max(values) / min(v for v in values if v > 0) if any(values) else 1
        use_log = may_log and spread >= 50

        if use_log:
            # Dots rather than bars. A bar states its magnitude through its
            # length measured from zero, and a log axis has no zero to
            # measure from - the left edge lands wherever the axis happens to
            # start, so the length would encode nothing. A dot claims only its
            # position, which is exactly what a log scale can support.
            axes.set_xscale("log")
            axes.plot(values, positions, "o", color=SERIES[0], markersize=9,
                      markeredgecolor=SURFACE, markeredgewidth=1.5,
                      linestyle="none", zorder=3)
            lower, upper = min(values), max(values)
            axes.set_xlim(lower / 3, upper * 6)
            _style_axes(axes, grid_axis="y")
        else:
            axes.barh(positions, values, color=SERIES[0], height=0.62,
                      edgecolor=SURFACE, linewidth=SEGMENT_GAP, zorder=2)
            _style_axes(axes, grid_axis="x")
            axes.set_xlim(0, max(values) * 1.18 if max(values) else 1)

        for position, value in zip(positions, values):
            label = f"{value:.3f}" if value < 100 else f"{value:.0f}"
            # A static PNG has no tooltip, so every value is labelled: the
            # label is standing in for the table view a reader would otherwise
            # hover for.
            axes.text(value * 1.35 if use_log else value + max(values) * 0.02,
                      position, label, va="center", fontsize=8.5,
                      color=INK_SECONDARY)

        axes.set_yticks(positions)
        axes.set_yticklabels(names, fontsize=9, color=INK)
        axes.invert_yaxis()
        caption = title + (" - log scale" if use_log else "")
        if lower_better:
            caption += ", lower is better"
        axes.set_title(caption, fontsize=11, color=INK, loc="left",
                       pad=10, fontweight="medium")

    figure.suptitle("Model comparison", fontsize=14, color=INK, x=0.012,
                    ha="left", y=0.995, fontweight="medium")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    # Room between the two rows, so the lower titles clear the upper panels'
    # tick labels rather than sitting on top of them.
    figure.subplots_adjust(hspace=0.42, wspace=0.28)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return path


def plot_confusion_matrix(result, class_names, path: Path) -> Path:
    """One labelled matrix, coloured by share of the true class.

    Colour encodes the row-normalised value and the annotation carries the raw
    count. Colouring by raw count instead would wash out every row of a small
    class: on a 40/40/20 split the smaller class can never reach the same
    counts as the larger ones, so its row would read as uniformly pale no
    matter how well the model handled it.
    """
    plt.rcParams["font.family"] = FONT
    matrix = np.asarray(result.confusion_matrix, dtype=float)
    row_totals = matrix.sum(axis=1, keepdims=True)
    # A class with no test images would divide by zero; it also has nothing to
    # show, so it stays at zero.
    shares = np.divide(matrix, row_totals, out=np.zeros_like(matrix),
                       where=row_totals > 0)

    size = max(3.6, 0.62 * len(class_names) + 2.2)
    figure, axes = plt.subplots(figsize=(size + 1.4, size))
    figure.patch.set_facecolor(SURFACE)

    image = axes.imshow(shares, cmap=BLUES, vmin=0.0, vmax=1.0)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            # Ink flips on the darker cells so the count stays legible.
            colour = "#ffffff" if shares[row, column] > 0.55 else INK
            axes.text(column, row, f"{int(matrix[row, column])}",
                      ha="center", va="center", fontsize=10, color=colour)

    axes.set_xticks(range(len(class_names)))
    axes.set_yticks(range(len(class_names)))
    axes.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9, color=INK)
    axes.set_yticklabels(class_names, fontsize=9, color=INK)
    axes.set_xlabel("Predicted", fontsize=10, color=INK_SECONDARY)
    axes.set_ylabel("True", fontsize=10, color=INK_SECONDARY)
    axes.set_title(result.name, fontsize=12, color=INK, loc="left",
                   pad=12, fontweight="medium")

    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.tick_params(colors=INK_MUTED, length=0)
    # Hairline separators between cells, matching the gap used elsewhere.
    axes.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    axes.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
    axes.grid(which="minor", color=SURFACE, linewidth=SEGMENT_GAP)
    axes.tick_params(which="minor", length=0)

    bar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    bar.set_label("Share of true class", fontsize=9, color=INK_SECONDARY)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    bar.outline.set_visible(False)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return path


def save_all(results, split, output_dir: Path) -> dict:
    """Write every figure section 9 requires, returning what was written."""
    output_dir = Path(output_dir)
    matrices_dir = output_dir / "confusion_matrices"

    written = {
        "class_distribution": plot_class_distribution(
            split, output_dir / "class_distribution.png"),
        "model_comparison": plot_model_comparison(
            results, output_dir / "model_comparison.png"),
        "confusion_matrices": {},
    }

    for result in results:
        # A model that failed has no matrix to draw; its row still appears in
        # the summary table carrying the error.
        if not result.succeeded:
            continue
        written["confusion_matrices"][result.key] = plot_confusion_matrix(
            result, split.dataset.class_names, matrices_dir / f"{result.key}.png")

    return written
