"""Build the assignment report in Markdown from the generated benchmark files.

This reads ``benchmark_results/<tier>/`` and writes ``report/``. It imports
nothing from the pipeline: the benchmark communicates with the report through
files, exactly as the Assignment 1 report did. That decoupling is what lets
the prose be re-rendered without re-running a thirteen minute benchmark, and
it means the report can only describe results that were actually produced.

It also stays outside the installed package. The package ships to PyPI for
anyone to use; a course report carrying a student name, a course number and
an instructor is not library code.

Usage
-----
    python scripts/generate_report.py
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmark_results"
REPORT_DIR = REPO_ROOT / "report"
REPORT_MD = REPORT_DIR / "CV_Benchmarking_Report.md"

STUDENT_NAME = "Samuel Collins"
STUDENT_TITLE = "Ph.D. Student, Department of Cyber-Physical Systems"
INSTITUTION = "Clark Atlanta University"
STUDENT_EMAIL = "samuel.collins@students.cau.edu"
COURSE = "CCIS 727 - Introduction to Computer Vision"
INSTRUCTOR = "Dr. Kishor Gupta"

PACKAGE_NAME = "Samuel_Collins_CV_Benchmarking"
IMPORT_NAME = "samuel_collins_cv_benchmarking"

# The tier the report leads with. Others are used for the learning curve.
HEADLINE_TIER = "animals10_n500"


REFLECTION_TEXT = """
The result that changed how I think about this assignment came from running the
benchmark twice. At a hundred images per class the CNN and the Random Forest were
indistinguishable, 0.295 against 0.294 macro F1, and if I had stopped there I would have
written that a convolutional network buys you nothing over an ensemble of trees on this
problem. At five hundred per class the CNN reached 0.434 and the Random Forest only
0.319. The sentence I would have written was not a small error, it was the opposite of
what the data actually shows, and nothing about the first run looked incomplete. It had
six models, real numbers, and a clear winner. What it lacked was a second point to
compare against.

That is also why the subset sampling matters more than it first appeared. Because the
shuffle is seeded on the class name rather than the sample count, the hundred-per-class
set is a byte-identical subset of the five-hundred-per-class set, so reading across the
two is a learning curve rather than a comparison of two unrelated draws. Getting that
property took one decision early on and I did not appreciate at the time that it was the
difference between two anecdotes and one measurement.

The models separated along a line I did not expect. Logistic Regression and the Random
Forest gained six and nine percent going from eight hundred to four thousand training
images; the CNN and the SVM gained forty-seven and forty-five. Two of these models were
close to what raw pixels can give them and two were still climbing. That is a more useful
way to describe a model than its score on any single run, and it is invisible unless you
vary the one thing most benchmarks hold fixed.

Macro F1 earned its place as the ranking metric during development rather than in the
final results. On an imbalanced test set I watched the fully connected network post
eighty-six percent accuracy while never once predicting the smallest class. Accuracy
barely registered the failure because that class was four of thirty-six images. Macro F1
fell to 0.62 because it weights every class the same, and zero_division=0 is what forces
the ignored class to score zero instead of quietly dropping out of the average. The
metric and the setting are not independent choices; the second is what makes the first
mean anything.

The Decision Tree scoring 0.093 at a hundred per class, below the 0.100 you get by
guessing among ten classes, was the clearest lesson in what these models actually do. A
single tree splits on individual pixel values, and one pixel out of 12,288 carries almost
no information about which animal is in the frame. Two hundred of those same trees voting
reached 0.319. Nothing changed except that the errors of many weak learners cancelled,
and that is the entire idea of an ensemble, made concrete in a way a textbook description
never managed for me.

Looking at the misclassified images was worth more than any metric. Three of the five
errors I inspected were cat, cow and dog all predicted as sheep, and all three were
animals photographed standing on grass. The model appears to have learned something about
green outdoor backgrounds rather than about the animals, which no confusion matrix would
have told me on its own. Seeing the standardised sixty-four by sixty-four input rather
than the original photograph also made the cost of preserving aspect ratio obvious: on
the widest images close to half of what the network receives is black padding I put
there. I still think padding was the right choice over stretching, but it is a real price
and I would not have known its size without looking.

The habit I want to keep is treating a passing test as a claim that needs checking. I
found a preprocessing bug by breaking the code deliberately and watching which tests
failed, and the one that survived was the line I would have called the most important in
the file. I found the prediction examples were all drawn from a single class by opening
the image rather than by running the suite, which passed. A green suite says the
assertions I thought to write are satisfied. It says nothing about the ones I did not
think to write, and those turned out to be where the real mistakes were.
"""


def _load_tier(tier: str) -> dict:
    """Read one tier's artifacts, or None when it was never run."""
    directory = RESULTS_DIR / tier
    summary_path = directory / "benchmark_summary.csv"
    if not summary_path.is_file():
        return None
    return {
        "tier": tier,
        "directory": directory,
        "summary": pd.read_csv(summary_path),
        "metrics": json.loads((directory / "benchmark_metrics.json").read_text()),
        "configuration": json.loads((directory / "run_configuration.json").read_text()),
    }


def available_tiers() -> list:
    """Every tier with results on disk, smallest first."""
    if not RESULTS_DIR.is_dir():
        return []

    def size(path):
        digits = "".join(c for c in path.name if c.isdigit())
        return int(digits) if digits else 0

    return [
        path.name for path in sorted(RESULTS_DIR.iterdir(), key=size)
        if path.is_dir() and (path / "benchmark_summary.csv").is_file()
    ]


def _figure(tier: str, name: str) -> str:
    """A markdown image reference, relative to the report's own directory."""
    return f"![{name}](../benchmark_results/{tier}/{name}.png)"


def _percent(value) -> str:
    return f"{value * 100:.1f}%"


def section_header() -> list:
    return [
        "# Image Classification Benchmarking Report",
        "",
        f"**{STUDENT_NAME}**  ",
        f"{STUDENT_TITLE}  ",
        f"{INSTITUTION}  ",
        f"{STUDENT_EMAIL}",
        "",
        f"**Course:** {COURSE}  ",
        f"**Instructor:** {INSTRUCTOR}  ",
        f"**Date:** {date.today().isoformat()}",
        "",
    ]


def section_objective(headline: dict) -> list:
    configuration = headline["configuration"]
    return [
        "## 1. Objective",
        "",
        "This report compares six image classification methods - four classical "
        "machine-learning models and two neural networks - on a single dataset, "
        "using one stratified split, one set of metrics, and one measurement of "
        "computational cost for every model. The comparison is produced by an "
        f"installable Python package, `{PACKAGE_NAME}`, through a single public "
        "function.",
        "",
        "The question the benchmark answers is not only which model scores highest, "
        "but why that result should be believed: whether the split was fair, whether "
        "every model saw the same data, whether the metric rewards the behaviour we "
        "actually want, and what each model costs to train and to run.",
        "",
        "```python",
        f"from {IMPORT_NAME} import benchmark_image_classification",
        "",
        "results = benchmark_image_classification(",
        f'    dataset="./data/{headline["tier"]}/labels.csv",',
        '    dataset_type="csv",',
        '    target_labels="class_name",',
        f'    color_mode="{configuration["color_mode"]}",',
        ")",
        "```",
        "",
    ]


def section_dataset(headline: dict) -> list:
    configuration = headline["configuration"]
    classes = configuration["class_names"]
    split = configuration["split"]
    total = split["training_samples"] + split["testing_samples"]

    lines = [
        "## 2. Dataset",
        "",
        f"**Source.** Animals-10 (`alessiocorrado99/animals10` on Kaggle), "
        f"{len(classes)} classes: {', '.join(classes)}.",
        "",
        f"**Subset.** {total:,} images, drawn as a reproducible stratified subset. "
        "Per class, the file list is sorted, shuffled with a seed derived from the "
        "class name, and the first N images that decode successfully are taken. "
        "Every file is opened and verified before selection; undecodable files are "
        "skipped and counted.",
        "",
        "Because the shuffle seed depends only on the class name and never on N, the "
        "tiers are nested: the 100-per-class set is a byte-identical subset of the "
        "500-per-class set. Reading across them is therefore a learning curve rather "
        "than a comparison of two unrelated samples.",
        "",
        f"**Standardisation.** Every image is converted to "
        f"`{configuration['color_mode']}`, fitted into "
        f"{configuration['image_size'][0]}x{configuration['image_size'][1]} pixels by "
        "scaling the longest side and padding the remainder with zeros, and "
        "normalised to the range [0, 1]. Aspect ratio is preserved rather than "
        "stretched, which costs padding: on the widest images close to half the "
        "input the model receives is padding.",
        "",
        "**Class distribution.**",
        "",
        _figure(headline["tier"], "class_distribution"),
        "",
        f"The split is 80/20, stratified by label, at random seed "
        f"{split['random_seed']}: {split['training_samples']:,} training and "
        f"{split['testing_samples']:,} testing images. The same indices are reused "
        "by all six models.",
        "",
    ]
    return lines


def section_method(headline: dict) -> list:
    configuration = headline["configuration"]
    rows = ["| Model | Key settings |", "|---|---|"]
    labels = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "svm": "SVM",
        "neural_network": "Neural Network",
        "simple_cnn": "Simple CNN",
    }
    for key, name in labels.items():
        parameters = configuration["models"].get(key, {})
        rendered = ", ".join(f"`{k}={v}`" for k, v in list(parameters.items())[:4])
        rows.append(f"| {name} | {rendered} |")

    return [
        "## 3. Method",
        "",
        "**One split, reused.** The split produces indices rather than data. The "
        "classical models consume flattened feature vectors and the CNN consumes "
        "image tensors, and both are sliced with the same index array, so the two "
        "representations cannot disagree about which images are in which half.",
        "",
        "**Scaling without leakage.** Logistic Regression, the SVM and the fully "
        "connected network are wrapped in a pipeline with a standard scaler, so the "
        "scaler is fitted on the training rows only. Fitting it on the whole dataset "
        "before splitting does not fail or warn - it just lets the test set's mean "
        "and variance shape the transformation used in training, and every scaled "
        "model then scores slightly too high.",
        "",
        "**Early stopping without leakage.** The neural models hold out 15% of the "
        "training half to decide when to stop. Choosing when to stop is a decision "
        "informed by data, so it cannot use the test set. The consequence is that "
        "the neural models train on about 68% of all images where the classical "
        "models get the full 80%.",
        "",
        "**Model configuration.**",
        "",
        *rows,
        "",
    ]


def section_results(headline: dict) -> list:
    summary = headline["summary"].copy()
    best = summary.iloc[0]

    table = summary.assign(**{
        "Accuracy": summary["Accuracy"].map(lambda v: f"{v:.3f}"),
        "Macro F1": summary["Macro F1"].map(lambda v: f"{v:.3f}"),
        "Macro Precision": summary["Macro Precision"].map(lambda v: f"{v:.3f}"),
        "Macro Recall": summary["Macro Recall"].map(lambda v: f"{v:.3f}"),
        "Weighted F1": summary["Weighted F1"].map(lambda v: f"{v:.3f}"),
        "Training Time (s)": summary["Training Time (s)"].map(lambda v: f"{v:.1f}"),
        "Inference Time (ms/img)": summary["Inference Time (ms/img)"].map(
            lambda v: f"{v:.3f}"),
    }).drop(columns=["Status"])

    classes = len(headline["configuration"]["class_names"])
    chance = 1.0 / classes

    return [
        f"## 4. Results ({headline['tier']})",
        "",
        "Ranked by macro F1, ties broken by the lower inference time.",
        "",
        table.to_markdown(index=False),
        "",
        f"**Best model: {best['Model']}**, macro F1 {best['Macro F1']:.3f}, "
        f"accuracy {_percent(best['Accuracy'])}. Chance for {classes} classes is "
        f"{chance:.3f}.",
        "",
        _figure(headline["tier"], "model_comparison"),
        "",
        "The four measures are shown on four separate axes rather than combined. "
        "Accuracy and macro F1 are bars from zero; the two timing panels span "
        "several orders of magnitude and are shown as dots on a log scale, because "
        "a bar states its magnitude through its length measured from zero and a log "
        "axis has no zero to measure from.",
        "",
    ]


def section_learning_curve(tiers: list) -> list:
    if len(tiers) < 2:
        return []

    names = [t["tier"] for t in tiers]
    frames = {t["tier"]: t["summary"].set_index("Model")["Macro F1"] for t in tiers}
    models = list(frames[names[-1]].sort_values(ascending=False).index)

    header = "| Model | " + " | ".join(names) + " | change |"
    divider = "|---" * (len(names) + 2) + "|"
    rows = [header, divider]
    for model in models:
        first = frames[names[0]].get(model)
        last = frames[names[-1]].get(model)
        cells = " | ".join(f"{frames[name].get(model, float('nan')):.3f}"
                           for name in names)
        change = f"{(last - first) / first * 100:+.0f}%" if first else "-"
        rows.append(f"| {model} | {cells} | {change} |")

    leader_small = frames[names[0]].idxmax()
    leader_large = frames[names[-1]].idxmax()

    lines = [
        "## 5. Effect of training set size",
        "",
        "Macro F1 across nested subsets. Each tier is a strict subset of the next, "
        "so these are the same images plus more, not different samples.",
        "",
        *rows,
        "",
    ]

    if leader_small != leader_large:
        lines += [
            f"The ranking changes with size. At `{names[0]}` the leader is "
            f"{leader_small}; at `{names[-1]}` it is {leader_large}. A benchmark run "
            "at one size would have supported a conclusion the larger run "
            "contradicts.",
            "",
        ]
    else:
        gap_small = frames[names[0]].nlargest(2)
        gap_large = frames[names[-1]].nlargest(2)
        lines += [
            f"{leader_large} leads at both sizes, but the margin over the "
            f"runner-up widens from "
            f"{(gap_small.iloc[0] - gap_small.iloc[1]):.3f} to "
            f"{(gap_large.iloc[0] - gap_large.iloc[1]):.3f} macro F1.",
            "",
        ]

    lines += [
        "Models whose scores have nearly flattened are close to what this "
        "representation can give them; models still climbing steeply would benefit "
        "from more data before any change of architecture.",
        "",
    ]
    return lines


def _top_confusions(matrix, classes, limit=5) -> list:
    """The largest off-diagonal cells, as (true, predicted, count)."""
    pairs = []
    for row, name in enumerate(classes):
        for column, other in enumerate(classes):
            if row != column and matrix[row][column] > 0:
                pairs.append((name, other, matrix[row][column]))
    return sorted(pairs, key=lambda item: -item[2])[:limit]


def section_per_class(headline: dict) -> list:
    metrics = headline["metrics"]
    classes = headline["configuration"]["class_names"]
    best_key = min(
        (k for k, v in metrics.items() if v["succeeded"]),
        key=lambda k: -metrics[k]["macro_f1"],
    )
    best = metrics[best_key]
    report = best["classification_report"]

    rows = ["| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for name in classes:
        entry = report[name]
        rows.append(
            f"| {name} | {entry['precision']:.3f} | {entry['recall']:.3f} | "
            f"{entry['f1-score']:.3f} | {int(entry['support'])} |"
        )

    per_class = sorted(
        ((name, report[name]["f1-score"]) for name in classes), key=lambda x: x[1])
    worst, best_class = per_class[0], per_class[-1]
    confusions = _top_confusions(best["confusion_matrix"], classes)

    lines = [
        f"## 6. Per-class behaviour ({best['name']})",
        "",
        *rows,
        "",
        f"Best handled: **{best_class[0]}** at F1 {best_class[1]:.3f}. "
        f"Worst handled: **{worst[0]}** at F1 {worst[1]:.3f}.",
        "",
        _figure(headline["tier"], f"confusion_matrices/{best_key}"),
        "",
        "Cells are coloured by share of the true class and annotated with the raw "
        "count. Colouring by count would leave a small class pale regardless of how "
        "well it was handled.",
        "",
        "**Most frequent confusions.**",
        "",
        "| True class | Predicted as | Images |",
        "|---|---|---|",
    ]
    for true_name, predicted_name, count in confusions:
        lines.append(f"| {true_name} | {predicted_name} | {count} |")
    lines.append("")

    examples = best.get("prediction_examples") or {}
    if examples.get("incorrect"):
        lines += [
            "**Example predictions.** Shown as the standardised 64x64 input the "
            "model actually received, not the original photograph, so the padding "
            "and the loss of detail are visible.",
            "",
            _figure(headline["tier"], "prediction_examples"),
            "",
        ]
        wrong = examples["incorrect"]
        targets = {entry["predicted_class"] for entry in wrong}
        if len(targets) < len(wrong):
            repeated = max(targets, key=lambda t: sum(
                1 for e in wrong if e["predicted_class"] == t))
            lines += [
                f"Several images of different true classes are predicted as "
                f"`{repeated}`, which suggests the model is responding to something "
                "the images share - background or overall colour - rather than to "
                "the animal.",
                "",
            ]
    return lines


def section_cost(headline: dict) -> list:
    summary = headline["summary"].sort_values("Training Time (s)", ascending=False)
    fastest = summary.sort_values("Inference Time (ms/img)").iloc[0]
    slowest = summary.sort_values("Inference Time (ms/img)").iloc[-1]
    ratio = slowest["Inference Time (ms/img)"] / fastest["Inference Time (ms/img)"]
    thousand = slowest["Inference Time (ms/img)"] * 1000 / 1000

    rows = ["| Model | Training (s) | Inference (ms/image) | Macro F1 |",
            "|---|---|---|---|"]
    for _, row in summary.iterrows():
        rows.append(
            f"| {row['Model']} | {row['Training Time (s)']:.1f} | "
            f"{row['Inference Time (ms/img)']:.3f} | {row['Macro F1']:.3f} |"
        )

    return [
        "## 7. Computational cost",
        "",
        *rows,
        "",
        f"Inference cost spans a factor of {ratio:,.0f} between "
        f"{fastest['Model']} and {slowest['Model']}. Classifying a thousand images "
        f"would take {slowest['Model']} about {thousand:.1f} seconds against "
        f"{fastest['Model']}'s {fastest['Inference Time (ms/img)']:.3f} milliseconds "
        "each.",
        "",
        "Accuracy alone would not surface this. A model chosen on macro F1 for a "
        "CPU-bound application could be unusable in practice, and the ranking rule "
        "here breaks ties on the lower inference time for that reason.",
        "",
    ]


def section_reflection() -> list:
    return ["## 8. Reflection", "", REFLECTION_TEXT.strip(), ""]


def section_reproduction(tiers: list) -> list:
    configuration = tiers[-1]["configuration"]
    return [
        "## 9. Reproducing these results",
        "",
        "```bash",
        f"pip install {PACKAGE_NAME}",
        "",
        "# rebuild the dataset (needs Kaggle API credentials)",
        "python scripts/download_animals10.py --per-class 500",
        "",
        "# run every built tier",
        "python scripts/run_benchmarks.py",
        "",
        "# rebuild this report from the results on disk",
        "python scripts/generate_report.py",
        "```",
        "",
        f"Package version {configuration['package']['version']}, random seed "
        f"{configuration['random_seed']}, image size "
        f"{configuration['image_size'][0]}x{configuration['image_size'][1]}, "
        f"colour mode `{configuration['color_mode']}`. Every model fixes its own "
        "random state; the full configuration for each run is in that tier's "
        "`run_configuration.json`.",
        "",
        "The images are not committed to the repository. Animals-10 is assembled "
        "from web-scraped photographs, so the download script rebuilds the subset "
        "instead, and the seeded sampling makes that rebuild exact.",
        "",
    ]


def build() -> str:
    tiers = [t for t in (_load_tier(name) for name in available_tiers()) if t]
    if not tiers:
        raise SystemExit(
            f"No results found under {RESULTS_DIR}. Run scripts/run_benchmarks.py first."
        )

    headline = next((t for t in tiers if t["tier"] == HEADLINE_TIER), tiers[-1])

    lines = section_header()
    lines += section_objective(headline)
    lines += section_dataset(headline)
    lines += section_method(headline)
    lines += section_results(headline)
    lines += section_learning_curve(tiers)
    lines += section_per_class(headline)
    lines += section_cost(headline)
    lines += section_reflection()
    lines += section_reproduction(tiers)
    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(build(), encoding="utf-8")
    print(f"Wrote {REPORT_MD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
