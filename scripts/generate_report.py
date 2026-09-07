"""Build the assignment report in Markdown from the generated benchmark files.

This reads ``benchmark_results/<dataset>_n<size>_<color_mode>/`` and writes
``report/``. It imports nothing from the pipeline: the benchmark communicates
with the report through files, exactly as the Assignment 1 report did. That
decoupling is what lets the prose be re-rendered without re-running a
thirteen minute benchmark, and it means the report can only describe results
that were actually produced.

It also stays outside the installed package. The package ships to PyPI for
anyone to use; a course report carrying a student name, a course number and
an instructor is not library code.

Run directories encode three things - dataset, subset size, colour mode - and
every comparison in the report holds two of them fixed and varies the third.
Reading across runs that differ in more than one is how a confounded
comparison gets written up as a finding.

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

# Measured over the built subsets with scripts/_subset.py:describe_shapes.
# Recorded here because data/ is not committed, so the report has to be
# rebuildable without it.
DATASETS = {
    "animals10": {
        "title": "Animals-10",
        "source": "`alessiocorrado99/animals10` (Kaggle)",
        "native": "mostly 4:3 photographs, median aspect ratio 1.37",
        "padding": "27.0% mean, 40.0% at the 90th percentile, 76.3% worst",
        "note": "Only 7.3% of the images are square, so preserving aspect "
                "ratio costs more than a quarter of the 64x64 canvas.",
    },
    "intel": {
        "title": "Intel Image Classification",
        "source": "`puneet6060/intel-image-classification` (Kaggle)",
        "native": "150x150, square (99.6% exactly)",
        "padding": "0.1% mean, 0.0% at the 90th percentile",
        "note": "Larger than the 64x64 internal size and square, so every "
                "image downscales into the canvas with no padding.",
    },
}

# The run the report leads with.
HEADLINE = "animals10_n500_rgb"


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

Adding the second dataset taught me something about my own reasoning rather than about
the models. Intel Image Classification is square, so nothing is lost to the padding that
costs Animals-10 twenty-seven percent of its canvas, and the Intel scores came back far
higher across the board. I read that as the padding being expensive, which is what I had
gone looking for. It is not what the numbers say. Intel has six classes where Animals-10
has ten, so chance is 0.167 rather than 0.100, and once both are expressed as a multiple
of chance the two datasets are close and the CNN is very slightly better on the padded
one. My comparison had two variables moving and I had assigned the whole difference to
the one I was interested in.

The same thing happened with colour. On Animals-10 the Decision Tree scored slightly
worse in RGB than in grayscale, and I had a tidy explanation ready about a single tree
overfitting the extra channels with no ensemble to average the mistake away. On Intel the
same model gained twenty-nine percent from colour. The explanation was not wrong so much
as not general: Intel's classes separate on colour directly, blue sea and white glacier
and green forest, where one threshold on one channel carries real information, while
brown animals photographed on green grass give a lone tree several thousand mostly noisy
columns. Two datasets turned a plausible story into a specific claim, and I would not
have caught it with one.

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


# --------------------------------------------------------------------------
# Reading the runs
# --------------------------------------------------------------------------

def parse_run_name(name: str) -> dict:
    """Split ``<dataset>_n<size>_<color_mode>`` into its three parts.

    Parsed from the right, because a dataset name can itself end in digits -
    "animals10" would otherwise be mistaken for a size.
    """
    parts = name.split("_")
    if len(parts) < 3 or not parts[-2].startswith("n"):
        return None
    try:
        size = int(parts[-2][1:])
    except ValueError:
        return None
    return {"dataset": "_".join(parts[:-2]), "size": size, "color_mode": parts[-1]}


def load_runs() -> dict:
    """Every run on disk, keyed by directory name."""
    runs = {}
    if not RESULTS_DIR.is_dir():
        return runs
    for directory in sorted(RESULTS_DIR.iterdir()):
        summary_path = directory / "benchmark_summary.csv"
        if not directory.is_dir() or not summary_path.is_file():
            continue
        parsed = parse_run_name(directory.name)
        if parsed is None:
            continue
        configuration = json.loads((directory / "run_configuration.json").read_text())
        runs[directory.name] = {
            "name": directory.name,
            **parsed,
            "summary": pd.read_csv(summary_path),
            "metrics": json.loads((directory / "benchmark_metrics.json").read_text()),
            "configuration": configuration,
            "classes": configuration["class_names"],
        }
    return runs


def markdown_table(frame) -> str:
    """Render a DataFrame as a Markdown table.

    Hand-rolled rather than using ``DataFrame.to_markdown``, which needs the
    optional `tabulate` package. The report should rebuild from a checkout
    with only the package's own dependencies installed.
    """
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "---|" * len(columns)]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def figure(run: dict, name: str) -> str:
    return f"![{name}](../benchmark_results/{run['name']}/{name}.png)"


def macro_f1(run: dict) -> pd.Series:
    return run["summary"].set_index("Model")["Macro F1"]


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

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


def section_objective(runs: dict) -> list:
    datasets = sorted({run["dataset"] for run in runs.values()})
    return [
        "## 1. Objective",
        "",
        "This report compares six image classification methods - four classical "
        "machine-learning models and two neural networks - through a single "
        f"public function in an installable package, `{PACKAGE_NAME}`.",
        "",
        f"Every model in a given run sees one stratified split, one set of "
        f"metrics and one measurement of cost. Across runs, exactly one thing "
        f"changes at a time: the size of the training set, the colour mode, or "
        f"the dataset. There are {len(runs)} runs over "
        f"{len(datasets)} datasets.",
        "",
        "```python",
        f"from {IMPORT_NAME} import benchmark_image_classification",
        "",
        "results = benchmark_image_classification(",
        '    dataset="./data/animals10_n500/labels.csv",',
        '    dataset_type="csv",',
        '    target_labels="class_name",',
        '    color_mode="rgb",',
        ")",
        "```",
        "",
    ]


def section_datasets(runs: dict) -> list:
    lines = ["## 2. Datasets", ""]
    for key in sorted({run["dataset"] for run in runs.values()}):
        info = DATASETS.get(key, {})
        example = next(r for r in runs.values() if r["dataset"] == key)
        lines += [
            f"### {info.get('title', key)}",
            "",
            f"- **Source:** {info.get('source', key)}",
            f"- **Classes ({len(example['classes'])}):** "
            f"{', '.join(example['classes'])}",
            f"- **Native size:** {info.get('native', 'varies')}",
            f"- **Padding at 64x64:** {info.get('padding', 'unmeasured')}",
            "",
            info.get("note", ""),
            "",
        ]
    lines += [
        "**Sampling.** Per class, the file list is sorted, shuffled with a seed "
        "derived from the class name, and the first N images that decode "
        "successfully are taken. Because the seed never depends on N, a smaller "
        "subset is a byte-identical prefix of a larger one, which is what makes "
        "two sizes comparable as a learning curve.",
        "",
        "**Standardisation.** Every image is fitted into 64x64 by scaling the "
        "longest side and padding the remainder with zeros, then normalised to "
        "[0, 1]. Aspect ratio is preserved rather than stretched.",
        "",
    ]
    return lines


def section_method() -> list:
    return [
        "## 3. Method",
        "",
        "**One split, reused.** The split produces indices rather than data. The "
        "classical models consume flattened feature vectors and the CNN consumes "
        "image tensors; both are sliced with the same index array, so the two "
        "representations cannot disagree about which images are in which half.",
        "",
        "**Scaling without leakage.** Logistic Regression, the SVM and the fully "
        "connected network are wrapped in a pipeline with a standard scaler, so "
        "it is fitted on training rows only. Fitting it before the split does not "
        "fail or warn - it simply lets the test set's statistics shape the "
        "training transformation, and every scaled model then scores slightly "
        "too high.",
        "",
        "**Early stopping without leakage.** The neural models hold out 15% of "
        "the training half to decide when to stop, because choosing when to stop "
        "is a decision informed by data and cannot use the test set. The "
        "consequence is that the neural models train on about 68% of all images "
        "where the classical models get the full 80%.",
        "",
        "**Ranking.** By macro F1, ties broken by lower inference time. Macro F1 "
        "rather than accuracy because it weights every class equally: a model "
        "that ignores a small class can still post high accuracy.",
        "",
    ]


def section_run(run: dict, index: int) -> list:
    """The per-dataset presentation section 11 asks for."""
    summary = run["summary"]
    best_row = summary.iloc[0]
    classes = run["classes"]
    chance = 1 / len(classes)
    split = run["configuration"]["split"]
    metrics = run["metrics"]
    best_key = min((k for k, v in metrics.items() if v["succeeded"]),
                   key=lambda k: -metrics[k]["macro_f1"])

    table = summary.drop(columns=["Status"]).copy()
    for column in table.columns[1:]:
        decimals = 1 if "Training" in column else 3
        table[column] = table[column].map(lambda v: f"{v:.{decimals}f}")

    lines = [
        f"## 4.{index} {DATASETS.get(run['dataset'], {}).get('title', run['dataset'])}"
        f" - {run['color_mode']}, {run['size']} per class",
        "",
        "```python",
        "results = benchmark_image_classification(",
        f'    dataset="./data/{run["dataset"]}_n{run["size"]}/labels.csv",',
        '    dataset_type="csv",',
        '    target_labels="class_name",',
        f'    color_mode="{run["color_mode"]}",',
        ")",
        "```",
        "",
        f"{split['training_samples'] + split['testing_samples']:,} images, "
        f"{len(classes)} classes, split "
        f"{split['training_samples']:,} training / {split['testing_samples']:,} "
        f"testing at seed {split['random_seed']}.",
        "",
        figure(run, "class_distribution"),
        "",
        markdown_table(table),
        "",
        f"**Best: {best_row['Model']}**, macro F1 {best_row['Macro F1']:.3f} "
        f"({best_row['Macro F1'] / chance:.1f}x the {chance:.3f} chance level for "
        f"{len(classes)} classes).",
        "",
        figure(run, "model_comparison"),
        "",
        figure(run, f"confusion_matrices/{best_key}"),
        "",
        figure(run, "prediction_examples"),
        "",
    ]

    confusions = top_confusions(metrics[best_key]["confusion_matrix"], classes)
    if confusions:
        lines += ["**Most frequent confusions.**", "",
                  "| True class | Predicted as | Images |", "|---|---|---|"]
        for true_name, predicted_name, count in confusions:
            lines.append(f"| {true_name} | {predicted_name} | {count} |")
        lines.append("")
    return lines


def top_confusions(matrix, classes, limit=5) -> list:
    pairs = [
        (classes[row], classes[column], matrix[row][column])
        for row in range(len(classes))
        for column in range(len(classes))
        if row != column and matrix[row][column] > 0
    ]
    return sorted(pairs, key=lambda item: -item[2])[:limit]


def section_size_experiment(runs: dict) -> list:
    """Same dataset, same colour mode, different subset size."""
    groups = {}
    for run in runs.values():
        groups.setdefault((run["dataset"], run["color_mode"]), []).append(run)
    groups = {k: sorted(v, key=lambda r: r["size"])
              for k, v in groups.items() if len(v) > 1}
    if not groups:
        return []

    lines = [
        "## 5. Experiment: training set size",
        "",
        "Same dataset, same colour mode, nested subsets - the smaller set is a "
        "strict prefix of the larger, so this is a learning curve rather than "
        "two unrelated samples.",
        "",
    ]
    for (dataset, mode), members in groups.items():
        sizes = [f"{r['size']}/class" for r in members]
        scores = {r["size"]: macro_f1(r) for r in members}
        models = list(scores[members[-1]["size"]].sort_values(ascending=False).index)

        lines += [
            f"**{DATASETS.get(dataset, {}).get('title', dataset)}, {mode}**",
            "",
            "| Model | " + " | ".join(sizes) + " | change |",
            "|---" * (len(sizes) + 2) + "|",
        ]
        for model in models:
            first = scores[members[0]["size"]].get(model)
            last = scores[members[-1]["size"]].get(model)
            cells = " | ".join(f"{scores[r['size']].get(model):.3f}" for r in members)
            change = f"{(last - first) / first * 100:+.0f}%" if first else "-"
            lines.append(f"| {model} | {cells} | {change} |")
        lines.append("")

        leader_first = scores[members[0]["size"]].idxmax()
        leader_last = scores[members[-1]["size"]].idxmax()
        if leader_first != leader_last:
            lines += [
                f"The ranking changes with size: {leader_first} leads at the "
                f"smaller subset, {leader_last} at the larger. A benchmark run "
                "at one size would have supported a conclusion the other "
                "contradicts.",
                "",
            ]
        else:
            gaps = [scores[r["size"]].nlargest(2) for r in members]
            lines += [
                f"{leader_last} leads at both sizes, but its margin over the "
                f"runner-up widens from {gaps[0].iloc[0] - gaps[0].iloc[1]:.3f} "
                f"to {gaps[-1].iloc[0] - gaps[-1].iloc[1]:.3f} macro F1. Models "
                "whose scores have flattened are near what raw pixels can give "
                "them; models still climbing would gain more from data than from "
                "a change of architecture.",
                "",
            ]
    return lines


def section_colour_experiment(runs: dict) -> list:
    """Same dataset, same size, different colour mode."""
    pairs = {}
    for run in runs.values():
        pairs.setdefault((run["dataset"], run["size"]), {})[run["color_mode"]] = run
    pairs = {k: v for k, v in pairs.items() if {"rgb", "grayscale"} <= set(v)}
    if not pairs:
        return []

    lines = [
        "## 6. Experiment: colour",
        "",
        "Identical images in both columns; only `color_mode` differs. Grayscale "
        "gives 4,096 features per image against RGB's 12,288.",
        "",
        "| Model | " + " | ".join(
            f"{DATASETS.get(d, {}).get('title', d)}" for d, _ in pairs) + " |",
        "|---" * (len(pairs) + 1) + "|",
    ]

    any_run = next(iter(next(iter(pairs.values())).values()))
    models = list(macro_f1(any_run).index)
    for model in models:
        cells = []
        for key in pairs:
            grayscale = macro_f1(pairs[key]["grayscale"]).get(model)
            rgb = macro_f1(pairs[key]["rgb"]).get(model)
            cells.append(f"{(rgb - grayscale) / grayscale * 100:+.0f}%")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")

    # Where the datasets disagree about a model, say so rather than
    # generalising from one of them.
    disagreements = []
    for model in models:
        deltas = []
        for key in pairs:
            grayscale = macro_f1(pairs[key]["grayscale"]).get(model)
            rgb = macro_f1(pairs[key]["rgb"]).get(model)
            deltas.append((rgb - grayscale) / grayscale)
        if min(deltas) < 0 < max(deltas):
            disagreements.append((model, deltas))

    for model, deltas in disagreements:
        names = [DATASETS.get(d, {}).get("title", d) for d, _ in pairs]
        described = ", ".join(f"{n} {d*100:+.0f}%" for n, d in zip(names, deltas))
        lines += [
            f"**{model} disagrees across datasets** ({described}), so the effect "
            "belongs to the data rather than to the model. Where a class is "
            "separable by colour directly, one threshold on one channel is "
            "informative; where it is not, the extra channels are mostly noise "
            "to a model with no ensemble to average them away.",
            "",
        ]

    # The cost side generalises even where the accuracy side does not.
    lines += ["**Cost.** RGB triples the feature count, and the SVM pays more "
              "than three times for it:", ""]
    lines += ["| Dataset | SVM training, grayscale | SVM training, RGB | factor |",
              "|---|---|---|---|"]
    for key in pairs:
        grayscale = pairs[key]["grayscale"]["summary"].set_index("Model")
        rgb = pairs[key]["rgb"]["summary"].set_index("Model")
        g = grayscale.loc["SVM", "Training Time (s)"]
        r = rgb.loc["SVM", "Training Time (s)"]
        title = DATASETS.get(key[0], {}).get("title", key[0])
        lines.append(f"| {title} | {g:.1f} s | {r:.1f} s | {r / g:.0f}x |")
    lines.append("")
    return lines


def section_dataset_experiment(runs: dict) -> list:
    """Different datasets at the same size and colour mode."""
    groups = {}
    for run in runs.values():
        groups.setdefault((run["size"], run["color_mode"]), []).append(run)
    comparable = [members for members in groups.values()
                  if len({r["dataset"] for r in members}) > 1]
    if not comparable:
        return []

    # Prefer the RGB pairing. Iteration order over the groups is not
    # meaningful, and picking arbitrarily once silently compared the two
    # grayscale runs while the surrounding prose described the RGB ones.
    comparable.sort(key=lambda members: members[0]["color_mode"] != "rgb")
    members = sorted(comparable[0], key=lambda r: r["dataset"])
    lines = [
        "## 7. Experiment: dataset and padding",
        "",
        "The two datasets differ in how much of the 64x64 canvas survives "
        "standardisation: Animals-10 loses 27% to padding, Intel essentially "
        "none. They also differ in class count, which has to be accounted for "
        "before the padding question can be asked at all.",
        "",
    ]

    def short(run):
        return DATASETS.get(run["dataset"], {}).get("title", run["dataset"])

    raw_headers = " | ".join(f"{short(r)} ({len(r['classes'])}c)" for r in members)
    lift_headers = " | ".join(f"{short(r)} x chance" for r in members)
    lines += [
        "| Model | " + raw_headers + " | " + lift_headers + " |",
        "|---" * (2 * len(members) + 1) + "|",
    ]

    models = list(macro_f1(members[0]).sort_values(ascending=False).index)
    lifts = {r["name"]: [] for r in members}
    for model in models:
        raw, normalised = [], []
        for run in members:
            score = macro_f1(run).get(model)
            chance = 1 / len(run["classes"])
            raw.append(f"{score:.3f}")
            normalised.append(f"{score / chance:.2f}")
            lifts[run["name"]].append(score / chance)
        lines.append(f"| {model} | " + " | ".join(raw) + " | " +
                     " | ".join(normalised) + " |")
    lines.append("")

    means = {name: sum(values) / len(values) for name, values in lifts.items()}
    ordered = sorted(means, key=means.get, reverse=True)
    described = ", ".join(
        f"{DATASETS.get(parse_run_name(n)['dataset'], {}).get('title', n)} "
        f"{means[n]:.2f}x" for n in ordered)

    lines += [
        f"Raw scores are much higher on the dataset with fewer classes, which is "
        f"what a lower chance level buys before any model does anything. "
        f"Expressed as a multiple of chance the two are close ({described}), and "
        "on that basis the padded dataset is not obviously the harder one.",
        "",
        "**This is evidence about padding, not a measurement of it.** Class "
        "count, task difficulty and dataset size all differ between these runs, "
        "so the honest conclusion is that padding is not the dominant "
        "limitation here, not that it is free. Isolating it would need the same "
        "dataset run with and without padding, or a subset of Animals-10 cut to "
        "the same class count.",
        "",
    ]
    return lines


def section_cost(runs: dict) -> list:
    run = runs[HEADLINE] if HEADLINE in runs else list(runs.values())[-1]
    summary = run["summary"].sort_values("Inference Time (ms/img)")
    fastest, slowest = summary.iloc[0], summary.iloc[-1]
    ratio = slowest["Inference Time (ms/img)"] / fastest["Inference Time (ms/img)"]

    rows = ["| Model | Training (s) | Inference (ms/image) | Macro F1 |",
            "|---|---|---|---|"]
    for _, row in run["summary"].iterrows():
        rows.append(f"| {row['Model']} | {row['Training Time (s)']:.1f} | "
                    f"{row['Inference Time (ms/img)']:.3f} | {row['Macro F1']:.3f} |")

    return [
        "## 8. Computational cost",
        "",
        f"From {run['name']}:",
        "",
        *rows,
        "",
        f"Inference cost spans a factor of {ratio:,.0f} between "
        f"{fastest['Model']} and {slowest['Model']}. Classifying a thousand "
        f"images would take {slowest['Model']} about "
        f"{slowest['Inference Time (ms/img)']:.1f} seconds against "
        f"{fastest['Model']}'s {fastest['Inference Time (ms/img)']:.3f} seconds.",
        "",
        "Accuracy alone would not surface this. A model chosen on macro F1 for a "
        "CPU-bound application could be unusable in practice, which is why the "
        "ranking breaks ties on the lower inference time.",
        "",
    ]


def section_reflection() -> list:
    return ["## 9. Reflection", "", REFLECTION_TEXT.strip(), ""]


def section_reproduction(runs: dict) -> list:
    configuration = list(runs.values())[0]["configuration"]
    return [
        "## 10. Reproducing these results",
        "",
        "```bash",
        f"pip install {PACKAGE_NAME}",
        "",
        "# rebuild the datasets (needs Kaggle API credentials)",
        "python scripts/download_animals10.py --per-class 500",
        "python scripts/download_animals10.py --per-class 100",
        "python scripts/download_intel.py --per-class 500",
        "",
        "# run every combination",
        "python scripts/run_benchmarks.py",
        "python scripts/run_benchmarks.py --color-mode grayscale",
        "",
        "# rebuild this report from the results on disk",
        "python scripts/generate_report.py",
        "```",
        "",
        f"Package version {configuration['package']['version']}, random seed "
        f"{configuration['random_seed']}, image size "
        f"{configuration['image_size'][0]}x{configuration['image_size'][1]}. "
        "Every model fixes its own random state; each run's full configuration "
        "is in its `run_configuration.json`.",
        "",
        "The images are not committed. Both datasets are third-party "
        "collections, so the download scripts rebuild the subsets instead, and "
        "the seeded sampling makes that rebuild exact.",
        "",
    ]


def build() -> str:
    runs = load_runs()
    if not runs:
        raise SystemExit(
            f"No results under {RESULTS_DIR}. Run scripts/run_benchmarks.py first."
        )

    # Headline first, then grouped by dataset with RGB leading each group and
    # larger subsets before smaller. Alphabetical order would put grayscale
    # ahead of rgb, which reverses the primary and supporting runs.
    mode_rank = {"rgb": 0, "grayscale": 1}
    ordered = sorted(
        runs.values(),
        key=lambda r: (r["name"] != HEADLINE, r["dataset"],
                       mode_rank.get(r["color_mode"], 2), -r["size"]),
    )

    lines = section_header()
    lines += section_objective(runs)
    lines += section_datasets(runs)
    lines += section_method()
    for index, run in enumerate(ordered, start=1):
        lines += section_run(run, index)
    lines += section_size_experiment(runs)
    lines += section_colour_experiment(runs)
    lines += section_dataset_experiment(runs)
    lines += section_cost(runs)
    lines += section_reflection()
    lines += section_reproduction(runs)
    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(build(), encoding="utf-8")
    print(f"Wrote {REPORT_MD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
