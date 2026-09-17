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

Run directories encode three things - dataset, subset size, color mode - and
every comparison in the report holds two of them fixed and varies the third.
Reading across runs that differ in more than one is how a confounded
comparison gets written up as a finding.

Usage
-----
    python scripts/generate_report.py
"""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report_part2  # noqa: E402

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

The same thing happened with color. On Animals-10 the Decision Tree scored slightly
worse in RGB than in grayscale, and I had a tidy explanation ready about a single tree
overfitting the extra channels with no ensemble to average the mistake away. On Intel the
same model gained twenty-nine percent from color. The explanation was not wrong so much
as not general: Intel's classes separate on color directly, blue sea and white glacier
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
reached 0.319. Nothing changed except that the errors of many weak learners canceled,
and that is the entire idea of an ensemble, made concrete in a way a textbook description
never managed for me.

Looking at the misclassified images was worth more than any metric. Three of the five
errors I inspected were cat, cow and dog all predicted as sheep, and all three were
animals photographed standing on grass. The model appears to have learned something about
green outdoor backgrounds rather than about the animals, which no confusion matrix would
have told me on its own. Seeing the standardized sixty-four by sixty-four input rather
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

# Part 2's half of the reflection, kept in its own constant so the Part 1 text
# stays exactly as it was written.
PART2_REFLECTION_TEXT = """
Part 2 tested that habit immediately, and I failed it for most of a day. The training
loop reported 96.67 percent validation accuracy for a ResNet18 whose true accuracy was
10.00 percent, and I believed the number long enough to build on it. It was predicting a
single class for every image in the test set. The cause was one argument,
non_blocking=True, on the copy that moves labels to the GPU. That copy is only safe from
pinned host memory, the image cache is not pinned, and on Metal the label tensor arrived
before its contents did: indices around 6.8e18 for a ten-class problem. Cross-entropy
indexes its target directly, so on a CPU an index that large raises immediately, and on
Metal it reads out of bounds and returns a number anyway. The loop trained against
corrupt labels and then scored itself against the same corrupt labels, and the two halves
of that agreed with each other perfectly.

What makes it worth writing down is that the bug produced a better number than the truth.
A crash I would have found in a minute. A validation accuracy of 0.9667 beside a loss of
0.0001 looks like a network that is working, and the only reason I caught it is that the
test accuracy underneath was exactly 0.1000 with a macro F1 of 0.0182, which is the
arithmetic of predicting one class out of ten and not a number a real model produces.
Every training curve and every table in this report would have inherited it silently. The
loop now checks its own reported accuracy against an independent evaluation of the same
weights, once on the device and once on CPU, and a label outside the valid range raises
instead of being trained on. I had been treating a green test suite as the thing to be
skeptical of, when the more dangerous object was a metric a component computes about
itself.

The prescribed learning rate taught me something I had not gone looking for. At AdamW
with lr 0.001, AlexNet sat at chance for all twenty epochs with its training loss pinned
at 2.3026, which is ln(10) and exactly what a network emits when its output is uniform,
and VGG16 crawled to 38.9 percent. Every other architecture trained at the same rate
without difficulty. The two that failed are the two that predate batch normalization, and
with no normalization layers to absorb it a 0.001 step on ImageNet weights destroys the
pretrained features before the first epoch is out. Rerun at 0.0001 they reach 90.0 and
94.4 percent, and VGG16 finishes as the most accurate model in the whole benchmark, above
EfficientNet-B0 and ConvNeXt-Tiny.

Had I reported the first run I would have written that AlexNet and VGG16 are obsolete
designs that modern architectures have left behind. That sentence would have been about
my optimizer settings rather than about the architectures, and it is the Intel padding
mistake from Part 1 wearing a different costume: a difference I was ready to attribute to
the variable I found interesting, which actually belonged to something else in the setup.
The assignment's instruction to document every learning-rate change is what forced me to
look, and what turned a nuisance into the most interesting result in Part 2.

The memory measurement took three attempts and the first two both produced numbers I
would have published. Reading the allocator pool gave a column that rose monotonically in
the order the architectures happened to run in, so DenseNet121 appeared to need 12.7
gigabytes, which was simply everything allocated before it. Sampling live allocation
between batches gave a real quantity but the wrong one: after a forward pass the
intermediate tensors are already freed, so what is left is the weights, and the column
matched checkpoint size at a correlation of 1.000 to within 0.23 megabytes. It was
section 18 wearing a different heading, and it would have carried its own weight in the
deployment ranking as though it were independent information. Only the third attempt,
sampling inside the forward pass, measured the working set a deployment target actually
has to fit.

That third measurement reversed a recommendation I had already half written.
EfficientNet-B0 holds 16 megabytes of weights and needs roughly 840 megabytes of
activations to run a batch, about fifty times its own size; DenseNet121 is forty times,
MobileNetV3 thirty. AlexNet, the second largest model here by weight, has the smallest
working set of all nine and the lowest single-image latency, because an 11x11 stride-4
first convolution collapses the spatial dimensions before there is much feature map to
carry. The architectures marketed as efficient are efficient in parameters, and parameters
are not what occupies memory at inference. An embedded board picked on checkpoint size
alone would not run the model I would have recommended, and none of the metrics the
assignment's table asks for would have shown me that.

The last thing Part 2 changed is how I read a leaderboard. VGG16 is the most accurate
model in this benchmark and ranks fifth on the deployment score, because it scores the
maximum on accuracy and zero on throughput, size and memory at once. YOLO reaches 93.9
percent from 1.5 million parameters and a 3.2 megabyte checkpoint, within half a point of
the winner at a hundredth of the storage. ConvNeXt-Tiny selected its first epoch and then
trained nineteen more while its validation loss climbed, and only finished second because
the rule that keeps the best checkpoint rather than the last one is in the protocol. Three
different models win the three criteria, and the spread between them in accuracy is far
smaller than the spread in what they cost. I came into this assignment thinking
architecture selection was a question with a single answer per dataset. It is a question
about which constraint binds on the hardware you actually have, and the benchmark's job is
to tell you where each model sits rather than to crown one.
"""


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

def american_date() -> str:
    """Month day, year - the convention a US reader expects in a report.

    Built by hand rather than with strftime("%B %-d, %Y"); the no-pad flag is
    a GNU extension and is not portable.
    """
    today = date.today()
    return f"{today.strftime('%B')} {today.day}, {today.year}"


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
        f"**Date:** {american_date()}",
        "",
    ]


def section_objective(runs: dict) -> list:
    datasets = sorted({run["dataset"] for run in runs.values()})
    return [
        "## 1. Objective",
        "",
        "This report compares sixteen image classification methods on one "
        "problem: four classical machine-learning models, two neural network "
        "baselines, and ten deep CNN architectures spanning 2012 to 2024. All "
        "of them are driven through a single public function in an installable "
        f"package, `{PACKAGE_NAME}`.",
        "",
        "It is in two parts. **Part 1**, sections 1 to 8, benchmarks the "
        "classical models and the two neural baselines on flattened 64x64 "
        "pixels, and varies training set size, color mode and dataset one at a "
        "time. **Part 2**, sections 9 to 18, adds AlexNet, VGG16, GoogLeNet, "
        "ResNet18, ResNet50, DenseNet121, MobileNetV3, EfficientNet-B0, "
        "ConvNeXt-Tiny and a YOLO classifier at 224x224 and compares them "
        "directly against those baselines - on the identical images, the same "
        "seed, the same held-out test half, and the same scoring code.",
        "",
        f"Every model in a given run sees one stratified split, one set of "
        f"metrics and one measurement of cost. Across the Part 1 runs, exactly "
        f"one thing changes at a time: the size of the training set, the color "
        f"mode, or the dataset. There are {len(runs)} runs over "
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
        "**Standardization.** Every image is fitted into 64x64 by scaling the "
        "longest side and padding the remainder with zeros, then normalized to "
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
    """Same dataset, same color mode, different subset size."""
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
        "Same dataset, same color mode, nested subsets - the smaller set is a "
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


def section_color_experiment(runs: dict) -> list:
    """Same dataset, same size, different color mode."""
    pairs = {}
    for run in runs.values():
        pairs.setdefault((run["dataset"], run["size"]), {})[run["color_mode"]] = run
    pairs = {k: v for k, v in pairs.items() if {"rgb", "grayscale"} <= set(v)}
    if not pairs:
        return []

    lines = [
        "## 6. Experiment: color",
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
    # generalizing from one of them.
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
            "separable by color directly, one threshold on one channel is "
            "informative; where it is not, the extra channels are mostly noise "
            "to a model with no ensemble to average them away.",
            "",
        ]

    # The cost side generalizes even where the accuracy side does not.
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
    """Different datasets at the same size and color mode."""
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
        "standardization: Animals-10 loses 27% to padding, Intel essentially "
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
        raw, normalized = [], []
        for run in members:
            score = macro_f1(run).get(model)
            chance = 1 / len(run["classes"])
            raw.append(f"{score:.3f}")
            normalized.append(f"{score / chance:.2f}")
            lifts[run["name"]].append(score / chance)
        lines.append(f"| {model} | " + " | ".join(raw) + " | " +
                     " | ".join(normalized) + " |")
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


def section_reflection(number: int = 9) -> list:
    """Both halves of the reflection, Part 1's text unchanged.

    Part 2's paragraphs follow on from Part 1's closing line about treating a
    passing test as a claim that needs checking, which is the thing Part 2
    went on to test.
    """
    lines = [f"## {number}. Reflection", "", REFLECTION_TEXT.strip(), ""]
    if PART2_REFLECTION_TEXT.strip():
        lines += [PART2_REFLECTION_TEXT.strip(), ""]
    return lines


def section_reproduction(runs: dict, number: int = 10) -> list:
    """One closing section covering both parts.

    Part 2's commands are folded in as a subsection rather than given a
    section of their own: a reader looking for how to rerun this work should
    find one place that tells them, not two in different halves of the
    document.
    """
    configuration = list(runs.values())[0]["configuration"]
    return [
        f"## {number}. Reproducing these results",
        "",
        "### Part 1: the traditional ML and baseline neural benchmark",
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
        f"The Part 1 results were produced under package version "
        f"{configuration['package']['version']} at random seed "
        f"{configuration['random_seed']}, image size "
        f"{configuration['image_size'][0]}x{configuration['image_size'][1]}; "
        "the package is now at 2.0.0, which adds the Part 2 architectures "
        "without changing anything Part 1 depends on. Every model fixes its "
        "own random state, and each run's full configuration is in its "
        "`run_configuration.json`.",
        "",
        "The images are not committed. Both datasets are third-party "
        "collections, so the download scripts rebuild the subsets instead, and "
        "the seeded sampling makes that rebuild exact.",
        "",
    ]


def em_dashes(text: str) -> str:
    """Replace spaced hyphens with em dashes outside fenced code blocks.

    American style sets a parenthetical dash closed up as an em dash. The
    replacement skips code fences, where a hyphen is a command-line flag or
    an operator rather than punctuation, and skips table rows, whose pipes
    and dashes are structure.
    """
    lines, inside_fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            inside_fence = not inside_fence
            lines.append(line)
            continue
        if inside_fence or line.lstrip().startswith("|"):
            lines.append(line)
            continue
        lines.append(line.replace(" - ", " \u2014 "))
    return "\n".join(lines)


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
    lines += section_color_experiment(runs)
    lines += section_dataset_experiment(runs)
    lines += section_cost(runs)

    # Part 2 sits after the Part 1 study and before the closing material. Its
    # reproduction block is held back and placed with Part 1's at the end, so
    # the document has one place where the commands live rather than two.
    #
    # Numbering continues across the seam rather than restarting: Part 1 ends
    # at section 8, so Part 2 is numbered from 9 and the closing sections
    # follow it. The count comes from Part 2 itself, so adding a section there
    # does not silently leave two sections sharing a number here.
    part2 = report_part2.sections(include_reproduction=False, start_number=9)
    next_number = 9
    if part2:
        lines += part2
        next_number = 9 + report_part2.section_count()

    lines += section_reflection(next_number)
    lines += section_reproduction(runs, next_number + 1)
    lines += report_part2.reproduction_only()
    return em_dashes("\n".join(lines))


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(build(), encoding="utf-8")
    print(f"Wrote {REPORT_MD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
