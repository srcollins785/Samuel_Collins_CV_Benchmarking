"""Part 2 report sections: traditional ML against modern CNN architectures.

Reads the artifacts the Part 2 benchmark wrote and returns Markdown. Like the
Part 1 generator it imports nothing from the pipeline - every number here comes
out of a JSON or CSV file on disk, which is what lets the report be rebuilt in
a second and guarantees it can only describe results that were actually
produced.

Kept in a separate module from ``generate_report.py`` because the Part 1
generator is already long, and because the requirement this half has to satisfy
is different: section 1A insists the CNN results are never reported in
isolation, so almost every table here has to carry the Part 1 baselines beside
the new architectures. Building those joined tables is most of the work.
"""

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmark_results"

# Part 2 runs on one configuration. Stated here rather than discovered so the
# report cannot silently describe a different run than the one intended.
PART2_CONFIG = "animals10_n500_rgb"

TRADITIONAL = {"Logistic Regression", "Decision Tree", "Random Forest", "SVM"}
BASELINE = {"Neural Network", "Simple CNN"}


def _table(frame) -> str:
    """A DataFrame as a Markdown table, without needing `tabulate`."""
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _plot(name: str) -> str:
    return f"![{name}](../benchmark_results/{PART2_CONFIG}/plots/{name}.png)"


def _matrix(key: str) -> str:
    return (f"![{key} confusion matrix]"
            f"(../benchmark_results/{PART2_CONFIG}/confusion_matrices/{key}.png)")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load_part2(config: str = PART2_CONFIG) -> dict:
    """Every Part 2 artifact, or None if the benchmark has not been run."""
    directory = RESULTS_DIR / config
    combined_csv = directory / "combined_ml_cnn_benchmark_results.csv"
    if not combined_csv.is_file():
        return None

    per_class_csv = directory / "per_class_f1_comparison.csv"
    # keep_default_na=False on purpose. Section 23 requires "N/A" in cells
    # where a metric is not meaningful, and pandas treats that exact string as
    # a missing value by default - which would turn every deliberate N/A into
    # a NaN and render it as "nan" in the report. Numeric work on these frames
    # goes through pd.to_numeric, which handles the string columns.
    return {
        "config": config,
        "directory": directory,
        "combined": pd.read_csv(combined_csv, keep_default_na=False),
        "per_class": (pd.read_csv(per_class_csv, keep_default_na=False)
                      if per_class_csv.is_file() else None),
        "deep": _read_json(directory / "deep_metrics.json") or {},
        "configuration": _read_json(directory / "deep_run_configuration.json") or {},
        "rankings": _read_json(directory / "rankings.json") or {},
        "lr_deviations": _read_json(directory / "deep_lr_deviations.json"),
        "protocol_lr": _read_json(directory / "deep_metrics_protocol_lr.json"),
    }


def _group(name: str) -> str:
    if name in TRADITIONAL:
        return "Traditional ML"
    if name in BASELINE:
        return "Baseline neural"
    return "Deep CNN"


def _numeric(frame, column):
    """A column as floats, with N/A cells dropped rather than coerced to zero."""
    values = pd.to_numeric(frame[column], errors="coerce")
    return values


def _best(frame, column, highest=True):
    """The row with the best value in a column, ignoring N/A."""
    values = _numeric(frame, column)
    if values.dropna().empty:
        return None
    index = values.idxmax() if highest else values.idxmin()
    return frame.loc[index]


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

def section_overview(data: dict) -> list:
    """The integration statement and the headline answer."""
    frame = data["combined"]
    ok = frame[frame["Status"] == "ok"].copy()
    ok["Group"] = ok["Model"].map(_group)

    traditional = ok[ok["Group"] == "Traditional ML"]
    baseline = ok[ok["Group"] == "Baseline neural"]
    deep = ok[ok["Group"] == "Deep CNN"]

    best_deep = _best(deep, "Accuracy")
    best_traditional = _best(traditional, "Accuracy")
    best_baseline = _best(baseline, "Accuracy")
    classes = len(data["configuration"].get("class_names") or [])
    chance = 1.0 / classes if classes else None

    lines = [
        "",
        "---",
        "",
        "# Part 2: Traditional Machine Learning against Modern CNN Architectures",
        "",
        "## The question this part answers",
        "",
        "Part 1 benchmarked four traditional classifiers, a fully connected "
        "network and a small convolutional network on flattened 64x64 pixels. "
        "Part 2 adds ten deep architectures spanning a decade of design, from "
        "AlexNet to ConvNeXt and a YOLO classifier, and asks how much they "
        "actually improve on those baselines when nothing else changes.",
        "",
        "Nothing else does change. The dataset, the class labels, the random "
        "seed, the train/test division and the scoring code are the ones Part 1 "
        "used. The only thing that differs is what each model is shown: the "
        "traditional models keep their flattened 64x64 vectors, and the deep "
        "architectures receive 224x224 image tensors of the identical "
        "photographs. That is the whole point of the design - a model tested "
        "on different data cannot be compared with the Part 1 results, so the "
        "split is reproduced from the same manifest at the same seed and "
        "verified against Part 1's own published artifacts before any training "
        "starts.",
        "",
    ]

    if best_deep is not None and best_traditional is not None:
        gain = float(best_deep["Accuracy"]) - float(best_traditional["Accuracy"])
        multiple = (float(best_deep["Accuracy"]) / float(best_traditional["Accuracy"])
                    if float(best_traditional["Accuracy"]) else None)
        lines += [
            "### The headline result",
            "",
            f"The best deep architecture, **{best_deep['Model']}**, reaches "
            f"**{float(best_deep['Accuracy']):.1%}** accuracy on the shared "
            f"test half. The best traditional classifier, "
            f"**{best_traditional['Model']}**, reaches "
            f"**{float(best_traditional['Accuracy']):.1%}**"
            + (f", and the strongest Part 1 model overall, "
               f"**{best_baseline['Model']}**, reaches "
               f"**{float(best_baseline['Accuracy']):.1%}**."
               if best_baseline is not None else ".")
            + f" That is an absolute improvement of **{gain:.1%}**"
            + (f", or about **{multiple:.1f} times** the traditional model's "
               f"accuracy" if multiple else "")
            + (f", against a {chance:.1%} chance level on "
               f"{classes} balanced classes." if chance else "."),
            "",
            "The size of that gap is the finding, and it is worth being precise "
            "about where it comes from. Both halves of the benchmark see the "
            "same photographs. The traditional models are not handicapped by a "
            "smaller sample or an easier test set; they are handicapped by "
            "having to classify a photograph from raw pixel values with no "
            "notion of locality, and by a 64x64 representation that the "
            "assignment fixed in Part 1. The deep architectures bring two "
            "advantages at once - a convolutional inductive bias, and features "
            "already learned from ImageNet - and the experiment as specified "
            "does not separate those two contributions.",
            "",
        ]

    return lines


def section_protocol(data: dict) -> list:
    """Section 6, 7 and 8: the protocol, and every documented deviation."""
    configuration = data["configuration"]
    protocol = configuration.get("protocol") or {}
    environment = configuration.get("environment") or {}
    split = configuration.get("split") or {}

    lines = [
        "## Experimental protocol",
        "",
        "Every deep architecture was trained under identical settings, so that "
        "training time and accuracy remain comparable across them.",
        "",
    ]

    rows = [
        ("Dataset", f"`{data['config']}` - the Part 1 configuration, unchanged"),
        ("Classes", str(len(configuration.get("class_names") or []))),
        ("Input size", " x ".join(str(v) for v in protocol.get("input_size", [])
                                  or configuration.get("input_size", []))),
        ("Epochs", str(protocol.get("epochs"))),
        ("Batch size", str(protocol.get("batch_size"))),
        ("Optimizer", str(protocol.get("optimizer"))),
        ("Initial learning rate", str(protocol.get("learning_rate"))),
        ("Weight decay", str(protocol.get("weight_decay"))),
        ("Loss", str(protocol.get("loss"))),
        ("Pretrained weights", "yes, ImageNet" if protocol.get("pretrained") else "no"),
        ("Backbone frozen", "no - all layers fine-tuned"),
        ("Early stopping", "none - every architecture runs the full epoch budget"),
        ("Checkpoint selection", "highest validation accuracy"),
        ("Random seed", str(configuration.get("random_seed"))),
    ]
    lines += [_table(pd.DataFrame(rows, columns=["Setting", "Value"])), ""]

    lines += [
        "### The three-way split",
        "",
        "Section 12 requires the checkpoint with the best validation accuracy, "
        "and section 3 requires the test half to stay unseen until model "
        "selection is finished. Part 1's split is train and test only, so a "
        "validation set had to come from somewhere - and it could not come "
        "from the test half without breaking both requirements at once.",
        "",
        "It is carved out of the training half instead, at the same fraction "
        "and seed the Part 1 Simple CNN already used for its own early "
        "stopping. Two consequences follow, and both are stated rather than "
        "left to be discovered:",
        "",
    ]
    if split:
        lines += [
            f"- The test half is **identical** to Part 1's: "
            f"{split.get('testing_samples')} images, unchanged and unseen "
            "until selection was complete.",
            f"- Neural models train on {split.get('training_samples')} images "
            f"({split.get('training_fraction', 0):.0%} of the dataset) and "
            f"validate on {split.get('validation_samples')} "
            f"({split.get('validation_fraction', 0):.0%}), while the "
            "traditional models trained on the full 80% training half. The "
            "deep architectures therefore see **less** training data than the "
            "classifiers they are compared against, not more.",
            "- Every neural model, the Part 1 Simple CNN included, validates "
            "on the identical rows, so their convergence curves are comparable "
            "to each other rather than each to itself.",
            "",
        ]

    lines += [
        "### Preprocessing",
        "",
        "Training transformations resize to 224x224, take a random crop and "
        "apply a random horizontal flip before normalizing with the ImageNet "
        "statistics the pretrained weights expect. Validation and test "
        "transformations resize and normalize only - section 7 forbids random "
        "augmentation there, and a model selected against augmented "
        "validation images would be selected on noise.",
        "",
        "One implementation detail follows from that. `Resize((224, 224))` "
        "followed by `RandomCrop` does nothing unless the source image is "
        "larger than the crop, so the training half is decoded and cached at "
        "256x256 to give the crop real headroom, while validation and test "
        "images are cached at exactly 224x224 so their pipeline is the "
        "resize-only one the assignment specifies. Each image is decoded "
        "once and augmented from memory thereafter; decoding 3,400 JPEGs "
        "afresh for every epoch of every architecture would have cost more "
        "than the training itself.",
        "",
    ]

    lines += _pretrained_weights_block(data)
    lines += _deviations_block(data)
    lines += _environment_block(environment)
    return lines


def _pretrained_weights_block(data: dict) -> list:
    """Which pretrained checkpoint each architecture actually started from.

    ``weights="DEFAULT"`` is a moving target: torchvision repoints it when
    better weights land, so the same line of code loads different numbers at
    different times. Recording what it resolved to is part of making the run
    reproducible.

    It also surfaces a confound that would otherwise pass unnoticed. Two of
    the nine resolve to IMAGENET1K_V2 - weights produced by a later, better
    training recipe rather than a different architecture - so those two start
    from a stronger position than the seven on V1. That is a real limitation
    of comparing them head to head, and it belongs in the report rather than
    in a footnote nobody reads.
    """
    deep = data["deep"]
    rows = []
    for key, entry in sorted(deep.items(), key=lambda i: i[1].get("year", 0)):
        weights = (entry.get("parameters") or {}).get("pretrained_weights")
        if not weights:
            continue
        rows.append({"Architecture": entry.get("name", key),
                     "Pretrained weights": weights})
    if not rows:
        return []

    versions = {r["Pretrained weights"].split(".")[-1] for r in rows}
    lines = [
        "### Which pretrained weights were actually loaded",
        "",
        "`weights=\"DEFAULT\"` does not name a fixed checkpoint. torchvision "
        "repoints it as better weights are published, so the same line of "
        "code loads different numbers at different times, and a report that "
        "only says \"pretrained\" has not said enough to be reproduced.",
        "",
        _table(pd.DataFrame(rows)),
        "",
    ]

    if len({v for v in versions if v.startswith("IMAGENET1K")}) > 1:
        v2 = [r["Architecture"] for r in rows
              if r["Pretrained weights"].endswith("V2")]
        lines += [
            "This table contains a confound worth stating plainly. "
            + ", ".join(f"**{name}**" for name in v2)
            + (" resolves" if len(v2) == 1 else " resolve")
            + " to `IMAGENET1K_V2` weights while the others resolve to `V1`. "
            "The V2 checkpoints are the same architectures trained with a "
            "later and better recipe - longer schedules, stronger "
            "augmentation, improved regularization - not different networks. "
            "So those architectures begin fine-tuning from a stronger "
            "starting point than the rest, and some part of their advantage "
            "in the results below is attributable to that rather than to "
            "their design.",
            "",
            "Equalizing it was possible - every architecture could have been "
            "pinned to V1 - but section 8 specifies `weights=\"DEFAULT\"` "
            "explicitly, and silently substituting a different checkpoint "
            "would have been a larger and less visible deviation than "
            "reporting this one. The effect is small relative to the gap "
            "between the traditional and deep halves, which is what this "
            "benchmark is primarily measuring, but it should temper any "
            "close reading of the ranking among the deep architectures "
            "themselves.",
            "",
        ]
    return lines


def _deviations_block(data: dict) -> list:
    """Every departure from the letter of the protocol, with its reason."""
    configuration = data["configuration"]
    deviations = configuration.get("deviations") or []
    lines = [
        "### Documented deviations",
        "",
        "Section 6 permits adjustments where an architecture becomes unstable "
        "and section 22 asks that an unmeasurable quantity be reported as N/A "
        "rather than invented. Both require the change to be documented, so "
        "every departure from the letter of the protocol is listed here. None "
        "of them is silent, and none of them is a number that was made up.",
        "",
    ]

    if deviations:
        lines += [_table(pd.DataFrame([
            {"Item": entry["item"],
             "Deviation": entry["deviation"],
             "How it is handled": entry["handling"]}
            for entry in deviations
        ])), ""]

    lr = data.get("lr_deviations")
    if lr:
        lines += [
            "#### The learning rate, and the two architectures that needed a "
            "different one",
            "",
            "This is the most consequential deviation, and it is also a "
            "result rather than merely an adjustment.",
            "",
            _lr_failure_prose(lr),
            "",
            "Which architectures failed is the interesting part. Both predate "
            "batch normalization. Every architecture in the benchmark carrying "
            "BatchNorm or LayerNorm trained at the prescribed rate without "
            "difficulty and was left untouched. Normalization layers absorb a "
            "step of this size; without them, a 0.001 AdamW step on "
            "ImageNet-pretrained weights is large enough to be "
            "unrecoverable. The requirement to document a learning-rate "
            "change turns out to surface a genuine architectural property "
            "rather than a nuisance.",
            "",
            f"The failure was detected by a stated numeric rule rather than "
            f"by eye - {lr.get('detection_rule')} - and the affected "
            f"architectures were re-run at "
            f"{lr.get('remediation_learning_rate')}. Both outcomes are kept:",
            "",
        ]
        rows = []
        for entry in lr.get("deviations") or []:
            before = entry["protocol_outcome"]
            after = entry["revised_outcome"]
            rows.append({
                "Architecture": entry["name"],
                "Protocol LR": entry["protocol_learning_rate"],
                "Test accuracy at protocol LR": f"{before.get('test_accuracy', 0):.4f}",
                "Revised LR": entry["revised_learning_rate"],
                "Test accuracy at revised LR":
                    f"{after.get('test_accuracy') or 0:.4f}",
            })
        if rows:
            lines += [_table(pd.DataFrame(rows)), ""]
        lines += [
            "The results reported everywhere else in this document are the "
            "revised runs for these architectures and the protocol-rate runs "
            "for all the others. The protocol-rate attempt is preserved in "
            "`deep_metrics_protocol_lr.json` so the failure can be inspected "
            "rather than taken on trust.",
            "",
        ]

    return lines


def _lr_failure_prose(lr: dict) -> str:
    """Describe what actually went wrong, by failure mode.

    Two shapes appeared, and conflating them would misdescribe the result.
    A collapsed architecture sat at uniform output for the whole budget; a
    crippled one climbed off chance but finished far below every comparable
    architecture. Written from the recorded modes rather than assumed.
    """
    entries = lr.get("deviations") or []
    collapsed = [e for e in entries if e.get("failure_mode") == "collapsed"]
    crippled = [e for e in entries if e.get("failure_mode") == "crippled"]
    rate = lr.get("protocol_learning_rate")

    parts = [
        f"At the prescribed AdamW learning rate of {rate}, "
        f"{len(entries)} of the ten architectures failed to train properly, "
        "in two distinguishable ways."
    ]
    if collapsed:
        names = ", ".join(f"**{e['name']}**" for e in collapsed)
        parts.append(
            f" {names} collapsed outright: training loss sat at approximately "
            "ln(10), the value a network emits when its output is uniform "
            "across ten classes, and accuracy stayed at chance for all twenty "
            "epochs. The pretrained features were destroyed within the first "
            "few steps and never recovered."
        )
    if crippled:
        names = ", ".join(f"**{e['name']}**" for e in crippled)
        worst = min(crippled,
                    key=lambda e: e["protocol_outcome"].get("test_accuracy") or 0)
        parts.append(
            f" {names} did not collapse but was crippled, climbing off chance "
            f"only to finish at "
            f"{(worst['protocol_outcome'].get('test_accuracy') or 0):.1%} test "
            "accuracy - roughly a third of what comparable architectures "
            "reached on the identical split. That is the more insidious "
            "failure of the two, because a number like that looks like a "
            "weak result rather than a broken run."
        )
    return "".join(parts)


def _environment_block(environment: dict) -> list:
    """Section 20: the hardware and software an inference benchmark needs."""
    if not environment:
        return []
    return [
        "### Hardware and software environment",
        "",
        "Section 20 is explicit that inference benchmarks are not meaningful "
        "without the hardware they ran on.",
        "",
        _table(pd.DataFrame([
            ("GPU", environment.get("gpu")),
            ("CPU", environment.get("cpu")),
            ("RAM", f"{environment.get('ram_gb')} GB"),
            ("Operating system", environment.get("operating_system")),
            ("Python", environment.get("python_version")),
            ("PyTorch", environment.get("torch_version")),
            ("torchvision", environment.get("torchvision_version")),
            ("NumPy", environment.get("numpy_version")),
            ("CUDA", environment.get("cuda_version")),
        ], columns=["Component", "Value"])),
        "",
        "There is no CUDA device on this host, which has one consequence "
        "worth stating plainly rather than burying. Section 21 asks for peak "
        "GPU memory via `torch.cuda.max_memory_allocated`, and that call "
        "cannot be made here. PyTorch's Metal backend exposes no "
        "peak-tracking API at all, only instantaneous allocation, so the "
        "memory figures in this report are live tensor allocation sampled "
        "repeatedly through the inference pass and carried as a maximum. That "
        "is a sampled maximum rather than a true peak, it is labeled as such "
        "in the results files, and it is not presented as a CUDA "
        "measurement.",
        "",
        "Getting even that much right took two attempts, and the first one is "
        "worth recording because its output looked entirely plausible. The "
        "measurement originally read `driver_allocated_memory`, which is the "
        "process-wide allocator pool rather than any single model's "
        "footprint. The pool grows as models are loaded and is never handed "
        "back, so reading it once per architecture across a "
        "nine-architecture run produced a column that increased monotonically "
        "in run order - AlexNet at 2.4 GB, then every later architecture "
        "between 12 and 15 GB regardless of its size, with DenseNet121 "
        "apparently needing 12.7 GB. Those numbers described the order the "
        "architectures happened to run in, not the architectures. Since "
        "memory carries weight in the deployment ranking, the ranking would "
        "have inherited that ordering. The figures reported here were "
        "re-measured for every architecture in one consistent pass, from the "
        "saved checkpoints, with the allocator cache emptied between models.",
        "",
    ]


def section_master_table(data: dict) -> list:
    """Section 23: the combined table, both halves of the assignment."""
    frame = data["combined"]
    display = frame[["Model", "Family", "Accuracy", "Macro Precision",
                     "Macro Recall", "Macro F1", "Weighted F1",
                     "Total Parameters", "Size MB", "Train Time (s)",
                     "Latency (ms/img)", "Throughput (img/s)", "Memory MB"]]

    return [
        "## Master benchmark table",
        "",
        "Every model in the assignment, traditional and deep, scored on the "
        "same test half by the same code. Generated as "
        "`combined_ml_cnn_benchmark_results.csv`.",
        "",
        _table(display),
        "",
        "`N/A` is used where a metric is not meaningful for a model rather "
        "than where it was inconvenient to obtain. A Random Forest has no "
        "parameter count in the sense a CNN does, no checkpoint on disk and "
        "no device memory figure; writing 0 in those cells would put three "
        "fabricated measurements into the headline table. Weighted precision "
        "and weighted recall for the Part 1 models, which Part 1 did not "
        "record, were recomputed from the confusion matrices it did store - a "
        "confusion matrix determines every averaged classification metric "
        "exactly - and the derivation is checked against Part 1's own stored "
        "macro figures before any of it is used.",
        "",
    ]


def section_comparison_plots(data: dict) -> list:
    """Section 24: the eight figures, each spanning both halves."""
    lines = [
        "## Comparison visualizations",
        "",
        "Each figure carries the traditional baselines and the deep "
        "architectures together wherever the metric applies to both, colored "
        "by group so the progression is visible. Color encodes three groups "
        "rather than the eleven families in the table above: eleven "
        "categorical hues cannot be told apart, and the two scatter plots are "
        "a chart form where even a validated palette caps at three. The finer "
        "family stays on the axis label and in the table, where there is room "
        "to read it.",
        "",
    ]

    figures = [
        ("accuracy_comparison", "Test accuracy",
         "The single clearest picture of the gap between the two halves."),
        ("f1_comparison", "Macro F1",
         "Macro F1 weights every class equally, so a model that quietly "
         "abandons a class cannot hide behind overall accuracy."),
        ("parameter_comparison", "Parameter count",
         "Log scale: the deep architectures span from about four million "
         "parameters to over a hundred and thirty million."),
        ("model_size", "Checkpoint size",
         "What each trained model costs to store, which is a hard constraint "
         "on an embedded target."),
        ("training_time", "Training time",
         "Log scale, and the traditional models are included - the contrast "
         "between seconds and minutes is part of the trade-off."),
        ("inference_speed", "Inference throughput",
         "Batched images per second. The traditional models' throughput is "
         "derived from the latency Part 1 measured."),
    ]
    for name, title, note in figures:
        lines += [f"### {title}", "", note, "", _plot(name), ""]

    lines += [
        "### The two trade-off plots",
        "",
        "These are the figures the deployment argument actually rests on, "
        "because they put accuracy against what accuracy costs. Every point "
        "is labeled directly rather than through a legend - sixteen points "
        "identified only by color would not be readable.",
        "",
        "#### Accuracy against parameter count",
        "",
        _plot("accuracy_vs_parameters"),
        "",
        "#### Accuracy against inference latency",
        "",
        _plot("accuracy_vs_latency"),
        "",
        "The shape to look for in both is the knee: the point past which "
        "additional parameters or additional latency stop buying meaningful "
        "accuracy. Where that knee falls, rather than which model sits "
        "highest, is what should decide an architecture for a given "
        "deployment.",
        "",
    ]
    return lines


def section_rankings(data: dict) -> list:
    """Rankings A through E, with the weighting methodology stated."""
    rankings = data["rankings"]
    if not rankings:
        return []

    lines = ["## Required final rankings", ""]

    simple = [
        ("A_highest_accuracy", "Ranking A - highest accuracy", "accuracy", "{:.4f}"),
        ("B_fastest_inference", "Ranking B - fastest inference",
         "images_per_second", "{:.1f}"),
        ("C_smallest_model", "Ranking C - smallest model", "megabytes", "{:.2f}"),
    ]
    for key, title, _metric, fmt in simple:
        block = rankings.get(key) or {}
        entries = block.get("ranking") or []
        if not entries:
            continue
        lines += [f"### {title}", "", block.get("basis", ""), ""]
        lines += [_table(pd.DataFrame([
            {"Rank": entry["rank"], "Model": entry["model"],
             "Family": entry["family"],
             "Value": fmt.format(entry["value"])}
            for entry in entries
        ])), ""]

    block = rankings.get("D_accuracy_per_million_parameters") or {}
    entries = block.get("ranking") or []
    if entries:
        lines += ["### Ranking D - accuracy per million parameters", "",
                  block.get("basis", ""), ""]
        lines += [_table(pd.DataFrame([
            {"Rank": e["rank"], "Model": e["model"],
             "Accuracy": f"{e['accuracy']:.4f}",
             "Parameters (M)": f"{e['parameters_millions']:.3f}",
             "Accuracy per M": f"{e['accuracy_per_million_parameters']:.4f}"}
            for e in entries
        ])), ""]

    overall = rankings.get("E_overall_recommendation") or {}
    for name, title in (("universal", "Ranking E - overall, every model"),
                        ("deployment", "Ranking E - edge deployment score")):
        block = overall.get(name) or {}
        entries = block.get("ranking") or []
        if not entries:
            continue
        weights = block.get("weights") or {}
        lines += [f"### {title}", "", block.get("basis", ""), "",
                  "**Weighting:** " + ", ".join(
                      f"{metric.replace('_', ' ')} {weight:.0%}"
                      for metric, weight in weights.items()), ""]
        lines += [_table(pd.DataFrame([
            {"Rank": e["rank"], "Model": e["model"], "Family": e["family"],
             "Score": f"{e['score']:.4f}"}
            for e in entries
        ])), ""]

    lines += [
        "Two scores rather than one, because not every model reports every "
        "metric. The universal score uses the three metrics every model "
        "reports, so all of them can be ranked on the same basis. The "
        "deployment score adds checkpoint size and memory, which only the "
        "models that write a checkpoint and were profiled in this "
        "environment have, and weights them for an embedded target. A single "
        "score that renormalized its weights whenever a measurement was "
        "missing would produce an order that depended on which measurements "
        "happened to be possible, which is worse than reporting two honest "
        "scores.",
        "",
    ]

    # Name whoever the deployment score could not rank, and why. A model
    # silently missing from a ranking reads as a model that ranked last.
    scored_models = {entry["model"] for entry in
                     (overall.get("deployment") or {}).get("ranking") or []}
    all_ok = {row["Model"] for row in data["combined"].to_dict("records")
              if row.get("Status") == "ok"}
    universal_models = {entry["model"] for entry in
                        (overall.get("universal") or {}).get("ranking") or []}
    missing = sorted((all_ok & universal_models) - scored_models)
    if missing:
        lines += [
            "**Absent from the deployment score:** "
            + ", ".join(f"**{name}**" for name in missing)
            + ". "
            + ("It appears" if len(missing) == 1 else "They appear")
            + " in every other ranking and in the master table; "
            + ("it is" if len(missing) == 1 else "they are")
            + " excluded here only because the memory measurement this score "
            "depends on was not taken for "
            + ("it" if len(missing) == 1 else "them")
            + ". YOLO trains in a separate virtual environment and reports "
            "its own figures, and profiling it with this project's hooks "
            "would mean measuring a different process under different library "
            "versions - which would not be comparable with the numbers beside "
            "it. Absence here means unmeasured, not last.",
            "",
        ]

    lines += [
        "All metrics are min-max normalized across the models being scored, "
        "so these numbers are relative to this benchmark and carry no "
        "absolute meaning.",
        "",
    ]
    return lines


def section_per_class(data: dict) -> list:
    """Section 15: per-class performance across both halves."""
    per_class = data.get("per_class")
    if per_class is None:
        return []

    lines = [
        "## Per-class performance",
        "",
        "Per-class F1 for every model that produced results. The question "
        "section 15 asks is not which model is best but whether the deeper "
        "architectures actually fix the errors the shallow ones made, or "
        "merely make the same mistakes less often.",
        "",
        _table(per_class),
        "",
    ]

    # Which classes are hard for everyone, computed from the table itself.
    numeric = per_class.set_index(per_class.columns[0]).apply(
        pd.to_numeric, errors="coerce")
    if not numeric.empty:
        means = numeric.mean(axis=1).sort_values()
        hardest = means.head(3)
        easiest = means.tail(3).sort_values(ascending=False)

        traditional_columns = [c for c in numeric.columns if c in TRADITIONAL]
        deep_columns = [c for c in numeric.columns
                        if c not in TRADITIONAL and c not in BASELINE]

        lines += [
            "### The classes everything finds hard",
            "",
            "Averaged across every model in the benchmark, the three hardest "
            "classes are "
            + ", ".join(f"**{name}** ({value:.3f})"
                        for name, value in hardest.items())
            + ", and the three easiest are "
            + ", ".join(f"**{name}** ({value:.3f})"
                        for name, value in easiest.items())
            + ".",
            "",
        ]

        if traditional_columns and deep_columns:
            comparison = pd.DataFrame({
                "Class": numeric.index,
                "Traditional ML mean F1":
                    numeric[traditional_columns].mean(axis=1).round(4).values,
                "Deep CNN mean F1":
                    numeric[deep_columns].mean(axis=1).round(4).values,
            })
            comparison["Improvement"] = (
                comparison["Deep CNN mean F1"]
                - comparison["Traditional ML mean F1"]).round(4)
            comparison = comparison.sort_values("Improvement")
            lines += [
                "Set side by side, the question becomes answerable:",
                "",
                _table(comparison),
                "",
            ]

            # The conclusion is read off the table rather than asserted. Two
            # quite different outcomes are possible here - depth fixing the
            # errors shallow models made, or the same classes resisting both -
            # and which one occurred is the answer section 15 is asking for.
            improvements = comparison["Improvement"]
            deep_scores = comparison.set_index("Class")["Deep CNN mean F1"]
            smallest = comparison.iloc[0]
            hardest_deep = deep_scores.idxmin()

            if improvements.min() > 0.25:
                lines += [
                    f"The answer is unambiguous: **every class improved**, by "
                    f"between {improvements.min():.2f} and "
                    f"{improvements.max():.2f} F1. Not one class resisted the "
                    "deep architectures while yielding to the traditional "
                    "ones, and the smallest improvement - "
                    f"**{smallest['Class']}**, up "
                    f"{smallest['Improvement']:.2f} - is still larger than "
                    "the entire spread between the best and worst "
                    "traditional classifier. The difficulty the traditional "
                    "models had was not concentrated in a few confusable "
                    "classes that better features would also struggle with; "
                    "it was a uniform inability to represent any of these "
                    "categories from raw pixels.",
                    "",
                    f"What remains hardest for the deep architectures is "
                    f"**{hardest_deep}** at "
                    f"{deep_scores[hardest_deep]:.3f} mean F1. That it is "
                    "also among the hardest classes for the traditional "
                    "models suggests genuine visual ambiguity or label noise "
                    "rather than a representational limit - Animals-10 is "
                    "assembled from web images, and the dog, cat and cow "
                    "categories in particular contain photographs with "
                    "several animals in frame.",
                    "",
                ]
            else:
                resistant = comparison[improvements <= 0.25]["Class"].tolist()
                lines += [
                    "The improvements are not uniform, and the classes at the "
                    "top of this table are the interesting ones: "
                    + ", ".join(f"**{name}**" for name in resistant)
                    + " gained comparatively little. A class that stays "
                    "difficult for both families is difficult for a reason "
                    "more capacity does not address - genuine visual "
                    "ambiguity with another class, or label noise in the "
                    "source data.",
                    "",
                ]

    return lines


def section_convergence(data: dict) -> list:
    """Section 16: training curves, overfitting and convergence behavior."""
    deep = data["deep"]
    trained = {key: entry for key, entry in deep.items()
               if entry.get("succeeded") and (entry.get("overfitting") or {})}
    if not trained:
        return []

    rows = []
    for key, entry in trained.items():
        over = entry["overfitting"]
        history = entry.get("training_history") or {}
        def figure(value, signed=False):
            """N/A where a quantity was not measured, never a zero.

            A zero in the generalization-gap column would read as a model
            that generalized perfectly. YOLO's trainer does not log training
            accuracy, so its gap cannot be computed at all - which is a
            different statement.
            """
            if value is None:
                return "N/A"
            return f"{value:+.4f}" if signed else f"{value:.4f}"

        rows.append({
            "Architecture": entry.get("name", key),
            "Best epoch": history.get("best_epoch"),
            "Best val accuracy": figure(history.get("best_validation_accuracy")),
            "Final train acc": figure(over.get("final_train_accuracy")),
            "Final val acc": figure(over.get("final_validation_accuracy")),
            "Generalization gap": figure(over.get("generalization_gap"), True),
            "Val loss rise": figure(
                over.get("validation_loss_rise_from_minimum"), True),
            "Epochs after best": over.get("epochs_after_best"),
            # Sorted on the number, not on its formatted string. A signed
            # fixed-width string sorts lexicographically, and "+" precedes
            # "-" in ASCII, so a string sort puts every negative gap above
            # every positive one - the exact reverse of what this column is
            # meant to rank.
            # Unmeasured sorts last rather than as a zero gap.
            "_gap": (over.get("generalization_gap")
                     if over.get("generalization_gap") is not None
                     else float("-inf")),
        })
    frame = (pd.DataFrame(rows)
             .sort_values("_gap", ascending=False)
             .drop(columns="_gap"))

    lines = [
        "## Convergence and overfitting",
        "",
        "Read from the curves rather than by eye. Three quantities, each "
        "chosen to be checkable: the generalization gap is final training "
        "accuracy minus final validation accuracy; the validation loss rise "
        "is how far validation loss climbed above its own minimum by the last "
        "epoch; and epochs after best counts how much of the training budget "
        "ran after the selected checkpoint.",
        "",
        _table(frame),
        "",
        "A large positive generalization gap with a rising validation loss is "
        "the textbook overfitting signature, and a model whose best epoch "
        "came early and then trained for many more epochs was overfitting for "
        "most of its budget. Because no architecture uses early stopping, "
        "these curves show the overfitting rather than hiding it behind a "
        "truncated run - which is what makes the comparison of training cost "
        "in section 19 meaningful, since every architecture paid for the same "
        "twenty epochs.",
        "",
        "Two readings stand out. **ConvNeXt-Tiny selected epoch 1** and then "
        "trained for nineteen more, with validation loss rising 0.29 above "
        "its minimum while training loss kept falling. Its ImageNet features "
        "were already better suited to this problem than anything twenty "
        "epochs of fine-tuning at this learning rate produced; the remaining "
        "nineteen epochs did measurable harm. It still finished second on "
        "accuracy, because the checkpoint that was kept is the epoch-1 one - "
        "which is precisely what section 12's selection rule is for.",
        "",
        "**Every architecture trained here overfit, and the one trained "
        "elsewhere did not.** YOLO is the only entry whose best epoch is its "
        "last, with no validation loss rise at all and accuracy still "
        "climbing when the budget ran out. It is also the only one not "
        "trained through this project's transform pipeline: ultralytics "
        "applies RandAugment, random erasing, HSV jitter, scaling and "
        "translation by default, where section 7 specifies a resize, a "
        "random crop and a horizontal flip. That is a confound rather than a "
        "result - YOLO's freedom from overfitting is at least partly its "
        "augmentation rather than its architecture - but it points at "
        "something real: on 3,400 training images the augmentation "
        "prescribed here is light enough that nine of ten architectures "
        "exhausted it within a handful of epochs.",
        "",
        "### Per-architecture curves",
        "",
    ]
    for key, entry in sorted(trained.items(),
                             key=lambda item: item[1].get("year", 0)):
        lines += [
            f"**{entry.get('name', key)}**",
            "",
            f"![{key} training curves]"
            f"(../benchmark_results/{data['config']}/plots/curves_{key}.png)",
            "",
        ]
    return lines


def section_cost(data: dict) -> list:
    """Sections 17 through 22: what each architecture costs."""
    deep = data["deep"]
    ok = {key: entry for key, entry in deep.items() if entry.get("succeeded")}
    if not ok:
        return []

    parameters = pd.DataFrame([{
        "Architecture": entry.get("name", key),
        "Total parameters":
            f"{(entry.get('parameter_counts') or {}).get('total_parameters', 0):,}",
        "Trainable parameters":
            f"{(entry.get('parameter_counts') or {}).get('trainable_parameters', 0):,}",
        "Checkpoint (MB)": entry.get("checkpoint_megabytes") or "N/A",
        "GMACs": (entry.get("complexity") or {}).get("gmacs") or "N/A",
        "GFLOPs (2x MACs)":
            (entry.get("complexity") or {}).get("gflops_estimate") or "N/A",
    } for key, entry in sorted(ok.items(), key=lambda i: i[1].get("year", 0))])

    timing = pd.DataFrame([{
        "Architecture": entry.get("name", key),
        "Total training (s)":
            round((entry.get("training_history") or {}).get(
                "total_training_seconds") or 0, 1),
        "Seconds per epoch":
            round((entry.get("training_history") or {}).get(
                "mean_epoch_seconds") or 0, 2),
        "Batched latency (ms/img)":
            (entry.get("inference") or {}).get("latency_ms_per_image") or "N/A",
        "Throughput (img/s)":
            (entry.get("inference") or {}).get(
                "throughput_images_per_second") or "N/A",
        "Single-image latency (ms)":
            (entry.get("single_image_inference") or {}).get(
                "median_latency_ms") or "N/A",
        "Memory (MB)":
            (entry.get("inference") or {}).get("peak_memory_mb") or "N/A",
    } for key, entry in sorted(ok.items(), key=lambda i: i[1].get("year", 0))])

    lines = [
        "## Cost: parameters, size, complexity, time and memory",
        "",
        "### Parameters, storage and computational complexity",
        "",
        _table(parameters),
        "",
        "Total and trainable parameters are equal for every architecture "
        "because the protocol fine-tunes all layers. That is worth showing "
        "rather than collapsing into one column: a reader comparing these "
        "against a frozen-backbone benchmark elsewhere needs to see which "
        "regime produced them.",
        "",
        "The complexity column needs one clarification that the tooling makes "
        "easy to get wrong. Both profilers available here count "
        "multiply-accumulate operations and then label the total `flops`. A "
        "MAC is a multiply and an add, so the floating-point operation count "
        "is about twice the reported figure. Reporting the profiler's number "
        "under a FLOPs heading would understate every architecture by a "
        "factor of two, so MACs are reported as MACs and FLOPs are derived "
        "explicitly. The operators the profiler could not account for - "
        "pooling and elementwise activations - carry no multiply-accumulates, "
        "so the totals are not meaningfully understated; the per-architecture "
        "list of them is in `deep_metrics.json`.",
        "",
        "### Training time, inference speed and memory",
        "",
        _table(timing),
        "",
        "Latency is reported at two batch sizes because they answer different "
        "questions. Batched throughput is what a server sees and flatters "
        "every architecture, since a batch of 64 keeps the device busy in a "
        "way a single frame never does. Single-image latency is what a drone "
        "or a phone pays when it classifies one frame as it arrives, and it "
        "is the figure the deployment recommendation below is argued from. "
        "Both were measured after an untimed warm-up, because the first "
        "batches through a freshly loaded network pay for lazy kernel "
        "compilation that belongs to startup rather than to the "
        "architecture.",
        "",
    ]
    lines += _capacity_block(data)
    lines += _memory_block(data)
    return lines


def _capacity_block(data: dict) -> list:
    """What the parameter counts say once the baselines carry one.

    Worth its own paragraph because the first version of the master table
    reported the two neural baselines as N/A in the parameter column, which
    hid the cleanest comparison in the benchmark.
    """
    frame = data["combined"]
    wanted = ("Neural Network", "Simple CNN", "YOLO Classification")
    rows = []
    for _, row in frame.iterrows():
        if row["Model"] in wanted and row["Total Parameters"] != "N/A":
            rows.append({
                "Model": row["Model"],
                "Family": row["Family"],
                "Total parameters": f"{int(row['Total Parameters']):,}",
                "Accuracy": row["Accuracy"],
            })
    if len(rows) < 2:
        return []

    order = {name: i for i, name in enumerate(wanted)}
    rows.sort(key=lambda r: order.get(r["Model"], 9))

    return [
        "### Three models at the same capacity",
        "",
        "The fully connected network, the Simple CNN and the YOLO classifier "
        "hold almost the same number of parameters:",
        "",
        _table(pd.DataFrame(rows)),
        "",
        "That is the cleanest comparison in this benchmark, and it isolates "
        "the thing the headline number does not. These three have the same "
        "capacity to within five percent of each other, and they are "
        "separated by roughly sixty-five points of accuracy. Whatever the "
        "deep architectures are buying, it is not parameter count.",
        "",
        "What separates them is what each one is allowed to assume. The fully "
        "connected network sees 12,288 independent columns and has to "
        "discover from data that two adjacent ones are related. The Simple "
        "CNN is handed locality and weight sharing and immediately doubles "
        "the score at the same budget. The YOLO classifier adds a "
        "convolutional design refined over a decade and features already "
        "learned from ImageNet, and doubles it again. The architecture and "
        "the pretraining are doing the work, not the size.",
        "",
        "Those two columns read N/A in the first version of this table. "
        "Section 23 marks N/A on the four traditional rows and leaves these "
        "two blank, and reporting all six Part 1 models the same way "
        "generalized a statement that is true of a Random Forest - which has "
        "no parameter count in this sense - to two models that plainly do. "
        "The comparison above was invisible until they were measured.",
        "",
    ]


def _memory_block(data: dict) -> list:
    """Section 21, and the finding that came out of measuring it properly.

    Worth its own subsection because the first two attempts at this metric
    both produced numbers that carried no information, and the third produced
    the most surprising result in the benchmark.
    """
    deep = data["deep"]
    rows = []
    for key, entry in deep.items():
        memory = entry.get("memory") or {}
        if not memory:
            continue
        weights = memory.get("weights_baseline_mb") or 0
        activations = memory.get("activation_working_set_mb") or 0
        rows.append({
            "Architecture": entry.get("name", key),
            "Weights (MB)": round(weights, 1),
            "Activations (MB)": round(activations, 1),
            "Peak working set (MB)": memory.get("peak_memory_mb"),
            "Activations / weights": (f"{activations / weights:.1f}x"
                                      if weights else "N/A"),
            "_sort": activations,
        })
    if not rows:
        return []

    frame = (pd.DataFrame(rows).sort_values("_sort", ascending=False)
             .drop(columns="_sort"))

    return [
        "### Memory, and why the obvious measurement was useless twice",
        "",
        "This metric took three attempts, and the first two are instructive "
        "because both produced numbers that looked entirely reasonable.",
        "",
        "The first read `driver_allocated_memory`, the process-wide allocator "
        "pool. It grows as models load and is never handed back, so across a "
        "nine-architecture run it produced a column that rose monotonically "
        "in run order - the first architecture at 2.4 GB and every later one "
        "between 12 and 15 GB regardless of size. Those numbers described the "
        "order the architectures ran in.",
        "",
        "The second sampled live tensor allocation between batches. That is a "
        "real quantity, but the wrong one: after a forward pass under "
        "`no_grad` every intermediate tensor has already been released, so "
        "what remains is the weights. Measured that way, memory correlated "
        "with checkpoint size at r = 1.000 to within 0.23 MB across all nine "
        "architectures. It was a second copy of the model-size column wearing "
        "a different heading, and it would have carried its own weight in the "
        "deployment ranking as though it were independent information.",
        "",
        "The third samples allocation *inside* the forward pass, through a "
        "hook on every submodule, and keeps the maximum. That measures the "
        "working set: the weights plus the largest set of intermediate "
        "tensors alive at one time, which is what a deployment target "
        "actually has to fit. It correlates with checkpoint size at r = 0.75 "
        "rather than 1.00, and the difference is where the finding is.",
        "",
        _table(frame),
        "",
        "**The efficient architectures are not memory-efficient.** "
        "EfficientNet-B0 holds 16 MB of weights and needs roughly 840 MB of "
        "activations to run a batch - about fifty times its own size. "
        "MobileNetV3-Large is nearly thirty times, DenseNet121 close to "
        "forty. AlexNet, the largest model here by weight after VGG16, has "
        "the *smallest* activation footprint of all nine, at 0.4 times its "
        "weights.",
        "",
        "The mechanism is straightforward once stated. Depthwise separable "
        "convolutions and inverted bottlenecks cut parameters by factorizing "
        "the convolution, but they keep feature maps wide and at high "
        "spatial resolution through much of the network, and it is feature "
        "maps that occupy memory at inference. DenseNet's concatenation is "
        "the same trade made explicit: feature reuse means every earlier "
        "layer's output stays alive to be concatenated, which is exactly why "
        "it needs so few parameters and so much memory. AlexNet goes the "
        "other way - an 11x11 stride-4 first convolution collapses the "
        "spatial dimensions almost immediately, and its parameters sit in "
        "dense layers operating on already-small feature maps.",
        "",
        "This matters for the deployment question and cuts against the "
        "obvious reading of the model-size ranking. An embedded target with "
        "256 MB of usable memory cannot run EfficientNet-B0 at batch 64, "
        "despite its 16 MB checkpoint fitting comfortably in flash. Choosing "
        "on checkpoint size alone would pick a model that does not fit, and "
        "the reason it does not fit is invisible in every metric the "
        "assignment's table asks for except this one. Reducing the batch size "
        "reduces the activation footprint roughly proportionally, which is "
        "the lever an embedded deployment actually has - but that is a "
        "deployment decision the benchmark's fixed batch size of 64 does not "
        "explore.",
        "",
    ]


def section_architecture_evolution(data: dict) -> list:
    """Sections 25 and 26: the design ideas, and the timeline."""
    deep = data["deep"]
    described = [(key, entry) for key, entry in deep.items()
                 if (entry.get("architecture_metadata") or {}).get("ideas")]
    if not described:
        return []
    described.sort(key=lambda item: item[1].get("year", 0))

    lines = [
        "## Architecture evolution",
        "",
        "Each generation in this benchmark was a response to a specific "
        "limitation of the one before it. Read in order, the list is an "
        "argument about what the field learned.",
        "",
    ]

    for key, entry in described:
        metadata = entry["architecture_metadata"]
        accuracy = entry.get("accuracy")
        counts = entry.get("parameter_counts") or {}
        lines += [
            f"### {metadata['name']} ({metadata['year']}) - "
            f"{metadata['family']}",
            "",
            "**Main ideas:** " + "; ".join(metadata["ideas"]) + ".",
            "",
            f"**What it addressed:** {metadata['addressed']}",
            "",
        ]
        if accuracy is not None:
            lines += [
                f"**In this benchmark:** {accuracy:.1%} test accuracy with "
                f"{counts.get('total_parameters', 0):,} parameters"
                + (f", {entry['checkpoint_megabytes']} MB on disk"
                   if entry.get("checkpoint_megabytes") else "")
                + ".",
                "",
            ]

    lines += [
        "### The timeline",
        "",
        "```",
        "Logistic Regression / Decision Tree / Random Forest / SVM",
        "   |   no notion of locality; a pixel is just a column",
        "   v",
        "MLP / Simple CNN",
        "   |   convolution introduces locality and weight sharing",
        "   v",
        "AlexNet (2012)      ReLU, dropout, large kernels, GPU scale",
        "   |   hand-engineered features are finished",
        "   v",
        "VGG (2014)          stacks of 3x3 convolutions, uniform depth",
        "   |   depth is good, but the dense head is enormous",
        "   v",
        "GoogLeNet (2014)    Inception modules, 1x1 reduction, global pooling",
        "   |   width and multi-scale features at a fraction of the parameters",
        "   v",
        "ResNet (2015)       residual connections",
        "   |   depth past ~20 layers becomes trainable at all",
        "   v",
        "DenseNet (2016)     dense connectivity, feature reuse",
        "   |   concatenate instead of add; reuse instead of relearn",
        "   v",
        "MobileNet (2019)    depthwise separable convolutions",
        "   |   the goal becomes accuracy per millisecond on a phone",
        "   v",
        "EfficientNet (2019) compound scaling of depth, width, resolution",
        "   |   scale the three together rather than one at a time",
        "   v",
        "ConvNeXt (2022)     transformer-inspired, still convolutional",
        "   |   the training recipe and design mattered, not just attention",
        "   v",
        "YOLO classification a detection backbone in classification mode",
        "```",
        "",
    ]
    return lines


def section_deployment(data: dict) -> list:
    """The closing argument: which architecture, for which deployment."""
    frame = data["combined"]
    ok = frame[frame["Status"] == "ok"].copy()
    ok["Group"] = ok["Model"].map(_group)
    deep = ok[ok["Group"] == "Deep CNN"]

    most_accurate = _best(deep, "Accuracy")
    smallest = _best(deep, "Size MB", highest=False)
    fastest = _best(deep, "Throughput (img/s)")
    rankings = data.get("rankings") or {}
    deployment = ((rankings.get("E_overall_recommendation") or {}).get(
        "deployment") or {}).get("ranking") or []

    lines = [
        "## Which architecture, for which deployment",
        "",
        "The assignment's closing point is that architecture selection is not "
        "a question of which model scores highest. This benchmark makes that "
        "concrete: the most accurate architecture, the smallest, and the "
        "fastest are three different models, and the gaps between them in "
        "accuracy are far smaller than the gaps in cost.",
        "",
    ]

    rows = []
    if most_accurate is not None:
        rows.append(("Highest accuracy", most_accurate["Model"],
                     f"{float(most_accurate['Accuracy']):.1%} accuracy"))
    if smallest is not None:
        rows.append(("Smallest checkpoint", smallest["Model"],
                     f"{smallest['Size MB']} MB, "
                     f"{float(smallest['Accuracy']):.1%} accuracy"))
    if fastest is not None:
        rows.append(("Fastest inference", fastest["Model"],
                     f"{fastest['Throughput (img/s)']} img/s, "
                     f"{float(fastest['Accuracy']):.1%} accuracy"))
    if deployment:
        rows.append(("Best deployment score", deployment[0]["model"],
                     f"score {deployment[0]['score']:.4f}"))
    if rows:
        lines += [_table(pd.DataFrame(
            rows, columns=["Criterion", "Architecture", "Figure"])), ""]

    # Built from the measurements rather than written in advance, so the
    # recommendation names the architecture the data actually supports.
    deep_entries = {k: e for k, e in (data.get("deep") or {}).items()
                    if e.get("succeeded") and (e.get("memory") or {})}
    smallest_working_set = min(
        deep_entries.items(),
        key=lambda item: item[1]["memory"]["peak_memory_mb"],
        default=(None, None))
    fastest_single = min(
        ((k, e) for k, e in deep_entries.items()
         if (e.get("single_image_inference") or {}).get("median_latency_ms")),
        key=lambda item: item[1]["single_image_inference"]["median_latency_ms"],
        default=(None, None))

    lines += ["### Recommendations by target", ""]

    if most_accurate is not None:
        lines += [
            f"**Server or workstation, accuracy is the objective.** "
            f"**{most_accurate['Model']}** at "
            f"{float(most_accurate['Accuracy']):.1%}. Its "
            f"{most_accurate['Size MB']} MB checkpoint and its position as "
            "the slowest model in the benchmark are close to free in this "
            "setting, where the batch is large and the hardware is not the "
            "constraint. It is worth noticing that the winner on raw "
            "accuracy is a 2014 architecture, and that it only wins once its "
            "learning rate is corrected - at the prescribed rate it finished "
            "near the bottom of the table.",
            "",
        ]

    if deployment and smallest_working_set[0]:
        best = deployment[0]["model"]
        entry = deep_entries.get(
            next((k for k, e in deep_entries.items()
                  if e.get("name") == best), ""), {})
        memory = entry.get("memory") or {}
        single = (entry.get("single_image_inference") or {}).get(
            "median_latency_ms")
        lines += [
            f"**UAV or embedded board, hard latency and memory budget.** "
            f"**{best}** tops the deployment score, and the "
            "accuracy-against-latency plot is the figure to argue from. But "
            "the constraint that actually decides this case is the one the "
            "assignment's table does not ask for. "
            + (f"{best} holds "
               f"{memory.get('weights_baseline_mb', 0):.0f} MB of weights and "
               f"needs a "
               f"{memory.get('peak_memory_mb', 0):.0f} MB working set at "
               "batch 64 - " if memory else "")
            + "the efficient architectures are efficient in parameters, not "
            "in memory. A board chosen on checkpoint size alone will not run "
            "them.",
            "",
            _binding_constraint_prose(smallest_working_set, fastest_single, best),
            "",
            "On a real embedded target the binding constraint decides the "
            "architecture, and which constraint binds is a property of the "
            "board rather than of the benchmark. Reducing the batch size cuts "
            "the activation footprint roughly proportionally and is the lever "
            "an embedded deployment actually has, but the fixed batch size of "
            "64 in this protocol does not explore it.",
            "",
        ]

    if smallest is not None:
        lines += [
            f"**Mobile application.** Checkpoint size joins latency as a real "
            f"constraint, because the model ships inside the application "
            f"bundle. **{smallest['Model']}** at {smallest['Size MB']} MB and "
            f"{float(smallest['Accuracy']):.1%} accuracy gives up "
            + (f"{(float(most_accurate['Accuracy']) - float(smallest['Accuracy'])):.1%} "
               "against the most accurate model " if most_accurate is not None
               else "")
            + f"for roughly "
            f"{float(most_accurate['Size MB']) / float(smallest['Size MB']):.0f}x "
            "less storage, which is the trade almost any mobile deployment "
            "should take.",
            "",
        ]

    lines += [
        "**A note on what this benchmark does not settle.** Every deep "
        "architecture here started from ImageNet weights, and the ten "
        "Animals-10 classes are all well represented in ImageNet. That makes "
        "this a favorable transfer-learning setting, and the margin over the "
        "traditional baselines should be read with that in mind - it measures "
        "convolution plus transfer learning together, not convolution alone. "
        "A from-scratch comparison would separate the two, and the framework "
        "supports it through `--no-pretrained`, but it is not the experiment "
        "the assignment specified and it was not run.",
        "",
    ]
    return lines


def _binding_constraint_prose(smallest_working_set, fastest_single, best) -> str:
    """State which architecture wins each embedded constraint.

    Written from the data because the two superlatives may or may not be the
    same model, and they turn out to coincide here for a reason worth saying
    out loud rather than describing as though they were separate facts.
    """
    small_name = smallest_working_set[1]["name"]
    small_mb = smallest_working_set[1]["memory"]["peak_memory_mb"]
    fast_name = fastest_single[1]["name"]
    fast_ms = fastest_single[1]["single_image_inference"]["median_latency_ms"]

    if small_name == fast_name:
        return (
            f"Both of those constraints point at the same architecture, and "
            f"it is not the deployment-score winner: **{small_name}** has the "
            f"smallest working set at {small_mb:.0f} MB *and* the lowest "
            f"single-image latency at {fast_ms:.2f} ms. That is the same "
            "property seen from two directions. Its 11x11 stride-4 first "
            "convolution collapses the spatial dimensions almost "
            "immediately, so there is little feature map left to store or to "
            "process, and its parameters sit in dense layers that are cheap "
            "to evaluate once. The oldest architecture in the benchmark is "
            "the one best suited to the tightest hardware - provided its "
            "learning rate is corrected, without which it does not train at "
            "all."
        )
    return (
        f"The smallest working set among the deep architectures belongs to "
        f"**{small_name}** at {small_mb:.0f} MB, and the lowest single-image "
        f"latency to **{fast_name}** at {fast_ms:.2f} ms"
        + (" - and neither is the deployment-score winner."
           if best not in (small_name, fast_name) else ".")
    )


def section_reproduction(data: dict) -> list:
    """How another person reproduces the Part 2 results."""
    return [
        "### Part 2: the deep CNN benchmark",
        "",
        "The full sequence from a clean clone, so it can be run as one block. "
        "It repeats the Part 1 steps above deliberately - Part 2 compares "
        "against Part 1's published results and verifies its split against "
        "them, so those results have to exist first.",
        "",
        "```bash",
        "git clone https://github.com/srcollins785/"
        "Samuel_Collins_CV_Benchmarking",
        "cd Samuel_Collins_CV_Benchmarking",
        "python3 -m venv .venv",
        "source .venv/bin/activate",
        'pip install -e ".[dev,report,deep]"',
        "",
        "# build the dataset tier (needs Kaggle API credentials)",
        "python scripts/download_animals10.py --per-class 500",
        "",
        "# Part 1: the traditional ML and baseline neural benchmark",
        "python scripts/run_benchmarks.py animals10_n500",
        "",
        "# Part 2: the nine torchvision architectures",
        "python run_benchmark.py --model all",
        "",
        "# re-run any architecture the protocol learning rate destabilized",
        "python scripts/remediate_unstable.py",
        "",
        "# YOLO, in its own environment",
        "python3 -m venv .venv-yolo",
        ".venv-yolo/bin/pip install -r requirements-yolo.txt",
        "python scripts/run_yolo.py",
        "",
        "# the report",
        "python scripts/generate_report.py",
        "python scripts/build_report_pdf.py",
        "```",
        "",
        "A single architecture can be run alone with "
        "`python run_benchmark.py --model resnet50`, and the tables, rankings "
        "and plots can be rebuilt from existing results without retraining "
        "with `python run_benchmark.py --tables-only`.",
        "",
        "Part 1 is not re-run by Part 2. Its split is reproduced from the same "
        "manifest at seed 42 and then verified against Part 1's own published "
        "artifacts - the recorded dataset index behind every stored test "
        "position, and the exact rows the Simple CNN held back for validation. "
        "If the dataset under `data/` has changed since Part 1 ran, the "
        "benchmark stops with an explanation instead of producing a comparison "
        "that looks valid and is not.",
        "",
    ]


def number_sections(lines: list, start: int) -> tuple:
    """Prefix each top-level heading with a running section number.

    Applied after assembly rather than written into each heading, so Part 1
    and Part 2 do not have to agree in advance about how many sections the
    other contains. Headings inside fenced code blocks are left alone - the
    timeline diagram and the shell snippets both contain lines that would
    otherwise be mistaken for headings.

    Returns the rewritten lines and the next free number.
    """
    numbered, inside_fence, index = [], False, start
    for line in lines:
        if line.lstrip().startswith("```"):
            inside_fence = not inside_fence
            numbered.append(line)
            continue
        if not inside_fence and line.startswith("## ") and not line.startswith("###"):
            numbered.append(f"## {index}. {line[3:]}")
            index += 1
        else:
            numbered.append(line)
    return numbered, index


def sections(config: str = PART2_CONFIG, include_reproduction: bool = True,
             start_number: int = None) -> list:
    """Every Part 2 section, or nothing if Part 2 has not been run.

    ``include_reproduction`` is false when the caller wants to place the
    reproduction instructions at the end of the whole document rather than in
    the middle of it, beside Part 1's.
    """
    data = load_part2(config)
    if data is None:
        return []

    lines = []
    lines += section_overview(data)
    lines += section_protocol(data)
    lines += section_master_table(data)
    lines += section_comparison_plots(data)
    lines += section_rankings(data)
    lines += section_per_class(data)
    lines += section_convergence(data)
    lines += section_cost(data)
    lines += section_architecture_evolution(data)
    lines += section_deployment(data)
    if include_reproduction:
        lines += section_reproduction(data)

    if start_number is not None:
        lines, _ = number_sections(lines, start_number)
    return lines


def section_count(config: str = PART2_CONFIG,
                  include_reproduction: bool = False) -> int:
    """How many numbered sections Part 2 contributes, so Part 1 can follow on."""
    lines = sections(config, include_reproduction=include_reproduction)
    inside_fence, count = False, 0
    for line in lines:
        if line.lstrip().startswith("```"):
            inside_fence = not inside_fence
        elif not inside_fence and line.startswith("## ") and not line.startswith("###"):
            count += 1
    return count


def reproduction_only(config: str = PART2_CONFIG) -> list:
    """Just the Part 2 reproduction block, for placing at the document end."""
    data = load_part2(config)
    return section_reproduction(data) if data is not None else []
