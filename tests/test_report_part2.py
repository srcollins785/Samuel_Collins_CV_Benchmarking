"""Tests for the Part 2 report sections.

The report is the deliverable, and it is assembled from roughly a dozen
artifact files whose fields are optional in different combinations. A missing
key raises a KeyError at the moment the report is built, which is the worst
time to find out, so these tests build the report from synthetic artifacts
covering the awkward cases: an architecture that failed, a metric that does
not apply, a learning-rate deviation present and absent.

The substantive assertions are about honesty rather than wording. Section 23's
N/A cells must survive into the rendered tables rather than becoming "nan",
and the learning-rate prose must describe the failure modes that actually
occurred rather than assuming one of them.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import report_part2  # noqa: E402

CLASSES = ["cat", "dog"]


def _deep_entry(name, key, accuracy, **extra):
    entry = {
        "name": name, "family": "Residual CNN", "year": 2015,
        "kind": "deep_cnn", "succeeded": True, "error": None,
        "accuracy": accuracy, "macro_precision": accuracy,
        "macro_recall": accuracy, "macro_f1": accuracy,
        "weighted_precision": accuracy, "weighted_recall": accuracy,
        "weighted_f1": accuracy,
        "parameter_counts": {"total_parameters": 11_181_642,
                             "trainable_parameters": 11_181_642,
                             "total_parameters_millions": 11.182,
                             "frozen_parameters": 0},
        "complexity": {"macs": 1_819_000_000, "gmacs": 1.819,
                       "flops_estimate": 3_638_000_000,
                       "gflops_estimate": 3.638, "profiler": "fvcore",
                       "counts": "multiply-accumulate operations (MACs)",
                       "unsupported_operators": ["aten::max_pool2d"],
                       "note": "counts MACs"},
        "inference": {"images": 1000, "latency_ms_per_image": 1.2,
                      "throughput_images_per_second": 830.0,
                      "peak_memory_mb": 512.0, "batch_size": 64,
                      "memory_measurement": "mps_live_allocation_sampled_maximum"},
        "single_image_inference": {"median_latency_ms": 6.5, "batch_size": 1},
        "checkpoint_megabytes": 44.8,
        "confusion_matrix": [[400, 100], [80, 420]],
        "classification_report": {"cat": {"f1-score": 0.82},
                                  "dog": {"f1-score": 0.84}},
        "per_class_accuracy": {"cat": 0.8, "dog": 0.84},
        "training_history": {
            "total_training_seconds": 150.0, "mean_epoch_seconds": 7.5,
            "epochs_run": 20, "epochs_requested": 20, "best_epoch": 13,
            "best_validation_accuracy": accuracy,
            "checkpoint_megabytes": 44.8,
            "per_epoch": [
                {"epoch": n, "train_loss": 1.0 / n,
                 "validation_loss": 1.0 / n + 0.05,
                 "train_accuracy": min(0.99, 0.5 + n * 0.02),
                 "validation_accuracy": accuracy,
                 "epoch_seconds": 7.5, "learning_rate": 0.001}
                for n in range(1, 21)],
        },
        "overfitting": {"generalization_gap": 0.06,
                        "final_train_accuracy": 0.95,
                        "final_validation_accuracy": accuracy,
                        "epochs_after_best": 7,
                        "validation_loss_rise_from_minimum": 0.02,
                        "minimum_validation_loss": 0.1,
                        "minimum_validation_loss_epoch": 13},
        "parameters": {"architecture": key, "learning_rate": 0.001,
                       "optimizer": "AdamW"},
        "architecture_metadata": {
            "key": key, "name": name, "family": "Residual CNN", "year": 2015,
            "ideas": ["residual connections", "skip paths"],
            "addressed": "Made depth trainable.",
        },
    }
    entry.update(extra)
    return entry


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    """A synthetic result directory, with the awkward cases present."""
    config = "animals10_n500_rgb"
    directory = tmp_path / config
    (directory / "plots").mkdir(parents=True)
    (directory / "confusion_matrices").mkdir()

    # Metrics derived from the matrix rather than hand-written. The loader
    # checks that its own derivation reproduces what Part 1 stored, and
    # invented-but-plausible numbers here would trip that check - which is
    # what happened the first time this fixture was written.
    from samuel_collins_cv_benchmarking import combined as _combined

    part1_matrix = [[300, 200], [220, 280]]
    part1 = _combined.metrics_from_confusion_matrix(part1_matrix)
    (directory / "benchmark_metrics.json").write_text(json.dumps({
        "logistic_regression": {
            "name": "Logistic Regression", "succeeded": True, "error": None,
            "accuracy": part1["accuracy"],
            "macro_precision": part1["macro_precision"],
            "macro_recall": part1["macro_recall"],
            "macro_f1": part1["macro_f1"],
            "weighted_f1": part1["weighted_f1"],
            "training_time_seconds": 12.0,
            "inference_time_ms_per_image": 0.03,
            "confusion_matrix": part1_matrix,
            "classification_report": {"cat": {"f1-score": 0.59},
                                      "dog": {"f1-score": 0.57}},
            "training_history": {},
        },
    }), encoding="utf-8")

    (directory / "deep_metrics.json").write_text(json.dumps({
        "resnet18": _deep_entry("ResNet18", "resnet18", 0.91),
        # A failed architecture: its row must survive into the report.
        "convnext_tiny": {"name": "ConvNeXt-Tiny", "family": "Modern CNN",
                          "year": 2022, "succeeded": False,
                          "error": "RuntimeError: out of memory"},
    }), encoding="utf-8")

    (directory / "deep_run_configuration.json").write_text(json.dumps({
        "part": 2, "color_mode": "rgb", "input_size": [224, 224],
        "random_seed": 42, "class_names": CLASSES,
        "split": {"training_samples": 3400, "validation_samples": 600,
                  "testing_samples": 1000, "training_fraction": 0.68,
                  "validation_fraction": 0.12, "testing_fraction": 0.2,
                  "validation_fraction_of_training": 0.15,
                  "random_seed": 42, "stratified": True},
        "protocol": {"epochs": 20, "batch_size": 64, "optimizer": "AdamW",
                     "learning_rate": 0.001, "weight_decay": 0.01,
                     "loss": "CrossEntropyLoss", "pretrained": True,
                     "input_size": [224, 224]},
        "device": "mps",
        "environment": {"gpu": "Apple GPU", "cpu": "Apple M4 Max",
                        "ram_gb": 48.0, "operating_system": "Darwin",
                        "python_version": "3.9.6", "torch_version": "2.8.0",
                        "torchvision_version": "0.23.0",
                        "numpy_version": "2.0.2",
                        "cuda_version": "not applicable - no CUDA device",
                        "cuda_available": False, "mps_available": True},
        "deviations": [{"item": "GPU memory", "deviation": "no CUDA",
                        "handling": "reported as MPS, labeled"}],
    }), encoding="utf-8")

    from samuel_collins_cv_benchmarking import combined
    rows = combined.combined_rows(directory)
    combined.write_combined_csv(
        rows, directory / "combined_ml_cnn_benchmark_results.csv")
    combined.write_per_class_csv(
        directory, directory / "per_class_f1_comparison.csv", CLASSES)
    (directory / "rankings.json").write_text(
        json.dumps(combined.all_rankings(rows)), encoding="utf-8")

    monkeypatch.setattr(report_part2, "RESULTS_DIR", tmp_path)
    return config


# -- loading ----------------------------------------------------------------

def test_absent_part2_results_produce_no_sections(tmp_path, monkeypatch):
    """The Part 1 report must still build before Part 2 has been run."""
    monkeypatch.setattr(report_part2, "RESULTS_DIR", tmp_path)
    assert report_part2.load_part2("nothing_here") is None
    assert report_part2.sections("nothing_here") == []


def test_na_cells_survive_into_the_report(artifacts):
    """pandas treats "N/A" as missing by default; that must not happen here."""
    data = report_part2.load_part2(artifacts)
    frame = data["combined"]
    row = frame[frame["Model"] == "Logistic Regression"].iloc[0]
    assert row["Total Parameters"] == "N/A"
    assert row["Size MB"] == "N/A"


# -- the sections build -----------------------------------------------------

def test_every_section_renders(artifacts):
    lines = report_part2.sections(artifacts)
    assert lines
    text = "\n".join(lines)
    for heading in ("Part 2", "Experimental protocol", "Master benchmark table",
                    "Comparison visualizations", "Required final rankings",
                    "Per-class performance", "Convergence and overfitting",
                    "Architecture evolution", "Which architecture"):
        assert heading in text, f"missing section: {heading}"


def test_the_report_states_the_headline_comparison(artifacts):
    text = "\n".join(report_part2.sections(artifacts))
    assert "ResNet18" in text
    assert "Logistic Regression" in text
    # The gap between the two halves is the finding; it must be quantified.
    assert "91.0%" in text or "0.91" in text


def test_a_failed_architecture_keeps_its_row(artifacts):
    """Section 1A: the comparison must not quietly shrink."""
    data = report_part2.load_part2(artifacts)
    models = set(data["combined"]["Model"])
    assert "ConvNeXt-Tiny" in models
    row = data["combined"][data["combined"]["Model"] == "ConvNeXt-Tiny"].iloc[0]
    assert "RuntimeError" in row["Status"]


def test_reproduction_block_is_separable(artifacts):
    """Held out of the body so it can join Part 1's at the document end.

    The whole document should have one section telling a reader how to rerun
    the work, with a subsection per part - not a Part 2 recipe buried in the
    middle and a Part 1 one at the end.
    """
    body = "\n".join(report_part2.sections(artifacts, include_reproduction=False))
    tail = "\n".join(report_part2.reproduction_only(artifacts))

    assert "the deep CNN benchmark" not in body
    assert "run_benchmark.py --model all" not in body
    assert "### Part 2: the deep CNN benchmark" in tail
    assert "run_benchmark.py --model all" in tail
    # A subsection, so it nests under Part 1's numbered closing section.
    assert not tail.lstrip().startswith("## ")


def test_part_two_sections_can_be_numbered_from_an_offset(artifacts):
    """Numbering continues across the seam instead of restarting at one."""
    numbered = report_part2.sections(artifacts, include_reproduction=False,
                                     start_number=9)
    headings = [line for line in numbered
                if line.startswith("## ") and not line.startswith("###")]
    assert headings[0].startswith("## 9. ")
    assert headings[1].startswith("## 10. ")
    # The count Part 1 uses to place the closing sections must match reality.
    assert report_part2.section_count(artifacts) == len(headings)


def test_numbering_leaves_fenced_blocks_alone(artifacts):
    """The timeline diagram contains lines a naive pass would rewrite."""
    numbered = report_part2.sections(artifacts, include_reproduction=False,
                                     start_number=9)
    text = "\n".join(numbered)
    # The timeline is inside a fence and must survive verbatim.
    assert "Logistic Regression / Decision Tree / Random Forest / SVM" in text


def test_environment_section_states_the_absence_of_cuda(artifacts):
    text = "\n".join(report_part2.sections(artifacts))
    assert "no CUDA" in text or "not applicable" in text
    assert "sampled" in text.lower()


def test_complexity_discussion_explains_macs_versus_flops(artifacts):
    """The factor of two has to be visible, not assumed."""
    text = "\n".join(report_part2.sections(artifacts))
    assert "multiply-accumulate" in text.lower()
    assert "twice" in text.lower() or "2x MACs" in text


# -- the learning-rate deviation prose --------------------------------------

def test_no_lr_section_when_nothing_was_remediated(artifacts):
    text = "\n".join(report_part2.sections(artifacts))
    assert "two architectures that needed a different one" not in text


def test_lr_prose_describes_both_failure_modes(artifacts, tmp_path):
    """Collapsed and crippled are different failures; don't conflate them."""
    directory = tmp_path / artifacts
    (directory / "deep_lr_deviations.json").write_text(json.dumps({
        "detection_rule": "below 1.5x chance or 50% of the cohort best",
        "protocol_learning_rate": 0.001,
        "remediation_learning_rate": 0.0001,
        "architectures_left_at_protocol_rate": ["resnet18"],
        "deviations": [
            {"architecture": "alexnet", "name": "AlexNet",
             "failure_mode": "collapsed",
             "protocol_learning_rate": 0.001,
             "protocol_outcome": {"test_accuracy": 0.10, "macro_f1": 0.018,
                                  "best_validation_accuracy": 0.10,
                                  "final_train_loss": 2.3028,
                                  "final_train_accuracy": 0.092},
             "revised_learning_rate": 0.0001,
             "revised_outcome": {"test_accuracy": 0.83, "macro_f1": 0.83,
                                 "best_validation_accuracy": 0.84,
                                 "best_epoch": 18},
             "reason": "uniform output", "justification": "section 6 permits"},
            {"architecture": "vgg16", "name": "VGG16",
             "failure_mode": "crippled",
             "protocol_learning_rate": 0.001,
             "protocol_outcome": {"test_accuracy": 0.389, "macro_f1": 0.380,
                                  "best_validation_accuracy": 0.403,
                                  "final_train_loss": 1.6477,
                                  "final_train_accuracy": 0.418},
             "revised_learning_rate": 0.0001,
             "revised_outcome": {"test_accuracy": 0.90, "macro_f1": 0.90,
                                 "best_validation_accuracy": 0.91,
                                 "best_epoch": 19},
             "reason": "far below cohort", "justification": "section 6 permits"},
        ],
    }), encoding="utf-8")

    text = "\n".join(report_part2.sections(artifacts))
    assert "AlexNet" in text and "VGG16" in text
    assert "collapsed" in text.lower()
    assert "crippled" in text.lower()
    # Both outcomes reported, so the deviation is auditable.
    assert "0.1000" in text or "10.0%" in text
    assert "38.9%" in text or "0.3890" in text


def test_lr_prose_handles_only_collapsed_failures(artifacts, tmp_path):
    """The prose is generated from the modes present, not a fixed sentence."""
    directory = tmp_path / artifacts
    (directory / "deep_lr_deviations.json").write_text(json.dumps({
        "detection_rule": "below 1.5x chance",
        "protocol_learning_rate": 0.001,
        "remediation_learning_rate": 0.0001,
        "architectures_left_at_protocol_rate": [],
        "deviations": [
            {"architecture": "alexnet", "name": "AlexNet",
             "failure_mode": "collapsed", "protocol_learning_rate": 0.001,
             "protocol_outcome": {"test_accuracy": 0.10, "macro_f1": 0.018,
                                  "final_train_loss": 2.3, "final_train_accuracy": 0.09},
             "revised_learning_rate": 0.0001,
             "revised_outcome": {"test_accuracy": 0.83, "macro_f1": 0.83,
                                 "best_epoch": 18},
             "reason": "uniform", "justification": "permitted"},
        ],
    }), encoding="utf-8")

    text = "\n".join(report_part2.sections(artifacts))
    assert "collapsed" in text.lower()
    assert "crippled" not in text.lower()
