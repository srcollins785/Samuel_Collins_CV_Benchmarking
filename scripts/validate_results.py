"""Audit the published results for internal consistency.

Written after two errors got through the first round of checking, both of the
same kind. The original verification asserted that fields were present and
positive: `total_parameters > 0` across the deep architectures. That passes
happily while `trainable_parameters` is zero, which is what YOLO was recording,
and it says nothing at all about models it does not enumerate, which is how the
two neural baselines ended up with no parameter count.

So the checks here are about whether numbers are *possible*, not whether they
exist. A field can be present, positive, and still impossible next to the field
beside it. Most of what follows is an identity that has to hold between two
recorded quantities, and the useful ones are the identities that are cheap to
state and hard to satisfy by accident:

  weighted recall == accuracy            exactly, for single-label multiclass
  confusion matrix total == test images  the matrix covers the whole test half
  trainable + frozen == total            and trainable > 0 if the model trained
  throughput == 1000 / latency           they are derived from one measurement
  peak memory >= weights                 a forward pass cannot use less
  sum(epoch_seconds) ~= total time       the clock and the epochs agree
  best_epoch is argmax(validation)       selection did what it claims

    python scripts/validate_results.py
    python scripts/validate_results.py --config animals10_n500_rgb
"""

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_CONFIG = "animals10_n500_rgb"

# Published complexity figures, to catch a profiler that silently mismeasures.
# Tolerance is loose: these are the literature's numbers for the same
# architectures at 224x224, and small differences are expected from the
# replaced head and from what each profiler counts.
PUBLISHED_GMACS = {
    "alexnet": 0.71, "vgg16": 15.5, "googlenet": 1.5, "resnet18": 1.8,
    "resnet50": 4.1, "densenet121": 2.9, "mobilenet_v3_large": 0.22,
    "efficientnet_b0": 0.39, "convnext_tiny": 4.5,
}

TRADITIONAL = {"Logistic Regression", "Decision Tree", "Random Forest", "SVM"}
PARAMETERIZED_BASELINES = {"Neural Network", "Simple CNN"}


class Audit:
    def __init__(self):
        self.failures = []
        self.passes = 0

    def check(self, ok, label, detail=""):
        if ok:
            self.passes += 1
        else:
            self.failures.append((label, detail))

    def close(self, ok, label, detail=""):
        self.check(ok, label, detail)

    def report(self):
        print(f"\n{self.passes} checks passed, {len(self.failures)} failed")
        for label, detail in self.failures:
            print(f"  FAIL  {label}")
            if detail:
                print(f"        {detail}")
        return not self.failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    arguments = parser.parse_args()

    d = REPO_ROOT / "benchmark_results" / arguments.config
    a = Audit()

    deep = json.loads((d / "deep_metrics.json").read_text())
    part1 = json.loads((d / "benchmark_metrics.json").read_text())
    conf = json.loads((d / "deep_run_configuration.json").read_text())
    rankings = json.loads((d / "rankings.json").read_text())
    combined = pd.read_csv(d / "combined_ml_cnn_benchmark_results.csv",
                           keep_default_na=False)

    classes = conf["class_names"]
    n_test = conf["split"]["testing_samples"]

    # -- metric identities, per model ---------------------------------------
    for source, name in ((deep, "deep"), (part1, "part1")):
        for key, e in source.items():
            if not e.get("succeeded"):
                continue
            label = e.get("name", key)
            cm = np.asarray(e.get("confusion_matrix") or [], dtype=float)
            if cm.size:
                a.check(int(cm.sum()) == n_test,
                        f"{label}: confusion matrix covers the test half",
                        f"matrix totals {int(cm.sum())}, test half is {n_test}")
                acc = float(np.trace(cm) / cm.sum())
                a.check(math.isclose(acc, e["accuracy"], abs_tol=1e-9),
                        f"{label}: accuracy matches its confusion matrix",
                        f"recorded {e['accuracy']:.6f}, matrix gives {acc:.6f}")

            # weighted recall is accuracy, exactly, for single-label problems.
            if e.get("weighted_recall") is not None:
                a.check(math.isclose(e["weighted_recall"], e["accuracy"],
                                     abs_tol=1e-9),
                        f"{label}: weighted recall equals accuracy",
                        f"recall {e['weighted_recall']:.6f} vs accuracy "
                        f"{e['accuracy']:.6f}")

            for m in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                      "weighted_precision", "weighted_recall", "weighted_f1"):
                v = e.get(m)
                if v is not None:
                    a.check(0.0 <= v <= 1.0, f"{label}: {m} within [0,1]",
                            f"got {v}")

    # -- parameter accounting ------------------------------------------------
    for key, e in deep.items():
        if not e.get("succeeded"):
            continue
        label = e.get("name", key)
        pc = e.get("parameter_counts") or {}
        total, trainable = pc.get("total_parameters"), pc.get("trainable_parameters")
        frozen = pc.get("frozen_parameters")
        a.check(bool(total), f"{label}: total parameters recorded", f"got {total}")
        if total:
            a.check(bool(trainable) and trainable > 0,
                    f"{label}: trainable parameters are non-zero",
                    f"trainable={trainable} against total={total:,}; a model "
                    "that trained cannot have zero trainable parameters")
            if trainable is not None and frozen is not None:
                a.check(trainable + frozen == total,
                        f"{label}: trainable + frozen == total",
                        f"{trainable} + {frozen} != {total}")
            a.check(trainable is not None and trainable <= total,
                    f"{label}: trainable does not exceed total")

    # -- the neural baselines are parameterized and must report it ----------
    for _, row in combined.iterrows():
        model = row["Model"]
        if model in PARAMETERIZED_BASELINES:
            a.check(row["Total Parameters"] != "N/A",
                    f"{model}: parameter count reported",
                    "a fully connected network and a CNN both have a "
                    "parameter count; N/A belongs to the four traditional "
                    "models only, which is what the section 23 template marks")
            a.check(row["Size MB"] != "N/A",
                    f"{model}: model size reported")
        if model in TRADITIONAL:
            a.check(row["Total Parameters"] == "N/A",
                    f"{model}: parameter count is N/A, as it should be")

    # -- cost measurements ---------------------------------------------------
    for key, e in deep.items():
        if not e.get("succeeded"):
            continue
        label = e.get("name", key)
        inf = e.get("inference") or {}
        lat, thr = inf.get("latency_ms_per_image"), inf.get("throughput_images_per_second")
        if lat and thr:
            a.check(math.isclose(thr, 1000.0 / lat, rel_tol=0.02),
                    f"{label}: throughput is the reciprocal of latency",
                    f"{thr} vs {1000.0/lat:.2f}")
        if inf.get("images"):
            a.check(inf["images"] >= 1000,
                    f"{label}: inference timed over at least 1000 images",
                    f"got {inf['images']}")

        mem = e.get("memory") or {}
        if mem:
            a.check(mem["peak_memory_mb"] >= mem["weights_baseline_mb"],
                    f"{label}: peak memory is at least the weight footprint")
            a.check(mem["activation_working_set_mb"] > 0,
                    f"{label}: activation working set is positive")

        h = e.get("training_history") or {}
        per_epoch = h.get("per_epoch") or []
        if per_epoch:
            a.check(len(per_epoch) == h.get("epochs_run"),
                    f"{label}: epochs_run matches the recorded epochs")
            secs = sum(r.get("epoch_seconds") or 0 for r in per_epoch)
            total_s = h.get("total_training_seconds")
            if total_s and secs:
                a.check(secs <= total_s * 1.15,
                        f"{label}: epoch times fit inside the total",
                        f"epochs sum to {secs:.1f}s, total recorded {total_s:.1f}s")
            accs = [r.get("validation_accuracy") for r in per_epoch
                    if r.get("validation_accuracy") is not None]
            if accs and h.get("best_epoch"):
                best_idx = accs.index(max(accs)) + 1
                a.check(h["best_epoch"] == best_idx,
                        f"{label}: best_epoch is the argmax of validation accuracy",
                        f"recorded {h['best_epoch']}, argmax is {best_idx}")

        cp = e.get("checkpoint_megabytes")
        path = d / "checkpoints" / f"best_{key}.pt"
        if cp and path.is_file():
            actual = path.stat().st_size / 1e6
            a.check(math.isclose(cp, actual, rel_tol=0.02),
                    f"{label}: checkpoint size matches the file",
                    f"recorded {cp} MB, file is {actual:.2f} MB")

        gm = (e.get("complexity") or {}).get("gmacs")
        if gm and key in PUBLISHED_GMACS:
            a.check(math.isclose(gm, PUBLISHED_GMACS[key], rel_tol=0.15),
                    f"{label}: GMACs near the published figure",
                    f"measured {gm}, published about {PUBLISHED_GMACS[key]}")

    # -- the combined table and the rankings agree ---------------------------
    ok_models = {e.get("name", k) for k, e in {**part1, **deep}.items()
                 if e.get("succeeded")}
    table_models = set(combined[combined["Status"] == "ok"]["Model"])
    a.check(ok_models == table_models,
            "every successful model appears in the master table",
            f"missing from table: {sorted(ok_models - table_models)}; "
            f"extra: {sorted(table_models - ok_models)}")

    acc_rank = [r["model"] for r in rankings["A_highest_accuracy"]["ranking"]]
    a.check(len(acc_rank) == len(table_models),
            "the accuracy ranking covers every successful model",
            f"{len(acc_rank)} ranked against {len(table_models)} models")
    values = [r["value"] for r in rankings["A_highest_accuracy"]["ranking"]]
    a.check(values == sorted(values, reverse=True),
            "the accuracy ranking is ordered")

    # -- figures the report links to must exist -----------------------------
    report = (REPO_ROOT / "report" / "CV_Benchmarking_Report.md").read_text()
    import re
    missing = []
    for rel in re.findall(r"\]\((\.\./benchmark_results/[^)]+)\)", report):
        if not (REPO_ROOT / "report" / rel).resolve().is_file():
            missing.append(rel)
    a.check(not missing, "every figure the report links to exists",
            f"missing: {missing[:4]}")

    ok = a.report()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
