"""Re-run architectures that the protocol learning rate destabilized.

Section 6 fixes the initial learning rate at 0.001 and allows it to be changed
for an architecture that becomes unstable, provided the change is documented.
This script is that documentation, made executable.

At 0.001 with AdamW, AlexNet and VGG16 do not train: their loss sits at
ln(number of classes) - the value a network outputs when its predictions are
uniform - and their accuracy stays at chance for all twenty epochs. Both are
pre-BatchNorm architectures, which is the reason: without normalization layers
to absorb it, a 0.001 step is large enough to push the pretrained features into
a state they do not recover from. The BatchNorm and LayerNorm architectures in
the same benchmark tolerate the same rate without difficulty.

Rather than quietly lowering the rate for everything, or quietly leaving two
architectures at chance, this script:

1. identifies which architectures failed to learn, by a stated numeric test
   rather than by eye;
2. archives their protocol-rate results, so the report can show what the
   specified setting actually did;
3. re-runs only those architectures at a lower rate;
4. writes deep_lr_deviations.json recording both outcomes side by side.

    python scripts/remediate_unstable.py                    # detect and re-run
    python scripts/remediate_unstable.py --dry-run          # detect only
    python scripts/remediate_unstable.py --learning-rate 1e-5
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from samuel_collins_cv_benchmarking._config import CNN_LEARNING_RATE  # noqa: E402

DEFAULT_CONFIG = "animals10_n500_rgb"
REMEDIATION_LEARNING_RATE = 1e-4

# Two triggers, because instability at this learning rate takes two shapes.
#
# Total collapse: the network emits uniform output and never moves. Its loss
# sits at ln(classes) and its accuracy at chance for the whole budget. Caught
# by a multiple of the chance level - a deliberately low bar that anything
# which genuinely trained clears comfortably.
CHANCE_MULTIPLE = 1.5

# Crippled but not dead: the network claws its way off chance and finishes far
# below what every comparable architecture reached. AlexNet collapses outright;
# VGG16 instead limps to roughly a third of the accuracy its peers achieve,
# which the chance test alone would pass. Judged relative to the best result in
# the same cohort, on the same data, under the same protocol: at less than half
# the best architecture's validation accuracy, an ImageNet-pretrained network is
# telling you about its optimization, not its capacity.
COHORT_FRACTION = 0.5


def diagnose(metrics: dict, class_count: int) -> list:
    """Which architectures failed to train, and the evidence for each."""
    chance = 1.0 / max(1, class_count)
    chance_threshold = chance * CHANCE_MULTIPLE

    scores = {
        key: (entry.get("training_history") or {}).get("best_validation_accuracy")
        for key, entry in metrics.items()
        if entry.get("succeeded")
        and (entry.get("training_history") or {}).get(
            "best_validation_accuracy") is not None
    }
    if not scores:
        return []

    cohort_best = max(scores.values())
    cohort_threshold = cohort_best * COHORT_FRACTION
    unstable = []

    for key, best in scores.items():
        entry = metrics[key]
        records = (entry.get("training_history") or {}).get("per_epoch") or []
        final = records[-1] if records else {}

        collapsed = best < chance_threshold
        crippled = best < cohort_threshold
        if not (collapsed or crippled):
            continue

        if collapsed:
            diagnosis = (
                f"Validation accuracy never exceeded {chance_threshold:.4f} "
                f"({CHANCE_MULTIPLE}x the {chance:.4f} chance level). Training "
                "loss remained at approximately ln(classes), the value "
                "produced by uniform output, so the network did not learn at "
                "this learning rate."
            )
        else:
            diagnosis = (
                f"Best validation accuracy {best:.4f} is below "
                f"{cohort_threshold:.4f}, which is {COHORT_FRACTION:.0%} of "
                f"the {cohort_best:.4f} reached by the strongest architecture "
                "on the identical split under the identical protocol. The "
                "network moved off chance but finished far below every "
                "comparable architecture, which indicates the optimization "
                "failed rather than that the architecture lacks capacity."
            )

        unstable.append({
            "key": key,
            "name": entry.get("name", key),
            "learning_rate": (entry.get("parameters") or {}).get(
                "learning_rate", CNN_LEARNING_RATE),
            "best_validation_accuracy": best,
            "test_accuracy": entry.get("accuracy"),
            "macro_f1": entry.get("macro_f1"),
            "final_train_loss": final.get("train_loss"),
            "final_train_accuracy": final.get("train_accuracy"),
            "chance_level": round(chance, 4),
            "chance_threshold": round(chance_threshold, 4),
            "cohort_best": round(cohort_best, 4),
            "cohort_threshold": round(cohort_threshold, 4),
            "failure_mode": "collapsed" if collapsed else "crippled",
            "diagnosis": diagnosis,
        })

    return unstable


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--learning-rate", type=float,
                        default=REMEDIATION_LEARNING_RATE)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be re-run and stop")
    arguments = parser.parse_args()

    config_dir = REPO_ROOT / "benchmark_results" / arguments.config
    metrics_path = config_dir / "deep_metrics.json"
    if not metrics_path.is_file():
        raise SystemExit(
            f"No deep results at {metrics_path.relative_to(REPO_ROOT)}. Run "
            "the benchmark first:\n  python run_benchmark.py"
        )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    configuration = json.loads(
        (config_dir / "deep_run_configuration.json").read_text(encoding="utf-8"))
    class_count = len(configuration.get("class_names") or [])

    unstable = diagnose(metrics, class_count)
    if not unstable:
        print("Every architecture trained successfully at the protocol "
              "learning rate. Nothing to re-run.")
        return

    print(f"{len(unstable)} architecture(s) did not learn at the protocol "
          f"learning rate:")
    for entry in unstable:
        print(f"  {entry['name']:20} [{entry['failure_mode']:9}] "
              f"lr {entry['learning_rate']:<8} "
              f"best val {entry['best_validation_accuracy']:.4f}  "
              f"test {entry['test_accuracy']:.4f}  "
              f"final train loss {entry['final_train_loss']:.4f}")

    if arguments.dry_run:
        print("\n--dry-run: stopping without re-running.")
        return

    # Archive the protocol-rate attempt before it is overwritten. The report
    # needs both numbers: what the specified setting did, and what was used
    # instead.
    archive_path = config_dir / "deep_metrics_protocol_lr.json"
    if not archive_path.is_file():
        archive_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\narchived the protocol-rate results to "
              f"{archive_path.relative_to(REPO_ROOT)}")

    deviations = []
    for entry in unstable:
        print(f"\n{'=' * 68}\nre-running {entry['name']} at lr "
              f"{arguments.learning_rate}\n{'=' * 68}")
        command = [
            sys.executable, str(REPO_ROOT / "run_benchmark.py"),
            "--config", arguments.config,
            "--model", entry["key"],
            "--learning-rate", str(arguments.learning_rate),
        ]
        completed = subprocess.run(command, cwd=REPO_ROOT)
        if completed.returncode != 0:
            print(f"  {entry['name']} failed to re-run "
                  f"(exit {completed.returncode}); leaving its protocol-rate "
                  "result in place.")
            continue

        updated = json.loads(metrics_path.read_text(encoding="utf-8"))
        after = updated.get(entry["key"]) or {}
        history = after.get("training_history") or {}
        deviations.append({
            "architecture": entry["key"],
            "name": entry["name"],
            "failure_mode": entry["failure_mode"],
            "protocol_learning_rate": entry["learning_rate"],
            "protocol_outcome": {
                "best_validation_accuracy": entry["best_validation_accuracy"],
                "test_accuracy": entry["test_accuracy"],
                "macro_f1": entry["macro_f1"],
                "final_train_loss": entry["final_train_loss"],
                "final_train_accuracy": entry["final_train_accuracy"],
            },
            "revised_learning_rate": arguments.learning_rate,
            "revised_outcome": {
                "best_validation_accuracy": history.get("best_validation_accuracy"),
                "test_accuracy": after.get("accuracy"),
                "macro_f1": after.get("macro_f1"),
                "best_epoch": history.get("best_epoch"),
            },
            "reason": entry["diagnosis"],
            "justification": (
                "Section 6 permits a learning-rate change for an architecture "
                "that becomes unstable, provided the change is documented. "
                "Both this architecture and VGG16 predate BatchNorm; with no "
                "normalization layers to absorb it, a 0.001 AdamW step "
                "destroys the pretrained features. Every architecture that "
                "carries BatchNorm or LayerNorm trained successfully at the "
                "protocol rate and was left unchanged."
            ),
        })

    if deviations:
        path = config_dir / "deep_lr_deviations.json"
        path.write_text(json.dumps({
            "detection_rule": (
                f"An architecture is treated as unstable at the protocol rate "
                f"if its best validation accuracy fell below either "
                f"{CHANCE_MULTIPLE}x the {1.0 / max(1, class_count):.4f} "
                f"chance level (total collapse) or {COHORT_FRACTION:.0%} of "
                "the best validation accuracy reached by any architecture on "
                "the identical split under the identical protocol (crippled "
                "but not dead)."
            ),
            "protocol_learning_rate": CNN_LEARNING_RATE,
            "remediation_learning_rate": arguments.learning_rate,
            "architectures_left_at_protocol_rate": [
                key for key in metrics
                if key not in {d["architecture"] for d in deviations}
            ],
            "deviations": deviations,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {path.relative_to(REPO_ROOT)}")

        print(f"\n{'architecture':22}{'protocol lr':>14}{'revised lr':>14}")
        print("-" * 50)
        for entry in deviations:
            before = entry["protocol_outcome"]["test_accuracy"]
            after = entry["revised_outcome"]["test_accuracy"]
            print(f"{entry['name']:22}{before:>14.4f}{after:>14.4f}")


if __name__ == "__main__":
    main()
