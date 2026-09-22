"""Measure the parameter count and stored size of the two neural baselines.

Section 23's table marks N/A in the parameter column for the four traditional
models only. The fully connected network and the Simple CNN are both
parameterized networks and belong in that column with real numbers; the first
version of the combined table reported all six Part 1 models as N/A, which
generalized a statement that is true of a Random Forest to two models it is
false of.

Part 1 did not record either quantity, so they are measured here rather than
recovered. The models are refitted on the same split at the same seed, which
reproduces them exactly - the parameter count is fixed by the architecture and
the input dimension regardless, and the serialized size follows from it.

Written to its own artifact rather than merged into benchmark_metrics.json.
That file is Part 1's published record, the combined table checks its own
derivations against it, and appending to it would make a verified artifact
depend on a later run.

What "size" means here is worth stating. The Simple CNN is measured the way
section 18 measures every deep architecture, as a saved state dict. The MLP is
measured as the whole fitted pipeline including its StandardScaler, because the
scaler holds a mean and a scale for each of the 12,288 input features and you
cannot deploy the classifier without it. Reporting the bare classifier would
understate what the model actually costs to ship.

    python scripts/measure_baselines.py
"""

import argparse
import json
import pickle
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

DEFAULT_CONFIG = "animals10_n500_rgb"


def mlp_parameter_count(pipeline) -> int:
    """Weights plus biases across every layer of a fitted MLPClassifier."""
    classifier = pipeline.named_steps["classifier"]
    return (sum(w.size for w in classifier.coefs_)
            + sum(b.size for b in classifier.intercepts_))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    arguments = parser.parse_args()

    from run_benchmark import parse_config, rebuild_split, verify_against_part1
    from samuel_collins_cv_benchmarking.neural_models import (
        neural_network_spec,
        simple_cnn_spec,
    )

    tier, color_mode = parse_config(arguments.config)
    config_dir = REPO_ROOT / "benchmark_results" / arguments.config

    print(f"configuration: {arguments.config}")
    split = rebuild_split(tier, color_mode)
    verified = verify_against_part1(split, config_dir)
    print(f"  split verified: {verified['test_positions_checked']} test "
          f"positions match")

    measured = {}

    # -- the fully connected network ----------------------------------------
    spec = neural_network_spec()
    model = spec()
    model.fit(split.features_train, split.labels_train)
    total = mlp_parameter_count(model)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=True) as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        size = Path(handle.name).stat().st_size / 1e6
    measured[spec.key] = {
        "name": spec.name,
        "total_parameters": int(total),
        # Nothing is frozen: every weight is fitted.
        "trainable_parameters": int(total),
        "frozen_parameters": 0,
        "total_parameters_millions": round(total / 1e6, 3),
        "size_megabytes": round(size, 2),
        "size_measured_as": (
            "the pickled fitted pipeline, scaler included - the scaler holds a "
            "mean and a scale per input feature and the classifier cannot be "
            "deployed without it"
        ),
        "architecture": "12288 - 128 - 64 - 10 fully connected",
    }
    print(f"  {spec.name:20} {total:>10,} parameters   {size:6.2f} MB")

    # -- the simple CNN ------------------------------------------------------
    import torch

    spec = simple_cnn_spec()
    cnn = spec()
    cnn.fit(split.images_train, split.labels_train)
    total = sum(p.numel() for p in cnn.model_.parameters())
    trainable = sum(p.numel() for p in cnn.model_.parameters() if p.requires_grad)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as handle:
        torch.save(cnn.model_.state_dict(), handle.name)
        size = Path(handle.name).stat().st_size / 1e6
    measured[spec.key] = {
        "name": spec.name,
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
        "total_parameters_millions": round(total / 1e6, 3),
        "size_megabytes": round(size, 2),
        "size_measured_as": (
            "a saved state dict, the same way section 18 measures every deep "
            "architecture"
        ),
        "architecture": (
            "conv32-pool-conv64-pool-dense128-dropout0.30-linear10 at 64x64"
        ),
    }
    print(f"  {spec.name:20} {total:>10,} parameters   {size:6.2f} MB")

    path = config_dir / "baseline_model_sizes.json"
    path.write_text(json.dumps({
        "note": (
            "Measured by scripts/measure_baselines.py. Part 1 did not record "
            "parameter counts or model sizes; these two models are "
            "parameterized networks and belong in section 23's parameter "
            "column, where the four traditional models correctly read N/A. "
            "Written separately so Part 1's published benchmark_metrics.json "
            "stays the record it was."
        ),
        "models": measured,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
