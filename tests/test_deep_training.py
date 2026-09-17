"""Tests for the shared deep training loop.

The important test in this file is
``test_reported_accuracy_matches_an_independent_evaluation``, and it exists
because of a real bug that this suite did not originally catch.

The training loop moved its batches to the device with
``non_blocking=True``. That is only safe from pinned host memory, and the
image cache is not pinned. On MPS the asynchronous copy returned int64 label
tensors whose contents had not arrived - indices around 6.8e18 for a ten-class
problem. Cross-entropy indexes its target directly, so on CPU that raises an
IndexError, but on MPS it read out of bounds and returned a plausible number.
The loop therefore trained against corrupt targets *and* scored itself against
them, and the symptom was not a crash: it was a validation accuracy of 0.9667
for a model whose true accuracy was 0.1000, with a loss near zero.

That is the worst shape a bug can have. It fabricated a number that was better
than the truth, in the one place nobody re-checks, and every training curve and
every table in the report would have inherited it.

So the loop's own reported metric is now checked against an independent
evaluation of the same weights on the same data, and a label tensor that did
not survive the trip to the device raises instead of being trained on.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from samuel_collins_cv_benchmarking import deep_data, deep_models, deep_training

CLASSES = ["alpha", "beta", "gamma"]
PER_CLASS = 12


@pytest.fixture(scope="module")
def loaders(tmp_path_factory):
    """A small file-backed dataset, run through the real pipeline."""
    from PIL import Image

    from samuel_collins_cv_benchmarking.benchmark import make_split
    from samuel_collins_cv_benchmarking.data_loader import load_folder
    from samuel_collins_cv_benchmarking.preprocessing import preprocess

    root = tmp_path_factory.mktemp("training_images") / "images"
    generator = np.random.default_rng(0)
    for band, name in zip((35, 120, 205), CLASSES):
        (root / name).mkdir(parents=True)
        for number in range(PER_CLASS):
            pixels = np.clip(
                generator.normal(band, 10, (300, 300, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(root / name / f"{name}_{number:02d}.png")

    split = make_split(preprocess(load_folder(root, CLASSES), "rgb"))
    return deep_data.build_loaders(split, "rgb")


@pytest.fixture(scope="module")
def trained(loaders, tmp_path_factory):
    """One architecture trained briefly, with its history and checkpoint."""
    checkpoint = tmp_path_factory.mktemp("checkpoints") / "best_resnet18.pt"
    model = deep_models.get_model("resnet18", num_classes=len(CLASSES),
                                  pretrained=False)
    history = deep_training.train_model(
        model, loaders["train"], loaders["validation"],
        epochs=3, device=deep_models.device_for(),
        checkpoint_path=checkpoint)
    return model, history, checkpoint


# -- the regression test ----------------------------------------------------

def test_reported_accuracy_matches_an_independent_evaluation(trained, loaders):
    """The loop's own metric must survive an independent recount.

    Evaluated twice more: once on the training device and once on CPU, both
    outside the training loop. A loop that scores itself against corrupted
    labels passes neither. The tolerance is exact on the comparison itself -
    these are counts of correct predictions, not floating-point reductions -
    with a small allowance for device arithmetic changing an individual
    logit's argmax.
    """
    import copy

    model, history, _ = trained
    device = deep_models.device_for()

    def independent(module, loader, where):
        module = module.to(where).eval()
        correct = seen = 0
        with torch.no_grad():
            for images, labels in loader:
                predicted = module(images.to(where)).argmax(dim=1).cpu()
                correct += int((predicted == labels).sum())
                seen += len(labels)
        return correct / max(1, seen)

    on_device = independent(model, loaders["validation"], device)
    on_cpu = independent(copy.deepcopy(model).cpu(), loaders["validation"],
                         torch.device("cpu"))

    # The selected weights are restored before train_model returns, so the
    # best recorded validation accuracy is what these should reproduce.
    reported = history["best_validation_accuracy"]

    assert on_device == pytest.approx(reported, abs=0.02), (
        f"the loop reported {reported:.4f} but an independent evaluation on "
        f"{device} finds {on_device:.4f}")
    assert on_cpu == pytest.approx(reported, abs=0.02), (
        f"the loop reported {reported:.4f} but an independent evaluation on "
        f"cpu finds {on_cpu:.4f}")


def test_a_confident_model_cannot_have_collapsed_predictions(trained, loaders):
    """The exact shape of the original bug, as a conditional invariant.

    Training against corrupt targets produced a model that predicted one
    class for every input *while reporting 0.9667 validation accuracy*. Those
    two facts cannot both be true, and that contradiction is the thing worth
    asserting.

    Deliberately not an unconditional "predictions must be diverse": this
    fixture trains a randomly initialized network for three epochs on a few
    dozen synthetic images, and a genuinely weak model collapsing onto one
    class is a legitimate outcome there. Asserting otherwise would make the
    test fail for a reason that has nothing to do with the bug it guards.
    """
    model, history, _ = trained
    predictions, truth = deep_training.predict(
        model, loaders["test"], deep_models.device_for())

    assert len(predictions) == len(truth)

    chance = 1.0 / len(CLASSES)
    reported = history["best_validation_accuracy"]
    if reported > chance * 1.5:
        assert len(np.unique(predictions)) > 1, (
            f"the loop reported {reported:.4f} validation accuracy, well "
            f"above the {chance:.4f} chance level, yet every test image "
            "received the same label - a model cannot be both accurate and "
            "collapsed, so one of these numbers is fabricated"
        )


# -- the guard ---------------------------------------------------------------

def test_guard_rejects_labels_that_did_not_survive_the_copy():
    corrupted = torch.tensor([0, 1, 6_815_625_949_482_245_429], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="out of range"):
        deep_training._assert_labels_valid(corrupted, 10, torch.device("cpu"))


def test_guard_rejects_negative_labels():
    with pytest.raises(RuntimeError, match="out of range"):
        deep_training._assert_labels_valid(
            torch.tensor([-1, 0, 1]), 10, torch.device("cpu"))


def test_guard_accepts_the_full_valid_range():
    """A label equal to classes - 1 is valid and must not trip the guard."""
    deep_training._assert_labels_valid(
        torch.tensor([0, 9]), 10, torch.device("cpu"))


def test_no_asynchronous_device_copies_remain():
    """No ``non_blocking`` copy may reappear in the deep modules.

    A source-level assertion because the failure it prevents is invisible at
    runtime: the copy succeeds, the numbers look reasonable, and only a
    line-by-line recount reveals that they are wrong. The comment explaining
    the hazard is allowed to mention it; a live call is not.
    """
    package = Path(deep_training.__file__).parent
    offenders = []
    for path in sorted(package.glob("deep_*.py")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if "non_blocking" in stripped and not stripped.startswith("#"):
                offenders.append(f"{path.name}:{number}")
    assert not offenders, f"asynchronous device copies found at {offenders}"


# -- section 11: what the history records -----------------------------------

def test_history_records_every_field_section_11_requires(trained):
    _, history, _ = trained
    required = {"epoch", "train_loss", "validation_loss", "train_accuracy",
                "validation_accuracy", "epoch_seconds", "learning_rate"}
    for record in history["per_epoch"]:
        assert required <= set(record), f"missing {required - set(record)}"


def test_every_requested_epoch_runs(trained):
    """No early stopping, so training time stays comparable across models."""
    _, history, _ = trained
    assert history["epochs_run"] == history["epochs_requested"] == 3
    assert history["early_stopping"] is False


def test_epochs_are_numbered_from_one_and_timed(trained):
    _, history, _ = trained
    assert [r["epoch"] for r in history["per_epoch"]] == [1, 2, 3]
    assert all(r["epoch_seconds"] > 0 for r in history["per_epoch"])


# -- section 12: model selection --------------------------------------------

def test_selected_epoch_is_the_best_validation_accuracy(trained):
    """Section 12 selects on validation accuracy, not the final epoch."""
    _, history, _ = trained
    accuracies = [r["validation_accuracy"] for r in history["per_epoch"]]
    assert history["best_validation_accuracy"] == max(accuracies)
    assert history["best_epoch"] == accuracies.index(max(accuracies)) + 1
    assert history["selection_criterion"] == "highest_validation_accuracy"


def test_checkpoint_is_written_and_measured(trained):
    """Section 18 reports the checkpoint's size, so the file has to exist."""
    _, history, checkpoint = trained
    assert Path(checkpoint).is_file()
    assert history["checkpoint_megabytes"] > 0
    assert history["checkpoint_megabytes"] == pytest.approx(
        Path(checkpoint).stat().st_size / 1e6, abs=0.01)


def test_checkpoint_holds_the_selected_weights(trained):
    """The saved file must be the selected epoch, not the last one."""
    model, history, checkpoint = trained
    saved = torch.load(checkpoint, map_location="cpu")
    current = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    assert set(saved) == set(current)
    for key, value in saved.items():
        assert torch.allclose(value.float(), current[key].float(), atol=1e-5), key


# -- section 19 and 21: cost bookkeeping ------------------------------------

def test_training_time_is_recorded_in_both_forms(trained):
    """Section 19 asks for seconds per epoch and the total."""
    _, history, _ = trained
    assert history["total_training_seconds"] > 0
    assert history["mean_epoch_seconds"] > 0
    assert history["mean_epoch_seconds"] == pytest.approx(
        float(np.mean([r["epoch_seconds"] for r in history["per_epoch"]])))


def test_memory_measurement_is_labeled_not_implied(trained):
    """Section 21 wants CUDA peak memory; this host has none, so say which."""
    _, history, _ = trained
    assert history["memory_measurement"] in {
        "cuda_peak_allocated",
        "mps_live_allocation_sampled_maximum",
        "not_applicable_cpu",
    }


def test_protocol_settings_are_recorded_with_the_run(trained):
    _, history, _ = trained
    assert history["optimizer"] == "AdamW"
    assert history["batch_size"] == 64
    assert history["learning_rate"] > 0


# -- section 16: the overfitting read-out -----------------------------------

def test_overfitting_signals_are_computed_from_the_curves(trained):
    _, history, _ = trained
    signals = deep_training.overfitting_signals(history)

    final = history["per_epoch"][-1]
    assert signals["generalization_gap"] == pytest.approx(
        round(final["train_accuracy"] - final["validation_accuracy"], 4))
    assert signals["epochs_after_best"] == 3 - history["best_epoch"]
    assert signals["validation_loss_rise_from_minimum"] >= 0
    assert 1 <= signals["minimum_validation_loss_epoch"] <= 3


def test_overfitting_signals_on_empty_history():
    assert deep_training.overfitting_signals({}) == {}
    assert deep_training.overfitting_signals({"per_epoch": []}) == {}
