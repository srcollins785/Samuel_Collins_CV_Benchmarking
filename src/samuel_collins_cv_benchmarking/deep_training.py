"""The one training loop every deep architecture goes through.

Nine architectures, one loop. That is not tidiness - it is the reason the
numbers can be compared at all. An architecture with its own training path
would be timed by different code, stopped by a different rule and scored
against a different validation set, and none of those differences would be
visible in the results table. Part 1 made the same choice for the same reason
(see ``evaluation.evaluate_model``); this is its 224x224 counterpart.

Three things here are easy to get wrong and quiet when you do.

*Asynchronous devices make naive timing meaningless.* Both CUDA and MPS queue
work and return control immediately, so a ``perf_counter`` around a training
step measures how long it took to *submit* the work, not to do it. Every
epoch time in section 19 and every latency in section 20 would come out far
too low, and the result would look like extraordinary hardware rather than a
bug. Every timing boundary in this file is preceded by an explicit
synchronize.

*Section 12 selects on validation accuracy, not validation loss.* Part 1's
Simple CNN kept the weights with the lowest validation loss; section 12 asks
for the highest validation accuracy. Those two disagree in practice - loss
often keeps improving after accuracy has plateaued, and late in training loss
can rise while accuracy holds. The deep models follow section 12 because it is
specified, and both metrics are recorded every epoch so the report can state
plainly that the two model families used different selection criteria rather
than implying one rule governed everything.

*No early stopping.* Every architecture runs the full 20 epochs. Stopping
early would make the training-time column in section 19 measure "how quickly
this model stopped improving" rather than "what 20 epochs of this
architecture costs", and it would hide exactly the overfitting section 16
asks us to look for. Overfitting is meant to be read off the gap between the
training and validation curves, not suppressed before it can appear.
"""

import time
from pathlib import Path

import numpy as np

from ._config import (
    CNN_BATCH_SIZE,
    CNN_EPOCHS,
    CNN_LEARNING_RATE,
    CNN_WEIGHT_DECAY,
    RANDOM_SEED,
)


# -- device bookkeeping ------------------------------------------------------

def synchronize(device) -> None:
    """Wait for queued device work to finish.

    Called before reading any clock. On CPU it is a no-op because CPU work is
    already synchronous by the time the Python call returns.
    """
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device) -> None:
    """Start a fresh memory measurement.

    On CUDA this resets the true peak counter. On MPS there is no peak
    counter to reset, so the allocator cache is emptied instead - which is
    what makes the following measurement describe *this* model rather than
    everything the process has allocated so far.
    """
    import torch

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


def device_memory_mb(device) -> float:
    """Device memory in MB, or 0.0 where the concept does not apply.

    CUDA reports a true peak through ``max_memory_allocated``. MPS has no
    peak-tracking API in torch 2.8, so this reports live tensor allocation
    and the caller samples it repeatedly, keeping the maximum.

    It deliberately reports ``current_allocated_memory`` rather than
    ``driver_allocated_memory``. The driver figure is the process-wide
    allocator pool: it grows as models are loaded and is never handed back,
    so reading it once per architecture in a nine-architecture run produces a
    column that increases monotonically in run order and describes the run's
    history rather than any model's footprint. The first attempt at this did
    exactly that - DenseNet121 appeared to need 12.7 GB, which was simply
    everything allocated before it. Live allocation, measured after emptying
    the cache, describes the model actually in memory.

    It remains a sampled maximum rather than a true peak, and it is labeled
    as such everywhere it is reported. Section 21 asks for CUDA peak memory
    and this host has no CUDA device; inventing a number in its place would
    be worse than saying what was measured instead.
    """
    import torch

    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e6
    return 0.0


def memory_measurement_kind(device) -> str:
    """How the memory number in the results should be read."""
    if device.type == "cuda":
        return "cuda_peak_allocated"
    if device.type == "mps":
        return "mps_live_allocation_sampled_maximum"
    return "not_applicable_cpu"


# -- one pass over a loader --------------------------------------------------

def _assert_labels_valid(labels, class_count: int, device) -> None:
    """Fail loudly if a label tensor did not survive the trip to the device.

    Cross-entropy indexes its target directly. On CPU an out-of-range target
    raises; on MPS it reads out of bounds and returns a number, so a corrupt
    label tensor shows up as a loss and an accuracy that look real. This
    turns that class of failure back into an exception.
    """
    import torch

    low = int(torch.min(labels))
    high = int(torch.max(labels))
    if low < 0 or high >= class_count:
        raise RuntimeError(
            f"Label values out of range on {device}: saw [{low}, {high}] for a "
            f"{class_count}-class problem. The label tensor did not arrive "
            "intact - this is what an asynchronous host-to-device copy of "
            "non-pinned memory looks like. Training against these values "
            "would produce a model and a metric that are both meaningless."
        )


def _run_epoch(model, loader, criterion, device, optimizer=None) -> tuple:
    """One pass: training when given an optimizer, evaluation when not.

    Returns ``(mean_loss, accuracy)``. Sharing this between the training and
    validation passes means the two losses are computed identically and the
    curves in section 16 are directly comparable to each other.
    """
    import torch

    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, seen = 0.0, 0, 0
    checked_labels = False
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels in loader:
            # Blocking copies, deliberately. non_blocking=True is only safe
            # from pinned host memory, and these tensors come from an
            # in-memory uint8 cache that is not pinned. On MPS the
            # asynchronous path returned int64 label tensors whose contents
            # had not landed yet - garbage indices like 6.8e18 against ten
            # classes. CrossEntropyLoss then read out of bounds, which CPU
            # raises on and MPS silently tolerates, so the loop trained
            # against corrupt targets and scored itself against them too.
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            # Checked once per pass, on the first batch only, so the cost is
            # negligible. A label outside [0, classes) means the tensor
            # reaching the device is not the tensor the loader produced;
            # without this the failure is a plausible-looking accuracy rather
            # than an error, which is far more expensive to find.
            if not checked_labels:
                _assert_labels_valid(labels, logits.shape[1], device)
                checked_labels = True

            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            # .item() forces a sync on its own, but the loss is needed as a
            # Python float for the history either way.
            total_loss += float(loss.item()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += len(labels)

    return total_loss / max(1, seen), correct / max(1, seen)


# -- the training function section 11 specifies -----------------------------

def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int = CNN_EPOCHS,
    learning_rate: float = CNN_LEARNING_RATE,
    device=None,
    checkpoint_path=None,
    on_epoch=None,
) -> dict:
    """Fine-tune one architecture and record everything section 11 asks for.

    Parameters
    ----------
    model
        From :func:`~.deep_models.get_model`, already on the target device or
        about to be moved there.
    train_loader, val_loader
        From :func:`~.deep_data.build_loaders`. ``val_loader`` comes out of
        the training half, never the test half.
    epochs
        Section 6 sets a minimum of 20 and this runs exactly that many.
    learning_rate
        Section 6 sets 0.001. Passed rather than read from config so an
        unstable architecture can be rerun at a documented lower rate.
    device
        Where to train. Defaults to the best available.
    checkpoint_path
        Where to write the best-validation-accuracy weights (section 12). The
        file's size on disk is what section 18 reports, so this is written
        even though the weights are also kept in memory.
    on_epoch
        Called with each epoch's record, for progress output during a run
        that takes minutes per architecture.

    Returns
    -------
    dict
        Per-epoch records, the selected epoch, total and mean training time,
        and the memory measurement with its kind.
    """
    import torch
    import torch.nn as nn

    from .deep_models import device_for

    device = device or device_for()
    # Seeded here as well as in the factory: dropout masks and any remaining
    # stochastic layer should replay identically between runs.
    torch.manual_seed(RANDOM_SEED)

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=CNN_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    reset_peak_memory(device)
    history = []
    best_accuracy, best_epoch, best_state = -1.0, None, None
    peak_memory = 0.0

    synchronize(device)
    training_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        synchronize(device)
        epoch_start = time.perf_counter()

        train_loss, train_accuracy = _run_epoch(
            model, train_loader, criterion, device, optimizer)

        # Validation runs every epoch because section 12 selects on it. It is
        # the training half's hold-out, so this does not touch the test set.
        validation_loss, validation_accuracy = _run_epoch(
            model, val_loader, criterion, device)

        synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start
        peak_memory = max(peak_memory, device_memory_mb(device))

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "epoch_seconds": epoch_seconds,
            # Recorded per epoch because section 11 asks for it. Constant
            # here - no scheduler - and a constant column is the honest way
            # to show that rather than omitting it.
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)

        # Section 12: keep the best validation accuracy, not the last epoch.
        # Strictly greater, so the earliest epoch reaching a given accuracy
        # wins and the choice does not depend on later ties.
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = {key: value.detach().clone().cpu()
                          for key, value in model.state_dict().items()}

        if on_epoch is not None:
            on_epoch(record)

    synchronize(device)
    total_seconds = time.perf_counter() - training_start

    # Restore the selected weights before the caller evaluates on the test
    # set. Without this, the test metrics would describe epoch 20 rather than
    # the model section 12 told us to select.
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_megabytes = None
    if checkpoint_path is not None and best_state is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_path)
        checkpoint_megabytes = round(checkpoint_path.stat().st_size / 1e6, 2)

    epoch_times = [record["epoch_seconds"] for record in history]
    return {
        "per_epoch": history,
        "epochs_run": len(history),
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy if best_epoch else None,
        "selection_criterion": "highest_validation_accuracy",
        "total_training_seconds": total_seconds,
        "mean_epoch_seconds": float(np.mean(epoch_times)) if epoch_times else None,
        "learning_rate": learning_rate,
        "optimizer": "AdamW",
        "weight_decay": CNN_WEIGHT_DECAY,
        "batch_size": CNN_BATCH_SIZE,
        "device": str(device),
        "peak_memory_mb": round(peak_memory, 1),
        "memory_measurement": memory_measurement_kind(device),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_megabytes": checkpoint_megabytes,
        "early_stopping": False,
    }


def predict(model, loader, device=None) -> tuple:
    """Predictions and ground truth for a whole loader, in loader order.

    Used on the test half *after* :func:`train_model` has restored the
    selected weights, which is the order section 12 requires: model selection
    finishes, then the test set is touched once.

    Returns
    -------
    tuple
        ``(predictions, truth)`` as int64 arrays.
    """
    import torch

    from .deep_models import device_for

    device = device or device_for()
    model = model.to(device).eval()

    predictions, truth = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            truth.append(labels.numpy())

    if not predictions:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return (np.concatenate(predictions).astype(np.int64),
            np.concatenate(truth).astype(np.int64))


def overfitting_signals(history: dict) -> dict:
    """Read section 16's diagnoses off the curves rather than by eye.

    Three quantities, each chosen to be checkable rather than impressionistic:

    *generalization gap* - final training accuracy minus final validation
    accuracy. Large and positive is the classic overfitting signature.

    *epochs since best* - how many epochs ran after the selected one. A model
    that peaked at epoch 4 and then ran 16 more was overfitting for most of
    its training budget.

    *validation loss rise* - final validation loss minus its minimum. Rising
    validation loss while training loss still falls is the textbook picture,
    and it is the signal that survives when accuracy is noisy.
    """
    records = history.get("per_epoch") or []
    if not records:
        return {}

    final = records[-1]
    validation_losses = [record["validation_loss"] for record in records]
    best_validation_loss = min(validation_losses)

    return {
        "generalization_gap": round(
            final["train_accuracy"] - final["validation_accuracy"], 4),
        "final_train_accuracy": round(final["train_accuracy"], 4),
        "final_validation_accuracy": round(final["validation_accuracy"], 4),
        "epochs_after_best": len(records) - (history.get("best_epoch") or len(records)),
        "validation_loss_rise_from_minimum": round(
            final["validation_loss"] - best_validation_loss, 4),
        "minimum_validation_loss": round(best_validation_loss, 4),
        "minimum_validation_loss_epoch": int(
            validation_losses.index(best_validation_loss) + 1),
    }
