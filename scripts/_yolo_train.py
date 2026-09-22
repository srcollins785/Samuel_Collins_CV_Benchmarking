"""Train and predict with YOLO classification. Runs in .venv-yolo only.

Deliberately minimal. It knows nothing about the benchmark - it is handed a
directory of images already split into train/val/test, trains on it, and writes
one prediction per test image. Every metric is then computed back in the main
environment by the same code that scores the other nine architectures, so the
comparison does not depend on ultralytics and this package agreeing about how
to average an F1 score.

Invoked by scripts/run_yolo.py; not intended to be run by hand.
"""

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="dataset root with train/val/test")
    parser.add_argument("--output", required=True, help="where to write predictions JSON")
    parser.add_argument("--model", default="yolo11n-cls.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", required=True, help="ultralytics run directory")
    arguments = parser.parse_args()

    import torch
    from ultralytics import YOLO

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = 0

    data = Path(arguments.data)
    model = YOLO(arguments.model)

    started = time.perf_counter()
    model.train(
        data=str(data),
        epochs=arguments.epochs,
        batch=arguments.batch,
        imgsz=arguments.imgsz,
        # Aligned with the shared protocol in section 6 as far as ultralytics
        # allows. Its augmentation defaults still differ from the torchvision
        # pipeline, which is recorded as a deviation rather than hidden.
        optimizer="AdamW",
        lr0=arguments.lr0,
        seed=arguments.seed,
        device=device,
        workers=0,
        pretrained=True,
        project=arguments.project,
        name="train",
        exist_ok=True,
        plots=False,
        verbose=True,
    )
    training_seconds = time.perf_counter() - started

    weights = Path(arguments.project) / "train" / "weights" / "best.pt"
    trained = YOLO(str(weights))

    # Class index order as ultralytics learned it, so the caller can map names
    # back to its own label encoding rather than assuming the two agree.
    names = trained.names
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]

    test_files = sorted(
        path for path in (data / "test").rglob("*")
        if path.is_file() and not path.name.startswith("."))

    # Timed separately from training, after a warm-up, matching how the other
    # architectures are measured in section 20.
    for path in test_files[:3]:
        trained.predict(str(path), imgsz=arguments.imgsz, device=device, verbose=False)

    predictions = {}
    inference_started = time.perf_counter()
    batch_size = 64
    for start in range(0, len(test_files), batch_size):
        chunk = test_files[start:start + batch_size]
        results = trained.predict([str(p) for p in chunk], imgsz=arguments.imgsz,
                                  device=device, verbose=False)
        for path, result in zip(chunk, results):
            predictions[path.name] = names[int(result.probs.top1)]
    inference_seconds = time.perf_counter() - inference_started

    # Counted from the training configuration, not from requires_grad on the
    # reloaded model. ultralytics returns best.pt in inference mode with grads
    # disabled, so counting requires_grad there reports zero trainable
    # parameters for a network that was fully fine-tuned - which is what this
    # script did originally. No freeze argument is passed to train(), so every
    # layer was optimized; the per-group learning rates in results.csv confirm
    # it, all three decaying across all twenty epochs.
    parameters = sum(p.numel() for p in trained.model.parameters())
    froze_layers = False  # no freeze= argument is passed to model.train()
    trainable = 0 if froze_layers else parameters

    Path(arguments.output).write_text(json.dumps({
        "predictions": predictions,
        "class_names": list(names),
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "images": len(test_files),
        "total_parameters": int(parameters),
        "trainable_parameters": int(trainable),
        "trainable_counted_from": (
            "training configuration - no freeze argument was passed, so every "
            "layer was optimized. requires_grad on the reloaded best.pt reads "
            "zero because ultralytics returns it in inference mode."
        ),
        "checkpoint_bytes": weights.stat().st_size if weights.is_file() else None,
        "checkpoint_path": str(weights),
        "device": str(device),
        "environment": {
            "python_version": __import__("platform").python_version(),
            "torch_version": torch.__version__,
            "ultralytics_version": __import__("ultralytics").__version__,
            "numpy_version": __import__("numpy").__version__,
        },
        "protocol": {
            "model": arguments.model,
            "epochs": arguments.epochs,
            "batch": arguments.batch,
            "imgsz": arguments.imgsz,
            "optimizer": "AdamW",
            "lr0": arguments.lr0,
            "seed": arguments.seed,
        },
    }, indent=2), encoding="utf-8")
    print(f"wrote {len(predictions)} predictions to {arguments.output}")


if __name__ == "__main__":
    main()
