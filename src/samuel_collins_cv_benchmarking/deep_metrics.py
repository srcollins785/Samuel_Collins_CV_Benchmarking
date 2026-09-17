"""Cost measurements: complexity, size, latency, memory, environment.

Section 20 says an inference benchmark is not meaningful without the hardware
it ran on, and section 22 says to report N/A and explain the limitation rather
than fabricate a measurement a profiler could not take. Both instructions
point the same way, so this module is written to be conservative: every number
it returns carries enough context to be read correctly, and anything it cannot
measure comes back as ``None`` with a reason attached.

Two measurement details matter more than they look.

*Profilers report MACs, not FLOPs.* Both fvcore and thop count
multiply-accumulate operations and then use the word "flops" for the total. A
MAC is a multiply and an add, so the true floating-point operation count is
about twice the reported figure. Reporting fvcore's number under a "FLOPs"
heading would understate every architecture by 2x. The results therefore name
the quantity ``macs`` and derive ``flops`` from it explicitly, so the factor
of two is visible rather than assumed.

*Latency depends entirely on batch size, so both are measured.* A batched
throughput figure is what a server sees; a single-image latency is what a
drone or a phone sees, and they can differ by an order of magnitude on the
same weights. Section 20 asks for latency and throughput, and the deployment
discussion in section 27 is about edge hardware, so reporting only the batched
number would answer the wrong question. Both appear, each labeled with the
batch size that produced it.
"""

import platform
import subprocess
import time

import numpy as np

from ._config import (
    CNN_BATCH_SIZE,
    CNN_INFERENCE_MIN_IMAGES,
    CNN_INFERENCE_WARMUP_BATCHES,
    CNN_IMAGE_SIZE,
)
from .deep_training import (
    device_memory_mb,
    memory_measurement_kind,
    reset_peak_memory,
    synchronize,
)

# Single-image latency is measured over this many images rather than all
# 1,000. Batch-1 inference is slow by construction, the figure stabilizes
# well before 200 samples, and nine architectures pay this cost each.
SINGLE_IMAGE_SAMPLES = 200


# -- the environment section 20 requires ------------------------------------

def _shell(command: list) -> str:
    """Best-effort single-line command output, empty string on any failure."""
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5)
        return completed.stdout.strip()
    except Exception:
        return ""


def _total_ram_gb():
    """Installed RAM in GB, or None where it cannot be read portably."""
    system = platform.system()
    if system == "Darwin":
        raw = _shell(["sysctl", "-n", "hw.memsize"])
        return round(int(raw) / 1024 ** 3, 1) if raw.isdigit() else None
    if system == "Linux":
        try:
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) / 1024 ** 2, 1)
        except Exception:
            return None
    return None


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        name = _shell(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name
    return platform.processor() or platform.machine() or "unknown"


def environment_report() -> dict:
    """GPU, CPU, RAM, OS and library versions.

    Section 20 lists these as required alongside any inference benchmark. The
    ``gpu`` and ``cuda_version`` entries say plainly that there is no CUDA
    device here rather than leaving a blank that could be read as an omission.
    """
    import torch

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda or "unknown"
    elif torch.backends.mps.is_available():
        gpu = f"Apple Silicon integrated GPU via Metal ({_cpu_name()})"
        cuda_version = "not applicable - no CUDA device on this host"
    else:
        gpu = "none - CPU only"
        cuda_version = "not applicable - no CUDA device on this host"

    return {
        "gpu": gpu,
        "cpu": _cpu_name(),
        "ram_gb": _total_ram_gb(),
        "operating_system": f"{platform.system()} {platform.release()} "
                            f"({platform.machine()})",
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": _torchvision_version(),
        "cuda_version": cuda_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "numpy_version": np.__version__,
    }


def _torchvision_version() -> str:
    try:
        import torchvision

        return torchvision.__version__
    except Exception:
        return "not installed"


# -- section 22: computational complexity -----------------------------------

def complexity(model, image_size: tuple = CNN_IMAGE_SIZE) -> dict:
    """Multiply-accumulates for one forward pass, with the profiler named.

    Tries fvcore first because it reports which operators it could not
    account for, which is exactly what section 22 asks to be disclosed;
    falls back to thop, which is quieter but gives no such list. Profiling
    runs on CPU regardless of the training device, since the operator
    coverage of these profilers is a property of the graph rather than the
    backend and CPU avoids any accelerator-specific tracing gaps.

    Returns
    -------
    dict
        ``macs``, ``gmacs``, ``flops_estimate`` (2x MACs), the profiler used,
        and either the unsupported-operator list or the reason no measurement
        was possible.
    """
    import copy
    import logging

    import torch

    # A deep copy on CPU, so profiling cannot disturb the model being trained
    # or leave it on the wrong device.
    probe = copy.deepcopy(model).cpu().eval()
    inputs = torch.randn(1, 3, *image_size)

    # fvcore logs one warning per unsupported operator at INFO level, which
    # for ConvNeXt is a wall of output. The list is captured and reported
    # instead, which is more useful than printing it nine times.
    for name in ("fvcore.nn.jit_analysis", "fvcore.nn.jit_handles"):
        logging.getLogger(name).setLevel(logging.ERROR)

    try:
        from fvcore.nn import FlopCountAnalysis

        analysis = FlopCountAnalysis(probe, inputs)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        macs = int(analysis.total())
        unsupported = sorted(analysis.unsupported_ops().keys())
        if macs > 0:
            return {
                "macs": macs,
                "gmacs": round(macs / 1e9, 3),
                # Named an estimate because it is a convention (1 MAC = 2
                # FLOPs), not a measurement.
                "flops_estimate": macs * 2,
                "gflops_estimate": round(macs * 2 / 1e9, 3),
                "profiler": "fvcore.nn.FlopCountAnalysis",
                "counts": "multiply-accumulate operations (MACs)",
                "unsupported_operators": unsupported,
                "note": (
                    "fvcore labels its total 'flops' but counts MACs. FLOPs "
                    "are derived as 2x MACs."
                    + (f" Operators not accounted for: {', '.join(unsupported)}."
                       if unsupported else "")
                ),
            }
    except Exception as error:
        fvcore_error = f"{type(error).__name__}: {error}"
    else:
        fvcore_error = "fvcore returned a total of zero"

    try:
        from thop import profile

        macs, _ = profile(probe, inputs=(inputs,), verbose=False)
        macs = int(macs)
        return {
            "macs": macs,
            "gmacs": round(macs / 1e9, 3),
            "flops_estimate": macs * 2,
            "gflops_estimate": round(macs * 2 / 1e9, 3),
            "profiler": "thop.profile",
            "counts": "multiply-accumulate operations (MACs)",
            "unsupported_operators": [],
            "note": (
                "Measured with thop after fvcore failed "
                f"({fvcore_error}). thop counts MACs; FLOPs are 2x MACs. "
                "thop silently ignores operators it does not know, so this "
                "total may understate architectures using uncommon layers."
            ),
        }
    except Exception as error:
        return {
            "macs": None,
            "gmacs": None,
            "flops_estimate": None,
            "gflops_estimate": None,
            "profiler": None,
            "counts": None,
            "unsupported_operators": [],
            # Section 22: report N/A and say why. No fabricated figure.
            "note": (
                "Not available. fvcore failed "
                f"({fvcore_error}) and thop failed "
                f"({type(error).__name__}: {error})."
            ),
        }


# -- section 20: inference speed --------------------------------------------

def inference_benchmark(
    model,
    loader,
    device,
    warmup_batches: int = CNN_INFERENCE_WARMUP_BATCHES,
) -> dict:
    """Time batched inference over a whole test loader.

    A warm-up of ``warmup_batches`` runs untimed first. The first batches
    through a freshly loaded network pay for lazy kernel compilation and
    allocator growth, costs that belong to startup rather than to the
    architecture, and including them would make every model look slower by an
    amount that varies with nothing interesting.

    Returns
    -------
    dict
        Images timed, total seconds, mean latency per image, throughput, the
        batch size that produced them, and peak memory during inference.
    """
    import torch

    model = model.to(device).eval()
    batch_size = getattr(loader, "batch_size", CNN_BATCH_SIZE)

    with torch.no_grad():
        for position, (images, _) in enumerate(loader):
            if position >= warmup_batches:
                break
            model(images.to(device))
    synchronize(device)

    reset_peak_memory(device)
    counted = 0
    peak = 0.0
    started = time.perf_counter()
    with torch.no_grad():
        for images, _ in loader:
            model(images.to(device))
            counted += len(images)
            # Sampled every batch and kept as a maximum. Reading it once
            # after the loop would catch a moment when the last batch's
            # activations had already been released, understating the
            # footprint of the pass.
            peak = max(peak, device_memory_mb(device))
    # The clock stops only after the device has actually finished; without
    # this the figure would describe submission rather than computation.
    synchronize(device)
    elapsed = time.perf_counter() - started
    peak = max(peak, device_memory_mb(device))

    return {
        "images": counted,
        "meets_minimum_images": counted >= CNN_INFERENCE_MIN_IMAGES,
        "minimum_images_required": CNN_INFERENCE_MIN_IMAGES,
        "total_seconds": round(elapsed, 4),
        "latency_ms_per_image": round(elapsed * 1000 / max(1, counted), 4),
        "throughput_images_per_second": round(counted / elapsed, 2) if elapsed else None,
        "batch_size": batch_size,
        "warmup_batches": warmup_batches,
        "device": str(device),
        "peak_memory_mb": round(peak, 1),
        "memory_measurement": memory_measurement_kind(device),
    }


def single_image_latency(
    model,
    loader,
    device,
    samples: int = SINGLE_IMAGE_SAMPLES,
) -> dict:
    """Latency for one image at a time - the edge-deployment number.

    Batched throughput flatters every architecture, because a batch of 64
    keeps the device busy in a way a single frame never does. A UAV
    classifying one frame as it arrives pays this cost instead, so this is
    the figure the deployment recommendations in section 27 should be read
    against.

    Reports the median as well as the mean: batch-1 timings are noisy, one
    scheduling hiccup moves a mean, and the median says what a typical frame
    actually costs.
    """
    import torch

    model = model.to(device).eval()

    # Pull enough individual images out of the loader to time.
    images = []
    for batch, _ in loader:
        for position in range(len(batch)):
            images.append(batch[position:position + 1])
            if len(images) >= samples:
                break
        if len(images) >= samples:
            break

    if not images:
        return {"samples": 0, "note": "no test images available to time"}

    with torch.no_grad():
        for _ in range(min(CNN_INFERENCE_WARMUP_BATCHES * 5, len(images))):
            model(images[0].to(device))
    synchronize(device)

    timings = []
    with torch.no_grad():
        for image in images:
            on_device = image.to(device)
            synchronize(device)
            started = time.perf_counter()
            model(on_device)
            synchronize(device)
            timings.append((time.perf_counter() - started) * 1000)

    array = np.array(timings)
    return {
        "samples": len(timings),
        "batch_size": 1,
        "mean_latency_ms": round(float(array.mean()), 4),
        "median_latency_ms": round(float(np.median(array)), 4),
        "p95_latency_ms": round(float(np.percentile(array, 95)), 4),
        "throughput_images_per_second": round(1000 / float(array.mean()), 2),
        "device": str(device),
    }


def peak_activation_memory(model, loader, device, batches: int = 3) -> dict:
    """Peak device memory *during* a forward pass, not between passes.

    Necessary because the obvious measurement is not the interesting one.
    Sampling allocation between batches - after ``model(images)`` returns -
    catches the weights with every intermediate activation already released,
    so it reproduces the checkpoint size almost exactly. Measured that way
    across these nine architectures the memory column correlated with
    checkpoint size at r = 1.000 to within 0.23 MB, which makes it a second
    copy of section 18 rather than an answer to section 21.

    What a deployment target actually has to fit is the working set: the
    weights plus the largest intermediate tensors alive at once. So this
    registers a forward hook on every submodule and samples allocation as the
    pass proceeds, keeping the maximum.

    Deliberately a separate pass from :func:`inference_benchmark`. Hooks add a
    Python callback per module per batch, which would inflate the latency
    figures that section 20 reports - so timing runs unhooked and this runs
    over a few batches afterward.

    Returns
    -------
    dict
        Peak and baseline allocation in MB, their difference (the activation
        working set), how it was measured, and how many batches it covered.
    """
    import torch

    model = model.to(device).eval()
    reset_peak_memory(device)

    # Baseline: weights resident, nothing in flight.
    with torch.no_grad():
        synchronize(device)
    baseline = device_memory_mb(device)

    peak = baseline
    handles = []

    def sample(_module, _inputs, _output):
        nonlocal peak
        current = device_memory_mb(device)
        if current > peak:
            peak = current

    for module in model.modules():
        handles.append(module.register_forward_hook(sample))

    try:
        with torch.no_grad():
            for position, (images, _) in enumerate(loader):
                if position >= batches:
                    break
                model(images.to(device))
                synchronize(device)
                peak = max(peak, device_memory_mb(device))
    finally:
        for handle in handles:
            handle.remove()

    return {
        "peak_memory_mb": round(peak, 1),
        "weights_baseline_mb": round(baseline, 1),
        "activation_working_set_mb": round(max(0.0, peak - baseline), 1),
        "batches_sampled": batches,
        "batch_size": getattr(loader, "batch_size", CNN_BATCH_SIZE),
        "measurement": memory_measurement_kind(device),
        "note": (
            "Sampled inside the forward pass via per-module hooks and kept as "
            "a maximum. Measured in a pass separate from the timed one, since "
            "the hooks themselves cost time. Not a hardware counter: on this "
            "backend no true peak-memory API exists."
        ),
    }


def checkpoint_megabytes(path) -> float:
    """Saved checkpoint size in MB (section 18), or None if absent."""
    from pathlib import Path

    if path is None:
        return None
    path = Path(path)
    return round(path.stat().st_size / 1e6, 2) if path.is_file() else None
