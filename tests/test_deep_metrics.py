"""Tests for the cost measurements: complexity, latency, memory, environment.

The theme is that a measurement has to be readable correctly, not just
present. Two specifics drive most of these tests.

Profilers count multiply-accumulates and then label the total "flops". A MAC
is a multiply and an add, so the true floating-point count is about twice the
reported figure, and publishing the profiler's number under a FLOPs heading
would understate every architecture by half. So the results name the quantity
MACs and derive FLOPs explicitly, and that relationship is asserted.

Section 22 says to report N/A and explain the limitation when a profiler
cannot measure something, rather than fabricate a figure. So the failure path
returns None with a reason attached, and that is asserted too.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from samuel_collins_cv_benchmarking import deep_metrics, deep_models


# -- section 20: the environment -------------------------------------------

def test_environment_reports_everything_section_20_requires():
    report = deep_metrics.environment_report()
    for key in ("gpu", "cpu", "ram_gb", "operating_system", "python_version",
                "torch_version", "cuda_version"):
        assert report.get(key) is not None, f"missing {key}"


def test_environment_states_cuda_absence_rather_than_leaving_it_blank():
    """A blank could be read as an omission; this has to be explicit."""
    report = deep_metrics.environment_report()
    if not report["cuda_available"]:
        assert "not applicable" in report["cuda_version"].lower()


def test_environment_versions_are_real_versions():
    report = deep_metrics.environment_report()
    assert report["torch_version"] == torch.__version__
    assert report["numpy_version"] == np.__version__


# -- section 22: complexity -------------------------------------------------

@pytest.fixture(scope="module")
def small_model():
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4))


def test_complexity_reports_macs_and_derives_flops(small_model):
    result = deep_metrics.complexity(small_model, (64, 64))
    assert result["macs"] > 0
    assert result["flops_estimate"] == result["macs"] * 2
    # Compared against the same rounding the reporter applies. macs is exact,
    # so these two are exact checks rather than tolerances.
    assert result["gmacs"] == round(result["macs"] / 1e9, 3)
    assert result["gflops_estimate"] == round(result["macs"] * 2 / 1e9, 3)


def test_complexity_names_the_quantity_it_counted(small_model):
    """"flops" from a profiler means MACs. Say so where it is reported."""
    result = deep_metrics.complexity(small_model, (64, 64))
    assert "multiply-accumulate" in result["counts"].lower()
    assert "mac" in result["note"].lower()
    assert result["profiler"]


def test_complexity_leaves_the_model_where_it_found_it(small_model):
    """Profiling must not move or mutate the model being benchmarked."""
    original = [p.detach().clone() for p in small_model.parameters()]
    deep_metrics.complexity(small_model, (64, 64))
    for before, after in zip(original, small_model.parameters()):
        assert torch.equal(before, after.detach().cpu())


def test_complexity_reports_na_with_a_reason_when_it_cannot_measure():
    """Section 22: N/A and an explanation, never an invented number."""

    class Unprofilable(nn.Module):
        def forward(self, x):
            raise RuntimeError("this model cannot be traced")

    result = deep_metrics.complexity(Unprofilable(), (32, 32))
    assert result["macs"] is None
    assert result["flops_estimate"] is None
    assert result["profiler"] is None
    assert "not available" in result["note"].lower()


def test_complexity_lists_unaccounted_operators(small_model):
    """Disclosure, per section 22 - the list may be empty but must exist."""
    result = deep_metrics.complexity(small_model, (64, 64))
    assert isinstance(result["unsupported_operators"], list)


# -- section 20: inference speed -------------------------------------------

@pytest.fixture(scope="module")
def tiny_loader():
    from torch.utils.data import DataLoader, TensorDataset

    images = torch.randn(96, 3, 64, 64)
    labels = torch.randint(0, 4, (96,))
    return DataLoader(TensorDataset(images, labels), batch_size=32)


def test_inference_benchmark_is_internally_consistent(small_model, tiny_loader):
    result = deep_metrics.inference_benchmark(
        small_model, tiny_loader, torch.device("cpu"), warmup_batches=1)

    assert result["images"] == 96
    assert result["total_seconds"] > 0
    # Each field is rounded independently from the raw elapsed time, so the
    # reported latency cannot be reconstructed exactly from the reported
    # total. Checked to within the precision the fields are published at.
    assert result["latency_ms_per_image"] == pytest.approx(
        result["total_seconds"] * 1000 / 96, abs=0.01)
    assert result["throughput_images_per_second"] == pytest.approx(
        96 / result["total_seconds"], rel=0.01)


def test_inference_benchmark_records_its_batch_size_and_warmup(
        small_model, tiny_loader):
    """Latency is meaningless without the batch size that produced it."""
    result = deep_metrics.inference_benchmark(
        small_model, tiny_loader, torch.device("cpu"), warmup_batches=1)
    assert result["batch_size"] == 32
    assert result["warmup_batches"] == 1


def test_inference_benchmark_flags_a_test_set_below_the_minimum(
        small_model, tiny_loader):
    """Section 20 asks for at least 1,000 images; 96 is not that."""
    result = deep_metrics.inference_benchmark(
        small_model, tiny_loader, torch.device("cpu"), warmup_batches=1)
    assert result["meets_minimum_images"] is False
    assert result["minimum_images_required"] == 1000


def test_memory_measurement_is_always_labeled(small_model, tiny_loader):
    result = deep_metrics.inference_benchmark(
        small_model, tiny_loader, torch.device("cpu"), warmup_batches=1)
    assert result["memory_measurement"] in {
        "cuda_peak_allocated",
        "mps_live_allocation_sampled_maximum",
        "not_applicable_cpu",
    }


def test_single_image_latency_reports_a_median_as_well_as_a_mean(
        small_model, tiny_loader):
    """Batch-1 timings are noisy; a median says what a typical frame costs."""
    result = deep_metrics.single_image_latency(
        small_model, tiny_loader, torch.device("cpu"), samples=12)

    assert result["samples"] == 12
    assert result["batch_size"] == 1
    assert result["mean_latency_ms"] > 0
    assert result["median_latency_ms"] > 0
    assert result["p95_latency_ms"] >= result["median_latency_ms"]
    assert result["throughput_images_per_second"] == pytest.approx(
        1000 / result["mean_latency_ms"], rel=0.01)


def test_single_image_latency_handles_an_empty_loader(small_model):
    from torch.utils.data import DataLoader, TensorDataset

    empty = DataLoader(TensorDataset(torch.empty(0, 3, 64, 64),
                                     torch.empty(0, dtype=torch.int64)))
    result = deep_metrics.single_image_latency(
        small_model, empty, torch.device("cpu"))
    assert result["samples"] == 0


# -- section 18: checkpoint size -------------------------------------------

def test_checkpoint_size_is_measured_from_the_file(tmp_path, small_model):
    path = tmp_path / "model.pt"
    torch.save(small_model.state_dict(), path)
    assert deep_metrics.checkpoint_megabytes(path) == pytest.approx(
        path.stat().st_size / 1e6, abs=0.01)


def test_checkpoint_size_is_none_when_there_is_no_file(tmp_path):
    assert deep_metrics.checkpoint_megabytes(None) is None
    assert deep_metrics.checkpoint_megabytes(tmp_path / "absent.pt") is None
