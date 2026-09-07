"""Tests for the public entry point.

Covers the assignment's requirements that folder, CSV, JSON/JSONL and array
inputs are all accepted, that the summary contains six model rows, that all
required output files are generated, that the installed package exposes the
student-specific import path, and that the metadata carries the required
project name and a valid version.

These are the end-to-end tests, and they earn their keep. Every module below
had a passing unit suite when the first full run still failed: the CNN's
validation guard checked that it had enough training *samples* without
checking that a 15 percent slice of them could hold every class. Nothing
short of running the whole pipeline would have found it.
"""

import json
import re

import numpy as np
import pytest

import samuel_collins_cv_benchmarking as package
from samuel_collins_cv_benchmarking import benchmark_image_classification


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Run inside a temporary directory.

    The results land in ./benchmark_results, which section 2.3 says the caller
    must not have to configure, so the test changes directory instead.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def array_dataset():
    """Separable grayscale images, big enough for every model to run."""
    generator = np.random.default_rng(0)
    bands = [40] * 30 + [130] * 30 + [220] * 30
    tensor = np.stack([
        generator.integers(low, low + 30, size=(20, 20)).astype(np.uint8)
        for low in bands
    ])
    labels = ["cat"] * 30 + ["dog"] * 30 + ["horse"] * 30
    return tensor, labels


# --------------------------------------------------------------------------
# Package metadata
# --------------------------------------------------------------------------

class TestPackageMetadata:

    def test_the_student_specific_import_works(self):
        # Section 10: this exact import must work after installation.
        from samuel_collins_cv_benchmarking import (  # noqa: F401
            benchmark_image_classification as imported,
        )

        assert callable(imported)

    def test_the_package_declares_a_valid_version(self):
        # Section 12: the metadata must contain a valid version.
        assert re.fullmatch(r"\d+\.\d+\.\d+", package.__version__)

    def test_the_distribution_name_matches_the_required_pattern(self):
        from samuel_collins_cv_benchmarking._config import DISTRIBUTION_NAME

        assert DISTRIBUTION_NAME == "Samuel_Collins_CV_Benchmarking"

    def test_the_import_name_is_the_lowercase_form(self):
        from samuel_collins_cv_benchmarking._config import DISTRIBUTION_NAME

        assert package.__name__ == DISTRIBUTION_NAME.lower()


# --------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------

class TestArgumentValidation:

    def test_rejects_an_unknown_dataset_type(self, workspace):
        with pytest.raises(ValueError, match="dataset_type must be one of"):
            benchmark_image_classification(np.zeros((4, 8, 8)), "spreadsheet",
                                           ["a", "a", "b", "b"], "rgb")

    def test_rejects_an_unknown_colour_mode(self, workspace):
        with pytest.raises(ValueError, match="color_mode must be one of"):
            benchmark_image_classification(np.zeros((4, 8, 8)), "array",
                                           ["a", "a", "b", "b"], "cmyk")

    def test_validates_before_doing_any_work(self, workspace):
        # No output directory should appear when the arguments are wrong.
        with pytest.raises(ValueError):
            benchmark_image_classification(np.zeros((4, 8, 8)), "array",
                                           ["a", "a", "b", "b"], "cmyk")
        assert not (workspace / "benchmark_results").exists()


# --------------------------------------------------------------------------
# All four dataset organizations
# --------------------------------------------------------------------------

class TestEveryDatasetType:

    def test_folder_input(self, workspace, fixtures_dir):
        results = benchmark_image_classification(
            str(fixtures_dir / "mini"), "folder", ["cat", "dog", "horse"], "rgb")
        assert results["dataset_information"]["dataset_type"] == "folder"
        assert results["dataset_information"]["number_of_images"] == 15

    def test_csv_input(self, workspace, fixtures_dir):
        results = benchmark_image_classification(
            str(fixtures_dir / "mini_labels.csv"), "csv", "class_name", "rgb")
        assert results["dataset_information"]["number_of_images"] == 15

    def test_json_input(self, workspace, fixtures_dir):
        results = benchmark_image_classification(
            str(fixtures_dir / "mini_labels.json"), "json", "class_name", "rgb")
        assert results["dataset_information"]["number_of_images"] == 15

    def test_jsonl_input(self, workspace, fixtures_dir):
        results = benchmark_image_classification(
            str(fixtures_dir / "mini_labels.jsonl"), "json", "class_name", "rgb")
        assert results["dataset_information"]["number_of_images"] == 15

    def test_array_input(self, workspace, array_dataset):
        tensor, labels = array_dataset
        results = benchmark_image_classification(tensor, "array", labels, "grayscale")
        assert results["dataset_information"]["number_of_images"] == 90

    def test_dataframe_input(self, workspace, array_dataset):
        import pandas as pd

        tensor, labels = array_dataset
        frame = pd.DataFrame({"image": list(tensor), "class_name": labels})
        results = benchmark_image_classification(frame, "array", "class_name", "grayscale")
        assert results["dataset_information"]["number_of_images"] == 90

    def test_the_four_organizations_describe_the_same_dataset(self, workspace, fixtures_dir):
        # Section 3: the package changes how it reads the data, not what the
        # data is.
        folder = benchmark_image_classification(
            str(fixtures_dir / "mini"), "folder", ["cat", "dog", "horse"], "rgb")
        csv = benchmark_image_classification(
            str(fixtures_dir / "mini_labels.csv"), "csv", "class_name", "rgb")
        for key in ("number_of_images", "number_of_classes", "class_names",
                    "class_distribution"):
            assert folder["dataset_information"][key] == csv["dataset_information"][key]


# --------------------------------------------------------------------------
# Colour modes
# --------------------------------------------------------------------------

class TestColourModes:

    def test_grayscale_gives_one_channel(self, workspace, array_dataset):
        tensor, labels = array_dataset
        results = benchmark_image_classification(tensor, "array", labels, "grayscale")
        assert results["dataset_information"]["image_shape"] == [64, 64, 1]

    def test_rgb_gives_three_channels(self, workspace, array_dataset):
        tensor, labels = array_dataset
        results = benchmark_image_classification(tensor, "array", labels, "rgb")
        assert results["dataset_information"]["image_shape"] == [64, 64, 3]


# --------------------------------------------------------------------------
# The returned dictionary
# --------------------------------------------------------------------------

class TestReturnedResult:

    @pytest.fixture(scope="class")
    def results(self, tmp_path_factory, array_dataset):
        import os

        tensor, labels = array_dataset
        directory = tmp_path_factory.mktemp("run")
        previous = os.getcwd()
        os.chdir(directory)
        try:
            return benchmark_image_classification(tensor, "array", labels, "grayscale")
        finally:
            os.chdir(previous)

    def test_carries_every_required_key(self, results):
        # Section 9's return contract.
        for key in ("package_information", "summary", "best_model",
                    "dataset_information", "split_information", "model_results",
                    "confusion_matrices", "classification_reports", "warnings"):
            assert key in results

    def test_package_information_names_the_distribution_and_version(self, results):
        assert results["package_information"]["name"] == "Samuel_Collins_CV_Benchmarking"
        assert re.fullmatch(r"\d+\.\d+\.\d+", results["package_information"]["version"])

    def test_summary_contains_six_model_rows(self, results):
        # Section 12 asks for exactly this.
        assert len(results["summary"]) == 6

    def test_model_results_cover_all_six(self, results):
        assert set(results["model_results"]) == {
            "logistic_regression", "decision_tree", "random_forest",
            "svm", "neural_network", "simple_cnn",
        }

    def test_best_model_is_one_of_them(self, results):
        names = {entry["name"] for entry in results["model_results"].values()}
        assert results["best_model"] in names

    def test_split_information_records_the_required_seed(self, results):
        assert results["split_information"]["random_seed"] == 42
        assert results["split_information"]["training_samples"] == 72
        assert results["split_information"]["testing_samples"] == 18

    def test_all_three_distributions_are_reported(self, results):
        # Section 5.
        assert results["dataset_information"]["class_distribution"]
        assert results["split_information"]["training_distribution"]
        assert results["split_information"]["testing_distribution"]

    def test_every_model_has_a_confusion_matrix_and_report(self, results):
        for key in results["model_results"]:
            if results["model_results"][key]["succeeded"]:
                assert key in results["confusion_matrices"]
                assert key in results["classification_reports"]

    def test_prediction_examples_are_included(self, results):
        # Section 11 asks for examples of correct and incorrect predictions.
        for entry in results["model_results"].values():
            assert "prediction_examples" in entry

    def test_every_successful_model_predicted_every_test_image(self, results):
        expected = results["split_information"]["testing_samples"]
        for key, matrix in results["confusion_matrices"].items():
            assert np.asarray(matrix).sum() == expected


# --------------------------------------------------------------------------
# The written output directory
# --------------------------------------------------------------------------

class TestWrittenArtifacts:

    @pytest.fixture
    def run(self, workspace, array_dataset):
        tensor, labels = array_dataset
        benchmark_image_classification(tensor, "array", labels, "grayscale")
        return workspace / "benchmark_results"

    def test_writes_the_required_top_level_files(self, run):
        # Section 9's result directory.
        for name in ("benchmark_summary.csv", "benchmark_metrics.json",
                     "run_configuration.json", "class_distribution.png",
                     "model_comparison.png"):
            assert (run / name).is_file(), name

    def test_writes_a_confusion_matrix_per_model(self, run):
        matrices = sorted(p.stem for p in (run / "confusion_matrices").glob("*.png"))
        assert matrices == ["decision_tree", "logistic_regression", "neural_network",
                            "random_forest", "simple_cnn", "svm"]

    def test_writes_a_classification_report_per_model(self, run):
        reports = sorted(p.stem for p in (run / "classification_reports").glob("*.csv"))
        assert reports == ["decision_tree", "logistic_regression", "neural_network",
                           "random_forest", "simple_cnn", "svm"]

    def test_summary_csv_has_six_rows(self, run):
        import pandas as pd

        assert len(pd.read_csv(run / "benchmark_summary.csv")) == 6

    def test_metrics_json_is_readable(self, run):
        metrics = json.loads((run / "benchmark_metrics.json").read_text())
        assert len(metrics) == 6
        for entry in metrics.values():
            assert "macro_f1" in entry

    def test_run_configuration_records_reproducibility_evidence(self, run):
        # Section 9: image size, colour mode, seed, split, model parameters
        # and package version.
        configuration = json.loads((run / "run_configuration.json").read_text())
        assert configuration["image_size"] == [64, 64]
        assert configuration["random_seed"] == 42
        assert configuration["color_mode"] == "grayscale"
        assert configuration["package"]["version"] == package.__version__
        assert len(configuration["models"]) == 6

    def test_configuration_records_every_model_parameter_set(self, run):
        configuration = json.loads((run / "run_configuration.json").read_text())
        assert configuration["models"]["random_forest"]["n_estimators"] == 200
        assert configuration["models"]["simple_cnn"]["pretrained"] is False

    def test_files_and_returned_result_agree(self, workspace, array_dataset):
        # The files are written from the results already computed, so the two
        # cannot disagree - checked rather than assumed.
        tensor, labels = array_dataset
        results = benchmark_image_classification(tensor, "array", labels, "grayscale")
        metrics = json.loads(
            (workspace / "benchmark_results" / "benchmark_metrics.json").read_text())
        for key, entry in results["model_results"].items():
            if entry["succeeded"]:
                assert metrics[key]["macro_f1"] == entry["macro_f1"]


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

class TestReproducibility:

    def test_two_runs_give_the_same_scores(self, tmp_path_factory, array_dataset):
        import os

        tensor, labels = array_dataset
        scores = []
        for name in ("first", "second"):
            directory = tmp_path_factory.mktemp(name)
            previous = os.getcwd()
            os.chdir(directory)
            try:
                results = benchmark_image_classification(
                    tensor, "array", labels, "grayscale")
            finally:
                os.chdir(previous)
            scores.append({key: entry["macro_f1"]
                           for key, entry in results["model_results"].items()})
        assert scores[0] == scores[1]
