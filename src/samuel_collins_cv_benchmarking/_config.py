"""Internal constants.

The assignment requires that users of the package configure nothing beyond the
four public parameters, so image size, seed and split live here rather than in
the public signature.
"""

IMAGE_SIZE = (64, 64)
RANDOM_SEED = 42
TEST_SIZE = 0.20
RESULTS_DIR = "benchmark_results"
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DATASET_TYPES = ("folder", "csv", "json", "array")
COLOR_MODES = ("grayscale", "rgb")
