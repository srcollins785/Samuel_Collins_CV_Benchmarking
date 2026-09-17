"""Internal constants.

The assignment requires that users of the package configure nothing beyond the
four public parameters, so image size, seed and split live here rather than in
the public signature.
"""

DISTRIBUTION_NAME = "Samuel_Collins_CV_Benchmarking"
# Kept here rather than in __init__ so benchmark.py can read it without
# importing the package root, which would be circular.
VERSION = "2.0.0"

IMAGE_SIZE = (64, 64)
RANDOM_SEED = 42
TEST_SIZE = 0.20
RESULTS_DIR = "benchmark_results"
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DATASET_TYPES = ("folder", "csv", "json", "array")
COLOR_MODES = ("grayscale", "rgb")


# ---------------------------------------------------------------------------
# Part 2: the deep CNN extension.
#
# Everything above describes the Part 1 benchmark and must not change. The
# Part 1 numbers were produced with those constants, and section 1A of the
# extension requires the traditional ML baselines to stay directly comparable,
# which they only are if the split that produced them is reproducible from the
# same values. So the deep CNN protocol is stated additively below rather than
# by editing anything above it.
#
# These names carry the CNN_ prefix for the same reason: IMAGE_SIZE is the
# 64x64 representation the classical models and the Simple CNN consume, and a
# reader who sees a bare IMAGE_SIZE somewhere in the deep code should be able
# to tell immediately that it is the wrong one.
# ---------------------------------------------------------------------------

# Section 7. The input resolution the pretrained ImageNet backbones expect.
CNN_IMAGE_SIZE = (224, 224)

# Section 7 asks for Resize((224, 224)) followed by RandomCrop(...), which is
# a no-op unless the source image is larger than the crop. Caching the
# training half at 256x256 gives the random crop real headroom, which is what
# the transform is there to provide. Validation and test images are cached at
# CNN_IMAGE_SIZE instead, so their transform is exactly the resize-only
# pipeline section 7 requires and carries no augmentation at all.
CNN_CACHE_SIZE = (256, 256)

# Section 6's common experimental protocol, applied identically to every
# architecture so that training time and accuracy stay comparable.
CNN_EPOCHS = 20
CNN_BATCH_SIZE = 64
CNN_LEARNING_RATE = 1e-3
CNN_OPTIMIZER = "AdamW"
# Section 6 does not name a weight decay. This is torch's AdamW default,
# recorded explicitly rather than left implicit so the report can state the
# full optimizer configuration without anyone reading the torch source.
CNN_WEIGHT_DECAY = 0.01
CNN_LOSS = "CrossEntropyLoss"

# Section 8. Pretrained ImageNet weights, with the final classification layer
# replaced, and every layer left trainable - section 8 and learning objective
# 5 both ask for fine-tuning rather than frozen feature extraction.
CNN_PRETRAINED = True
CNN_FREEZE_BACKBONE = False

# Section 7's Normalize(...). These are the ImageNet statistics the pretrained
# weights were trained under; substituting the dataset's own statistics would
# put the inputs in a different range from the one the weights expect and cost
# accuracy for no benefit.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# The validation fraction is carved out of the *training* half only, so the
# 80/20 train/test division above is untouched and the test set stays unseen
# until model selection is complete (sections 3 and 12).
#
# This value must equal neural_models.VALIDATION_FRACTION. Both the Simple CNN
# and the deep architectures then validate on the identical rows at the
# identical seed, which is what lets the report compare their convergence
# curves to each other rather than only to themselves. A test asserts the two
# agree, so this cannot drift unnoticed.
CNN_VALIDATION_FRACTION = 0.15

# Device order, best first. MPS is chosen on this machine; the CUDA entry is
# kept ahead of it so the same code reproduces on a CUDA host without edits,
# and whichever device is actually selected is recorded in the run
# configuration. Note that the Part 1 Simple CNN deliberately stays on CPU for
# cross-machine determinism - ten deep networks at 224x224 cannot afford that,
# and the difference is a documented deviation rather than an oversight.
CNN_DEVICE_PREFERENCE = ("cuda", "mps", "cpu")

# Section 20. Inference is timed over at least 1,000 test images, after a
# warm-up that is not timed: the first batch through a freshly loaded network
# pays for lazy kernel compilation and allocator growth, and including it
# would make every model look slower than it is by an amount that depends on
# nothing interesting.
CNN_INFERENCE_MIN_IMAGES = 1000
CNN_INFERENCE_WARMUP_BATCHES = 3

# Subdirectories written inside a configuration's result directory. The
# assignment's suggested layout puts these at the repository root, but the
# Part 1 pipeline already groups every artifact for one configuration under
# benchmark_results/<configuration>/, and keeping that grouping means a reader
# can see at a glance which dataset and color mode any checkpoint belongs to.
CHECKPOINTS_DIR = "checkpoints"
LOGS_DIR = "logs"
PLOTS_DIR = "plots"

# Section 23 names this file explicitly. It is the one table that must contain
# both the Part 1 traditional ML baselines and the Part 2 deep architectures.
COMBINED_RESULTS_CSV = "combined_ml_cnn_benchmark_results.csv"

# The ten architectures section 5 requires, in the order the report presents
# them - roughly chronological, so the combined table reads as the
# architectural timeline section 26 asks for rather than as an arbitrary list.
CNN_ARCHITECTURES = (
    "alexnet",
    "vgg16",
    "googlenet",
    "resnet18",
    "resnet50",
    "densenet121",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "convnext_tiny",
    "yolo_cls",
)

# YOLO is the one architecture that cannot run in this environment. On macOS
# ultralytics excludes every numpy 2.0-2.3.4 release, and numpy >= 2.3.5
# requires Python 3.11+, so installing it here would downgrade numpy beneath
# the environment that produced the Part 1 results and the test suite. It runs
# instead as a subprocess in its own virtual environment, writing the same
# artifacts this package writes, which the report already reads from disk.
YOLO_MODEL = "yolo11n-cls.pt"
YOLO_REQUIREMENTS = "requirements-yolo.txt"
