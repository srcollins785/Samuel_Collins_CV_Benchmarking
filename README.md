# Samuel_Collins_CV_Benchmarking

Benchmark traditional machine-learning classifiers and modern CNN
architectures on one image-classification problem. Give it a labeled image
dataset in any of four organizations; it standardizes the images, builds one
stratified split, trains every model on that split, and saves comparable
metrics, plots and reports.

Sixteen models on a single shared split:

| Group | Models |
|---|---|
| Traditional ML | Logistic Regression, Decision Tree, Random Forest, SVM |
| Baseline neural | Neural Network (MLP), Simple CNN |
| Deep CNN | AlexNet, VGG16, GoogLeNet, ResNet18, ResNet50, DenseNet121, MobileNetV3-Large, EfficientNet-B0, ConvNeXt-Tiny, YOLO classification |

**PyPI:** https://pypi.org/project/Samuel_Collins_CV_Benchmarking/ · version 2.0.0
· MIT licensed · 545 tests

## Installation

```bash
pip install Samuel_Collins_CV_Benchmarking
```

Then:

```python
from samuel_collins_cv_benchmarking import benchmark_image_classification
```

From a clone, for development. The virtual environment matters on macOS,
where the system `python3` and a Homebrew `python3` are different
interpreters with different packages installed:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,report,deep]"
```

The `deep` extra adds torchvision and the two complexity profilers. It is
optional and imported lazily, so a plain `pip install
Samuel_Collins_CV_Benchmarking` does not pull a gigabyte of pretrained-model
machinery onto someone who only wants the traditional classifiers.

To reproduce the published numbers rather than install the newest packages,
use the pinned set instead:

```bash
pip install -r requirements.txt
pip install -e .
```

Inside the environment, plain `python`, `pytest` and `twine` resolve to the
right interpreter, and everything below can be run without a path prefix.

## Running everything

```bash
python scripts/run_all.py                 # tests, benchmarks, report, PDF
python scripts/run_all.py --report-only   # just rebuild the report
```

### The deep CNN benchmark

```bash
python run_benchmark.py                   # all nine torchvision architectures
python run_benchmark.py --model resnet50   # one architecture
python run_benchmark.py --model all        # explicit form of the default
python run_benchmark.py --tables-only      # rebuild tables and plots only
```

The deep architectures run on the `animals10_n500` RGB configuration - the
same one Part 1 benchmarked, so the two halves are directly comparable. RGB
because every pretrained backbone expects three channels; the grayscale path
refuses rather than replicating one channel into three and quietly weakening
transfer learning.

Part 1 is **not** re-run. Its split is reproduced from the same manifest at
seed 42 and then verified against Part 1's own published artifacts - the
recorded dataset index behind every stored test position, and the exact rows
the Simple CNN held back for validation. If the dataset under `data/` has
changed since Part 1 ran, the benchmark stops with an explanation instead of
producing a comparison that looks valid and is not.

```bash
# re-run any architecture the protocol learning rate destabilized
python scripts/remediate_unstable.py --dry-run   # report only
python scripts/remediate_unstable.py             # detect and re-run
```

At the prescribed AdamW learning rate of 0.001, the two pre-BatchNorm
architectures do not train: AlexNet collapses to chance and VGG16 is crippled.
Every architecture carrying BatchNorm or LayerNorm trains at that rate without
difficulty. This script detects the failures by a stated numeric rule, archives
the protocol-rate results, re-runs only the affected architectures at a lower
rate, and writes `deep_lr_deviations.json` recording both outcomes. The report
reads that file, so the deviation is documented rather than silently applied.

### YOLO classification

YOLO needs its own environment. `ultralytics` cannot be installed alongside the
main one: this project runs on Python 3.9, and on macOS ultralytics excludes
every numpy 2.0 through 2.3.4 release while numpy >= 2.3.5 requires Python
3.11+. Installing it would resolve numpy down from 2.0.2 and silently change
the environment that produced the Part 1 results and the test suite.

```bash
python3 -m venv .venv-yolo                 # use a Python 3.11+ interpreter
.venv-yolo/bin/pip install -r requirements-yolo.txt
python scripts/run_yolo.py
```

It exports the identical split as a symlink tree, trains in the isolated
environment as a subprocess, and brings the predictions back to be scored by
the same code that scores every other architecture - so the metrics are
computed identically even though the training environment differs.

## Usage

```python
from samuel_collins_cv_benchmarking import benchmark_image_classification

results = benchmark_image_classification(
    dataset="./data/animals10_n500/images",
    dataset_type="folder",
    target_labels=["cat", "dog", "horse"],
    color_mode="rgb",
)
```

### Parameters

| Parameter | Meaning |
|---|---|
| `dataset` | Dataset root directory, CSV/JSON/JSONL manifest path, Pandas DataFrame, or NumPy image array |
| `dataset_type` | One of `"folder"`, `"csv"`, `"json"`, `"array"` |
| `target_labels` | Class-folder names, manifest label-field name, DataFrame label column, or a label vector |
| `color_mode` | `"grayscale"` for one channel, `"rgb"` for three |

Image size (64x64), random seed (42), split ratio (80/20) and output location
are internal constants — callers configure nothing beyond the four parameters
above.

## The four dataset organizations

**1. Class folders** — each subfolder name is the label. PNG, JPG, JPEG, BMP
and TIFF are supported.

```
data/animals10_n500/images/
├── cat/cat_001.jpeg
├── dog/dog_001.jpeg
└── horse/horse_001.jpeg
```

```python
benchmark_image_classification(
    dataset="./data/animals10_n500/images", dataset_type="folder",
    target_labels=["cat", "dog", "horse"], color_mode="rgb")
```

**2. CSV manifest** — an `image_path` column plus a label column named by
`target_labels`. Relative paths resolve from the manifest's own location.

```csv
image_path,class_name
images/cat/cat_001.jpeg,cat
```

```python
benchmark_image_classification(
    dataset="./data/animals10_n500/labels.csv", dataset_type="csv",
    target_labels="class_name", color_mode="rgb")
```

**3. JSON / JSONL manifest** — a JSON list of records, or one JSON record per
line. Every record carries `image_path` and the label field.

```json
[{"image_path": "images/cat/cat_001.jpeg", "class_name": "cat"}]
```

```python
benchmark_image_classification(
    dataset="./data/animals10_n500/labels.json", dataset_type="json",
    target_labels="class_name", color_mode="rgb")
```

**4. NumPy array / in-memory** — `dataset` is the image tensor, `target_labels`
the label vector. Accepted shapes: `(N, H, W)`, `(N, H, W, 1)`, `(N, H, W, 3)`.

```python
benchmark_image_classification(
    dataset=X_images, dataset_type="array",
    target_labels=y_labels, color_mode="grayscale")
```

## Datasets

**RGB - Animals-10.** 10 classes. The images are **not committed to this
repository**: Animals-10 is assembled from web-scraped photographs, so
redistributing it here is not appropriate. Rebuild it in one command (needs
Kaggle API credentials at `~/.kaggle/kaggle.json`):

```bash
python scripts/download_animals10.py                 # 500/class -> data/animals10_n500
python scripts/download_animals10.py --per-class 10  # fast smoke set
python scripts/download_animals10.py --per-class 100 --out data/custom
```

Each tier lands in its own directory containing `images/` plus `labels.csv`,
`labels.json` and `labels.jsonl`, so several sizes coexist:

```
data/animals10_n500/
├── images/<class>/<class>_001.jpeg
├── labels.csv
├── labels.json
└── labels.jsonl
```

### Sampling method

Reported subset: **500 images per class, 5,000 total**, drawn from
`alessiocorrado99/animals10`. Per class, the file list is sorted, shuffled with
`random.Random(<english class name>)`, and the first N images that decode
successfully are taken. Every file is opened, verified and RGB-converted before
selection; undecodable files are skipped and counted.

The seed depends only on the class name, never on N, so the tiers are **nested**:
`n10` is byte-identical to the first 10 images of `n100`, filenames included.
A comparison across tiers is therefore a genuine learning curve rather than
three unrelated samples.

Source folder names are Italian and are mapped to English (`cane`->`dog`,
`gatto`->`cat`, `ragno`->`spider`, ...). The per-class ceiling is set by the
smallest class, elephant, at 1,446 images.

**Grayscale.** The same images through `color_mode="grayscale"`, which is the
single-channel demonstration. Running both isolates what color contributes:
on Animals-10 it is worth about a quarter of the CNN's macro F1, and it
costs the SVM roughly forty times its training run.

## Results

Two runs over nested subsets of Animals-10, ten classes, RGB at 64x64. The
tiers share one fixed ordering, so `n100` is a byte-identical subset of
`n500` and reading across them is a learning curve rather than a comparison
of unrelated samples. Chance for ten classes is 0.100.

| Model | Macro F1 @ 100/class | Macro F1 @ 500/class | change |
|---|---|---|---|
| Simple CNN | 0.295 | **0.434** | +47% |
| SVM | 0.242 | 0.352 | +45% |
| Random Forest | 0.294 | 0.319 | +9% |
| Neural Network | 0.178 | 0.281 | +58% |
| Logistic Regression | 0.213 | 0.225 | +6% |
| Decision Tree | 0.093 | 0.180 | +94% |

At 100 images per class the CNN and Random Forest are tied. At 500 the CNN
leads by 36%, and Logistic Regression and Random Forest have nearly
flattened while the CNN and SVM are still climbing steeply. Reporting only
the smaller run would have supported the conclusion that a CNN and an
ensemble of trees are equivalent here - true at that size, and misleading as
a finding.

Cost at 500 per class tells a different story from accuracy alone:

| Model | Training | Inference |
|---|---|---|
| SVM | 547.4 s | 150.1 ms/image |
| Simple CNN | 44.8 s | 0.43 ms/image |
| Decision Tree | 20.2 s | 0.002 ms/image |
| Logistic Regression | 9.5 s | 0.040 ms/image |
| Random Forest | 3.3 s | 0.028 ms/image |
| Neural Network | 1.3 s | 0.025 ms/image |

The SVM buys third place at 75,000 times the Decision Tree's inference cost:
classifying a thousand images would take it two and a half minutes against
Random Forest's 0.03 seconds for a slightly better score.

Full outputs are under `benchmark_results/<tier>/`. Reproduce them with:

```bash
python scripts/download_animals10.py --per-class 500
python scripts/run_benchmarks.py
```

The 500/class run takes about 13 minutes, 9 of which are the SVM.

## Output layout

One directory per configuration, holding both halves of the benchmark:

```
benchmark_results/animals10_n500_rgb/
    benchmark_summary.csv                       Part 1 comparison table
    benchmark_metrics.json                      Part 1 full results
    deep_metrics.json                           Part 2 full results
    deep_run_configuration.json                 protocol, environment, deviations
    deep_lr_deviations.json                     learning-rate changes, if any
    combined_ml_cnn_benchmark_results.csv        the master table, all 16 models
    per_class_f1_comparison.csv                 per-class F1, every model
    rankings.json                               rankings A through E
    confusion_matrices/*.png                    one per model, both halves
    classification_reports/*.csv                one per model, both halves
    plots/                                      comparison figures + training curves
    checkpoints/best_*.pt                       selected weights per architecture
    logs/*.jsonl                                one line per epoch, written live
```

The per-epoch logs are committed - they are the record of what each run
actually did. The checkpoints are not: the nine together are about 1.5 GB and
VGG16's alone exceeds GitHub's 100 MB per-file limit. They are regenerated by
`python run_benchmark.py`, and their sizes are recorded in `deep_metrics.json`
and the master table, so every reported number survives without the weights
being in version control.

`N/A` appears wherever a metric is not meaningful for a model rather than
wherever it was inconvenient to obtain. A Random Forest has no parameter count
in the sense a CNN does, no checkpoint on disk and no device memory figure, and
a zero in those cells would be a fabricated measurement that looks exactly like
a real one.

## Tests

```bash
pytest tests/
```

`tests/fixtures/` holds small committed datasets so the suite runs in a clean
checkout with no downloads:

| Fixture | Contents |
|---|---|
| `mini/` | 3 classes x 5 images, pristine. One image per required extension (`.jpeg`, `.jpg`, `.png`, `.bmp`, `.tiff`) at five different dimensions, so resizing and aspect handling are exercised. |
| `broken/` | 3 classes x 3 images plus one undecodable file and, in the manifests, one row pointing at a file that is not on disk. Covers the skip-and-report path. |

Each tree has matching `*_labels.csv`, `.json` and `.jsonl` manifests, so all
four dataset organizations can be tested against committed data.

## License

MIT. See [LICENSE](LICENSE). The license covers this source code, not the
third-party image datasets it consumes.
