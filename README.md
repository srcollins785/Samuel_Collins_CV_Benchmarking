# Samuel_Collins_CV_Benchmarking

Benchmark classical machine-learning and neural-network image classifiers
through a single public function. Give it a labeled image dataset in any of
four organizations; it standardizes the images, builds one stratified split,
trains every model on that split, and saves comparable metrics, plots and
reports.

> **Status: scaffolding.** The package layout, metadata and test fixtures are
> in place. The loaders, models and evaluation pipeline are not implemented
> yet — `benchmark_image_classification()` validates its arguments and then
> raises `NotImplementedError`.

## Installation

```bash
pip install Samuel_Collins_CV_Benchmarking
```

From a clone, for development:

```bash
pip install -e ".[dev]"
```

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

**Grayscale - not yet added.**

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
