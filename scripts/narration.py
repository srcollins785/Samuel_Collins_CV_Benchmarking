"""Narration for the demonstration video, keyed to the slide timings.

Each entry starts when the slide it belongs to appears. The timings come from
scripts/_demo_timeline.json, which build_demo_video.py writes, so the two
cannot drift: change a duration in the video and add_narration.py reports
which lines no longer fit.

Written in the first person, since it is Samuel's demonstration. Read at
about 150 words a minute, which is what the slot lengths assume.
"""

# One line per slide, in the order the slides appear. There are no timings
# here on purpose: build_demo_video.py reads how long each line takes to
# speak and holds its slide at least that long, so the two cannot drift.
NARRATION = [
    (
     "This is Samuel Collins Computer Vision Benchmarking, version one point "
     "zero point one. It compares six image classification models through a "
     "single public function. Everything you will see in this video is real "
     "output from real runs."),

    (
     "First, installation. The package is published on the Python Package "
     "Index."),

    (
     "This is a clean virtual environment. Pip resolves the six dependencies: "
     "numpy, pandas, Pillow, scikit-learn, matplotlib and PyTorch."),

    (
     "Version one point zero point one, MIT licensed."),

    (
     "The distribution is named for me, as the assignment requires. The import "
     "name is its lowercase form."),

    (
     "One public function, and four parameters: the dataset, its type, the "
     "target labels, and the color mode."),

    (
     "Now a real benchmark. One thousand images across ten classes, split "
     "eighty twenty at seed forty two, and that one split is reused by all six "
     "models."),

    (
     "Here it reads a CSV manifest: an image path column and a label column."),

    (
     "The same images again, this time as class folders, one folder per class."),

    (
     "And again as JSON, and as JSON Lines."),

    (
     "And finally as a NumPy array already in memory, with no files at all."),

    (
     "Look at the macro F one. Folder, CSV, JSON and JSON Lines describing the "
     "same thousand images all return zero point two nine four six six zero. "
     "Identical to six decimal places. The package changes how it reads the "
     "data, not how it evaluates the models."),

    (
     "The assignment asks for two demonstrations, one single channel and one "
     "RGB. These are the same images through both color modes."),

    (
     "In RGB each classical model sees twelve thousand two hundred and eighty "
     "eight features per image."),

    (
     "In grayscale, four thousand and ninety six. A third as many."),

    (
     "Color is worth about a quarter of the CNN's macro F one here, and it "
     "costs the support vector machine roughly forty times its training run. "
     "On my second dataset the decision tree reverses sign, so that effect "
     "belongs to the data, not the model."),

    (
     "Every run writes a complete record of itself."),

    (
     "Eighteen files: the summary table, per model metrics, the configuration, "
     "the class level reports, and the figures."),

    (
     "The summary is ranked by macro F one, with ties broken by the lower "
     "inference time."),

    (
     "The class distribution. The stratified split preserves each class's "
     "share in both halves."),

    (
     "Accuracy, macro F one, training time and inference time on four separate "
     "axes, never two scales on one chart. The timing panels are dots on a log "
     "scale, because a bar measures its length from zero and a log axis "
     "has no zero."),

    (
     "The confusion matrix is colored by share of the true class, so a small "
     "class is not washed out by a large one. Cat and dog confuse each other; "
     "cow, horse and elephant all collapse toward sheep."),

    (
     "These are correct and incorrect predictions, shown as the standardized "
     "sixty four by sixty four input the model actually received. The black "
     "padding is visible because it is part of what the network sees. Three of "
     "these errors are animals photographed on grass."),

    (
     "Reproducibility is recorded rather than claimed."),

    (
     "The configuration file carries the image size, the color mode, the seed, "
     "the split and every model's parameters. Re-running reproduces every "
     "score exactly."),

    (
     "A second dataset, so the padding question can be asked at all. Intel "
     "Image Classification is square at one hundred and fifty pixels, where "
     "Animals ten loses twenty seven percent of its canvas to padding."),

    (
     "Higher scores here, but six classes rather than ten, so chance is zero "
     "point one six seven and not zero point one. Normalized by chance the two "
     "datasets are close."),

    (
     "Its confusions are glacier against mountain, and sea against glacier. "
     "Scenes that share color and texture."),

    (
     "Six runs in total: two datasets, two color modes, two subset sizes. Each "
     "comparison changes one variable and holds the rest fixed."),

    (
     "This is the same benchmark at one hundred images per class. Here the CNN "
     "and the random forest are tied, zero point two nine five against zero "
     "point two nine four."),

    (
     "Finally, the repository."),

    (
     "Eight modules under source, six runnable examples, and seven test files. "
     "Only the source directory ships to the Package Index."),

    (
     "Three hundred and seventy three tests, all passing."),

    (
     "The finding I would most want to point at: at one hundred images per "
     "class the CNN and the random forest were tied. At five hundred the CNN "
     "leads by thirty six percent. A single run would have supported the "
     "opposite conclusion."),

    (
     "And cost is not accuracy. The support vector machine takes five hundred "
     "and forty seven seconds to train, and a hundred and fifty milliseconds "
     "per image at inference, for third place. That is why ties are broken on "
     "the faster model."),

    (
     "The package is on the Python Package Index, the code is on GitHub, and "
     "the full report is in the repository. Thank you."),
]
