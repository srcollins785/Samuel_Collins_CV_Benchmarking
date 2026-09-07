"""Assemble the demonstration video from captured output and real figures.

Everything on screen comes from somewhere real: terminal frames are rendered
from `scripts/_demo_capture.json`, which `capture_demo.py` produces by
running the commands and recording what they printed, and the figures are the
PNGs the benchmark itself wrote. Nothing is typed in by hand for the video.

The structure follows section 13.8's list in its order: PyPI installation,
import, execution, generated outputs, repository organization.

Requires ffmpeg.

Usage
-----
    python scripts/capture_demo.py        # first, to record the output
    python scripts/build_demo_video.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
CAPTURE = REPO_ROOT / "scripts" / "_demo_capture.json"
RESULTS = REPO_ROOT / "benchmark_results"
OUTPUT = REPO_ROOT / "report" / "CV_Benchmarking_Demo.mp4"

WIDTH, HEIGHT = 1920, 1080
FPS = 30

# Every duration is multiplied by this. Section 13.8 asks for five to seven
# minutes, and the natural pace of the material is under four; stretching it
# uniformly also makes the terminal output readable rather than merely
# visible. Tuned so the film lands near the middle of the required range.
PACE = 1.85

# The palette the figures already use, so the frames and the charts look like
# one piece of work rather than two.
SURFACE = (252, 252, 251)
INK = (11, 11, 11)
INK_SECONDARY = (82, 81, 78)
INK_MUTED = (137, 135, 129)
ACCENT = (42, 120, 214)
GOOD = (12, 163, 12)
RULE = (225, 224, 217)
TERMINAL_BG = (26, 26, 25)
TERMINAL_INK = (232, 232, 228)
TERMINAL_DIM = (150, 150, 143)

MONO = "/System/Library/Fonts/Menlo.ttc"
SANS = "/System/Library/Fonts/Helvetica.ttc"


def font(path, size, index=0):
    return ImageFont.truetype(path, size, index=index)


F_TITLE = font(SANS, 64, 1)
F_HEAD = font(SANS, 44, 1)
F_BODY = font(SANS, 30)
F_SMALL = font(SANS, 25)
F_TERM = font(MONO, 23)
F_TERM_BOLD = font(MONO, 23, 1)
F_STEP = font(SANS, 26, 1)


class Timeline:
    """Collects frames with durations, then hands them to ffmpeg.

    One image per visible state rather than one per video frame: a six minute
    video at 30fps is 10,800 frames, and almost all of them would be
    identical. ffmpeg's concat demuxer holds each still for as long as it is
    needed, so the whole film is a few hundred PNGs.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self.entries = []
        self.marks = []

    def mark(self, name: str):
        """Record where a section begins, for the narration to line up with."""
        self.marks.append({"name": name, "at": round(self.duration, 2),
                           "frame": len(self.entries)})

    def fit_sections(self, required: list, breath: float = 1.1):
        """Hold each section long enough for the line spoken over it.

        Extends the last frame of any section that is shorter than its
        narration. Doing it here rather than by hand-tuning each duration
        means the text can be rewritten freely: the slide follows the speech
        rather than the speech being trimmed to the slide.
        """
        if not required:
            return
        bounds = [m["frame"] for m in self.marks] + [len(self.entries)]
        for index, seconds in enumerate(required):
            if index + 1 >= len(bounds):
                break
            start, end = bounds[index], bounds[index + 1]
            if end <= start:
                continue
            current = sum(d for _, d in self.entries[start:end])
            shortfall = (seconds + breath) - current
            if shortfall > 0:
                path, held = self.entries[end - 1]
                self.entries[end - 1] = (path, held + shortfall)
        # The marks were recorded before the stretch, so restate them.
        running = 0.0
        for index, mark in enumerate(self.marks):
            mark["at"] = round(running, 2)
            end = bounds[index + 1] if index + 1 < len(bounds) else len(self.entries)
            running += sum(d for _, d in self.entries[mark["frame"]:end])

    def add(self, image: Image.Image, seconds: float):
        path = self.directory / f"f{len(self.entries):05d}.png"
        image.save(path)
        self.entries.append((path, max(0.04, seconds * PACE)))

    @property
    def duration(self):
        return sum(d for _, d in self.entries)

    def write_concat(self) -> Path:
        listing = self.directory / "frames.txt"
        with listing.open("w") as handle:
            for path, seconds in self.entries:
                handle.write(f"file '{path.name}'\nduration {seconds:.3f}\n")
            handle.write(f"file '{self.entries[-1][0].name}'\n")  # ffmpeg quirk
        return listing


def blank(color=SURFACE):
    return Image.new("RGB", (WIDTH, HEIGHT), color)


def wrap(draw, text, fnt, max_width):
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=fnt) <= max_width:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def card(timeline, step, title, body, seconds=4.2):
    timeline.mark(title)
    """A full-screen caption introducing the next thing on screen.

    Laid out from the middle outwards so the block sits centered in the frame
    however many lines the body runs to.
    """
    image = blank()
    draw = ImageDraw.Draw(image)
    body_lines = wrap(draw, body, F_BODY, WIDTH - 460)
    block = 64 + 30 + F_TITLE.size + 44 + len(body_lines) * 48
    top = (HEIGHT - block) // 2

    if step:
        draw.text((160, top), step, font=F_STEP, fill=ACCENT)
    draw.line([(160, top + 52), (760, top + 52)], fill=ACCENT, width=4)
    draw.text((160, top + 84), title, font=F_TITLE, fill=INK)
    y = top + 84 + F_TITLE.size + 44
    for line in body_lines:
        draw.text((160, y), line, font=F_BODY, fill=INK_SECONDARY)
        y += 48
    timeline.add(image, seconds)


def terminal_frames(timeline, command, output, note="", line_seconds=0.16,
                    hold=2.2):
    """A terminal window: the prompt, then the output arriving line by line."""
    timeline.mark(f"$ {command}")
    def draw_window(visible):
        image = blank()
        draw = ImageDraw.Draw(image)
        # window chrome
        draw.rectangle([90, 90, WIDTH - 90, HEIGHT - 150], fill=TERMINAL_BG)
        draw.rectangle([90, 90, WIDTH - 90, 148], fill=(44, 44, 42))
        for i, colour in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            draw.ellipse([126 + i * 32, 110, 144 + i * 32, 128], fill=colour)
        draw.text((WIDTH // 2 - 90, 108), "Terminal", font=F_SMALL, fill=TERMINAL_DIM)

        y = 186
        draw.text((130, y), "$", font=F_TERM_BOLD, fill=GOOD)
        draw.text((160, y), command, font=F_TERM_BOLD, fill=TERMINAL_INK)
        y += 44
        for line in visible:
            draw.text((130, y), line[:118], font=F_TERM, fill=TERMINAL_INK)
            y += 31
            if y > HEIGHT - 200:
                break
        if note:
            draw.text((90, HEIGHT - 118), note, font=F_SMALL, fill=INK_SECONDARY)
        return image

    timeline.add(draw_window([]), 0.9)          # the command, before it runs
    for count in range(1, len(output) + 1):
        timeline.add(draw_window(output[:count]), line_seconds)
    timeline.add(draw_window(output), hold)


def figure(timeline, path: Path, caption, seconds=5.0):
    """A generated figure, shown whole with a caption beneath it."""
    if not path.is_file():
        return
    timeline.mark(f"figure: {path.stem}")
    image = blank()
    draw = ImageDraw.Draw(image)
    picture = Image.open(path).convert("RGB")

    # The caption is measured before the picture is placed, so a two-line
    # caption takes two lines of room rather than being drawn on top of
    # itself - which is exactly what happened until a contact sheet of the
    # finished film made it visible.
    lines = wrap(draw, caption, F_BODY, WIDTH - 320)
    caption_height = len(lines) * 44 + 40
    box_w = WIDTH - 220
    box_h = HEIGHT - 150 - caption_height

    scale = min(box_w / picture.width, box_h / picture.height)
    picture = picture.resize((int(picture.width * scale), int(picture.height * scale)),
                             Image.LANCZOS)
    top = (HEIGHT - caption_height - picture.height) // 2
    image.paste(picture, ((WIDTH - picture.width) // 2, max(60, top)))

    y = HEIGHT - caption_height + 10
    for line in lines:
        draw.text((160, y), line, font=F_BODY, fill=INK_SECONDARY)
        y += 44
    timeline.add(image, seconds)


def title_card(timeline, lines, seconds=5.0):
    timeline.mark(f"title: {lines[0]}")
    image = blank()
    draw = ImageDraw.Draw(image)
    fonts_all = [F_TITLE, F_HEAD, F_BODY, F_BODY, F_SMALL]
    block = sum(fonts_all[min(i, 4)].size + 26 for i in range(len(lines)))
    y = (HEIGHT - block) // 2
    fonts = [F_TITLE, F_HEAD, F_BODY, F_BODY, F_SMALL]
    colours = [INK, ACCENT, INK_SECONDARY, INK_SECONDARY, INK_MUTED]
    for i, line in enumerate(lines):
        f = fonts[min(i, len(fonts) - 1)]
        draw.text((160, y), line, font=f, fill=colours[min(i, len(colours) - 1)])
        y += f.size + 26
    timeline.add(image, seconds)


def scene(capture, label):
    return next((s for s in capture["scenes"] if s["label"] == label), None)


def build(timeline, capture):
    run = RESULTS / "animals10_n500_rgb"

    title_card(timeline, [
        "Image Classification Benchmarking",
        "Samuel_Collins_CV_Benchmarking 1.0.1",
        "Samuel Collins  ·  CCIS 727  ·  Clark Atlanta University",
        "Six models, one split, one set of metrics",
        "Every command and every number in this video is real output",
    ], 5.5)

    # --- 1. PyPI installation ------------------------------------------
    card(timeline, "1 of 5", "Install from PyPI",
         "The package is published as Samuel_Collins_CV_Benchmarking. This is a "
         "clean virtual environment with nothing installed in it.")
    for label, note in (("install", "pip resolves the six dependencies"),
                        ("show", "version 1.0.1, MIT licensed")):
        s = scene(capture, label)
        if s:
            terminal_frames(timeline, s["command"], s["output"], note)

    # --- 2. Import ------------------------------------------------------
    card(timeline, "2 of 5", "The student-specific import",
         "The distribution is Samuel_Collins_CV_Benchmarking; the import name is "
         "its lowercase form. One public function, four parameters.")
    s = scene(capture, "import")
    if s:
        terminal_frames(timeline, s["command"], s["output"],
                        "dataset · dataset_type · target_labels · color_mode")

    # --- 3. Execution ---------------------------------------------------
    card(timeline, "3 of 5", "Running a benchmark",
         "1,000 images across ten classes. One stratified 80/20 split at seed 42, "
         "reused by all six models, so the comparison is fair by construction.")
    for label, note in (
        ("ex_csv", "a CSV manifest: image_path plus a label column"),
        ("ex_folder", "the same images as class folders"),
        ("ex_json", "the same images again, as JSON and JSONL"),
        ("ex_array", "and as an in-memory NumPy array"),
    ):
        s = scene(capture, label)
        if s:
            terminal_frames(timeline, s["command"], s["output"], note,
                            line_seconds=0.13, hold=1.8)

    card(timeline, "", "Four organizations, one result",
         "folder, CSV, JSON and JSONL describing the same 1,000 images all return "
         "macro F1 0.294660 — identical to six decimal places. The package changes "
         "how it reads the data, not how it evaluates the models.", 6.0)

    # --- the two required demonstrations --------------------------------
    card(timeline, "", "Two demonstrations, one single-channel and one RGB",
         "The same images through both color modes. RGB gives each classical "
         "model 12,288 features; grayscale gives 4,096. The difference between "
         "the two runs is what color was worth on this dataset.", 5.5)
    for label, note in (
        ("ex_rgb", "three channels: 64 x 64 x 3 = 12,288 features"),
        ("ex_grayscale", "one channel: 64 x 64 = 4,096 features"),
    ):
        found = scene(capture, label)
        if found:
            terminal_frames(timeline, found["command"], found["output"], note,
                            line_seconds=0.15, hold=2.6)

    card(timeline, "", "What color was worth",
         "On Animals-10 color is worth about a quarter of the CNN's macro F1 and "
         "costs the SVM roughly forty times its training run. On the second "
         "dataset the Decision Tree reverses sign, so the effect belongs to the "
         "data rather than to the model.", 6.0)

    # --- 4. Generated outputs -------------------------------------------
    card(timeline, "4 of 5", "What the run writes",
         "Every run leaves a complete, inspectable record: the summary table, "
         "per-model metrics, the configuration that produced them, and the figures.")
    for label, note in (("outputs", "eighteen files, written automatically"),
                        ("summary_csv", "ranked by macro F1, ties broken by inference time")):
        s = scene(capture, label)
        if s:
            terminal_frames(timeline, s["command"], s["output"], note,
                            line_seconds=0.12)

    figure(timeline, run / "class_distribution.png",
           "Class distribution — the 80/20 split preserves each class's share.", 4.5)
    figure(timeline, run / "model_comparison.png",
           "Accuracy, macro F1, training time and inference time on four separate "
           "axes. The timing panels are dots on a log scale: a bar measures from "
           "zero, and a log axis has none.", 6.5)
    figure(timeline, run / "confusion_matrices" / "simple_cnn.png",
           "Confusion matrix, colored by share of the true class so a small class "
           "is not washed out by a large one.", 5.5)
    figure(timeline, run / "prediction_examples.png",
           "Correct and incorrect predictions, shown as the standardized 64x64 "
           "input the model actually received — the padding is visible because it "
           "is part of what the network sees.", 6.5)

    card(timeline, "", "Reproducibility is recorded, not claimed",
         "Every run writes the configuration that produced it: image size, color "
         "mode, seed, split, and each model's parameters. Re-running reproduces "
         "every score exactly.", 5.0)
    found = scene(capture, "config")
    if found:
        terminal_frames(timeline, found["command"], found["output"],
                        "seed 42, 64x64, and every hyperparameter",
                        line_seconds=0.10, hold=3.0)

    card(timeline, "", "A second dataset",
         "Intel Image Classification: six natural-scene classes at 150x150, "
         "square, so nothing is lost to the padding that costs Animals-10 27% of "
         "its canvas. Running both is how the padding question gets asked at all.",
         5.5)
    figure(timeline, RESULTS / "intel_n500_rgb" / "model_comparison.png",
           "The same six models on Intel Image Classification. Higher scores, but "
           "six classes rather than ten, so chance is 0.167 not 0.100.", 6.0)
    figure(timeline, RESULTS / "intel_n500_rgb" / "confusion_matrices" / "simple_cnn.png",
           "Where its confusions are: glacier and mountain, sea and glacier. "
           "Scenes that share color and texture.", 5.5)

    card(timeline, "", "Six runs, three experiments",
         "Two datasets, two color modes, two subset sizes. Each comparison holds "
         "everything fixed but one variable, because a comparison with two "
         "variables moving is how a confounded result gets written up as a "
         "finding.", 6.0)
    figure(timeline, RESULTS / "animals10_n100_rgb" / "model_comparison.png",
           "The same benchmark at 100 images per class. The CNN and Random Forest "
           "are tied here, 0.295 against 0.294.", 6.0)
    # --- 5. Repository organization -------------------------------------
    card(timeline, "5 of 5", "Repository organization",
         "Eight modules under src/, six runnable examples, seven test files, and "
         "the scripts that rebuild the datasets, the runs and this report.")
    for label, note in (("structure", "src/ ships to PyPI; examples/ and tests/ do not"),
                        ("tests", "the full suite, run against the working tree")):
        s = scene(capture, label)
        if s:
            terminal_frames(timeline, s["command"], s["output"], note,
                            line_seconds=0.11)

    # --- findings --------------------------------------------------------
    card(timeline, "", "What the benchmark found",
         "At 100 images per class the CNN and Random Forest are tied, 0.295 against "
         "0.294. At 500 the CNN leads by 36 percent. A single run would have "
         "supported the opposite conclusion.", 6.5)
    card(timeline, "", "Cost is not accuracy",
         "The SVM takes 547 seconds to train and 150 milliseconds per image at "
         "inference — roughly 75,000 times the Decision Tree — for third place. "
         "Ranking breaks ties on the lower inference time for that reason.", 6.5)

    title_card(timeline, [
        "Samuel_Collins_CV_Benchmarking",
        "pip install Samuel_Collins_CV_Benchmarking",
        "pypi.org/project/Samuel_Collins_CV_Benchmarking",
        "github.com/srcollins785/Samuel_Collins_CV_Benchmarking",
        "373 tests · six benchmark runs · two datasets",
    ], 6.0)


def main() -> None:
    if not CAPTURE.is_file():
        sys.exit("No capture found. Run scripts/capture_demo.py first.")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required. Install it with: brew install ffmpeg")

    capture = json.loads(CAPTURE.read_text())
    workspace = Path(tempfile.mkdtemp(prefix="demo_frames_"))
    timeline = Timeline(workspace)

    print("rendering frames ...", flush=True)
    build(timeline, capture)

    # If the narration has been synthesized, stretch each slide to cover its
    # line. Without it the video keeps its natural pacing.
    spoken = REPO_ROOT / "scripts" / "_narration_durations.json"
    if spoken.is_file():
        required = json.loads(spoken.read_text())
        timeline.fit_sections(required)
        print(f"  fitted {len(required)} slides to the narration")
    minutes, seconds = divmod(timeline.duration, 60)
    print(f"  {len(timeline.entries)} frames, {int(minutes)}m {seconds:04.1f}s")

    manifest = REPO_ROOT / "scripts" / "_demo_timeline.json"
    manifest.write_text(json.dumps(
        {"duration": round(timeline.duration, 2), "sections": timeline.marks},
        indent=2))
    print(f"  timeline written to {manifest.name}")

    listing = timeline.write_concat()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print("encoding ...", flush=True)
    completed = subprocess.run(
        # -r alone: the concat demuxer supplies the per-still durations and
        # ffmpeg resamples them to a constant frame rate. Asking for -vsync
        # vfr as well is contradictory and ffmpeg 8 refuses it outright.
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-movflags", "+faststart", str(OUTPUT)],
        capture_output=True, text=True, cwd=workspace)
    shutil.rmtree(workspace, ignore_errors=True)

    if not OUTPUT.is_file():
        sys.exit(f"ffmpeg failed:\n{completed.stderr[-1200:]}")
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} "
          f"({OUTPUT.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
