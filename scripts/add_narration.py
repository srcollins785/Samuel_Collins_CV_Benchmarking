"""Synthesize the narration and mux it onto the demonstration video.

Each line is spoken separately and placed at its own start time, rather than
read as one continuous take, so a line that runs long cannot push everything
after it out of sync with the slides.

Before rendering anything it checks every line against the gap before the
next one and reports any that overrun. An overrun is not fatal - speech
simply overlaps the following slide - but it is the thing most likely to
make the result feel wrong, so it is reported rather than discovered.

Usage
-----
    python scripts/add_narration.py --list-voices
    python scripts/add_narration.py --voice "Ava (Premium)"
    python scripts/add_narration.py --voice Ava --rate 165 --check-only
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from narration import NARRATION  # noqa: E402

VIDEO = REPO_ROOT / "report" / "CV_Benchmarking_Demo.mp4"
OUTPUT = REPO_ROOT / "report" / "CV_Benchmarking_Demo_narrated.mp4"


def available_voices() -> list:
    result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
    voices = []
    for line in result.stdout.splitlines():
        if " en_" in line:
            voices.append(line.split(" en_")[0].strip())
    return voices


def duration_of(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True)
    return float(result.stdout.strip() or 0)


def synthesize(voice: str, rate: int, directory: Path) -> list:
    """Speak each line separately; return (start, path, spoken length)."""
    clips, paths = [], []
    for index, text in enumerate(NARRATION):
        path = directory / f"line{index:03d}.aiff"
        completed = subprocess.run(
            ["say", "-v", voice, "-r", str(rate), "-o", str(path), text],
            capture_output=True, text=True)
        if completed.returncode != 0 or not path.is_file():
            sys.exit(f"say failed for voice {voice!r}:\n{completed.stderr.strip()}")
        clips.append(duration_of(path))
        paths.append(path)
    return clips, paths


def slide_starts() -> list:
    """Where each slide begins in the finished video."""
    import json
    manifest = REPO_ROOT / "scripts" / "_demo_timeline.json"
    if not manifest.is_file():
        sys.exit("No timeline. Run scripts/build_demo_video.py first.")
    return [s["at"] for s in json.loads(manifest.read_text())["sections"]]


def report_fit(spoken, starts) -> int:
    """Show each line against the slide it is spoken over."""
    print(f"\n{'start':>8}  {'spoken':>7}  {'slide':>7}  line")
    overruns = 0
    for index, seconds in enumerate(spoken):
        if index >= len(starts):
            break
        slot = (starts[index + 1] - starts[index]) if index + 1 < len(starts) else 12.0
        flag = ""
        if seconds > slot + 0.05:
            flag = f"  OVER by {seconds - slot:.1f}s"
            overruns += 1
        minutes, second = divmod(starts[index], 60)
        print(f"{int(minutes)}:{second:05.2f}  {seconds:6.1f}s  {slot:6.1f}s  "
              f"{NARRATION[index][:40]}...{flag}")
    print(f"\n  {len(spoken)} lines, {overruns} overrunning")
    return overruns


def build_track(paths, starts, total: float, directory: Path) -> Path:
    """Lay every clip onto one bed at the moment its slide appears."""
    track = directory / "narration.m4a"
    inputs, filters, labels = [], [], []
    for index, path in enumerate(paths):
        offset = int(starts[index] * 1000)
        inputs += ["-i", str(path)]
        filters.append(f"[{index}:a]adelay={offset}|{offset}[a{index}]")
        labels.append(f"[a{index}]")
    graph = (";".join(filters) + ";" + "".join(labels)
             + f"amix=inputs={len(paths)}:normalize=0[out]")

    completed = subprocess.run(
        ["ffmpeg", "-y", *inputs, "-filter_complex", graph, "-map", "[out]",
         "-t", str(total), "-c:a", "aac", "-b:a", "192k", str(track)],
        capture_output=True, text=True)
    if not track.is_file():
        sys.exit(f"could not build the narration track:\n{completed.stderr[-900:]}")
    return track


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--rate", type=int, default=170,
                        help="words per minute (default 170)")
    parser.add_argument("--emit-durations", action="store_true",
                        help="synthesize and record how long each line takes, "
                             "so the video can be rebuilt to fit it")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--list-voices", action="store_true")
    arguments = parser.parse_args()

    if arguments.list_voices:
        for voice in available_voices():
            print(" ", voice)
        return

    # say accepts any name and silently substitutes the system default, so an
    # unknown voice has to be rejected here or it passes unnoticed.
    if arguments.voice not in available_voices():
        print(f"Voice {arguments.voice!r} is not installed. `say` would fall "
              f"back to the default without saying so. Installed English "
              f"voices:\n", file=sys.stderr)
        for voice in available_voices():
            print("  ", voice, file=sys.stderr)
        sys.exit(1)

    workspace = Path(tempfile.mkdtemp(prefix="narration_"))
    try:
        print(f"synthesizing {len(NARRATION)} lines with {arguments.voice} "
              f"at {arguments.rate} wpm ...", flush=True)
        spoken, paths = synthesize(arguments.voice, arguments.rate, workspace)

        if arguments.emit_durations:
            target = REPO_ROOT / "scripts" / "_narration_durations.json"
            target.write_text(json.dumps([round(s, 3) for s in spoken], indent=2))
            print(f"  wrote {target.name}; now rebuild the video so each slide "
                  f"covers its line")
            return

        starts = slide_starts()
        overruns = report_fit(spoken, starts)
        if arguments.check_only:
            return

        if not VIDEO.is_file():
            sys.exit(f"{VIDEO.name} is missing. Run build_demo_video.py first.")
        total = duration_of(VIDEO)
        track = build_track(paths, starts, total, workspace)

        print("\nmuxing ...", flush=True)
        completed = subprocess.run(
            ["ffmpeg", "-y", "-i", str(VIDEO), "-i", str(track),
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-map", "0:v:0", "-map", "1:a:0", "-shortest",
             "-movflags", "+faststart", str(OUTPUT)],
            capture_output=True, text=True)
        if not OUTPUT.is_file():
            sys.exit(f"mux failed:\n{completed.stderr[-900:]}")
        print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} "
              f"({OUTPUT.stat().st_size / 1_000_000:.1f} MB)")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
