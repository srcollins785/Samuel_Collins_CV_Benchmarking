"""Run the demonstration commands and record exactly what they print.

The video is assembled from this capture rather than from typed-out text, so
every line on screen is output a command actually produced. If a command
changes its output, re-running this changes the video.

Writes scripts/_demo_capture.json.

Usage
-----
    python scripts/capture_demo.py
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CAPTURE = REPO_ROOT / "scripts" / "_demo_capture.json"

# Noise that is real but not worth screen time: a spurious BLAS warning from
# numpy on Apple silicon, and pip's notice about its own version.
NOISE = re.compile(
    r"matmul|ret = a @ b|raw_prediction|grad\[|warnings\.warn|findfont|"
    r"You are using pip version|consider upgrading|^\s*$"
)


def run(command, cwd=None, keep=None, label=None, actual=None, shown=False):
    """Run a command and record it with what it printed.

    ``command`` is what the video displays; ``actual`` is what runs, and
    differs only where an absolute interpreter or venv path is unavoidable.
    The output is whatever the command produced - never edited, only trimmed
    to the lines the frame has room for.
    """
    print(f"  $ {command}", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(actual or command, shell=True, cwd=cwd or REPO_ROOT,
                               capture_output=True, text=True)
    elapsed = time.perf_counter() - started

    lines = [l.rstrip() for l in (completed.stdout + completed.stderr).splitlines()]
    lines = [l for l in lines if l.strip() and not NOISE.search(l)]
    if keep:
        lines = lines[:keep] if keep > 0 else lines[keep:]

    print(f"    -> {len(lines)} line(s), {elapsed:.1f}s", flush=True)
    return {
        "label": label or "",
        "command": command,
        "output": lines,
        "seconds": round(elapsed, 1),
        "returncode": completed.returncode,
    }


def main() -> None:
    """Capture the demonstration, using commands a person would actually type.

    Earlier versions ran inline `python -c` blobs. The output was real, but
    the command on screen was not something anyone would write, and the video
    should show a viewer what they would do themselves. These are the same
    example scripts the repository ships.
    """
    scenes = []
    venv = Path(tempfile.mkdtemp(prefix="demo_venv_")) / "venv"

    print("\n[1] install from PyPI into a clean environment", flush=True)
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    pip, python = venv / "bin" / "pip", venv / "bin" / "python"
    scenes.append(run(f'pip install Samuel_Collins_CV_Benchmarking',
                      shown=True, actual=f'"{pip}" install Samuel_Collins_CV_Benchmarking',
                      keep=-6, label="install"))
    scenes.append(run("pip show Samuel_Collins_CV_Benchmarking",
                      actual=f'"{pip}" show Samuel_Collins_CV_Benchmarking',
                      keep=6, label="show"))

    print("\n[2] the student-specific import", flush=True)
    one_liner = ("from samuel_collins_cv_benchmarking import "
                 "benchmark_image_classification as run; print(run)")
    scenes.append(run(f"python -c \"{one_liner}\"",
                      actual=f'"{python}" -c "{one_liner}"', label="import"))

    # The examples write benchmark_results relative to the working directory,
    # so they run somewhere disposable rather than into the repository's own
    # committed results.
    workspace = Path(tempfile.mkdtemp(prefix="demo_run_"))
    print("\n[3] each dataset organization, via the shipped examples", flush=True)
    # CSV runs last so the summary table captured next belongs to the run the
    # video has just been discussing, rather than to whichever example
    # happened to finish most recently.
    for name, label in [("folder_input_example", "ex_folder"),
                        ("json_input_example", "ex_json"),
                        ("array_input_example", "ex_array"),
                        ("csv_input_example", "ex_csv")]:
        scenes.append(run(
            f"python examples/{name}.py",
            actual=f'PYTHONWARNINGS=ignore "{python}" '
                   f'"{REPO_ROOT}/examples/{name}.py"',
            cwd=workspace, keep=-12, label=label))

    print("\n[4] the two color-mode demonstrations", flush=True)
    for name, label in [("rgb_example", "ex_rgb"),
                        ("grayscale_example", "ex_grayscale")]:
        scenes.append(run(
            f"python examples/{name}.py",
            actual=f'PYTHONWARNINGS=ignore "{python}" '
                   f'"{REPO_ROOT}/examples/{name}.py"',
            cwd=workspace, keep=-10, label=label))

    print("\n[5] the run configuration, for reproducibility", flush=True)
    scenes.append(run(
        "cat benchmark_results/run_configuration.json | head -24",
        cwd=workspace, label="config"))

    print("\n[6] the generated outputs", flush=True)
    scenes.append(run("find benchmark_results -type f | sort",
                      cwd=workspace, label="outputs"))
    scenes.append(run(
        "column -s, -t < benchmark_results/benchmark_summary.csv | cut -c1-104",
        cwd=workspace, label="summary_csv"))

    print("\n[5] repository organization", flush=True)
    scenes.append(run("find src examples tests -name '*.py' | sort", label="structure"))
    scenes.append(run("pytest tests/ -q",
                      actual=f'"{REPO_ROOT}/.venv/bin/python" -m pytest tests/ -q',
                      keep=-4, label="tests"))

    shutil.rmtree(venv.parent, ignore_errors=True)
    CAPTURE.write_text(json.dumps({"scenes": scenes}, indent=2))
    print(f"\nwrote {CAPTURE.relative_to(REPO_ROOT)} "
          f"({sum(len(s['output']) for s in scenes)} captured lines)")


if __name__ == "__main__":
    main()
