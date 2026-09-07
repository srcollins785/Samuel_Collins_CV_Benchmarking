"""Run the whole pipeline: tests, benchmarks, report, PDF.

Each step is a separate script invoked as a subprocess, and the first failure
stops the run. Keeping them separate rather than importing one another means
any step can be run alone - which matters, because the benchmarks take about
twenty minutes and the report takes one second, and most of the time it is
the report that changed.

Steps
-----
1. pytest                    the suite must pass before results are believed
2. run_benchmarks.py         every built tier, RGB
3. run_benchmarks.py         every built tier, grayscale
4. generate_report.py        Markdown, from the files the benchmarks wrote
5. build_report_pdf.py       PDF, via headless Chrome

Usage
-----
    python scripts/run_all.py                 # everything, about 25 minutes
    python scripts/run_all.py --report-only   # steps 4-5, about 5 seconds
    python scripts/run_all.py --skip-tests    # everything except step 1
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "benchmark_results"


def steps(skip_tests: bool, report_only: bool) -> list:
    """The pipeline, as (label, command, rough duration) triples."""
    everything = [
        ("tests", [sys.executable, "-m", "pytest", "tests/", "-q"], "1 minute"),
        ("benchmarks (rgb)",
         [sys.executable, str(SCRIPTS / "run_benchmarks.py")], "20 minutes"),
        ("benchmarks (grayscale)",
         [sys.executable, str(SCRIPTS / "run_benchmarks.py"),
          "--color-mode", "grayscale"], "3 minutes"),
        ("report (markdown)",
         [sys.executable, str(SCRIPTS / "generate_report.py")], "1 second"),
        ("report (pdf)",
         [sys.executable, str(SCRIPTS / "build_report_pdf.py")], "5 seconds"),
    ]
    if report_only:
        return everything[3:]
    if skip_tests:
        return everything[1:]
    return everything


def preflight(report_only: bool) -> None:
    """Fail before a twenty minute run rather than during it."""
    if report_only:
        if not any(RESULTS_DIR.glob("*/benchmark_summary.csv")):
            sys.exit(
                f"--report-only needs existing results under "
                f"{RESULTS_DIR.relative_to(REPO_ROOT)}, and there are none. "
                "Run without the flag first."
            )
        return

    tiers = [p.name for p in sorted(DATA_DIR.glob("*")) if (p / "labels.csv").is_file()] \
        if DATA_DIR.is_dir() else []
    if not tiers:
        sys.exit(
            "No dataset tiers found under data/. Build at least one first:\n"
            "  python scripts/download_animals10.py --per-class 500\n"
            "  python scripts/download_intel.py --per-class 500\n"
            "Both need Kaggle API credentials at ~/.kaggle/kaggle.json."
        )
    print(f"tiers to benchmark: {', '.join(tiers)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true",
                        help="regenerate the report from existing results")
    parser.add_argument("--skip-tests", action="store_true",
                        help="skip the test suite")
    arguments = parser.parse_args()

    preflight(arguments.report_only)
    pipeline = steps(arguments.skip_tests, arguments.report_only)

    print(f"\n{len(pipeline)} step(s):", flush=True)
    for label, _, duration in pipeline:
        print(f"  - {label} (about {duration})", flush=True)

    started = time.perf_counter()
    for number, (label, command, _) in enumerate(pipeline, start=1):
        print(f"\n{'=' * 68}\n[{number}/{len(pipeline)}] {label}\n{'=' * 68}", flush=True)
        step_started = time.perf_counter()
        # cwd is the repo root so the scripts' relative paths resolve the same
        # way whether this is run from here or from anywhere else.
        completed = subprocess.run(command, cwd=REPO_ROOT)
        elapsed = time.perf_counter() - step_started

        if completed.returncode != 0:
            sys.exit(
                f"\n{label} failed with exit code {completed.returncode} "
                f"after {elapsed:.1f}s. Stopping; later steps would report "
                "results this one did not produce."
            )
        print(f"\n{label}: {elapsed:.1f}s", flush=True)

    total = time.perf_counter() - started
    report = REPO_ROOT / "report" / "CV_Benchmarking_Report.pdf"
    print(f"\n{'=' * 68}", flush=True)
    print(f"pipeline complete in {total / 60:.1f} minutes", flush=True)
    if report.is_file():
        print(f"report: {report.relative_to(REPO_ROOT)} "
              f"({report.stat().st_size / 1_000_000:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
