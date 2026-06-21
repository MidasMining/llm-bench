#!/usr/bin/env python3
"""
run_all.py — Canonical benchmarking entry point (v3.0)
=====================================================
Orchestrates all standalone benchmark tools into a single versioned run
with structured output and durable metadata.

Does NOT duplicate any benchmark logic — pure orchestration.

Usage:
    python run_all.py \\
      --run-name b70-qwen3-32b-vllm-xpu \\
      --model Qwen/Qwen3-32B \\
      --api-url http://127.0.0.1:8000/v1 \\
      --hardware "Arc Pro B70" --backend "vLLM XPU"

    # Skip legs or continue on error:
    python run_all.py --run-name quick-test --model m --api-url http://... \\
      --skip-parallel --skip-context --continue-on-error

    # Generate unified report at end:
    python run_all.py --run-name full --model m --api-url http://... --report
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

VERSION = "3.0"


def build_run_dir(base: str, run_name: str) -> Path:
    """Create a versioned timestamped run directory."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dirname = f"{run_name}-v{VERSION}-{ts}"
    run_dir = Path(base) / dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("quality", "decode", "parallel", "context", "report"):
        (run_dir / sub).mkdir(exist_ok=True)
    return run_dir


def write_metadata(run_dir: Path, args: argparse.Namespace,
                   legs_run: list, legs_skipped: list) -> Path:
    """Write RUN_METADATA.json."""
    meta = {
        "bench_version": VERSION,
        "run_name": args.run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "api_url": args.api_url,
        "hardware": args.hardware,
        "backend": args.backend,
        "driver": args.driver,
        "host": platform.node(),
        "notes": args.notes,
        "command": " ".join(sys.argv),
        "tools": {
            "quality": "compare_models.py",
            "decode": "decode_rate_bench.py v2.0",
            "parallel": "parallel_benchmark.py v2.0",
            "context": "long_context_test.py",
        },
        "legs_run": legs_run,
        "legs_skipped": legs_skipped,
    }
    path = run_dir / "RUN_METADATA.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path


def run_leg(name: str, cmd: list, run_dir: Path,
            continue_on_error: bool) -> bool:
    """Run a benchmark leg via subprocess. Returns True on success."""
    print(f"\n{'=' * 70}")
    print(f"  LEG: {name.upper()}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n*** {name} failed (exit {result.returncode}) ***")
        if not continue_on_error:
            print("Stopping. Use --continue-on-error to keep going.")
            sys.exit(result.returncode)
        return False
    return True


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=f"llm-bench orchestrator v{VERSION} — canonical entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python run_all.py --run-name b70-qwen3 --model Qwen/Qwen3-32B \\
    --api-url http://127.0.0.1:8000/v1 --hardware "Arc B70" --backend "vLLM XPU"

  python run_all.py --run-name quick --model m --api-url http://... \\
    --skip-parallel --skip-context --continue-on-error --report
""",
    )

    # Required
    parser.add_argument("--run-name", required=True,
                        help="Short identifier (used in directory name)")
    parser.add_argument("--model", required=True, help="Model ID")
    parser.add_argument("--api-url", required=True, help="OpenAI-compatible API base URL")

    # Metadata
    parser.add_argument("--hardware", default=None, help="Hardware description")
    parser.add_argument("--backend", default=None, help="Inference backend description")
    parser.add_argument("--driver", default=None, help="GPU driver version")
    parser.add_argument("--notes", default=None, help="Free-form notes")

    # Quality leg
    parser.add_argument("--tests", nargs="+", default=None,
                        help="Quality tests to run (default: all)")

    # Decode leg
    parser.add_argument("--decode-targets", nargs="+", type=int,
                        default=[500, 2000, 4000, 6000],
                        help="Decode bench input-token targets (default: 500 2000 4000 6000)")
    parser.add_argument("--decode-runs", type=int, default=3,
                        help="Runs per decode target (default: 3)")

    # Parallel leg
    parser.add_argument("--parallel-max-concurrent", type=int, default=64,
                        help="Max concurrency (default: 64)")
    parser.add_argument("--parallel-levels", default="1,2,4,8,16,32,64",
                        help="Concurrency levels (default: 1,2,4,8,16,32,64)")

    # Context leg
    parser.add_argument("--context-size", type=int, default=32000,
                        help="Long-context test token count (default: 32000)")

    # Skip flags
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--skip-parallel", action="store_true")
    parser.add_argument("--skip-context", action="store_true")

    # Behavior
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue to next leg on failure")
    parser.add_argument("--report", action="store_true",
                        help="Generate unified report after benchmarks")
    parser.add_argument("--results-base", default="results",
                        help="Base directory for run output (default: results)")

    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {VERSION}")

    args = parser.parse_args()

    # --- Build run directory ---
    run_dir = build_run_dir(args.results_base, args.run_name)
    print(f"llm-bench orchestrator v{VERSION}")
    print(f"Run directory: {run_dir}")

    legs_run = []
    legs_skipped = []
    coe = args.continue_on_error

    # --- LEG 1: Quality ---
    if not args.skip_quality:
        cmd = [
            sys.executable, str(script_dir / "compare_models.py"),
            "--model", args.model,
            "--api-url", args.api_url,
            "--output", str(run_dir / "quality"),
        ]
        if args.tests:
            cmd += ["--tests"] + args.tests
        if run_leg("quality", cmd, run_dir, coe):
            legs_run.append("quality")
        else:
            legs_skipped.append("quality (failed)")
    else:
        legs_skipped.append("quality")

    # --- LEG 2: Decode rate ---
    if not args.skip_decode:
        cmd = [
            sys.executable, str(script_dir / "decode_rate_bench.py"),
            "--model", args.model,
            "--api-url", args.api_url,
            "--output", str(run_dir / "decode"),
            "--targets",
        ] + [str(t) for t in args.decode_targets] + [
            "--runs", str(args.decode_runs),
        ]
        if run_leg("decode", cmd, run_dir, coe):
            legs_run.append("decode")
        else:
            legs_skipped.append("decode (failed)")
    else:
        legs_skipped.append("decode")

    # --- LEG 3: Parallel ---
    if not args.skip_parallel:
        cmd = [
            sys.executable, str(script_dir / "parallel_benchmark.py"),
            "--model", args.model,
            "--api-url", args.api_url,
            "--max-concurrent", str(args.parallel_max_concurrent),
            "--concurrency-levels", args.parallel_levels,
            "--output", str(run_dir / "parallel"),
        ]
        if run_leg("parallel", cmd, run_dir, coe):
            legs_run.append("parallel")
        else:
            legs_skipped.append("parallel (failed)")
    else:
        legs_skipped.append("parallel")

    # --- LEG 4: Long context ---
    if not args.skip_context:
        cmd = [
            sys.executable, str(script_dir / "long_context_test.py"),
            "--model", args.model,
            "--api-url", args.api_url,
            "--context-size", str(args.context_size),
            "--output", str(run_dir / "context" / "long_context_results.json"),
        ]
        if run_leg("context", cmd, run_dir, coe):
            legs_run.append("context")
        else:
            legs_skipped.append("context (failed)")
    else:
        legs_skipped.append("context")

    # --- Write metadata (after legs so we know which ran) ---
    meta_path = write_metadata(run_dir, args, legs_run, legs_skipped)
    print(f"\nMetadata: {meta_path}")

    # --- Optional report ---
    if args.report:
        cmd = [
            sys.executable, str(script_dir / "report.py"),
            "--run-dir", str(run_dir),
            "--format", "markdown",
        ]
        out_path = run_dir / "report" / "report.md"
        print(f"\n{'=' * 70}")
        print(f"  REPORT")
        print(f"{'=' * 70}\n")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            with open(out_path, "w") as f:
                f.write(result.stdout)
            print(f"Report saved to: {out_path}")
            # Also print to terminal
            print(result.stdout)
        else:
            print("Report generation failed or produced no output.")
            if result.stderr:
                print(result.stderr)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")
    print(f"Run directory: {run_dir}")
    print(f"Legs run:      {', '.join(legs_run) if legs_run else '(none)'}")
    if legs_skipped:
        print(f"Legs skipped:  {', '.join(legs_skipped)}")


if __name__ == "__main__":
    main()
