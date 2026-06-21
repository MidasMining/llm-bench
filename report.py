#!/usr/bin/env python3
"""Unified benchmark report generator.

Reads JSON results from all benchmark tools and produces a combined summary
showing quality, decode rate, prefill latency, and concurrency scaling in
one view.

Usage:
  # From a run_all.py run directory:
  python report.py --run-dir results/a4000-qwen3-v3.0-20260621-184233/

  # Auto-discover results for a model:
  python report.py --model seed-oss

  # Specify results directories:
  python report.py --compare-dir results/8xA4000 --decode-dir results/decode \\
      --parallel-dir results/parallel

  # Generate markdown report:
  python report.py --model seed-oss --format markdown > report.md
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def find_latest_result(directory: str, prefix: str, model_filter: str = None) -> Optional[Dict]:
    """Find the most recent JSON result file matching prefix in directory."""
    pattern = os.path.join(directory, f"{prefix}*.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            # Apply model filter if specified
            if model_filter:
                model_name = ""
                if isinstance(data, list):
                    # compare_models output is a list
                    model_name = data[0].get("name", "") if data else ""
                else:
                    model_name = data.get("model", "")
                if model_filter.lower() not in model_name.lower():
                    continue
            return {"file": f, "data": data}
        except (json.JSONDecodeError, IndexError):
            continue
    return None


def find_all_results(directory: str, prefix: str) -> List[Dict]:
    """Find all JSON result files matching prefix in directory."""
    pattern = os.path.join(directory, f"{prefix}*.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    results = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            results.append({"file": f, "data": data})
        except json.JSONDecodeError:
            continue
    return results


def format_quality_section(data: Dict) -> str:
    """Format quality benchmark results."""
    lines = []

    if isinstance(data, list):
        # Multi-model comparison
        for model_data in data:
            lines.append(format_single_model_quality(model_data))
    else:
        lines.append(format_single_model_quality(data))

    return "\n".join(lines)


def format_single_model_quality(model_data: Dict) -> str:
    """Format quality results for a single model."""
    lines = []
    name = model_data.get("name", "Unknown")
    seq = model_data.get("sequential", {})
    tests = seq.get("tests", [])

    total_score = sum(t.get("score", 0) for t in tests)
    total_max = sum(t.get("max_score", 0) for t in tests)
    pct = (total_score / total_max * 100) if total_max > 0 else 0
    throughput = seq.get("throughput", 0)
    avg_ttft = seq.get("avg_ttft_seconds", 0)

    lines.append(f"  Model: {name}")
    lines.append(f"  Quality: {total_score}/{total_max} ({pct:.1f}%)")
    lines.append(f"  Throughput (sequential): {throughput:.1f} t/s")
    if avg_ttft > 0:
        lines.append(f"  Avg TTFT (prefill): {avg_ttft:.3f}s")

    # Per-test breakdown
    lines.append(f"  {'Test':<30} {'Score':<10} {'TTFT':<8} {'Time'}")
    lines.append(f"  {'-'*60}")
    for t in tests:
        ttft_str = f"{t.get('ttft_seconds', 0):.2f}s" if t.get('ttft_seconds', 0) > 0 else "N/A"
        status = "PASS" if t.get("passed") else "FAIL"
        lines.append(
            f"  {t['name']:<30} "
            f"{int(t.get('score', 0))}/{int(t.get('max_score', 0))} {status:<5} "
            f"{ttft_str:<8} {t.get('time_seconds', 0):.1f}s"
        )

    return "\n".join(lines)


def format_decode_section(data: Dict) -> str:
    """Format decode rate benchmark results."""
    lines = []
    model = data.get("model", "Unknown")
    tag = data.get("tag", "")
    gpu = data.get("gpu_info", {}).get("name", "Unknown GPU")
    summary = data.get("summary", {})
    results = data.get("results", [])

    lines.append(f"  Model: {model}")
    if tag:
        lines.append(f"  Tag: {tag}")
    lines.append(f"  GPU: {gpu}")
    lines.append(f"  Median decode rate: {summary.get('median_decode_rate_toks', 0):.1f} t/s")

    prefill = summary.get("prefill_scaling", {})
    if prefill:
        lines.append(
            f"  Prefill scaling: {prefill.get('ttft_at_min_s', 0):.3f}s "
            f"@ {prefill.get('min_context', 0)} tokens → "
            f"{prefill.get('ttft_at_max_s', 0):.3f}s "
            f"@ {prefill.get('max_context', 0)} tokens"
        )

    stable = summary.get("decode_stable", False)
    lines.append(f"  Decode stable across context: {'Yes' if stable else 'No'}")

    # Per-context table
    lines.append(f"\n  {'Context':<9} {'TTFT':<10} {'Decode':<10} {'Rate'}")
    lines.append(f"  {'-'*45}")
    for r in results:
        lines.append(
            f"  {r['target_tokens']:<9} "
            f"{r['ttft_s']['median']:.3f}s    "
            f"{r['decode_sec']['median']:.3f}s    "
            f"{r['decode_rate_toks']['median']:.1f} t/s"
        )

    return "\n".join(lines)


def format_parallel_section(data: Dict) -> str:
    """Format parallel benchmark results."""
    lines = []
    model = data.get("model", "Unknown")
    seq = data.get("sequential", {})
    parallel = data.get("parallel", [])
    peak = data.get("peak", {})

    lines.append(f"  Model: {model}")
    lines.append(f"  Sequential: {seq.get('throughput_toks', 0):.1f} t/s "
                 f"(TTFT: {seq.get('avg_ttft_s', 0):.3f}s)")

    # Scaling table
    lines.append(f"\n  {'Conc':<6} {'Throughput':<12} {'Per-Req':<10} {'TTFT':<10} {'p95 Lat'}")
    lines.append(f"  {'-'*55}")
    for r in parallel:
        marker = " ← peak" if r["concurrency"] == peak.get("concurrency") else ""
        ttft_str = f"{r.get('avg_ttft_s', 0):.3f}s" if r.get('avg_ttft_s', 0) > 0 else "N/A"
        lines.append(
            f"  {r['concurrency']:<6} "
            f"{r['throughput_toks']:<12.1f} "
            f"{r['per_request_toks']:<10.1f} "
            f"{ttft_str:<10} "
            f"{r.get('p95_latency_s', 0):.1f}s{marker}"
        )

    lines.append(f"\n  Peak: {peak.get('throughput_toks', 0):.1f} t/s "
                 f"@ concurrency={peak.get('concurrency', '?')} "
                 f"({peak.get('speedup_vs_sequential', 0):.1f}x speedup)")

    return "\n".join(lines)


def discover_run_dir(run_dir: str, model_filter: str = None) -> tuple:
    """Auto-discover results within a run_all.py structured run directory.

    Expected structure:
        run_dir/
        ├── RUN_METADATA.json
        ├── quality/     → comparison_*.json
        ├── decode/      → decode_rate_*.json
        └── parallel/    → parallel_benchmark_*.json

    Returns (compare_result, decode_result, parallel_result).
    """
    compare_result = decode_result = parallel_result = None

    quality_dir = os.path.join(run_dir, "quality")
    if os.path.isdir(quality_dir):
        compare_result = find_latest_result(quality_dir, "comparison", model_filter)

    decode_dir = os.path.join(run_dir, "decode")
    if os.path.isdir(decode_dir):
        decode_result = find_latest_result(decode_dir, "decode_rate", model_filter)

    parallel_dir = os.path.join(run_dir, "parallel")
    if os.path.isdir(parallel_dir):
        parallel_result = find_latest_result(parallel_dir, "parallel_benchmark", model_filter)

    return compare_result, decode_result, parallel_result


def generate_report(compare_result, decode_result, parallel_result,
                    format_type="text", run_metadata=None):
    """Generate unified report from available results."""
    sections = []

    header = "LLM BENCHMARK REPORT"
    sep = "=" * 70

    if format_type == "markdown":
        sections.append(f"# {header}\n")
    else:
        sections.append(sep)
        sections.append(f"  {header}")
        sections.append(sep)

    # Run metadata (from run_all.py)
    if run_metadata:
        if format_type == "markdown":
            sections.append("\n## Run Info\n")
            sections.append(f"- **Run name**: {run_metadata.get('run_name', '?')}")
            sections.append(f"- **Model**: {run_metadata.get('model', '?')}")
            sections.append(f"- **Hardware**: {run_metadata.get('hardware', '?')}")
            sections.append(f"- **Backend**: {run_metadata.get('backend', '?')}")
            sections.append(f"- **Created**: {run_metadata.get('created_at', '?')}")
            sections.append(f"- **Bench version**: {run_metadata.get('bench_version', '?')}")
        else:
            sections.append(f"\n{'─'*70}")
            sections.append("RUN INFO")
            sections.append(f"{'─'*70}")
            sections.append(f"  Run name:  {run_metadata.get('run_name', '?')}")
            sections.append(f"  Model:     {run_metadata.get('model', '?')}")
            sections.append(f"  Hardware:  {run_metadata.get('hardware', '?')}")
            sections.append(f"  Backend:   {run_metadata.get('backend', '?')}")
            sections.append(f"  Created:   {run_metadata.get('created_at', '?')}")
            sections.append(f"  Version:   {run_metadata.get('bench_version', '?')}")

    # Quality section
    if compare_result:
        if format_type == "markdown":
            sections.append("\n## Quality & Correctness\n")
            sections.append(f"*Source: {compare_result['file']}*\n")
            sections.append("```")
        else:
            sections.append(f"\n{'─'*70}")
            sections.append("QUALITY & CORRECTNESS (compare_models.py)")
            sections.append(f"{'─'*70}")
        sections.append(format_quality_section(compare_result["data"]))
        if format_type == "markdown":
            sections.append("```")

    # Decode rate section
    if decode_result:
        if format_type == "markdown":
            sections.append("\n## Decode Rate (Prefill vs Decode Isolated)\n")
            sections.append(f"*Source: {decode_result['file']}*\n")
            sections.append("```")
        else:
            sections.append(f"\n{'─'*70}")
            sections.append("DECODE RATE — PREFILL vs DECODE ISOLATED (decode_rate_bench.py)")
            sections.append(f"{'─'*70}")
        sections.append(format_decode_section(decode_result["data"]))
        if format_type == "markdown":
            sections.append("```")

    # Parallel section
    if parallel_result:
        if format_type == "markdown":
            sections.append("\n## Concurrency Scaling\n")
            sections.append(f"*Source: {parallel_result['file']}*\n")
            sections.append("```")
        else:
            sections.append(f"\n{'─'*70}")
            sections.append("CONCURRENCY SCALING (parallel_benchmark.py)")
            sections.append(f"{'─'*70}")
        sections.append(format_parallel_section(parallel_result["data"]))
        if format_type == "markdown":
            sections.append("```")

    # Key metrics summary
    if format_type == "markdown":
        sections.append("\n## Key Metrics Summary\n")
        sections.append("| Metric | Value |")
        sections.append("|--------|-------|")
    else:
        sections.append(f"\n{'─'*70}")
        sections.append("KEY METRICS SUMMARY")
        sections.append(f"{'─'*70}")

    metrics = []
    if compare_result:
        data = compare_result["data"]
        if isinstance(data, list):
            data = data[0]
        seq = data.get("sequential", {})
        tests = seq.get("tests", [])
        total_score = sum(t.get("score", 0) for t in tests)
        total_max = sum(t.get("max_score", 0) for t in tests)
        pct = (total_score / total_max * 100) if total_max > 0 else 0
        metrics.append(("Quality Score", f"{total_score}/{total_max} ({pct:.0f}%)"))
        if seq.get("avg_ttft_seconds", 0) > 0:
            metrics.append(("Avg TTFT (quality tests)", f"{seq['avg_ttft_seconds']:.3f}s"))

    if decode_result:
        summary = decode_result["data"].get("summary", {})
        metrics.append(("Decode Rate (isolated)", f"{summary.get('median_decode_rate_toks', 0):.1f} t/s"))
        prefill = summary.get("prefill_scaling", {})
        if prefill:
            metrics.append(("TTFT @ min context", f"{prefill.get('ttft_at_min_s', 0):.3f}s"))
            metrics.append(("TTFT @ max context", f"{prefill.get('ttft_at_max_s', 0):.3f}s"))

    if parallel_result:
        peak = parallel_result["data"].get("peak", {})
        seq = parallel_result["data"].get("sequential", {})
        metrics.append(("Sequential Throughput", f"{seq.get('throughput_toks', 0):.1f} t/s"))
        metrics.append(("Peak Concurrent", f"{peak.get('throughput_toks', 0):.1f} t/s @ C={peak.get('concurrency', '?')}"))
        metrics.append(("Speedup", f"{peak.get('speedup_vs_sequential', 0):.1f}x"))

    for label, value in metrics:
        if format_type == "markdown":
            sections.append(f"| {label} | {value} |")
        else:
            sections.append(f"  {label:<35} {value}")

    if format_type != "markdown":
        sections.append(f"\n{sep}")

    # Historical clarity note
    note = ("Note: throughput numbers from bench versions before v2.0 counted "
            "prompt+completion tokens and should not be compared to v2+ "
            "completion-only decode rates. See REFACTORING.md for details.")
    if format_type == "markdown":
        sections.append(f"\n---\n\n*{note}*")
    else:
        sections.append(f"\n  {note}")

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark report — combines quality, decode, and concurrency results."
    )
    parser.add_argument("--model", type=str, help="Model name filter (substring match)")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="run_all.py run directory (auto-discovers quality/decode/parallel subdirs)")
    parser.add_argument("--compare-dir", type=str, default="results",
                        help="Directory to search for comparison results")
    parser.add_argument("--decode-dir", type=str, default="results/decode",
                        help="Directory to search for decode rate results")
    parser.add_argument("--parallel-dir", type=str, default="results/parallel",
                        help="Directory to search for parallel benchmark results")
    parser.add_argument("--format", choices=["text", "markdown"], default="text",
                        help="Output format")
    args = parser.parse_args()

    run_metadata = None

    if args.run_dir:
        # Structured run directory mode
        if not os.path.isdir(args.run_dir):
            print(f"Run directory not found: {args.run_dir}", file=sys.stderr)
            sys.exit(1)

        # Load metadata if available
        meta_path = os.path.join(args.run_dir, "RUN_METADATA.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                run_metadata = json.load(f)

        compare_result, decode_result, parallel_result = discover_run_dir(
            args.run_dir, args.model
        )

        if not any([compare_result, decode_result, parallel_result]):
            print(f"No results found in run directory: {args.run_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        # Legacy: search across common result locations
        compare_dirs = [args.compare_dir, "results/8xA4000", "results/compare"]
        decode_dirs = [args.decode_dir]
        parallel_dirs = [args.parallel_dir, "results/8xA4000"]

        compare_result = None
        for d in compare_dirs:
            if os.path.isdir(d):
                result = find_latest_result(d, "comparison", args.model)
                if result:
                    compare_result = result
                    break

        decode_result = None
        for d in decode_dirs:
            if os.path.isdir(d):
                result = find_latest_result(d, "decode_rate", args.model)
                if result:
                    decode_result = result
                    break

        parallel_result = None
        for d in parallel_dirs:
            if os.path.isdir(d):
                result = find_latest_result(d, "parallel_benchmark", args.model)
                if result:
                    parallel_result = result
                    break

        if not any([compare_result, decode_result, parallel_result]):
            print("No results found. Run benchmarks first or check --*-dir paths.", file=sys.stderr)
            print(f"\nSearched:", file=sys.stderr)
            print(f"  compare: {compare_dirs}", file=sys.stderr)
            print(f"  decode:  {decode_dirs}", file=sys.stderr)
            print(f"  parallel: {parallel_dirs}", file=sys.stderr)
            sys.exit(1)

    report = generate_report(compare_result, decode_result, parallel_result,
                             args.format, run_metadata=run_metadata)
    print(report)


if __name__ == "__main__":
    main()
