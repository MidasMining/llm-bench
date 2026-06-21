#!/usr/bin/env python3
"""Single-stream decode-rate benchmark with streaming SSE parsing.

Isolates decode rate from prefill cost by parsing chat-completion SSE events:
  TTFT = time to first content delta (prefill latency)
  decode_rate = (completion_tokens - 1) / (last_event_time - first_token_time)

Sweeps a range of input context sizes by repeating a deterministic filler so
prefill scales while decode workload is fixed (max_tokens=500). Useful for
comparing attention-kernel scaling vs context length without conflating prefill.

Outputs structured JSON results alongside the human-readable table.

Usage:
  python decode_rate_bench.py \\
      --api-url http://localhost:8100/v1 --model Qwen3.6-V100 \\
      --tag "FLASH_ATTN_V100 + graphs" \\
      --targets 500 2000 4000 6000

  # Save to results directory:
  python decode_rate_bench.py --model seed-oss --output results/decode/

Defaults match a typical V100 vLLM serve.
"""
import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import requests

VERSION = "2.0"

SENTENCE = "The quick brown fox jumps over the lazy dog. "  # ~10 tokens

PROMPT_TEMPLATE = (
    "Read the following text and produce a detailed analytical commentary "
    "in at least 500 words about its repetitive structure, linguistic "
    "features, and potential uses in NLP benchmarking.\n\nTEXT:\n{body}"
    "\n\nAnalysis:"
)


def make_prompt(target_input_tokens: int) -> str:
    body = SENTENCE * (target_input_tokens // 10)
    return PROMPT_TEMPLATE.format(body=body)


def stream_call(api_url: str, model: str, prompt: str, max_tokens: int, timeout: int = 900):
    """Returns dict with prompt_tokens, completion_tokens, ttft, decode_sec, total_sec."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Disable reasoning so we can measure raw decode rate against a fixed
        # output budget; reasoning models otherwise emit a variable number of
        # think-tokens before answering.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    ttft = None
    last_event_t = None
    pt = ct = 0

    with requests.post(
        f"{api_url}/chat/completions",
        json=payload,
        stream=True,
        timeout=timeout,
        headers={"Accept": "text/event-stream"},
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if not raw.startswith("data: "):
                continue
            payload_str = raw[6:]
            if payload_str == "[DONE]":
                break
            try:
                ev = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                pt = ev["usage"].get("prompt_tokens", pt)
                ct = ev["usage"].get("completion_tokens", ct)
            choices = ev.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                # Reasoning models emit `reasoning` deltas as well as `content`.
                # Count any non-empty decode emission as a "decode event".
                emitted = (delta.get("content")
                           or delta.get("reasoning")
                           or delta.get("reasoning_content"))
                if emitted and ttft is None:
                    ttft = time.time() - t0
                if emitted is not None:
                    last_event_t = time.time()

    total = time.time() - t0
    if ttft is None:
        ttft = total
    if last_event_t is None:
        decode_sec = 0.0
    else:
        decode_sec = max(last_event_t - (t0 + ttft), 1e-3)
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "ttft": ttft,
        "decode_sec": decode_sec,
        "total_sec": total,
    }


def detect_gpu() -> Dict[str, Any]:
    """Detect GPU information using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'name': parts[0] if len(parts) > 0 else 'Unknown',
                'memory_mb': int(parts[1]) if len(parts) > 1 else 0,
                'driver': parts[2] if len(parts) > 2 else 'Unknown',
                'detected': True
            }
    except Exception:
        pass
    return {'name': 'Unknown', 'memory_mb': 0, 'driver': 'Unknown', 'detected': False}


def main():
    ap = argparse.ArgumentParser(
        description="Single-stream decode-rate benchmark isolating prefill from decode."
    )
    ap.add_argument("--api-url", default="http://localhost:8100/v1")
    ap.add_argument("--model", default="Qwen3.6-V100")
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--targets", type=int, nargs="+", default=[500, 2000, 4000, 6000])
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--output", type=str, default=None,
                    help="Directory to save JSON results (default: results/decode/)")
    ap.add_argument("--no-json", action="store_true",
                    help="Suppress JSON output (stdout table only)")
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_info = detect_gpu()

    print(f"=== {args.tag} ===")
    print(f"api={args.api_url} model={args.model}")
    if gpu_info['detected']:
        print(f"gpu={gpu_info['name']} ({gpu_info['memory_mb']}MB)")
    print()

    # warmup
    w = stream_call(args.api_url, args.model, "Hello, brief response please.", 20)
    print(f"warmup: pt={w['prompt_tokens']} ct={w['completion_tokens']} "
          f"ttft={w['ttft']:.2f}s decode={w['decode_sec']:.2f}s total={w['total_sec']:.2f}s\n")

    print(f"{'target':>7} {'prompt':>7} {'comp':>5} "
          f"{'ttft':>7} {'decode':>9} {'rate':>10}")

    # Collect structured results per target context size
    target_results = []

    for target in args.targets:
        prompt = make_prompt(target)
        runs_data = []
        for i in range(args.runs):
            r = stream_call(args.api_url, args.model, prompt, args.max_tokens)
            rate = ((r["completion_tokens"] - 1) / r["decode_sec"]
                    if r["decode_sec"] > 0 and r["completion_tokens"] > 1 else 0.0)
            runs_data.append({
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "ttft_s": round(r["ttft"], 4),
                "decode_sec": round(r["decode_sec"], 4),
                "total_sec": round(r["total_sec"], 4),
                "decode_rate_toks": round(rate, 2),
            })

        rates = [rd["decode_rate_toks"] for rd in runs_data]
        ttfts = [rd["ttft_s"] for rd in runs_data]
        decode_secs = [rd["decode_sec"] for rd in runs_data]

        summary = {
            "target_tokens": target,
            "actual_prompt_tokens": runs_data[-1]["prompt_tokens"],
            "completion_tokens": runs_data[-1]["completion_tokens"],
            "ttft_s": {
                "mean": round(statistics.mean(ttfts), 4),
                "median": round(statistics.median(ttfts), 4),
                "min": round(min(ttfts), 4),
                "max": round(max(ttfts), 4),
            },
            "decode_sec": {
                "mean": round(statistics.mean(decode_secs), 4),
                "median": round(statistics.median(decode_secs), 4),
                "min": round(min(decode_secs), 4),
                "max": round(max(decode_secs), 4),
            },
            "decode_rate_toks": {
                "mean": round(statistics.mean(rates), 2),
                "median": round(statistics.median(rates), 2),
                "min": round(min(rates), 2),
                "max": round(max(rates), 2),
            },
            "runs": runs_data,
        }
        target_results.append(summary)

        # Print human-readable row (median values)
        print(f"{target:>7} {summary['actual_prompt_tokens']:>7} "
              f"{summary['completion_tokens']:>5} "
              f"{summary['ttft_s']['median']:>5.2f}s "
              f"{summary['decode_sec']['median']:>7.2f}s "
              f"{summary['decode_rate_toks']['median']:>8.1f} t/s")

    # Compute overall summary across all context sizes
    all_rates = [t["decode_rate_toks"]["median"] for t in target_results]
    overall_decode_rate = round(statistics.mean(all_rates), 2) if all_rates else 0.0

    print(f"\n{'--- summary ---':>43}")
    print(f"{'median decode rate across contexts:':>43} {overall_decode_rate:.1f} t/s")
    print(f"{'prefill scaling (TTFT min→max):':>43} "
          f"{target_results[0]['ttft_s']['median']:.3f}s → "
          f"{target_results[-1]['ttft_s']['median']:.3f}s")

    # Build JSON output
    if not args.no_json:
        output_data = {
            "version": VERSION,
            "benchmark": "decode_rate",
            "timestamp": timestamp,
            "tag": args.tag,
            "model": args.model,
            "api_url": args.api_url,
            "gpu_info": gpu_info,
            "config": {
                "max_tokens": args.max_tokens,
                "runs_per_target": args.runs,
                "targets": args.targets,
                "reasoning_disabled": True,
            },
            "warmup": {
                "prompt_tokens": w["prompt_tokens"],
                "completion_tokens": w["completion_tokens"],
                "ttft_s": round(w["ttft"], 4),
                "decode_sec": round(w["decode_sec"], 4),
            },
            "results": target_results,
            "summary": {
                "median_decode_rate_toks": overall_decode_rate,
                "prefill_scaling": {
                    "min_context": target_results[0]["target_tokens"],
                    "max_context": target_results[-1]["target_tokens"],
                    "ttft_at_min_s": target_results[0]["ttft_s"]["median"],
                    "ttft_at_max_s": target_results[-1]["ttft_s"]["median"],
                },
                "decode_stable": (
                    max(all_rates) - min(all_rates) < 0.1 * statistics.mean(all_rates)
                    if all_rates and statistics.mean(all_rates) > 0 else False
                ),
            },
        }

        # Determine output path
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = Path("results/decode")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Filename: model slug + timestamp
        model_slug = args.model.split("/")[-1].replace(" ", "_")
        output_file = output_dir / f"decode_rate_{model_slug}_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
