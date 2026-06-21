#!/usr/bin/env python3
"""
Parallel Tool-Call Benchmark v1.0
=================================
Tests LLM performance under both sequential and parallel workloads,
simulating real agentic tool-calling patterns.

Two test legs:
1. Sequential: One request at a time (latency-focused)
2. Parallel: Concurrent requests (throughput-focused)

Auto-detects optimal concurrency for the GPU being tested.

Usage:
    python parallel_benchmark.py --model seed-oss --api-url http://localhost:8100/v1
    python parallel_benchmark.py --model seed-oss --api-url http://localhost:8100/v1 --max-concurrent 32
"""

import argparse
import asyncio
import aiohttp
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import statistics

VERSION = "2.0"

# ============================================================================
# COLORS
# ============================================================================

class Colors:
    HEADER = '\033[1;93m'
    PASS = '\033[92m'
    FAIL = '\033[91m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    MAGENTA = '\033[95m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

C = Colors()

# ============================================================================
# TEST PROMPTS - Varying complexity for realistic tool-call simulation
# ============================================================================

TOOL_CALL_PROMPTS = {
    "simple_lookup": {
        "name": "Simple Lookup",
        "prompt": "What is the capital of France? Answer in one word.",
        "max_tokens": 32,
        "expected_pattern": "paris",
    },
    "code_review_short": {
        "name": "Code Review (Short)",
        "prompt": """Review this Python function for bugs:

```python
def divide(a, b):
    return a / b
```

List any issues in 2-3 sentences.""",
        "max_tokens": 128,
        "expected_pattern": "zero|division|exception",
    },
    "analysis_medium": {
        "name": "Analysis (Medium)",
        "prompt": """Analyze this code for security issues:

```python
import sqlite3

def get_user(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE name = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()
```

Explain the vulnerability and how to fix it.""",
        "max_tokens": 512,
        "expected_pattern": "sql injection|parameterized|sanitize",
    },
    "complex_debug": {
        "name": "Complex Debug",
        "prompt": """Debug this race condition:

```python
import threading

class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        current = self.count
        time.sleep(0.001)  # Simulate work
        self.count = current + 1

counter = Counter()
threads = [threading.Thread(target=counter.increment) for _ in range(100)]
for t in threads: t.start()
for t in threads: t.join()
print(counter.count)  # Expected: 100, Actual: varies
```

Explain the bug and provide a fix.""",
        "max_tokens": 768,
        "expected_pattern": "lock|mutex|atomic|thread.*safe",
    },
}

# Heavier prompts for stress testing (from practical tests)
HEAVY_PROMPTS = {
    "nightmare_byte_order": {
        "name": "Nightmare: Byte Order",
        "prompt": """In Bitcoin mining, the block header contains a prev_hash field.
A common bug occurs when developers do:

```python
prev_hash = bytes.fromhex(template['previousblockhash'])
```

What bug might this cause in the mining process and how would you fix it?""",
        "max_tokens": 512,
        "expected_pattern": "reverse|endian|byte.?order|\\[::-1\\]",
    },
    "file_creation": {
        "name": "File Creation Task",
        "prompt": """Create a bash script that:
1. Takes a directory path as argument
2. Finds all .py files recursively
3. Counts total lines of code (excluding blank lines and comments)
4. Outputs the result as JSON: {"files": N, "lines": M}

Provide the complete script.""",
        "max_tokens": 512,
        "expected_pattern": "find|wc|grep|json",
    },
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RequestResult:
    prompt_name: str
    success: bool
    tokens: int  # completion tokens only (output)
    latency: float  # seconds
    prompt_tokens: int = 0
    total_tokens: int = 0  # prompt + completion (for reference)
    ttft: float = 0.0  # time to first token (prefill latency)
    error: Optional[str] = None

@dataclass
class ConcurrencyResult:
    concurrency: int
    total_requests: int
    successful: int
    failed: int
    total_tokens: int  # completion tokens (output only)
    total_time: float
    aggregate_throughput: float  # tok/s (completion tokens / wall time)
    avg_latency: float
    p50_latency: float
    p95_latency: float
    per_request_throughput: float  # tok/s per request
    avg_ttft: float = 0.0  # average time to first token across requests
    p50_ttft: float = 0.0

@dataclass
class BenchmarkResult:
    model: str
    gpu_info: str
    timestamp: str
    sequential_results: List[ConcurrencyResult]
    parallel_results: List[ConcurrencyResult]
    optimal_concurrency: int
    peak_throughput: float
    summary: Dict[str, Any]

# ============================================================================
# ASYNC HTTP CLIENT
# ============================================================================

async def call_model_async(
    session: aiohttp.ClientSession,
    api_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    timeout: int = 300
) -> RequestResult:
    """Make async streaming API call to model.

    Uses SSE streaming to measure TTFT (time to first token) and properly
    separates prompt_tokens from completion_tokens. Throughput is based on
    completion tokens only (output generation), not inflated by prompt size.
    """
    start_time = time.time()
    ttft = 0.0

    try:
        async with session.post(
            f"{api_url}/chat/completions",
            json={
                'model': model_id,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': max_tokens,
                'temperature': 0.0,
                'stream': True,
                'stream_options': {'include_usage': True},
            },
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers={"Accept": "text/event-stream"},
        ) as response:
            if response.status != 200:
                body = await response.text()
                return RequestResult(
                    prompt_name="",
                    success=False,
                    tokens=0,
                    latency=time.time() - start_time,
                    error=f"HTTP {response.status}: {body[:200]}"
                )

            pt = ct = 0
            first_token_seen = False

            async for raw_line in response.content:
                line = raw_line.decode('utf-8', errors='replace').strip()
                if not line or not line.startswith("data: "):
                    continue
                payload_str = line[6:]
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
                if choices and not first_token_seen:
                    delta = choices[0].get("delta") or {}
                    emitted = (delta.get("content")
                               or delta.get("reasoning")
                               or delta.get("reasoning_content"))
                    if emitted:
                        ttft = time.time() - start_time
                        first_token_seen = True

            latency = time.time() - start_time
            if not first_token_seen:
                ttft = latency

            return RequestResult(
                prompt_name="",
                success=True,
                tokens=ct,  # completion tokens only
                latency=latency,
                prompt_tokens=pt,
                total_tokens=pt + ct,
                ttft=ttft,
            )

    except asyncio.TimeoutError:
        return RequestResult(
            prompt_name="",
            success=False,
            tokens=0,
            latency=time.time() - start_time,
            error="Timeout"
        )
    except Exception as e:
        return RequestResult(
            prompt_name="",
            success=False,
            tokens=0,
            latency=time.time() - start_time,
            error=str(e)
        )

async def run_concurrent_requests(
    api_url: str,
    model_id: str,
    prompts: List[Dict],
    concurrency: int,
    timeout: int = 300
) -> ConcurrencyResult:
    """Run multiple requests concurrently and measure performance."""

    # Create request list - repeat prompts to reach concurrency level
    requests = []
    for i in range(concurrency):
        prompt_info = prompts[i % len(prompts)]
        requests.append({
            'prompt': prompt_info['prompt'],
            'max_tokens': prompt_info['max_tokens'],
            'name': prompt_info['name']
        })

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        start_time = time.time()

        # Launch all requests concurrently
        tasks = [
            call_model_async(
                session, api_url, model_id,
                req['prompt'], req['max_tokens'], timeout
            )
            for req in requests
        ]

        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

    # Analyze results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    # Use completion tokens only for throughput (not prompt tokens)
    total_tokens = sum(r.tokens for r in successful)
    latencies = [r.latency for r in successful] if successful else [0]
    ttfts = [r.ttft for r in successful if r.ttft > 0]

    aggregate_throughput = total_tokens / total_time if total_time > 0 else 0
    avg_latency = statistics.mean(latencies) if latencies else 0
    p50_latency = statistics.median(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else avg_latency
    avg_ttft = statistics.mean(ttfts) if ttfts else 0
    p50_ttft = statistics.median(ttfts) if ttfts else 0

    per_request_throughput = aggregate_throughput / concurrency if concurrency > 0 else 0

    return ConcurrencyResult(
        concurrency=concurrency,
        total_requests=len(results),
        successful=len(successful),
        failed=len(failed),
        total_tokens=total_tokens,
        total_time=total_time,
        aggregate_throughput=aggregate_throughput,
        avg_latency=avg_latency,
        p50_latency=p50_latency,
        p95_latency=p95_latency,
        per_request_throughput=per_request_throughput,
        avg_ttft=avg_ttft,
        p50_ttft=p50_ttft,
    )

# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def detect_optimal_concurrency(
    api_url: str,
    model_id: str,
    max_concurrent: int = 64
) -> int:
    """Auto-detect optimal concurrency by testing increasing batch sizes."""

    print(f"\n{C.CYAN}Auto-detecting optimal concurrency...{C.RESET}")

    prompts = list(TOOL_CALL_PROMPTS.values())[:2]  # Use simple prompts for detection
    test_levels = [1, 2, 4, 8, 16, 32, 64]
    test_levels = [c for c in test_levels if c <= max_concurrent]

    best_throughput = 0
    optimal = 1
    results = []

    for conc in test_levels:
        print(f"  Testing concurrency={conc}...", end=" ", flush=True)

        result = asyncio.run(run_concurrent_requests(
            api_url, model_id, prompts, conc, timeout=120
        ))

        results.append(result)
        print(f"{result.aggregate_throughput:.1f} tok/s")

        if result.aggregate_throughput > best_throughput:
            best_throughput = result.aggregate_throughput
            optimal = conc

        # Stop if throughput starts declining significantly
        if len(results) > 2:
            if result.aggregate_throughput < results[-2].aggregate_throughput * 0.9:
                break

        # Stop if we hit errors
        if result.failed > result.successful:
            print(f"  {C.YELLOW}Too many failures at concurrency={conc}, stopping{C.RESET}")
            break

    print(f"\n{C.PASS}Optimal concurrency: {optimal} ({best_throughput:.1f} tok/s){C.RESET}")
    return optimal

def run_sequential_benchmark(
    api_url: str,
    model_id: str,
    prompts: List[Dict],
    timeout: int = 300
) -> ConcurrencyResult:
    """Run prompts one at a time (sequential/latency mode)."""

    results = []
    total_start = time.time()

    for prompt_info in prompts:
        result = asyncio.run(run_concurrent_requests(
            api_url, model_id, [prompt_info], 1, timeout
        ))
        results.append(result)

    total_time = time.time() - total_start
    total_tokens = sum(r.total_tokens for r in results)
    latencies = [r.avg_latency for r in results]

    return ConcurrencyResult(
        concurrency=1,
        total_requests=len(prompts),
        successful=sum(r.successful for r in results),
        failed=sum(r.failed for r in results),
        total_tokens=total_tokens,
        total_time=total_time,
        aggregate_throughput=total_tokens / total_time if total_time > 0 else 0,
        avg_latency=statistics.mean(latencies) if latencies else 0,
        p50_latency=statistics.median(latencies) if latencies else 0,
        p95_latency=sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else 0,
        per_request_throughput=total_tokens / total_time if total_time > 0 else 0
    )

def run_parallel_benchmark(
    api_url: str,
    model_id: str,
    prompts: List[Dict],
    concurrency_levels: List[int],
    timeout: int = 300
) -> List[ConcurrencyResult]:
    """Run parallel benchmark at various concurrency levels."""

    results = []

    for conc in concurrency_levels:
        print(f"  Concurrency {conc}...", end=" ", flush=True)

        result = asyncio.run(run_concurrent_requests(
            api_url, model_id, prompts, conc, timeout
        ))

        results.append(result)

        status = C.PASS if result.failed == 0 else C.YELLOW
        print(f"{status}{result.aggregate_throughput:.1f} tok/s "
              f"({result.successful}/{result.total_requests} ok, "
              f"avg {result.avg_latency:.1f}s){C.RESET}")

    return results

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Parallel Tool-Call Benchmark")
    parser.add_argument('--model', required=True, help='Model ID')
    parser.add_argument('--api-url', required=True, help='API base URL')
    parser.add_argument('--max-concurrent', type=int, default=64, help='Max concurrency to test')
    parser.add_argument('--timeout', type=int, default=300, help='Request timeout (seconds)')
    parser.add_argument('--output', type=str, help='Output directory for results')
    parser.add_argument('--skip-detection', action='store_true', help='Skip auto-detection')
    parser.add_argument('--concurrency-levels', type=str, default='1,2,4,8,16,32',
                        help='Comma-separated concurrency levels to test')
    args = parser.parse_args()

    print(f"{C.BOLD}{C.HEADER}Parallel Tool-Call Benchmark v{VERSION}{C.RESET}")
    print(f"Model: {args.model}")
    print(f"API: {args.api_url}")
    print(f"Max Concurrent: {args.max_concurrent}")

    # Parse concurrency levels
    concurrency_levels = [int(x) for x in args.concurrency_levels.split(',')]
    concurrency_levels = [c for c in concurrency_levels if c <= args.max_concurrent]

    # Prepare prompts
    all_prompts = list(TOOL_CALL_PROMPTS.values()) + list(HEAVY_PROMPTS.values())

    # ========================================================================
    # LEG 1: SEQUENTIAL (Single Request at a Time)
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"{C.BOLD}LEG 1: SEQUENTIAL (Latency Mode){C.RESET}")
    print(f"{'='*70}")
    print("Running all prompts one at a time...")

    seq_result = run_sequential_benchmark(
        args.api_url, args.model, all_prompts, args.timeout
    )

    print(f"\n{C.CYAN}Sequential Results:{C.RESET}")
    print(f"  Total requests: {seq_result.total_requests}")
    print(f"  Successful: {seq_result.successful}")
    print(f"  Completion tokens: {seq_result.total_tokens}")
    print(f"  Total time: {seq_result.total_time:.1f}s")
    print(f"  {C.BOLD}Throughput: {seq_result.aggregate_throughput:.1f} tok/s (completion only){C.RESET}")
    print(f"  Avg latency: {seq_result.avg_latency:.1f}s")
    if seq_result.avg_ttft > 0:
        print(f"  Avg TTFT: {seq_result.avg_ttft:.2f}s")

    # ========================================================================
    # LEG 2: PARALLEL (Throughput Mode)
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"{C.BOLD}LEG 2: PARALLEL (Throughput Mode){C.RESET}")
    print(f"{'='*70}")

    # Auto-detect optimal concurrency if not skipped
    if not args.skip_detection:
        optimal_conc = detect_optimal_concurrency(
            args.api_url, args.model, args.max_concurrent
        )
        # Add optimal to test levels if not present
        if optimal_conc not in concurrency_levels:
            concurrency_levels.append(optimal_conc)
            concurrency_levels.sort()
    else:
        optimal_conc = max(concurrency_levels)

    print(f"\nTesting concurrency levels: {concurrency_levels}")

    parallel_results = run_parallel_benchmark(
        args.api_url, args.model, all_prompts, concurrency_levels, args.timeout
    )

    # Find peak throughput
    peak_result = max(parallel_results, key=lambda r: r.aggregate_throughput)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"{C.BOLD}{C.HEADER}BENCHMARK SUMMARY{C.RESET}")
    print(f"{'='*70}")

    print(f"\n{C.CYAN}Sequential (Batch=1):{C.RESET}")
    print(f"  Throughput: {seq_result.aggregate_throughput:.1f} tok/s")
    print(f"  Avg Latency: {seq_result.avg_latency:.1f}s")

    print(f"\n{C.CYAN}Parallel Scaling:{C.RESET}")
    print(f"  {'Conc':<6} {'Throughput':<12} {'Per-Req':<10} {'TTFT':<8} {'Latency':<10} {'Status'}")
    print(f"  {'-'*60}")

    for r in parallel_results:
        status = f"{C.PASS}OK{C.RESET}" if r.failed == 0 else f"{C.YELLOW}{r.failed} failed{C.RESET}"
        marker = " ← peak" if r.concurrency == peak_result.concurrency else ""
        ttft_str = f"{r.avg_ttft:.2f}s" if r.avg_ttft > 0 else "N/A"
        print(f"  {r.concurrency:<6} {r.aggregate_throughput:<12.1f} "
              f"{r.per_request_throughput:<10.1f} {ttft_str:<8} {r.avg_latency:<10.1f} {status}{marker}")

    print(f"\n{C.BOLD}Peak Performance:{C.RESET}")
    print(f"  Concurrency: {peak_result.concurrency}")
    print(f"  Throughput: {C.PASS}{peak_result.aggregate_throughput:.1f} tok/s{C.RESET}")
    print(f"  Per-request: {peak_result.per_request_throughput:.1f} tok/s")
    print(f"  Speedup vs sequential: {peak_result.aggregate_throughput / seq_result.aggregate_throughput:.1f}x")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "version": VERSION,
        "benchmark": "parallel",
        "model": args.model,
        "api_url": args.api_url,
        "timestamp": timestamp,
        "notes": {
            "throughput_metric": "completion_tokens_only",
            "description": "Throughput based on output (completion) tokens / wall time. "
                           "Prompt tokens excluded. TTFT measured via streaming.",
        },
        "sequential": {
            "throughput_toks": round(seq_result.aggregate_throughput, 2),
            "avg_latency_s": round(seq_result.avg_latency, 3),
            "avg_ttft_s": round(seq_result.avg_ttft, 4),
            "completion_tokens": seq_result.total_tokens,
            "total_time_s": round(seq_result.total_time, 3),
        },
        "parallel": [
            {
                "concurrency": r.concurrency,
                "throughput_toks": round(r.aggregate_throughput, 2),
                "per_request_toks": round(r.per_request_throughput, 2),
                "avg_latency_s": round(r.avg_latency, 3),
                "p50_latency_s": round(r.p50_latency, 3),
                "p95_latency_s": round(r.p95_latency, 3),
                "avg_ttft_s": round(r.avg_ttft, 4),
                "p50_ttft_s": round(r.p50_ttft, 4),
                "successful": r.successful,
                "failed": r.failed,
            }
            for r in parallel_results
        ],
        "peak": {
            "concurrency": peak_result.concurrency,
            "throughput_toks": round(peak_result.aggregate_throughput, 2),
            "speedup_vs_sequential": round(
                peak_result.aggregate_throughput / seq_result.aggregate_throughput, 2
            ) if seq_result.aggregate_throughput > 0 else 0,
        }
    }

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"parallel_benchmark_{timestamp}.json"
    else:
        output_file = Path(f"parallel_benchmark_{timestamp}.json")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{C.PASS}Results saved to: {output_file}{C.RESET}")
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
