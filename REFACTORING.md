# Refactoring Notes — llm-bench v3.0

## What Changed

### `run_all.py` (new)
Canonical orchestrator that invokes all standalone benchmark tools in sequence,
writing results to a versioned timestamped directory with `RUN_METADATA.json`.

### `compare_models.py` (stripped)
Now **quality-only**: runs the 22-check practical rubric, records per-test TTFT
and sequential throughput as metadata. Removed:
- Embedded parallel benchmark (ParallelResult, call_model_async, run_parallel_requests, run_parallel_benchmark)
- Embedded long-context test (LONG_CONTEXT_TEST dict, run_long_context_test)
- `--parallel`, `--parallel-only`, `--long-context`, `--max-concurrent` CLI args
- `parallel_results` field from ModelResults dataclass
- `"parallel"` key from JSON output
- `asyncio` / `aiohttp` imports

Those capabilities now live in their standalone tools:
- **parallel_benchmark.py** — concurrent throughput with auto-detection
- **long_context_test.py** — needle-in-haystack, multi-file understanding, cross-reference
- **decode_rate_bench.py** — isolated decode rate at various context lengths

### `report.py` (updated)
Added `--run-dir` flag to auto-discover results within a structured run directory
(as created by `run_all.py`). Falls back to existing `--compare-dir`/`--decode-dir`/
`--parallel-dir` behavior.

---

## Why

`compare_models.py` grew into a Swiss army knife that ran quality tests, parallel
throughput benchmarks, and long-context tests all in one process. The standalone
tools (`parallel_benchmark.py`, `decode_rate_bench.py`, `long_context_test.py`)
superseded those embedded implementations with better metrics:

| Feature | Old (embedded) | New (standalone) |
|---------|----------------|------------------|
| TTFT measurement | Not in parallel | Per-request TTFT |
| Throughput metric | Mixed prompt+completion tokens | Completion-only |
| Decode isolation | Not available | decode_rate_bench.py |
| Concurrency auto-detect | Basic | Full with early-stop |

Keeping dead code in `compare_models.py` created confusion about which numbers
to trust and which tool to use.

---

## Historical Clarity — Why Old Numbers Are Murky

### v1.0 era (Jan 2026)
- **Throughput** = `total_tokens / time` (prompt + completion tokens counted)
- **No TTFT** measurement
- **No prefill/decode isolation**
- Numbers looked higher because prompt tokens inflated the count

### v2.0 era (Feb 2026)
- Throughput switched to **completion tokens only**
- TTFT measured via streaming in all tools
- `decode_rate_bench.py` introduced to isolate decode from prefill
- `parallel_benchmark.py` introduced with proper per-request TTFT

### v3.0 (current)
- `run_all.py` orchestrates all tools with versioned run directories
- `compare_models.py` stripped to quality-only
- `RUN_METADATA.json` records exact tool versions and parameters

**Do not directly compare old total-token throughput numbers to v2+ decode-rate
numbers.** They measure fundamentally different things. A model that showed
"150 t/s" in v1.0 might show "45 t/s decode rate" in v2.0+ — that's not a
regression, it's an honest measurement.

---

## Version Timeline

| Version | Date | Entry point | Throughput metric | TTFT |
|---------|------|-------------|-------------------|------|
| 1.0 | Jan 2026 | compare_models.py | total tokens / time | No |
| 2.0 | Feb 2026 | compare_models.py + standalone tools | completion tokens / time | Yes |
| 3.0 | Jun 2026 | **run_all.py** (orchestrator) | completion tokens / time | Yes |

---

## Migration Guide

### Old: `compare_models.py --parallel`
```bash
# Before (v2.x):
python compare_models.py --model m --api-url http://... --parallel --max-concurrent 64

# After (v3.0) — standalone:
python parallel_benchmark.py --model m --api-url http://... --max-concurrent 64

# After (v3.0) — orchestrated:
python run_all.py --run-name test --model m --api-url http://... --skip-quality --skip-decode --skip-context
```

### Old: `compare_models.py --long-context`
```bash
# Before (v2.x):
python compare_models.py --model m --api-url http://... --long-context

# After (v3.0) — standalone:
python long_context_test.py --model m --api-url http://... --context-size 32000

# After (v3.0) — orchestrated:
python run_all.py --run-name test --model m --api-url http://... --skip-quality --skip-decode --skip-parallel
```

### Old: `compare_models.py --parallel-only`
```bash
# Before (v2.x):
python compare_models.py --model m --api-url http://... --parallel-only

# After (v3.0):
python parallel_benchmark.py --model m --api-url http://... --max-concurrent 64
```

### Full orchestrated run (replaces ad-hoc shell scripts)
```bash
python run_all.py \
  --run-name b70-qwen3-32b-vllm-xpu \
  --model Qwen/Qwen3-32B \
  --api-url http://127.0.0.1:8000/v1 \
  --hardware "Arc Pro B70" --backend "vLLM XPU" \
  --report
```
