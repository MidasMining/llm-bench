# LLM BENCHMARK REPORT


## Run Info

- **Run name**: b70-moe-xpugraph
- **Model**: Qwen3.6-35B-A3B
- **Hardware**: 1x Arc Pro B70 32GB (BMG-G31, Xe2)
- **Backend**: vLLM XPU Graph FULL (native source, patched MoE+CCL)
- **Created**: 2026-06-22T00:06:23
- **Bench version**: 3.0

## Quality & Correctness

*Source: results/b70-moe-xpugraph-v3.0-20260621-235506/quality/comparison_20260622_000225.json*

```
  Model: Qwen3.6-35B-A3B
  Quality: 20/22 (90.9%)
  Throughput (sequential): 82.0 t/s
  Avg TTFT (prefill): 0.189s
  Test                           Score      TTFT     Time
  ------------------------------------------------------------
  ZMQ Listener Bug               2/2 PASS  0.18s    86.8s
  PPLNS Mining Pool Bugs         3/3 PASS  0.19s    86.9s
  Payment System Bugs            2/4 FAIL  0.16s    86.9s
  Stratum Protocol Bugs          5/5 PASS  0.24s    86.9s
  HiveOS Wrapper Creation        8/8 PASS  0.18s    86.7s
```

## Decode Rate (Prefill vs Decode Isolated)

*Source: results/b70-moe-xpugraph-v3.0-20260621-235506/decode/decode_rate_Qwen3.6-35B-A3B_20260622_000226.json*

```
  Model: Qwen3.6-35B-A3B
  Tag: bench
  GPU: Unknown
  Median decode rate: 70.8 t/s
  Prefill scaling: 0.115s @ 500 tokens → 2.038s @ 14000 tokens
  Decode stable across context: Yes

  Context   TTFT       Decode     Rate
  ---------------------------------------------
  500       0.115s    6.968s    71.6 t/s
  2000      0.294s    6.913s    72.2 t/s
  4000      0.513s    6.979s    71.5 t/s
  8000      1.078s    7.148s    69.8 t/s
  14000     2.038s    5.158s    68.6 t/s
```

## Concurrency Scaling

*Source: results/b70-moe-xpugraph-v3.0-20260621-235506/parallel/parallel_benchmark_20260622_000558.json*

```
  Model: Qwen3.6-35B-A3B
  Sequential: 71.1 t/s (TTFT: 0.000s)

  Conc   Throughput   Per-Req    TTFT       p95 Lat
  -------------------------------------------------------
  1      60.3         60.3       0.110s     0.5s
  2      75.0         37.5       0.208s     2.1s
  4      99.0         24.8       0.286s     14.5s
  8      167.0        20.9       0.228s     15.7s
  16     297.9        18.6       0.289s     21.4s ← peak

  Peak: 297.9 t/s @ concurrency=16 (4.2x speedup)
```

## Key Metrics Summary

| Metric | Value |
|--------|-------|
| Quality Score | 20/22 (91%) |
| Avg TTFT (quality tests) | 0.189s |
| Decode Rate (isolated) | 70.8 t/s |
| TTFT @ min context | 0.115s |
| TTFT @ max context | 2.038s |
| Sequential Throughput | 71.1 t/s |
| Peak Concurrent | 297.9 t/s @ C=16 |
| Speedup | 4.2x |

---

*Note: throughput numbers from bench versions before v2.0 counted prompt+completion tokens and should not be compared to v2+ completion-only decode rates. See REFACTORING.md for details.*
