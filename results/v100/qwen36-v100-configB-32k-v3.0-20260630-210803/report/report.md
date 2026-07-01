# LLM BENCHMARK REPORT


## Run Info

- **Run name**: qwen36-v100-configB-32k
- **Model**: Qwen3.6-V100
- **Hardware**: V100 PG503 32GB SM70 (H12D-10, EPYC 32c/64t / 125GB)
- **Backend**: vLLM 0.1.dev15586 test-vibha-wht + TQ tq-t3nc + FLASH_ATTN_V100 + SM70 TurboMind GEMM
- **Created**: 2026-06-30T21:24:03
- **Bench version**: 3.0

## Quality & Correctness

*Source: results/v100/qwen36-v100-configB-32k-v3.0-20260630-210803/quality/comparison_20260630_211658.json*

```
  Model: Qwen3.6-V100
  Quality: 22/22 (100.0%)
  Throughput (sequential): 64.0 t/s
  Avg TTFT (prefill): 1.472s
  Test                           Score      TTFT     Time
  ------------------------------------------------------------
  ZMQ Listener Bug               2/2 PASS  1.41s    106.7s
  PPLNS Mining Pool Bugs         3/3 PASS  1.66s    112.6s
  Payment System Bugs            4/4 PASS  1.64s    90.0s
  Stratum Protocol Bugs          5/5 PASS  2.00s    114.3s
  HiveOS Wrapper Creation        8/8 PASS  0.65s    110.0s
```

## Decode Rate (Prefill vs Decode Isolated)

*Source: results/v100/qwen36-v100-configB-32k-v3.0-20260630-210803/decode/decode_rate_Qwen3.6-V100_20260630_211658.json*

```
  Model: Qwen3.6-V100
  Tag: bench
  GPU: NVIDIA RTX A4000
  Median decode rate: 56.3 t/s
  Prefill scaling: 1.234s @ 500 tokens → 5.403s @ 6000 tokens
  Decode stable across context: No

  Context   TTFT       Decode     Rate
  ---------------------------------------------
  500       1.234s    8.438s    59.1 t/s
  2000      2.271s    8.689s    57.4 t/s
  4000      3.879s    9.033s    55.2 t/s
  6000      5.403s    9.371s    53.2 t/s
```

## Concurrency Scaling

*Source: results/v100/qwen36-v100-configB-32k-v3.0-20260630-210803/parallel/parallel_benchmark_20260630_212330.json*

```
  Model: Qwen3.6-V100
  Sequential: 57.5 t/s (TTFT: 0.000s)

  Conc   Throughput   Per-Req    TTFT       p95 Lat
  -------------------------------------------------------
  1      51.9         51.9       0.107s     0.6s
  2      61.8         30.9       0.276s     2.6s
  4      80.5         20.1       0.774s     17.9s
  8      119.9        15.0       1.048s     21.9s
  16     196.3        12.3       2.388s     32.4s
  64     291.8        4.6        19.878s    88.3s ← peak

  Peak: 291.8 t/s @ concurrency=64 (5.1x speedup)
```

## Key Metrics Summary

| Metric | Value |
|--------|-------|
| Quality Score | 22/22 (100%) |
| Avg TTFT (quality tests) | 1.472s |
| Decode Rate (isolated) | 56.3 t/s |
| TTFT @ min context | 1.234s |
| TTFT @ max context | 5.403s |
| Sequential Throughput | 57.5 t/s |
| Peak Concurrent | 291.8 t/s @ C=64 |
| Speedup | 5.1x |

---

*Note: throughput numbers from bench versions before v2.0 counted prompt+completion tokens and should not be compared to v2+ completion-only decode rates. See REFACTORING.md for details.*
