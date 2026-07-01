# LLM BENCHMARK REPORT


## Run Info

- **Run name**: qwen36-v100-mtp-pilot-8k
- **Model**: Qwen3.6-V100
- **Hardware**: V100 PG503 32GB SM70 (H12D-10, EPYC 32c/64t / 125GB)
- **Backend**: vLLM 0.1.dev15586 test-vibha-wht + Qwen3_5MoeMTP + fp16 KV
- **Created**: 2026-06-30T22:23:47
- **Bench version**: 3.0

## Quality & Correctness

*Source: results/v100/qwen36-v100-mtp-pilot-8k-v3.0-20260630-214922/quality/comparison_20260630_221216.json*

```
  Model: Qwen3.6-V100
  Quality: 22/22 (100.0%)
  Throughput (sequential): 22.4 t/s
  Avg TTFT (prefill): 1.379s
  Test                           Score      TTFT     Time
  ------------------------------------------------------------
  ZMQ Listener Bug               2/2 PASS  0.58s    77.9s
  PPLNS Mining Pool Bugs         3/3 PASS  1.34s    266.0s
  Payment System Bugs            4/4 PASS  1.23s    350.0s
  Stratum Protocol Bugs          5/5 PASS  3.40s    298.8s
  HiveOS Wrapper Creation        8/8 PASS  0.35s    381.0s
```

## Decode Rate (Prefill vs Decode Isolated)

*Source: results/v100/qwen36-v100-mtp-pilot-8k-v3.0-20260630-214922/decode/decode_rate_Qwen3.6-V100_20260630_221216.json*

```
  Model: Qwen3.6-V100
  Tag: bench
  GPU: NVIDIA RTX A4000
  Median decode rate: 30.2 t/s
  Prefill scaling: 0.567s @ 500 tokens → 8.751s @ 6000 tokens
  Decode stable across context: No

  Context   TTFT       Decode     Rate
  ---------------------------------------------
  500       0.567s    9.549s    52.3 t/s
  2000      3.677s    15.588s    32.0 t/s
  4000      6.182s    23.620s    21.1 t/s
  6000      8.751s    32.507s    15.3 t/s
```

## Concurrency Scaling

*Source: results/v100/qwen36-v100-mtp-pilot-8k-v3.0-20260630-214922/parallel/parallel_benchmark_20260630_222320.json*

```
  Model: Qwen3.6-V100
  Sequential: 64.8 t/s (TTFT: 0.000s)

  Conc   Throughput   Per-Req    TTFT       p95 Lat
  -------------------------------------------------------
  1      64.4         64.4       0.120s     0.5s
  2      82.9         41.5       0.156s     1.9s
  4      90.4         22.6       0.428s     15.9s
  64     123.5        1.9        93.602s    196.3s ← peak

  Peak: 123.5 t/s @ concurrency=64 (1.9x speedup)
```

## Key Metrics Summary

| Metric | Value |
|--------|-------|
| Quality Score | 22/22 (100%) |
| Avg TTFT (quality tests) | 1.379s |
| Decode Rate (isolated) | 30.2 t/s |
| TTFT @ min context | 0.567s |
| TTFT @ max context | 8.751s |
| Sequential Throughput | 64.8 t/s |
| Peak Concurrent | 123.5 t/s @ C=64 |
| Speedup | 1.9x |

---

*Note: throughput numbers from bench versions before v2.0 counted prompt+completion tokens and should not be compared to v2+ completion-only decode rates. See REFACTORING.md for details.*
