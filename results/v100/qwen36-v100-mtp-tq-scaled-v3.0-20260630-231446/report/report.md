# LLM BENCHMARK REPORT


## Run Info

- **Run name**: qwen36-v100-mtp-tq-scaled
- **Model**: Qwen3.6-V100
- **Hardware**: V100 PG503 32GB SM70 (H12D-10, EPYC 32c/64t / 125GB)
- **Backend**: vLLM 0.1.dev15586 test-vibha-wht + Qwen3_5MoeMTP + tq-t3nc (turboquant_attn.py patched for graph-safe .item())
- **Created**: 2026-06-30T23:25:16
- **Bench version**: 3.0

## Quality & Correctness

*Source: results/v100/qwen36-v100-mtp-tq-scaled-v3.0-20260630-231446/quality/comparison_20260630_232023.json*

```
  Model: Qwen3.6-V100
  Quality: 1/22 (4.5%)
  Throughput (sequential): 105.8 t/s
  Avg TTFT (prefill): 1.365s
  Test                           Score      TTFT     Time
  ------------------------------------------------------------
  ZMQ Listener Bug               0/2 FAIL  0.56s    66.2s
  PPLNS Mining Pool Bugs         0/3 FAIL  1.26s    67.4s
  Payment System Bugs            0/4 FAIL  1.22s    67.1s
  Stratum Protocol Bugs          1/5 FAIL  3.45s    70.0s
  HiveOS Wrapper Creation        0/8 FAIL  0.33s    65.8s
```

## Decode Rate (Prefill vs Decode Isolated)

*Source: results/v100/qwen36-v100-mtp-tq-scaled-v3.0-20260630-231446/decode/decode_rate_Qwen3.6-V100_20260630_232023.json*

```
  Model: Qwen3.6-V100
  Tag: bench
  GPU: NVIDIA RTX A4000
  Median decode rate: 91.6 t/s
  Prefill scaling: 0.555s @ 500 tokens → 10.533s @ 6000 tokens
  Decode stable across context: Yes

  Context   TTFT       Decode     Rate
  ---------------------------------------------
  500       0.555s    5.353s    93.2 t/s
  2000      3.769s    5.356s    93.2 t/s
  4000      7.156s    5.466s    91.3 t/s
  6000      10.533s    5.632s    88.6 t/s
```

## Concurrency Scaling

*Source: results/v100/qwen36-v100-mtp-tq-scaled-v3.0-20260630-231446/parallel/parallel_benchmark_20260630_232406.json*

```
  Model: Qwen3.6-V100
  Sequential: 89.8 t/s (TTFT: 0.000s)

  Conc   Throughput   Per-Req    TTFT       p95 Lat
  -------------------------------------------------------
  1      73.6         73.6       0.085s     0.4s
  2      95.9         48.0       0.131s     1.7s
  4      144.4        36.1       0.309s     10.0s
  8      217.5        27.2       0.940s     12.1s
  16     341.6        21.4       1.481s     18.6s ← peak

  Peak: 341.6 t/s @ concurrency=16 (3.8x speedup)
```

## Key Metrics Summary

| Metric | Value |
|--------|-------|
| Quality Score | 1/22 (5%) |
| Avg TTFT (quality tests) | 1.365s |
| Decode Rate (isolated) | 91.6 t/s |
| TTFT @ min context | 0.555s |
| TTFT @ max context | 10.533s |
| Sequential Throughput | 89.8 t/s |
| Peak Concurrent | 341.6 t/s @ C=16 |
| Speedup | 3.8x |

---

*Note: throughput numbers from bench versions before v2.0 counted prompt+completion tokens and should not be compared to v2+ completion-only decode rates. See REFACTORING.md for details.*
