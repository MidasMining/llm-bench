# LLM BENCHMARK REPORT


## Run Info

- **Run name**: b70-moe-parallel-sweep
- **Model**: Qwen3.6-35B-A3B
- **Hardware**: 1x Arc Pro B70 32GB (BMG-G31, Xe2)
- **Backend**: vLLM XPU Graph FULL (native source, patched MoE+CCL)
- **Created**: 2026-06-22T00:55:04
- **Bench version**: 3.0

## Concurrency Scaling

*Source: results/b70-moe-parallel-sweep-v3.0-20260622-005131/parallel/parallel_benchmark_20260622_005504.json*

```
  Model: Qwen3.6-35B-A3B
  Sequential: 71.5 t/s (TTFT: 0.000s)

  Conc   Throughput   Per-Req    TTFT       p95 Lat
  -------------------------------------------------------
  1      59.5         59.5       0.116s     0.5s
  2      74.8         37.4       0.209s     2.1s
  4      99.6         24.9       0.219s     14.5s
  8      166.8        20.9       0.234s     15.7s
  16     296.8        18.6       0.294s     21.5s
  24     470.8        19.6       0.357s     20.9s
  32     494.5        15.4       0.417s     25.2s
  48     747.6        15.6       0.526s     26.4s
  64     820.3        12.8       0.636s     31.8s ← peak

  Peak: 820.3 t/s @ concurrency=64 (11.5x speedup)
```

## Key Metrics Summary

| Metric | Value |
|--------|-------|
| Sequential Throughput | 71.5 t/s |
| Peak Concurrent | 820.3 t/s @ C=64 |
| Speedup | 11.5x |

---

*Note: throughput numbers from bench versions before v2.0 counted prompt+completion tokens and should not be compared to v2+ completion-only decode rates. See REFACTORING.md for details.*
