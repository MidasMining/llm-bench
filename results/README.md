# Results Directory Structure

Benchmark results are organized by test type and hardware/model configuration.

## Directory Convention

```
results/
├── decode/          # decode_rate_bench.py output (prefill vs decode isolated)
├── parallel/        # parallel_benchmark.py output (concurrency scaling)
├── compare/         # compare_models.py output (quality scoring, general)
├── coding/          # coding_benchmark.py output (BAM filtering tasks)
├── 8xA4000/         # Hardware-specific: 8x RTX A4000 sweep (all test types)
├── gen4/            # PCIe Gen4 bandwidth testing
├── tp8/             # Tensor-parallel=8 scaling results
├── mistral35/       # Model-specific: Mistral-3.5 runs
└── qwen3-235b-thinking/  # Model-specific: Qwen3-235B reasoning model
```

## Naming Convention

Files follow the pattern: `<benchmark>_<timestamp>.json`

- `comparison_YYYYMMDD_HHMMSS.json` — from `compare_models.py`
- `decode_rate_<model>_YYYYMMDD_HHMMSS.json` — from `decode_rate_bench.py`
- `parallel_benchmark_YYYYMMDD_HHMMSS.json` — from `parallel_benchmark.py`
- `coding-bench-<model>-<run>-<task>.json` — from `coding_benchmark.py`

## Where to Put New Results

- **General sweeps** (multiple models, same hardware): `results/<hardware>/`
- **Single model deep-dive**: `results/<model-slug>/`
- **Test-type specific** (decode-only, parallel-only): `results/<test-type>/`
- Default output (no `--output` flag):
  - `decode_rate_bench.py` → `results/decode/`
  - `parallel_benchmark.py` → current directory (use `--output results/parallel/`)
  - `compare_models.py` → requires `--output` flag

## Reading Results

Use `report.py` to generate a unified view:

```bash
# Find latest results for a model across all directories:
python report.py --model seed-oss

# Specify directories explicitly:
python report.py --compare-dir results/8xA4000 --decode-dir results/decode
```

## JSON Schema Notes

All result files include:
- `version` — benchmark tool version that produced the result
- `timestamp` — when the benchmark ran
- `model` — model identifier

Since v2.0 (June 2026), throughput metrics are based on **completion tokens only**
(not prompt+completion). Earlier results may include prompt tokens in throughput
calculations. Check the `version` field or presence of `notes.throughput_metric`
to distinguish.
