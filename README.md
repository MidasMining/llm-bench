# LLM Benchmark Suite v2.0

Comprehensive benchmarking suite for evaluating local LLM models on real-world coding and debugging tasks.

## Features

- **Standard Benchmarks**: 46 tests across code, reasoning, knowledge, tool use, speed, and context
- **Practical Debugging**: Real bugs from production mining pool code
- **Long Context Tests**: Evaluation from 32K to 1M tokens
- **Multi-Model Comparison**: Side-by-side comparison with detailed reports

## File Structure

```
llm-benchmark/
├── compare_models.py          # Main comparison harness (v2.0)
├── long_context_test.py       # Long context evaluation
├── models.yaml                # Model configuration
├── INSTALL_MODELS.md          # Installation instructions
├── README.md                  # This file
├── README_COMPARISON.md       # Detailed comparison docs
└── practical/                 # Practical debugging tests
    ├── run_practical.py       # Test runner
    ├── zmq_test.py           # Easy (1 bug)
    ├── pplns_test.py         # Hard (3 bugs)
    ├── expert_test.py        # Expert (4 bugs)
    ├── nightmare_test.py     # Nightmare (5 bugs)
    ├── nightmare_followup.py # Follow-up for missed bugs
    └── hiveos_wrapper_test.py # File creation (8 criteria)
```

## Quick Start

### 1. Test Against Running Model

```bash
# Run all practical tests against model on port 8000
python compare_models.py --model seed-oss --api-url http://localhost:8000/v1

# Run specific tests
python compare_models.py --model seed-oss --tests nightmare hiveos_wrapper
```

### 2. Run Individual Tests

```bash
cd practical/

# Run single test
python zmq_test.py --api-url http://localhost:8000/v1 --model seed-oss

# Run all practical tests
python run_practical.py --api-url http://localhost:8000/v1 --model seed-oss
```

### 3. Multi-Model Comparison

```bash
# With config file
python compare_models.py --config models.yaml

# Quick mode (skip slow tests)
python compare_models.py --config models.yaml --quick
```

## Practical Tests

| Test | Difficulty | Bugs | Description |
|------|------------|------|-------------|
| `zmq_test.py` | Easy | 1 | Threading bug in ZMQ listener |
| `pplns_test.py` | Hard | 3 | Config mismatch, hardcoded values |
| `expert_test.py` | Expert | 4 | Race condition, SQL injection, float precision |
| `nightmare_test.py` | Nightmare | 5 | Crypto byte order, info leak, memory leak |
| `hiveos_wrapper_test.py` | Practical | 8 | Multi-file creation following conventions |

## Models Supported

| Model | Params | Active | Architecture | Quantization |
|-------|--------|--------|--------------|--------------|
| Seed-OSS-36B | 36B | 36B | Dense Transformer | AWQ INT4 |
| GLM-4.7-Flash | 30B | ~3B | MoE + MLA | NVFP4 |
| GPT-OSS-20B | 21B | 3.6B | MoE (32 experts) | MXFP4 |
| Devstral-Small-24B | 24B | 24B | Mistral Dense | AWQ INT4 |
| Qwen3-30B-A3B | 30B | 3B | MoE (128 experts) | AWQ INT4 |
| Nemotron-3-Nano | 31.6B | 3.6B | Hybrid Mamba-MoE | FP8 |

See `INSTALL_MODELS.md` for detailed installation instructions.

## Benchmark Results

All benchmarks run on RTX 5090 (32GB GDDR7) with vLLM 0.14.0, temperature=0.0, max_tokens=8192, timeout=600s.

### Score Summary

| Model | ZMQ (2) | PPLNS (3) | Payment (4) | Stratum (5) | HiveOS (8) | **Total** | **Score** | **Throughput** |
|-------|---------|-----------|-------------|-------------|------------|-----------|-----------|----------------|
| **Seed-OSS-36B** | 2/2 | 3/3 | 4/4 | 5/5 | 8/8 | **22/22** | **100.0%** | 38.4 t/s |
| **Qwen3-30B-A3B** | 2/2 | 3/3 | 4/4 | 5/5 | 8/8 | **22/22** | **100.0%** | 31.2 t/s |
| Devstral-Small-24B | 2/2 | 3/3 | 4/4 | 4/5 | 8/8 | 21/22 | 95.5% | 53.6 t/s |
| GLM-4.7-Flash | 0/2* | 3/3 | 0/4* | 0/5* | 8/8 | 11/22 | 50.0% | 4.4 t/s |
| GPT-OSS-20B | 2/2 | 3/3 | 4/4 | 0/5** | 0/8** | 9/22 | 40.9% | 26.0 t/s |
| Nemotron-3-Nano | - | - | - | - | - | N/A | OOM | N/A |

\* = Timed out at 600s (model generates extensive thinking tokens before output)
\*\* = 0 tokens returned (model error or timeout)

### Time Per Test (seconds)

| Model | ZMQ | PPLNS | Payment | Stratum | HiveOS | **Total** |
|-------|-----|-------|---------|---------|--------|-----------|
| Devstral-Small-24B | 13.4 | 203.7 | 76.0 | 43.0 | 23.3 | **359** |
| GPT-OSS-20B | 134.4 | 133.3 | 105.1 | 183.3 | 183.0 | **739** |
| Qwen3-30B-A3B | 29.4 | 114.9 | 158.2 | 312.6 | 311.6 | **926** |
| Seed-OSS-36B | 240.3 | 241.5 | 241.0 | 242.1 | 184.9 | **1150** |
| GLM-4.7-Flash | 600* | 540.2 | 600* | 600* | 265.0 | **2605** |

### Key Findings

1. **Seed-OSS-36B** and **Qwen3-30B-A3B** both achieved perfect 100% scores, finding all bugs across all difficulty levels
2. **Devstral-Small-24B** had the highest throughput (53.6 t/s) and near-perfect accuracy (95.5%), missing only the byte order/endianness check in the nightmare test
3. **GLM-4.7-Flash** suffered from 600s timeouts on 3/5 tests due to extensive thinking token generation; the 2 tests that completed scored perfectly
4. **GPT-OSS-20B** was fast and scored well on easy/medium tests but failed on the nightmare and HiveOS practical tests
5. **Nemotron-3-Nano FP8** (30.52 GiB) could not fit on a single RTX 5090 (32 GB)

### VRAM Usage

| Model | Quantization | Model Size | VRAM Used | Fits 32GB? |
|-------|-------------|------------|-----------|------------|
| GPT-OSS-20B | MXFP4 | 12.9 GB | ~30 GB | Yes |
| Devstral-Small-24B | Compressed Tensors | 15.2 GB | ~28 GB | Yes |
| Qwen3-30B-A3B | AWQ INT4 | 16 GB | ~26 GB | Yes |
| GLM-4.7-Flash | NVFP4 | 20 GB | ~31 GB | Yes |
| Seed-OSS-36B | AWQ INT4 | 20 GB | ~30 GB | Yes |
| Nemotron-3-Nano | FP8 | 30.5 GB | 30.5 GB | No (OOM) |

## Configuration

Edit `models.yaml` to customize:

```yaml
models:
  - name: "Seed-OSS-36B"
    api_url: "http://localhost:8000/v1"
    model_id: "seed-oss"

settings:
  temperature: 0.0
  max_tokens: 8192
  timeout: 600
```

## Command Reference

```bash
# List available tests
python compare_models.py --list-tests

# Run all tests
python compare_models.py --model seed-oss

# Run specific tests
python compare_models.py --tests zmq pplns expert

# Quick mode (easy tests only)
python compare_models.py --quick

# Include long context tests
python compare_models.py --long-context

# Save results to file
python compare_models.py --output ./results
```

## Requirements

```bash
pip install pyyaml requests
```

## License

MIT License

---

*Created: January 2026*
*For: Mining Pool Development & AI Model Evaluation*
