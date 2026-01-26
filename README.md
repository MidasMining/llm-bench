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

| Model | Context | Architecture | Notes |
|-------|---------|--------------|-------|
| Seed-OSS-36B | 128K | Dense Transformer | Current baseline |
| GLM-4.7-Flash | 128K | MoE (30B/3B active) | Purpose-built for agentic coding |
| Nemotron-3-Nano | 1M | Hybrid Mamba-MoE | Fastest throughput |

See `INSTALL_MODELS.md` for detailed installation instructions.

## Example Output

```
========================================
Testing: Seed-OSS-36B
API: http://localhost:8000/v1
========================================

Running: ZMQ Listener Bug (Easy)
  PASS 1/1 found (15.2s, 1847 tokens)
    [PASS] socket_thread_issue
    [PASS] context_thread_safety

Running: Stratum Protocol Bugs (Nightmare)
  FAIL 3/5 found (45.3s, 6053 tokens)
    [FAIL] byte_order
    [FAIL] info_leak
    [PASS] race_stale
    [PASS] memory_leak
    [PASS] input_validation

----------------------------------------
SUMMARY
----------------------------------------
  ZMQ Listener Bug                 PASS           1/1
  PPLNS Mining Pool Bugs           PASS           3/3
  Payment System Bugs              PASS           4/4
  Stratum Protocol Bugs            FAIL           3/5
  HiveOS Wrapper Creation          PASS           7/8

  Total: 18/21 (85.7%)
  Time: 156.3s
```

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
