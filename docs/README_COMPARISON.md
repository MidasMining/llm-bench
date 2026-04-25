# LLM Model Comparison Test Harness v1.0

Unified test framework for comparing multiple LLM models across benchmarks.

## Overview

This harness runs your existing benchmark suite against multiple models and produces side-by-side comparison reports.

## Files Included

| File | Description |
|------|-------------|
| `compare_models.py` | Main comparison script |
| `long_context_test.py` | Long context evaluation (32K-1M tokens) |
| `models.yaml` | Configuration for Seed, GLM, Nemotron |
| `INSTALL_MODELS.md` | Step-by-step installation guide |

## Requirements

### Existing Benchmark Suite

This harness integrates with your existing `llm-benchmark` setup:

```
llm-benchmark/
├── benchmark.py           # Standard 46-test benchmark (existing)
├── practical/             # Practical debugging tests (existing)
│   ├── zmq_test.py
│   ├── pplns_test.py
│   ├── expert_test.py
│   ├── nightmare_test.py
│   └── hiveos_wrapper_test.py
├── compare_models.py      # NEW: Comparison harness
├── long_context_test.py   # NEW: Long context tests
├── models.yaml            # NEW: Model configuration
└── INSTALL_MODELS.md      # NEW: Installation guide
```

### Python Dependencies

```bash
pip install pyyaml requests
```

## Quick Start

### 1. Install New Models

Follow `INSTALL_MODELS.md` to download and deploy:
- GLM-4.7-Flash
- Nemotron-3-Nano

### 2. Run Comparison

**Option A: Test against running endpoints**

Start each model server, then:

```bash
# Test all models (assumes servers on ports 8000, 8001, 8002)
python compare_models.py --config models.yaml --mode parallel
```

**Option B: Sequential testing (one GPU pool)**

```bash
# Harness starts/stops models automatically
python compare_models.py --config models.yaml --mode sequential
```

### 3. View Results

```bash
# Reports saved to ./comparison_results/
ls comparison_results/
# comparison_20260120_123456.md   # Markdown report
# comparison_20260120_123456.json # Detailed JSON
```

## Configuration

Edit `models.yaml` to customize:

```yaml
models:
  - name: "Seed-OSS-36B"
    api_url: "http://localhost:8000/v1"
    model_id: "seed-oss"
    context_limit: 131072
    
tests:
  standard:
    enabled: true
  practical:
    enabled: true
  long_context:
    enabled: true
    context_sizes: [32000, 64000, 128000, 256000]
```

## Test Categories

### Standard Benchmark (46 tests)
- Code generation (10)
- Reasoning (10)
- Knowledge (10)
- Tool use (10)
- Speed (3)
- Context (3)

### Practical Debugging
| Test | Difficulty | What it Tests |
|------|------------|---------------|
| ZMQ | Easy | Threading bugs |
| PPLNS | Hard | Config/logic errors |
| Expert | Expert | Race conditions, SQL injection |
| Nightmare | Nightmare | Crypto byte order, OPSEC |
| HiveOS | Practical | Multi-file creation |

### Long Context
| Size | Tests |
|------|-------|
| 32K | Needle-in-haystack, cross-file reasoning |
| 64K | Same as above |
| 128K | Seed/GLM limit |
| 256K | Nemotron only |
| 512K | Nemotron only |

## Command Reference

```bash
# Full comparison (all tests, all models)
python compare_models.py --config models.yaml

# Quick mode (skip long context)
python compare_models.py --config models.yaml --quick

# Specific tests only
python compare_models.py --config models.yaml --tests practical nightmare

# Specific models only
python compare_models.py --config models.yaml --models "GLM-4.7-Flash"

# Custom output directory
python compare_models.py --config models.yaml --output ./my_results
```

## Output Example

```
╔═══════════════════════════════════════════════════════════════╗
║                 MODEL COMPARISON RESULTS                      ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  STANDARD BENCHMARKS                                          ║
║  ───────────────────────────────────────────────────────────  ║
║                  Seed-OSS    GLM-4.7    Nemotron    Winner    ║
║  Code            80.0%       85.0%      78.0%       GLM       ║
║  Reasoning       90.0%       88.0%      92.0%       Nemotron  ║
║  Knowledge       100.0%      98.0%      96.0%       Seed      ║
║                                                               ║
║  PRACTICAL DEBUGGING                                          ║
║  ───────────────────────────────────────────────────────────  ║
║  Nightmare       3/5         4/5        4/5         GLM/Nem   ║
║  Byte order bug  MISSED      FOUND      FOUND                 ║
║                                                               ║
║  LONG CONTEXT                                                 ║
║  ───────────────────────────────────────────────────────────  ║
║  256K tokens     SKIP        SKIP       PASS        Nemotron  ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

## Merging with Existing Setup

Copy these new files to your existing `llm-benchmark` directory:

```bash
# On your H12D-8D machine
cd ~/AI/llm-benchmark

# Copy new files
cp /path/to/new/compare_models.py .
cp /path/to/new/long_context_test.py .
cp /path/to/new/models.yaml .
cp /path/to/new/INSTALL_MODELS.md .

# Verify structure
ls -la
# Should show both existing (benchmark.py, practical/) and new files
```

## Workflow

1. **Keep Seed running** on port 8000 (your current setup)
2. **Download GLM & Nemotron** following INSTALL_MODELS.md
3. **Test one at a time** (sequential mode) or deploy multiple endpoints
4. **Run comparison** and review reports
5. **Decide** based on your specific workload results

---

*Generated: January 2026*
*For: Midas (THT Mining Pool Development)*
