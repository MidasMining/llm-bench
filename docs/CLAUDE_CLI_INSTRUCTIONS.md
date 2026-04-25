# Claude CLI Instructions: Model Comparison Setup

## Overview

Step-by-step instructions for Claude CLI to install GLM-4.7-Flash and Nemotron-3-Nano models and run the comparison test harness.

---

## Phase 1: Environment Verification

First, verify the environment and existing setup.

### Task 1.1: Check Current Environment

```bash
# Check CUDA and driver version
nvidia-smi

# Check Python environment
which python3
python3 --version

# Check vLLM installation
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')"

# Check available disk space for models
df -h /mnt/data
```

**Expected:**
- CUDA 12.1+
- Python 3.10+
- vLLM installed
- 150GB+ free disk space

### Task 1.2: Verify Existing Seed-OSS Setup

```bash
# Check if Seed-OSS is running
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Quick test
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"seed-oss","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}' \
  | python3 -m json.tool
```

---

## Phase 2: Install New Models

### Task 2.1: Upgrade vLLM for New Model Support

Both GLM-4.7-Flash and Nemotron-3-Nano require recent vLLM builds.

```bash
# Activate your vLLM environment
source ~/vllm-env/bin/activate

# Upgrade vLLM to nightly (includes GLM and Nemotron support)
pip uninstall vllm -y
pip install -U vllm --pre \
    --index-url https://pypi.org/simple \
    --extra-index-url https://wheels.vllm.ai/nightly

# Upgrade transformers for chat templates
pip install -U transformers

# Install hf_transfer for faster downloads
pip install hf_transfer

# Verify
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')"
```

### Task 2.2: Download GLM-4.7-Flash

```bash
# Enable fast transfers
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/mnt/data/huggingface

# Download GLM-4.7-Flash (~60GB)
huggingface-cli download zai-org/GLM-4.7-Flash \
    --local-dir /mnt/data/models/GLM-4.7-Flash \
    --local-dir-use-symlinks False

# Verify download
ls -la /mnt/data/models/GLM-4.7-Flash/
```

**Expected files:**
- config.json
- model-*.safetensors (multiple shards)
- tokenizer.json, tokenizer_config.json

### Task 2.3: Download Nemotron-3-Nano

```bash
# Download Nemotron-3-Nano FP8 (~32GB, recommended)
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
    --local-dir /mnt/data/models/Nemotron-3-Nano-FP8 \
    --local-dir-use-symlinks False

# Verify download
ls -la /mnt/data/models/Nemotron-3-Nano-FP8/
```

---

## Phase 3: Deploy Models for Testing

Since you have a single GPU pool (4× A4000), test models sequentially.

### Task 3.1: Test GLM-4.7-Flash

```bash
# Stop any running vLLM servers first
pkill -f "vllm serve" || true
sleep 5

# Start GLM-4.7-Flash
vllm serve /mnt/data/models/GLM-4.7-Flash \
    --tensor-parallel-size 4 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --served-model-name glm-4.7-flash \
    --host 0.0.0.0 \
    --port 8000

# In another terminal, verify:
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

### Task 3.2: Test Nemotron-3-Nano

```bash
# Stop GLM server first
pkill -f "vllm serve" || true
sleep 5

# Start Nemotron-3-Nano
vllm serve /mnt/data/models/Nemotron-3-Nano-FP8 \
    --tensor-parallel-size 4 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.95 \
    --served-model-name nemotron-3-nano \
    --host 0.0.0.0 \
    --port 8000

# In another terminal, verify:
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

---

## Phase 4: Run Comparison Tests

### Task 4.1: Extract Test Harness

```bash
cd ~/AI

# Extract the benchmark suite
unzip llm-benchmark-v2.zip -d .

# Verify structure
ls -la llm-benchmark/
ls -la llm-benchmark/practical/
```

### Task 4.2: Run Practical Tests (Per Model)

For each model, start the server, then run tests:

```bash
cd ~/AI/llm-benchmark

# Test 1: Seed-OSS (start seed server first)
python compare_models.py \
    --model seed-oss \
    --api-url http://localhost:8000/v1 \
    --output ./results/seed-oss

# Test 2: GLM-4.7-Flash (start glm server first)
python compare_models.py \
    --model glm-4.7-flash \
    --api-url http://localhost:8000/v1 \
    --output ./results/glm

# Test 3: Nemotron-3-Nano (start nemotron server first)
python compare_models.py \
    --model nemotron-3-nano \
    --api-url http://localhost:8000/v1 \
    --output ./results/nemotron
```

### Task 4.3: Run Individual Practical Tests

```bash
cd ~/AI/llm-benchmark/practical

# Quick test - ZMQ only
python zmq_test.py --api-url http://localhost:8000/v1 --model seed-oss

# Nightmare test - most challenging
python nightmare_test.py --api-url http://localhost:8000/v1 --model seed-oss

# All practical tests
python run_practical.py --api-url http://localhost:8000/v1 --model seed-oss
```

### Task 4.4: Run Nightmare Follow-up (If Bugs Missed)

```bash
# If nightmare test misses byte order or info leak bugs
python nightmare_followup.py --api-url http://localhost:8000/v1 --model seed-oss
```

---

## Phase 5: Compare Results

### Task 5.1: Review JSON Results

```bash
cd ~/AI/llm-benchmark/results

# View results
cat seed-oss/comparison_*.json | python3 -m json.tool
cat glm/comparison_*.json | python3 -m json.tool
cat nemotron/comparison_*.json | python3 -m json.tool
```

### Task 5.2: Create Comparison Summary

After testing all three models, compare:

```bash
cd ~/AI/llm-benchmark

# List all result files
find results/ -name "*.json" -exec basename {} \; | sort

# Compare nightmare test scores across models
echo "=== Nightmare Test Scores ==="
for model in seed-oss glm nemotron; do
    score=$(cat results/$model/comparison_*.json 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
for t in data[0].get('tests', []):
    if 'Nightmare' in t.get('name', ''):
        print(f'{t[\"score\"]}/{t[\"max_score\"]}')" 2>/dev/null || echo "N/A")
    echo "$model: $score"
done
```

---

## Quick Reference Commands

### Start Models

```bash
# Seed-OSS
vllm serve Seed-OSS/Seed-OSS-36B-Instruct-AWQ --tensor-parallel-size 4 --max-model-len 131072 --port 8000

# GLM-4.7-Flash
vllm serve /mnt/data/models/GLM-4.7-Flash --tensor-parallel-size 4 --max-model-len 131072 --tool-call-parser glm47 --reasoning-parser glm45 --port 8000

# Nemotron-3-Nano
vllm serve /mnt/data/models/Nemotron-3-Nano-FP8 --tensor-parallel-size 4 --max-model-len 262144 --port 8000
```

### Run Tests

```bash
# All practical tests
python compare_models.py --model MODEL_NAME --api-url http://localhost:8000/v1

# Quick test (zmq + pplns only)
python compare_models.py --model MODEL_NAME --quick

# Specific tests
python compare_models.py --model MODEL_NAME --tests nightmare hiveos_wrapper

# List available tests
python compare_models.py --list-tests
```

### Check Server Status

```bash
# Model list
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Quick completion test
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MODEL_NAME","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

---

## Troubleshooting

### Out of Memory

```bash
# Reduce context length
--max-model-len 65536

# Reduce GPU utilization
--gpu-memory-utilization 0.85
```

### Model Not Loading

```bash
# Check model path
ls -la /mnt/data/models/

# Use explicit local path
vllm serve /mnt/data/models/GLM-4.7-Flash ...
```

### Chat Template Errors

```bash
# Ensure latest transformers
pip install -U transformers

# Or install from git
pip install git+https://github.com/huggingface/transformers.git
```

### Slow Performance

```bash
# Check GPU utilization
nvidia-smi -l 1

# Enable tensor parallelism properly
--tensor-parallel-size 4
```

---

## Expected Test Results

| Test | Seed-OSS | Target |
|------|----------|--------|
| ZMQ (Easy) | 1/1 | 1/1 |
| PPLNS (Hard) | 3/3 | 3/3 |
| Expert (Expert) | 4/4+ | 4/4 |
| Nightmare (Nightmare) | 3-5/5 | 5/5 |
| HiveOS (Practical) | 6-8/8 | 8/8 |

**Key Nightmare Bugs to Watch:**
1. Byte order (prev_hash reversal) - Seed missed this
2. Info leak (logging before submit) - Seed missed this
3. Race/stale condition
4. Memory leak (OrderedDict)
5. Input validation (hex parsing)

---

*Generated: January 2026*
*Hardware: 4-6× RTX A4000*
