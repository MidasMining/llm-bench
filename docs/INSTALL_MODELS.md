# Model Installation Instructions for Claude-CLI

Instructions for installing and deploying GLM-4.7-Flash and Nemotron-3-Nano on 4-6× RTX A4000 GPUs.

---

## Prerequisites

### Hardware Requirements

| Model | Min VRAM | Recommended | Your Setup (96GB) |
|-------|----------|-------------|-------------------|
| GLM-4.7-Flash (BF16) | 60GB | 64GB+ | ✓ 4 cards |
| GLM-4.7-Flash (AWQ) | 20GB | 32GB+ | ✓ 2 cards |
| Nemotron-3-Nano (FP8) | 32GB | 48GB+ | ✓ 2-4 cards |
| Nemotron-3-Nano (BF16) | 60GB | 64GB+ | ✓ 4 cards |

### Software Requirements

```bash
# Check CUDA version (need 12.1+)
nvidia-smi

# Check Python version (need 3.10+)
python3 --version
```

---

## Part 1: Environment Setup

### Step 1.1: Create Fresh vLLM Environment

```bash
# Create new conda environment (recommended)
conda create -n vllm-compare python=3.11 -y
conda activate vllm-compare

# Or use existing vllm-env
source ~/vllm-env/bin/activate
```

### Step 1.2: Install vLLM Nightly (Required for New Models)

Both GLM-4.7-Flash and Nemotron-3-Nano require bleeding-edge vLLM:

```bash
# Uninstall existing vLLM
pip uninstall vllm -y

# Install nightly build with new model support
pip install -U vllm --pre \
    --index-url https://pypi.org/simple \
    --extra-index-url https://wheels.vllm.ai/nightly

# Install latest transformers (required for chat templates)
pip install git+https://github.com/huggingface/transformers.git

# Verify installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

### Step 1.3: Install Additional Dependencies

```bash
pip install pyyaml requests huggingface_hub hf_transfer
```

---

## Part 2: Download Models

### Step 2.1: Set Up Hugging Face Cache

```bash
# Set cache directory (adjust path as needed)
export HF_HOME=/mnt/data/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1

# Login to Hugging Face (if needed for gated models)
huggingface-cli login
```

### Step 2.2: Download GLM-4.7-Flash

```bash
# Download GLM-4.7-Flash (~60GB)
huggingface-cli download zai-org/GLM-4.7-Flash \
    --local-dir /mnt/data/models/GLM-4.7-Flash \
    --local-dir-use-symlinks False

# Verify download
ls -la /mnt/data/models/GLM-4.7-Flash/
```

Expected files:
- `config.json`
- `model-*.safetensors` (multiple shards)
- `tokenizer.json`
- `tokenizer_config.json`

### Step 2.3: Download Nemotron-3-Nano

Choose one variant:

**Option A: FP8 Quantized (Recommended - Smaller, Faster)**
```bash
# Download Nemotron-3-Nano FP8 (~32GB)
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
    --local-dir /mnt/data/models/Nemotron-3-Nano-FP8 \
    --local-dir-use-symlinks False
```

**Option B: BF16 Full Precision**
```bash
# Download Nemotron-3-Nano BF16 (~60GB)
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --local-dir /mnt/data/models/Nemotron-3-Nano-BF16 \
    --local-dir-use-symlinks False
```

---

## Part 3: Deploy Models with vLLM

### Step 3.1: Deploy GLM-4.7-Flash

**Basic deployment (4× A4000):**

```bash
# Start GLM-4.7-Flash server
vllm serve zai-org/GLM-4.7-Flash \
    --tensor-parallel-size 4 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --served-model-name glm-4.7-flash \
    --host 0.0.0.0 \
    --port 8000
```

**With speculative decoding (faster):**

```bash
vllm serve zai-org/GLM-4.7-Flash \
    --tensor-parallel-size 4 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --served-model-name glm-4.7-flash \
    --host 0.0.0.0 \
    --port 8000
```

### Step 3.2: Deploy Nemotron-3-Nano

**FP8 deployment (2-4× A4000):**

```bash
# Start Nemotron-3-Nano FP8 server
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
    --tensor-parallel-size 4 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.95 \
    --served-model-name nemotron-3-nano \
    --host 0.0.0.0 \
    --port 8000
```

**BF16 deployment (4× A4000):**

```bash
# Start Nemotron-3-Nano BF16 server
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --tensor-parallel-size 4 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95 \
    --served-model-name nemotron-3-nano \
    --host 0.0.0.0 \
    --port 8000
```

**Extended context (512K+ tokens):**

```bash
# For very long context (requires careful VRAM management)
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
    --tensor-parallel-size 4 \
    --max-model-len 524288 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32768 \
    --served-model-name nemotron-3-nano \
    --host 0.0.0.0 \
    --port 8000
```

---

## Part 4: Verify Deployment

### Step 4.1: Check Server Status

```bash
# Check if server is running
curl http://localhost:8000/v1/models | python -m json.tool
```

Expected output:
```json
{
  "object": "list",
  "data": [
    {
      "id": "glm-4.7-flash",
      "object": "model",
      "owned_by": "vllm"
    }
  ]
}
```

### Step 4.2: Test Basic Completion

```bash
# Test GLM-4.7-Flash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7-flash",
    "messages": [{"role": "user", "content": "Write a Python function to reverse a string"}],
    "max_tokens": 200,
    "temperature": 0
  }' | python -m json.tool
```

### Step 4.3: Check VRAM Usage

```bash
# Monitor GPU memory
watch -n 1 nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

---

## Part 5: Run Comparison Tests

### Step 5.1: Sequential Mode (One Model at a Time)

If you have a single GPU pool, run models sequentially:

```bash
cd ~/AI/llm-benchmark

# Test GLM-4.7-Flash (start server first, then run)
python compare_models.py \
    --config models.yaml \
    --mode parallel \
    --models "GLM-4.7-Flash" \
    --output ./results/glm

# Stop server, start Nemotron, then:
python compare_models.py \
    --config models.yaml \
    --mode parallel \
    --models "Nemotron-3-Nano" \
    --output ./results/nemotron
```

### Step 5.2: Quick Test (Practical Tests Only)

```bash
# Skip long-running tests
python compare_models.py \
    --config models.yaml \
    --mode parallel \
    --tests practical \
    --quick
```

### Step 5.3: Full Comparison

```bash
# Run all tests (takes ~30-60 minutes per model)
python compare_models.py \
    --config models.yaml \
    --mode sequential \
    --output ./comparison_results
```

---

## Part 6: Systemd Services (Optional)

### Step 6.1: Create GLM-4.7-Flash Service

```bash
sudo tee /etc/systemd/system/glm-4.7-flash.service << 'EOF'
[Unit]
Description=GLM-4.7-Flash vLLM Server
After=network.target

[Service]
Type=simple
User=llm
WorkingDirectory=/home/llm
Environment="PATH=/home/llm/vllm-env/bin:/usr/local/bin:/usr/bin"
Environment="HF_HOME=/mnt/data/huggingface"
ExecStart=/home/llm/vllm-env/bin/vllm serve zai-org/GLM-4.7-Flash \
    --tensor-parallel-size 4 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --served-model-name glm-4.7-flash \
    --host 0.0.0.0 \
    --port 8001
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable glm-4.7-flash
```

### Step 6.2: Create Nemotron Service

```bash
sudo tee /etc/systemd/system/nemotron-3-nano.service << 'EOF'
[Unit]
Description=Nemotron-3-Nano vLLM Server
After=network.target

[Service]
Type=simple
User=llm
WorkingDirectory=/home/llm
Environment="PATH=/home/llm/vllm-env/bin:/usr/local/bin:/usr/bin"
Environment="HF_HOME=/mnt/data/huggingface"
ExecStart=/home/llm/vllm-env/bin/vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
    --tensor-parallel-size 4 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.95 \
    --served-model-name nemotron-3-nano \
    --host 0.0.0.0 \
    --port 8002
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable nemotron-3-nano
```

---

## Part 7: Troubleshooting

### Issue: "CUDA out of memory"

```bash
# Reduce context length
--max-model-len 65536

# Reduce GPU memory utilization
--gpu-memory-utilization 0.85

# Enable chunked prefill
--enable-chunked-prefill --max-num-batched-tokens 16384
```

### Issue: "Model not found"

```bash
# Check model path
ls -la /mnt/data/models/

# Use local path instead of HF name
vllm serve /mnt/data/models/GLM-4.7-Flash ...
```

### Issue: "Chat template error"

```bash
# Update transformers
pip install git+https://github.com/huggingface/transformers.git

# Or specify chat template manually
--chat-template /path/to/template.jinja
```

### Issue: "Speculative decoding not supported"

```bash
# Remove speculative config for unsupported models
# Nemotron uses Mamba layers which may not support MTP yet
```

### Issue: "Slow startup"

```bash
# Pre-download model
huggingface-cli download <model-name> --local-dir /mnt/data/models/<name>

# Use local path
vllm serve /mnt/data/models/<name> ...
```

---

## Part 8: Model-Specific Notes

### GLM-4.7-Flash

**Thinking Modes:**
- Interleaved thinking (default): Model thinks between responses
- Preserved thinking: Maintains context across turns
- Turn-level thinking: Thinks at start of each turn

**Enable thinking:**
```python
# In API call
messages = [
    {"role": "system", "content": "Think step by step."},
    {"role": "user", "content": "..."}
]
```

**Tool calling:**
```bash
# Already enabled with --enable-auto-tool-choice
# Parser: --tool-call-parser glm47
```

### Nemotron-3-Nano

**Hybrid Architecture:**
- 23 Mamba-2 layers (efficient long context)
- 6 Attention layers (precise reasoning)
- 128 MoE experts, 6 active

**Reasoning control:**
```python
# Enable reasoning traces
messages = [
    {"role": "user", "content": "..."},
]
# Model will include <think>...</think> tags
```

**1M context:**
```bash
# Requires careful memory management
--max-model-len 1048576 \
--gpu-memory-utilization 0.90 \
--enable-chunked-prefill \
--max-num-batched-tokens 8192
```

---

## Quick Reference Card

### Start Seed-OSS (Current)
```bash
vllm serve Seed-OSS/Seed-OSS-36B-Instruct-AWQ --tensor-parallel-size 4 --max-model-len 131072 --port 8000
```

### Start GLM-4.7-Flash
```bash
vllm serve zai-org/GLM-4.7-Flash --tensor-parallel-size 4 --max-model-len 131072 --tool-call-parser glm47 --reasoning-parser glm45 --port 8000
```

### Start Nemotron-3-Nano
```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 --tensor-parallel-size 4 --max-model-len 262144 --port 8000
```

### Run Benchmarks
```bash
python compare_models.py --config models.yaml --mode parallel
```

---

*Generated: January 2026*
*Hardware: 4-6× RTX A4000 (64-96GB VRAM)*
