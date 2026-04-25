# SGLang + GLM-4.7-Flash on 4x A4000 - Quick Start

## Tested Configuration (2026-02-04)

**Model**: `QuantTrio/GLM-4.7-Flash-AWQ` (30B MoE, 3B active)
**Framework**: SGLang 0.3.2.dev9039+pr.17247
**Hardware**: 4x NVIDIA RTX A4000 (16GB each)
**Performance**: 293 tok/s peak @ concurrency 8, **113 tok/s single-request**

## Installation (Specific Versions Required!)

```bash
# Use uv for reliable installation
pip install uv

# Install SGLang PR version with GLM-4.7 support
uv pip install "sglang==0.3.2.dev9039+pr-17247.g90c446848" \
  --extra-index-url https://sgl-project.github.io/whl/pr/ \
  -p /home/llm/hf-env

# Install specific transformers commit
uv pip install "git+https://github.com/huggingface/transformers.git@76732b4e7120808ff989edbd16401f61fa6a0afa" \
  -p /home/llm/hf-env
```

## Launch Command

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PATH="/home/llm/hf-env/bin:$PATH"

python -m sglang.launch_server \
  --model-path QuantTrio/GLM-4.7-Flash-AWQ \
  --tp-size 4 \
  --mem-fraction-static 0.90 \
  --context-length 65536 \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend triton \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --max-running-requests 64 \
  --host 0.0.0.0 \
  --port 8200
```

## Benchmark Results

| Concurrency | Throughput | Per-Request | Latency |
|-------------|------------|-------------|---------|
| 1 | 113 t/s | 113 t/s | 0.4s |
| 2 | 147 t/s | 74 t/s | 1.0s |
| 4 | 181 t/s | 45 t/s | 4.7s |
| 8 | **293 t/s** | 37 t/s | 4.9s |
| 16 | 136 t/s | 8.5 t/s | 35.6s |
| 32 | 265 t/s | 8.3 t/s | 35.2s |

**Peak**: 293 tok/s @ concurrency 8

## Comparison: SGLang GLM vs vLLM Seed-OSS

| Metric | SGLang + GLM-4.7-Flash | vLLM + Seed-OSS-36B |
|--------|------------------------|---------------------|
| Model Type | 30B MoE (3B active) | 36B Dense |
| Single-request | **113 t/s** | 42 t/s |
| Peak throughput | 293 t/s (C=8) | **771 t/s** (C=64) |
| KV Cache | **355K tokens** | 123K tokens |
| Best For | Interactive, low latency | Swarms, batch jobs |

## Key Insights

1. **GLM is 2.7x faster per-request** - MoE with only 3B active params
2. **Seed-OSS scales better at high concurrency** - Dense model batches efficiently
3. **FP8 KV Cache** (`--kv-cache-dtype fp8_e5m2`) gives 2.9x more token capacity
4. **CUDA graphs** only captured for bs 1,2,4,8 - higher concurrency falls back

## When to Use Which

- **GLM-4.7-Flash + SGLang**: Coding assistants, chat, interactive use
- **Seed-OSS-36B + vLLM**: Agent swarms, batch processing, high throughput

## Notes

- SGLang PR version doesn't work with Seed-OSS (OOM during model init)
- vLLM 0.14 doesn't support GLM-4.7-Flash (needs dev transformers)
- Keep separate venvs if you need both frameworks
