# Seed-OSS-36B on 4x A4000 - Quick Start

## Tested Configuration (2026-01-31)

**Model**: `QuantTrio/Seed-OSS-36B-Instruct-AWQ`
**Hardware**: 4x NVIDIA RTX A4000 (16GB each)
**Performance**: 771 tok/s peak @ concurrency 64

## Launch Command

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PATH="/home/llm/hf-env/bin:$PATH"

/home/llm/hf-env/bin/vllm serve QuantTrio/Seed-OSS-36B-Instruct-AWQ \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --attention-backend FLASHINFER \
  --port 8100 \
  --served-model-name seed-oss
```

Or use the startup script:
```bash
/tmp/start-vllm-flashinfer.sh
```

## Context vs Concurrency Tradeoff

| --max-model-len | Max Concurrent Requests | Use Case |
|-----------------|-------------------------|----------|
| 8192 | 15 | High-throughput API |
| 16384 | ~7 | General use (recommended) |
| 32768 | ~4 | Longer documents |
| 65536 | ~2 | Long context tasks |

## Test Server

```bash
curl http://localhost:8100/v1/models
```

## Run Benchmark

```bash
cd /home/llm/llm-bench
/home/llm/hf-env/bin/python parallel_benchmark.py \
  --api-url http://localhost:8100/v1 \
  --model seed-oss \
  --max-concurrent 128
```

## Shutdown

```bash
pkill -f vllm
# or
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs kill -9
```

## Key Notes

- **TP=4 required** - TP=2 causes OOM during sampler warmup
- **--enforce-eager required** - CUDA graph capture fails on A4000s
- **FlashInfer needs ninja in PATH** - already in hf-env/bin
- **FLASHINFER vs FLASH_ATTN** - identical memory, FlashInfer ~0.7% faster
- **Model cached at**: `~/.cache/huggingface/hub/models--QuantTrio--Seed-OSS-36B-Instruct-AWQ`
