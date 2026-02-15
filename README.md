# LLM Benchmark Suite v2.1

Comprehensive benchmarking suite for evaluating local LLM models on real-world coding and debugging tasks. Tests models on production mining pool code with bugs ranging from easy threading issues to expert-level crypto byte order vulnerabilities.

## Test Suite

| Test | Difficulty | Checks | Description |
|------|------------|--------|-------------|
| ZMQ Listener | Easy | 2 | Threading bug in ZMQ socket listener |
| PPLNS Mining Pool | Hard | 3 | Config mismatch, hardcoded values, unused variables |
| Payment System | Expert | 4 | Race condition, SQL injection, float precision, atomicity |
| Stratum Protocol | Nightmare | 5 | Byte order/endianness, info leak, stale data, memory leak, input validation |
| HiveOS Wrapper | Practical | 8 | Multi-file creation (manifest, config, run, stats scripts) |

**Total: 22 checks across 5 practical debugging tests**

---

## 8x RTX A4000 Results (128GB VRAM)

**Platform**: 8x NVIDIA RTX A4000 (16GB each), AMD EPYC 7532, PCIe Gen4
**Software**: vLLM 0.14, SGLang 0.3.2, Python 3.12, CUDA 12.8
**Date**: February 12-15, 2026

### Model Rankings

| # | Model | Quant | TP | Quality | Single t/s | Peak t/s | Context | Framework |
|---|-------|-------|-----|---------|-----------|----------|---------|-----------|
| 1 | **Devstral-2-123B** | AWQ | 8 | **100%** (22/22) | 41 | 300 @ C=32 | 32K | vLLM |
| 2 | **Nemotron-3-Nano-30B** | BF16 | 8 | **100%** (22/22) | **205** | **1628** @ C=32 | 16K | vLLM |
| 3 | **Qwen3-Coder-30B-A3B** | AWQ | 4 | **100%** (22/22) | 184 | 1025 @ C=32 | 32K | vLLM |
| 4 | **GLM-4.7-Flash** | AWQ | 4 | **100%** (22/22) | 101 | 566 @ C=8 | 65K | SGLang |
| 5 | **Magistral-Small-2509** | BF16 | 8 | 95.5% (21/22) | 88 | 1071 @ C=32 | 131K | vLLM |
| 6 | **Magistral-Small-2506** | AWQ | 8 | 95.5% (21/22) | 156 | **1831** @ C=32 | 32K | vLLM |
| 7 | Qwen3-32B | AWQ | 8 | 95.5% (21/22) | 78 | 1013 @ C=32 | 32K | vLLM |
| 8 | EXAONE-4.0-32B | GPTQ g32 | 8 | 95.5% (21/22) | 110 | 719 @ C=64 | 131K | vLLM |
| 9 | Qwen3-30B-A3B | AWQ | 4 | 95.5% (21/22) | 178 | 1575 @ C=32 | 32K | vLLM |
| 10 | Devstral-Small-2-24B | AWQ | 8 | 95.5% (21/22) | 148 | 1452 @ C=32 | 32K | vLLM |
| 11 | Seed-OSS-36B | AWQ | 8 | 90.9% (20/22) | 88 | 1163 @ C=32 | 32K | vLLM |
| 12 | Qwen3-30B-A3B-Thinking | AWQ | 4 | 81.8% (18/22) | 160 | 1031 @ C=32 | 32K | vLLM |
| 13 | **Nanbeige4.1-3B** | BF16 | 4 | 77.3% (17/22) | 187 | 1239 @ C=64 | 131K | vLLM |
| 14 | DS-R1-Distill-Qwen-32B | AWQ | 8 | 54.5% (12/22) | 78 | 992 @ C=32 | 32K | vLLM |
| 15 | DS-R1-Distill-Llama-70B | AWQ | 8 | 45.5% (10/22) | 57 | 540 @ C=32 | 16K | vLLM |
| 16 | GPT-OSS-20B | MXFP4 | 8 | 40.9% (9/22) | 52 | 933 @ C=16 | 8K | vLLM |

### Category Winners

| Category | Model | Why |
|----------|-------|-----|
| **Best Quality** | Devstral-2-123B | Only model to score 5/5 on Stratum byte order test |
| **Best Overall** | Nemotron-3-Nano-30B | 100% quality + 205 t/s single + 1628 t/s peak |
| **Best Code Model** | Qwen3-Coder-30B-A3B | 100% quality, purpose-built for code tasks |
| **Best Throughput** | Magistral-Small-2506 AWQ | 1831 t/s peak, 95.5% quality, only 14GB |
| **Best Long Context** | Magistral-Small-2509 | 131K context, 95.5% quality, working reasoning |
| **Best Reasoning** | Qwen3-32B | 95.5% quality with `<think>` mode |

### Detailed Test Results

| Model | ZMQ (2) | PPLNS (3) | Payment (4) | Stratum (5) | HiveOS (8) | Total |
|-------|:-------:|:---------:|:-----------:|:-----------:|:----------:|:-----:|
| Devstral-2-123B | 2/2 | 3/3 | 4/4 | **5/5** | 8/8 | **22/22** |
| Nemotron-3-Nano-30B | 2/2 | 3/3 | 4/4 | **5/5** | 8/8 | **22/22** |
| Qwen3-Coder-30B-A3B | 2/2 | 3/3 | 4/4 | **5/5** | 8/8 | **22/22** |
| GLM-4.7-Flash | 2/2 | 3/3 | 4/4 | **5/5** | 8/8 | **22/22** |
| Magistral-Small-2509 | 2/2 | 3/3 | 4/4 | 4/5 | 8/8 | 21/22 |
| Magistral-Small-2506 | 2/2 | 3/3 | 4/4 | 4/5 | 8/8 | 21/22 |
| Qwen3-32B | 2/2 | 3/3 | 4/4 | 4/5 | 8/8 | 21/22 |
| EXAONE-4.0-32B | 2/2 | 3/3 | 4/4 | 4/5 | 8/8 | 21/22 |
| Qwen3-30B-A3B | 2/2 | 3/3 | 4/4 | 4/5 | 8/8 | 21/22 |
| Devstral-Small-2-24B | 2/2 | 3/3 | 4/4 | 4/5 | 8/8 | 21/22 |
| Seed-OSS-36B | 2/2 | 3/3 | 4/4 | 3/5 | 8/8 | 20/22 |
| Qwen3-30B-A3B-Think | 2/2 | 2/3 | 4/4 | 4/5 | 6/8 | 18/22 |
| **Nanbeige4.1-3B** | 2/2 | 3/3 | 3/4 | 4/5 | 5/8 | **17/22** |
| DS-R1-Distill-Qwen-32B | 2/2 | 2/3 | 0/4 | 0/5 | 8/8 | 12/22 |
| DS-R1-Distill-Llama-70B | 2/2 | 0/3 | 0/4 | 0/5 | 8/8 | 10/22 |
| GPT-OSS-20B | 0/2 | 3/3 | 0/4 | 0/5 | 6/8 | 9/22 |

### Throughput Scaling (tokens/second)

| Concurrency | Nemotron | Magistral-2506 | Qwen3-30B | Devstral-S | Nanbeige-3B | Seed-OSS | Qwen3-Coder | Magistral-2509 | Qwen3-32B | EXAONE | Devstral-2 |
|:-----------:|:--------:|:--------------:|:---------:|:----------:|:-----------:|:--------:|:-----------:|:--------------:|:---------:|:------:|:----------:|
| 1 | 205 | 156 | 178 | 180 | 249 | 78 | 184 | 88 | 78 | 110 | 41 |
| 2 | 327 | 275 | 347 | 186 | 251 | 168 | 333 | 135 | 170 | 142 | 82 |
| 4 | 571 | 524 | 621 | 425 | 279 | 373 | 462 | 229 | 331 | 161 | 124 |
| 8 | 765 | 839 | 869 | 691 | 453 | 616 | 589 | 399 | 571 | 279 | 185 |
| 16 | 1158 | 1266 | 1208 | 1051 | 737 | 923 | 840 | 649 | 825 | 444 | 271 |
| **32** | **1628** | **1831** | **1575** | **1452** | 1044 | **1163** | **1025** | **1071** | **1013** | 636 | **300** |
| **64** | - | - | - | - | **1239** | - | - | - | - | **719** | - |

### Failed / Incompatible Models

| Model | Issue |
|-------|-------|
| GPT-OSS-120B FP16 | OOM - repo is actually FP16, not quantized (15.6GB/GPU) |
| GLM-4.5-Air AWQ | Marlin kernel error: `size_n=2736 not divisible by tile_n=64` |
| Qwen3-Next-80B-A3B AWQ | 2 KV heads (max TP=2), 24.5GB/GPU exceeds 16GB |
| Qwen3-Coder-Next AWQ | 2 KV heads (max TP=2), needs vLLM 0.15+ |
| EXAONE-4.0-32B AWQ/GPTQ g128 | Marlin min_thread_k=128 alignment fails at TP>2 (3424%128≠0). Fixed with custom GPTQ g32 |

---

## Key Findings

### 1. Quality: Non-Reasoning Models Win on Structured Tasks
The top 4 models by quality (100%) are all non-reasoning: Devstral-2-123B, Nemotron-3-Nano-30B, Qwen3-Coder-30B-A3B, and GLM-4.7-Flash. Reasoning models (DeepSeek-R1 distillations) score worst due to over-reasoning and poor structured output.

### 2. The Hardest Test: Stratum Byte Order
Only 4 models found the byte order/endianness bug in the Stratum protocol test. This is the strongest differentiator between 95.5% and 100% quality models.

### 3. Nemotron-3-Nano-30B: Best Overall
Mamba+MoE hybrid architecture achieves 100% quality AND fastest single-request speed (205 t/s). At 59GB BF16, it uses ~7.4GB/GPU at TP=8. No reasoning mode but the raw quality and speed are unmatched.

### 4. Magistral-2506 AWQ: Throughput King
At only 14GB (24B params), it achieves 1831 t/s peak - highest of any model. With only ~1.75GB/GPU for weights, it has massive KV cache headroom.

### 5. fp8 KV Cache Doubles Capacity
After patching two vLLM bugs (flashinfer positional args + compressed-tensors false rejection), fp8 KV cache works for all quant formats:
- Devstral-2-123B: 16K -> 32K context
- Seed-OSS-36B: 32K -> 64K context
- Magistral: 1.25M tokens, 38x concurrency

### 6. SGLang vs vLLM: vLLM Wins
Tested 3 models on SGLang - all performed worse than vLLM (Nemotron -47% peak, Qwen3-Coder quality collapsed to 54.5%). Exception: GLM-4.7-Flash works ONLY on SGLang.

### 7. Pipeline Parallelism Not Beneficial
EXAONE TP=2+PP=4 (8 GPUs): quality dropped 18pp, throughput dropped 26%. Only gained context length. Pipeline latency overhead outweighs memory savings.

### 8. Nanbeige4.1-3B: 3B Reasoning Model Punches Above Its Weight
At only 3B parameters (~6GB BF16), Nanbeige4.1-3B scores 77.3% - matching EXAONE-4.0-32B (10x larger). Uses `<think>` reasoning tags, 131K context, and achieves 187 t/s single-request. Fits on a single 16GB GPU. Best quality-per-parameter ratio in the benchmark.

### 9. Custom Quantization Can Unlock TP
EXAONE-4.0-32B was stuck at TP=2 with AWQ/GPTQ g128 (Marlin requires `size_k % 128 == 0`, but 27392/8=3424, 3424%128≠0). Custom GPTQ with `group_size=32, desc_act=False` + `--dtype float16` enables Marlin for aligned layers and Exllama fallback for misaligned ones. Result: TP=8, 110 t/s single (+67%), quality preserved at 95.5%.

---

## TP Compatibility Reference

| Model | Q Heads | KV Heads | Max TP | Type | Notes |
|-------|---------|----------|--------|------|-------|
| Devstral-2-123B | 96 | 8 | 8 | Dense | 123B, tight VRAM |
| Magistral-Small | 32 | 8 | 8 | Dense | 24B, both 2506/2509 |
| Qwen3-32B | 64 | 8 | 8 | Dense | Reasoning |
| Seed-OSS-36B | 80 | 8 | 8 | Dense | Reasoning |
| Nemotron-3-Nano-30B | 32 | 8 | 8 | Mamba+MoE | --trust-remote-code |
| Devstral-Small-2-24B | 32 | 8 | 8 | Dense | Mistral3 |
| GPT-OSS-20B | 64 | 8 | 8 | MoE | enforce-eager only |
| Qwen3-30B-A3B | 32 | 4 | 4 | MoE | 128 experts, 8 active |
| Qwen3-Coder-30B-A3B | 32 | 4 | 4 | MoE | 128 experts, 8 active |
| GLM-4.7-Flash | 20 | 20 | 4 | MoE | SGLang only |
| Nanbeige4.1-3B | 20 | 4 | 4 | Dense | 3B reasoning, fits single GPU |
| EXAONE-4.0-32B | 40 | 8 | 8 | Dense | Custom GPTQ g32 + float16 for TP=8 |

---

## Recommended Configurations

### Daily Coding (quality-first, C=1-2)
```bash
# Qwen3-Coder-30B-A3B x2 replicas (100% quality, 184 t/s each)
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 --max-model-len 32768 --port 8000

CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 --max-model-len 32768 --port 8001
```

### Long Context Reasoning
```bash
# Magistral-Small-2509 BF16 (95.5%, 131K context, [THINK] reasoning)
vllm serve mistralai/Magistral-Small-2509 \
  --tensor-parallel-size 8 --gpu-memory-utilization 0.90 \
  --max-num-seqs 48 --tokenizer-mode mistral --config-format mistral \
  --load-format mistral --reasoning-parser mistral --port 8000
```

### High-Throughput Batch
```bash
# Magistral-Small-2506 AWQ (1831 t/s peak, 95.5% quality)
vllm serve abhishekchohan/Magistral-Small-2506-AWQ \
  --tensor-parallel-size 8 --gpu-memory-utilization 0.90 \
  --max-num-seqs 48 --max-model-len 32768 --port 8000
```

### Maximum Quality
```bash
# Devstral-2-123B with fp8 KV (100% quality, 32K context)
vllm serve cyankiwi/Devstral-2-123B-Instruct-2512-AWQ-4bit \
  --tensor-parallel-size 8 --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 --max-model-len 32768 \
  --kv-cache-dtype fp8_e5m2 --port 8000
```

---

## vLLM Patches Required

Two bugs in vLLM 0.14 must be patched for fp8 KV cache to work:

### Patch 1: FlashInfer positional arg bug
**File**: `vllm/v1/attention/backends/flashinfer.py` line 1590
**Issue**: flashinfer 0.6.1 added `o_data_type` parameter to `plan()`, shifting all positional args
**Fix**: Change positional arguments to keyword arguments in `fast_plan_decode`

### Patch 2: Compressed-tensors false fp8 KV rejection
**File**: `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py` line 179
**Issue**: `get_quant_method()` unconditionally returns `CompressedTensorsKVCacheMethod` for all Attention layers, even when `kv_cache_scheme` is null
**Fix**:
```python
# Before (broken):
if isinstance(layer, Attention):
    return CompressedTensorsKVCacheMethod(self)

# After (fixed):
if isinstance(layer, Attention):
    if self.kv_cache_scheme is not None:
        return CompressedTensorsKVCacheMethod(self)
    return None
```

---

## Quick Start

```bash
# Install
pip install pyyaml requests

# Run against a model server
python compare_models.py --model <model-id> --api-url http://localhost:8000/v1

# With parallel throughput test
python compare_models.py --model <model-id> --api-url http://localhost:8000/v1 --parallel

# With server config metadata
python compare_models.py --model <model-id> --api-url http://localhost:8000/v1 \
  --parallel --tp 8 --framework vllm --quant-method awq \
  --output ./results-8xA4000
```

## File Structure

```
llm-bench/
├── compare_models.py              # Main benchmark harness (v2.1)
├── parallel_benchmark.py          # Standalone throughput benchmark
├── models.yaml                    # Model configurations
├── CONSOLIDATED-FINDINGS.md       # Detailed analysis and findings
├── results-8xA4000/               # 8x RTX A4000 benchmark results (22 JSON files)
├── results-gen4/                   # PCIe Gen4 test results
├── results/                        # Legacy single-GPU results
└── practical/                      # Individual test scripts
```

## Historical Results

### RTX 5090 (Single GPU, 32GB)

| Model | Quality | Throughput |
|-------|:-------:|-----------|
| Seed-OSS-36B AWQ | 100% (22/22) | 38.4 t/s |
| Qwen3-30B-A3B AWQ | 100% (22/22) | 31.2 t/s |
| Devstral-Small-24B | 95.5% (21/22) | 53.6 t/s |

### Tesla V100 (Single GPU, 32GB)

| Model | Quant | Quality | Throughput |
|-------|-------|:-------:|-----------|
| Seed-OSS-36B | GPTQ | 95.5% (21/22) | 48.3 t/s |
| Seed-OSS-36B | AWQ | 95.5% (21/22) | 7.0 t/s |

**V100 Finding**: GPTQ is 7x faster than AWQ on Volta (native CUDA kernels vs Triton fallback).

---

## License

MIT License

*Created: January 2026 | Updated: February 2026*
*For: Mining Pool Development & AI Model Evaluation*
