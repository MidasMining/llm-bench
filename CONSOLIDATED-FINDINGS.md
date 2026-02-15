# Consolidated LLM Benchmark Findings
**Updated: February 15, 2026**
**Platform: 8x RTX A4000 (16GB each, 128GB total), EPYC 7532 (Rome), PCIe Gen4**

---

## Full Model Benchmark (Feb 12-14, 2026)

### Results Summary (8x RTX A4000)

| Model | Quant | TP | Reasoning | Quality | Single t/s | Peak t/s | Context |
|-------|-------|-----|-----------|---------|-----------|----------|---------|
| **Devstral-2-123B** | AWQ | 8 | No | **100%** (22/22) | 41 | 300 @ C=32 | 32K† |
| **Nemotron-3-Nano-30B** | BF16 | 8 | No | **100%** (22/22) | **205** | 1628 @ C=32 | 16K |
| **Qwen3-Coder-30B-A3B** | AWQ | 4 | No | **100%** (22/22) | 184 | 1025 @ C=32 | 32K |
| **GLM-4.7-Flash** | AWQ | 4 | No | **100%** (22/22) | 101 | 566 @ C=8 | 65K |
| **Magistral-Small-2509** | BF16 | 8 | Yes§ | **95.5%** (21/22) | 88 | 1071 @ C=32 | 131K |
| **Magistral-Small-2506** | AWQ | 8 | No* | **95.5%** (21/22) | 156 | **1831** @ C=32 | 32K |
| **Qwen3-32B** | AWQ | 8 | Yes | 95.5% (21/22) | 78 | 1013 @ C=32 | 32K |
| **EXAONE-4.0-32B** | AWQ | 2 | No‡ | 95.5% (21/22) | 66 | 748 @ C=32 | 32K |
| **Nanbeige4.1-3B** | BF16 | 4 | Yes¶ | 77.3% (17/22) | 187 | 1239 @ C=64 | 131K |
| Qwen3-30B-A3B | AWQ | 4 | No | 95.5% (21/22) | 178 | 1575 @ C=32 | 32K |
| Devstral-Small-2-24B | AWQ | 8 | No | 95.5% (21/22) | 148 | 1452 @ C=32 | 32K |
| **Seed-OSS-36B** | AWQ | 8 | Yes | 90.9% (20/22) | 88 | 1163 @ C=32 | 32K |
| Qwen3-30B-A3B-Thinking | AWQ | 4 | Yes | 81.8% (18/22) | 160 | 1031 @ C=32 | 32K |
| DS-R1-Distill-Qwen-32B | AWQ | 8 | Yes | 54.5% (12/22) | 78 | 992 @ C=32 | 32K |
| DS-R1-Distill-Llama-70B | AWQ | 8 | Yes | 45.5% (10/22) | 57 | 540 @ C=32 | 16K |
| GPT-OSS-20B | MXFP4 | 8 | No | 40.9%** | 52 | 933 @ C=16 | 8K |

§ Magistral-Small-2509 has proper [THINK]/[/THINK] special tokens, requires --tokenizer-mode mistral --config-format mistral --load-format mistral --reasoning-parser mistral
\* Magistral-2506 has reasoning capability but community AWQ quant dropped `begin_think` control tokens
\** Quality unreliable - reasoning parser strips content to None on most tests
† Devstral-2-123B: 32K context / 32 max-seqs with fp8 KV cache (was 16K/16 without). Requires compressed-tensors patch.
‡ EXAONE has `<think>` tokens but not in chat template; limited to TP=2 (AWQ intermediate_size alignment)
¶ Nanbeige4.1-3B uses `<think>` tags (Qwen3-style reasoning), only 3B params, LlamaForCausalLM architecture

### Category Winners

| Category | Model | Why |
|----------|-------|-----|
| **Best Quality** | Devstral-2-123B | 100% quality (22/22 including 5/5 Stratum!), only 100% on byte order |
| **Best Overall** | Nemotron-3-Nano-30B | 100% quality, 205 t/s single, 1628 t/s peak |
| **Best Code Model** | Qwen3-Coder-30B-A3B | 100% quality, purpose-built for code |
| **Best Peak Throughput** | Magistral-Small-2506 | 1831 t/s peak, 95.5% quality, only 14GB |
| **Best Long Context** | GLM-4.7-Flash | 65K context, 100% quality |
| **Best Reasoning** | Seed-OSS-36B | 90.9% quality with `<seed:think>` reasoning, 60 t/s with thinking |
| **Best Reasoning (Quality)** | Qwen3-32B | 95.5% quality with `<think>` mode, but 78 t/s with thinking |

### Failed / Marginal Models

| Model | Quant | Size | Failure | Error |
|-------|-------|------|---------|-------|
| GPT-OSS-120B | FP16*** | ~33GB | OOM | Repo is actually FP16 (no quantization_config), 15.6GB/GPU |
| GLM-4.5-Air | AWQ | ~59GB | Kernel error | Marlin: `size_n=2736 not divisible by tile_n=64` |
| Qwen3-Next-80B-A3B | AWQ | ~49GB | OOM | 2 KV heads (max TP=2), 24.5GB/GPU exceeds 16GB |
| Qwen3-Coder-Next | AWQ | ~45GB | Won't fit | 2 KV heads (max TP=2), needs vLLM 0.15+ |
| EXAONE-4.0-32B | AWQ | 18GB | TP limited | intermediate_size=27392 not AWQ-aligned at TP>2 |
| Qwen3-30B-A3B (stelterlab) | AWQ | ~17GB | CUDA error | Illegal memory access during CUDA graph compile |
| Qwen3-480B-Coder | AWQ | ~236GB | Not attempted | Exceeds 128GB total VRAM |

---

## Key Findings

### 1. Reasoning Models Trade Speed for Thinking
- Models with reasoning enabled (Qwen3-32B, Seed-OSS, Qwen3-30B-Think) generate invisible thinking tokens
- Seed-OSS-36B: 88 t/s without thinking → ~60 t/s effective with `<seed:think>` reasoning
- Qwen3-32B: 78 t/s total throughput includes thinking overhead
- **Quality impact varies**: Qwen3-32B maintained 95.5%, but Qwen3-30B-Thinking dropped to 81.8%

### 2. DeepSeek-R1 Distillation Fails Practical Tests
- Both R1-Distill-Qwen-32B (54.5%) and R1-Distill-Llama-70B (45.5%) scored poorly
- Models over-reason and produce verbose outputs that don't match expected structured formats
- Good for open-ended discussion but bad for structured debugging tasks
- **Not recommended for coding workflows**

### 3. Magistral-Small-2506 is a Hidden Gem
- Only 14GB AWQ (24B params) but achieves **1831 t/s peak** (highest of any model!)
- 95.5% quality matches much larger models
- At TP=8 with only ~1.75GB/GPU for weights, has massive KV cache headroom
- Could potentially serve 128K+ context at low concurrency
- Reasoning capability exists but broken in community AWQ quant

### 4. Nemotron-3-Nano-30B Remains Overall Champion
- **100% quality + fastest single-request (205 t/s)** remains unbeaten
- Mamba+MoE hybrid architecture is uniquely efficient
- BF16 at 59GB limits context to 16K at TP=8 with max-num-seqs 48
- No reasoning mode available

### 5. Non-Reasoning Models Consistently Score Higher
- Top 3 by quality are all non-reasoning: Nemotron (100%), Qwen3-Coder (100%), GLM-4.7-Flash (100%)
- Reasoning mode introduces parsing overhead that can hurt structured test scores
- For coding tasks, a fast non-reasoning model + human reasoning may outperform model reasoning

### 6. Devstral-2-123B: Quality King, Now with fp8 KV Cache
- Only model to score **5/5 on Stratum Protocol** (byte order/endianness test)
- At 123B dense parameters (78GB AWQ), it's the largest model that fits on 8x A4000
- **With fp8 KV cache** (after compressed-tensors patch): 32K context, 32 max-seqs, 300 t/s peak
- **Without fp8 KV**: 16K context, 16 max-seqs, 282 t/s peak
- Quality preserved at 100% (22/22) with fp8 KV - no accuracy loss
- Demonstrates that bigger models DO score better on the hardest tests

### 7. EXAONE-4.0-32B: AWQ Alignment Limits TP, PP Doesn't Help
- 95.5% quality at 32B dense, has `<think>` tokens but not in default chat template
- AWQ quantized with intermediate_size=27392, which doesn't divide evenly at TP>2
- Stuck at TP=2 → only uses 2 of 8 GPUs → 66 t/s single, 748 t/s peak
- **PP=4 tested (TP=2 PP=4 = 8 GPUs)**: quality dropped to 77.3%, peak dropped to 556 t/s (-26%), only benefit was 131K context (vs 32K). Pipeline latency overhead hurts both speed and quality. Not recommended.
- **Official GPTQ also fails at TP=8**: Same alignment issue (Marlin: 3424%128≠0, Exllama: blocked by desc_act+TP, basic GPTQ: also alignment constrained). Fix requires custom GPTQ with `group_size=32, desc_act=False`

### 9. fp8 KV Cache Now Works on vLLM (After Two Patches)
- **Patch 1**: Fixed flashinfer `non_blocking=None` bug (positional arg mismatch with o_data_type)
  - File: `vllm/v1/attention/backends/flashinfer.py` line 1590 - changed positional to keyword args
- **Patch 2**: Fixed compressed-tensors false rejection of fp8 KV cache
  - File: `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py` line 179
  - Bug: `get_quant_method()` unconditionally returned `CompressedTensorsKVCacheMethod` for all Attention layers, even when `kv_cache_scheme` is null
  - Fix: Check `if self.kv_cache_scheme is not None` before returning KV cache quant method
- Magistral at TP=8: 11.97 GiB KV per GPU, 1.25M tokens, 38x concurrency at 32K
- Seed-OSS-36B: unlocked **64K context** (up from 32K) with fp8 KV, 730K tokens, 11x @ 64K
- **Devstral-2-123B**: unlocked **32K context** (up from 16K) with fp8 KV, 227K tokens, quality preserved at 100%
- All AWQ/GPTQ/compressed-tensors quants now work with fp8 KV cache

### 10. SGLang vs vLLM: vLLM Wins for Most Models
- Tested 3 models on SGLang (Devstral-2-123B, Nemotron-3-Nano-30B, Qwen3-Coder-30B-A3B)
- **Devstral-2-123B**: OOM on SGLang at both 0.85 and 0.90 mem-fraction-static (model too large for SGLang memory management)
- **Nemotron-3-Nano-30B**: Quality 100% (same), but speed 160 t/s single (vs 205 vLLM, -22%), peak 862 t/s (vs 1628 vLLM, -47%). Triton backend incompatible (Mamba hybrid), no fp8 KV available
- **Qwen3-Coder-30B-A3B**: Quality collapsed to **54.5%** (vs 100% vLLM!), speed 105 t/s (vs 184 vLLM, -43%), peak 645 t/s (vs 1025 vLLM, -37%). Quality regression likely caused by SGLang default chat sampling params (repetition_penalty=1.05, temperature=0.7) overriding benchmark settings
- **Exception**: GLM-4.7-Flash works ONLY on SGLang (GLMForCausalLM architecture) - 100% quality, 101 t/s
- **Conclusion**: Use vLLM for all models except GLM family. SGLang offers no speed or quality advantage on this hardware

### 11. Nanbeige4.1-3B: Best Quality-per-Parameter Ratio
- **77.3% quality at only 3B parameters** - matches EXAONE-4.0-32B (10x larger)
- Uses `<think>` reasoning (like Qwen3), 131K context, LlamaForCausalLM architecture
- 187 t/s single-request, 1239 t/s peak @ C=64 (on TP=4, only 4 GPUs)
- At ~6GB BF16, fits on a **single 16GB GPU** - no multi-GPU needed
- 20 attention heads limits TP to 4 max (same as GLM-4.7-Flash)
- Failed: atomicity bug (Payment), memory leak pruning (Stratum), some HiveOS wrapper checks
- **Best use case**: Edge deployment, resource-constrained environments, or high-throughput reasoning at minimal cost

### 12. The "2 KV Head Problem" Blocks Many New Models
- Qwen3-Next-80B, Qwen3-Coder-Next, MiniCPM-SALA all have only 2 KV heads
- Max TP=2, making them impractical on multi-GPU setups with 16GB/card
- This is a growing trend in newer architectures (linear attention, DeltaNet hybrids)

---

## Workflow-Optimized Configurations

Based on Midas workflow profile (C=1-2 typical, quality-first, code-focused):

### Daily Operations (Stages 1-5)
**Primary: Qwen3-Coder-30B-A3B × 2 replicas**
```bash
# Replica 1: GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 --max-model-len 32768 \
  --port 8000 --disable-log-requests

# Replica 2: GPUs 4-7
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 --max-model-len 32768 \
  --port 8001 --disable-log-requests
```
- 100% quality, 184 t/s per replica, 32K context
- Handles C=2 natively (one request per replica)

### Pool Development (Stage 6 - extended context)
**Seed-OSS-36B with reasoning + fp8 KV cache**
```bash
vllm serve QuantTrio/Seed-OSS-36B-Instruct-AWQ \
  --tensor-parallel-size 8 --gpu-memory-utilization 0.90 \
  --max-num-seqs 24 --max-model-len 65536 \
  --kv-cache-dtype fp8_e5m2 \
  --port 8000 --disable-log-requests --reasoning-parser seed_oss
```
- 90.9% quality with `<seed:think>` reasoning for architecture discussions
- Extended context (65K) enabled by fp8 KV cache (730K token budget, 11x concurrency)
- ~60 t/s effective with reasoning
- NOTE: Requires flashinfer.py patch (see Known Issues)

### High-Throughput Batch (research sprints)
**Magistral-Small-2506**
```bash
vllm serve abhishekchohan/Magistral-Small-2506-AWQ \
  --tensor-parallel-size 8 --gpu-memory-utilization 0.90 \
  --max-num-seqs 48 --max-model-len 32768 \
  --port 8000 --disable-log-requests
```
- 95.5% quality, 156 t/s single, 1831 t/s peak
- Best for batch analysis across multiple coins

---

## TP Compatibility Reference

| Model | Q Heads | KV Heads | Max TP | TP Used | Type | Notes |
|-------|---------|----------|--------|---------|------|-------|
| Devstral-2-123B | 96 | 8 | 8 | 8 | Dense | 123B params, Ministral3 arch, tight VRAM |
| Magistral-Small-2506 | 32 | 8 | 8 | 8 | Dense | 24B params, Mistral arch |
| Qwen3-32B | 64 | 8 | 8 | 8 | Dense | Reasoning mode |
| DS-R1-Distill-Qwen-32B | 40 | 8 | 8 | 8 | Dense | R1 reasoning |
| DS-R1-Distill-Llama-70B | 64 | 8 | 8 | 8 | Dense | R1 reasoning, Llama3 base |
| Seed-OSS-36B | 80 | 8 | 8 | 8 | Dense | Reasoning mode |
| Nemotron-3-Nano-30B | 32 | 8 | 8 | 8 | Mamba+MoE | --trust-remote-code |
| Devstral-Small-2-24B | 32 | 8 | 8 | 8 | Dense | Mistral3 arch |
| GPT-OSS-20B | 64 | 8 | 8 | 8 | MoE | enforce-eager only |
| Qwen3-30B-A3B | 32 | 4 | 4 | 4 | MoE | 128E/8A |
| Qwen3-Coder-30B-A3B | 32 | 4 | 4 | 4 | MoE | 128E/8A |
| Qwen3-30B-A3B-Thinking | 32 | 4 | 4 | 4 | MoE | Thinking variant |
| GLM-4.7-Flash | 20 | 20 | 4 | 4 | MoE | SGLang only |
| Nanbeige4.1-3B | 20 | 4 | 4 | 4 | Dense | 3B reasoning model, LlamaForCausalLM |
| EXAONE-4.0-32B | 40 | 8 | 2* | 2 | Dense | AWQ intermediate_size not aligned at TP>2 |

---

## Parallel Throughput Scaling (All Models)

### Non-Reasoning Models
| C | Seed-OSS | Qwen3-Coder | Qwen3-30B | Devstral-S | Nemotron | GLM-4.7 | Magistral | Devstral-2 | EXAONE | Nanbeige-3B |
|:-:|:--------:|:-----------:|:---------:|:----------:|:--------:|:-------:|:---------:|:----------:|:------:|:-----------:|
| 1 | 78 | 184 | 178 | 180 | 205 | 101 | 156 | 46 | 66 | 249 |
| 2 | 168 | 333 | 347 | 186 | 327 | 188 | 275 | 76 | 147 | 251 |
| 4 | 373 | 462 | 621 | 425 | 571 | 356 | 524 | 126 | 255 | 279 |
| 8 | 616 | 589 | 869 | 691 | 765 | **566** | 839 | 188 | 385 | 453 |
| 16 | 923 | 840 | 1208 | 1051 | 1158 | 255* | 1266 | **282** | 575 | 737 |
| 32 | **1163** | **1025** | **1575** | **1452** | **1628** | 511 | **1831** | - | **748** | 1044 |
| 64 | - | - | - | - | - | - | - | - | - | **1239** |

### Reasoning Models
| C | Qwen3-32B | Qwen3-30B-Think | DS-R1-32B | DS-R1-70B |
|:-:|:---------:|:---------------:|:---------:|:---------:|
| 1 | 78 | 160 | 78 | 57 |
| 2 | 170 | 300 | 172 | 115 |
| 4 | 331 | 438 | 333 | 230 |
| 8 | 571 | 648 | 564 | 335 |
| 16 | 825 | 841 | 799 | 432 |
| 32 | **1013** | **1031** | **992** | **540** |

\* GLM-4.7-Flash throughput drops at C=16 due to CUDA graph cliff on SGLang TP=4

---

## Reasoning Model Comparison

| Model | Quality | Reasoning Style | Thinking Speed | Discussion Quality | Structured Tasks |
|-------|---------|----------------|---------------|-------------------|-----------------|
| **Seed-OSS-36B** | 90.9% | `<seed:think>` tags | ~60 t/s | Good - methodical analysis | Good |
| **Qwen3-32B** | 95.5% | `<think>` tags | ~78 t/s total | Good | Good |
| Qwen3-30B-A3B-Think | 81.8% | `<think>` tags | ~160 t/s total | Decent | Weak on some |
| DS-R1-Distill-Qwen-32B | 54.5% | `<think>` tags | ~78 t/s total | Verbose | Poor |
| **Nanbeige4.1-3B** | 77.3% | `<think>` tags | ~187 t/s total | Decent - concise | Decent |
| DS-R1-Distill-Llama-70B | 45.5% | `<think>` tags | ~57 t/s total | Verbose | Very poor |

**Recommendation**: For reasoning/discussion, use Seed-OSS-36B (best think quality at practical speed) or Qwen3-32B (higher structured quality but slower).

---

## Known Issues & Workarounds

1. **FlashInfer 0.6.1 fp8 KV cache on vLLM - FIXED** - Root cause: positional arg mismatch in `vllm/v1/attention/backends/flashinfer.py:1590` where `self.plan()` was called positionally but flashinfer 0.6.1 added `o_data_type` parameter, shifting all args by 1. Fix: use keyword arguments. Patch applied to local install.
2. **flashinfer-cubin version mismatch** - cubin 0.6.1 won't upgrade to 0.6.3 (download hangs)
3. **GLM models need dev transformers** - Must swap `transformers 5.0.0.dev0` for SGLang
4. **Nemotron needs --trust-remote-code** - Custom Mamba+MoE architecture
5. **vLLM warmup OOM at --max-num-seqs 64** - Use 48 max for TP=8
6. **Magistral AWQ reasoning broken** - Community quant drops `begin_think` control tokens
7. **DeepSeek-R1 distilled models** - Over-reason on structured tasks, bad quality scores
8. **Qwen3-Coder has no thinking mode** - Despite Qwen3 family, Coder variant removed thinking

---

## Result Files
All raw benchmark JSON files are in `/home/llm/llm-bench/results-8xA4000/`:

### Round 1 (Feb 12-13) - Base models
- `comparison_20260212_194040.json` - Seed-OSS-36B AWQ
- `comparison_20260212_234515.json` - Qwen3-Coder-30B-A3B AWQ
- `comparison_20260212_235209.json` - Qwen3-30B-A3B AWQ
- `comparison_20260212_235841.json` - Devstral-Small-2-24B AWQ
- `comparison_20260213_011549.json` - Nemotron-3-Nano-30B BF16
- `comparison_20260213_013232.json` - GLM-4.7-Flash AWQ
- `comparison_20260213_083837.json` - GPT-OSS-20B MXFP4

### Round 2 (Feb 13-14) - Reasoning models
- `comparison_20260213_131314.json` - Qwen3-32B AWQ (reasoning)
- `comparison_20260214_002057.json` - DeepSeek-R1-Distill-Qwen-32B AWQ
- `comparison_20260214_011427.json` - Qwen3-30B-A3B-Thinking-2507 AWQ
- `comparison_20260214_013641.json` - DeepSeek-R1-Distill-Llama-70B AWQ
- `comparison_20260214_020200.json` - Magistral-Small-2506 AWQ

### Round 3 (Feb 14) - Extended models
- `comparison_20260214_144229.json` - EXAONE-4.0-32B AWQ (TP=2)
- `comparison_20260214_155110.json` - Devstral-2-123B AWQ (TP=8)

### Round 4 (Feb 14) - SGLang cross-framework comparison
- `comparison_20260214_202736.json` - Nemotron-3-Nano-30B BF16 (SGLang TP=8)
- `comparison_20260214_220116.json` - Qwen3-Coder-30B-A3B AWQ (SGLang TP=4, fp8 KV)

### Round 5 (Feb 15) - PP tests + fp8 KV compressed-tensors patch
- `comparison_20260214_222425.json` - EXAONE-4.0-32B AWQ (TP=2 PP=4, all 8 GPUs)
- `comparison_20260215_004318.json` - Devstral-2-123B AWQ (fp8 KV cache, 32K context)
- `comparison_20260215_015429.json` - Magistral-Small-2509 BF16 (mistral format, 131K context)

### Round 6 (Feb 15) - Small reasoning model
- `comparison_20260215_034513.json` - Nanbeige4.1-3B BF16 (TP=4, 131K ctx, reasoning)

---

## Previous Benchmark Results (Historical)

### Single-GPU Tests (5090, 32GB, Jan 26)
| Model | Quant | Quality Score | Throughput |
|-------|-------|:------------:|------------|
| Seed-OSS-36B | AWQ | 100% (22/22) | 38.4 t/s |
| Qwen3-30B-A3B | AWQ | 100% (22/22) | 31.2 t/s |
| Devstral-Small-24B | compressed-tensors | 95.5% (21/22) | 53.6 t/s |
| GLM-4.7-Flash | NVFP4 | 50% (11/22) | 4.4 t/s |
| GPT-OSS-20B | MXFP4 | 40.9% (9/22) | 26.0 t/s |
