# Model Benchmark Results — 8x RTX A4000 (128GB VRAM)
Last updated: 2026-04-09

## Hardware
- 8x NVIDIA RTX A4000 (16GB each, 128GB total)
- AMD EPYC 7532, PCIe Gen4
- 503GB system RAM
- vLLM PR #38479 fork (nightly ~0.19-dev) with TurboQuant patches

## Benchmarks
- **Standard**: 14 practical debugging checks (ZMQ, PPLNS, Payment, Stratum) — tests general reasoning
- **BWA-MEM2**: 30-point domain knowledge test (bioinformatics, Nextflow, systems engineering) — tests domain expertise and evidence-based reasoning
- **Calibration**: Grok-class = 25-30/30 BWA-MEM2, Trinity-Large (398B hosted) ≈ 30/30

---

## Results Summary

### Tier 1: Production Candidates

| Model | Params (Total/Active) | Quant | TP | Standard (14) | BWA-MEM2 (30) | Speed (t/s) | KV Cache | Notes |
|-------|----------------------|-------|-----|---------------|---------------|-------------|----------|-------|
| **Cascade-2 + TQ** | 30B / 3B MoE+Mamba | BF16 | 8 | **100%** | ~8/30 est. | **221** | 878K (TQ 2x) | Production. Fastest. TQ lossless. |
| **MiniMax M2.5-REAP-139B** | 139B / 10B MoE | AWQ INT4 | 8 | **100%** | **24/30 (80%)** | 119 | 139K | Best domain. TQ breaks it. |
| **Nemotron-3-Super-120B** | 120B / 12B MoE+Mamba | AWQ INT4 | 8+EP | **100%** | **23/30 (77%)** | 100 | 142K | Close second domain. 32K ctx now works. |
| **GLM-4.7-Flash** | 30B / 3.6B MoE | AWQ INT4 | 4 | **100%** | **21/30 (70%)** | 69 | — | Good domain. TP=4 only (20 heads). |

### Tier 2: Tested, Lower Domain Scores

| Model | Params (Total/Active) | Quant | TP | Standard (14) | BWA-MEM2 (30) | Speed (t/s) | Notes |
|-------|----------------------|-------|-----|---------------|---------------|-------------|-------|
| Qwen3.5-122B-A10B | 122B / 10B MoE | GPTQ INT4 | 8 | **100%** | 15/30 (50%) | 117 | Got retry isolation right |
| Qwen3.5-35B-A3B | 35B / 3B MoE | BF16 | 8 | **100%** | 12/30 (40%) | 168 | Mentioned pipe truncation |
| Qwen3.5-27B | 27B dense | BF16 | 4-8 | **100%** | 8/30 (27%) | 58 | Worst domain knowledge |

### V100 (Single GPU, 32GB) — vLLM-tq + 1Cat TurboMind SM70 GEMM port

| Model | Params | Quant | Standard (21) | BWA-MEM2 (30) | Speed (t/s) | KV | Notes |
|-------|--------|-------|---------------|---------------|-------------|-----|-------|
| **Qwen3.6-35B-A3B-AWQ** | 35B / 3B MoE | AWQ INT4 | **22/21 (105%)** | **18/30 (60%)** | 49.4 @ 14K | tq-t3nc (155K) | V100+TQ+TurboMind GEMM. Beats 35B-class band (+6) and Qwen3.5-122B (+3 BWA). |

**V100 stack notes:**
- Hardware: 1× Tesla V100 32GB (SM70 / Volta)
- Attention: TurboQuant tq-t3nc (3-bit MSE keys + 3-bit values + norm correction, ~5× KV compression, 155K context)
- Linear: 1Cat TurboMind SM70 m8n8k4 WMMA GEMM (replaces AWQ-Marlin which requires SM75+)
- Decode kernel optimization: D-light autotune of TQ Triton stage1 (num_warps 4→2, +1-2% e2e)
- Optional: Path B-scratch flash decode via `VLLM_TQ_FLASH_DECODE=1` (+4% at 14K, -5% at 4K — workload-dependent switch)
- Standard bench needs `enable_thinking=False` or `max_tokens >= 8000` for reasoning models; default 4096 truncates Qwen3.6 mid-think.

### Tier 3: From Original Benchmark (Feb 2026, vLLM 0.14)

| Model | Params | Quant | Standard (22) | Speed (t/s) | Notes |
|-------|--------|-------|---------------|-------------|-------|
| Nemotron-Cascade-2-30B | 30B / 3B | BF16 | **100% (22/22)** | 249 | Current production model |
| Nemotron-3-Nano-30B | 30B / 3B | BF16 | **100% (22/22)** | 262 | Previous production, same arch |
| Devstral-2-123B | 123B dense | AWQ | **100% (22/22)** | 46/300 | Best quality (5/5 Stratum) |
| Qwen3-Coder-30B-A3B | 30B / 3B | AWQ | **100% (22/22)** | 184/1025 | Best code model |
| GLM-4.7-Flash | 30B / 3.6B | AWQ | **100% (22/22)** | 101/566 | Best long context (65K) |
| Magistral-Small-2506 | ~30B | AWQ | 95.5% | 156/1831 | Best peak throughput |
| QwQ-32B | 32B | AWQ | 95.5% | 102/733 | Reasoning, <think> tags |
| GLM-4.5-Air | 30B / 3B | AWQ FP16Mix | 95.5% | 87/724 | 5/5 Stratum |

### Failed to Load

| Model | Params | Issue |
|-------|--------|-------|
| GPT-OSS-120B | 117B / 5B MoE | MXFP4 OOM — needs Hopper/Blackwell tensor cores |
| Cohere Command A Reasoning | 111B dense | Missing chat template (gated model, needs HF auth) |
| Llama 4 Scout | 109B / 17B MoE | AWQ weight key mismatch (shared_expert.activation_fn.scales) |

---

## TurboQuant Compatibility

| Model | tq3 + 2-bit V | tq3 + 4-bit V | tq3 + FP8 V | tq4 + FP8 V | Best TQ Config |
|-------|--------------|--------------|-------------|-------------|---------------|
| **Cascade-2** | 71.4% | 85.7% | **100%** | — | tq3 + FP8 values (2x compression) |
| **MiniMax M2.5-REAP** | — | — | 50% | 28.6% | **None — TQ breaks this model** |
| Super-120B | — | — | — | — | Not tested (only 8 attn layers, minimal benefit) |

### Key TQ Findings
- FP8 values = lossless on Cascade-2; value precision is the bottleneck
- MiniMax is incompatible with TQ regardless of key/value bit width — architecture mismatch
- TQ is model-dependent: works on Nemotron hybrid arch, fails on MiniMax MoE
- CUDA TQ decode overhead is ~11% (fundamental, not optimizable)
- head_dim=128 works with both turbo3 and turbo4

---

## BWA-MEM2 Domain Test — Key Patterns

### What separates good from bad scores:

| Dimension | Models that got it right | Models that got it wrong |
|-----------|------------------------|------------------------|
| Root cause (pipe corruption) | MiniMax (2/3), Super (2/3) | All Qwen3.5 (0-1/3), GLM (1/3) |
| Retry is correct fix | GLM, MiniMax, Super | All Qwen3.5 (said "don't retry") |
| Nextflow retry isolation | Qwen3.5-122B, GLM, MiniMax, Super | Qwen3.5-27B, 35B |
| No fabricated references | MiniMax, Super, GLM | Qwen3.5-27B (confused BWA tools) |
| Scope to BWAMEM2_MEM | All models | — |

### Correlation: Model size vs domain score
| Size range | Historical band | Notable outlier |
|-----------|-----------------|-----------------|
| 27-35B | 8-12/30 | **Qwen3.6-35B-A3B-AWQ: 18/30** (V100, this update) |
| 120-139B | 21-24/30 | — |
| 398B (Trinity, hosted) | ~30/30 | — |

Qwen3.6-35B-A3B raises the 35B-class ceiling by +6. Improvements come from systems-reasoning dimensions (retry strategy, scope, isolation) rather than the headline domain identification (AVX-512 high-core pipe corruption — still missed).

---

## Production Configurations

### Current Production: Cascade-2 + TurboQuant
```bash
# /home/llm/run-tq3-production.sh
export TQ_VALUE_BITS=8
vllm serve /home/llm/models/Nemotron-Cascade-2-30B-A3B \
  --tensor-parallel-size 8 --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 --max-model-len 131072 --port 8000 \
  --trust-remote-code --kv-cache-dtype tq3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin .../nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3
```
- 221 t/s, 878K token KV cache, 131K context, 100% standard quality
- Best for: Dowsing (structured extraction), fast chat, tool calling

### Domain Expert: MiniMax M2.5-REAP-139B (no TQ)
```bash
vllm serve /home/llm/models/MiniMax-M2.5-REAP-139B-AWQ \
  --tensor-parallel-size 8 --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 --max-model-len 32768 --port 8001 \
  --trust-remote-code
```
- 119 t/s, 139K token KV cache, 32K context, 24/30 domain quality
- Best for: Code review, bioinformatics analysis, domain-expert tasks

### Alternative Domain: Super-120B
```bash
vllm serve /home/llm/models/Nemotron-3-Super-120B-AWQ-4bit \
  --tensor-parallel-size 8 --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 --max-model-len 32768 --port 8001 \
  --trust-remote-code --enable-expert-parallel \
  --mamba-ssm-cache-dtype float16
```
- 100 t/s, 142K token KV cache, 32K context, 23/30 domain quality
- Same Nemotron family as production model
