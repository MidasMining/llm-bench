# vLLM compressed-tensors fp8 KV cache patch

**Status:** STILL BROKEN upstream as of vLLM **v0.20.0** (verified 2026-04-25
against `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`
line 193-194 — unconditional return without the `kv_cache_scheme is not None`
check). Apply this patch to any vLLM install used for TurboQuant workloads
that combine compressed-tensors-quantized models with `--kv-cache-dtype fp8_e5m2`.

If/when upstream lands the fix, move this file to `archive/patches/` alongside
the flashinfer one.

## What it patches

**File:** `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`
(line ~179 in v0.14, line ~193 in v0.20.0)

**Bug:** `get_quant_method()` unconditionally returns `CompressedTensorsKVCacheMethod`
for any `Attention` layer, even when the model's `kv_cache_scheme` is null.
This causes models quantized with compressed-tensors (but without baked-in
KV-cache quantization) to reject runtime fp8 KV cache settings.

**Fix:** Check `if self.kv_cache_scheme is not None` before returning the
KV cache quant method.

### Upstream (broken)
```python
if isinstance(layer, Attention):
    return CompressedTensorsKVCacheMethod(self)
```

### Patched
```python
if isinstance(layer, Attention):
    if self.kv_cache_scheme is not None:
        return CompressedTensorsKVCacheMethod(self)
    return None
```

## How to apply

`compressed_tensors_v0.14_patched.py` is the full patched file from vLLM 0.14.
For other vLLM versions, apply the two-line change shown above by hand — the
surrounding code has only drifted cosmetically across releases.

```bash
# Example for vLLM 0.14 install at /home/llm/hf-env/
cp compressed_tensors_v0.14_patched.py \
   /home/llm/hf-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py
```

## Impact (why TurboQuant needs this)

With the patch applied, fp8 KV cache works on all compressed-tensors-quantized
models — which unlocks dramatically more KV headroom for long-context evals:

- **Devstral-2-123B AWQ**: 16K → 32K context, 16 → 32 max-seqs, quality preserved at 100%
- **Seed-OSS-36B AWQ**: 32K → 64K context, 730K token budget, 11x concurrency at 64K
- **Magistral-2506 AWQ**: 11.97 GiB KV/GPU, 1.25M tokens, 38x concurrency at 32K

Without the patch, these models fall back to bf16 KV cache and lose roughly half
their effective context budget at the same VRAM.
