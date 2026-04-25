# Archived: FlashInfer fp8 positional arg fix

**Status:** FIXED upstream in vLLM **v0.18.0** and still fixed at **v0.20.0**
(verified 2026-04-25 against `vllm/v1/attention/backends/flashinfer.py`).
This patch is no longer needed on any current vLLM release.

**Kept here for reproducibility** of TurboQuant benchmarks that ran on vLLM 0.14
before the upstream fix landed.

## What it patched

**File:** `vllm/v1/attention/backends/flashinfer.py` (line ~1590 in v0.14)

**Bug:** `self.plan()` was called with positional args. FlashInfer 0.6.1 added
an `o_data_type` parameter, shifting all arguments by 1, which caused fp8 KV
cache to fail silently on every quantized model.

**Fix:** Changed positional args to keyword args in the `self.plan()` call,
including `o_data_type=o_data_type` explicitly.

### v0.14 ORIGINAL (broken)
```python
self.plan(
    indptr_cpu,
    indices,
    last_page_len_cpu,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    pos_encoding_mode,
    ...
)
```

### v0.14 PATCHED
```python
self.plan(
    indptr=indptr_cpu,
    indices=indices,
    last_page_len=last_page_len_cpu,
    num_qo_heads=num_qo_heads,
    num_kv_heads=num_kv_heads,
    head_dim=head_dim,
    page_size=page_size,
    pos_encoding_mode=pos_encoding_mode,
    window_left=window_left,
    logits_soft_cap=logits_soft_cap,
    q_data_type=q_data_type,
    kv_data_type=kv_data_type,
    data_type=data_type,
    sm_scale=sm_scale,
    rope_scale=rope_scale,
    rope_theta=rope_theta,
    non_blocking=non_blocking,
    block_tables=None,
    seq_lens=None,
)
```

### v0.18.0+ upstream (already fixed, no patch needed)
Line 1687 in v0.18.0, line 1772 in v0.20.0: same keyword-arg pattern with
`o_data_type=o_data_type` included by upstream.
