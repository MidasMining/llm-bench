#!/usr/bin/env python3
"""Single-stream decode-rate benchmark with streaming SSE parsing.

Isolates decode rate from prefill cost by parsing chat-completion SSE events:
  TTFT = time to first content delta
  decode_rate = (completion_tokens - 1) / (last_event_time - first_token_time)

Sweeps a range of input context sizes by repeating a deterministic filler so
prefill scales while decode workload is fixed (max_tokens=500). Useful for
comparing attention-kernel scaling vs context length without conflating prefill.

Usage:
  python decode_rate_bench.py \\
      --api-url http://localhost:8100/v1 --model Qwen3.6-V100 \\
      --tag "FLASH_ATTN_V100 + graphs" \\
      --targets 500 2000 4000 6000

Defaults match a typical V100 vLLM serve.
"""
import argparse
import json
import statistics
import time

import requests

SENTENCE = "The quick brown fox jumps over the lazy dog. "  # ~10 tokens

PROMPT_TEMPLATE = (
    "Read the following text and produce a detailed analytical commentary "
    "in at least 500 words about its repetitive structure, linguistic "
    "features, and potential uses in NLP benchmarking.\n\nTEXT:\n{body}"
    "\n\nAnalysis:"
)


def make_prompt(target_input_tokens: int) -> str:
    body = SENTENCE * (target_input_tokens // 10)
    return PROMPT_TEMPLATE.format(body=body)


def stream_call(api_url: str, model: str, prompt: str, max_tokens: int, timeout: int = 900):
    """Returns dict with prompt_tokens, completion_tokens, ttft, decode_sec, total_sec."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Disable reasoning so we can measure raw decode rate against a fixed
        # output budget; reasoning models otherwise emit a variable number of
        # think-tokens before answering.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    ttft = None
    last_event_t = None
    pt = ct = 0

    with requests.post(
        f"{api_url}/chat/completions",
        json=payload,
        stream=True,
        timeout=timeout,
        headers={"Accept": "text/event-stream"},
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if not raw.startswith("data: "):
                continue
            payload_str = raw[6:]
            if payload_str == "[DONE]":
                break
            try:
                ev = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                pt = ev["usage"].get("prompt_tokens", pt)
                ct = ev["usage"].get("completion_tokens", ct)
            choices = ev.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                # Reasoning models emit `reasoning` deltas as well as `content`.
                # Count any non-empty decode emission as a "decode event".
                emitted = (delta.get("content")
                           or delta.get("reasoning")
                           or delta.get("reasoning_content"))
                if emitted and ttft is None:
                    ttft = time.time() - t0
                if emitted is not None:
                    last_event_t = time.time()

    total = time.time() - t0
    if ttft is None:
        ttft = total
    if last_event_t is None:
        decode_sec = 0.0
    else:
        decode_sec = max(last_event_t - (t0 + ttft), 1e-3)
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "ttft": ttft,
        "decode_sec": decode_sec,
        "total_sec": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default="http://localhost:8100/v1")
    ap.add_argument("--model", default="Qwen3.6-V100")
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--targets", type=int, nargs="+", default=[500, 2000, 4000, 6000])
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    print(f"=== {args.tag} ===")
    print(f"api={args.api_url} model={args.model}\n")

    # warmup
    w = stream_call(args.api_url, args.model, "Hello, brief response please.", 20)
    print(f"warmup: pt={w['prompt_tokens']} ct={w['completion_tokens']} "
          f"ttft={w['ttft']:.2f}s decode={w['decode_sec']:.2f}s total={w['total_sec']:.2f}s\n")

    print(f"{'target':>7} {'prompt':>7} {'comp':>5} "
          f"{'ttft':>7} {'decode':>9} {'rate':>10}")

    for target in args.targets:
        prompt = make_prompt(target)
        rates = []
        last_pt = ct = 0
        ttft_avg = decode_avg = 0.0
        for i in range(args.runs):
            r = stream_call(args.api_url, args.model, prompt, args.max_tokens)
            last_pt = r["prompt_tokens"]
            ct = r["completion_tokens"]
            rate = (ct - 1) / r["decode_sec"] if r["decode_sec"] > 0 and ct > 1 else 0.0
            rates.append(rate)
            ttft_avg += r["ttft"]
            decode_avg += r["decode_sec"]
        ttft_avg /= args.runs
        decode_avg /= args.runs
        avg_rate = statistics.mean(rates)
        print(f"{target:>7} {last_pt:>7} {ct:>5} "
              f"{ttft_avg:>5.2f}s {decode_avg:>7.2f}s "
              f"{avg_rate:>8.1f} t/s")


if __name__ == "__main__":
    main()
