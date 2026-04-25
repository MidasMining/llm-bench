# Tool-Call Reliability Benchmark

> **!!! CONFIGURE BEFORE USE !!!** This benchmark hands the model an SSH-capable
> bash tool pointed at a real target host and lets it run arbitrary commands
> there (including container stop/rm/run). You must supply the target host,
> SSH user, and SSH password yourself — no defaults are baked into the script.
> Provide via `--target-host` / `--target-user` / `--target-password`, or via
> the env vars `BENCH_TARGET_HOST` / `BENCH_TARGET_USER` / `BENCH_TARGET_PASSWORD`.
> Point this at a **disposable test rig only**, never production.

Measures whether a vLLM-served model can reliably emit a sequence of well-formed tool calls to complete a multi-step SSH diagnostic + remediation task on a remote host.

The canonical scenario is the Nosana `queryAaaa ETIMEOUT` fix on a HiveOS GPU rig (5 steps: confirm error → check IPv6 → inspect env → recreate container with `NODE_OPTIONS=--dns-result-order=ipv4first` → verify). Other prompts can be plugged in via `--prompt-file`.

## How it works

1. Sends the prompt to the chat completions endpoint with a single `bash` tool defined.
2. Each turn, parses the model's tool calls, **actually executes** them via `subprocess`, and feeds stdout+stderr back as the tool result.
3. Loops up to `--max-turns` (default 15) or until the model stops emitting tool calls.
4. Post-check: SSHs to the target and inspects whether `NODE_OPTIONS` ended up set on the `nosana-node` container.
5. Scores 5 canonical steps based on tool-call content and writes a JSON result.

## Requirements

- vLLM endpoint serving the model with `--enable-auto-tool-choice` and a tool parser appropriate for the model (e.g. `qwen3_xml` for Nemotron, `minimax_m2` for MiniMax).
- `sshpass` and `ssh` on the local host (for executing the model's SSH commands and the post-check).
- `requests` Python package.
- A reachable test target. **Do not point this at a production rig** — the model will stop, remove, and recreate containers. Use the disposable VM fixture (see `../projects/turboquant/nemo-tq-benchmark.md` for the snapshot-revert workflow).

## Usage

```bash
python3 tool_call_benchmark.py \
  --model <MODEL_ID_OR_PATH> \
  --api-url http://localhost:8001/v1/chat/completions \
  --target-host <TARGET_HOST> \
  --target-user <TARGET_USER> --target-password <TARGET_PASSWORD> \
  --variant <run-label> \
  --output-dir ./results
```

Or, to keep credentials out of shell history:

```bash
export BENCH_TARGET_HOST=<TARGET_HOST>
export BENCH_TARGET_USER=<TARGET_USER>
export BENCH_TARGET_PASSWORD=<TARGET_PASSWORD>
python3 tool_call_benchmark.py --model <MODEL_ID_OR_PATH> --variant <run-label>
```

Options:

| Flag | Default | Notes |
|---|---|---|
| `--model` | required | Model id as exposed by the vLLM endpoint |
| `--api-url` | `http://localhost:8000/v1/chat/completions` | OpenAI-compatible chat completions URL |
| `--target-host` | env `BENCH_TARGET_HOST` (required) | IP/hostname the model should SSH into |
| `--target-user` | env `BENCH_TARGET_USER` (required) | SSH user on the target |
| `--target-password` | env `BENCH_TARGET_PASSWORD` (required) | Pass empty string for key-based auth |
| `--variant` | required | Becomes part of the output filename |
| `--prompt-file` | bundled `prompts/nosana_aaaa_fix.txt` | `{TARGET_HOST}` is substituted in |
| `--output-dir` | `.` | JSON result destination |
| `--max-turns` | `15` | Hard cap on conversation turns |
| `--temperature` / `--top-p` / `--max-tokens` | `0.6` / `0.95` / `2048` | Generation params |
| `--request-timeout` | `120` | Per-request HTTP timeout (seconds) |

## Output

A JSON file `tool-bench-<variant>.json` containing:
- `tool_calls` — list of `{turn, command, output}` for each executed tool call
- `steps_completed` (0-5) and `steps_detail` (which canonical steps were attempted)
- `malformed_xml` — count of tool calls whose argument JSON failed to parse
- `final_check.node_options_present` — whether the fix actually landed
- `model_response_texts` — any non-tool text the model emitted
- `turns`, `start_time`, `end_time`

## Reset between runs

The benchmark mutates the target rig (it removes and recreates a container). Before re-running on the same target, restore it to the broken baseline. The recommended workflow is `virsh snapshot-revert` on the test VM — see `../projects/turboquant/nemo-tq-benchmark.md` for the full setup.

## Caveats

- This benchmark measures **end-to-end task completion under tool use**, not raw token throughput or quality on static prompts. It exercises long-context instruction adherence, tool argument well-formedness, and recovery from error output.
- The `score_steps` heuristic is keyword-based — a model that solves the task with unconventional commands (e.g. uses `nsenter` instead of `docker exec`) may score lower than its actual completion warrants. Treat the JSON as evidence to read, not just sum.
- The post-check only verifies `NODE_OPTIONS`. If a model claims success without applying the fix, that lie surfaces here.
- Variance run-to-run is real. See `../projects/turboquant/nemo-tq-benchmark.md` for an example where the same model on the same prompt produced different tool-routing shapes (`n=1` is not enough).
