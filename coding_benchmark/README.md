# Coding Reliability Benchmark

> **!!! CONFIGURE BEFORE USE !!!** This benchmark hands the model file-IO and
> shell-command tools confined to a sandbox tempdir on the **local machine**
> (no SSH, no remote host). Only obvious risk is a model deciding to run
> something heavy or long; default per-command timeout is 120s.

Measures whether a vLLM-served model can complete a multi-step coding task
end-to-end: read requirements, plan, write Python, run it, verify the
output, and report success only after self-checking. Different shape from
`tool_call_benchmark/` (which is sysadmin-style: SSH → inspect → execute).

The canonical task is BAM filtering with pysam — chosen because pysam has a
real, mechanically introspectable API surface (so confabulated method names
get caught), the task has natural multi-step structure (parse → filter →
write → sort → index), and outcomes are deterministically verifiable
against a gold reference.

## How it works

1. Generate a deterministic synthetic input BAM (~100 reads across
   chr1/chr2/chr3 with varied MAPQ and flags) and a gold output BAM by
   applying the scenario's filter directly.
2. Send the task prompt to the chat completions endpoint with four tools:
   `submit_plan`, `write_file`, `read_file`, `run_command`. The runner gates
   the file/exec tools — they return an error until `submit_plan` has been
   called. All paths are confined to the sandbox workdir.
3. Loop up to `--max-turns` (default 20). Each turn the model either emits
   tool calls or stops voluntarily.
4. After the loop, score the run on five axes (see below) and write a JSON
   result.

## Scoring axes

Each returns a score in `[0, 1]`; the result JSON contains the breakdown so
you can read evidence rather than just a number.

| Axis | What it catches | Mechanism |
|---|---|---|
| **api_correctness** | confabulated pysam method/attr names | AST-walks `solution.py`, checks every `pysam.X.Y...` chain via `getattr` against installed pysam |
| **functional** | plausible-looking but wrong output | Compares model's `output.bam` to gold by `(chrom, pos, flag, qname)`; exact-order match = 1.0, set-jaccard otherwise |
| **plan** | never planned / planned late | Did the model call `submit_plan` first with ≥3 steps? 1.0 yes, 0.5 if late or too few, 0.0 if never. Revision count is logged separately (not penalized). |
| **plan_adherence** | planned but didn't follow through | For each step in the FINAL submitted plan, did any later non-plan tool call reference its content (substring-match informative tokens)? Score = matched / total. |
| **required_files** | missing sort/index/stats outputs | Checks each required filename exists in the workdir |
| **voluntary_termination** | claims success without verifying / oscillates | Boolean: did the model stop on its own, or hit `--max-turns`? |

## Requirements

Pysam is **not** installed in `/home/llm/hf-env/`. Use a venv that has it:

```bash
python3 -m venv ~/coding-bench-env
~/coding-bench-env/bin/pip install pysam requests
# samtools is invoked via pysam.sort/pysam.index — install via apt or conda
```

The model endpoint just needs vLLM with `--enable-auto-tool-choice` and a
tool parser appropriate for the model.

## Usage

```bash
~/coding-bench-env/bin/python coding_benchmark.py \
  --model <MODEL_ID_OR_PATH> \
  --api-url http://localhost:8000/v1/chat/completions \
  --scenario t2-flag-chrom \
  --variant <run-label> \
  --output-dir ./sample_results
```

Options:

| Flag | Default | Notes |
|---|---|---|
| `--model` | required | Model id as exposed by the vLLM endpoint |
| `--api-url` | `http://localhost:8000/v1/chat/completions` | OpenAI-compatible chat completions URL |
| `--scenario` | required | One of `t1-mapq`, `t2-flag-chrom` |
| `--variant` | required | Becomes part of the output filename |
| `--output-dir` | `.` | JSON result destination |
| `--max-turns` | `20` | Hard cap on conversation turns |
| `--temperature` / `--top-p` / `--max-tokens` | `0.6` / `0.95` / `4096` | Generation params |
| `--request-timeout` | `180` | Per-request HTTP timeout (seconds) |
| `--keep-workdir` | off | Preserve sandbox tempdir for debugging |

## Scenarios

| Key | Difficulty | What the model has to do |
|---|---|---|
| `t1-mapq` | easy | Filter reads by `MAPQ >= 30`, write coordinate-sorted+indexed BAM |
| `t2-flag-chrom` | medium | T1 + flag bitmask filter (paired & not duplicate) + chromosome subset (chr1/chr2) + a `output.stats.txt` with input/output counts |

Add new scenarios by extending `SCENARIOS` in `scenarios.py` — provide a
`filter_fn` Python lambda (used to build the gold BAM) plus a
human-readable `filter_description` (used in the prompt) and the list of
required output filenames.

## Output

JSON file `coding-bench-<variant>-<scenario>.json` containing:

- `tool_calls` — per-turn record of every tool invocation, args, and output
- `scoring.api_correctness` — `total_pysam_refs`, `real_count`, `fake_count`, `fake_refs` (the names the model invented)
- `scoring.functional` — `match`, `model_count`, `gold_count`, plus `missing_count`/`extra_count` if the contents differ
- `scoring.plan` — extracted plan items
- `scoring.required_files` — which expected outputs are missing
- `voluntary_termination` — did the model stop on its own
- `turns`, `malformed_args`, `start_time`, `end_time`

## Caveats

- The plan-integrity score is heuristic. We can detect "did the model emit a
  plan in the right shape." We **cannot** mechanically grade whether the
  later code actually follows that plan — read the `tool_calls` log if you
  care about that signal specifically.
- `api_correctness` only catches confabulation of attribute *names*, not
  signatures. A model that calls `pysam.AlignmentFile(...)` with wrong
  positional args still scores 1.0 on that axis (and the `functional` axis
  picks up the resulting failure).
- The synthetic BAM is small and deterministic by seed. It exercises every
  filter criterion at least a few times, but it is not a stress test for
  large-input handling.
- Variance run-to-run is real. `n=1` is not enough for a confident
  capability claim — run each scenario at least 3-5 times per model and
  read the distribution.
