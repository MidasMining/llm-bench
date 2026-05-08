#!/usr/bin/env python3
"""Coding-task reliability benchmark for vLLM-served models.

!!! CONFIGURE BEFORE USE !!!
This benchmark is local-only — no SSH, no remote host. The model gets
file-IO and shell-command tools confined to a sandbox tempdir. Tasks
involve writing a pysam-based BAM filter, running it, verifying output,
and reporting success.

Required:
  --model         model id as known to the vLLM endpoint
  --variant       label written into the result JSON filename
  --scenario      one of the scenario keys defined in scenarios.py

Requirements on the host:
  - pysam (in the venv that runs this script)
  - samtools (pysam.sort/pysam.index call out to it)

Example:
  python3 coding_benchmark.py \\
    --model <MODEL_ID_OR_PATH> \\
    --api-url http://localhost:8000/v1/chat/completions \\
    --scenario t2-flag-chrom \\
    --variant <run-label>
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

import scenarios as scenarios_mod
import scoring


def run_command(cmd, cwd, timeout=120):
    try:
        r = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        out = (r.stdout + r.stderr)[:4000]
        return out, r.returncode
    except subprocess.TimeoutExpired:
        return f"command timed out after {timeout}s", -1
    except Exception as e:
        return f"error: {e}", -1


def safe_path(workdir, raw):
    """Resolve `raw` relative to workdir; reject if it escapes."""
    workdir = Path(workdir).resolve()
    p = Path(raw)
    target = (workdir / p).resolve() if not p.is_absolute() else p.resolve()
    try:
        target.relative_to(workdir)
    except ValueError:
        return None
    return target


def execute_tool(name, args, workdir):
    if name == "write_file":
        path = safe_path(workdir, args.get("path", ""))
        if path is None:
            return "error: path is outside the workdir"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = args.get("content", "")
        path.write_text(content)
        return f"wrote {len(content)} bytes to {path.relative_to(workdir)}"
    if name == "read_file":
        path = safe_path(workdir, args.get("path", ""))
        if path is None:
            return "error: path is outside the workdir"
        try:
            return path.read_text()[:4000]
        except FileNotFoundError:
            return f"error: file not found: {args.get('path')}"
    if name == "run_command":
        out, rc = run_command(args.get("command", ""), workdir)
        return f"[exit {rc}]\n{out}"
    return f"error: unknown tool {name}"


TOOLS = [
    {"type": "function", "function": {
        "name": "submit_plan",
        "description": (
            "Submit your plan before doing any other work. You MUST call this first; "
            "write_file, read_file, and run_command will return an error until you do. "
            "Provide 3-7 ordered steps describing your approach. "
            "If you later encounter unexpected issues (failed tests, missing methods, wrong assumptions), "
            "call this tool again with revised steps and a brief revision_reason — revisions are logged but not penalized."
        ),
        "parameters": {
            "type": "object", "required": ["steps"],
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered list of plan steps (one sentence each).",
                },
                "revision_reason": {
                    "type": "string",
                    "description": "If revising a previously submitted plan, briefly explain why.",
                },
            },
        },
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write a file. Paths are relative to the workdir; cannot escape it.",
        "parameters": {
            "type": "object", "required": ["path", "content"],
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workdir"},
                "content": {"type": "string", "description": "File contents to write"},
            },
        },
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file. Paths are relative to the workdir.",
        "parameters": {
            "type": "object", "required": ["path"],
            "properties": {"path": {"type": "string"}},
        },
    }},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Run a bash command in the workdir. Returns stdout+stderr and the exit code.",
        "parameters": {
            "type": "object", "required": ["command"],
            "properties": {"command": {"type": "string"}},
        },
    }},
]


def build_prompt(scenario, workdir):
    here = Path(__file__).parent
    text = (here / "prompts" / "pysam_filter_task.txt").read_text()
    return (text
            .replace("{INPUT_BAM_PATH}", str(Path(workdir) / "input.bam"))
            .replace("{OUTPUT_BAM_PATH}", "output.bam")
            .replace("{FILTER_CRITERIA}", scenario["filter_description"])
            .replace("{EXTRA_REQUIREMENTS}", scenario["extra_requirements"])
            .replace("{WORKDIR}", str(workdir)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--scenario", required=True, choices=list(scenarios_mod.SCENARIOS.keys()))
    p.add_argument("--variant", required=True, help="Label written into the result JSON filename")
    p.add_argument("--output-dir", default=".", help="Where to write the JSON result")
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--request-timeout", type=int, default=180)
    p.add_argument("--keep-workdir", action="store_true",
                   help="Keep the sandbox workdir after the run (for debugging)")
    args = p.parse_args()

    scenario = scenarios_mod.SCENARIOS[args.scenario]

    here = Path(__file__).parent
    sys.path.insert(0, str(here))
    import fixtures

    workdir = Path(tempfile.mkdtemp(prefix=f"coding_bench_{args.scenario}_"))
    print(f"=== Coding Benchmark — {scenario['name']} ===")
    print(f"Model: {args.model}")
    print(f"Workdir: {workdir}")
    print()

    try:
        input_bam = workdir / "input.bam"
        gold_bam = workdir / ".gold.bam"
        fixtures.make_test_bam(str(input_bam))
        fixtures.make_gold(str(input_bam), str(gold_bam), scenario["filter_fn"])

        prompt = build_prompt(scenario, workdir)
        messages = [
            {"role": "system",
             "content": "You are a careful Python developer. Use the provided tools to complete the task. Always verify your own work before claiming success."},
            {"role": "user", "content": prompt},
        ]

        log = {
            "variant": args.variant, "model": args.model, "scenario": args.scenario,
            "scenario_name": scenario["name"],
            "workdir": str(workdir), "turns": 0,
            "tool_calls": [], "model_response_texts": [],
            "plan_history": [],
            "voluntary_termination": None, "malformed_args": 0,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        plan_submitted = False

        for turn in range(args.max_turns):
            log["turns"] = turn + 1
            print(f"--- Turn {turn + 1} ---")
            # Force submit_plan until one has been submitted. After that, free choice.
            # This makes the bench universal across models — the "plan first" constraint
            # lives at the protocol level, not the prompt level.
            tool_choice = ({"type": "function", "function": {"name": "submit_plan"}}
                           if not plan_submitted else "auto")
            try:
                resp = requests.post(args.api_url, json={
                    "model": args.model, "messages": messages, "tools": TOOLS,
                    "tool_choice": tool_choice,
                    "temperature": args.temperature, "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                }, timeout=args.request_timeout)
                data = resp.json()
            except Exception as e:
                log["api_error"] = str(e)
                print(f"API error: {e}")
                break

            if "error" in data:
                log["api_error"] = str(data["error"])
                print(f"API error: {data['error']}")
                break

            choice = data["choices"][0]
            msg = choice["message"]
            if msg.get("content"):
                log["model_response_texts"].append(msg["content"])
                print(f"Model: {msg['content'][:300]}")

            if msg.get("tool_calls"):
                messages.append(msg)
                for tc in msg["tool_calls"]:
                    fn = tc["function"]
                    try:
                        tool_args = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        log["malformed_args"] += 1
                        log["tool_calls"].append({
                            "turn": turn + 1, "tool": fn["name"],
                            "error": "malformed JSON args",
                            "raw_args": fn["arguments"][:500],
                        })
                        messages.append({"role": "tool", "tool_call_id": tc["id"],
                                         "content": "error: arguments JSON did not parse"})
                        continue
                    print(f"  Tool: {fn['name']}({json.dumps(tool_args)[:160]})")
                    if fn["name"] == "submit_plan":
                        steps = tool_args.get("steps", []) or []
                        reason = tool_args.get("revision_reason")
                        log["plan_history"].append({
                            "turn": turn + 1, "steps": steps, "revision_reason": reason,
                        })
                        plan_submitted = True
                        rev = len(log["plan_history"]) - 1
                        output = (f"plan accepted ({len(steps)} steps)" if rev == 0
                                  else f"plan revised — revision #{rev} accepted ({len(steps)} steps)")
                    elif not plan_submitted:
                        output = ("error: submit_plan must be called first. "
                                  "Submit your approach as 3-7 steps via submit_plan(steps=[...]) "
                                  "before using write_file, read_file, or run_command.")
                    else:
                        output = execute_tool(fn["name"], tool_args, workdir)
                    print(f"  Output: {output[:200]}")
                    log["tool_calls"].append({
                        "turn": turn + 1, "tool": fn["name"], "args": tool_args,
                        "output": output[:1500],
                    })
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": output})
            else:
                messages.append(msg)
                log["voluntary_termination"] = True
                print("Model stopped (no tool calls)")
                break

        if log["voluntary_termination"] is None:
            log["voluntary_termination"] = False  # hit max_turns

        log["scoring"] = {}
        log["scoring"]["api_correctness"] = scoring.check_pysam_api(str(workdir / "solution.py"))
        out_bam = workdir / "output.bam"
        if out_bam.exists():
            log["scoring"]["functional"] = scoring.check_functional(str(out_bam), str(gold_bam))
        else:
            log["scoring"]["functional"] = {"error": "no output.bam produced", "score": 0.0, "match": False}
        log["scoring"]["plan"] = scoring.check_plan(log["plan_history"], log["tool_calls"])
        log["scoring"]["plan_adherence"] = scoring.check_plan_adherence(log["plan_history"], log["tool_calls"])
        log["scoring"]["required_files"] = scoring.check_required_files(workdir, scenario["required_output_files"])

        log["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        os.makedirs(args.output_dir, exist_ok=True)
        outfile = os.path.join(args.output_dir, f"coding-bench-{args.variant}-{args.scenario}.json")
        with open(outfile, "w") as f:
            json.dump(log, f, indent=2, default=str)

        print("\n=== Summary ===")
        for axis, result in log["scoring"].items():
            score = result.get("score", "?")
            print(f"  {axis}: {score}")
        print(f"  voluntary_termination: {log['voluntary_termination']}")
        print(f"  turns: {log['turns']}, malformed_args: {log['malformed_args']}")
        print(f"\nResults: {outfile}")
    finally:
        if args.keep_workdir:
            print(f"workdir preserved: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
