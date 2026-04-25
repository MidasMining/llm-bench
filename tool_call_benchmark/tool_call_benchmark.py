#!/usr/bin/env python3
"""Tool-call reliability benchmark for vLLM-served models.

!!! CONFIGURE BEFORE USE !!!
This benchmark SSHes into a target host and lets the model execute arbitrary
bash commands there (including container stop/rm/run). Required per-environment
inputs — no defaults are baked in:
  --target-host       IP/hostname of a *disposable* test rig (NOT production)
  --target-user       SSH user on the target
  --target-password   SSH password on the target (or use key-based auth and
                      pass an empty string; sshpass will be skipped)
Or set via env vars: BENCH_TARGET_HOST, BENCH_TARGET_USER, BENCH_TARGET_PASSWORD.

Sends a multi-step SSH task prompt to a vLLM endpoint with a `bash` tool
definition, executes the model's tool calls against a real target host, and
records:
- Number of tool calls made
- Number of canonical steps completed (out of 5)
- Whether tool-call argument JSON was well-formed
- Whether the fix landed (post-check on the target)

Designed for the Nosana queryAaaa ETIMEOUT scenario but the prompt and
post-check are pluggable.

Example:
  python3 tool_call_benchmark.py \\
    --model <MODEL_ID_OR_PATH> \\
    --api-url http://localhost:8001/v1/chat/completions \\
    --target-host <TARGET_HOST> \\
    --target-user <TARGET_USER> --target-password <TARGET_PASSWORD> \\
    --variant <run-label>
"""

import argparse
import json
import os
import subprocess
import sys
import time

import requests


def execute_command(cmd, timeout=60):
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return (result.stdout + result.stderr)[:4000]
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error: {e}"


def redact(text, secrets):
    """Replace each secret string in text with <REDACTED>. Empty/None secrets are skipped."""
    for s in secrets:
        if s:
            text = text.replace(s, "<REDACTED>")
    return text


def score_steps(tool_calls, target_host):
    """Walk the recorded tool calls and mark which canonical steps were attempted."""
    steps = {
        "1_ssh_logs": False,
        "2_ipv6_check": False,
        "3_inspect_env": False,
        "4_recreate_container": False,
        "5_monitor_logs": False,
    }
    for tc in tool_calls:
        cmd = tc["command"]
        if target_host not in cmd:
            continue
        if "logs" in cmd and ("nosana-node" in cmd or "tail" in cmd):
            if not steps["1_ssh_logs"]:
                steps["1_ssh_logs"] = True
            elif steps["4_recreate_container"]:
                steps["5_monitor_logs"] = True
        if "ip -6" in cmd or "ping6" in cmd or "ipv6" in cmd.lower():
            steps["2_ipv6_check"] = True
        if "inspect" in cmd and "Env" in cmd:
            steps["3_inspect_env"] = True
        if "podman run" in cmd and "NODE_OPTIONS" in cmd:
            steps["4_recreate_container"] = True
    return steps


def post_check_node_options(target_host, target_user, target_password):
    """Verify whether NODE_OPTIONS=--dns-result-order=ipv4first landed on the target."""
    cmd = (
        f'sshpass -p \'{target_password}\' ssh -o StrictHostKeyChecking=no '
        f'-o UserKnownHostsFile=/dev/null {target_user}@{target_host} '
        f'"sudo docker exec podman podman inspect nosana-node --format \'{{{{json .Config.Env}}}}\' 2>/dev/null"'
    )
    raw = execute_command(cmd)
    return ("dns-result-order" in raw), raw


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Model id as known to the vLLM endpoint")
    p.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--target-host", default=os.environ.get("BENCH_TARGET_HOST"),
                   help="IP or hostname the model should SSH into. "
                        "Falls back to env BENCH_TARGET_HOST. No baked-in default.")
    p.add_argument("--target-user", default=os.environ.get("BENCH_TARGET_USER"),
                   help="SSH user on the target. Falls back to env BENCH_TARGET_USER.")
    p.add_argument("--target-password", default=os.environ.get("BENCH_TARGET_PASSWORD"),
                   help="SSH password on the target. Falls back to env BENCH_TARGET_PASSWORD. "
                        "Pass empty string if using key-based auth.")
    p.add_argument("--variant", required=True, help="Label written into the result JSON filename")
    p.add_argument(
        "--prompt-file",
        default=None,
        help="Path to prompt template. Defaults to prompts/nosana_aaaa_fix.txt next to this script. "
             "{TARGET_HOST} is substituted before sending.",
    )
    p.add_argument("--output-dir", default=".", help="Where to write the JSON result")
    p.add_argument("--max-turns", type=int, default=15)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--request-timeout", type=int, default=120)
    args = p.parse_args()

    missing = [name for name, val in (
        ("--target-host (or env BENCH_TARGET_HOST)", args.target_host),
        ("--target-user (or env BENCH_TARGET_USER)", args.target_user),
    ) if not val]
    if args.target_password is None:
        missing.append("--target-password (or env BENCH_TARGET_PASSWORD; pass empty string for key auth)")
    if missing:
        sys.exit("ERROR: missing required credentials — set " + ", ".join(missing))

    if args.prompt_file is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.prompt_file = os.path.join(here, "prompts", "nosana_aaaa_fix.txt")
    with open(args.prompt_file) as f:
        task_prompt = f.read().replace("{TARGET_HOST}", args.target_host)

    tools = [{
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command on the local machine and return stdout+stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute"}
                },
                "required": ["command"],
            },
        },
    }]

    messages = [
        {"role": "system", "content": "You are an expert Linux sysadmin. Use the bash tool to execute commands. Complete the task step by step, verifying each step before proceeding."},
        {"role": "user", "content": task_prompt},
    ]

    log = {
        "variant": args.variant,
        "model": args.model,
        "api_url": args.api_url,
        "target_host": args.target_host,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tool_calls": [],
        "malformed_xml": 0,
        "turns": 0,
        "steps_completed": 0,
        "final_check": None,
        "model_response_texts": [],
    }

    print(f"=== Tool-Call Reliability Benchmark — Variant: {args.variant} ===")
    print(f"Model: {args.model}")
    print(f"Endpoint: {args.api_url}")
    print(f"Target: {args.target_user}@{args.target_host}")
    print(f"Max turns: {args.max_turns}")
    print()

    for turn in range(args.max_turns):
        log["turns"] = turn + 1
        print(f"--- Turn {turn + 1} ---")
        try:
            response = requests.post(args.api_url, json={
                "model": args.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            }, timeout=args.request_timeout)
            data = response.json()
        except Exception as e:
            print(f"API error: {e}")
            log["model_response_texts"].append(f"API error: {e}")
            break

        if "error" in data:
            print(f"API error: {data['error']}")
            log["model_response_texts"].append(f"API error: {data['error']}")
            break

        choice = data["choices"][0]
        msg = choice["message"]
        finish_reason = choice.get("finish_reason", "unknown")

        if msg.get("content"):
            print(f"Model: {msg['content'][:500]}")
            log["model_response_texts"].append(msg["content"])

        if msg.get("tool_calls"):
            messages.append(msg)
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                cmd = None
                try:
                    cmd = json.loads(fn["arguments"]).get("command", "")
                except json.JSONDecodeError:
                    print(f"  MALFORMED tool call arguments: {fn['arguments'][:200]}")
                    log["malformed_xml"] += 1
                if cmd:
                    print(f"  Tool call: bash({cmd[:120]})")
                    output = execute_command(cmd)
                    print(f"  Output: {output[:200]}")
                    log["tool_calls"].append({
                        "turn": turn + 1,
                        "command": redact(cmd, [args.target_password]),
                        "output": redact(output[:500], [args.target_password]),
                    })
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": output})
                else:
                    messages.append({
                        "role": "tool", "tool_call_id": tc["id"],
                        "content": "Error: could not parse command from tool call arguments",
                    })
        else:
            messages.append(msg)
            print(f"Model finished ({finish_reason}) — no tool calls")
            break

    print("\n=== Benchmark Complete ===")

    print("\n--- Post-check ---")
    has_node_options, raw = post_check_node_options(
        args.target_host, args.target_user, args.target_password
    )
    log["final_check"] = {
        "node_options_present": has_node_options,
        "raw": redact(raw[:500], [args.target_password]),
    }
    print(f"NODE_OPTIONS set: {has_node_options}")
    print(f"Raw: {raw[:300]}")

    steps = score_steps(log["tool_calls"], args.target_host)
    log["steps_completed"] = sum(steps.values())
    log["steps_detail"] = steps
    log["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"\nSteps completed: {log['steps_completed']}/5")
    print(f"Tool calls: {len(log['tool_calls'])}")
    print(f"Malformed argument JSON: {log['malformed_xml']}")
    print(f"Steps: {json.dumps(steps, indent=2)}")

    os.makedirs(args.output_dir, exist_ok=True)
    outfile = os.path.join(args.output_dir, f"tool-bench-{args.variant}.json")
    with open(outfile, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nResults: {outfile}")


if __name__ == "__main__":
    main()
