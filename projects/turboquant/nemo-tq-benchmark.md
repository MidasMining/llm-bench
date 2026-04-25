# Nemo Tool-Call Reliability Benchmark — KV Cache Impact Test

## Purpose

Measure whether the current production KV cache setting (`tq-t3nc`, 3-bit TurboQuant keys, 4× compression) degrades opencode+Nemotron-Cascade-2's ability to complete a multi-step remote-execution task. Compare completion rate across different `--kv-cache-dtype` settings using the same prompt, same target, same harness.

**Status of prior runs (2026-04-19, under `tq-t3nc`):**
- Run 1 (pre-AGENTS.md fix): 5 tool calls, 0/5 steps completed, false success claim, wrong SSH user on 3/5 calls.
- Run 2 (post-AGENTS.md fix): 4 tool calls, 2/5 steps completed, terminated mid-task after emitting a malformed tool-call XML (bare `<parameter=...>` without opening `<function=bash>`).

## Hypothesis

Under aggressive KV quantization, structured-output emission (tool-call XML) and long-prompt instruction adherence may degrade. Benchmark: same test under `auto` (BF16) and `fp8` KV cache should yield higher completion rate if KV quant is the bottleneck.

## Test target

- **Rig (current, 2026-04-24 onward):** `Z-TestDummy` VM at `192.168.122.136` on the EPYC LLM host. Host libvirt domain: `Z-TestDummy`. HiveOS guest, `user`/`1`, passwordless sudo. Nested `docker → podman → nosana-node` stack matches production exactly; image is pinned as `localhost/nosana-node:broken-testfixture` (digest `sha256:87be673e609fb8622990b3ec19ebe6fd105982b9eff7e7e0b7e0930d0889b683`) and launched with `--pull=never` so upstream fixes can't drift the fixture. The `ctr.log` has a synthesized `queryAaaa ETIMEOUT nosana.mypinata.cloud` fixture (12 matches) seeded to match the historical error pattern verbatim.
- **Why a VM instead of a production rig:** running on the fleet means un-fixing the rig between model runs, sacrificing earnings, and risking `--pull=always` image drift that breaks comparability across runs. The VM is isolated, idempotent, and snapshotable.
- **Prompt edit needed:** the prompt says "SSH into my HiveOS worker at 192.168.1.52" — for test runs, substitute `192.168.122.136`. Everything else in the prompt remains valid (same podman stack, same log pattern, same fix recipe).
- **Historical target (preserved for reference):** `192.168.1.52` (H12D-02, 5080 PNY) and alternates `.54`, `.60`, `.115`, `.117`. Prior Nemotron Cascade-2 and MiniMax runs targeted `.52` live — do not re-run on production rigs.

## Exact prompt (stored at `/tmp/nemo-prompt.txt`)

Inline copy in case `/tmp/` was cleared:

```
SSH into my HiveOS worker at 192.168.1.52. The pasted prompt is the task relevant to that rig:

**Prompt for LLM technician:**

> I have a Nosana GPU worker node running on a HiveOS rig. The node runs inside a nested container stack: the host runs Docker, Docker runs a `podman` container, and inside that podman runs the `nosana-node` container (Node.js application).
>
> Jobs are failing with this error in the nosana-node logs:
> ```
> Error finishing job BANjb64QDk6jzqFzYrX4gpcW3EMpwKowwmoksLYKquTG
> Message: Failed to finish job: Error: queryAaaa ETIMEOUT nosana.mypinata.cloud
> ```
>
> The error happens during job result upload. The GPU computation finishes fine. My network does not have IPv6 connectivity — there are no IPv6 addresses on any host interface, and no IPv6 routes. However, DNS servers still return AAAA records for `nosana.mypinata.cloud`.
>
> To access the logs:
> ```bash
> sudo docker exec podman podman logs --tail=100 nosana-node 2>&1
> ```
> Note: the output contains ANSI escape codes and carriage returns instead of newlines. Strip them with: `sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | tr '\r' '\n'`
>
> The fix that has worked on other rigs is to recreate the nosana-node container with the environment variable `NODE_OPTIONS=--dns-result-order=ipv4first`, which tells Node.js to prefer IPv4 addresses when DNS returns both A and AAAA records.
>
> To recreate the container:
> ```bash
> sudo docker exec podman podman stop nosana-node
> sudo docker exec podman podman rm nosana-node
> sudo docker exec -it podman podman run --pull=always --name nosana-node \
>   --network NOSANA_GATEWAY --interactive -t \
>   --volume /root/.nosana/:/root/.nosana/ \
>   --mount type=bind,source=/root/../podman.sock,target=/root/.nosana/podman/podman.sock \
>   -e CLI_VERSION= \
>   -e NODE_OPTIONS=--dns-result-order=ipv4first \
>   docker.io/nosana/nosana-node:latest start --network mainnet
> ```
>
> Please:
> 1. SSH into the rig and confirm the AAAA timeout error exists in the logs
> 2. Verify the rig has no IPv6 connectivity (`ip -6 addr show`, `ping6 -c1 google.com`)
> 3. Check if `NODE_OPTIONS` is already set in the current container (`sudo docker exec podman podman inspect nosana-node --format "{{json .Config.Env}}"`)
> 4. If not set, apply the fix by recreating the container with the command above
> 5. Monitor logs for 5 minutes to confirm no new AAAA timeout errors appear
```

## Pre-check (confirm rig is a valid test target)

```bash
# Verify NODE_OPTIONS absent and nosana-node container exists
IP=52
sshpass -p 1 ssh -o StrictHostKeyChecking=no user@192.168.1.$IP \
  "sudo docker exec podman podman inspect nosana-node --format '{{json .Config.Env}}'" \
  | grep -o 'NODE_OPTIONS[^"]*' \
  || echo "NODE_OPTIONS absent — valid test target"
```

If NODE_OPTIONS is already SET (from a prior passing run), pick a different IP from the list above.

## Harness state

- opencode CLI: `~/.opencode/bin/opencode` (v1.4.11)
- Global config: `~/.config/opencode/opencode.json` → provider `vllm`, model `Nemotron-Cascade-2-30B-A3B`
- Global instructions (IMPORTANT): `~/.config/opencode/AGENTS.md` — contains SSH credential rules, verification discipline, HiveOS specifics. **Do NOT modify this file for the benchmark.** It was the same file both benchmark runs used.
- vLLM endpoint: `http://127.0.0.1:8000/v1`
- vLLM launch script: `/home/llm/run-wht-production.sh` (current: `--kv-cache-dtype tq-t3nc`)

## Exact launch command

```bash
# Session is fresh (opencode run creates a new session each call).
# nohup + backgrounding avoids TTY dependencies.
nohup bash -c '~/.opencode/bin/opencode run \
  --model "vllm//home/llm/models/Nemotron-Cascade-2-30B-A3B" \
  "$(cat /tmp/nemo-prompt.txt)" \
  > /tmp/nemo-benchmark-run.log 2>&1; \
  echo "---EXIT=$?---" >> /tmp/nemo-benchmark-run.log' \
  > /dev/null 2>&1 &
disown
```

Wait for completion (typically 2-15 minutes, driven by how many tool calls nemo makes):

```bash
until grep -q 'EXIT=' /tmp/nemo-benchmark-run.log 2>/dev/null; do sleep 30; done
```

## Evaluation rubric (score each variant separately)

For each KV cache variant, record:

| Metric | How to measure |
|---|---|
| Tool calls made | `grep -c '^\$ ' /tmp/nemo-benchmark-run.log` |
| Steps completed (0-5) | Manual review — see step definitions below |
| SSH user correctness | `grep -c 'root@\|llm@\|admin@' /tmp/nemo-benchmark-run.log` (should be 0) |
| Malformed tool-call XML | `grep -c '<parameter=' /tmp/nemo-benchmark-run.log` (should be 0 in non-tool-call positions) |
| False success claim | Did final text claim the fix was applied when `NODE_OPTIONS` is still absent post-run? |
| NODE_OPTIONS applied | Post-check (below) |

**Step definitions (from the prompt):**
1. Confirm AAAA timeout error exists in logs
2. Verify no IPv6 connectivity (both `ip -6 addr show` AND `ping6 -c1 google.com`)
3. Check NODE_OPTIONS env var
4. Recreate container with the fix
5. Monitor logs 5 minutes

## Post-check (did the fix actually land?)

```bash
IP=52
ENV=$(sshpass -p 1 ssh -o StrictHostKeyChecking=no user@192.168.1.$IP \
  "sudo docker exec podman podman inspect nosana-node --format '{{json .Config.Env}}'")
echo "$ENV" | grep -o 'NODE_OPTIONS[^"]*' || echo "NODE_OPTIONS still absent"
```

If `NODE_OPTIONS=--dns-result-order=ipv4first` is present → step 4 succeeded.

## KV cache variants to compare

Minimum set:

1. **`tq-t3nc`** (current production baseline — 2/5 steps, malformed XML on last call)
2. **`auto`** (default — BF16 or FP16, whichever vLLM picks)
3. **`fp8`** (requires patches 1 & 2 — see MEMORY.md fp8 KV patches section)

To switch:

```bash
# Stop current vLLM
pgrep -f 'vllm serve' | xargs -r kill
# Edit /home/llm/run-wht-production.sh line 9 to the desired --kv-cache-dtype
# Relaunch
bash /home/llm/run-wht-production.sh &
```

Wait for vLLM to report ready (look for `Application startup complete` in its log) before running the benchmark.

**IMPORTANT:** After the benchmark suite completes, restore production config to whatever was the pre-benchmark state.

## Between-run reset

Each benchmark run must start from the same baseline (NODE_OPTIONS absent, `queryAaaa ETIMEOUT` fixture visible in `podman logs`). Use the VM snapshot to revert atomically instead of recreating the container — the fixture's `ctr.log` seeding is re-created automatically.

```bash
# From EPYC LLM host
echo 7355608 | sudo -S virsh snapshot-revert Z-TestDummy --snapshot-name clean-baseline --running
# Verify baseline post-revert
sshpass -p 1 ssh -o StrictHostKeyChecking=no user@192.168.122.136 \
  'sudo docker exec podman podman logs --tail=40 nosana-node 2>&1 | sed "s/\x1b\[[0-9;]*[a-zA-Z]//g" | tr "\r" "\n" | grep -c queryAaaa'
# Expect: 12
```

If the snapshot is unavailable, fall back to manual reset on the VM:
```bash
sshpass -p 1 ssh -o StrictHostKeyChecking=no user@192.168.122.136 bash -s <<'EOF'
sudo docker exec podman podman stop nosana-node || true
sudo docker exec podman podman rm -f nosana-node || true
sudo docker exec podman podman run -d --pull=never --name nosana-node \
  --network NOSANA_GATEWAY \
  --volume /root/.nosana/:/root/.nosana/ \
  --mount type=bind,source=/root/../podman.sock,target=/root/.nosana/podman/podman.sock \
  -e CLI_VERSION= \
  localhost/nosana-node:broken-testfixture start --network mainnet
# Re-seed ctr.log fixture
CID=$(sudo docker exec podman podman inspect nosana-node --format '{{.Id}}')
sudo docker cp /tmp/ctr-fixture.log podman:/tmp/ctr-fixture.log
sudo docker exec podman sh -c "cat /tmp/ctr-fixture.log >> /var/lib/containers/storage/overlay-containers/$CID/userdata/ctr.log"
EOF
```

Note: drop `-it` / `-t` (non-interactive SSH). Use `-d` for detached. Use `--pull=never` and the pinned local tag to preserve the exact broken image digest.

## Expected outcome

If 3-bit KV is the bottleneck, `auto` or `fp8` should complete 4-5/5 steps with zero malformed XML, zero wrong-user SSH attempts. If all three variants produce the same 2/5 completion, the model (Cascade-2 30B-A3B, 3B active) is the ceiling and the fix is a larger model, not a KV dtype change.

## Reference: prior runs artifacts

- Run 1 transcript (SQLite extraction): `/tmp/opencode-session.txt` (may be cleared)
- Run 2 log: `/tmp/nemo-retest2.log` (may be cleared)
- Production vLLM launch script: `/home/llm/run-wht-production.sh`
- AGENTS.md harness: `~/.config/opencode/AGENTS.md`

## Cross-model data point: MiniMax M2.7 REAP 172B AWQ (2026-04-24)

Run against the same rig (.52), same prompt, same AGENTS.md harness, same `tq-t3nc` KV cache setting.

**vLLM launch:**
```
vllm serve /home/llm/models/MiniMax-M2.7-REAP-172B-AWQ \
  --tensor-parallel-size 8 --gpu-memory-utilization 0.92 \
  --max-num-seqs 4 --max-model-len 65536 --port 8002 \
  --trust-remote-code --kv-cache-dtype tq-t3nc \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2
```

**Opencode config:** provider `vllm-minimax` on `http://127.0.0.1:8002/v1`, model `/home/llm/models/MiniMax-M2.7-REAP-172B-AWQ`.

**Results vs Nemotron-Cascade-2 baseline (both @ `tq-t3nc`):**

| Metric | MiniMax M2.7 REAP | Nemotron-Cascade-2 |
|---|---|---|
| Tool calls | 40 | 4 |
| Steps completed | 4/5 | 2/5 |
| SSH user correct | 40/40 (100%) | 2/5 |
| Malformed XML | 0 | 1 (terminated run) |
| False success claim | No | Yes |
| NODE_OPTIONS applied | **✅ verified on rig** | ❌ |

**Observations:**
- Zero malformed tool-call XML across 40 calls (minimax_m2 parser solid).
- Credential discipline perfect — never guessed a different user.
- Hit the same `podman logs` buffering/hang trap Nemotron did, but iterated with varied approaches instead of bailing out.
- Honest verification discipline: when `podman logs` kept hanging, it reported "could not verify via logs" rather than claiming "no errors visible."
- Minor rule violation: `--interactive -t` in the podman run command (AGENTS.md says strip `-it`), but outer `docker exec -d` saved it.

**Log artifact:** `/tmp/minimax-benchmark-run.log`

**Interpretation:** At the same `tq-t3nc` KV setting, the 172B REAP model dramatically outperforms the 30B-A3B Nemotron. This is a cross-model signal, NOT a within-model TQ variant test — so it doesn't isolate whether KV quantization is the bottleneck for Nemotron specifically. The tq-t3nc / auto / fp8 variant test on Nemotron is still the clean experiment.

## Cross-model data point: Nemotron-3-Super-120B AWQ (2026-04-24)

First benchmark against the new VM test rig (`192.168.122.136`, `Z-TestDummy`) instead of a live fleet rig. Snapshot-revert between runs replaces the old un-fix/re-fix cycle. Prompt identical except IP replaced. Same `tq-t3nc` KV cache setting as Nemotron/MiniMax runs.

**vLLM launch (port 8001, from `/home/llm/run-super120b-tq-131k.sh`):**
```
vllm serve /home/llm/models/Nemotron-3-Super-120B-AWQ-4bit \
  --tensor-parallel-size 8 --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 --max-model-len 131072 --port 8001 \
  --trust-remote-code --enable-expert-parallel \
  --mamba-ssm-cache-dtype float16 --kv-cache-dtype tq-t3nc \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser-plugin .../super_v3_reasoning_parser.py \
  --reasoning-parser super_v3
```

**opencode provider:** `vllm-super//home/llm/models/Nemotron-3-Super-120B-AWQ-4bit`

| Metric | Super 120B | MiniMax 172B | Nemotron Cascade-2 30B |
|---|---|---|---|
| Tool calls (ssh) | 27 | 40 | 4 |
| Steps completed (of 5) | 5 | 4 | 2 |
| Fix applied to rig | yes | yes | no (terminated) |
| Fix verified post-run | yes (`NODE_OPTIONS=--dns-result-order=ipv4first` in env) | yes | n/a |
| Wrong-user SSH attempts | 0 | 0 | 0 |
| Malformed tool XML | 0 | 0 | 1 (bare `<parameter=>`) |
| AGENTS.md violations | 1 (`-it` in first recreate; self-recovered with `-d`) | 1 (`--interactive -t`; covered by outer `-d`) | 3 (wrong user, false success) |

**Observations:**
- Early SSH calls failed because Super omitted `-o StrictHostKeyChecking=no` on a fresh known_hosts — self-diagnosed after several silent failures and added the flag.
- Hit the `-it` trap on first recreate (exactly what AGENTS.md warns about), got the TTY error, and on retry correctly used `-d` without `-it`. This is the ideal failure-recovery pattern.
- Monitor phase (step 5): attempted a 10× / 5× loop with `sleep 30` that kept hitting opencode's internal bash timeout. Fell back to a single spot-check, which showed no new `queryAaaa` errors — reasonable adaptation but skimps the "5 minutes" requirement.
- Final summary is factually correct and matches the verified VM state.

**Log artifact:** `/tmp/super-tq-vm-run1.log` (175 lines)

**VM-rig validation:** This run is the first proof that the VM fixture reproduces the benchmark conditions faithfully — Super saw the `queryAaaa ETIMEOUT` pattern in `podman logs`, found `NODE_OPTIONS` absent, and successfully applied the fix. Fixture + snapshot workflow works end-to-end.

### Super run 2 (2026-04-24, post-snapshot-revert)

Re-run on the same VM after `virsh snapshot-revert Z-TestDummy clean-baseline --running`. Same vLLM endpoint, same prompt, same KV cache. Goal: variance check — does Super behave consistently?

| Metric | Run 1 | Run 2 |
|---|---|---|
| Steps completed (of 5) | 5 | 5 |
| Fix applied + verified | yes | yes (`NODE_OPTIONS=--dns-result-order=ipv4first` confirmed in env) |
| SSH tool calls | 27 | 14 |
| **Local-shell missteps** | 0 | **22** (ran `docker ps`, `ps aux | grep podman`, `sudo docker exec podman` against the EPYC host instead of SSH-ing) |
| Wrong-user SSH attempts | 0 | 0 |
| Malformed tool XML | 0 | 0 |
| Final summary correctness | matched VM state | matched VM state |

**New failure mode in run 2:** Super dropped the `sshpass -p '1' ssh user@...` wrapper on multiple commands and ran them on the EPYC LLM host directly. Saw the host's `ollama-mirror`, `ollama-registry`, `claudish-alpha`, `claudish-beta` containers; hit a `sudo: a terminal is required` prompt; tried `echo '1' | sudo -S docker exec podman ...` (wrong sudo password — local user is `llm`/`7355608`, not `user`/`1`). Recovered by re-prefixing later commands with `sshpass`. Despite the missteps, completion was correct.

**Possible cause:** AGENTS.md is loaded into the system prompt; the prompt opens with "SSH into my HiveOS worker at 192.168.122.136" but Super may have lost the SSH-prefix discipline as the conversation grew. This is a long-context / instruction-adherence regression that opcode's prompt template doesn't pin to every tool call.

**Variance signal:** Same model, same prompt, same KV cache, same target — but tool-use shape diverged significantly (run 1 was clean SSH-only, run 2 wandered into local exec). This argues for `n≥3` per variant before conclusions, not `n=1`. The model is non-deterministic on tool-routing even when output is "correct."

**Log artifact:** `/tmp/super-tq-vm-run2.log` (280 lines)
