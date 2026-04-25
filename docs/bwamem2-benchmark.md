# LLM Bench — Evaluation Case: BWA-MEM2 Alignment Failure Analysis

## Metadata

| Field | Value |
|-------|-------|
| Category | Technical Review — Bioinformatics / Systems Engineering |
| Difficulty | Hard |
| Domain Knowledge Required | Genomics pipelines, Nextflow, containerized workflows, Linux systems |
| Evaluation Type | Open-ended analysis with factual constraints |
| Source | Real production failure — genomics compute platform, April 2026 |
| Time to Complete | N/A (single-turn response) |

---

## Prompt

Present the following to the model under evaluation. No additional context should be provided beyond what is in the prompt.

```
# BWA-MEM2 Alignment Failure — Review Request

## System Context

Genomics compute platform running nf-core/sarek 3.8.1 on bare-metal AMD EPYC Genoa 96-core nodes. Nextflow runs locally on compute nodes, placed there by HashiCorp Nomad. All pipeline containers pulled from a local Docker registry. Two identical Genoa nodes (H13-01, H13-02), each with 460GB NVMe scratch and 125GB RAM.

Pipeline: FASTQ → FASTQC → BWA-MEM2 alignment → MarkDuplicates → BQSR → DeepVariant/GATK HC → MultiQC

## What Happened

Job cc8197f8 (DeepVariant, WES, GIAB HG002) failed at the Aligning Reads stage on H13-02 after 14 minutes. 31 of 33 Nextflow tasks completed successfully. BWA-MEM2 runs 8 scatter tasks sequentially (~27s each). Tasks 1-6 succeeded. Tasks 7 and 8 failed with different errors:

**Task 7:**
```
[E::read_ncigar] No CIGAR operations
samtools sort: truncated file. Aborting
```
BWA-MEM2 output a malformed SAM record (no CIGAR), causing samtools sort to abort on the truncated pipe.

**Task 8:**
```
[W::sam_hdr_create] Ignored @SQ SN:HLA-C*16:01: : bad or missing LN tag
samtools sort: failed to change sort order header to 'SO:coordinate'
```
samtools sort failed to parse the SAM header because of a malformed HLA contig line.

## What We Ruled Out

- **Not disk space** — 195GB free (56% used)
- **Not OOM** — no kills in dmesg, each BWA-MEM2 task uses ~27GB RSS
- **Not the per-job work directory change** — 31 tasks ran correctly in the new path
- **Not node-specific (yet)** — only tested on H13-02 so far; previous successful runs were on H13-01
- Container versions: bwa-mem2:2.2.1, samtools bundled in the same container

## Previous Successful Runs on Same Data

| Job | Node | Caller | Duration | Result |
|-----|------|--------|----------|--------|
| 0ec08f21 | H13-02 | DeepVariant | 2h 08m | Complete |
| f8f81c8e | H13-02 | DeepVariant | ~2h | Complete |
| 9fe76819 | H13-01 | HaplotypeCaller | ~1h 30m | Complete |

Same sample, same containers, same config. The failure is intermittent.

## Proposed Fix

Add `errorStrategy 'retry'` with `maxRetries 2` to the BWA-MEM2 process in the Nextflow config:

```groovy
withName: 'NFCORE_SAREK:SAREK:ALIGNMENT:BWAMEM2_MEM' {
    container = '<DOCKER_REGISTRY>:5000/bwa-mem2:2.2.1--hd03093a_5'
    errorStrategy = 'retry'
    maxRetries = 2
}
```

## Questions for Review

1. Is `errorStrategy 'retry'` the right approach, or should we investigate the BWA-MEM2 build/version first?
2. Should the retry be scoped to just BWAMEM2_MEM, or applied more broadly to all alignment processes?
3. The HLA header issue (Task 8) looks different from the CIGAR issue (Task 7) — could these be two separate bugs, or is one causing the other?
4. Any concerns about retrying alignment tasks specifically (e.g., could a partial/corrupt output from a failed attempt interfere with the retry)?
5. Should we pin a different BWA-MEM2 version or samtools version to avoid this class of error?
```

---

## Scoring Rubric

### Scoring Dimensions

Each dimension is scored 0-3:
- **0** = Incorrect, harmful, or fabricated
- **1** = Partially correct but with significant errors
- **2** = Correct with minor gaps
- **3** = Fully correct and well-reasoned

Total possible score: **30 points** (10 dimensions × 3 points each)

---

### Dimension 1: Root Cause Identification (0-3)

**What the model should identify:**
The failure is an intermittent pipe corruption issue between BWA-MEM2 and samtools sort. BWA-MEM2 on high-core AVX-512 systems produces SAM output at a rate that can occasionally cause truncated or malformed pipe output. This is a known class of issue in the bioinformatics community.

| Score | Criteria |
|-------|----------|
| 3 | Correctly identifies pipe corruption between BWA-MEM2 and samtools sort under high-throughput AVX-512 conditions. Notes this is a known intermittent issue. |
| 2 | Identifies pipe/streaming issue but lacks specificity about AVX-512 or high-core correlation. |
| 1 | Identifies it as a BWA-MEM2 problem but attributes to wrong cause (e.g., reference corruption, disk I/O, memory). |
| 0 | Attributes to hardware fault, reference dictionary corruption, or other fabricated root cause without evidence. |

**Red flags (automatic score reduction):**
- Claiming the reference FASTA/FAI/dict is corrupt (same reference succeeded on same node multiple times)
- Claiming hardware memory bit-flips or NUMA faults without evidence
- Claiming concurrent task I/O collision (tasks run sequentially, confirmed in the prompt)

---

### Dimension 2: Error Relationship Analysis (0-3)

**Question 3 asks whether the two errors are related.**

| Score | Criteria |
|-------|----------|
| 3 | Correctly identifies both errors as symptoms of the same upstream pipe corruption. The HLA header warning is common with hg38 ALT contigs and becomes fatal when the pipe is already in a bad state. |
| 2 | Says they are related but doesn't explain the mechanism clearly. |
| 1 | Says they are unrelated bugs requiring separate fixes. |
| 0 | Invents a causal mechanism not supported by the evidence (e.g., claims one task's output corrupted the other's). |

---

### Dimension 3: Retry Strategy Assessment (0-3)

**Question 1 asks whether retry is the right approach.**

| Score | Criteria |
|-------|----------|
| 3 | Affirms retry as the correct immediate fix. Notes it's the standard pattern in nf-core pipelines. Recommends investigating root cause in parallel but not blocking the fix on it. |
| 2 | Affirms retry but adds unnecessary caveats that suggest misunderstanding of Nextflow's retry isolation. |
| 1 | Recommends against retry or says it's insufficient without additional validation steps. |
| 0 | Claims retry will propagate corrupted data downstream, or recommends rebuilding the pipeline instead. |

**Key fact the model must get right:** Nextflow retries run in a fresh work directory. Failed task outputs are discarded. Downstream processes never see partial output from a failed attempt. This is how nf-core/sarek handles alignment retries in production.

---

### Dimension 4: Retry Scope (0-3)

**Question 2 asks about scoping.**

| Score | Criteria |
|-------|----------|
| 3 | Recommends scoping narrowly to BWAMEM2_MEM only. Explains that broader application would mask real failures in other processes. |
| 2 | Recommends narrow scope but doesn't explain why. |
| 1 | Recommends applying retry broadly to all alignment or all processes. |
| 0 | Recommends a global retry strategy or doesn't address scope. |

---

### Dimension 5: Retry Safety / Output Isolation (0-3)

**Question 4 asks about corrupted output interfering with retries.**

| Score | Criteria |
|-------|----------|
| 3 | Correctly states low/no concern. Explains that Nextflow uses unique per-task directories, failed outputs are discarded, and retries start clean. References how nf-core/sarek handles this. |
| 2 | Says it's safe but doesn't explain the mechanism. |
| 1 | Expresses significant concern and recommends manual cleanup or validation between retries. |
| 0 | Claims retries will use or be contaminated by the failed attempt's output. Recommends custom BAM validation scripts between retry attempts. |

**Red flag:** Recommending `samtools quickcheck` or file-size validation between retry attempts indicates the model doesn't understand Nextflow's retry isolation. This adds unnecessary complexity and reveals a fundamental misunderstanding.

---

### Dimension 6: Version Pinning Recommendation (0-3)

**Question 5 asks about version changes.**

| Score | Criteria |
|-------|----------|
| 3 | Recommends staying on current version for now. Notes the issue is not version-specific — it's a class of problem that exists across BWA-MEM2 releases on high-core systems. Suggests version investigation as a parallel long-term effort. |
| 2 | Recommends staying but with less clear reasoning. |
| 1 | Recommends an immediate version change without strong justification. |
| 0 | Cites specific bug numbers, version releases, or fixes that cannot be verified or are fabricated. Recommends downgrading to original BWA (different tool entirely). |

**Red flags (automatic 0):**
- Citing specific GitHub issue numbers without flagging uncertainty
- Claiming a specific BWA-MEM2 version "fixes" this issue without verifiable evidence
- Confusing BWA-MEM2 with original BWA (bwa-0.7.x) — these are different tools
- Recommending a version that doesn't exist

---

### Dimension 7: Stack Awareness (0-3)

**Does the model respect the described system architecture?**

| Score | Criteria |
|-------|----------|
| 3 | All recommendations use Nextflow config overrides (`withName` blocks), reference nf-core/sarek correctly, and stay within the described stack. |
| 2 | Mostly correct but includes minor references to tools not in the stack. |
| 1 | Mixes in significant references to wrong tools (Snakemake, Flask, Celery, etc.) or writes invalid Nextflow syntax. |
| 0 | Provides implementation code in the wrong framework, writes custom process definitions for a community pipeline (can only override config, not redefine processes), or invents Nextflow directives that don't exist. |

**Red flags:**
- Writing `process BWA_EXAMPLE { ... }` blocks (can't redefine nf-core/sarek processes)
- Referencing Snakemake, WDL, or CWL when the system uses Nextflow
- Using `errorStrategy 'bypass'` (not a valid Nextflow error strategy)
- Writing Nextflow DSL2 syntax that doesn't compile

---

### Dimension 8: Evidence-Based Reasoning (0-3)

**Does the model reason from the evidence provided, or invent scenarios?**

| Score | Criteria |
|-------|----------|
| 3 | All claims tie back to evidence in the prompt. Uncertainty is flagged. No fabricated references. |
| 2 | Mostly evidence-based with minor speculation clearly marked. |
| 1 | Significant speculation presented as fact. Some fabricated details. |
| 0 | Fabricates bug numbers, version releases, or technical claims. Presents speculation as established fact. |

**Key evidence the model should use:**
- Same sample/container/config succeeded multiple times → intermittent, not systematic
- Tasks run sequentially → no concurrent I/O collision
- 31 of 33 tasks succeeded → not a global node or config problem
- No OOM, no disk space issue → ruled out resource exhaustion

---

### Dimension 9: Actionability (0-3)

**Are the recommendations concrete and implementable?**

| Score | Criteria |
|-------|----------|
| 3 | Provides a clear, minimal fix (config change) that can be implemented immediately. Separates immediate fix from longer-term investigation. |
| 2 | Provides a fix but includes unnecessary steps or complexity. |
| 1 | Recommendations are vague, overly complex, or require significant additional work before the fix can be applied. |
| 0 | Recommendations require rebuilding containers, rewriting pipeline code, or implementing custom validation frameworks. |

---

### Dimension 10: Conciseness and Signal-to-Noise (0-3)

**Does the model stay focused on the questions asked?**

| Score | Criteria |
|-------|----------|
| 3 | Answers the 5 questions directly. Minimal tangential content. Clear structure. |
| 2 | Answers the questions but includes moderate tangential content. |
| 1 | Answers are buried in extensive tangential analysis, checklists, or implementation code. |
| 0 | Response is dominated by irrelevant implementation, wrong-stack code, or fabricated troubleshooting steps. |

---

## Reference Answer (Gold Standard)

The following is a condensed reference answer representing a high-quality response. Models are not expected to match it word-for-word — this defines the factual boundaries.

**Root cause:** Intermittent pipe corruption between BWA-MEM2 and samtools sort. Known class of issue on high-core AVX-512 systems where BWA-MEM2 produces SAM output faster than the pipe consumer can process. Both errors (missing CIGAR, malformed HLA header) are symptoms of the same upstream pipe truncation.

**Q1 — Retry vs investigate:** Retry is the correct immediate fix. It's the standard pattern in nf-core pipelines. Investigate root cause (version, build flags, thread count) in parallel but don't block the fix.

**Q2 — Scope:** Narrow to `NFCORE_SAREK:SAREK:ALIGNMENT:BWAMEM2_MEM` only. Broader application masks real failures in other processes.

**Q3 — Related errors:** Yes — both are downstream effects of a corrupted/truncated SAM stream. The HLA header warning is common with hg38 ALT contigs and becomes fatal when combined with pipe corruption.

**Q4 — Retry safety:** Low concern. Nextflow uses unique per-task work directories. Failed task outputs are discarded entirely. Retries start in a fresh directory. Downstream processes never see partial output. This is how nf-core/sarek handles retries in production.

**Q5 — Version pinning:** Not immediately. BWA-MEM2 2.2.1 is current and matches what sarek 3.8.1 ships. The issue is not version-specific — it's a class of problem across BWA-MEM2 releases on very high core count systems. Retry first, investigate version changes as a separate effort. If retry rate exceeds ~5% over 10-20 jobs, consider alternative aligners (DragMap is supported in sarek).

**Implementation:** Single config change in `sarek_compute.config`:
```groovy
withName: 'NFCORE_SAREK:SAREK:ALIGNMENT:BWAMEM2_MEM' {
    errorStrategy = 'retry'
    maxRetries = 2
}
```

---

## Anti-Patterns (Automatic Score Penalties)

The following patterns indicate a low-quality response regardless of other content:

| Anti-Pattern | Why It's Wrong | Penalty |
|-------------|----------------|---------|
| Citing specific GitHub issue numbers without uncertainty flags | Likely fabricated — models confabulate plausible references | -3 from Dimension 8 |
| Recommending custom `process` blocks for nf-core/sarek | Cannot redefine community pipeline processes — only config overrides | -3 from Dimension 7 |
| Writing `errorStrategy 'bypass'` | Not a valid Nextflow error strategy | -3 from Dimension 7 |
| Claiming retry propagates corrupted BAMs downstream | Contradicts Nextflow's per-task isolation model | -3 from Dimension 5 |
| Recommending reference dictionary regeneration as the fix | Same reference succeeded on same node — not the cause | -3 from Dimension 1 |
| Confusing BWA (bwa-0.7.x) with BWA-MEM2 (2.x) | Different tools | -3 from Dimension 8 |
| Recommending Snakemake/WDL/CWL solutions | Wrong workflow engine | -3 from Dimension 7 |
| Writing bash monitoring/validation wrapper scripts | Unnecessary complexity — Nextflow handles retry isolation | -3 from Dimension 9 |

---

## Calibration Notes

**High-quality response (25-30):** Grok-class. Identifies root cause correctly, affirms retry, scopes narrowly, explains Nextflow retry isolation, stays within the stack, no fabricated references.

**Medium-quality response (15-24):** Gets the general direction right (retry is appropriate, errors are related) but includes some wrong-stack code, unnecessary complexity, or unverified claims.

**Low-quality response (0-14):** Fundamentally misunderstands the system. Fabricates references, recommends wrong tools, claims retry is unsafe, or provides implementation code that doesn't compile. Nemo-class responses typically score 5-10 due to correct high-level pattern recognition undermined by fabricated details and wrong-stack implementation.
