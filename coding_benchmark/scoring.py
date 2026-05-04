"""Scoring helpers for the coding benchmark.

Each helper returns a dict with at minimum a `score` field in [0, 1].
The runner aggregates these into the result JSON.
"""

import ast
import importlib
import re
from pathlib import Path


def check_pysam_api(file_path):
    """AST-walk the model's solution and verify every `pysam.X.Y...` reference
    actually exists in the installed pysam package.

    A reference is "real" if every attribute in the chain resolves via getattr().
    Returns:
      - total_pysam_refs, real_count, fake_count, fake_refs (list of strings)
      - score = real / total (1.0 if no pysam refs at all)
    """
    try:
        src = Path(file_path).read_text()
    except FileNotFoundError:
        return {"error": f"file not found: {file_path}", "score": 0.0}
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {"error": f"syntax error: {e}", "score": 0.0}

    pysam = importlib.import_module("pysam")

    chains = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain = []
        cur = node
        while isinstance(cur, ast.Attribute):
            chain.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name) and cur.id == "pysam":
            chain.reverse()
            chains.append(chain)

    real, fake = [], []
    for chain in chains:
        obj = pysam
        valid = True
        for attr in chain:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                valid = False
                break
        ref = "pysam." + ".".join(chain)
        (real if valid else fake).append(ref)

    total = len(chains)
    return {
        "total_pysam_refs": total,
        "real_count": len(real),
        "fake_count": len(fake),
        "fake_refs": sorted(set(fake)),
        "score": (len(real) / total) if total else 1.0,
    }


def check_functional(model_output_bam, gold_bam):
    """Compare model's output BAM to the gold BAM by (chrom, pos, flag, qname).

    Exact match (same order) → score 1.0.
    Same set but different order → score = jaccard, with `match=False` so the
    sort requirement is visible in the result.
    """
    import pysam
    try:
        with pysam.AlignmentFile(model_output_bam, "rb") as a:
            a_reads = [(r.reference_name, r.reference_start, r.flag, r.query_name) for r in a]
        with pysam.AlignmentFile(gold_bam, "rb") as b:
            b_reads = [(r.reference_name, r.reference_start, r.flag, r.query_name) for r in b]
    except Exception as e:
        return {"error": f"failed to open BAMs: {e}", "score": 0.0, "match": False}

    if a_reads == b_reads:
        return {"score": 1.0, "match": True, "model_count": len(a_reads), "gold_count": len(b_reads)}

    a_set, b_set = set(a_reads), set(b_reads)
    union = a_set | b_set
    jaccard = (len(a_set & b_set) / len(union)) if union else 0.0
    return {
        "score": jaccard,
        "match": False,
        "model_count": len(a_reads),
        "gold_count": len(b_reads),
        "missing_count": len(b_set - a_set),
        "extra_count": len(a_set - b_set),
        "note": "score < 1.0 because exact-order match failed (sort requirement) or contents differ",
    }


def check_plan(plan_history, tool_calls):
    """Score whether the model submitted a plan via the submit_plan tool first.

    Plan is now structured tool-call data, not free-form text — no regex
    parsing needed. Scoring:
      - 0.0 if submit_plan never called
      - 0.5 if called but final plan has <3 steps, OR called after another tool
      - 1.0 if submit_plan was the first tool used and final plan has ≥3 steps
    """
    if not plan_history:
        return {
            "score": 0.0, "plan_count": 0, "revision_count": 0,
            "was_first_tool": False, "note": "submit_plan never called",
        }

    was_first = bool(tool_calls) and tool_calls[0].get("tool") == "submit_plan"
    final = plan_history[-1]
    final_count = len(final["steps"])
    rev_count = len(plan_history) - 1

    if was_first and final_count >= 3:
        score = 1.0
    elif final_count >= 1:
        score = 0.5
    else:
        score = 0.0

    return {
        "score": score,
        "plan_count": final_count,
        "revision_count": rev_count,
        "was_first_tool": was_first,
        "final_steps": final["steps"],
        "revision_reasons": [p.get("revision_reason") for p in plan_history[1:]],
    }


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "and", "or", "but", "if", "then", "else",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "into",
    "this", "that", "these", "those", "it", "its", "my", "your", "our",
    "we", "i", "you", "they", "them", "as", "also", "just", "so", "very",
    "all", "each", "any", "such", "than", "too", "not", "no", "yes",
    "step", "steps", "plan", "approach", "first", "second", "third",
    "next", "then", "use", "using", "need", "needs", "want", "wants",
    "ensure", "make", "set", "get",
}


def _content_tokens(text):
    """Lowercase word-tokenize, drop stopwords and tokens shorter than 3 chars."""
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) >= 3}


def check_plan_adherence(plan_history, tool_calls):
    """For each step in the FINAL submitted plan, did any later non-plan tool
    call reference its content?

    Heuristic — extracts informative tokens from each step (drop stopwords,
    keep nouns/verbs), then substring-matches them against the joined
    content of all non-plan tool calls (args + outputs). A step "matches"
    if at least one of its informative tokens appears.
    """
    if not plan_history:
        return {"score": 0.0, "note": "no plan to compare against"}
    final_steps = plan_history[-1]["steps"]
    if not final_steps:
        return {"score": 0.0, "note": "final plan has 0 steps"}

    haystack_parts = []
    for tc in tool_calls:
        if tc.get("tool") == "submit_plan":
            continue
        args = tc.get("args", {}) or {}
        for key in ("path", "command", "content"):
            v = args.get(key)
            if v:
                haystack_parts.append(str(v))
        out = tc.get("output", "")
        if out:
            haystack_parts.append(out)
    haystack = "\n".join(haystack_parts).lower()

    matches = []
    for step in final_steps:
        tokens = _content_tokens(step)
        if not tokens:
            matches.append({"step": step, "matched_tokens": [], "matched": False})
            continue
        hits = sorted(t for t in tokens if t in haystack)
        matches.append({
            "step": step, "matched_tokens": hits[:5], "matched": bool(hits),
        })

    matched_count = sum(1 for m in matches if m["matched"])
    return {
        "score": matched_count / len(final_steps),
        "matched_count": matched_count,
        "total_steps": len(final_steps),
        "matches": matches,
    }


def check_required_files(workdir, required):
    """Score = (files present) / (files required)."""
    from pathlib import Path
    workdir = Path(workdir)
    missing = [f for f in required if not (workdir / f).exists()]
    return {
        "required": required,
        "missing": missing,
        "score": 1.0 - (len(missing) / len(required)) if required else 1.0,
    }
