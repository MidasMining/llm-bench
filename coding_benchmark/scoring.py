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


def check_plan(messages):
    """Look at the first non-empty assistant text and count numbered/bulleted plan items.

    Heuristic — we can't mechanically grade "did the model follow its plan,"
    but we CAN see whether a plan was emitted and whether it matches the
    requested 3-5 step shape.
    """
    first_text = None
    for m in messages:
        if m.get("role") == "assistant" and m.get("content"):
            first_text = m["content"]
            break

    if not first_text:
        return {"score": 0.0, "plan_items": [], "note": "no opening text from model"}

    items = []
    for line in first_text.split("\n"):
        m = re.match(r"^\s*(?:\d+[\.\)]|[-*])\s+(.+)$", line)
        if m:
            items.append(m.group(1).strip())

    if 3 <= len(items) <= 8:
        score = 1.0
    elif 1 <= len(items) < 3:
        score = 0.5
    else:
        score = 0.0

    return {"plan_count": len(items), "plan_items": items[:10], "score": score}


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
