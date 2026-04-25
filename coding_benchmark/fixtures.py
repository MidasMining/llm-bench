"""Synthetic BAM fixtures for the coding benchmark.

make_test_bam() emits a deterministic ~100-read BAM across chr1/chr2/chr3
with varied MAPQ and flag combinations so every scenario's filter
discriminates meaningfully.

make_gold() applies a scenario's filter_fn to produce a sorted+indexed
reference output for diff-based scoring.
"""

import os
import random


def make_test_bam(path, n_reads=100, seed=42):
    import pysam
    rng = random.Random(seed)
    header = {
        "HD": {"VN": "1.6", "SO": "unsorted"},
        "SQ": [
            {"SN": "chr1", "LN": 1_000_000},
            {"SN": "chr2", "LN": 1_000_000},
            {"SN": "chr3", "LN": 1_000_000},
        ],
    }
    flag_choices = [0x2, 0x2 | 0x400, 0x0, 0x10, 0x10 | 0x2, 0x400, 0x2 | 0x10]
    mapq_choices = [5, 10, 20, 30, 40, 60]

    with pysam.AlignmentFile(path, "wb", header=header) as out:
        for i in range(n_reads):
            a = pysam.AlignedSegment()
            a.query_name = f"read_{i:04d}"
            a.reference_id = rng.choice([0, 1, 2])
            a.reference_start = rng.randint(0, 999_000)
            a.mapping_quality = rng.choice(mapq_choices)
            a.flag = rng.choice(flag_choices)
            a.query_sequence = "ACGT" * 25
            a.query_qualities = pysam.qualitystring_to_array("I" * 100)
            a.cigarstring = "100M"
            out.write(a)


def make_gold(input_path, gold_path, filter_fn):
    """Apply filter_fn to input_path, write coordinate-sorted+indexed gold BAM."""
    import pysam
    unsorted = gold_path + ".unsorted.bam"
    with pysam.AlignmentFile(input_path, "rb") as in_, \
            pysam.AlignmentFile(unsorted, "wb", template=in_) as out:
        for r in in_:
            if filter_fn(r):
                out.write(r)
    pysam.sort("-o", gold_path, unsorted)
    pysam.index(gold_path)
    os.unlink(unsorted)
