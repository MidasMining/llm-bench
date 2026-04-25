"""Scenario definitions for the coding benchmark.

Each scenario describes a BAM filtering task at a specific difficulty tier.
The runner generates a deterministic synthetic input BAM, applies the
scenario's filter_fn to produce a gold BAM, then asks the model to reproduce
the filter using pysam.
"""


SCENARIOS = {
    "t1-mapq": {
        "name": "T1: MAPQ filter only",
        "filter_description": "MAPQ score >= 30",
        "filter_fn": lambda r: r.mapping_quality >= 30,
        "extra_requirements": (
            "The output BAM must be coordinate-sorted and indexed "
            "(an `output.bam.bai` file must exist next to it)."
        ),
        "required_output_files": ["output.bam", "output.bam.bai"],
    },
    "t2-flag-chrom": {
        "name": "T2: MAPQ + flag mask + chromosome subset + stats",
        "filter_description": (
            "ALL of these must be true for a read to pass:\n"
            "  - MAPQ score >= 30\n"
            "  - Read is properly paired (flag has bit 0x2 set)\n"
            "  - Read is NOT a PCR/optical duplicate (flag does NOT have bit 0x400 set)\n"
            "  - Read maps to chromosome `chr1` or `chr2`"
        ),
        "filter_fn": lambda r: (
            r.mapping_quality >= 30
            and bool(r.flag & 0x2)
            and not bool(r.flag & 0x400)
            and r.reference_name in ("chr1", "chr2")
        ),
        "extra_requirements": (
            "The output BAM must be coordinate-sorted and indexed.\n"
            "Also write a file `output.stats.txt` containing exactly two lines:\n"
            "  input_count: <total reads in the input BAM>\n"
            "  output_count: <reads in your filtered output>\n"
        ),
        "required_output_files": ["output.bam", "output.bam.bai", "output.stats.txt"],
    },
}
