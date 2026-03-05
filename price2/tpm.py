"""TPM (Transcripts Per Million) conversion for PRICE2 output.

Reads activity-based result files produced by the ORF deconvolution
pipeline and writes corresponding files with TPM-normalised values.
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_tpm_output(o_dir: str) -> None:
    """Convert activity values to TPM and write a parallel TSV file.

    Reads ``final_orfs.tsv`` (and, if present, ``final_regions.tsv``)
    from ``<o_dir>/regions_activities/``, converts activity values
    (reads per nucleotide) to Transcripts Per Million (TPM) per run,
    and writes the results to ``final_orfs_tpm.tsv`` (and
    ``final_regions_tpm.tsv``).

    TPM for region *i* in run *j* is defined as::

        TPM_ij = (activity_ij / sum_i(activity_ij)) * 1e6

    Parameters
    ----------
    o_dir : str
        Path to the PRICE2 output directory.
    """
    ra_dir = os.path.join(o_dir, "regions_activities")
    meta_cols = [
        "orf_id",
        "gene_id",
        "transcript_id",
        "locus_id",
        "genomic_region",
        "orf_type",
    ]
    region_meta_cols = [
        "region_id",
        "gene_id",
        "transcript_id",
        "locus_id",
        "genomic_region",
        "orf_type",
    ]

    for suffix, id_col, cols in [
        ("orfs", "orf_id", meta_cols),
        ("regions", "region_id", region_meta_cols),
    ]:
        in_path = os.path.join(ra_dir, f"final_{suffix}.tsv")
        out_path = os.path.join(ra_dir, f"final_{suffix}_tpm.tsv")

        if not os.path.exists(in_path):
            continue

        df = pd.read_csv(in_path, sep="\t")
        run_cols = [c for c in df.columns if c not in cols]

        if not run_cols:
            continue

        activities = df[run_cols].values.astype(np.float64)
        col_sums = activities.sum(axis=0)
        col_sums[col_sums == 0] = 1.0  # avoid division by zero
        tpm = (activities / col_sums) * 1e6

        df_tpm = df[cols].copy()
        df_tpm[run_cols] = tpm

        df_tpm.to_csv(out_path, sep="\t", index=False, float_format="%.2e")
