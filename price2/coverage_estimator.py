"""Estimating a coverage model from a Ribo-seq BAM.

Assigns each uniquely mapping read to a P-site codon on the single coding
transcript it projects onto (the cleavage model answers which codon,
:meth:`~price2.cleavage_model.CleavageModel.p_site_codon`), and accumulates
the P-site histograms around the CDS start and stop codons that
:meth:`~price2.coverage_model.CoverageModel.from_histograms` turns into
enrichment factors.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pysam

from price2.bam import is_unique, iter_mapped
from price2.cleavage_model import CleavageModel
from price2.coverage_model import (
    HIST_SIZE,
    START_CODON_IDX,
    STOP_HIST_OFFSET,
)
from price2.genomic_features import Transcript
from price2.reference_annotation import ReferenceAnnotation
from price2.ribo_seq_alignment import RiboSeqAlignment

# Window (in CDS positions) around the start codon used to select reads.
_START_WINDOW: tuple[int, int] = (-30, 330)

# Window (in CDS positions relative to CDS end) used to select reads for
# the stop-codon histogram.
_STOP_WINDOW: tuple[int, int] = (-330, 30)


class PSiteAssignment(NamedTuple):
    """Where a read sits on the one coding transcript it was assigned to."""

    #: The coding transcript.
    transcript: Transcript
    #: ``(start, end)`` of the read in CDS coordinates (0-based, half-open).
    iv_on_cds: tuple[int, int]
    #: CDS coordinate of the inferred P-site.
    p_site: int


def _try_assign_p_site(
    aln: RiboSeqAlignment,
    ra: ReferenceAnnotation,
    cm: CleavageModel,
) -> PSiteAssignment | None:
    """Attempt to assign *aln* to a unique P-site on a coding transcript.

    Parameters
    ----------
    aln : RiboSeqAlignment
        Ribo-seq alignment to process.
    ra : ReferenceAnnotation
        Reference annotation used to look up overlapping coding transcripts.
    cm : CleavageModel
        Cleavage model providing per-codon likelihoods.

    Returns
    -------
    PSiteAssignment or None
        The assignment when the read maps onto exactly one CDS and the
        cleavage model names a dominant P-site codon; ``None`` otherwise.
    """
    transcript_candidates = ra.collect_coding_transcripts(aln.genomic_region)
    if not transcript_candidates:
        return None

    # Project the read onto CDS coordinates for every candidate transcript.
    # Accept only reads that map to exactly one unique CDS interval; a second
    # successful projection already disqualifies the read.
    transcript = None
    iv_on_cds = None
    for tr in transcript_candidates:
        iv_on_exons = tr.exons.try_map_to_local(aln.genomic_region)
        if iv_on_exons is None:
            continue
        if transcript is not None:
            return None
        transcript = tr
        iv_on_cds = (
            iv_on_exons[0] - tr.annotated_cds_iv[0],
            iv_on_exons[1] - tr.annotated_cds_iv[0],
        )

    if transcript is None:
        return None

    frame = iv_on_cds[0] % 3
    winner = cm.p_site_codon(len(aln), frame, aln.untemplated_addition)
    if winner is None:
        return None

    p_site_cds_pos = iv_on_cds[0] + (-frame) % 3 + winner * 3
    return PSiteAssignment(transcript, iv_on_cds, p_site_cds_pos)


def build_histograms(
    ra: ReferenceAnnotation,
    bam: pysam.AlignmentFile,
    cm: CleavageModel,
    region: tuple[str, int, int] | None = None,
    end_to_end: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate P-site counts around the CDS start and stop codons.

    Both histograms are filled in a single pass: assigning a read to a P-site
    is by far the most expensive step and its result serves both.

    A read contributes to the start histogram when its P-site falls within
    :data:`_START_WINDOW` of CDS position 0 and its matching range does not
    span the CDS end, and to the stop histogram when its P-site falls within
    :data:`_STOP_WINDOW` of the CDS end and its matching range does not span
    the CDS start.  The two conditions are not exclusive.

    Only uniquely mapping reads (``NH`` tag equal to 1, or absent) are
    counted, as in :class:`~price2.cleavage_model.CleavageModel`.

    Parameters
    ----------
    ra : ReferenceAnnotation
        Reference annotation.
    bam : pysam.AlignmentFile
        Open, coordinate-sorted BAM file.
    cm : CleavageModel
        Cleavage model used to assign reads to P-site positions.
    region : tuple[str, int, int] or None, optional
        Restrict the pass to ``(contig, start, end)``.  Requires *bam* to be
        indexed.  Only reads whose leftmost mapped base lies in ``[start, end)``
        are counted, so the histograms of a set of regions tiling the genome sum
        to those of the whole file.  When *None* the whole file is scanned.
    end_to_end : bool, optional
        When ``True`` the BAM was mapped with ``--alignEndsType EndToEnd``;
        the untemplated addition is recovered from the 5'-terminal mismatch
        instead of a soft-clip (see :func:`price2.bam.footprint`).

    Returns
    -------
    start_hist : np.ndarray
        Histogram of shape ``(HIST_SIZE,)``; index :data:`START_CODON_IDX`
        corresponds to CDS position 0.
    stop_hist : np.ndarray
        Histogram of shape ``(HIST_SIZE,)``; index :data:`STOP_PEAK_IDX`
        corresponds to CDS position ``len(cds) − 3``.
    """
    start_hist = np.zeros(HIST_SIZE)
    stop_hist = np.zeros(HIST_SIZE)
    start_lo, start_hi = _START_WINDOW
    stop_lo, stop_hi = _STOP_WINDOW

    for raw_aln in iter_mapped(bam, region):
        # Multimapping reads would contribute the same footprint to every
        # locus they align to, biasing the metagene profile towards whatever
        # is repeated in the genome.  Skipped before the alignment is built,
        # as the cleavage model does with its own reads.
        if not is_unique(raw_aln):
            continue

        aln = RiboSeqAlignment.from_pysam(raw_aln, end_to_end=end_to_end)
        if aln is None:
            continue
        assigned = _try_assign_p_site(aln, ra, cm)
        if assigned is None:
            continue
        iv_on_cds, p_site = assigned.iv_on_cds, assigned.p_site
        coding_length = assigned.transcript.coding_length

        if start_lo < iv_on_cds[0] < start_hi and not (
            iv_on_cds[0] < coding_length < iv_on_cds[1]
        ):
            idx = p_site // 3 + START_CODON_IDX
            if 0 <= idx < HIST_SIZE:
                start_hist[idx] += 1

        dist_to_end = iv_on_cds[1] - coding_length
        if stop_lo < dist_to_end < stop_hi and not (iv_on_cds[0] < 0 < iv_on_cds[1]):
            idx = p_site // 3 - coding_length // 3 + STOP_HIST_OFFSET
            if 0 <= idx < HIST_SIZE:
                stop_hist[idx] += 1

    return start_hist, stop_hist
