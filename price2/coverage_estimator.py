"""Estimating a coverage model from a Ribo-seq BAM.

Assigns each uniquely mapping read to a P-site codon on the single coding
transcript it projects onto, using the cleavage model's per-codon
likelihoods, and accumulates the P-site histograms around the CDS start and
stop codons that :meth:`~price2.coverage_model.CoverageModel.from_histograms`
turns into enrichment factors.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pysam

from price2.bam import is_unique, iter_mapped
from price2.cleavage_model import CleavageModel, read_in_cds_likelihood
from price2.coverage_model import (
    HIST_SIZE,
    START_CODON_IDX,
    STOP_HIST_OFFSET,
)
from price2.reference_annotation import ReferenceAnnotation
from price2.ribo_seq_alignment import RiboSeqAlignment

# A read is assigned to a P-site only if the most likely codon carries at
# least this fraction of the total per-codon likelihood.
_MIN_DOMINANT_FRACTION: float = 0.8

# Minimum absolute likelihood for any codon to be considered.
_MIN_CODON_LIKELIHOOD: float = 0.01

# Window (in CDS positions) around the start codon used to select reads.
_START_WINDOW: tuple[int, int] = (-30, 330)

# Window (in CDS positions relative to CDS end) used to select reads for
# the stop-codon histogram.
_STOP_WINDOW: tuple[int, int] = (-330, 30)

#: Sentinel stored in the P-site table for reads that carry no dominant codon.
_REJECT: int = -1


def _p_site_table(cm: CleavageModel) -> list[Optional[int]]:
    """Return the memo table for *cm*, creating it on first use.

    The table is indexed by ``(length * 3 + frame) * 2 + oua`` and holds the
    winning codon index, :data:`_REJECT`, or ``None`` when the entry has not
    been computed yet.  Its length bounds the read length at
    ``len(pl) + len(pr) + 4``, the longest read the cleavage model can produce
    (see :class:`~price2.cleavage_model.CleavageModel`); longer reads have zero
    likelihood under the model.
    """
    try:
        return cm._p_site_table_cache
    except AttributeError:
        max_len = len(cm.pl) + len(cm.pr) + 4
        table: list[Optional[int]] = [None] * (max_len * 6)
        cm._p_site_table_cache = table
        return table


def _compute_p_site_codon(
    cm: CleavageModel, length: int, frame: int, oua: bool
) -> int:
    """Return the index of the codon carrying the P-site, or :data:`_REJECT`.

    The per-codon likelihood vector of a read depends only on its matching
    length, its reading frame and whether it carries an untemplated addition —
    never on *where* the read sits.  Both acceptance criteria
    (:data:`_MIN_CODON_LIKELIHOOD`, :data:`_MIN_DOMINANT_FRACTION`) and the
    winning codon are therefore functions of that triple alone.

    Relative to the *read start* the CDS codon boundaries sit at ``f0``,
    ``f0 + 3``, ... with ``f0 = (-frame) % 3``, which is the offset convention
    :func:`read_in_cds_likelihood` uses internally.  Only codons that fit
    entirely inside the read can carry the P-site.
    """
    f0 = (-frame) % 3
    n_codons = (length - f0) // 3
    if n_codons <= 0:
        return _REJECT

    likelihoods = np.array(
        [
            read_in_cds_likelihood(
                cm.pl, cm.pr, cm.pu, length, frame, oua, f0 + 3 * i, f0 + 3 * i + 3
            )
            for i in range(n_codons)
        ]
    )

    if likelihoods.max() < _MIN_CODON_LIKELIHOOD:
        return _REJECT

    likelihoods /= likelihoods.sum()

    if likelihoods.max() < _MIN_DOMINANT_FRACTION:
        return _REJECT

    return int(np.argmax(likelihoods))


def _p_site_codon(cm: CleavageModel, length: int, frame: int, oua: int) -> int:
    """Memoised :func:`_compute_p_site_codon`."""
    table = _p_site_table(cm)
    key = (length * 3 + frame) * 2 + oua
    if key >= len(table):
        # Longer than any read the cleavage model can generate.
        return _REJECT
    winner = table[key]
    if winner is None:
        winner = _compute_p_site_codon(cm, length, frame, bool(oua))
        table[key] = winner
    return winner


def _try_assign_p_site(
    aln: RiboSeqAlignment,
    ra: ReferenceAnnotation,
    cm: CleavageModel,
) -> Optional[tuple[object, tuple[int, int], int]]:
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
    tuple or None
        ``(transcript, iv_on_cds, p_site_cds_pos)`` when the read can be
        unambiguously assigned to a single CDS interval and a dominant P-site
        position; ``None`` otherwise.

        * *transcript* – the coding transcript the read was assigned to.
        * *iv_on_cds* – ``(start, end)`` of the read projected onto CDS
          coordinates (0-based, half-open).
        * *p_site_cds_pos* – CDS coordinate of the inferred P-site.
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
        try:
            iv_on_exons = tr.exons.map_to_local(aln.genomic_region)
        except ValueError:
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
    winner = _p_site_codon(cm, len(aln), frame, int(aln.untemplated_addition))
    if winner == _REJECT:
        return None

    p_site_cds_pos = iv_on_cds[0] + (-frame) % 3 + winner * 3
    return transcript, iv_on_cds, p_site_cds_pos


def build_histograms(
    ra: ReferenceAnnotation,
    bam: pysam.AlignmentFile,
    cm: CleavageModel,
    region: Optional[tuple[str, int, int]] = None,
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
        instead of a soft-clip (see :meth:`RiboSeqAlignment.from_pysam`).

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

        result = _try_assign_p_site(
            RiboSeqAlignment.from_pysam(raw_aln, end_to_end=end_to_end), ra, cm
        )
        if result is None:
            continue
        transcript, iv_on_cds, p_site = result

        coding_length = transcript.coding_length

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
