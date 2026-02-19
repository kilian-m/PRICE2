"""Coverage model for Ribo-seq data.

Models the elevated ribosome footprint density at ORF start codons and at
the codon immediately upstream of stop codons, relative to average ORF-body
coverage.  The resulting scale factors are used to weight the expected
coverage profile when deconvolving overlapping ORFs.
"""

import warnings
from enum import Enum
from typing import Optional

import HTSeq
import numpy as np

from price2.cleavage_model import CleavageModel, read_in_cds_likelihood
from price2.reference_annotation import ReferenceAnnotation
from price2.ribo_seq_alignment import RiboSeqAlignment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Size of the accumulation arrays used while building the P-site histograms.
_HIST_SIZE: int = 300

# Index in the start-histogram that corresponds to CDS position 0 (start codon).
_START_CODON_IDX: int = 30

# Slice over the ORF body in the start-histogram (positions 3, 6, …, 198 of CDS).
_START_BODY_SLICE: slice = slice(33, 230, 3)

# Index in the stop-histogram that corresponds to the last sense codon before
# the stop codon (CDS position len(cds) − 3).
_STOP_PEAK_IDX: int = 217

# Offset applied when filling the stop-histogram:
#   index = p_site_cds_pos − len(cds) + _STOP_HIST_OFFSET
# so that CDS position len(cds) maps to index _STOP_HIST_OFFSET.
_STOP_HIST_OFFSET: int = 220

# Slice over the ORF body in the stop-histogram.
_STOP_BODY_SLICE: slice = slice(_STOP_PEAK_IDX - 198, _STOP_PEAK_IDX, 3)

# Minimum number of reads required for reliable factor estimation.
_MIN_READS: int = 100

# A read is assigned to a P-site only if the most likely codon carries at
# least this fraction of the total per-codon likelihood.
_MIN_DOMINANT_FRACTION: float = 0.8

# Minimum absolute likelihood for any codon to be considered.
_MIN_CODON_LIKELIHOOD: float = 0.01

# Window (in CDS positions) around the start codon used to select reads.
_START_WINDOW: tuple[int, int] = (-30, 200)

# Window (in CDS positions relative to CDS end) used to select reads for
# the stop-codon histogram.
_STOP_WINDOW: tuple[int, int] = (-200, 30)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


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
    # Accept only reads that map to exactly one unique CDS interval.
    cds_intervals: set[tuple[int, int]] = set()
    transcript = None
    for tr in transcript_candidates:
        try:
            iv_on_cds = tr.cds.induce(aln.genomic_region)
        except ValueError:
            continue
        cds_intervals.add(iv_on_cds)
        transcript = tr

    if len(cds_intervals) != 1:
        return None

    iv_on_cds = cds_intervals.pop()
    n_codons = (iv_on_cds[1] - iv_on_cds[0]) // 3
    if n_codons == 0:
        return None

    frame = iv_on_cds[0] % 3

    # Compute the likelihood that the P-site lies in each codon spanned by
    # the read.
    likelihoods = np.array(
        [
            read_in_cds_likelihood(
                cm.pl,
                cm.pr,
                cm.pu,
                len(aln),
                frame,
                aln.untemplated_addition,
                frame % 3 + 3 * i,
                frame % 3 + 3 * i + 3,
            )
            for i in range(n_codons)
        ]
    )

    if likelihoods.max() < _MIN_CODON_LIKELIHOOD:
        return None

    likelihoods /= likelihoods.sum()

    if likelihoods.max() < _MIN_DOMINANT_FRACTION:
        return None

    p_site_cds_pos = iv_on_cds[0] + int(np.argmax(likelihoods)) * 3
    return transcript, iv_on_cds, p_site_cds_pos


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------


class CoveragePosition(Enum):
    """Position category of a codon relative to its ORF.

    Members
    -------
    start
        The start (initiator) codon.
    middle
        Any codon in the ORF body.
    stop
        The codon immediately upstream of the stop codon.
    """

    start = 0
    middle = 1
    stop = 2


class CoverageModel:
    """Position-specific ribosome footprint enrichment model.

    Estimates coverage scale factors at the start codon and at the last
    sense codon before the stop codon, relative to the average coverage over
    the ORF body.  Both factors are derived from annotated CDS regions in
    *ra* and the cleavage model *cm*.

    Parameters
    ----------
    ra : ReferenceAnnotation
        Parsed reference annotation.
    sample_bam_path : str
        Path to the BAM file for the Ribo-seq sample.
    cm : CleavageModel
        Cleavage model used to assign reads to P-site positions.

    Attributes
    ----------
    start_factor : float
        Enrichment at the start codon relative to the ORF body.  Always >= 1.
    stop_factor : float
        Enrichment one codon upstream of the stop codon relative to the ORF
        body.  Always >= 1.
    """

    def __init__(
        self,
        ra: ReferenceAnnotation,
        sample_bam_path: str,
        cm: CleavageModel,
    ) -> None:
        start_hist = self._build_start_histogram(ra, sample_bam_path, cm)
        stop_hist = self._build_stop_histogram(ra, sample_bam_path, cm)
        self.start_factor = self._compute_start_factor(start_hist, sample_bam_path)
        self.stop_factor = self._compute_stop_factor(stop_hist, sample_bam_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_start_histogram(
        ra: ReferenceAnnotation,
        bam_path: str,
        cm: CleavageModel,
    ) -> np.ndarray:
        """Accumulate P-site counts relative to the start codon.

        Only reads whose P-site falls within ``_START_WINDOW`` of CDS
        position 0, and that do not span the CDS end, are included.

        Parameters
        ----------
        ra : ReferenceAnnotation
            Reference annotation.
        bam_path : str
            Path to the BAM file.
        cm : CleavageModel
            Cleavage model.

        Returns
        -------
        np.ndarray
            Histogram of shape ``(_HIST_SIZE,)``.  Index
            ``_START_CODON_IDX`` corresponds to CDS position 0.
        """
        hist = np.zeros(_HIST_SIZE)
        for raw_aln in HTSeq.BAM_Reader(bam_path):
            result = _try_assign_p_site(RiboSeqAlignment(raw_aln), ra, cm)
            if result is None:
                continue
            transcript, iv_on_cds, p_site = result

            lo, hi = _START_WINDOW
            if not (lo < iv_on_cds[0] < hi):
                continue

            # Exclude reads whose matching range spans the CDS end.
            if iv_on_cds[0] < len(transcript.cds) < iv_on_cds[1]:
                continue

            hist[p_site + _START_CODON_IDX] += 1

        return hist

    @staticmethod
    def _build_stop_histogram(
        ra: ReferenceAnnotation,
        bam_path: str,
        cm: CleavageModel,
    ) -> np.ndarray:
        """Accumulate P-site counts relative to the stop codon.

        Only reads whose P-site falls within ``_STOP_WINDOW`` of the CDS end,
        and that do not span the CDS start, are included.

        Parameters
        ----------
        ra : ReferenceAnnotation
            Reference annotation.
        bam_path : str
            Path to the BAM file.
        cm : CleavageModel
            Cleavage model.

        Returns
        -------
        np.ndarray
            Histogram of shape ``(_HIST_SIZE,)``.  Index ``_STOP_PEAK_IDX``
            corresponds to CDS position ``len(cds) − 3``.
        """
        hist = np.zeros(_HIST_SIZE)
        for raw_aln in HTSeq.BAM_Reader(bam_path):
            result = _try_assign_p_site(RiboSeqAlignment(raw_aln), ra, cm)
            if result is None:
                continue
            transcript, iv_on_cds, p_site = result

            dist_to_end = iv_on_cds[1] - len(transcript.cds)
            lo, hi = _STOP_WINDOW
            if not (lo < dist_to_end < hi):
                continue

            # Exclude reads whose matching range spans the CDS start.
            if iv_on_cds[0] < 0 < iv_on_cds[1]:
                continue

            hist[p_site - len(transcript.cds) + _STOP_HIST_OFFSET] += 1

        return hist

    @staticmethod
    def _compute_start_factor(hist: np.ndarray, bam_path: str) -> float:
        """Compute the start-codon enrichment factor from *hist*.

        Parameters
        ----------
        hist : np.ndarray
            Start-codon P-site histogram as returned by
            :meth:`_build_start_histogram`.
        bam_path : str
            BAM path used only for warning messages.

        Returns
        -------
        float
            Enrichment factor >= 1.
        """
        peak_count = hist[_START_CODON_IDX]
        body_counts = hist[_START_BODY_SLICE]

        if peak_count < _MIN_READS:
            warnings.warn(
                f"Only {int(peak_count)} reads at start codon position. "
                "Low evidence for coverage model."
            )
        if body_counts.sum() < _MIN_READS:
            warnings.warn(
                f"Only {int(body_counts.sum())} reads at middle codon positions. "
                f"Low evidence for coverage model.  sample: {bam_path}"
            )

        with np.errstate(divide="raise"):
            try:
                factor = float(peak_count / body_counts.mean())
            except (FloatingPointError, ZeroDivisionError):
                factor = 1.0

        return max(1.0, factor)

    @staticmethod
    def _compute_stop_factor(hist: np.ndarray, bam_path: str) -> float:
        """Compute the stop-codon enrichment factor from *hist*.

        Parameters
        ----------
        hist : np.ndarray
            Stop-codon P-site histogram as returned by
            :meth:`_build_stop_histogram`.
        bam_path : str
            BAM path used only for warning messages.

        Returns
        -------
        float
            Enrichment factor >= 1.
        """
        peak_count = hist[_STOP_PEAK_IDX]
        body_counts = hist[_STOP_BODY_SLICE]

        if peak_count < _MIN_READS:
            warnings.warn(
                f"Only {int(peak_count)} reads at stop codon position. "
                f"Low evidence for coverage model.  sample: {bam_path}"
            )
        if body_counts.sum() < _MIN_READS:
            warnings.warn(
                f"Only {int(body_counts.sum())} reads at middle codon positions. "
                f"Low evidence for coverage model.  sample: {bam_path}"
            )

        with np.errstate(divide="raise"):
            try:
                factor = float(peak_count / body_counts.mean())
            except (FloatingPointError, ZeroDivisionError):
                factor = 1.0

        return max(1.0, factor)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_coverage_factor(self, position: CoveragePosition) -> float:
        """Return the coverage scale factor for *position*.

        Parameters
        ----------
        position : CoveragePosition
            Position category (start, middle, or stop).

        Returns
        -------
        float
            ``start_factor`` for ``CoveragePosition.start``,
            ``stop_factor`` for ``CoveragePosition.stop``, and
            ``1.0`` for ``CoveragePosition.middle``.
        """
        if position == CoveragePosition.start:
            return self.start_factor
        if position == CoveragePosition.stop:
            return self.stop_factor
        return 1.0
