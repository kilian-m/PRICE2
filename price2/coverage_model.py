"""Coverage model for Ribo-seq data.

Models the elevated ribosome footprint density at ORF start codons and at
the codon immediately upstream of stop codons, relative to average ORF-body
coverage.  The resulting scale factors are used to weight the expected
coverage profile when deconvolving overlapping ORFs.
"""

import logging
import warnings
from enum import Enum
from typing import Optional

import matplotlib.pyplot as plt
import pysam
import numpy as np
from scipy.stats import trim_mean

from price2.cleavage_model import CleavageModel, read_in_cds_likelihood
from price2.reference_annotation import ReferenceAnnotation
from price2.ribo_seq_alignment import RiboSeqAlignment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Size of the accumulation arrays used while building the P-site histograms.
# Each bin corresponds to one codon (3 nt).
_HIST_SIZE: int = 120

# Index in the start-histogram that corresponds to CDS position 0 (start codon).
_START_CODON_IDX: int = 10

# Slice over the ORF body in the start-histogram (codons 1 .. 100 of CDS).
_START_BODY_SLICE: slice = slice(11, 111)

# Index in the stop-histogram that corresponds to the last sense codon before
# the stop codon (CDS position len(cds) − 3).
_STOP_PEAK_IDX: int = 110

# Offset applied when filling the stop-histogram:
#   index = p_site_cds_pos // 3 − len(cds) // 3 + _STOP_HIST_OFFSET
# so that CDS codon len(cds)//3 maps to index _STOP_HIST_OFFSET.
_STOP_HIST_OFFSET: int = 111

# Slice over the ORF body in the stop-histogram.
_STOP_BODY_SLICE: slice = slice(_STOP_PEAK_IDX - 100, _STOP_PEAK_IDX)

# Minimum number of reads required for reliable factor estimation.
_MIN_READS: int = 100

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
    cds_intervals: list[tuple[int, int]] = []
    transcripts = []
    for tr in transcript_candidates:
        try:
            iv_on_exons = tr.exons.map_to_local(aln.genomic_region)
            iv_on_cds = (
                iv_on_exons[0] - tr.annotated_cds_iv[0],
                iv_on_exons[1] - tr.annotated_cds_iv[0],
            )
        except ValueError:
            continue
        cds_intervals.append(iv_on_cds)
        transcripts.append(tr)

    if len(cds_intervals) != 1:
        return None

    iv_on_cds = cds_intervals[0]
    transcript = transcripts[0]
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

    Models elevated ribosome footprint density at ORF start codons and at
    the codon immediately upstream of stop codons, relative to average
    ORF-body coverage.

    Parameters
    ----------
    start_factor : float
        Enrichment at the start codon relative to the ORF body.  Always >= 1.
    stop_factor : float
        Enrichment one codon upstream of the stop codon relative to the ORF
        body.  Always >= 1.
    start_hist : np.ndarray or None, optional
        P-site count histogram around the start codon.  Optional; only
        needed for plotting.
    stop_hist : np.ndarray or None, optional
        P-site count histogram around the stop codon.  Optional; only
        needed for plotting.
    """

    def __init__(
        self,
        start_factor: float,
        stop_factor: float,
        start_hist: Optional[np.ndarray] = None,
        stop_hist: Optional[np.ndarray] = None,
    ) -> None:
        self.start_factor = start_factor
        self.stop_factor = stop_factor
        if start_hist is not None:
            self.start_hist = start_hist
        if stop_hist is not None:
            self.stop_hist = stop_hist

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_bam(
        cls,
        ra: ReferenceAnnotation,
        sample_bam_path: str,
        cm: CleavageModel,
    ) -> "CoverageModel":
        """Estimate a coverage model from a BAM file.

        Parameters
        ----------
        ra : ReferenceAnnotation
            Parsed reference annotation.
        sample_bam_path : str
            Path to the BAM file for the Ribo-seq sample.
        cm : CleavageModel
            Cleavage model used to assign reads to P-site positions.

        Returns
        -------
        CoverageModel
            Model with all attributes (including optional histograms) set.
        """
        start_hist = cls._build_start_histogram(ra, sample_bam_path, cm)
        stop_hist = cls._build_stop_histogram(ra, sample_bam_path, cm)
        start_factor = cls._compute_start_factor(start_hist, sample_bam_path)
        stop_factor = cls._compute_stop_factor(stop_hist, sample_bam_path)
        return cls(start_factor, stop_factor, start_hist, stop_hist)

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
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for raw_aln in bam.fetch(until_eof=True):
                if raw_aln.is_unmapped:
                    continue
                result = _try_assign_p_site(
                    RiboSeqAlignment.from_pysam(raw_aln), ra, cm
                )
                if result is None:
                    continue
                transcript, iv_on_cds, p_site = result

                lo, hi = _START_WINDOW
                if not (lo < iv_on_cds[0] < hi):
                    continue

                # Exclude reads whose matching range spans the CDS end.
                if iv_on_cds[0] < transcript.coding_length < iv_on_cds[1]:
                    continue

                idx = p_site // 3 + _START_CODON_IDX
                if 0 <= idx < _HIST_SIZE:
                    hist[idx] += 1

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
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for raw_aln in bam.fetch(until_eof=True):
                if raw_aln.is_unmapped:
                    continue
                result = _try_assign_p_site(
                    RiboSeqAlignment.from_pysam(raw_aln), ra, cm
                )
                if result is None:
                    continue
                transcript, iv_on_cds, p_site = result

                dist_to_end = iv_on_cds[1] - transcript.coding_length
                lo, hi = _STOP_WINDOW
                if not (lo < dist_to_end < hi):
                    continue

                # Exclude reads whose matching range spans the CDS start.
                if iv_on_cds[0] < 0 < iv_on_cds[1]:
                    continue

                idx = p_site // 3 - transcript.coding_length // 3 + _STOP_HIST_OFFSET
                if 0 <= idx < _HIST_SIZE:
                    hist[idx] += 1

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
            logger.warning(
                "Only %d reads at start codon position. "
                "Low evidence for coverage model.",
                int(peak_count),
                bam_path,
            )
        if body_counts.sum() < _MIN_READS:
            logger.warning(
                "Only %d reads at middle codon positions. "
                "Low evidence for coverage model.  sample: %s",
                int(body_counts.sum()),
                bam_path,
            )

        with np.errstate(divide="raise"):
            try:
                factor = float(
                    peak_count / trim_mean(body_counts, proportiontocut=0.25)
                )
            except (FloatingPointError, ValueError):
                factor = 1.0
            except ZeroDivisionError:
                factor = 1000.0

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
            logger.warning(
                "Only %d reads at stop codon position. "
                "Low evidence for coverage model.  sample: %s",
                int(peak_count),
                bam_path,
            )
        if body_counts.sum() < _MIN_READS:
            logger.warning(
                "Only %d reads at middle codon positions. "
                "Low evidence for coverage model.  sample: %s",
                int(body_counts.sum()),
                bam_path,
            )

        with np.errstate(divide="raise"):
            try:
                factor = float(
                    peak_count / trim_mean(body_counts, proportiontocut=0.25)
                )
            except (FloatingPointError, ValueError):
                factor = 1.0
            except ZeroDivisionError:
                factor = 1000.0

        return max(1.0, factor)

    #: TSV header line produced by :meth:`to_files`.
    TSV_HEADER: str = "dataset_id\tstart_factor\tstop_factor"

    def to_files(
        self,
        dataset_id: str,
        tsv_fh,
        npz_data: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        """Write the model to open file handles.

        Appends one TSV line with the obligatory attributes to *tsv_fh*.
        If *npz_data* is provided, optional arrays (``start_hist``,
        ``stop_hist``) are added to the dict for later ``np.savez``.

        Parameters
        ----------
        dataset_id : str
            Sample identifier placed in the first TSV column.
        tsv_fh : file-like
            Writable text file handle (header already written).
        npz_data : dict[str, np.ndarray] or None, optional
            Accumulator dict for optional arrays.
        """
        tsv_fh.write(f"{dataset_id}\t{self.start_factor:.6g}\t{self.stop_factor:.6g}\n")

        if npz_data is not None:
            if hasattr(self, "start_hist"):
                npz_data[f"{dataset_id}_start_hist"] = self.start_hist
            if hasattr(self, "stop_hist"):
                npz_data[f"{dataset_id}_stop_hist"] = self.stop_hist

    @classmethod
    def from_files(
        cls, tsv_path: str, npz_path: Optional[str] = None
    ) -> "dict[str, CoverageModel]":
        """Load models from a TSV file and optionally an NPZ file.

        Parameters
        ----------
        tsv_path : str
            Path to the ``coverage_models.tsv`` file (obligatory
            attributes: ``start_factor``, ``stop_factor``).
        npz_path : str or None, optional
            Path to the ``coverage_models.npz`` file.  When provided,
            optional attributes (``start_hist``, ``stop_hist``) are
            attached to the corresponding models.

        Returns
        -------
        dict[str, CoverageModel]
            Mapping of dataset identifier to the reconstructed model.
        """
        models: dict[str, CoverageModel] = {}
        with open(tsv_path) as fh:
            next(fh)  # skip header
            for line in fh:
                parts = line.strip().split("\t")
                dataset_id = parts[0]
                models[dataset_id] = cls(float(parts[1]), float(parts[2]))

        if npz_path is not None:
            data = np.load(npz_path)
            for dataset_id, model in models.items():
                key_sh = f"{dataset_id}_start_hist"
                key_eh = f"{dataset_id}_stop_hist"
                if key_sh in data:
                    model.start_hist = data[key_sh]
                if key_eh in data:
                    model.stop_hist = data[key_eh]

        return models

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_histograms(
        cls,
        start_hist: np.ndarray,
        stop_hist: np.ndarray,
        run_id: str = "",
    ) -> "CoverageModel":
        """Construct a CoverageModel from pre-computed P-site histograms.

        Skips BAM file I/O; derives the start and stop enrichment factors
        directly from the supplied histograms.

        Parameters
        ----------
        start_hist : np.ndarray
            P-site count histogram around the start codon, shape
            ``(_HIST_SIZE,)``.
        stop_hist : np.ndarray
            P-site count histogram around the stop codon, shape
            ``(_HIST_SIZE,)``.
        run_id : str, optional
            Sample identifier used in warning messages.

        Returns
        -------
        CoverageModel
            Model with all attributes (including optional histograms) set.
        """
        start_factor = cls._compute_start_factor(start_hist, run_id)
        stop_factor = cls._compute_stop_factor(stop_hist, run_id)
        return cls(start_factor, stop_factor, start_hist, stop_hist)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot(
        self,
        axes: Optional[tuple] = None,
    ) -> plt.Figure:
        """Plot the start- and stop-codon P-site histograms.

        Two side-by-side panels: CDS start (left) and CDS stop (right).
        Each panel highlights the peak position in red and annotates the
        corresponding enrichment factor.

        Parameters
        ----------
        axes : tuple of (Axes, Axes) or None, optional
            A ``(ax_start, ax_stop)`` tuple to draw on.  A new figure is
            created when *None*.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the two panels.
        """
        if axes is None:
            fig, (ax_start, ax_stop) = plt.subplots(1, 2, figsize=(14, 5))
        else:
            ax_start, ax_stop = axes
            fig = ax_start.get_figure()

        # Start-codon histogram (x-axis in codon positions)
        x_start = np.arange(_HIST_SIZE) - _START_CODON_IDX

        # Determine which body positions survive the 50% trim.
        start_body_idx = np.arange(_HIST_SIZE)[_START_BODY_SLICE]
        start_body_vals = self.start_hist[_START_BODY_SLICE]
        q25, q75 = np.percentile(start_body_vals, [25, 75])
        start_kept = set(
            start_body_idx[(start_body_vals >= q25) & (start_body_vals <= q75)]
        )

        colors_start = []
        for i in range(_HIST_SIZE):
            if i == _START_CODON_IDX:
                colors_start.append("tab:red")
            elif i in start_kept:
                colors_start.append("steelblue")
            elif i in set(start_body_idx):
                colors_start.append("lightsteelblue")
            else:
                colors_start.append("lightsteelblue")

        ax_start.bar(x_start, self.start_hist, width=1, color=colors_start)
        ax_start.set_xlabel("Codon position relative to translation start")
        ax_start.set_ylabel("P-site read count")
        ax_start.set_title("CDS start")
        ax_start.set_xlim(-12, 105)
        ax_start.text(
            0.95,
            0.95,
            f"start factor = {self.start_factor:.2f}",
            transform=ax_start.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Stop-codon histogram (x-axis in codon positions)
        x_stop = np.arange(_HIST_SIZE) - _STOP_HIST_OFFSET

        # Determine which body positions survive the 50% trim.
        stop_body_idx = np.arange(_HIST_SIZE)[_STOP_BODY_SLICE]
        stop_body_vals = self.stop_hist[_STOP_BODY_SLICE]
        q25, q75 = np.percentile(stop_body_vals, [25, 75])
        stop_kept = set(
            stop_body_idx[(stop_body_vals >= q25) & (stop_body_vals <= q75)]
        )

        colors_stop = []
        for i in range(_HIST_SIZE):
            if i == _STOP_PEAK_IDX:
                colors_stop.append("tab:red")
            elif i in stop_kept:
                colors_stop.append("steelblue")
            elif i in set(stop_body_idx):
                colors_stop.append("lightsteelblue")
            else:
                colors_stop.append("lightsteelblue")

        ax_stop.bar(x_stop, self.stop_hist, width=1, color=colors_stop)
        ax_stop.set_xlabel("Codon position relative to translation end")
        ax_stop.set_ylabel("P-site read count")
        ax_stop.set_title("CDS stop")
        ax_stop.set_xlim(-105, 12)
        ax_stop.text(
            0.3,
            0.95,
            f"stop factor = {self.stop_factor:.2f}",
            transform=ax_stop.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        return fig

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
