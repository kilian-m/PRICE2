"""Coverage model for Ribo-seq data.

Models the elevated ribosome footprint density at ORF start codons and at
the codon immediately upstream of stop codons, relative to average ORF-body
coverage.  The resulting scale factors are used to weight the expected
coverage profile when deconvolving overlapping ORFs.
"""

import importlib
import logging
from typing import Optional

import numpy as np
from scipy.stats import trim_mean

logger = logging.getLogger(__name__)

#: Names that moved to other modules, resolved lazily for older imports.
_MOVED = {
    "CoveragePosition": "price2.coverage_position",
    "build_histograms": "price2.coverage_estimator",
    "_try_assign_p_site": "price2.coverage_estimator",
}


def __getattr__(name: str):
    if name in _MOVED:
        return getattr(importlib.import_module(_MOVED[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Size of the accumulation arrays used while building the P-site histograms.
# Each bin corresponds to one codon (3 nt).
HIST_SIZE: int = 120

# Index in the start-histogram that corresponds to CDS position 0 (start codon).
START_CODON_IDX: int = 10

# Slice over the ORF body in the start-histogram (codons 1 .. 100 of CDS).
START_BODY_SLICE: slice = slice(11, 111)

# Index in the stop-histogram that corresponds to the last sense codon before
# the stop codon (CDS position len(cds) − 3).
STOP_PEAK_IDX: int = 110

# Offset applied when filling the stop-histogram:
#   index = p_site_cds_pos // 3 − len(cds) // 3 + STOP_HIST_OFFSET
# so that CDS codon len(cds)//3 maps to index STOP_HIST_OFFSET.
STOP_HIST_OFFSET: int = 111

# Slice over the ORF body in the stop-histogram.
STOP_BODY_SLICE: slice = slice(STOP_PEAK_IDX - 100, STOP_PEAK_IDX)

# Minimum number of reads required for reliable factor estimation.
MIN_READS: int = 100

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
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _enrichment_factor(
        hist: np.ndarray, peak_idx: int, body: slice, label: str, run_id: str
    ) -> float:
        """Enrichment of the P-site count at *peak_idx* over the ORF body.

        The body coverage is the 25 %-trimmed mean of ``hist[body]``.  A
        body without coverage gives a factor of 1, and the factor is
        floored at 1 so a depleted peak never down-weights the position.

        Parameters
        ----------
        hist : np.ndarray
            P-site histogram as returned by :func:`build_histograms`.
        peak_idx : int
            Histogram index of the peak position.
        body : slice
            Histogram slice over the ORF body.
        label : str
            Name of the peak position used in warning messages.
        run_id : str
            Sample identifier used only in warning messages.

        Returns
        -------
        float
            Enrichment factor >= 1.
        """
        peak_count = hist[peak_idx]
        body_counts = hist[body]

        if peak_count < MIN_READS:
            logger.warning(
                "Only %d reads at %s position. "
                "Low evidence for coverage model.  sample: %s",
                int(peak_count),
                label,
                run_id,
            )
        if body_counts.sum() < MIN_READS:
            logger.warning(
                "Only %d reads at middle codon positions. "
                "Low evidence for coverage model.  sample: %s",
                int(body_counts.sum()),
                run_id,
            )

        with np.errstate(divide="raise"):
            try:
                factor = float(
                    peak_count / trim_mean(body_counts, proportiontocut=0.25)
                )
            except (FloatingPointError, ValueError):
                factor = 1.0

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
            ``(HIST_SIZE,)``.
        stop_hist : np.ndarray
            P-site count histogram around the stop codon, shape
            ``(HIST_SIZE,)``.
        run_id : str, optional
            Sample identifier used in warning messages.

        Returns
        -------
        CoverageModel
            Model with all attributes (including optional histograms) set.
        """
        start_factor = cls._enrichment_factor(
            start_hist, START_CODON_IDX, START_BODY_SLICE, "start codon", run_id
        )
        stop_factor = cls._enrichment_factor(
            stop_hist, STOP_PEAK_IDX, STOP_BODY_SLICE, "stop codon", run_id
        )
        return cls(start_factor, stop_factor, start_hist, stop_hist)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_plausible(self) -> bool:
        """Whether both histograms carry enough reads for a trustworthy model.

        Requires at least :data:`MIN_READS` P-sites on the start codon, on
        the stop peak, and over each ORF body.
        """
        return bool(
            self.start_hist[START_CODON_IDX] >= MIN_READS
            and self.start_hist[START_BODY_SLICE].sum() >= MIN_READS
            and self.stop_hist[STOP_PEAK_IDX] >= MIN_READS
            and self.stop_hist[STOP_BODY_SLICE].sum() >= MIN_READS
        )

    def plot(self, axes: Optional[tuple] = None):
        """Plot the start- and stop-codon P-site histograms.

        See :func:`price2.plotting.plot_coverage`.
        """
        from price2.plotting import plot_coverage

        return plot_coverage(self, axes)

