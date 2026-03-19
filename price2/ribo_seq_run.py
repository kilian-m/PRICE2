"""Ribo-seq run representation and construction from BAM files.

A :class:`RiboSeqRun` bundles the run identifier with the estimated
:class:`~price2.cleavage_model.CleavageModel` and
:class:`~price2.coverage_model.CoverageModel` derived from the mapped reads.
The module also provides helpers that construct
:class:`RiboSeqRun` objects from BAM files, including downsampling to at most
10 million reads before model estimation.

BAM files are assumed to be coordinate-sorted and indexed.
"""

import logging
import logging.handlers
import multiprocessing
import os
import subprocess

import numpy as np
import pysam

from price2.cleavage_model import CleavageEstimator, CleavageModel, _MIN_COUNTED_ALNS
from price2.coverage_model import (
    CoverageModel,
    _MIN_READS,
    _START_CODON_IDX,
    _START_BODY_SLICE,
    _STOP_PEAK_IDX,
    _STOP_BODY_SLICE,
)
from price2.reference_annotation import ReferenceAnnotation

logger = logging.getLogger(__name__)


def _init_worker_logging(log_queue, log_level: str) -> None:
    """Configure logging in a Pool worker process."""
    worker_logger = logging.getLogger("price2")
    if not worker_logger.handlers:
        worker_logger.addHandler(logging.handlers.QueueHandler(log_queue))
        worker_logger.setLevel(log_level)
        worker_logger.propagate = False


class RiboSeqRun:
    """A single Ribo-seq run with its associated models.

    Parameters
    ----------
    run_id : str
        Unique sample identifier (typically the BAM filename without
        the ``.bam`` extension).
    cleavage_model : CleavageModel
        Cleavage site probability model estimated from this run.
    coverage_model : CoverageModel
        Coverage scale-factor model estimated from this run.
    read_count : int, optional
        Total number of mapped reads in the original BAM file.
    cleavage_counted_reads : int, optional
        Number of reads used for cleavage model estimation.
    is_high_quality : bool, optional
        Whether the cleavage model passed quality checks (peak at position 12
        and peak probability >= 0.3).
    """

    def __init__(
        self,
        run_id: str,
        cleavage_model: CleavageModel,
        coverage_model: CoverageModel,
        read_count: int = 0,
        cleavage_counted_reads: int = 0,
        is_high_quality: bool = True,
    ) -> None:
        self.id = run_id
        self.cleavage_model = cleavage_model
        self.coverage_model = coverage_model
        self.read_count = read_count
        self.cleavage_counted_reads = cleavage_counted_reads
        self.is_high_quality = is_high_quality

    def __repr__(self) -> str:
        return f"RiboSeqRun(id={self.id!r}, read_count={self.read_count})"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RiboSeqRun):
            return NotImplemented
        return self.id == other.id


def ribo_seq_runs_from_bams(
    bam_dir: str,
    bam_ids: set[str],
    wdir: str,
    ref_annotation: ReferenceAnnotation,
    processes: int = 32,
    log_level: str = "INFO",
    high_quality_only: bool = False,
) -> list[RiboSeqRun]:
    """Build :class:`RiboSeqRun` objects from a collection of BAM files.

    Each BAM file is processed in a separate worker process.  A temporary
    ``sample_bam/`` sub-directory under *wdir* is created to hold
    downsampled BAM files during model estimation and removed afterwards.

    Parameters
    ----------
    bam_dir : str
        Directory containing the BAM files.
    bam_ids : set[str]
        Set of sample identifiers.  The corresponding BAM filenames are
        expected to be ``<id>.bam``.
    wdir : str
        Working directory used for temporary files.
    ref_annotation : ReferenceAnnotation
        Parsed reference annotation used during model estimation.
    processes : int, optional
        Maximum number of worker processes.  Defaults to 32.
    high_quality_only : bool, optional
        If True, exclude runs whose cleavage model failed quality checks
        (peak not at position 12 or peak probability < 0.3).  Defaults to
        False.

    Returns
    -------
    list[RiboSeqRun]
        One :class:`RiboSeqRun` per BAM file, in unspecified order.
    """
    os.makedirs(f"{wdir}/sample_bam", exist_ok=True)
    bam_files = [f"{bam_id}.bam" for bam_id in bam_ids]

    if bam_files:
        ctx = multiprocessing.get_context("forkserver")
        price2_logger = logging.getLogger("price2")
        manager = multiprocessing.Manager()
        log_queue = manager.Queue()
        listener = logging.handlers.QueueListener(
            log_queue, *price2_logger.handlers, respect_handler_level=True
        )
        listener.start()
        try:
            with ctx.Pool(
                processes,
                initializer=_init_worker_logging,
                initargs=(log_queue, log_level),
            ) as pool:
                ribo_seq_runs: list[RiboSeqRun] = pool.starmap(
                    ribo_seq_run_from_bam,
                    [
                        (bam_dir, bam_file, wdir, ref_annotation)
                        for bam_file in bam_files
                    ],
                )
        finally:
            listener.stop()
            manager.shutdown()
    else:
        ribo_seq_runs = []

    os.rmdir(f"{wdir}/sample_bam")
    if high_quality_only:
        filtered = [r for r in ribo_seq_runs if r.is_high_quality]
        excluded = [r for r in ribo_seq_runs if not r.is_high_quality]
        if excluded:
            logger.warning(
                "Excluding %d low-quality run(s): %s",
                len(excluded),
                ", ".join(r.id for r in excluded),
            )
        return filtered
    return ribo_seq_runs


def ribo_seq_run_from_bam(
    bam_dir: str,
    bam_file: str,
    wdir: str,
    ref_annotation: ReferenceAnnotation,
) -> RiboSeqRun:
    """Build a :class:`RiboSeqRun` from a single BAM file.

    Reads the total read count from the BAM file, creates a downsampled copy
    capped at 10 million reads, estimates the cleavage and coverage models
    from that sample, and then removes the temporary file.

    Parameters
    ----------
    bam_dir : str
        Directory containing *bam_file*.
    bam_file : str
        BAM filename (e.g. ``"sample1.bam"``).
    wdir : str
        Working directory; a ``sample_bam/`` sub-directory must already exist
        here.
    ref_annotation : ReferenceAnnotation
        Parsed reference annotation used during model estimation.

    Returns
    -------
    RiboSeqRun
        Fully initialised run object with estimated models.
    """
    run_id = bam_file.split(".")[0]
    bam_file_path = f"{bam_dir}/{bam_file}"

    # Count total reads and derive the downsampling fraction.
    with pysam.AlignmentFile(bam_file_path, "rb") as bam:
        read_count = bam.count()
    fraction_of_reads = min(10_000_000 / read_count, 0.99)

    # Write downsampled BAM to a temporary file.
    sample_bam_file = f"{wdir}/sample_bam/{bam_file}"
    subprocess.run(
        [
            "samtools",
            "view",
            "-b",
            "-s",
            str(fraction_of_reads),
            "-o",
            sample_bam_file,
            bam_file_path,
        ],
        check=True,
    )

    # Estimate the cleavage model.
    ce = CleavageEstimator()
    ce.collect_data(ref_annotation, sample_bam_file)
    ce.correct_table()
    cleavage_model = ce.run()

    # Estimate the coverage model.
    coverage_model = CoverageModel.from_bam(
        ref_annotation,
        sample_bam_file,
        cleavage_model,
    )

    os.remove(sample_bam_file)

    max_pos = int(np.argmax(cleavage_model.pl))
    max_prob = float(cleavage_model.pl[max_pos])
    cleavage_ok = (
        (max_pos == 12) and (max_prob >= 0.3) and (ce.counted_alns >= _MIN_COUNTED_ALNS)
    )

    start_hist = coverage_model.start_hist
    stop_hist = coverage_model.stop_hist
    coverage_ok = (
        start_hist[_START_CODON_IDX] >= _MIN_READS
        and start_hist[_START_BODY_SLICE].sum() >= _MIN_READS
        and stop_hist[_STOP_PEAK_IDX] >= _MIN_READS
        and stop_hist[_STOP_BODY_SLICE].sum() >= _MIN_READS
    )
    is_high_quality = cleavage_ok and coverage_ok

    return RiboSeqRun(
        run_id,
        cleavage_model,
        coverage_model,
        read_count=read_count,
        cleavage_counted_reads=ce.counted_alns,
        is_high_quality=is_high_quality,
    )


# ---------------------------------------------------------------------------
# Dataset model I/O
# ---------------------------------------------------------------------------


def save_dataset_models(
    runs: list[RiboSeqRun],
    output_dir: str,
    save_optional: bool = True,
) -> None:
    """Save cleavage and coverage model summaries, plots, and optional data.

    Creates a ``dataset_models/`` sub-directory under *output_dir* and
    writes:

    * ``cleavage_models.tsv`` – obligatory attributes (pl, pr, pu).
    * ``cleavage_models.pdf`` – multi-page diagnostic plots (requires
      optional cleavage attributes).
    * ``coverage_models.tsv`` – obligatory attributes (start_factor,
      stop_factor).
    * ``coverage_models.pdf`` – multi-page histogram plots (requires
      optional coverage attributes).

    When *save_optional* is True, also writes:

    * ``cleavage_models.npz`` – optional arrays (dist_starts, table).
    * ``coverage_models.npz`` – optional arrays (start_hist, stop_hist).

    Parameters
    ----------
    runs : list[RiboSeqRun]
        Ribo-seq runs whose models should be exported.
    output_dir : str
        Root output directory (``config.o_dir``).
    save_optional : bool, optional
        Whether to write ``.npz`` files with optional model attributes
        (default True).
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    dm_dir = os.path.join(output_dir, "dataset_models")
    os.makedirs(dm_dir, exist_ok=True)

    # --- Cleavage TSV + optional NPZ ---
    cleavage_tsv = os.path.join(dm_dir, "cleavage_models.tsv")
    cleavage_npz_data: dict[str, np.ndarray] = {} if save_optional else None
    with open(cleavage_tsv, "w") as fh:
        fh.write(CleavageModel.TSV_HEADER + "\n")
        for run in runs:
            run.cleavage_model.to_files(run.id, fh, cleavage_npz_data)
    if cleavage_npz_data:
        np.savez(os.path.join(dm_dir, "cleavage_models.npz"), **cleavage_npz_data)

    # --- Cleavage PDF ---
    path = os.path.join(dm_dir, "cleavage_models.pdf")
    with PdfPages(path) as pdf:
        for run in runs:
            fig = run.cleavage_model.plot_full()
            fig.suptitle(run.id, fontsize=14, fontweight="bold")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    # --- Coverage TSV + optional NPZ ---
    coverage_tsv = os.path.join(dm_dir, "coverage_models.tsv")
    coverage_npz_data: dict[str, np.ndarray] = {} if save_optional else None
    with open(coverage_tsv, "w") as fh:
        fh.write(CoverageModel.TSV_HEADER + "\n")
        for run in runs:
            run.coverage_model.to_files(run.id, fh, coverage_npz_data)
    if coverage_npz_data:
        np.savez(os.path.join(dm_dir, "coverage_models.npz"), **coverage_npz_data)

    # --- Coverage PDF ---
    path = os.path.join(dm_dir, "coverage_models.pdf")
    with PdfPages(path) as pdf:
        for run in runs:
            fig = run.coverage_model.plot()
            fig.suptitle(run.id, fontsize=14, fontweight="bold")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def load_cleavage_models(
    dm_dir: str,
    load_optional: bool = False,
) -> dict[str, CleavageModel]:
    """Load cleavage models from a ``dataset_models/`` directory.

    Parameters
    ----------
    dm_dir : str
        Path to a ``dataset_models/`` directory.
    load_optional : bool, optional
        Whether to load optional attributes from the ``.npz`` file
        (default False).

    Returns
    -------
    dict[str, CleavageModel]
        Mapping of dataset identifier to the reconstructed model.
    """
    tsv = os.path.join(dm_dir, "cleavage_models.tsv")
    npz = os.path.join(dm_dir, "cleavage_models.npz")
    return CleavageModel.from_files(tsv, npz if load_optional and os.path.exists(npz) else None)


def load_coverage_models(
    dm_dir: str,
    load_optional: bool = False,
) -> dict[str, CoverageModel]:
    """Load coverage models from a ``dataset_models/`` directory.

    Parameters
    ----------
    dm_dir : str
        Path to a ``dataset_models/`` directory.
    load_optional : bool, optional
        Whether to load optional attributes from the ``.npz`` file
        (default False).

    Returns
    -------
    dict[str, CoverageModel]
        Mapping of dataset identifier to the reconstructed model.
    """
    tsv = os.path.join(dm_dir, "coverage_models.tsv")
    npz = os.path.join(dm_dir, "coverage_models.npz")
    return CoverageModel.from_files(tsv, npz if load_optional and os.path.exists(npz) else None)
