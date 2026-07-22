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
import multiprocessing as mp
import os
import shutil
import subprocess

import numba
import numpy as np
import pysam

from price2.cleavage_model import CleavageEstimator, CleavageModel, _MIN_COUNTED_ALNS
from price2.coverage_model import (
    CoverageModel,
    build_histograms,
    _HIST_SIZE,
    _MIN_READS,
    _START_CODON_IDX,
    _START_BODY_SLICE,
    _STOP_PEAK_IDX,
    _STOP_BODY_SLICE,
)
from price2.reference_annotation import ReferenceAnnotation

logger = logging.getLogger(__name__)

#: Width of the genomic windows the coverage pass is split into.  Reads cluster
#: on coding exons, so windows carry very unequal loads; they are handed out
#: one at a time and this size keeps the tail short without drowning the pool in
#: per-window index lookups.
_COVERAGE_WINDOW = 5_000_000

#: Read cap for model estimation.  The downsampled BAM holds at most this many
#: reads.
_MAX_SAMPLED_READS = 10_000_000


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
    high_quality_only: bool = False,
    end_to_end: bool = False,
) -> list[RiboSeqRun]:
    """Build :class:`RiboSeqRun` objects from a collection of BAM files.

    Runs in two phases.  Every BAM is first downsampled and its cleavage model
    fitted by a worker of its own.  The coverage histograms are then accumulated
    by workers that each take one genomic window of one sample BAM; there are
    far more windows than BAM files, so this phase — which dominates the run
    time — keeps every core busy.

    Workers are forked so they share the reference annotation without pickling
    it, which is also why the annotation is indexed up front.  Nothing in this
    process may start the parallel EM before the second pool is forked: numba's
    OpenMP threading layer does not survive a fork.

    A temporary ``sample_bam/`` sub-directory under *wdir* holds the downsampled
    BAM files for the duration and is removed afterwards.

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
    end_to_end : bool, optional
        When True the BAM files were mapped with ``--alignEndsType EndToEnd``;
        the untemplated addition is recovered from the 5'-terminal mismatch
        instead of a soft-clip.  Defaults to False.

    Returns
    -------
    list[RiboSeqRun]
        One :class:`RiboSeqRun` per BAM file, ordered by BAM filename.
    """
    global _WORKER_RA, _WORKER_CLEAVAGE, _WORKER_END_TO_END

    sample_dir = f"{wdir}/sample_bam"
    os.makedirs(sample_dir, exist_ok=True)
    bam_files = sorted(f"{bam_id}.bam" for bam_id in bam_ids)

    if not bam_files:
        os.rmdir(sample_dir)
        return []

    # Build the CDS index once, before forking: every worker queries it per read
    # and would otherwise build its own copy.
    ref_annotation.build_cds_index()

    # The context must be requested explicitly: importing
    # ``orf_activity_estimator`` sets the global start method to ``forkserver``,
    # under which workers would not inherit the module globals below.
    ctx = mp.get_context("fork")
    _WORKER_RA = ref_annotation
    _WORKER_END_TO_END = end_to_end
    fitted: list[tuple[str, int, str, int, CleavageModel]] = []
    try:
        # Phase one runs one worker per BAM, so each may use several threads of
        # its own for samtools and for the EM.
        threads = max(1, processes // len(bam_files))
        with ctx.Pool(
            min(processes, len(bam_files)),
            initializer=_init_cleavage_worker,
            initargs=(threads,),
        ) as pool:
            fitted = pool.starmap(
                _sample_and_fit_cleavage,
                [(bam_dir, bam_file, sample_dir) for bam_file in bam_files],
            )

        _WORKER_CLEAVAGE = {run_id: cm for run_id, _, _, _, cm in fitted}
        tasks = [
            (run_id, sample_bam, window)
            for run_id, _, sample_bam, _, _ in fitted
            for window in _coverage_windows(sample_bam)
        ]
        histograms = {
            run_id: (np.zeros(_HIST_SIZE), np.zeros(_HIST_SIZE))
            for run_id, _, _, _, _ in fitted
        }
        with ctx.Pool(min(processes, len(tasks))) as pool:
            for run_id, start_hist, stop_hist in pool.imap_unordered(
                _coverage_window, tasks, chunksize=1
            ):
                histograms[run_id][0][:] += start_hist
                histograms[run_id][1][:] += stop_hist
    finally:
        _WORKER_RA = None
        _WORKER_CLEAVAGE = {}
        _WORKER_END_TO_END = False
        # Also clears the samples of a phase that died half-way, which would
        # otherwise mask the original exception with a "directory not empty".
        shutil.rmtree(sample_dir, ignore_errors=True)

    ribo_seq_runs = [
        _assemble_run(
            run_id,
            read_count,
            counted_alns,
            cleavage_model,
            CoverageModel.from_histograms(*histograms[run_id], run_id),
        )
        for run_id, read_count, _, counted_alns, cleavage_model in fitted
    ]

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


def _assemble_run(
    run_id: str,
    read_count: int,
    counted_alns: int,
    cleavage_model: CleavageModel,
    coverage_model: CoverageModel,
) -> RiboSeqRun:
    """Bundle the fitted models of one run and score their quality."""
    max_pos = int(np.argmax(cleavage_model.pl))
    max_prob = float(cleavage_model.pl[max_pos])
    cleavage_ok = (
        (max_pos == 12) and (max_prob >= 0.3) and (counted_alns >= _MIN_COUNTED_ALNS)
    )

    start_hist = coverage_model.start_hist
    stop_hist = coverage_model.stop_hist
    coverage_ok = (
        start_hist[_START_CODON_IDX] >= _MIN_READS
        and start_hist[_START_BODY_SLICE].sum() >= _MIN_READS
        and stop_hist[_STOP_PEAK_IDX] >= _MIN_READS
        and stop_hist[_STOP_BODY_SLICE].sum() >= _MIN_READS
    )

    return RiboSeqRun(
        run_id,
        cleavage_model,
        coverage_model,
        read_count=read_count,
        cleavage_counted_reads=counted_alns,
        is_high_quality=cleavage_ok and coverage_ok,
    )


def _coverage_windows(sample_bam: str) -> list[tuple[str, int, int]]:
    """Tile every contig of *sample_bam* with :data:`_COVERAGE_WINDOW` windows."""
    with pysam.AlignmentFile(sample_bam, "rb") as bam:
        references = bam.references
        lengths = bam.lengths
    return [
        (contig, start, min(start + _COVERAGE_WINDOW, length))
        for contig, length in zip(references, lengths)
        for start in range(0, length, _COVERAGE_WINDOW)
    ]


# ---------------------------------------------------------------------------
# Worker state and tasks
# ---------------------------------------------------------------------------

#: Set in the parent before a pool is forked, and read by its workers.  This is
#: what keeps the reference annotation and the cleavage models out of the task
#: pickles.
_WORKER_RA: ReferenceAnnotation | None = None
_WORKER_CLEAVAGE: dict[str, CleavageModel] = {}
#: EndToEnd untemplated-addition detection, set in the parent before forking.
_WORKER_END_TO_END: bool = False

#: Per-worker state, populated after the fork.
_WORKER_BAM: dict[str, pysam.AlignmentFile] = {}
_WORKER_THREADS: int = 1


def _worker_bam(path: str) -> pysam.AlignmentFile:
    """Return a per-worker BAM handle, so the index is loaded only once."""
    handle = _WORKER_BAM.get(path)
    if handle is None:
        handle = pysam.AlignmentFile(path, "rb")
        _WORKER_BAM[path] = handle
    return handle


def _init_cleavage_worker(threads: int) -> None:
    """Cap each worker's thread budget so the pool does not oversubscribe."""
    global _WORKER_THREADS

    _WORKER_THREADS = threads
    numba.set_num_threads(threads)


def _sample_and_fit_cleavage(
    bam_dir: str, bam_file: str, sample_dir: str
) -> tuple[str, int, str, int, CleavageModel]:
    """Downsample one BAM and fit its cleavage model.

    Returns
    -------
    tuple
        ``(run_id, read_count, sample_bam_path, counted_alns, cleavage_model)``.
    """
    run_id = os.path.splitext(bam_file)[0]
    bam_file_path = f"{bam_dir}/{bam_file}"

    # Count total reads and derive the downsampling fraction.  ``mapped`` is
    # read from the BAM index; it equals ``count()``, which would decompress
    # every record.
    with pysam.AlignmentFile(bam_file_path, "rb") as bam:
        read_count = bam.mapped
    fraction_of_reads = min(_MAX_SAMPLED_READS / read_count, 0.99)

    sample_bam = f"{sample_dir}/{bam_file}"
    subprocess.run(
        [
            "samtools",
            "view",
            "-@",
            str(_WORKER_THREADS),
            "-b",
            "-s",
            str(fraction_of_reads),
            "-o",
            sample_bam,
            bam_file_path,
        ],
        check=True,
    )
    # The coverage phase fetches windows of the sample by genomic coordinate.
    pysam.index("-@", str(_WORKER_THREADS), sample_bam)

    estimator = CleavageEstimator()
    estimator.collect_data(_WORKER_RA, sample_bam, end_to_end=_WORKER_END_TO_END)
    estimator.correct_table()
    return run_id, read_count, sample_bam, estimator.counted_alns, estimator.run()


def _coverage_window(
    task: tuple[str, str, tuple[str, int, int]]
) -> tuple[str, np.ndarray, np.ndarray]:
    """Accumulate the coverage histograms of one genomic window of one run."""
    run_id, sample_bam, window = task
    start_hist, stop_hist = build_histograms(
        _WORKER_RA,
        _worker_bam(sample_bam),
        _WORKER_CLEAVAGE[run_id],
        window,
        end_to_end=_WORKER_END_TO_END,
    )
    return run_id, start_hist, stop_hist


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
