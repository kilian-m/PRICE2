"""Ribo-seq run representation and construction from BAM files.

A :class:`RiboSeqRun` bundles the run identifier with the estimated
:class:`~price2.cleavage_model.CleavageModel` and
:class:`~price2.coverage_model.CoverageModel` derived from the mapped reads.
The module also constructs :class:`RiboSeqRun` objects from BAM files,
downsampling each to at most 10 million reads before model estimation
(:mod:`price2.dataset_models` exports the fitted models).

BAM files are assumed to be coordinate-sorted and indexed.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import shutil
import subprocess
from typing import NamedTuple

import numba
import numpy as np
import pysam

from price2.bam import cached_alignment_file
from price2.cleavage_estimator import MIN_COUNTED_ALNS, CleavageEstimator
from price2.cleavage_model import CleavageModel
from price2.coverage_estimator import build_histograms
from price2.coverage_model import HIST_SIZE, CoverageModel
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
        If True, exclude runs that fail the quality gate of
        :func:`_assemble_run` (an implausible cleavage or coverage model, or
        fewer than :data:`MIN_COUNTED_ALNS` counted alignments).  Defaults
        to False.
    end_to_end : bool, optional
        When True the BAM files were mapped with ``--alignEndsType EndToEnd``;
        the untemplated addition is recovered from the 5'-terminal mismatch
        instead of a soft-clip.  Defaults to False.

    Returns
    -------
    list[RiboSeqRun]
        One :class:`RiboSeqRun` per BAM file, ordered by BAM filename.
    """
    sample_dir = f"{wdir}/sample_bam"
    os.makedirs(sample_dir, exist_ok=True)
    bam_files = sorted(f"{bam_id}.bam" for bam_id in bam_ids)

    if not bam_files:
        os.rmdir(sample_dir)
        return []

    # ``fork`` so that the workers inherit the annotation and the models
    # through the initializer's arguments without pickling them; the
    # deconvolution pool uses ``forkserver`` instead.
    ctx = mp.get_context("fork")
    try:
        fitted = _fit_cleavage_models(
            ctx, bam_dir, bam_files, sample_dir, ref_annotation, end_to_end, processes
        )
        histograms = _coverage_histograms(
            ctx, fitted, ref_annotation, end_to_end, processes
        )
    finally:
        # Also clears the samples of a phase that died half-way, which would
        # otherwise mask the original exception with a "directory not empty".
        shutil.rmtree(sample_dir, ignore_errors=True)

    ribo_seq_runs = [
        _assemble_run(
            fit.run_id,
            fit.read_count,
            fit.counted_alns,
            fit.cleavage_model,
            CoverageModel.from_histograms(*histograms[fit.run_id], fit.run_id),
        )
        for fit in fitted
    ]
    if not high_quality_only:
        return ribo_seq_runs

    excluded = [r for r in ribo_seq_runs if not r.is_high_quality]
    if excluded:
        logger.warning(
            "Excluding %d low-quality run(s): %s",
            len(excluded),
            ", ".join(r.id for r in excluded),
        )
    return [r for r in ribo_seq_runs if r.is_high_quality]


def _fit_cleavage_models(
    ctx,
    bam_dir: str,
    bam_files: list[str],
    sample_dir: str,
    ref_annotation: ReferenceAnnotation,
    end_to_end: bool,
    processes: int,
) -> list[FittedRun]:
    """Downsample every BAM and fit its cleavage model, one worker per BAM.

    Each worker may use several threads of its own for samtools and for
    the EM, since there are fewer BAMs than cores.
    """
    threads = max(1, processes // len(bam_files))
    with ctx.Pool(
        min(processes, len(bam_files)),
        initializer=_init_worker,
        initargs=(ref_annotation, end_to_end, {}, threads),
    ) as pool:
        return pool.starmap(
            _sample_and_fit_cleavage,
            [(bam_dir, bam_file, sample_dir) for bam_file in bam_files],
        )


def _coverage_histograms(
    ctx,
    fitted: list[FittedRun],
    ref_annotation: ReferenceAnnotation,
    end_to_end: bool,
    processes: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The start and stop P-site histograms of every fitted run.

    Accumulated by workers that each take one genomic window of one sample
    BAM: there are far more windows than BAM files, so this phase — which
    dominates the run time — keeps every core busy.
    """
    cleavage_models = {fit.run_id: fit.cleavage_model for fit in fitted}
    tasks = [
        (fit.run_id, fit.sample_bam, window)
        for fit in fitted
        for window in _coverage_windows(fit.sample_bam)
    ]
    histograms = {
        fit.run_id: (np.zeros(HIST_SIZE), np.zeros(HIST_SIZE)) for fit in fitted
    }
    with ctx.Pool(
        min(processes, len(tasks)),
        initializer=_init_worker,
        initargs=(ref_annotation, end_to_end, cleavage_models, None),
    ) as pool:
        for run_id, start_hist, stop_hist in pool.imap_unordered(
            _coverage_window, tasks, chunksize=1
        ):
            histograms[run_id][0][:] += start_hist
            histograms[run_id][1][:] += stop_hist
    return histograms


def _assemble_run(
    run_id: str,
    read_count: int,
    counted_alns: int,
    cleavage_model: CleavageModel,
    coverage_model: CoverageModel,
) -> RiboSeqRun:
    """Bundle the fitted models of one run and score their quality."""
    cleavage_ok = (
        cleavage_model.is_plausible() and counted_alns >= MIN_COUNTED_ALNS
    )
    coverage_ok = coverage_model.is_plausible()

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

#: What every worker of the two pools reads, set by :func:`_init_worker`
#: (the pools' ``initializer``).  Passing these as initializer arguments of
#: a forked pool keeps the reference annotation and the cleavage models out
#: of the task pickles.
_WORKER_RA: ReferenceAnnotation | None = None
_WORKER_CLEAVAGE: dict[str, CleavageModel] = {}
#: EndToEnd untemplated-addition detection.
_WORKER_END_TO_END: bool = False
#: Threads for samtools and the EM.
_WORKER_THREADS: int = 1


def _init_worker(
    ref_annotation: ReferenceAnnotation,
    end_to_end: bool,
    cleavage_models: dict[str, CleavageModel],
    threads: int | None,
) -> None:
    """Give a worker its inputs; *threads* caps its budget so the pool does
    not oversubscribe (``None`` for a phase without threaded work)."""
    global _WORKER_RA, _WORKER_END_TO_END, _WORKER_CLEAVAGE, _WORKER_THREADS

    _WORKER_RA = ref_annotation
    _WORKER_END_TO_END = end_to_end
    _WORKER_CLEAVAGE = cleavage_models
    if threads is not None:
        _WORKER_THREADS = threads
        numba.set_num_threads(threads)


class FittedRun(NamedTuple):
    """What :func:`_sample_and_fit_cleavage` returns for one BAM."""

    run_id: str
    #: Mapped reads in the original BAM.
    read_count: int
    #: The downsampled, indexed BAM the coverage phase reads.
    sample_bam: str
    #: Alignments the cleavage estimator counted.
    counted_alns: int
    cleavage_model: CleavageModel


def _sample_and_fit_cleavage(
    bam_dir: str, bam_file: str, sample_dir: str
) -> FittedRun:
    """Downsample one BAM and fit its cleavage model."""
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
    return FittedRun(
        run_id, read_count, sample_bam, estimator.counted_alns, estimator.run()
    )


def _coverage_window(
    task: tuple[str, str, tuple[str, int, int]]
) -> tuple[str, np.ndarray, np.ndarray]:
    """Accumulate the coverage histograms of one genomic window of one run."""
    run_id, sample_bam, window = task
    start_hist, stop_hist = build_histograms(
        _WORKER_RA,
        cached_alignment_file(sample_bam),
        _WORKER_CLEAVAGE[run_id],
        window,
        end_to_end=_WORKER_END_TO_END,
    )
    return run_id, start_hist, stop_hist
