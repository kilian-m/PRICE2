"""Parallel ORF activity estimation across genomic loci.

:class:`ORFActivityEstimator` fans the loci of a run out over a
``pebble.ProcessPool`` — once, for the classic single pass, or repeatedly
inside the multimapping-EM outer loop it drives in
:meth:`~ORFActivityEstimator.run_multimap_em`; :func:`process_loc` is what
one worker does with one locus: load it, generate and filter ORF candidates,
build the equivalence groups, deconvolve, and render the results for the
parent to write.  Each locus runs in an isolated worker so that timeouts and
crashes stay contained.

The workers are started with the ``forkserver`` method: numba's JIT state
and SQLite handles are not safe to ``fork``.  :func:`init_worker` runs once
per worker process and installs the :class:`WorkerContext` every locus of
that worker shares.
"""

from __future__ import annotations

import csv
import logging
import logging.handlers
import multiprocessing as mp
import os
import resource
import time
import traceback
from concurrent.futures import TimeoutError, as_completed
from contextlib import contextmanager
from dataclasses import dataclass

from pebble import ProcessPool
from pebble.common import CONSTS as _pebble_consts
from pyfaidx import Fasta
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from price2 import database, export, mu_solver, multimap, run_state
from price2.export import OutputText
from price2.config import Config
from price2.equivalence_groups import make_equivalence_groups
from price2.layout import RunLayout
from price2.locus import Locus
from price2.orf_candidates import build_rgrs
from price2.pipeline import run_stage
from price2.ribo_seq_run import RiboSeqRun

logger = logging.getLogger(__name__)

_MP_CONTEXT = mp.get_context("forkserver")

#: pebble's default channel lock timeout is 60 s; with ``forkserver`` and many
#: workers (e.g. 80) the result-pipe mutex can be contended for longer, which
#: makes workers exit with "Abnormal termination".  600 s gives ample headroom.
_PEBBLE_CHANNEL_LOCK_TIMEOUT = 600

#: Thread pools of the numerical libraries a worker may load.  Every worker
#: is a process of its own and the pool already fills the cores, so each
#: library gets one thread unless the environment says otherwise.
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def limit_library_threads() -> None:
    """Default the BLAS/OpenMP thread counts to one for this process.

    Read by the libraries when they initialise, so call it before numpy is
    imported to be sure it takes effect — ``price.py`` does at start-up, and
    the worker processes inherit the result.
    """
    for name in _THREAD_VARIABLES:
        os.environ.setdefault(name, "1")


#: The columns of ``performance_measurements.tsv``, in order.  Every row is
#: written against this list (a locus that lacks a value leaves the field
#: empty), so the columns line up whichever path a locus took.
PERF_COLUMNS = (
    "loc_id", "chrom", "strand", "start", "end",
    "db_time", "load_reads_time",
    "build_rgrs_time", "unfiltered_rgr_count",
    "well_fitting_reads_time",
    "coverage_filter_time", "filtered_coverage_rgr_count",
    "deconvolution_filter_time", "filtered_deconvolution_rgr_count",
    "eg_time", "eg_count",
    "route_reads_time", "read_count", "mstep_pruned_orf_count",
    "optimization_time", "irls_outer_iterations", "filtered_deconvoluted_rgr_count",
    "likelihood_ratio_time", "filtered_lrt_rgr_count", "orf_count",
    "activity_time",
    "gene_number", "transcripts_number", "exon_length",
    "max_rss_mb", "overall_time",
)


# --------------------------------------------------------------------------- #
# What a worker knows and what a job asks of it
# --------------------------------------------------------------------------- #


@dataclass
class WorkerContext:
    """State shared by every locus a worker process handles.

    Attributes
    ----------
    config : Config
        The run configuration.
    layout : RunLayout
        The run's files.
    genome : pyfaidx.Fasta
        The reference genome, opened once per worker.
    runs : list[RiboSeqRun]
        The Ribo-seq runs with their models, loaded once per worker (they
        do not change during the deconvolution, and their model tables are
        memoised on the objects' identity).
    """

    config: Config
    layout: RunLayout
    genome: Fasta
    runs: list[RiboSeqRun]


_CONTEXT: WorkerContext | None = None


def init_worker(config: Config, log_queue) -> None:
    """Prepare a worker process: route its logs to the parent, open the genome.

    Passed to the pool as ``initializer``; in-process callers (tests) call it
    once before :func:`process_loc`.

    Parameters
    ----------
    config : Config
        The configuration the workers run with.
    log_queue : multiprocessing.Queue
        Queue drained by the parent's :class:`logging.handlers.QueueListener`.
    """
    global _CONTEXT
    limit_library_threads()
    worker_logger = logging.getLogger("price2")
    if not worker_logger.handlers:
        worker_logger.addHandler(logging.handlers.QueueHandler(log_queue))
        worker_logger.setLevel(config.log_level)
        worker_logger.propagate = False
    mu_solver.set_kernel(config.mu_kernel)
    layout = config.layout
    _CONTEXT = WorkerContext(
        config, layout, Fasta(config.fasta_path), _load_runs(layout.db_path)
    )


def _context() -> WorkerContext:
    if _CONTEXT is None:
        raise RuntimeError("process_loc called before init_worker")
    return _CONTEXT


@dataclass(frozen=True)
class LocusJob:
    """One locus to process, and in which pass.

    Parameters
    ----------
    locus_id : str
        The locus id stored in the database.
    em_iteration : int or None
        Multimapping-EM iteration; ``None`` for the classic single pass.
    em_final : bool
        ``True`` for the classic pass or the final EM pass (full
        deconvolution, filtering, export and resume bookkeeping); ``False``
        for a light intermediate EM pass that only writes activities and λ
        for the E-step.
    """

    locus_id: str
    em_iteration: int | None = None
    em_final: bool = True

    @property
    def em_mode(self) -> bool:
        return self.em_iteration is not None

    @property
    def em_light(self) -> bool:
        return self.em_mode and not self.em_final


@dataclass
class LocusResult:
    """What a full pass hands back to the parent for one locus.

    Parameters
    ----------
    locus_id : str
        The locus.
    outputs : dict[str, OutputText]
        Rows to append to the files of ``regions_activities/``.
    perf : dict or None
        The locus's row of ``performance_measurements.tsv``; ``None`` for a
        locus that was skipped before any work was measured.
    """

    locus_id: str
    outputs: dict[str, OutputText]
    perf: dict | None = None


class PerfLog(dict):
    """Per-locus timing and count statistics (``performance_measurements.tsv``)."""

    @contextmanager
    def timed(self, key: str):
        """Record the wall-clock seconds spent in the block under *key*."""
        start = time.time()
        try:
            yield
        finally:
            self[key] = time.time() - start


# --------------------------------------------------------------------------- #
# The parent: fan-out
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EmCheckpoint:
    """Where an interrupted multimapping EM continues.

    Parameters
    ----------
    start_iteration : int
        The iteration to run next.
    finished : set[str]
        Loci whose light M-step of that iteration already completed.
    final_only : bool
        The loop had already converged; only the final full pass is left.
    """

    start_iteration: int
    finished: set[str]
    final_only: bool


def _em_checkpoint(db_path: str, resume: bool) -> EmCheckpoint:
    """Read the EM checkpoint, or reset the EM state for a fresh loop."""
    point = multimap.em_resume_point(db_path) if resume else None
    if point is None:
        # Clear any per-iteration state from a previous run so a warm re-run
        # cannot consume stale λ / weights / activities.
        multimap.reset_em_state(db_path)
        run_state.clear_progress(db_path, run_state.EM_STAGE)
        return EmCheckpoint(0, set(), False)
    start_iteration, finished = point
    # The loop has already ended if its last run recorded the iteration the
    # final pass consumes and the checkpoint still sits there.
    stored_final = run_state.read_progress(db_path, run_state.EM_STAGE).get(
        run_state.FINAL_ITERATION
    )
    return EmCheckpoint(start_iteration, finished, stored_final == str(start_iteration))


class ORFActivityEstimator:
    """Orchestrate parallel ORF deconvolution over all genomic loci.

    Each locus stored in the database is dispatched to an isolated worker
    process; the workers hand their rendered rows back and this process is
    the only writer of the output files.

    Parameters
    ----------
    config : Config
        Parsed PRICE configuration object.

    Attributes
    ----------
    loci_ids : list of str
        Every locus id stored in the database, in dispatch order
        (``config.dispatch_order``): database order, or the loci with the
        most stored read bytes first.
    locus_timeout : int
        Wall-clock budget for one locus, in seconds:
        ``config.timeout`` per Ribo-seq run, capped at ``config.timeout_cap``.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.db_path = config.layout.db_path
        self._pool: ProcessPool | None = None
        self._log_queue = None
        self._peak_rss: tuple[float, str] = (0.0, "")

        with database.connect(self.db_path) as db:
            loci_ids = [
                locus_id
                for locus_id, in db.execute("SELECT locus_id FROM loci")
            ]
            n_runs = db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            read_bytes = dict(
                db.execute(
                    "SELECT locus_id, SUM(LENGTH(reads_blob)) FROM reads "
                    "GROUP BY locus_id"
                )
            )
        # A stable sort keeps database order among equals, so the fan-out is
        # the same from run to run.
        if config.dispatch_order == "largest":
            loci_ids = sorted(
                loci_ids, key=lambda locus_id: -read_bytes.get(locus_id, 0)
            )
        self.loci_ids = loci_ids

        # A locus is solved for every dataset at once — one activity column
        # per run — so its cost grows with how many there are.  The budget is
        # therefore per sample: an absolute one would abandon loci a wide run
        # could still have finished, while being needlessly generous to a
        # narrow one.  The cap keeps a wide panel from spending hours on one
        # locus.
        self.locus_timeout = config.timeout * max(1, n_runs)
        if config.timeout_cap:
            self.locus_timeout = min(self.locus_timeout, config.timeout_cap)
        logger.info(
            "per-locus timeout: %d s (%d s per sample, %d sample(s), cap %d s)",
            self.locus_timeout,
            config.timeout,
            max(1, n_runs),
            config.timeout_cap,
        )
        memory_gb = _physical_memory_gb()
        if memory_gb:
            logger.info(
                "%.0f GB of memory for %d worker(s): %.1f GB each; a worker's "
                "peak RSS grows with the number of samples (max_rss_mb in "
                "performance_measurements.tsv)",
                memory_gb,
                config.processes,
                memory_gb / max(1, config.processes),
            )

    @contextmanager
    def worker_pool(self):
        """Hold one process pool and log listener open for the block.

        The multimapping EM runs ~20 light M-steps plus the final full pass;
        creating and joining a pool and a manager each time costs ~1 s of
        wall per iteration, so the EM wraps its whole loop in this context.
        """
        _pebble_consts.channel_lock_timeout = _PEBBLE_CHANNEL_LOCK_TIMEOUT
        price2_logger = logging.getLogger("price2")
        manager = _MP_CONTEXT.Manager()
        log_queue = manager.Queue()
        listener = logging.handlers.QueueListener(
            log_queue, *price2_logger.handlers, respect_handler_level=True
        )
        listener.start()
        pool = ProcessPool(
            max_workers=self.config.processes,
            max_tasks=self.config.worker_max_tasks,
            initializer=init_worker,
            initargs=(self.config, log_queue),
            context=_MP_CONTEXT,
        )
        self._pool = pool
        self._log_queue = log_queue
        try:
            yield
        finally:
            pool.close()
            pool.join()
            listener.stop()
            manager.shutdown()
            self._pool = None
            self._log_queue = None

    def run_multimap_em(self, resume: bool = False) -> None:
        """Drive the multimapping-EM outer loop around the per-locus fan-out.

        Each iteration runs a light M-step fan-out (one interleaved Huber
        reweight per locus, warm-started, writing activities and per-slot λ),
        then a single global E-step that re-normalises each multimapping
        read's fractional weight across its loci.  The loop stops when the
        E-step's weight change falls below ``config.em_tol`` or after
        ``config.em_max_iter`` iterations, followed by one final full M-step
        (filtering + activity estimation + export) using the converged
        weights.

        Parameters
        ----------
        resume : bool, optional
            Continue an interrupted EM from its last checkpoint instead of
            restarting it (see :func:`price2.multimap.em_resume_point`).  The
            caller sets this from the run's
            :class:`~price2.run_state.ResumePlan`.
        """
        db_path = self.db_path

        if not multimap.has_multimap_index(db_path):
            logger.warning(
                "multimap_em is enabled but no populated linkage index was "
                "found in %s. Either no read maps to >=2 in-locus slots, or "
                "price.db was collected with multimap_em disabled -- in which "
                "case its multimapping reads were discarded as well, and only a "
                "cold re-collection with multimap_em=true can restore them. "
                "Running a single classic pass instead.",
                db_path,
            )
            self.run_orf_deconvolution()
            return

        database.enable_wal(db_path)
        # Fails here, with the reason, rather than in the E-step after a fan-out.
        multimap.load_linkage(db_path)
        checkpoint = _em_checkpoint(db_path, resume)
        # Loci with no multimap slots do not change across EM iterations, so
        # the light passes only need to touch the loci that carry slots.
        slot_loci = multimap.slot_locus_ids(db_path)
        if checkpoint.final_only:
            logger.info(
                "EM already converged; resuming at the final full M-step "
                "(iteration %d).",
                checkpoint.start_iteration,
            )
        elif resume and (checkpoint.start_iteration or checkpoint.finished):
            logger.info(
                "resuming the multimapping EM at iteration %d (%d of %d slot "
                "loci already done).",
                checkpoint.start_iteration,
                len(checkpoint.finished & slot_loci),
                len(slot_loci),
            )

        # One worker pool for the whole EM: every M-step would otherwise
        # rebuild it, paying a fresh pool + manager per iteration.
        with self.worker_pool():
            final_iteration = checkpoint.start_iteration
            if not checkpoint.final_only:
                final_iteration = self._em_loop(checkpoint, slot_loci)
                # From here a resume can skip straight to the final pass.
                run_state.write_progress(
                    db_path,
                    run_state.EM_STAGE,
                    {run_state.FINAL_ITERATION: str(final_iteration)},
                )
            run_stage(
                "EM final full M-step",
                lambda: self.run_orf_deconvolution(
                    em_iteration=final_iteration, em_final=True
                ),
            )

    def _em_loop(self, checkpoint: EmCheckpoint, slot_loci: set[str]) -> int:
        """Alternate light M-steps and E-steps; return the final pass's iteration."""
        config, db_path = self.config, self.db_path
        finished = checkpoint.finished
        last_iteration = checkpoint.start_iteration
        for iteration in range(checkpoint.start_iteration, config.em_max_iter):
            last_iteration = iteration
            # Only the resumed iteration has loci already behind it; every
            # later one starts empty.
            subset = slot_loci - finished
            finished = set()
            if subset:
                run_stage(
                    f"EM iteration {iteration} light M-step",
                    lambda: self.run_orf_deconvolution(
                        em_iteration=iteration, em_final=False, loci_subset=subset
                    ),
                )
            else:
                logger.info(
                    "EM iteration %d light M-step was already complete.", iteration
                )
            delta = multimap.e_step(db_path, iteration=iteration)
            logger.info(
                "EM iteration %d: read mass reassigned (L1 fraction) = %.3e",
                iteration,
                delta,
            )
            if delta < config.em_tol:
                logger.info(
                    "EM converged after %d iteration(s) (tol=%.1e).",
                    iteration + 1,
                    config.em_tol,
                )
                break
        return last_iteration + 1

    def run_orf_deconvolution(
        self,
        em_iteration: int | None = None,
        em_final: bool = True,
        loci_subset: set | None = None,
    ) -> None:
        """Run the per-locus ORF deconvolution over the loci of the run.

        Dispatches each locus to a worker process.  On a full pass the loci
        recorded as finished in ``price.db`` are skipped, so that an
        interrupted run resumes where it stopped.  The workers hand their
        rows back and this process, the only writer, appends them to
        ``<o_dir>/regions_activities/`` and records the locus as done;
        failed loci are recorded in ``failed_loci.txt`` there.

        Uses the pool held open by :meth:`worker_pool` when the caller has
        one, otherwise starts a pool for this call alone.

        Parameters
        ----------
        em_iteration : int, optional
            Multimapping-EM iteration index.  ``None`` (default) runs the
            classic single-pass pipeline.
        em_final : bool, optional
            See :class:`LocusJob`.  Intermediate passes neither skip nor
            record finished loci.
        loci_subset : set, optional
            Restrict the fan-out to these locus ids.  Light EM passes pass
            the loci that carry multimap slots; the others cannot change
            between iterations and are computed once, in the final pass.
        """
        if self._pool is None:
            with self.worker_pool():
                self._run_loci(em_iteration, em_final, loci_subset)
        else:
            self._run_loci(em_iteration, em_final, loci_subset)

    def _run_loci(
        self, em_iteration: int | None, em_final: bool, loci_subset: set | None
    ) -> None:
        layout = self.config.layout
        os.makedirs(layout.regions_activities_dir, exist_ok=True)

        # Dispatched in a fixed order (``loci_ids``), so the fan-out (and
        # with it the performance log and the first locus to time out) is
        # the same from run to run.
        excluded: set[str] = set()
        if loci_subset is not None:
            excluded |= set(self.loci_ids) - loci_subset
        # Resume-skip bookkeeping only applies to full passes; light EM
        # passes intentionally re-run every locus each iteration.
        if em_final:
            excluded |= set(
                run_state.read_progress(self.db_path, run_state.DECONVOLUTION_STAGE)
            )
        loci_ids = [locus_id for locus_id in self.loci_ids if locus_id not in excluded]

        price2_logger = logging.getLogger("price2")
        pbar = tqdm(total=len(loci_ids), disable=not logger.isEnabledFor(logging.INFO))
        writer = export.OutputWriter(layout.regions_activities_dir)
        progress = run_state.ProgressRecorder(
            self.db_path, run_state.DECONVOLUTION_STAGE
        )
        futures = {
            self._pool.schedule(
                process_loc,
                args=[LocusJob(locus_id, em_iteration, em_final)],
                timeout=self.locus_timeout,
            ): locus_id
            for locus_id in loci_ids
        }
        with logging_redirect_tqdm(loggers=[price2_logger]), progress:
            for future in as_completed(futures):
                locus_id = futures[future]
                try:
                    result = future.result()
                except TimeoutError:
                    reason = (
                        f"abandoned after {self.locus_timeout} s "
                        f"(config.timeout = {self.config.timeout} s per run)"
                    )
                    logger.error("locus %s %s", locus_id, reason)
                    with open(layout.failed_loci_path, "a") as fh:
                        fh.write(f"{locus_id}\n{reason}\n\n")
                except Exception as exc:
                    logger.error("locus %s failed: %s", locus_id, exc)
                    with open(layout.failed_loci_path, "a") as fh:
                        fh.write(f"{locus_id}\n{exc}\n{traceback.format_exc()}\n\n")
                else:
                    if result is not None:
                        self._record(result, writer, progress)
                finally:
                    pbar.update(1)
        pbar.close()
        if self._peak_rss[0]:
            logger.info(
                "peak worker RSS so far: %.0f MB (at locus %s)", *self._peak_rss
            )

    def _record(
        self,
        result: LocusResult,
        writer: export.OutputWriter,
        progress: run_state.ProgressRecorder,
    ) -> None:
        """Write a finished locus's rows, then mark it done.

        The order matters for resuming: a locus is only recorded as
        finished once its rows are on disk, and
        :func:`price2.run_state.repair_outputs` drops rows of unrecorded loci.
        """
        layout = self.config.layout
        writer.write(result.outputs)
        if result.perf is not None:
            rss = result.perf.get("max_rss_mb", 0.0)
            if rss > self._peak_rss[0]:
                self._peak_rss = (rss, result.locus_id)
            if self.config.export_performance_measurements:
                _append_performance(layout.performance_path, result.perf)
        progress.mark(result.locus_id)


# --------------------------------------------------------------------------- #
# The worker: one locus
# --------------------------------------------------------------------------- #


def process_loc(job: LocusJob) -> LocusResult | None:
    """Process one locus: filter ORFs, deconvolve, render the outputs.

    Runs in a worker process prepared by :func:`init_worker`.  A light EM
    pass writes its activities and λ for the E-step to the database and
    returns ``None``; a full pass continues through the likelihood-ratio
    filter and the final activity estimate and returns the rows for the
    parent to write.

    Parameters
    ----------
    job : LocusJob
        The locus and the pass to run.
    """
    ctx = _context()
    config, layout = ctx.config, ctx.layout
    perf = PerfLog(loc_id=job.locus_id)
    outputs: dict[str, OutputText] = {}
    t_start = time.time()

    runs = ctx.runs
    with perf.timed("db_time"):
        loc, prepared = _load_locus(job, ctx, perf)
    if not prepared and not _prepare_locus(job, ctx, loc, runs, perf, outputs):
        # No transcript survived: nothing to solve, nothing to write.
        return None if job.em_light else LocusResult(job.locus_id, {})
    # The prepared state is persisted after ``assign_reads_to_egs`` below, so
    # that the routing it builds is stored with it.
    save_prepared = job.em_light and job.em_iteration == 0

    slots = _apply_em_state(job, loc, runs, layout.db_path)
    mm_data = slots.by_run() if slots is not None else None
    with perf.timed("route_reads_time"):
        loc.assign_reads_to_egs(runs, mm_data)
    perf["read_count"] = sum(loc.counted_reads.values())

    if job.em_light:
        _light_mstep(
            job, loc, runs, slots, mm_data, save_prepared, config, layout, perf
        )
        return None

    _full_pass(loc, runs, config, perf, outputs)
    # The worker's high-water mark, not this locus's alone; its maximum over
    # the loci is what sizing ``processes`` by memory needs.
    perf["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    perf["overall_time"] = time.time() - t_start
    outputs.update(export.final_outputs(loc, config, runs))
    return LocusResult(job.locus_id, outputs, dict(perf))


def _load_runs(db_path: str) -> list[RiboSeqRun]:
    with database.connect(db_path) as db:
        return [
            database.unpickle_blob(blob)
            for _, blob in db.execute("SELECT * FROM runs")
        ]


def _load_locus(
    job: LocusJob, ctx: WorkerContext, perf: PerfLog
) -> tuple[Locus, bool]:
    """Return the locus to work on and whether it is already prepared.

    In EM mode the weight-independent prepared state (RGRs, filters, read
    routing) stored by the first light pass is reused: only the fractional
    response changes between iterations.  An intermediate light pass loads
    just the routing (arrays only) rather than the locus's whole object
    graph.  On a miss — classic mode, the first light iteration, or a
    non-multimapping locus in the final pass — the pre-RGR skeleton is
    loaded and still has to be prepared.
    """
    config, db_path = ctx.config, ctx.layout.db_path
    loc = None
    if job.em_light and job.em_iteration > 0:
        loc = multimap.load_light_locus(db_path, job.locus_id)
    if loc is None and job.em_mode:
        loc = multimap.load_prepared_locus(db_path, job.locus_id)
    prepared = loc is not None
    if loc is None:
        with database.connect(db_path) as db:
            row = db.execute(
                "SELECT loc_blob FROM loci WHERE locus_id = ?", (job.locus_id,)
            ).fetchone()
        loc = database.unpickle_blob(row[0])
    perf["chrom"] = loc.iv.chrom
    perf["strand"] = loc.iv.strand
    perf["start"] = loc.iv.start
    perf["end"] = loc.iv.end

    if prepared:
        # Reads are excluded from the prepared blob.  An intermediate light
        # M-step derives the response entirely from the routing, so only
        # iteration 0 and the final pass reload them.
        with perf.timed("load_reads_time"):
            if not (job.em_light and job.em_iteration > 0):
                loc.get_reads_from_db(
                    db_path, drop_multimappers=not config.multimap_em
                )
        # Keep the perf columns aligned with the prepare path; the skipped
        # stages report zero time and the counts come off the routing.
        n_rgrs = loc.routing.num_rgrs
        perf.update(
            build_rgrs_time=0.0,
            unfiltered_rgr_count=n_rgrs,
            well_fitting_reads_time=0.0,
            filtered_coverage_rgr_count=n_rgrs,
            coverage_filter_time=0.0,
            filtered_deconvolution_rgr_count=n_rgrs,
            deconvolution_filter_time=0.0,
            eg_time=0.0,
            eg_count=loc.routing.n_rows,
        )
    return loc, prepared


def _prepare_locus(
    job: LocusJob,
    ctx: WorkerContext,
    loc: Locus,
    runs: list[RiboSeqRun],
    perf: PerfLog,
    outputs: dict[str, OutputText],
) -> bool:
    """Generate the ORF candidates, filter them and build the equivalence groups.

    Adds the intermediate tables of each stage to *outputs* when
    ``export_all_steps`` is set.  Returns ``False`` when the locus keeps no
    transcript and is to be skipped.
    """
    config, db_path = ctx.config, ctx.layout.db_path
    export_steps = config.export_all_steps and not job.em_light

    with perf.timed("build_rgrs_time"):
        min_explained_reads = config.min_explained_reads_per_run * len(runs)
        has_transcripts = build_rgrs(
            loc, db_path, ctx.genome, config, min_explained_reads
        )
    if not has_transcripts:
        return False
    if export_steps:
        outputs.update(export.step_outputs(loc, config, "all"))

    with perf.timed("load_reads_time"):
        loc.get_reads_from_db(db_path, drop_multimappers=not config.multimap_em)

    perf["unfiltered_rgr_count"] = len(loc.rgrs)
    with perf.timed("well_fitting_reads_time"):
        if config.coverage_filter or config.deconvolution_filter:
            loc.make_well_fitting_reads(runs)

    with perf.timed("coverage_filter_time"):
        if config.coverage_filter:
            loc.coverage_filter_rgrs(config)
        if export_steps:
            outputs.update(export.step_outputs(loc, config, "coverage_filtered"))
        perf["filtered_coverage_rgr_count"] = len(loc.rgrs)

    with perf.timed("deconvolution_filter_time"):
        if config.deconvolution_filter:
            loc.deconvolution_filter_rgrs(config)
        perf["filtered_deconvolution_rgr_count"] = len(loc.rgrs)
    if export_steps:
        outputs.update(export.step_outputs(loc, config, "deconvolution_filtered"))

    loc.update_transcript_rgrs()
    with perf.timed("eg_time"):
        loc.egs = make_equivalence_groups(loc, runs)
    perf["eg_count"] = loc.egs.n_rows
    return True


def _apply_em_state(
    job: LocusJob, loc: Locus, runs: list[RiboSeqRun], db_path: str
) -> multimap.LocusSlots | None:
    """Warm-start the activities and load the locus's slots with their weights.

    The RGR set is final here (all pre-deconvolution filters have run), so
    warm-start activities align by ``rgr.id`` and the weights apply per
    multimapping slot.  ``None`` outside EM mode (classic full counting)
    and for a locus without multimapping slots.
    """
    if not job.em_mode:
        return None
    if job.em_iteration > 0:
        warm = multimap.load_warm_activities(
            db_path, job.locus_id, job.em_iteration - 1
        )
        if warm is not None:
            loc.set_warm_start(warm, len(runs))
    return multimap.load_locus_mm_data(db_path, job.locus_id, job.em_iteration)


def _light_mstep(
    job: LocusJob,
    loc: Locus,
    runs: list[RiboSeqRun],
    slots: multimap.LocusSlots | None,
    mm_data: dict | None,
    save_prepared: bool,
    config: Config,
    layout: RunLayout,
    perf: PerfLog,
) -> None:
    """One Huber reweight, then persist activities and λ for the E-step."""
    db_path = layout.db_path
    loc.deconvolve(config, runs=runs, max_outer=config.em_huber_steps, prune=False)
    # After the first M-step, drop ORFs that are inactive in every run.  They
    # rarely revive in later M-steps, so pruning them now shrinks the design
    # matrix every later iteration and the final pass rebuild.  The routing
    # and the prepared locus are rebuilt to match before they are persisted
    # below.
    if save_prepared and config.em_prune_after_first_mstep:
        perf["mstep_pruned_orf_count"] = loc.prune_inactive_orfs(
            config, runs, mm_data
        )
    if save_prepared:
        multimap.save_prepared_locus(db_path, job.locus_id, loc)
        # The design matrix of the routing as persisted (rebuilt here if the
        # pruning above changed the routing), so no later pass builds it.
        multimap.save_locus_routing(
            db_path, job.locus_id, loc, loc.sparse_system(runs).X
        )
    multimap.write_locus_em_output(
        db_path,
        job.locus_id,
        job.em_iteration,
        loc.activities_by_id(),
        loc.compute_multimap_lambdas(runs),
        slots,
    )


def _full_pass(
    loc: Locus,
    runs: list[RiboSeqRun],
    config: Config,
    perf: PerfLog,
    outputs: dict[str, OutputText],
) -> None:
    """Group-LASSO deconvolution, likelihood-ratio filter and final estimate."""
    with perf.timed("optimization_time"):
        loc.deconvolve(config, runs=runs)
    perf["irls_outer_iterations"] = loc.irls_outer_iterations
    perf["filtered_deconvoluted_rgr_count"] = len(loc.rgrs)
    if config.export_all_steps:
        outputs.update(export.step_outputs(loc, config, "deconvoluted"))

    if config.likelihood_ratio_filter:
        with perf.timed("likelihood_ratio_time"):
            loc.likelihood_ratio_filtering(config, runs)
        perf["filtered_lrt_rgr_count"] = len(loc.rgrs)
        perf["orf_count"] = sum(1 for rgr in loc.rgrs if rgr.is_orf)

    with perf.timed("activity_time"):
        loc.estimate_activities(runs, config)

    perf["gene_number"] = len(loc.gene_ids_complete)
    perf["transcripts_number"] = loc.transcripts_number
    perf["exon_length"] = loc.exon_length


def _physical_memory_gb() -> float:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
    except (ValueError, OSError, AttributeError):
        return 0.0


def _perf_field(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3e}"
    return "" if value is None else str(value)


def _append_performance(path: str, perf: dict) -> None:
    """Append one locus's row to ``performance_measurements.tsv``.

    The row is written against :data:`PERF_COLUMNS`, so the columns are the
    same for every locus; a key outside the list is a programming error.
    """
    unknown = sorted(set(perf) - set(PERF_COLUMNS))
    if unknown:
        raise KeyError(f"performance keys missing from PERF_COLUMNS: {unknown}")
    header = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        if header:
            writer.writerow(PERF_COLUMNS)
        writer.writerow(_perf_field(perf.get(column)) for column in PERF_COLUMNS)
