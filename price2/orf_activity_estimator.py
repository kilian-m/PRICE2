"""Parallel ORF activity estimation across genomic loci.

:class:`ORFActivityEstimator` fans the loci of a run out over a
``pebble.ProcessPool``; :func:`process_loc` is what one worker does with one
locus: load it, generate and filter ORF candidates, build the equivalence
groups, deconvolve, and write the results.  Each locus runs in an isolated
worker so that timeouts and crashes stay contained.

The workers are started with the ``forkserver`` method: numba's JIT state
and SQLite handles are not safe to ``fork``.  :func:`init_worker` runs once
per worker process and installs the :class:`WorkerContext` every locus of
that worker shares.
"""

from __future__ import annotations

import logging
import logging.handlers
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import TimeoutError, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace

import pandas as pd
from pebble import ProcessPool
from pebble.common import CONSTS as _pebble_consts
from pyfaidx import Fasta
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from price2 import database, export, multimap
from price2.export import OutputText
from price2.config import Config
from price2.equivalence_groups import make_equivalence_groups
from price2.layout import RunLayout
from price2.locus import Locus
from price2.orf_candidates import build_rgrs
from price2.ribo_seq_run import RiboSeqRun

logger = logging.getLogger(__name__)

_MP_CONTEXT = mp.get_context("forkserver")

#: pebble's default channel lock timeout is 60 s; with ``forkserver`` and many
#: workers (e.g. 80) the result-pipe mutex can be contended for longer, which
#: makes workers exit with "Abnormal termination".  600 s gives ample headroom.
_PEBBLE_CHANNEL_LOCK_TIMEOUT = 600


# --------------------------------------------------------------------------- #
# What a worker knows and what a job asks of it
# --------------------------------------------------------------------------- #


@dataclass
class WorkerContext:
    """State shared by every locus a worker process handles.

    Attributes
    ----------
    config : Config
        The run configuration, including a broker queue when a GPU broker
        is running.
    layout : RunLayout
        The run's files.
    genome : pyfaidx.Fasta
        The reference genome, opened once per worker.
    """

    config: Config
    layout: RunLayout
    genome: Fasta


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
    worker_logger = logging.getLogger("price2")
    if not worker_logger.handlers:
        worker_logger.addHandler(logging.handlers.QueueHandler(log_queue))
        worker_logger.setLevel(config.log_level)
        worker_logger.propagate = False
    _CONTEXT = WorkerContext(config, config.layout, Fasta(config.fasta_path))


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


class ORFActivityEstimator:
    """Orchestrate parallel ORF deconvolution over all genomic loci.

    Each locus stored in the database is dispatched to an isolated worker
    process; every result is written to disk by the worker itself.

    Parameters
    ----------
    config : Config
        Parsed PRICE configuration object.

    Attributes
    ----------
    loci_ids : list of str
        Every locus id stored in the database.
    locus_timeout : int
        Wall-clock budget for one locus, in seconds:
        ``config.timeout`` per Ribo-seq run.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.db_path = config.layout.db_path
        # The configuration handed to the workers; a running GPU broker adds
        # its request queue to this copy, never to ``config`` itself.
        self._worker_config = config
        self._broker = None
        self._pool: ProcessPool | None = None
        self._log_queue = None

        with database.connect(self.db_path) as db:
            self.loci_ids = [
                locus_id
                for locus_id, in db.execute("SELECT locus_id FROM loci")
            ]
            n_runs = db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

        # A locus is solved for every dataset at once — one activity column
        # per run — so its cost grows with how many there are.  The budget is
        # therefore per sample: an absolute one would abandon loci a wide run
        # could still have finished, while being needlessly generous to a
        # narrow one.
        self.locus_timeout = config.timeout * max(1, n_runs)
        logger.info(
            "per-locus timeout: %d s (%d s per sample, %d sample(s))",
            self.locus_timeout,
            config.timeout,
            max(1, n_runs),
        )

    @contextmanager
    def gpu_broker_pool(self):
        """Hold one GPU broker pool open for the duration of the block.

        The broker serves every worker over shared memory from a single CUDA
        context per broker process (see :mod:`price2.gpu_broker`); starting
        it once around the whole multimapping EM avoids paying that context
        per M-step.  A no-op when the broker is disabled, cannot start, or is
        already running.
        """
        started = self._broker is None and self._start_gpu_broker()
        try:
            yield
        finally:
            if started:
                self._broker.stop()
                self._broker = None
                self._worker_config = self.config

    def _start_gpu_broker(self) -> bool:
        config = self.config
        if config.inner_solver != "mu" or not config.mu_broker:
            return False
        try:
            from price2.gpu_broker import GpuBroker

            broker = GpuBroker(
                n_procs=config.mu_broker_procs,
                n_streams=config.mu_broker_streams,
                dtype_str=config.mu_dtype,
            )
            broker.start()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "GPU broker unavailable (%s); workers use the configured "
                "per-worker MU path",
                exc,
            )
            return False
        self._broker = broker
        self._worker_config = replace(config, mu_broker_req_q=broker.req_q)
        logger.info(
            "GPU deconvolution broker pool started (%d procs x %d streams, %s)",
            config.mu_broker_procs,
            config.mu_broker_streams,
            config.mu_dtype,
        )
        return True

    @contextmanager
    def worker_pool(self):
        """Hold one process pool and log listener open for the block.

        The multimapping EM runs ~20 light M-steps plus the final full pass;
        creating and joining a pool and a manager each time costs ~1 s of
        wall per iteration, so the EM wraps its whole loop in this context.
        Start a GPU broker (:meth:`gpu_broker_pool`) before the pool: the
        workers receive its queue at start-up.
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
            initargs=(self._worker_config, log_queue),
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

    def run_orf_deconvolution(
        self,
        em_iteration: int | None = None,
        em_final: bool = True,
        loci_subset: set | None = None,
    ) -> None:
        """Run the per-locus ORF deconvolution over the loci of the run.

        Dispatches each locus to a worker process.  On a full pass the loci
        listed in ``<w_dir>/processed_loci.txt`` are skipped, so that an
        interrupted run resumes where it stopped.  The workers hand their
        rows back and this process, the only writer, appends them to
        ``<o_dir>/regions_activities/`` and records the locus as done;
        failed loci are recorded in ``failed_loci.txt`` there.

        Uses the pool held open by :meth:`worker_pool` when the caller has
        one, otherwise starts a broker and a pool for this call alone.

        Parameters
        ----------
        em_iteration : int, optional
            Multimapping-EM iteration index.  ``None`` (default) runs the
            classic single-pass pipeline.
        em_final : bool, optional
            See :class:`LocusJob`.  Intermediate passes do not touch
            ``processed_loci.txt``.
        loci_subset : set, optional
            Restrict the fan-out to these locus ids.  Light EM passes pass
            the loci that carry multimap slots; the others cannot change
            between iterations and are computed once, in the final pass.
        """
        if self._pool is None:
            with self.gpu_broker_pool(), self.worker_pool():
                self._run_loci(em_iteration, em_final, loci_subset)
        else:
            self._run_loci(em_iteration, em_final, loci_subset)

    def _run_loci(
        self, em_iteration: int | None, em_final: bool, loci_subset: set | None
    ) -> None:
        layout = self.config.layout
        os.makedirs(layout.regions_activities_dir, exist_ok=True)

        loci_ids = set(self.loci_ids)
        if loci_subset is not None:
            loci_ids &= loci_subset
        # Resume-skip bookkeeping only applies to full passes; light EM
        # passes intentionally re-run every locus each iteration.
        if em_final and os.path.exists(layout.processed_loci_path):
            with open(layout.processed_loci_path) as fh:
                loci_ids -= {line.strip() for line in fh if line.strip()}

        price2_logger = logging.getLogger("price2")
        log_level = logging.getLevelName(self.config.log_level)
        pbar = tqdm(total=len(loci_ids), disable=log_level > logging.INFO)
        writer = export.OutputWriter(layout.regions_activities_dir)
        futures = {
            self._pool.schedule(
                process_loc,
                args=[LocusJob(locus_id, em_iteration, em_final)],
                timeout=self.locus_timeout,
            ): locus_id
            for locus_id in loci_ids
        }
        with logging_redirect_tqdm(loggers=[price2_logger]):
            for future in as_completed(futures):
                locus_id = futures[future]
                try:
                    result = future.result()
                except (TimeoutError, Exception) as exc:
                    logger.error("locus %s failed: %s", locus_id, exc)
                    with open(layout.failed_loci_path, "a") as fh:
                        fh.write(f"{locus_id}\n{exc}\n{traceback.format_exc()}\n\n")
                else:
                    if result is not None:
                        self._record(result, writer)
                finally:
                    pbar.update(1)
        pbar.close()

    def _record(self, result: LocusResult, writer: export.OutputWriter) -> None:
        """Write a finished locus's rows, then mark it done.

        The order matters for resuming: a locus is only listed in
        ``processed_loci.txt`` once its rows are on disk, and
        :func:`price2.run_state.repair_outputs` drops rows of unlisted loci.
        """
        layout = self.config.layout
        writer.write(result.outputs)
        if result.perf is not None and self.config.export_performance_measurements:
            _append_performance(layout.performance_path, result.perf)
        export.append_line(layout.processed_loci_path, result.locus_id)


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

    with perf.timed("db_time"):
        runs = _load_runs(layout.db_path)
        loc, prepared = _load_locus(job, ctx, perf)
    if not prepared and not _prepare_locus(job, ctx, loc, runs, perf, outputs):
        # No transcript survived: nothing to solve, nothing to write.
        return None if job.em_light else LocusResult(job.locus_id, {})
    # The prepared state is persisted after ``assign_reads_to_egs`` below, so
    # that the routing it builds is stored with it.
    save_prepared = job.em_light and job.em_iteration == 0

    slots = _apply_em_state(job, loc, runs, layout.db_path)
    mm_data = slots.by_run() if slots is not None else None
    with perf.timed("proc_reads_2_time"):
        loc.assign_reads_to_egs(runs, mm_data)
    perf["read_count"] = sum(loc.counted_reads.values())

    if job.em_light:
        _light_mstep(
            job, loc, runs, slots, mm_data, save_prepared, config, layout, perf
        )
        return None

    _full_pass(loc, runs, config, perf, outputs)
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
            assign_reads_time=0.0,
            filtered_coverage_rgr_count=n_rgrs,
            coverage_filter_time=0.0,
            filtered_deconvolution_rgr_count=n_rgrs,
            filter_2_time=0.0,
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
    with perf.timed("assign_reads_time"):
        if config.coverage_filter or config.deconvolution_filter:
            loc.make_well_fitting_reads(runs)

    with perf.timed("coverage_filter_time"):
        if config.coverage_filter:
            loc.coverage_filter_rgrs(config)
        if export_steps:
            outputs.update(export.step_outputs(loc, config, "coverage_filtered"))
        perf["filtered_coverage_rgr_count"] = len(loc.rgrs)

    with perf.timed("filter_2_time"):
        if config.deconvolution_filter:
            loc.deconvolution_filter_rgrs(config)
        perf["filtered_deconvolution_rgr_count"] = len(loc.rgrs)
    if export_steps:
        outputs.update(export.step_outputs(loc, config, "deconvolution_filtered"))

    loc.update_transcript_rgrs()
    with perf.timed("eg_time"):
        loc.egs = make_equivalence_groups(loc, runs)
    perf["eg_count"] = sum(len(egs) for egs in loc.egs.values())
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
        multimap.save_locus_routing(db_path, job.locus_id, loc)
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
        perf["orf_count"] = sum(1 for rgr in loc.rgrs if rgr.type == "ORF")

    with perf.timed("activity_time"):
        loc.estimate_activities(runs, config)

    perf["gene_number"] = len(loc.gene_ids_complete)
    perf["transcripts_number"] = loc.transcripts_number
    perf["exon_length"] = loc.exon_length


def _append_performance(path: str, perf: dict) -> None:
    header = not os.path.exists(path)
    with open(path, "a") as fh:
        fh.write(
            pd.DataFrame([perf]).to_csv(
                header=header, index=False, float_format="{:.2e}".format, sep="\t"
            )
        )
