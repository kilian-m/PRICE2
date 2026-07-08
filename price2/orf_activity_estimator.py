"""Parallel ORF activity estimation across genomic loci.

This module orchestrates the per-locus deconvolution pipeline:
loading data from SQLite, filtering ORF candidates, constructing
equivalence groups, running group-LASSO optimisation and writing
results.  Each locus is processed in an isolated worker process via
``pebble.ProcessPool`` so that timeouts and crashes are isolated.
"""

import glob
import logging
import logging.handlers
import os
import sqlite3 as sql
import time
import traceback
from concurrent.futures import TimeoutError, as_completed
from pickle import loads

import multiprocessing as mp
import pandas as pd
from filelock import FileLock
from pebble import ProcessPool
from pyfaidx import Fasta
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from price2 import multimap
from price2.data_collector import build_rgrs
from price2.equivalence_groups import make_equivalence_groups

# Must be set before any ProcessPool is created.  ``forkserver`` is
# required because numba JIT state and SQLite handles are not safe to
# fork directly.
mp.set_start_method("forkserver", force=True)

# Increase pebble's channel lock timeout.  The default is 60 s; with
# forkserver and many workers (e.g. 80) the result-pipe mutex can be
# contended long enough to exceed that, causing workers to exit with
# code 1 ("Abnormal termination").  600 s gives ample headroom.
from pebble.common import CONSTS as _pebble_consts

_pebble_consts.channel_lock_timeout = 600

logger = logging.getLogger(__name__)

# Worker-local pyfaidx handle cache.  ``max_tasks=1`` means each worker
# handles one locus, so this typically opens once per worker; the cache
# keeps the code defensive if that ever changes.
_genome_cache: dict[str, Fasta] = {}


def _get_genome(path: str) -> Fasta:
    """Return a worker-local :class:`pyfaidx.Fasta` for *path*."""
    handle = _genome_cache.get(path)
    if handle is None:
        handle = Fasta(path)
        _genome_cache[path] = handle
    return handle


class ORFActivityEstimator:
    """Orchestrate parallel ORF deconvolution over all genomic loci.

    Each locus stored in the PRICE SQLite database is dispatched to an
    isolated worker process.  Results are accumulated in
    ``self.loci`` and performance statistics in
    ``self.performance_df``.

    Parameters
    ----------
    config : Config
        Parsed PRICE configuration object.
    """

    def __init__(self, config) -> None:
        """Initialise estimator and load locus IDs from the database.

        Parameters
        ----------
        config : Config
            Parsed PRICE configuration object.
        """
        self.config = config
        self.db_path = f"{config.w_dir}/price.db"

        db = sql.connect(self.db_path)
        cur = db.cursor()

        cur.execute("SELECT locus_id FROM loci")
        self.loci_ids = [id for id, in cur.fetchall()]

    def run_orf_deconvolution(
        self,
        em_iteration: int | None = None,
        em_final: bool = True,
        loci_subset: set | None = None,
    ) -> None:
        """Run the full per-locus ORF deconvolution pipeline.

        Dispatches each locus to a worker process via
        :class:`pebble.ProcessPool`.  Already-processed loci (detected
        via ``performance_measurements.tsv``) are skipped so that the
        run can be resumed after a crash.

        Results are written incrementally to
        ``<o_dir>/regions_activities/results.tsv``.  Failed loci are
        logged to ``<o_dir>/regions_activities/failed_loci.txt``.

        When ``config.save_memory`` is ``False`` the in-memory
        attributes ``self.loci`` and ``self.performance_df`` are
        populated after the pool finishes.

        Parameters
        ----------
        em_iteration : int, optional
            Multimapping-EM iteration index.  ``None`` (default) runs the
            classic single-pass pipeline.  When set, workers load this
            iteration's fractional weights and the previous iteration's
            activities (warm start).
        em_final : bool, optional
            ``True`` for a classic run or the final EM pass (full
            deconvolution + filtering + export + resume bookkeeping).
            ``False`` for a light intermediate EM pass (one Huber
            reweight, no pruning/export; writes activities and λ for the
            E-step).  Intermediate passes do not touch
            ``processed_loci.txt``.
        loci_subset : set, optional
            When given, restrict the fan-out to these locus ids.  Light EM
            passes pass the set of loci that carry multimap slots, since
            loci with no multimapping reads never change across EM
            iterations and only need computing once (in the final pass).
        """
        self.loci = {}
        loci_ids = set(self.loci_ids)
        if loci_subset is not None:
            loci_ids = loci_ids & loci_subset

        performance_measurements = {}

        db = sql.connect(self.db_path)
        cur = db.cursor()
        cur.execute("SELECT run_id FROM runs")
        run_ids = [id for id, in cur.fetchall()]
        db.close()

        ra_dir = os.path.join(self.config.o_dir, "regions_activities")
        os.makedirs(ra_dir, exist_ok=True)

        # Resume-skip bookkeeping only applies to full passes; light EM
        # passes intentionally re-run every locus each iteration.
        if em_final:
            processed_loci_path = os.path.join(
                self.config.w_dir, "processed_loci.txt"
            )
            if os.path.exists(processed_loci_path):
                with open(processed_loci_path) as fh:
                    processed_loc_ids = {
                        line.strip() for line in fh if line.strip()
                    }
                loci_ids = loci_ids - processed_loc_ids
            elif os.path.exists(
                f"{self.config.o_dir}/performance_measurements.tsv"
            ):
                processed_loc_ids = set(
                    pd.read_csv(
                        f"{self.config.o_dir}/performance_measurements.tsv",
                        sep="\t",
                    )["loc_id"]
                )
                loci_ids = loci_ids - processed_loc_ids

        loci_ids = list(loci_ids)
        log_level_num = logging.getLevelName(self.config.log_level)
        pbar = tqdm(total=len(loci_ids), disable=log_level_num > logging.INFO)

        # Optional single-context GPU deconvolution broker: started once here and
        # shared across the whole worker pool via ``config.mu_broker_req_q`` (a
        # Manager queue proxy that pickles to each process_loc worker). One broker
        # holds one CUDA context and serves every worker over shared memory, so
        # device VRAM does not scale with ``config.processes``. Falls back to the
        # configured per-worker MU path if the broker cannot start.
        broker = None
        self.config.mu_broker_req_q = None
        if (
            getattr(self.config, "inner_solver", "lbfgs") == "mu"
            and getattr(self.config, "mu_broker", False)
        ):
            try:
                from price2.gpu_broker import GpuBroker

                broker = GpuBroker(
                    n_procs=getattr(self.config, "mu_broker_procs", 4),
                    n_streams=getattr(self.config, "mu_broker_streams", 2),
                    dtype_str=getattr(self.config, "mu_dtype", "float32"),
                )
                broker.start()
                self.config.mu_broker_req_q = broker.req_q
                logger.info(
                    "GPU deconvolution broker pool started (%d procs x %d streams, %s)",
                    getattr(self.config, "mu_broker_procs", 4),
                    getattr(self.config, "mu_broker_streams", 2),
                    getattr(self.config, "mu_dtype", "float32"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "GPU broker unavailable (%s); workers use the configured "
                    "per-worker MU path",
                    exc,
                )
                broker = None
                self.config.mu_broker_req_q = None

        price2_logger = logging.getLogger("price2")
        manager = mp.Manager()
        log_queue = manager.Queue()
        listener = logging.handlers.QueueListener(
            log_queue, *price2_logger.handlers, respect_handler_level=True
        )
        listener.start()
        try:
            with ProcessPool(max_workers=self.config.processes, max_tasks=1) as pool:
                futures = {
                    pool.schedule(
                        process_loc,
                        args=[
                            (
                                locus_id,
                                self.config,
                                log_queue,
                                em_iteration,
                                em_final,
                            )
                        ],
                        timeout=self.config.timeout,
                    ): locus_id
                    for locus_id in loci_ids
                }

                results = {}
                with logging_redirect_tqdm(loggers=[price2_logger]):
                    for fut in as_completed(futures):
                        loc_id = futures[fut]
                        try:
                            result = fut.result()
                            if not self.config.save_memory:
                                results[loc_id] = result
                        except (TimeoutError, Exception) as e:
                            stack = traceback.format_exc()
                            logger.error("locus %s failed: %s", loc_id, e)
                            lock = FileLock(f"{ra_dir}/failed_loci.txt.lock")
                            with lock:
                                with open(f"{ra_dir}/failed_loci.txt", "a") as f:
                                    f.write(f"{loc_id}\n{str(e)}\n{stack}\n\n")
                            if not self.config.save_memory:
                                results[loc_id] = e
                        finally:
                            pbar.update(1)
        finally:
            listener.stop()
            manager.shutdown()
            if broker is not None:
                broker.stop()
                self.config.mu_broker_req_q = None

        pbar.close()

        if not self.config.save_memory:
            self.loci = {}
            self.failed_loci = {}
            for loc_id, result in results.items():
                if isinstance(result, Exception):
                    self.failed_loci[loc_id] = result
                else:
                    self.loci[loc_id] = result[0]
                    performance_measurements[loc_id] = result[1]

            self.performance_df = pd.DataFrame(performance_measurements).T

        lock_files = glob.glob(os.path.join(ra_dir, "*.lock"))
        for lock_file in lock_files:
            os.remove(lock_file)


def process_loc(arguments: tuple):
    """Process a single locus: filter ORFs, deconvolve, write output.

    This function is executed in a separate worker process by
    :meth:`ORFActivityEstimator.run_orf_deconvolution`.

    Parameters
    ----------
    arguments : tuple
        A ``(locus_id, config, log_queue)`` triple where *locus_id* is
        the string identifier stored in the SQLite database, *config* is
        the :class:`~price2.config.Config` instance, and *log_queue* is
        a :class:`multiprocessing.Queue` connected to the main-process
        :class:`~logging.handlers.QueueListener`.

    Returns
    -------
    tuple[Locus, dict] or None
        ``(locus, performance_measurements)`` when
        ``config.save_memory`` is ``False``, otherwise ``None``.
    """
    loc_id, config, log_queue, em_iteration, em_final = arguments
    em_mode = em_iteration is not None
    em_light = em_mode and not em_final

    # Route all price2 log records back to the main process.
    worker_logger = logging.getLogger("price2")
    if not worker_logger.handlers:
        worker_logger.addHandler(logging.handlers.QueueHandler(log_queue))
        worker_logger.setLevel(config.log_level)
        worker_logger.propagate = False

    performance_measurements: dict = {}
    performance_measurements["loc_id"] = loc_id
    t_start = time.time()
    t1 = time.time()

    if not hasattr(config, "base_o_dir"):
        config.base_o_dir = os.path.join(config.o_dir, "regions_activities")

    # --- Load runs (always needed) ---
    db_path = f"{config.w_dir}/price.db"
    db = sql.connect(db_path, timeout=120)
    cur = db.cursor()
    cur.execute("PRAGMA busy_timeout = 120000")
    cur.execute("SELECT * FROM runs")
    runs = [loads(blob) for id, blob in cur.fetchall()]
    db.close()

    # In EM mode, reuse the weight-independent prepared state (RGRs,
    # filters, equivalence-group geometry) cached by the first light pass;
    # only the fractional response y changes between iterations, so later
    # iterations and the final pass skip ORF generation, both filter passes
    # and the EG build.  ``None`` on a miss: classic mode, the first light
    # iteration, or a non-multimapping locus in the final pass.
    loc = multimap.load_prepared_locus(db_path, loc_id) if em_mode else None

    if loc is not None:
        performance_measurements["chrom"] = loc.iv.chrom
        performance_measurements["strand"] = loc.iv.strand
        performance_measurements["start"] = loc.iv.start
        performance_measurements["end"] = loc.iv.end
        performance_measurements["db_time"] = time.time() - t1
        # Reads are excluded from the cache blob; reload them.
        t1 = time.time()
        loc.get_reads_from_db(db_path)
        performance_measurements["load_reads_time"] = time.time() - t1
        # Keep perf columns aligned with the full-prepare path (the skipped
        # stages report zero time).
        performance_measurements["build_rgrs_time"] = 0.0
        performance_measurements["unfiltered_rgr_count"] = len(loc.rgr_set)
        performance_measurements["assign_reads_time"] = 0.0
        performance_measurements["filtered_coverage_rgr_count"] = len(loc.rgr_set)
        performance_measurements["coverage_filter_time"] = 0.0
        performance_measurements["filtered_deconvolution_rgr_count"] = len(
            loc.rgr_set
        )
        performance_measurements["filter_2_time"] = 0.0
        performance_measurements["eg_time"] = 0.0
        performance_measurements["eg_count"] = sum(
            len(egs) for egs in loc.egs.values()
        )
    else:
        # --- Load locus skeleton from database ---
        db = sql.connect(db_path, timeout=120)
        cur = db.cursor()
        cur.execute("PRAGMA busy_timeout = 120000")
        cur.execute("SELECT * FROM loci WHERE locus_id = ?", (loc_id,))
        loc = loads(cur.fetchone()[1])
        performance_measurements["chrom"] = loc.iv.chrom
        performance_measurements["strand"] = loc.iv.strand
        performance_measurements["start"] = loc.iv.start
        performance_measurements["end"] = loc.iv.end
        db.close()

        t2 = time.time()
        performance_measurements["db_time"] = t2 - t1

        # --- Build ORF candidates (formerly DataCollector.collect_loci) ---
        t1 = time.time()
        genome = _get_genome(config.fasta_path)
        min_explained_reads = config.min_explained_reads_per_run * len(runs)
        has_transcripts = build_rgrs(
            loc, db_path, genome, config, min_explained_reads
        )
        performance_measurements["build_rgrs_time"] = time.time() - t1
        if not has_transcripts:
            if not em_light:
                processed_loci_path = os.path.join(
                    config.w_dir, "processed_loci.txt"
                )
                lock = FileLock(processed_loci_path + ".lock")
                with lock:
                    with open(processed_loci_path, "a") as f:
                        f.write(loc_id + "\n")
            return None if config.save_memory else (loc, performance_measurements)

        if config.export_all_steps and not em_light:
            if config.export_gtf:
                loc.to_gtf(
                    f"{config.base_o_dir}/all",
                    write_orfs=config.export_orfs,
                    write_loci=config.export_loci,
                    write_transcripts=config.export_transcripts,
                )
            if config.export_tsv and config.export_orfs:
                loc.to_tsv(f"{config.base_o_dir}/all")
            if config.export_bed and config.export_orfs:
                loc.to_bed(f"{config.base_o_dir}/all")
        loc.rgr_filter_sets = {}

        # --- Load reads ---
        t1 = time.time()
        loc.get_reads_from_db(db_path)
        t2 = time.time()
        performance_measurements["load_reads_time"] = t2 - t1

        # --- Assign reads to ORF candidates ---
        t1 = time.time()
        performance_measurements["unfiltered_rgr_count"] = len(loc.rgr_set)
        loc.rgr_filter_sets["unfiltered"] = loc.rgr_set

        if config.coverage_filter or config.deconvolution_filter:
            loc.make_well_fitting_reads(runs)

        t2 = time.time()
        performance_measurements["assign_reads_time"] = t2 - t1

        # --- Coverage filter ---
        t1 = time.time()

        if config.coverage_filter:
            loc.coverage_filter_rgrs(config)

        if config.export_all_steps and not em_light:
            if config.export_gtf:
                loc.to_gtf(
                    f"{config.base_o_dir}/coverage_filtered",
                    write_orfs=config.export_orfs,
                    write_loci=config.export_loci,
                    write_transcripts=config.export_transcripts,
                )
            if config.export_tsv and config.export_orfs:
                loc.to_tsv(f"{config.base_o_dir}/coverage_filtered")
            if config.export_bed and config.export_orfs:
                loc.to_bed(f"{config.base_o_dir}/coverage_filtered")
        loc.rgr_filter_sets["coverage_filtered"] = loc.rgr_set

        performance_measurements["filtered_coverage_rgr_count"] = len(loc.rgr_set)
        t2 = time.time()
        performance_measurements["coverage_filter_time"] = t2 - t1

        # --- Deconvolution filter ---
        t1 = time.time()
        if config.deconvolution_filter:
            loc.deconvolution_filter_rgrs(config)

        loc.rgr_filter_sets["deconvolution_filtered"] = loc.rgr_set
        performance_measurements["filtered_deconvolution_rgr_count"] = len(
            loc.rgr_set
        )
        t2 = time.time()
        performance_measurements["filter_2_time"] = t2 - t1

        if config.export_all_steps and not em_light:
            if config.export_gtf:
                loc.to_gtf(
                    f"{config.base_o_dir}/deconvolution_filtered",
                    write_orfs=config.export_orfs,
                    write_loci=config.export_loci,
                    write_transcripts=config.export_transcripts,
                )
            if config.export_tsv and config.export_orfs:
                loc.to_tsv(f"{config.base_o_dir}/deconvolution_filtered")
            if config.export_bed and config.export_orfs:
                loc.to_bed(f"{config.base_o_dir}/deconvolution_filtered")

        # --- Equivalence groups ---
        for tr in loc.transcripts:
            tr.update_with_filtered_orfs(loc.rgr_set)

        t1 = time.time()
        loc.egs = make_equivalence_groups(loc, runs)
        t2 = time.time()

        performance_measurements["eg_time"] = t2 - t1
        performance_measurements["eg_count"] = sum(
            len(egs) for egs in loc.egs.values()
        )

        # --- Cache prepared state for subsequent EM iterations ---
        if em_light and em_iteration == 0:
            multimap.save_prepared_locus(db_path, loc_id, loc)

    # --- Multimapping EM: warm start + fractional read weights ---
    # The RGR set is final here (all pre-deconvolution filters have run),
    # so warm-start activities align by rgr.id and fractional weights can
    # be applied per multimapping slot.
    mm_data = None
    if em_mode:
        if em_iteration > 0:
            warm = multimap.load_warm_activities(
                db_path, loc_id, em_iteration - 1
            )
            if warm is not None:
                loc.set_warm_start(warm, len(runs))
        mm_data = multimap.load_locus_mm_data(db_path, loc_id, em_iteration)

    # --- Assign reads to equivalence groups ---
    t1 = time.time()
    loc.assign_reads_to_egs(runs, mm_data)
    t2 = time.time()
    performance_measurements["proc_reads_2_time"] = t2 - t1
    performance_measurements["read_count"] = sum(
        rc for _, rc in loc.counted_reads.items()
    )

    # --- EM light M-step: one Huber reweight, emit λ + activities, stop ---
    if em_light:
        loc.deconvolve(
            config, runs=runs, max_outer=config.em_huber_steps, prune=False
        )
        lambdas = loc.compute_multimap_lambdas(runs)
        multimap.write_locus_em_output(
            db_path,
            loc_id,
            em_iteration,
            loc.activities_by_id(),
            lambdas,
        )
        return None if config.save_memory else (loc, performance_measurements)

    # --- Group-LASSO optimisation ---
    loc.deconvolve(config, runs=runs)
    performance_measurements["irls_outer_iterations"] = getattr(
        loc, "irls_outer_iterations", 0
    )

    loc.rgr_filter_sets["deconvoluted"] = loc.rgr_set
    performance_measurements["filtered_deconvoluted_rgr_count"] = len(loc.rgr_set)
    t2 = time.time()
    performance_measurements["optimization_time"] = t2 - t1

    if config.export_all_steps and not em_light:
        if config.export_gtf:
            loc.to_gtf(
                f"{config.base_o_dir}/deconvoluted",
                write_orfs=config.export_orfs,
                write_loci=config.export_loci,
                write_transcripts=config.export_transcripts,
            )
        if config.export_tsv and config.export_orfs:
            loc.to_tsv(f"{config.base_o_dir}/deconvoluted")
        if config.export_bed and config.export_orfs:
            loc.to_bed(f"{config.base_o_dir}/deconvoluted")

    # --- Likelihood-ratio filter ---
    if config.likelihood_ratio_filter:
        t1 = time.time()
        loc.likelihood_ratio_filtering(config, runs)
        loc.rgr_filter_sets["likelihood_ratio_filtered"] = loc.rgr_set
        t2 = time.time()
        performance_measurements["likelihood_ratio_time"] = t2 - t1
        performance_measurements["filtered_lrt_rgr_count"] = len(loc.rgr_set)
        performance_measurements["orf_count"] = sum(
            1 for rgr in loc.rgr_set if rgr.type == "ORF"
        )

    # --- Estimate activities ---
    t1 = time.time()
    loc.estimate_activities(runs, config)
    t2 = time.time()
    performance_measurements["activity_time"] = t2 - t1

    # --- Collect final statistics ---
    performance_measurements["gene_number"] = len(loc.gene_ids_complete)
    performance_measurements["transcripts_number"] = loc.transcripts_number
    t_end = time.time()
    performance_measurements["overall_time"] = t_end - t_start
    performance_measurements["exon_length"] = loc.exon_length

    if config.export_tsv:
        if config.export_orfs:
            loc.to_tsv(f"{config.base_o_dir}/", runs=runs)
        if config.export_regions and not loc.result_df.empty:
            loc.to_tsv(f"{config.base_o_dir}/", runs=runs, include_noise=True)

    if config.export_gtf:
        loc.to_gtf(
            f"{config.base_o_dir}/",
            write_orfs=config.export_orfs,
            write_loci=config.export_loci,
            write_transcripts=config.export_transcripts,
        )

    if config.export_bed:
        if config.export_orfs:
            loc.to_bed(f"{config.base_o_dir}/")
        if config.export_regions and not loc.result_df.empty:
            loc.to_bed(f"{config.base_o_dir}/", include_noise=True)

    if config.export_performance_measurements:
        perf_path = f"{config.o_dir}/performance_measurements.tsv"
        header = not os.path.exists(perf_path)
        lock = FileLock(perf_path + ".lock")
        with lock:
            with open(perf_path, "a") as f:
                f.write(
                    pd.DataFrame([performance_measurements]).to_csv(
                        header=header,
                        index=False,
                        float_format="{:.2e}".format,
                        sep="\t",
                    )
                )

    processed_loci_path = os.path.join(config.w_dir, "processed_loci.txt")
    lock = FileLock(processed_loci_path + ".lock")
    with lock:
        with open(processed_loci_path, "a") as f:
            f.write(loc_id + "\n")

    if config.save_memory:
        return
    else:
        return (loc, performance_measurements)
