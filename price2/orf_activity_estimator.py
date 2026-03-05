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
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from price2.equivalence_groups import make_equivalence_groups

# Must be set before any ProcessPool is created.  ``forkserver`` is
# required because numba JIT state and SQLite handles are not safe to
# fork directly.
mp.set_start_method("forkserver", force=True)

logger = logging.getLogger(__name__)


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
        if config.loci_subset > 0:
            self.loci_ids = self.loci_ids[: config.loci_subset]

    def run_orf_deconvolution(self) -> None:
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
        """
        self.loci = {}
        loci_ids = set(self.loci_ids)

        performance_measurements = {}

        db = sql.connect(self.db_path)
        cur = db.cursor()
        cur.execute("SELECT run_id FROM runs")
        run_ids = [id for id, in cur.fetchall()]
        db.close()

        ra_dir = os.path.join(self.config.o_dir, "regions_activities")
        os.makedirs(ra_dir, exist_ok=True)

        if os.path.exists(f"{self.config.o_dir}/performance_measurements.tsv"):
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
                        args=[(locus_id, self.config, log_queue)],
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
    loc_id, config, log_queue = arguments

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

    # --- Load locus and runs from database ---
    db_path = f"{config.w_dir}/price.db"
    db = sql.connect(db_path)
    cur = db.cursor()
    cur.execute("SELECT * FROM loci WHERE locus_id = ?", (loc_id,))
    loc = loads(cur.fetchone()[1])
    performance_measurements["chrom"] = loc.iv.chrom
    performance_measurements["strand"] = loc.iv.strand
    performance_measurements["start"] = loc.iv.start
    performance_measurements["end"] = loc.iv.end

    cur.execute("SELECT * FROM runs")
    runs = [loads(blob) for id, blob in cur.fetchall()]
    db.close()

    t2 = time.time()
    performance_measurements["db_time"] = t2 - t1

    if config.verbose_gtf:
        loc.to_gtf(f"{config.base_o_dir}/all")
        loc.to_tsv(f"{config.base_o_dir}/all")
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

    if config.verbose_gtf:
        loc.to_gtf(f"{config.base_o_dir}/coverage_filtered")
        loc.to_tsv(f"{config.base_o_dir}/coverage_filtered")
    loc.rgr_filter_sets["coverage_filtered"] = loc.rgr_set

    performance_measurements["filtered_coverage_rgr_count"] = len(loc.rgr_set)
    t2 = time.time()
    performance_measurements["coverage_filter_time"] = t2 - t1

    # --- Deconvolution filter ---
    t1 = time.time()
    if config.deconvolution_filter:
        loc.deconvolution_filter_rgrs(config)

    loc.rgr_filter_sets["deconvolution_filtered"] = loc.rgr_set
    performance_measurements["filtered_deconvolution_rgr_count"] = len(loc.rgr_set)
    t2 = time.time()
    performance_measurements["filter_2_time"] = t2 - t1

    if config.verbose_gtf:
        loc.to_gtf(f"{config.base_o_dir}/deconvolution_filtered")
        loc.to_tsv(f"{config.base_o_dir}/deconvolution_filtered")

    # --- Equivalence groups ---
    for tr in loc.transcripts:
        tr.update_with_filtered_orfs(loc.rgr_set)

    t1 = time.time()
    loc.egs = make_equivalence_groups(loc, runs)
    t2 = time.time()

    performance_measurements["eg_time"] = t2 - t1

    # --- Assign reads to equivalence groups ---
    t1 = time.time()
    loc.assign_reads_to_egs(runs)
    t2 = time.time()
    performance_measurements["proc_reads_2_time"] = t2 - t1
    performance_measurements["read_count"] = sum(
        rc for _, rc in loc.counted_reads.items()
    )

    # --- Group-LASSO optimisation ---
    loc.deconvolve(config, runs=runs)

    loc.rgr_filter_sets["deconvoluted"] = loc.rgr_set
    performance_measurements["filtered_deconvoluted_rgr_count"] = len(loc.rgr_set)
    t2 = time.time()
    performance_measurements["optimization_time"] = t2 - t1

    if config.verbose_gtf:
        loc.to_gtf(f"{config.base_o_dir}/deconvoluted")
        loc.to_tsv(f"{config.base_o_dir}/deconvoluted")

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
    performance_measurements["gene_number"] = len(
        {rgr.transcript.gene_id for rgr in loc.rgr_set_complete}
    )
    performance_measurements["transcripts_number"] = loc.transcripts_number
    t_end = time.time()
    performance_measurements["overall_time"] = t_end - t_start
    performance_measurements["exon_length"] = loc.exon_length

    if not loc.result_df.empty:
        loc.to_tsv(
            f"{config.base_o_dir}/final",
            runs=runs,
            include_noise=True,
        )
    loc.to_gtf(
        f"{config.base_o_dir}/final",
        write_orfs=True,
        write_loci=True,
        write_transcripts=True,
    )

    loc.to_tsv(f"{config.base_o_dir}/final", runs=runs)

    if not os.path.exists(f"{config.o_dir}/performance_measurements.tsv"):
        header = True
    else:
        header = False
    lock = FileLock(f"{config.o_dir}/performance_measurements.tsv.lock")
    with lock:
        with open(f"{config.o_dir}/performance_measurements.tsv", "a") as f:
            f.write(
                pd.DataFrame([performance_measurements]).to_csv(
                    header=header,
                    index=False,
                    float_format="{:.2e}".format,
                    sep="\t",
                )
            )

    if config.save_memory:
        return
    else:
        return (loc, performance_measurements)
