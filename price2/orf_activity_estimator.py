import os
import shutil
import glob
import sqlite3 as sql
import numpy as np
import pandas as pd
import numba as nb

from pickle import loads, dumps
import zlib
from filelock import FileLock

# import multiprocessing
# from multiprocessing import Pool

from price2.locus import Locus
from price2.ribo_seq_alignment import RiboSeqAlignment
from price2.equivalence_groups import make_equivalence_groups

from tqdm import tqdm

import time

from pebble import ProcessPool
from concurrent.futures import TimeoutError


class ORFActivityEstimator:

    def __init__(self, config):
        self.config = config
        self.db_path = f"{config.w_dir}/price.db"

        db = sql.connect(self.db_path)
        cur = db.cursor()

        cur.execute("SELECT locus_id FROM loci")
        self.loci_ids = [id for id, in cur.fetchall()]
        if config.loci_subset > 0:
            self.loci_ids = self.loci_ids[: config.loci_subset]

    def run_orf_deconvolution(self):

        self.loci = {}
        loci_ids = set(self.loci_ids)

        performance_measurements = {}

        db = sql.connect(self.db_path)
        cur = db.cursor()
        cur.execute("SELECT run_id FROM runs")
        run_ids = [id for id, in cur.fetchall()]
        db.close()

        if not os.path.exists(f"{self.config.o_dir}/results.tsv"):
            lock = FileLock(f"{self.config.o_dir}/results.tsv.lock")
            with lock:
                with open(f"{self.config.o_dir}/results.tsv", "a") as f:
                    f.write("orf_id\t")
                    f.write("\t".join(run_ids))
                    f.write("\n")

        if os.path.exists(f"{self.config.o_dir}/performance_measurements.tsv"):
            processed_loc_ids = set(
                pd.read_csv(
                    f"{self.config.o_dir}/performance_measurements.tsv",
                    sep="\t",
                )["loc_id"]
            )
            loci_ids = loci_ids - processed_loc_ids

        pbar = tqdm(total=len(loci_ids))

        with ProcessPool(max_workers=self.config.processes) as pool:
            future = pool.map(
                process_loc,
                [
                    (
                        locus_id,
                        self.config,
                    )
                    for locus_id in loci_ids
                ],
                timeout=self.config.timeout,
            )

            iterator = future.result()

            results = []
            while True:
                try:
                    result = next(iterator)
                    results.append(result)
                    pbar.update(1)
                except TimeoutError as e:
                    results.append(e)
                    pbar.update(1)
                except StopIteration:
                    pbar.close()
                    break
                except Exception as e:
                    results.append(e)
                    pbar.update(1)

        self.loci = {}
        self.failed_loci = {}
        for loc_id, result in zip(loci_ids, results):
            if isinstance(result, Exception):
                self.failed_loci[loc_id] = result
            else:
                self.loci[loc_id] = result[0]
                performance_measurements[loc_id] = result[1]

        self.performance_df = pd.DataFrame(performance_measurements).T

        lock_files = glob.glob(os.path.join(self.config.o_dir, "*.lock"))
        for lock_file in lock_files:
            os.remove(lock_file)


def process_loc(arguments):
    nb.set_num_threads(1)

    (
        loc_id,
        config,
    ) = arguments

    performance_measurements = {}  # performance measurement
    performance_measurements["loc_id"] = loc_id  # performance measurement
    t_start = time.time()  # performance measurement
    t1 = time.time()  # performance measurement

    if not hasattr(config, "base_o_dir"):
        config.base_o_dir = config.o_dir

    ###############################
    ### get data from databases ###
    ###############################
    db_path = f"{config.w_dir}/price.db"
    db = sql.connect(db_path)
    cur = db.cursor()
    cur.execute("""SELECT * FROM loci WHERE locus_id = ?""", (loc_id,))
    loc = loads(cur.fetchone()[1])
    performance_measurements["chrom"] = loc.iv.chrom
    performance_measurements["strand"] = loc.iv.strand
    performance_measurements["start"] = loc.iv.start
    performance_measurements["end"] = loc.iv.end

    cur.execute("SELECT * FROM runs")
    runs = [loads(blob) for id, blob in cur.fetchall()]
    db.close()

    t2 = time.time()  # performance measurement
    performance_measurements["db_time"] = t2 - t1  # performance measurement

    if config.verbose_gtf:
        loc.to_gtf(f"{config.base_o_dir}/all.gtf")
    loc.rgr_filter_sets = {}

    ##########################
    ### load reads to list ###
    ##########################

    t1 = time.time()  # performance measurement

    loc.get_reads_from_db(db_path)

    t2 = time.time()  # performance measurement
    performance_measurements["load_reads_time"] = t2 - t1  # performance measurement

    ######################################
    ### assign reads to ORF candidates ###
    ######################################

    t1 = time.time()  # performance measurement
    unfiltered_rgr_count = len(loc.rgr_set)  # performance measurement
    performance_measurements["unfiltered_rgr_count"] = (
        unfiltered_rgr_count  # performance measurement
    )
    loc.rgr_filter_sets["unfiltered"] = loc.rgr_set

    if config.coverage_filter or config.deconvolution_filter:
        loc.make_well_fitting_reads(runs)

    t2 = time.time()  # performance measurement
    performance_measurements["assign_reads_time"] = t2 - t1  # performance measurement

    ###############################################
    ### filter ORF candidates based on coverage ###
    ###############################################

    t1 = time.time()  # performance measurement

    if config.coverage_filter:
        loc.coverage_filter_rgrs(config)

    if config.verbose_gtf:
        loc.to_gtf(f"{config.base_o_dir}/coverage_filtered")
    loc.rgr_filter_sets["coverage_filtered"] = loc.rgr_set

    performance_measurements["filtered_coverage_rgr_count"] = len(
        loc.rgr_set
    )  # performance measurement
    t2 = time.time()  # performance measurement
    performance_measurements["coverage_filter_time"] = (
        t2 - t1
    )  # performance measurement

    ##############################################################
    ### deconvolution filter ORF candidates ending in one stop ###
    ##############################################################

    t1 = time.time()  # performance measurement
    if config.deconvolution_filter:
        loc.deconvolution_filter_rgrs(config)

    loc.rgr_filter_sets["deconvolution_filtered"] = loc.rgr_set
    performance_measurements["filtered_deconvolution_rgr_count"] = len(
        loc.rgr_set
    )  # performance measurement
    t2 = time.time()  # performance measurement
    performance_measurements["filter_2_time"] = t2 - t1  # performance measurement

    if config.verbose_gtf:
        loc.to_gtf(f"{config.base_o_dir}/deconvolution_filtered")

    ###################################
    ### generate equivalence groups ###
    ###################################

    for tr in loc.transcripts:
        tr.update_with_filtered_orfs(loc.rgr_set)

    t1 = time.time()  # performance measurement
    loc.egs = make_equivalence_groups(loc, runs)
    t2 = time.time()  # performance measurement
    performance_measurements["eg_time"] = t2 - t1  # performance measurement

    ##########################################
    ### assign reads to equivalence groups ###
    ##########################################

    t1 = time.time()  # performance measurement
    loc.assign_reads_to_egs(runs)
    t2 = time.time()  # performance measurement
    performance_measurements["proc_reads_2_time"] = t2 - t1  # performance measurement

    ############################
    ### run the optimization ###
    ############################
    loc.deconvolve(
        config,
        runs=runs,
    )

    # performance_measurements["opt_time"] = opt_time  # performance measurement
    # performance_measurements["data_time"] = data_time  # performance measurement

    loc.rgr_filter_sets["deconvoluted"] = loc.rgr_set
    performance_measurements["filtered_deconvoluted_rgr_count"] = len(loc.rgr_set)
    # performance_measurements["iterations"] = loc.optimization_result.nit
    t2 = time.time()  # performance measurement
    performance_measurements["optimization_time"] = t2 - t1  # performance measurement

    ####################################
    ### filter with likelihood ratio ###
    ####################################

    if config.verbose_gtf:
        loc.to_gtf(f"{config.base_o_dir}/deconvoluted")

    if config.likelihood_ratio_filter:
        t1 = time.time()  # performance measurement

        loc.likelihood_ratio_filtering(
            config,
            runs,
        )

        loc.rgr_filter_sets["likelihood_ratio_filtered"] = loc.rgr_set
        t2 = time.time()  # performance measurement
        performance_measurements["likelihood_ratio_time"] = (
            t2 - t1
        )  # performance measurement

        performance_measurements["filtered_lrt_rgr_count"] = len(loc.rgr_set)
        performance_measurements["orf_count"] = len(
            [1 for rgr in loc.rgr_set if rgr.type == "ORF"]
        )

    ##########################
    ### estimate activites ###
    ##########################

    t1 = time.time()  # performance measurement
    loc.estimate_activities(runs, config)

    t2 = time.time()  # performance measurement
    performance_measurements["activity_time"] = t2 - t1  # performance measurement

    ######################
    ### return results ###
    ######################

    # compute number of genes and transcripts
    performance_measurements["gene_number"] = len(
        {rgr.transcript.gene_id for rgr in loc.rgr_set_complete}
    )  # performance measurement

    performance_measurements["transcripts_number"] = (
        loc.transcripts_number
    )  # performance measurement

    del loc.egs
    del loc.rsas_dict

    t_end = time.time()  # performance measurement
    performance_measurements["overall_time"] = (
        t_end - t_start
    )  # performance measurement

    performance_measurements["exon_length"] = loc.exon_length  # performance measurement

    if config.o_dir:
        if not loc.result_df.empty:
            lock = FileLock(f"{config.base_o_dir}/results.tsv.lock")
            with lock:
                with open(f"{config.base_o_dir}/results.tsv", "a") as f:
                    f.write(
                        loc.result_df.to_csv(
                            header=False,
                            index=True,
                            float_format="{:.2e}".format,
                            sep="\t",
                        )
                    )
        loc.to_gtf(
            f"{config.base_o_dir}/final",
            write_orfs=True,
            write_loci=True,
            write_transcripts=True,
        )
        if not os.path.exists(f"{config.base_o_dir}/performance_measurements.tsv"):
            header = True
        else:
            header = False
        lock = FileLock(f"{config.base_o_dir}/performance_measurements.tsv.lock")
        with lock:
            with open(f"{config.base_o_dir}/performance_measurements.tsv", "a") as f:
                f.write(
                    pd.DataFrame([performance_measurements]).to_csv(
                        header=header,
                        index=False,
                        float_format="{:.2e}".format,
                        sep="\t",
                    )
                )

    return (loc, performance_measurements)
