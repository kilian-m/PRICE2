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

from multiprocessing import Pool

from price2.locus import Locus
from price2.ribo_seq_alignment import RiboSeqAlignment

from tqdm import tqdm

import time


class ORFActivityEstimator:

    def __init__(self, config):
        self.config = config
        self.db_path = f"{config['w_dir']}/price.db"

        db = sql.connect(self.db_path)
        cur = db.cursor()

        cur.execute("SELECT locus_id FROM loci")
        self.loci_ids = [id for id, in cur.fetchall()]
        if config["loci_subset"] > 0:
            self.loci_ids = self.loci_ids[: config["loci_subset"]]

    def run_orf_deconvolution(self):

        self.loci = {}
        loci_ids = set(self.loci_ids)

        performance_measurements = []

        db = sql.connect(self.db_path)
        cur = db.cursor()
        cur.execute("SELECT run_id FROM runs")
        run_ids = [id for id, in cur.fetchall()]
        db.close()

        if not os.path.exists(f"{self.config['o_dir']}/results.tsv"):
            lock = FileLock(f"{self.config['o_dir']}/results.tsv.lock")
            with lock:
                with open(f"{self.config['o_dir']}/results.tsv", "a") as f:
                    f.write("orf_id\t")
                    f.write("\t".join(run_ids))
                    f.write("\n")
            lock = FileLock(f"{self.config['o_dir']}/rpkm.tsv.lock")
            with lock:
                with open(f"{self.config['o_dir']}/rpkm.tsv", "a") as f:
                    f.write("orf_id\t")
                    f.write("\t".join(run_ids))
                    f.write("\n")
        if not os.path.exists(f"{self.config['o_dir']}/performance_measurements.tsv"):
            lock = FileLock(f"{self.config['o_dir']}/performance_measurements.tsv.lock")
            with lock:
                with open(
                    f"{self.config['o_dir']}/performance_measurements.tsv", "a"
                ) as f:
                    f.write(
                        "\t".join(
                            [
                                "loc_id",
                                "exon_length",
                                "read_count",
                                "overall_time",
                                "db_time",
                                "load_reads_time",
                                "proc_reads_time",
                                "filter_time",
                                "eg_time",
                                "proc_reads_2_time",
                                "prep_data_time",
                                "opt_time",
                                "likelihood_ratio_time",
                                "unfiltered_rgr_count",
                                "filtered_coverage_rgr_count",
                                "filtered_optimization_rgr_count",
                                "filtered_lrt_rgr_count",
                                "λ",
                                "num_transcripts",
                                "gene_number",
                                "transcripts_number",
                                "initial_iterations",
                                "all_iterations",
                            ]
                        )
                    )
                    f.write("\n")
        else:

            processed_loc_ids = set(
                pd.read_csv(
                    f"{self.config['o_dir']}/performance_measurements.tsv",
                    sep="\t",
                )["loc_id"]
            )
            loci_ids = loci_ids - processed_loc_ids

        with Pool(self.config["processes"]) as pool:
            map_iterator = pool.imap_unordered(
                process_loc,
                [
                    (
                        locus_id,
                        self.config,
                    )
                    for locus_id in loci_ids
                ],
            )

            for result in tqdm(map_iterator, total=len(loci_ids)):
                loc = result[0]
                self.loci[loc.id] = loc
                performance_measurements.append(result[1])

        self.performance_df = pd.DataFrame(
            performance_measurements,
            columns=[
                "loc_id",
                "exon_length",
                "read_count",
                "overall_time",
                "db_time",
                "load_reads_time",
                "proc_reads_time",
                "filter_time",
                "eg_time",
                "proc_reads_2_time",
                "prep_data_time",
                "opt_time",
                "likelihood_ratio_time",
                "unfiltered_rgr_count",
                "filtered_coverage_rgr_count",
                "filtered_optimization_rgr_count",
                "filtered_lrt_rgr_count",
                "λ",
                "num_transcripts",
                "gene_number",
                "transcripts_number",
                "initial_iterations",
                "all_iterations",
            ],
        )

        lock_files = glob.glob(os.path.join(self.config["o_dir"], "*.lock"))
        for lock_file in lock_files:
            os.remove(lock_file)


def process_loc(arguments):

    t_start = time.time()  # performance measurement

    t1 = time.time()  # performance measurement

    (
        loc_id,
        config,
    ) = arguments

    nb.set_num_threads(1)

    lower_λ, upper_λ, number_λs = config["λs"]

    if not "base_o_dir" in config:
        config["base_o_dir"] = f"{config['o_dir']}"

    ###############################
    ### get data from databases ###
    ###############################
    db_path = f"{config['w_dir']}/price.db"
    db = sql.connect(db_path)
    cur = db.cursor()
    cur.execute("""SELECT * FROM loci WHERE locus_id = ?""", (loc_id,))
    loc = loads(cur.fetchone()[1])

    cur.execute("SELECT * FROM runs")
    runs = [loads(blob) for id, blob in cur.fetchall()]
    db.close()

    t2 = time.time()  # performance measurement
    open_databases_t = t2 - t1  # performance measurement

    if config["verbose_gtf"]:
        loc.to_gtf(f"{config['base_o_dir']}/all.gtf")

    ##########################
    ### load reads to list ###
    ##########################

    t1 = time.time()  # performance measurement

    loc.get_reads_from_db(db_path)

    t2 = time.time()  # performance measurement
    load_reads_t = t2 - t1  # performance measurement

    ###############################################################################
    ### assign reads to ORF candidates, filter ORF candidates based on coverage ###
    ###############################################################################

    t1 = time.time()  # performance measurement
    unfiltered_rgr_count = len(loc.rgr_set)  # performance measurement

    for tr in loc.transcripts:
        tr.update_with_filtered_orfs(loc.rgr_set)

    loc.make_well_fitting_reads(runs)

    if config["coverage_filter"]:
        loc.coverage_filter_rgrs(config["min_well_fitting_reads_per_length"])
    else:
        for c, rgr in enumerate(loc.rgr_set):
            rgr.index = c

    filtered_1_rgr_count = len(loc.rgr_set)  # performance measurement
    t2 = time.time()  # performance measurement
    process_reads_1_t = t2 - t1  # performance measurement

    if config["verbose_gtf"]:
        loc.to_gtf(f"{config['base_o_dir']}/coverage_filtered")

    ################################################
    ### filter ORF candidates ending in one stop ###
    ################################################

    t1 = time.time()  # performance measurement
    if config["deconvolution_filter"]:
        loc.deconvolution_filter_rgrs(
            config["deconvolution_filter_min_activity_fraction"]
        )
    else:
        for c, rgr in enumerate(loc.rgr_set):
            rgr.index = c

    filtered_2_rgr_count = len(loc.rgr_set)  # performance measurement
    t2 = time.time()  # performance measurement
    filter_2_t = t2 - t1  # performance measurement

    if config["verbose_gtf"]:
        loc.to_gtf(f"{config['base_o_dir']}/deconvolution_filtered")

    ###################################
    ### generate equivalence groups ###
    ###################################

    for tr in loc.transcripts:
        tr.update_with_filtered_orfs(loc.rgr_set)

    t1 = time.time()  # performance measurement
    loc.make_equivalence_groups_precise(runs)
    t2 = time.time()  # performance measurement
    generate_equivalence_t = t2 - t1  # performance measurement

    ##########################################
    ### assign reads to equivalence groups ###
    ##########################################

    t1 = time.time()  # performance measurement
    loc.assign_reads_to_egs(runs)
    t2 = time.time()  # performance measurement
    process_reads_2_t = t2 - t1  # performance measurement

    loc.counted_reads = {}
    for run in runs:
        loc.counted_reads[run.id] = 0
        for v in loc.egs[run].values():
            loc.counted_reads[run.id] += v.read_count

    ##############################################################################
    ##### prepare data to be used efficiently by the numba objective function ####
    ##############################################################################

    t1 = time.time()  # performance measurement
    for c, rgr in enumerate(loc.rgr_set):
        rgr.index = c
    objective_args = loc.to_objective_args(runs)

    t2 = time.time()  # performance measurement
    prepare_data_t = t2 - t1  # performance measurement

    ############################
    ### run the optimization ###
    ############################

    t1 = time.time()  # performance measurement

    loc.deconvolve(
        *objective_args,
        config["ftol"],
        config["gtol"],
        config["callback"],
        config["callback_args"],
        config["pseudo_min"],
        lower_λ=config["λs"][0],
        upper_λ=config["λs"][1],
        number_λs=config["λs"][2],
    )

    t2 = time.time()  # performance measurement
    optimization_t = t2 - t1  # performance measurement

    ####################################
    ### filter with likelihood ratio ###
    ####################################

    loc.collapse_egs(runs=runs, min_activity_threshold=config["min_activity_threshold"])
    objective_args = loc.to_objective_args(runs)

    if config["verbose_gtf"]:
        loc.to_gtf(f"{config['base_o_dir']}/deconvoluted")

    t1 = time.time()  # performance measurement
    try:
        loc.likelihood_ratio_filtering(
            *objective_args,
            α=config["α"],
            ftol=config["ftol"],
            gtol=config["gtol"],
        )
    except ValueError:
        print(f"ValueError in likelihood ratio filtering of {loc.id}")
    t2 = time.time()  # performance measurement
    likelihood_ratio_t = t2 - t1  # performance measurement

    ######################
    ### return results ###
    ######################
    loc.keep_rgrs = list(loc.rgr_set)

    filtered_lrt = len(loc.keep_rgr_inds)

    # compute number of genes and transcripts
    gene_number = len({rgr.transcript.gene_id for rgr in loc.rgr_set_complete})
    transcripts_number = loc.transcripts_number

    # number of iterations
    # initial
    initial_iterations = list(loc.optimization_results.values())[0].nit

    all_iterations = sum([r.nit for r in loc.optimization_results.values()])

    t_end = time.time()  # performance measurement
    overall_t = t_end - t_start  # performance measurement

    exon_length = loc.exon_length
    performance_measurements = [
        loc_id,
        exon_length,
        loc.read_count,
        overall_t,
        open_databases_t,
        load_reads_t,
        process_reads_1_t,
        filter_2_t,
        generate_equivalence_t,
        process_reads_2_t,
        prepare_data_t,
        optimization_t,
        likelihood_ratio_t,
        unfiltered_rgr_count,
        filtered_1_rgr_count,
        filtered_2_rgr_count,
        filtered_lrt,
        loc.best_λ,
        len(loc.noise_rgr_indices),
        gene_number,
        transcripts_number,
        initial_iterations,
        all_iterations,
    ]

    loc.compute_rpkm(runs)

    if config["o_dir"]:
        if not loc.result_df.empty:
            lock = FileLock(f"{config['base_o_dir']}/results.tsv.lock")
            with lock:
                with open(f"{config['base_o_dir']}/results.tsv", "a") as f:
                    f.write(
                        loc.result_df.to_csv(
                            header=False,
                            index=True,
                            float_format="{:.2e}".format,
                            sep="\t",
                        )
                    )
            lock = FileLock(f"{config['base_o_dir']}/rpkm.tsv.lock")
            with lock:
                with open(f"{config['base_o_dir']}/rpkm.tsv", "a") as f:
                    f.write(
                        loc.result_df.to_csv(
                            header=False,
                            index=True,
                            float_format="{:.2e}".format,
                            sep="\t",
                        )
                    )
        loc.to_gtf(
            f"{config['base_o_dir']}/final",
            write_orfs=True,
            write_loci=True,
            write_transcripts=True,
        )
        lock = FileLock(f"{config['base_o_dir']}/performance_measurements.tsv.lock")
        with lock:
            with open(f"{config['base_o_dir']}/performance_measurements.tsv", "a") as f:
                f.write(
                    pd.DataFrame([performance_measurements]).to_csv(
                        header=False,
                        index=False,
                        float_format="{:.2e}".format,
                        sep="\t",
                    )
                )

    del loc.egs
    del loc.rsas_dict

    return (loc, performance_measurements)
