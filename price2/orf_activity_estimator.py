import os
import sqlite3
import numpy as np
import pandas as pd

from pickle import loads, dumps
import zlib

from multiprocessing import Pool

from price2.locus import Locus
from price2.ribo_seq_alignment import RiboSeqAlignment

from tqdm import notebook

import time


class ORFActivityEstimator:

    def __init__(
        self,
        reads_db_path: str,
        loci_db_path: str,
        runs_db_path: str,
    ):
        self.reads_db_path = reads_db_path
        self.loci_db_path = loci_db_path
        self.runs_db_path = runs_db_path

        loci_db = sqlite3.connect(loci_db_path)
        cur = loci_db.cursor()

        cur.execute("SELECT locus_id FROM loci")
        self.loci_ids = [id for id, in cur.fetchall()]

    def run_orf_deconvolution(
        self,
        tolerance: float = 1e-7,  # 1e-6
        processes: int = 32,
        num_loci: int = 100,
        coverage_filter: bool = True,
        deconvolution_filter: bool = True,
        bg=None,  # for testing
        regularizaion: bool = True,
        lower_λ=3,
        upper_λ=13,
    ):
        self.loci = {}
        loci_ids = self.loci_ids[:num_loci]

        performance_measurements = []

        if regularizaion:
            f = process_loc
        else:
            f = process_loc_unregularized

        with Pool(processes) as pool:
            map_iterator = pool.imap_unordered(
                f,
                [
                    (
                        locus_id,
                        self.reads_db_path,
                        self.loci_db_path,
                        self.runs_db_path,
                        tolerance,
                        coverage_filter,
                        deconvolution_filter,
                        # bg.activity_matrices[locus_id].iloc[:,0].to_dict(), # for testing,
                        lower_λ,
                        upper_λ,
                    )
                    for locus_id in loci_ids
                ],
            )

            for result in notebook.tqdm(map_iterator, total=len(loci_ids)):
                loc = result[0]
                self.loci[loc.id] = loc
                performance_measurements.append(result[1])

        self.performance_df = pd.DataFrame(
            performance_measurements,
            columns=[
                "loc_id",
                "overall_time",
                "db_time",
                "load_reads_time",
                "proc_reads_time",
                "filter_time",
                "eg_time",
                "proc_reads_2_time",
                "prep_data_time",
                "opt_time",
                "unfiltered_rgr_count",
                "filtered_1_rgr_count",
                "filtered_2_rgr_count",
            ],
        )


def process_loc(arguments):

    t_start = time.time()  # performance measurement

    t1 = time.time()  # performance measurement

    (
        loc_id,
        reads_db_path,
        loci_db_path,
        runs_db_path,
        tolerance,
        coverage_filter,
        deconvolution_filter,
        # true_activities,
        lower_λ,
        upper_λ,
    ) = arguments

    ###############################
    ### get data from databases ###
    ###############################

    # load locus
    loci_db = sqlite3.connect(loci_db_path)
    cur = loci_db.cursor()
    cur.execute("""SELECT * FROM loci WHERE locus_id = ?""", (loc_id,))
    loc = loads(cur.fetchone()[1])
    loci_db.close()

    # load runs
    run_db = sqlite3.connect(runs_db_path)
    cur = run_db.cursor()
    cur.execute("SELECT * FROM runs")
    runs = [loads(blob) for id, blob in cur.fetchall()]
    run_db.close()

    t2 = time.time()  # performance measurement
    open_databases_t = t2 - t1  # performance measurement

    ##########################
    ### load reads to list ###
    ##########################

    t1 = time.time()  # performance measurement

    loc.get_reads_from_db(reads_db_path)

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

    if coverage_filter:
        loc.coverage_filter_rgrs()
    else:
        for c, rgr in enumerate(loc.rgr_set):
            rgr.index = c

    filtered_1_rgr_count = len(loc.rgr_set)  # performance measurement
    t2 = time.time()  # performance measurement
    process_reads_1_t = t2 - t1  # performance measurement

    ################################################
    ### filter ORF candidates ending in one stop ###
    ################################################

    t1 = time.time()  # performance measurement
    if deconvolution_filter:
        loc.deconvolution_filter_rgrs()
    else:
        for c, rgr in enumerate(loc.rgr_set):
            rgr.index = c

    filtered_2_rgr_count = len(loc.rgr_set)  # performance measurement
    t2 = time.time()  # performance measurement
    filter_2_t = t2 - t1  # performance measurement

    if True:  # write filtered ORFs to database for gtf
        keep_ORFs_ids = [rgr.id for rgr in loc.rgr_set if rgr.type == "ORF"]
        loc.keep_ORFs_ids = keep_ORFs_ids
        loci_db = sqlite3.connect(loci_db_path)
        cur = loci_db.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS loci_orfs (
                    locus_id text,
                    orfs blob
                    )"""
        )

        cur.execute(
            """INSERT INTO loci_orfs VALUES (?, ?)""", (loc.id, dumps(keep_ORFs_ids))
        )

        loci_db.commit()
        loci_db.close()

    ###################################
    ### generate equivalence groups ###
    ###################################

    # t1 = time.time() # performance measurement
    # loc.make_compatibility_groups()
    # t2 = time.time() # performance measurement
    # generate_compatibility_t = t2 - t1 # performance measurement

    for tr in loc.transcripts:
        # tr.orf_set = tr.orf_set & loc.rgr_set
        tr.update_with_filtered_orfs(loc.rgr_set)

    t1 = time.time()  # performance measurement
    # loc.make_equivalence_groups(runs)
    loc.make_equivalence_groups_precise(runs)
    t2 = time.time()  # performance measurement
    generate_equivalence_t = t2 - t1  # performance measurement
    print(loc_id, "generate_equivalence_groups", generate_equivalence_t)

    ###############
    ### testing ###
    ###############
    # assign read counts to equivalence groups
    # rgr_dict = {rgr.id: rgr for rgr in loc.rgr_set}
    # true_activities
    # active_rgrs = {rgr_dict[x].id for x in true_activities.keys()}
    # true_lengths = {rgr_id: len(rgr_dict[rgr_id]) for rgr_id in active_rgrs}
    ## rcf: read count fraction
    # rgr_2_rcf = {rgr_id: true_lengths[rgr_id] * true_activities[rgr_id] for rgr_id in active_rgrs}

    ###################
    ### end testing ###
    ###################

    ##########################################
    ### assign reads to equivalence groups ###
    ##########################################

    t1 = time.time()  # performance measurement
    loc.assign_reads_to_egs(runs)
    t2 = time.time()  # performance measurement
    process_reads_2_t = t2 - t1  # performance measurement
    print(f"{loc_id} assign_reads_to_egs: {process_reads_2_t}")
    loc.counted_reads = {}
    for run in runs:
        loc.counted_reads[run.id] = 0
        for v in loc.egs[run].values():
            loc.counted_reads[run.id] += v.read_count

    ##############################################################################
    ##### prepare data to be used efficiently by the numba objective function ####
    ##############################################################################

    t1 = time.time()  # performance measurement
    run_read_counts, cm_lut, egs, num_rgrs, rgr_lengths, initial_guess = (
        loc.to_objective_args(runs)
    )
    t2 = time.time()  # performance measurement
    prepare_data_t = t2 - t1  # performance measurement

    ############################
    ### run the optimization ###
    ############################

    t1 = time.time()  # performance measurement

    loc.deconvolve(
        run_read_counts,
        cm_lut,
        egs,
        num_rgrs,
        rgr_lengths,
        initial_guess,
        tolerance,
        lower_λ,
        upper_λ,
    )

    ta = np.zeros((len(loc.rgr_set), 2))

    rgr_dict = {rgr.id: rgr for rgr in loc.rgr_set}

    l = len(loc.rgr_set) / sum([len(rgr) for rgr in loc.rgr_set])

    t2 = time.time()  # performance measurement
    optimization_t = t2 - t1  # performance measurement

    ######################
    ### return results ###
    ######################
    loc.keep_rgrs = list(loc.rgr_set)
    t_end = time.time()  # performance measurement
    overall_t = t_end - t_start  # performance measurement

    performance_measurements = [
        loc_id,
        overall_t,
        open_databases_t,
        load_reads_t,
        process_reads_1_t,
        filter_2_t,
        generate_equivalence_t,
        process_reads_2_t,
        prepare_data_t,
        optimization_t,
        unfiltered_rgr_count,
        filtered_1_rgr_count,
        filtered_2_rgr_count,
    ]

    temp = loc.optimization_results[1e5].x
    results = temp.reshape((len(temp) // len(runs), len(runs)))
    temp = {}
    print(len(loc.rgr_set))
    for rgr in loc.rgr_set:
        temp[rgr.index] = rgr
    rgr_ids = [temp[i].id for i in range(len(temp))]
    run_ids = [run.id for run in runs]
    loc.result_df = pd.DataFrame(results, index=rgr_ids, columns=run_ids)
    loc.result_df["mean"] = loc.result_df.mean(axis=1)
    loc.result_df.sort_values(by="mean", inplace=True, ascending=False)
    loc.result_df.drop("mean", axis=1, inplace=True)

    # del loc.cgs
    # del loc.egs
    # del loc.rsas_dict

    return (loc, performance_measurements)


def process_loc_unregularized(arguments):

    t_start = time.time()  # performance measurement

    t1 = time.time()  # performance measurement

    (
        loc_id,
        reads_db_path,
        loci_db_path,
        runs_db_path,
        tolerance,
        coverage_filter,
        deconvolution_filter,
        # true_activities,
    ) = arguments

    ###############################
    ### get data from databases ###
    ###############################

    # load locus
    loci_db = sqlite3.connect(loci_db_path)
    cur = loci_db.cursor()
    cur.execute("""SELECT * FROM loci WHERE locus_id = ?""", (loc_id,))
    loc = loads(cur.fetchone()[1])
    loci_db.close()

    # load runs
    run_db = sqlite3.connect(runs_db_path)
    cur = run_db.cursor()
    cur.execute("SELECT * FROM runs")
    runs = [loads(blob) for id, blob in cur.fetchall()]
    run_db.close()

    t2 = time.time()  # performance measurement
    open_databases_t = t2 - t1  # performance measurement

    ##########################
    ### load reads to list ###
    ##########################

    t1 = time.time()  # performance measurement

    loc.get_reads_from_db(reads_db_path)

    t2 = time.time()  # performance measurement
    load_reads_t = t2 - t1  # performance measurement

    ###############################################################################
    ### assign reads to ORF candidates, filter ORF candidates based on coverage ###
    ###############################################################################

    t1 = time.time()  # performance measurement
    unfiltered_rgr_count = len(loc.rgr_set)  # performance measurement

    loc.make_well_fitting_reads(runs)

    if coverage_filter:
        loc.coverage_filter_rgrs()
    else:
        for c, rgr in enumerate(loc.rgr_set):
            rgr.index = c

    filtered_1_rgr_count = len(loc.rgr_set)  # performance measurement
    t2 = time.time()  # performance measurement
    process_reads_1_t = t2 - t1  # performance measurement

    ################################################
    ### filter ORF candidates ending in one stop ###
    ################################################

    t1 = time.time()  # performance measurement
    if deconvolution_filter:
        loc.deconvolution_filter_rgrs()
    else:
        for c, rgr in enumerate(loc.rgr_set):
            rgr.index = c

    filtered_2_rgr_count = len(loc.rgr_set)  # performance measurement
    t2 = time.time()  # performance measurement
    filter_2_t = t2 - t1  # performance measurement

    if True:  # write filtered ORFs to database for gtf
        keep_ORFs_ids = [rgr.id for rgr in loc.rgr_set if rgr.type == "ORF"]
        loc.keep_ORFs_ids = keep_ORFs_ids
        loci_db = sqlite3.connect(loci_db_path)
        cur = loci_db.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS loci_orfs (
                    locus_id text,
                    orfs blob
                    )"""
        )

        cur.execute(
            """INSERT INTO loci_orfs VALUES (?, ?)""", (loc.id, dumps(keep_ORFs_ids))
        )

        loci_db.commit()
        loci_db.close()

    #####################################################
    ### generate compatibility and equivalence groups ###
    #####################################################

    t1 = time.time()  # performance measurement
    loc.make_compatibility_groups_old()
    t2 = time.time()  # performance measurement
    generate_compatibility_t = t2 - t1  # performance measurement

    t1 = time.time()  # performance measurement
    loc.make_equivalence_groups_old(runs)
    t2 = time.time()  # performance measurement
    generate_equivalence_t = t2 - t1  # performance measurement

    ###############
    ### testing ###
    ###############
    # assign read counts to equivalence groups
    # rgr_dict = {rgr.id: rgr for rgr in loc.rgr_set}
    # true_activities
    # active_rgrs = {rgr_dict[x].id for x in true_activities.keys()}
    # true_lengths = {rgr_id: len(rgr_dict[rgr_id]) for rgr_id in active_rgrs}
    ## rcf: read count fraction
    # rgr_2_rcf = {rgr_id: true_lengths[rgr_id] * true_activities[rgr_id] for rgr_id in active_rgrs}

    ###################
    ### end testing ###
    ###################

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
    run_read_counts, cm_lut, egs, num_rgrs, rgr_lengths, initial_guess = (
        loc.to_objective_args(runs)
    )
    t2 = time.time()  # performance measurement
    prepare_data_t = t2 - t1  # performance measurement

    ############################
    ### run the optimization ###
    ############################

    t1 = time.time()  # performance measurement

    ta = np.zeros((len(loc.rgr_set), 2))

    rgr_dict = {rgr.id: rgr for rgr in loc.rgr_set}

    l = len(loc.rgr_set) / sum([len(rgr) for rgr in loc.rgr_set])

    # for k, v in true_activities.items():
    #    ta[rgr_dict[k].index] = true_activities[k] * l

    # initial_guess = ta.flatten()

    loc.deconvolve_unregularized(
        run_read_counts,
        cm_lut,
        egs,
        num_rgrs,
        rgr_lengths,
        initial_guess,
        tolerance,
    )

    t2 = time.time()  # performance measurement
    optimization_t = t2 - t1  # performance measurement

    ######################
    ### return results ###
    ######################
    loc.keep_rgrs = list(loc.rgr_set)
    t_end = time.time()  # performance measurement
    overall_t = t_end - t_start  # performance measurement

    performance_measurements = [
        loc_id,
        overall_t,
        open_databases_t,
        load_reads_t,
        process_reads_1_t,
        filter_2_t,
        generate_compatibility_t,
        generate_equivalence_t,
        process_reads_2_t,
        prepare_data_t,
        optimization_t,
        unfiltered_rgr_count,
        filtered_1_rgr_count,
        filtered_2_rgr_count,
    ]

    temp = loc.optimization_results[0].x
    results = temp.reshape((len(temp) // len(runs), len(runs)))
    temp = {}
    for rgr in loc.rgr_set:
        temp[rgr.index] = rgr
    rgr_ids = [temp[i].id for i in range(len(temp))]
    run_ids = [run.id for run in runs]
    loc.result_df = pd.DataFrame(results, index=rgr_ids, columns=run_ids)
    loc.result_df["mean"] = loc.result_df.mean(axis=1)
    loc.result_df.sort_values(by="mean", inplace=True, ascending=False)
    loc.result_df.drop("mean", axis=1, inplace=True)

    # del loc.cgs
    # del loc.egs
    # del loc.rsas_dict

    return (loc, performance_measurements)


def compute_likelihood(arguments):

    (
        loc_id,
        reads_db_path,
        loci_db_path,
        runs_db_path,
        estimated_activities,
        λ,
    ) = arguments

    ###############################
    ### get data from databases ###
    ###############################

    # load locus
    loci_db = sqlite3.connect(loci_db_path)
    cur = loci_db.cursor()
    cur.execute("""SELECT * FROM loci WHERE locus_id = ?""", (loc_id,))
    loc = loads(cur.fetchone()[1])
    loci_db.close()

    # load runs
    run_db = sqlite3.connect(runs_db_path)
    cur = run_db.cursor()
    cur.execute("SELECT * FROM runs")
    runs = [loads(blob) for id, blob in cur.fetchall()]
    run_db.close()

    ##########################
    ### load reads to list ###
    ##########################

    loc.get_reads_from_db(reads_db_path)

    ###############################################################################
    ### assign reads to ORF candidates, filter ORF candidates based on coverage ###
    ###############################################################################

    loc.make_well_fitting_reads(runs)

    for c, rgr in enumerate(loc.rgr_set):
        rgr.index = c

    #####################################################
    ### generate compatibility and equivalence groups ###
    #####################################################

    loc.make_compatibility_groups()
    loc.make_equivalence_groups(runs)

    ##########################################
    ### assign reads to equivalence groups ###
    ##########################################

    loc.assign_reads_to_egs(runs)

    ##############################################################################
    ##### prepare data to be used efficiently by the numba objective function ####
    ##############################################################################

    run_read_counts, cm_lut, egs, num_rgrs, rgr_lengths, _ = loc.to_objective_args(runs)

    ############################
    ### run the optimization ###
    ############################

    from .locus import objective_function_likelihood, objective_function_penalty

    ll = objective_function_likelihood(
        estimated_activities,
        run_read_counts,
        cm_lut,
        egs,
        num_rgrs,
        rgr_lengths,
        λ,
    )

    pe = objective_function_penalty(
        estimated_activities,
        run_read_counts,
        cm_lut,
        egs,
        num_rgrs,
        rgr_lengths,
        λ,
    )

    ######################
    ### return results ###
    ######################

    del loc.cgs
    del loc.egs
    del loc.rsas_dict

    return ll, pe
