# imports
import argparse
import json
import os
import shutil
import time

import HTSeq
import numba as nb

nb.set_num_threads(1)

from reference_annotation import ReferenceAnnotation
from ribo_seq_run import ribo_seq_runs_from_bams
from data_collector import DataCollector
from orf_activity_estimator import ORFActivityEstimator

if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser(description="Estimate ORF activities.")
    parser.add_argument("--config", type=str, help="Path to JSON config file.")

    args = parser.parse_args()

    config = {
        "processes": 80,
        "warm_start": True,
        "callback": True,
        "callback_args": (0.001, 0.001),
        "min_explained_reads_per_run": 5,
        "min_well_fitting_reads_per_length": 0.1,
        "deconvolution_filter_min_activity_fraction": 0.2,
        "ftol": 1e-6,
        "gtol": 1e-6,
        "deconvolution_filter": True,
        "coverage_filter": True,
        "α": 1e-10,
        "verbose_gtf": True,
        "loci_subset": 0,
        "callback": True,
        "callback_args": (0.001, 0.001),
        "pseudo_min": 1e-14,
        "min_activity_threshold": 0.01,
    }

    with open(args.config, "r") as f:
        json_config = json.load(f)
    config.update(json_config)

    wdir = config["w_dir"]
    odir = config["o_dir"]

    if not config["warm_start"]:
        if os.path.exists(wdir):
            shutil.rmtree(wdir)
        if os.path.exists(odir):
            shutil.rmtree(odir)
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print("load reference annotation...", end="", flush=True)
    start = time.time()
    ref_annotation = ReferenceAnnotation(config["gtf_path"])
    end = time.time()
    print(f"{end - start:.2e} seconds", flush=True)

    print("load genome...", end="", flush=True)
    start = time.time()
    genome = dict((s.name, s) for s in HTSeq.FastaReader(config["fasta_path"]))
    end = time.time()
    print(f"{end - start:.2e} seconds", flush=True)

    #######################
    ### Data Collection ###
    #######################
    db_path = f"{wdir}/price.db"
    data_collector = DataCollector(ref_annotation, genome, config)

    print("compute cleavage models...", end="", flush=True)
    start = time.time()
    data_collector.collect_runs()
    end = time.time()
    print(f"{end - start:.2e} seconds", flush=True)

    print("collect_mappings...", end="", flush=True)
    start = time.time()
    data_collector.collect_mappings()
    end = time.time()
    print(f"{end - start:.2e} seconds", flush=True)

    print("generate ORFs and save loci...", end="", flush=True)
    start = time.time()
    data_collector.collect_loci()
    end = time.time()
    print(f"{end - start:.2e} seconds", flush=True)

    #############################
    ### run ORF deconvolution ###
    #############################

    orf_activity_estimator = ORFActivityEstimator(config)
    print(
        f"\nrun ORF deconvolution for {len(data_collector.runs)} runs and {len(orf_activity_estimator.loci_ids)} loci in {config['processes']} processes...",
        flush=True,
    )
    start = time.time()
    orf_activity_estimator.run_orf_deconvolution()
    end = time.time()
    print(f"{end - start:.2e} seconds", flush=True)
