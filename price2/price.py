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
from data_collector import DataCollector
from orf_activity_estimator import ORFActivityEstimator
from config import Config


if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser(description="Estimate ORF activities.")
    parser.add_argument("--config", type=str, help="Path to JSON config file.")

    args = parser.parse_args()

    config = Config.make_config(config=args.config)

    wdir = config.w_dir
    odir = config.o_dir

    if not config.warm_start:
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
    ref_annotation = ReferenceAnnotation(config.gtf_path)
    end = time.time()
    print(f"{end - start:.2e} seconds", flush=True)

    print("load genome...", end="", flush=True)
    start = time.time()
    genome = dict((s.name, s) for s in HTSeq.FastaReader(config.fasta_path))
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
        f"\nrun ORF deconvolution for {len(data_collector.runs)} runs and {len(orf_activity_estimator.loci_ids)} loci in {config.processes} processes...",
        flush=True,
    )
    start = time.time()
    orf_activity_estimator.run_orf_deconvolution()
    end = time.time()
    print(f"{end - start:.2e} seconds", flush=True)

    with open(f"{odir}/failed_loci.txt", "w") as f:
        for loc_id, e in orf_activity_estimator.failed_loci.items():
            if hasattr(e, "traceback"):
                f.write(f"{loc_id}\n{e.traceback}\n\n\n")
            else:
                f.write(f"{loc_id}\n{e}\n\n\n")

    print(f"orf deconvolution done in {end - start:.2e} seconds")
    print(f"successful loci: {len(orf_activity_estimator.loci)}")
    print(f"failed loci: {len(orf_activity_estimator.failed_loci)}")
