"""PRICE2: Probabilistic Ribosome Profiling Inference of Coding Elements.

Main entry point for the PRICE2 pipeline.  Orchestrates the full analysis
from raw Ribo-seq BAM files to a table of active translons:

1. Parse reference annotation (GTF) and genome (FASTA).
2. Collect Ribo-seq runs and estimate cleavage/coverage models.
3. Map reads to loci and generate ORF candidates.
4. Run group-LASSO ORF deconvolution in parallel.
"""

import argparse
import logging
import os
import shutil
import sys
import time

import HTSeq

from price2.config import Config
from price2.data_collector import DataCollector
from price2.ribo_seq_run import save_dataset_models
from price2.orf_activity_estimator import ORFActivityEstimator
from price2.reference_annotation import ReferenceAnnotation
from price2.tpm import generate_tpm_output

logger = logging.getLogger("price2.price")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup_logging(config: Config) -> None:
    """Configure the ``price2`` logger.

    Attaches a :class:`logging.StreamHandler` writing to *stderr* to the
    ``price2`` root logger.  The level is taken from
    ``config.log_level`` (standard Python level name, e.g. ``"INFO"`` or
    ``"DEBUG"``).

    Parameters
    ----------
    config : Config
        Fully populated configuration object.
    """
    price2_logger = logging.getLogger("price2")
    price2_logger.setLevel(config.log_level)
    price2_logger.propagate = False

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(config.log_level)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    price2_logger.addHandler(handler)


def _timed(label: str, fn, *args, **kwargs):
    """Run *fn* with *args*/*kwargs*, logging *label* and elapsed time.

    Parameters
    ----------
    label : str
        Message logged before the call.
    fn : callable
        Function to call.
    *args, **kwargs
        Forwarded to *fn*.

    Returns
    -------
    object
        Whatever *fn* returns.
    """
    start = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - start
    if label:
        logger.info("%s%.2e seconds", label, elapsed)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None
        Argument list; defaults to ``sys.argv[1:]`` when *None*.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attribute ``config`` (path string).
    """
    parser = argparse.ArgumentParser(
        description="PRICE2 — estimate ORF activities from Ribo-seq data."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file.",
    )
    return parser.parse_args(argv)


def setup_directories(config: Config) -> None:
    """Create working and output directories from scratch.

    Any pre-existing working and output directories are deleted before
    recreation, ensuring a clean run.

    Parameters
    ----------
    config : Config
        Fully populated configuration object.
    """
    for path in (config.w_dir, config.o_dir):
        if os.path.exists(path):
            shutil.rmtree(path)

    for path in (config.w_dir, config.o_dir):
        os.makedirs(path, exist_ok=True)


def load_genome(fasta_path: str) -> dict:
    """Load the reference genome from a FASTA file.

    Parameters
    ----------
    fasta_path : str
        Path to the genome FASTA file.

    Returns
    -------
    dict
        Mapping from sequence name to ``HTSeq.Sequence`` object.
    """
    return {seq.name: seq for seq in HTSeq.FastaReader(fasta_path)}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(config: Config) -> None:
    """Execute the full PRICE2 pipeline.

    Parameters
    ----------
    config : Config
        Fully populated configuration object.
    """
    setup_directories(config)

    ref_annotation = _timed(
        "load reference annotation... ",
        ReferenceAnnotation,
        config.gtf_path,
    )

    genome = _timed(
        "load genome... ",
        load_genome,
        config.fasta_path,
    )

    # --- Data collection ---
    data_collector = DataCollector(ref_annotation, genome, config)

    _timed(
        "compute cleavage and coverage models... ",
        data_collector.collect_runs,
    )

    _timed(
        "save dataset model summaries... ",
        save_dataset_models,
        data_collector.runs,
        config.o_dir,
    )

    _timed(
        "collect mappings... ",
        data_collector.collect_mappings,
    )

    _timed(
        "generate ORFs and save loci... ",
        data_collector.collect_loci,
    )

    # --- ORF deconvolution ---
    orf_activity_estimator = ORFActivityEstimator(config)

    n_runs = len(data_collector.runs)
    n_loci = len(orf_activity_estimator.loci_ids)
    n_proc = config.processes
    logger.info(
        "run ORF deconvolution for %d run(s) and %d loci in %d process(es)...",
        n_runs,
        n_loci,
        n_proc,
    )

    _timed("", orf_activity_estimator.run_orf_deconvolution)

    # --- TPM output ---
    _timed(
        "generate TPM output... ",
        generate_tpm_output,
        config.o_dir,
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for the PRICE2 command-line interface.

    Parameters
    ----------
    argv : list[str] | None
        Argument list; defaults to ``sys.argv[1:]`` when *None*.
    """
    args = parse_args(argv)
    config = Config.make_config(config=args.config)
    setup_logging(config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
