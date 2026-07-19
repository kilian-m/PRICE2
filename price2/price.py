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
from pyfaidx import Fasta

from price2 import multimap
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


def setup_directories(config: Config) -> bool:
    """Create working and output directories.

    When ``config.warm_start`` is ``True`` and a populated SQLite
    database already exists in ``w_dir``, only the output directory is
    cleared and ``processed_loci.txt`` is removed so that deconvolution
    reruns all loci.  If the database is absent the run falls back to a
    full cold start (both directories are wiped and recreated).

    When ``config.warm_start`` is ``False`` both directories are always
    wiped and recreated.

    Parameters
    ----------
    config : Config
        Fully populated configuration object.

    Returns
    -------
    bool
        ``True`` when data collection will be skipped (warm start
        effective), ``False`` when a full cold run is required.
    """
    db_path = os.path.join(config.w_dir, "price.db")
    warm = config.warm_start and os.path.exists(db_path)

    if warm:
        if os.path.exists(config.o_dir):
            shutil.rmtree(config.o_dir)
        os.makedirs(config.o_dir, exist_ok=True)
        processed_loci_path = os.path.join(config.w_dir, "processed_loci.txt")
        if os.path.exists(processed_loci_path):
            os.remove(processed_loci_path)
    else:
        for path in (config.w_dir, config.o_dir):
            if os.path.exists(path):
                shutil.rmtree(path)
        for path in (config.w_dir, config.o_dir):
            os.makedirs(path, exist_ok=True)

    return warm


def load_genome(fasta_path: str) -> Fasta:
    """Open the reference genome FASTA via :mod:`pyfaidx`.

    Returns an ``mmap``-backed handle so that the OS shares a single
    page cache across worker processes.  The ``.fai`` index is built
    on first access if not already present.

    Parameters
    ----------
    fasta_path : str
        Path to the genome FASTA file.

    Returns
    -------
    pyfaidx.Fasta
        Chromosome-keyed indexed FASTA handle.
    """
    return Fasta(fasta_path)


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
    warm = setup_directories(config)

    if warm:
        logger.info(
            "warm start: reusing existing database in %s",
            config.w_dir,
        )
    else:
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

        if config.export_dataset_models:
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

        if config.multimap_em:
            _timed(
                "build multimapping linkage index... ",
                multimap.build_multimap_index,
                f"{config.w_dir}/price.db",
                processes=config.processes,
            )

    # --- ORF deconvolution ---
    orf_activity_estimator = ORFActivityEstimator(config)

    n_loci = len(orf_activity_estimator.loci_ids)
    n_proc = config.processes
    logger.info(
        "run ORF deconvolution for %d loci in %d process(es)...",
        n_loci,
        n_proc,
    )

    if config.multimap_em:
        _run_em_deconvolution(config, orf_activity_estimator)
    else:
        _timed("", orf_activity_estimator.run_orf_deconvolution)

    # --- TPM output ---
    _timed(
        "generate TPM output... ",
        generate_tpm_output,
        config.o_dir,
        export_tsv=config.export_tsv,
    )


def _run_em_deconvolution(
    config: Config,
    estimator: ORFActivityEstimator,
) -> None:
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
    config : Config
        Fully populated configuration object.
    estimator : ORFActivityEstimator
        Estimator bound to the run's database.
    """
    db_path = f"{config.w_dir}/price.db"

    if not multimap.has_multimap_index(db_path):
        logger.warning(
            "multimap_em is enabled but no populated linkage index was "
            "found in %s. Either no read maps to >=2 in-locus slots, or "
            "price.db was collected with multimap_em disabled (re-run a "
            "cold collection with multimap_em=true to build the linkage). "
            "Running a single classic pass instead.",
            db_path,
        )
        _timed("", estimator.run_orf_deconvolution)
        return

    multimap.enable_wal(db_path)
    # Clear any per-iteration state from a previous run so a warm re-run
    # cannot consume stale λ / weights / activities.
    multimap.reset_em_state(db_path)

    # Loci with no multimap slots do not change across EM iterations, so
    # the light passes only need to touch the loci that carry slots.
    slot_loci = multimap.slot_locus_ids(db_path)

    last_it = 0
    # One broker pool for the whole EM: every M-step would otherwise rebuild it,
    # paying a CUDA context per broker process per iteration.  Likewise hold one
    # worker pool + log listener open across every M-step (and the final pass)
    # instead of spawning and joining a fresh 40-worker pool + manager each time.
    with estimator.gpu_broker_pool(), estimator.worker_pool():
        for it in range(config.em_max_iter):
            last_it = it
            _timed(
                f"EM iteration {it} light M-step... ",
                estimator.run_orf_deconvolution,
                em_iteration=it,
                em_final=False,
                loci_subset=slot_loci,
            )
            delta = multimap.e_step(db_path, iteration=it)
            logger.info(
                "EM iteration %d: read mass reassigned (L1 fraction) = %.3e",
                it,
                delta,
            )
            if delta < config.em_tol:
                logger.info(
                    "EM converged after %d iteration(s) (tol=%.1e).",
                    it + 1,
                    config.em_tol,
                )
                break

        # Final full M-step with the converged fractional weights.
        _timed(
            "EM final full M-step... ",
            estimator.run_orf_deconvolution,
            em_iteration=last_it + 1,
            em_final=True,
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
