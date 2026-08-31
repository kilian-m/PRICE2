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
from price2 import run_state
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


def setup_directories(config: Config) -> run_state.ResumePlan:
    """Create the working and output directories and decide what to reuse.

    With ``config.warm_start`` enabled (the default) an existing working
    directory is picked up where the previous invocation stopped: the data
    collection resumes run by run and locus by locus, the multimapping EM
    resumes at its last checkpointed iteration, and the final
    deconvolution resumes at the loci not yet in ``processed_loci.txt``.
    What may be reused is decided by :func:`price2.run_state.plan_resume`
    from the configuration fingerprints stored in the database; a stage
    whose options changed starts over.

    With ``config.warm_start`` disabled both directories are wiped and
    recreated, as is any run whose ``w_dir`` holds no database yet.

    Parameters
    ----------
    config : Config
        Fully populated configuration object.

    Returns
    -------
    price2.run_state.ResumePlan
        What the run may skip.  A cold start returns a plan that skips
        nothing.

    Raises
    ------
    price2.run_state.IncompatibleRunStateError
        When the existing database was collected under different
        collection options (see :func:`price2.run_state.plan_resume`).
    """
    db_path = os.path.join(config.w_dir, "price.db")

    if not (config.warm_start and os.path.exists(db_path)):
        for path in (config.w_dir, config.o_dir):
            if os.path.exists(path):
                shutil.rmtree(path)
        for path in (config.w_dir, config.o_dir):
            os.makedirs(path, exist_ok=True)
        run_state.record_configuration(config, db_path)
        return run_state.ResumePlan(
            skip_collection=False,
            reuse_deconvolution=False,
            reason="cold start",
        )

    plan = run_state.plan_resume(config, db_path)
    os.makedirs(config.o_dir, exist_ok=True)
    processed_loci_path = os.path.join(config.w_dir, "processed_loci.txt")

    if plan.reuse_deconvolution and not _outputs_resumable(
        config, processed_loci_path
    ):
        plan = run_state.ResumePlan(
            skip_collection=plan.skip_collection,
            reuse_deconvolution=False,
            reason="reusing the collected data; the deconvolution starts over",
        )

    if not plan.reuse_deconvolution:
        if os.path.exists(config.o_dir):
            shutil.rmtree(config.o_dir)
        os.makedirs(config.o_dir, exist_ok=True)
        if os.path.exists(processed_loci_path):
            os.remove(processed_loci_path)

    run_state.record_configuration(config, db_path)
    return plan


def _outputs_resumable(config: Config, processed_loci_path: str) -> bool:
    """Reconcile an existing output directory with the finished-locus list.

    Parameters
    ----------
    config : Config
        Fully populated configuration object.
    processed_loci_path : str
        Path to ``processed_loci.txt`` in the working directory.

    Returns
    -------
    bool
        ``True`` when the outputs were made consistent and the finished
        loci may be skipped; ``False`` when the deconvolution has to start
        over.
    """
    ra_dir = os.path.join(config.o_dir, "regions_activities")
    if os.path.exists(processed_loci_path) and not os.path.isdir(ra_dir):
        # The results those loci produced are gone; skipping them now would
        # silently drop them from the output.
        logger.warning(
            "%s lists finished loci but %s no longer exists; the "
            "deconvolution starts over.",
            processed_loci_path,
            ra_dir,
        )
        return False

    if not run_state.repair_outputs(config.o_dir, processed_loci_path):
        logger.warning(
            "the outputs in %s cannot be reconciled with %s; the "
            "deconvolution starts over.",
            config.o_dir,
            processed_loci_path,
        )
        return False

    return True


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
    plan = setup_directories(config)
    db_path = os.path.join(config.w_dir, "price.db")
    logger.info("%s (%s)", plan.reason, config.w_dir)

    if not plan.skip_collection:
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

        # Unlike the stages above, this one is not repeatable: it consumes
        # the spilled alignments and deletes them, so re-running it after a
        # successful build would replace a valid index with an empty one.
        # A populated index with no spill left beside it is therefore taken
        # as already built; a spill that is still there means alignments
        # have been collected since, and the index is rebuilt to include
        # them (that also covers a build interrupted before its cleanup).
        if config.multimap_em and (
            not multimap.has_multimap_index(db_path)
            or os.path.isdir(multimap.spill_dir(db_path))
        ):
            _timed(
                "build multimapping linkage index... ",
                multimap.build_multimap_index,
                f"{config.w_dir}/price.db",
                processes=config.processes,
            )

        run_state.write_state(db_path, collection_complete="1")

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
        _run_em_deconvolution(
            config, orf_activity_estimator, resume=plan.reuse_deconvolution
        )
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
    resume: bool = False,
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
    resume : bool, optional
        Continue an interrupted EM from its last checkpoint instead of
        restarting it (see :func:`price2.multimap.em_resume_point`).  The
        caller sets this from the run's :class:`~price2.run_state.ResumePlan`.
    """
    db_path = f"{config.w_dir}/price.db"

    if not multimap.has_multimap_index(db_path):
        logger.warning(
            "multimap_em is enabled but no populated linkage index was "
            "found in %s. Either no read maps to >=2 in-locus slots, or "
            "price.db was collected with multimap_em disabled -- in which "
            "case its multimapping reads were discarded as well, and only a "
            "cold re-collection with multimap_em=true can restore them. "
            "Running a single classic pass instead.",
            db_path,
        )
        _timed("", estimator.run_orf_deconvolution)
        return

    multimap.enable_wal(db_path)

    checkpoint = multimap.em_resume_point(db_path) if resume else None
    if checkpoint is None:
        # Clear any per-iteration state from a previous run so a warm re-run
        # cannot consume stale λ / weights / activities.
        multimap.reset_em_state(db_path)
        run_state.write_state(db_path, em_final_iteration="")
        start_iteration, finished = 0, set()
    else:
        start_iteration, finished = checkpoint

    # Loci with no multimap slots do not change across EM iterations, so
    # the light passes only need to touch the loci that carry slots.
    slot_loci = multimap.slot_locus_ids(db_path)

    # The loop has already ended if its last run recorded the iteration the
    # final pass consumes and the checkpoint still sits there; go straight to
    # the final pass rather than paying another M-step and E-step for nothing.
    final_iteration = start_iteration
    stored_final = run_state.read_state(db_path).get("em_final_iteration")
    skip_loop = checkpoint is not None and stored_final == str(start_iteration)
    if skip_loop:
        logger.info(
            "EM already converged; resuming at the final full M-step "
            "(iteration %d).",
            start_iteration,
        )
    elif checkpoint is not None:
        logger.info(
            "resuming the multimapping EM at iteration %d (%d of %d slot "
            "loci already done).",
            start_iteration,
            len(finished & slot_loci),
            len(slot_loci),
        )

    last_it = start_iteration
    # One broker pool for the whole EM: every M-step would otherwise rebuild it,
    # paying a CUDA context per broker process per iteration.  Likewise hold one
    # worker pool + log listener open across every M-step (and the final pass)
    # instead of spawning and joining a fresh 40-worker pool + manager each time.
    with estimator.gpu_broker_pool(), estimator.worker_pool():
        if not skip_loop:
            for it in range(start_iteration, config.em_max_iter):
                last_it = it
                # Only the resumed iteration has loci already behind it; every
                # later one starts empty.
                subset = slot_loci - finished
                finished = set()
                if subset:
                    _timed(
                        f"EM iteration {it} light M-step... ",
                        estimator.run_orf_deconvolution,
                        em_iteration=it,
                        em_final=False,
                        loci_subset=subset,
                    )
                else:
                    logger.info(
                        "EM iteration %d light M-step was already complete.",
                        it,
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

            final_iteration = last_it + 1
            # From here a resume can skip straight to the final pass.
            run_state.write_state(
                db_path, em_final_iteration=str(final_iteration)
            )

        # Final full M-step with the converged fractional weights.
        _timed(
            "EM final full M-step... ",
            estimator.run_orf_deconvolution,
            em_iteration=final_iteration,
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
    try:
        run_pipeline(config)
    except run_state.IncompatibleRunStateError as exc:
        logger.error("%s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
