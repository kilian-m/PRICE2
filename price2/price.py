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
from dataclasses import dataclass

from pyfaidx import Fasta

from price2 import database
from price2 import multimap
from price2 import run_state
from price2.config import Config
from price2.data_collector import DataCollector
from price2.ribo_seq_run import save_dataset_models
from price2.orf_activity_estimator import ORFActivityEstimator
from price2.pipeline import Stage, run_stage, run_stages
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
    db_path = config.layout.db_path

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
    processed_loci_path = config.layout.processed_loci_path

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
    ra_dir = config.layout.regions_activities_dir
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
    db_path = config.layout.db_path
    logger.info("%s (%s)", plan.reason, config.w_dir)

    if not plan.skip_collection:
        run_stages(_collection_stages(config))
        run_state.write_state(db_path, collection_complete="1")

    estimator = ORFActivityEstimator(config)
    logger.info(
        "run ORF deconvolution for %d loci in %d process(es)...",
        len(estimator.loci_ids),
        config.processes,
    )
    if config.multimap_em:
        _run_em_deconvolution(config, estimator, resume=plan.reuse_deconvolution)
    else:
        run_stage("ORF deconvolution", estimator.run_orf_deconvolution)

    run_stage(
        "generate TPM output",
        lambda: generate_tpm_output(config.o_dir, export_tsv=config.export_tsv),
    )


class _Collection:
    """The data-collection stages and the objects they share."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.collector: DataCollector | None = None

    def load_inputs(self) -> None:
        reference = run_stage(
            "load reference annotation",
            lambda: ReferenceAnnotation(self.config.gtf_path),
        )
        genome = run_stage("load genome", lambda: load_genome(self.config.fasta_path))
        self.collector = DataCollector(reference, genome, self.config)

    def collect_runs(self) -> None:
        self.collector.collect_runs()

    def save_models(self) -> None:
        save_dataset_models(self.collector.runs, self.config.o_dir)

    def collect_mappings(self) -> None:
        self.collector.collect_mappings()

    def collect_loci(self) -> None:
        self.collector.collect_loci()

    def build_multimap_index(self) -> None:
        multimap.build_multimap_index(
            self.config.layout.db_path, processes=self.config.processes
        )

    def multimap_index_built(self) -> bool:
        # Unlike the other stages this one is not repeatable: it consumes the
        # spilled alignments and deletes them, so re-running it after a
        # successful build would replace a valid index with an empty one.  A
        # populated index with no spill left beside it is therefore taken as
        # built; a spill that is still there means alignments have been
        # collected since (or a build was interrupted before its cleanup), and
        # the index is rebuilt to include them.
        db_path = self.config.layout.db_path
        return multimap.has_multimap_index(db_path) and not os.path.isdir(
            multimap.spill_dir(db_path)
        )


def _collection_stages(config: Config) -> list[Stage]:
    """The stages that fill ``price.db``, in order."""
    collection = _Collection(config)
    return [
        Stage("load inputs", collection.load_inputs),
        Stage("compute cleavage and coverage models", collection.collect_runs),
        Stage(
            "save dataset model summaries",
            collection.save_models,
            enabled=config.export_dataset_models,
        ),
        Stage("collect mappings", collection.collect_mappings),
        Stage("generate ORFs and save loci", collection.collect_loci),
        Stage(
            "build multimapping linkage index",
            collection.build_multimap_index,
            enabled=config.multimap_em,
            done=collection.multimap_index_built,
        ),
    ]


@dataclass(frozen=True)
class EmCheckpoint:
    """Where an interrupted multimapping EM continues.

    Parameters
    ----------
    start_iteration : int
        The iteration to run next.
    finished : set[str]
        Loci whose light M-step of that iteration already completed.
    final_only : bool
        The loop had already converged; only the final full pass is left.
    """

    start_iteration: int
    finished: set[str]
    final_only: bool


def _em_checkpoint(db_path: str, resume: bool) -> EmCheckpoint:
    """Read the EM checkpoint, or reset the EM state for a fresh loop."""
    point = multimap.em_resume_point(db_path) if resume else None
    if point is None:
        # Clear any per-iteration state from a previous run so a warm re-run
        # cannot consume stale λ / weights / activities.
        multimap.reset_em_state(db_path)
        run_state.write_state(db_path, em_final_iteration="")
        return EmCheckpoint(0, set(), False)
    start_iteration, finished = point
    # The loop has already ended if its last run recorded the iteration the
    # final pass consumes and the checkpoint still sits there.
    stored_final = run_state.read_state(db_path).get("em_final_iteration")
    return EmCheckpoint(start_iteration, finished, stored_final == str(start_iteration))


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
    db_path = config.layout.db_path

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
        run_stage("ORF deconvolution", estimator.run_orf_deconvolution)
        return

    database.enable_wal(db_path)
    checkpoint = _em_checkpoint(db_path, resume)
    # Loci with no multimap slots do not change across EM iterations, so the
    # light passes only need to touch the loci that carry slots.
    slot_loci = multimap.slot_locus_ids(db_path)
    if checkpoint.final_only:
        logger.info(
            "EM already converged; resuming at the final full M-step "
            "(iteration %d).",
            checkpoint.start_iteration,
        )
    elif resume and (checkpoint.start_iteration or checkpoint.finished):
        logger.info(
            "resuming the multimapping EM at iteration %d (%d of %d slot "
            "loci already done).",
            checkpoint.start_iteration,
            len(checkpoint.finished & slot_loci),
            len(slot_loci),
        )

    # One broker pool and one worker pool for the whole EM: every M-step
    # would otherwise rebuild them, paying a CUDA context per broker process
    # and a fresh pool + manager per iteration.
    with estimator.gpu_broker_pool(), estimator.worker_pool():
        final_iteration = checkpoint.start_iteration
        if not checkpoint.final_only:
            final_iteration = _em_loop(
                config, estimator, db_path, checkpoint, slot_loci
            )
            # From here a resume can skip straight to the final pass.
            run_state.write_state(db_path, em_final_iteration=str(final_iteration))
        run_stage(
            "EM final full M-step",
            lambda: estimator.run_orf_deconvolution(
                em_iteration=final_iteration, em_final=True
            ),
        )


def _em_loop(
    config: Config,
    estimator: ORFActivityEstimator,
    db_path: str,
    checkpoint: EmCheckpoint,
    slot_loci: set[str],
) -> int:
    """Alternate light M-steps and E-steps; return the final pass's iteration."""
    finished = checkpoint.finished
    last_iteration = checkpoint.start_iteration
    for iteration in range(checkpoint.start_iteration, config.em_max_iter):
        last_iteration = iteration
        # Only the resumed iteration has loci already behind it; every later
        # one starts empty.
        subset = slot_loci - finished
        finished = set()
        if subset:
            run_stage(
                f"EM iteration {iteration} light M-step",
                lambda: estimator.run_orf_deconvolution(
                    em_iteration=iteration, em_final=False, loci_subset=subset
                ),
            )
        else:
            logger.info(
                "EM iteration %d light M-step was already complete.", iteration
            )
        delta = multimap.e_step(db_path, iteration=iteration)
        logger.info(
            "EM iteration %d: read mass reassigned (L1 fraction) = %.3e",
            iteration,
            delta,
        )
        if delta < config.em_tol:
            logger.info(
                "EM converged after %d iteration(s) (tol=%.1e).",
                iteration + 1,
                config.em_tol,
            )
            break
    return last_iteration + 1


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
