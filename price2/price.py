"""PRICE2: Probabilistic Ribosome Profiling Inference of Coding Elements.

Main entry point for the PRICE2 pipeline.  Orchestrates the full analysis
from raw Ribo-seq BAM files to a table of active translons:

1. Parse the reference annotation (GTF).
2. Collect Ribo-seq runs and estimate cleavage/coverage models.
3. Map reads to loci and generate ORF candidates.
4. Run group-LASSO ORF deconvolution in parallel.
"""

import argparse
import logging
import os
import shutil
import sys

# Before numpy is imported anywhere: every worker process is single-threaded
# by design (the pools fill the cores), so the numerical libraries get one
# thread each unless the environment says otherwise.  The workers inherit it.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from price2 import multimap  # noqa: E402
from price2 import run_state
from price2.config import Config
from price2.data_collector import DataCollector
from price2.dataset_models import save_dataset_models
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
    deconvolution resumes at the loci not yet recorded as finished.
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

    run_state.migrate_legacy_progress(db_path, config.w_dir)
    plan = run_state.plan_resume(config, db_path)
    os.makedirs(config.o_dir, exist_ok=True)

    if plan.reuse_deconvolution and not _outputs_resumable(config):
        plan = run_state.ResumePlan(
            skip_collection=plan.skip_collection,
            reuse_deconvolution=False,
            reason="reusing the collected data; the deconvolution starts over",
        )

    if not plan.reuse_deconvolution:
        if os.path.exists(config.o_dir):
            shutil.rmtree(config.o_dir)
        os.makedirs(config.o_dir, exist_ok=True)
        run_state.clear_progress(
            db_path, run_state.DECONVOLUTION_STAGE, run_state.EM_STAGE
        )

    run_state.record_configuration(config, db_path)
    return plan


def _outputs_resumable(config: Config) -> bool:
    """Reconcile an existing output directory with the finished loci.

    Parameters
    ----------
    config : Config
        Fully populated configuration object.

    Returns
    -------
    bool
        ``True`` when the outputs were made consistent and the finished
        loci may be skipped; ``False`` when the deconvolution has to start
        over.
    """
    db_path = config.layout.db_path
    done = set(run_state.read_progress(db_path, run_state.DECONVOLUTION_STAGE))
    ra_dir = config.layout.regions_activities_dir
    if done and not os.path.isdir(ra_dir):
        # The results those loci produced are gone; skipping them now would
        # silently drop them from the output.
        logger.warning(
            "%s records %d finished loci but %s no longer exists; the "
            "deconvolution starts over.",
            db_path,
            len(done),
            ra_dir,
        )
        return False

    if not run_state.repair_outputs(config.o_dir, done):
        logger.warning(
            "the outputs in %s cannot be reconciled with the finished loci "
            "recorded in %s; the deconvolution starts over.",
            config.o_dir,
            db_path,
        )
        return False

    return True


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
        run_state.write_progress(
            db_path, run_state.COLLECTION_STAGE, [run_state.COMPLETE]
        )

    run_stages(_deconvolution_stages(config, plan))


class _Collection:
    """The data-collection stages and the objects they share."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.collector: DataCollector | None = None

    def load_inputs(self) -> None:
        reference = run_stage(
            "load reference annotation",
            lambda: ReferenceAnnotation.load_cached(
                self.config.gtf_path, self.config.w_dir
            ),
        )
        self.collector = DataCollector(reference, self.config)

    def collect_runs(self) -> None:
        self.collector.collect_runs()

    def save_models(self) -> None:
        save_dataset_models(
            self.collector.runs, self.config.layout.dataset_models_dir
        )

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


def _deconvolution_stages(config: Config, plan: run_state.ResumePlan) -> list[Stage]:
    """The stages that turn ``price.db`` into the output tables, in order."""
    estimator = ORFActivityEstimator(config)
    logger.info(
        "run ORF deconvolution for %d loci in %d process(es)...",
        len(estimator.loci_ids),
        config.processes,
    )
    return [
        Stage(
            "ORF deconvolution with the multimapping EM",
            lambda: estimator.run_multimap_em(resume=plan.reuse_deconvolution),
            enabled=config.multimap_em,
        ),
        Stage(
            "ORF deconvolution",
            estimator.run_orf_deconvolution,
            enabled=not config.multimap_em,
        ),
        Stage(
            "generate TPM output",
            lambda: generate_tpm_output(config.o_dir, export_tsv=config.export_tsv),
        ),
    ]


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
