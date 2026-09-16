from __future__ import annotations

from dataclasses import MISSING, Field, dataclass, field
import json
import logging
import os

from price2.layout import RunLayout

logger = logging.getLogger(__name__)

#: Options that earlier releases accepted and that no longer do anything.  They
#: are dropped with a warning so that an old configuration file still loads.
_OBSOLETE_FIELDS: frozenset[str] = frozenset(
    {"l_file", "memory_limit_gb", "save_memory", "multimap_background"}
)

# The scope of an option says what a change of it invalidates when a run is
# resumed (:mod:`price2.run_state` fingerprints the options by scope).  It is
# recorded on the field, next to the option, with :func:`option`.
#: The option decides the content of ``price.db``.
COLLECTION = "collection"
#: The option changes a result but not the database (the default scope).
DECONVOLUTION = "deconvolution"
#: The option only says where things are or how fast to go.
RUNTIME = "runtime"


def option(
    default: object = MISSING,
    *,
    scope: str = DECONVOLUTION,
    path: bool = False,
    **kwargs: object,
) -> object:
    """Declare a configuration field with its scope.

    Parameters
    ----------
    default : object
        The default value; omitted for a required option.
    scope : str
        One of :data:`COLLECTION`, :data:`DECONVOLUTION`, :data:`RUNTIME`.
    path : bool
        The value is a path to an input, to be fingerprinted by its
        basename so that a relocated analysis stays resumable.
    **kwargs : object
        Passed on to :func:`dataclasses.field`.
    """
    if scope not in (COLLECTION, DECONVOLUTION, RUNTIME):
        raise ValueError(f"unknown option scope {scope!r}")
    return field(default=default, metadata={"scope": scope, "path": path}, **kwargs)


def option_scope(f: Field) -> str:
    """The scope of a :class:`Config` field (:data:`DECONVOLUTION` unless declared)."""
    return f.metadata.get("scope", DECONVOLUTION)


def is_path_option(f: Field) -> bool:
    """Whether a :class:`Config` field is an input path (see :func:`option`)."""
    return bool(f.metadata.get("path", False))


@dataclass
class Config:
    """Configuration for a PRICE2 run.

    All options are loaded from a JSON file or keyword arguments via
    :meth:`make_config`.  Every option is documented at its field below; the
    reasoning behind the tuned defaults is in ``docs/tuning.md``.  Path
    options left empty are derived from ``base_dir`` in :meth:`__post_init__`::

        base_dir/
        ├── o_dir/      # output
        ├── w_dir/      # working directory, holds price.db
        └── bam_dir/    # mapped Ribo-seq BAM files

    Every option carries a scope (:func:`option`) that says what a change of
    it invalidates in a resumed run: the collected database, only the
    deconvolution (the default), or nothing (:mod:`price2.run_state`).
    """

    # ------------------------------------------------------------------ #
    # Required                                                             #
    # ------------------------------------------------------------------ #
    #: Root directory of the run; the empty path options below default to
    #: its subdirectories.
    base_dir: str = option(scope=RUNTIME)

    # ------------------------------------------------------------------ #
    # Paths (derived from base_dir when left empty)                       #
    # ------------------------------------------------------------------ #
    #: Output directory (``<base_dir>/o_dir``).
    o_dir: str = option("", scope=RUNTIME)
    #: Working directory holding ``price.db`` (``<base_dir>/w_dir``).
    w_dir: str = option("", scope=RUNTIME)
    #: Reference annotation, GTF.
    gtf_path: str = option("", scope=COLLECTION, path=True)
    #: Reference genome, FASTA.
    fasta_path: str = option("", scope=COLLECTION, path=True)
    #: Directory of the coordinate-sorted, indexed BAM files
    #: (``<base_dir>/bam_dir``).
    bam_dir: str = option("", scope=COLLECTION, path=True)
    #: The ``{bam_id}.bam`` files of ``bam_dir`` to use; ``None`` uses all.
    bam_ids: list[str] | None = option(None, scope=COLLECTION)

    # ------------------------------------------------------------------ #
    # Read mapping                                                          #
    # ------------------------------------------------------------------ #
    #: How STAR aligned the read ends.  ``"local"`` (``--alignEndsType
    #: Local``) leaves the reverse-transcription untemplated nucleotide as a
    #: 1-nt 5' soft-clip that PRICE2 reads directly.  ``"endtoend"`` has no
    #: soft-clips: the nucleotide is recovered from a 5'-terminal mismatch
    #: (needs the ``MD`` tag) and trimmed off the footprint, reproducing the
    #: Local geometry.  The two are close but not bit-identical, as the
    #: ~1-2 % of such reads whose extra mismatch trips STAR's
    #: ``outFilterMismatchNmax`` never reach PRICE2 under EndToEnd.
    align_ends_type: str = option("local", scope=COLLECTION)

    # ------------------------------------------------------------------ #
    # Parallelism & runtime                                                #
    # ------------------------------------------------------------------ #
    #: Worker processes.  A fixed default, not the host's core count.
    processes: int = option(80, scope=RUNTIME)
    #: Per-locus wall-clock budget in seconds *per Ribo-seq run*: a locus is
    #: abandoned after ``timeout * len(runs)`` seconds, since one locus is
    #: solved for all runs at once ...
    timeout: int = option(180, scope=RUNTIME)
    #: ... but never given more than this many seconds in total (``0``: no
    #: cap).  The per-run budget suits panels of a few runs; with a hundred
    #: it would let one locus run for hours, and a locus that needs them is
    #: one that will not finish anyway.  Abandoned loci are named in the log
    #: and in ``failed_loci.txt``.
    timeout_cap: int = option(3600, scope=RUNTIME)
    #: The order the loci are handed to the workers.  ``"database"`` keeps
    #: the order of ``price.db``, which interleaves heavy and light loci;
    #: ``"largest"`` puts the loci with the most stored reads first, which
    #: shortens a pass's tail but makes every worker stream its largest
    #: matrices at the same time and, measured, doubles a pass instead
    #: (memory bandwidth; ``docs/tuning.md``).
    dispatch_order: str = option("database", scope=RUNTIME)
    #: Floor on the activities during optimisation, guarding ``log(0)`` in
    #: the likelihood.  Do not lower it.
    pseudo_min: float = 1e-14
    #: Loci a worker handles before it is replaced (``0``: never).  Recycling
    #: bounds the memory growth from the occasional huge locus; a respawn
    #: costs ~2 CPU-seconds (see ``docs/tuning.md``).
    worker_max_tasks: int = option(1000, scope=RUNTIME)

    # ------------------------------------------------------------------ #
    # Transcript pre-filtering                                             #
    # ------------------------------------------------------------------ #
    #: A transcript is kept only if it explains at least this many reads per
    #: run (``min_explained_reads_per_run * len(runs)`` in total).
    min_explained_reads_per_run: int = 5

    # ------------------------------------------------------------------ #
    # Coverage filter                                                      #
    # ------------------------------------------------------------------ #
    #: Remove ORF candidates with too few well-fitting reads per nucleotide.
    coverage_filter: bool = True
    #: Its threshold on well-fitting reads per nucleotide, taken as the
    #: maximum over the runs.
    min_well_fitting_reads_per_length: float = 0.01

    # ------------------------------------------------------------------ #
    # Deconvolution filter                                                 #
    # ------------------------------------------------------------------ #
    #: Before the main deconvolution, remove ORF candidates whose activity
    #: within their stop-codon group is negligible.
    deconvolution_filter: bool = True
    #: The activity below which that filter drops a candidate; also what
    #: counts as "active" in the IRLS-Huber stopping rule.
    deconvolution_filter_min_activity: float = 0.01

    # ------------------------------------------------------------------ #
    # Main optimisation                                                    #
    # ------------------------------------------------------------------ #
    #: Convergence of the main optimisation: stop once no active activity
    #: changed by more than this relative amount.
    stop_factor_relative: float = 0.01
    #: ``ftol`` / ``gtol`` / ``maxls`` of ``scipy.optimize.minimize`` on the
    #: legacy L-BFGS-B path (``inner_solver="lbfgs"``); zero tolerances leave
    #: ``stop_factor_relative`` in charge.  Ignored by the multiplicative
    #: updates.
    ftol: float = 0
    gtol: float = 0
    maxls: int = 200
    #: Group-LASSO penalty λ, calibrated for ``inner_solver="mu"`` by an
    #: AIC/BIC scan (``docs/tuning.md``).
    lam: float = 10

    # ------------------------------------------------------------------ #
    # RGR activity thresholds                                              #
    # ------------------------------------------------------------------ #
    #: An RGR must reach this activity in at least one run to be retained.
    rgr_min_activity: float = 0.01
    #: An ORF must reach this fraction of its locus's canonical ORF activity
    #: to be retained.
    min_activity_fraction: float = 0.1

    # ------------------------------------------------------------------ #
    # IRLS-Huber                                                           #
    # ------------------------------------------------------------------ #
    #: Huber constant of the robust reweighting: low is more robust, high
    #: approaches the plain maximum-likelihood fit.
    irls_huber_c: float = 3.0
    #: Cap on the IRLS-Huber outer iterations.
    irls_huber_max_outer: int = 10
    #: Stop the outer loop once the relative L2 change of the weights falls
    #: below this.
    irls_huber_tol: float = 1e-4
    #: Also stop once the set of RGRs above
    #: ``deconvolution_filter_min_activity`` has not changed for
    #: ``irls_active_patience`` outer iterations: the call set converges long
    #: before the weight norm does (``docs/tuning.md``).
    irls_stop_on_active_set: bool = True
    irls_active_patience: int = 3

    # ------------------------------------------------------------------ #
    # Count distribution for the deconvolution likelihood                  #
    # ------------------------------------------------------------------ #
    #: Count model of every deconvolution solve: ``"poisson"`` (variance =
    #: mean) or ``"nb"``, a negative binomial with variance ``μ + μ²/θ`` for
    #: overdispersed counts; ``θ → ∞`` recovers the Poisson.
    distribution: str = "poisson"
    #: Negative-binomial dispersion ``θ`` (the "size"); smaller means more
    #: overdispersion.  Ignored for ``"poisson"``.
    nb_dispersion: float = 10.0

    # ------------------------------------------------------------------ #
    # Inner solver for the group-LASSO deconvolution                       #
    # ------------------------------------------------------------------ #
    #: ``"mu"``: multiplicative (weighted Richardson-Lucy + group-LASSO)
    #: updates at every solve site, converging to the optimum that L-BFGS-B
    #: stalls short of and hence to a sparser call set; ``"lbfgs"``: the
    #: legacy scipy L-BFGS-B path.  See ``docs/tuning.md``.
    inner_solver: str = "mu"

    # GPU offload of the multiplicative updates.  Off by default: the solves
    # are too small for the GPU to pay off end to end (``docs/tuning.md``).
    # Both paths need PyTorch built against CUDA, which is not a declared
    # dependency, and fall back to the CPU updates when it is missing.
    #: Run the multiplicative updates on the GPU when available and the
    #: system has at least ``mu_gpu_min_rows`` rows; every worker gets its
    #: own CUDA context, so VRAM scales with ``processes``.
    mu_gpu: bool = option(False, scope=RUNTIME)
    #: Below this row count the CPU is faster than the transfer.
    mu_gpu_min_rows: int = option(50_000, scope=RUNTIME)
    #: GPU dtype of the updates, ``"float32"`` or ``"float64"``.
    mu_dtype: str = "float32"
    #: Iteration cap and relative-change tolerance of the multiplicative
    #: inner loop (CPU and GPU).
    mu_inner_max_iter: int = 3000
    mu_inner_tol: float = 1e-5
    #: The CPU update loop: ``"numba"`` runs each update as scipy's C mat-vecs
    #: plus two compiled element-wise passes, ``"numpy"`` as the chain of
    #: numpy calls it replaced.  Both compute the same iterates; the kernel
    #: only removes the per-call overhead that dominates small solves.
    mu_kernel: str = option("numba", scope=RUNTIME)
    #: Serve the GPU updates from broker processes holding one CUDA context
    #: each, shared by the worker pool over shared memory, so VRAM does not
    #: scale with ``processes``.  Requires ``inner_solver="mu"``; falls back
    #: to the per-worker path when the broker cannot start.
    mu_broker: bool = option(False, scope=RUNTIME)
    #: Broker processes (independent CUDA contexts and GILs).
    mu_broker_procs: int = option(4, scope=RUNTIME)
    #: CUDA stream threads per broker process; GIL-bound, so scale
    #: ``mu_broker_procs`` first.
    mu_broker_streams: int = option(2, scope=RUNTIME)
    #: Runtime only: the request queue of a running broker, set on the copy
    #: of the configuration handed to the workers.  Never read from a file.
    mu_broker_req_q: object = option(None, scope=RUNTIME, repr=False, compare=False)

    # ------------------------------------------------------------------ #
    # Likelihood-ratio filter                                              #
    # ------------------------------------------------------------------ #
    #: Apply a likelihood-ratio test as the final filtering step.
    likelihood_ratio_filter: bool = True
    #: Significance level of that test.
    likelihood_ratio_alpha: float = 1e-10
    #: The test's refits are per run (the objective is a sum over the runs):
    #: a reduced model is re-solved only in the runs where clamping the
    #: candidate lowers the log-likelihood by more than this many nats; the
    #: refit could regain at most that much in every other run, so their fit
    #: is kept.  ``0`` refits every run the candidate touches at all.
    likelihood_ratio_run_tol: float = 1e-6
    #: A reduced refit stops as soon as the candidate is not significant
    #: (the drop verdict is final: the updates only raise the likelihood),
    #: and otherwise once the log-likelihood gains less than this many nats
    #: between two checks, ten updates apart.  ``0`` iterates every keep
    #: verdict to the solver's own tolerance.
    likelihood_ratio_ll_tol: float = 1e-3

    # ------------------------------------------------------------------ #
    # Run selection & logging                                              #
    # ------------------------------------------------------------------ #
    #: Exclude the Ribo-seq runs that fail the model quality gate: an
    #: implausible cleavage or coverage model, or too few counted alignments
    #: (see ``ribo_seq_run._assemble_run``).
    high_quality_runs_only: bool = option(False, scope=COLLECTION)
    #: Level of the ``price2`` logger (``"DEBUG"``, ``"INFO"``, ...).
    log_level: str = option("INFO", scope=RUNTIME)

    # ------------------------------------------------------------------ #
    # ORF candidate generation                                             #
    # ------------------------------------------------------------------ #
    #: Codons accepted as translation starts and stops when generating the
    #: ORF candidates.
    start_codons: tuple[str, ...] = ("ATG", "CTG", "GTG", "TTG")
    stop_codons: tuple[str, ...] = ("TAA", "TAG", "TGA")

    # ------------------------------------------------------------------ #
    # Multimapping EM                                                       #
    # ------------------------------------------------------------------ #
    #: Run the deconvolution inside an EM outer loop that fractionally
    #: re-assigns each multimapping read across the loci it maps to
    #: (E-step), with the per-locus optimisation as the M-step; collection
    #: then also records the per-alignment linkage.  When ``False``,
    #: alignments with ``NH > 1`` are discarded, at collection and again
    #: when reads are loaded, so a database collected with the EM behaves
    #: the same.  (Classic PRICE2 counted such reads at full weight in every
    #: locus they overlap.)
    multimap_em: bool = option(True, scope=COLLECTION)
    #: Backstop cap on the EM outer iterations; ``em_tol`` is meant to end
    #: the loop, which converges linearly in ~18-20 iterations on tested
    #: data (``docs/tuning.md``).
    em_max_iter: int = 30
    #: The loop stops once the L1 fraction of read mass reassigned between
    #: successive E-steps falls below this.
    em_tol: float = 1e-3
    #: IRLS-Huber reweight steps per light M-step; one keeps Huber and the
    #: EM in a single shared loop.
    em_huber_steps: int = 1
    #: Drop the ORF candidates the first light M-step finds inactive (below
    #: ``rgr_min_activity`` in every run) from all later EM iterations.  A
    #: pruned ORF cannot come back, which changes the call set slightly;
    #: kept switchable for A/B tests.
    em_prune_after_first_mstep: bool = True

    # ------------------------------------------------------------------ #
    # Resuming                                                             #
    # ------------------------------------------------------------------ #
    #: Continue an interrupted run: reuse ``w_dir`` and its database and let
    #: every stage pick up where it stopped (collection run by run and locus
    #: by locus, the EM at its last checkpoint, the final deconvolution at
    #: the loci not yet recorded as finished).  A stage whose options
    #: changed starts over; a changed option that decides the database's
    #: content stops the run instead (:mod:`price2.run_state`).  ``False``
    #: wipes both directories first.
    warm_start: bool = option(True, scope=RUNTIME)

    # ------------------------------------------------------------------ #
    # Export options                                                        #
    # ------------------------------------------------------------------ #
    #: Write ``dataset_models/`` with the cleavage and coverage model tables
    #: and plots.
    export_dataset_models: bool = True
    #: Write ``performance_measurements.tsv`` with per-locus timing and
    #: filtering statistics.
    export_performance_measurements: bool = False
    #: Write the tables after every filtering step, not only the final one.
    export_all_steps: bool = False
    #: Output formats.
    export_tsv: bool = True
    export_gtf: bool = False
    export_bed: bool = True
    #: Table contents: the ORFs; all regions (ORFs and NOISE, as
    #: ``*_regions`` files beside ``*_orfs``); and, in the GTF, the locus
    #: intervals and the transcript/NOISE entries.
    export_orfs: bool = True
    export_regions: bool = False
    export_loci: bool = False
    export_transcripts: bool = False

    # ------------------------------------------------------------------ #
    # Derived                                                              #
    # ------------------------------------------------------------------ #

    @property
    def layout(self) -> RunLayout:
        """The run's files and directories, derived from ``w_dir``/``o_dir``."""
        return RunLayout(self.w_dir, self.o_dir)

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def make_config(cls, **kwargs: object) -> Config:
        """Create a :class:`Config` from keyword arguments and/or a JSON file.

        When ``config`` is present in *kwargs* its value is treated as a
        path to a JSON configuration file.  Values supplied directly as
        keyword arguments take precedence over those in the file.
        Options that earlier releases accepted (see ``_OBSOLETE_FIELDS``)
        are dropped with a warning; any other unknown key is an error, so
        that a misspelled option cannot silently run on its default.

        Parameters
        ----------
        **kwargs : object
            Arbitrary keyword arguments.  The special key ``config`` may
            point to a JSON file path; all remaining keys must correspond
            to :class:`Config` field names.

        Returns
        -------
        Config
            A fully initialised :class:`Config` instance.

        Raises
        ------
        ValueError
            When a key is neither a :class:`Config` field nor an obsolete
            option.

        Examples
        --------
        >>> cfg = Config.make_config(config="run.json", lam=50)
        >>> cfg = Config.make_config(base_dir="/data/run1", lam=200)
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        config_path = kwargs.pop("config", None)
        if config_path is not None:
            with open(config_path, "r") as f:  # type: ignore[arg-type]
                json_dict = json.load(f)
            kwargs = {**json_dict, **kwargs}

        unknown = sorted(set(kwargs) - known_fields - _OBSOLETE_FIELDS)
        if unknown:
            raise ValueError(
                f"unknown configuration option(s): {', '.join(unknown)}"
            )
        obsolete = sorted(set(kwargs) & _OBSOLETE_FIELDS)
        if obsolete:
            logger.warning(
                "ignoring obsolete configuration option(s): %s",
                ", ".join(obsolete),
            )
        return cls(**{k: v for k, v in kwargs.items() if k in known_fields})

    def __post_init__(self) -> None:
        """Resolve the empty path fields under ``base_dir`` and validate the options.

        Raises
        ------
        ValueError
            For an option outside its allowed values or range.  Checking
            here, rather than where the option is first read, fails the run
            before any work is done instead of inside a worker hours later
            (an unknown ``inner_solver``, say, would otherwise silently
            select L-BFGS-B).
        """
        if self.o_dir == "":
            self.o_dir = os.path.join(self.base_dir, "o_dir")
        if self.w_dir == "":
            self.w_dir = os.path.join(self.base_dir, "w_dir")
        if self.bam_dir == "":
            self.bam_dir = os.path.join(self.base_dir, "bam_dir")
        self.start_codons = _codons("start_codons", self.start_codons)
        self.stop_codons = _codons("stop_codons", self.stop_codons)
        self._validate()

    def _validate(self) -> None:
        for name, allowed in _CHOICES.items():
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(
                    f"{name} must be one of {', '.join(map(repr, allowed))}, "
                    f"got {value!r}"
                )
        if not isinstance(logging.getLevelName(self.log_level), int):
            raise ValueError(f"log_level is not a logging level: {self.log_level!r}")
        for name in _POSITIVE:
            _check_number(name, getattr(self, name), strict=True)
        for name in _NON_NEGATIVE:
            _check_number(name, getattr(self, name), strict=False)
        for name in _INTEGER:
            if not isinstance(getattr(self, name), int):
                raise ValueError(
                    f"{name} must be an integer, got {getattr(self, name)!r}"
                )
        if self.bam_ids is not None and not all(
            isinstance(bam_id, str) and bam_id for bam_id in self.bam_ids
        ):
            raise ValueError(f"bam_ids must be a list of names, got {self.bam_ids!r}")


#: Options restricted to a fixed set of values.
_CHOICES: dict[str, tuple[str, ...]] = {
    "align_ends_type": ("local", "endtoend"),
    "distribution": ("poisson", "nb"),
    "inner_solver": ("mu", "lbfgs"),
    "mu_dtype": ("float32", "float64"),
    "mu_kernel": ("numba", "numpy"),
    "dispatch_order": ("largest", "database"),
}

#: Numeric options that must be greater than zero.
_POSITIVE: tuple[str, ...] = (
    "processes",
    "timeout",
    "pseudo_min",
    "min_well_fitting_reads_per_length",
    "deconvolution_filter_min_activity",
    "stop_factor_relative",
    "maxls",
    "rgr_min_activity",
    "min_activity_fraction",
    "irls_huber_c",
    "irls_huber_max_outer",
    "irls_huber_tol",
    "irls_active_patience",
    "nb_dispersion",
    "mu_inner_max_iter",
    "mu_inner_tol",
    "mu_broker_procs",
    "mu_broker_streams",
    "likelihood_ratio_alpha",
    "em_max_iter",
    "em_tol",
    "em_huber_steps",
)

#: Numeric options that must not be negative.
_NON_NEGATIVE: tuple[str, ...] = (
    "timeout_cap",
    "worker_max_tasks",
    "min_explained_reads_per_run",
    "ftol",
    "gtol",
    "lam",
    "mu_gpu_min_rows",
    "likelihood_ratio_run_tol",
    "likelihood_ratio_ll_tol",
)

#: Options that are counts: an integer, not merely a number.
_INTEGER: tuple[str, ...] = (
    "processes",
    "timeout_cap",
    "worker_max_tasks",
    "maxls",
    "irls_huber_max_outer",
    "irls_active_patience",
    "mu_gpu_min_rows",
    "mu_inner_max_iter",
    "mu_broker_procs",
    "mu_broker_streams",
    "em_max_iter",
    "em_huber_steps",
)

_NUCLEOTIDES = frozenset("ACGT")


def _check_number(name: str, value: object, *, strict: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if value < 0 or (strict and value == 0):
        bound = "greater than zero" if strict else "zero or more"
        raise ValueError(f"{name} must be {bound}, got {value!r}")


def _codons(name: str, codons: object) -> tuple[str, ...]:
    """Return *codons* as a tuple of upper-case triplets, or raise."""
    if isinstance(codons, str) or not isinstance(codons, (list, tuple)) or not codons:
        raise ValueError(f"{name} must be a non-empty list of codons, got {codons!r}")
    result = tuple(str(codon).upper() for codon in codons)
    bad = [codon for codon in result if len(codon) != 3 or set(codon) - _NUCLEOTIDES]
    if bad:
        raise ValueError(f"{name} holds invalid codon(s): {', '.join(bad)}")
    return result
