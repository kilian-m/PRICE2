from __future__ import annotations

from dataclasses import dataclass
import json


@dataclass
class Config:
    """Configuration for a PRICE2 run.

    All parameters are loaded from a JSON file via :meth:`make_config`.
    Paths that are left as empty strings are derived automatically from
    ``base_dir`` in :meth:`__post_init__`.

    The directory layout assumed by default::

        base_dir/
        ├── o_dir/      # output
        ├── w_dir/      # working / SQLite database
        ├── bam_dir/    # mapped Ribo-seq BAM files
        └── logs/       # log files

    Parameters
    ----------
    base_dir : str
        Root directory for the run.  All relative paths default to
        subdirectories of this directory (required).
    o_dir : str
        Output directory.  Defaults to ``<base_dir>/o_dir``.
    w_dir : str
        Working directory (holds ``price.db`` SQLite database).
        Defaults to ``<base_dir>/w_dir``.
    l_file : str
        Log file path.  Defaults to ``<base_dir>/logs/process.log``.
    gtf_path : str
        Path to the reference annotation GTF file.
    fasta_path : str
        Path to the reference genome FASTA file.
    bam_dir : str
        Directory containing mapped Ribo-seq BAM files.
        Defaults to ``<base_dir>/bam_dir``.
    bam_ids : list[str] or None
        Optional subset of BAM file identifiers to process.  When
        ``None`` all BAM files found in ``bam_dir`` are used.
    processes : int
        Number of parallel worker processes.
    worker_max_tasks : int
        Number of loci a worker process handles before being replaced
        (``0`` means never replaced).
    timeout : int
        Per-locus timeout in seconds.
    memory_limit_gb : int
        Per-worker memory limit in gigabytes.
    pseudo_min : float
        Lower bound applied to activity estimates during optimisation to
        avoid ``log(0)`` numerical instability.
    min_explained_reads_per_run : int
        Minimum number of reads that must be explained by a single
        transcript per run for that transcript to be retained.
    coverage_filter : bool
        Enable the coverage filter, which removes ORF candidates with
        insufficient average well-fitting read coverage.
    min_well_fitting_reads_per_length : float
        Coverage-filter threshold: minimum well-fitting reads per
        nucleotide length (taken as the maximum across all runs).
    deconvolution_filter : bool
        Enable the pre-deconvolution filter that removes ORF candidates
        with low estimated activity within stop-codon groups.
    deconvolution_filter_min_activity : float
        Minimum activity threshold used by the deconvolution filter.
    stop_factor_relative : float
        Convergence criterion for the main optimisation: stop when the
        relative change in every activity falls below this value.
    ftol : float
        ``ftol`` tolerance passed to ``scipy.optimize.minimize``
        (L-BFGS-B).  Set to 0 to rely solely on ``stop_factor_relative``.
    gtol : float
        ``gtol`` tolerance passed to ``scipy.optimize.minimize``
        (L-BFGS-B).  Set to 0 to rely solely on ``stop_factor_relative``.
    maxls : int
        Maximum number of line-search steps in all optimisations.
    lam : float
        Group-LASSO regularisation strength λ.
    rgr_min_activity : float
        Minimum activity (in at least one run) for an RGR to be retained.
    min_activity_fraction : float
        Minimum activity expressed as a fraction of the canonical ORF
        activity for an ORF to be retained.
    likelihood_ratio_filter : bool
        Apply a likelihood-ratio test as the final filtering step.
    likelihood_ratio_alpha : float
        Significance threshold for the likelihood-ratio test.
    high_quality_runs_only : bool
        When ``True``, exclude Ribo-seq runs whose cleavage model failed
        quality checks (peak not at position 12 or peak probability < 0.3).
    save_memory : bool
        When ``True``, locus objects are not stored in the master process
        to reduce peak memory usage.
    log_level : str
        Logging verbosity level for the ``price2`` logger.  Accepts
        standard Python level names (``"DEBUG"``, ``"INFO"``,
        ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``).
    start_codons : tuple[str, ...]
        Codons accepted as translation start sites for ORF candidate
        generation.
    stop_codons : tuple[str, ...]
        Codons accepted as translation stop sites for ORF candidate
        generation.
    warm_start : bool
        When ``True``, skip data collection and reuse the existing working
        directory (``w_dir``) and its SQLite database.  The pipeline
        resumes directly at ORF deconvolution.  Requires a prior run to
        have completed the data-collection phase (default ``False``).
    export_dataset_models : bool
        Write the ``dataset_models/`` directory with cleavage and coverage
        model summaries and plots (default ``True``).
    export_performance_measurements : bool
        Write ``performance_measurements.tsv`` with per-locus timing and
        filtering statistics (default ``False``).
    export_all_steps : bool
        When ``True``, write output files for every filtering step.
        When ``False`` (default) only the final results are exported.
    export_tsv : bool
        Write TSV output files (default ``True``).
    export_gtf : bool
        Write GTF output files (default ``False``).
    export_bed : bool
        Write BED12 output files (default ``True``).
    export_orfs : bool
        Include ORF entries in output files (default ``True``).
    export_regions : bool
        Include all regions (ORFs + NOISE) in output files (default
        ``False``).  Produces ``*_regions.*`` files alongside
        ``*_orfs.*``.
    export_loci : bool
        Include locus interval entries in GTF output (default ``False``).
    export_transcripts : bool
        Include transcript/NOISE entries in GTF output (default
        ``False``).
    """

    # ------------------------------------------------------------------ #
    # Required                                                             #
    # ------------------------------------------------------------------ #
    base_dir: str

    # ------------------------------------------------------------------ #
    # Paths (derived from base_dir when left empty)                       #
    # ------------------------------------------------------------------ #
    o_dir: str = ""
    w_dir: str = ""
    l_file: str = ""
    gtf_path: str = ""
    fasta_path: str = ""
    bam_dir: str = ""
    bam_ids: list[str] | None = None

    # ------------------------------------------------------------------ #
    # Parallelism & runtime                                                #
    # ------------------------------------------------------------------ #
    processes: int = 80
    timeout: int = 60 * 30
    memory_limit_gb: int = 5
    pseudo_min: float = 1e-14
    #: Loci a worker handles before it is replaced (``0`` = never replaced).
    #: A fresh process costs ~2 CPU-seconds before it does any useful work:
    #: numba cache load, imports, and a cold interpreter/allocator on its first
    #: locus.  At ``1`` that was charged to every one of the ~39K loci in every
    #: EM iteration.  Reuse amortises it (measured ``0.70 + 1.97/worker_max_tasks``
    #: CPU-seconds per locus) while still recycling processes often enough to
    #: bound RSS growth from the occasional very large locus.
    worker_max_tasks: int = 50

    # ------------------------------------------------------------------ #
    # Transcript pre-filtering                                             #
    # ------------------------------------------------------------------ #
    min_explained_reads_per_run: int = 5

    # ------------------------------------------------------------------ #
    # Coverage filter                                                      #
    # ------------------------------------------------------------------ #
    coverage_filter: bool = True
    min_well_fitting_reads_per_length: float = 0.01

    # ------------------------------------------------------------------ #
    # Deconvolution filter                                                 #
    # ------------------------------------------------------------------ #
    deconvolution_filter: bool = True
    deconvolution_filter_min_activity: float = 0.01

    # ------------------------------------------------------------------ #
    # Main optimisation                                                    #
    # ------------------------------------------------------------------ #
    stop_factor_relative: float = 0.01
    ftol: float = 0
    gtol: float = 0
    maxls: int = 200
    #: Group-LASSO penalty λ. Recalibrated from 100 -> 10 after the switch to
    #: ``inner_solver="mu"``: the tighter solver applies the penalty cleanly, and
    #: an AIC/BIC scan (playground/lambda_aic) shows λ=100 over-penalises
    #: (ΔAIC≈5e3 on tiny chr22; BIC-optimal λ=10, AIC-optimal λ=3). λ=10 is the
    #: conservative (BIC) choice; confirm on a production-scale dataset.
    lam: float = 10

    # ------------------------------------------------------------------ #
    # RGR activity thresholds                                              #
    # ------------------------------------------------------------------ #
    rgr_min_activity: float = 0.01
    min_activity_fraction: float = 0.1

    # ------------------------------------------------------------------ #
    # IRLS-Huber                                                           #
    # ------------------------------------------------------------------ #
    irls_huber_c: float = (
        3.0  # low values -> more robust | high values -> more like MLE
    )
    irls_huber_max_outer: int = 10
    irls_huber_tol: float = 1e-4
    #: Experimental alternative stopping rule for the IRLS-Huber outer loop.
    #: The default ``irls_huber_tol`` measures the L2 change of the whole
    #: weight vector, which shrinks only geometrically (group-LASSO drags many
    #: coordinates slowly toward ``pseudo_min``) and so rarely fires before
    #: ``irls_huber_max_outer`` on read-dense loci. When ``True`` the loop
    #: instead stops once the set of ORFs above
    #: ``deconvolution_filter_min_activity`` (i.e. the ones that survive
    #: filtering) is unchanged for ``irls_active_patience`` consecutive outer
    #: iterations — the reported call set converges far earlier than the raw
    #: parameter norm.
    irls_stop_on_active_set: bool = True
    irls_active_patience: int = 3

    # ------------------------------------------------------------------ #
    # Count distribution for the deconvolution likelihood                  #
    # ------------------------------------------------------------------ #
    #: Count model assumed for the read counts in every deconvolution solve.
    #: ``"poisson"`` (default) is the classic PRICE2 model (variance = mean).
    #: ``"nb"`` uses a negative-binomial likelihood (variance = μ + μ²/θ) to
    #: absorb the overdispersion typical of Ribo-seq counts; ``θ`` is the fixed
    #: global dispersion set by ``nb_dispersion``. As ``θ → ∞`` the NB model
    #: collapses back to the Poisson, so ``"nb"`` with a large ``nb_dispersion``
    #: reproduces the Poisson results. The choice threads through every solve
    #: site (deconvolution filter, group-LASSO IRLS-Huber deconvolution, the
    #: weighted likelihood-ratio filter and final activity estimation) and both
    #: the CPU and GPU multiplicative-update paths.
    distribution: str = "poisson"
    #: Negative-binomial dispersion ``θ`` (a.k.a. the "size"/``r`` parameter),
    #: used only when ``distribution == "nb"``. Smaller values mean stronger
    #: overdispersion; larger values approach the Poisson limit. Ignored for the
    #: Poisson model.
    nb_dispersion: float = 10.0

    # ------------------------------------------------------------------ #
    # Inner solver for the group-LASSO Poisson deconvolution              #
    # ------------------------------------------------------------------ #
    #: ``"mu"`` (default) uses multiplicative (weighted Richardson-Lucy +
    #: group-LASSO) updates at every solve site; they converge to the true
    #: optimum that scipy L-BFGS-B stalls short of. ``"lbfgs"`` selects the legacy
    #: L-BFGS-B path. Two consequences of the default:
    #:  (1) it CHANGES the call set vs the legacy loose L-BFGS-B — the converged
    #:      solve is sparser (~19% fewer ORFs on Yewdell). ``lam`` was
    #:      recalibrated for it (100 -> 10) via an AIC/BIC scan; see
    #:      playground/lambda_aic.
    #:  (2) on CPU it is slower than L-BFGS-B (tight convergence costs iterations).
    #: See playground/deconvolution_performance.
    inner_solver: str = "mu"
    #
    # GPU offload of the multiplicative updates (``mu_gpu`` / ``mu_broker``)
    # -------------------------------------------------------------------
    # BOTH ARE OFF BY DEFAULT: in the tests run so far the GPU did not speed the
    # pipeline up meaningfully. The individual deconvolution solves are small
    # (sparse mat-vecs over a few 10k rows), so kernel-launch and host<->device
    # transfer overhead eats most of the per-solve gain, and the CPU worker pool
    # already parallelises across loci — the end-to-end wall time barely moves,
    # while the GPU paths add CUDA contexts, VRAM pressure and (for the broker)
    # a shared-memory IPC layer. Turn them on only if you have re-measured on
    # your own data and see a real win.
    #
    # Requirements when you DO enable them: an NVIDIA GPU + driver and PyTorch
    # built against CUDA (developed/tested with torch 2.5.1+cu121). torch is NOT
    # a declared dependency — it is absent from pyproject.toml and price2.yml,
    # so install it separately, e.g.
    #     pip install torch --index-url https://download.pytorch.org/whl/cu121
    # Both paths degrade gracefully: if torch or CUDA is unavailable the solves
    # silently fall back to the NumPy CPU multiplicative updates.
    #
    #: Use the GPU (PyTorch + CUDA) multiplicative-update path when available
    #: and the system is at least ``mu_gpu_min_rows`` rows; else run on CPU.
    #: Gives each worker process its own CUDA context (VRAM scales with
    #: ``processes``); see ``mu_broker`` for the shared-context variant.
    mu_gpu: bool = False
    #: Only offload to the GPU above this row count (transfer overhead makes
    #: small systems faster on the CPU).
    mu_gpu_min_rows: int = 50_000
    #: GPU dtype for the multiplicative updates (``"float32"`` or ``"float64"``).
    mu_dtype: str = "float32"
    #: Multiplicative-update inner-loop cap and relative-change tolerance.
    mu_inner_max_iter: int = 3000
    mu_inner_tol: float = 1e-5
    #: Route GPU deconvolution through a single-context broker process instead of
    #: giving each worker its own CUDA context. Requires ``inner_solver="mu"``.
    #: One broker holds one context and serves the whole worker pool over shared
    #: memory, so device VRAM does not scale with ``processes`` (see
    #: playground/deconvolution_performance/gpu_broker). Falls back to the
    #: configured per-worker path (CPU MU, or per-worker GPU when ``mu_gpu``)
    #: if the broker cannot start (no PyTorch/CUDA).
    #: OFF by default — see the GPU-offload note above ``mu_gpu``: no meaningful
    #: end-to-end speedup in the small tests, and it needs a CUDA PyTorch.
    mu_broker: bool = False
    #: Number of broker PROCESSES (independent CUDA contexts / GILs). A single
    #: process is GIL-bound and cannot saturate the GPU or feed a large worker
    #: pool; a few processes fix that while keeping the context count bounded.
    mu_broker_procs: int = 4
    #: CUDA stream-threads per broker process (intra-process overlap; limited by
    #: that process's GIL, so scale mu_broker_procs first).
    mu_broker_streams: int = 2

    # ------------------------------------------------------------------ #
    # Likelihood-ratio filter                                              #
    # ------------------------------------------------------------------ #
    likelihood_ratio_filter: bool = True
    likelihood_ratio_alpha: float = 1e-10
    high_quality_runs_only: bool = False
    save_memory: bool = (
        True  # Do not disable this in normal mode, it can clog the pipe shifting results from workers to the master process and cause deadlocks. only use with few workers
    )
    log_level: str = "INFO"

    # ------------------------------------------------------------------ #
    # ORF candidate generation                                             #
    # ------------------------------------------------------------------ #
    start_codons: tuple[str, ...] = ("ATG", "CTG", "GTG", "TTG")
    stop_codons: tuple[str, ...] = ("TAA", "TAG", "TGA")

    # ------------------------------------------------------------------ #
    # Multimapping EM                                                       #
    # ------------------------------------------------------------------ #
    # When ``True`` (default) the deconvolution runs inside an EM outer loop
    # that fractionally re-assigns each multimapping read across the loci it
    # maps to (E-step), with the existing per-locus optimisation as the
    # M-step.  When ``False`` behaviour is identical to classic PRICE2
    # (every read counted at full weight at every locus it overlaps).  Note:
    # enabling this makes read collection record per-alignment multimap
    # linkage (extra time/disk) and adds the EM outer iterations below.
    multimap_em: bool = True
    #: Safety cap on EM outer iterations (light M-step + global E-step). The
    #: EM converges only *linearly* (the L1 read-mass reassigned per E-step
    #: shrinks by a roughly constant factor ~0.8 each iteration), so reaching
    #: ``em_tol`` takes ~18-20 iterations on tested data. The loop is meant to
    #: stop on ``em_tol``; this cap is a backstop, not the normal exit — set it
    #: high enough that the tolerance governs. (A cap of 10 truncated the loop
    #: at ~5x em_tol, i.e. before it converged; see
    #: playground/em_stopping_criterion.)
    em_max_iter: int = 30
    #: Convergence tolerance on the L1 fraction of read mass reassigned between
    #: successive E-steps; the loop stops once below this. This — not
    #: ``em_max_iter`` — should be what ends the loop in a normal run.
    em_tol: float = 1e-3
    #: Number of interleaved IRLS-Huber reweight steps per light M-step.
    #: One keeps Huber and EM in a single shared loop (as intended); the
    #: robust weights refine together with the fractional assignments.
    em_huber_steps: int = 1
    #: Cache the weight-independent read → design-matrix-row routing and the
    #: geometry of ``X`` beside each prepared locus.  Only the fractional
    #: weights change between light M-steps, so an iteration then reduces to a
    #: weighted ``bincount`` plus a vectorised ``X.data`` rebuild, instead of
    #: re-deriving ``get_rgr_frame_covpos`` for every read and walking every
    #: design-matrix cell in Python.  The cache holds no locus objects, so an
    #: intermediate pass loads it alone rather than unpickling the locus.
    #: Costs ~65% more space per prepared locus (~95 KB on top of ~148 KB).
    eg_cache: bool = True
    #: Background rate for a multimapping read's alignments that fall
    #: outside any annotated locus.  ``0.0`` (default) means "loci-only":
    #: weight is normalised across the annotated loci a read hits and
    #: intergenic alignments are ignored.  A positive value discounts
    #: repeat-heavy reads (currently reserved; intergenic alignments are
    #: not recorded, so only loci-only normalisation is active).
    multimap_background: float = 0.0

    # ------------------------------------------------------------------ #
    # Export options                                                        #
    # ------------------------------------------------------------------ #
    warm_start: bool = False

    # ------------------------------------------------------------------ #
    # Export options                                                        #
    # ------------------------------------------------------------------ #
    export_dataset_models: bool = True
    export_performance_measurements: bool = False
    export_all_steps: bool = False
    export_tsv: bool = True
    export_gtf: bool = False
    export_bed: bool = True
    export_orfs: bool = True
    export_regions: bool = False
    export_loci: bool = False
    export_transcripts: bool = False

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def make_config(cls, **kwargs: object) -> Config:
        """Create a :class:`Config` from keyword arguments and/or a JSON file.

        When ``config`` is present in *kwargs* its value is treated as a
        path to a JSON configuration file.  Values supplied directly as
        keyword arguments take precedence over those in the file.
        Unknown keys are silently ignored.

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

        Examples
        --------
        >>> cfg = Config.make_config(config="run.json", lam=50)
        >>> cfg = Config.make_config(base_dir="/data/run1", lam=200)
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        if "config" in kwargs:
            with open(kwargs["config"], "r") as f:  # type: ignore[arg-type]
                json_dict = json.load(f)
            kwargs = {**json_dict, **kwargs}

        filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_fields}
        return cls(**filtered_kwargs)

    def __post_init__(self) -> None:
        """Resolve empty path fields to their default locations under ``base_dir``."""
        if self.o_dir == "":
            self.o_dir = f"{self.base_dir}/o_dir"
        if self.w_dir == "":
            self.w_dir = f"{self.base_dir}/w_dir"
        if self.bam_dir == "":
            self.bam_dir = f"{self.base_dir}/bam_dir"
        if self.l_file == "":
            self.l_file = f"{self.base_dir}/logs/process.log"
