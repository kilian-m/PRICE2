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
    timeout : int
        Per-locus timeout in seconds.
    memory_limit_gb : int
        Per-worker memory limit in gigabytes.
    warm_start : bool
        When ``True``, reuse cached intermediate results stored in the
        working directory.  When ``False``, recompute from scratch.
    verbose_gtf : bool
        Write GTF files of retained ORFs after each filtering step.
    pseudo_min : float
        Lower bound applied to activity estimates during optimisation to
        avoid ``log(0)`` numerical instability.
    loci_subset : int
        Limit the number of loci to process (0 = no limit).  Intended
        for debugging only.
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
    rgrs_to_remove_fraction : float
        Fraction of RGRs that must have low activity before iterative
        removal is triggered during main deconvolution.
    rgr_min_activity : float
        Minimum activity (in at least one run) for an RGR to be retained.
    min_activity_fraction : float
        Minimum activity expressed as a fraction of the canonical ORF
        activity for an ORF to be retained.
    likelihood_ratio_filter : bool
        Apply a likelihood-ratio test as the final filtering step.
    likelihood_ratio_alpha : float
        Significance threshold for the likelihood-ratio test.
    save_memory : bool
        When ``True``, locus objects are not stored in the master process
        to reduce peak memory usage.
    start_codons : tuple[str, ...]
        Codons accepted as translation start sites for ORF candidate
        generation.
    stop_codons : tuple[str, ...]
        Codons accepted as translation stop sites for ORF candidate
        generation.
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
    warm_start: bool = True
    verbose_gtf: bool = True
    pseudo_min: float = 1e-14
    loci_subset: int = 0

    # ------------------------------------------------------------------ #
    # Transcript pre-filtering                                             #
    # ------------------------------------------------------------------ #
    min_explained_reads_per_run: int = 5

    # ------------------------------------------------------------------ #
    # Coverage filter                                                      #
    # ------------------------------------------------------------------ #
    coverage_filter: bool = True
    min_well_fitting_reads_per_length: float = 0.1

    # ------------------------------------------------------------------ #
    # Deconvolution filter                                                 #
    # ------------------------------------------------------------------ #
    deconvolution_filter: bool = True
    deconvolution_filter_min_activity: float = 0.1

    # ------------------------------------------------------------------ #
    # Main optimisation                                                    #
    # ------------------------------------------------------------------ #
    stop_factor_relative: float = 0.01
    ftol: float = 0
    gtol: float = 0
    maxls: int = 200
    lam: float = 100

    # ------------------------------------------------------------------ #
    # RGR activity thresholds                                              #
    # ------------------------------------------------------------------ #
    rgrs_to_remove_fraction: float = 0.5
    rgr_min_activity: float = 0.01
    min_activity_fraction: float = 0.1

    # ------------------------------------------------------------------ #
    # Likelihood-ratio filter                                              #
    # ------------------------------------------------------------------ #
    likelihood_ratio_filter: bool = True
    likelihood_ratio_alpha: float = 1e-10
    save_memory: bool = True

    # ------------------------------------------------------------------ #
    # ORF candidate generation                                             #
    # ------------------------------------------------------------------ #
    start_codons: tuple[str, ...] = ("ATG", "CTG", "GTG", "ACG")
    stop_codons: tuple[str, ...] = ("TAA", "TAG", "TGA")

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
