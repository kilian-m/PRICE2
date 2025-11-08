from dataclasses import dataclass
import json


@dataclass
class Config:
    # REQUIRED
    # base directory
    # generally the folder structure is:
    # base_dir/
    #   o_dir/
    #   w_dir/
    base_dir: str

    # output directory
    o_dir: str = ""
    # working directory
    w_dir: str = ""
    # log file
    l_file: str = ""
    # reference annotation gtf path
    gtf_path: str = ""
    # reference genome fasta path
    fasta_path: str = ""
    # bam directory with mapped ribo-seq runs
    bam_dir: str = ""

    # number of parallel processes
    processes: int = 80
    # timeout for each locus in seconds
    timeout: int = 60 * 30
    # recompute everything if false
    warm_start: bool = True
    # if true, write gtf file of retained ORFs for each intermediate step
    verbose_gtf: bool = True
    # lower bounds for activity estimates in optimization processes because of numerical instability at 0
    pseudo_min: float = 1e-14
    # number of loci to process, for debugging, set to 0 for all, REMOVE
    loci_subset: int = 0  # 0 for all / temporary

    # ORF candidate filtering based on well fitting reads
    # coverage filter removes ORFs with low average coverage of well fitting reads (consider max across all runs)
    coverage_filter: bool = True
    # threshold for coverage filter
    min_well_fitting_reads_per_length: float = 0.1  # 1

    # only keep the set of transcripts that are necessary to explain the most observed reads
    # number of reads that need to be explained by a single transcript per run to be retained
    min_explained_reads_per_run: int = 5

    # consider ORFs that are compatible with the same exons and ending in the same stop codon
    # deconvolute on these stop groups based on well fitting reads (within one each run)
    # remove ORFs with low estimated activity in all runs
    deconvolution_filter: bool = True
    # threshold for deconvolution filter
    deconvolution_filter_min_activity: float = 0.1

    # main ORF deconvolution is done by maximizing a regularized poisson likelihood with regards to ORF activities
    # stop iteration if the relative change in each activity is less than this value (or the activity is small)
    stop_factor_relative: float = 0.01
    # ftol and gtol for scipy.optimize.minimize as alternatives to stop_factor_relative
    ftol: float = 0  # 1e-10  # 1e-6  #
    gtol: float = 0  # 1e-10  # 1e-6  #
    # maximal steps in line search in all optimizations
    maxls: int = 200
    # regularization strength lamba, 100 seems to work well in simulated data
    lam: float = 100

    # in main deconvolution rgrs with low activity are removed
    # fraction of rgrs that need to have low activity to start this process
    rgrs_to_remove_fraction: float = 0.5
    # minimal activity (in at least one run) for rgrs to be retained, used at various places
    rgr_min_activity: float = 0.01
    # minimal fraction of the canonical ORF activity to be retained
    min_activity_fraction: float = 0.1  # 0.1

    # apply likelihood ratio
    likelihood_ratio_filter: bool = True
    # threshold for likelihood ratio
    likelihood_ratio_alpha: float = 1e-10
    # loci information is not saved in master processes to save memory
    safe_memory: bool = False

    @classmethod
    def make_config(cls, **kwargs):
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        if "config" in kwargs:
            with open(kwargs["config"], "r") as f:
                json_dict = json.load(f)
            kwargs = {**json_dict, **kwargs}

        filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_fields}

        return cls(**filtered_kwargs)

    def __post_init__(self):
        if self.o_dir == "":
            self.o_dir = f"{self.base_dir}/o_dir"
        if self.w_dir == "":
            self.w_dir = f"{self.base_dir}/w_dir"
        if self.bam_dir == "":
            self.bam_dir = f"{self.base_dir}/bam_dir"
        if self.l_file == "":
            self.l_file = f"{self.base_dir}/logs/process.log"
