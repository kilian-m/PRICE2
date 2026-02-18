import HTSeq
import pysam
import os

from multiprocessing import Pool

from price2.cleavage_model import CleavageModel, CleavageEstimator
from price2.reference_annotation import ReferenceAnnotation
from price2.coverage_model import CoverageModel

# assumes sorted and indexed bam file


class RiboSeqRun:
    id: str
    bam_reader: HTSeq.BAM_Reader
    cleavage_model: CleavageModel
    read_count: int

    def __init__(
        self,
        id: str,
        directory: str,
        cleavage_model: CleavageModel,
        coverage_model: CoverageModel,
        read_count: int = 0,
    ) -> None:
        self.id = id
        # bam_file_path = f'{directory}/{id}.bam'
        self.cleavage_model = cleavage_model
        self.coverage_model = coverage_model
        self.read_count = read_count

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: "RiboSeqRun") -> bool:
        if not isinstance(other, type(self)):
            False
        return self.id == other.id


def ribo_seq_runs_from_bams(
    bam_dir: str,
    bam_ids: set[str],
    wdir: str,
    ref_annotation: ReferenceAnnotation,
    processes: int = 32,
) -> list[RiboSeqRun]:

    os.makedirs(f"{wdir}/sample_bam", exist_ok=True)
    bam_files = [f"{bam_id}.bam" for bam_id in bam_ids]

    if len(bam_ids) > 0:
        with Pool(processes) as pool:
            ribo_seq_runs = pool.starmap(
                ribo_seq_run_from_bam,
                [(bam_dir, bam_file, wdir, ref_annotation) for bam_file in bam_files],
            )
    else:
        ribo_seq_runs = []

    os.rmdir(f"{wdir}/sample_bam")

    return ribo_seq_runs


def ribo_seq_run_from_bam(
    bam_dir: str,
    bam_file: str,
    wdir: str,
    ref_annotation: ReferenceAnnotation,
) -> RiboSeqRun:
    id = bam_file.split(".")[0]
    bam_file_path = f"{bam_dir}/{bam_file}"

    # sample BAM file
    read_count = pysam.AlignmentFile(bam_file_path, "rb").count()
    sample_bam_file = f"{wdir}/sample_bam/{bam_file}"
    open(sample_bam_file, "w").close()
    fraction_of_reads = min(10_000_000 / read_count, 0.99)
    pysam.view(
        "-s",
        str(fraction_of_reads),
        "-o",
        sample_bam_file,
        bam_file_path,
        save_stdout=sample_bam_file,
    )

    # estimate cleavage model
    ce = CleavageEstimator()
    ce.collect_data(ref_annotation, sample_bam_file)
    ce.correct_table()
    cleavage_model = ce.run()

    # estimate coverage model
    coverageModel = CoverageModel(
        ref_annotation,
        sample_bam_file,
        cleavage_model,
    )

    # remove sample BAM file
    os.remove(sample_bam_file)

    return RiboSeqRun(id, bam_dir, cleavage_model, coverageModel, read_count=read_count)
