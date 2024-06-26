import HTSeq
import pysam
import os

from multiprocessing import Pool

from .cleavage_model import CleavageModel, CleavageEstimator
from .reference_annotation import ReferenceAnnotation

# assumes sorted and indexed bam file

class RiboSeqRun:
    id: str
    bam_reader: HTSeq.BAM_Reader
    cleavage_model: CleavageModel
    read_count: int


    def __init__(self, id: str, directory: str, cleavage_model: CleavageModel) -> None:
        self.id = id
        bam_file_path = f'{directory}/{id}.bam'
        self.cleavage_model = cleavage_model
        self.read_count = 0

    
    def __hash__(self) -> int:
        return hash(self.id)
        

def ribo_seq_runs_from_bams(bam_dir: str,
                            wdir: str,
                            ref_annotation: ReferenceAnnotation,
                            processes: int = 32,
                            ) -> list[RiboSeqRun]:
    
    os.makedirs(f'{wdir}/sample_bam', exist_ok=True)
    bam_files = [f for f in os.listdir(bam_dir) if f.endswith('.bam')]

    with Pool(processes) as pool:
        ribo_seq_runs = pool.starmap(ribo_seq_run_from_bam, [(bam_dir, bam_file, wdir, ref_annotation) for bam_file in bam_files])

    os.rmdir(f'{wdir}/sample_bam')

    return ribo_seq_runs


def ribo_seq_run_from_bam(bam_dir: str,
                          bam_file: str,
                          wdir: str,
                          ref_annotation: ReferenceAnnotation,
                          ) -> RiboSeqRun:
    id = bam_file.split('.')[0]
    bam_file_path = f'{bam_dir}/{bam_file}'
    read_count = pysam.AlignmentFile(bam_file_path, 'rb').count()
    sample_bam_file = f'{wdir}/sample_bam/{bam_file}'
    open(sample_bam_file, 'w').close()
    fraction_of_reads = 100_000 / read_count
    pysam.view('-s', str(fraction_of_reads), '-o', sample_bam_file, bam_file_path, save_stdout=sample_bam_file)

    ce = CleavageEstimator()

    ce.collect_data(ref_annotation, HTSeq.BAM_Reader(sample_bam_file))
    ce.correct_table()
    
    cleavage_model = ce.run()

    os.remove(sample_bam_file)

    return RiboSeqRun(id, bam_dir, cleavage_model)