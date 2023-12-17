import HTSeq
import pysam
import os

from .cleavage_model import CleavageModel, CleavageEstimator
from .reference_annotation import ReferenceAnnotation

# assumes sorted and indexed bam file

class RiboSeqRun:
    id: str
    bam_reader: HTSeq.BAM_Reader
    cleavage_model: CleavageModel
    read_count: int


    def __init__(self, id: str, directory: str, ra: ReferenceAnnotation, cleavage_model: CleavageModel=None) -> None:
        self.id = id
        bam_file_path = f'{directory}/{id}.bam'
        print(bam_file_path)
        self.bam_reader = HTSeq.BAM_Reader(bam_file_path)

        self.read_count = pysam.AlignmentFile(bam_file_path, 'rb').count()

        if cleavage_model:
            self.cleavage_model = cleavage_model
            return

        # make sample bam file with 100_000 reads
        sample_bam_file = f'{directory}/{id}_sample.bam'
        open(sample_bam_file, 'w').close()
        fraction_of_reads = 100_000 / self.read_count
        pysam.view('-s', str(fraction_of_reads), '-o', sample_bam_file, bam_file_path, save_stdout=sample_bam_file)

        ce = CleavageEstimator()

        ce.collect_data(ra, HTSeq.BAM_Reader(sample_bam_file))
        ce.correct_table()
        self.cleavage_model = ce.run()

        os.remove(sample_bam_file)

    
    def __hash__(self) -> int:
        return hash(self.id)
        

        
