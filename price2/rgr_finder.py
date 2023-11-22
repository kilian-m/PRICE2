import HTSeq

from .genomic_region import GenomicRegion
from .genomic_features import Transcript, ReadGeneratingRegion

class RGRFinder:
    transcripts: dict[str:Transcript]
    rgr_set: set[ReadGeneratingRegion]
    rgr_intervals: HTSeq.GenomicArrayOfSets
    region_intervals: HTSeq.GenomicArrayOfSets
        
    def __init__(self, transcripts: dict[str:Transcript], genome: dict) -> None:
        self.rgr_set = set()
        self.transcripts = transcripts

        self.rgr_intervals = HTSeq.GenomicArrayOfSets("auto", stranded=True, storage="step")
        self.region_intervals = HTSeq.GenomicArrayOfSets("auto", stranded=True, storage="step")
        self.compatibility_groups = dict()

        for transcript in self.transcripts.values():
            noise = ReadGeneratingRegion(
                'NOISE',
                transcript,
                (0, len(transcript.exons)),
            )
            if not noise in self.rgr_set:
                self.rgr_set.add(noise)
                for iv in noise.genomic_region.intervals:
                    self.rgr_intervals[iv] += noise
            
            seq = transcript.exons.get_sequence(genome)
            for orf_iv_on_transcript in RGRFinder.find_orfs(seq):
                orf = ReadGeneratingRegion(
                    'ORF',
                    transcript,
                    orf_iv_on_transcript,
                )
                if not orf in self.rgr_set:
                    self.rgr_set.add(orf)
                    for iv in orf.genomic_region.intervals:
                        self.rgr_intervals[iv] += orf


                    
    # find ORFs based on transcript sequence
    # return list of intervals of all ORFs with min length
    def find_orfs(
            seq: str,
            min_length: int=0, 
            start_codons: set=set(['ATG',]),# 'CTG', 'GTG', 'ACG']),
            stop_codons: set=set(['TAA', 'TAG', 'TGA']) ) -> list[tuple[int, int]]:

        orf_iv_on_transcript = []
        for i in range(3):
            starts = []
            for j in range(i, len(seq), 3):
                if seq[j:j+3] in start_codons:
                    starts.append(j)
                if seq[j:j+3] in stop_codons:
                    for start in starts:
                        if j - start >= min_length:
                            orf_iv_on_transcript.append((start,j+3))
                    starts = []
        return orf_iv_on_transcript

    
    def collect_rgrs(self, genomic_region: GenomicRegion) -> set[ReadGeneratingRegion]:
        rgrs = set()
        for query_iv in genomic_region.intervals:
            for subject_iv, rgr_set in self.rgr_intervals[query_iv].steps():
                rgrs |= rgr_set
        return rgrs

