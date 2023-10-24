import HTSeq

from .genomic_region import GenomicRegion
from .genomic_features import Transcript, ORF, Locus, CompatibilityGroup

class ORFFinder:
    transcripts: dict[str:Transcript]
    orf_set: set[ORF]
    orf_intervals: HTSeq.GenomicArrayOfSets
    region_intervals: HTSeq.GenomicArrayOfSets
    loci_set: set[Locus]
    loci_intervals: HTSeq.GenomicArray
    compatibility_groups: dict[frozenset[ORF]: CompatibilityGroup]
        
    def __init__(self, transcripts: dict[str:Transcript], genome: dict) -> None:
        self.orf_set = set()
        self.transcripts = transcripts

        self.orf_intervals = HTSeq.GenomicArrayOfSets("auto", stranded=True, storage="step")
        self.region_intervals = HTSeq.GenomicArrayOfSets("auto", stranded=True, storage="step")
        loci_intervals_binary = HTSeq.GenomicArray("auto", stranded=True, storage="step", typecode='b')
        self.compatibility_groups = dict()

        for transcript in self.transcripts.values():
            for exon_iv in transcript.exons.intervals:
                self.region_intervals[exon_iv] += transcript

            seq = transcript.exons.get_sequence(genome)
            for orf_iv_on_transcript in ORFFinder.find_orfs(seq):
                orf = ORF(
                    transcript,
                    orf_iv_on_transcript,
                )
                if not orf in self.orf_set:
                    self.orf_set.add(orf)
                    for iv in orf.genomic_region.intervals:
                        self.orf_intervals[iv] += orf
            
            loci_intervals_binary[transcript.iv] = True


        self.loci_intervals = HTSeq.GenomicArray("auto", stranded=True, storage="step", typecode='O')
        self.loci_set = set()
        for iv, step in loci_intervals_binary.steps():
            if step:
                locus = Locus(iv)
                self.loci_intervals[iv] = locus
                self.loci_set.add(locus)

                for step_iv, step_set in self.orf_intervals[iv].steps():
                    fr_set = frozenset(step_set)
                    if not fr_set in locus.compatibility_groups:
                        cc = CompatibilityGroup()
                        locus.compatibility_groups[fr_set] = cc
                        self.compatibility_groups[fr_set] = cc
                    locus.compatibility_groups[fr_set].length += step_iv.length


                    
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
    

    def collect_regions(self, genomic_region: GenomicRegion) -> set[ORF | Transcript]:
        regions = set()
        for query_iv in genomic_region.intervals:
            for subject_iv, region_set in self.region_intervals[query_iv].steps():
                regions |= region_set
        return regions
    

    def get_locus(self, genomic_region: GenomicRegion) -> Locus:
        loci = set()
        for iv in genomic_region.intervals:
            for step_iv, locus in self.loci_intervals[iv].steps():
                loci.add(locus)
        if len(loci) != 1:
            return
        return list(loci)[0]

