import HTSeq

from .genomic_region import GenomicRegion
from .genomic_features import Transcript

class ReferenceAnnotation:
    cds_intervals: HTSeq.GenomicArrayOfSets
    ti_sites: HTSeq.GenomicArray
    transcripts: dict[str: Transcript]
    transcript_intervals: HTSeq.GenomicArrayOfSets

    def __init__(self, gtf_path: str):
        self.cds_intervals = HTSeq.GenomicArrayOfSets("auto", stranded=True)
        #self.ti_sites = HTSeq.GenomicArray("auto", stranded=True, storage="step", typecode='b')
        self.ti_sites_new = HTSeq.GenomicArrayOfSets("auto", stranded=True)
        self.transcripts = {}
        self.transcript_intervals = HTSeq.GenomicArrayOfSets("auto", stranded=True)

        gtf_file = HTSeq.GFF_Reader(gtf_path)
        
        for feature in gtf_file:
            if feature.type == 'transcript':
                if not feature.attr['transcript_id'] in self.transcripts:
                    self.transcripts[feature.attr['transcript_id']] = Transcript(feature)
            elif feature.type == 'exon':
                self.transcripts[feature.attr['transcript_id']].add_exon(feature)
                self.transcript_intervals[feature.iv] += self.transcripts[feature.attr['transcript_id']]
            elif feature.type =='CDS':
                self.transcripts[feature.attr['transcript_id']].add_cds_region(feature)
                self.cds_intervals[feature.iv] += self.transcripts[feature.attr['transcript_id']]
                #self.ti_sites[feature.iv.start_as_pos] = True
                iv = HTSeq.GenomicInterval(feature.iv.chrom, feature.iv.start, feature.iv.start+1, feature.iv.strand)
                gr = GenomicRegion([iv], chrom=feature.iv.chrom, strand=feature.iv.strand)
                #self.ti_sites_new[iv] += self.transcripts[feature.attr['transcript_id']]
                self.ti_sites_new[iv] += gr

        for transcript in self.transcripts.values():
            transcript.cds_regions_to_cds_intervals()
    

    def collect_coding_transcripts(self, region: GenomicRegion):
        transcripts = set()
        for interval in region.intervals:
            for iv, value in self.cds_intervals[interval].steps():
                transcripts |= value
        return transcripts
    
    
    def collect_transcripts(self, region: GenomicRegion):
        transcripts = set()
        for interval in region.intervals:
            for iv, value in self.transcript_intervals[interval].steps():
                transcripts |= value
        return transcripts
        