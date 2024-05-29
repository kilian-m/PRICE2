import HTSeq

from .genomic_region import GenomicRegion



class Transcript:
    type: str = 'transcript'
    id: str
    gene_id: str
    exon_length: int
    exons: GenomicRegion
    five_prime_utr: GenomicRegion
    cds: GenomicRegion
    three_prime_utr: GenomicRegion
    iv: HTSeq._HTSeq.GenomicInterval
    annotated_cds_iv: tuple[int, int] # start and end of annotated CDS regions in transcript coordinates (assumes only one CDS per transcript)

    def __init__(self, feature: HTSeq.features.GenomicFeature):
        self.id = feature.attr['transcript_id']
        self.gene_id = feature.attr['gene_id']
        self.iv = feature.iv
        self.exons = GenomicRegion([], chrom=feature.iv.chrom, strand=feature.iv.strand)
        self.coding_length = 0
        self.exon_length = 0

    def add_exon(self, exon: HTSeq.features.GenomicFeature) -> None:
        self.exons.add_interval(exon.iv)
        self.exon_length += exon.iv.length

    def add_region(self, region: HTSeq.features.GenomicFeature) -> None:
        if region.type == 'five_prime_UTR':
            if not hasattr(self, 'five_prime_utr'):
                self.five_prime_utr = GenomicRegion([], chrom=region.iv.chrom, strand=region.iv.strand)
            self.five_prime_utr.add_interval(region.iv)
        elif region.type == 'CDS':
            if not hasattr(self, 'cds'):
                self.cds = GenomicRegion([], chrom=region.iv.chrom, strand=region.iv.strand)
            self.cds.add_interval(region.iv)
            self.coding_length += region.iv.length
        elif region.type == 'three_prime_UTR':
            if not hasattr(self, 'three_prime_utr'):
                self.three_prime_utr = GenomicRegion([], chrom=region.iv.chrom, strand=region.iv.strand)
            self.three_prime_utr.add_interval(region.iv)



    def cds_regions_to_cds_intervals(self) -> None:
        try:
            self.annotated_cds_iv = self.exons.induce(self.cds)
        except AttributeError:
            self.annotated_cds_iv = None


class ReadGeneratingRegion:
    type: str # ORF or NOISE
    genomic_region: GenomicRegion
    transcript_id: str
    #transcript: Transcript

    def __init__(self, type: str, transcript: Transcript, 
                 iv_on_transcript: tuple[int, int]=None,
                 genomic_region: GenomicRegion=None
                 ):
        if iv_on_transcript is None and genomic_region is None:
            raise ValueError('Either iv_on_transcript or genomic_region must be provided.')
        self.type = type
        #self.transcript = transcript

        if genomic_region is None:
            self.genomic_region = transcript.exons.map(iv_on_transcript)
            self.transcript_id = transcript.id
        else:
            self.genomic_region = genomic_region
            self.transcript_id = transcript

    
    def __eq__(self, other: 'ReadGeneratingRegion') -> bool:
        return (self.type == other.type) and (self.genomic_region == other.genomic_region)
    
    
    def __hash__(self) -> int:
        return hash(self.genomic_region)


    def __len__(self) -> int:
        return len(self.genomic_region)
    
    
    def __repr__(self) -> str:
        return f'{self.type} on region {self.genomic_region}'
    

    def to_serializable(self) -> tuple:
        t = (self.type, self.transcript_id, self.genomic_region.strand, self.genomic_region.chrom, 
             [(x.start, x.end) for x in self.genomic_region.intervals])
        return t
    

    def from_serializable(t: tuple) -> 'ReadGeneratingRegion':
        type, transcript_id, strand, chrom, intervals = t
        genomic_region = GenomicRegion([HTSeq.GenomicInterval(chrom, x[0], x[1], strand) for x in intervals], chrom, strand)
        return ReadGeneratingRegion(type, transcript_id, genomic_region=genomic_region)


