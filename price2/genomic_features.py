import HTSeq

from .genomic_region import GenomicRegion



class Transcript:
    type: str = 'transcript'
    id: str
    coding_length: int
    exon_length: int
    exons: GenomicRegion
    cds: GenomicRegion
    iv: HTSeq._HTSeq.GenomicInterval
    annotated_cds_iv: tuple[int, int] # start and end of annotated CDS regions in transcript coordinates (assumes only one CDS per transcript)

    def __init__(self, feature: HTSeq.features.GenomicFeature):
        self.id = feature.attr['transcript_id']
        self.iv = feature.iv
        self.exons = GenomicRegion([], chrom=feature.iv.chrom, strand=feature.iv.strand)
        self.coding_length = 0
        self.exon_length = 0

    def add_exon(self, exon: HTSeq.features.GenomicFeature) -> None:
        self.exons.add_interval(exon.iv)
        self.exon_length += exon.iv.length

    def add_cds_region(self, cds_region: HTSeq.features.GenomicFeature) -> None:
        try:
            self.cds.add_interval(cds_region.iv)
        except AttributeError:
            self.cds = GenomicRegion([], chrom=cds_region.iv.chrom, strand=cds_region.iv.strand)
            self.cds.add_interval(cds_region.iv)
        self.coding_length += cds_region.iv.length

    def cds_regions_to_cds_intervals(self) -> None:
        try:
            self.annotated_cds_iv = self.exons.induce(self.cds)
        except AttributeError:
            self.annotated_cds_iv = None


class ReadGeneratingRegion:
    type: str
    genomic_region: GenomicRegion
    transcript: Transcript
    iv_on_transcript: tuple[int, int]
    average_activity: float

    def __init__(self, type: str, transcript: Transcript, iv_on_transcript: tuple[int, int],):
        self.type = type
        self.transcript = transcript
        self.iv_on_transcript = iv_on_transcript
        self.genomic_region = transcript.exons.map(iv_on_transcript)

    
    def __eq__(self, other: 'ReadGeneratingRegion') -> bool:
        return (self.type == other.type) and (self.genomic_region == other.genomic_region)
    
    
    def __hash__(self) -> int:
        return hash(self.genomic_region)


    def __len__(self) -> int:
        return len(self.genomic_region)
    
    
    def __repr__(self) -> str:
        return f'{self.type} on region {self.genomic_region}'

