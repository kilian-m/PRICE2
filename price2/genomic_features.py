import HTSeq
from HTSeq import GenomicInterval

from price2.genomic_region import GenomicRegion


class Transcript:
    type: str = "transcript"
    id: str
    gene_id: str
    exon_length: int
    exons: GenomicRegion
    five_prime_utr: GenomicRegion
    cds: GenomicRegion
    three_prime_utr: GenomicRegion
    iv: HTSeq._HTSeq.GenomicInterval
    annotated_cds_iv: tuple[
        int, int
    ]  # start and end of annotated CDS regions in transcript coordinates (assumes only one CDS per transcript)

    def __init__(self, feature: HTSeq.features.GenomicFeature):
        self.id = feature.attr["transcript_id"]
        self.gene_id = feature.attr["gene_id"]
        self.iv = feature.iv
        self.exons = GenomicRegion([], chrom=feature.iv.chrom, strand=feature.iv.strand)
        self.coding_length = 0
        self.exon_length = 0
        self.orf_set = set()

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: "Transcript") -> bool:
        return self.id == other.id

    def __len__(self) -> int:
        return len(self.exons)

    def add_exon(self, exon: HTSeq.features.GenomicFeature) -> None:
        self.exons.add_interval(exon.iv)
        self.exon_length += exon.iv.length

    def add_region(self, region: HTSeq.features.GenomicFeature) -> None:
        if region.type == "five_prime_utr":
            if not hasattr(self, "five_prime_utr"):
                self.five_prime_utr = GenomicRegion(
                    [], chrom=region.iv.chrom, strand=region.iv.strand
                )
            self.five_prime_utr.add_interval(region.iv)
        elif region.type == "CDS":
            if not hasattr(self, "cds"):
                self.cds = GenomicRegion(
                    [], chrom=region.iv.chrom, strand=region.iv.strand
                )
            self.cds.add_interval(region.iv)
            self.coding_length += region.iv.length
        elif region.type == "three_prime_utr":
            if not hasattr(self, "three_prime_utr"):
                self.three_prime_utr = GenomicRegion(
                    [], chrom=region.iv.chrom, strand=region.iv.strand
                )
            self.three_prime_utr.add_interval(region.iv)

    def cds_regions_to_cds_intervals(self) -> None:
        try:
            self.annotated_cds_iv = self.exons.induce(self.cds)
        except AttributeError:
            self.annotated_cds_iv = None

    def add_orf(self, orf: "ReadGeneratingRegion") -> None:
        self.orf_set.add(orf)
        # iv = GenomicInterval(".", orf.iv_on_transcript[0], orf.iv_on_transcript[1], ".")
        # self.orf_intervals[iv.start % 3][iv] += orf

    def update_with_filtered_orfs(self, rgr_set: set["ReadGeneratingRegion"]) -> None:
        self.orf_set = self.orf_set & rgr_set
        self.orf_intervals = (
            HTSeq.GenomicArrayOfSets("auto", stranded=False),
            HTSeq.GenomicArrayOfSets("auto", stranded=False),
            HTSeq.GenomicArrayOfSets("auto", stranded=False),
        )
        for orf in self.orf_set:
            iv = GenomicInterval(
                ".", orf.iv_on_transcript[0], orf.iv_on_transcript[1], "."
            )
            self.orf_intervals[iv.start % 3][iv] += orf


class ReadGeneratingRegion:
    type: str  # ORF or NOISE
    genomic_region: GenomicRegion
    transcript_id: str
    transcript: Transcript

    def __init__(
        self,
        type: str,
        transcript: Transcript,
        id: str,
        iv_on_transcript: tuple[int, int] = None,
        genomic_region: GenomicRegion = None,
    ):
        if iv_on_transcript is None and genomic_region is None:
            raise ValueError(
                "Either iv_on_transcript or genomic_region must be provided."
            )
        self.type = type
        self.read_count = 0
        self.id = id
        self.transcript = transcript

        if genomic_region is None:
            self.genomic_region = transcript.exons.map(
                (iv_on_transcript[0], iv_on_transcript[1] - 3)
            )
            if self.type == "ORF":
                self.full_genomic_region = transcript.exons.map(iv_on_transcript)
            else:
                self.full_genomic_region = self.genomic_region
            self.transcript_id = transcript.id
            self.iv_on_transcript = iv_on_transcript
            # for read simulation:
            self.dist_to_transcript_start = iv_on_transcript[0]
            self.dist_to_transcript_end = transcript.exon_length - iv_on_transcript[1]
        else:
            self.genomic_region = genomic_region
            self.transcript_id = transcript

    def __eq__(self, other: "ReadGeneratingRegion") -> bool:
        return (self.type == other.type) and (
            self.genomic_region == other.genomic_region
        )

    def __hash__(self) -> int:
        return hash(self.genomic_region)

    def __len__(self) -> int:
        return len(self.genomic_region)

    def __repr__(self) -> str:
        return f"{self.type} {self.id} on region {self.genomic_region}"

    def effective_length(
        self, upstream_cleavage: int = 12, downstream_cleavage: int = 17
    ) -> int:
        return (
            len(self.genomic_region)
            - max(0, upstream_cleavage - self.dist_to_transcript_start)
            - max(0, downstream_cleavage - self.dist_to_transcript_end)
        )

    def to_gtf(self, loc_id: str) -> str:
        seq_id = self.genomic_region.chrom
        source = "PRICE2"
        if self.type == "ORF":
            typ = "CDS"
        else:
            typ = "exon"
        # typ = self.type # NOISE, ORF
        # start
        # end
        score = "."
        strand = self.genomic_region.strand
        phase = "."
        attributes = f'gene_id "{self.transcript.gene_id}"; transcript_id "{self.id}"; loc_id "{loc_id}";'
        if hasattr(self, "log_p_value"):
            attributes += f'log_p_value "{self.log_p_value}";'
        if hasattr(self, "mean_rpkm"):
            attributes += f'mean_rpkm "{self.mean_rpkm}";'
        if hasattr(self, "mean_activity"):
            attributes += f'mean_activity "{self.mean_activity}";'
        # s = f'{seq_id}\t{source}\t{typ}\t{self.genomic_region.intervals[0].start}\t{self.genomic_region.intervals[-1].end}\t{score}\t{strand}\t{phase}\t{attributes}\n'
        s = ""
        for interval in self.full_genomic_region.intervals:
            s += f"{seq_id}\t{source}\texon\t{interval.start+1}\t{interval.end}\t{score}\t{strand}\t{phase}\t{attributes}\n"
            if typ == "CDS":
                s += f"{seq_id}\t{source}\tCDS\t{interval.start+1}\t{interval.end}\t{score}\t{strand}\t{phase}\t{attributes}\n"

        return s
