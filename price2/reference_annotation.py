import HTSeq

from price2.genomic_region import GenomicRegion
from price2.genomic_features import Transcript


class ReferenceAnnotation:
    cds_intervals: HTSeq.GenomicArrayOfSets
    transcripts: dict[str:Transcript]
    transcript_intervals: HTSeq.GenomicArrayOfSets

    def __init__(self, gtf_path: str):  # , canonical: bool = False
        chromosomes = set()
        self.cds_intervals = HTSeq.GenomicArrayOfSets("auto", stranded=True)
        self.transcripts = {}
        self.transcript_intervals = HTSeq.GenomicArrayOfSets("auto", stranded=True)

        gtf_file = HTSeq.GFF_Reader(gtf_path)

        for feature in gtf_file:
            if not "transcript_id" in feature.attr:
                continue

            if (chr := feature.iv.chrom) not in chromosomes:
                chromosomes.add(chr)
                self.cds_intervals.add_chrom(chr)
                self.transcript_intervals.add_chrom(chr)

            if feature.type == "transcript":
                if not feature.attr["transcript_id"] in self.transcripts:
                    self.transcripts[feature.attr["transcript_id"]] = Transcript(
                        feature
                    )
                continue
            if not feature.attr["transcript_id"] in self.transcripts:
                continue
            if feature.type == "exon":
                self.transcripts[feature.attr["transcript_id"]].add_exon(feature)
                self.transcript_intervals[feature.iv] += self.transcripts[
                    feature.attr["transcript_id"]
                ]
            elif feature.type in ["CDS", "five_prime_utr", "three_prime_utr"]:
                self.transcripts[feature.attr["transcript_id"]].add_region(feature)
                if feature.type == "CDS":
                    self.cds_intervals[feature.iv] += self.transcripts[
                        feature.attr["transcript_id"]
                    ]

        for transcript in self.transcripts.values():
            transcript.cds_regions_to_cds_intervals()

    def collect_coding_transcripts(self, region: GenomicRegion):
        transcripts = set()
        for interval in region.intervals:
            try:
                for iv, value in self.cds_intervals[interval].steps():
                    transcripts |= value
            except KeyError:
                pass
        return transcripts

    def collect_transcripts(self, region: GenomicRegion):
        transcripts = set()
        for interval in region.intervals:
            for iv, value in self.transcript_intervals[interval].steps():
                transcripts |= value
        return transcripts
