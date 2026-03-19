"""Reference annotation loading and interval indexing from GTF files."""

import logging

import HTSeq

from price2.genomic_region import GenomicRegion
from price2.genomic_features import Transcript

logger = logging.getLogger(__name__)


class ReferenceAnnotation:
    """Index a GTF reference annotation for fast interval queries.

    Parses a GTF file and builds interval-indexed data structures for
    transcripts, exons, and CDS regions, enabling efficient lookup of
    transcripts overlapping a given genomic region.

    Attributes
    ----------
    cds_intervals : HTSeq.GenomicArrayOfSets
        Interval index mapping genomic positions to transcripts whose CDS
        overlaps those positions.
    transcripts : dict[str, Transcript]
        Mapping from transcript ID to :class:`~price2.genomic_features.Transcript`
        object.
    transcript_intervals : HTSeq.GenomicArrayOfSets
        Interval index mapping genomic positions to transcripts whose exons
        overlap those positions.
    """

    cds_intervals: HTSeq.GenomicArrayOfSets
    transcripts: dict[str, Transcript]
    transcript_intervals: HTSeq.GenomicArrayOfSets

    def __init__(self, gtf_path: str) -> None:
        """Parse a GTF file and build genomic interval indexes.

        Parameters
        ----------
        gtf_path : str
            Path to the GTF annotation file.
        """
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
            elif feature.type == "CDS":
                self.transcripts[feature.attr["transcript_id"]].add_cds_region(feature)

        for transcript in self.transcripts.values():
            transcript.cds_regions_to_cds_intervals()
            if transcript.annotated_cds_iv is not None:
                for interval in transcript._cds.intervals:
                    self.cds_intervals[interval] += transcript

        logger.info("Loaded %d transcripts from %s", len(self.transcripts), gtf_path)

    def collect_coding_transcripts(self, region: GenomicRegion) -> set[Transcript]:
        """Return all transcripts with a CDS overlapping *region*.

        Parameters
        ----------
        region : GenomicRegion
            The genomic region to query.

        Returns
        -------
        set[Transcript]
            Transcripts whose CDS intervals overlap any exon of *region*.
        """
        transcripts = set()
        for interval in region.intervals:
            try:
                for iv, value in self.cds_intervals[interval].steps():
                    transcripts |= value
            except KeyError:
                pass
        return transcripts

    def collect_transcripts(self, region: GenomicRegion) -> set[Transcript]:
        """Return all transcripts with an exon overlapping *region*.

        Parameters
        ----------
        region : GenomicRegion
            The genomic region to query.

        Returns
        -------
        set[Transcript]
            Transcripts whose exon intervals overlap any exon of *region*.
        """
        transcripts = set()
        for interval in region.intervals:
            for iv, value in self.transcript_intervals[interval].steps():
                transcripts |= value
        return transcripts
