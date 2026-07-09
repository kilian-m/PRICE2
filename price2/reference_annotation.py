"""Reference annotation loading and interval indexing from GTF files."""

import logging
from bisect import bisect_left, bisect_right

import HTSeq

from price2.genomic_region import GenomicRegion
from price2.genomic_features import Transcript

logger = logging.getLogger(__name__)

#: Flattened CDS interval index: per ``(chromosome, strand)``, the start position
#: of every step and the transcripts covering it (``None`` where there are none).
CdsIndex = dict[tuple[str, str], tuple[list[int], list[frozenset | None]]]


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

    #: Flattened ``cds_intervals``; see :meth:`build_cds_index`.
    _cds_index: CdsIndex | None = None

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

    def build_cds_index(self) -> None:
        """Flatten ``cds_intervals`` into a per-chromosome binary-search index.

        Walking an :class:`HTSeq.GenomicArrayOfSets` with ``steps()`` costs
        several microseconds per query, which dominates cleavage- and
        coverage-model estimation (one query per read).  The step boundaries
        never change after construction, so they are flattened once into a
        sorted list of step starts plus the transcript set of each step, which
        a query resolves with two binary searches.

        Equal transcript sets are interned, and steps covered by no CDS store
        ``None``, so the index adds little memory on top of the source array.
        Called automatically on the first query; call it explicitly in a parent
        process to share the index with forked workers.
        """
        if self._cds_index is not None:
            return

        index: CdsIndex = {}
        interned: dict[frozenset, frozenset] = {}
        for chrom, strand_vectors in self.cds_intervals.chrom_vectors.items():
            for strand, chrom_vector in strand_vectors.items():
                starts: list[int] = []
                step_sets: list[frozenset | None] = []
                for iv, transcripts in chrom_vector.steps():
                    if transcripts:
                        frozen = frozenset(transcripts)
                        frozen = interned.setdefault(frozen, frozen)
                    else:
                        frozen = None
                    starts.append(iv.start)
                    step_sets.append(frozen)
                index[(chrom, strand)] = (starts, step_sets)
        self._cds_index = index

    def _coding_transcripts_at(
        self, chrom: str, strand: str, start: int, end: int
    ) -> frozenset | None:
        """Transcripts whose CDS overlaps ``[start, end)``, or ``None``."""
        try:
            starts, step_sets = self._cds_index[(chrom, strand)]
        except KeyError:
            return None

        first = bisect_right(starts, start) - 1
        if first < 0:
            first = 0
        last = bisect_left(starts, end)

        if last - first == 1:
            return step_sets[first]

        found = None
        for i in range(first, last):
            step = step_sets[i]
            if step is None:
                continue
            found = step if found is None else found | step
        return found

    def collect_coding_transcripts(self, region: GenomicRegion) -> frozenset[Transcript]:
        """Return all transcripts with a CDS overlapping *region*.

        Parameters
        ----------
        region : GenomicRegion
            The genomic region to query.

        Returns
        -------
        frozenset[Transcript]
            Transcripts whose CDS intervals overlap any exon of *region*.
        """
        if self._cds_index is None:
            self.build_cds_index()

        found = None
        for interval in region.intervals:
            step = self._coding_transcripts_at(
                interval.chrom, interval.strand, interval.start, interval.end
            )
            if step is None:
                continue
            found = step if found is None else found | step
        return found if found is not None else frozenset()

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
