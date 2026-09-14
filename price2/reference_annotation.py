"""Reference annotation loading and interval indexing from GTF files.

Coordinates: the GTF is 1-based and closed; :class:`HTSeq.GFF_Reader`
converts every feature to the project's 0-based, half-open convention while
parsing, so nothing here converts again.
"""

from __future__ import annotations

import logging
from bisect import bisect_left, bisect_right
from collections.abc import Iterable

import HTSeq

from price2.genomic_region import GenomicRegion
from price2.genomic_features import Transcript

logger = logging.getLogger(__name__)

#: Flattened CDS interval index: per ``(chromosome, strand)``, the start position
#: of every step and the transcripts whose CDS covers it (``None`` where there
#: are none).  The first step starts at 0 and the last one is unbounded.
CdsIndex = dict[tuple[str, str], tuple[list[int], list[frozenset | None]]]


class ReferenceAnnotation:
    """Index a GTF reference annotation for fast interval queries.

    Parses a GTF file into :class:`~price2.genomic_features.Transcript`
    objects and builds two indexes over them: a stranded step array of the
    exons, from which the loci are cut, and a flattened index of the CDSs
    answering which coding transcripts a read overlaps.

    Attributes
    ----------
    transcripts : dict[str, Transcript]
        Mapping from transcript ID to :class:`~price2.genomic_features.Transcript`
        object.
    transcript_intervals : HTSeq.GenomicArrayOfSets
        Interval index mapping genomic positions to transcripts whose exons
        overlap those positions.
    """

    transcripts: dict[str, Transcript]
    transcript_intervals: HTSeq.GenomicArrayOfSets

    def __init__(self, gtf_path: str) -> None:
        """Parse a GTF file and build the indexes.

        Parameters
        ----------
        gtf_path : str
            Path to the GTF annotation file.
        """
        self.transcripts = {}
        self.transcript_intervals = HTSeq.GenomicArrayOfSets("auto", stranded=True)
        chromosomes: set[str] = set()

        for feature in HTSeq.GFF_Reader(gtf_path):
            transcript_id = feature.attr.get("transcript_id")
            if transcript_id is None:
                continue
            chrom = feature.iv.chrom
            if chrom not in chromosomes:
                chromosomes.add(chrom)
                self.transcript_intervals.add_chrom(chrom)

            if feature.type == "transcript":
                if transcript_id not in self.transcripts:
                    self.transcripts[transcript_id] = Transcript(feature)
                continue
            transcript = self.transcripts.get(transcript_id)
            if transcript is None:
                continue
            if feature.type == "exon":
                transcript.add_exon(feature)
                self.transcript_intervals[feature.iv] += transcript
            elif feature.type == "CDS":
                transcript.add_cds_region(feature)

        for transcript in self.transcripts.values():
            transcript.finalize()
        self._cds_index = _cds_index_from(self.transcripts.values())

        logger.info("Loaded %d transcripts from %s", len(self.transcripts), gtf_path)

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
        found = None
        for interval in region.intervals:
            step = self._coding_transcripts_at(
                region.chrom, region.strand, interval.start, interval.end
            )
            if step is None:
                continue
            found = step if found is None else found | step
        return found if found is not None else frozenset()


def _cds_index_from(transcripts: Iterable[Transcript]) -> CdsIndex:
    """Flatten the transcripts' CDS intervals into a per-chromosome step index.

    A sweep over the interval boundaries of each ``(chromosome, strand)``:
    every position where the set of covering CDSs changes starts a new
    step.  A query then resolves with two binary searches, which is what
    cleavage- and coverage-model estimation need (one query per read).
    Equal transcript sets are interned, and steps covered by no CDS store
    ``None``, so the index is small.
    """
    events: dict[tuple[str, str], list[tuple[int, int, Transcript]]] = {}
    for transcript in transcripts:
        if transcript.cds is None:
            continue
        key = (transcript.cds.chrom, transcript.cds.strand)
        bucket = events.setdefault(key, [])
        for interval in transcript.cds.intervals:
            bucket.append((interval.start, 1, transcript))
            bucket.append((interval.end, -1, transcript))

    index: CdsIndex = {}
    interned: dict[frozenset, frozenset] = {}
    for key, bucket in events.items():
        bucket.sort(key=lambda event: event[0])
        starts: list[int] = [0]
        step_sets: list[frozenset | None] = [None]
        active: set[Transcript] = set()
        i = 0
        while i < len(bucket):
            position = bucket[i][0]
            while i < len(bucket) and bucket[i][0] == position:
                _, delta, transcript = bucket[i]
                if delta > 0:
                    active.add(transcript)
                else:
                    active.discard(transcript)
                i += 1
            frozen = interned.setdefault(frozenset(active), frozenset(active)) if active else None
            if frozen == step_sets[-1]:
                continue
            if position == starts[-1]:
                step_sets[-1] = frozen
            else:
                starts.append(position)
                step_sets.append(frozen)
        index[key] = (starts, step_sets)
    return index
