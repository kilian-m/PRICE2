"""Genomic region representation for multi-exonic coordinate handling.

Provides the ``GenomicRegion`` class for representing potentially
multi-exonic genomic regions and performing coordinate transformations
between genomic and region-local coordinate systems.

Notes
-----
Intervals are stored in **chromosome order** (ascending start
coordinate), regardless of strand.  For negative-strand regions this
is the reverse of translation order.  The project convention is
0-based, half-open coordinates throughout.
"""

from __future__ import annotations

from typing import Literal

import HTSeq
import pandas as pd
from HTSeq import GenomicInterval

Strand = Literal["+", "-"]
"""Genomic strand: either ``'+'`` (forward) or ``'-'`` (reverse)."""


class GenomicRegion:
    """A possibly multi-exonic genomic region on a single chromosome.

    Intervals are stored in chromosome order (ascending start position).
    All coordinates are 0-based, half-open.

    Attributes
    ----------
    chrom : str
        Chromosome / reference sequence name.
    strand : Strand
        ``'+'`` or ``'-'``.
    intervals : list[GenomicInterval]
        Exonic intervals in chromosome order.
    length : int
        Total spliced length (sum of exon lengths).
    """

    chrom: str
    strand: Strand
    intervals: list[GenomicInterval]
    length: int

    def __init__(
        self,
        intervals: list[GenomicInterval] | None = None,
        chrom: str | None = None,
        strand: Strand | None = None,
        df: pd.DataFrame | None = None,
        gi_string: str | None = None,
    ) -> None:
        """Create a ``GenomicRegion``.

        Exactly one of *intervals*, *df*, or *gi_string* must be given.

        Parameters
        ----------
        intervals : list[GenomicInterval] | None
            Pre-built HTSeq ``GenomicInterval`` objects.
        chrom : str | None
            Chromosome name (inferred from *intervals* if omitted).
        strand : Strand | None
            Strand (inferred from *intervals* if omitted).
        df : pd.DataFrame | None
            DataFrame with columns ``chrom``, ``start``, ``end``,
            ``strand``.
        gi_string : str | None
            Compact string representation, e.g.
            ``"chr1+:100-200|300-400"``.
        """
        if df is not None:
            self.intervals = [
                HTSeq.GenomicInterval(
                    row["chrom"],
                    row["start"],
                    row["end"],
                    row["strand"],
                )
                for _, row in df.iterrows()
            ]
            self.chrom = self.intervals[0].chrom
            self.strand = self.intervals[0].strand
        elif gi_string is not None:
            chrom_strand, intervals_str = gi_string.split(":")
            chrom = chrom_strand[:-1]
            strand = chrom_strand[-1]
            intervals = [
                HTSeq.GenomicInterval(chrom, int(start), int(end), strand)
                for start, end in (x.split("-") for x in intervals_str.split("|"))
            ]
            if strand == "-":
                intervals = intervals[::-1]
            self.chrom = chrom
            self.strand = strand
            self.intervals = intervals
        else:
            self.chrom = chrom if chrom else intervals[0].chrom
            self.strand = strand if strand else intervals[0].strand
            self.intervals = intervals

        self.length = sum(iv.end - iv.start for iv in self.intervals)
        self.hash = hash((self.strand, self.chrom, tuple(self.intervals)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GenomicRegion):
            return False
        return (
            self.chrom == other.chrom
            and self.intervals == other.intervals
            and self.strand == other.strand
        )

    def __hash__(self) -> int:
        # NOTE: hash changes when intervals are added via add_interval.
        return self.hash

    def __str__(self) -> str:
        intervals_str = "|".join(f"{iv.start}-{iv.end}" for iv in self.intervals)
        return f"{self.chrom}{self.strand}:{intervals_str}"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        if self.length == 0:
            self.length = sum(iv.end - iv.start for iv in self.intervals)
        return self.length

    def add_interval(self, interval: HTSeq.GenomicInterval) -> None:
        """Append an exon interval and recompute the hash.

        Parameters
        ----------
        interval : HTSeq.GenomicInterval
            Interval to append.

        Notes
        -----
        Calling this method invalidates any previously stored hash.
        """
        self.intervals.append(interval)
        self.hash = hash((self.strand, self.chrom, tuple(self.intervals)))

    def induce(self, other: GenomicRegion) -> tuple[int, int]:
        """Express *other* as region-local coordinates within *self*.

        *other* must be a sub-region of *self* (same exon structure
        in the overlapping part).

        Parameters
        ----------
        other : GenomicRegion
            Region whose coordinates should be expressed
            relative to *self*.

        Returns
        -------
        tuple[int, int]
            ``(start, end)`` in spliced region-local coordinates
            (0-based, half-open).

        Raises
        ------
        ValueError
            If *other* is on a different chromosome/strand or
            is not compatible with *self*'s exon structure.
        """
        if self.chrom != other.chrom or self.strand != other.strand:
            raise ValueError("Query region is not compatible with reference " "region.")
        ref_intervals = self.intervals
        query_intervals = other.intervals

        if self.strand == "-":
            exons_end = ref_intervals[0].end
            ref_intervals = [
                HTSeq.GenomicInterval(
                    ri.chrom,
                    exons_end - ri.end,
                    exons_end - ri.start,
                    ri.strand,
                )
                for ri in ref_intervals
            ]
            query_intervals = [
                HTSeq.GenomicInterval(
                    qi.chrom,
                    exons_end - qi.end,
                    exons_end - qi.start,
                    qi.strand,
                )
                for qi in query_intervals
            ]

        # overlapping part
        s = max(ref_intervals[0].start, query_intervals[0].start)
        e = min(ref_intervals[-1].end, query_intervals[-1].end)
        cut_query = cut_intervals(query_intervals, s, e)
        cut_ref = cut_intervals(ref_intervals, s, e)
        if cut_query != cut_ref:
            raise ValueError("Query region is not compatible with reference " "region.")

        # upstream part
        upstream_query = cut_intervals(query_intervals, query_intervals[0].start, s)
        if len(upstream_query) > 1:
            raise ValueError("Query region is not compatible with reference " "region.")

        # downstream part
        downstream_query = cut_intervals(query_intervals, e, query_intervals[-1].end)
        if len(downstream_query) > 1:
            raise ValueError("Query region is not compatible with reference " "region.")

        upstream_ref = cut_intervals(ref_intervals, ref_intervals[0].start, s)

        if len(upstream_query) == 1:
            start = query_intervals[0].start - ref_intervals[0].start
        elif ref_intervals[-1].end <= query_intervals[0].start:
            start = (
                sum(x.end - x.start for x in ref_intervals)
                + query_intervals[0].start
                - ref_intervals[-1].end
            )
        else:
            start = sum(x[1] - x[0] for x in upstream_ref)
        end = start + sum(iv.end - iv.start for iv in query_intervals)
        return (start, end)

    def map(self, iv_in_region: tuple[int, int]) -> GenomicRegion:
        """Map region-local coordinates back to genomic coordinates.

        Parameters
        ----------
        iv_in_region : tuple[int, int]
            ``(start, end)`` in spliced region-local coordinates
            (0-based, half-open).

        Returns
        -------
        GenomicRegion
            New ``GenomicRegion`` covering the mapped genomic
            intervals.
        """
        start_in_region, end_in_region = iv_in_region
        if self.strand == "+":
            s = 0  # cumulative exon length from transcript start
            idx = 0
            region_starts: list[int] = []
            region_ends: list[int] = []
            while (
                idx < len(self.intervals) - 1
                and s + self.intervals[idx].length < start_in_region
            ):
                s += self.intervals[idx].length
                idx += 1
            region_starts.append(self.intervals[idx].start + start_in_region - s)

            while (
                idx < len(self.intervals) - 1
                and s + self.intervals[idx].length < end_in_region
            ):
                region_ends.append(self.intervals[idx].end)
                s += self.intervals[idx].length
                idx += 1
                region_starts.append(self.intervals[idx].start)
            region_ends.append(self.intervals[idx].start + end_in_region - s)

            ivs = zip(region_starts, region_ends)

        elif self.strand == "-":
            s = 0
            idx = 0
            region_starts = []
            region_ends = []

            while (
                idx < len(self.intervals) - 1
                and s + self.intervals[idx].length < start_in_region
            ):
                s += self.intervals[idx].length
                idx += 1
            region_starts.append(self.intervals[idx].end - (start_in_region - s))

            while (
                idx < len(self.intervals) - 1
                and s + self.intervals[idx].length < end_in_region
            ):
                region_ends.append(self.intervals[idx].start)
                s += self.intervals[idx].length
                idx += 1
                region_starts.append(self.intervals[idx].end)
            region_ends.append(self.intervals[idx].end - (end_in_region - s))

            ivs = zip(region_ends, region_starts)

        region_intervals = [
            HTSeq.GenomicInterval(self.chrom, rs, re, self.strand)
            for rs, re in ivs
            if rs != re
        ]
        return GenomicRegion(
            intervals=region_intervals,
            strand=self.strand,
            chrom=self.chrom,
        )

    def get_sequence(self, genome: dict[str, HTSeq._HTSeq.Sequence]) -> str:
        """Extract the nucleotide sequence for this region.

        On the negative strand each exon is reverse-complemented so
        that the returned string is in 5'->3' (translation) order.

        Parameters
        ----------
        genome : dict[str, HTSeq._HTSeq.Sequence]
            Mapping of chromosome names to genome sequences.

        Returns
        -------
        str
            Spliced nucleotide sequence.
        """
        parts: list[str] = []
        if self.strand == "+":
            for iv in self.intervals:
                parts.append(str(genome[self.chrom][iv.start : iv.end]))
        elif self.strand == "-":
            for iv in self.intervals:
                parts.append(
                    str(genome[self.chrom][iv.start : iv.end].get_reverse_complement())
                )
        return "".join(parts)

    def contains_to_stop(self, other: GenomicRegion) -> bool:
        """Check whether *other* is contained in *self* sharing the same stop end.

        The two regions must share the same 3'-most exon
        boundaries; *other* may start later (for ``+``) or
        earlier (for ``-``) than *self*.

        Parameters
        ----------
        other : GenomicRegion
            Candidate sub-region.

        Returns
        -------
        bool
            ``True`` if *other* is contained in *self* and they
            share the same stop-codon end.
        """
        if self.strand != other.strand or self.chrom != other.chrom:
            return False
        if len(self.intervals) < len(other.intervals):
            return False

        s_ivs = self.intervals[::-1]
        o_ivs = other.intervals[::-1]

        for i in range(len(o_ivs)):
            s_iv = s_ivs[i]
            o_iv = o_ivs[i]
            if s_iv == o_iv:
                continue
            if i + 1 < len(o_ivs):
                return False
            if self.strand == "-":
                if s_iv.start != o_iv.start or s_iv.end < o_iv.end:
                    return False
            else:
                if s_iv.end != o_iv.end or s_iv.start > o_iv.start:
                    return False
        return True

    def truncate(self, length: int, end: str) -> GenomicRegion:
        """Remove nucleotides from the 5' or 3' end.

        Parameters
        ----------
        length : int
            Number of nucleotides to remove.
        end : str
            Which end to truncate: ``"5'"`` or ``"3'"``.

        Returns
        -------
        GenomicRegion
            Truncated region.

        Raises
        ------
        ValueError
            If *length* is non-positive, exceeds the region
            length, or *end* is invalid.
        """
        if length <= 0:
            raise ValueError("Length must be positive.")
        if length >= self.length:
            raise ValueError("Cannot truncate more than the total length.")

        if end == "5'":
            new_start = length
            new_end = self.length
        elif end == "3'":
            new_start = 0
            new_end = self.length - length
        else:
            raise ValueError('end must be "5\'" or "3\'".')

        return self.map((new_start, new_end))


def cut_intervals(
    interval_list: list[GenomicInterval],
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    """Clip a list of intervals to ``[start, end)``.

    Parameters
    ----------
    interval_list : list[GenomicInterval]
        Intervals to clip.
    start : int
        Inclusive lower bound.
    end : int
        Exclusive upper bound.

    Returns
    -------
    list[tuple[int, int]]
        Clipped ``(start, end)`` tuples for intervals that overlap
        ``[start, end)``.
    """
    result: list[tuple[int, int]] = []
    for iv in interval_list:
        if iv.end <= start or end <= iv.start:
            continue
        clipped_start = max(iv.start, start)
        clipped_end = min(iv.end, end)
        result.append((clipped_start, clipped_end))
    return result
