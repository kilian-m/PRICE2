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
from pyfaidx import Fasta

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
        bed_line: str | None = None,
    ) -> None:
        """Create a ``GenomicRegion``.

        Exactly one of *intervals*, *df*, *gi_string*, or *bed_line* must
        be given.

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
        bed_line : str | None
            A single tab-separated BED line.  BED6 (6 columns) produces a
            single-exon region; BED12 (12 columns) produces a multi-exonic
            region from the block columns.  Coordinates are 0-based
            half-open as per the BED specification.  Strand defaults to
            ``'+'`` when the strand column is absent (BED3/BED4/BED5).
        """
        if df is not None:
            intervals = [
                HTSeq.GenomicInterval(
                    row["chrom"],
                    row["start"],
                    row["end"],
                    row["strand"],
                )
                for _, row in df.iterrows()
            ]
            self.chrom = intervals[0].chrom
            self.strand = intervals[0].strand
        elif gi_string is not None:
            chrom_strand, intervals_str = gi_string.split(":")
            chrom = chrom_strand[:-1]
            strand = chrom_strand[-1]
            intervals = [
                HTSeq.GenomicInterval(chrom, int(start), int(end), strand)
                for start, end in (x.split("-") for x in intervals_str.split("|"))
            ]
            self.chrom = chrom
            self.strand = strand
        elif bed_line is not None:
            fields = bed_line.rstrip("\n").split("\t")
            chrom = fields[0]
            chrom_start = int(fields[1])
            chrom_end = int(fields[2])
            strand = fields[5] if len(fields) > 5 else "+"
            if len(fields) >= 12:
                block_count = int(fields[9])
                block_sizes = [int(x) for x in fields[10].rstrip(",").split(",")]
                block_starts = [int(x) for x in fields[11].rstrip(",").split(",")]
                intervals = [
                    HTSeq.GenomicInterval(
                        chrom,
                        chrom_start + block_starts[i],
                        chrom_start + block_starts[i] + block_sizes[i],
                        strand,
                    )
                    for i in range(block_count)
                ]
            else:
                intervals = [
                    HTSeq.GenomicInterval(chrom, chrom_start, chrom_end, strand)
                ]
            self.chrom = chrom
            self.strand = strand
        else:
            self.chrom = chrom if chrom else intervals[0].chrom
            self.strand = strand if strand else intervals[0].strand
        self.intervals = intervals

        for i in range(1, len(self.intervals)):
            if self.intervals[i - 1].end > self.intervals[i].start:
                raise ValueError(
                    "Intervals must be non-overlapping and in chromosome order."
                )

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
        # self.intervals.append(interval)
        self.hash = hash((self.strand, self.chrom, tuple(self.intervals)))
        self.length += interval.end - interval.start
        if self.strand == "+":
            self.intervals.append(interval)
        else:
            self.intervals.insert(0, interval)

    def map_to_local(self, other: GenomicRegion) -> tuple[int, int]:
        """Map *other* into the local spliced coordinate system of *self*.

        Computes the ``(start, end)`` position of *other* within *self*'s
        concatenated exon space (0-based, half-open).  Local position 0
        corresponds to the first nucleotide of *self* in chromosome order,
        which is the 3'-most nucleotide for negative-strand regions.

        *other* must be fully contained in *self*: every interval of *other*
        must lie within exactly one interval of *self*, with no gaps (skipped
        self-intervals) between consecutive matches.

        Parameters
        ----------
        other : GenomicRegion
            Region to project into local coordinates.  Must share the same
            chromosome and strand as *self*.

        Returns
        -------
        tuple[int, int]
            ``(local_start, local_end)`` in *self*'s spliced coordinate space.

        Raises
        ------
        ValueError
            If *other* is ``None`` or has a different chromosome or strand or is not fully contained in *self*.
        """
        if other is None:
            raise ValueError("Cannot map None region to local coordinates.")

        if self.chrom != other.chrom or self.strand != other.strand:
            raise ValueError("Cannot map region with different chromosome or strand.")

        # self_ivs = sorted(self.intervals, key=lambda iv: iv.start)
        # other_ivs = sorted(other.intervals, key=lambda iv: iv.start)

        j = 0
        cum_len = 0  # cumulative spliced length of self-intervals before index j
        prev_j = None
        prev_other_end = None
        local_start = None
        local_end = None

        for other_iv in other.intervals:
            a, b = other_iv.start, other_iv.end

            # Advance j past self-intervals that end at or before the start
            # of the current other-interval (no overlap possible).
            while j < len(self.intervals) and self.intervals[j].end <= a:
                cum_len += self.intervals[j].end - self.intervals[j].start
                j += 1

            if j >= len(self.intervals):
                raise ValueError(
                    "Other region extends beyond the bounds of this region."
                )

            x, y = self.intervals[j].start, self.intervals[j].end

            # other-interval must be fully contained within this self-interval.
            if not (x <= a and b <= y):
                raise ValueError(
                    "Other region is not fully contained within this region."
                )

            # Contiguity check: no self-intervals were skipped between matches.
            if prev_j is not None and j > prev_j + 1:
                raise ValueError(
                    "Other region spans a junction not present in this region."
                )

            # Same-exon junction: consecutive read exons within one
            # self-interval means a spurious splice inside an exon.
            if prev_j is not None and j == prev_j:
                raise ValueError(
                    "Read has a splice junction inside a reference exon."
                )

            # Junction boundary check: when consecutive other-intervals
            # map to consecutive self-intervals, the splice sites must
            # align exactly.
            if prev_j is not None and j == prev_j + 1:
                if prev_other_end != self.intervals[prev_j].end:
                    raise ValueError(
                        "Read junction does not match reference exon boundary."
                    )
                if a != x:
                    raise ValueError(
                        "Read junction does not match reference exon boundary."
                    )

            offset_start = cum_len + (a - x)
            offset_end = cum_len + (b - x)

            if local_start is None:
                local_start = offset_start
            local_end = offset_end
            prev_j = j
            prev_other_end = b

        # For negative strand, local position 0 is the 5' end (highest
        # genomic coordinate), so flip the chromosome-order offsets.
        if self.strand == "-":
            local_start, local_end = self.length - local_end, self.length - local_start

        return (local_start, local_end)

    def map_to_global(self, iv: tuple[int, int]) -> GenomicRegion:
        """Map region-local coordinates back to genomic coordinates.

        Parameters
        ----------
        iv : tuple[int, int]
            ``(start, end)`` in spliced region-local coordinates
            (0-based, half-open).

        Returns
        -------
        GenomicRegion
            New ``GenomicRegion`` covering the mapped genomic intervals.

        Raises
        ------
        ValueError
            If *start* is negative or *end* exceeds the region length.
        """
        start, end = iv
        if start < 0:
            raise ValueError(
                f"Interval start {start} is before the reference (must be >= 0)."
            )
        if end > len(self):
            raise ValueError(
                f"Interval end {end} exceeds the reference length {len(self)}."
            )

        # For negative strand, local position 0 is the 5' end (highest
        # genomic coordinate).  Flip to chromosome-order offsets first.
        if self.strand == "-":
            start, end = self.length - end, self.length - start

        result_ivs: list[tuple[int, int]] = []
        cumulative = 0

        for interval in self.intervals:
            iv_len = interval.length
            local_end = cumulative + iv_len
            overlap_start = max(start, cumulative)
            overlap_end = min(end, local_end)
            if overlap_start < overlap_end:
                global_start = interval.start + (overlap_start - cumulative)
                global_end = interval.start + (overlap_end - cumulative)
                result_ivs.append((global_start, global_end))
            cumulative += iv_len
            if cumulative >= end:
                break

        region_intervals = [
            HTSeq.GenomicInterval(self.chrom, rs, re, self.strand)
            for rs, re in result_ivs
            if rs != re
        ]
        return GenomicRegion(
            intervals=region_intervals,
            strand=self.strand,
            chrom=self.chrom,
        )

    def get_sequence(self, genome: Fasta) -> str:
        """Extract the nucleotide sequence for this region.

        On the negative strand each exon is reverse-complemented so
        that the returned string is in 5'->3' (translation) order.

        Parameters
        ----------
        genome : pyfaidx.Fasta
            Indexed FASTA handle keyed by chromosome name.

        Returns
        -------
        str
            Spliced nucleotide sequence.
        """
        chrom = genome[self.chrom]
        parts: list[str] = []
        if self.strand == "+":
            for iv in self.intervals:
                parts.append(str(chrom[iv.start : iv.end]))
        elif self.strand == "-":
            for iv in self.intervals[::-1]:
                parts.append(str(-chrom[iv.start : iv.end]))
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

        if self.strand == "+":
            s_ivs = self.intervals[::-1]
            o_ivs = other.intervals[::-1]
        else:
            s_ivs = self.intervals
            o_ivs = other.intervals

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
