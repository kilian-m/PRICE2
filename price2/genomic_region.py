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

from collections.abc import Sequence
from typing import Literal, NamedTuple

from pyfaidx import Fasta

Strand = Literal["+", "-"]
"""Genomic strand: either ``'+'`` (forward) or ``'-'`` (reverse)."""


class Interval(NamedTuple):
    """A ``[start, end)`` stretch of a region's chromosome, 0-based half-open.

    The chromosome and strand live on the :class:`GenomicRegion`; the
    interval is a plain pair, hashable and cheap to build and to pickle.
    """

    start: int
    end: int


def _interval(iv: object) -> Interval:
    """*iv* as an :class:`Interval`: a pair, or anything with ``start``/``end``
    (an ``HTSeq.GenomicInterval``, say)."""
    if isinstance(iv, Interval):
        return iv
    if isinstance(iv, (tuple, list)):
        return Interval(int(iv[0]), int(iv[1]))
    return Interval(iv.start, iv.end)


class GenomicRegion:
    """A possibly multi-exonic genomic region on a single chromosome.

    Intervals are stored in chromosome order (ascending start position).
    All coordinates are 0-based, half-open.  A region is immutable once
    built: its intervals are a tuple and its hash is fixed at construction,
    so it can serve as a dictionary key or set member from the start.

    Attributes
    ----------
    chrom : str
        Chromosome / reference sequence name.
    strand : Strand
        ``'+'`` or ``'-'``.
    intervals : tuple[Interval, ...]
        Exonic intervals in chromosome order.
    length : int
        Total spliced length (sum of exon lengths).
    """

    chrom: str
    strand: Strand
    intervals: tuple[Interval, ...]
    length: int

    def __init__(
        self,
        intervals: Sequence[object],
        chrom: str | None = None,
        strand: Strand | None = None,
    ) -> None:
        """Create a ``GenomicRegion``.

        Parameters
        ----------
        intervals : sequence
            ``(start, end)`` pairs in chromosome order, or objects carrying
            ``start`` and ``end`` such as ``HTSeq.GenomicInterval``.
        chrom : str | None
            Chromosome name; inferred from the first interval when omitted,
            which then has to carry it (an HTSeq interval does).
        strand : Strand | None
            Strand; inferred like *chrom* when omitted.

        Raises
        ------
        ValueError
            If the intervals overlap or are out of chromosome order.
        """
        self.chrom = chrom if chrom else intervals[0].chrom
        self.strand = strand if strand else intervals[0].strand
        self.intervals = tuple(_interval(iv) for iv in intervals)

        for i in range(1, len(self.intervals)):
            if self.intervals[i - 1].end > self.intervals[i].start:
                raise ValueError(
                    "Intervals must be non-overlapping and in chromosome order."
                )

        self.length = sum(iv.end - iv.start for iv in self.intervals)
        self.hash = hash((self.strand, self.chrom, self.intervals))

    def __setstate__(self, state: dict) -> None:
        """Restore a pickle; earlier releases stored HTSeq intervals in a list.

        The hash is recomputed rather than restored: it involves string
        hashes, which differ between interpreter processes.
        """
        self.__dict__.update(state)
        self.intervals = tuple(_interval(iv) for iv in self.intervals)
        self.hash = hash((self.strand, self.chrom, self.intervals))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GenomicRegion):
            return False
        return (
            self.chrom == other.chrom
            and self.intervals == other.intervals
            and self.strand == other.strand
        )

    def __hash__(self) -> int:
        return self.hash

    def __str__(self) -> str:
        intervals_str = "|".join(f"{iv.start}-{iv.end}" for iv in self.intervals)
        return f"{self.chrom}{self.strand}:{intervals_str}"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return self.length

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
            If *other* is ``None`` or has a different chromosome or strand or
            is not fully contained in *self*.  Callers that expect the
            failure — every "does this read fit this transcript?" loop —
            use :meth:`try_map_to_local` instead.
        """
        if other is None:
            raise ValueError("Cannot map None region to local coordinates.")
        span = self._map_to_local(other)
        if isinstance(span, str):
            raise ValueError(span)
        return span

    def try_map_to_local(self, other: GenomicRegion) -> tuple[int, int] | None:
        """:meth:`map_to_local`, returning ``None`` where it would raise.

        For the loops that test a read against every candidate transcript:
        most candidates fail, and an exception per failure costs more than
        the mapping itself.
        """
        span = self._map_to_local(other)
        return None if isinstance(span, str) else span

    def _map_to_local(self, other: GenomicRegion) -> tuple[int, int] | str:
        """The span of *other* in *self*, or the reason there is none."""
        if self.chrom != other.chrom or self.strand != other.strand:
            return "Cannot map region with different chromosome or strand."

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
                return "Other region extends beyond the bounds of this region."

            x, y = self.intervals[j].start, self.intervals[j].end

            # other-interval must be fully contained within this self-interval.
            if not (x <= a and b <= y):
                return "Other region is not fully contained within this region."

            # Contiguity check: no self-intervals were skipped between matches.
            if prev_j is not None and j > prev_j + 1:
                return "Other region spans a junction not present in this region."

            # Same-exon junction: consecutive read exons within one
            # self-interval means a spurious splice inside an exon.
            if prev_j is not None and j == prev_j:
                return "Read has a splice junction inside a reference exon."

            # Junction boundary check: when consecutive other-intervals
            # map to consecutive self-intervals, the splice sites must
            # align exactly.
            if prev_j is not None and j == prev_j + 1:
                if prev_other_end != self.intervals[prev_j].end or a != x:
                    return "Read junction does not match reference exon boundary."

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
            iv_len = interval.end - interval.start
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

        return GenomicRegion(result_ivs, chrom=self.chrom, strand=self.strand)

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
        if self.strand == "+":
            parts = [str(chrom[iv.start : iv.end]) for iv in self.intervals]
        else:
            parts = [str(-chrom[iv.start : iv.end]) for iv in self.intervals[::-1]]
        return "".join(parts)

    def _from_stop(self) -> list[tuple[int, int]]:
        """The intervals from the 3' end backwards, as if on the ``+`` strand.

        Negating the coordinates of a ``-`` strand region turns it into a
        ``+`` strand region read in the opposite direction, so the strand
        cases of :meth:`contains_to_stop` collapse into one.
        """
        if self.strand == "+":
            return [(iv.start, iv.end) for iv in self.intervals[::-1]]
        return [(-iv.end, -iv.start) for iv in self.intervals]

    def contains_to_stop(self, other: GenomicRegion) -> bool:
        """Check whether *other* is contained in *self* sharing the same stop end.

        Seen from the 3' end, every interval of *other* but its 5'-most one
        has to be an interval of *self*, and that 5'-most one has to be the
        3' part of its partner: *other* may start later than *self* but
        must otherwise follow its splicing.

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

        o_ivs = other._from_stop()
        for i, (s_iv, o_iv) in enumerate(zip(self._from_stop(), o_ivs)):
            if s_iv == o_iv:
                continue
            if i + 1 < len(o_ivs):
                return False
            if s_iv[1] != o_iv[1] or s_iv[0] > o_iv[0]:
                return False
        return True
