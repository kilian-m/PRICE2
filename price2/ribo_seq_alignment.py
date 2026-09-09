"""Ribo-seq alignment representation.

Provides the :class:`RiboSeqAlignment` class, which wraps a single
mapped Ribo-seq fragment and exposes strand-aware coordinate helpers
used during cleavage-model estimation and read-to-ORF assignment.

Notes
-----
Coordinates follow the project convention: **0-based, half-open**
intervals stored in **chromosome order**.  Negative-strand alignments
are therefore stored in reverse-complement order relative to the
direction of translation.
"""

from __future__ import annotations

import HTSeq
import pysam

from price2.genomic_region import GenomicRegion


def five_prime_terminal_mismatch(
    aln: "pysam.AlignedSegment", is_minus: bool
) -> bool:
    """Return whether the read's 5'-most aligned base mismatches the reference.

    Used for ``--alignEndsType EndToEnd`` BAMs, where the untemplated
    reverse-transcription nucleotide is not soft-clipped but force-aligned as a
    single 5'-terminal mismatch.  The test reads the ``MD`` optional tag in
    ``O(1)``: a terminal mismatch shows up as a zero-length match run adjacent
    to the terminal reference base.  On the ``+`` strand the read's 5' end is
    the leftmost reference base, so ``MD`` begins ``0<base>``; on the ``-``
    strand it is the rightmost, so ``MD`` ends ``<base>0``.

    Parameters
    ----------
    aln : pysam.AlignedSegment
        The aligned read.  Must carry an ``MD`` tag (STAR
        ``--outSAMattributes ... MD``); absent it, ``False`` is returned.
    is_minus : bool
        ``True`` for reverse-strand reads.

    Returns
    -------
    bool
        ``True`` when the 5'-most aligned base is a mismatch.
    """
    try:
        md = aln.get_tag("MD")
    except KeyError:
        return False
    if is_minus:
        return len(md) >= 2 and md[-1] == "0" and md[-2].isalpha()
    return len(md) >= 2 and md[0] == "0" and md[1].isalpha()


def trim_five_prime_base(
    intervals: list[HTSeq.GenomicInterval], is_minus: bool
) -> list[HTSeq.GenomicInterval]:
    """Drop the single 5'-most reference base from a chromosome-ordered footprint.

    Reproduces what ``--alignEndsType Local`` does implicitly (the soft-clipped
    RT base never enters the footprint) for an EndToEnd read whose RT base was
    force-aligned.  On ``+`` strand the 5' end is the first interval's start;
    on ``-`` strand it is the last interval's end.  An interval reduced to
    length zero is dropped.

    Parameters
    ----------
    intervals : list[HTSeq.GenomicInterval]
        Footprint intervals in chromosome order.
    is_minus : bool
        ``True`` for reverse-strand reads.

    Returns
    -------
    list[HTSeq.GenomicInterval]
        A new list with one 5'-end base removed.
    """
    if not intervals:
        return intervals
    if is_minus:
        iv = intervals[-1]
        if iv.end - iv.start <= 1:
            return intervals[:-1]
        trimmed = HTSeq.GenomicInterval(iv.chrom, iv.start, iv.end - 1, iv.strand)
        return intervals[:-1] + [trimmed]
    iv = intervals[0]
    if iv.end - iv.start <= 1:
        return intervals[1:]
    trimmed = HTSeq.GenomicInterval(iv.chrom, iv.start + 1, iv.end, iv.strand)
    return [trimmed] + intervals[1:]


class RiboSeqAlignment:
    """A single mapped Ribo-seq read fragment.

    Wraps genomic position information for one alignment and exposes
    helpers for cleavage-model estimation and ORF assignment.

    Attributes
    ----------
    genomic_region : GenomicRegion
        Spliced genomic region covered by matching (``M``) cigar
        operations, stored in chromosome order.
    untemplated_addition : bool
        Whether a 1-nt soft-clipped untemplated addition was detected
        at the 5' end of the read (i.e. the end closest to the mRNA
        5' cap).
    mapping_positions : int
        Number of genomic loci the fragment maps to (``NH`` tag).
    read_count : int
        Collapsed read count; ``1`` for single alignments loaded
        directly from a BAM file.
    """

    genomic_region: GenomicRegion
    untemplated_addition: bool
    mapping_positions: int
    read_count: int

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_region(
        cls,
        genomic_region: GenomicRegion,
        untemplated_addition: bool,
        unique: bool,
        read_count: int,
    ) -> "RiboSeqAlignment":
        """Construct a :class:`RiboSeqAlignment` from pre-built attributes.

        Used by the vectorized loader in
        :meth:`~price2.locus.Locus.get_reads_from_db`.

        Parameters
        ----------
        genomic_region : GenomicRegion
            Spliced genomic region of the alignment, in chromosome order.
        untemplated_addition : bool
            Whether a 1-nt 5' untemplated addition was detected.
        unique : bool
            ``True`` when the fragment maps to exactly one locus
            (``mapping_positions`` is set to 1, otherwise 2).
        read_count : int
            Collapsed read count.

        Returns
        -------
        RiboSeqAlignment
            Fully initialised instance.
        """
        obj: RiboSeqAlignment = cls.__new__(cls)
        obj.genomic_region = genomic_region
        obj.untemplated_addition = untemplated_addition
        obj.mapping_positions = 1 if unique else 2
        obj.read_count = read_count
        return obj

    @classmethod
    def from_pysam(
        cls, aln: "pysam.AlignedSegment", end_to_end: bool = False
    ) -> "RiboSeqAlignment":
        """Construct a :class:`RiboSeqAlignment` from a pysam alignment.

        Parameters
        ----------
        aln : pysam.AlignedSegment
            A single aligned record returned by
            :meth:`pysam.AlignmentFile.fetch`.
        end_to_end : bool, optional
            When ``True`` the BAM was mapped with ``--alignEndsType EndToEnd``:
            the untemplated addition is recovered from the 5'-terminal mismatch
            (via the ``MD`` tag) rather than a soft-clip, and that base is
            trimmed off the footprint.  Defaults to ``False`` (soft-clip mode).

        Returns
        -------
        RiboSeqAlignment
            Fully initialised instance.
        """
        obj: RiboSeqAlignment = cls.__new__(cls)
        obj._init_from_pysam_alignment(aln, end_to_end=end_to_end)
        return obj

    def _init_from_pysam_alignment(
        self, aln: "pysam.AlignedSegment", end_to_end: bool = False
    ) -> None:
        """Initialise from a pysam AlignedSegment object.

        Reads the ``NH`` optional tag for multimapper count, detects the
        untemplated addition, and walks the CIGAR string to collect ``M``
        (match) intervals as :class:`HTSeq.GenomicInterval` objects.

        In the default soft-clip mode (``--alignEndsType Local``) the
        untemplated addition is a 1-nt 5'-end soft-clip.  When *end_to_end* is
        ``True`` (``--alignEndsType EndToEnd``) there are no soft-clips: the
        untemplated addition is instead read off the 5'-terminal mismatch and
        that base is trimmed from the footprint so the stored geometry matches
        the Local case.
        """
        try:
            self.mapping_positions = aln.get_tag("NH")
        except KeyError:
            self.mapping_positions = 1
        self.read_count = 1

        is_minus = aln.is_reverse
        strand = "-" if is_minus else "+"
        cigar = aln.cigartuples  # list of (op_code, length) or None

        if end_to_end:
            # EndToEnd: RT base is force-aligned as a 5'-terminal mismatch.
            self.untemplated_addition = five_prime_terminal_mismatch(aln, is_minus)
        elif not is_minus:
            # Local: 1-nt soft-clip at the 5' end (first cigar op on + strand).
            self.untemplated_addition = bool(
                cigar and cigar[0][0] == 4 and cigar[0][1] == 1
            )
        else:
            # Local: 1-nt soft-clip at the 5' end (last cigar op on - strand).
            self.untemplated_addition = bool(
                cigar and cigar[-1][0] == 4 and cigar[-1][1] == 1
            )

        # Walk CIGAR to collect M (op=0) intervals in reference coordinates.
        # ops that consume reference: M(0), D(2), N(3); skip I(1), S(4), H(5), P(6).
        chrom = aln.reference_name
        pos = aln.reference_start
        intervals: list[HTSeq.GenomicInterval] = []
        for op, length in (cigar or []):
            if op == 0:  # M
                intervals.append(
                    HTSeq.GenomicInterval(chrom, pos, pos + length, strand)
                )
                pos += length
            elif op in (2, 3):  # D / N
                pos += length

        # Under EndToEnd the force-aligned RT base is part of the M footprint;
        # drop it so the 5' end matches what Local's soft-clip would have left.
        if end_to_end and self.untemplated_addition:
            intervals = trim_five_prime_base(intervals, is_minus)

        self.genomic_region = GenomicRegion(intervals)

    # ------------------------------------------------------------------
    # Special methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"RiboSeqAlignment({self.genomic_region})"

    def __len__(self) -> int:
        return len(self.genomic_region)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def unique(self) -> bool:
        """Return ``True`` if the fragment maps to exactly one locus.

        Returns
        -------
        bool
            ``True`` when :attr:`mapping_positions` equals 1.
        """
        return self.mapping_positions == 1

