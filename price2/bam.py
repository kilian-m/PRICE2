"""Reading alignments from BAM files.

The small set of BAM conventions PRICE2 relies on, in one place: how a file
handle is cached inside a worker process, which alignments count as mapped
and as uniquely mapping, how the untemplated 5' addition is recognised under
the two STAR end-alignment modes and what a read's footprint is once it has
been removed, and how the ``MD`` tag is detected.
"""

from __future__ import annotations

from collections.abc import Iterator

import pysam

#: Per-process cache of open BAM handles, keyed by path, so that a worker
#: parses each index only once.
_HANDLES: dict[str, pysam.AlignmentFile] = {}


def cached_alignment_file(path: str, *, exclusive: bool = False) -> pysam.AlignmentFile:
    """Return this process's open handle for *path*, opening it once.

    Parameters
    ----------
    path : str
        Path to a coordinate-sorted, indexed BAM file.
    exclusive : bool, optional
        Close every other cached handle first.  Workers that sweep one file
        at a time use this to keep a single handle open.
    """
    handle = _HANDLES.get(path)
    if handle is None:
        if exclusive:
            for stale in _HANDLES.values():
                stale.close()
            _HANDLES.clear()
        handle = pysam.AlignmentFile(path, "rb")
        _HANDLES[path] = handle
    return handle


def is_unique(aln: pysam.AlignedSegment) -> bool:
    """Whether the alignment is its read's only one (``NH == 1`` or no ``NH``)."""
    try:
        return aln.get_tag("NH") == 1
    except KeyError:
        return True


def iter_mapped(
    bam: pysam.AlignmentFile, region: tuple[str, int, int] | None = None
) -> Iterator[pysam.AlignedSegment]:
    """Yield the mapped alignments of *bam*, optionally of one region.

    Parameters
    ----------
    bam : pysam.AlignmentFile
        Open BAM file; must be indexed when *region* is given.
    region : tuple[str, int, int], optional
        ``(contig, start, end)``.  Only alignments whose leftmost mapped
        base lies in ``[start, end)`` are yielded, so the alignments of a
        set of regions tiling the genome partition those of the file.
    """
    if region is None:
        for aln in bam.fetch(until_eof=True):
            if not aln.is_unmapped:
                yield aln
        return
    contig, lo, hi = region
    for aln in bam.fetch(contig, lo, hi):
        if not aln.is_unmapped and lo <= aln.reference_start < hi:
            yield aln


def five_prime_terminal_mismatch(aln: pysam.AlignedSegment, is_minus: bool) -> bool:
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
    """
    try:
        md = aln.get_tag("MD")
    except KeyError:
        return False
    if is_minus:
        return len(md) >= 2 and md[-1] == "0" and md[-2].isalpha()
    return len(md) >= 2 and md[0] == "0" and md[1].isalpha()


def trim_five_prime_base(
    blocks: list[tuple[int, int]], is_minus: bool
) -> list[tuple[int, int]]:
    """Drop the single 5'-most reference base from a read's reference blocks.

    The EndToEnd counterpart of Local's implicit soft-clip removal: when the
    force-aligned RT base is recovered as a 5'-terminal mismatch it must be
    stripped from the footprint so the stored geometry matches the Local
    case.  On the ``+`` strand the 5' end is the first block's start; on the
    ``-`` strand it is the last block's end.  A block reduced to length zero
    is dropped.

    Parameters
    ----------
    blocks : list of (int, int)
        Reference blocks in chromosome order, as
        :meth:`pysam.AlignedSegment.get_blocks` returns them.
    is_minus : bool
        ``True`` for reverse-strand reads.

    Returns
    -------
    list of (int, int)
        A new block list with one 5'-end base removed.
    """
    if not blocks:
        return blocks
    if is_minus:
        start, end = blocks[-1]
        if end - start <= 1:
            return blocks[:-1]
        return blocks[:-1] + [(start, end - 1)]
    start, end = blocks[0]
    if end - start <= 1:
        return blocks[1:]
    return [(start + 1, end)] + blocks[1:]


def footprint(
    aln: pysam.AlignedSegment, end_to_end: bool = False
) -> tuple[list[tuple[int, int]], bool] | None:
    """The reference blocks of an alignment and its untemplated-addition flag.

    This is the one place that decides what a Ribo-seq read's footprint is.
    In the default soft-clip mode (STAR ``--alignEndsType Local``) the
    untemplated RT addition is a 1-nt soft-clip at the read's 5' end and
    never enters the aligned blocks.  Under ``EndToEnd`` there are no
    soft-clips: the addition is force-aligned as a 5'-terminal mismatch,
    recognised through the ``MD`` tag (:func:`five_prime_terminal_mismatch`)
    and trimmed off the blocks (:func:`trim_five_prime_base`) so that both
    modes store the same geometry.

    Parameters
    ----------
    aln : pysam.AlignedSegment
        A mapped record.
    end_to_end : bool, optional
        Whether the BAM was mapped with ``--alignEndsType EndToEnd``.

    Returns
    -------
    tuple[list[tuple[int, int]], bool] or None
        The blocks in chromosome order and whether the read carried an
        untemplated addition; ``None`` when nothing of the read is aligned
        (no CIGAR, or an EndToEnd read that consisted of its RT base only).
    """
    cigar = aln.cigartuples
    if not cigar:
        return None
    blocks = aln.get_blocks()
    is_minus = aln.is_reverse
    if end_to_end:
        untemplated_addition = five_prime_terminal_mismatch(aln, is_minus)
        if untemplated_addition:
            blocks = trim_five_prime_base(blocks, is_minus)
    elif is_minus:
        # The 5' end of a reverse-strand read is its last CIGAR operation.
        untemplated_addition = cigar[-1] == (4, 1)
    else:
        untemplated_addition = cigar[0] == (4, 1)
    if not blocks:
        return None
    return blocks, untemplated_addition


def first_mapped_read_has_md(bam_path: str) -> bool | None:
    """Return whether the first mapped read of a BAM carries an ``MD`` tag.

    ``None`` when the file cannot be opened or contains no mapped reads.
    STAR writes ``MD`` for every alignment or for none, so the first mapped
    read is representative of the whole file.
    """
    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for aln in iter_mapped(bam):
                return aln.has_tag("MD")
    except (OSError, ValueError):
        return None
    return None
