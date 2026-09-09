"""Reading alignments from BAM files.

The small set of BAM conventions PRICE2 relies on, in one place: how a file
handle is cached inside a worker process, which alignments count as mapped
and as uniquely mapping, and how the ``MD`` tag is detected.
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
