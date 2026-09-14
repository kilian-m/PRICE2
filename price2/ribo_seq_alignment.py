"""Ribo-seq alignment representation.

:class:`RiboSeqAlignment` is one mapped Ribo-seq fragment as the worker-side
code sees it: its spliced footprint as a :class:`~price2.genomic_region.GenomicRegion`,
whether it carried a 5' untemplated addition, whether it maps uniquely, and
how many identical reads it stands for.  How those facts are read off a BAM
record is the business of :func:`price2.bam.footprint`.

Notes
-----
Coordinates follow the project convention: **0-based, half-open**
intervals stored in **chromosome order**.  Negative-strand alignments
are therefore stored in reverse-complement order relative to the
direction of translation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pysam

from price2 import bam
from price2.genomic_region import GenomicRegion


@dataclass(slots=True, eq=False)
class RiboSeqAlignment:
    """A single mapped Ribo-seq read fragment.

    Attributes
    ----------
    genomic_region : GenomicRegion
        Spliced genomic region covered by the aligned blocks, stored in
        chromosome order (the untemplated addition, if any, is not part of
        it).
    untemplated_addition : bool
        Whether a 1-nt untemplated addition was detected at the 5' end of
        the read (i.e. the end closest to the mRNA 5' cap).
    unique : bool
        Whether the fragment maps to exactly one genomic locus (``NH == 1``
        or no ``NH`` tag).
    read_count : int
        Collapsed read count; ``1`` for a single alignment loaded directly
        from a BAM file.
    """

    genomic_region: GenomicRegion
    untemplated_addition: bool
    unique: bool
    read_count: int = 1

    @classmethod
    def from_pysam(
        cls, aln: pysam.AlignedSegment, end_to_end: bool = False
    ) -> RiboSeqAlignment | None:
        """Build the alignment of one BAM record, or ``None`` if it has no footprint.

        Parameters
        ----------
        aln : pysam.AlignedSegment
            A mapped record returned by :meth:`pysam.AlignmentFile.fetch`.
        end_to_end : bool, optional
            Whether the BAM was mapped with ``--alignEndsType EndToEnd``; see
            :func:`price2.bam.footprint`.
        """
        found = bam.footprint(aln, end_to_end)
        if found is None:
            return None
        blocks, untemplated_addition = found
        chrom = aln.reference_name
        strand = "-" if aln.is_reverse else "+"
        region = GenomicRegion(blocks, chrom=chrom, strand=strand)
        return cls(region, untemplated_addition, bam.is_unique(aln))

    def __len__(self) -> int:
        return len(self.genomic_region)
