"""Genomic features for Ribo-seq ORF deconvolution.

Defines :class:`Transcript` and :class:`ReadGeneratingRegion` (RGR),
the two core objects that represent annotated transcripts and candidate
translated regions used throughout the PRICE2 pipeline.

Notes
-----
All coordinates follow the project convention: **0-based, half-open**
intervals stored in **chromosome order**.
"""

from __future__ import annotations

import enum

import HTSeq

from price2.genomic_region import GenomicRegion


class RGRType(str, enum.Enum):
    """Type of a :class:`ReadGeneratingRegion`.

    Inherits from ``str`` so that comparisons with plain string
    literals (e.g. ``rgr.type == "ORF"``) keep working.
    """

    ORF = "ORF"
    NOISE = "NOISE"


class Transcript:
    """An annotated transcript built from a GTF feature.

    Stores transcript metadata and accumulates exonic intervals, UTR
    and CDS sub-regions, and associated :class:`ReadGeneratingRegion`
    objects.

    Attributes
    ----------
    type : str
        Always ``"transcript"``.
    id : str
        Ensembl transcript identifier.
    gene_id : str
        Ensembl gene identifier.
    biotype : str
        Transcript biotype (e.g. ``"protein_coding"``), read from the
        ``transcript_biotype`` (Ensembl) or ``transcript_type`` (GENCODE)
        GTF attribute, or ``"unknown"`` if neither is present.
    iv : HTSeq.GenomicInterval
        Genomic interval of the whole transcript locus.
    exons : GenomicRegion
        Multi-exonic region built from individual exon features.
    exon_length : int
        Total spliced length (bp).
    coding_length : int
        Total CDS length (bp).
    annotated_cds_iv : tuple[int, int] | None
        Start and end of the annotated CDS in transcript
        (spliced) coordinates, or ``None`` if no CDS is annotated.
    orf_set : set[ReadGeneratingRegion]
        ORF-type RGRs associated with this transcript.
    rgr_set : set[ReadGeneratingRegion]
        All RGRs (ORFs and NOISE) associated with this transcript.
    """

    type: str = "transcript"

    def __init__(self, feature: HTSeq.features.GenomicFeature) -> None:
        """Initialise from a GTF ``transcript`` feature.

        Parameters
        ----------
        feature : HTSeq.features.GenomicFeature
            A transcript-level GTF feature as parsed by HTSeq.
        """
        self.id: str = feature.attr["transcript_id"]
        self.gene_id: str = feature.attr["gene_id"]
        self.iv: HTSeq.GenomicInterval = feature.iv
        self.exons: GenomicRegion = GenomicRegion(
            [], chrom=feature.iv.chrom, strand=feature.iv.strand
        )
        self.coding_length: int = 0
        self.exon_length: int = 0
        self.orf_set: set[ReadGeneratingRegion] = set()
        self.rgr_set: set[ReadGeneratingRegion] = set()
        self.biotype: str = feature.attr.get(
            "transcript_biotype",
            feature.attr.get("transcript_type", "unknown"),
        )
        self.annotated_cds_iv: tuple[int, int] | None = None

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transcript):
            return NotImplemented
        return self.id == other.id

    def __len__(self) -> int:
        return len(self.exons)

    def __repr__(self) -> str:
        return f"Transcript({self.id!r})"

    def add_exon(self, exon: HTSeq.features.GenomicFeature) -> None:
        """Add an exon interval to this transcript.

        Parameters
        ----------
        exon : HTSeq.features.GenomicFeature
            An exon-level GTF feature as parsed by HTSeq.
        """
        self.exons.add_interval(exon.iv)
        self.exon_length += exon.iv.length

    def add_cds_region(self, region: HTSeq.features.GenomicFeature) -> None:
        """Add a CDS sub-region to this transcript.

        Accumulates intervals for ``CDS`` features.

        Parameters
        ----------
        region : HTSeq.features.GenomicFeature
            A sub-transcript GTF feature (``CDS``) as parsed by HTSeq.
            or ``three_prime_utr``) as parsed by HTSeq.
        """

        if not hasattr(self, "_cds"):
            self._cds: GenomicRegion = GenomicRegion(
                [], chrom=region.iv.chrom, strand=region.iv.strand
            )
        self._cds.add_interval(region.iv)
        self.coding_length += region.iv.length

    def cds_regions_to_cds_intervals(self) -> None:
        """Compute annotated CDS position in transcript coordinates.

        Populates :attr:`annotated_cds_iv` with a ``(start, end)``
        tuple in spliced transcript coordinates, or ``None`` if no CDS
        has been added to this transcript.
        """
        try:
            if len(self._cds) % 3 != 0:
                del self._cds
                self.coding_length = 0
                self.annotated_cds_iv = None
            else:
                self.annotated_cds_iv = self.exons.map_to_local(self._cds)
        except AttributeError:
            self.annotated_cds_iv = None

    def add_orf(self, orf: ReadGeneratingRegion) -> None:
        """Register an ORF-type RGR with this transcript.

        Parameters
        ----------
        orf : ReadGeneratingRegion
            The ORF to associate with this transcript.
        """
        self.orf_set.add(orf)
        self.rgr_set.add(orf)

    def update_with_filtered_orfs(self, rgr_set: set[ReadGeneratingRegion]) -> None:
        """Restrict :attr:`orf_set` and :attr:`rgr_set` to the survivors of a filter.

        Parameters
        ----------
        rgr_set : set[ReadGeneratingRegion]
            The surviving RGRs after a filtering step.
        """
        self.orf_set = self.orf_set & rgr_set
        self.rgr_set = self.rgr_set & rgr_set


class ReadGeneratingRegion:
    """A candidate translated region that can generate Ribo-seq reads.

    An RGR is either an ORF (type ``"ORF"``) or a background noise
    region (type ``"NOISE"``).  It stores both genomic and
    transcript-local coordinates and participates in the group-LASSO
    deconvolution.

    Attributes
    ----------
    type : RGRType
        ``RGRType.ORF`` or ``RGRType.NOISE``.
    id : str
        Unique identifier for this RGR.
    transcript : Transcript
        The parent transcript.
    transcript_id : str
        Identifier of the parent transcript.
    genomic_region : GenomicRegion
        The coding body (stop codon excluded) as a genomic region.
    full_genomic_region : GenomicRegion
        The coding body including the stop codon (ORFs only;
        equals :attr:`genomic_region` for NOISE regions).
    iv_on_transcript : tuple[int, int]
        Spliced transcript coordinates ``(start, end)`` of the coding
        body, 0-based half-open.
    dist_to_transcript_start : int
        Distance in nt from the RGR start to the transcript 5' end.
    dist_to_transcript_end : int
        Distance in nt from the RGR end to the transcript 3' end.
    read_count : int
        Observed read count (populated externally).
    orf_type : str | None
        ORF type classification (e.g. ``'cORF'``, ``'uORF'``), or
        ``None`` for NOISE regions or before classification.
    """

    def __init__(
        self,
        type: RGRType | str,
        transcript: Transcript,
        id: str,
        iv_on_transcript: tuple[int, int],
    ) -> None:
        """Create a ReadGeneratingRegion.

        Genomic coordinates are derived from the parent transcript's exon
        structure.

        Parameters
        ----------
        type : RGRType | str
            ``RGRType.ORF`` or ``RGRType.NOISE`` (plain strings are
            also accepted for backward compatibility).
        transcript : Transcript
            The parent transcript.
        id : str
            Unique identifier for this RGR.
        iv_on_transcript : tuple[int, int]
            Spliced transcript coordinates ``(start, end)``, 0-based
            half-open.
        """
        self.type: RGRType = RGRType(type)
        self.read_count: int = 0
        self.id: str = id
        self.transcript: Transcript = transcript
        self.orf_type: str | None = None

        self.genomic_region: GenomicRegion = transcript.exons.map_to_global(
            iv_on_transcript
        )
        if self.type == "ORF":
            self.full_genomic_region: GenomicRegion = (
                transcript.exons.map_to_global(
                    (iv_on_transcript[0], iv_on_transcript[1] + 3)
                )
            )
        else:
            self.full_genomic_region = self.genomic_region
        self.transcript_id: str = transcript.id
        self.iv_on_transcript: tuple[int, int] = iv_on_transcript
        self.dist_to_transcript_start: int = iv_on_transcript[0]
        self.dist_to_transcript_end: int = (
            transcript.exon_length - iv_on_transcript[1]
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReadGeneratingRegion):
            return NotImplemented
        return (self.type == other.type) and (
            self.genomic_region == other.genomic_region
        )

    def __hash__(self) -> int:
        return hash(self.genomic_region)

    def __len__(self) -> int:
        return len(self.genomic_region)

    def __repr__(self) -> str:
        return f"{self.type} {self.id} on region {self.genomic_region}"

