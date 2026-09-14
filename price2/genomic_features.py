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

import dataclasses
import enum

import HTSeq

from price2.genomic_region import GenomicRegion


class RGRType(str, enum.Enum):
    """Type of a :class:`ReadGeneratingRegion`.

    Inherits from ``str`` so that it renders as its value in the output
    tables and a plain string is accepted at construction; code asks
    :attr:`ReadGeneratingRegion.is_orf` rather than comparing strings.
    """

    ORF = "ORF"
    NOISE = "NOISE"


class Transcript:
    """An annotated transcript built from a GTF feature.

    Stores transcript metadata, the exon and CDS regions collected from the
    GTF features, and the associated :class:`ReadGeneratingRegion` objects.
    The GTF parser adds the exon and CDS features one by one
    (:meth:`add_exon`, :meth:`add_cds_region`) and then calls
    :meth:`finalize`, which builds the two regions; :attr:`exons` and
    :attr:`cds` are ``None`` until then.

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
        Multi-exonic region built from the exon features.
    cds : GenomicRegion | None
        The annotated CDS, or ``None`` when there is none (or its length
        is not a multiple of three).
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
        self.exons: GenomicRegion | None = None
        self.cds: GenomicRegion | None = None
        self.coding_length: int = 0
        self.exon_length: int = 0
        self.orf_set: set[ReadGeneratingRegion] = set()
        self.rgr_set: set[ReadGeneratingRegion] = set()
        self.biotype: str = feature.attr.get(
            "transcript_biotype",
            feature.attr.get("transcript_type", "unknown"),
        )
        self.annotated_cds_iv: tuple[int, int] | None = None
        # The GTF features, until ``finalize`` builds the regions.
        self._exon_ivs: list[HTSeq.GenomicInterval] = []
        self._cds_ivs: list[HTSeq.GenomicInterval] = []

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
        """Add an exon feature to this transcript.

        Parameters
        ----------
        exon : HTSeq.features.GenomicFeature
            An exon-level GTF feature as parsed by HTSeq.
        """
        self._exon_ivs.append(exon.iv)
        self.exon_length += exon.iv.length

    def add_cds_region(self, region: HTSeq.features.GenomicFeature) -> None:
        """Add a CDS feature to this transcript.

        Parameters
        ----------
        region : HTSeq.features.GenomicFeature
            A ``CDS`` GTF feature as parsed by HTSeq.
        """
        self._cds_ivs.append(region.iv)
        self.coding_length += region.iv.length

    def finalize(self) -> None:
        """Build :attr:`exons` and :attr:`cds` once every feature has been added.

        The features are sorted into chromosome order (a ``-`` strand GTF
        lists them in translation order).  A CDS whose length is not a
        multiple of three is discarded as unannotated; otherwise
        :attr:`annotated_cds_iv` receives its position in spliced transcript
        coordinates.
        """
        chrom, strand = self.iv.chrom, self.iv.strand
        by_start = sorted(self._exon_ivs, key=lambda iv: iv.start)
        self.exons = GenomicRegion(by_start, chrom=chrom, strand=strand)
        if self._cds_ivs and self.coding_length % 3 == 0:
            by_start = sorted(self._cds_ivs, key=lambda iv: iv.start)
            self.cds = GenomicRegion(by_start, chrom=chrom, strand=strand)
            self.annotated_cds_iv = self.exons.map_to_local(self.cds)
        else:
            self.cds = None
            self.coding_length = 0
            self.annotated_cds_iv = None
        del self._exon_ivs, self._cds_ivs

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


@dataclasses.dataclass(eq=False, repr=False)
class ReadGeneratingRegion:
    """A candidate translated region that can generate Ribo-seq reads.

    An RGR is either an ORF (type ``"ORF"``) or a background noise
    region (type ``"NOISE"``).  It stores both genomic and
    transcript-local coordinates and participates in the group-LASSO
    deconvolution.

    The genomic coordinates are derived from the parent transcript's exon
    structure when the region is created; :attr:`orf_type` and
    :attr:`index` are assigned later by the locus that owns the region.

    **Identity.**  Two RGRs are equal when they have the same type and the
    same coding body (:attr:`genomic_region`); the ``id``, the parent
    transcript, the ORF type and the index take no part.  This is what
    lets :func:`~price2.orf_candidates.make_rgrs` deduplicate the same ORF
    found on several transcripts of a locus.
    """

    #: ``RGRType.ORF`` or ``RGRType.NOISE`` (a plain string is coerced).
    type: RGRType
    #: The parent transcript.
    transcript: Transcript
    #: Unique identifier for this RGR.
    id: str
    #: Spliced transcript coordinates ``(start, end)`` of the coding body,
    #: 0-based half-open, stop codon excluded.
    iv_on_transcript: tuple[int, int]
    #: ORF type classification (e.g. ``'cORF'``, ``'uORF'``), or ``None``
    #: for NOISE regions or before classification.
    orf_type: str | None = None
    #: Position in ``Locus.rgrs``, which addresses the region's
    #: design-matrix column block and its row of the activity matrix;
    #: ``-1`` until the locus assigns it.
    index: int = -1
    #: The coding body (stop codon excluded) as a genomic region.
    genomic_region: GenomicRegion = dataclasses.field(init=False)
    #: The coding body including the stop codon (ORFs only; equals
    #: :attr:`genomic_region` for NOISE regions).
    full_genomic_region: GenomicRegion = dataclasses.field(init=False)
    #: Distance in nt from the RGR start to the transcript 5' end.
    dist_to_transcript_start: int = dataclasses.field(init=False)
    #: Distance in nt from the RGR end to the transcript 3' end.
    dist_to_transcript_end: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.type = RGRType(self.type)
        exons = self.transcript.exons
        start, end = self.iv_on_transcript
        self.genomic_region = exons.map_to_global(self.iv_on_transcript)
        if self.type is RGRType.ORF:
            self.full_genomic_region = exons.map_to_global((start, end + 3))
        else:
            self.full_genomic_region = self.genomic_region
        self.dist_to_transcript_start = start
        self.dist_to_transcript_end = self.transcript.exon_length - end

    @property
    def is_orf(self) -> bool:
        """Whether this is an ORF candidate (else a NOISE region)."""
        return self.type is RGRType.ORF

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReadGeneratingRegion):
            return NotImplemented
        return (self.type == other.type) and (
            self.genomic_region == other.genomic_region
        )

    def __hash__(self) -> int:
        return hash((self.type, self.genomic_region))

    def __len__(self) -> int:
        return len(self.genomic_region)

    def __repr__(self) -> str:
        return f"{self.type} {self.id} on region {self.genomic_region}"

