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

import HTSeq
from HTSeq import GenomicInterval

from price2.genomic_region import GenomicRegion


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
        Transcript biotype (e.g. ``"protein_coding"``).
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
        self.biotype: str = feature.attr.get("transcript_biotype", "unknown")
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

    def add_region(self, region: HTSeq.features.GenomicFeature) -> None:
        """Add a UTR or CDS sub-region to this transcript.

        Accumulates intervals for ``five_prime_utr``, ``CDS``, and
        ``three_prime_utr`` features.

        Parameters
        ----------
        region : HTSeq.features.GenomicFeature
            A sub-transcript GTF feature (``five_prime_utr``, ``CDS``,
            or ``three_prime_utr``) as parsed by HTSeq.
        """
        if region.type == "five_prime_utr":
            if not hasattr(self, "five_prime_utr"):
                self.five_prime_utr: GenomicRegion = GenomicRegion(
                    [], chrom=region.iv.chrom, strand=region.iv.strand
                )
            self.five_prime_utr.add_interval(region.iv)
        elif region.type == "CDS":
            if not hasattr(self, "cds"):
                self.cds: GenomicRegion = GenomicRegion(
                    [], chrom=region.iv.chrom, strand=region.iv.strand
                )
            self.cds.add_interval(region.iv)
            self.coding_length += region.iv.length
        elif region.type == "three_prime_utr":
            if not hasattr(self, "three_prime_utr"):
                self.three_prime_utr: GenomicRegion = GenomicRegion(
                    [], chrom=region.iv.chrom, strand=region.iv.strand
                )
            self.three_prime_utr.add_interval(region.iv)

    def cds_regions_to_cds_intervals(self) -> None:
        """Compute annotated CDS position in transcript coordinates.

        Populates :attr:`annotated_cds_iv` with a ``(start, end)``
        tuple in spliced transcript coordinates, or ``None`` if no CDS
        has been added to this transcript.
        """
        try:
            self.annotated_cds_iv = self.exons.induce(self.cds)
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

    def update_with_filtered_orfs(
        self, rgr_set: set[ReadGeneratingRegion]
    ) -> None:
        """Restrict RGRs to the given set and rebuild interval indices.

        After filtering, HTSeq ``GenomicArrayOfSets`` interval indices
        are rebuilt for both ORF-only and all-RGR lookups.

        Parameters
        ----------
        rgr_set : set[ReadGeneratingRegion]
            The surviving RGRs after a filtering step.
        """
        self.orf_set = self.orf_set & rgr_set
        self.rgr_set = self.rgr_set & rgr_set

        self.orf_intervals: tuple[
            HTSeq.GenomicArrayOfSets,
            HTSeq.GenomicArrayOfSets,
            HTSeq.GenomicArrayOfSets,
        ] = (
            HTSeq.GenomicArrayOfSets("auto", stranded=False),
            HTSeq.GenomicArrayOfSets("auto", stranded=False),
            HTSeq.GenomicArrayOfSets("auto", stranded=False),
        )
        for orf in self.orf_set:
            iv = GenomicInterval(
                ".", orf.iv_on_transcript[0], orf.iv_on_transcript[1], "."
            )
            self.orf_intervals[iv.start % 3][iv] += orf

        self.rgr_intervals: tuple[
            HTSeq.GenomicArrayOfSets,
            HTSeq.GenomicArrayOfSets,
            HTSeq.GenomicArrayOfSets,
        ] = (
            HTSeq.GenomicArrayOfSets("auto", stranded=False),
            HTSeq.GenomicArrayOfSets("auto", stranded=False),
            HTSeq.GenomicArrayOfSets("auto", stranded=False),
        )
        for rgr in self.rgr_set:
            iv = GenomicInterval(
                ".", rgr.iv_on_transcript[0], rgr.iv_on_transcript[1], "."
            )
            if rgr.type == "ORF":
                self.rgr_intervals[iv.start % 3][iv] += rgr
            else:
                for i in range(3):
                    self.rgr_intervals[i][iv] += rgr


class ReadGeneratingRegion:
    """A candidate translated region that can generate Ribo-seq reads.

    An RGR is either an ORF (type ``"ORF"``) or a background noise
    region (type ``"NOISE"``).  It stores both genomic and
    transcript-local coordinates and participates in the group-LASSO
    deconvolution.

    Attributes
    ----------
    type : str
        ``"ORF"`` or ``"NOISE"``.
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
    """

    def __init__(
        self,
        type: str,
        transcript: Transcript,
        id: str,
        iv_on_transcript: tuple[int, int] | None = None,
        genomic_region: GenomicRegion | None = None,
    ) -> None:
        """Create a ReadGeneratingRegion.

        Exactly one of *iv_on_transcript* or *genomic_region* must be
        provided.  If *iv_on_transcript* is given, genomic coordinates
        are derived from the parent transcript's exon structure.

        Parameters
        ----------
        type : str
            ``"ORF"`` or ``"NOISE"``.
        transcript : Transcript
            The parent transcript.
        id : str
            Unique identifier for this RGR.
        iv_on_transcript : tuple[int, int] | None
            Spliced transcript coordinates ``(start, end)``, 0-based
            half-open.
        genomic_region : GenomicRegion | None
            Pre-computed genomic region (overrides coordinate mapping).

        Raises
        ------
        ValueError
            If neither *iv_on_transcript* nor *genomic_region* is given.
        """
        if iv_on_transcript is None and genomic_region is None:
            raise ValueError(
                "Either iv_on_transcript or genomic_region must be provided."
            )
        self.type: str = type
        self.read_count: int = 0
        self.id: str = id
        self.transcript: Transcript = transcript

        if genomic_region is None:
            self.genomic_region: GenomicRegion = transcript.exons.map(
                iv_on_transcript
            )
            if self.type == "ORF":
                self.full_genomic_region: GenomicRegion = transcript.exons.map(
                    (iv_on_transcript[0], iv_on_transcript[1] + 3)
                )
            else:
                self.full_genomic_region = self.genomic_region
            self.transcript_id: str = transcript.id
            self.iv_on_transcript: tuple[int, int] = iv_on_transcript
            self.dist_to_transcript_start: int = iv_on_transcript[0]
            self.dist_to_transcript_end: int = (
                transcript.exon_length - iv_on_transcript[1]
            )
        else:
            self.genomic_region = genomic_region
            self.transcript_id = transcript  # type: ignore[assignment]

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

    def effective_length(
        self,
        upstream_cleavage: int = 12,
        downstream_cleavage: int = 17,
    ) -> int:
        """Compute the number of positions that can yield reads.

        Positions near transcript ends may not be covered by reads
        because cleavage sites extend beyond the transcript boundary.
        This method returns the length of the RGR after trimming those
        inaccessible positions at each end.

        Parameters
        ----------
        upstream_cleavage : int, optional
            Maximum upstream cleavage offset (default 12).
        downstream_cleavage : int, optional
            Maximum downstream cleavage offset (default 17).

        Returns
        -------
        int
            Effective observable length (≥ 0).
        """
        return max(
            0,
            len(self.genomic_region)
            - max(0, upstream_cleavage - self.dist_to_transcript_start)
            - max(0, downstream_cleavage - self.dist_to_transcript_end),
        )

    def to_gtf(self, loc_id: str) -> str:
        """Serialise this RGR as GTF-formatted lines.

        Produces one ``exon`` line per interval and, for ORF-type RGRs,
        an additional ``CDS`` line per interval.  Coordinates are
        converted to 1-based, closed (GTF) convention.

        Parameters
        ----------
        loc_id : str
            Locus identifier included in the ``loc_id`` GTF attribute.

        Returns
        -------
        str
            One or more tab-separated GTF record lines, each terminated
            by ``\\n``.
        """
        seq_id = self.genomic_region.chrom
        source = "PRICE2"
        typ = "CDS" if self.type == "ORF" else "exon"
        score = "."
        strand = self.genomic_region.strand
        phase = "."
        attributes = (
            f'gene_id "{self.transcript.gene_id}"; '
            f'transcript_id "{self.id}"; '
            f'loc_id "{loc_id}";'
        )
        if hasattr(self, "log_p_value"):
            attributes += f' log_p_value "{self.log_p_value}";'
        if hasattr(self, "mean_rpkm"):
            attributes += f' mean_rpkm "{self.mean_rpkm}";'
        if hasattr(self, "mean_activity"):
            attributes += f' mean_activity "{self.mean_activity}";'
        s = ""
        for interval in self.full_genomic_region.intervals:
            s += (
                f"{seq_id}\t{source}\texon\t"
                f"{interval.start + 1}\t{interval.end}\t"
                f"{score}\t{strand}\t{phase}\t{attributes}\n"
            )
            if typ == "CDS":
                s += (
                    f"{seq_id}\t{source}\tCDS\t"
                    f"{interval.start + 1}\t{interval.end}\t"
                    f"{score}\t{strand}\t{phase}\t{attributes}\n"
                )
        return s

    def to_tsv_line(self, loc_id: str) -> str:
        """Serialise this RGR as a single TSV line.

        Columns: ``id``, ``gene_id``, ``loc_id``, ``location``.

        Parameters
        ----------
        loc_id : str
            Locus identifier.

        Returns
        -------
        str
            Tab-separated line terminated by ``\\n``.
        """
        return (
            f"{self.id}\t{self.transcript.gene_id}\t{loc_id}\t"
            f"{self.genomic_region}\n"
        )
