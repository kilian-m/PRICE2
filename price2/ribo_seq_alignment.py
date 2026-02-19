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
import pandas as pd

from price2.genomic_region import GenomicRegion
from price2.genomic_features import Transcript, ReadGeneratingRegion
from price2.reference_annotation import ReferenceAnnotation


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
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        data: pd.DataFrame | HTSeq._HTSeq.SAM_Alignment | dict,
    ) -> None:
        """Construct a :class:`RiboSeqAlignment`.

        Parameters
        ----------
        data : pd.DataFrame or HTSeq.SAM_Alignment or dict
            Source data.  Accepted forms:

            ``pd.DataFrame``
                Each row is one exonic interval of a collapsed
                alignment.  Required columns: ``chrom``, ``start``,
                ``end``, ``strand``, ``unique``, ``untemplated_addition``,
                ``count``.
            ``HTSeq.SAM_Alignment``
                A single alignment parsed from a BAM/SAM file.
            ``dict``
                Pre-built mapping with keys ``genomic_region``,
                ``untemplated_addition``, and ``mapping_positions``.
                An optional ``read_count`` key may be supplied.
        """
        if isinstance(data, pd.DataFrame):
            self._init_from_dataframe(data)
        elif isinstance(data, HTSeq._HTSeq.SAM_Alignment):
            self._init_from_sam_alignment(data)
        elif isinstance(data, dict):
            self._init_from_dict(data)
        else:
            raise TypeError(
                f"Unsupported data type for RiboSeqAlignment: {type(data)}"
            )

    def _init_from_dataframe(self, df: pd.DataFrame) -> None:
        """Initialise from a collapsed-alignment DataFrame."""
        first = df.iloc[0]
        self.mapping_positions = 1 if first["unique"] else 2  # TODO: propagate exact NH
        self.untemplated_addition = bool(first["untemplated_addition"])
        self.read_count = int(first["count"])

        intervals = [
            HTSeq.GenomicInterval(
                row["chrom"], row["start"], row["end"], row["strand"]
            )
            for _, row in df.iterrows()
        ]
        self.genomic_region = GenomicRegion(intervals)

    def _init_from_sam_alignment(self, sa: HTSeq._HTSeq.SAM_Alignment) -> None:
        """Initialise from an HTSeq SAM_Alignment object."""
        self.mapping_positions = sa.optional_field("NH")
        self.read_count = 1

        strand = sa.iv.strand
        if strand == "+":
            self.untemplated_addition = bool(
                sa.cigar[0].type == "S" and sa.cigar[0].size == 1
            )
        else:
            self.untemplated_addition = bool(
                sa.cigar[-1].type == "S" and sa.cigar[-1].size == 1
            )

        intervals = [op.ref_iv for op in sa.cigar if op.type == "M"]
        if strand == "-":
            intervals = intervals[::-1]

        self.genomic_region = GenomicRegion(intervals)

    def _init_from_dict(self, data: dict) -> None:
        """Initialise from a pre-built attribute dictionary."""
        self.genomic_region = data["genomic_region"]
        self.untemplated_addition = data["untemplated_addition"]
        self.mapping_positions = data["mapping_positions"]
        self.read_count = data.get("read_count", 1)

    # ------------------------------------------------------------------
    # Special methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"RiboSeqAlignment({self.genomic_region})"

    def __len__(self) -> int:
        return len(self.genomic_region)

    def __hash__(self) -> int:
        return hash(self.genomic_region) + hash(self.untemplated_addition)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RiboSeqAlignment):
            return NotImplemented
        return (
            self.genomic_region == other.genomic_region
            and self.untemplated_addition == other.untemplated_addition
            and self.mapping_positions == other.mapping_positions
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def matching_length(self) -> int:
        """Total number of matched (``M``) bases.

        Equivalent to the spliced length of :attr:`genomic_region`.
        """
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

    def get_first_pos_in_genomic_coords(self) -> int:
        """Return the lowest genomic start coordinate of the alignment.

        For positive-strand reads this is the leftmost covered base;
        for negative-strand reads it is also the leftmost base (i.e.
        the 3' end of the read in translation order).

        Returns
        -------
        int
            0-based start coordinate of the first interval (chromosome
            order).
        """
        if self.genomic_region.strand == "+":
            return self.genomic_region.intervals[0].start
        else:
            return self.genomic_region.intervals[-1].start

    def get_last_pos_in_genomic_coords(self) -> int:
        """Return the highest genomic end coordinate of the alignment.

        The returned value is the half-open end of the last interval in
        chromosome order.

        Returns
        -------
        int
            0-based, half-open end coordinate of the last interval
            (chromosome order).
        """
        if self.genomic_region.strand == "+":
            return self.genomic_region.intervals[-1].end
        else:
            return self.genomic_region.intervals[0].end

    def is_countable(
        self,
        reference_annotation: ReferenceAnnotation,
        max_considered_len: int = 50,
        min_considered_len: int = 15,
        min_dist_to_start: int = 30,
        min_dist_to_end: int = 30,
    ) -> bool:
        """Return whether this alignment should be counted for cleavage-model estimation.

        An alignment is countable when it is uniquely mapped, falls
        within the accepted length range, and is not too close to any
        annotated translation initiation site (TIS).

        Parameters
        ----------
        reference_annotation : ReferenceAnnotation
            Reference annotation used to locate annotated TIS positions.
        max_considered_len : int, optional
            Maximum accepted matching length in nucleotides.
        min_considered_len : int, optional
            Minimum accepted matching length in nucleotides.
        min_dist_to_start : int, optional
            Minimum required distance (nt) to the nearest annotated TIS.
        min_dist_to_end : int, optional
            Minimum required distance (nt) to the nearest CDS end.

        Returns
        -------
        bool
            ``True`` if all criteria are satisfied.
        """
        if not self.unique():
            return False
        if self.close_to_any_tis(reference_annotation, min_dist_to_start):
            return False
        if not min_considered_len <= self.matching_length <= max_considered_len:
            return False
        return True

    def dist_to_cds_start(self, transcript: Transcript) -> int | None:
        """Return distance from the alignment to the CDS start in transcript coordinates.

        Parameters
        ----------
        transcript : Transcript
            Transcript whose CDS is used as the reference.

        Returns
        -------
        int or None
            0-based distance in spliced transcript coordinates, or
            ``None`` if the alignment does not overlap the CDS.
        """
        interval = transcript.cds.induce(self.genomic_region)
        if not interval:
            return None
        return interval[0]

    def dist_to_cds_end(self, transcript: Transcript) -> int | None:
        """Return distance from the alignment to the CDS end in transcript coordinates.

        Parameters
        ----------
        transcript : Transcript
            Transcript whose CDS is used as the reference.

        Returns
        -------
        int or None
            Remaining CDS bases after the alignment in spliced
            transcript coordinates, or ``None`` if the alignment does
            not overlap the CDS.
        """
        interval = transcript.cds.induce(self.genomic_region)
        if not interval:
            return None
        return transcript.cds.length - interval[1]

    def get_frames(
        self, reference_annotation: ReferenceAnnotation
    ) -> tuple[int, int, int] | None:
        """Return per-frame read counts across all overlapping coding transcripts.

        Parameters
        ----------
        reference_annotation : ReferenceAnnotation
            Reference annotation used to find overlapping coding
            transcripts.

        Returns
        -------
        tuple[int, int, int] or None
            A 3-tuple ``(count_frame0, count_frame1, count_frame2)``
            tallied over all overlapping coding transcripts, or ``None``
            if no coding transcript overlaps.
        """
        transcript_candidates = reference_annotation.collect_coding_transcripts(
            self.genomic_region
        )
        if not transcript_candidates:
            return None
        frames = [0, 0, 0]
        for transcript in transcript_candidates:
            frame = transcript.cds.induce(self.genomic_region)[0] % 3
            frames[frame] += 1
        return tuple(frames)

    def get_frame(self, reference_annotation: ReferenceAnnotation) -> int | None:
        """Return the unambiguous reading frame index, or ``None`` if ambiguous.

        Parameters
        ----------
        reference_annotation : ReferenceAnnotation
            Reference annotation used to find overlapping coding
            transcripts.

        Returns
        -------
        int or None
            ``0``, ``1``, or ``2`` when all overlapping transcripts
            agree on a single frame; ``None`` otherwise (including when
            no transcript overlaps).
        """
        frames = self.get_frames(reference_annotation)
        if frames is None:
            return None
        f0, f1, f2 = frames
        if f0 and not f1 and not f2:
            return 0
        if not f0 and f1 and not f2:
            return 1
        if not f0 and not f1 and f2:
            return 2
        return None
