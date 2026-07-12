"""Locus-level ORF deconvolution for PRICE2.

Defines the :class:`Locus` class that aggregates overlapping transcripts
into a single genomic unit, generates ORF candidates, runs group-LASSO
penalised maximum-likelihood estimation, and applies filtering steps
(coverage, deconvolution, likelihood-ratio) to identify actively
translated regions.

Standalone helper functions for ORF detection and the optimisation
:class:`Callback` are also provided.
"""

from __future__ import annotations

import bisect
import logging
import math
import os
import sqlite3 as sql
import time
import zlib
from collections import defaultdict
from pickle import loads

import HTSeq
import numpy as np
from pyfaidx import Fasta

logger = logging.getLogger(__name__)
import pandas as pd
from filelock import FileLock
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, vstack as sp_vstack
from scipy.stats import chi2
from scipy.special import gammaln

from price2 import multimap
from price2.config import Config
from price2.coverage_model import CoveragePosition
from price2.equivalence_groups import EquivalenceGroup
from price2.genomic_features import ReadGeneratingRegion, Transcript
from price2.genomic_region import GenomicRegion
from price2.ribo_seq_alignment import RiboSeqAlignment
from price2.ribo_seq_run import RiboSeqRun

# Priority-ordered levels for ORF type assignment.  Within each level,
# having more than one matching label across compatible transcripts yields
# "other ORF"; the first level with exactly one match wins.
ORF_TYPES_LEVELS: list[set[str]] = [
    {"cORF"},
    {"N terminal extended cORF", "N terminal truncated cORF"},
    {
        "+1 uoORF",
        "+2 uoORF",
        "+1 doORF",
        "+2 doORF",
        "+1 iORF",
        "+2 iORF",
    },
    {"uORF", "dORF"},
    {"pcRNA-ORF", "lincRNA-ORF", "varRNA-ORF"},
]


def get_orf_type(
    orf: ReadGeneratingRegion,
    transcripts: set[Transcript],
) -> str:
    """Classify an ORF relative to the annotated CDSs of compatible transcripts.

    For each transcript in *transcripts* that contains the ORF's genomic
    footprint, the relationship between the ORF interval and the annotated
    CDS (in spliced transcript coordinates) is determined.  When an ORF is
    compatible with multiple transcripts the assignments are reconciled
    using :data:`ORF_TYPES_LEVELS`: the highest-priority level that has
    exactly one matching label is used; conflicting labels at the same
    level yield ``"other ORF"``.

    Parameters
    ----------
    orf : ReadGeneratingRegion
        An ORF-type RGR whose type should be classified.
    transcripts : set[Transcript]
        All transcripts belonging to the locus.

    Returns
    -------
    str
        ORF type label, e.g. ``'cORF'``, ``'uORF'``, ``'+0 iORF'``, or
        ``'other ORF'``.
    """
    assignments: dict[Transcript, str] = {}

    for tr in transcripts:
        try:
            orf_interval = tr.exons.map_to_local(orf.genomic_region)
        except ValueError:
            continue

        if tr.annotated_cds_iv is not None:
            cds_interval = tr.annotated_cds_iv
            orf_start, orf_end = orf_interval
            cds_start, cds_end = cds_interval

            if orf_interval == cds_interval:
                label = "cORF"
            elif orf_end == cds_end and orf_start < cds_start:
                label = "N terminal extended cORF"
            elif orf_end == cds_end and orf_start > cds_start:
                label = "N terminal truncated cORF"
            elif orf_end <= cds_start:
                label = "uORF"
            elif orf_start >= cds_end:
                label = "dORF"
            else:
                frame = (orf_start - cds_start) % 3
                if orf_start < cds_start < orf_end < cds_end:
                    label = f"+{frame} uoORF"
                elif cds_start < orf_start < cds_end < orf_end:
                    label = f"+{frame} doORF"
                elif cds_start < orf_start and orf_end < cds_end:
                    label = f"+{frame} iORF"
                else:
                    label = "other ORF"

        elif tr.biotype == "protein_coding":
            label = "pcRNA-ORF"
        elif tr.biotype == "lincRNA":
            label = "lincRNA-ORF"
        else:
            label = "varRNA-ORF"

        assignments[tr] = label

    if not assignments:
        return "other ORF"

    label_set = set(assignments.values())
    for level in ORF_TYPES_LEVELS:
        matches = label_set & level
        if len(matches) > 1:
            return "other ORF"
        if len(matches) == 1:
            return matches.pop()

    return "other ORF"


class EgRoutingCache:
    """Weight-independent per-locus state reused across EM iterations.

    Between multimapping-EM iterations only the fractional read weights
    change: the RGR set, the equivalence groups and hence the design matrix
    ``X`` are fixed (the light M-step runs with ``prune=False``).  This
    caches everything a light M-step would otherwise re-derive from the
    reads — the read → design-matrix-row routing and the geometry of ``X`` —
    so an iteration reduces to a weighted ``bincount`` plus a vectorised
    rebuild of ``X.data``.

    Invalidated (set to ``None`` on the locus) whenever the RGR set changes,
    since that re-indexes RGRs and collapses equivalence groups.

    It holds only arrays, ints and strings — no RGR, transcript or
    equivalence-group objects — so it can be pickled on its own.  A light
    M-step therefore loads just this blob (a few numpy ``memcpy``s) instead
    of unpickling the locus's whole object graph, which is ~96% of the cost
    of restoring a prepared locus.

    Attributes
    ----------
    n_rows : int
        Number of design-matrix rows (non-empty equivalence groups).
    n_reads : dict[str, int]
        Reads per run at build time; guards against a changed read order.
    eg_row : dict[str, numpy.ndarray]
        Per run, the row each read feeds: ``>=0`` a row index, ``-1`` the
        read's key is absent from ``egs`` (uncounted), ``-2`` the read is
        compatible with no RGR.
    counts0 : dict[str, numpy.ndarray]
        Per run, each read's raw (unweighted) count.
    mm_idx, mm_gk, mm_base : dict[str, numpy.ndarray]
        Per run, the read positions that carry a multimapping slot, their
        group keys, and their baseline cross-locus mass.
    slot_gk, slot_rl, slot_oua, slot_nnz : dict[str, numpy.ndarray]
        Per run, one entry per multimapping slot: group key, read length,
        untemplated-addition flag, and number of compatible RGR cells.
    slot_rgr, slot_code : dict[str, numpy.ndarray]
        Per run, the flattened slot cells: RGR index and packed
        ``frame * 3 + coverage_position``.  Replaces the object-valued
        ``mm_slots`` routing that :meth:`compute_multimap_lambdas` used.
    cell_rgr, cell_code : numpy.ndarray
        One entry per design-matrix cell: the RGR index, and
        ``frame * 3 + coverage_position`` packed into a byte.
    row_nnz, row_len, row_rl, row_oua, row_run : numpy.ndarray
        Per row: cell count, EG length, read length, untemplated-addition
        flag, and run index.
    num_rgrs : int
        RGR count at build time (design-matrix column blocks).
    rgr_ids : tuple[str, ...]
        RGR identifiers, indexed by ``rgr.index``.
    rgr_lengths : numpy.ndarray
        RGR lengths, indexed by ``rgr.index``.
    """

    __slots__ = (
        "n_rows", "n_reads", "eg_row", "counts0", "mm_idx", "mm_gk",
        "mm_base", "slot_gk", "slot_rl", "slot_oua", "slot_nnz",
        "slot_rgr", "slot_code", "cell_rgr", "cell_code", "row_nnz",
        "row_len", "row_rl", "row_oua", "row_run", "num_rgrs",
        "rgr_ids", "rgr_lengths",
    )

    def __init__(self, **fields: object) -> None:
        for name, value in fields.items():
            setattr(self, name, value)


class Locus:
    """A genomic locus containing overlapping transcripts and ORF candidates.

    A locus aggregates one or more transcripts whose exons overlap on the
    same strand into a single unit of analysis.  It generates candidate
    :class:`ReadGeneratingRegion` objects (ORFs and noise regions),
    constructs equivalence groups from mapped Ribo-seq reads, and runs
    group-LASSO penalised Poisson-likelihood optimisation to identify
    actively translated ORFs.

    Attributes
    ----------
    iv : HTSeq.GenomicInterval
        Genomic interval spanning the locus.
    id : str
        Unique identifier of the form ``"loc_<N>"``.
    transcripts : set[Transcript]
        Transcripts assigned to this locus.
    transcript_intervals : HTSeq.GenomicArrayOfSets
        Stranded genomic array mapping positions to overlapping
        transcripts.
    rgr_set : set[ReadGeneratingRegion]
        Current set of ORF and noise RGR candidates.
    rgr_intervals : HTSeq.GenomicArrayOfSets
        Stranded genomic array mapping positions to overlapping RGRs.
    egs : dict[RiboSeqRun, dict]
        Per-run equivalence groups built during read assignment.
    read_counts : dict[RiboSeqRun, int]
        Number of reads assigned to this locus per run.
    exon_length : int
        Total exonic length (bp) covered by the locus.
    result : np.ndarray | None
        Activity matrix of shape ``(n_rgrs, n_runs)`` after
        deconvolution, or ``None`` before estimation.
    """

    iv: HTSeq.GenomicInterval
    id: str
    transcripts: set[Transcript]
    transcript_intervals: HTSeq.GenomicArrayOfSets
    rgr_set: set[ReadGeneratingRegion]
    rgr_intervals: HTSeq.GenomicArrayOfSets
    egs: dict[RiboSeqRun, dict]
    read_counts: dict[RiboSeqRun, int]
    exon_length: int
    result: np.ndarray | None

    def __init__(
        self,
        iv: HTSeq.GenomicInterval,
        transcript_intervals: HTSeq.GenomicArrayOfSets,
        loci_number: int,
    ) -> None:
        """Initialise a Locus from a genomic interval.

        Parameters
        ----------
        iv : HTSeq.GenomicInterval
            Genomic interval spanning all transcripts in the locus.
        transcript_intervals : HTSeq.GenomicArrayOfSets
            Genome-wide stranded array mapping positions to transcript
            sets; only the portion overlapping *iv* is retained.
        loci_number : int
            Sequential counter used to build :attr:`id`.
        """
        self.iv = iv
        self.read_counts: dict[RiboSeqRun, int] = {}
        self.uncounted_reads = 0
        self.eg_cache: EgRoutingCache | None = None
        self._eg_y: np.ndarray | None = None
        self.id = f"loc_{loci_number}"

        self.transcript_intervals = HTSeq.GenomicArrayOfSets(
            "auto", stranded=True, storage="step"
        )

        for iv, val in transcript_intervals[self.iv].steps():
            self.transcript_intervals[iv] = val

        self.transcripts: set[Transcript] = set()

        self.exon_length = 0
        for iv, value in self.transcript_intervals.steps():
            self.transcripts |= value
            if value:
                self.exon_length += iv.length

    def __repr__(self) -> str:
        return f"Locus({self.iv})"

    @property
    def transcript_breakpoint_index(
        self,
    ) -> tuple[list[int], list[int], list[set]]:
        """Lazily built sorted breakpoint index of ``transcript_intervals``.

        Iterates the step-array once and caches three parallel lists:
        ``bp_starts``, ``bp_ends``, and ``bp_sets`` (only non-empty steps).
        Use :func:`bisect.bisect_right` on *bp_ends* to find overlapping
        entries for a query ``[q_start, q_end)`` interval in O(log B)
        instead of a full step-array traversal.

        The cache is excluded from pickle (``__getstate__``) so it does not
        inflate stored locus blobs.

        Returns
        -------
        tuple[list[int], list[int], list[set]]
            ``(bp_starts, bp_ends, bp_sets)`` — parallel lists over
            all non-empty breakpoints, sorted by start position.
        """
        try:
            return self._transcript_breakpoint_index
        except AttributeError:
            bp_starts: list[int] = []
            bp_ends: list[int] = []
            bp_sets: list[set] = []
            for iv, ts in self.transcript_intervals.steps():
                if ts:
                    bp_starts.append(iv.start)
                    bp_ends.append(iv.end)
                    bp_sets.append(ts)
            self._transcript_breakpoint_index = (bp_starts, bp_ends, bp_sets)
            return self._transcript_breakpoint_index

    @property
    def transcript_junction_index(self) -> dict[tuple[int, int], tuple]:
        """Lazily built map from intron to the transcripts that splice it.

        Keyed by ``(donor_exon_end, acceptor_exon_start)`` — the intron of
        a pair of *consecutive* exons — with the flanking exon bounds
        attached.  A two-block read maps into a transcript exactly when its
        gap is one of that transcript's introns and its outer ends stay
        inside the flanking exons, which is what
        :meth:`~price2.genomic_region.GenomicRegion.map_to_local` checks
        one candidate at a time; the index turns that into a dict lookup.

        Excluded from pickle (see :meth:`__getstate__`).

        Returns
        -------
        dict
            ``{(intron_start, intron_end): ((transcript, donor_exon_start,
            acceptor_exon_end), ...)}``.
        """
        try:
            return self._transcript_junction_index
        except AttributeError:
            index: dict[tuple[int, int], list] = {}
            for transcript in self.transcripts:
                exons = transcript.exons.intervals
                for donor, acceptor in zip(exons, exons[1:]):
                    index.setdefault((donor.end, acceptor.start), []).append(
                        (transcript, donor.start, acceptor.end)
                    )
            self._transcript_junction_index = {
                k: tuple(v) for k, v in index.items()
            }
            return self._transcript_junction_index

    @property
    def has_abutting_exons(self) -> bool:
        """Whether any transcript has two exons that touch (``end == start``).

        The single-block read fast path in
        :func:`~price2.data_collector.collect_mappings` infers "this
        transcript covers the read within one exon" from the read being
        covered by a contiguous run of breakpoint steps.  That inference
        holds only because a transcript's exons are separated by introns,
        so a contiguous covered stretch cannot straddle two of them.
        Abutting exons would break it; annotations containing them fall
        back to :meth:`~price2.genomic_region.GenomicRegion.map_to_local`.

        Excluded from pickle (see :meth:`__getstate__`).
        """
        try:
            return self._has_abutting_exons
        except AttributeError:
            self._has_abutting_exons = any(
                donor.end == acceptor.start
                for transcript in self.transcripts
                for donor, acceptor in zip(
                    transcript.exons.intervals, transcript.exons.intervals[1:]
                )
            )
            return self._has_abutting_exons

    def __getstate__(self) -> dict:
        """Return pickle state, excluding lazily-rebuilt caches."""
        state = self.__dict__.copy()
        state.pop("_transcript_breakpoint_index", None)
        state.pop("_transcript_junction_index", None)
        state.pop("_has_abutting_exons", None)
        state.pop("_rgr_intervals", None)
        return state

    @property
    def rgr_intervals(self) -> HTSeq.GenomicArrayOfSets:
        """Per-position ``GenomicArrayOfSets`` over the current RGR set.

        Built lazily from :attr:`rgr_set` on first access and cached; the cache
        is invalidated whenever the RGR set changes (see :meth:`remove_rgrs`).
        Nothing in the core pipeline reads this structure, so it is normally
        never materialised — building it eagerly in :meth:`make_rgrs` was an
        O(n_rgr^2) hot-spot on dense loci (hundreds of seconds on the largest
        Yewdell locus) for no downstream benefit.
        """
        ri = getattr(self, "_rgr_intervals", None)
        if ri is None:
            ri = HTSeq.GenomicArrayOfSets("auto", stranded=True, storage="step")
            for rgr in self.rgr_set:
                for iv in rgr.genomic_region.intervals:
                    ri[iv] += rgr
            self._rgr_intervals = ri
        return ri

    def make_rgrs(
        self,
        genome: Fasta,
        config: Config,
        min_length_to_end: int = 30,
    ) -> None:
        """Generate ORF and noise ReadGeneratingRegions for this locus.

        For each transcript, noise regions are created upstream and
        downstream of the annotated CDS (if present) or spanning the
        full transcript otherwise.  ORF candidates are found by scanning
        the spliced transcript sequence for start/stop codon pairs.
        Duplicate RGRs (identical genomic footprint) are deduplicated,
        keeping the copy with the longest flanking context.

        Parameters
        ----------
        genome : pyfaidx.Fasta
            Indexed FASTA handle keyed by chromosome name.
        config : Config
            Configuration providing ``start_codons`` and ``stop_codons``.
        min_length_to_end : int
            Minimum combined length of ORF plus flanking transcript
            distance (in nucleotides) for an ORF to be retained.
        """
        self.rgr_set: set[ReadGeneratingRegion] = set()
        orf_dict: dict[ReadGeneratingRegion, ReadGeneratingRegion] = {}
        noise_dict: dict[ReadGeneratingRegion, ReadGeneratingRegion] = {}

        for transcript in self.transcripts:
            if (transcript.annotated_cds_iv is not None) and (
                (cds_start := transcript.annotated_cds_iv[0]) > 5
            ):
                # cds_start = transcript.exons.map_to_local(transcript.cds)[0]
                noise1 = ReadGeneratingRegion(
                    "NOISE",
                    transcript,
                    f"{transcript.id}_a",
                    (0, cds_start),
                )

                noise2 = ReadGeneratingRegion(
                    "NOISE",
                    transcript,
                    f"{transcript.id}_b",
                    (cds_start, len(transcript.exons)),
                )

                for noise in [noise1, noise2]:
                    if noise not in noise_dict:
                        noise_dict[noise] = noise
                    else:
                        existing = noise_dict[noise]
                        existing_span = (
                            existing.dist_to_transcript_end
                            + existing.dist_to_transcript_start
                        )
                        new_span = (
                            noise.dist_to_transcript_end
                            + noise.dist_to_transcript_start
                        )
                        if new_span > existing_span:
                            noise_dict[noise] = noise

            else:
                noise = ReadGeneratingRegion(
                    "NOISE",
                    transcript,
                    transcript.id,
                    (0, len(transcript.exons)),
                )
                if noise not in noise_dict:
                    noise_dict[noise] = noise
                else:
                    existing = noise_dict[noise]
                    existing_span = (
                        existing.dist_to_transcript_end
                        + existing.dist_to_transcript_start
                    )
                    new_span = (
                        noise.dist_to_transcript_end
                        + noise.dist_to_transcript_start
                    )
                    if new_span > existing_span:
                        noise_dict[noise] = noise

            seq = transcript.exons.get_sequence(genome)
            c = 0
            for orf_iv_on_transcript in find_orfs(
                seq, config.start_codons, config.stop_codons
            ):
                rgr_iv_on_transcript = (
                    orf_iv_on_transcript[0],
                    orf_iv_on_transcript[1] - 3,
                )  # remove stop codon
                c += 1
                orf = ReadGeneratingRegion(
                    "ORF",
                    transcript,
                    f"{transcript.id}_{c:04d}",
                    rgr_iv_on_transcript,
                )
                if len(orf) + orf.dist_to_transcript_end < min_length_to_end:
                    continue
                if len(orf) + orf.dist_to_transcript_start < min_length_to_end:
                    continue
                if orf not in orf_dict:
                    orf_dict[orf] = orf
                else:
                    existing = orf_dict[orf]
                    existing_span = (
                        existing.dist_to_transcript_end
                        + existing.dist_to_transcript_start
                    )
                    new_span = orf.dist_to_transcript_end + orf.dist_to_transcript_start
                    if new_span > existing_span:
                        orf_dict[orf] = orf

        for noise in noise_dict.values():
            noise.transcript.rgr_set.add(noise)
        self.rgr_set |= set(noise_dict.values())
        for orf in orf_dict.values():
            orf.orf_type = get_orf_type(orf, self.transcripts)
            orf.transcript.add_orf(orf)
        self.rgr_set |= set(orf_dict.values())

        # ``rgr_intervals`` (a per-position GenomicArrayOfSets over every RGR)
        # is built lazily on first access (see the ``rgr_intervals`` property)
        # rather than eagerly here.  Building it eagerly is O(n_rgr^2) for the
        # heavily-overlapping ORF candidates of large loci (~275 s on the
        # densest Yewdell locus alone) and nothing in the pipeline reads it, so
        # the eager build was pure overhead on the critical path.
        self._rgr_intervals = None

        # ``rgr.index`` addresses the design-matrix column blocks and the rows
        # of ``result``, so every RGR carries one from the moment the set is
        # built.  ``remove_rgrs`` re-densifies them after a removal.
        for c, rgr in enumerate(self.rgr_set):
            rgr.index = c

        self.gene_ids_complete = {
            rgr.transcript.gene_id for rgr in self.rgr_set
        }

    def get_rgr_frame_covpos(
        self,
        rsa: RiboSeqAlignment,
        run: RiboSeqRun,
        overlap_likelihood_ratio_threshold: float = 0.2,
    ) -> frozenset[tuple[ReadGeneratingRegion, int | None, CoveragePosition]] | None:
        """Determine which RGRs a read alignment is compatible with.

        For each RGR overlapping the read, compute the reading frame and
        coverage-profile position (start / middle / stop).  Partial
        overlaps are kept only when the cleavage-model probability ratio
        exceeds *overlap_likelihood_ratio_threshold*.

        Parameters
        ----------
        rsa : RiboSeqAlignment
            A single Ribo-seq read alignment.
        run : RiboSeqRun
            The Ribo-seq run that produced *rsa*.
        overlap_likelihood_ratio_threshold : float
            Minimum ratio of partial-overlap cleavage probability to
            full-overlap probability for a partial overlap to be accepted.

        Returns
        -------
        frozenset[tuple[ReadGeneratingRegion, int | None, CoveragePosition]] or None
            Set of ``(rgr, frame, coverage_position)`` tuples, or
            ``None`` when no compatible RGR is found.
        """

        overlap_transcripts = set(self.transcripts)

        rgr_frame_covpos = set()

        bp_starts, bp_ends, bp_sets = self.transcript_breakpoint_index
        for query_iv in rsa.genomic_region.intervals:
            i = bisect.bisect_right(bp_ends, query_iv.start)
            while i < len(bp_starts) and bp_starts[i] < query_iv.end:
                overlap_transcripts &= bp_sets[i]
                i += 1

        # Hoist per-read / per-run invariants out of the transcript x rgr
        # loops.  ``len(rsa) == len(rsa.genomic_region)`` (a cached value) and
        # the untemplated-addition flag are the same for every candidate RGR,
        # so compute them once.  The full-overlap cleavage likelihood is an
        # exact lookup-table entry (verified bit-identical to
        # ``CleavageModel.pmf`` across the whole domain), so index the LUT
        # directly instead of paying a pmf() call frame per RGR — only the
        # partial-overlap (region-bounded) likelihoods still call pmf.
        pmf = run.cleavage_model.pmf
        cds_lut = run.cleavage_model.cds_lut
        noise_lut = run.cleavage_model.noise_lut
        lut_len = cds_lut.shape[0]
        read_length = len(rsa)
        oua = rsa.untemplated_addition
        oua_i = int(oua)
        thr = overlap_likelihood_ratio_threshold
        cov_middle = CoveragePosition.middle
        cov_start = CoveragePosition.start
        cov_stop = CoveragePosition.stop
        in_lut = read_length < lut_len
        noise_cl = noise_lut[read_length, oua_i] if in_lut else 0.0

        for tr in overlap_transcripts:
            try:
                rsa_iv_on_tr = tr.exons.map_to_local(rsa.genomic_region)
            except ValueError:
                continue
            rsa_lo, rsa_hi = rsa_iv_on_tr
            for rgr in tr.rgr_set:
                rgr_lo, rgr_hi = rgr.iv_on_transcript
                # full overlap with orf
                if rgr.type == "NOISE":
                    frame = None
                    if (rgr_lo <= rsa_lo) and (rgr_hi >= rsa_hi):
                        if noise_cl > 0:
                            rgr_frame_covpos.add((rgr, frame, cov_middle))
                    elif (rsa_lo <= rgr_lo <= rsa_hi) or (
                        rsa_lo <= rgr_hi <= rsa_hi
                    ):
                        region_start = rgr_lo - rsa_lo
                        region_end = rgr_hi - rsa_lo
                        if (
                            ol := pmf(
                                read_length,
                                oua,
                                frame,
                                region_start=region_start,
                                region_end=region_end,
                            )
                        ) > 0:
                            cl = noise_cl
                            if cl == 0:
                                continue
                            if ol / cl > thr:
                                rgr_frame_covpos.add(
                                    (
                                        rgr,
                                        frame,
                                        cov_middle,
                                    )
                                )
                elif rgr.type == "ORF":
                    orf = rgr
                    if (rgr_lo <= rsa_lo) and (rgr_hi >= rsa_hi):
                        frame = (rsa_lo - rgr_lo) % 3
                        if (cds_lut[read_length, frame, oua_i] if in_lut else 0.0) > 0:
                            rgr_frame_covpos.add((orf, frame, cov_middle))
                    # part overlap with orf
                    elif (rsa_lo <= rgr_lo <= rsa_hi) or (
                        rsa_lo <= rgr_hi <= rsa_hi
                    ):
                        frame = (rsa_lo - rgr_lo) % 3
                        cl = cds_lut[read_length, frame, oua_i] if in_lut else 0.0
                        cl_ok = not cl == 0
                        # consider overlap likelihood
                        # compute at which position in the read the orf starts
                        region_start = rgr_lo + 3 - rsa_lo
                        # compute at which position in the read the orf ends
                        region_end = rgr_hi - 3 - rsa_lo
                        if (
                            ol := pmf(
                                read_length,
                                oua,
                                frame,
                                region_start=region_start,
                                region_end=region_end,
                            )
                        ) > 0:
                            if cl_ok and (ol / cl > thr):
                                rgr_frame_covpos.add(
                                    (
                                        orf,
                                        frame,
                                        cov_middle,
                                    )
                                )
                        # consider coverage profile - start
                        # compute where the orf starts relative to the read
                        start_position = (
                            rgr_lo - rsa_lo,
                            rgr_lo + 3 - rsa_lo,
                        )
                        if (
                            ol := pmf(
                                read_length,
                                oua,
                                frame,
                                region_start=start_position[0],
                                region_end=start_position[1],
                            )
                        ) > 0:
                            if cl_ok and (
                                # ol * run.coverage_model.start_factor / cl
                                ol / cl
                                > thr
                            ):
                                rgr_frame_covpos.add(
                                    (orf, frame, cov_start)
                                )

                        # consider coverage profile - stop
                        # compute where the orf ends relative to the read
                        stop_position = (
                            rgr_hi - 3 - rsa_lo,
                            rgr_hi - rsa_lo,
                        )
                        if (
                            ol := pmf(
                                read_length,
                                oua,
                                frame,
                                region_start=stop_position[0],
                                region_end=stop_position[1],
                            )
                        ) > 0:
                            if cl_ok and (
                                # ol * run.coverage_model.stop_factor / cl
                                ol / cl
                                > thr
                            ):
                                rgr_frame_covpos.add(
                                    (orf, frame, cov_stop)
                                )

        if not rgr_frame_covpos:
            return

        rgr_frame_covpos = frozenset(rgr_frame_covpos)
        return rgr_frame_covpos

    def gtf_line(self) -> str:
        """Return a single GTF line describing this locus."""
        seq_id = self.iv.chrom
        source = "PRICE2"
        typ = "locus"
        start = self.iv.start
        end = self.iv.end
        score = "."
        strand = self.iv.strand
        phase = "."
        attributes = f'locus_id "{self.id}";'

        return f"{seq_id}\t{source}\t{typ}\t{start}\t{end}\t{score}\t{strand}\t{phase}\t{attributes}\n"

    def to_gtf(
        self,
        prefix: str,
        write_loci: bool = False,
        write_transcripts: bool = False,
        write_orfs: bool = True,
    ) -> None:
        """Append locus features to GTF files.

        Files are created or appended to with file-lock protection for
        concurrent writes from multiple worker processes.

        Parameters
        ----------
        prefix : str
            Path prefix; files are named ``<prefix>_loci.gtf``,
            ``<prefix>_transcripts.gtf``, and ``<prefix>_orfs.gtf``.
        write_loci : bool
            Write the locus interval.
        write_transcripts : bool
            Write noise (transcript-level) RGRs.
        write_orfs : bool
            Write ORF-type RGRs.
        """
        sep = "" if prefix.endswith("/") else "_"
        if write_loci:
            path = f"{prefix}{sep}loci.gtf"
            lock = FileLock(path + ".lock")
            with lock:
                with open(path, "a") as f:
                    f.write(self.gtf_line())

        if write_transcripts:
            path = f"{prefix}{sep}transcripts.gtf"
            lock = FileLock(path + ".lock")
            with lock:
                with open(path, "a") as f:
                    for rgr in self.rgr_set:
                        if rgr.type == "NOISE":
                            f.write(rgr.to_gtf(self.id))

        if write_orfs:
            path = f"{prefix}{sep}orfs.gtf"
            lock = FileLock(path + ".lock")
            with lock:
                with open(path, "a") as f:
                    for rgr in self.rgr_set:
                        if rgr.type == "ORF":
                            f.write(rgr.to_gtf(self.id))

    def to_tsv(
        self,
        prefix: str,
        runs: list | None = None,
        include_noise: bool = False,
    ) -> None:
        """Append region results to a TSV file.

        When *runs* is provided the output includes a header row
        (written only once) and per-run activity columns taken from
        :attr:`result_df`.  Without *runs* only the four annotation
        columns are written (used for intermediate/verbose output).

        The file is named ``<prefix>_regions.tsv`` when *include_noise*
        is ``True``, otherwise ``<prefix>_orfs.tsv``.

        Parameters
        ----------
        prefix : str
            Path prefix for the output file.
        runs : list[RiboSeqRun] | None
            Ribo-seq runs whose IDs become the activity columns.
        include_noise : bool
            When ``True`` NOISE regions are written alongside ORFs.
        """
        suffix = "regions" if include_noise else "orfs"
        id_col = "region_id" if include_noise else "orf_id"
        sep = "" if prefix.endswith("/") else "_"
        path = f"{prefix}{sep}{suffix}.tsv"
        lock = FileLock(path + ".lock")
        with lock:
            # Write header once when activity columns are requested.
            if runs is not None and not os.path.exists(path):
                run_ids = [run.id for run in runs]
                with open(path, "w") as f:
                    f.write(
                        f"{id_col}\tgene_id\ttranscript_id\tlocus_id"
                        f"\tgenomic_region\torf_type\t" + "\t".join(run_ids) + "\n"
                    )

            with open(path, "a") as f:
                for rgr in self.rgr_set:
                    if not include_noise and rgr.type != "ORF":
                        continue
                    if runs is not None and hasattr(self, "result_df"):
                        activities = self.result_df.loc[rgr.id]
                        activity_str = "\t".join(f"{v:.2e}" for v in activities)
                        orf_type_str = rgr.orf_type if rgr.orf_type is not None else ""
                        f.write(
                            f"{rgr.id}\t{rgr.transcript.gene_id}"
                            f"\t{rgr.transcript.id}"
                            f"\t{self.id}\t{rgr.full_genomic_region}"
                            f"\t{orf_type_str}\t{activity_str}\n"
                        )
                    else:
                        f.write(rgr.to_tsv_line(self.id))

    def to_bed(
        self,
        prefix: str,
        include_noise: bool = False,
    ) -> None:
        """Append region results to a BED12 file.

        Parameters
        ----------
        prefix : str
            Path prefix; the file is named ``<prefix>_regions.bed`` when
            *include_noise* is ``True``, otherwise ``<prefix>_orfs.bed``.
        include_noise : bool
            When ``True`` NOISE regions are written alongside ORFs.
        """
        suffix = "regions" if include_noise else "orfs"
        sep = "" if prefix.endswith("/") else "_"
        path = f"{prefix}{sep}{suffix}.bed"
        lock = FileLock(path + ".lock")
        with lock:
            with open(path, "a") as f:
                for rgr in self.rgr_set:
                    if not include_noise and rgr.type != "ORF":
                        continue
                    f.write(rgr.to_bed_line())

    def to_fasta(
        self,
        prefix: str,
        genome: Fasta,
    ) -> None:
        """Append ORF sequences (with flanking context) to a FASTA file.

        Each ORF is extended by up to 14 nt upstream and 20 nt downstream
        on its parent transcript before extracting the spliced sequence.

        Parameters
        ----------
        prefix : str
            Directory prefix; the file is ``<prefix>/orfs.fasta``.
        genome : pyfaidx.Fasta
            Indexed FASTA handle keyed by chromosome name.
        """

        path = f"{prefix}/orfs.fasta"
        for rgr in self.rgr_set:
            if rgr.type != "ORF":
                continue
            iv_on_transcript = rgr.iv_on_transcript
            iv_on_transcript = (
                max(0, iv_on_transcript[0] - 14),
                min(len(rgr.transcript), iv_on_transcript[1] + 20),
            )
            gr = rgr.transcript.exons.map_to_global(iv_on_transcript)
            seq = gr.get_sequence(genome).upper()

            lock = FileLock(path + ".lock")
            with lock:
                with open(path, "a") as f:
                    f.write(f">{rgr.id}|{rgr.transcript.gene_id}|{rgr.transcript.id}\n")
                    for i in range(0, len(seq), 60):
                        f.write(seq[i : i + 60] + "\n")

    # ------------------------------------------------------------------ #
    # ORF activity estimation                                              #
    # ------------------------------------------------------------------ #

    def get_reads_from_db(self, db_path: str) -> None:
        """Load reads for this locus from the SQLite database.

        Populates :attr:`rsas_dict` (mapping run id to a list of
        :class:`RiboSeqAlignment` objects) and :attr:`run_read_count`.

        Parameters
        ----------
        db_path : str
            Path to the ``price.db`` SQLite database.
        """
        db = sql.connect(db_path, timeout=120)
        cur = db.cursor()
        cur.execute("PRAGMA busy_timeout = 120000")
        reads_dfs = cur.execute(
            """
            SELECT * FROM reads 
            WHERE locus_id = ?
            """,
            (self.id,),
        )
        self.run_read_count = {}
        self.rsas_dict = {}
        chrom = self.iv.chrom
        strand = self.iv.strand
        # Memoize GenomicRegion objects by their interval-coordinate signature.
        # Reads are exact-deduplicated within a run at collection time, but the
        # same coordinates recur across runs (typically 40-80% of reads); sharing
        # one immutable GenomicRegion across runs avoids rebuilding its intervals
        # and hash.  Scoped per locus, so it is freed when the locus is done.
        region_cache: dict[tuple, GenomicRegion] = {}
        for _, run_id, blob in reads_dfs:
            reads_df = loads(zlib.decompress(blob))

            # Vectorized: pull columns into numpy arrays and locate per-read
            # boundaries from is_first_iv, avoiding groupby / iterrows / per-row
            # Series construction (the dominant cost of read loading).
            is_first = reads_df["is_first_iv"].to_numpy()
            starts = reads_df["start"].to_numpy()
            ends = reads_df["end"].to_numpy()
            uas = reads_df["untemplated_addition"].to_numpy()
            uniques = reads_df["unique"].to_numpy()
            counts = reads_df["count"].to_numpy()

            boundaries = np.flatnonzero(is_first)
            read_ends = np.append(boundaries[1:], len(is_first))

            rsas_run = []
            for b, e in zip(boundaries.tolist(), read_ends.tolist()):
                if e - b == 1:
                    sig = (int(starts[b]), int(ends[b]))
                else:
                    sig = tuple(
                        (int(starts[j]), int(ends[j])) for j in range(b, e)
                    )
                gr = region_cache.get(sig)
                if gr is None:
                    intervals = [
                        HTSeq.GenomicInterval(
                            chrom, int(starts[j]), int(ends[j]), strand
                        )
                        for j in range(b, e)
                    ]
                    gr = GenomicRegion(
                        intervals=intervals, chrom=chrom, strand=strand
                    )
                    region_cache[sig] = gr
                rsas_run.append(
                    RiboSeqAlignment.from_region(
                        gr,
                        untemplated_addition=bool(uas[b]),
                        unique=bool(uniques[b]),
                        read_count=int(counts[b]),
                    )
                )
            self.rsas_dict[run_id] = rsas_run

            self.run_read_count[run_id] = int(
                counts[boundaries].astype(np.int64).sum()
            )

    def make_well_fitting_reads(self, runs: list[RiboSeqRun]) -> None:
        """Count well-fitting reads per RGR and run.

        A read is *well-fitting* when its length and untemplated-addition
        status match a high-probability entry in the run's cleavage
        model.  Results are stored in :attr:`wfr_df` (a DataFrame
        indexed by RGR id with one column per run) and :attr:`wfr_count`.

        Parameters
        ----------
        runs : list[RiboSeqRun]
            Ribo-seq runs to process.
        """
        well_fitting_rcs = {}
        for run in runs:
            well_fitting_rcs[run.id] = {}
            for rgr in self.rgr_set:
                if rgr.type == "ORF":
                    well_fitting_rcs[run.id][rgr.id] = 0
        self.wfr_count = 0
        for run in runs:
            well_fitting_indices = run.cleavage_model.get_high_prob_indices()
            well_fitting_length_oua = {(l, oua) for l, f, oua in well_fitting_indices}
            for rsa in self.rsas_dict[run.id]:
                if (
                    len(rsa),
                    int(rsa.untemplated_addition),
                ) not in well_fitting_length_oua:
                    continue
                rgr_frame_covpos = self.get_rgr_frame_covpos(rsa, run)
                if not rgr_frame_covpos:
                    continue

                # One RGR can appear under several coverage positions for the
                # same read (a read spanning a short ORF overlaps both its
                # start- and stop-codon regions), so deduplicate before
                # counting: a read contributes its count once per RGR.
                orf_ids = {
                    rgr.id
                    for rgr, _frame, _covpos in rgr_frame_covpos
                    if rgr.type != "NOISE"
                }
                if orf_ids:
                    self.wfr_count += rsa.read_count
                for rgr_id in orf_ids:
                    well_fitting_rcs[run.id][rgr_id] += rsa.read_count

        self.wfr_df = (
            pd.DataFrame.from_dict(well_fitting_rcs).replace(np.nan, 0).astype(np.int32)
        )

    def coverage_filter_rgrs(self, config: Config) -> None:
        """Remove ORFs with insufficient well-fitting read coverage.

        For each ORF RGR the per-nucleotide well-fitting read count is
        computed across all runs.  ORFs whose maximum across runs falls
        below ``config.min_well_fitting_reads_per_length`` are removed.

        Parameters
        ----------
        config : Config
            Configuration providing the coverage threshold.
        """

        rgr_lengths = {rgr.id: len(rgr.genomic_region) for rgr in self.rgr_set}

        rgr_lengths = pd.Series(rgr_lengths).reindex(self.wfr_df.index)
        wfr_df_rel = self.wfr_df.div(rgr_lengths, axis=0)

        rgrs_to_remove_ids = set(
            wfr_df_rel[
                wfr_df_rel.max(axis=1) <= config.min_well_fitting_reads_per_length
            ].index
        )
        rgrs_to_remove = {
            rgr
            for rgr in self.rgr_set
            if rgr.id in rgrs_to_remove_ids and rgr.type == "ORF"
        }

        self.remove_rgrs(rgrs_to_remove)

    def deconvolution_filter_rgrs(self, config: Config) -> None:
        """Remove ORFs that are inactive within their stop-codon group.

        ORFs sharing the same stop codon are grouped, split into
        compatible optimisation groups, and each group is deconvolved.
        ORFs with estimated activity below
        ``config.deconvolution_filter_min_activity`` in every run are
        removed.

        Parameters
        ----------
        config : Config
            Configuration providing filter thresholds.
        """

        tmp = self.make_stop_groups()
        optimization_groups = self.split_stop_groups(tmp)

        rgr_ids_to_remove = set()
        for opt_group in optimization_groups:
            rgr_ids_to_remove |= self.deconvolute_opt_group(opt_group, config)

        rgrs_to_remove = {
            rgr
            for rgr in self.rgr_set
            if rgr.type == "ORF" and rgr.id in rgr_ids_to_remove
        }

        self.remove_rgrs(rgrs_to_remove)

    def make_stop_groups(
        self,
    ) -> dict[int, list[ReadGeneratingRegion]]:
        """Group ORF RGRs by their stop-codon position.

        Noise RGRs are excluded.  Groups with a single member are
        dropped since they cannot be deconvolved.

        Returns
        -------
        dict[int, list[ReadGeneratingRegion]]
            Mapping from stop-codon genomic position to the list of
            ORF RGRs ending there.
        """
        stop_groups = {}
        for rgr in self.rgr_set:
            if rgr.type == "NOISE":
                continue
            if rgr.genomic_region.strand == "+":
                stop = rgr.genomic_region.intervals[-1].end
            else:
                stop = rgr.genomic_region.intervals[0].start
            try:
                stop_groups[stop].append(rgr)
            except KeyError:
                stop_groups[stop] = [rgr]

        stop_groups = {k: v for k, v in stop_groups.items() if len(v) > 1}

        return stop_groups

    def split_stop_groups(
        self,
        stop_groups: dict[int, list[ReadGeneratingRegion]],
    ) -> list[set[ReadGeneratingRegion]]:
        """Split stop groups into splice-compatible optimisation groups.

        RGRs sharing a stop codon may have incompatible exon–intron
        structures.  This method partitions each stop group into
        maximal subsets of mutually compatible RGRs.

        Parameters
        ----------
        stop_groups : dict[int, list[ReadGeneratingRegion]]
            Stop groups produced by :meth:`make_stop_groups`.

        Returns
        -------
        list[set[ReadGeneratingRegion]]
            Each element is a set of compatible RGRs to deconvolve
            together.
        """
        optimization_groups: list[set[ReadGeneratingRegion]] = []
        for stop_group in stop_groups.values():
            rgrs = list(stop_group)
            containment_dict: dict[ReadGeneratingRegion, set[ReadGeneratingRegion]] = {}
            for rgr in rgrs:
                containment_dict[rgr] = {
                    other
                    for other in rgrs
                    if rgr.genomic_region.contains_to_stop(other.genomic_region)
                }

            remaining = list(containment_dict.values())
            while remaining:
                big_set = max(remaining, key=len)
                optimization_groups.append(big_set)
                remaining = [s for s in remaining if not s.issubset(big_set)]

        return optimization_groups

    def deconvolute_opt_group(
        self,
        opt_group: set[ReadGeneratingRegion],
        config: Config,
    ) -> set[str]:
        """Deconvolve a single optimisation group and return ORF ids to remove.

        Each run is independently optimised using L-BFGS-B.  An ORF is
        kept if its estimated activity exceeds
        ``config.deconvolution_filter_min_activity`` in at least one run.

        Parameters
        ----------
        opt_group : set[ReadGeneratingRegion]
            Set of compatible RGRs to deconvolve together.
        config : Config
            Configuration providing filter thresholds.

        Returns
        -------
        set[str]
            RGR identifiers that should be removed.
        """
        rgr_indices_to_keep: list[set[int]] = []
        sorted_rgrs = sorted(opt_group, key=len, reverse=True)
        rgr_indices = {rgr.id: i for i, rgr in enumerate(sorted_rgrs)}

        min_reads = self.wfr_df.sum().sum() / self.wfr_df.shape[1] * 0.1

        number_of_runs = self.wfr_df.shape[1]
        theta = _distribution_theta(config)

        for run_idx in range(number_of_runs):
            rgr_read_counts = self.wfr_df.iloc[:, run_idx].to_dict()

            # skip if the locus is probably not expressed in this run
            if sum(rgr_read_counts.values()) < min_reads:
                continue
            egs: dict[frozenset[str], tuple[int, int]] = {}
            s: set[str] = set()
            for j in range(len(sorted_rgrs) - 1):
                s.add(sorted_rgrs[j].id)
                length = len(sorted_rgrs[j]) - len(sorted_rgrs[j + 1])
                # Reads compatible with the shorter RGR are almost always also
                # compatible with the longer one that contains it, making this
                # difference the count of reads unique to the longer RGR.  The
                # partial-overlap likelihood test bounds each RGR by its own
                # coordinates, so compatibility is not strictly monotone and the
                # difference can turn slightly negative.  A count cannot be
                # negative, and the negative-binomial denominator
                # ``X^T(w (y + theta) / (theta + delta))`` would flip sign and
                # diverge if it were.
                rc = max(
                    rgr_read_counts[sorted_rgrs[j].id]
                    - rgr_read_counts[sorted_rgrs[j + 1].id],
                    0,
                )
                egs[frozenset(s)] = (length, rc)

            s.add(sorted_rgrs[-1].id)
            length = len(sorted_rgrs[-1])
            rc = rgr_read_counts[sorted_rgrs[-1].id]
            egs[frozenset(s)] = (length, rc)

            bounds = [(config.pseudo_min, None) for _ in range(len(egs))]
            initial_guess = np.full(len(egs), 0.1)

            eg_lengths = np.array([egs[eg][0] for eg in egs])
            eg_read_counts = np.array([egs[eg][1] for eg in egs], dtype=np.float64)

            # Build sparse design matrix: X[row, rgr_idx] = eg_length for each RGR in that EG
            n_rgrs_filter = len(sorted_rgrs)
            rows_f: list[int] = []
            cols_f: list[int] = []
            data_f: list[float] = []
            for i, eg in enumerate(egs):
                for rgr_id_str in eg:
                    rows_f.append(i)
                    cols_f.append(rgr_indices[rgr_id_str])
                    data_f.append(float(eg_lengths[i]))
            X_filter = csr_matrix(
                (data_f, (rows_f, cols_f)),
                shape=(len(egs), n_rgrs_filter),
                dtype=np.float64,
            )

            if getattr(config, "inner_solver", "lbfgs") == "mu":
                from price2 import mu_solver

                result_x = mu_solver.mu_inner_cpu(
                    X_filter, X_filter.T.tocsr(), eg_read_counts,
                    np.ones(X_filter.shape[0]), initial_guess, 0.0,
                    len(initial_guess), 1, config.pseudo_min,
                    getattr(config, "mu_inner_max_iter", 3000),
                    getattr(config, "mu_inner_tol", 1e-5), theta=theta)
            else:
                result_x = minimize(
                    poisson_nll_grad,
                    initial_guess,
                    args=(X_filter, eg_read_counts, theta),
                    bounds=bounds,
                    method="L-BFGS-B",
                    jac=True,
                    options={"maxiter": 10_000},
                ).x

            rgr_indices_to_keep_one_run = set(
                np.where(result_x >= config.deconvolution_filter_min_activity)[0]
            )
            rgr_indices_to_keep.append(rgr_indices_to_keep_one_run)

        try:
            all_kept = set.union(*rgr_indices_to_keep)
        except TypeError:
            all_kept = set()

        return {k for k, v in rgr_indices.items() if v not in all_kept}

    def assign_reads_to_egs(
        self,
        runs: list[RiboSeqRun],
        mm_data: dict | None = None,
        build_cache: bool = False,
    ) -> None:
        """Assign reads to their equivalence groups.

        Each read is matched to its ``(rgr_frame_covpos, length, oua)``
        key and added to the corresponding :class:`EquivalenceGroup`.
        Reads whose key is absent (due to earlier filtering) are counted
        in :attr:`uncounted_reads`.

        When ``mm_data`` is supplied (multimapping EM mode), each
        cross-locus multimapping read contributes a *fractional* count at
        this locus instead of its full count.  For a slot with collapsed
        count ``c``, baseline cross-locus mass ``base`` and current
        fractional weight ``weight`` the effective contribution is
        ``c - base + weight`` (single-slot multimappers keep full weight;
        cross-locus reads are down-weighted so their total mass across all
        their loci sums to one).  The routing needed by the E-step is
        cached in :attr:`mm_slots`.

        Parameters
        ----------
        runs : list[RiboSeqRun]
            Ribo-seq runs to process.
        mm_data : dict, optional
            ``{run_id: {group_key: (base, weight)}}`` for this locus's
            multimapping slots, or ``None`` for classic full-weight
            counting.
        build_cache : bool, optional
            Record the weight-independent read routing and design-matrix
            geometry in :attr:`eg_cache` while assigning.  Later EM
            iterations then take :meth:`_assign_reads_from_cache`, which
            skips the per-read :meth:`get_rgr_frame_covpos` recomputation.
        """
        cache = getattr(self, "eg_cache", None)
        if cache is not None and not build_cache:
            self._assign_reads_from_cache(runs, mm_data, cache)
            return

        # Row layout of the design matrix, in the order ``egs_to_sparse``
        # emits rows.  Built up-front so each read can record the row it
        # feeds instead of re-deriving its equivalence-group key later.
        if build_cache:
            row_of_key: dict = {}
            n_rows = 0
            for run in runs:
                d: dict = {}
                for key in self.egs[run]:
                    if not key[0]:
                        continue
                    d[key] = n_rows
                    n_rows += 1
                row_of_key[run.id] = d
            eg_row: dict = {}
            counts0: dict = {}
            mm_idx: dict = {}
            mm_gk: dict = {}
            mm_base: dict = {}

        # {run_id: {group_key: (rgr_frame_covpos, read_length, oua)}} —
        # the compatibility routing the E-step needs to recompute λ.
        self.mm_slots = {run.id: {} for run in runs}
        for run in runs:
            run_id = run.id
            run_mm = mm_data.get(run_id) if mm_data else None

            if build_cache:
                n_reads = len(self.rsas_dict[run_id])
                rows_arr = np.full(n_reads, -2, dtype=np.int32)
                counts_arr = np.zeros(n_reads, dtype=np.float64)
                run_rows = row_of_key[run_id]
                mi: list = []
                mg: list = []
                mb: list = []

            for i, rsa in enumerate(self.rsas_dict[run_id]):
                rgr_frame_covpos = self.get_rgr_frame_covpos(rsa, run)
                read_count = rsa.read_count

                if build_cache:
                    counts_arr[i] = rsa.read_count
                    if rgr_frame_covpos:
                        rows_arr[i] = run_rows.get(
                            (
                                rgr_frame_covpos,
                                len(rsa),
                                rsa.untemplated_addition,
                            ),
                            -1,
                        )

                if run_mm is not None and not rsa.unique():
                    gk = multimap.alignment_group_key(rsa)
                    slot = run_mm.get(gk)
                    if slot is not None:
                        base, weight = slot
                        # ``base`` is summed over the spilled alignments at
                        # indexing time, independently of the collapsed slot
                        # count ``read_count``; floor the non-cross-locus
                        # remainder at zero so that any disagreement between
                        # the two can never drive the Poisson response
                        # negative.
                        read_count = max(0.0, read_count - base) + weight
                        if build_cache:
                            mi.append(i)
                            mg.append(gk)
                            mb.append(base)
                        if rgr_frame_covpos:
                            self.mm_slots[run_id][gk] = (
                                rgr_frame_covpos,
                                len(rsa),
                                rsa.untemplated_addition,
                            )

                if not rgr_frame_covpos:
                    continue

                try:
                    self.egs[run][
                        (rgr_frame_covpos, len(rsa), rsa.untemplated_addition)
                    ].read_count += read_count
                    run.read_count += read_count
                except KeyError:
                    self.uncounted_reads += read_count

                try:
                    self.read_counts[run] += read_count
                except KeyError:
                    self.read_counts[run] = read_count

            if build_cache:
                eg_row[run_id] = rows_arr
                counts0[run_id] = counts_arr
                mm_idx[run_id] = np.array(mi, dtype=np.int32)
                mm_gk[run_id] = np.array(mg, dtype=np.int64)
                mm_base[run_id] = np.array(mb, dtype=np.float64)

        self.counted_reads = {}
        for run in runs:
            self.counted_reads[run.id] = 0
            for v in self.egs[run].values():
                self.counted_reads[run.id] += v.read_count

        if build_cache:
            self.eg_cache = self._make_eg_cache(
                runs, n_rows, eg_row, counts0, mm_idx, mm_gk, mm_base
            )

    def _make_eg_cache(
        self,
        runs: list[RiboSeqRun],
        n_rows: int,
        eg_row: dict,
        counts0: dict,
        mm_idx: dict,
        mm_gk: dict,
        mm_base: dict,
    ) -> EgRoutingCache:
        """Freeze the design-matrix geometry into a compact cell encoding.

        Storing ``X`` itself would add ~1.1 MB per locus; instead the cells
        are stored as ``(rgr index, frame*3 + coverage position)`` pairs plus
        per-row ``(length, read length, oua, run)``, from which ``X.data``
        is recomputed with a handful of vectorised look-ups.

        The multimapping-slot routing is encoded the same way, so that the
        cache holds no references to RGR objects and can be loaded without
        the rest of the locus.
        """
        cell_rgr: list = []
        cell_code: list = []
        row_nnz: list = []
        row_len: list = []
        row_rl: list = []
        row_oua: list = []
        row_run: list = []
        for run_index, run in enumerate(runs):
            for (rgr_frame_covpos, read_length, oua), eg in self.egs[run].items():
                if not rgr_frame_covpos:
                    continue
                row_nnz.append(len(rgr_frame_covpos))
                row_len.append(eg.length)
                row_rl.append(read_length)
                row_oua.append(int(oua))
                row_run.append(run_index)
                for rgr, frame, cov_pos in rgr_frame_covpos:
                    cell_rgr.append(rgr.index)
                    cell_code.append(
                        (3 if frame is None else frame) * 3 + cov_pos.value
                    )

        slot_gk: dict = {}
        slot_rl: dict = {}
        slot_oua: dict = {}
        slot_nnz: dict = {}
        slot_rgr: dict = {}
        slot_code: dict = {}
        for run in runs:
            gks, rls, ouas, nnzs, srgr, scode = [], [], [], [], [], []
            for gk, (rfc, read_length, oua) in self.mm_slots[run.id].items():
                gks.append(gk)
                rls.append(read_length)
                ouas.append(int(oua))
                nnzs.append(len(rfc))
                for rgr, frame, cov_pos in rfc:
                    srgr.append(rgr.index)
                    scode.append((3 if frame is None else frame) * 3 + cov_pos.value)
            slot_gk[run.id] = np.array(gks, dtype=np.int64)
            slot_rl[run.id] = np.array(rls, dtype=np.int32)
            slot_oua[run.id] = np.array(ouas, dtype=np.uint8)
            slot_nnz[run.id] = np.array(nnzs, dtype=np.int32)
            slot_rgr[run.id] = np.array(srgr, dtype=np.int32)
            slot_code[run.id] = np.array(scode, dtype=np.uint8)

        num_rgrs = len(self.rgr_set)
        rgr_ids: list = [None] * num_rgrs
        rgr_lengths = np.empty(num_rgrs, dtype=np.int64)
        for rgr in self.rgr_set:
            rgr_ids[rgr.index] = rgr.id
            rgr_lengths[rgr.index] = len(rgr)

        return EgRoutingCache(
            n_rows=n_rows,
            n_reads={r.id: len(self.rsas_dict[r.id]) for r in runs},
            eg_row=eg_row,
            counts0=counts0,
            mm_idx=mm_idx,
            mm_gk=mm_gk,
            mm_base=mm_base,
            slot_gk=slot_gk,
            slot_rl=slot_rl,
            slot_oua=slot_oua,
            slot_nnz=slot_nnz,
            slot_rgr=slot_rgr,
            slot_code=slot_code,
            num_rgrs=num_rgrs,
            rgr_ids=tuple(rgr_ids),
            rgr_lengths=rgr_lengths,
            cell_rgr=np.array(cell_rgr, dtype=np.int32),
            cell_code=np.array(cell_code, dtype=np.uint8),
            row_nnz=np.array(row_nnz, dtype=np.int32),
            row_len=np.array(row_len, dtype=np.int64),
            row_rl=np.array(row_rl, dtype=np.int32),
            row_oua=np.array(row_oua, dtype=np.uint8),
            row_run=np.array(row_run, dtype=np.uint8),
        )

    def _assign_reads_from_cache(
        self,
        runs: list[RiboSeqRun],
        mm_data: dict | None,
        cache: EgRoutingCache,
    ) -> None:
        """Re-derive the response ``y`` from cached routing (no per-read work).

        Only the fractional weights change between EM iterations, so the
        per-read equivalence-group routing recorded in *cache* stays valid and
        the response reduces to a weighted ``bincount`` over rows.
        """
        y = np.zeros(cache.n_rows, dtype=np.float64)
        self.read_counts = {}
        self.counted_reads = {}
        self.uncounted_reads = 0.0

        for run in runs:
            run_id = run.id
            if len(self.rsas_dict[run_id]) != cache.n_reads[run_id]:
                raise RuntimeError(
                    f"locus {self.id}: cached routing has "
                    f"{cache.n_reads[run_id]} reads for run {run_id} but "
                    f"{len(self.rsas_dict[run_id])} were loaded"
                )
            counts = cache.counts0[run_id].copy()
            run_mm = mm_data.get(run_id) if mm_data else None
            idx = cache.mm_idx[run_id]
            if run_mm is not None and idx.size:
                weights = np.fromiter(
                    (run_mm[gk][1] for gk in cache.mm_gk[run_id]),
                    dtype=np.float64,
                    count=idx.size,
                )
                counts[idx] = (
                    np.maximum(0.0, counts[idx] - cache.mm_base[run_id]) + weights
                )

            rows = cache.eg_row[run_id]
            counted = rows >= 0
            y += np.bincount(
                rows[counted], weights=counts[counted], minlength=cache.n_rows
            )
            counted_sum = float(counts[counted].sum())
            uncounted = rows == -1
            self.uncounted_reads += float(counts[uncounted].sum())
            self.counted_reads[run_id] = counted_sum
            run.read_count += counted_sum
            # ``read_counts`` gains an entry only when the run has at least one
            # read compatible with some RGR, matching the uncached path.
            compatible = counted | uncounted
            if compatible.any():
                self.read_counts[run] = float(counts[compatible].sum())

        # Downstream consumers (the final pass's pruning, the likelihood-ratio
        # test, ``estimate_activities``) read counts off the EG objects.  A
        # light M-step loads only the cache, so there are no EG objects and
        # nothing downstream that reads them.
        if getattr(self, "egs", None):
            i = 0
            for run in runs:
                for key, eg in self.egs[run].items():
                    if not key[0]:
                        continue
                    eg.read_count = y[i]
                    i += 1
        self._eg_y = y

    def _design_matrix_from_cache(
        self,
        cache: EgRoutingCache,
        cm_lut: np.ndarray,
        coverage_params: np.ndarray,
        num_runs: int,
    ) -> csr_matrix:
        """Rebuild ``X`` from the cached cell encoding, fully vectorised."""
        nnz_per_row = cache.row_nnz
        run_c = np.repeat(cache.row_run, nnz_per_row)
        rl_c = np.repeat(cache.row_rl, nnz_per_row)
        oua_c = np.repeat(cache.row_oua, nnz_per_row)
        len_c = np.repeat(cache.row_len, nnz_per_row)
        code = cache.cell_code.astype(np.int64)
        frame_c, cov_c = code // 3, code % 3

        data = (
            len_c
            * cm_lut[run_c, rl_c, frame_c, oua_c]
            * coverage_params[run_c, cov_c]
        )
        rows_idx = np.repeat(np.arange(cache.n_rows, dtype=np.int64), nnz_per_row)
        cols_idx = cache.cell_rgr.astype(np.int64) * num_runs + run_c
        # COO construction (as in ``egs_to_sparse``) so that a row touching the
        # same RGR at several coverage positions sums those cells.
        return csr_matrix(
            (data, (rows_idx, cols_idx)),
            shape=(cache.n_rows, cache.num_rgrs * num_runs),
            dtype=np.float64,
        )

    def to_sparse_args(
        self,
        runs: list[RiboSeqRun],
    ) -> dict:
        """Build argument dictionary for the sparse-matrix objective functions.

        Assembles cleavage-model look-up tables, coverage-model parameters,
        a CSR sparse design matrix, the response vector, and an initial-guess
        vector.

        Parameters
        ----------
        runs : list[RiboSeqRun]
            Ribo-seq runs to include.

        Returns
        -------
        dict
            Keys: ``X``, ``y``, ``cleavage_model``, ``coverage_model``,
            ``num_rgrs``, ``rgr_lengths``, ``num_runs``, ``initial_guess``.
        """
        num_runs = len(runs)

        cm_lut = np.zeros((num_runs, runs[0].cleavage_model.cds_lut.shape[0], 4, 2))
        for i, run in enumerate(runs):
            cm_lut[i, :, 3, :] = run.cleavage_model.noise_lut
            cm_lut[i, :, :3, :] = run.cleavage_model.cds_lut

        coverage_params = np.zeros((num_runs, 3))
        for i, run in enumerate(runs):
            coverage_params[i, 0] = run.coverage_model.start_factor
            coverage_params[i, 1] = 1
            coverage_params[i, 2] = run.coverage_model.stop_factor

        # Must be aligned with ``rgr.index`` (the design-matrix column blocks
        # and the rows of ``result``), not with ``rgr_set`` iteration order:
        # the two are different permutations, and ``deconvolve`` multiplies
        # ``rgr_lengths`` against the index-aligned activity matrix.
        cache = getattr(self, "eg_cache", None)
        if cache is not None:
            num_rgrs = cache.num_rgrs
            rgr_lengths = cache.rgr_lengths
        else:
            num_rgrs = len(self.rgr_set)
            rgr_lengths = np.empty(num_rgrs, dtype=np.int64)
            for rgr in self.rgr_set:
                rgr_lengths[rgr.index] = len(rgr)

        if hasattr(self, "result"):
            initial_guess = self.result
        else:
            initial_guess = np.ones((num_rgrs, num_runs))
        initial_guess = initial_guess.flatten()

        # ``X`` depends only on the equivalence-group geometry and the
        # cleavage/coverage models, all fixed across EM iterations, so a
        # cached locus rebuilds it vectorised instead of walking every cell
        # in Python.  ``y`` came out of the cached read routing.
        y = getattr(self, "_eg_y", None)
        if cache is not None and y is not None:
            X = self._design_matrix_from_cache(
                cache, cm_lut, coverage_params, num_runs
            )
        else:
            X, y = egs_to_sparse(
                self.egs, runs, cm_lut, coverage_params, num_rgrs, num_runs
            )

        return {
            "X": X,
            "y": y,
            "cleavage_model": cm_lut,
            "coverage_model": coverage_params,
            "num_rgrs": num_rgrs,
            "rgr_lengths": rgr_lengths,
            "num_runs": num_runs,
            "initial_guess": initial_guess,
        }

    def deconvolve(
        self,
        config: Config,
        runs: list[RiboSeqRun],
        max_outer: int | None = None,
        prune: bool = True,
    ) -> tuple[float, float]:
        """IRLS deconvolution with Huber weights on Pearson residuals.

        At each outer iteration:
          1. Compute fitted values δ = X @ w and Pearson residuals.
          2. Compute Huber weights: ω_i = min(1, c / |r_i|).
          3. Solve weighted group-LASSO Poisson NLL.

        Converges when the relative change in w falls below
        ``config.irls_huber_tol``.

        Warm-starts from ``self.result`` when present (via
        :meth:`to_sparse_args`); with no prior result the initial guess is
        all ones, identical to a cold start.

        Parameters
        ----------
        config : Config
            Configuration providing ``irls_huber_c``,
            ``irls_huber_max_outer``, ``irls_huber_tol``, and
            standard optimisation parameters.
        runs : list[RiboSeqRun]
            Ribo-seq runs to deconvolve jointly.
        max_outer : int, optional
            Cap on IRLS-Huber outer iterations.  ``None`` uses
            ``config.irls_huber_max_outer``.  The EM light M-step passes a
            small value (e.g. 1) so one Huber reweight interleaves with
            each global E-step.
        prune : bool, optional
            When ``True`` (default) low-activity ORFs are removed after
            the solve.  The EM light M-step passes ``False`` to keep the
            ORF set — and hence the E-step targets and warm-start layout —
            fixed across iterations.

        Returns
        -------
        tuple[float, float]
            ``(opt_time, data_time)`` — wall-clock seconds spent in
            optimisation vs. data preparation.
        """
        logger = logging.getLogger("price2")
        opt_time = 0.0
        data_time = 0.0

        # ── Build sparse system ──────────────────────────────────────────
        s1 = time.time()
        args_dict = self.to_sparse_args(runs)
        X = args_dict["X"]
        y = args_dict["y"]
        num_rgrs = args_dict["num_rgrs"]
        num_runs = args_dict["num_runs"]
        rgr_lengths = args_dict["rgr_lengths"]
        s2 = time.time()
        data_time += s2 - s1

        n = num_rgrs * num_runs
        bounds = [(config.pseudo_min, None)] * n
        c = config.irls_huber_c
        theta = _distribution_theta(config)

        w_current = args_dict["initial_guess"]

        n_outer = (
            config.irls_huber_max_outer if max_outer is None else max_outer
        )

        # Inner-solver setup. "mu" swaps the scipy L-BFGS-B inner solve for
        # multiplicative (weighted Richardson-Lucy) updates, optionally on GPU.
        use_mu = getattr(config, "inner_solver", "lbfgs") == "mu"
        if use_mu:
            from price2 import mu_solver

            XT = X.T.tocsr()
            gpu = None
            if (
                getattr(config, "mu_gpu", False)
                and X.shape[0] >= getattr(config, "mu_gpu_min_rows", 50_000)
            ):
                try:
                    gpu = mu_solver.GpuMuSolver(
                        X, y, getattr(config, "mu_dtype", "float32")
                    )
                except Exception as exc:  # torch/CUDA missing -> CPU fallback
                    logger.warning("GPU MU unavailable (%s); using CPU", exc)

        # Optional single-context GPU broker: when config carries a broker
        # request queue, ship the whole system to the broker (one shared CUDA
        # context for the entire pool) instead of each worker holding its own.
        # The broker runs the full IRLS-Huber MU loop and returns w.
        broker_req_q = getattr(config, "mu_broker_req_q", None)
        use_broker = (
            use_mu and broker_req_q is not None
            and X.shape[0] >= getattr(config, "mu_gpu_min_rows", 50_000)
        )

        s1 = time.time()
        # When the broker is active it runs the full IRLS-Huber MU loop on the
        # GPU and returns w directly, so the local Python loop below is skipped
        # (n_outer -> 0). Honour the caller's outer budget: the EM light M-step
        # passes a small max_outer (already folded into n_outer above) instead
        # of the config maximum.
        if use_broker:
            from price2.gpu_broker import BrokerClient, Params as _BParams
            _bp = _BParams(
                num_rgrs=num_rgrs, num_runs=num_runs, lam=config.lam,
                pseudo_min=config.pseudo_min, huber_c=c,
                max_outer=n_outer,
                huber_tol=config.irls_huber_tol,
                mu_inner_max_iter=getattr(config, "mu_inner_max_iter", 3000),
                mu_inner_tol=getattr(config, "mu_inner_tol", 1e-5),
                theta=theta)
            w_current = BrokerClient(broker_req_q).solve(X, XT, y, _bp, w0=w_current)
            broker_outer = n_outer
            n_outer = 0
        # Active-set-plateau stopping criterion (flag-gated). The IRLS-Huber
        # rel-change metric keeps shrinking geometrically long after the set of
        # ORFs above the activity filter has stabilised; stop once that set is
        # unchanged for `irls_active_patience` consecutive outer iterations.
        _use_active_stop = getattr(config, "irls_stop_on_active_set", False)
        _active_patience = getattr(config, "irls_active_patience", 2)
        _thr_hi = getattr(config, "deconvolution_filter_min_activity", 0.1)
        _prev_active = None
        _stable_count = 0
        outer = -1
        for outer in range(n_outer):
            # Fitted values and Huber weights on (NB-aware) Pearson residuals
            delta = np.asarray(X @ w_current).ravel()
            weights = _huber_weights(y, delta, c, theta)

            # Solve weighted group-LASSO count-model NLL
            if use_mu:
                mi = getattr(config, "mu_inner_max_iter", 3000)
                mt = getattr(config, "mu_inner_tol", 1e-5)
                if gpu is not None:
                    w_new = gpu.solve(weights, w_current, config.lam,
                                      num_rgrs, num_runs, config.pseudo_min, mi, mt,
                                      theta=theta)
                else:
                    w_new = mu_solver.mu_inner_cpu(
                        X, XT, y, weights, w_current, config.lam,
                        num_rgrs, num_runs, config.pseudo_min, mi, mt,
                        theta=theta)
            else:
                cb = Callback(w_current, config)
                result = minimize(
                    weighted_poisson_nll_grad_lasso,
                    w_current,
                    args=(X, y, weights, config.lam, num_rgrs, num_runs, theta),
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                    callback=cb,
                    options={
                        "maxiter": 10_000,
                        "ftol": config.ftol,
                        "gtol": config.gtol,
                        "maxls": config.maxls,
                    },
                )
                w_new = result.x

            rel_change = np.linalg.norm(w_new - w_current) / max(
                np.linalg.norm(w_current), 1e-14
            )
            w_current = w_new
            if rel_change < config.irls_huber_tol:
                break
            if _use_active_stop:
                _g = np.sqrt(
                    (w_new.reshape(num_rgrs, num_runs) ** 2).sum(axis=1)
                )
                _active = frozenset(np.nonzero(_g > _thr_hi)[0].tolist())
                if _active == _prev_active:
                    _stable_count += 1
                else:
                    _stable_count = 0
                _prev_active = _active
                if _stable_count >= _active_patience:
                    break

        s2 = time.time()
        opt_time += s2 - s1
        self.irls_outer_iterations = (
            broker_outer if use_broker else outer + 1
        )
        logger.debug(
            "IRLS-Huber: converged in %d outer iterations (c=%.1f)",
            outer + 1,
            c,
        )

        # ── Store result ─────────────────────────────────────────────────
        result_matrix = w_current.reshape(num_rgrs, num_runs)
        result_matrix[result_matrix <= config.pseudo_min] = 0
        self.result = result_matrix

        # Store Huber weights for use in weighted LRT
        delta = np.asarray(X @ w_current).ravel()
        self.irls_huber_weights = _huber_weights(y, delta, c, theta)

        # ── Post-optimisation RGR removal (same as deconvolve) ───────────
        if prune:
            x = self.result
            x_t = x.T
            canonical_indices = (rgr_lengths * x_t).argmax(axis=1)
            min_activities = np.maximum(
                x_t[np.arange(x_t.shape[0]), canonical_indices]
                * config.min_activity_fraction,
                config.rgr_min_activity,
            )
            self.rgr_indices_to_remove = set(
                np.where(np.all(x < min_activities, axis=1))[0]
            )
            rgrs_to_remove = set(
                [
                    rgr
                    for rgr in self.rgr_set
                    if rgr.index in self.rgr_indices_to_remove
                    and rgr.type == "ORF"
                ]
            )
            self.remove_rgrs(rgrs_to_remove, runs=runs)

        return opt_time, data_time

    def activities_by_id(self) -> dict:
        """Return the current activity matrix keyed by stable ``rgr.id``.

        Keying by ``rgr.id`` (rather than the volatile ``rgr.index``,
        which index densification reassigns) lets the activities be
        reloaded as a warm start in the next EM iteration.

        Returns
        -------
        dict
            ``{rgr_id: numpy.ndarray of shape (num_runs,)}``.
        """
        cache = getattr(self, "eg_cache", None)
        if cache is not None:
            return {
                rgr_id: self.result[index].copy()
                for index, rgr_id in enumerate(cache.rgr_ids)
            }
        return {rgr.id: self.result[rgr.index].copy() for rgr in self.rgr_set}

    def set_warm_start(self, activities: dict, num_runs: int) -> None:
        """Seed :attr:`result` from persisted per-``rgr.id`` activities.

        RGRs without a stored activity (new to this iteration) default to
        ones.  Must be called once the RGR set is final for the iteration
        (i.e. after all pre-deconvolution filters).

        Parameters
        ----------
        activities : dict
            ``{rgr_id: numpy.ndarray}`` from a previous M-step.
        num_runs : int
            Number of Ribo-seq runs (columns of the activity matrix).
        """
        cache = getattr(self, "eg_cache", None)
        if cache is not None:
            result = np.ones((cache.num_rgrs, num_runs))
            for index, rgr_id in enumerate(cache.rgr_ids):
                a = activities.get(rgr_id)
                if a is not None:
                    result[index] = a
        else:
            result = np.ones((len(self.rgr_set), num_runs))
            for rgr in self.rgr_set:
                a = activities.get(rgr.id)
                if a is not None:
                    result[rgr.index] = a
        self.result = result

    def compute_multimap_lambdas(self, runs: list[RiboSeqRun]) -> list:
        """Compute the per-slot origin rate ``λ`` for multimapping reads.

        For each recorded multimapping slot, ``λ`` is the read's design-
        matrix row *without* the geometric ``length`` factor dotted with
        the current activities — i.e. ``Σ cleavage · coverage · activity``
        over the read's compatible ORFs, which is the per-read expected
        rate the E-step normalises across a read's loci
        (``λ = δ_EG / length_EG``).

        Parameters
        ----------
        runs : list[RiboSeqRun]
            Ribo-seq runs, in the order used to build :attr:`result`.

        Returns
        -------
        list of (run_id, group_key, lam)
            One entry per multimapping slot recorded at this locus.
        """
        cache = getattr(self, "eg_cache", None)
        if cache is None and not any(self.mm_slots.get(run.id) for run in runs):
            return []

        num_runs = len(runs)
        cm_lut = np.zeros(
            (num_runs, runs[0].cleavage_model.cds_lut.shape[0], 4, 2)
        )
        for i, run in enumerate(runs):
            cm_lut[i, :, 3, :] = run.cleavage_model.noise_lut
            cm_lut[i, :, :3, :] = run.cleavage_model.cds_lut

        coverage_params = np.zeros((num_runs, 3))
        for i, run in enumerate(runs):
            coverage_params[i, 0] = run.coverage_model.start_factor
            coverage_params[i, 1] = 1
            coverage_params[i, 2] = run.coverage_model.stop_factor

        if cache is not None:
            return self._multimap_lambdas_from_cache(
                runs, cache, cm_lut, coverage_params
            )

        out = []
        for run_index, run in enumerate(runs):
            for gk, (rfc, read_length, oua) in self.mm_slots[run.id].items():
                oua_int = int(oua)
                lam = 0.0
                for rgr, frame, cov_pos in rfc:
                    f = 3 if frame is None else frame
                    lam += (
                        cm_lut[run_index, read_length, f, oua_int]
                        * coverage_params[run_index, cov_pos.value]
                        * self.result[rgr.index, run_index]
                    )
                out.append((run.id, gk, float(lam)))
        return out

    def _multimap_lambdas_from_cache(
        self,
        runs: list[RiboSeqRun],
        cache: EgRoutingCache,
        cm_lut: np.ndarray,
        coverage_params: np.ndarray,
    ) -> list:
        """Vectorised :meth:`compute_multimap_lambdas` over the cached slots."""
        out: list = []
        for run_index, run in enumerate(runs):
            gks = cache.slot_gk[run.id]
            if gks.size == 0:
                continue
            nnz = cache.slot_nnz[run.id]
            read_length = np.repeat(cache.slot_rl[run.id], nnz)
            oua = np.repeat(cache.slot_oua[run.id], nnz)
            code = cache.slot_code[run.id].astype(np.int64)
            contribution = (
                cm_lut[run_index, read_length, code // 3, oua]
                * coverage_params[run_index, code % 3]
                * self.result[cache.slot_rgr[run.id], run_index]
            )
            slot_of_cell = np.repeat(np.arange(gks.size), nnz)
            lam = np.bincount(
                slot_of_cell, weights=contribution, minlength=gks.size
            )
            out.extend(
                (run.id, int(gk), float(value)) for gk, value in zip(gks, lam)
            )
        return out

    def remove_rgrs(
        self,
        rgrs_to_remove: set[ReadGeneratingRegion],
        runs: list[RiboSeqRun] | None = None,
    ) -> None:
        """Remove a set of RGRs and update all dependent data structures.

        Updates :attr:`rgr_set`, re-indexes remaining RGRs, invalidates the
        lazy :attr:`rgr_intervals` cache, collapses equivalence groups (if
        present), and re-slices :attr:`result` (if present).

        Parameters
        ----------
        rgrs_to_remove : set[ReadGeneratingRegion]
            RGRs to discard.
        runs : list[RiboSeqRun] or None
            Required when equivalence groups need collapsing.
        """
        old_rgr_set = self.rgr_set
        self.rgr_set = self.rgr_set - rgrs_to_remove

        # The cached routing keys off rgr.index and the equivalence-group
        # layout, both of which this method invalidates.
        self.eg_cache = None
        self._eg_y = None

        # Rows of ``result`` are keyed by the *old* rgr.index, which the
        # re-indexing below overwrites.  ``rgr_set`` iteration order is not
        # the index order (it changes across a pickle round-trip), so the old
        # indices have to be captured rather than re-derived by enumeration.
        old_indices = (
            {rgr: rgr.index for rgr in old_rgr_set}
            if hasattr(self, "result")
            else None
        )

        # rgr indices
        for c, rgr in enumerate(self.rgr_set):
            rgr.index = c

        # Invalidate the lazily-built rgr_intervals cache; it is rebuilt from
        # the current rgr_set on next access (nothing in the pipeline reads it).
        self._rgr_intervals = None

        # egs
        if hasattr(self, "egs"):
            self.collapse_egs(runs)

        # results
        if hasattr(self, "result"):
            index_array = np.zeros(len(self.rgr_set), dtype=int)
            for rgr in self.rgr_set:
                index_array[rgr.index] = old_indices[rgr]
            self.result = self.result[index_array]

    def collapse_egs(
        self,
        runs: list[RiboSeqRun],
    ) -> None:
        """Collapse equivalence groups after RGR removal.

        Rebuilds the per-run equivalence-group dictionaries, merging
        entries whose keys become identical after removed RGRs are
        dropped from the key's ``rgr_frame_covpos`` frozenset.

        Two memory optimisations:

        * an ``old_to_new`` cache maps each old key to its remapped key, so
          the new ``rgr_frame_covpos`` frozenset is materialised only once
          per distinct old key (rather than once per (old key, run));
        * per-run dicts are rebuilt one at a time and the old dict for
          that run is released immediately, bounding the doubled-allocation
          transient to a single run instead of the full ``len(runs)``.

        Parameters
        ----------
        runs : list[RiboSeqRun]
            Ribo-seq runs whose EGs should be rebuilt.
        """
        rgr_set = self.rgr_set
        old_to_new: dict[tuple, tuple] = {}

        new_egs: dict = {}
        for run in runs:
            old_run_egs = self.egs.pop(run)
            new_run_egs: dict = defaultdict(EquivalenceGroup)
            for old_eg_key, old_eg in old_run_egs.items():
                new_eg_key = old_to_new.get(old_eg_key)
                if new_eg_key is None:
                    rgr_frame_covpos, read_length, oua = old_eg_key
                    new_rgr_frame_covpos = frozenset(
                        (rgr, frame, covpos)
                        for rgr, frame, covpos in rgr_frame_covpos
                        if rgr in rgr_set
                    )
                    new_eg_key = (new_rgr_frame_covpos, read_length, oua)
                    old_to_new[old_eg_key] = new_eg_key

                new_eg = new_run_egs[new_eg_key]
                new_eg.length += old_eg.length
                new_eg.read_count += old_eg.read_count

            new_egs[run] = new_run_egs

        self.egs = new_egs

    def likelihood_ratio_filtering(
        self,
        config: Config,
        runs: list[RiboSeqRun],
    ) -> None:
        """Likelihood-ratio test filtering using Huber weights.

        Uses the Huber weights from the IRLS-Huber deconvolution to
        compute a weighted Poisson log-likelihood for both the full and
        reduced models, so that outlier EGs contribute less to the test
        statistic.

        For each ORF, a Wilks test compares the full log-likelihood with
        the log-likelihood obtained when that ORF's activity is clamped
        to ``config.pseudo_min``.  If the drop is not significant at
        level ``config.likelihood_ratio_alpha``, the ORF is removed.
        ORFs are tested in order of ascending total activity so that the
        weakest candidates are evaluated first.

        Parameters
        ----------
        config : Config
            Configuration providing convergence tolerances and
            significance threshold.
        runs : list[RiboSeqRun]
            Ribo-seq runs to include.
        """

        weights = self.irls_huber_weights
        theta = _distribution_theta(config)

        def run_weighted_likelihood_optimization(initial_guess, bounds, optim_args):
            X_lr, y_lr, ftol, gtol = optim_args
            if getattr(config, "inner_solver", "lbfgs") == "mu":
                # Weighted Poisson MLE via MU. A (pmin, pmin) box pins a coord to
                # ~0 (the reduced hypothesis); map those to a fixed_mask.
                import types

                from price2 import mu_solver

                pmin = config.pseudo_min
                fixed = np.array(
                    [b[1] is not None and b[1] <= pmin for b in bounds])
                w_lr = mu_solver.mu_inner_cpu(
                    X_lr, XT_lr, y_lr, weights,
                    np.asarray(initial_guess, dtype=np.float64), 0.0,
                    len(initial_guess), 1, pmin,
                    getattr(config, "mu_inner_max_iter", 3000),
                    getattr(config, "mu_inner_tol", 1e-5),
                    fixed_mask=fixed, theta=theta)
                optimization_result = types.SimpleNamespace(x=w_lr, success=True)
            else:
                cb = Callback(initial_guess, config)
                optimization_result = minimize(
                    weighted_poisson_nll_grad,
                    initial_guess,
                    args=(X_lr, y_lr, weights, theta),
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                    callback=cb,
                    options={
                        "maxiter": 10_000,
                        "ftol": ftol,
                        "gtol": gtol,
                        "maxls": config.maxls,
                    },
                )
                if cb.success:
                    optimization_result.success = True
                if not optimization_result.success:
                    raise RuntimeError(
                        f"Weighted LRT filtering failed to converge. "
                        f"{optimization_result.message}"
                    )
            ll = weighted_poisson_log_likelihood_sparse(
                optimization_result.x, X_lr, y_lr, weights, theta
            )
            return optimization_result, ll

        sparse_args = self.to_sparse_args(runs)
        X_lr = sparse_args["X"]
        # Transpose once for the MU LRT solver (referenced by the closure above);
        # only needed when inner_solver="mu".
        XT_lr = (X_lr.T.tocsr()
                 if getattr(config, "inner_solver", "lbfgs") == "mu" else None)
        y_lr = sparse_args["y"]
        num_rgrs = sparse_args["num_rgrs"]
        initial_guess = sparse_args["initial_guess"]
        num_runs = sparse_args["num_runs"]
        args = (X_lr, y_lr, config.ftol, config.gtol)

        # Recompute weights on the current sparse system
        c = config.irls_huber_c
        delta = np.asarray(X_lr @ initial_guess).ravel()
        weights = _huber_weights(y_lr, delta, c, theta)

        noise_rgr_indices = {rgr.index for rgr in self.rgr_set if rgr.type == "NOISE"}
        test_rgr_indices = {rgr.index for rgr in self.rgr_set if rgr.type == "ORF"}
        keep_rgr_indices = noise_rgr_indices | test_rgr_indices

        self.rgr_dict = {rgr.index: rgr for rgr in self.rgr_set}

        shape = initial_guess.reshape(num_rgrs, -1).shape

        t = np.empty((), dtype=object)
        t[()] = (config.pseudo_min, None)
        bounds = list(np.full(initial_guess.shape, t))

        optim_args = args

        optimization_result, full_log_likelihood = run_weighted_likelihood_optimization(
            initial_guess, bounds, optim_args
        )

        self.final_result = optimization_result

        initial_guess = optimization_result.x

        rgr_ind_list = list(test_rgr_indices)
        try:
            act_sum = self.result[np.array(rgr_ind_list)].sum(axis=1)
            rgr_ind_list = np.array(rgr_ind_list)[np.argsort(act_sum)]
        except IndexError:
            rgr_ind_list = []

        full_rgr_ind = keep_rgr_indices
        full_activities = initial_guess
        for rgr_ind in rgr_ind_list:
            reduced_rgr_ind = full_rgr_ind - {rgr_ind}

            reduced_activities = full_activities.copy().reshape(num_rgrs, -1)
            reduced_activities[rgr_ind] = config.pseudo_min
            reduced_activities = reduced_activities.flatten()

            reduced_log_likelihood = weighted_poisson_log_likelihood_sparse(
                reduced_activities, X_lr, y_lr, weights, theta
            )

            log_p = wilks_test_p(
                full_log_likelihood,
                reduced_log_likelihood,
                df_diff=num_runs,
            )
            if log_p > np.log(config.likelihood_ratio_alpha):
                full_rgr_ind.remove(rgr_ind)
                full_activities = reduced_activities
                full_log_likelihood = reduced_log_likelihood

            else:
                # optimize full model
                t = np.empty((), dtype=object)
                t[()] = (config.pseudo_min, config.pseudo_min)
                bounds = np.full(shape, t)
                t[()] = (config.pseudo_min, None)
                for index in full_rgr_ind:
                    bounds[index] = t
                bounds = list(bounds.flatten())

                optimization_result, full_log_likelihood = (
                    run_weighted_likelihood_optimization(
                        full_activities, bounds, optim_args
                    )
                )
                full_activities = optimization_result.x

                # optimize reduced model
                t = np.empty((), dtype=object)
                t[()] = (config.pseudo_min, config.pseudo_min)
                bounds = np.full(shape, t)
                t[()] = (config.pseudo_min, None)
                for index in reduced_rgr_ind:
                    bounds[index] = t
                bounds = list(bounds.flatten())
                optimization_result_reduced, reduced_log_likelihood = (
                    run_weighted_likelihood_optimization(
                        full_activities, bounds, optim_args
                    )
                )

                log_p = wilks_test_p(
                    full_log_likelihood,
                    reduced_log_likelihood,
                    df_diff=num_runs,
                )
                if log_p > np.log(config.likelihood_ratio_alpha):
                    full_rgr_ind.remove(rgr_ind)
                    full_activities = optimization_result_reduced.x
                    full_log_likelihood = reduced_log_likelihood

        t = np.empty((), dtype=object)
        t[()] = (config.pseudo_min, config.pseudo_min)
        bounds = np.full(shape, t)
        t[()] = (config.pseudo_min, None)
        for index in full_rgr_ind:
            bounds[index] = t
        bounds = list(bounds.flatten())

        optimization_result, full_log_likelihood = run_weighted_likelihood_optimization(
            full_activities, bounds, optim_args
        )
        full_activities = optimization_result.x

        self.final_result = optimization_result

        tmp = self.final_result.x.reshape(num_rgrs, num_runs)
        tmp[tmp <= config.pseudo_min] = 0
        self.result = tmp
        with np.errstate(invalid="ignore"):
            tmp = tmp / tmp.sum(axis=0)
            tmp[np.isnan(tmp)] = 0

        rgrs_to_remove = set()
        for rgr in self.rgr_set:
            if rgr.index not in keep_rgr_indices and rgr.type != "NOISE":
                rgrs_to_remove.add(rgr)

        self.remove_rgrs(rgrs_to_remove, runs=runs)

        self.runs = runs

    def estimate_activities(
        self,
        runs: list[RiboSeqRun],
        config: Config,
    ) -> None:
        """Estimate final ORF activities without regularisation.

        Iteratively optimises the unregularised Poisson log-likelihood.
        After each round, ORFs whose activity is below
        ``config.rgr_min_activity`` in every run are removed.  The loop
        continues until no more ORFs are removed.  Results are stored
        in :attr:`result` and :attr:`result_df`.

        Parameters
        ----------
        runs : list[RiboSeqRun]
            Ribo-seq runs to estimate activities for.
        config : Config
            Configuration providing convergence and threshold parameters.
        """
        theta = _distribution_theta(config)
        rgrs_removed = True
        while rgrs_removed:
            args_dict = self.to_sparse_args(runs)
            X_ea = args_dict["X"]
            y_ea = args_dict["y"]
            obj_fn = poisson_nll_grad
            obj_args = (X_ea, y_ea, theta)

            num_runs = args_dict["num_runs"]
            num_rgrs = args_dict["num_rgrs"]
            initial_guess = args_dict["initial_guess"]

            bounds = [(config.pseudo_min, None)] * len(initial_guess)

            if getattr(config, "inner_solver", "lbfgs") == "mu":
                # Unregularised Poisson MLE via Richardson-Lucy (weights=1, lam=0),
                # consistent with the MU group-LASSO deconvolution.
                from price2 import mu_solver

                XT_ea = X_ea.T.tocsr()
                w_ea = mu_solver.mu_inner_cpu(
                    X_ea, XT_ea, y_ea, np.ones(X_ea.shape[0]),
                    initial_guess, 0.0, len(initial_guess), 1, config.pseudo_min,
                    getattr(config, "mu_inner_max_iter", 3000),
                    getattr(config, "mu_inner_tol", 1e-5), theta=theta)
                tmp = w_ea.copy()
            else:
                cb = Callback(initial_guess, config)
                optimization_result = minimize(
                    obj_fn,
                    initial_guess,
                    args=obj_args,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                    callback=cb,
                    options={
                        "maxiter": 10_000,
                        "gtol": config.gtol,
                        "ftol": config.ftol,
                        "maxls": config.maxls,
                    },
                )
                if cb.success:
                    optimization_result.success = True
                if not optimization_result.success:
                    raise RuntimeError(f"Activity estimation failed to converge.")
                tmp = optimization_result.x.copy()
            tmp = tmp.reshape(num_rgrs, num_runs)
            tmp[tmp <= config.pseudo_min] = 0
            self.result = tmp

            rgr_indices_to_remove = set(
                np.where(np.all(self.result < config.rgr_min_activity, axis=1))[0]
            )
            rgrs_to_remove = set(
                [
                    rgr
                    for rgr in self.rgr_set
                    if rgr.index in rgr_indices_to_remove and rgr.type == "ORF"
                ]
            )
            if rgrs_to_remove:
                self.remove_rgrs(rgrs_to_remove, runs=runs)
            else:
                rgrs_removed = False

        run_ids = [run.id for run in runs]
        temp = {rgr.index: rgr for rgr in self.rgr_set}
        rgr_ids = [temp[i].id for i in range(len(temp))]
        self.result_df = pd.DataFrame(self.result, index=rgr_ids, columns=run_ids)


# ------------------------------------------------------------------ #
# Statistical tests                                                    #
# ------------------------------------------------------------------ #


def _chi2_logsf_asymptotic(x: float, k: int) -> float:
    """Asymptotic log upper-tail of chi-squared for large x.

    Uses the divergent asymptotic series for the upper regularized
    incomplete gamma function Q(s, z) with s = k/2, z = x/2:

        log Q(s, z) = -z + (s-1) log(z) - log Γ(s)
                     + log(1 + (s-1)/z + (s-1)(s-2)/z² + ...)

    The series is truncated before the terms start to grow.
    """
    s = 0.5 * k
    z = 0.5 * x
    leading = -z + (s - 1.0) * np.log(z) - gammaln(s)
    term = 1.0
    total = 1.0
    prev_abs = 1.0
    for n in range(1, 64):
        term *= (s - n) / z
        if abs(term) > prev_abs:
            break
        total += term
        prev_abs = abs(term)
        if abs(term) < 1e-16 * abs(total):
            break
    return leading + np.log(total)


def wilks_test_p(
    log_likelihood_full: float,
    log_likelihood_reduced: float,
    df_diff: int = 1,
) -> float:
    """Compute the log p-value for a Wilks likelihood-ratio test.

    Parameters
    ----------
    log_likelihood_full : float
        Log-likelihood of the full model.
    log_likelihood_reduced : float
        Log-likelihood of the reduced model.
    df_diff : int
        Difference in degrees of freedom.

    Returns
    -------
    float
        Log p-value (use ``np.exp(result)`` for the p-value).
    """
    λ = -2 * (log_likelihood_reduced - log_likelihood_full)
    if λ <= 0:
        return 0.0
    logp = chi2.logsf(λ, df_diff)
    if not np.isfinite(logp):
        logp = _chi2_logsf_asymptotic(λ, df_diff)
    return logp


# ------------------------------------------------------------------ #
# Sparse-matrix objective functions                                    #
# ------------------------------------------------------------------ #


def egs_to_sparse(
    locus_egs: dict,
    runs: list[RiboSeqRun],
    cm_lut: np.ndarray,
    coverage_params: np.ndarray,
    num_rgrs: int,
    num_runs: int,
) -> tuple[csr_matrix, np.ndarray]:
    """Convert locus equivalence groups to a sparse CSR design matrix.

    Builds the design matrix ``X`` and response vector ``y`` for the
    identity-link Poisson GLM directly from the locus's native EG
    dictionary, avoiding Numba typed-List construction entirely.

    Each row corresponds to one ``(EG, run)`` pair.  Column
    ``rgr_index * num_runs + run_index`` receives the value::

        length * cm_lut[run, read_length, frame, oua] * coverage_params[run, cov_pos]

    Parameters
    ----------
    locus_egs : dict
        ``Locus.egs`` — mapping from :class:`RiboSeqRun` to a dict of
        ``(rgr_frame_covpos, read_length, oua) -> EquivalenceGroup``.
    runs : list[RiboSeqRun]
        Ordered list of runs (determines run indices).
    cm_lut : np.ndarray, shape ``(num_runs, max_read_len, 4, 2)``
        Cleavage-model look-up table.
    coverage_params : np.ndarray, shape ``(num_runs, 3)``
        Coverage-model factors.
    num_rgrs : int
        Number of RGRs (= number of column groups).
    num_runs : int
        Number of Ribo-seq runs.

    Returns
    -------
    X : csr_matrix, shape ``(n_EGs_total, num_rgrs * num_runs)``
        Sparse design matrix.
    y : np.ndarray, shape ``(n_EGs_total,)``
        Observed read counts.
    """
    # Pass 1: count rows (non-empty EGs) and total non-zeros so we can
    # pre-size numpy arrays.  Building Python int/float lists with one
    # entry per CSR cell costs ~80 B/cell on CPython and dominates peak
    # RSS at this stage; numpy buffers are 12 B/cell instead.
    n_rows = 0
    nnz = 0
    for run in runs:
        for (rgr_frame_covpos, _, _), _ in locus_egs[run].items():
            sz = len(rgr_frame_covpos)
            if sz == 0:
                continue
            n_rows += 1
            nnz += sz

    rows_idx = np.empty(nnz, dtype=np.int64)
    cols_idx = np.empty(nnz, dtype=np.int64)
    data = np.empty(nnz, dtype=np.float64)
    y = np.empty(n_rows, dtype=np.float64)

    # Pass 2: populate.
    row = 0
    cell = 0
    for run_index, run in enumerate(runs):
        for (rgr_frame_covpos, read_length, oua), eg in locus_egs[run].items():
            if not rgr_frame_covpos:
                continue
            y[row] = eg.read_count
            length = eg.length
            oua_int = int(oua)
            for rgr, frame, cov_pos in rgr_frame_covpos:
                f = 3 if frame is None else frame
                rows_idx[cell] = row
                cols_idx[cell] = rgr.index * num_runs + run_index
                data[cell] = (
                    length
                    * cm_lut[run_index, read_length, f, oua_int]
                    * coverage_params[run_index, cov_pos.value]
                )
                cell += 1
            row += 1

    n_cols = num_rgrs * num_runs
    X = csr_matrix(
        (data, (rows_idx, cols_idx)), shape=(n_rows, n_cols), dtype=np.float64
    )
    return X, y


def _distribution_theta(config: Config) -> float | None:
    """Return the negative-binomial dispersion θ, or ``None`` for Poisson.

    Reads ``config.distribution``; when it is ``"nb"`` the fixed global
    ``config.nb_dispersion`` is returned, otherwise ``None`` (the classic
    Poisson model).  Every deconvolution solve site funnels the count-model
    choice through this single helper, so the flag has one source of truth.

    Parameters
    ----------
    config : Config
        Parsed PRICE configuration object.

    Returns
    -------
    float or None
        The dispersion θ for the negative-binomial model, or ``None`` when
        the Poisson model is selected.
    """
    if getattr(config, "distribution", "poisson") == "nb":
        return float(getattr(config, "nb_dispersion", 10.0))
    return None


def _huber_weights(
    y: np.ndarray,
    delta: np.ndarray,
    c: float,
    theta: float | None = None,
) -> np.ndarray:
    """Huber weights ``ω_i = min(1, c / |r_i|)`` on Pearson residuals.

    The standardised residual is ``r_i = (y_i − δ_i) / √v_i`` with the
    count-model variance ``v_i``: ``δ_i`` for the Poisson model
    (``theta is None``) and ``δ_i + δ_i² / θ`` for the negative binomial.
    Using the NB variance keeps the robustness threshold ``c`` on the same
    standardised scale as the NB likelihood, so overdispersed-but-inlying
    observations are not spuriously down-weighted.

    Parameters
    ----------
    y : np.ndarray, shape ``(n_samples,)``
        Observed read counts.
    delta : np.ndarray, shape ``(n_samples,)``
        Fitted means ``δ = X @ w``.
    c : float
        Huber tuning constant.
    theta : float or None, optional
        Negative-binomial dispersion.  ``None`` selects the Poisson variance.

    Returns
    -------
    np.ndarray, shape ``(n_samples,)``
        Per-observation Huber weights in ``(0, 1]``.
    """
    delta_safe = np.maximum(delta, 1e-14)
    if theta is None:
        var = delta_safe
    else:
        var = delta_safe + delta_safe**2 / theta
    pearson_r = (y - delta_safe) / np.sqrt(var)
    abs_r = np.abs(pearson_r)
    return np.where(abs_r <= c, 1.0, c / np.maximum(abs_r, 1e-14))


def poisson_nll_grad(
    w: np.ndarray,
    X: csr_matrix,
    y: np.ndarray,
    theta: float | None = None,
) -> tuple[float, np.ndarray]:
    """Identity-link Poisson (or negative-binomial) NLL and gradient.

    With ``theta is None`` (default) this is the classic identity-link
    Poisson model.  When ``theta`` is a positive float the identity-link
    **negative-binomial** NLL and gradient are returned instead, with the
    same mean but variance ``δ + δ²/θ``.  As ``θ → ∞`` the two coincide.

    Model:  δ = X @ w  (mean of the count model)
    Poisson loss:  Σ_i [δ_i − y_i · ln δ_i]        (zero-zero pairs excluded)
    Poisson score: r_i = 1 − y_i / δ_i
    NB loss:       Σ_i [(y_i + θ) · ln(θ + δ_i) − y_i · ln δ_i]
    NB score:      r_i = (y_i + θ) / (θ + δ_i) − y_i / δ_i
    Grad (both):   X.T @ r

    (Constants that do not depend on ``w`` are dropped from the loss; they
    do not affect the gradient or the minimiser.)

    Parameters
    ----------
    w : np.ndarray, shape ``(n_features,)``
        Current activity estimate (must satisfy ``w_j > 0`` via bounds).
    X : csr_matrix, shape ``(n_samples, n_features)``
        Non-negative sparse design matrix.
    y : np.ndarray, shape ``(n_samples,)``
        Observed read counts.
    theta : float or None, optional
        Negative-binomial dispersion.  ``None`` selects the Poisson model.

    Returns
    -------
    loss : float
    grad : np.ndarray, shape ``(n_features,)``
    """
    delta = np.asarray(X @ w).ravel()
    active = ~((delta == 0.0) & (y == 0.0))
    d_act = delta[active]
    y_act = y[active]
    if theta is None:
        loss = float(d_act.sum() - (y_act * np.log(d_act)).sum())
        r_act = 1.0 - y_act / d_act
    else:
        loss = float(
            ((y_act + theta) * np.log(theta + d_act) - y_act * np.log(d_act)).sum()
        )
        r_act = (y_act + theta) / (theta + d_act) - y_act / d_act
    r = np.zeros(len(y), dtype=np.float64)
    r[active] = r_act
    grad = np.asarray(X.T @ r).ravel()
    return loss, grad


def weighted_poisson_nll_grad(
    w: np.ndarray,
    X: csr_matrix,
    y: np.ndarray,
    weights: np.ndarray,
    theta: float | None = None,
) -> tuple[float, np.ndarray]:
    """Weighted identity-link Poisson (or negative-binomial) NLL and gradient.

    Poisson loss:  Σ_i ω_i · [δ_i − y_i · ln δ_i]
    Poisson grad:  X.T @ [ω_i · (1 − y_i / δ_i)]
    NB loss:       Σ_i ω_i · [(y_i + θ) · ln(θ + δ_i) − y_i · ln δ_i]
    NB grad:       X.T @ [ω_i · ((y_i + θ) / (θ + δ_i) − y_i / δ_i)]

    Parameters
    ----------
    w : np.ndarray, shape ``(n_features,)``
    X : csr_matrix, shape ``(n_samples, n_features)``
    y : np.ndarray, shape ``(n_samples,)``
    weights : np.ndarray, shape ``(n_samples,)``
        Per-observation Huber weights in [0, 1].
    theta : float or None, optional
        Negative-binomial dispersion.  ``None`` selects the Poisson model.

    Returns
    -------
    loss : float
    grad : np.ndarray, shape ``(n_features,)``
    """
    delta = np.asarray(X @ w).ravel()
    active = ~((delta == 0.0) & (y == 0.0))
    d_act = delta[active]
    y_act = y[active]
    w_act = weights[active]
    if theta is None:
        loss = float((w_act * (d_act - y_act * np.log(d_act))).sum())
        r_act = w_act * (1.0 - y_act / d_act)
    else:
        loss = float(
            (
                w_act
                * ((y_act + theta) * np.log(theta + d_act) - y_act * np.log(d_act))
            ).sum()
        )
        r_act = w_act * ((y_act + theta) / (theta + d_act) - y_act / d_act)
    r = np.zeros(len(y), dtype=np.float64)
    r[active] = r_act
    grad = np.asarray(X.T @ r).ravel()
    return loss, grad


def weighted_poisson_nll_grad_lasso(
    w: np.ndarray,
    X: csr_matrix,
    y: np.ndarray,
    weights: np.ndarray,
    lam: float,
    num_rgrs: int,
    num_runs: int,
    theta: float | None = None,
) -> tuple[float, np.ndarray]:
    """Weighted Poisson (or negative-binomial) NLL with group-LASSO penalty.

    Parameters
    ----------
    w : np.ndarray, shape ``(num_rgrs * num_runs,)``
    X : csr_matrix
    y : np.ndarray
    weights : np.ndarray, shape ``(n_samples,)``
    lam : float
    num_rgrs : int
    num_runs : int
    theta : float or None, optional
        Negative-binomial dispersion.  ``None`` selects the Poisson model.

    Returns
    -------
    loss : float
    grad : np.ndarray, shape ``(num_rgrs * num_runs,)``
    """
    loss, grad = weighted_poisson_nll_grad(w, X, y, weights, theta)
    W = w.reshape(num_rgrs, num_runs)
    norms = np.sqrt((W**2).sum(axis=1))
    safe_norms = np.maximum(norms, 1e-300)
    loss += lam * norms.sum()
    grad_penalty = lam * (W / safe_norms[:, None])
    grad = grad + grad_penalty.ravel()
    return loss, grad


def weighted_poisson_log_likelihood_sparse(
    w: np.ndarray,
    X: csr_matrix,
    y: np.ndarray,
    weights: np.ndarray,
    theta: float | None = None,
) -> float:
    """Weighted Poisson (or negative-binomial) log-likelihood for LRTs.

    The full log-likelihood (including the ``y``-dependent normalising
    constants) is returned so the Wilks statistic is comparable across
    models.  The constants cancel in the full-vs-reduced difference, but
    are kept so the absolute value is a genuine log-likelihood.

    Parameters
    ----------
    w : np.ndarray, shape ``(n_features,)``
    X : csr_matrix
    y : np.ndarray
    weights : np.ndarray, shape ``(n_samples,)``
    theta : float or None, optional
        Negative-binomial dispersion.  ``None`` selects the Poisson model.

    Returns
    -------
    float
        Weighted log-likelihood value.
    """
    delta = np.asarray(X @ w).ravel()
    active = ~((delta == 0.0) & (y == 0.0))
    d_act, y_act, w_act = delta[active], y[active], weights[active]
    if theta is None:
        return float(
            (w_act * (y_act * np.log(d_act) - d_act - gammaln(y_act + 1))).sum()
        )
    return float(
        (
            w_act
            * (
                gammaln(y_act + theta)
                - gammaln(theta)
                - gammaln(y_act + 1)
                + theta * np.log(theta)
                + y_act * np.log(d_act)
                - (y_act + theta) * np.log(theta + d_act)
            )
        ).sum()
    )


# ------------------------------------------------------------------ #
# ORF detection                                                        #
# ------------------------------------------------------------------ #


def find_orfs(
    seq: str,
    start_codons: list[str] | tuple[str, ...] = ("ATG",),
    stop_codons: list[str] | tuple[str, ...] = ("TAA", "TAG", "TGA"),
    min_length: int = 0,
) -> list[tuple[int, int]]:
    """Find all ORFs in a transcript sequence.

    Scans all three reading frames for start/stop codon pairs and
    returns 0-based, half-open intervals **including** the stop codon.

    Parameters
    ----------
    seq : str
        Spliced transcript nucleotide sequence.
    start_codons : list[str] or tuple[str, ...]
        Codons accepted as translation initiation sites.
    stop_codons : list[str] or tuple[str, ...]
        Codons accepted as translation termination sites.
    min_length : int
        Minimum ORF length (nt, excluding stop codon) to report.

    Returns
    -------
    list[tuple[int, int]]
        ``(start, end)`` intervals in transcript coordinates.  *end*
        includes the 3-nt stop codon.
    """
    start_codons_set = set(start_codons)
    stop_codons_set = set(stop_codons)
    orf_iv_on_transcript: list[tuple[int, int]] = []
    for i in range(3):
        starts: list[int] = []
        for j in range(i, len(seq), 3):
            codon = seq[j : j + 3]
            if codon in start_codons_set:
                starts.append(j)
            if codon in stop_codons_set:
                for start in starts:
                    if j - start >= min_length:
                        orf_iv_on_transcript.append((start, j + 3))
                starts = []
    return orf_iv_on_transcript


# ------------------------------------------------------------------ #
# Optimisation callback                                                #
# ------------------------------------------------------------------ #


class Callback:
    """Convergence callback for L-BFGS-B optimisation.

    Monitors the relative change in activity estimates between
    iterations and raises ``StopIteration`` when all active parameters
    have converged according to ``config.stop_factor_relative``.

    Attributes
    ----------
    success : bool
        ``True`` when convergence was reached.
    """

    success: bool

    def __init__(
        self,
        initial_guess: np.ndarray,
        config: Config,
    ) -> None:
        self.config = config
        self.previous = initial_guess
        self.success = False

    def __call__(self, new: np.ndarray) -> None:
        """Evaluate convergence after an L-BFGS-B iteration.

        Raises
        ------
        StopIteration
            When convergence is detected.
        """
        tmp = self.previous / new
        if not np.any(
            (
                (
                    ((1 - self.config.stop_factor_relative) > tmp)
                    | (tmp > (1 + self.config.stop_factor_relative))
                )
                & (new > self.config.rgr_min_activity)
            )
        ):
            self.success = True
            raise StopIteration
        else:
            self.previous = new
