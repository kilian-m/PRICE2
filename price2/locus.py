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

from price2.config import Config
from price2.coverage_model import CoveragePosition
from price2.equivalence_groups import EquivalenceGroup
from price2.genomic_features import ReadGeneratingRegion, Transcript
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

    def __getstate__(self) -> dict:
        """Return pickle state, excluding the breakpoint index cache."""
        state = self.__dict__.copy()
        state.pop("_transcript_breakpoint_index", None)
        return state

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

        self.rgr_intervals = HTSeq.GenomicArrayOfSets(
            "auto", stranded=True, storage="step"
        )
        for rgr in self.rgr_set:
            for iv in rgr.genomic_region.intervals:
                self.rgr_intervals[iv] += rgr

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

        for tr in overlap_transcripts:
            try:
                rsa_iv_on_tr = tr.exons.map_to_local(rsa.genomic_region)
            except ValueError:
                continue
            for rgr in tr.rgr_set:
                # full overlap with orf
                if rgr.type == "NOISE":
                    frame = None
                    if (rgr.iv_on_transcript[0] <= rsa_iv_on_tr[0]) and (
                        rgr.iv_on_transcript[1] >= rsa_iv_on_tr[1]
                    ):
                        if (
                            run.cleavage_model.pmf(
                                len(rsa.genomic_region), rsa.untemplated_addition, frame
                            )
                            > 0
                        ):
                            rgr_frame_covpos.add((rgr, frame, CoveragePosition.middle))
                    elif (
                        rsa_iv_on_tr[0] <= rgr.iv_on_transcript[0] <= rsa_iv_on_tr[1]
                    ) or (
                        rsa_iv_on_tr[0] <= rgr.iv_on_transcript[1] <= rsa_iv_on_tr[1]
                    ):
                        region_start = rgr.iv_on_transcript[0] - rsa_iv_on_tr[0]
                        region_end = rgr.iv_on_transcript[1] - rsa_iv_on_tr[0]
                        if (
                            ol := run.cleavage_model.pmf(
                                len(rsa),
                                rsa.untemplated_addition,
                                frame,
                                region_start=region_start,
                                region_end=region_end,
                            )
                        ) > 0:
                            cl = run.cleavage_model.pmf(
                                len(rsa), rsa.untemplated_addition, frame
                            )
                            if cl == 0:
                                continue
                            if ol / cl > overlap_likelihood_ratio_threshold:
                                rgr_frame_covpos.add(
                                    (
                                        rgr,
                                        frame,
                                        CoveragePosition.middle,
                                    )
                                )
                elif rgr.type == "ORF":
                    orf = rgr
                    if (
                        orf.iv_on_transcript[0] <= rsa_iv_on_tr[0]
                        and orf.iv_on_transcript[1] >= rsa_iv_on_tr[1]
                    ):
                        frame = (rsa_iv_on_tr[0] - orf.iv_on_transcript[0]) % 3
                        if (
                            run.cleavage_model.pmf(
                                len(rsa.genomic_region), rsa.untemplated_addition, frame
                            )
                            > 0
                        ):
                            rgr_frame_covpos.add((orf, frame, CoveragePosition.middle))
                    # part overlap with orf
                    elif (
                        rsa_iv_on_tr[0] <= orf.iv_on_transcript[0] <= rsa_iv_on_tr[1]
                    ) or (
                        rsa_iv_on_tr[0] <= orf.iv_on_transcript[1] <= rsa_iv_on_tr[1]
                    ):
                        frame = (rsa_iv_on_tr[0] - orf.iv_on_transcript[0]) % 3
                        # consider overlap likelihood
                        # compute at which position in the read the orf starts
                        region_start = orf.iv_on_transcript[0] + 3 - rsa_iv_on_tr[0]
                        # compute at which position in the read the orf ends
                        region_end = orf.iv_on_transcript[1] - 3 - rsa_iv_on_tr[0]
                        if (
                            ol := run.cleavage_model.pmf(
                                len(rsa),
                                rsa.untemplated_addition,
                                frame,
                                region_start=region_start,
                                region_end=region_end,
                            )
                        ) > 0:
                            cl = run.cleavage_model.pmf(
                                len(rsa), rsa.untemplated_addition, frame
                            )
                            if (not cl == 0) and (
                                ol / cl > overlap_likelihood_ratio_threshold
                            ):
                                rgr_frame_covpos.add(
                                    (
                                        orf,
                                        frame,
                                        CoveragePosition.middle,
                                    )
                                )
                        # consider coverage profile - start
                        # compute where the orf starts relative to the read
                        start_position = (
                            orf.iv_on_transcript[0] - rsa_iv_on_tr[0],
                            orf.iv_on_transcript[0] + 3 - rsa_iv_on_tr[0],
                        )
                        if (
                            ol := run.cleavage_model.pmf(
                                len(rsa),
                                rsa.untemplated_addition,
                                frame,
                                region_start=start_position[0],
                                region_end=start_position[1],
                            )
                        ) > 0:
                            cl = run.cleavage_model.pmf(
                                len(rsa), rsa.untemplated_addition, frame
                            )
                            if (not cl == 0) and (
                                # ol * run.coverage_model.start_factor / cl
                                ol / cl
                                > overlap_likelihood_ratio_threshold
                            ):
                                rgr_frame_covpos.add(
                                    (orf, frame, CoveragePosition.start)
                                )

                        # consider coverage profile - stop
                        # compute where the orf ends relative to the read
                        stop_position = (
                            orf.iv_on_transcript[1] - 3 - rsa_iv_on_tr[0],
                            orf.iv_on_transcript[1] - rsa_iv_on_tr[0],
                        )
                        if (
                            ol := run.cleavage_model.pmf(
                                len(rsa),
                                rsa.untemplated_addition,
                                frame,
                                region_start=stop_position[0],
                                region_end=stop_position[1],
                            )
                        ) > 0:
                            cl = run.cleavage_model.pmf(
                                len(rsa), rsa.untemplated_addition, frame
                            )
                            if (not cl == 0) and (
                                # ol * run.coverage_model.stop_factor / cl
                                ol / cl
                                > overlap_likelihood_ratio_threshold
                            ):
                                rgr_frame_covpos.add(
                                    (orf, frame, CoveragePosition.stop)
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
        for _, run_id, blob in reads_dfs:
            rsas_run = []
            reads_df = loads(zlib.decompress(blob))
            reads_df["chrom"] = self.iv.chrom
            reads_df["strand"] = self.iv.strand
            reads_df["read_id"] = reads_df["is_first_iv"].cumsum()
            for _, read_df in reads_df.groupby("read_id"):
                rsa = RiboSeqAlignment(read_df)
                rsas_run.append(rsa)
            self.rsas_dict[run_id] = rsas_run

            self.run_read_count[run_id] = reads_df[reads_df["is_first_iv"]][
                "count"
            ].sum()

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

                counted = False
                for rgr, frame, covpos in rgr_frame_covpos:
                    if rgr.type == "NOISE":
                        continue

                    well_fitting_rcs[run.id][rgr.id] += rsa.read_count
                    if not counted:
                        self.wfr_count += rsa.read_count
                        counted = True

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
                rc = (
                    rgr_read_counts[sorted_rgrs[j].id]
                    - rgr_read_counts[sorted_rgrs[j + 1].id]
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

            result = minimize(
                poisson_nll_grad,
                initial_guess,
                args=(X_filter, eg_read_counts),
                bounds=bounds,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": 10_000},
            )

            rgr_indices_to_keep_one_run = set(
                np.where(result.x >= config.deconvolution_filter_min_activity)[0]
            )
            rgr_indices_to_keep.append(rgr_indices_to_keep_one_run)

        try:
            all_kept = set.union(*rgr_indices_to_keep)
        except TypeError:
            all_kept = set()

        return {k for k, v in rgr_indices.items() if v not in all_kept}

    def assign_reads_to_egs(self, runs: list[RiboSeqRun]) -> None:
        """Assign reads to their equivalence groups.

        Each read is matched to its ``(rgr_frame_covpos, length, oua)``
        key and added to the corresponding :class:`EquivalenceGroup`.
        Reads whose key is absent (due to earlier filtering) are counted
        in :attr:`uncounted_reads`.

        Parameters
        ----------
        runs : list[RiboSeqRun]
            Ribo-seq runs to process.
        """
        for run in runs:
            run_id = run.id

            for rsa in self.rsas_dict[run_id]:
                rgr_frame_covpos = self.get_rgr_frame_covpos(rsa, run)
                read_count = rsa.read_count
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
                    self.read_counts[run] = int(read_count)

        self.counted_reads = {}
        for run in runs:
            self.counted_reads[run.id] = 0
            for v in self.egs[run].values():
                self.counted_reads[run.id] += v.read_count

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

        num_rgrs = len(self.rgr_set)
        rgr_lengths = np.array([len(rgr) for rgr in self.rgr_set])

        if hasattr(self, "result"):
            initial_guess = self.result
        else:
            initial_guess = np.ones((num_rgrs, num_runs))
        initial_guess = initial_guess.flatten()

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
    ) -> tuple[float, float]:
        """IRLS deconvolution with Huber weights on Pearson residuals.

        At each outer iteration:
          1. Compute fitted values δ = X @ w and Pearson residuals.
          2. Compute Huber weights: ω_i = min(1, c / |r_i|).
          3. Solve weighted group-LASSO Poisson NLL.

        Converges when the relative change in w falls below
        ``config.irls_huber_tol``.

        Parameters
        ----------
        config : Config
            Configuration providing ``irls_huber_c``,
            ``irls_huber_max_outer``, ``irls_huber_tol``, and
            standard optimisation parameters.
        runs : list[RiboSeqRun]
            Ribo-seq runs to deconvolve jointly.

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

        w_current = np.ones(n)

        s1 = time.time()
        for outer in range(config.irls_huber_max_outer):
            # Compute fitted values and Pearson residuals
            delta = np.asarray(X @ w_current).ravel()
            delta_safe = np.maximum(delta, 1e-14)
            pearson_r = (y - delta_safe) / np.sqrt(delta_safe)

            # Huber weights: ω_i = min(1, c / |r_i|)
            abs_r = np.abs(pearson_r)
            weights = np.where(abs_r <= c, 1.0, c / np.maximum(abs_r, 1e-14))

            # Solve weighted group-LASSO Poisson NLL
            cb = Callback(
                w_current,
                num_runs,
                config,
                remove_rgrs=False,
                rgr_lengths=rgr_lengths,
            )
            result = minimize(
                weighted_poisson_nll_grad_lasso,
                w_current,
                args=(X, y, weights, config.lam, num_rgrs, num_runs),
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

        s2 = time.time()
        opt_time += s2 - s1
        self.irls_outer_iterations = outer + 1
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
        delta_safe = np.maximum(delta, 1e-14)
        pearson_r = (y - delta_safe) / np.sqrt(delta_safe)
        abs_r = np.abs(pearson_r)
        self.irls_huber_weights = np.where(
            abs_r <= c, 1.0, c / np.maximum(abs_r, 1e-14)
        )

        # ── Post-optimisation RGR removal (same as deconvolve) ───────────
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
                if rgr.index in self.rgr_indices_to_remove and rgr.type == "ORF"
            ]
        )
        self.remove_rgrs(rgrs_to_remove, runs=runs)

        return opt_time, data_time

    def remove_rgrs(
        self,
        rgrs_to_remove: set[ReadGeneratingRegion],
        runs: list[RiboSeqRun] | None = None,
    ) -> None:
        """Remove a set of RGRs and update all dependent data structures.

        Updates :attr:`rgr_set`, re-indexes remaining RGRs, rebuilds
        :attr:`rgr_intervals`, collapses equivalence groups (if present),
        and re-slices :attr:`result` (if present).

        Parameters
        ----------
        rgrs_to_remove : set[ReadGeneratingRegion]
            RGRs to discard.
        runs : list[RiboSeqRun] or None
            Required when equivalence groups need collapsing.
        """
        old_rgr_set = self.rgr_set
        self.rgr_set = self.rgr_set - rgrs_to_remove

        # rgr indices
        for c, rgr in enumerate(self.rgr_set):
            rgr.index = c

        # rgr_intervals
        rgr_intervals = HTSeq.GenomicArrayOfSets("auto", stranded=True, storage="step")
        for step, step_set in self.rgr_intervals.steps():
            rgr_intervals[step] = step_set & self.rgr_set
        self.rgr_intervals = rgr_intervals

        # egs
        if hasattr(self, "egs"):
            self.collapse_egs(runs)

        # results
        if hasattr(self, "result"):
            index_array = np.zeros(len(self.rgr_set), dtype=int)
            for c, rgr in enumerate(old_rgr_set):
                if rgr in self.rgr_set:
                    index_array[rgr.index] = c
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

        def run_weighted_likelihood_optimization(initial_guess, bounds, optim_args):
            X_lr, y_lr, ftol, gtol = optim_args
            cb = Callback(initial_guess, num_runs, config)
            optimization_result = minimize(
                weighted_poisson_nll_grad,
                initial_guess,
                args=(X_lr, y_lr, weights),
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
                optimization_result.x, X_lr, y_lr, weights
            )
            return optimization_result, ll

        sparse_args = self.to_sparse_args(runs)
        X_lr = sparse_args["X"]
        y_lr = sparse_args["y"]
        num_rgrs = sparse_args["num_rgrs"]
        initial_guess = sparse_args["initial_guess"]
        num_runs = sparse_args["num_runs"]
        args = (X_lr, y_lr, config.ftol, config.gtol)

        # Recompute weights on the current sparse system
        delta = np.asarray(X_lr @ initial_guess).ravel()
        delta_safe = np.maximum(delta, 1e-14)
        pearson_r = (y_lr - delta_safe) / np.sqrt(delta_safe)
        abs_r = np.abs(pearson_r)
        c = config.irls_huber_c
        weights = np.where(abs_r <= c, 1.0, c / np.maximum(abs_r, 1e-14))

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
                reduced_activities, X_lr, y_lr, weights
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
        rgrs_removed = True
        while rgrs_removed:
            args_dict = self.to_sparse_args(runs)
            X_ea = args_dict["X"]
            y_ea = args_dict["y"]
            obj_fn = poisson_nll_grad
            obj_args = (X_ea, y_ea)

            num_runs = args_dict["num_runs"]
            num_rgrs = args_dict["num_rgrs"]
            initial_guess = args_dict["initial_guess"]

            bounds = [(config.pseudo_min, None)] * len(initial_guess)

            cb = Callback(initial_guess, num_runs, config)
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


def poisson_nll_grad(
    w: np.ndarray,
    X: csr_matrix,
    y: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Identity-link Poisson NLL and gradient using a sparse design matrix.

    Model:  δ = X @ w
    Loss:   Σ_i [δ_i − y_i · ln δ_i]  (zero-zero pairs excluded)
    Grad:   X.T @ r,  r_i = 1 − y_i / δ_i

    Parameters
    ----------
    w : np.ndarray, shape ``(n_features,)``
        Current activity estimate (must satisfy ``w_j > 0`` via bounds).
    X : csr_matrix, shape ``(n_samples, n_features)``
        Non-negative sparse design matrix.
    y : np.ndarray, shape ``(n_samples,)``
        Observed read counts.

    Returns
    -------
    loss : float
    grad : np.ndarray, shape ``(n_features,)``
    """
    delta = np.asarray(X @ w).ravel()
    active = ~((delta == 0.0) & (y == 0.0))
    d_act = delta[active]
    y_act = y[active]
    loss = float(d_act.sum() - (y_act * np.log(d_act)).sum())
    r = np.zeros(len(y), dtype=np.float64)
    r[active] = 1.0 - y_act / d_act
    grad = np.asarray(X.T @ r).ravel()
    return loss, grad


def weighted_poisson_nll_grad(
    w: np.ndarray,
    X: csr_matrix,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Weighted identity-link Poisson NLL and gradient.

    Loss:   Σ_i ω_i · [δ_i − y_i · ln δ_i]
    Grad:   X.T @ [ω_i · (1 − y_i / δ_i)]

    Parameters
    ----------
    w : np.ndarray, shape ``(n_features,)``
    X : csr_matrix, shape ``(n_samples, n_features)``
    y : np.ndarray, shape ``(n_samples,)``
    weights : np.ndarray, shape ``(n_samples,)``
        Per-observation Huber weights in [0, 1].

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
    loss = float((w_act * (d_act - y_act * np.log(d_act))).sum())
    r = np.zeros(len(y), dtype=np.float64)
    r[active] = w_act * (1.0 - y_act / d_act)
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
) -> tuple[float, np.ndarray]:
    """Weighted Poisson NLL with group-LASSO penalty.

    Parameters
    ----------
    w : np.ndarray, shape ``(num_rgrs * num_runs,)``
    X : csr_matrix
    y : np.ndarray
    weights : np.ndarray, shape ``(n_samples,)``
    lam : float
    num_rgrs : int
    num_runs : int

    Returns
    -------
    loss : float
    grad : np.ndarray, shape ``(num_rgrs * num_runs,)``
    """
    loss, grad = weighted_poisson_nll_grad(w, X, y, weights)
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
) -> float:
    """Weighted Poisson log-likelihood for likelihood-ratio tests.

    Parameters
    ----------
    w : np.ndarray, shape ``(n_features,)``
    X : csr_matrix
    y : np.ndarray
    weights : np.ndarray, shape ``(n_samples,)``

    Returns
    -------
    float
        Weighted log-likelihood value.
    """
    delta = np.asarray(X @ w).ravel()
    active = ~((delta == 0.0) & (y == 0.0))
    d_act, y_act, w_act = delta[active], y[active], weights[active]
    return float((w_act * (y_act * np.log(d_act) - d_act - gammaln(y_act + 1))).sum())


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
    Optionally identifies low-activity RGRs for early removal.

    Attributes
    ----------
    success : bool
        ``True`` when convergence was reached.
    rgr_indices_to_remove : set[int]
        Indices of RGRs flagged for removal (only populated when
        *remove_rgrs* is ``True``).
    """

    success: bool
    rgr_indices_to_remove: set[int]

    def __init__(
        self,
        initial_guess: np.ndarray,
        number_samples: int,
        config: Config,
        rgr_lengths: np.ndarray | None = None,
        remove_rgrs: bool = False,
    ) -> None:
        self.config = config
        self.previous = initial_guess
        self.initial_guess = initial_guess
        self.success = False
        self.number_samples = number_samples
        self.rgr_lengths = rgr_lengths
        self.rgr_indices_to_remove: set[int] = set()
        self.remove_rgrs = remove_rgrs

    def __call__(self, new: np.ndarray) -> None:
        """Evaluate convergence after an L-BFGS-B iteration.

        Raises
        ------
        StopIteration
            When convergence is detected or enough RGRs are flagged
            for removal.
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

        if self.remove_rgrs:
            x = new.reshape((-1, self.number_samples))

            if self.rgr_lengths is None:
                raise ValueError("rgr_lengths must be provided to remove rgrs.")
            x_t = x.T
            canonical_indices = (self.rgr_lengths * x_t).argmax(axis=1)
            min_activities = np.maximum(
                x_t[np.arange(x_t.shape[0]), canonical_indices]
                * self.config.min_activity_fraction,
                self.config.rgr_min_activity,
            )
            self.rgr_indices_to_remove = set(
                np.where(np.all(x < min_activities, axis=1))[0]
            )
            if x.shape[0] * self.config.rgrs_to_remove_fraction < len(
                self.rgr_indices_to_remove
            ):
                raise StopIteration
