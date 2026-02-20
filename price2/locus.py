"""Locus-level ORF deconvolution for PRICE2.

Defines the :class:`Locus` class that aggregates overlapping transcripts
into a single genomic unit, generates ORF candidates, runs group-LASSO
penalised maximum-likelihood estimation, and applies filtering steps
(coverage, deconvolution, likelihood-ratio) to identify actively
translated regions.

Standalone helper functions for ORF detection, Numba-accelerated
objective functions, and the optimisation :class:`Callback` are also
provided.
"""

from __future__ import annotations

import math
import sqlite3 as sql
import time
import warnings
import zlib
from collections import defaultdict
from pickle import loads

import HTSeq
import numpy as np
import pandas as pd
from filelock import FileLock
from numba import jit
from numba.core.errors import NumbaTypeSafetyWarning
from numba.typed import List
from scipy.optimize import minimize
from scipy.stats import chi2

from price2.config import Config
from price2.coverage_model import CoveragePosition
from price2.equivalence_groups import EquivalenceGroup
from price2.genomic_features import ReadGeneratingRegion, Transcript
from price2.ribo_seq_alignment import RiboSeqAlignment
from price2.ribo_seq_run import RiboSeqRun

warnings.simplefilter("ignore", category=NumbaTypeSafetyWarning)


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

    def make_rgrs(
        self,
        genome: dict[str, HTSeq.Sequence],
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
        genome : dict[str, HTSeq.Sequence]
            Chromosome-keyed genome sequences.
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
            if (
                hasattr(transcript, "cds")
                and (cds_start := transcript.exons.induce(transcript.cds)[0]) > 5
            ):
                cds_start = transcript.exons.induce(transcript.cds)[0]
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
                self.rgr_set.add(noise)
                transcript.rgr_set.add(noise)

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
            orf.transcript.add_orf(orf)
        self.rgr_set |= set(orf_dict.values())

        self.rgr_intervals = HTSeq.GenomicArrayOfSets(
            "auto", stranded=True, storage="step"
        )
        for rgr in self.rgr_set:
            for iv in rgr.genomic_region.intervals:
                self.rgr_intervals[iv] += rgr

        self.rgr_set_complete = self.rgr_set

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

        for query_iv in rsa.genomic_region.intervals:
            for subject_iv, tr_set in self.transcript_intervals[query_iv].steps():
                overlap_transcripts &= tr_set

        for tr in overlap_transcripts:
            try:
                rsa_iv_on_tr = tr.exons.induce(rsa.genomic_region)
            except ValueError:
                continue
            # rgr_frame_covpos.add((tr.noise, None, CoveragePosition.middle))
            # for orf in tr.orf_set:
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
        if write_loci:
            path = f"{prefix}_loci.gtf"
            lock = FileLock(path + ".lock")
            with lock:
                with open(path, "a") as f:
                    f.write(self.gtf_line())

        if write_transcripts:
            path = f"{prefix}_transcripts.gtf"
            lock = FileLock(path + ".lock")
            with lock:
                with open(path, "a") as f:
                    for rgr in self.rgr_set:
                        if rgr.type == "NOISE":
                            f.write(rgr.to_gtf(self.id))

        if write_orfs:
            path = f"{prefix}_orfs.gtf"
            lock = FileLock(path + ".lock")
            with lock:
                with open(path, "a") as f:
                    for rgr in self.rgr_set:
                        if rgr.type == "ORF":
                            f.write(rgr.to_gtf(self.id))

    def to_tsv(self, prefix: str) -> None:
        """Append ORF results to a TSV file.

        Parameters
        ----------
        prefix : str
            Path prefix; the file is named ``<prefix>_orfs.tsv``.
        """
        path = f"{prefix}_orfs.tsv"
        lock = FileLock(path + ".lock")
        with lock:
            with open(path, "a") as f:
                for rgr in self.rgr_set:
                    if rgr.type == "ORF":
                        f.write(rgr.to_tsv_line(self.id))

    def to_fasta(
        self,
        prefix: str,
        genome: dict[str, HTSeq.Sequence],
    ) -> None:
        """Append ORF sequences (with flanking context) to a FASTA file.

        Each ORF is extended by up to 14 nt upstream and 20 nt downstream
        on its parent transcript before extracting the spliced sequence.

        Parameters
        ----------
        prefix : str
            Directory prefix; the file is ``<prefix>/orfs.fasta``.
        genome : dict[str, HTSeq.Sequence]
            Chromosome-keyed genome sequences.
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
            gr = rgr.transcript.exons.map(iv_on_transcript)
            seqs = []
            if gr.strand == "+":
                for iv in gr.intervals:
                    seqs.append(genome[iv.chrom][iv.start : iv.end].seq.upper())
            else:
                for iv in gr.intervals:
                    seqs.append((-genome[iv.chrom][iv.start : iv.end]).seq.upper())
            seq = "".join(seqs)

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
        db = sql.connect(db_path)
        cur = db.cursor()
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
                stop = rgr.genomic_region.intervals[-1].start
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
            eg_read_counts = np.array([egs[eg][1] for eg in egs])
            eg_rgr_ids = List(
                [np.array([rgr_indices[rgr] for rgr in eg]) for eg in egs]
            )

            result = minimize(
                filter_objective_numba,
                initial_guess,
                args=(eg_lengths, eg_read_counts, eg_rgr_ids),
                bounds=bounds,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": 10_000, "maxfun": 1e6},
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
                    self.egs[run][
                        (rgr_frame_covpos, len(rsa), rsa.untemplated_addition)
                    ].reads.add(rsa)
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

    def to_deconvolution_args(
        self,
        runs: list[RiboSeqRun],
    ) -> dict:
        """Build argument dictionary for the Numba objective functions.

        Assembles cleavage-model look-up tables, coverage-model parameters,
        equivalence groups, and an initial-guess vector into a single
        dictionary consumed by :func:`objective_function` and related
        functions.

        Parameters
        ----------
        runs : list[RiboSeqRun]
            Ribo-seq runs to include.

        Returns
        -------
        dict
            Keys: ``cleavage_model``, ``coverage_model``, ``egs``,
            ``num_rgrs``, ``rgr_lengths``, ``num_runs``,
            ``initial_guess``.
        """
        rgrs = list(self.rgr_set)

        num_runs = len(runs)

        cm_lut = np.zeros((len(runs), runs[0].cleavage_model.cds_lut.shape[0], 4, 2))
        for i, run in enumerate(runs):
            cm_lut[i, :, 3, :] = run.cleavage_model.noise_lut
            cm_lut[i, :, :3, :] = run.cleavage_model.cds_lut

        coverage_params = np.zeros((len(runs), 3))
        for i, run in enumerate(runs):
            coverage_params[i, 0] = run.coverage_model.start_factor
            coverage_params[i, 1] = 1
            coverage_params[i, 2] = run.coverage_model.stop_factor

        num_rgrs = len(rgrs)
        rgr_lengths = np.array([len(rgr) for rgr in rgrs])

        if hasattr(self, "result"):
            initial_guess = self.result
        else:
            initial_guess = np.ones((num_rgrs, len(runs)))
        initial_guess = initial_guess.flatten()

        egs = List()
        for run in runs:
            l = List()
            for (rgr_frame_covpos, read_length, oua), eg in self.egs[run].items():
                if not rgr_frame_covpos:
                    continue
                temp = List()
                for rgr, frame, covpos in rgr_frame_covpos:
                    if frame is None:
                        frame = 3
                    temp.append((rgr.index, frame, covpos.value))
                l.append((eg.length, eg.read_count, read_length, int(oua), temp))
            if l:
                egs.append(l)

        return {
            "cleavage_model": cm_lut,
            "coverage_model": coverage_params,
            "egs": egs,
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
        """Run group-LASSO penalised Poisson-likelihood deconvolution.

        Iteratively optimises ORF activities.  Between iterations, ORFs
        whose activity falls below the canonical-ORF-relative threshold
        are removed to speed convergence.

        Parameters
        ----------
        config : Config
            Configuration providing regularisation and convergence
            parameters.
        runs : list[RiboSeqRun]
            Ribo-seq runs to deconvolve jointly.

        Returns
        -------
        tuple[float, float]
            ``(opt_time, data_time)`` — wall-clock seconds spent in
            optimisation vs. data preparation.
        """
        opt_time = 0.0
        data_time = 0.0

        while True:
            s1 = time.time()
            deconvolution_args = self.to_deconvolution_args(runs)
            cm_lut = deconvolution_args["cleavage_model"]
            coverage_params = deconvolution_args["coverage_model"]
            egs = deconvolution_args["egs"]
            num_rgrs = deconvolution_args["num_rgrs"]
            num_runs = deconvolution_args["num_runs"]
            rgr_lengths = deconvolution_args["rgr_lengths"]
            initial_guess = deconvolution_args["initial_guess"]
            s2 = time.time()
            data_time += s2 - s1
            s1 = time.time()

            bounds = [(config.pseudo_min, None)] * len(initial_guess)

            cb = Callback(
                initial_guess,
                num_runs,
                config,
                remove_rgrs=True,
                rgr_lengths=rgr_lengths,
            )

            optimization_result = minimize(
                objective_function,
                initial_guess,
                args=(
                    num_runs,
                    cm_lut,
                    coverage_params,
                    egs,
                    num_rgrs,
                    config.lam,
                ),
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
            s2 = time.time()
            opt_time += s2 - s1

            if cb.success or optimization_result.success:
                break
            elif cb.rgr_indices_to_remove:
                self.result = optimization_result.x.reshape(-1, num_runs)
                rgrs_to_remove = set(
                    [
                        rgr
                        for rgr in self.rgr_set
                        if rgr.index in cb.rgr_indices_to_remove and rgr.type == "ORF"
                    ]
                )
                self.remove_rgrs(rgrs_to_remove, runs=runs)
            else:
                raise RuntimeError(
                    f"Optimization stopped unexpectedly. {optimization_result.message}"
                )

        tmp = optimization_result.x.copy()
        tmp = tmp.reshape(-1, num_runs)

        result = optimization_result
        result.x = result.x.reshape(-1, num_runs)
        result.x[result.x <= config.pseudo_min] = 0

        self.result = result.x

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

        Parameters
        ----------
        runs : list[RiboSeqRun]
            Ribo-seq runs whose EGs should be rebuilt.
        """
        new_egs = {}
        for run in runs:
            new_egs[run] = defaultdict(EquivalenceGroup)
            for old_eg_key in self.egs[run]:
                (rgr_frame_covpos, read_length, oua) = old_eg_key
                new_rgr_frame_covpos = set()
                for rgr, frame, covpos in rgr_frame_covpos:
                    if rgr in self.rgr_set:
                        new_rgr_frame_covpos.add((rgr, frame, covpos))
                new_rgr_frame_covpos = frozenset(new_rgr_frame_covpos)
                new_eg_key = (new_rgr_frame_covpos, read_length, oua)

                new_eg = new_egs[run][new_eg_key]
                old_eg = self.egs[run][old_eg_key]

                new_eg.length += old_eg.length

                new_eg.read_count += old_eg.read_count
                new_eg.reads |= old_eg.reads

        self.egs = new_egs

    def likelihood_ratio_filtering(
        self,
        config: Config,
        runs: list[RiboSeqRun],
    ) -> None:
        """Apply likelihood-ratio test filtering to remove non-significant ORFs.

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

        def run_likelihood_optimization(initial_guess, bounds, optim_args):
            num_runs, cm_lut, cov_params, egs, ftol, gtol = optim_args

            cb = Callback(initial_guess, num_runs, config)
            optimization_result = minimize(
                objective_function_wo_regularization,
                initial_guess,
                args=(
                    num_runs,
                    cm_lut,
                    cov_params,
                    egs,
                ),
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
                    f"Likelihood ratio filtering failed to converge. "
                    f"{optimization_result.message}"
                )

            ll = log_likelihood(
                optimization_result.x,
                num_runs,
                cm_lut,
                cov_params,
                egs,
            )
            return optimization_result, ll

        deconvolution_args = self.to_deconvolution_args(runs)

        cm_lut = deconvolution_args["cleavage_model"]
        coverage_params = deconvolution_args["coverage_model"]
        egs = deconvolution_args["egs"]
        num_rgrs = deconvolution_args["num_rgrs"]
        initial_guess = deconvolution_args["initial_guess"]
        num_runs = deconvolution_args["num_runs"]

        args = (
            num_runs,
            cm_lut,
            coverage_params,
            egs,
        )

        noise_rgr_indices = {rgr.index for rgr in self.rgr_set if rgr.type == "NOISE"}
        test_rgr_indices = {rgr.index for rgr in self.rgr_set if rgr.type == "ORF"}
        keep_rgr_indices = noise_rgr_indices | test_rgr_indices

        self.rgr_dict = {rgr.index: rgr for rgr in self.rgr_set}

        shape = initial_guess.reshape(num_rgrs, -1).shape

        t = np.empty((), dtype=object)
        t[()] = (config.pseudo_min, None)
        bounds = list(np.full(initial_guess.shape, t))

        optim_args = (
            num_runs,
            cm_lut,
            coverage_params,
            egs,
            config.ftol,
            config.gtol,
        )

        optimization_result, full_log_likelihood = run_likelihood_optimization(
            initial_guess, bounds, optim_args
        )

        self.final_result = optimization_result

        initial_guess = optimization_result.x

        rgr_ind_list = list(test_rgr_indices)
        try:
            # sort by activity
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

            reduced_log_likelihood = log_likelihood(reduced_activities, *args)

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

                optimization_result, full_log_likelihood = run_likelihood_optimization(
                    full_activities, bounds, optim_args
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
                    run_likelihood_optimization(full_activities, bounds, optim_args)
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

        optimization_result, full_log_likelihood = run_likelihood_optimization(
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
            deconvolution_args = self.to_deconvolution_args(runs)

            num_runs = deconvolution_args["num_runs"]
            cm_lut = deconvolution_args["cleavage_model"]
            coverage_params = deconvolution_args["coverage_model"]
            egs = deconvolution_args["egs"]
            num_rgrs = deconvolution_args["num_rgrs"]
            initial_guess = deconvolution_args["initial_guess"]

            bounds = [(config.pseudo_min, None)] * len(initial_guess)

            cb = Callback(initial_guess, num_runs, config)
            optimization_result = minimize(
                objective_function_wo_regularization,
                initial_guess,
                args=(
                    num_runs,
                    cm_lut,
                    coverage_params,
                    egs,
                ),
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


def wilks_test(
    log_likelihood_full: float,
    log_likelihood_reduced: float,
    df_diff: int = 1,
    α: float = 1e-5,
) -> bool:
    """Perform a Wilks likelihood-ratio test.

    Returns ``True`` when the null hypothesis (that the reduced model
    is sufficient) should be rejected.

    Parameters
    ----------
    log_likelihood_full : float
        Log-likelihood of the full (unrestricted) model.
    log_likelihood_reduced : float
        Log-likelihood of the reduced (restricted) model.
    df_diff : int
        Difference in degrees of freedom between the two models.
    α : float
        Significance level.

    Returns
    -------
    bool
        ``True`` if the full model is significantly better.
    """
    return 2 * (log_likelihood_full - log_likelihood_reduced) > chi2.ppf(1 - α, df_diff)


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
    return chi2.logsf(λ, df_diff)


# ------------------------------------------------------------------ #
# Numba-accelerated objective functions                                #
# ------------------------------------------------------------------ #


@jit(nopython=True, cache=True, parallel=False)
def filter_objective_numba(
    x: np.ndarray,
    eg_lengths: np.ndarray,
    eg_read_counts: np.ndarray,
    eg_rgr_ids: List,
) -> tuple[float, np.ndarray]:
    """Poisson negative log-likelihood for the deconvolution filter.

    Used by :meth:`Locus.deconvolute_opt_group` to quickly evaluate
    stop-codon group deconvolution per run.

    Parameters
    ----------
    x : np.ndarray
        Activity vector, shape ``(n_egs,)``.
    eg_lengths : np.ndarray
        Equivalence-group lengths, shape ``(n_egs,)``.
    eg_read_counts : np.ndarray
        Observed read counts per EG, shape ``(n_egs,)``.
    eg_rgr_ids : numba.typed.List
        Each element is an ``np.ndarray`` of RGR indices belonging to
        that equivalence group.

    Returns
    -------
    tuple[float, np.ndarray]
        ``(loss, gradient)``.
    """

    loss = 0
    grads = np.zeros_like(x)

    for eg_len, eg_rc, eg_rgrs in zip(eg_lengths, eg_read_counts, eg_rgr_ids):
        activity = 0
        δ_derived = np.zeros_like(x)
        for rgr_id in eg_rgrs:
            activity += x[rgr_id]
            δ_derived[rgr_id] += eg_len

        δ = eg_len * activity
        y = eg_rc

        if y == 0 and δ == 0:
            continue

        loss += δ - y * np.log(δ)
        grads += -y * δ_derived / δ + δ_derived

    return loss, grads


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


# compute the negative log likelihood + a penalty term
# and the gradient with respect to each activity parameter
@jit(nopython=True, cache=True, parallel=False)
def objective_function(
    x: np.ndarray,
    num_runs: int,
    cm_lut: np.ndarray,
    coverage_params: np.ndarray,
    egs,
    num_rgrs: int,
    λ: float,
) -> tuple[float, np.ndarray]:
    """Negative Poisson log-likelihood with group-LASSO penalty.

    Parameters
    ----------
    x : np.ndarray
        Flattened activity vector, shape ``(num_rgrs * num_runs,)``.
    num_runs : int
        Number of Ribo-seq runs.
    cm_lut : np.ndarray
        Cleavage model look-up table, shape
        ``(num_runs, max_read_length, 4, 2)``.
    coverage_params : np.ndarray
        Coverage model parameters, shape ``(num_runs, 3)``.
    egs : numba.typed.List
        Per-run equivalence groups.
    num_rgrs : int
        Number of RGRs.
    λ : float
        Group-LASSO regularisation strength.

    Returns
    -------
    tuple[float, np.ndarray]
        ``(loss, gradient)``.
    """

    loss = 0
    grads = np.zeros_like(x)

    for run_index in range(num_runs):
        for eg in egs[run_index]:
            activity = 0
            δ_derived = np.zeros_like(x)
            for rgr_index, frame, cov_pos in eg[4]:
                activity += (
                    x[rgr_index * num_runs + run_index]
                    * cm_lut[run_index, eg[2], frame, eg[3]]
                    * coverage_params[run_index, cov_pos]
                )
                δ_derived[rgr_index * num_runs + run_index] += (
                    cm_lut[run_index, eg[2], frame, eg[3]]
                    * coverage_params[run_index, cov_pos]
                )

            δ = eg[0] * activity
            δ_derived *= eg[0]

            y = eg[1]

            if y == 0 and δ == 0:
                continue

            loss += δ - y * np.log(δ)
            grads += -y * δ_derived / δ + δ_derived

    penalty = 0
    for rgr_index in range(num_rgrs):
        s = 0
        for run_index in range(num_runs):
            s += x[rgr_index * num_runs + run_index] ** 2
        s_sqrt = s**0.5
        penalty += s_sqrt
        for run_index in range(num_runs):
            grads[rgr_index * num_runs + run_index] += (
                λ * x[rgr_index * num_runs + run_index] / s_sqrt
            )

    return loss + λ * penalty, grads


@jit(nopython=True, cache=True, parallel=False)
def objective_function_wo_regularization(
    x: np.ndarray,
    num_runs: int,
    cm_lut: np.ndarray,
    coverage_params: np.ndarray,
    egs,
) -> tuple[float, np.ndarray]:
    """Negative Poisson log-likelihood without regularisation.

    Same as :func:`objective_function` but omits the group-LASSO
    penalty term.  Used during likelihood-ratio filtering and final
    activity estimation.

    Parameters
    ----------
    x : np.ndarray
        Flattened activity vector.
    num_runs : int
        Number of Ribo-seq runs.
    cm_lut : np.ndarray
        Cleavage model look-up table.
    coverage_params : np.ndarray
        Coverage model parameters.
    egs : numba.typed.List
        Per-run equivalence groups.

    Returns
    -------
    tuple[float, np.ndarray]
        ``(loss, gradient)``.
    """

    loss = 0
    grads = np.zeros_like(x)

    for run_index in range(num_runs):
        for eg in egs[run_index]:
            activity = 0
            δ_derived = np.zeros_like(x)
            for rgr_index, frame, cov_pos in eg[4]:
                activity += (
                    x[rgr_index * num_runs + run_index]
                    * cm_lut[run_index, eg[2], frame, eg[3]]
                    * coverage_params[run_index, cov_pos]
                )
                δ_derived[rgr_index * num_runs + run_index] += (
                    cm_lut[run_index, eg[2], frame, eg[3]]
                    * coverage_params[run_index, cov_pos]
                )

            δ = eg[0] * activity
            δ_derived *= eg[0]

            y = eg[1]

            if y == 0 and δ == 0:
                continue

            loss += δ - y * np.log(δ)
            grads += -y * δ_derived / δ + δ_derived

    return loss, grads


@jit(nopython=True, cache=True, parallel=False)
def log_likelihood(
    x: np.ndarray,
    num_runs: int,
    cm_lut: np.ndarray,
    coverage_params: np.ndarray,
    egs,
) -> float:
    """Compute the Poisson log-likelihood (without penalty).

    Unlike :func:`objective_function_wo_regularization`, this returns
    the *positive* log-likelihood and does not compute the gradient.
    Used for likelihood-ratio tests.

    Parameters
    ----------
    x : np.ndarray
        Flattened activity vector.
    num_runs : int
        Number of Ribo-seq runs.
    cm_lut : np.ndarray
        Cleavage model look-up table.
    coverage_params : np.ndarray
        Coverage model parameters.
    egs : numba.typed.List
        Per-run equivalence groups.

    Returns
    -------
    float
        Log-likelihood value.
    """

    ll = 0

    for run_index in range(num_runs):
        for eg in egs[run_index]:
            activity = 0

            for rgr_index, frame, covpos in eg[4]:
                activity += (
                    x[rgr_index * num_runs + run_index]
                    * cm_lut[run_index, eg[2], frame, eg[3]]
                    * coverage_params[run_index, covpos]
                )

            δ = eg[0] * activity

            y = eg[1]
            if y == 0 and δ == 0:
                continue

            ll += y * np.log(δ) - δ - math.lgamma(y + 1)

    return ll
