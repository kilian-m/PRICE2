from pickle import loads
import sqlite3 as sql
import zlib
import HTSeq
import pandas as pd
import numpy as np
import math
from scipy.stats import chi2
import numba as nb

from filelock import FileLock

from scipy.optimize import minimize
from numba import jit
from numba.typed import List

from collections import defaultdict


from numba.core.errors import NumbaTypeSafetyWarning
import warnings

warnings.simplefilter("ignore", category=NumbaTypeSafetyWarning)

from price2.genomic_features import ReadGeneratingRegion, Transcript
from price2.ribo_seq_alignment import RiboSeqAlignment
from price2.ribo_seq_run import RiboSeqRun
from price2.coverage_model import CoveragePosition
from price2.equivalence_groups import EquivalenceGroup


class Locus:
    # permanent properties
    iv: HTSeq.GenomicInterval
    read_count: int
    read_counts: dict[RiboSeqRun, int]
    rgr_set = set[ReadGeneratingRegion]
    transcript_intervals: HTSeq.GenomicArrayOfSets
    transcripts: set[Transcript]

    egs: dict[EquivalenceGroup, EquivalenceGroup]

    def __init__(
        self,
        iv: HTSeq.GenomicInterval,
        transcript_intervals: HTSeq.GenomicArrayOfSets,
        loci_number: int,
    ) -> None:
        self.iv = iv
        self.read_counts = dict()

        self.uncounted_reads = 0
        self.times = dict()
        self.id = f"loc_{loci_number}"

        self.transcript_intervals = HTSeq.GenomicArrayOfSets(
            "auto", stranded=True, storage="step"
        )

        for iv, val in transcript_intervals[self.iv].steps():
            self.transcript_intervals[iv] = val

        self.transcripts = set()

        self.exon_length = 0
        for iv, value in self.transcript_intervals.steps():
            self.transcripts |= value
            if value:
                self.exon_length += iv.length

        self.lls = []  # for plotting

    def __repr__(self) -> str:
        return f"Locus({self.iv})"

    def make_rgrs(
        self,
        genome: dict[str : HTSeq._HTSeq.Sequence],
        config,
        min_length_to_end: int = 30,
    ) -> None:
        self.rgr_set = set()
        orf_dict = dict()
        noise_dict = dict()

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
                    if not noise in noise_dict:
                        noise_dict[noise] = noise
                    else:
                        alt_noise = noise_dict[noise]
                        tmp1 = (
                            alt_noise.dist_to_transcript_end
                            + alt_noise.dist_to_transcript_start
                        )

                        tmp2 = (
                            noise.dist_to_transcript_end
                            + noise.dist_to_transcript_start
                        )
                        if tmp2 > tmp1:
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
                if not orf in orf_dict:
                    orf_dict[orf] = orf
                else:
                    alt_orf = orf_dict[orf]
                    tmp1 = (
                        alt_orf.dist_to_transcript_end
                        + alt_orf.dist_to_transcript_start
                    )

                    tmp2 = orf.dist_to_transcript_end + orf.dist_to_transcript_start
                    if tmp2 > tmp1:
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
        l = list(self.rgr_set)
        for i, rgr in enumerate(l):
            for iv in rgr.genomic_region.intervals:
                self.rgr_intervals[iv] += rgr

        self.rgr_set_complete = self.rgr_set

    def make_equivalence_groups_precise(self, runs: list[RiboSeqRun]) -> None:
        # keep because this is definitely correct
        warnings.warn(
            "make_equivalence_groups_precise is deprecated", DeprecationWarning
        )
        self.egs = dict()

        for run in runs:
            self.egs[run] = dict()
            lengths = dict()
            steps = list(self.transcript_intervals.steps())
            if self.iv.strand == "-":
                steps = steps[::-1]
            for step_iv, transcripts in steps:
                if not transcripts:
                    continue
                for i in range(step_iv.length):
                    reads = set()
                    for transcript in transcripts:
                        if not transcript in lengths:
                            lengths[transcript] = 0
                        for read_length in run.cleavage_model.non_zero_lengths:
                            for oua in [True, False]:
                                iv_on_transcript = (
                                    lengths[transcript],
                                    lengths[transcript] + read_length,
                                )
                                if iv_on_transcript[0] < 0 or iv_on_transcript[1] > len(
                                    transcript
                                ):
                                    continue
                                read_gr = transcript.exons.map(iv_on_transcript)
                                rsa = RiboSeqAlignment(
                                    {
                                        "mapping_positions": 1,
                                        "genomic_region": read_gr,
                                        "untemplated_addition": oua,
                                    }
                                )
                                reads.add(rsa)
                        lengths[transcript] += 1

                    for rsa in reads:
                        rgr_frame_covpos = self.get_rgr_frame_covpos(rsa, run)
                        if not rgr_frame_covpos:
                            continue
                        if (
                            not (rgr_frame_covpos, len(rsa), rsa.untemplated_addition)
                            in self.egs[run]
                        ):
                            self.egs[run][
                                (rgr_frame_covpos, len(rsa), rsa.untemplated_addition)
                            ] = EquivalenceGroup()
                        self.egs[run][
                            (rgr_frame_covpos, len(rsa), rsa.untemplated_addition)
                        ].length += 1

    def get_rgr_frame_covpos(
        self,
        rsa: RiboSeqAlignment,
        run: RiboSeqRun,
        overlap_likelihood_ratio_threshold: float = 0.2,
    ) -> frozenset[tuple[ReadGeneratingRegion, int | None]]:

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

    def get_rgr_frames_alt(
        self,
        rsa: RiboSeqAlignment,
        run: RiboSeqRun,
        overlap_likelihood_ratio_threshold: float = 0.5,
    ) -> frozenset[tuple[ReadGeneratingRegion, int | None]]:
        overlap_transcripts = set(self.transcripts)

        rgr_frame = set()

        for query_iv in rsa.genomic_region.intervals:
            for subject_iv, tr_set in self.transcript_intervals[query_iv].steps():
                overlap_transcripts &= tr_set

        for tr in overlap_transcripts:
            try:
                rsa_iv_on_tr = tr.exons.induce(rsa.genomic_region)
            except ValueError:
                continue
            rgr_frame.add((tr.noise, None))

            rsa_iv = HTSeq.GenomicInterval(
                ".",
                rsa_iv_on_tr[0],
                rsa_iv_on_tr[1],
                ".",
            )
            for phase in range(3):

                step_sets = [
                    step_set for _, step_set in tr.orf_intervals[phase][rsa_iv].steps()
                ]

                full_overlap = set.intersection(*step_sets)
                if full_overlap:
                    orf = next(iter(full_overlap))
                    frame = (rsa_iv_on_tr[0] - orf.iv_on_transcript[0]) % 3
                    if (
                        run.cleavage_model.pmf(
                            len(rsa.genomic_region), rsa.untemplated_addition, frame
                        )
                        > 0
                    ):
                        for orf in full_overlap:
                            rgr_frame.add((orf, frame))
                processed_orfs = full_overlap

                for step_set in step_sets:
                    orfs = step_set - processed_orfs
                    if not orfs:
                        continue
                    orf = next(iter(orfs))
                    frame = (rsa_iv_on_tr[0] - orf.iv_on_transcript[0]) % 3
                    region_start = orf.iv_on_transcript[0] - rsa_iv_on_tr[0]
                    region_end = orf.iv_on_transcript[1] - rsa_iv_on_tr[0]

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
                        if ol / cl > overlap_likelihood_ratio_threshold:
                            for orf in orfs:
                                rgr_frame.add((orf, frame))
                    processed_orfs |= orfs

        if not rgr_frame:
            return

        rgr_frame = frozenset(rgr_frame)
        return rgr_frame

    def gtf_line(self):
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
        self, prefix, write_loci=False, write_transcripts=False, write_orfs=True
    ):
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

    def to_tsv(self, prefix):
        path = f"{prefix}_orfs.tsv"
        lock = FileLock(path + ".lock")
        with lock:
            with open(path, "a") as f:
                for rgr in self.rgr_set:
                    if rgr.type == "ORF":
                        f.write(rgr.to_tsv_line(self.id))

    def to_fasta(self, prefix, genome):

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

    #############################################
    ### functions for orf activity estimation ###
    #############################################

    # get reads from database
    def get_reads_from_db(self, db_path) -> dict[str : list[RiboSeqAlignment]]:
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

    def make_well_fitting_reads(self, runs):
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

    # filter RGRs based on coverage
    # compute coverage with well fitting reads for each RGR and each run
    # remove RGRs max coverage below threshold
    def coverage_filter_rgrs(self, config):

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

    def deconvolution_filter_rgrs(self, config):

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

    # assign rgrs to groups corresponding to one stop codon
    # do not consider NOISE rgrs and groups with a single rgr
    def make_stop_groups(self):
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

    # a stop group (a set of RGRs ending in one stop) can be incompatible if the RGRs contain different intron-exon boundaries
    # this function splits stop groups into optimization groups (sets of compatible RGRs)
    def split_stop_groups(self, stop_groups):
        optimization_groups = []
        for stop_group in stop_groups.values():
            l = list(stop_group)
            containment_dict = {}
            for rgr in l:
                containment_dict[rgr] = set()
                for other_rgr in l:
                    if rgr.genomic_region.contains_to_stop(other_rgr.genomic_region):
                        containment_dict[rgr].add(other_rgr)

            l = list(containment_dict.values())
            while l:
                # get biggest set from list
                big_set = max(l, key=len)
                optimization_groups.append(big_set)
                new_l = []
                for s in l:
                    if not s.issubset(big_set):
                        new_l.append(s)
                l = new_l

        return optimization_groups

    def deconvolute_opt_group(
        self,
        opt_group,
        config,
    ):
        # rgr_indices_to_remove = []
        rgr_indices_to_keep = []
        rgr_read_counts = dict(self.wfr_df.sum(axis=1))
        l = list(opt_group)
        l.sort(key=len, reverse=True)
        rgr_indices = {rgr.id: i for i, rgr in enumerate(l)}

        min_reads = self.wfr_df.sum().sum() / self.wfr_df.shape[1] * 0.1

        number_of_runs = self.wfr_df.shape[1]

        # iterate over runs
        for i in range(number_of_runs):  # len(self.wfr_df.iloc[0])):

            rgr_read_counts = self.wfr_df.iloc[:, i].to_dict()

            # skip if the locus is probably not expressed in this run
            if sum(rgr_read_counts.values()) < min_reads:
                continue
            egs = {}
            s = set()
            for j in range(len(l) - 1):
                s.add(l[j].id)
                length = len(l[j]) - len(l[j + 1])
                rc = rgr_read_counts[l[j].id] - rgr_read_counts[l[j + 1].id]
                egs[frozenset(s)] = (length, rc)

            s.add(l[-1].id)
            length = len(l[-1])
            rc = rgr_read_counts[l[-1].id]
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
            rgr_indices_to_keep = set.union(*rgr_indices_to_keep)
        except TypeError:
            rgr_indices_to_remove = set(rgr_indices.keys())

        return set([k for k, v in rgr_indices.items() if v not in rgr_indices_to_keep])

    def assign_reads_to_egs(self, runs):
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

    def to_deconvolution_args(self, runs):
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
                    if frame == None:
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
        config,
        runs,
    ):
        import time

        opt_time = 0
        data_time = 0

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

    def deconvolve_unregularized(self, config, runs):
        deconvolution_args = self.to_deconvolution_args(runs)
        run_read_counts = deconvolution_args["run_read_counts"]
        cm_lut = deconvolution_args["cleavage_model"]
        coverage_params = deconvolution_args["coverage_model"]
        egs = deconvolution_args["egs"]
        num_rgrs = deconvolution_args["num_rgrs"]
        initial_guess = deconvolution_args["initial_guess"]
        num_runs = deconvolution_args["num_runs"]

        bounds = [(config.pseudo_min, None)] * len(initial_guess)

        cb = Callback(initial_guess, num_runs, config)

        result = minimize(
            objective_function_wo_regularization,
            initial_guess,
            args=(
                run_read_counts,
                cm_lut,
                coverage_params,
                egs,
            ),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            callback=cb,
            options={"maxiter": 10_000, "ftol": config.ftol, "gtol": config.gtol},
        )

        result.x = result.x.reshape(num_rgrs, len(self.counted_reads))
        result.x[result.x <= config.pseudo_min] = 0

        self.result = result.x

        keep_rgr_indices = set(np.where(self.result > 0)[0])
        keep_rgr_indices |= set(
            [rgr.index for rgr in self.rgr_set if rgr.type == "NOISE"]
        )
        rgrs_to_remove = set(
            [rgr for rgr in self.rgr_set if rgr.index not in keep_rgr_indices]
        )
        self.remove_rgrs(rgrs_to_remove, runs=runs)

    def remove_rgrs(self, rgrs_to_remove, runs=None):
        # rgr_set
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
        runs,
    ):
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

    def likelihood_ratio_filtering_depracated(
        self,
        config,
        runs,
    ):
        deconvolution_args = self.to_deconvolution_args(runs)

        cm_lut = deconvolution_args["cleavage_model"]
        coverage_params = deconvolution_args["coverage_model"]
        egs = deconvolution_args["egs"]
        num_rgrs = deconvolution_args["num_rgrs"]
        initial_guess = deconvolution_args["initial_guess"]
        num_runs = deconvolution_args["num_runs"]

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
                print(optimization_result.message)
                raise RuntimeError(
                    f"Likelihood ratio filtering failed to converge. {optimization_result.message}"
                )

            ll = log_likelihood(
                optimization_result.x,
                num_runs,
                cm_lut,
                cov_params,
                egs,
            )
            return optimization_result, ll

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

        optimization_result, log_likelihood_all = run_likelihood_optimization(
            initial_guess, bounds, optim_args
        )

        opt_up_to_date = True

        self.final_result = optimization_result

        initial_guess = optimization_result.x

        rgr_ind_list = list(test_rgr_indices)
        try:
            # sort by activity
            act_sum = self.result[np.array(rgr_ind_list)].sum(axis=1)
            rgr_ind_list = np.array(rgr_ind_list)[np.argsort(act_sum)]
        except IndexError:
            rgr_ind_list = []

        for rgr_ind in rgr_ind_list:
            free_rgr_ind = keep_rgr_indices - {rgr_ind}
            initial_guess_int = initial_guess.copy().reshape(num_rgrs, -1)

            initial_guess_int[rgr_ind] = config.pseudo_min
            initial_guess_int = initial_guess_int.flatten()

            # test if removing the does not significantly decrease the likelihood even without optimization
            log_likelihood_int = log_likelihood(
                initial_guess_int,
                num_runs,
                cm_lut,
                coverage_params,
                egs,
            )

            log_p = wilks_test_p(
                log_likelihood_all,
                log_likelihood_int,
                df_diff=num_runs,
            )

            self.rgr_dict[rgr_ind].log_p_value = log_p
            if log_p > np.log(config.likelihood_ratio_alpha):
                # remove rgr
                keep_rgr_indices.remove(rgr_ind)
                initial_guess = initial_guess_int
                log_likelihood_all = log_likelihood_int
                opt_up_to_date = False
                continue

            else:
                if not opt_up_to_date:
                    t = np.empty((), dtype=object)
                    t[()] = (config.pseudo_min, config.pseudo_min)
                    bounds = np.full(shape, t)
                    t[()] = (config.pseudo_min, None)
                    for index in keep_rgr_indices:
                        bounds[index] = t
                    bounds = list(bounds.flatten())
                    optimization_result, log_likelihood_all = (
                        run_likelihood_optimization(initial_guess, bounds, optim_args)
                    )
                    opt_up_to_date = True
                    self.final_result = optimization_result
                    initial_guess = optimization_result.x

                t = np.empty((), dtype=object)
                t[()] = (config.pseudo_min, config.pseudo_min)
                bounds = np.full(shape, t)
                t[()] = (config.pseudo_min, None)
                for index in free_rgr_ind:
                    bounds[index] = t
                bounds = list(bounds.flatten())

                optimization_result_int, log_likelihood_int = (
                    run_likelihood_optimization(initial_guess_int, bounds, optim_args)
                )

                log_p = wilks_test_p(
                    log_likelihood_all,
                    log_likelihood_int,
                    df_diff=num_runs,
                )

                self.rgr_dict[rgr_ind].log_p_value = log_p
                if log_p > np.log(config.likelihood_ratio_alpha):
                    keep_rgr_indices.remove(rgr_ind)
                    initial_guess = optimization_result_int.x
                    log_likelihood_all = log_likelihood_int
                    self.final_result = optimization_result_int

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

    def likelihood_ratio_filtering(
        self,
        config,
        runs,
    ):

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
            # print(optimization_result.message, flush=True)
            # print(optimization_result.nit, flush=True)
            if cb.success:
                optimization_result.success = True
            if not optimization_result.success:
                print(optimization_result.message)
                raise RuntimeError(
                    f"Likelihood ratio filtering failed to converge. {optimization_result.message}"
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
                # print(
                #     f"Removing rgr {self.rgr_dict[rgr_ind].id} without re-optimization",
                #     flush=True,
                # )
                # remove rgr
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
                    # print(
                    #     f"Removing rgr {self.rgr_dict[rgr_ind].id} after re-optimization",
                    #     flush=True,
                    # )
                    full_rgr_ind.remove(rgr_ind)
                    full_activities = optimization_result_reduced.x
                    full_log_likelihood = reduced_log_likelihood
                # else:
                #     print(f"Keeping rgr {self.rgr_dict[rgr_ind].id}", flush=True)

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

    #         free_rgr_ind = keep_rgr_indices - {rgr_ind}
    #         initial_guess_int = initial_guess.copy().reshape(num_rgrs, -1)
    #
    #         initial_guess_int[rgr_ind] = config.pseudo_min
    #         initial_guess_int = initial_guess_int.flatten()
    #
    #         # test if removing the rgr does not significantly decrease the likelihood even without optimization
    #         log_likelihood_int = log_likelihood(
    #             initial_guess_int,
    #             num_runs,
    #             cm_lut,
    #             coverage_params,
    #             egs,
    #         )
    #
    #         log_p = wilks_test_p(
    #             log_likelihood_all,
    #             log_likelihood_int,
    #             df_diff=num_runs,
    #         )
    #
    #         self.rgr_dict[rgr_ind].log_p_value = log_p
    #         if log_p > np.log(config.likelihood_ratio_alpha):
    #             # remove rgr
    #             keep_rgr_indices.remove(rgr_ind)
    #             initial_guess = initial_guess_int
    #             log_likelihood_all = log_likelihood_int
    #
    #         else:
    #             # optimize current model
    #             t = np.empty((), dtype=object)
    #             t[()] = (config.pseudo_min, config.pseudo_min)
    #             bounds = np.full(shape, t)
    #             t[()] = (config.pseudo_min, None)
    #             for index in keep_rgr_indices:
    #                 bounds[index] = t
    #             bounds = list(bounds.flatten())
    #             optimization_result, log_likelihood_all = run_likelihood_optimization(
    #                 initial_guess, bounds, optim_args
    #             )
    #             self.final_result = optimization_result
    #             initial_guess = optimization_result.x
    #
    #             # optimize reduced model
    #             t = np.empty((), dtype=object)
    #             t[()] = (config.pseudo_min, config.pseudo_min)
    #             bounds = np.full(shape, t)
    #             t[()] = (config.pseudo_min, None)
    #             for index in free_rgr_ind:
    #                 bounds[index] = t
    #             bounds = list(bounds.flatten())
    #
    #             optimization_result_int, log_likelihood_int = (
    #                 run_likelihood_optimization(initial_guess_int, bounds, optim_args)
    #             )
    #
    #             log_p = wilks_test_p(
    #                 log_likelihood_all,
    #                 log_likelihood_int,
    #                 df_diff=num_runs,
    #             )
    #
    #             self.rgr_dict[rgr_ind].log_p_value = log_p
    #             if log_p > np.log(config.likelihood_ratio_alpha):
    #                 keep_rgr_indices.remove(rgr_ind)
    #                 initial_guess = optimization_result_int.x
    #                 log_likelihood_all = log_likelihood_int
    #                 self.final_result = optimization_result_int
    #
    #     tmp = self.final_result.x.reshape(num_rgrs, num_runs)
    #     tmp[tmp <= config.pseudo_min] = 0
    #     self.result = tmp
    #     with np.errstate(invalid="ignore"):
    #         tmp = tmp / tmp.sum(axis=0)
    #         tmp[np.isnan(tmp)] = 0
    #
    #     rgrs_to_remove = set()
    #     for rgr in self.rgr_set:
    #         if rgr.index not in keep_rgr_indices and rgr.type != "NOISE":
    #             rgrs_to_remove.add(rgr)
    #
    #     self.remove_rgrs(rgrs_to_remove, runs=runs)
    #
    #     self.runs = runs

    def estimate_activities(self, runs, config):
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


# reject null if true
def wilks_test(log_likelihood_full, log_likelihood_reduced, df_diff=1, α=1e-5) -> bool:
    # reject null hypothesis if the test statistic is greater than the critical value
    return 2 * (log_likelihood_full - log_likelihood_reduced) > chi2.ppf(1 - α, df_diff)


def wilks_test_p(log_likelihood_full, log_likelihood_reduced, df_diff=1) -> float:
    λ = -2 * (log_likelihood_reduced - log_likelihood_full)
    log_p_value = chi2.logsf(λ, df_diff)
    return log_p_value


@jit(nopython=True, cache=True, parallel=False)
def filter_objective_numba(x, eg_lengths, eg_read_counts, eg_rgr_ids):

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


# find ORFs based on transcript sequence
# return list of intervals of all ORFs with min length
def find_orfs(
    seq: str,
    start_codons: list = ["ATG"],
    stop_codons: list = ["TAA", "TAG", "TGA"],
    min_length: int = 0,
) -> list[tuple[int, int]]:
    start_codons = set(start_codons)
    stop_codons = set(stop_codons)
    orf_iv_on_transcript = []
    for i in range(3):
        starts = []
        for j in range(i, len(seq), 3):
            if seq[j : j + 3] in start_codons:
                starts.append(j)
            if seq[j : j + 3] in stop_codons:
                for start in starts:
                    if j - start >= min_length:
                        orf_iv_on_transcript.append(
                            (start, j + 3)
                        )  # with +3 including stop codon
                starts = []
    return orf_iv_on_transcript


class Callback:
    def __init__(
        self, initial_guess, number_samples, config, rgr_lengths=None, remove_rgrs=False
    ):
        self.config = config
        self.previous = initial_guess
        self.initial_guess = initial_guess
        self.success = False
        self.number_samples = number_samples
        self.rgr_lengths = rgr_lengths
        self.rgr_indices_to_remove = set()
        self.remove_rgrs = remove_rgrs

    def __call__(self, new):
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
    x,
    num_runs,
    cm_lut,
    coverage_params,
    egs,  # egs: length, read_count, read_length, oua, rgr_frame_covpos
    num_rgrs,
    λ: float,
) -> float:

    # num_runs = len(run_read_counts)

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


# compute the negative log likelihood
# and the gradient regarding each activity parameter
@jit(nopython=True, cache=True, parallel=False)
def objective_function_wo_regularization(
    x,
    num_runs,
    cm_lut,
    coverage_params,
    egs,  # egs: length, read_count, read_length, oua, rgr_frame_covpos
) -> float:

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


# compute the negative log likelihood + a penalty term
# and the gradient regarding each activity parameter
@jit(nopython=True, cache=True, parallel=False)
def log_likelihood(
    x,
    num_runs,
    cm_lut,
    coverage_params,
    egs,  # egs: length, read_count, read_length, oua, rgr_frame_covpos
) -> float:

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

            # ll += y * np.log(δ) - δ - math.log(math.factorial(y))
            ll += y * np.log(δ) - δ - math.lgamma(y + 1)

    return ll


# def run_likelihood_optimization(
#     initial_guess,
#     bounds,
#     num_runs,
#     cm_lut,
#     cov_params,
#     egs,
#     config,
# ):
#
#     cb = Callback(initial_guess, num_runs, config)
#     optimization_result = minimize(
#         objective_function_wo_regularization,
#         initial_guess,
#         args=(
#             num_runs,
#             cm_lut,
#             cov_params,
#             egs,
#         ),
#         method="L-BFGS-B",
#         jac=True,
#         bounds=bounds,
#         callback=cb,
#         options={
#             "maxiter": 10_000,
#             "ftol": config.ftol,
#             "gtol": config.gtol,
#             "maxls": config.maxls,
#         },
#     )
#     if cb.success:
#         optimization_result.success = True
#     if not optimization_result.success:
#         raise RuntimeError(f"Likelihood ratio filtering failed to converge")
#
#     ll = log_likelihood(
#         optimization_result.x,
#         num_runs,
#         cm_lut,
#         cov_params,
#         egs,
#     )
#     return optimization_result, ll
#
