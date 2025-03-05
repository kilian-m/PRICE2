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
from scipy.sparse import block_diag, coo_array
from scipy.stats import poisson
from skglm import GroupLasso
from numba import jit
from numba.typed import List

from skglm.datafits import QuadraticGroup
from skglm.penalties import WeightedGroupL2
from skglm.solvers import GroupBCD
from skglm import GeneralizedLinearEstimator

from skglm.utils.data import grp_converter

from numba.core.errors import NumbaTypeSafetyWarning
import warnings

warnings.simplefilter("ignore", category=NumbaTypeSafetyWarning)

from price2.genomic_features import ReadGeneratingRegion, Transcript
from price2.ribo_seq_alignment import RiboSeqAlignment
from price2.ribo_seq_run import RiboSeqRun


class EquivalenceGroup:
    length: int
    read_count: int

    def __init__(
        self,
        length: int = 0,
        read_count: int = 0,
    ) -> None:
        self.length = length
        self.read_count = read_count


class Locus:
    # permanent properties
    iv: HTSeq.GenomicInterval
    read_count: int
    read_counts: dict[RiboSeqRun, int]
    rgr_set = set[ReadGeneratingRegion]
    # rgr_intervals: HTSeq.GenomicArrayOfSets
    transcripts: set[Transcript]

    egs: dict[EquivalenceGroup, EquivalenceGroup]

    def __init__(
        self,
        iv: HTSeq.GenomicInterval,
        transcript_intervals: HTSeq.GenomicArrayOfSets,
        loci_number: int,
    ) -> None:
        self.iv = iv
        self.read_count = 0
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
        self, genome: dict[str : HTSeq._HTSeq.Sequence], min_length_to_end: int = 30
    ) -> None:
        self.rgr_set = set()
        orf_dict = dict()

        for transcript in self.transcripts:
            noise = ReadGeneratingRegion(
                "NOISE",
                transcript,
                transcript.id,
                (0, len(transcript.exons)),
            )
            transcript.noise = noise
            if not noise in self.rgr_set:
                self.rgr_set.add(noise)

            seq = transcript.exons.get_sequence(genome)
            c = 0
            for orf_iv_on_transcript in find_orfs(seq):
                c += 1
                # orf_iv_on_transcript[1] -= 3
                orf = ReadGeneratingRegion(
                    "ORF",
                    transcript,
                    f"{transcript.id}_{c:04d}",
                    orf_iv_on_transcript,
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
        for orf in orf_dict.values():
            # orf.transcript.orf_set.add(orf)
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
                        rgr_frame = self.get_rgr_frames(rsa, run)
                        if not rgr_frame:
                            continue
                        if (
                            not (rgr_frame, len(rsa), rsa.untemplated_addition)
                            in self.egs[run]
                        ):
                            self.egs[run][
                                (rgr_frame, len(rsa), rsa.untemplated_addition)
                            ] = EquivalenceGroup()
                        self.egs[run][
                            (rgr_frame, len(rsa), rsa.untemplated_addition)
                        ].length += 1

    def get_rgr_frames(
        self,
        rsa: RiboSeqAlignment,
        run: RiboSeqRun,
        overlap_likelihood_ratio_threshold: float = 0.9,  # .1 .5 .9
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
            for orf in tr.orf_set:
                # full overlap with orf
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
                        rgr_frame.add((orf, frame))
                # part overlap with orf
                elif (
                    rsa_iv_on_tr[0] <= orf.iv_on_transcript[0] <= rsa_iv_on_tr[1]
                ) or (rsa_iv_on_tr[0] <= orf.iv_on_transcript[1] <= rsa_iv_on_tr[1]):
                    frame = (rsa_iv_on_tr[0] - orf.iv_on_transcript[0]) % 3
                    # compute at which position in the read the orf starts
                    region_start = orf.iv_on_transcript[0] - rsa_iv_on_tr[0]
                    # compute at which position in the read the orf ends
                    region_end = orf.iv_on_transcript[1] - rsa_iv_on_tr[0]
                    if (
                        ol := run.cleavage_model.pmf(
                            len(rsa),
                            rsa.untemplated_addition,
                            frame,
                            region_start=region_start,
                            region_end=region_end,
                        )
                        > 0
                    ):
                        cl = run.cleavage_model.pmf(
                            len(rsa), rsa.untemplated_addition, frame
                        )
                        if ol / cl > overlap_likelihood_ratio_threshold:
                            rgr_frame.add(
                                (orf, (rsa_iv_on_tr[0] - orf.iv_on_transcript[0]) % 3)
                            )

        if not rgr_frame:
            return

        rgr_frame = frozenset(rgr_frame)
        return rgr_frame

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
                        > 0
                    ):
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

    # def to_gtf(self, rgrs=None) -> str:
    #    seq_id = self.iv.chrom
    #    source = "PRICE2"
    #    typ = "exon"
    #    start = self.iv.start
    #    end = self.iv.end
    #    score = "."
    #    strand = self.iv.strand
    #    phase = "."
    #    attributes = f'gene_id "{self.id}";'
    #
    #    s = f"{seq_id}\t{source}\t{typ}\t{start}\t{end}\t{score}\t{strand}\t{phase}\t{attributes}\n"
    #
    #    if isinstance(rgrs, type(None)):
    #        for rgr in self.rgr_set:
    #            s += rgr.to_gtf()
    #    else:
    #        rgr_dict = {rgr.id: rgr for rgr in self.rgr_set}
    #        for rgr in self.rgr_set:
    #            if rgr.type == "NOISE":
    #                s += rgr.to_gtf()
    #        for rgr in rgrs:
    #            s += rgr_dict[rgr].to_gtf()
    #
    #    return s

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

    def to_gtf(self, path):
        lock = FileLock(path + ".lock")
        with lock:
            with open(path, "a") as f:
                f.write(self.gtf_line())
                for rgr in self.rgr_set:
                    f.write(rgr.to_gtf(self.id))

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

    def make_well_fitting_reads(self, runs):
        well_fitting_rcs = {}
        for run in runs:
            well_fitting_rcs[run.id] = {}
            for rgr in self.rgr_set:
                if rgr.type == "ORF":
                    well_fitting_rcs[run.id][rgr.id] = 0

        for run in runs:
            well_fitting_indices = run.cleavage_model.get_high_prob_indices()
            well_fitting_length_oua = {(l, oua) for l, f, oua in well_fitting_indices}
            for rsa in self.rsas_dict[run.id]:
                if (
                    len(rsa),
                    int(rsa.untemplated_addition),
                ) not in well_fitting_length_oua:
                    continue
                rgr_frame = self.get_rgr_frames(rsa, run)
                if not rgr_frame:
                    continue
                for rgr, frame in rgr_frame:
                    if rgr.type == "NOISE":
                        continue
                    if (
                        len(rsa),
                        int(rsa.untemplated_addition),
                        frame,
                    ) in well_fitting_indices:
                        well_fitting_rcs[run.id][rgr.id] += rsa.read_count

        self.wfr_df = (
            pd.DataFrame.from_dict(well_fitting_rcs).replace(np.nan, 0).astype(np.int32)
        )

    # filter RGRs based on coverage
    # compute coverage with well fitting reads for each RGR and each run
    # remove RGRs max coverage below threshold
    def coverage_filter_rgrs(self, threshold=0.01):

        rgr_lengths = {rgr.id: len(rgr.genomic_region) for rgr in self.rgr_set}

        rgr_lengths = pd.Series(rgr_lengths).reindex(self.wfr_df.index)
        wfr_df_rel = self.wfr_df.div(rgr_lengths, axis=0)

        keep_ORFs_ids = set(wfr_df_rel[wfr_df_rel.max(axis=1) > threshold].index)

        rgr_set = set()
        for rgr in self.rgr_set:
            if rgr.id in keep_ORFs_ids:
                rgr_set.add(rgr)
            elif rgr.type == "NOISE":
                rgr_set.add(rgr)

        self.rgr_set = rgr_set
        for c, rgr in enumerate(self.rgr_set):
            rgr.index = c

    def deconvolution_filter_rgrs(self, min_activity_fraction):

        tmp = self.make_stop_groups()
        optimization_groups = self.split_stop_groups(tmp)

        remove_rgrs = set()
        for opt_group in optimization_groups:
            remove_rgrs |= self.deconvolute_opt_group(opt_group, min_activity_fraction)

        rgr_set = set()
        for rgr in self.rgr_set:
            if rgr.type == "NOISE":
                rgr_set.add(rgr)
            elif not rgr.id in remove_rgrs:
                rgr_set.add(rgr)

        self.rgr_set = rgr_set
        for c, rgr in enumerate(self.rgr_set):
            rgr.index = c

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
        self, opt_group, min_activity_fraction=0.1, pseudo_min=1e-14
    ):
        rgr_indices_to_remove = []
        rgr_read_counts = dict(self.wfr_df.sum(axis=1))
        l = list(opt_group)
        l.sort(key=len, reverse=True)
        rgr_indices = {rgr.id: i for i, rgr in enumerate(l)}

        min_reads = self.wfr_df.sum().sum() / self.wfr_df.shape[1] * 0.1

        # iterate over runs
        for i in range(len(self.wfr_df.iloc[0])):

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

            bounds = [(pseudo_min, np.inf) for _ in range(len(egs))]
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

            rgr_indices_to_remove_one_run = set(
                np.where(result.x / result.x.sum() < min_activity_fraction)[0]
            )
            rgr_indices_to_remove.append(rgr_indices_to_remove_one_run)

        try:
            rgr_indices_to_remove = set.intersection(*rgr_indices_to_remove)
        except TypeError:
            rgr_indices_to_remove = set()

        return set([k for k, v in rgr_indices.items() if v in rgr_indices_to_remove])

    def assign_reads_to_egs(self, runs):
        for run in runs:
            run_id = run.id

            for rsa in self.rsas_dict[run_id]:
                rgr_frame = self.get_rgr_frames(rsa, run)
                read_count = rsa.read_count
                if not rgr_frame:
                    continue

                try:
                    self.egs[run][
                        (rgr_frame, len(rsa), rsa.untemplated_addition)
                    ].read_count += read_count
                    run.read_count += read_count
                except KeyError:
                    self.uncounted_reads += read_count

                try:
                    self.read_counts[run] += read_count
                except KeyError:
                    self.read_counts[run] = int(read_count)

    def to_objective_args(self, runs):
        rgrs = list(self.rgr_set)

        run_read_counts = np.array([run.read_count for run in runs])

        cm_lut = np.zeros((len(runs), runs[0].cleavage_model.cds_lut.shape[0], 4, 2))
        for i, run in enumerate(runs):
            cm_lut[i, :, 3, :] = run.cleavage_model.noise_lut / 3
            cm_lut[i, :, :3, :] = run.cleavage_model.cds_lut

        egs = []
        for run in runs:
            l = []

            for (rgr_frame, read_length, oua), eg in self.egs[run].items():
                temp = []
                for rgr, frame in rgr_frame:
                    if frame == None:
                        frame = 3
                    temp.append((rgr.index, frame))
                l.append((eg.length, eg.read_count, read_length, int(oua), temp))
            egs.append(l)

        num_rgrs = len(rgrs)
        rgr_lengths = np.array([len(rgr) for rgr in rgrs])

        # initial_guess = np.full((num_rgrs, len(runs)), 1/sum([len(rgr) for rgr in rgrs]))
        initial_guess = np.full((num_rgrs, len(runs)), 1 / sum(rgr_lengths))

        initial_guess = initial_guess.flatten()

        egs_unconverted = egs
        egs = List()

        for run in egs_unconverted:
            l = List()
            for eg in run:
                temp = List()
                for rgr, frame in eg[4]:
                    temp.append((rgr, frame))
                l.append((eg[0], eg[1], eg[2], eg[3], temp))

            egs.append(l)

        return (run_read_counts, cm_lut, egs, num_rgrs, rgr_lengths, initial_guess)

    def get_regression_parameters(self, runs):

        # create y
        ys = []
        for run in runs:
            yt = np.zeros(len(self.egs[run]))
            for i, eg in enumerate(self.egs[run].values()):
                yt[i] = eg.read_count
            ys.append(yt)
        y = np.concatenate(ys)

        # create x
        xs = []
        for run in runs:
            xt = np.zeros((len(self.egs[run]), len(self.rgr_set)))
            for i, ((rgr_frame, read_length, oua), eg) in enumerate(
                self.egs[run].items()
            ):
                for rgr, frame in rgr_frame:
                    xt[i, rgr.index] = (
                        eg.length
                        * run.read_count
                        * run.cleavage_model.pmf(read_length, oua, frame)
                    )
            xs.append(coo_array(xt))
        x = block_diag(xs)

        # create groups
        groups = [[] for _ in range(len(self.rgr_set))]
        for j in range(len(self.rgr_set)):
            for i in range(len(runs)):
                groups[j].append(i * len(self.rgr_set) + j)

        return x, y, groups

    def deconvolve(
        self,
        run_read_counts,
        cm_lut,
        egs,
        num_rgrs,
        rgr_lengths,
        initial_guess,
        ftol,
        gtol,
        callback,
        callback_args,
        pseudo_min,
        lower_λ=5,
        upper_λ=5,
        number_λs=1,
    ):
        # if num_rgrs < 100:
        #    nb.set_num_threads(1)
        # else:
        #    nb.set_num_threads(16)

        λs = np.logspace(lower_λ, upper_λ, number_λs)

        self.initial_guesses = []
        self.cb_dict = {}
        self.optimization_results = {}
        self.regularization_dict = {}
        self.λ_2_normal_result = {}
        self.λ_2_result = {}

        bounds = [(pseudo_min, None)] * len(initial_guess)

        for λ in λs:
            self.initial_guesses.append(initial_guess)
            if callback:
                self.cb = Callback(
                    num_rgrs,
                    len(run_read_counts),
                    initial_guess=initial_guess,
                    rel_thr=callback_args[0],
                    abs_thr=callback_args[1],
                )
                self.cb.args = (run_read_counts, cm_lut, egs, num_rgrs, λ)
            else:
                self.cb = None
            self.cb_dict[λ] = self.cb

            optimization_result = minimize(
                objective_function,
                initial_guess,
                args=(
                    run_read_counts,
                    cm_lut,
                    egs,
                    num_rgrs,
                    λ,
                ),
                method="L-BFGS-B",
                jac=True,
                callback=self.cb,
                bounds=bounds,
                options={"maxiter": 10_000, "gtol": gtol, "ftol": ftol},
            )

            self.optimization_results[λ] = optimization_result
            initial_guess = optimization_result.x

            ### compute BIC
            # number of observations = number of reads
            # dof = |non-zero activities|
            # BIC = -2 * log(likelihood) + log(observations) * dof

            ll = log_likelihood(optimization_result.x, run_read_counts, cm_lut, egs)

            with np.errstate(invalid="ignore"):
                tmp = optimization_result.x.reshape(num_rgrs, len(self.counted_reads))
                tmp = tmp / tmp.sum(axis=0)
                dof = (tmp > 1e-2).sum()

            self.read_count = sum(self.read_counts.values())

            bic = -2 * ll + np.log(self.read_count) * dof

            self.regularization_dict[λ] = (
                bic,
                -2 * ll,
                np.log(self.read_count) * dof,
            )
            if callback:
                del self.cb.args

            self.λ_2_result[λ] = optimization_result
            tmp = optimization_result.x.reshape(num_rgrs, len(run_read_counts))
            tmp[tmp <= pseudo_min] = 0
            with np.errstate(invalid="ignore"):
                tmp = tmp / tmp.sum(axis=0)
                tmp[np.isnan(tmp)] = 0
            normalized_result = tmp

            self.λ_2_normal_result[λ] = normalized_result

        self.best_λ = min(self.regularization_dict, key=self.regularization_dict.get)
        self.best_normal_result = self.λ_2_normal_result[self.best_λ]
        self.best_result = self.λ_2_result[self.best_λ]

    def deconvolve_wo_regularization(
        self,
        run_read_counts,
        cm_lut,
        egs,
        num_rgrs,
        rgr_lengths,
        initial_guess,
        ftol,
        gtol,
        pseudo_min=1e-14,
    ):
        # if num_rgrs < 100:
        #    nb.set_num_threads(1)
        # else:
        #    nb.set_num_threads(16)

        self.initial_guesses = []
        self.cb_dict = {}
        self.optimization_results = {}
        self.regularization_dict = {}
        self.λ_2_normal_result = {}
        self.λ_2_result = {}

        self.best_λ = 0

        bounds = [(pseudo_min, None)] * len(initial_guess)

        self.initial_guesses.append(initial_guess)
        self.cb = Callback(num_rgrs, len(run_read_counts), initial_guess=initial_guess)
        self.cb.args = (run_read_counts, cm_lut, egs, num_rgrs, 0)

        optimization_result = minimize(
            objective_function_wo_regularization,
            initial_guess,
            args=(
                run_read_counts,
                cm_lut,
                egs,
            ),
            method="L-BFGS-B",
            jac=True,
            # callback=self.cb,
            bounds=bounds,
            options={"maxiter": 10_000, "gtol": gtol, "ftol": ftol},
        )

        ll = log_likelihood(optimization_result.x, run_read_counts, cm_lut, egs)

        self.read_count = sum(self.read_counts.values())

        del self.cb.args

        tmp = optimization_result.x.reshape(num_rgrs, len(run_read_counts))
        tmp[tmp <= pseudo_min] = 0
        with np.errstate(invalid="ignore"):
            tmp = tmp / tmp.sum(axis=0)
            tmp[np.isnan(tmp)] = 0
        normalized_result = tmp

        self.optimization_results[0] = optimization_result
        self.best_normal_result = normalized_result
        self.best_result = optimization_result

    def deconvolve_regression(
        self,
        runs,
        lower_λ=-2,
        upper_λ=7,
        number_λs=100,
    ):
        λs = np.logspace(lower_λ, upper_λ, number_λs)

        x, y, g = self.get_regression_parameters(runs)

        self.r_optimization_results = {}
        self.r_normal_results = {}
        self.r_ll = {}
        self.r_n_features = {}
        self.r_bics = {}
        self.r_dof = {}

        for λ in λs:

            grp_indices, grp_ptr = grp_converter(g, len(y))
            group_sizes = np.diff(grp_ptr)
            weights = np.ones(len(group_sizes))

            gle = GeneralizedLinearEstimator(
                datafit=QuadraticGroup(
                    grp_ptr=grp_ptr,
                    grp_indices=grp_indices,
                ),
                penalty=WeightedGroupL2(
                    alpha=λ,
                    weights=weights,
                    grp_indices=grp_indices,
                    grp_ptr=grp_ptr,
                    positive=True,
                ),
                solver=GroupBCD(warm_start=True),
            )

            try:
                gle.coef_ = initial_guess
            except NameError:
                pass
            gle.fit(x, y)
            initial_guess = gle.coef_

            result = gle.coef_.reshape((len(runs), -1)).T
            self.r_optimization_results[λ] = result

            with np.errstate(invalid="ignore"):
                tmp = result / result.sum(axis=0)
                tmp[np.isnan(tmp)] = 0

            self.r_normal_results[λ] = tmp

            ll = sum(poisson.logpmf(y, gle.predict(x)))
            self.r_ll[λ] = ll

            # compute BIC
            n_samples = x.shape[0]
            # n_features = np.sum(group_lasso.coef_ != 0)
            n_features = sum(result.sum(axis=1) != 0)
            dof = np.log(n_samples) * n_features
            self.r_dof[λ] = dof
            self.r_n_features[λ] = n_features

            # bic = n_samples * ll + np.log(n_samples) * n_features
            bic = -2 * ll + np.log(n_samples) * n_features
            self.r_bics[λ] = bic

        self.r_best_λ = min(self.r_bics, key=self.r_bics.get)
        self.r_best_result = self.r_optimization_results[self.r_best_λ]

    def collapse_egs(
        self,
        runs,
        min_activity_threshold=0.01,
    ):
        self.noise_rgr_indices = set(
            [rgr.index for rgr in self.rgr_set if rgr.type == "NOISE"]
        )
        test_rgr_indices = (
            set(np.where(self.best_normal_result > min_activity_threshold)[0])
            - self.noise_rgr_indices
        )
        keep_rgr_inds = self.noise_rgr_indices | test_rgr_indices
        self.rgr_set = set([rgr for rgr in self.rgr_set if rgr.index in keep_rgr_inds])

        new_egs = {}
        for run in runs:
            new_egs[run] = {}
            for old_eg_key in self.egs[run]:
                (rgr_frames, read_length, oua) = old_eg_key
                new_rgr_frames = set()
                for rgr, frame in rgr_frames:
                    if rgr.index in keep_rgr_inds:
                        new_rgr_frames.add((rgr, frame))
                new_rgr_frames = frozenset(new_rgr_frames)
                new_eg_key = (new_rgr_frames, read_length, oua)
                try:
                    new_egs[run][new_eg_key].length += self.egs[run][old_eg_key].length
                    new_egs[run][new_eg_key].read_count += self.egs[run][
                        old_eg_key
                    ].read_count
                except KeyError:
                    new_egs[run][new_rgr_frames, read_length, oua] = EquivalenceGroup(
                        self.egs[run][old_eg_key].length,
                        self.egs[run][old_eg_key].read_count,
                    )

        self.egs = new_egs
        for c, rgr in enumerate(self.rgr_set):
            rgr.index = c

    def likelihood_ratio_filtering(
        self,
        run_read_counts,
        cm_lut,
        egs,
        num_rgrs,
        rgr_lengths,
        initial_guess,
        ftol=1e-6,
        gtol=1e-6,
        pseudo_min=1e-14,
        α=1e-5,
    ):

        noise_rgr_indices = {rgr.index for rgr in self.rgr_set if rgr.type == "NOISE"}
        test_rgr_indices = {rgr.index for rgr in self.rgr_set if rgr.type == "ORF"}
        keep_rgr_indices = noise_rgr_indices | test_rgr_indices

        shape = initial_guess.reshape(num_rgrs, -1).shape

        t = np.empty((), dtype=object)
        t[()] = (pseudo_min, None)
        bounds = list(np.full(initial_guess.shape, t))

        optimization_result = minimize(
            objective_function,
            initial_guess,
            args=(
                run_read_counts,
                cm_lut,
                egs,
                num_rgrs,
                1,
            ),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 10_000, "ftol": ftol, "gtol": gtol},
        )

        self.final_result = optimization_result

        initial_guess = optimization_result.x

        log_likelihood_all = log_likelihood(
            optimization_result.x,
            run_read_counts,
            cm_lut,
            egs,
        )

        rgr_ind_list = list(test_rgr_indices)
        try:
            act_sum = self.best_normal_result[np.array(rgr_ind_list)].sum(axis=1)
            rgr_ind_list = np.array(rgr_ind_list)[np.argsort(act_sum)]
        except IndexError:
            rgr_ind_list = []

        for rgr_ind in rgr_ind_list:
            free_rgr_ind = keep_rgr_indices - {rgr_ind}
            initial_guess_int = initial_guess.copy().reshape(num_rgrs, -1)
            initial_guess_int[rgr_ind] = pseudo_min
            initial_guess_int = initial_guess_int.flatten()

            t = np.empty((), dtype=object)
            t[()] = (pseudo_min, pseudo_min)
            bounds = np.full(shape, t)
            t[()] = (pseudo_min, None)
            for index in free_rgr_ind:
                bounds[index] = t
            bounds = list(bounds.flatten())

            optimization_result_int = minimize(
                objective_function,
                initial_guess_int,
                args=(
                    run_read_counts,
                    cm_lut,
                    egs,
                    num_rgrs,
                    1,
                ),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 10_000, "ftol": ftol, "gtol": gtol},
            )
            log_likelihood_int = log_likelihood(
                optimization_result_int.x,
                run_read_counts,
                cm_lut,
                egs,
            )

            if not wilks_test(
                log_likelihood_all,
                log_likelihood_int,
                df_diff=len(run_read_counts),  # number of datasets
                α=α,
            ):
                # remove rgr
                keep_rgr_indices.remove(rgr_ind)
                initial_guess = optimization_result_int.x
                log_likelihood_all = log_likelihood_int
                self.final_result = optimization_result_int

        tmp = self.final_result.x.reshape(num_rgrs, len(run_read_counts))
        tmp[tmp <= pseudo_min] = 0
        with np.errstate(invalid="ignore"):
            tmp = tmp / tmp.sum(axis=0)
            tmp[np.isnan(tmp)] = 0
        self.final_normalized_result = tmp
        self.keep_rgr_inds = keep_rgr_indices

    def result_matrix(self):
        return self.best_result.x.reshape(len(self.rgr_set), -1)

    def compute_rpkm(self, runs):
        # make dataframe
        temp = {rgr.index: rgr for rgr in self.rgr_set}
        rgr_ids = [temp[i].id for i in range(len(temp))]
        run_ids = [run.id for run in runs]

        try:
            self.final_normalized_result
        except AttributeError:
            print(self.id)

        self.result_df = pd.DataFrame(
            self.final_normalized_result, index=rgr_ids, columns=run_ids
        )
        self.result_df["mean"] = self.result_df.mean(axis=1)
        self.result_df.sort_values(by="mean", inplace=True, ascending=False)
        self.result_df.drop("mean", axis=1, inplace=True)
        self.result_df = self.result_df[self.result_df.sum(axis=1) > 0]

        s = set(self.result_df.index)
        self.rgr_dict = {}
        for rgr in self.rgr_set:
            if rgr.id in s:
                self.rgr_dict[rgr.id] = rgr
        self.rgr_set = {rgr for rgr in self.rgr_set if rgr.id in s}
        for c, rgr in enumerate(self.rgr_set):
            rgr.index = c

        read_counts = {}
        rpkm_dict = {}
        for rgr in self.rgr_dict.values():
            read_counts[rgr.id] = {}
            rpkm_dict[rgr.id] = {}
        for run in runs:
            s = sum(
                [
                    rgr.effective_length() * self.result_df[run.id][rgr.id]
                    for rgr in self.rgr_dict.values()
                ]
            )
            for rgr in self.rgr_dict.values():
                if s == 0:
                    read_counts[rgr.id][run.id] = 0
                else:
                    try:
                        rc = self.read_counts[run]
                    except KeyError:
                        rc = 0
                    read_counts[rgr.id][run.id] = (
                        rgr.effective_length()
                        * self.result_df[run.id][rgr.id]
                        / s
                        # * run.read_count
                        * rc
                        # * self.read_counts[run]
                    )
                rpkm_dict[rgr.id][run.id] = (
                    read_counts[rgr.id][run.id] / (run.read_count / 1e6)
                ) / rgr.effective_length()
        self.rpkm_df = pd.DataFrame(rpkm_dict).T


# reject null if true
def wilks_test(log_likelihood_full, log_likelihood_reduced, df_diff=1, α=1e-5) -> bool:
    # reject null hypothesis if the test statistic is greater than the critical value
    return 2 * (log_likelihood_full - log_likelihood_reduced) > chi2.ppf(1 - α, df_diff)


@jit(nopython=True, cache=True, parallel=False)
def filter_objective_numba(x, eg_lengths, eg_read_counts, eg_rgr_ids):

    Σ_rc = sum(eg_read_counts)
    loss = 0
    grads = np.zeros_like(x)

    for eg_len, eg_rc, eg_rgrs in zip(eg_lengths, eg_read_counts, eg_rgr_ids):
        activity = 0
        δ_derived = np.zeros_like(x)
        for rgr_id in eg_rgrs:
            activity += x[rgr_id]
            δ_derived[rgr_id] += Σ_rc * eg_len

        δ = Σ_rc * eg_len * activity
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
    min_length: int = 0,
    start_codons: set = set(["ATG", "CTG", "GTG", "ACG"]),
    stop_codons: set = set(["TAA", "TAG", "TGA"]),
) -> list[tuple[int, int]]:
    orf_iv_on_transcript = []
    for i in range(3):
        starts = []
        for j in range(i, len(seq), 3):
            if seq[j : j + 3] in start_codons:
                starts.append(j)
            if seq[j : j + 3] in stop_codons:
                for start in starts:
                    if j - start >= min_length:
                        orf_iv_on_transcript.append((start, j))  # j + 3
                starts = []
    return orf_iv_on_transcript


class Callback:
    def __init__(
        self,
        num_rgrs,
        num_runs,
        initial_guess,
        pseudo_min=1e-14,
        rel_thr=0.001,
        abs_thr=0.001,
    ):
        self.num_rgrs = num_rgrs
        self.num_runs = num_runs
        self.pseudo_min = pseudo_min
        self.rel_thr = rel_thr
        self.abs_thr = abs_thr

        tmp = initial_guess.reshape(self.num_rgrs, self.num_runs)
        tmp[tmp <= self.pseudo_min] = 0
        with np.errstate(invalid="ignore"):
            tmp = tmp / tmp.sum(axis=0)
            tmp[np.isnan(tmp)] = 0
        normalized_result = tmp

        self.intermediate_normalized_results = [normalized_result]
        # self.ll = []

    def __call__(self, x):
        tmp = x.reshape(self.num_rgrs, self.num_runs)
        # self.ll.append(objective_function(x, *self.args)[0])

        tmp[tmp <= self.pseudo_min] = 0
        with np.errstate(invalid="ignore"):
            tmp = tmp / tmp.sum(axis=0)
            tmp[np.isnan(tmp)] = 0

        self.intermediate_normalized_results.append(tmp)

        l = 5
        if len(self.intermediate_normalized_results) > l:
            if cb_stop(
                np.array(self.intermediate_normalized_results[-l:]),
                self.rel_thr,
                self.abs_thr,
            ):
                raise StopIteration

            # tmp = np.array(self.ll[-l:])
            # if np.abs(((tmp[0 : l - 1] - tmp[1:l]) / tmp[1:l])).mean() < 0.001:
            #    raise StopIteration


@jit(nopython=True, cache=True, parallel=False)
def cb_stop(x, rel_thr, abs_thr):
    relative_activity_change = 0
    max_absolute_activity_change = 0
    for i in range(-len(x) + 1, 0):
        relative_activity_change += (
            np.exp(np.abs(np.log(x[i - 1].sum() / x[i].sum()))) - 1
        )
        max_absolute_activity_change += np.max(
            np.abs(x[i - 1] / x[i - 1].sum(axis=0) - (x[i] / x[i].sum(axis=0)))
        )
    if relative_activity_change < rel_thr and max_absolute_activity_change < abs_thr:
        return True
    else:
        return False


# compute the negative log likelihood + a penalty term
# and the gradient with respect to each activity parameter
@jit(nopython=True, parallel=False, cache=True)
def objective_function(
    x,
    run_read_counts,
    cm_lut,
    egs,  # egs: length, read_count, read_length, oua, rgr_frame
    num_rgrs,
    λ: float,
) -> float:

    num_runs = len(run_read_counts)

    loss = 0
    grads = np.zeros_like(x)

    for run_index in range(num_runs):
        for eg in egs[run_index]:
            activity = 0
            δ_derived = np.zeros_like(x)
            for rgr_index, frame in eg[4]:
                activity += (
                    x[rgr_index * num_runs + run_index]
                    * cm_lut[run_index, eg[2], frame, eg[3]]
                )
                δ_derived[rgr_index * num_runs + run_index] += cm_lut[
                    run_index, eg[2], frame, eg[3]
                ]

            δ = run_read_counts[run_index] * eg[0] * activity
            δ_derived *= run_read_counts[run_index] * eg[0]

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
@jit(nopython=True, parallel=False, cache=True)
def objective_function_wo_regularization(
    x,
    run_read_counts,
    cm_lut,
    egs,  # egs: length, read_count, read_length, oua, rgr_frame
) -> float:

    num_runs = len(run_read_counts)

    loss = 0
    grads = np.zeros_like(x)

    for run_index in range(num_runs):
        for eg in egs[run_index]:
            activity = 0
            δ_derived = np.zeros_like(x)
            for rgr_index, frame in eg[4]:
                activity += (
                    x[rgr_index * num_runs + run_index]
                    * cm_lut[run_index, eg[2], frame, eg[3]]
                )
                δ_derived[rgr_index * num_runs + run_index] += cm_lut[
                    run_index, eg[2], frame, eg[3]
                ]

            δ = run_read_counts[run_index] * eg[0] * activity
            δ_derived *= run_read_counts[run_index] * eg[0]

            y = eg[1]

            if y == 0 and δ == 0:
                # print("δ == 0")
                # print(run_read_counts[run_index])
                continue

            loss += δ - y * np.log(δ)
            grads += -y * δ_derived / δ + δ_derived

    return loss, grads


# compute the negative log likelihood + a penalty term
# and the gradient regarding each activity parameter
@jit(nopython=True, cache=True, parallel=False)
def log_likelihood(
    x,
    run_read_counts,
    cm_lut,
    egs,  # egs: length, read_count, read_length, oua, rgr_frame
) -> float:

    num_runs = len(run_read_counts)

    ll = 0

    for run_index in range(num_runs):
        for eg in egs[run_index]:
            activity = 0

            for rgr_index, frame in eg[4]:
                activity += (
                    x[rgr_index * num_runs + run_index]
                    * cm_lut[run_index, eg[2], frame, eg[3]]
                )

            δ = run_read_counts[run_index] * eg[0] * activity

            y = eg[1]
            if y == 0 and δ == 0:
                continue

            # ll += y * np.log(δ) - δ - math.log(math.factorial(y))
            ll += y * np.log(δ) - δ - math.lgamma(y + 1)

    return ll


@jit(nopython=True, cache=True, parallel=False)
def run_orf_deconvolution_em_numba(
    cm_lut: np.ndarray,
    egs,
    rgr_lengths: np.ndarray,
    num_rgrs: int,
    iterations: int,
    activity_change_cutoff: float,
) -> None:

    activities = np.full(num_rgrs, 1 / num_rgrs)

    num_egs = len(egs)

    for i in range(iterations):
        rgr_read_counts = np.zeros(num_rgrs)
        # E-step
        for eg_index in range(num_egs):
            read_count = egs[eg_index][1]
            likelihoods = np.empty(len(egs[eg_index][4]))
            for j, (rgr_index, frame) in enumerate(egs[eg_index][4]):
                likelihoods[j] = (
                    activities[rgr_index]
                    * cm_lut[egs[eg_index][2], frame, egs[eg_index][3]]
                )
            likelihood_sum = likelihoods.sum()
            if likelihood_sum > 0:
                p = likelihoods / likelihood_sum
            else:
                p = np.full(len(likelihoods), 1 / len(egs[eg_index][4]))
            for j, (rgr_index, frame) in enumerate(egs[eg_index][4]):
                rgr_read_counts[rgr_index] += read_count * p[j]

        # M-step
        new_activities = np.zeros(num_rgrs)
        for rgr_index in range(num_rgrs):
            new_activities[rgr_index] = (
                rgr_read_counts[rgr_index] / rgr_lengths[rgr_index]
            )
        new_activities /= new_activities.sum()

        if i > 1:
            activity_change = sum(np.abs(new_activities - activities))
            if activity_change < activity_change_cutoff:
                break

        activities = new_activities

    return activities
