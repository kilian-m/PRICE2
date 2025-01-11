from pickle import loads
import sqlite3
import zlib
import HTSeq
import pandas as pd
import numpy as np

from scipy.optimize import minimize
from numba import jit, prange
from numba.typed import List

from numba.core.errors import NumbaTypeSafetyWarning
import warnings

warnings.simplefilter("ignore", category=NumbaTypeSafetyWarning)

from .genomic_features import ReadGeneratingRegion, Transcript
from .ribo_seq_alignment import RiboSeqAlignment
from .ribo_seq_run import RiboSeqRun


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

        # for iv, value in transcript_intervals[self.iv].steps():
        for iv, value in self.transcript_intervals.steps():
            self.transcripts |= value

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
            # for step_iv, transcripts in self.transcript_intervals.steps():
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

    def to_gtf(self, rgrs=None) -> str:
        seq_id = self.iv.chrom
        source = "PRICE2"
        typ = "exon"
        start = self.iv.start
        end = self.iv.end
        score = "."
        strand = self.iv.strand
        phase = "."
        attributes = f'gene_id "{self.id}";'

        s = f"{seq_id}\t{source}\t{typ}\t{start}\t{end}\t{score}\t{strand}\t{phase}\t{attributes}\n"

        if isinstance(rgrs, type(None)):
            for rgr in self.rgr_set:
                s += rgr.to_gtf(self.id)
        else:
            rgr_dict = {rgr.id: rgr for rgr in self.rgr_set}
            for rgr in self.rgr_set:
                if rgr.type == "NOISE":
                    s += rgr.to_gtf(self.id)
            for rgr in rgrs:
                s += rgr_dict[rgr].to_gtf(self.id)

        return s

    #############################################
    ### functions for orf activity estimation ###
    #############################################

    # get reads from database
    def get_reads_from_db(self, reads_db_path) -> dict[str : list[RiboSeqAlignment]]:
        read_db = sqlite3.connect(reads_db_path)
        read_cursor = read_db.cursor()
        reads_dfs = read_cursor.execute(
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

    def coverage_filter_rgrs(self):

        rgr_lengths = {rgr.id: len(rgr.genomic_region) for rgr in self.rgr_set}

        rgr_lengths = pd.Series(rgr_lengths).reindex(self.wfr_df.index)
        wfr_df_rel = self.wfr_df.div(rgr_lengths, axis=0)

        # TODO: make this an adjustable parameter
        keep_ORFs_ids = set(wfr_df_rel[wfr_df_rel.max(axis=1) > 0.003].index)

        rgr_set = set()
        for rgr in self.rgr_set:
            if rgr.id in keep_ORFs_ids:
                rgr_set.add(rgr)
            elif rgr.type == "NOISE":
                rgr_set.add(rgr)

        self.rgr_set = rgr_set
        for c, rgr in enumerate(self.rgr_set):
            rgr.index = c

    def deconvolution_filter_rgrs(self):

        tmp = self.make_stop_groups()
        optimization_groups = self.split_stop_groups(tmp)

        remove_rgrs = set()
        for opt_group in optimization_groups:
            remove_rgrs |= self.deconvolute_opt_group(opt_group)

        # remove_rgrs = self.get_rgrs_to_remove(self.rgr_set, self.wfr_df)

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

    def deconvolute_opt_group(self, opt_group, min_activity_fraction=0.05):
        rgr_indices_to_remove = []
        rgr_read_counts = dict(self.wfr_df.sum(axis=1))
        l = list(opt_group)
        l.sort(key=len, reverse=True)
        rgr_indices = {rgr.id: i for i, rgr in enumerate(l)}

        min_reads = self.wfr_df.sum().sum() / self.wfr_df.shape[1] * 0.1

        for i in range(len(self.wfr_df.iloc[0])):

            rgr_read_counts = self.wfr_df.iloc[:, i].to_dict()
            if sum(rgr_read_counts.values()) < min_reads:
                continue
            egs = {}
            s = set()
            for i in range(len(l) - 1):
                s.add(l[i].id)
                length = len(l[i]) - len(l[i + 1])
                rc = rgr_read_counts[l[i].id] - rgr_read_counts[l[i + 1].id]
                egs[frozenset(s)] = (length, rc)

            s.add(l[-1].id)
            length = len(l[-1])
            rc = rgr_read_counts[l[-1].id]
            egs[frozenset(s)] = (length, rc)

            bounds = [(1e-14, np.inf) for _ in range(len(egs))]
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
        # rgr_lengths = np.array(rgr_lengths)

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

    def deconvolve(
        self,
        run_read_counts,
        cm_lut,
        egs,
        num_rgrs,
        rgr_lengths,
        initial_guess,
        tolerance,
        lower_λ=3,
        upper_λ=13,
    ):
        number_λs = upper_λ - lower_λ + 11
        λs = np.logspace(lower_λ, upper_λ, number_λs)

        self.initial_guesses = []
        self.cb_dict = {}
        self.optimization_results = {}
        self.regularization_dict = {}
        self.λ_2_result = {}

        bounds = [(1e-14, None)] * len(initial_guess)

        for λ in λs:
            self.initial_guesses.append(initial_guess)
            self.cb = Callback(
                num_rgrs, len(run_read_counts), initial_guess=initial_guess
            )
            self.cb_dict[λ] = self.cb
            self.cb.args = (run_read_counts, cm_lut, egs, num_rgrs, rgr_lengths, λ)

            optimization_result = minimize(
                objective_function,  # _new # _constraint # _normal
                initial_guess,
                args=(
                    run_read_counts,
                    cm_lut,
                    egs,
                    num_rgrs,
                    # rgr_lengths,
                    λ,
                ),  # rgr_lengths
                method="L-BFGS-B",
                jac=True,
                # callback=self.cb,
                bounds=bounds,
                options={"maxiter": 10_000, "ftol": tolerance},
            )

            self.optimization_results[λ] = optimization_result
            initial_guess = optimization_result.x

            ### compute BIC
            # number of observations = number of reads
            # dof = |non-zero activities|
            # BIC = -2 * log(likelihood) + log(observations) * dof

            log_likelihood = -objective_function(
                optimization_result.x, run_read_counts, cm_lut, egs, num_rgrs, 0
            )[0]
            # dof = sum(optimization_result.x > 2 * 1e-14)

            with np.errstate(invalid="ignore"):
                tmp = optimization_result.x.reshape(num_rgrs, len(run_read_counts))
                tmp = tmp / tmp.sum(axis=0)
                dof = (tmp < 1e-2).sum()

            number_observations = sum(run_read_counts)
            bic = -2 * log_likelihood + np.log(number_observations) * dof

            self.regularization_dict[λ] = (
                bic,
                -2 * log_likelihood,
                np.log(number_observations) * dof,
            )

            del self.cb.args

            tmp = optimization_result.x.reshape(num_rgrs, len(run_read_counts))
            tmp[tmp <= 1e-14] = 0
            with np.errstate(invalid="ignore"):
                tmp = tmp / tmp.sum(axis=0)
                tmp[np.isnan(tmp)] = 0
            normalized_result = tmp

            self.λ_2_result[λ] = normalized_result

        self.best_λ = min(self.regularization_dict, key=self.regularization_dict.get)
        self.best_result = self.λ_2_result[self.best_λ]


@jit(nopython=True, cache=True)
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
    def __init__(self, num_rgrs, num_runs, initial_guess):
        self.num_rgrs = num_rgrs
        self.num_runs = num_runs

        tmp = initial_guess.reshape(self.num_rgrs, self.num_runs)
        tmp[tmp <= 1e-10] = 0
        with np.errstate(invalid="ignore"):
            tmp = tmp / tmp.sum(axis=0)
            tmp[np.isnan(tmp)] = 0
        normalized_result = tmp

        self.intermediate_results = [normalized_result]
        self.ll = []

    def __call__(self, x):
        tmp = x.reshape(self.num_rgrs, self.num_runs)
        self.ll.append(objective_function(x, *self.args)[0])

        tmp[tmp <= 1e-10] = 0
        with np.errstate(invalid="ignore"):
            tmp = tmp / tmp.sum(axis=0)
            tmp[np.isnan(tmp)] = 0

        self.intermediate_results.append(tmp)

        l = 10
        if len(self.intermediate_results) > l:

            tmp = np.array(self.ll[-l:])
            if np.abs(((tmp[0 : l - 1] - tmp[1:l]) / tmp[1:l])).mean() < 0.01:
                raise StopIteration

            if cb_stop(np.array(self.intermediate_results[-l:])):
                raise StopIteration


@jit(nopython=True, cache=True)
def cb_stop(x):
    s1 = 0
    s2 = 0
    for i in range(-len(x) + 1, 0):
        s1 += np.exp(np.abs(np.log(x[i - 1].sum() / x[i].sum()))) - 1
        s2 += np.max(
            np.abs(x[i - 1] / x[i - 1].sum(axis=0) - (x[i] / x[i].sum(axis=0)))
        )
    if s1 < 0.001 and s2 < 0.001:
        return True
    else:
        return False


# compute the negative log likelihood + a penalty term
# and the gradient regarding each activity parameter
@jit(nopython=True, parallel=True, cache=True)
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

    for run_index in prange(num_runs):
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
    for rgr_index in prange(num_rgrs):
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


# objective function where the penalty for each RGR scales with the length
@jit(nopython=True, parallel=True, cache=True)
def objective_function_length_penalty(
    x,
    run_read_counts,
    cm_lut,
    egs,  # egs: length, read_count, read_length, oua, rgr_frame
    num_rgrs,
    rgr_lengths,
    λ: float,
) -> float:

    num_runs = len(run_read_counts)

    loss = 0
    grads = np.zeros_like(x)

    for run_index in prange(num_runs):
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
    for rgr_index in prange(num_rgrs):
        s = 0
        for run_index in range(num_runs):
            s += x[rgr_index * num_runs + run_index] ** 2
        s_sqrt = s**0.5
        penalty += s_sqrt * rgr_lengths[rgr_index]  # * rgr_length
        for run_index in range(num_runs):
            grads[rgr_index * num_runs + run_index] += (
                λ
                * rgr_lengths[rgr_index]
                * x[rgr_index * num_runs + run_index]
                / s_sqrt
            )

    return loss + λ * penalty, grads


@jit(nopython=True, cache=True)
def run_orf_deconvolution_em_numba(
    cm_lut: np.ndarray,  # v
    egs,  # v
    rgr_lengths: np.ndarray,
    num_rgrs: int,  # v
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
