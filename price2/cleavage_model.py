import numpy as np
from numba import njit
import HTSeq
import pysam
import pickle

import os

from .reference_annotation import ReferenceAnnotation
from .ribo_seq_alignment import RiboSeqAlignment

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class CleavageModel:

    def __init__(self, pl: np.array, pr: np.array, pu: float) -> None:
        self.pl = pl
        self.pr = pr
        self.pu = pu
        self.cds_lut = np.zeros(
            (len(self.pl) + len(self.pr) + 3, 3, 2), dtype=np.float64
        )
        for length in range(len(self.pl) + len(self.pr) + 3):
            for frame in range(3):
                for oua in range(2):
                    self.cds_lut[length, frame, oua] = read_in_cds_likelihood(
                        pl=self.pl,
                        pr=self.pr,
                        pu=self.pu,
                        length=length,
                        frame=frame,
                        oua=oua,
                        region_start=0,
                        region_end=10**10,
                    )
        self.noise_lut = np.zeros(
            (len(self.pl) + len(self.pr) + 3, 2), dtype=np.float64
        )
        for length in range(len(self.pl) + len(self.pr) + 3):
            for oua in range(2):
                self.noise_lut[length, oua] = read_in_noise_likelihood(
                    pl=self.pl,
                    pr=self.pr,
                    pu=self.pu,
                    length=length,
                    oua=oua,
                    region_start=0,
                    region_end=10**10,
                )

        self.non_zero_lengths = np.nonzero(self.noise_lut.sum(axis=1))[0]

    def pmf(
        self,
        length: int,
        oua: bool,
        frame: int = None,
        region_start: int = 0,
        region_end: int = 10**10,
    ) -> float:
        # region_start relative to read start
        # region_end relative to read start
        # length is the matching length of the alignment
        if length > len(self.pl) + len(self.pr) + 3 + int(oua):
            return 0

        if frame == None:  # noise
            if region_start == 0 and region_end == 10**10:
                temp = self.noise_lut[length, int(oua)]
            else:
                temp = read_in_noise_likelihood(
                    self.pl,
                    self.pr,
                    self.pu,
                    length,
                    oua,
                    region_start,
                    region_end,
                )
            return temp / 3
        else:  # CDS
            if region_start == 0 and region_end == 10**10:
                return self.cds_lut[length, frame, int(oua)]
            f0 = (-frame) % 3
            f1 = (f0 - 1) % 3

            if region_start and f0 != region_start % 3:
                raise ValueError("region_start and frame are not compatible")

            if region_end < 10**10 and f0 != region_end % 3:
                raise ValueError("region_end and frame are not compatible")

            return read_in_cds_likelihood(
                self.pl,
                self.pr,
                self.pu,
                length,
                frame,
                oua,
                region_start,
                region_end,
            )

    def rvs(self, size: int = 1):
        pl = np.random.choice(np.arange(len(self.pl)), size=size, p=self.pl)
        pr = np.random.choice(np.arange(len(self.pr)), size=size, p=self.pr)
        u = np.random.choice([True, False], size=size, p=[self.pu, 1 - self.pu])
        return (pl, pr, u)

    # get all read variants, meaning all valid upstream and downstream cleavages
    # but only the most likely combination for each read length
    def get_read_variants(self):
        length_dict = {}  # length -> (likelihood, (pl, pr, u))
        for i in range(len(self.pl)):
            for j in range(len(self.pr)):
                for u in [True, False]:
                    length = i + 3 + j + int(u)

    def get_high_prob_indices(self, prob_sum: float = 0.3):

        lut = self.cds_lut.copy()
        s = 0
        max_prob_positions = []
        while s < 0.3:
            index = np.unravel_index(lut.argmax(), lut.shape)
            s += lut[index]
            max_prob_positions.append(index)
            lut[index] = 0

        return max_prob_positions

    def plot(self, ax=None) -> None:
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.bar(range(-len(self.pl) + 1, 1), self.pl[::-1])
        ax.bar(range(len(self.pr)), self.pr)
        ax.set_xlim(-30, 25)
        ax.set_ylim(0, 0.8)
        fill = Rectangle((-5, 0.7), self.pu * 25, 0.05, fill=True, facecolor="tab:red")
        frame = Rectangle(
            (-5, 0.7), 25, 0.05, fill=False, edgecolor="black", linewidth=2
        )
        ax.add_patch(fill)
        ax.add_patch(frame)
        ax.set_xlabel("position relative to p-site")
        ax.set_ylabel("cleavage probability")


@njit
def read_in_cds_likelihood(
    pl: np.ndarray,
    pr: np.ndarray,
    pu: float,
    length: int,
    frame: int,
    oua: bool,
    region_start: int = 0,
    region_end: int = 10 * 10,
) -> float:

    f0 = (-frame) % 3

    start_index = max(f0, region_start, length - len(pr) - 2)  # -3
    if start_index % 3 == f0 % 3:
        pass
    elif start_index % 3 == (f0 + 1) % 3:
        start_index += 2
    elif start_index % 3 == (f0 + 2) % 3:
        start_index += 1

    i = np.arange(start_index, min(len(pl), length - 2, region_end - 2), 3)

    l = (pl[i] * pr[length - i - 3]).sum()

    if oua:
        l *= pu * 3 / 4

    else:
        # assume there is no ua
        l *= 1 - pu
        # assume there is an ua
        length -= 1
        region_start -= 1
        region_end -= 1
        frame = (frame + 1) % 3

        f0 = (-frame) % 3

        start_index = max(f0, region_start, length - len(pr) - 3)
        if start_index % 3 == f0 % 3:
            pass
        elif start_index % 3 == (f0 + 1) % 3:
            start_index += 2
        elif start_index % 3 == (f0 + 2) % 3:
            start_index += 1

        i = np.arange(start_index, min(len(pl), length - 2, region_end - 2), 3)

        l += (pl[i] * pr[length - i - 3]).sum() * pu * 1 / 4

    return l


@njit
def read_in_noise_likelihood(
    pl: np.ndarray,
    pr: np.ndarray,
    pu: float,
    length: int,
    oua: bool,
    region_start: int = 0,
    region_end: int = 10 * 10,
) -> float:

    i = np.arange(
        max(0, region_start, length - len(pr) - 2),
        min(len(pl), length - 2, region_end - 2),
    )
    l = (pl[i] * pr[length - i - 3]).sum()

    if oua:
        l *= pu * 3 / 4

    else:
        # assume there is no ua
        l *= 1 - pu

        # assume there is an ua
        length -= 1
        region_start -= 1
        region_end -= 1

        i = np.arange(
            max(0, region_start, length - 3 - len(pr)),
            min(len(pl), length - 2, region_end - 2),
        )
        l += (pl[i] * pr[length - i - 3]).sum() * pu * 1 / 4
    return l


class CleavageEstimator:

    def __init__(
        self,
        repeats: int = 1_000,
        maxiter: int = 100,
        delta_cutoff: float = 0.001,
        seed: int = 42,
    ) -> None:

        self.table = np.zeros(shape=(100, 3, 2, 1), dtype=np.int32)
        self.obs_min_len = 15
        self.obs_max_len = 40
        self.seed = seed
        self.repeats = repeats
        self.c = 0
        self.maxiter = maxiter
        self.delta_cutoff = delta_cutoff

    def collect_data(
        self,
        reference_annotation: ReferenceAnnotation,
        alignments: HTSeq.BAM_Reader,
        min_considered_length: int = 15,
        max_considered_length: int = 40,
        min_dist_to_start: int = 30,
        min_dist_to_end: int = 30,
        sufficient_counted_alns: int = 100_000,
    ) -> None:
        self.table = np.zeros(shape=(self.obs_max_len + 10, 3, 2, 1), dtype=np.int32)
        self.dist_starts = np.zeros(shape=(200), dtype=np.int32)
        self.outside_cds = 0
        self.not_countable = 0
        self.counted_alns = 0
        for aln in alignments:
            aln = RiboSeqAlignment(aln)

            if not aln.unique():
                self.not_countable += 1
                continue
            # if aln.close_to_any_tis(reference_annotation):
            #    self.not_countable += 1
            #    continue
            if not min_considered_length <= len(aln) < max_considered_length:
                self.not_countable += 1
                continue
            transcript_candidates = reference_annotation.collect_coding_transcripts(
                aln.genomic_region
            )
            if len(transcript_candidates) == 0:
                self.outside_cds += 1
                continue
            frame = None
            dist_to_start = None

            # get frame
            for tr in transcript_candidates:
                try:
                    iv_on_cds = tr.cds.induce(aln.genomic_region)
                except ValueError:
                    self.not_countable += 1
                    continue

                if (
                    iv_on_cds[0] > min_dist_to_start
                    and tr.coding_length - iv_on_cds[1] > min_dist_to_end
                ):
                    new_frame = iv_on_cds[0] % 3

                    if frame is None:
                        frame = new_frame
                    elif frame != new_frame:
                        self.not_countable += 1
                        break

            else:
                if not frame is None:
                    self.table[len(aln), frame, int(aln.untemplated_addition), 0] += 1
                    self.counted_alns += 1
                    if self.counted_alns >= sufficient_counted_alns:
                        break

            # get dist_to_start
            for tr in transcript_candidates:
                try:
                    new_dist_to_start = tr.cds.induce(aln.genomic_region)[0]
                except ValueError:
                    new_dist_to_start = None

                if type(new_dist_to_start) == int:
                    if dist_to_start is None:
                        dist_to_start = new_dist_to_start
                    elif dist_to_start != new_dist_to_start:
                        break
            else:
                if (type(dist_to_start) == int) and (-100 < dist_to_start < 100):
                    self.dist_starts[dist_to_start + 100] += 1

    def correct_table(self) -> None:
        temp_table = self.table.copy()
        (self.table[:, 0, :, :], self.table[:, 1, :, :], self.table[:, 2, :, :]) = (
            temp_table[:, 0, :, :],
            temp_table[:, 2, :, :],
            temp_table[:, 1, :, :],
        )

    def run(self) -> CleavageModel:

        self.best_ll, self.best_u, self.best_pl, self.best_pr = repeat(
            self.repeats,
            self.obs_max_len,
            self.obs_min_len,
            self.table,
            self.maxiter,
            self.c,
            self.delta_cutoff,
            self.seed,
        )
        shift = self.compute_shift()
        self.correct_max_pos(shift)
        self.regularize()
        return CleavageModel(self.best_pl, self.best_pr, self.best_u)

    def regularize(self, keep_prob: float = 0.9):
        pl = self.best_pl.copy()
        self.best_pl = select_and_scale(pl, keep_prob)
        pr = self.best_pr.copy()
        self.best_pr = select_and_scale(pr, keep_prob)

    def compute_shift(self, range_start: int = -25, range_end: int = -5) -> int:
        pl_rev = self.best_pl[::-1]
        pl_maxpos = pl_rev.argmax()

        overlayed = np.zeros(range_end - range_start)
        for i in range(range_start, range_end):
            overlayed[i - range_start] = sum(
                pl_rev
                * self.dist_starts[
                    100 + i - pl_maxpos : 100 + i - pl_maxpos + len(pl_rev)
                ]
            )

        overlayed_diff = overlayed[3:] - overlayed[:-3]
        return -(overlayed_diff.argmax() - pl_maxpos + range_start + len(pl_rev) + 2)

    def correct_max_pos(self, shift: int) -> None:
        pl = np.zeros(len(self.best_pl))
        for i in range(len(self.best_pl)):
            if i - shift >= 0 and i - shift < len(self.best_pl):
                pl[i] = self.best_pl[i - shift]

        pr = np.zeros(len(self.best_pr))
        for i in range(len(self.best_pr)):
            if i + shift >= 0 and i + shift < len(self.best_pl):
                pr[i] = self.best_pr[i + shift]

        pl = pl / pl.sum()
        pr = pr / pr.sum()

        self.best_pl = pl
        self.best_pr = pr


@njit
def compute_ll(
    table,
    obs_min_len,
    obs_max_len,
    pl,
    pr,
    u,
    c,
) -> float:
    ll = 0
    for length in range(obs_min_len, obs_max_len + 1):
        for frame in range(3):
            frame1 = (frame - 1) % 3

            untemplated_addition = 1
            n = table[length, frame, untemplated_addition, c]

            if n > 0:
                i = np.arange(frame1, min(len(pl), length - 3), 3)
                p = (pl[i] * pr[length - i - 3 - 1]).sum()
                ll += n * np.log(p)

            untemplated_addition = 0
            n = table[length, frame, untemplated_addition, c]

            if n > 0:
                i = np.arange(frame1, min(len(pl), length - 3), 3)
                p = (pl[i] * pr[length - i - 3 - 1]).sum() * u
                i = np.arange(frame, min(len(pl), length - 2), 3)
                p += (pl[i] * pr[length - i - 3]).sum() * (1 - u)
                ll += n * np.log(p)
    return ll


@njit
def repeat(
    repeats,
    obs_max_len,
    obs_min_len,
    table,
    maxiter,
    c,
    delta_cutoff,
    seed=42,
):
    np.random.seed(seed)

    total = table[obs_min_len : obs_max_len + 1, :, :, c].sum()

    best_ll = -np.inf

    for rep in range(max(repeats, 1)):

        pl = np.random.rand(obs_max_len + 1)
        pr = np.random.rand(obs_max_len + 1)

        # maxpos = 12
        # pl[maxpos-1]*=4
        # pl[maxpos]*=10
        # pl[maxpos+1]*=4

        pl = pl / pl.sum()
        pr = pr / pr.sum()

        N = table[obs_min_len : obs_max_len + 1, :, 1, c].sum()
        u = N * 4 / 3

        N += table[obs_min_len : obs_max_len + 1, :, 0, c].sum()
        u /= N

        better_ll = -np.inf

        for it in range(maxiter):
            eps = 1e-14

            ql0 = np.zeros(obs_max_len + 1)
            qr0 = np.zeros(obs_max_len + 1)
            ql1 = np.zeros(obs_max_len + 1)
            qr1 = np.zeros(obs_max_len + 1)

            qu = 0

            for length in range(obs_min_len, obs_max_len + 1):
                for frame in range(3):
                    untemplated_addition = 1

                    n = table[length, frame, untemplated_addition, c]

                    frame1 = (frame - 1) % 3
                    left = np.arange(frame, length - 3, 3)
                    left1 = np.arange(frame1, length - 3, 3)
                    sum = eps

                    sum += (pl[left1] * pr[length - left1 - 3 - 1]).sum()
                    s = pl[left1] * pr[length - left1 - 3 - 1] / sum * n
                    ql1[left1 + 1] += s
                    qr1[length - left1 - 1 - 3] += s

                    qu += n

                    untemplated_addition = 0

                    n = table[length, frame, untemplated_addition, c]

                    sum0 = eps
                    sum1 = eps

                    prop = u / (4 - 3 * u)

                    sum1 += (pl[left1] * pr[length - left1 - 3 - 1] * prop).sum()

                    sum0 += (pl[left] * pr[length - left - 3] * (1 - prop)).sum()
                    sum = sum1 + sum0

                    s = pl[left1] * pr[length - left1 - 3 - 1] * prop / sum * n
                    ql1[left1] += s
                    qr1[length - left1 - 1 - 3] += s

                    s = pl[left] * pr[length - left - 3] * (1 - prop) / sum * n
                    ql0[left] += s
                    qr0[length - left - 3] += s

                    qu += sum1 / sum * n

            old_pl = pl.copy()
            old_pr = pr.copy()
            for i in range(obs_max_len + 1):
                pl[i] = ((ql1[i + 1] if i + 1 < len(ql1) else 0) + ql0[i]) / total
                pr[i] = (qr1[i] + qr0[i]) / total

            N = table[obs_min_len : obs_max_len + 1, :, :, c].sum()
            u = qu / N

            model_change = (np.absolute(old_pl - pl)).sum() + (
                np.absolute(old_pr - pr)
            ).sum()
            if model_change < delta_cutoff:
                better_ll = compute_ll(table, obs_min_len, obs_max_len, pl, pr, u, c)
                better_u = u
                break

        else:
            better_ll = compute_ll(table, obs_min_len, obs_max_len, pl, pr, u, c)
            better_u = u

        if better_ll > best_ll:
            best_ll = better_ll
            best_u = better_u
            best_pl = pl
            best_pr = pr

    return best_ll, best_u, best_pl, best_pr


def to_file(file_path: str, d: dict[str, CleavageModel]) -> None:
    # convert dict to pickleable list
    l = []
    for k, v in d.items():
        l.append((k, v.pl, v.pr, v.pu))
    with open(file_path, "wb") as f:
        pickle.dump(l, f)


def from_file(file_path: str) -> dict[str, CleavageModel]:
    with open(file_path, "rb") as f:
        l = pickle.load(f)
    d = {}
    for k, pl, pr, pu in l:
        d[k] = CleavageModel(pl, pr, pu)
    return d


def cleavage_models_from_bams(
    bam_dir, wdir, ref_annotation, cm_pickle: str
) -> dict[str, CleavageModel]:
    os.makedirs(f"{wdir}/sample_bam", exist_ok=True)
    if not os.path.exists(cm_pickle):
        d = {}
    else:
        d = from_file(cm_pickle)
    for bam_file in os.listdir(bam_dir):
        if bam_file.endswith(".bam"):
            id = bam_file.split(".")[0]
            if id in d:
                continue
            bam_file_path = f"{bam_dir}/{bam_file}"
            read_count = pysam.AlignmentFile(bam_file_path, "rb").count()
            sample_bam_file = f"{wdir}/sample_bam/{bam_file}"
            open(sample_bam_file, "w").close()
            fraction_of_reads = 100_000 / read_count
            pysam.view(
                "-s",
                str(fraction_of_reads),
                "-o",
                sample_bam_file,
                bam_file_path,
                save_stdout=sample_bam_file,
            )

            ce = CleavageEstimator()

            ce.collect_data(ref_annotation, HTSeq.BAM_Reader(sample_bam_file))
            ce.correct_table()
            cleavage_model = ce.run()
            d[id] = cleavage_model
            os.remove(sample_bam_file)

    to_file(cm_pickle, d)
    os.rmdir(f"{wdir}/sample_bam")
    return d


def select_and_scale(arr, k):
    # Step 1: Sort the array in descending order while keeping track of original indices
    sorted_indices = np.argsort(arr)[::-1]
    sorted_arr = arr[sorted_indices]

    # Step 2: Select elements until their sum is at least k
    cumulative_sum = 0
    selected_indices = []
    for i, elem in enumerate(sorted_arr):
        if cumulative_sum >= k:
            break
        selected_indices.append(sorted_indices[i])
        cumulative_sum += elem

    # Step 3: Reconstruct the array with zeros and place selected elements in their original positions
    result = np.zeros_like(arr)
    for idx in selected_indices:
        result[idx] = arr[idx]
    # Step 4: Normalize the selected elements
    selected_sum = result.sum()
    if selected_sum > 0:
        result = result / selected_sum

    return result
