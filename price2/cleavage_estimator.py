"""Estimating a cleavage model from a Ribo-seq BAM.

:class:`CleavageEstimator` tallies, over the reads that map uniquely into
annotated CDSs, how the read ends sit relative to the codon boundaries, and
fits the left/right cleavage distributions and the untemplated-addition
probability of a :class:`~price2.cleavage_model.CleavageModel` to that table
by EM with random restarts (the numba kernels below).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pysam
from numba import njit, prange

from price2.bam import iter_mapped
from price2.cleavage_model import (
    MIN_PEAK_PROBABILITY,
    PLAUSIBLE_P_SITE_OFFSETS,
    CleavageModel,
)
from price2.reference_annotation import ReferenceAnnotation
from price2.ribo_seq_alignment import RiboSeqAlignment

logger = logging.getLogger(__name__)

#: Minimum number of counted alignments for a reliable cleavage model.
MIN_COUNTED_ALNS: int = 100_000


class CleavageEstimator:
    """EM-based estimator for cleavage model parameters.

    Learns left/right cleavage distributions and the
    untemplated-addition probability from reads that map
    unambiguously to annotated CDS regions.

    The defaults were chosen by tracing 200 restarts to full convergence on
    three count tables (a 100k-read sample, a 37M-read genome-wide table, and a
    deliberately ill-conditioned synthetic one).  The EM needs a few thousand
    iterations, not a hundred: with ``delta_cutoff=1e-3`` it stops after ~80-120
    iterations and *no* restart gets within 1 nat of the best attainable
    likelihood, so the old ``repeats=1000`` merely sampled 1000 barely-started
    runs.  Once each restart is run to convergence, ~50 restarts suffice for 99%
    confidence of reaching the best basin, and the peak-informed initialisation
    (see :func:`repeat`) roughly halves that again.

    Parameters
    ----------
    repeats : int, optional
        Number of random restarts for the EM algorithm.
    maxiter : int, optional
        Maximum EM iterations per restart.
    delta_cutoff : float, optional
        Convergence threshold on parameter change (L1, on ``pl`` and ``pr``).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        repeats: int = 100,
        maxiter: int = 10_000,
        delta_cutoff: float = 1e-8,
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
        sample_bam_path: str,
        min_considered_length: int = 15,
        max_considered_length: int = 40,
        min_dist_to_start: int = 30,
        min_dist_to_end: int = 30,
        min_counted_alns: int = MIN_COUNTED_ALNS,
        end_to_end: bool = False,
    ) -> None:
        """Collect read-length / frame / UTA counts from a BAM file.

        Iterates over *every* uniquely-mapped read that falls within a CDS
        region and tallies its length, reading frame and untemplated-addition
        status into ``self.table``, plus its read-start-to-CDS-start distance
        into ``self.dist_starts``.  The whole file is scanned to convergence on
        the true distributions: sampling only a genomic prefix of a
        coordinate-sorted BAM would bias the counts towards the reads at the
        start of the file, so any downsampling must happen before this call.

        Parameters
        ----------
        reference_annotation : ReferenceAnnotation
            Parsed reference annotation.
        sample_bam_path : str
            Path to the BAM file.
        min_considered_length : int, optional
            Minimum read length to consider.
        max_considered_length : int, optional
            Maximum read length to consider.
        min_dist_to_start : int, optional
            Minimum distance from CDS start to count a read.
        min_dist_to_end : int, optional
            Minimum distance from CDS end to count a read.
        min_counted_alns : int, optional
            Warn when fewer than this many alignments are tallied into
            ``table``; below it the dataset is too small for a reliable fit.
        end_to_end : bool, optional
            When ``True`` the BAM was mapped with ``--alignEndsType EndToEnd``;
            the untemplated addition is recovered from the 5'-terminal mismatch
            instead of a soft-clip (see :func:`price2.bam.footprint`).
        """
        self.table = np.zeros(shape=(self.obs_max_len + 10, 3, 2, 1), dtype=np.int32)
        self.dist_starts = np.zeros(shape=(200,), dtype=np.int32)
        self.outside_cds = 0
        self.not_unique = 0
        self.not_countable = 0
        self.bad_length = 0
        self.counted_alns = 0
        with pysam.AlignmentFile(sample_bam_path, "rb") as bam:
            for raw_aln in iter_mapped(bam):
                aln = RiboSeqAlignment.from_pysam(raw_aln, end_to_end=end_to_end)
                if aln is None:
                    continue
                if not aln.unique:
                    self.not_unique += 1
                    continue
                if not min_considered_length <= len(aln) < max_considered_length:
                    self.bad_length += 1
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
                    if tr.annotated_cds_iv is None:
                        continue
                    try:
                        iv_on_tr = tr.exons.map_to_local(aln.genomic_region)
                        iv_on_cds = (
                            iv_on_tr[0] - tr.annotated_cds_iv[0],
                            iv_on_tr[1] - tr.annotated_cds_iv[0],
                        )
                    except ValueError:
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
                    if frame is not None:
                        self.table[
                            len(aln),
                            frame,
                            int(aln.untemplated_addition),
                            0,
                        ] += 1
                        self.counted_alns += 1

                # get dist_to_start
                for tr in transcript_candidates:
                    if not tr.annotated_cds_iv:
                        continue
                    try:
                        new_dist_to_exon_start = tr.exons.map_to_local(
                            aln.genomic_region
                        )[0]
                        new_dist_to_cds_start = (
                            new_dist_to_exon_start - tr.annotated_cds_iv[0]
                        )
                    except ValueError:
                        new_dist_to_cds_start = None

                    new_dist_to_start = new_dist_to_cds_start

                    if isinstance(new_dist_to_start, int):
                        if dist_to_start is None:
                            dist_to_start = new_dist_to_start
                        elif dist_to_start != new_dist_to_start:
                            break
                else:
                    if isinstance(dist_to_start, int) and (-100 < dist_to_start < 100):
                        self.dist_starts[dist_to_start + 100] += 1

        if self.counted_alns < min_counted_alns:
            logger.warning(
                "Not enough alignments counted: %d < %d for %s\n"
                "  not_unique: %d\n"
                "  not_countable: %d\n"
                "  outside_cds: %d\n"
                "  bad_length: %d",
                self.counted_alns,
                min_counted_alns,
                sample_bam_path,
                self.not_unique,
                self.not_countable,
                self.outside_cds,
                self.bad_length,
            )

    def correct_table(self) -> None:
        """Swap frame-1 and frame-2 columns in the count table."""
        temp_table = self.table.copy()
        (
            self.table[:, 0, :, :],
            self.table[:, 1, :, :],
            self.table[:, 2, :, :],
        ) = (
            temp_table[:, 0, :, :],
            temp_table[:, 2, :, :],
            temp_table[:, 1, :, :],
        )

    def run(self, regularize: bool = True) -> CleavageModel:
        """Run EM estimation and return the fitted cleavage model.

        Parameters
        ----------
        regularize : bool, optional
            Whether to apply regularisation (default True).

        Returns
        -------
        CleavageModel
            The fitted cleavage model.
        """
        self.best_ll, self.best_u, self.best_pl, self.best_pr = repeat(
            self.repeats,
            self.obs_max_len,
            self.obs_min_len,
            self.table,
            self.maxiter,
            self.c,
            self.delta_cutoff,
            self.seed,
            self.init_peak(),
        )
        shift = self.compute_shift()
        self.correct_max_pos(shift)
        if regularize:
            self.regularize()

        max_pos = int(np.argmax(self.best_pl))
        max_prob = float(self.best_pl[max_pos])
        if max_pos not in PLAUSIBLE_P_SITE_OFFSETS:
            logger.warning(
                "Unusual cleavage model: Upstream cleavage peak is at position %d, "
                "expected one of %s. ",
                max_pos,
                sorted(PLAUSIBLE_P_SITE_OFFSETS),
            )
        if max_prob < MIN_PEAK_PROBABILITY:
            logger.warning(
                "Low quality dataset: Upstream cleavage peak probability is %.3f",
                max_prob,
            )

        dist_starts = getattr(self, "dist_starts", None)
        table = getattr(self, "table", None)
        return CleavageModel(
            self.best_pl,
            self.best_pr,
            self.best_u,
            dist_starts=dist_starts.copy() if dist_starts is not None else None,
            table=table.copy() if table is not None else None,
        )

    def regularize(self, keep_prob: float = 0.9) -> None:
        """Zero out low-probability entries and re-normalise.

        Parameters
        ----------
        keep_prob : float, optional
            Cumulative probability mass to retain (default 0.9).
        """
        self.best_pl = select_and_scale(self.best_pl.copy(), keep_prob)
        self.best_pr = select_and_scale(self.best_pr.copy(), keep_prob)

    def _height(self, offset: int) -> float:
        """Read-start count at ``offset`` nt upstream of the CDS start.

        ``dist_starts`` index 100 holds a read start sitting on the CDS start,
        so an offset of ``o`` upstream sits at index ``100 - o``.
        """
        dist_starts = self.dist_starts
        idx = 100 - offset
        return float(dist_starts[idx]) if 0 <= idx < len(dist_starts) else 0.0

    def _reading_frame(self, min_offset: int = 6, max_offset: int = 25) -> int:
        """Reading frame (``offset % 3``) of the P-site relative to the CDS.

        The read-start metagene around the CDS start often carries a second comb
        of peaks one nt away from the true one -- the untemplated-addition
        shadow, whose reads map one base off after their extra 5' base is soft
        clipped (or, under ``EndToEnd``, matched).  That shadow can be as tall as
        or taller than the real comb (e.g. SRR13202602), so the frame cannot be
        read off the single tallest bar.  Cross-correlating the *whole* fitted
        ``pl`` against the start-region histogram instead integrates over the
        comb, and the frame whose correlation is largest is the true one.

        Parameters
        ----------
        min_offset, max_offset : int, optional
            Inclusive window of read-start-to-CDS-start distances to score.

        Returns
        -------
        int
            The P-site reading frame (0, 1 or 2).
        """
        pl = self.best_pl
        peak = int(np.argmax(pl))
        k = np.arange(len(pl))

        def correlation(offset: int) -> float:
            # Place ``pl`` with its peak at ``offset`` and correlate it with the
            # metagene, counting only positions inside the start-region window.
            idx = 100 - (offset - peak + k)
            inside = (100 - max_offset <= idx) & (idx <= 100 - min_offset)
            return float(np.dot(pl[inside], self.dist_starts[idx[inside]]))

        offsets = range(min_offset, max_offset + 1)
        frame_score = {
            f: max((correlation(o) for o in offsets if o % 3 == f), default=0.0)
            for f in range(3)
        }
        return max(frame_score, key=frame_score.get)

    def _onset_offset(
        self,
        frame: Optional[int] = None,
        min_offset: int = 6,
        max_offset: int = 25,
        default: int = 12,
    ) -> int:
        """Read-start-to-CDS-start distance at the translation onset.

        Within a reading frame the metagene is a 3-nt-periodic comb of peaks --
        the start codon and every downstream in-frame codon -- of similar
        height, so a plain ``argmax`` often lands on a downstream codon and
        reports an offset that is 3, 6, ... nt too small (e.g. 10 instead of
        13).  The start codon is the *onset* of that comb: the in-frame position
        whose count jumps up the most over its next-upstream (``offset + 3``)
        neighbour, which still lies in the 5' UTR and is near-empty.  Selecting
        on that jump rather than on the raw height recovers the true offset even
        when two in-frame peaks are nearly tied.

        Parameters
        ----------
        frame : int or None, optional
            Reading frame to search.  When ``None`` the frame is taken from the
            tallest peak (used only to seed the EM, before ``pl`` is fitted);
            :meth:`compute_shift` passes the frame from :meth:`_reading_frame`.
        min_offset, max_offset : int, optional
            Inclusive range of read-start-to-CDS-start distances to search.
        default : int, optional
            Returned when no start-distance histogram was collected, or it is
            empty across the search range (the canonical P-site offset).

        Returns
        -------
        int
            Offset from the read start to the P-site at the translation onset.
        """
        dist_starts = getattr(self, "dist_starts", None)
        if dist_starts is None:
            return default

        offsets = range(min_offset, max_offset + 1)
        if sum(self._height(o) for o in offsets) == 0:
            return default

        if frame is None:
            frame = max(offsets, key=self._height) % 3
        comb = [o for o in offsets if o % 3 == frame % 3]
        return max(comb, key=lambda o: self._height(o) - self._height(o + 3))

    def init_peak(self, default: int = 12) -> int:
        """Expected position of the ``pl`` peak, for initialising the EM.

        Delegates to :meth:`_onset_offset`; the most frequent distance from a
        read start to the CDS start is, up to sign, the most likely left
        cleavage.  ``pl`` is not fitted yet, so the frame is taken from the
        tallest peak -- good enough for a starting point, since
        :meth:`compute_shift` re-anchors the final model.  Falls back to
        *default* when no start-distance histogram was collected.

        Returns
        -------
        int
            Offset from the read start to the P-site at the ``pl`` peak.
        """
        return self._onset_offset(default=default)

    def compute_shift(self) -> int:
        """Shift that anchors the fitted ``pl`` peak to the P-site offset.

        :func:`repeat` fixes the *shape* of ``pl``/``pr`` but leaves their
        absolute position free: shifting ``pl`` one codon right and ``pr`` one
        codon left leaves every footprint likelihood unchanged.  This resolves
        that gauge freedom by moving the ``pl`` peak onto the onset offset --
        the reading frame from :meth:`_reading_frame` (robust to the
        untemplated-addition shadow) combined with the in-frame onset from
        :meth:`_onset_offset` (robust to 3-nt periodicity) -- so the model's
        P-site matches the observed read-start-to-CDS-start distance.

        Returns
        -------
        int
            Shift to pass to :meth:`correct_max_pos`.
        """
        dist_starts = getattr(self, "dist_starts", None)
        if dist_starts is None:
            return 0
        onset = self._onset_offset(frame=self._reading_frame())
        return onset - int(np.argmax(self.best_pl))

    def correct_max_pos(self, shift: int) -> None:
        """Shift pl and pr arrays and re-normalise.

        Parameters
        ----------
        shift : int
            Number of positions to shift.
        """
        n = len(self.best_pl)
        pl = np.zeros(n)
        for i in range(n):
            if 0 <= i - shift < n:
                pl[i] = self.best_pl[i - shift]

        pr = np.zeros(len(self.best_pr))
        for i in range(len(self.best_pr)):
            if 0 <= i + shift < n:
                pr[i] = self.best_pr[i + shift]

        self.best_pl = pl / pl.sum()
        self.best_pr = pr / pr.sum()


@njit(cache=True)
def compute_ll(
    table: np.ndarray,
    obs_min_len: int,
    obs_max_len: int,
    pl: np.ndarray,
    pr: np.ndarray,
    u: float,
    c: int,
) -> float:
    """Compute the log-likelihood of the observed count table.

    Parameters
    ----------
    table : np.ndarray
        Observed counts, shape ``(max_len, 3 (frame), 2 (untemplated_addition), n_conditions)``.
    obs_min_len, obs_max_len : int
        Range of observed read lengths.
    pl, pr : np.ndarray
        Left / right cleavage distributions.
    u : float
        Untemplated-addition probability.
    c : int
        Condition index into the table's last axis.

    Returns
    -------
    float
        Log-likelihood.
    """
    ll = 0
    for length in range(obs_min_len, obs_max_len + 1):
        for frame in range(3):
            frame1 = (frame - 1) % 3

            untemplated_addition = 1
            n = table[length, frame, untemplated_addition, c]

            if n > 0:
                # Visible soft-clip UTA -> plain footprint geometry (see repeat()).
                # An UTA is present and mismatches the reference: prob u * 3/4.
                i = np.arange(frame, min(len(pl), length - 2), 3)
                p = (pl[i] * pr[length - i - 3]).sum() * u * 3 / 4
                ll += n * np.log(p)

            untemplated_addition = 0
            n = table[length, frame, untemplated_addition, c]

            if n > 0:
                # Either an UTA that matches the reference (prob u * 1/4, the
                # footprint is one shorter and one frame over), or no UTA at all.
                i = np.arange(frame1, min(len(pl), length - 3), 3)
                p = (pl[i] * pr[length - i - 3 - 1]).sum() * u / 4
                i = np.arange(frame, min(len(pl), length - 2), 3)
                p += (pl[i] * pr[length - i - 3]).sum() * (1 - u)
                ll += n * np.log(p)
    return ll


@njit(cache=True)
def _init_restarts(
    repeats: int, obs_max_len: int, peak: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the initial ``pl``/``pr`` of every restart.

    Kept separate from the EM itself so that the random draws stay sequential
    and the restarts remain reproducible no matter how they are scheduled.

    Returns
    -------
    pls, prs : np.ndarray
        Arrays of shape ``(repeats, obs_max_len + 1)``, one normalised
        distribution per restart.
    """
    np.random.seed(seed)
    pls = np.empty((repeats, obs_max_len + 1))
    prs = np.empty((repeats, obs_max_len + 1))
    for rep in range(repeats):
        pl = np.random.rand(obs_max_len + 1)
        pr = np.random.rand(obs_max_len + 1)

        pl[peak - 1] *= 4
        pl[peak] *= 10
        pl[peak + 1] *= 4

        pls[rep] = pl / pl.sum()
        prs[rep] = pr / pr.sum()
    return pls, prs


@njit(cache=True)
def _em_restart(
    obs_max_len: int,
    obs_min_len: int,
    table: np.ndarray,
    maxiter: int,
    c: int,
    delta_cutoff: float,
    total: int,
    pl: np.ndarray,
    pr: np.ndarray,
) -> tuple[float, float]:
    """Run the EM to convergence from one starting point.

    *pl* and *pr* are updated in place and hold the fitted distributions on
    return.

    Returns
    -------
    tuple[float, float]
        ``(log_likelihood, u)`` of the fitted model.
    """
    N = table[obs_min_len : obs_max_len + 1, :, 1, c].sum()
    u = N * 4 / 3

    N += table[obs_min_len : obs_max_len + 1, :, 0, c].sum()
    u /= N

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
                # left indexes the footprint left cleavage: it may reach
                # length - 3, which pairs with a right cleavage of 0.
                left = np.arange(frame, length - 2, 3)
                # left1 belongs to the one-shorter hidden-UTA footprint, so
                # it stops one codon earlier.
                left1 = np.arange(frame1, length - 3, 3)

                total_p = eps + (pl[left] * pr[length - left - 3]).sum()
                s = pl[left] * pr[length - left - 3] / total_p * n
                ql0[left] += s
                qr0[length - left - 3] += s

                qu += n

                untemplated_addition = 0

                n = table[length, frame, untemplated_addition, c]

                sum0 = eps
                sum1 = eps

                prop = u / (4 - 3 * u)

                sum1 += (pl[left1] * pr[length - left1 - 3 - 1] * prop).sum()

                sum0 += (pl[left] * pr[length - left - 3] * (1 - prop)).sum()
                total_p = sum1 + sum0

                s = pl[left1] * pr[length - left1 - 3 - 1] * prop / total_p * n
                ql1[left1] += s
                qr1[length - left1 - 1 - 3] += s

                s = pl[left] * pr[length - left - 3] * (1 - prop) / total_p * n
                ql0[left] += s
                qr0[length - left - 3] += s

                qu += sum1 / total_p * n

        old_pl = pl.copy()
        old_pr = pr.copy()
        for i in range(obs_max_len + 1):
            # ql0 and ql1 are both indexed by the footprint left cleavage:
            # the E-step pairs ql1[left1] with pr[length - left1 - 4].
            pl[i] = (ql1[i] + ql0[i]) / total
            pr[i] = (qr1[i] + qr0[i]) / total

        N = table[obs_min_len : obs_max_len + 1, :, :, c].sum()
        u = qu / N

        model_change = np.absolute(old_pl - pl).sum() + np.absolute(old_pr - pr).sum()
        if model_change < delta_cutoff:
            break

    return compute_ll(table, obs_min_len, obs_max_len, pl, pr, u, c), u


@njit(parallel=True, cache=True)
def _em_restarts(
    obs_max_len: int,
    obs_min_len: int,
    table: np.ndarray,
    maxiter: int,
    c: int,
    delta_cutoff: float,
    total: int,
    pls: np.ndarray,
    prs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run every restart in *pls*/*prs*, in parallel, updating them in place.

    Restarts are independent, so scheduling cannot change any single result.

    Returns
    -------
    lls, us : np.ndarray
        Per-restart log-likelihood and untemplated-addition probability.
    """
    repeats = pls.shape[0]
    lls = np.empty(repeats)
    us = np.empty(repeats)
    for rep in prange(repeats):
        lls[rep], us[rep] = _em_restart(
            obs_max_len,
            obs_min_len,
            table,
            maxiter,
            c,
            delta_cutoff,
            total,
            pls[rep],
            prs[rep],
        )
    return lls, us


def repeat(
    repeats: int,
    obs_max_len: int,
    obs_min_len: int,
    table: np.ndarray,
    maxiter: int,
    c: int,
    delta_cutoff: float,
    seed: int = 42,
    init_peak: int = 12,
) -> tuple:
    """Run the EM algorithm with multiple random restarts.

    Each restart starts from a uniform random ``pl``/``pr`` whose ``pl`` is then
    tilted towards *init_peak*.  The likelihood surface is riddled with local
    optima (on a genome-wide table only ~2% of purely random restarts reach the
    best one), and this tilt raises that to ~9-50% while cutting the iterations
    to convergence by up to 10x.  It only biases the starting point; the data
    still decide where the peak ends up.

    The restarts are drawn up front and then fitted in parallel; the result does
    not depend on the number of threads.

    Parameters
    ----------
    repeats : int
        Number of random restarts.
    obs_max_len, obs_min_len : int
        Observed read length range.
    table : np.ndarray
        Count table.
    maxiter : int
        Maximum iterations per restart.
    c : int
        Condition index.
    delta_cutoff : float
        Convergence threshold.
    seed : int, optional
        Random seed.
    init_peak : int, optional
        Expected position of the ``pl`` peak, used to tilt the initialisation.
        :meth:`CleavageEstimator.init_peak` derives it from the observed
        read-start-to-CDS-start histogram; 12 is the canonical value.

    Returns
    -------
    tuple
        ``(best_ll, best_u, best_pl, best_pr)``.
    """
    total = table[obs_min_len : obs_max_len + 1, :, :, c].sum()
    peak = min(max(init_peak, 1), obs_max_len - 1)

    pls, prs = _init_restarts(max(repeats, 1), obs_max_len, peak, seed)
    lls, us = _em_restarts(
        obs_max_len, obs_min_len, table, maxiter, c, delta_cutoff, total, pls, prs
    )

    best = int(np.argmax(lls))
    return lls[best], us[best], pls[best].copy(), prs[best].copy()


def select_and_scale(arr: np.ndarray, keep_prob: float) -> np.ndarray:
    """Keep the largest elements up to *keep_prob* mass, zero the rest.

    Elements are selected in descending order until their cumulative
    sum reaches *keep_prob*, then the result is re-normalised.

    Parameters
    ----------
    arr : np.ndarray
        Input probability distribution.
    keep_prob : float
        Cumulative probability mass to retain.

    Returns
    -------
    np.ndarray
        Filtered and re-normalised distribution.
    """
    sorted_indices = np.argsort(arr)[::-1]
    sorted_arr = arr[sorted_indices]

    cumulative_sum = 0.0
    selected_indices: list[int] = []
    for i, elem in enumerate(sorted_arr):
        if cumulative_sum >= keep_prob:
            break
        selected_indices.append(sorted_indices[i])
        cumulative_sum += elem

    result = np.zeros_like(arr)
    for idx in selected_indices:
        result[idx] = arr[idx]

    total = result.sum()
    if total > 0:
        result /= total
    return result
