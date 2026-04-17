"""Cleavage site estimation for Ribo-seq data.

Provides the cleavage model (probability distributions for left and right
cleavage positions relative to the P-site) and an EM-based estimator that
learns the model parameters from mapped Ribo-seq reads.
"""

import logging
import pickle
from functools import lru_cache
from typing import Optional

import pysam
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from numba import njit

from price2.reference_annotation import ReferenceAnnotation
from price2.ribo_seq_alignment import RiboSeqAlignment

logger = logging.getLogger(__name__)

# Minimum number of alignments required for reliable cleavage model estimation.
_MIN_COUNTED_ALNS: int = 100_000


class CleavageModel:
    """Ribosome cleavage model for Ribo-seq reads.

    Models the probability of left (upstream) and right (downstream)
    cleavage positions relative to the P-site, plus the probability
    of an untemplated addition (UTA).

    Parameters
    ----------
    pl : np.ndarray
        Left cleavage probability distribution, shape (n_left,).
    pr : np.ndarray
        Right cleavage probability distribution, shape (n_right,).
    pu : float
        Probability of an untemplated addition.
    """

    def __init__(
        self,
        pl: np.ndarray,
        pr: np.ndarray,
        pu: float,
        dist_starts: Optional[np.ndarray] = None,
        table: Optional[np.ndarray] = None,
    ) -> None:
        self.pl = pl
        self.pr = pr
        self.pu = pu
        if dist_starts is not None:
            self.dist_starts = dist_starts
        if table is not None:
            self.table = table
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
        self.fill_dist_to_orf_start()
        self.fill_dist_to_orf_end()

    def pmf(
        self,
        length: int,
        oua: bool,
        frame: Optional[int] = None,
        region_start: int = 0,
        region_end: int = 10**10,
    ) -> float:
        """Compute the probability of observing a read.

        Parameters
        ----------
        length : int
            Matching length of the alignment.
        oua : bool
            Whether the read has an untemplated addition.
        frame : int or None, optional
            Reading frame (0, 1, 2) for CDS reads, or None for
            noise reads.
        region_start : int, optional
            Start of the region relative to the read start.
        region_end : int, optional
            End of the region relative to the read start.

        Returns
        -------
        float
            Probability of the read under the model.
        """
        # region_start relative to read start
        # region_end relative to read start
        # length is the matching length of the alignment
        if length >= len(self.pl) + len(self.pr) + 3 + int(oua):
            return 0

        if frame is None:  # noise
            if region_start == 0 and region_end == 10**10:
                return self.noise_lut[length, int(oua)]
            return read_in_noise_likelihood(
                self.pl,
                self.pr,
                self.pu,
                length,
                oua,
                region_start,
                region_end,
            )
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

    def rvs(self, size: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw random cleavage samples from the model.

        Parameters
        ----------
        size : int, optional
            Number of samples to draw (default 1).

        Returns
        -------
        tuple of np.ndarray
            ``(left_positions, right_positions, uta_flags)``.
        """
        pl = np.random.choice(np.arange(len(self.pl)), size=size, p=self.pl)
        pr = np.random.choice(np.arange(len(self.pr)), size=size, p=self.pr)
        u = np.random.choice([True, False], size=size, p=[self.pu, 1 - self.pu])
        return (pl, pr, u)

    def get_high_prob_indices(self, prob_sum: float = 0.3) -> list[tuple[int, ...]]:
        """Return CDS LUT indices covering the highest-probability entries.

        Greedily selects entries from ``cds_lut`` until their cumulative
        probability reaches *prob_sum*.

        Parameters
        ----------
        prob_sum : float, optional
            Cumulative probability threshold (default 0.3).

        Returns
        -------
        list of tuple
            Indices ``(length, frame, oua)`` of the selected entries.
        """
        lut = self.cds_lut.copy()
        cumulative = 0.0
        max_prob_positions: list[tuple[int, ...]] = []
        while cumulative < prob_sum:
            index = np.unravel_index(lut.argmax(), lut.shape)
            cumulative += lut[index]
            max_prob_positions.append(index)
            lut[index] = 0
        return max_prob_positions

    def fill_dist_to_orf_start(
        self, overlap_likelihood_ratio_thresh: float = 0.2
    ) -> None:
        """Pre-compute minimum distance from read start to ORF start.

        For every valid ``(read_length, oua, frame)`` combination,
        determine the farthest upstream position where the
        likelihood ratio still exceeds *overlap_likelihood_ratio_thresh*.

        Parameters
        ----------
        overlap_likelihood_ratio_thresh : float, optional
            Likelihood ratio threshold (default 0.2).
        """
        self.dist_to_orf_start: dict[tuple[int, bool, Optional[int]], int] = {}
        for read_length in self.non_zero_lengths:
            for oua in [True, False]:
                for frame in [None, 0, 1, 2]:
                    cl = self.pmf(read_length, oua, frame)
                    if cl == 0:
                        continue
                    if isinstance(frame, int):
                        positions = np.arange(-frame % 3, read_length, 3)
                    else:
                        positions = np.arange(read_length)
                    likelihoods = np.empty(positions.shape)
                    for i, pos in enumerate(positions):
                        ol = self.pmf(
                            read_length,
                            oua,
                            frame,
                            region_start=pos,
                        )
                        likelihoods[i] = ol / cl
                    try:
                        position = -positions[
                            likelihoods > overlap_likelihood_ratio_thresh
                        ].max()
                        self.dist_to_orf_start[(read_length, oua, frame)] = position
                    except ValueError:
                        pass

    def get_dist_to_orf_start(
        self,
        read_length: int,
        oua: bool,
        frame: Optional[int],
    ) -> int:
        """Return minimum distance from read start to ORF start.

        Lazily initialises the lookup via
        :meth:`fill_dist_to_orf_start` on first access.
        """
        try:
            return self.dist_to_orf_start[(read_length, oua, frame)]
        except AttributeError:
            self.fill_dist_to_orf_start()
            return self.dist_to_orf_start[(read_length, oua, frame)]

    def fill_dist_to_orf_end(
        self, overlap_likelihood_ratio_thresh: float = 0.2
    ) -> None:
        """Pre-compute minimum distance from read start to ORF end.

        For every valid ``(read_length, oua, frame)`` combination,
        determine the farthest downstream position where the
        likelihood ratio still exceeds *overlap_likelihood_ratio_thresh*.

        Parameters
        ----------
        overlap_likelihood_ratio_thresh : float, optional
            Likelihood ratio threshold (default 0.2).
        """
        self.dist_to_orf_end: dict[tuple[int, bool, Optional[int]], int] = {}
        for read_length in self.non_zero_lengths:
            for oua in [True, False]:
                for frame in [None, 0, 1, 2]:
                    cl = self.pmf(read_length, oua, frame)
                    if cl == 0:
                        continue
                    if isinstance(frame, int):
                        positions = np.arange(-frame % 3, read_length, 3)
                    else:
                        positions = np.arange(read_length)
                    likelihoods = np.empty(positions.shape)
                    for i, pos in enumerate(positions):
                        if frame is None:
                            ol = self.pmf(
                                read_length,
                                oua,
                                frame,
                                region_end=pos + 1,
                            )
                        else:
                            ol = self.pmf(
                                read_length,
                                oua,
                                frame,
                                region_end=pos + 3,
                            )
                        likelihoods[i] = ol / cl
                    try:
                        position = -positions[
                            likelihoods > overlap_likelihood_ratio_thresh
                        ].min()
                        self.dist_to_orf_end[(read_length, oua, frame)] = position
                    except ValueError:
                        pass

    def get_dist_to_orf_end(
        self,
        read_length: int,
        oua: bool,
        frame: Optional[int],
    ) -> int:
        """Return minimum distance from read start to ORF end.

        Lazily initialises the lookup via
        :meth:`fill_dist_to_orf_end` on first access.
        """
        try:
            return self.dist_to_orf_end[(read_length, oua, frame)]
        except AttributeError:
            self.fill_dist_to_orf_end()
            return self.dist_to_orf_end[(read_length, oua, frame)]

    def plot(self, ax: Optional[plt.Axes] = None) -> None:
        """Plot the cleavage model.

        Shows left/right cleavage distributions as bar charts and
        the untemplated-addition probability as a red bar.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axes to draw on. A new figure is created when *None*.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.bar(range(-len(self.pl) + 1, 1), self.pl[::-1])
        ax.bar(range(len(self.pr)), self.pr)
        ax.set_xlim(-30, 25)
        ax.set_ylim(0, 1)
        fill = Rectangle(
            (-5, 0.9),
            self.pu * 25,
            0.05,
            fill=True,
            facecolor="tab:red",
        )
        border = Rectangle(
            (-5, 0.9),
            25,
            0.05,
            fill=False,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(fill)
        ax.add_patch(border)
        ax.text(
            0.6,
            0.85,
            f"UTA prob = {self.pu:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax.set_xlabel("position relative to P-site")
        ax.set_ylabel("cleavage probability")

    def plot_full(self, fig: Optional[plt.Figure] = None) -> plt.Figure:
        """Plot a 3-panel diagnostic figure for the cleavage model.

        Panel 1: left/right cleavage distributions and UTA probability.
        Panel 2: P-site read-start distance to CDS start histogram
                 (requires ``dist_starts`` attribute).
        Panel 3: read-length / reading-frame count table
                 (requires ``table`` attribute).

        Parameters
        ----------
        fig : matplotlib.figure.Figure or None, optional
            Figure to draw on.  A new 3-axes figure is created when *None*.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the three panels.
        """
        if fig is None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        else:
            axes = fig.axes

        ax0, ax1, ax2 = axes[0], axes[1], axes[2]

        # Panel 1: read length / frame distribution
        lo, hi = 20, min(40, self.table.shape[0])
        x = np.arange(lo, hi)
        bar_w = 0.25
        ax0.bar(x - bar_w, self.table[lo:hi, 0, 0, 0], bar_w, label="frame 0")
        ax0.bar(x, self.table[lo:hi, 1, 0, 0], bar_w, label="frame 1")
        ax0.bar(x + bar_w, self.table[lo:hi, 2, 0, 0], bar_w, label="frame 2")
        ax0.set_xlabel("read length")
        ax0.set_ylabel("read count")
        ax0.set_title("read length / frame distribution")
        ax0.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax0.legend()

        # Panel 2: P-site distribution around CDS start
        x = np.arange(len(self.dist_starts)) - 100
        ax1.bar(x, self.dist_starts, width=1, color="steelblue")
        ax1.set_xlabel("read-start position relative to CDS start")
        ax1.set_ylabel("read count")
        ax1.set_title("Read starts around CDS start")
        ax1.set_xlim(-50, 50)
        p_site_offset = int(np.argmax(self.pl))
        ax1.bar(
            -p_site_offset,
            self.dist_starts[-p_site_offset + 100],
            width=1,
            color="tab:red",
        )
        ax1.text(
            0.58,
            0.95,
            f"most frequent distance of\nread start to CDS start" f" = {p_site_offset}",
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Panel 3: cleavage distributions
        self.plot(ax=ax2)
        ax2.set_title("Cleavage distributions")

        return fig

    #: TSV header line produced by :meth:`to_files`.
    TSV_HEADER: str = "dataset_id\tpu\tpl\tpr"

    def to_files(
        self,
        dataset_id: str,
        tsv_fh,
        npz_data: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        """Write the model to open file handles.

        Appends one TSV line with the obligatory attributes to *tsv_fh*.
        If *npz_data* is provided, optional arrays (``dist_starts``,
        ``table``) are added to the dict for later ``np.savez``.

        Parameters
        ----------
        dataset_id : str
            Sample identifier placed in the first TSV column.
        tsv_fh : file-like
            Writable text file handle (header already written).
        npz_data : dict[str, np.ndarray] or None, optional
            Accumulator dict for optional arrays.
        """
        pl_str = ",".join(f"{v:.6g}" for v in self.pl)
        pr_str = ",".join(f"{v:.6g}" for v in self.pr)
        tsv_fh.write(f"{dataset_id}\t{self.pu:.6g}\t{pl_str}\t{pr_str}\n")

        if npz_data is not None:
            if hasattr(self, "dist_starts"):
                npz_data[f"{dataset_id}_dist_starts"] = self.dist_starts
            if hasattr(self, "table"):
                npz_data[f"{dataset_id}_table"] = self.table

    @classmethod
    def from_files(
        cls, tsv_path: str, npz_path: Optional[str] = None
    ) -> "dict[str, CleavageModel]":
        """Load models from a TSV file and optionally an NPZ file.

        Parameters
        ----------
        tsv_path : str
            Path to the ``cleavage_models.tsv`` file (obligatory
            attributes: ``pl``, ``pr``, ``pu``).
        npz_path : str or None, optional
            Path to the ``cleavage_models.npz`` file.  When provided,
            optional attributes (``dist_starts``, ``table``) are
            attached to the corresponding models.

        Returns
        -------
        dict[str, CleavageModel]
            Mapping of dataset identifier to the reconstructed model.
        """
        models: dict[str, CleavageModel] = {}
        with open(tsv_path) as fh:
            next(fh)  # skip header
            for line in fh:
                parts = line.strip().split("\t")
                dataset_id = parts[0]
                pu = float(parts[1])
                pl = np.array([float(v) for v in parts[2].split(",")])
                pr = np.array([float(v) for v in parts[3].split(",")])
                models[dataset_id] = cls(pl, pr, pu)

        if npz_path is not None:
            data = np.load(npz_path)
            for dataset_id, model in models.items():
                key_ds = f"{dataset_id}_dist_starts"
                key_t = f"{dataset_id}_table"
                if key_ds in data:
                    model.dist_starts = data[key_ds]
                if key_t in data:
                    model.table = data[key_t]

        return models

    @lru_cache(maxsize=None)
    def shift(self, read_length: int, oua: bool, frame: int) -> int:
        """Compute the most likely distance from read start to P-site.

        Parameters
        ----------
        read_length : int
            Matching length of the read.
        oua : bool
            Whether the read has an untemplated addition.
        frame : int
            Reading frame (0, 1, or 2).

        Returns
        -------
        int
            Offset from read start to the P-site.
        """
        f0 = (-frame) % 3

        pr = self.pr
        pl = self.pl

        i = np.arange(f0, read_length - 2, 3)

        likelihoods = pl[i] * pr[read_length - i - 3]

        if oua:
            if likelihoods.sum() == 0:
                raise ValueError("Read does not fit cleavage model")
            return likelihoods.argmax() * 3 + f0

        else:
            likelihoods *= 1 - self.pu

            read_length -= 1
            frame = (frame + 1) % 3
            f1 = (-frame) % 3

            i = np.arange(f1, read_length - 2, 3)
            likelihoods_ua = pl[i] * pr[read_length - i - 3] * self.pu * 1 / 4

            if len(likelihoods) == len(likelihoods_ua) + 1:
                likelihoods_ua = np.insert(likelihoods_ua, 0, 0)
            return (likelihoods + likelihoods_ua).argmax() * 3 + f0


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
    """Compute the likelihood of a read under the CDS model.

    Parameters
    ----------
    pl : np.ndarray
        Left cleavage probability distribution.
    pr : np.ndarray
        Right cleavage probability distribution.
    pu : float
        Untemplated addition probability.
    length : int
        Matching length of the alignment.
    frame : int
        Reading frame (0, 1, or 2).
    oua : bool
        Whether the read has an untemplated addition.
    region_start : int, optional
        Region start relative to the read start.
    region_end : int, optional
        Region end relative to the read start.

    Returns
    -------
    float
        Read likelihood under the CDS cleavage model.
    """
    f0 = (-frame) % 3

    start_index = max(f0, region_start, length - len(pr) - 2)
    if start_index % 3 == f0 % 3:
        pass
    elif start_index % 3 == (f0 + 1) % 3:
        start_index += 2
    elif start_index % 3 == (f0 + 2) % 3:
        start_index += 1

    i = np.arange(start_index, min(len(pl), length - 2, region_end - 2), 3)

    likelihood = (pl[i] * pr[length - i - 3]).sum()

    if oua:
        likelihood *= pu * 3 / 4

    else:
        # assume there is no ua
        likelihood *= 1 - pu
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

        likelihood += (pl[i] * pr[length - i - 3]).sum() * pu * 1 / 4

    return likelihood * 3


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
    """Compute the likelihood of a read under the noise model.

    Same as :func:`read_in_cds_likelihood` but without reading-frame
    constraints (the cleavage can occur at any position).

    Parameters
    ----------
    pl : np.ndarray
        Left cleavage probability distribution.
    pr : np.ndarray
        Right cleavage probability distribution.
    pu : float
        Untemplated addition probability.
    length : int
        Matching length of the alignment.
    oua : bool
        Whether the read has an untemplated addition.
    region_start : int, optional
        Region start relative to the read start.
    region_end : int, optional
        Region end relative to the read start.

    Returns
    -------
    float
        Read likelihood under the noise cleavage model.
    """
    i = np.arange(
        max(0, region_start, length - len(pr) - 2),
        min(len(pl), length - 2, region_end - 2),
    )
    likelihood = (pl[i] * pr[length - i - 3]).sum()

    if oua:
        likelihood *= pu * 3 / 4

    else:
        # assume there is no ua
        likelihood *= 1 - pu

        # assume there is an ua
        length -= 1
        region_start -= 1
        region_end -= 1

        i = np.arange(
            max(0, region_start, length - 2 - len(pr)),
            min(len(pl), length - 2, region_end - 2),
        )
        likelihood += (pl[i] * pr[length - i - 3]).sum() * pu * 1 / 4
    return likelihood


class CleavageEstimator:
    """EM-based estimator for cleavage model parameters.

    Learns left/right cleavage distributions and the
    untemplated-addition probability from reads that map
    unambiguously to annotated CDS regions.

    Parameters
    ----------
    repeats : int, optional
        Number of random restarts for the EM algorithm.
    maxiter : int, optional
        Maximum EM iterations per restart.
    delta_cutoff : float, optional
        Convergence threshold on parameter change.
    seed : int, optional
        Random seed for reproducibility.
    """

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
        sample_bam_path: str,
        min_considered_length: int = 15,
        max_considered_length: int = 40,
        min_dist_to_start: int = 30,
        min_dist_to_end: int = 30,
        sufficient_counted_alns: int = _MIN_COUNTED_ALNS,
    ) -> None:
        """Collect read-length / frame / UTA counts from a BAM file.

        Iterates over uniquely-mapped reads that fall within CDS
        regions and tallies their length, reading frame and
        untemplated-addition status into ``self.table``.

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
        sufficient_counted_alns : int, optional
            Stop after counting this many alignments.
        """
        self.table = np.zeros(shape=(self.obs_max_len + 10, 3, 2, 1), dtype=np.int32)
        self.dist_starts = np.zeros(shape=(200,), dtype=np.int32)
        self.outside_cds = 0
        self.not_unique = 0
        self.not_countable = 0
        self.bad_length = 0
        self.counted_alns = 0
        with pysam.AlignmentFile(sample_bam_path, "rb") as bam:
            for raw_aln in bam.fetch(until_eof=True):
                if raw_aln.is_unmapped:
                    continue
                aln = RiboSeqAlignment.from_pysam(raw_aln)

                if not aln.unique():
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
                        if self.counted_alns >= sufficient_counted_alns:
                            break

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

        if self.counted_alns < sufficient_counted_alns:
            logger.warning(
                "Not enough alignments counted: %d < %d for %s\n"
                "  not_unique: %d\n"
                "  not_countable: %d\n"
                "  outside_cds: %d\n"
                "  bad_length: %d",
                self.counted_alns,
                sufficient_counted_alns,
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
        )
        shift = self.compute_shift()
        self.correct_max_pos(shift)
        if regularize:
            self.regularize()

        max_pos = int(np.argmax(self.best_pl))
        max_prob = float(self.best_pl[max_pos])
        if max_pos != 12:
            logger.warning(
                "Unusual cleavage model: Upstream cleavage peak is at position %d, expected 12. ",
                max_pos,
            )
        if max_prob < 0.3:
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

    def compute_shift(self, range_start: int = -25, range_end: int = -5) -> int:
        """Compute the shift to align pl with the start-distance histogram.

        Uses cross-correlation (via convolution) to find the offset that
        best aligns the reversed left-cleavage distribution with the
        observed read-start-to-ORF-start distance histogram.

        Parameters
        ----------
        range_start : int, optional
            Start of the search range relative to the ORF start (default -25).
        range_end : int, optional
            End of the search range relative to the ORF start (default -5).

        Returns
        -------
        int
            Shift that aligns the cleavage model to the P-site.
        """
        # pl_rev = self.best_pl[::-1]
        dist_starts = self.dist_starts[100 + range_start : 100 + range_end]

        # Index within dist_starts that corresponds to the expected peak position
        expected_peak = -range_start

        # Best alignment offset from cross-correlation
        conv_offset = np.convolve(dist_starts, self.best_pl, mode="full").argmax()

        return expected_peak - conv_offset

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


@njit
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
    repeats: int,
    obs_max_len: int,
    obs_min_len: int,
    table: np.ndarray,
    maxiter: int,
    c: int,
    delta_cutoff: float,
    seed: int = 42,
) -> tuple:
    """Run the EM algorithm with multiple random restarts.

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

    Returns
    -------
    tuple
        ``(best_ll, best_u, best_pl, best_pr)``.
    """
    np.random.seed(seed)

    total = table[obs_min_len : obs_max_len + 1, :, :, c].sum()

    best_ll = -np.inf

    for rep in range(max(repeats, 1)):

        pl = np.random.rand(obs_max_len + 1)
        pr = np.random.rand(obs_max_len + 1)

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
                    total_p = eps

                    total_p += (pl[left1] * pr[length - left1 - 3 - 1]).sum()
                    s = pl[left1] * pr[length - left1 - 3 - 1] / total_p * n
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
                pl[i] = ((ql1[i + 1] if i + 1 < len(ql1) else 0) + ql0[i]) / total
                pr[i] = (qr1[i] + qr0[i]) / total

            N = table[obs_min_len : obs_max_len + 1, :, :, c].sum()
            u = qu / N

            model_change = (
                np.absolute(old_pl - pl).sum() + np.absolute(old_pr - pr).sum()
            )
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


def to_file(file_path: str, models: dict[str, CleavageModel]) -> None:
    """Serialise a dict of cleavage models to a pickle file.

    Parameters
    ----------
    file_path : str
        Output file path.
    models : dict[str, CleavageModel]
        Mapping of sample name to cleavage model.
    """
    records = [(name, m.pl, m.pr, m.pu) for name, m in models.items()]
    with open(file_path, "wb") as fh:
        pickle.dump(records, fh)


def from_file(file_path: str) -> dict[str, CleavageModel]:
    """Deserialise cleavage models from a pickle file.

    Parameters
    ----------
    file_path : str
        Input file path (written by :func:`to_file`).

    Returns
    -------
    dict[str, CleavageModel]
        Mapping of sample name to cleavage model.
    """
    with open(file_path, "rb") as fh:
        records = pickle.load(fh)
    return {name: CleavageModel(pl, pr, pu) for name, pl, pr, pu in records}


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
