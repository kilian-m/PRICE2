"""Cleavage site estimation for Ribo-seq data.

Provides the cleavage model (probability distributions for left and right
cleavage positions relative to the P-site) and an EM-based estimator that
learns the model parameters from mapped Ribo-seq reads.
"""

import importlib
import logging
from typing import Optional

import numpy as np
from numba import njit

logger = logging.getLogger(__name__)

#: Plausible P-site offsets (read-start-to-P-site distance) for a healthy
#: Ribo-seq dataset.  12 is canonical, but 11 and 13 are common and equally
#: valid depending on read-length range and RNase digestion; a peak outside
#: this range signals an unusual or low-quality library.
PLAUSIBLE_P_SITE_OFFSETS: frozenset[int] = frozenset({11, 12, 13})

#: Minimum probability mass on the upstream cleavage peak of a healthy model.
MIN_PEAK_PROBABILITY: float = 0.3

#: Names that moved to :mod:`price2.cleavage_estimator`, resolved lazily for
#: older imports.
_MOVED = {
    "CleavageEstimator",
    "compute_ll",
    "repeat",
    "select_and_scale",
    "_init_restarts",
    "_em_restart",
    "_em_restarts",
}


def __getattr__(name: str):
    if name in _MOVED:
        return getattr(importlib.import_module("price2.cleavage_estimator"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
        # Axis 0 size is len(pl) + len(pr) + 4 so the longest physically
        # possible read (len(pl) + len(pr) + 2 bases of cleavage + 1
        # untemplated addition) has a valid LUT entry.
        lut_len = len(self.pl) + len(self.pr) + 4
        self.cds_lut = np.zeros((lut_len, 3, 2), dtype=np.float64)
        for length in range(lut_len):
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
        self.noise_lut = np.zeros((lut_len, 2), dtype=np.float64)
        for length in range(lut_len):
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

    def is_plausible(self) -> bool:
        """Whether the upstream cleavage peak looks like a healthy library's.

        The peak must sit at one of :data:`PLAUSIBLE_P_SITE_OFFSETS` and carry
        at least :data:`MIN_PEAK_PROBABILITY` of the mass.
        """
        max_pos = int(np.argmax(self.pl))
        return (
            max_pos in PLAUSIBLE_P_SITE_OFFSETS
            and float(self.pl[max_pos]) >= MIN_PEAK_PROBABILITY
        )

    def plot(self, ax=None) -> None:
        """Plot the cleavage distributions.

        See :func:`price2.plotting.plot_cleavage`.
        """
        from price2.plotting import plot_cleavage

        plot_cleavage(self, ax)

    def plot_full(self, fig=None):
        """Three-panel diagnostic figure.

        See :func:`price2.plotting.plot_cleavage_full`.
        """
        from price2.plotting import plot_cleavage_full

        return plot_cleavage_full(self, fig)

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

@njit(cache=True)
def read_in_cds_likelihood(
    pl: np.ndarray,
    pr: np.ndarray,
    pu: float,
    length: int,
    frame: int,
    oua: bool,
    region_start: int = 0,
    region_end: int = 10**10,
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


@njit(cache=True)
def read_in_noise_likelihood(
    pl: np.ndarray,
    pr: np.ndarray,
    pu: float,
    length: int,
    oua: bool,
    region_start: int = 0,
    region_end: int = 10**10,
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


