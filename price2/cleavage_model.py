"""Cleavage site estimation for Ribo-seq data.

Provides the cleavage model (probability distributions for left and right
cleavage positions relative to the P-site) and its likelihood kernels; the
EM-based estimator that learns the parameters from mapped Ribo-seq reads
lives in :mod:`price2.cleavage_estimator`.

Conventions
-----------
**Read-relative offsets.**  ``region_start`` / ``region_end`` of
:meth:`CleavageModel.pmf`, :func:`read_in_cds_likelihood` and
:func:`read_in_noise_likelihood` are 0-based, half-open offsets from the read
start (the first aligned base is offset 0); ``length`` is the aligned length.
``region_end = UNBOUNDED`` means "no downstream bound", and a call with the
default region ``(0, UNBOUNDED)`` is served from the lookup tables.

**Frame.**  ``frame`` is the phase of the read start relative to the CDS start,
``(read_start - cds_start) % 3`` (``iv_on_cds[0] % 3`` in the estimator,
``(read_start - rgr_start) % 3`` in the read routing).  Seen from the read
start the codon boundaries therefore sit at ``f0 = (-frame) % 3``, ``f0 + 3``,
...; an in-frame region bound must be ``≡ f0 (mod 3)``, and a P-site offset of
``i`` nt puts the read in frame ``(-i) % 3``.  ``frame=None`` selects the
noise model, which has no frame.  The estimator's count table is collected
with column ``frame`` and, after ``CleavageEstimator.correct_table``, holds in
column ``c`` the reads with P-site offset ``≡ c (mod 3)``, i.e. frame
``(-c) % 3`` -- the convention of the EM and of :func:`read_in_cds_likelihood`.

**Distances to the ORF bounds.**  ``dist_to_orf_start[(length, oua, frame)]``
is ``-p`` for the largest region-start offset ``p`` at which the read keeps at
least the overlap likelihood ratio, i.e. the most upstream read start relative
to an ORF start that still plausibly overlaps it; ``dist_to_orf_end`` is
``-p`` for the smallest offset ``p`` of the start of the region's last base
(noise) or last codon (in frame) at which it still does, i.e. the most
downstream such read start.  Both are ``<= 0``; a missing key means no such
position.

**``dist_starts``.**  Histogram of the read-start position relative to the CDS
start: index ``DIST_STARTS_CENTRE + d`` counts the read starts ``d`` nt
downstream of the CDS start (``d < 0`` upstream), ``|d| < DIST_STARTS_CENTRE``.
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

#: A read is assigned a P-site codon (:meth:`CleavageModel.p_site_codon`) only
#: when some codon reaches this likelihood ...
P_SITE_MIN_CODON_LIKELIHOOD: float = 0.01
#: ... and the best codon carries this fraction of the total over the codons.
P_SITE_MIN_DOMINANT_FRACTION: float = 0.8

#: ``region_end`` value meaning "no downstream bound" (see *Conventions*).
UNBOUNDED: int = 10**10

#: Index of ``dist_starts`` that counts the read starts sitting on the CDS
#: start; the histogram has ``2 * DIST_STARTS_CENTRE`` bins.
DIST_STARTS_CENTRE: int = 100


def _count_table(table: np.ndarray) -> np.ndarray:
    """The estimator's count table as ``(length, frame, untemplated addition)``.

    Earlier releases carried a fourth "condition" axis of size one; tables
    they pickled or exported are squeezed to the three-axis layout.
    """
    table = np.asarray(table)
    if table.ndim == 4 and table.shape[3] == 1:
        return np.ascontiguousarray(table[:, :, :, 0])
    if table.ndim != 3:
        raise ValueError(f"a cleavage count table has 3 axes, got shape {table.shape}")
    return table

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
            self.table = _count_table(table)
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
                        region_end=UNBOUNDED,
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
                    region_end=UNBOUNDED,
                )

        self.non_zero_lengths = np.nonzero(self.noise_lut.sum(axis=1))[0]
        self.dist_to_orf_start = self._dist_to_orf_bound(end=False)
        self.dist_to_orf_end = self._dist_to_orf_bound(end=True)

    def pmf(
        self,
        length: int,
        oua: bool,
        frame: Optional[int] = None,
        region_start: int = 0,
        region_end: int = UNBOUNDED,
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
        if length >= len(self.pl) + len(self.pr) + 3 + int(oua):
            return 0

        if frame is None:  # noise
            if region_start == 0 and region_end == UNBOUNDED:
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
            if region_start == 0 and region_end == UNBOUNDED:
                return self.cds_lut[length, frame, int(oua)]
            f0 = (-frame) % 3

            if region_start and f0 != region_start % 3:
                raise ValueError("region_start and frame are not compatible")

            if region_end < UNBOUNDED and f0 != region_end % 3:
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

    def _dist_to_orf_bound(
        self, end: bool, overlap_likelihood_ratio_thresh: float = 0.2
    ) -> dict[tuple[int, bool, Optional[int]], int]:
        """Distances from the read start to the ORF start or end (*Conventions*).

        For every ``(read_length, oua, frame)`` with a non-zero unbounded
        likelihood, scan the region-start offsets (``end=False``) or the
        offsets of the region's last base / codon (``end=True``) and keep
        the farthest one at which the bounded-to-unbounded likelihood ratio
        still exceeds *overlap_likelihood_ratio_thresh*.  In-frame offsets
        step through the codon starts ``(-frame) % 3, +3, ...``; the noise
        model scans every base.

        Returns
        -------
        dict
            ``(read_length, oua, frame) -> -offset``; a combination without
            any qualifying offset is absent.
        """
        dist: dict[tuple[int, bool, Optional[int]], int] = {}
        for read_length in self.non_zero_lengths:
            for oua in (True, False):
                for frame in (None, 0, 1, 2):
                    cl = self.pmf(read_length, oua, frame)
                    if cl == 0:
                        continue
                    if frame is None:
                        positions = np.arange(read_length)
                        unit = 1
                    else:
                        positions = np.arange((-frame) % 3, read_length, 3)
                        unit = 3
                    ratios = np.empty(positions.shape)
                    for i, pos in enumerate(positions):
                        if end:
                            ol = self.pmf(read_length, oua, frame, region_end=pos + unit)
                        else:
                            ol = self.pmf(read_length, oua, frame, region_start=pos)
                        ratios[i] = ol / cl
                    kept = positions[ratios > overlap_likelihood_ratio_thresh]
                    if kept.size:
                        dist[(read_length, oua, frame)] = -(kept.min() if end else kept.max())
        return dist

    def get_dist_to_orf_start(
        self, read_length: int, oua: bool, frame: Optional[int]
    ) -> int:
        """Offset (``<= 0``) of the most upstream plausible read start relative
        to an ORF start; ``KeyError`` when none exists (*Conventions*)."""
        return self.dist_to_orf_start[(read_length, oua, frame)]

    def get_dist_to_orf_end(
        self, read_length: int, oua: bool, frame: Optional[int]
    ) -> int:
        """Offset (``<= 0``) of the most downstream plausible read start
        relative to the start of an ORF's last base (noise) or last codon (in
        frame); ``KeyError`` when none exists (*Conventions*)."""
        return self.dist_to_orf_end[(read_length, oua, frame)]

    def p_site_codon(
        self, length: int, frame: int, untemplated_addition: bool
    ) -> Optional[int]:
        """Index of the codon a read of this shape places its P-site on.

        The per-codon likelihood vector of a read depends only on its
        matching length, its reading frame and whether it carries an
        untemplated addition — never on *where* the read sits — so the
        answer is memoised per triple.  Relative to the read start the CDS
        codon boundaries sit at ``f0``, ``f0 + 3``, ... with
        ``f0 = (-frame) % 3`` (the *Conventions* section above); only codons
        that fit entirely inside the read can carry the P-site.

        Parameters
        ----------
        length : int
            Matching length of the read.
        frame : int
            Reading frame of the read start relative to the CDS.
        untemplated_addition : bool
            Whether the read carries a 5' untemplated addition.

        Returns
        -------
        int or None
            The index of the winning codon, counted from ``f0``; ``None``
            when no codon fits, no codon reaches
            :data:`P_SITE_MIN_CODON_LIKELIHOOD`, or the best one carries less
            than :data:`P_SITE_MIN_DOMINANT_FRACTION` of the total, and for
            reads longer than the model can produce.
        """
        # Memo on the instance, created lazily so that models pickled by an
        # earlier release work too; dropped again by ``__getstate__``.
        cache = self.__dict__.setdefault("_p_site_codons", {})
        key = (length, frame, bool(untemplated_addition))
        try:
            return cache[key]
        except KeyError:
            pass
        winner = None
        f0 = (-frame) % 3
        n_codons = (length - f0) // 3
        if n_codons > 0 and length < len(self.pl) + len(self.pr) + 4:
            likelihoods = np.array(
                [
                    read_in_cds_likelihood(
                        self.pl, self.pr, self.pu, length, frame, key[2],
                        f0 + 3 * i, f0 + 3 * i + 3,
                    )
                    for i in range(n_codons)
                ]
            )
            if likelihoods.max() >= P_SITE_MIN_CODON_LIKELIHOOD:
                likelihoods /= likelihoods.sum()
                if likelihoods.max() >= P_SITE_MIN_DOMINANT_FRACTION:
                    winner = int(np.argmax(likelihoods))
        cache[key] = winner
        return winner

    def __getstate__(self) -> dict:
        """Pickle without the P-site memo (and the memo of earlier releases)."""
        state = dict(self.__dict__)
        state.pop("_p_site_codons", None)
        state.pop("_p_site_table_cache", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore a pickle; earlier releases stored a 4-D count table."""
        self.__dict__.update(state)
        if "table" in state:
            self.table = _count_table(state["table"])

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
                    model.table = _count_table(data[key_t])

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
    region_end: int = UNBOUNDED,
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
    region_end: int = UNBOUNDED,
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


