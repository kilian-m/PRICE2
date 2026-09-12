"""Locus-level ORF deconvolution for PRICE2.

Defines the :class:`Locus` class that aggregates overlapping transcripts
into a single genomic unit, generates ORF candidates, runs group-LASSO
penalised maximum-likelihood estimation, and applies filtering steps
(coverage, deconvolution, likelihood-ratio) to identify actively
translated regions.

ORF candidate generation lives in :mod:`price2.orf_candidates`, read loading
and equivalence-group assignment in :mod:`price2.read_routing`.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

import HTSeq
import numpy as np

logger = logging.getLogger(__name__)
import pandas as pd
from scipy.sparse import csr_matrix

from price2 import likelihood
from price2 import read_routing
from price2 import solver
from price2.config import Config
from price2.equivalence_groups import CELL_CODES, EquivalenceGroup
from price2.genomic_features import ReadGeneratingRegion, Transcript
from price2.likelihood import (
    distribution_theta,
    huber_weights,
    weighted_poisson_log_likelihood_sparse,
    wilks_test_p,
)
from price2.ribo_seq_alignment import RiboSeqAlignment
from price2.ribo_seq_run import RiboSeqRun

# Re-exported for callers (and tests) that import the objectives from here.
_huber_weights = huber_weights
_distribution_theta = distribution_theta
poisson_nll_grad = likelihood.poisson_nll_grad
weighted_poisson_nll_grad = likelihood.weighted_poisson_nll_grad
weighted_poisson_nll_grad_lasso = likelihood.weighted_poisson_nll_grad_lasso

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
        self.id = f"loc_{loci_number}"
        self._init_state()

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

    def _init_state(self) -> None:
        """Reset everything a worker fills in after the skeleton is built."""
        # ORF candidates (``make_rgrs`` / ``build_rgrs``).
        self.rgr_set: set[ReadGeneratingRegion] = set()
        self.gene_ids_complete: set[str] = set()
        self.transcripts_number: int = 0
        # Reads (``get_reads_from_db``) and their counts per RGR
        # (``make_well_fitting_reads``).
        self.rsas_dict: dict[str, list[RiboSeqAlignment]] = {}
        self.run_read_count: dict[str, int] = {}
        self.wfr_df: pd.DataFrame | None = None
        # Equivalence groups and the reads assigned to them.
        self.egs: dict[RiboSeqRun, dict] | None = None
        self.mm_slots: dict[str, dict] = {}
        self.read_counts: dict[RiboSeqRun, float] = {}
        self.counted_reads: dict[str, float] = {}
        self.uncounted_reads: float = 0
        self.eg_cache: read_routing.EgRoutingCache | None = None
        self._eg_y: np.ndarray | None = None
        # Deconvolution results.
        self.result: np.ndarray | None = None
        self.result_df: pd.DataFrame | None = None
        self.irls_huber_weights: np.ndarray | None = None
        self.irls_outer_iterations: int = 0

    @classmethod
    def light(cls, locus_id: str, iv: HTSeq.GenomicInterval, cache) -> Locus:
        """A locus carrying only its routing cache (an intermediate EM pass).

        It has no transcripts, RGRs or equivalence groups (``rgr_set`` is
        ``None``); restoring those dominates the cost of loading a prepared
        locus, and a light M-step needs none of them.
        """
        loc = cls.__new__(cls)
        loc._init_state()
        loc.id = locus_id
        loc.iv = iv
        loc.rgr_set = None
        loc.eg_cache = cache
        return loc

    def __setstate__(self, state: dict) -> None:
        """Restore a pickle, filling in attributes older pickles lack."""
        self._init_state()
        self.__dict__.update(state)

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

    @property
    def transcript_junction_index(self) -> dict[tuple[int, int], tuple]:
        """Lazily built map from intron to the transcripts that splice it.

        Keyed by ``(donor_exon_end, acceptor_exon_start)`` — the intron of
        a pair of *consecutive* exons — with the flanking exon bounds
        attached.  A two-block read maps into a transcript exactly when its
        gap is one of that transcript's introns and its outer ends stay
        inside the flanking exons, which is what
        :meth:`~price2.genomic_region.GenomicRegion.map_to_local` checks
        one candidate at a time; the index turns that into a dict lookup.

        Excluded from pickle (see :meth:`__getstate__`).

        Returns
        -------
        dict
            ``{(intron_start, intron_end): ((transcript, donor_exon_start,
            acceptor_exon_end), ...)}``.
        """
        try:
            return self._transcript_junction_index
        except AttributeError:
            index: dict[tuple[int, int], list] = {}
            for transcript in self.transcripts:
                exons = transcript.exons.intervals
                for donor, acceptor in zip(exons, exons[1:]):
                    index.setdefault((donor.end, acceptor.start), []).append(
                        (transcript, donor.start, acceptor.end)
                    )
            self._transcript_junction_index = {
                k: tuple(v) for k, v in index.items()
            }
            return self._transcript_junction_index

    @property
    def has_abutting_exons(self) -> bool:
        """Whether any transcript has two exons that touch (``end == start``).

        The single-block read fast path in
        :func:`~price2.data_collector.collect_mappings` infers "this
        transcript covers the read within one exon" from the read being
        covered by a contiguous run of breakpoint steps.  That inference
        holds only because a transcript's exons are separated by introns,
        so a contiguous covered stretch cannot straddle two of them.
        Abutting exons would break it; annotations containing them fall
        back to :meth:`~price2.genomic_region.GenomicRegion.map_to_local`.

        Excluded from pickle (see :meth:`__getstate__`).
        """
        try:
            return self._has_abutting_exons
        except AttributeError:
            self._has_abutting_exons = any(
                donor.end == acceptor.start
                for transcript in self.transcripts
                for donor, acceptor in zip(
                    transcript.exons.intervals, transcript.exons.intervals[1:]
                )
            )
            return self._has_abutting_exons

    def __getstate__(self) -> dict:
        """Return pickle state, excluding lazily-rebuilt caches."""
        state = self.__dict__.copy()
        state.pop("_transcript_breakpoint_index", None)
        state.pop("_transcript_junction_index", None)
        state.pop("_has_abutting_exons", None)
        # Sets of RGRs can only be unpickled once the RGRs' state is restored
        # (their hash reads ``genomic_region``).  Emitting the transcripts
        # first pickles every RGR through ``Transcript.rgr_set`` before any
        # locus-level set refers to it.
        first = {
            name: state.pop(name)
            for name in ("iv", "id", "transcript_intervals", "transcripts")
            if name in state
        }
        return {**first, **state}

    # ------------------------------------------------------------------ #
    # Reads and equivalence groups (see :mod:`price2.read_routing`)        #
    # ------------------------------------------------------------------ #

    def get_reads_from_db(self, db_path: str, drop_multimappers: bool = False) -> None:
        """Load this locus's reads (see :func:`price2.read_routing.load_reads`)."""
        read_routing.load_reads(self, db_path, drop_multimappers)

    def compatible_cells(
        self, rsa: RiboSeqAlignment, run: RiboSeqRun
    ) -> frozenset[int] | None:
        """The packed ``(RGR, frame, coverage position)`` cells of a read.

        See :func:`price2.read_routing.rgr_compatibility`.
        """
        return read_routing.rgr_compatibility(self, rsa, run)

    def make_well_fitting_reads(self, runs: list[RiboSeqRun]) -> None:
        """Count well-fitting reads per RGR.

        See :func:`price2.read_routing.count_well_fitting_reads`.
        """
        read_routing.count_well_fitting_reads(self, runs)

    def assign_reads_to_egs(
        self,
        runs: list[RiboSeqRun],
        mm_data: dict | None = None,
        build_cache: bool = False,
    ) -> None:
        """Assign reads to equivalence groups.

        See :func:`price2.read_routing.assign_reads_to_egs`.
        """
        read_routing.assign_reads_to_egs(self, runs, mm_data, build_cache)

    def compute_multimap_lambdas(self, runs: list[RiboSeqRun]) -> list:
        """Per-slot origin rates (see :func:`price2.read_routing.multimap_lambdas`)."""
        return read_routing.multimap_lambdas(self, runs)

    # ------------------------------------------------------------------ #
    # ORF activity estimation                                              #
    # ------------------------------------------------------------------ #

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

        Each run is optimised on its own (see :func:`price2.solver.solve`).
        An ORF is
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
        theta = distribution_theta(config)

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
                # Reads compatible with the shorter RGR are almost always also
                # compatible with the longer one that contains it, making this
                # difference the count of reads unique to the longer RGR.  The
                # partial-overlap likelihood test bounds each RGR by its own
                # coordinates, so compatibility is not strictly monotone and the
                # difference can turn slightly negative.  A count cannot be
                # negative, and the negative-binomial denominator
                # ``X^T(w (y + theta) / (theta + delta))`` would flip sign and
                # diverge if it were.
                rc = max(
                    rgr_read_counts[sorted_rgrs[j].id]
                    - rgr_read_counts[sorted_rgrs[j + 1].id],
                    0,
                )
                egs[frozenset(s)] = (length, rc)

            s.add(sorted_rgrs[-1].id)
            length = len(sorted_rgrs[-1])
            rc = rgr_read_counts[sorted_rgrs[-1].id]
            egs[frozenset(s)] = (length, rc)

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

            result_x = solver.solve(
                X_filter,
                eg_read_counts,
                initial_guess,
                solver.SolveSpec(theta=theta, lbfgs_scipy_defaults=True),
                config,
            )

            rgr_indices_to_keep_one_run = set(
                np.where(result_x >= config.deconvolution_filter_min_activity)[0]
            )
            rgr_indices_to_keep.append(rgr_indices_to_keep_one_run)

        try:
            all_kept = set.union(*rgr_indices_to_keep)
        except TypeError:
            all_kept = set()

        return {k for k, v in rgr_indices.items() if v not in all_kept}

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

        # Must be aligned with ``rgr.index`` (the design-matrix column blocks
        # and the rows of ``result``), not with ``rgr_set`` iteration order:
        # the two are different permutations, and ``deconvolve`` multiplies
        # ``rgr_lengths`` against the index-aligned activity matrix.
        cache = self.eg_cache
        if cache is not None:
            num_rgrs = cache.num_rgrs
            rgr_lengths = cache.rgr_lengths
        else:
            num_rgrs = len(self.rgr_set)
            rgr_lengths = np.empty(num_rgrs, dtype=np.int64)
            for rgr in self.rgr_set:
                rgr_lengths[rgr.index] = len(rgr)

        if self.result is not None:
            initial_guess = self.result
        else:
            initial_guess = np.ones((num_rgrs, num_runs))
        initial_guess = initial_guess.flatten()

        # ``X`` depends only on the equivalence-group geometry and the
        # cleavage/coverage models, all fixed across EM iterations, so a
        # cached locus rebuilds it vectorised instead of walking every cell
        # in Python.  ``y`` came out of the cached read routing.
        y = self._eg_y
        if cache is not None and y is not None:
            X = read_routing.design_matrix_from_cache(
                cache, cm_lut, coverage_params, num_runs
            )
        else:
            X, y = read_routing.egs_to_sparse(
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
        max_outer: int | None = None,
        prune: bool = True,
    ) -> tuple[float, float]:
        """IRLS deconvolution with Huber weights on Pearson residuals.

        At each outer iteration:
          1. Compute fitted values δ = X @ w and Pearson residuals.
          2. Compute Huber weights: ω_i = min(1, c / |r_i|).
          3. Solve weighted group-LASSO Poisson NLL.

        Converges when the relative change in w falls below
        ``config.irls_huber_tol``.

        Warm-starts from ``self.result`` when present (via
        :meth:`to_sparse_args`); with no prior result the initial guess is
        all ones, identical to a cold start.

        Parameters
        ----------
        config : Config
            Configuration providing ``irls_huber_c``,
            ``irls_huber_max_outer``, ``irls_huber_tol``, and
            standard optimisation parameters.
        runs : list[RiboSeqRun]
            Ribo-seq runs to deconvolve jointly.
        max_outer : int, optional
            Cap on IRLS-Huber outer iterations.  ``None`` uses
            ``config.irls_huber_max_outer``.  The EM light M-step passes a
            small value (e.g. 1) so one Huber reweight interleaves with
            each global E-step.
        prune : bool, optional
            When ``True`` (default) low-activity ORFs are removed after
            the solve.  The EM light M-step passes ``False`` to keep the
            ORF set — and hence the E-step targets and warm-start layout —
            fixed across iterations.

        Returns
        -------
        tuple[float, float]
            ``(opt_time, data_time)`` — wall-clock seconds spent in
            optimisation vs. data preparation.
        """
        # ── Build sparse system ──────────────────────────────────────────
        s1 = time.time()
        args_dict = self.to_sparse_args(runs)
        X = args_dict["X"]
        y = args_dict["y"]
        num_rgrs = args_dict["num_rgrs"]
        num_runs = args_dict["num_runs"]
        rgr_lengths = args_dict["rgr_lengths"]
        data_time = time.time() - s1

        # ── Solve ────────────────────────────────────────────────────────
        s1 = time.time()
        fit = solver.irls_huber(
            X,
            y,
            args_dict["initial_guess"],
            config,
            num_rgrs,
            num_runs,
            max_outer=max_outer,
        )
        opt_time = time.time() - s1
        self.irls_outer_iterations = fit.outer_iterations
        logger.debug(
            "IRLS-Huber: converged in %d outer iterations (c=%.1f)",
            fit.outer_iterations,
            config.irls_huber_c,
        )

        # ── Store result ─────────────────────────────────────────────────
        result_matrix = fit.w.reshape(num_rgrs, num_runs)
        result_matrix[result_matrix <= config.pseudo_min] = 0
        self.result = result_matrix

        # Huber weights at the clamped solution, for the weighted LRT.
        theta = distribution_theta(config)
        delta = np.asarray(X @ result_matrix.ravel()).ravel()
        self.irls_huber_weights = huber_weights(
            y, delta, config.irls_huber_c, theta
        )

        # ── Post-optimisation RGR removal (same as deconvolve) ───────────
        if prune:
            x = self.result
            x_t = x.T
            canonical_indices = (rgr_lengths * x_t).argmax(axis=1)
            min_activities = np.maximum(
                x_t[np.arange(x_t.shape[0]), canonical_indices]
                * config.min_activity_fraction,
                config.rgr_min_activity,
            )
            rgr_indices_to_remove = set(
                np.where(np.all(x < min_activities, axis=1))[0]
            )
            rgrs_to_remove = set(
                [
                    rgr
                    for rgr in self.rgr_set
                    if rgr.index in rgr_indices_to_remove
                    and rgr.type == "ORF"
                ]
            )
            self.remove_rgrs(rgrs_to_remove, runs=runs)

        return opt_time, data_time

    def activities_by_id(self) -> dict:
        """Return the current activity matrix keyed by stable ``rgr.id``.

        Keying by ``rgr.id`` (rather than the volatile ``rgr.index``,
        which index densification reassigns) lets the activities be
        reloaded as a warm start in the next EM iteration.

        Returns
        -------
        dict
            ``{rgr_id: numpy.ndarray of shape (num_runs,)}``.
        """
        cache = self.eg_cache
        if cache is not None:
            return {
                rgr_id: self.result[index].copy()
                for index, rgr_id in enumerate(cache.rgr_ids)
            }
        return {rgr.id: self.result[rgr.index].copy() for rgr in self.rgr_set}

    def set_warm_start(self, activities: dict, num_runs: int) -> None:
        """Seed :attr:`result` from persisted per-``rgr.id`` activities.

        RGRs without a stored activity (new to this iteration) default to
        ones.  Must be called once the RGR set is final for the iteration
        (i.e. after all pre-deconvolution filters).

        Parameters
        ----------
        activities : dict
            ``{rgr_id: numpy.ndarray}`` from a previous M-step.
        num_runs : int
            Number of Ribo-seq runs (columns of the activity matrix).
        """
        cache = self.eg_cache
        if cache is not None:
            result = np.ones((cache.num_rgrs, num_runs))
            for index, rgr_id in enumerate(cache.rgr_ids):
                a = activities.get(rgr_id)
                if a is not None:
                    result[index] = a
        else:
            result = np.ones((len(self.rgr_set), num_runs))
            for rgr in self.rgr_set:
                a = activities.get(rgr.id)
                if a is not None:
                    result[rgr.index] = a
        self.result = result

    def remove_rgrs(
        self,
        rgrs_to_remove: set[ReadGeneratingRegion],
        runs: list[RiboSeqRun] | None = None,
    ) -> None:
        """Remove a set of RGRs and update all dependent data structures.

        Updates :attr:`rgr_set`, re-indexes remaining RGRs, collapses
        equivalence groups (if present), and re-slices :attr:`result` (if
        present).

        Parameters
        ----------
        rgrs_to_remove : set[ReadGeneratingRegion]
            RGRs to discard.
        runs : list[RiboSeqRun] or None
            Required when equivalence groups need collapsing.
        """
        old_rgr_set = self.rgr_set
        self.rgr_set = self.rgr_set - rgrs_to_remove

        # The cached routing keys off rgr.index and the equivalence-group
        # layout, both of which this method invalidates.
        self.eg_cache = None
        self._eg_y = None

        # The equivalence-group cells and the rows of ``result`` are keyed by
        # the *old* rgr.index, which the re-indexing below overwrites.
        # ``rgr_set`` iteration order is not the index order (it changes
        # across a pickle round-trip), so the old indices have to be captured
        # rather than re-derived by enumeration.
        old_index = {rgr: rgr.index for rgr in old_rgr_set}
        old_to_new = [-1] * len(old_rgr_set)
        for c, rgr in enumerate(self.rgr_set):
            old_to_new[old_index[rgr]] = c
            rgr.index = c

        # egs
        if self.egs is not None:
            self.collapse_egs(runs, old_to_new)

        # results
        if self.result is not None:
            index_array = np.zeros(len(self.rgr_set), dtype=int)
            for rgr in self.rgr_set:
                index_array[rgr.index] = old_index[rgr]
            self.result = self.result[index_array]

    def collapse_egs(
        self,
        runs: list[RiboSeqRun],
        old_to_new: list[int],
    ) -> None:
        """Collapse equivalence groups after RGR removal.

        The cells of a group key carry ``rgr.index`` (see
        :mod:`price2.equivalence_groups`), so every key is remapped to the
        survivors' new indices; the cells of removed RGRs are dropped, and
        entries whose keys become identical are merged.

        Two memory optimisations:

        * a cache maps each old key to its remapped key, so the new cell
          set is materialised only once per distinct old key (rather than
          once per (old key, run));
        * per-run dicts are rebuilt one at a time and the old dict for
          that run is released immediately, bounding the doubled-allocation
          transient to a single run instead of the full ``len(runs)``.

        Parameters
        ----------
        runs : list[RiboSeqRun]
            Ribo-seq runs whose EGs should be rebuilt.
        old_to_new : list[int]
            The new ``rgr.index`` of every old index, ``-1`` for a removed
            RGR.
        """
        # Old cell -> new cell (``-1`` for a removed RGR), one list lookup
        # per cell.
        cell_map = [
            -1 if new < 0 else new * CELL_CODES + code
            for new in old_to_new
            for code in range(CELL_CODES)
        ]
        key_map: dict[tuple, tuple] = {}

        new_egs: dict = {}
        for run in runs:
            old_run_egs = self.egs.pop(run)
            new_run_egs: dict = defaultdict(EquivalenceGroup)
            for old_eg_key, old_eg in old_run_egs.items():
                new_eg_key = key_map.get(old_eg_key)
                if new_eg_key is None:
                    cells, read_length, oua = old_eg_key
                    new_cells = frozenset(
                        c for c in map(cell_map.__getitem__, cells) if c >= 0
                    )
                    new_eg_key = (new_cells, read_length, oua)
                    key_map[old_eg_key] = new_eg_key

                new_eg = new_run_egs[new_eg_key]
                new_eg.length += old_eg.length
                new_eg.read_count += old_eg.read_count

            new_egs[run] = new_run_egs

        self.egs = new_egs

    def prune_inactive_orfs(
        self,
        config: Config,
        runs: list[RiboSeqRun],
        mm_data: dict | None = None,
    ) -> int:
        """Drop ORFs inactive in every run after an EM light M-step.

        A runtime heuristic for the multimapping EM: once the first light
        M-step has produced activity estimates, ORF-type RGRs whose activity
        is below ``config.rgr_min_activity`` in *every* run are very unlikely
        to revive in later M-steps.  Removing them here shrinks the design
        matrix that all subsequent EM iterations and the final full pass
        rebuild.

        Unlike the post-solve pruning in :meth:`deconvolve` (which also removes
        ORFs falling below a fraction of the locus's canonical activity), this
        uses the plain ``rgr_min_activity`` floor only, so it is deliberately
        conservative — it removes an ORF only when no run gives it appreciable
        activity.

        After removing the RGRs (via :meth:`remove_rgrs`, which re-indexes the
        survivors, re-slices :attr:`result` and collapses the equivalence
        groups) the read-routing cache is rebuilt over the pruned equivalence
        groups so that the prepared-locus blob and the
        :class:`EgRoutingCache` persisted for later passes describe the smaller
        system.

        Parameters
        ----------
        config : Config
            Provides ``rgr_min_activity``.
        runs : list[RiboSeqRun]
            Ribo-seq runs, in the order used to build :attr:`result`.
        mm_data : dict, optional
            This locus's multimapping-slot data, forwarded to the cache rebuild
            so fractional weights and slot routing are recorded.

        Returns
        -------
        int
            Number of ORF RGRs removed.
        """
        result = self.result
        if result is None:
            return 0

        # ``result`` rows are keyed by ``rgr.index`` (identical to the routing
        # cache's column order), so the inactive mask indexes RGRs directly.
        inactive = np.all(result < config.rgr_min_activity, axis=1)
        indices_to_remove = set(np.nonzero(inactive)[0].tolist())
        if not indices_to_remove:
            return 0
        rgrs_to_remove = {
            rgr
            for rgr in self.rgr_set
            if rgr.type == "ORF" and rgr.index in indices_to_remove
        }
        if not rgrs_to_remove:
            return 0

        # Removes the RGRs, re-indexes survivors, re-slices ``result`` and
        # collapses the equivalence groups (remapping every group key to the
        # new indices).  Also clears ``eg_cache`` / ``_eg_y``.
        self.remove_rgrs(rgrs_to_remove, runs=runs)

        # Keep the transcripts' ORF views consistent with the pruned RGR set
        # (mirrors the pre-``make_equivalence_groups`` step of a full prepare),
        # so the persisted locus is self-consistent.
        for tr in self.transcripts:
            tr.update_with_filtered_orfs(self.rgr_set)

        # Rebuild the routing cache over the collapsed equivalence groups.
        # ``collapse_egs`` already produced the correct group geometry and
        # merged read counts, but the per-read routing arrays the cache stores
        # must be re-derived from the reads.  Re-running ``assign_reads_to_egs``
        # with ``build_cache=True`` does exactly that; the group read counts are
        # zeroed first (and the plain-dict conversion restores the KeyError-
        # based "uncounted" handling that ``collapse_egs``'s ``defaultdict``
        # would otherwise mask) so the reassignment recomputes them cleanly.
        for run in runs:
            run_egs = self.egs[run]
            for eg in run_egs.values():
                eg.read_count = 0
            self.egs[run] = dict(run_egs)
            run.read_count = 0
        self.read_counts = {}
        self.counted_reads = {}
        self.uncounted_reads = 0
        self.assign_reads_to_egs(runs, mm_data, build_cache=True)

        return len(rgrs_to_remove)

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

        theta = distribution_theta(config)
        sparse_args = self.to_sparse_args(runs)
        X_lr = sparse_args["X"]
        y_lr = sparse_args["y"]
        num_rgrs = sparse_args["num_rgrs"]
        num_runs = sparse_args["num_runs"]
        initial_guess = sparse_args["initial_guess"]
        # Transposed once for the many MU solves below.
        XT_lr = X_lr.T.tocsr() if config.inner_solver == "mu" else None

        # Huber weights at the current fit, so that outlier EGs contribute
        # less to the test statistic.
        delta = np.asarray(X_lr @ initial_guess).ravel()
        weights = huber_weights(y_lr, delta, config.irls_huber_c, theta)

        def fit(
            w0: np.ndarray, kept: set[int] | None
        ) -> tuple[np.ndarray, float]:
            """Weighted MLE with every RGR outside *kept* pinned at zero."""
            fixed_mask = None
            if kept is not None:
                fixed = np.ones((num_rgrs, num_runs), dtype=bool)
                fixed[list(kept)] = False
                fixed_mask = fixed.ravel()
            w = solver.solve(
                X_lr,
                y_lr,
                w0,
                solver.SolveSpec(
                    theta=theta,
                    weights=weights,
                    fixed_mask=fixed_mask,
                    strict=True,
                ),
                config,
                XT=XT_lr,
            )
            log_likelihood = weighted_poisson_log_likelihood_sparse(
                w, X_lr, y_lr, weights, theta
            )
            return w, log_likelihood

        noise_rgr_indices = {
            rgr.index for rgr in self.rgr_set if rgr.type == "NOISE"
        }
        test_rgr_indices = {rgr.index for rgr in self.rgr_set if rgr.type == "ORF"}
        keep_rgr_indices = noise_rgr_indices | test_rgr_indices

        full_activities, full_log_likelihood = fit(initial_guess, None)

        rgr_ind_list = list(test_rgr_indices)
        try:
            act_sum = self.result[np.array(rgr_ind_list)].sum(axis=1)
            rgr_ind_list = np.array(rgr_ind_list)[np.argsort(act_sum)]
        except IndexError:
            rgr_ind_list = []

        # The same set object: removals below shrink ``keep_rgr_indices`` too.
        full_rgr_ind = keep_rgr_indices
        log_alpha = np.log(config.likelihood_ratio_alpha)
        for rgr_ind in rgr_ind_list:
            reduced_rgr_ind = full_rgr_ind - {rgr_ind}

            # Cheap test first: clamp this ORF without refitting.  The clamped
            # likelihood bounds the refit reduced likelihood from below, so a
            # "not significant" verdict here is final.
            reduced_activities = full_activities.copy().reshape(num_rgrs, -1)
            reduced_activities[rgr_ind] = config.pseudo_min
            reduced_activities = reduced_activities.flatten()
            reduced_log_likelihood = weighted_poisson_log_likelihood_sparse(
                reduced_activities, X_lr, y_lr, weights, theta
            )
            log_p = wilks_test_p(
                full_log_likelihood, reduced_log_likelihood, df_diff=num_runs
            )
            if log_p > log_alpha:
                full_rgr_ind.remove(rgr_ind)
                full_activities = reduced_activities
                full_log_likelihood = reduced_log_likelihood
                continue

            # Looks significant: confirm with properly refit full and reduced
            # models.
            full_activities, full_log_likelihood = fit(
                full_activities, full_rgr_ind
            )
            reduced_activities, reduced_log_likelihood = fit(
                full_activities, reduced_rgr_ind
            )
            log_p = wilks_test_p(
                full_log_likelihood, reduced_log_likelihood, df_diff=num_runs
            )
            if log_p > log_alpha:
                full_rgr_ind.remove(rgr_ind)
                full_activities = reduced_activities
                full_log_likelihood = reduced_log_likelihood

        full_activities, full_log_likelihood = fit(full_activities, full_rgr_ind)

        result = full_activities.reshape(num_rgrs, num_runs)
        result[result <= config.pseudo_min] = 0
        self.result = result

        rgrs_to_remove = {
            rgr
            for rgr in self.rgr_set
            if rgr.index not in keep_rgr_indices and rgr.type != "NOISE"
        }
        self.remove_rgrs(rgrs_to_remove, runs=runs)

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
        theta = distribution_theta(config)
        rgrs_removed = True
        while rgrs_removed:
            args_dict = self.to_sparse_args(runs)
            X_ea = args_dict["X"]
            y_ea = args_dict["y"]
            num_runs = args_dict["num_runs"]
            num_rgrs = args_dict["num_rgrs"]
            initial_guess = args_dict["initial_guess"]

            w = solver.solve(
                X_ea,
                y_ea,
                initial_guess,
                solver.SolveSpec(theta=theta, strict=True),
                config,
            )
            tmp = w.copy().reshape(num_rgrs, num_runs)
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
# ORF detection                                                        #
# ------------------------------------------------------------------ #


