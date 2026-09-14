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

import copy
import logging
import time
from dataclasses import dataclass

import HTSeq
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from price2 import likelihood
from price2 import read_routing
from price2 import solver
from price2.config import Config
from price2.genomic_features import ReadGeneratingRegion, Transcript
from price2.likelihood import (
    distribution_theta,
    huber_weights,
    weighted_poisson_log_likelihood_sparse,
    wilks_test_p,
)
from price2.ribo_seq_alignment import RiboSeqAlignment
from price2.ribo_seq_run import RiboSeqRun

logger = logging.getLogger(__name__)

# Re-exported for callers (and tests) that import the objectives from here.
_huber_weights = huber_weights
_distribution_theta = distribution_theta
poisson_nll_grad = likelihood.poisson_nll_grad
weighted_poisson_nll_grad = likelihood.weighted_poisson_nll_grad
weighted_poisson_nll_grad_lasso = likelihood.weighted_poisson_nll_grad_lasso


@dataclass(frozen=True)
class SparseSystem:
    """The linear system of one locus, as the solvers take it.

    Parameters
    ----------
    X : csr_matrix
        Design matrix, rows ``(EG, run)``, columns ``(RGR, run)``; the
        column blocks are in ``rgr.index`` order.
    y : np.ndarray
        Read counts per row, under the current read weights.
    num_rgrs, num_runs : int
        The group layout of the activities.
    rgr_lengths : np.ndarray
        Length of every RGR, by ``rgr.index``.
    initial_guess : np.ndarray
        Starting activities, flattened ``(num_rgrs, num_runs)``: the last
        result when there is one (a warm start), else all ones.
    """

    X: csr_matrix
    y: np.ndarray
    num_rgrs: int
    num_runs: int
    rgr_lengths: np.ndarray
    initial_guess: np.ndarray


class Locus:
    """A genomic locus containing overlapping transcripts and ORF candidates.

    A locus aggregates one or more transcripts whose exons overlap on the
    same strand into a single unit of analysis.  The collector builds it as
    a skeleton (the attributes below); a deconvolution worker then fills in
    the ORF candidates, the reads, their routing to the design-matrix rows
    and the activities — every one of those attributes is declared, with
    what fills it, in :meth:`_init_state`.

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
    exon_length : int
        Total exonic length (bp) covered by the locus.
    """

    iv: HTSeq.GenomicInterval
    id: str
    transcripts: set[Transcript]
    transcript_intervals: HTSeq.GenomicArrayOfSets
    exon_length: int

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
        # ORF candidates (``make_rgrs`` / ``build_rgrs``): the current ORF
        # and noise RGRs in ``rgr.index`` order — an RGR's position here is
        # its index, which addresses its design-matrix column block and its
        # row of ``result``; ``None`` on a light locus (``Locus.light``).
        self.rgrs: list[ReadGeneratingRegion] | None = []
        self.gene_ids_complete: set[str] = set()
        self.transcripts_number: int = 0
        # Reads (``get_reads_from_db``) and their counts per RGR
        # (``make_well_fitting_reads``).
        self.rsas_dict: dict[str, list[RiboSeqAlignment]] = {}
        self.run_read_count: dict[str, int] = {}
        self.wfr_df: pd.DataFrame | None = None
        # Equivalence groups: the geometry from ``make_equivalence_groups``
        # (``{run: {key: length}}``, consumed by ``assign_reads_to_egs``), then
        # the routing of the reads to the design-matrix rows — the sole
        # representation of the equivalence groups from then on — and the
        # response ``y`` under the current read weights, one entry per row.
        self.egs: dict[RiboSeqRun, dict] | None = None
        self.routing: read_routing.ReadRouting | None = None
        self.eg_read_counts: np.ndarray | None = None
        self.counted_reads: dict[str, float] = {}
        # Deconvolution results: the activity matrix ``(n_rgrs, n_runs)`` and
        # its rendering, the IRLS-Huber iteration count for the perf log.
        self.result: np.ndarray | None = None
        self.result_df: pd.DataFrame | None = None
        self.irls_outer_iterations: int = 0

    @classmethod
    def light(
        cls, locus_id: str, iv: HTSeq.GenomicInterval, routing: read_routing.ReadRouting
    ) -> Locus:
        """A locus carrying only its read routing (an intermediate EM pass).

        It has no transcripts or RGRs (``rgrs`` is ``None``); restoring
        those dominates the cost of loading a prepared locus, and a light
        M-step needs none of them.
        """
        loc = cls.__new__(cls)
        loc._init_state()
        loc.id = locus_id
        loc.iv = iv
        loc.rgrs = None
        loc.routing = routing
        return loc

    def prepared_copy(self) -> Locus:
        """A shallow copy without the reads, the routing and the response.

        This is the state persisted between EM passes (``prepared_loci``):
        everything that depends only on the raw reads and is identical in
        every iteration.  The routing is stored on its own
        (``prepared_loci_cache``) so a light pass can load it alone.
        """
        clone = copy.copy(self)
        clone.rsas_dict = {}
        clone.run_read_count = {}
        clone.routing = None
        clone.eg_read_counts = None
        clone.counted_reads = {}
        return clone

    def __setstate__(self, state: dict) -> None:
        """Restore a pickle, filling in attributes older pickles lack."""
        self._init_state()
        # Skeletons collected before the RGRs became an ordered list carry
        # an (empty) ``rgr_set``.
        state.pop("rgr_set", None)
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
        # first pickles every RGR through ``Transcript.rgr_set`` before
        # ``rgrs`` refers to it; an RGR reached through ``rgrs`` first would
        # be inserted into its transcript's set before its own state is set.
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
    ) -> None:
        """Route the reads (once) and compute the response.

        See :func:`price2.read_routing.assign_reads_to_egs`.
        """
        read_routing.assign_reads_to_egs(self, runs, mm_data)

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

        rgr_lengths = {rgr.id: len(rgr.genomic_region) for rgr in self.rgrs}

        rgr_lengths = pd.Series(rgr_lengths).reindex(self.wfr_df.index)
        wfr_df_rel = self.wfr_df.div(rgr_lengths, axis=0)

        rgrs_to_remove_ids = set(
            wfr_df_rel[
                wfr_df_rel.max(axis=1) <= config.min_well_fitting_reads_per_length
            ].index
        )
        rgrs_to_remove = {
            rgr
            for rgr in self.rgrs
            if rgr.id in rgrs_to_remove_ids and rgr.is_orf
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
            for rgr in self.rgrs
            if rgr.is_orf and rgr.id in rgr_ids_to_remove
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
        for rgr in self.rgrs:
            if not rgr.is_orf:
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

    def sparse_system(self, runs: list[RiboSeqRun]) -> SparseSystem:
        """The design matrix, response and starting point of the next solve.

        Built from the read routing under the current read weights and the
        runs' model tables.

        Parameters
        ----------
        runs : list[RiboSeqRun]
            Ribo-seq runs to include, in the column order of ``result``.
        """
        num_runs = len(runs)
        routing = self.routing
        num_rgrs = routing.num_rgrs
        if self.result is not None:
            initial_guess = self.result
        else:
            initial_guess = np.ones((num_rgrs, num_runs))
        cm_lut, coverage_params = read_routing.model_tables(runs)
        return SparseSystem(
            X=routing.design_matrix(cm_lut, coverage_params, num_runs),
            y=self.eg_read_counts,
            num_rgrs=num_rgrs,
            num_runs=num_runs,
            rgr_lengths=routing.rgr_lengths,
            initial_guess=initial_guess.flatten(),
        )

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
        :meth:`sparse_system`); with no prior result the initial guess is
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
        system = self.sparse_system(runs)
        data_time = time.time() - s1

        # ── Solve ────────────────────────────────────────────────────────
        s1 = time.time()
        fit = solver.irls_huber(
            system.X,
            system.y,
            system.initial_guess,
            config,
            system.num_rgrs,
            system.num_runs,
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
        result_matrix = fit.w.reshape(system.num_rgrs, system.num_runs)
        result_matrix[result_matrix <= config.pseudo_min] = 0
        self.result = result_matrix

        # ── Post-optimisation RGR removal (same as deconvolve) ───────────
        if prune:
            x = self.result
            x_t = x.T
            canonical_indices = (system.rgr_lengths * x_t).argmax(axis=1)
            min_activities = np.maximum(
                x_t[np.arange(x_t.shape[0]), canonical_indices]
                * config.min_activity_fraction,
                config.rgr_min_activity,
            )
            self.remove_rgrs(
                self._orfs_at(np.flatnonzero(np.all(x < min_activities, axis=1)))
            )

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
        return {
            rgr_id: self.result[index].copy()
            for index, rgr_id in enumerate(self._rgr_ids())
        }

    def _rgr_ids(self) -> tuple[str, ...]:
        """RGR ids by ``rgr.index``, off the routing for a light locus."""
        if self.rgrs is None:
            return self.routing.rgr_ids
        return tuple(rgr.id for rgr in self.rgrs)

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
        rgr_ids = self._rgr_ids()
        result = np.ones((len(rgr_ids), num_runs))
        for index, rgr_id in enumerate(rgr_ids):
            a = activities.get(rgr_id)
            if a is not None:
                result[index] = a
        self.result = result

    def _orfs_at(self, indices) -> set[ReadGeneratingRegion]:
        """The ORF-type RGRs among ``rgrs[i] for i in indices``."""
        return {
            rgr for rgr in map(self.rgrs.__getitem__, indices) if rgr.is_orf
        }

    def update_transcript_rgrs(self) -> None:
        """Restrict every transcript's ORF and RGR sets to :attr:`rgrs`.

        The equivalence-group build and the read routing iterate the
        transcript-side sets, so they have to agree with the locus; this is
        called once the pre-deconvolution filters are done and after the EM
        prunes ORFs.
        """
        kept = set(self.rgrs)
        for tr in self.transcripts:
            tr.update_with_filtered_orfs(kept)

    def remove_rgrs(self, rgrs_to_remove: set[ReadGeneratingRegion]) -> None:
        """Remove a set of RGRs and update all dependent data structures.

        Compacts :attr:`rgrs` (the survivors keep their relative order and
        take their new position as ``rgr.index``), rebuilds the routing (if
        present) with the response carried over — merged rows add up, as
        their reads are the same — and drops the removed rows of
        :attr:`result` (if present).

        Parameters
        ----------
        rgrs_to_remove : set[ReadGeneratingRegion]
            RGRs to discard.
        """
        kept = [rgr for rgr in self.rgrs if rgr not in rgrs_to_remove]
        old_to_new = [-1] * len(self.rgrs)
        for c, rgr in enumerate(kept):
            old_to_new[rgr.index] = c
            rgr.index = c
        self.rgrs = kept

        if self.routing is not None:
            self.routing, row_map = self.routing.without_rgrs(old_to_new)
            merged = row_map >= 0
            self.eg_read_counts = np.bincount(
                row_map[merged],
                weights=self.eg_read_counts[merged],
                minlength=self.routing.n_rows,
            )

        if self.result is not None:
            self.result = self.result[np.array(old_to_new) >= 0]

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
        survivors, re-slices :attr:`result` and rebuilds the routing) the
        reads are re-keyed against the merged groups and the response is
        recomputed, so that the routing persisted for the later passes is
        the one a rebuild from the reads would give.

        Parameters
        ----------
        config : Config
            Provides ``rgr_min_activity``.
        runs : list[RiboSeqRun]
            Ribo-seq runs, in the order used to build :attr:`result`.
        mm_data : dict, optional
            This locus's multimapping-slot data, for the recomputed response.

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
        rgrs_to_remove = self._orfs_at(np.flatnonzero(inactive))
        if not rgrs_to_remove:
            return 0

        self.remove_rgrs(rgrs_to_remove)

        # Keep the transcripts' ORF views consistent with the pruned RGRs
        # (mirrors the pre-``make_equivalence_groups`` step of a full prepare),
        # so the persisted locus is self-consistent.
        self.update_transcript_rgrs()

        # A read whose key matched no group before may match one of the
        # merged groups: re-key the reads and recompute the response, as
        # rebuilding the routing from the reads would.
        self.routing.rekey()
        self.assign_reads_to_egs(runs, mm_data)

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
        system = self.sparse_system(runs)
        X_lr, y_lr = system.X, system.y
        num_rgrs, num_runs = system.num_rgrs, system.num_runs
        initial_guess = system.initial_guess
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
            rgr.index for rgr in self.rgrs if not rgr.is_orf
        }
        test_rgr_indices = {rgr.index for rgr in self.rgrs if rgr.is_orf}
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

        self.remove_rgrs(
            self._orfs_at(set(range(len(self.rgrs))) - keep_rgr_indices)
        )

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
            system = self.sparse_system(runs)

            w = solver.solve(
                system.X,
                system.y,
                system.initial_guess,
                solver.SolveSpec(theta=theta, strict=True),
                config,
            )
            tmp = w.copy().reshape(system.num_rgrs, system.num_runs)
            tmp[tmp <= config.pseudo_min] = 0
            self.result = tmp

            rgrs_to_remove = self._orfs_at(
                np.flatnonzero(np.all(self.result < config.rgr_min_activity, axis=1))
            )
            if rgrs_to_remove:
                self.remove_rgrs(rgrs_to_remove)
            else:
                rgrs_removed = False

        self.result_df = pd.DataFrame(
            self.result,
            index=[rgr.id for rgr in self.rgrs],
            columns=[run.id for run in runs],
        )


# ------------------------------------------------------------------ #
# ORF detection                                                        #
# ------------------------------------------------------------------ #


