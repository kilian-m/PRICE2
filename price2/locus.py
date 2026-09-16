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
from dataclasses import dataclass, replace

import HTSeq
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import chi2

from price2 import likelihood
from price2 import mu_solver
from price2 import read_routing
from price2 import solver
from price2.config import Config
from price2.equivalence_groups import EquivalenceGroups
from price2.genomic_features import ReadGeneratingRegion, Transcript
from price2.likelihood import (
    distribution_theta,
    huber_weights,
    weighted_poisson_log_likelihood_rows,
    weighted_poisson_log_likelihood_sparse,
    wilks_test_p,
)
from price2.ribo_seq_alignment import RiboSeqAlignment
from price2.ribo_seq_run import RiboSeqRun

logger = logging.getLogger(__name__)

# Re-exported for callers (and tests) that import the objectives from here.
_huber_weights = huber_weights
_distribution_theta = distribution_theta


def _by_id(transcripts) -> list[Transcript]:
    """The transcripts sorted by id: a fixed order that no hash seed can change."""
    return sorted(transcripts, key=lambda transcript: transcript.id)
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
    design : solver.Design
        ``X`` and ``y`` with their derived arrays, shared by every solve on
        this routing and response (see :meth:`Locus.sparse_system`).
    row_run : np.ndarray
        The run of every row.
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
    design: solver.Design
    row_run: np.ndarray
    num_rgrs: int
    num_runs: int
    rgr_lengths: np.ndarray
    initial_guess: np.ndarray


@dataclass
class _DesignCache:
    """The design matrix of a routing, and the last solver system built on it.

    Identity-keyed: ``remove_rgrs`` replaces the routing and
    ``assign_reads_to_egs`` the response, so a stale entry is never reused.
    """

    routing: read_routing.ReadRouting
    X: csr_matrix
    y: np.ndarray | None = None
    design: solver.Design | None = None


class _LrtStop:
    """Ends a reduced refit of the likelihood-ratio filter once its verdict is settled.

    The multiplicative updates only raise the log-likelihood, so as soon as
    it reaches *target* — the value at which the candidate is not
    significant — the drop verdict is final.  A keep verdict converges on
    the same quantity: the refit ends once the log-likelihood gains less
    than *ll_tol* between two checks.
    """

    def __init__(
        self, log_likelihood, target: float, ll_tol: float
    ) -> None:
        self.log_likelihood = log_likelihood
        self.target = target
        self.ll_tol = ll_tol
        self.previous = -np.inf

    def __call__(self, w: np.ndarray) -> bool:
        value = self.log_likelihood(w)
        if value >= self.target or value - self.previous < self.ll_tol:
            return True
        self.previous = value
        return False


class _WeightedFit:
    """The weighted maximum-likelihood fit the likelihood-ratio filter repeats.

    Every call solves the same system under the same Huber weights; only
    the set of RGRs pinned at zero, and the runs re-solved, change between
    calls.  Without a penalty the objective is a sum over the runs — the
    design matrix is block-diagonal by run — so a fit restricted to some
    runs is exact for those runs and leaves the others untouched.

    Parameters
    ----------
    system : SparseSystem
        The locus's system on the surviving RGR set.
    weights : np.ndarray
        Huber weights at the current fit.
    theta : float or None
        Negative-binomial dispersion; ``None`` for the Poisson model.
    config : Config
        Solver settings.
    """

    def __init__(
        self,
        system: SparseSystem,
        weights: np.ndarray,
        theta: float | None,
        config: Config,
    ) -> None:
        self.system = system
        self.weights = weights
        self.theta = theta
        self.config = config
        self.num_rgrs, self.num_runs = system.num_rgrs, system.num_runs
        self.col_run = np.arange(system.X.shape[1]) % self.num_runs
        design = system.design
        # The Poisson updates run on the collapsed system: built once for
        # the weights, restricted per run subset without touching the full
        # matrix.  The negative binomial and L-BFGS-B slice the full system.
        self.collapsed = None
        if theta is None and config.inner_solver == "mu":
            self.collapsed = solver.Collapsed.build(design, weights)
            self.row_run_p = (
                system.row_run if design.all_counted else system.row_run[design.pos]
            )

    def __call__(
        self,
        w0: np.ndarray,
        kept: set[int] | None,
        runs: np.ndarray | None = None,
        stop_target: float | None = None,
    ) -> np.ndarray:
        """Weighted MLE with every RGR outside *kept* pinned at zero.

        Parameters
        ----------
        w0 : np.ndarray
            Starting activities.
        kept : set[int] or None
            The free RGRs; ``None`` leaves every RGR free.
        runs : np.ndarray of bool, optional
            The runs to re-solve; the activities of the others are returned
            as they are.  ``None`` solves every run.
        stop_target : float, optional
            End the solve once the log-likelihood of the re-solved runs
            reaches this (see :class:`_LrtStop`); only with *runs*.

        Returns
        -------
        np.ndarray
            The fitted activities.
        """
        fixed_mask = None
        if kept is not None:
            fixed = np.ones((self.num_rgrs, self.num_runs), dtype=bool)
            fixed[list(kept)] = False
            fixed_mask = fixed.ravel()
        spec = solver.SolveSpec(
            theta=self.theta, weights=self.weights, fixed_mask=fixed_mask, strict=True
        )
        if runs is None:
            return solver.solve(
                self.system.X,
                self.system.y,
                w0,
                spec,
                self.config,
                design=self.system.design,
                collapsed=self.collapsed,
            )

        cols = np.flatnonzero(runs[self.col_run])
        sub_spec = replace(
            spec, fixed_mask=None if fixed_mask is None else fixed_mask[cols]
        )
        stop = None
        if self.collapsed is not None:
            sub = self.collapsed.restrict(runs[self.row_run_p], cols)
            if stop_target is not None:
                stop = _LrtStop(
                    sub.log_likelihood, stop_target, self.config.likelihood_ratio_ll_tol
                )
            w_sub = solver.solve(
                None, None, w0[cols], sub_spec, self.config, collapsed=sub, stop=stop
            )
        else:
            rows = runs[self.system.row_run]
            X_sub = self.system.X[rows][:, cols]
            y_sub = self.system.y[rows]
            weights_sub = self.weights[rows]
            sub_spec = replace(sub_spec, weights=weights_sub)
            if stop_target is not None:
                stop = _LrtStop(
                    lambda w: weighted_poisson_log_likelihood_sparse(
                        w, X_sub, y_sub, weights_sub, self.theta
                    ),
                    stop_target,
                    self.config.likelihood_ratio_ll_tol,
                )
            w_sub = solver.solve(
                X_sub, y_sub, w0[cols], sub_spec, self.config,
                design=solver.Design(X_sub, y_sub), stop=stop,
            )
        w = w0.copy()
        w[cols] = w_sub
        return w

    def run_log_likelihoods(self, w: np.ndarray) -> np.ndarray:
        """The weighted log-likelihood at ``w``, one entry per run."""
        if self.collapsed is not None:
            row_terms, col_terms = self.collapsed.log_likelihood_terms(w)
            return np.bincount(
                self.row_run_p, weights=row_terms, minlength=self.num_runs
            ) - col_terms.reshape(self.num_rgrs, self.num_runs).sum(axis=0)
        rows = weighted_poisson_log_likelihood_rows(
            w, self.system.X, self.system.y, self.weights, self.theta
        )
        return np.bincount(self.system.row_run, weights=rows, minlength=self.num_runs)

    def log_likelihood(self, w: np.ndarray) -> float:
        return float(self.run_log_likelihoods(w).sum())


def _likelihood_ratio_prune(
    fit: _WeightedFit,
    activities: np.ndarray,
    kept: set[int],
    candidates: list[int],
    config: Config,
) -> tuple[set[int], np.ndarray]:
    """Drop the candidates whose removal does not significantly lower the fit.

    Each candidate gets a two-tier Wilks test against the current full
    model.  The cheap tier clamps the candidate's activities to
    ``config.pseudo_min`` without refitting; that likelihood bounds the
    refit reduced likelihood from below, so a "not significant" verdict is
    final.  Only a significant-looking candidate pays for the constrained
    refits (kept RGRs free, the candidate pinned) that confirm it.

    The refits are per run.  Clamping the candidate changes the fit only in
    the runs where it lowers the log-likelihood (by more than
    ``config.likelihood_ratio_run_tol``); the reduced model is re-solved in
    those runs alone, and the full model is re-solved in a run only after a
    candidate was dropped there without a refit.  A reduced refit stops as
    soon as its verdict is settled (:class:`_LrtStop`).  A dropped
    candidate stays pinned in every later fit.

    Parameters
    ----------
    fit : _WeightedFit
        The fit to repeat.
    activities : np.ndarray
        Starting point of the first full fit.
    kept : set[int]
        The RGR indices in the full model; shrunk in place as candidates
        are dropped.
    candidates : list[int]
        The ORF RGR indices to test, in test order.
    config : Config
        ``likelihood_ratio_alpha``, ``likelihood_ratio_run_tol`` and
        ``pseudo_min``.

    Returns
    -------
    kept : set[int]
        The surviving RGR indices (the same object).
    activities : np.ndarray
        The activities refit on them.
    """
    num_rgrs, num_runs = fit.num_rgrs, fit.num_runs
    log_alpha = np.log(config.likelihood_ratio_alpha)
    run_tol = config.likelihood_ratio_run_tol
    # The Wilks statistic below which a candidate is dropped; a reduced
    # refit whose log-likelihood comes within half of it of the full model's
    # can stop.
    critical = chi2.isf(config.likelihood_ratio_alpha, num_runs)
    if not np.isfinite(critical):
        critical = None

    def clamped(w: np.ndarray, rgr_ind: int) -> np.ndarray:
        reduced = w.reshape(num_rgrs, num_runs).copy()
        reduced[rgr_ind] = config.pseudo_min
        return reduced.ravel()

    activities = fit(activities, None)
    run_ll = fit.run_log_likelihoods(activities)
    # Runs where a candidate was clamped without re-solving the full model.
    stale = np.zeros(num_runs, dtype=bool)

    for rgr_ind in candidates:
        reduced = clamped(activities, rgr_ind)
        reduced_run_ll = fit.run_log_likelihoods(reduced)
        log_p = wilks_test_p(run_ll.sum(), reduced_run_ll.sum(), df_diff=num_runs)
        touched = run_ll - reduced_run_ll > run_tol
        if log_p > log_alpha:
            kept.remove(rgr_ind)
            activities, run_ll = reduced, reduced_run_ll
            stale |= touched
            continue

        if stale.any():
            activities = fit(activities, kept, runs=stale)
            run_ll = fit.run_log_likelihoods(activities)
            stale[:] = False
            reduced = clamped(activities, rgr_ind)
            reduced_run_ll = fit.run_log_likelihoods(reduced)
            touched = run_ll - reduced_run_ll > run_tol
        if touched.any():
            target = None
            if critical is not None:
                target = (
                    run_ll.sum() - 0.5 * critical - reduced_run_ll[~touched].sum()
                )
            reduced = fit(reduced, kept - {rgr_ind}, runs=touched, stop_target=target)
            reduced_run_ll = fit.run_log_likelihoods(reduced)
        log_p = wilks_test_p(run_ll.sum(), reduced_run_ll.sum(), df_diff=num_runs)
        if log_p > log_alpha:
            kept.remove(rgr_ind)
            activities, run_ll = reduced, reduced_run_ll
            # The refit may have stopped early; re-solve these runs before
            # they serve as the full model again.
            stale |= touched

    if stale.any():
        activities = fit(activities, kept, runs=stale)
    return kept, activities


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
    #: The annotated transcripts sorted by id; after ``build_rgrs`` the
    #: read-supported ones, in selection order.
    transcripts: list[Transcript]
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

        transcripts: set[Transcript] = set()
        self.exon_length = 0
        for iv, value in self.transcript_intervals.steps():
            transcripts |= value
            if value:
                self.exon_length += iv.length
        self.transcripts = _by_id(transcripts)

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
        # The reads' compatible cells (``read_compatibility``), computed once
        # from the reads and the candidates and kept through the filters.
        self._compat: read_routing.ReadCompatibility | None = None
        self.wfr_df: pd.DataFrame | None = None
        # Equivalence groups: the geometry from ``make_equivalence_groups``
        # (``EquivalenceGroups``, consumed by ``assign_reads_to_egs``), then
        # the routing of the reads to the design-matrix rows — the sole
        # representation of the equivalence groups from then on — and the
        # response ``y`` under the current read weights, one entry per row.
        self.egs: EquivalenceGroups | None = None
        self.routing: read_routing.ReadRouting | None = None
        self.eg_read_counts: np.ndarray | None = None
        self.counted_reads: dict[str, float] = {}
        # The design matrix of the current routing and the solver system on
        # the current response (``sparse_system``); never pickled.
        self._design_cache: _DesignCache | None = None
        # Deconvolution results: the activity matrix ``(n_rgrs, n_runs)`` and
        # its rendering, the IRLS-Huber iteration count for the perf log.
        self.result: np.ndarray | None = None
        self.result_df: pd.DataFrame | None = None
        self.irls_outer_iterations: int = 0

    @classmethod
    def light(
        cls,
        locus_id: str,
        iv: HTSeq.GenomicInterval,
        routing: read_routing.ReadRouting,
        design_matrix: csr_matrix | None = None,
    ) -> Locus:
        """A locus carrying only its read routing (an intermediate EM pass).

        It has no transcripts or RGRs (``rgrs`` is ``None``); restoring
        those dominates the cost of loading a prepared locus, and a light
        M-step needs none of them.  The design matrix stored with the
        routing, if any, spares the pass its construction.
        """
        loc = cls.__new__(cls)
        loc._init_state()
        loc.id = locus_id
        loc.iv = iv
        loc.rgrs = None
        loc.routing = routing
        loc.set_design_matrix(design_matrix)
        return loc

    def set_design_matrix(self, X: csr_matrix | None) -> None:
        """Adopt *X* as the design matrix of the current routing.

        For a design matrix persisted with the routing
        (:func:`price2.multimap.save_locus_routing`); ``None`` is a no-op.
        """
        if X is not None:
            self._design_cache = _DesignCache(self.routing, X)

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
        clone._compat = None
        clone.routing = None
        clone.eg_read_counts = None
        clone.counted_reads = {}
        clone._design_cache = None
        return clone

    def __setstate__(self, state: dict) -> None:
        """Restore a pickle, filling in attributes older pickles lack."""
        self._init_state()
        # Skeletons collected before the RGRs became an ordered list carry
        # an (empty) ``rgr_set``, and older ones a set of transcripts.
        state.pop("rgr_set", None)
        self.__dict__.update(state)
        if isinstance(self.transcripts, (set, frozenset)):
            self.transcripts = _by_id(self.transcripts)

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
        state.pop("_design_cache", None)
        state.pop("_compat", None)
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

    def read_compatibility(
        self, runs: list[RiboSeqRun]
    ) -> read_routing.ReadCompatibility:
        """The reads' compatible cells, computed on first use.

        Built on the current RGR candidates and valid after any of them are
        removed (see :class:`price2.read_routing.ReadCompatibility`), so the
        well-fitting counts and the routing share one computation.
        """
        if self._compat is None:
            self._compat = read_routing.ReadCompatibility.build(self, runs)
        return self._compat

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
            rgr_ids_to_remove |= self._filter_stop_group(opt_group, config)

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

    def _filter_stop_group(
        self,
        opt_group: set[ReadGeneratingRegion],
        config: Config,
    ) -> set[str]:
        """Solve one optimisation group of the deconvolution filter.

        The RGRs of the group share a stop codon and nest, longest first;
        the reads unique to each are the difference of its well-fitting
        count and the next shorter one's, and the system is the same tiny
        lower-triangular design in every run — only the counts differ.  All
        runs are therefore solved as the columns of one multiplicative-update
        loop (:func:`price2.mu_solver.mu_columns`); the legacy L-BFGS-B path
        solves them one by one.  An ORF is kept if its estimated activity
        exceeds ``config.deconvolution_filter_min_activity`` in at least one
        run; runs in which the locus is probably not expressed (fewer than
        a tenth of the average well-fitting reads) do not vote.

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
        sorted_rgrs = sorted(opt_group, key=len, reverse=True)
        ids = [rgr.id for rgr in sorted_rgrs]
        n = len(sorted_rgrs)
        theta = distribution_theta(config)

        wfr = self.wfr_df
        min_reads = wfr.to_numpy().sum() / wfr.shape[1] * 0.1
        expressed = np.flatnonzero(wfr.to_numpy().sum(axis=0) >= min_reads)
        if expressed.size == 0:
            return set(ids)

        counts = wfr.loc[ids].to_numpy(dtype=np.float64)[:, expressed]
        lengths = np.array([len(rgr) for rgr in sorted_rgrs], dtype=np.float64)
        # Row j of the system is the reads unique to the j longest RGRs: its
        # length is the difference to the next shorter RGR, its count the
        # difference of the well-fitting counts.  Reads compatible with the
        # shorter RGR are almost always also compatible with the longer one
        # that contains it, but the partial-overlap likelihood test bounds
        # each RGR by its own coordinates, so compatibility is not strictly
        # monotone and the difference can turn slightly negative.  A count
        # cannot be negative, and the negative-binomial denominator
        # ``X^T(w (y + theta) / (theta + delta))`` would flip sign and
        # diverge if it were.
        eg_lengths = np.append(lengths[:-1] - lengths[1:], lengths[-1])
        eg_counts = np.vstack([np.maximum(counts[:-1] - counts[1:], 0.0), counts[-1:]])
        # X[row, rgr] = eg_length for every RGR the row's reads fit: the
        # ``row + 1`` longest ones.
        X_filter = np.tril(np.ones((n, n))) * eg_lengths[:, None]
        initial_guess = np.full((n, expressed.size), 0.1)

        if config.inner_solver == "mu":
            mu_solver.set_kernel(config.mu_kernel)
            result = mu_solver.mu_columns(
                X_filter,
                eg_counts,
                initial_guess,
                theta=theta,
                pmin=config.pseudo_min,
                max_iter=config.mu_inner_max_iter,
                tol=config.mu_inner_tol,
            )
        else:
            X_sparse = csr_matrix(X_filter)
            result = np.column_stack(
                [
                    solver.solve(
                        X_sparse,
                        eg_counts[:, k],
                        initial_guess[:, k],
                        solver.SolveSpec(theta=theta, lbfgs_scipy_defaults=True),
                        config,
                    )
                    for k in range(expressed.size)
                ]
            )

        kept = (result >= config.deconvolution_filter_min_activity).any(axis=1)
        return {rgr_id for rgr_id, keep in zip(ids, kept) if not keep}

    def sparse_system(self, runs: list[RiboSeqRun]) -> SparseSystem:
        """The design matrix, response and starting point of the next solve.

        Built from the read routing under the current read weights and the
        runs' model tables.  The design matrix is kept for as long as the
        routing stays the same, and the solver system for as long as the
        response does too (``_design_cache``): the deconvolution, the
        likelihood-ratio filter and the final estimate all solve the same
        system, and a light EM pass solves it under a new response only.  A
        locus is always solved for the same runs, so they are not part of
        the key.

        Parameters
        ----------
        runs : list[RiboSeqRun]
            Ribo-seq runs to include, in the column order of ``result``.
        """
        num_runs = len(runs)
        routing, y = self.routing, self.eg_read_counts
        num_rgrs = routing.num_rgrs
        cache = self._design_cache
        if cache is None or cache.routing is not routing:
            cm_lut, coverage_params = read_routing.model_tables(runs)
            cache = _DesignCache(
                routing, routing.design_matrix(cm_lut, coverage_params, num_runs)
            )
            self._design_cache = cache
        if cache.design is None or cache.y is not y:
            cache.y = y
            cache.design = solver.Design(cache.X, y)
        if self.result is not None:
            initial_guess = self.result
        else:
            initial_guess = np.ones((num_rgrs, num_runs))
        return SparseSystem(
            X=cache.X,
            y=y,
            design=cache.design,
            row_run=routing.row_run,
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
    ) -> None:
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
        """
        system = self.sparse_system(runs)
        fit = solver.irls_huber(
            system.design,
            system.initial_guess,
            config,
            system.num_rgrs,
            system.num_runs,
            max_outer=max_outer,
        )
        self.irls_outer_iterations = fit.outer_iterations
        logger.debug(
            "IRLS-Huber: converged in %d outer iterations (c=%.1f)",
            fit.outer_iterations,
            config.irls_huber_c,
        )

        result_matrix = fit.w.reshape(system.num_rgrs, system.num_runs)
        result_matrix[result_matrix <= config.pseudo_min] = 0
        self.result = result_matrix

        # Drop the ORFs that are inactive in every run: below the configured
        # fraction of their run's most active ORF (by activity × length)
        # and below the absolute floor.
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
        # Huber weights at the current fit, so that outlier EGs contribute
        # less to the test statistic.
        delta = np.asarray(system.X @ system.initial_guess).ravel()
        weights = huber_weights(system.y, delta, config.irls_huber_c, theta)
        fit = _WeightedFit(system, weights, theta, config)

        noise_rgr_indices = {rgr.index for rgr in self.rgrs if not rgr.is_orf}
        test_rgr_indices = {rgr.index for rgr in self.rgrs if rgr.is_orf}
        kept, activities = _likelihood_ratio_prune(
            fit,
            system.initial_guess,
            noise_rgr_indices | test_rgr_indices,
            self._by_ascending_activity(test_rgr_indices),
            config,
        )

        result = activities.reshape(system.num_rgrs, system.num_runs)
        result[result <= config.pseudo_min] = 0
        self.result = result

        self.remove_rgrs(self._orfs_at(set(range(len(self.rgrs))) - kept))

    def _by_ascending_activity(self, indices: set[int]) -> list[int]:
        """*indices* ordered by their RGRs' total activity, weakest first."""
        if not indices:
            return []
        order = np.array(list(indices))
        return order[np.argsort(self.result[order].sum(axis=1))].tolist()

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
                design=system.design,
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


