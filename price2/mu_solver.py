"""Multiplicative-update inner solver for the group-LASSO Poisson deconvolution.

The default inner solver of every solve site (``config.inner_solver = "mu"``;
scipy's L-BFGS-B remains as the legacy path in :mod:`price2.solver`).  For the
identity-link Poisson NLL with a group-LASSO penalty and non-negativity,

    minimize over w >= pmin
        sum_i omega_i (delta_i - y_i log delta_i) + lam * sum_g ||w_g||_2,
    delta = X @ w,  X >= 0,

the weighted Richardson-Lucy / EM multiplicative update

    w <- w * (X^T (omega*y/delta)) / (X^T omega + lam * w/||w_g||)

is a majorisation-minimisation step: it preserves non-negativity, needs only two
sparse mat-vecs per iteration, and converges to the (unique) minimiser of this
convex problem. On the demanding PRICE2 loci this reaches the true optimum in a
fraction of the time scipy L-BFGS-B needs merely to approach it.

The update runs as a fused kernel (:func:`_kernel_loop`);
``config.mu_kernel = "numpy"`` selects the scipy mat-vec loop it replaced,
which computes the same iterates.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.special import gammaln

from price2.likelihood import group_lasso_penalty

#: Floor on ``omega * y / delta`` relative to ``omega * y``: ``delta`` is
#: floored at ``omega * y / OVERFLOW_CAP`` so that the ratio cannot overflow
#: the ``X^T`` mat-vec to ``+Inf``.
OVERFLOW_CAP = 1e200


def relative_change(w_new: np.ndarray, w: np.ndarray) -> float:
    """Relative L2 change ``‖w_new − w‖ / ‖w‖`` of one update step.

    The stopping rule of the multiplicative inner loop and of the IRLS-Huber
    outer loop (:func:`price2.solver.irls_huber`); ``‖w‖`` is floored at
    ``1e-14`` so an all-zero iterate cannot divide by zero.
    """
    return np.linalg.norm(w_new - w) / max(np.linalg.norm(w), 1e-14)


def _check_group_shape(
    lam: float, group_shape: tuple[int, int] | None, n: int
) -> None:
    if lam > 0.0:
        if group_shape is None:
            raise ValueError("a group-LASSO solve (lam > 0) needs group_shape")
        if group_shape[0] * group_shape[1] != n:
            raise ValueError(
                f"group_shape {group_shape} does not tile {n} activities"
            )


#: How often (in updates) a solve consults its ``stop`` callback.
STOP_EVERY = 10

#: Guard against ``log(0)`` in the objective and the ratio ``omega_y / delta``.
_TINY = 1e-300

#: The multiplicative update loops: ``"numba"`` is the fused
#: kernel (:func:`_kernel_loop`), ``"numpy"`` the numpy loop it replaced;
#: both produce the same iterates (see :func:`mu_poisson`).
KERNELS = ("numba", "numpy")
_KERNEL = ["numba"]


def set_kernel(name: str) -> None:
    """Select the update loop for this process (``config.mu_kernel``)."""
    if name not in KERNELS:
        raise ValueError(f"mu_kernel must be one of {KERNELS}, got {name!r}")
    _KERNEL[0] = name


def _use_numba() -> bool:
    return _KERNEL[0] == "numba"


# --------------------------------------------------------------------------- #
# The fused kernel
# --------------------------------------------------------------------------- #
#
# One update is two sparse mat-vecs and a dozen element-wise numpy calls, each
# with its own dispatch cost and temporary; on the small systems (the
# likelihood-ratio refits, the collapsed systems of read-poor loci) that
# overhead is most of an iteration.  The kernel keeps scipy's C mat-vec — a
# numba loop over a CSR gather is slower than it — called without the
# wrapper's checks, and does everything element-wise in two numba passes.  The
# arithmetic is that of the numpy path: the same products, the group norms
# adding their squares in numpy's pairwise order, so the iterates are the same
# to the last bit.  Only the stopping test's norms are summed in a different
# order, which can change *when* a solve stops by one iteration in the rare
# case that the relative change sits within a rounding error of the tolerance.


@njit(cache=True)
def _pairwise_sum_sq(a, lo, hi):
    """``Σ a[i]²`` over ``[lo, hi)`` in numpy's pairwise summation order."""
    n = hi - lo
    if n < 8:
        res = 0.0
        for i in range(lo, hi):
            res += a[i] * a[i]
        return res
    if n <= 128:
        r0 = a[lo] * a[lo]
        r1 = a[lo + 1] * a[lo + 1]
        r2 = a[lo + 2] * a[lo + 2]
        r3 = a[lo + 3] * a[lo + 3]
        r4 = a[lo + 4] * a[lo + 4]
        r5 = a[lo + 5] * a[lo + 5]
        r6 = a[lo + 6] * a[lo + 6]
        r7 = a[lo + 7] * a[lo + 7]
        i = lo + 8
        end = hi - (n % 8)
        while i < end:
            r0 += a[i] * a[i]
            r1 += a[i + 1] * a[i + 1]
            r2 += a[i + 2] * a[i + 2]
            r3 += a[i + 3] * a[i + 3]
            r4 += a[i + 4] * a[i + 4]
            r5 += a[i + 5] * a[i + 5]
            r6 += a[i + 6] * a[i + 6]
            r7 += a[i + 7] * a[i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < hi:
            res += a[i] * a[i]
            i += 1
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return _pairwise_sum_sq(a, lo, lo + n2) + _pairwise_sum_sq(a, lo + n2, hi)


@njit(cache=True)
def _mu_ratio(delta, omega_y, delta_floor, ratio):
    """Floor ``delta`` in place and set ``ratio = omega_y / delta``."""
    for i in range(delta.shape[0]):
        if delta[i] < delta_floor[i]:
            delta[i] = delta_floor[i]
        ratio[i] = omega_y[i] / delta[i]


@njit(cache=True)
def _mu_nb_ratio(delta, y, weights, theta, ratio):
    """``ratio = ω (y + θ) / (θ + δ)``, the negative-binomial denominator's vector."""
    for i in range(delta.shape[0]):
        ratio[i] = weights[i] * (y[i] + theta) / (theta + delta[i])


@njit(cache=True)
def _mu_update(w, num, den, lam, num_rgrs, num_runs, pmin, fixed):
    """``w ← max(w · num / (den + penalty), pmin)`` in place.

    ``den`` is the denominator's data term (``Xᵀω`` for the Poisson model);
    ``lam > 0`` adds the group-LASSO gradient ``lam · w_g / ‖w_g‖`` with the
    norms summed in numpy's pairwise order; ``fixed`` is a ``uint8`` mask of
    the coordinates held at ``pmin`` (empty for none).  Returns the squared
    norms of the step and of the previous iterate for the stopping rule.
    """
    has_fixed = fixed.shape[0] > 0
    sq_diff = 0.0
    sq_w = 0.0
    if lam > 0.0:
        for g in range(num_rgrs):
            lo = g * num_runs
            norm = np.sqrt(_pairwise_sum_sq(w, lo, lo + num_runs))
            if norm < 1e-300:
                norm = 1e-300
            for j in range(lo, lo + num_runs):
                d = den[j] + lam * (w[j] / norm)
                if d < 1e-300:
                    d = 1e-300
                w_new = w[j] * num[j] / d
                if w_new < pmin:
                    w_new = pmin
                if has_fixed and fixed[j]:
                    w_new = pmin
                diff = w_new - w[j]
                sq_diff += diff * diff
                sq_w += w[j] * w[j]
                w[j] = w_new
    else:
        for j in range(w.shape[0]):
            d = den[j]
            if d < 1e-300:
                d = 1e-300
            w_new = w[j] * num[j] / d
            if w_new < pmin:
                w_new = pmin
            if has_fixed and fixed[j]:
                w_new = pmin
            diff = w_new - w[j]
            sq_diff += diff * diff
            sq_w += w[j] * w[j]
            w[j] = w_new
    return sq_diff, sq_w


try:
    from scipy.sparse._sparsetools import csr_matvec as _csr_matvec
except ImportError:  # pragma: no cover - older scipy
    _csr_matvec = None


def _matvec(A: csr_matrix, x: np.ndarray, out: np.ndarray) -> np.ndarray:
    """``out = A @ x`` through scipy's C routine, without the wrapper's checks."""
    if _csr_matvec is None:
        out[:] = A @ x
        return out
    out[:] = 0.0
    _csr_matvec(A.shape[0], A.shape[1], A.indptr, A.indices, A.data, x, out)
    return out


def _kernel_loop(
    X, XT, omega_y, delta_floor, xt_omega, y, weights, theta,
    w, lam, group_shape, pmin, max_iter, tol, fixed_mask, stop,
):
    """The update loop of the fused kernel.

    Each update is two (three for the negative binomial) C mat-vecs and one
    numba pass over the rows and one over the columns; *stop* is consulted
    every :data:`STOP_EVERY` updates as in the numpy loop.
    """
    num_rgrs, num_runs = group_shape if lam > 0.0 else (1, w.shape[0])
    fixed = (
        np.zeros(0, dtype=np.uint8)
        if fixed_mask is None
        else np.ascontiguousarray(fixed_mask, dtype=np.uint8)
    )
    n_rows, n_cols = X.shape
    delta = np.empty(n_rows)
    ratio = np.empty(n_rows)
    num = np.empty(n_cols)
    den = xt_omega if theta is None else np.empty(n_cols)
    lam, pmin, tol = float(lam), float(pmin), float(tol)
    for iteration in range(1, max_iter + 1):
        _mu_ratio(_matvec(X, w, delta), omega_y, delta_floor, ratio)
        _matvec(XT, ratio, num)
        if theta is not None:
            _mu_nb_ratio(delta, y, weights, theta, ratio)
            _matvec(XT, ratio, den)
        sq_diff, sq_w = _mu_update(w, num, den, lam, num_rgrs, num_runs, pmin, fixed)
        if np.sqrt(sq_diff) / max(np.sqrt(sq_w), 1e-14) < tol:
            break
        if stop is not None and iteration % STOP_EVERY == 0 and stop(w):
            break
    return w


class Design:
    """A design matrix and response, with the rows that carry counts split out.

    Built once per (routing, response) and shared by every solve on it: the
    transpose, the mask of the rows with ``y > 0`` and the row-sliced matrix
    are computed on first use and kept.  For the Poisson model a row with
    ``y_i = 0`` enters the objective only through ``ω_i δ_i = ω_i X_i · w``,
    linear in ``w``, so the multiplicative updates fold every such row into
    the denominator ``Xᵀω`` — one full mat-vec per weight vector — and
    iterate on the counted rows alone (:class:`Collapsed`,
    :func:`mu_poisson`).  On PRICE2 loci 94-96 % of the rows are empty.

    Parameters
    ----------
    X : csr_matrix, shape ``(n_rows, n_cols)``
        Non-negative sparse design matrix.
    y : np.ndarray, shape ``(n_rows,)``
        Observed counts.
    XT : csr_matrix, optional
        ``X.T`` in CSR layout, if the caller already has it.
    """

    __slots__ = ("X", "y", "_XT", "_pos", "_Xp", "_XpT")

    def __init__(
        self, X: csr_matrix, y: np.ndarray, XT: csr_matrix | None = None
    ) -> None:
        self.X = X if isinstance(X, csr_matrix) else csr_matrix(X)
        self.y = np.asarray(y, dtype=np.float64).ravel()
        if self.y.shape[0] != self.X.shape[0]:
            raise ValueError(
                f"{self.X.shape[0]} design-matrix rows but {self.y.shape[0]} counts"
            )
        self._XT = XT
        self._pos = None
        self._Xp = None
        self._XpT = None

    @property
    def n_rows(self) -> int:
        return self.X.shape[0]

    @property
    def n_cols(self) -> int:
        return self.X.shape[1]

    @property
    def XT(self) -> csr_matrix:
        """``X.T`` in CSR layout."""
        if self._XT is None:
            self._XT = self.X.T.tocsr()
        return self._XT

    @property
    def pos(self) -> np.ndarray:
        """Boolean mask of the rows with a count."""
        if self._pos is None:
            self._pos = self.y > 0.0
        return self._pos

    @property
    def all_counted(self) -> bool:
        """Whether every row has a count (the collapse is then a no-op)."""
        return bool(self.pos.all())

    @property
    def Xp(self) -> csr_matrix:
        """The rows of ``X`` with a count (``X`` itself when that is all of them)."""
        if self._Xp is None:
            self._Xp = self.X if self.all_counted else self.X[self.pos]
        return self._Xp

    @property
    def XpT(self) -> csr_matrix:
        """``Xp.T`` in CSR layout."""
        if self._XpT is None:
            self._XpT = self.XT if self.all_counted else self.Xp.T.tocsr()
        return self._XpT

    @property
    def yp(self) -> np.ndarray:
        """The counts of the counted rows."""
        return self.y if self.all_counted else self.y[self.pos]


class Collapsed:
    """A weighted Poisson system with its zero-count rows folded away.

    What :func:`mu_poisson` iterates on: the counted rows ``Xp`` with their
    counts and weights, and the full denominator vector ``Xᵀω`` in which the
    zero-count rows survive only as the column sums ``Σ_{y_i = 0} ω_i X_i``.
    The weighted log-likelihood collapses the same way
    (:meth:`log_likelihood`), and restricting the system to a subset of its
    rows and columns (:meth:`restrict`) touches only the small arrays — how
    the likelihood-ratio filter refits one run at a time.

    Parameters
    ----------
    Xp, XpT : csr_matrix
        The counted rows and their transpose.
    yp, omega_p : np.ndarray
        Their counts and weights.
    Xt_omega : np.ndarray, shape ``(n_cols,)``
        ``Xᵀω`` over *all* rows.
    """

    __slots__ = ("Xp", "XpT", "yp", "omega_p", "Xt_omega", "_zero_colsum", "_const")

    def __init__(
        self,
        Xp: csr_matrix,
        XpT: csr_matrix,
        yp: np.ndarray,
        omega_p: np.ndarray,
        Xt_omega: np.ndarray,
    ) -> None:
        self.Xp = Xp
        self.XpT = XpT
        self.yp = yp
        self.omega_p = omega_p
        self.Xt_omega = Xt_omega
        self._zero_colsum = None
        self._const = None

    @classmethod
    def build(cls, design: Design, weights: np.ndarray | None) -> Collapsed:
        """Collapse *design* under the row weights ``weights`` (``None``: ones)."""
        if weights is None:
            weights = np.ones(design.n_rows)
        Xt_omega = np.asarray(design.XT @ weights).ravel()
        omega_p = weights if design.all_counted else weights[design.pos]
        return cls(design.Xp, design.XpT, design.yp, omega_p, Xt_omega)

    @property
    def n_cols(self) -> int:
        return self.Xt_omega.shape[0]

    @property
    def zero_colsum(self) -> np.ndarray:
        """``Σ_{y_i = 0} ω_i X_i``: the zero-count rows' share of ``Xᵀω``."""
        if self._zero_colsum is None:
            self._zero_colsum = self.Xt_omega - np.asarray(
                self.XpT @ self.omega_p
            ).ravel()
        return self._zero_colsum

    def restrict(self, rows: np.ndarray, cols: np.ndarray) -> Collapsed:
        """The system on a subset of its counted rows and its columns.

        Parameters
        ----------
        rows : np.ndarray of bool, shape ``(n_counted_rows,)``
            The counted rows to keep.
        cols : np.ndarray of int
            The columns to keep, in order.
        """
        Xp = self.Xp[rows][:, cols]
        return Collapsed(
            Xp, Xp.T.tocsr(), self.yp[rows], self.omega_p[rows], self.Xt_omega[cols]
        )

    def log_likelihood_terms(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The weighted Poisson log-likelihood, split by row and by column.

        Returns
        -------
        row_terms : np.ndarray, shape ``(n_counted_rows,)``
            ``ω_i (y_i ln δ_i − δ_i − ln Γ(y_i + 1))`` of the counted rows.
        col_terms : np.ndarray, shape ``(n_cols,)``
            ``c_j w_j`` with ``c = Σ_{y_i = 0} ω_i X_i``: what the zero-count
            rows subtract, attributed to the columns.
        """
        if self._const is None:
            self._const = self.omega_p * gammaln(self.yp + 1.0)
        delta = np.asarray(self.Xp @ w).ravel()
        row_terms = self.omega_p * (self.yp * np.log(delta) - delta) - self._const
        return row_terms, self.zero_colsum * w

    def log_likelihood(self, w: np.ndarray) -> float:
        """The weighted Poisson log-likelihood at ``w`` (with its constants)."""
        row_terms, col_terms = self.log_likelihood_terms(w)
        return float(row_terms.sum() - col_terms.sum())


def mu_poisson(
    system: Collapsed,
    w0: np.ndarray,
    *,
    lam: float = 0.0,
    group_shape: tuple[int, int] | None = None,
    pmin: float = 1e-14,
    max_iter: int = 3000,
    tol: float = 1e-5,
    fixed_mask: np.ndarray | None = None,
    stop: Callable[[np.ndarray], bool] | None = None,
) -> np.ndarray:
    """The Poisson multiplicative updates on a :class:`Collapsed` system.

    The update of :func:`mu_inner_cpu` with the zero-count rows folded into
    the denominator: ``w ← w · Xpᵀ(ω y / δ) / (Xᵀω + lam · w/‖w_g‖)``, with
    ``δ = Xp @ w`` over the counted rows only.  Every iterate is identical
    to the full update's — the folded rows contribute exact zeros to the
    numerator — at a fraction of the cost.

    Parameters
    ----------
    system : Collapsed
        The weighted system.
    w0, lam, group_shape, pmin, max_iter, tol, fixed_mask
        As in :func:`mu_inner_cpu`.
    stop : callable, optional
        Consulted with the current iterate every :data:`STOP_EVERY`
        updates; returning ``True`` ends the solve.  The likelihood-ratio
        filter uses it to stop a reduced refit once its verdict is settled.
    """
    _check_group_shape(lam, group_shape, len(w0))
    Xp, XpT = system.Xp, system.XpT
    omega_y = system.omega_p * system.yp
    # Per-row floor on delta so that omega_y/delta cannot exceed ~1e200 and
    # overflow the X^T mat-vec to +Inf. That Inf used to cascade into NaN one
    # iteration later — the group-norm penalty computes Inf/Inf —
    # which then silently poisons the whole solve (and, downstream, the EM
    # convergence metric). The floor is omega_y/1e200 (never below the original
    # 1e-300 div-by-zero guard); it is far below delta at the optimum, where
    # delta ~= y, so it never binds for a converged solution and the fixed point
    # / results are unchanged. It only bounds pathological transients where an
    # observed group (omega_y > 0) is momentarily predicted ~0.
    delta_floor = np.maximum(omega_y / OVERFLOW_CAP, _TINY)
    data_den = system.Xt_omega
    w = w0.astype(np.float64, copy=True)
    if fixed_mask is not None:
        w[fixed_mask] = pmin
    if _use_numba():
        return _kernel_loop(
            Xp, XpT, omega_y, delta_floor, data_den, None, None, None,
            w, lam, group_shape, pmin, max_iter, tol, fixed_mask, stop,
        )
    for iteration in range(1, max_iter + 1):
        delta = np.maximum(np.asarray(Xp @ w).ravel(), delta_floor)
        num = np.asarray(XpT @ (omega_y / delta)).ravel()
        if lam > 0.0:
            _, pen = group_lasso_penalty(w, lam, group_shape)
            den = np.maximum(data_den + pen, _TINY)
        else:
            den = np.maximum(data_den, _TINY)
        w_new = np.maximum(w * num / den, pmin)
        if fixed_mask is not None:
            w_new[fixed_mask] = pmin
        rel = relative_change(w_new, w)
        w = w_new
        if rel < tol:
            break
        if stop is not None and iteration % STOP_EVERY == 0 and stop(w):
            break
    return w


def mu_inner_cpu(
    X: csr_matrix | None,
    y: np.ndarray | None,
    w0: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    lam: float = 0.0,
    group_shape: tuple[int, int] | None = None,
    pmin: float = 1e-14,
    max_iter: int = 3000,
    tol: float = 1e-5,
    fixed_mask: np.ndarray | None = None,
    theta: float | None = None,
    XT: csr_matrix | None = None,
    design: Design | None = None,
    collapsed: Collapsed | None = None,
    stop: Callable[[np.ndarray], bool] | None = None,
) -> np.ndarray:
    """Weighted (+optional group-LASSO) count solve via multiplicative updates.

    Covers every solve in the pipeline once ``inner_solver="mu"``:

    * group-LASSO deconvolution: ``lam > 0``, ``weights`` = Huber weights;
    * deconvolution filter / final estimate: ``lam = 0``, no weights
      (Richardson-Lucy);
    * LRT: ``lam = 0``, Huber weights, ``fixed_mask`` pins the reduced-model
      coordinates at ``pmin`` (the L-BFGS-B ``(pmin, pmin)`` box), so those
      ORFs are held at ~0 exactly as the reduced hypothesis requires.

    The Poisson solve runs on the counted rows only (:func:`mu_poisson`);
    the negative binomial keeps every row, as its objective is not linear
    in the zero-count rows.

    Parameters
    ----------
    X : csr_matrix, shape ``(n_samples, n_features)``
        Non-negative sparse design matrix.  May be ``None`` with ``collapsed``.
    y : np.ndarray, shape ``(n_samples,)``
        Observed counts.  May be ``None`` with ``collapsed``.
    w0 : np.ndarray, shape ``(n_features,)``
        Starting point.
    weights : np.ndarray, optional
        Per-sample weights ``omega``; ``None`` means all ones.
    lam : float
        Group-LASSO penalty; ``0`` skips the group-norm term entirely.
    group_shape : (num_rgrs, num_runs), optional
        How ``w`` tiles into the penalty groups (one row per RGR, its
        activities across the runs).  Required when ``lam > 0``.
    pmin : float
        Floor on every activity (``config.pseudo_min``).
    max_iter, tol : int, float
        Iteration cap and relative-change stopping tolerance.
    fixed_mask : np.ndarray of bool, optional
        Coordinates held at ``pmin`` throughout.
    theta : float, optional
        ``None`` (default) is the Poisson multiplicative update; a positive
        ``theta`` selects the negative-binomial model.  The numerator
        ``X^T(ω y/δ)`` is identical for both; only the denominator data term
        differs — Poisson uses the iteration-invariant ``X^T ω``, whereas the
        negative binomial uses ``X^T(ω (y+θ)/(θ+δ))``, which depends on ``δ``
        and is recomputed each step.  As ``θ → ∞`` the two updates coincide.
        Both share the fixed point of the true optimum, so the update
        converges to the same solution the L-BFGS-B path finds.
    XT : csr_matrix, optional
        ``X.T`` in CSR layout; computed when omitted.  Callers that solve
        the same system repeatedly pass it in.
    design : Design, optional
        ``X`` and ``y`` with their derived arrays already built; callers
        that solve the same system repeatedly pass it in (it supersedes
        ``XT``).
    collapsed : Collapsed, optional
        The Poisson system already collapsed under ``weights`` (which the
        caller guarantees to match); supersedes ``X``, ``y`` and ``design``.
        Poisson only.
    stop : callable, optional
        Early-stopping check, see :func:`mu_poisson`.

    Returns
    -------
    np.ndarray, shape ``(n_features,)``
        The fitted activities, every entry at least ``pmin``.
    """
    settings = dict(
        lam=lam,
        group_shape=group_shape,
        pmin=pmin,
        max_iter=max_iter,
        tol=tol,
        fixed_mask=fixed_mask,
        stop=stop,
    )
    if theta is None:
        if collapsed is None:
            if design is None:
                design = Design(X, y, XT)
            collapsed = Collapsed.build(design, weights)
        return mu_poisson(collapsed, w0, **settings)
    if collapsed is not None:
        raise ValueError("a collapsed system is a Poisson system; theta must be None")
    if design is not None:
        X, y, XT = design.X, design.y, design.XT
    elif XT is None:
        XT = X.T.tocsr()
    if weights is None:
        weights = np.ones(X.shape[0])
    return _mu_negative_binomial(X, XT, y, weights, w0, theta, **settings)


def _mu_negative_binomial(
    X, XT, y, weights, w0, theta, *, lam, group_shape, pmin, max_iter, tol,
    fixed_mask, stop,
) -> np.ndarray:
    """The negative-binomial multiplicative updates over every row."""
    _check_group_shape(lam, group_shape, len(w0))
    omega_y = weights * y
    # See ``mu_poisson`` for the floor.
    delta_floor = np.maximum(omega_y / OVERFLOW_CAP, _TINY)
    w = w0.astype(np.float64, copy=True)
    if fixed_mask is not None:
        w[fixed_mask] = pmin
    if _use_numba():
        return _kernel_loop(
            X, XT, omega_y, delta_floor, None,
            np.asarray(y, dtype=np.float64), np.asarray(weights, dtype=np.float64),
            theta, w, lam, group_shape, pmin, max_iter, tol, fixed_mask, stop,
        )
    for iteration in range(1, max_iter + 1):
        delta = np.maximum(np.asarray(X @ w).ravel(), delta_floor)
        num = np.asarray(XT @ (omega_y / delta)).ravel()
        # NB denominator data term X^T(ω (y+θ)/(θ+δ)) depends on δ.
        data_den = np.asarray(XT @ (weights * (y + theta) / (theta + delta))).ravel()
        if lam > 0.0:
            _, pen = group_lasso_penalty(w, lam, group_shape)
            den = np.maximum(data_den + pen, _TINY)
        else:
            den = np.maximum(data_den, _TINY)
        w_new = np.maximum(w * num / den, pmin)
        if fixed_mask is not None:
            w_new[fixed_mask] = pmin
        rel = relative_change(w_new, w)
        w = w_new
        if rel < tol:
            break
        if stop is not None and iteration % STOP_EVERY == 0 and stop(w):
            break
    return w


@njit(cache=True)
def _mu_columns_kernel(X, Y, W, theta, pmin, max_iter, tol):
    """The loop of :func:`mu_columns`: every column to its own convergence."""
    n_rows, n_cols = X.shape
    n_resp = Y.shape[1]
    col_sum = np.empty(n_cols)
    for j in range(n_cols):
        s = 0.0
        for i in range(n_rows):
            s += X[i, j]
        col_sum[j] = s if s > 1e-300 else 1e-300
    delta = np.empty(n_rows)
    ratio = np.empty(n_rows)
    num = np.empty(n_cols)
    den = np.empty(n_cols)
    for c in range(n_resp):
        for _ in range(max_iter):
            for i in range(n_rows):
                s = 0.0
                for j in range(n_cols):
                    s += X[i, j] * W[j, c]
                floor = Y[i, c] / 1e200
                if floor < 1e-300:
                    floor = 1e-300
                if s < floor:
                    s = floor
                delta[i] = s
                ratio[i] = Y[i, c] / s
            for j in range(n_cols):
                s = 0.0
                for i in range(n_rows):
                    s += X[i, j] * ratio[i]
                num[j] = s
            if theta < 0.0:
                for j in range(n_cols):
                    den[j] = col_sum[j]
            else:
                for i in range(n_rows):
                    ratio[i] = (Y[i, c] + theta) / (theta + delta[i])
                for j in range(n_cols):
                    s = 0.0
                    for i in range(n_rows):
                        s += X[i, j] * ratio[i]
                    den[j] = s if s > 1e-300 else 1e-300
            sq_diff = 0.0
            sq_w = 0.0
            for j in range(n_cols):
                w_new = W[j, c] * num[j] / den[j]
                if w_new < pmin:
                    w_new = pmin
                diff = w_new - W[j, c]
                sq_diff += diff * diff
                sq_w += W[j, c] * W[j, c]
                W[j, c] = w_new
            norm_w = np.sqrt(sq_w)
            if norm_w < 1e-14:
                norm_w = 1e-14
            if np.sqrt(sq_diff) / norm_w < tol:
                break


def mu_columns(
    X: np.ndarray,
    Y: np.ndarray,
    W0: np.ndarray,
    *,
    theta: float | None = None,
    pmin: float = 1e-14,
    max_iter: int = 3000,
    tol: float = 1e-5,
) -> np.ndarray:
    """Unweighted, unpenalised updates for several responses sharing one ``X``.

    The deconvolution filter solves the same tiny nested system in every
    run, with the run's counts as the response.  Those solves are independent
    columns of one update — ``W ← W · Xᵀ(Y/Δ) / Xᵀ1`` with ``Δ = X @ W`` —
    so they run in one compiled loop over a dense ``X``, each column to its
    own convergence, instead of one scipy solve per run.

    Parameters
    ----------
    X : np.ndarray, shape ``(n_rows, n_cols)``
        Dense non-negative design matrix.
    Y : np.ndarray, shape ``(n_rows, n_responses)``
        One response per column.
    W0 : np.ndarray, shape ``(n_cols, n_responses)``
        Starting activities.
    theta, pmin, max_iter, tol
        As in :func:`mu_inner_cpu`.

    Returns
    -------
    np.ndarray, shape ``(n_cols, n_responses)``
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    W = np.array(W0, dtype=np.float64, order="C", copy=True)
    _mu_columns_kernel(
        X, Y, W, -1.0 if theta is None else float(theta), float(pmin),
        int(max_iter), float(tol),
    )
    return W
