"""One entry point for every count-model solve of the deconvolution.

Four places fit activities to read counts: the deconvolution filter, the
group-LASSO deconvolution, the likelihood-ratio filter and the final
activity estimate.  They differ only in what they set on a
:class:`SolveSpec`; the choice between the multiplicative-update solver
(``config.inner_solver = "mu"``) and scipy's L-BFGS-B is made once, in
:func:`solve`.

:func:`irls_huber` is the robust outer loop of the main deconvolution: it
alternates Huber reweighting with a group-LASSO :func:`solve`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from price2 import mu_solver
from price2.config import Config
from price2.likelihood import (
    distribution_theta,
    huber_weights,
    poisson_nll_grad,
    weighted_poisson_nll_grad,
    weighted_poisson_nll_grad_lasso,
)
from price2.mu_solver import Collapsed, Design

__all__ = [
    "Collapsed",
    "Design",
    "IrlsResult",
    "SolveSpec",
    "irls_huber",
    "solve",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SolveSpec:
    """What one solve minimises.

    Parameters
    ----------
    theta : float or None
        Negative-binomial dispersion; ``None`` for the Poisson model.
    weights : np.ndarray or None
        Per-row Huber weights; ``None`` weights every row equally.
    lam : float
        Group-LASSO strength; ``0`` is a plain maximum-likelihood fit.
    group_shape : tuple[int, int] or None
        ``(num_rgrs, num_runs)`` layout of ``w`` for the group penalty.
        Only needed when ``lam > 0``.
    fixed_mask : np.ndarray or None
        Boolean mask of coordinates pinned at ``config.pseudo_min``, i.e.
        held at zero.  The likelihood-ratio filter uses it for the reduced
        hypothesis.
    strict : bool
        Raise when L-BFGS-B reports failure to converge.  The
        multiplicative-update solver always returns its last iterate.
    lbfgs_scipy_defaults : bool
        Run L-BFGS-B with scipy's own tolerances and without the
        relative-change :class:`Callback`, as the deconvolution filter
        always has.  The other sites use ``config.ftol``/``gtol``/``maxls``
        and the callback.  Ignored by the multiplicative-update solver.
    """

    theta: float | None = None
    weights: np.ndarray | None = None
    lam: float = 0.0
    group_shape: tuple[int, int] | None = None
    fixed_mask: np.ndarray | None = None
    strict: bool = False
    lbfgs_scipy_defaults: bool = False


@dataclass
class IrlsResult:
    """Outcome of :func:`irls_huber`."""

    w: np.ndarray
    outer_iterations: int


def solve(
    X: csr_matrix | None,
    y: np.ndarray | None,
    w0: np.ndarray,
    spec: SolveSpec,
    config: Config,
    *,
    XT: csr_matrix | None = None,
    design: Design | None = None,
    collapsed: Collapsed | None = None,
    stop: Callable[[np.ndarray], bool] | None = None,
) -> np.ndarray:
    """Fit activities ``w`` to counts ``y`` under the model ``δ = X @ w``.

    Parameters
    ----------
    X : csr_matrix, shape ``(n_samples, n_features)``
        Non-negative sparse design matrix.
    y : np.ndarray, shape ``(n_samples,)``
        Observed read counts.
    w0 : np.ndarray, shape ``(n_features,)``
        Starting point.
    spec : SolveSpec
        The objective.
    config : Config
        Provides the solver choice, its tolerances and ``pseudo_min``.
    XT : csr_matrix, optional
        ``X.T`` in CSR layout; computed when omitted.  Callers that solve
        the same system repeatedly pass it in.
    design : Design, optional
        ``X`` and ``y`` with their derived arrays (transpose, counted rows)
        already built, for callers that solve the same system repeatedly.
    collapsed : Collapsed, optional
        A Poisson system already collapsed under ``spec.weights``, which the
        caller guarantees to match; ``X`` and ``y`` may then be ``None``.
        Only the multiplicative-update solver takes it.
    stop : callable, optional
        Early-stopping check of the multiplicative updates
        (:func:`price2.mu_solver.mu_poisson`); ignored by L-BFGS-B.

    Returns
    -------
    np.ndarray, shape ``(n_features,)``
        The fitted activities, every entry at least ``config.pseudo_min``.
    """
    if config.inner_solver == "mu":
        return _solve_mu(
            X, y, w0, spec, config, XT=XT, design=design,
            collapsed=collapsed, stop=stop,
        )
    if collapsed is not None:
        raise ValueError("a collapsed system needs the multiplicative-update solver")
    if design is not None:
        X, y = design.X, design.y
    return _solve_lbfgs(X, y, w0, spec, config)


def _solve_mu(X, y, w0, spec, config, *, XT, design, collapsed, stop):
    mu_solver.set_kernel(config.mu_kernel)
    settings = dict(
        weights=spec.weights,
        lam=spec.lam,
        group_shape=spec.group_shape,
        pmin=config.pseudo_min,
        max_iter=config.mu_inner_max_iter,
        tol=config.mu_inner_tol,
        theta=spec.theta,
    )
    return mu_solver.mu_inner_cpu(
        X, y, w0, fixed_mask=spec.fixed_mask, XT=XT, design=design,
        collapsed=collapsed, stop=stop, **settings
    )


def _solve_lbfgs(X, y, w0, spec, config):
    pmin = config.pseudo_min
    if spec.fixed_mask is None:
        bounds = [(pmin, None)] * len(w0)
    else:
        bounds = [(pmin, pmin) if fixed else (pmin, None) for fixed in spec.fixed_mask]
    if spec.lam > 0.0:
        num_rgrs, num_runs = spec.group_shape
        fun = weighted_poisson_nll_grad_lasso
        args = (X, y, spec.weights, spec.lam, num_rgrs, num_runs, spec.theta)
    elif spec.weights is not None:
        fun = weighted_poisson_nll_grad
        args = (X, y, spec.weights, spec.theta)
    else:
        fun = poisson_nll_grad
        args = (X, y, spec.theta)
    if spec.lbfgs_scipy_defaults:
        callback = None
        options = {"maxiter": 10_000}
    else:
        callback = Callback(w0, config)
        options = {
            "maxiter": 10_000,
            "ftol": config.ftol,
            "gtol": config.gtol,
            "maxls": config.maxls,
        }
    result = minimize(
        fun,
        w0,
        args=args,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        callback=callback,
        options=options,
    )
    converged = result.success or (callback is not None and callback.success)
    if spec.strict and not converged:
        raise RuntimeError(f"L-BFGS-B failed to converge: {result.message}")
    return result.x


class Callback:
    """Convergence callback for L-BFGS-B optimisation.

    Monitors the relative change in activity estimates between iterations
    and raises ``StopIteration`` once no active parameter (one above
    ``config.rgr_min_activity``) moved by more than
    ``config.stop_factor_relative``.

    Attributes
    ----------
    success : bool
        ``True`` when convergence was reached.
    """

    success: bool

    def __init__(self, initial_guess: np.ndarray, config: Config) -> None:
        self.config = config
        self.previous = initial_guess
        self.success = False

    def __call__(self, new: np.ndarray) -> None:
        """Evaluate convergence after an L-BFGS-B iteration.

        Raises
        ------
        StopIteration
            When convergence is detected.
        """
        config = self.config
        # The box bounds keep ``new`` at or above ``pseudo_min``; the floor
        # only guards a caller that passes an unbounded iterate.
        ratio = self.previous / np.maximum(new, config.pseudo_min)
        tolerance = config.stop_factor_relative
        active = new > config.rgr_min_activity
        moved = active & ((ratio < 1 - tolerance) | (ratio > 1 + tolerance))
        if not np.any(moved):
            self.success = True
            raise StopIteration
        self.previous = new


# --------------------------------------------------------------------------- #
# The robust outer loop of the main deconvolution
# --------------------------------------------------------------------------- #


def irls_huber(
    design: Design,
    w0: np.ndarray,
    config: Config,
    num_rgrs: int,
    num_runs: int,
    *,
    max_outer: int | None = None,
) -> IrlsResult:
    """Group-LASSO deconvolution by IRLS with Huber weights.

    Each outer iteration computes the fitted values ``δ = X @ w``, the Huber
    weights on their Pearson residuals, and a weighted group-LASSO
    :func:`solve`.  Two stopping rules run side by side: the relative change
    of ``w`` falling below ``config.irls_huber_tol``, and, when
    ``config.irls_stop_on_active_set`` is set, the set of RGRs above
    ``config.deconvolution_filter_min_activity`` staying unchanged for
    ``config.irls_active_patience`` iterations.

    Parameters
    ----------
    design : Design
        Design matrix (rows ``(EG, run)``, columns ``(RGR, run)``) and read
        counts per row.
    w0 : np.ndarray
        Starting activities (all ones for a cold start).
    config : Config
        Solver settings.
    num_rgrs, num_runs : int
        Group layout of ``w``.
    max_outer : int, optional
        Cap on outer iterations; ``None`` uses ``config.irls_huber_max_outer``.
        The EM light M-step passes ``1`` so one Huber reweight interleaves
        with each global E-step.

    Returns
    -------
    IrlsResult
    """
    theta = distribution_theta(config)
    c = config.irls_huber_c
    n_outer = config.irls_huber_max_outer if max_outer is None else max_outer
    use_mu = config.inner_solver == "mu"
    X, y = design.X, design.y
    XT = design.XT if use_mu else None

    spec = SolveSpec(
        theta=theta, lam=config.lam, group_shape=(num_rgrs, num_runs)
    )
    threshold = config.deconvolution_filter_min_activity
    prev_active = None
    stable_count = 0
    w = w0
    outer = -1
    for outer in range(n_outer):
        delta = np.asarray(X @ w).ravel()
        weights = huber_weights(y, delta, c, theta)
        w_new = solve(
            X, y, w, replace(spec, weights=weights), config, XT=XT, design=design,
        )

        rel_change = mu_solver.relative_change(w_new, w)
        w = w_new
        if rel_change < config.irls_huber_tol:
            break
        if config.irls_stop_on_active_set:
            norms = np.sqrt((w_new.reshape(num_rgrs, num_runs) ** 2).sum(axis=1))
            active = frozenset(np.nonzero(norms > threshold)[0].tolist())
            stable_count = stable_count + 1 if active == prev_active else 0
            prev_active = active
            if stable_count >= config.irls_active_patience:
                break

    return IrlsResult(w, outer + 1)
