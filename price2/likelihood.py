"""Count-model likelihoods for the deconvolution.

Every solve in PRICE2 fits the same identity-link model: the expected read
count of an equivalence group is ``δ = X @ w``, the product of the design
matrix and the activities.  The counts are Poisson (``theta is None``) or
negative-binomial with variance ``δ + δ²/θ`` (``theta > 0``); as ``θ → ∞``
the two coincide.  This module holds the objective functions, gradients,
log-likelihoods and robustness weights those solves share, and the Wilks
test the likelihood-ratio filter applies to them.

All functions are pure: they take ``(w, X, y, ...)`` arrays and return
numbers or arrays, so they can be tested against finite differences without
a locus.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import gammaln
from scipy.stats import chi2

from price2.config import Config


def distribution_theta(config: Config) -> float | None:
    """Return the negative-binomial dispersion θ, or ``None`` for Poisson.

    Every solve site funnels the count-model choice through this helper, so
    ``config.distribution`` has one source of truth.

    Parameters
    ----------
    config : Config
        Parsed PRICE configuration object.

    Returns
    -------
    float or None
        ``config.nb_dispersion`` when the negative-binomial model is
        selected, ``None`` for the Poisson model.
    """
    if config.distribution == "nb":
        return float(config.nb_dispersion)
    return None


def huber_weights(
    y: np.ndarray,
    delta: np.ndarray,
    c: float,
    theta: float | None = None,
) -> np.ndarray:
    """Huber weights ``ω_i = min(1, c / |r_i|)`` on Pearson residuals.

    The standardised residual is ``r_i = (y_i − δ_i) / √v_i`` with the
    count-model variance ``v_i``: ``δ_i`` for the Poisson model and
    ``δ_i + δ_i² / θ`` for the negative binomial.  Using the NB variance
    keeps the robustness threshold ``c`` on the same standardised scale as
    the NB likelihood, so overdispersed-but-inlying observations are not
    spuriously down-weighted.

    Parameters
    ----------
    y : np.ndarray, shape ``(n_samples,)``
        Observed read counts.
    delta : np.ndarray, shape ``(n_samples,)``
        Fitted means ``δ = X @ w``.
    c : float
        Huber tuning constant.
    theta : float or None, optional
        Negative-binomial dispersion.  ``None`` selects the Poisson variance.

    Returns
    -------
    np.ndarray, shape ``(n_samples,)``
        Per-observation Huber weights in ``(0, 1]``.
    """
    delta_safe = np.maximum(delta, 1e-14)
    if theta is None:
        var = delta_safe
    else:
        var = delta_safe + delta_safe**2 / theta
    pearson_r = (y - delta_safe) / np.sqrt(var)
    abs_r = np.abs(pearson_r)
    return np.where(abs_r <= c, 1.0, c / np.maximum(abs_r, 1e-14))


# --------------------------------------------------------------------------- #
# Objective functions
# --------------------------------------------------------------------------- #


def weighted_poisson_nll_grad(
    w: np.ndarray,
    X: csr_matrix,
    y: np.ndarray,
    weights: np.ndarray,
    theta: float | None = None,
) -> tuple[float, np.ndarray]:
    """Weighted identity-link Poisson (or negative-binomial) NLL and gradient.

    Model:  ``δ = X @ w`` (mean of the count model).

    Poisson loss:  ``Σ_i ω_i · [δ_i − y_i · ln δ_i]``
    Poisson grad:  ``X.T @ [ω_i · (1 − y_i / δ_i)]``
    NB loss:       ``Σ_i ω_i · [(y_i + θ) · ln(θ + δ_i) − y_i · ln δ_i]``
    NB grad:       ``X.T @ [ω_i · ((y_i + θ) / (θ + δ_i) − y_i / δ_i)]``

    Observations with ``δ_i = y_i = 0`` are excluded, and constants that do
    not depend on ``w`` are dropped from the loss; they affect neither the
    gradient nor the minimiser.

    Parameters
    ----------
    w : np.ndarray, shape ``(n_features,)``
        Current activity estimate (must satisfy ``w_j > 0`` via bounds).
    X : csr_matrix, shape ``(n_samples, n_features)``
        Non-negative sparse design matrix.
    y : np.ndarray, shape ``(n_samples,)``
        Observed read counts.
    weights : np.ndarray, shape ``(n_samples,)``
        Per-observation Huber weights in ``[0, 1]``.
    theta : float or None, optional
        Negative-binomial dispersion.  ``None`` selects the Poisson model.

    Returns
    -------
    loss : float
    grad : np.ndarray, shape ``(n_features,)``
    """
    delta = np.asarray(X @ w).ravel()
    active = ~((delta == 0.0) & (y == 0.0))
    d_act = delta[active]
    y_act = y[active]
    w_act = weights[active]
    if theta is None:
        loss = float((w_act * (d_act - y_act * np.log(d_act))).sum())
        r_act = w_act * (1.0 - y_act / d_act)
    else:
        loss = float(
            (
                w_act
                * ((y_act + theta) * np.log(theta + d_act) - y_act * np.log(d_act))
            ).sum()
        )
        r_act = w_act * ((y_act + theta) / (theta + d_act) - y_act / d_act)
    r = np.zeros(len(y), dtype=np.float64)
    r[active] = r_act
    grad = np.asarray(X.T @ r).ravel()
    return loss, grad


def poisson_nll_grad(
    w: np.ndarray,
    X: csr_matrix,
    y: np.ndarray,
    theta: float | None = None,
) -> tuple[float, np.ndarray]:
    """Unweighted identity-link Poisson (or negative-binomial) NLL and gradient.

    :func:`weighted_poisson_nll_grad` with every weight equal to one.

    Parameters
    ----------
    w : np.ndarray, shape ``(n_features,)``
        Current activity estimate (must satisfy ``w_j > 0`` via bounds).
    X : csr_matrix, shape ``(n_samples, n_features)``
        Non-negative sparse design matrix.
    y : np.ndarray, shape ``(n_samples,)``
        Observed read counts.
    theta : float or None, optional
        Negative-binomial dispersion.  ``None`` selects the Poisson model.

    Returns
    -------
    loss : float
    grad : np.ndarray, shape ``(n_features,)``
    """
    return weighted_poisson_nll_grad(w, X, y, np.ones(len(y)), theta)


def weighted_poisson_nll_grad_lasso(
    w: np.ndarray,
    X: csr_matrix,
    y: np.ndarray,
    weights: np.ndarray,
    lam: float,
    num_rgrs: int,
    num_runs: int,
    theta: float | None = None,
) -> tuple[float, np.ndarray]:
    """Weighted Poisson (or negative-binomial) NLL with a group-LASSO penalty.

    The penalty ``lam · Σ_g ‖w_g‖₂`` groups one RGR's activities across all
    runs, so an RGR is switched on or off for the whole panel.

    Parameters
    ----------
    w : np.ndarray, shape ``(num_rgrs * num_runs,)``
        Current activity estimate, RGR-major.
    X : csr_matrix
        Non-negative sparse design matrix.
    y : np.ndarray
        Observed read counts.
    weights : np.ndarray, shape ``(n_samples,)``
        Per-observation Huber weights.
    lam : float
        Penalty strength.
    num_rgrs, num_runs : int
        Group layout of ``w``.
    theta : float or None, optional
        Negative-binomial dispersion.  ``None`` selects the Poisson model.

    Returns
    -------
    loss : float
    grad : np.ndarray, shape ``(num_rgrs * num_runs,)``
    """
    loss, grad = weighted_poisson_nll_grad(w, X, y, weights, theta)
    W = w.reshape(num_rgrs, num_runs)
    norms = np.sqrt((W**2).sum(axis=1))
    safe_norms = np.maximum(norms, 1e-300)
    loss += lam * norms.sum()
    grad_penalty = lam * (W / safe_norms[:, None])
    grad = grad + grad_penalty.ravel()
    return loss, grad


def weighted_poisson_log_likelihood_sparse(
    w: np.ndarray,
    X: csr_matrix,
    y: np.ndarray,
    weights: np.ndarray,
    theta: float | None = None,
) -> float:
    """Weighted Poisson (or negative-binomial) log-likelihood for LRTs.

    The full log-likelihood, including the ``y``-dependent normalising
    constants, is returned so the Wilks statistic is comparable across
    models.  The constants cancel in the full-vs-reduced difference but keep
    the absolute value a genuine log-likelihood.

    Parameters
    ----------
    w : np.ndarray, shape ``(n_features,)``
    X : csr_matrix
    y : np.ndarray
    weights : np.ndarray, shape ``(n_samples,)``
    theta : float or None, optional
        Negative-binomial dispersion.  ``None`` selects the Poisson model.

    Returns
    -------
    float
        Weighted log-likelihood value.
    """
    delta = np.asarray(X @ w).ravel()
    active = ~((delta == 0.0) & (y == 0.0))
    d_act, y_act, w_act = delta[active], y[active], weights[active]
    if theta is None:
        return float(
            (w_act * (y_act * np.log(d_act) - d_act - gammaln(y_act + 1))).sum()
        )
    return float(
        (
            w_act
            * (
                gammaln(y_act + theta)
                - gammaln(theta)
                - gammaln(y_act + 1)
                + theta * np.log(theta)
                + y_act * np.log(d_act)
                - (y_act + theta) * np.log(theta + d_act)
            )
        ).sum()
    )


# --------------------------------------------------------------------------- #
# Likelihood-ratio test
# --------------------------------------------------------------------------- #


def _chi2_logsf_asymptotic(x: float, k: int) -> float:
    """Asymptotic log upper-tail of chi-squared for large x.

    Uses the divergent asymptotic series for the upper regularized
    incomplete gamma function Q(s, z) with s = k/2, z = x/2:

        log Q(s, z) = -z + (s-1) log(z) - log Γ(s)
                     + log(1 + (s-1)/z + (s-1)(s-2)/z² + ...)

    The series is truncated before the terms start to grow.
    """
    s = 0.5 * k
    z = 0.5 * x
    leading = -z + (s - 1.0) * np.log(z) - gammaln(s)
    term = 1.0
    total = 1.0
    prev_abs = 1.0
    for n in range(1, 64):
        term *= (s - n) / z
        if abs(term) > prev_abs:
            break
        total += term
        prev_abs = abs(term)
        if abs(term) < 1e-16 * abs(total):
            break
    return leading + np.log(total)


def wilks_test_p(
    log_likelihood_full: float,
    log_likelihood_reduced: float,
    df_diff: int = 1,
) -> float:
    """Compute the log p-value for a Wilks likelihood-ratio test.

    Parameters
    ----------
    log_likelihood_full : float
        Log-likelihood of the full model.
    log_likelihood_reduced : float
        Log-likelihood of the reduced model.
    df_diff : int
        Difference in degrees of freedom.

    Returns
    -------
    float
        Log p-value (use ``np.exp(result)`` for the p-value).
    """
    statistic = -2 * (log_likelihood_reduced - log_likelihood_full)
    if statistic <= 0:
        return 0.0
    logp = chi2.logsf(statistic, df_diff)
    if not np.isfinite(logp):
        logp = _chi2_logsf_asymptotic(statistic, df_diff)
    return logp
