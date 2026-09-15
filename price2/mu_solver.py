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
fraction of the time scipy L-BFGS-B needs merely to approach it, and it maps
directly onto the GPU (all ops are sparse mat-vecs + element-wise arithmetic).

The GPU path is optional: it is used only when ``config.mu_gpu`` is set and
PyTorch with CUDA is importable; otherwise the NumPy path runs on the CPU.
:mod:`price2.gpu_broker` carries a third copy of the update, run for a whole
IRLS-Huber loop on a shared GPU; it checks convergence only every few
iterations, so its results are close to but not identical with these two.
"""
from __future__ import annotations

import functools
import warnings

import numpy as np
from scipy.sparse import csr_matrix

from price2.likelihood import group_lasso_penalty

#: Floor on ``omega * y / delta`` relative to ``omega * y``: ``delta`` is
#: floored at ``omega * y / OVERFLOW_CAP`` so that the ratio cannot overflow
#: the ``X^T`` mat-vec to ``+Inf`` in float64 (and, for the GPU paths,
#: ``float32`` overflows at ~3.4e38, so a far smaller cap heads Inf off).
OVERFLOW_CAP = {"float64": 1e200, "float32": 1e20}


def silence_sparse_beta_warning() -> None:
    """Mute torch's "sparse CSR tensor support is in beta state" notice.

    torch emits it once per process, the first time a CSR tensor is built, which
    means one line of log per broker process and per GPU worker.  The design
    matrices are sparse by nature and CSR is the layout torch's own sparse
    matmul wants, so the notice is not actionable.

    Call this in any process that is about to build sparse tensors, before the
    first one.  The filter is process-global, so it must not be installed in a
    process that does not want it.
    """
    warnings.filterwarnings(
        "ignore",
        message="Sparse CSR tensor support is in beta state",
        category=UserWarning,
    )


@functools.cache
def _torch():
    """Import torch once per process; ``None`` when it or CUDA is unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
    except Exception:
        return None
    silence_sparse_beta_warning()
    return torch


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


def mu_inner_cpu(
    X: csr_matrix,
    y: np.ndarray,
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
) -> np.ndarray:
    """Weighted (+optional group-LASSO) count solve via multiplicative updates.

    Covers every solve in the pipeline once ``inner_solver="mu"``:

    * group-LASSO deconvolution: ``lam > 0``, ``weights`` = Huber weights;
    * deconvolution filter / final estimate: ``lam = 0``, no weights
      (Richardson-Lucy);
    * LRT: ``lam = 0``, Huber weights, ``fixed_mask`` pins the reduced-model
      coordinates at ``pmin`` (the L-BFGS-B ``(pmin, pmin)`` box), so those
      ORFs are held at ~0 exactly as the reduced hypothesis requires.

    Parameters
    ----------
    X : csr_matrix, shape ``(n_samples, n_features)``
        Non-negative sparse design matrix.
    y : np.ndarray, shape ``(n_samples,)``
        Observed counts.
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

    Returns
    -------
    np.ndarray, shape ``(n_features,)``
        The fitted activities, every entry at least ``pmin``.
    """
    _check_group_shape(lam, group_shape, len(w0))
    if XT is None:
        XT = X.T.tocsr()
    if weights is None:
        weights = np.ones(X.shape[0])
    poisson = theta is None
    omega_y = weights * y
    # Per-row floor on delta so that omega_y/delta cannot exceed ~1e200 and
    # overflow the X^T mat-vec to +Inf. That Inf used to cascade into NaN one
    # iteration later — the group-norm penalty computes Inf/Inf —
    # which then silently poisons the whole solve (and, downstream, the EM
    # convergence metric). The floor is omega_y/1e200 (never below the original
    # 1e-300 div-by-zero guard); it is far below delta at the optimum, where
    # delta ~= y, so it never binds for a converged solution and the fixed point
    # / results are unchanged. It only bounds pathological transients where an
    # observed group (omega_y > 0) is momentarily predicted ~0.
    delta_floor = np.maximum(omega_y / OVERFLOW_CAP["float64"], 1e-300)
    if poisson:
        # Poisson denominator data term X^T ω is constant across iterations.
        Xt_omega = np.asarray(XT @ weights).ravel()
    w = w0.astype(np.float64, copy=True)
    if fixed_mask is not None:
        w[fixed_mask] = pmin
    for _ in range(max_iter):
        delta = np.maximum(np.asarray(X @ w).ravel(), delta_floor)
        num = np.asarray(XT @ (omega_y / delta)).ravel()
        if poisson:
            data_den = Xt_omega
        else:
            # NB denominator data term X^T(ω (y+θ)/(θ+δ)) depends on δ.
            data_den = np.asarray(
                XT @ (weights * (y + theta) / (theta + delta))
            ).ravel()
        if lam > 0.0:
            _, pen = group_lasso_penalty(w, lam, group_shape)
            den = np.maximum(data_den + pen, 1e-300)
        else:
            den = np.maximum(data_den, 1e-300)
        w_new = np.maximum(w * num / den, pmin)
        if fixed_mask is not None:
            w_new[fixed_mask] = pmin
        rel = relative_change(w_new, w)
        w = w_new
        if rel < tol:
            break
    return w


class GpuMuSolver:
    """Reusable GPU multiplicative-update solver for one design matrix.

    Transfers ``X``, ``X^T`` and ``y`` to the GPU once and reuses them across
    all IRLS-Huber outer iterations of a locus.

    Parameters
    ----------
    X : csr_matrix
        Non-negative sparse design matrix.
    y : np.ndarray
        Observed counts.
    dtype_str : str
        ``"float32"`` or ``"float64"`` (``config.mu_dtype``).

    Raises
    ------
    RuntimeError
        When PyTorch with CUDA is unavailable.
    """

    def __init__(self, X: csr_matrix, y: np.ndarray, dtype_str: str = "float32"):
        torch = _torch()
        if torch is None:
            raise RuntimeError("GPU MU requested but torch/CUDA is unavailable")
        if dtype_str not in OVERFLOW_CAP:
            raise ValueError(f"mu_dtype must be float32 or float64, got {dtype_str!r}")
        self.t = torch
        self.dtype = getattr(torch, dtype_str)
        self.tiny = 1e-30 if self.dtype == torch.float32 else 1e-300
        self.cap = OVERFLOW_CAP[dtype_str]
        XT = X.T.tocsr()
        self.Xc = self._csr(X)
        self.XcT = self._csr(XT)
        self.yt = torch.from_numpy(np.ascontiguousarray(y)).to(self.dtype).cuda()

    def _csr(self, A):
        t = self.t
        return t.sparse_csr_tensor(
            t.from_numpy(A.indptr.astype(np.int64)),
            t.from_numpy(A.indices.astype(np.int64)),
            t.from_numpy(A.data).to(self.dtype),
            size=tuple(int(s) for s in A.shape), device="cuda", dtype=self.dtype)

    def solve(
        self,
        w0: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        lam: float = 0.0,
        group_shape: tuple[int, int] | None = None,
        pmin: float = 1e-14,
        max_iter: int = 3000,
        tol: float = 1e-5,
        theta: float | None = None,
    ) -> np.ndarray:
        """The update of :func:`mu_inner_cpu` on the GPU, for the held ``X``/``y``.

        Takes the same parameters except ``fixed_mask`` (the LRT solves stay
        on the CPU) and ``XT`` (held since construction); returns the
        activities as a float64 NumPy array.
        """
        _check_group_shape(lam, group_shape, len(w0))
        t = self.t
        poisson = theta is None
        if weights is None:
            weights = np.ones(len(self.yt))
        omega = t.from_numpy(np.ascontiguousarray(weights)).to(self.dtype).cuda()
        w = t.from_numpy(np.ascontiguousarray(w0)).to(self.dtype).cuda()
        omega_y = omega * self.yt
        # The per-element delta floor of mu_inner_cpu, with a dtype-aware cap.
        delta_floor = (omega_y * (1.0 / self.cap)).clamp_min(self.tiny)
        if poisson:
            # Poisson denominator data term X^T ω is constant across iterations.
            Xt_omega = t.mv(self.XcT, omega)
        for _ in range(max_iter):
            delta = t.maximum(t.mv(self.Xc, w), delta_floor)
            num = t.mv(self.XcT, omega_y / delta)
            if poisson:
                data_den = Xt_omega
            else:
                # NB denominator data term X^T(ω (y+θ)/(θ+δ)) depends on δ.
                data_den = t.mv(self.XcT, omega * (self.yt + theta) / (theta + delta))
            if lam > 0.0:
                W = w.view(group_shape)
                norms = (W * W).sum(1).sqrt()
                pen = (lam * (W / norms.clamp_min(self.tiny).view(-1, 1))).reshape(-1)
                den = (data_den + pen).clamp_min(self.tiny)
            else:
                den = data_den.clamp_min(self.tiny)
            w_new = (w * num / den).clamp_min(pmin)
            rel = (w_new - w).norm() / w.norm().clamp_min(1e-14)
            w = w_new
            if rel.item() < tol:
                break
        return w.double().cpu().numpy()
