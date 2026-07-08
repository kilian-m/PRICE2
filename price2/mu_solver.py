"""Multiplicative-update inner solver for the group-LASSO Poisson deconvolution.

Drop-in replacement for the scipy L-BFGS-B inner solve in
:meth:`Locus.deconvolve`. For the identity-link Poisson NLL with a group-LASSO
penalty and non-negativity,

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
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

_TORCH = None
_TORCH_OK = None


def _torch():
    """Lazily import torch; return the module or ``None`` if unavailable."""
    global _TORCH, _TORCH_OK
    if _TORCH_OK is None:
        try:
            import torch  # noqa: F401
            _TORCH = torch
            _TORCH_OK = torch.cuda.is_available()
        except Exception:
            _TORCH, _TORCH_OK = None, False
    return _TORCH if _TORCH_OK else None


def mu_inner_cpu(X, XT, y, weights, w0, lam, num_rgrs, num_runs, pmin,
                 max_iter=3000, tol=1e-5, fixed_mask=None, theta=None):
    """Weighted (+optional group-LASSO) count solve via multiplicative updates.

    Covers every solve in the pipeline once ``inner_solver="mu"``:
      * group-LASSO deconvolution: lam>0, weights=Huber weights;
      * deconvolution filter / final estimate: lam=0, weights=1 (Richardson-Lucy);
      * LRT: lam=0, weights=Huber weights, ``fixed_mask`` pins the reduced-model
        coords at ``pmin`` (the L-BFGS-B (pmin,pmin) box), so those ORFs are held
        at ~0 exactly as the reduced hypothesis requires.

    The group-norm term is skipped entirely when ``lam == 0``.

    ``theta is None`` (default) is the Poisson multiplicative update; a positive
    ``theta`` selects the negative-binomial model. The numerator ``X^T(ω y/δ)``
    is identical for both; only the denominator data term differs — Poisson uses
    the iteration-invariant ``X^T ω``, whereas the negative binomial uses
    ``X^T(ω (y+θ)/(θ+δ))``, which depends on ``δ`` and is recomputed each step.
    As ``θ → ∞`` the two updates coincide (``(y+θ)/(θ+δ) → 1``). Both share the
    fixed point of the true optimum (numerator = denominator at a stationary
    point), so the update converges to the same solution the L-BFGS-B path finds.
    """
    poisson = theta is None
    omega_y = weights * y
    if poisson:
        # Poisson denominator data term X^T ω is constant across iterations.
        Xt_omega = np.asarray(XT @ weights).ravel()
    w = w0.astype(np.float64, copy=True)
    if fixed_mask is not None:
        w[fixed_mask] = pmin
    for _ in range(max_iter):
        delta = np.maximum(np.asarray(X @ w).ravel(), 1e-300)
        num = np.asarray(XT @ (omega_y / delta)).ravel()
        if poisson:
            data_den = Xt_omega
        else:
            # NB denominator data term X^T(ω (y+θ)/(θ+δ)) depends on δ.
            data_den = np.asarray(
                XT @ (weights * (y + theta) / (theta + delta))
            ).ravel()
        if lam > 0.0:
            W = w.reshape(num_rgrs, num_runs)
            norms = np.sqrt((W**2).sum(axis=1))
            pen = lam * (W / np.maximum(norms, 1e-300)[:, None]).ravel()
            den = np.maximum(data_den + pen, 1e-300)
        else:
            den = np.maximum(data_den, 1e-300)
        w_new = np.maximum(w * num / den, pmin)
        if fixed_mask is not None:
            w_new[fixed_mask] = pmin
        rel = np.linalg.norm(w_new - w) / max(np.linalg.norm(w), 1e-14)
        w = w_new
        if rel < tol:
            break
    return w


class GpuMuSolver:
    """Reusable GPU multiplicative-update solver for one design matrix.

    Transfers ``X``, ``X^T`` and ``y`` to the GPU once and reuses them across
    all IRLS-Huber outer iterations of a locus.
    """

    def __init__(self, X: csr_matrix, y, dtype_str="float32"):
        torch = _torch()
        if torch is None:
            raise RuntimeError("GPU MU requested but torch/CUDA is unavailable")
        self.t = torch
        self.dtype = getattr(torch, dtype_str)
        self.tiny = 1e-30 if self.dtype == torch.float32 else 1e-300
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

    def solve(self, weights, w0, lam, num_rgrs, num_runs, pmin,
              max_iter=3000, tol=1e-5, theta=None):
        t = self.t
        poisson = theta is None
        omega = t.from_numpy(np.ascontiguousarray(weights)).to(self.dtype).cuda()
        w = t.from_numpy(np.ascontiguousarray(w0)).to(self.dtype).cuda()
        omega_y = omega * self.yt
        if poisson:
            # Poisson denominator data term X^T ω is constant across iterations.
            Xt_omega = t.mv(self.XcT, omega)
        for _ in range(max_iter):
            delta = t.mv(self.Xc, w).clamp_min(self.tiny)
            num = t.mv(self.XcT, omega_y / delta)
            if poisson:
                data_den = Xt_omega
            else:
                # NB denominator data term X^T(ω (y+θ)/(θ+δ)) depends on δ.
                data_den = t.mv(self.XcT, omega * (self.yt + theta) / (theta + delta))
            W = w.view(num_rgrs, num_runs)
            norms = (W * W).sum(1).sqrt()
            pen = (lam * (W / norms.clamp_min(self.tiny).view(-1, 1))).reshape(-1)
            den = (data_den + pen).clamp_min(self.tiny)
            w_new = (w * num / den).clamp_min(pmin)
            rel = (w_new - w).norm() / w.norm().clamp_min(1e-14)
            w = w_new
            if rel.item() < tol:
                break
        return w.double().cpu().numpy()
