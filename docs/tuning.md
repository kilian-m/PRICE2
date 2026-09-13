# Why the defaults are what they are

The tuning history behind the non-obvious defaults of `price2.config.Config`.
Each option is documented at its field; this file keeps the measurements and
the reasoning that would otherwise crowd the code.  The `playground/…`
directories named below live in the separate `price2-analysis` repository,
not in this one.

## `worker_max_tasks = 1000`

A fresh worker process costs about 2 CPU-seconds before it does any useful
work: the numba cache load, the imports, and a cold interpreter and allocator
on its first locus.  At `worker_max_tasks = 1` that was charged to every one of
the ~39K loci in every EM iteration.  Reusing workers amortises it (measured
`0.70 + 1.97 / worker_max_tasks` CPU-seconds per locus) while still recycling
processes often enough to bound RSS growth from the occasional very large
locus.

The value was then retuned from 50 to 1000.  The `0.70` per-locus figure
predates the routing-based light M-step of the multimapping EM, which reduced
the per-locus work of an EM iteration to a few milliseconds (a cached bincount
plus a small multiplicative-update solve).  Against that, recycling every 50
loci makes a worker spend far more time respawning than computing: at ~13K
slot loci per light iteration that is ~260 respawns of ~2 CPU-s each, about
13 s of wall time per iteration.  Measured on the Yewdell-scale three-BAM set
(`playground/pruning`), dropping the recycling cut the whole-deconvolution
wall time from 964 s to 454 s (2.1×, identical call set).  1000 still recycles
~13× per light iteration and ~48× in the final full pass, enough to bound RSS
from heavy loci, while making the respawn cost negligible against the light
pass.

## `inner_solver = "mu"` and `lam = 10`

The multiplicative (weighted Richardson-Lucy + group-LASSO) updates converge
to the true optimum that scipy's L-BFGS-B stalls short of.  Two consequences:

1. The call set changes versus the legacy loose L-BFGS-B: the converged solve
   is sparser, ~19 % fewer ORFs on Yewdell.  `lam` was recalibrated for it.
2. On the CPU it is slower than L-BFGS-B, because tight convergence costs
   iterations (`playground/deconvolution_performance`).

The penalty went from 100 to 10 after the switch: the tighter solver applies
the penalty cleanly, and an AIC/BIC scan (`playground/lambda_aic`) shows that
λ = 100 over-penalises (ΔAIC ≈ 5e3 on tiny chr22); the BIC-optimal value is
λ = 10 and the AIC-optimal λ = 3.  λ = 10 is the conservative (BIC) choice; it
should be confirmed on a production-scale dataset.

## GPU offload (`mu_gpu`, `mu_broker`) off by default

In the tests run so far the GPU did not speed the pipeline up meaningfully.
The individual deconvolution solves are small (sparse mat-vecs over a few 10k
rows), so kernel-launch and host-device transfer overhead eats most of the
per-solve gain, and the CPU worker pool already parallelises across loci; the
end-to-end wall time barely moves, while the GPU paths add CUDA contexts, VRAM
pressure and, for the broker, a shared-memory IPC layer
(`playground/deconvolution_performance/gpu_broker`).  Turn them on only after
re-measuring on your own data.

Both paths need an NVIDIA GPU with driver and a PyTorch built against CUDA
(developed with torch 2.5.1+cu121).  torch is deliberately not a declared
dependency, so install it separately, e.g.

    pip install torch --index-url https://download.pytorch.org/whl/cu121

When torch or CUDA is unavailable the solves silently fall back to the NumPy
CPU multiplicative updates.

## `irls_stop_on_active_set = True`

`irls_huber_tol` measures the L2 change of the whole weight vector, which
shrinks only geometrically (the group-LASSO drags many coordinates slowly
toward `pseudo_min`) and so rarely fires before `irls_huber_max_outer` on
read-dense loci.  Stopping instead once the set of ORFs above
`deconvolution_filter_min_activity`, the ones that survive filtering, is
unchanged for `irls_active_patience` consecutive outer iterations ends the loop
far earlier: the reported call set converges long before the raw parameter
norm.

## `em_max_iter = 30`, `em_tol = 1e-3`

The multimapping EM converges only linearly: the L1 read mass reassigned per
E-step shrinks by a roughly constant factor of ~0.8 per iteration, so reaching
`em_tol` takes ~18-20 iterations on tested data.  The tolerance is meant to end
the loop and the cap is a backstop, so the cap must be high enough that the
tolerance governs: a cap of 10 truncated the loop at ~5× `em_tol`, before it
had converged (`playground/em_stopping_criterion`).

## `em_prune_after_first_mstep = True`

ORF candidates that the first light M-step finds inactive in every run seldom
revive in later M-steps, so dropping them lets every later EM iteration and the
final full pass work on a smaller design matrix; when the pruning fires, the
locus's read routing is rebuilt before its prepared state is persisted.  It is
a runtime heuristic that changes the call set slightly (a pruned ORF cannot
come back), which is why it stays a separate switch for A/B testing.

## `distribution = "poisson"`

The negative-binomial likelihood (`"nb"`, variance `μ + μ²/θ` with the global
dispersion `nb_dispersion`) absorbs the overdispersion typical of Ribo-seq
counts and threads through every solve site: the deconvolution filter, the
group-LASSO IRLS-Huber deconvolution, the weighted likelihood-ratio filter and
the final activity estimation, on both the CPU and GPU multiplicative-update
paths.  As `θ → ∞` it collapses to the Poisson, so `"nb"` with a large
`nb_dispersion` reproduces the Poisson results.  The Poisson stays the default
as the classic PRICE2 model.
