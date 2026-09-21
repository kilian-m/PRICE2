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

The Poisson updates run on the counted rows only.  On real loci 94-96 % of
the equivalence-group rows carry no read, and for the Poisson objective such
a row is linear in the activities, so every zero-count row folds into the
denominator vector `Xᵀω` — one full mat-vec per Huber reweight — and the
inner loop iterates over the 4-6 % of rows with counts
(`mu_solver.Design` / `mu_solver.Collapsed`).  The iterates are identical to
the full update's; the negative binomial keeps every row.  The design matrix
of a locus is built once per routing and stored with it for the EM passes.

## `mu_kernel = "numba"`

One multiplicative update is two sparse mat-vecs and a dozen element-wise
operations.  As a chain of numpy calls each carries its own dispatch cost and
temporary, and on the small systems — the per-run refits of the
likelihood-ratio filter, the collapsed systems of read-poor loci — that
overhead was most of an iteration.  The default kernel keeps scipy's C
mat-vec (a compiled loop over a CSR gather is slower than it) and does the
element-wise work in two numba passes; the arithmetic is the same, down to
the order in which the group norms add their squares, so the iterates are
bit-identical to the numpy loop's and `"numpy"` is only kept for A/B runs.
Measured on random systems the kernel is 2-3× faster below ~50k nonzeros and
neutral above.  The deconvolution filter's nested systems, one per stop-codon
group, are solved for all runs as the columns of one compiled loop instead
of one scipy solve per run.

## `timeout_cap = 3600`

`timeout` is per run because a locus is solved for every run at once, but at
a hundred runs the product is hours per locus, and a locus that needs them
is a locus that will not finish anyway; the cap bounds the budget of any
locus.  With the collapsed updates and the per-run filter the slowest chr22
locus of the 48-sample panel takes under five minutes, so an hour is
generous.  Abandoned loci are named in the log and in `failed_loci.txt`.
`performance_measurements.tsv` records each worker's peak RSS
(`max_rss_mb`), which grows with the number of samples — size `processes`
so that the largest fits the machine.

## `dispatch_order = "database"`

Dispatching the loci largest first (`"largest"`, by the bytes of stored
reads) looked like the obvious way to shorten a pass's tail, where the last
heavy loci run alone.  Measured on the 6-BAM genome it does the opposite: a
light EM M-step over the 13k slot loci takes 26-32 s in database order and
55-120 s largest first, and the 48-sample final pass gained nothing.  With
80 workers all streaming their largest matrices at once the pass is bound by
memory bandwidth; database order interleaves heavy and light loci and keeps
the bandwidth shared.  The option stays for machines where that is not the
bottleneck.

## `likelihood_ratio_run_tol = 1e-6`, `likelihood_ratio_ll_tol = 1e-3`

The likelihood-ratio filter's refits are per run (`docs/deconvolution.md`
§5): a reduced model is re-solved only in the runs where clamping the
candidate lowered the log-likelihood by more than `likelihood_ratio_run_tol`
nats, and the refit stops as soon as the drop verdict is settled or, for a
keep, once the log-likelihood gains less than `likelihood_ratio_ll_tol` nats
between two checks ten updates apart.  With `0` for both the filter is the
unrestricted, fully converged test.  Note that the restricted refits converge
where the unrestricted ones stopped early — the relative change of a whole
activity vector barely moves when one ORF is pinned — so a few borderline
verdicts differ from those of earlier versions (0.4 % of the ORFs on the chr22
test set); with a tight `mu_inner_tol` the old and the new filter agree exactly
(`price2-analysis/playground/price2_performance/REPORT.md`).

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
the final activity estimation.  As `θ → ∞` it collapses to the Poisson, so `"nb"` with a large
`nb_dispersion` reproduces the Poisson results.  The Poisson stays the default
as the classic PRICE2 model.
