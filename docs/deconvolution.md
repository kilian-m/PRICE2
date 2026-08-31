# The PRICE2 deconvolution procedure

Five nested views, from the whole run down to the individual solves.
All node labels name the function that implements the step.

---

## 1. Whole run — the EM outer loop

`price.py :: run_pipeline` / `_run_em_deconvolution`

```mermaid
flowchart TD
    START(["price.py --config run.json"]) --> DC["<b>DataCollector</b><br/>cleavage + coverage models per run<br/>reads → loci → ORF candidates<br/>persisted to price.db"]
    DC --> MMQ{"config.multimap_em"}

    MMQ -->|"false"| CLASSIC["<b>unique-reads single pass</b><br/>run_orf_deconvolution()<br/>multimapping reads (NH &gt; 1) discarded<br/>at collection and again at read load"]
    CLASSIC --> TPM

    MMQ -->|"true"| IDX["<b>build_multimap_index</b><br/>keep reads with ≥ 2 in-locus slots<br/>slot = locus_id + group_key<br/>collapse identical slot-sets → MMG"]
    IDX --> RESET["<b>reset_em_state</b><br/>weights[0] = slot baseline<br/>i.e. full counts = classic behaviour"]
    subgraph EM ["EM outer loop — it = 0 … em_max_iter-1"]
        direction TB
        M["<b>light M-step</b> — parallel fan-out<br/>only over loci that carry slots<br/>1 Huber reweight, no pruning<br/>emits activities[it] and λ[it]"]
        M --> E["<b>E-step</b> — single global reduce<br/>multimap.e_step()<br/>emits weights[it+1]"]
        E --> CONV{"Δ &lt; em_tol<br/>Δ = Σ|w_new − w_old| / Σ w_new"}
        CONV -->|"no, it &lt; em_max_iter"| M
    end

    RESET --> M
    CONV -->|"yes — converged"| FINAL["<b>final full M-step</b><br/>all loci, converged weights<br/>full deconvolution + LRT<br/>+ activity estimation + export"]
    FINAL --> TPM["generate_tpm_output<br/>final_orfs_tpm.tsv"]
    TPM --> DONE(["done"])

    classDef em fill:#eef6ff,stroke:#4a7fb5
    classDef hot fill:#fff4e6,stroke:#c98a2b
    class M,E,CONV em
    class FINAL,CLASSIC hot
```

The **M-step** is the ordinary per-locus deconvolution, unchanged in shape — only its
response vector `y` carries fractional counts. The **E-step** is the only cross-locus
coupling, and it is a per-read normalisation, not a joint optimisation. Loci with no
multimapping slots cannot change between iterations, so the light passes skip them and
they are computed exactly once, in the final pass.

Defaults: `em_max_iter = 30` (backstop), `em_tol = 1e-3` (the normal exit),
`em_huber_steps = 1`.

---

## 2. One locus — the M-step worker

`orf_activity_estimator.py :: process_loc`, running in a `pebble` forkserver worker.

```mermaid
flowchart TD
    IN(["process_loc(locus_id, em_iteration, em_final)"]) --> RUNS["load runs from price.db"]
    RUNS --> CACHE{"EM mode and<br/>prepared_loci cache hit"}

    CACHE -->|"hit"| READS2["get_reads_from_db"]

    CACHE -->|"miss / classic"| SKEL["load Locus skeleton"]
    SKEL --> RGR["<b>build_rgrs</b><br/>enumerate ORF candidates on every<br/>transcript + one NOISE RGR per locus"]
    RGR --> READS["get_reads_from_db"]
    READS --> WFR["<b>make_well_fitting_reads</b><br/>per-RGR read counts, unweighted"]
    WFR --> F1["<b>coverage filter</b><br/>drop ORF if max over runs of<br/>well-fitting reads / length ≤ 0.1"]
    F1 --> F2["<b>deconvolution filter</b><br/>group ORFs by stop codon → split into<br/>splice-compatible optimisation groups →<br/>nested Poisson solve per run →<br/>drop ORF if activity &lt; 0.1 in every run"]
    F2 --> EGS["<b>make_equivalence_groups</b><br/>reads sharing one ORF-compatibility set,<br/>read length and 5' untemplated-addition<br/>state collapse into one EG = one matrix row"]
    EGS --> SAVE["save_prepared_locus<br/>only at EM iteration 0"]
    SAVE --> READS2

    READS2 --> WARM{"EM mode"}
    WARM -->|"yes"| WS["<b>set_warm_start</b> — only when it &gt; 0<br/>activities[it-1], keyed by rgr.id<br/><br/><b>load_locus_mm_data</b><br/>per-slot base and weight[it]"]
    WARM -->|"no — classic"| ASSIGN
    WS --> ASSIGN["<b>assign_reads_to_egs</b><br/>y_EG = Σ read counts<br/>multimapper contributes<br/>max(0, c − base) + weight"]

    ASSIGN --> LIGHT{"em_final"}

    LIGHT -->|"false — light pass"| DL["<b>deconvolve</b><br/>max_outer = em_huber_steps = 1<br/>prune = False"]
    DL --> LAM["<b>compute_multimap_lambdas</b><br/>λ per slot = δ_EG / length_EG"]
    LAM --> WRITE["write_locus_em_output<br/>activities[it] + λ[it] → price.db"]
    WRITE --> STOP(["return — E-step consumes λ"])

    LIGHT -->|"true — classic or final pass"| DF["<b>deconvolve</b> (see §3)<br/>full IRLS-Huber group-LASSO<br/>+ post-solve pruning"]
    DF --> LRT["<b>likelihood_ratio_filtering</b> (see §5)"]
    LRT --> EST["<b>estimate_activities</b><br/>unregularised Poisson MLE,<br/>iteratively drop ORFs below<br/>rgr_min_activity until stable"]
    EST --> EXP["export TSV / GTF / BED<br/>+ processed_loci.txt bookkeeping"]
    EXP --> OUT(["return"])

    classDef filt fill:#f3f0ff,stroke:#7a5cc4
    classDef solve fill:#fff4e6,stroke:#c98a2b
    class F1,F2,LRT filt
    class DL,DF,EST solve
```

Everything above `assign_reads_to_egs` depends only on **unweighted** reads, so it is
identical in every EM iteration. That is exactly the state cached in `prepared_loci`
and reused from iteration 1 onward — ORF generation, both filters and the EG DAG build
are the dominant per-locus cost.

---

## 3. The main deconvolution — IRLS with Huber weights

`locus.py :: deconvolve`

```mermaid
flowchart TD
    A(["deconvolve(config, runs, max_outer, prune)"]) --> B["<b>to_sparse_args</b> → egs_to_sparse<br/>row = one EG × run pair<br/>X[row, rgr·n_runs + run] =<br/>length · cleavage(len, frame, oua) · coverage(pos)<br/>y[row] = EG read count"]
    B --> C["w ← self.result if warm-started,<br/>else all ones"]

    subgraph IRLS ["IRLS outer loop — up to irls_huber_max_outer = 10"]
        direction TB
        D["<b>δ = X w</b>"]
        D --> H["<b>Huber weights</b><br/>r = (y − δ) / √v<br/>v = δ (Poisson) or δ + δ²/θ (NB)<br/>ω = min(1, c / |r|), c = irls_huber_c"]
        H --> S["<b>weighted group-LASSO solve</b><br/>min over w ≥ pseudo_min of<br/>Σ ω_i (δ_i − y_i log δ_i) + lam · Σ_g ‖w_g‖₂<br/>group g = one RGR across all runs"]
        S --> T{"stop?"}
        T -->|"rel change ≥ tol<br/>and active set still moving"| D
    end

    subgraph INNER ["inner solver — config.inner_solver"]
        direction TB
        MU["<b>mu</b> (default) — multiplicative update<br/>w ← w · Xᵀ(ω y / δ) / (Xᵀω + lam · w/‖w_g‖)<br/>majorisation-minimisation, keeps w ≥ 0<br/>CPU numpy · per-worker GPU · shared GPU broker"]
        LB["<b>lbfgs</b> — scipy L-BFGS-B on the<br/>analytic NLL + gradient, box bounds"]
    end

    C --> D
    S -.-> INNER
    T -->|"‖Δw‖/‖w‖ &lt; irls_huber_tol = 1e-4"| STORE
    T -->|"active set unchanged for<br/>irls_active_patience iterations"| STORE

    STORE["store result, clamp ≤ pseudo_min → 0<br/>store final ω for the weighted LRT"] --> P{"prune"}
    P -->|"False — light EM pass"| RET(["return"])
    P -->|"True"| PR["<b>post-solve pruning</b><br/>per run, canonical = longest·strongest ORF<br/>drop ORF if activity &lt; max(0.1 · canonical,<br/>rgr_min_activity) in <i>every</i> run"]
    PR --> RET

    classDef hot fill:#fff4e6,stroke:#c98a2b
    class S,MU,LB hot
```

Two stopping criteria run in parallel. The relative-change metric keeps shrinking
geometrically long after the biologically meaningful quantity — the *set* of ORFs above
the activity threshold — has stabilised, so `irls_stop_on_active_set` (default `True`)
exits once that set is unchanged for `irls_active_patience` consecutive iterations.

The group-LASSO penalty is what couples the runs: an ORF is either on or off across the
whole dataset panel, and only its magnitude varies per run.

---

## 4. The E-step — fractional reassignment of multimapping reads

`multimap.py :: e_step`, one global reduce between M-step fan-outs.

```mermaid
flowchart LR
    subgraph W ["workers — M-step, in parallel"]
        L1["locus ℓ₁<br/>λ per slot"]
        L2["locus ℓ₂<br/>λ per slot"]
        L3["locus ℓ₃<br/>λ per slot"]
    end

    L1 --> GL[("group_lambdas[it]<br/>one blob per locus")]
    L2 --> GL
    L3 --> GL

    GL --> J["join λ onto MMG membership<br/>multimap_group_slots × multimap_groups"]
    J --> R["<b>responsibility</b><br/>f(r, ℓ) = λ(r, ℓ) / Σ_ℓ' λ(r, ℓ')<br/>uniform 1/n when Σλ = 0<br/>read fits no ORF anywhere"]
    R --> ACC["<b>accumulate per slot</b><br/>w(ℓ) = Σ_MMG count · f<br/>groupby reduction, not a Python loop"]
    ACC --> DELTA["Δ = Σ|w_new − w_old| / Σ w_new"]
    DELTA --> GW[("group_weights[it+1]")]
    GW --> CONV{"Δ &lt; em_tol"}

    CONV -->|"no"| NEXT["next light M-step reads weight[it+1]<br/>as its fractional response y"]
    CONV -->|"yes"| FIN["final full M-step<br/>filtering, LRT, export"]

    classDef em fill:#eef6ff,stroke:#4a7fb5
    class R,ACC em
```

`λ(r, ℓ)` is the read's *origin rate* at locus ℓ under the current model: its
design-matrix row without the geometric length factor, dotted with the current
activities — `Σ cleavage · coverage · activity` over the read's compatible ORFs, which is
exactly `δ_EG / length_EG`.

Reads that share the same set of alignment slots behave identically here, so they are
collapsed into **multimap groups** (MMGs) once, at index-build time. Iteration 0 starts
from `weight = base`, i.e. the classic full double-count, which is why the first Δ is
large — it measures the one-off removal of that double count rather than a real move.

---

## 5. Likelihood-ratio filtering

`locus.py :: likelihood_ratio_filtering`, run only on the final pass.

```mermaid
flowchart TD
    A(["likelihood_ratio_filtering(config, runs)"]) --> B["rebuild X, y on the surviving RGR set<br/>recompute Huber ω at the current fit"]
    B --> C["<b>fit full model</b><br/>weighted, <i>unregularised</i> Poisson MLE<br/>lam = 0, ω from IRLS → ll_full"]
    C --> D["order ORFs by ascending total activity<br/>weakest candidates tested first"]

    subgraph LOOP ["for each ORF, in that order"]
        direction TB
        E["<b>cheap reduced likelihood</b><br/>clamp this ORF's activity to pseudo_min,<br/>no refit → ll_reduced"]
        E --> F["<b>Wilks test</b><br/>2 (ll_full − ll_reduced) ~ χ² with<br/>df = number of runs → log p"]
        F --> G{"log p &gt; log α<br/>α = likelihood_ratio_alpha = 1e-10"}
        G -->|"yes — ORF explains nothing"| DROP["<b>drop ORF</b><br/>accept the reduced fit as the new full"]
        G -->|"no — looks significant"| REFIT["<b>refit properly</b><br/>full model: kept ORFs free<br/>reduced model: this ORF pinned at pseudo_min<br/>all dropped ORFs pinned at pseudo_min"]
        REFIT --> H{"log p &gt; log α"}
        H -->|"yes"| DROP
        H -->|"no"| KEEP["<b>keep ORF</b>"]
    end

    D --> LOOP
    LOOP --> FIN["final refit on the kept set<br/>→ self.result"]
    FIN --> RM["remove_rgrs(dropped)<br/>re-index, collapse EGs, re-slice result"]
    RM --> OUT(["→ estimate_activities"])

    classDef filt fill:#f3f0ff,stroke:#7a5cc4
    class DROP,KEEP,F filt
```

The two-tier test is a cost optimisation, not a statistical one: the clamp-and-score pass
is a couple of sparse mat-vecs, and it settles the large majority of candidates. Only
when the cheap test says *significant* does the expensive pair of constrained refits run
to confirm it — the clamped likelihood is a lower bound on the properly refit reduced
likelihood, so a cheap "not significant" verdict can never be overturned by a refit.

The Huber weights ω are carried over from the deconvolution, so an EG that was an outlier
under the robust fit also contributes less to the test statistic. NOISE RGRs are never
tested and never dropped — they are the null sink that absorbs reads no ORF explains.

---

## Notation

| symbol | meaning |
|---|---|
| `w` | activity vector, one entry per (RGR, run) pair |
| `X` | sparse design matrix, rows = (EG, run), `X[i,j] = length · cleavage · coverage` |
| `y` | observed EG read counts — *fractional* for multimappers under EM |
| `δ = X w` | fitted per-EG expected read count |
| `ω` | Huber weight on the Pearson residual of an EG |
| `λ(r, ℓ)` | origin rate of multimap read `r` at locus `ℓ` = `δ_EG / length_EG` |
| `f(r, ℓ)` | responsibility — `r`'s fractional weight at `ℓ`, sums to 1 over loci |
| `lam` | group-LASSO penalty strength, `config.lam` |
| `θ` | negative-binomial dispersion, `None` for the Poisson model |
