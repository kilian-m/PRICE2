# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate environment (required before running anything)
conda activate price2

# Run the tool
python price2/price.py --config config.json

# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_cleavage_model.py -v
```

## Architecture

PRICE2 is a genomics pipeline that detects actively translated ORFs from multiple Ribo-seq datasets using group-LASSO penalized Poisson regression. `price.py` drives the run as a list of `pipeline.Stage`s in two phases:

**Data collection** (`data_collector.py`): estimates per-dataset models from the BAMs (`cleavage_estimator.py` / `coverage_estimator.py` fit the frozen `cleavage_model.py` / `coverage_model.py`; `bam.py` holds the BAM conventions, `plotting.py` the diagnostic plots), builds loci from the annotation, maps reads to them and, with `multimap_em`, spills multimapping alignments for the linkage index of the `multimap` package (`keys`, `spill`, `index`, `linkage`, `state`, `prepared`, `em`). Everything is persisted to `price.db` through `database.py` (the only place with SQL DDL, connections and blob codecs); `layout.py` names every file of a run; `run_state.py` fingerprints the configuration so a run can be resumed.

**Parallel deconvolution** (`orf_activity_estimator.py`): fans the loci out over a `pebble` pool started with `forkserver` (do not change to `fork`). Each worker runs `process_loc` for a `LocusJob`: `orf_candidates.py` generates and classifies the ORF candidates, `read_routing.py` loads the reads, decides which regions each read is compatible with and builds the equivalence groups and the sparse design matrix, `locus.py` applies the filters and orchestrates the solves, `solver.py` is the single solve entry point (multiplicative updates or L-BFGS-B) over the objectives in `likelihood.py`, and `export.py` renders the rows the parent writes. With `multimap_em`, light M-steps and `multimap.e_step` alternate before the final full pass.

**Core data structures:**
- `Locus` (`locus.py`): the transcripts of one genomic unit, its RGR candidates, equivalence groups and activity matrix; every attribute a worker fills in is declared in `_init_state`.
- `ReadGeneratingRegion` (RGR, `genomic_features.py`): A candidate translated region (ORF or NOISE type).
- Equivalence groups (`equivalence_groups.py`): reads compatible with the same ORF set — the rows of the sparse design matrix fed to the optimizer. `make_equivalence_groups` yields their geometry (`{run: {key: length}}`); `read_routing.ReadRouting` is the sole representation once the reads are routed (response, design matrix, multimapping rates and RGR removal all derive from its arrays), and the light EM M-steps load it alone.
- `CleavageModel` / `CoverageModel`: Per-dataset learned distributions used to compute per-read per-ORF likelihoods.

**Output**: Per-locus TSV/GTF/BED rows under `regions_activities/` (per filtering stage with `export_all_steps`), then aggregated TPM-normalized output (`orfs_tpm.tsv`, `regions_tpm.tsv`).

## Key Conventions

**Coordinates:** Always 0-based, half-open intervals. GTF input is 1-based and must be converted on load. Multi-exonic regions are stored in chromosome order — negative-strand regions are therefore in reverse translation order; account for this when computing reading frames.

**RGR indexing and equivalence-group keys:** `Locus.rgrs` is an ordered list and an RGR's position in it is `rgr.index`, which addresses its design-matrix column block and its row of `result`. An equivalence-group key is `(cells, read_length, oua)` where each cell is the packed int `rgr.index * 12 + frame_code * 3 + covpos` (`equivalence_groups.pack_cell`; `ReadRouting` stores the same cells per row and per read). Never drop RGRs by hand: `Locus.remove_rgrs` compacts the list and rebuilds the routing.

**Numerical stability:** Use `pseudo_min = 1e-14` to guard against `log(0)` in the Poisson likelihood. Do not remove or reduce this.

**Output files:** Workers never write to `o_dir`; they hand their rendered rows back and the parent process (`ORFActivityEstimator._record`) is the only writer of the output tables, `performance_measurements.tsv` and the per-locus progress (the `progress` table of `price.db`, via `run_state.ProgressRecorder`). Workers do write their own EM state to SQLite through `price2.database.connect` (WAL mode, busy timeout).

**Process model:** The per-run model estimation and the read mapping use `multiprocessing` pools started with `fork`, so the annotation, the models and the loci are inherited without pickling; the spill collapse, the multimap index build and the deconvolution use `forkserver` (numba's JIT state and SQLite handles are not fork-safe). The mapping pool and the multimap index pool are created before the parent opens `price.db`, so no SQLite connection crosses a fork. `price.py` pins the BLAS/OpenMP thread counts to one before numpy is imported; the parallelism is across loci. A driver script that starts the pipeline must guard its top level with `if __name__ == "__main__":`, because a `forkserver` worker re-imports the main module.

**Tuned defaults:** The non-obvious defaults in `config.py` (`lam`, `worker_max_tasks`, `dispatch_order`, `em_max_iter`, `irls_stop_on_active_set`, the `likelihood_ratio_*` tolerances) were measured; the reason is in the comment at each field. Do not retune them without a new measurement.

**Performance-sensitive code:** `locus.py` deconvolution uses `scipy.sparse` CSR matrices and BLAS-backed operations. Keep numerical code vectorized (numpy/scipy); avoid Python loops over reads or positions.

