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

PRICE2 is a genomics pipeline that detects actively translated ORFs from multiple Ribo-seq datasets using group-LASSO penalized Poisson regression. The pipeline has two main phases:

**Data collection** (`data_collector.py`): Processes BAM files to estimate per-dataset cleavage and coverage models (`cleavage_model.py`, `coverage_model.py`), maps reads to genomic loci, and generates ORF candidates. All intermediate results are persisted to a SQLite database (`price.db` in the working directory) to allow resumable runs.

**Parallel deconvolution** (`orf_activity_estimator.py`): Spawns worker processes (via `pebble` with `forkserver` — do not change to `fork`) to process each locus independently. Each worker loads its locus from SQLite, applies coverage and deconvolution filters, builds equivalence groups (reads sharing the same ORF compatibility set), then solves the group-LASSO optimization using IRLS-Huber robust regression with L-BFGS-B.

**Core data structures:**
- `Locus` (`locus.py`, ~1800 lines): Aggregates overlapping transcripts, generates ORF candidates, runs all filtering and deconvolution logic.
- `ReadGeneratingRegion` (RGR): A candidate translated region (ORF or NOISE type).
- `EquivalenceGroup`: Reads compatible with the same ORF set — the rows of the sparse design matrix fed to the optimizer.
- `CleavageModel` / `CoverageModel`: Per-dataset learned distributions used to compute per-read per-ORF likelihoods.

**Output**: Per-locus TSV/GTF files at each filtering stage under `regions_activities/`, then aggregated TPM-normalized output (`final_orfs_tpm.tsv`, `final_regions_tpm.tsv`).

## Key Conventions

**Coordinates:** Always 0-based, half-open intervals. GTF input is 1-based and must be converted on load. Multi-exonic regions are stored in chromosome order — negative-strand regions are therefore in reverse translation order; account for this when computing reading frames.

**Numerical stability:** Use `pseudo_min = 1e-14` to guard against `log(0)` in the Poisson likelihood. Do not remove or reduce this.

**SQLite concurrency:** Workers use `filelock` for safe concurrent access. Do not access SQLite from workers without holding the appropriate lock.

**Performance-sensitive code:** `locus.py` deconvolution uses `scipy.sparse` CSR matrices and BLAS-backed operations. Keep numerical code vectorized (numpy/scipy); avoid Python loops over reads or positions.

**Testing:** Tests live in `tests/`. Currently minimal coverage — do not apply TDD (this needs human supervision). Do not modify tests to make code pass — only fix tests if there is a genuine error in the test itself.

If anything in the code is unclear, ask the user what it is supposed to do rather than guessing.

## Code Style

- PEP 8, 88-character line limit
- Type hints on all public functions and methods
- NumPy-style docstrings on all public APIs
- Use classes for core domain objects, functions for utilities
- Use pytest functions (not test classes); use fixtures to share resources between tests
- No code in `__init__.py` files
- Prefer widely used packages (numpy, scipy, pandas, scikit-learn); avoid obscure dependencies
