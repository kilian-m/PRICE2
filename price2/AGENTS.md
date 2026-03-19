# AGENTS.md - PRICE2 Project Guide

## Project Overview

**PRICE2** is a bioinformatics tool that detects translated regions (translons) from multiple mapped Ribo-seq samples (BAM files). It implements a generative model of Ribo-seq data and uses group-LASSO penalized maximum likelihood estimation to find active translons, similar to isoform deconvolution from RNA-seq data.

**Python:** 3.12+  
**Package Manager:** conda (environment defined in `price2.yml`)

## Development Strategy

- Use a **test-driven development** strategy: write failing tests first, then implement solutions.
- Run the tests and ensure that they fail prior to generating any solutions.
- Write code that passes the tests.
- **IMPORTANT:** Do not modify the tests simply so that the code passes. Only modify the tests if you identify a specific error in the test.
- if code in unclear to you ask me what it is supposed to do
- **IMPORTANT:** I have not focused on testing the code before. This process needs human supervision. For now do not apply this strategy. Ignore testing.


## Module Structure

```
price2/
├── __init__.py                 # Keep empty
├── price.py                    # Main entry point (CLI)
├── config.py                   # Configuration dataclass (JSON-based)
├── reference_annotation.py     # GTF reference annotation parsing
├── fasta_reader.py             # Genome FASTA reading
├── genomic_region.py           # Genomic coordinate operations
├── genomic_features.py         # ReadGeneratingRegion, Transcript
├── ribo_seq_alignment.py       # Ribo-seq alignment handling
├── ribo_seq_run.py             # Ribo-seq run representation
├── cleavage_model.py           # Cleavage site estimation
├── coverage_model.py           # Coverage profile modeling
├── equivalence_groups.py       # Equivalence group construction
├── data_collector.py           # Data collection orchestration
├── locus.py                    # Locus class (core deconvolution logic)
├── orf_activity_estimator.py   # Parallel ORF activity estimation
└── AGENTS.md                   # This file
```

## Key Dependencies

```yaml
# Managed via conda (price2.yml)
- htseq          # Genomic interval operations, BAM/GTF I/O
- numpy          # Numerical operations
- scipy          # Optimization (L-BFGS-B), sparse matrices (csr_matrix), statistical tests (chi2)
- pandas         # Data structuring and output
- pebble         # Process pool with per-task timeouts
- filelock       # File-based locking for SQLite access
- tqdm           # Progress bars
- psutil         # Memory monitoring
- numba
```

Prefer widely used packages (numpy, pandas, scipy, scikit-learn). Avoid obscure packages from GitHub.

## Common Commands

```bash
# Run the tool
python price2/price.py --config config.json

# Run tests
pytest tests/

# Run a specific test file
pytest tests/test_cleavage_model.py

# Activate the conda environment
conda activate price2
```

## Code Style & Conventions

### General Principles
- Follow **PEP 8**
- Maximum line length: **88** characters (black default)
- Use **type hints** for all public functions and methods
- Use **NumPy-style docstrings** for all public APIs
- Write clean, modular code — prefer shorter functions/methods over longer ones
- Use **classes** for core domain objects, **functions** for utilities
- Use **pytest** for testing with functions (not classes); use fixtures to share resources
- Do not include any code in `__init__.py` files

### Docstring Example

```python
def estimate_activities(
    coverage: np.ndarray, model: np.ndarray, lam: float
) -> np.ndarray:
    """Estimate ORF activities via penalized maximum likelihood.

    Parameters
    ----------
    coverage : np.ndarray
        Observed read coverage, shape (n_positions,).
    model : np.ndarray
        Expected coverage model per ORF, shape (n_orfs, n_positions).
    lam : float
        Regularization strength for group-LASSO penalty.

    Returns
    -------
    np.ndarray
        Estimated activity for each ORF, shape (n_orfs,).
    """
```

### Performance-Sensitive Code
- The deconvolution core (`locus.py`) uses **`scipy.sparse` CSR matrices** and BLAS-backed `X.T @ r` operations
- Prefer pure numpy/scipy vectorized operations for all numerical code
- Current parallelism uses pebble ProcessPool with forkserver start method; any approach that works is acceptable

### Error Handling
- Validate inputs early
- Provide clear, actionable error messages
- Log errors appropriately

## Domain-Specific Knowledge

### Coordinate System
- The project uses **0-based, half-open** genomic coordinates throughout
- Multi-exonic regions are stored **in chromosome order** (i.e., negative-strand translons are in reverse of the translation direction)
- Be mindful of strand when manipulating genomic intervals

### Core Concepts
- **Translon / ORF:** An open reading frame that may be actively translated
- **Ribo-seq:** Ribosome profiling — sequencing of ribosome-protected RNA fragments
- **Deconvolution:** Separating overlapping ORF signals from shared read coverage, analogous to isoform quantification in RNA-seq
- **Locus:** A genomic region containing one or more overlapping transcripts, processed independently
- **Equivalence Group:** A set of reads that are compatible with the same set of ORFs
- **ReadGeneratingRegion (RGR):** A candidate translated region that could generate observed reads
- **Cleavage model:** Model of ribosome cleavage site positions relative to the A-site
- **Coverage model:** Expected read coverage profile along an ORF

### Data Flow
1. Parse reference annotation (GTF) and genome (FASTA)
2. Collect Ribo-seq runs (BAM files) and compute cleavage/coverage models
3. Map reads to genomic loci and generate ORF candidates
4. Store intermediate data in SQLite database (`price.db` in working directory)
5. Apply filtering steps (coverage filter, deconvolution filter)
6. Run parallel ORF deconvolution per locus (group-LASSO penalized Poisson likelihood)
7. Apply likelihood ratio filter
8. Output active translons

### Configuration
All parameters are managed via a `Config` dataclass loaded from a JSON file. Key parameters include regularization strength (`lam`), filtering thresholds, parallelism settings, and start/stop codons.

## Common Pitfalls

1. **Coordinate systems:** Always use 0-based half-open intervals. Never mix with 1-based coordinates from GTF without conversion.
2. **Strand handling:** Negative-strand regions are in chromosome order, not translation order. Be careful when computing reading frames.
3. **Numerical stability:** The optimization uses pseudo-minimum values (`pseudo_min = 1e-14`) to avoid log(0). Respect this in any numerical code.
4. **Memory:** Whole-genome processing can be memory-intensive. The project uses SQLite for intermediate storage and per-worker memory limits.
5. **Multiprocessing start method:** The project uses `forkserver` — do not switch to `fork` as it causes issues with SQLite.
6. **SQLite concurrency:** File locks (`filelock`) are used for safe concurrent SQLite access.

## Testing

### Test Structure
- Tests live in `tests/` at the project root
- Test data (BAM, GTF, SAM files) is in `tests/`
- Use pytest functions, not test classes
- Use pytest fixtures to share resources between tests

### Running Tests
before running any tests activate the conda environment 'price2'
```bash
pytest tests/
pytest tests/test_cleavage_model.py -v
```

## Validation Checklist

Before considering any component complete:
- [ ] Tests written and passing (TDD: tests written first)
- [ ] Function/method has type hints
- [ ] NumPy-style docstring is complete
- [ ] Error cases are handled
- [ ] Coordinate conventions are respected (0-based half-open)
- [ ] Performance is acceptable
