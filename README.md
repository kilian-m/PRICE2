<div align="center">
<h1> PRICE 2 </h>
</div>

## About
PRICE 2 is a tool to detect translated open reading frames based on multiple ribo-seq or QTI-seq datasets. It builds on ideas introduced in [PRICE](https://github.com/erhard-lab/gedi/wiki/Price) and the well established [generative model for isoform deconvolution](https://academic.oup.com/bioinformatics/article/25/8/1026/324948R).

## User Guide
### Installation
PRICE 2 is available on the [bioconda](https://bioconda.github.io/) channel and can be installed with conda/mamba/pixi:

```bash
conda install -c bioconda -c conda-forge price2
```

Alternatively, install the latest development version from source:

```bash
git clone https://github.com/kilian-m/PRICE2.git
cd PRICE2
conda env create -f price2/price2.yml
conda activate price2
pip install --no-deps .
```

### Data preparation
PRICE 2 requires a reference annotation in GTF format, a reference genome in FASTA format and one or more mapped ribo-seq datasets in BAM format. Before you apply PRICE 2 you should check the quality of your fastqs, remove rRNA reads and clip adapters. Use the [STAR aligner](https://github.com/alexdobin/STAR) to map the reads.

PRICE 2 relies on the untemplated nucleotide additions ("RT nucleotides") added by the reverse transcriptase. It supports two STAR read-end alignment modes, selected with the `align_ends_type` config option (default `local`):

- **`local`** (default) — map with `--alignEndsType Local`. The RT nucleotide is soft-clipped and read directly:
  ```bash
  --outSAMtype BAM SortedByCoordinate --alignEndsType Local --outSAMattributes nM MD NH
  ```
- **`endtoend`** — map with `--alignEndsType EndToEnd` (the mode most other ribo-seq ORF callers use, so you may already have BAMs of this type). Here the RT nucleotide is not soft-clipped but recovered from the 5'-terminal mismatch, so set `"align_ends_type": "endtoend"` in the config:
  ```bash
  --outSAMtype BAM SortedByCoordinate --alignEndsType EndToEnd --outSAMattributes nM MD NH
  ```
  Results are close to a `local` run but not identical: RT-carrying reads whose extra mismatch exceeds STAR's `--outFilterMismatchNmax` are dropped by the mapper.

The **`MD` tag is required** — always include it in `--outSAMattributes`. In `endtoend` mode PRICE 2 reads the RT nucleotide from it; if it is missing PRICE 2 warns and detects no untemplated additions.

The BAMs must also be **coordinate-sorted and indexed**: PRICE 2 fetches reads by genomic region, both when estimating the per-dataset models and when mapping reads to loci. The STAR options above already sort the output, so it only remains to index each file:

```bash
samtools index sample.bam   # writes sample.bam.bai next to it
```

### Running PRICE 2
PRICE 2 exposes many parameters but in most use cases the defaults should work well. We advice to set the parameters in a json config file (alternatively they can be set when calling PRICE 2 from the CLI).

```bash
price2 --config config.json
```

A minimal `config.json`:

```json
{
    "base_dir": "/path/to/analysis",
    "gtf_path": "/path/to/annotation.gtf",
    "fasta_path": "/path/to/genome.fa"
}
```

The required parameters are:
- `base_dir`: the directory where PRICE 2 will write its output and intermediate files. The following subdirectories should be present or will be created in `base_dir`:
        
        base_dir/
        ├── o_dir/      # output
        ├── w_dir/      # working / SQLite database
        ├── bam_dir/    # mapped Ribo-seq BAM files
        └── logs/       # log files
- `gtf_path`: path to the reference annotation in GTF format
- `fasta_path`: path to the reference genome in FASTA format

other frequently used parameters are:
- `bam_ids`: a list of identifiers for the BAM files. The BAM files should be named `{bam_id}.bam`, be located in `bam_dir` and each have its index (`{bam_id}.bam.bai`) alongside. If not provided, all BAM files in `bam_dir` will be used.
- `align_ends_type`: the STAR read-end alignment mode of your BAMs, `"local"` (default) or `"endtoend"` (see [Data preparation](#data-preparation)). Both require the `MD` tag in the BAMs.
- `processes`: the number of worker processes used for parallelization (default `80`). This is a fixed default, not the machine's core count, so set it explicitly to match the host you are running on.
- `timeout`: the wall-clock budget for one locus, in seconds **per Ribo-seq run** (default `180`). A locus is abandoned after `timeout` × (number of runs) seconds — 900 s for five datasets — and listed in `o_dir/regions_activities/failed_loci.txt`. It scales with the sample count because a locus is solved for all datasets at once.
- `warm_start`: continue an interrupted run instead of starting over (default `true`, see [Resuming an interrupted run](#resuming-an-interrupted-run)). Set it to `false` to force a clean run, which wipes `w_dir` and `o_dir` first.

### Resuming an interrupted run
A run that is cut short — by a wall-clock limit, a node failure or a `Ctrl-C` — is picked up where it stopped when you simply start it again with the same config. That is what `warm_start` does, and it is on by default. Each stage resumes at its own granularity:

- **data collection**: per Ribo-seq run for the models and the read mappings, per locus for the locus skeletons;
- **multimapping EM**: at the last checkpointed iteration, re-running only the loci of that iteration that had not finished, and skipping straight to the final pass when the EM had already converged;
- **final deconvolution**: at the loci not yet listed in `w_dir/processed_loci.txt`.

Before anything is written, the output files are reconciled with that list: a half-written trailing line is dropped, and so is every row belonging to a locus that is re-run, so no result is duplicated or lost. A resumed run reproduces the output of an uninterrupted one.

PRICE 2 records a fingerprint of your configuration in `price.db` to decide what may be reused:

- change an option that only affects the deconvolution (a filter, `lam`, an `export_*` selection, …) and the collected data is kept while the deconvolution starts over;
- change an option that decides the *content* of the database (`gtf_path`, `fasta_path`, `bam_dir`, `bam_ids`, `align_ends_type`, `high_quality_runs_only`, `multimap_em`) and the run stops with an error rather than silently mixing incompatible data or discarding a collection that can take days. Point the new configuration at a different `base_dir`, or set `warm_start` to `false` to collect it again.

Paths are compared by file name, so moving or staging an analysis directory elsewhere does not invalidate it.

### Output
All results are written to `o_dir`. The main tables live in `o_dir/regions_activities/`:

- `orfs.tsv`: one row per detected ORF, one column per Ribo-seq run, holding the estimated activity (reads per nucleotide).
- `orfs_tpm.tsv`: the same table with activities normalised to Transcripts Per Million per run.
- `orfs.bed`: the detected ORFs in BED format.

Both ORF tables carry the metadata columns `orf_id`, `gene_id`, `transcript_id`, `locus_id`, `genomic_region` and `orf_type`; the remaining columns are the Ribo-seq runs.

The learned per-dataset models are written to `o_dir/dataset_models/` as `cleavage_models.tsv` and `coverage_models.tsv`, alongside `cleavage_models.pdf` and `coverage_models.pdf` diagnostic plots.

## How it works
PRICE 2 implements a generative model for ribo-seq read counts. Each ORF is associated with an activity for each ribo-seq dataset. Reads are considered to be generated by ORFs according to probabilities computed from ORF activities and dataset characteristics. Reads that contain the same information (i.e. same dataset, length, untemplated addition status, phase and overlapping the same ORFs) are assigned to the same equivalence group.

The core of PRICE 2 is a Poisson regression with the read counts of the equivalence groups as the target variables and the activities as the coefficients. For a sparse solution a group LASSO penalty is applied to the coefficients such that the coefficients for one ORF in multiple datasets form a group. The penalised objective is minimised with a multiplicative-update solver.

Reads that map to several loci are not discarded. An Expectation-Maximisation loop wraps the per-locus deconvolution: reads compatible with more than one locus are assigned fractionally according to the current activity estimates (E-step), the per-locus deconvolutions are re-solved on those fractional counts (M-step), and the two alternate until both converge. Loci are still solved independently, so there is no joint optimisation across the genome. Setting `multimap_em` to `false` turns the loop off, and multimapping reads (`NH` > 1) are then discarded rather than counted at full weight in every locus they align to. The per-dataset cleavage and coverage models are always estimated from uniquely mapping reads only.

## License
PRICE 2 is released under the GNU General Public License v3.0 or later. See [LICENSE](LICENSE) for the full text.