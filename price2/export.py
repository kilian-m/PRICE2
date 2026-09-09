"""Writing loci and their regions to the TSV, GTF and BED outputs.

Worker processes append to shared files, so every write takes the file's
lock first.  The column layout of the TSV tables is defined here and reused
by :mod:`price2.tpm` and :mod:`price2.run_state`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from filelock import FileLock

from price2.config import Config

if TYPE_CHECKING:
    from price2.genomic_features import ReadGeneratingRegion
    from price2.locus import Locus
    from price2.ribo_seq_run import RiboSeqRun

#: Metadata columns of ``orfs.tsv``; the run ids follow as activity columns.
ORF_TABLE_COLUMNS: tuple[str, ...] = (
    "orf_id",
    "gene_id",
    "transcript_id",
    "locus_id",
    "genomic_region",
    "orf_type",
)
#: Metadata columns of ``regions.tsv`` (ORFs and NOISE regions together).
REGION_TABLE_COLUMNS: tuple[str, ...] = ("region_id",) + ORF_TABLE_COLUMNS[1:]

#: Sub-directories of ``regions_activities/`` written by ``export_all_steps``,
#: one per filtering stage.
STEP_NAMES = ("all", "coverage_filtered", "deconvolution_filtered", "deconvoluted")

_GTF_SOURCE = "PRICE2"

# Colours of the BED ``itemRgb`` column, matched against the ORF type in
# order; the first key contained in the type wins.
_BED_COLOURS: tuple[tuple[str, str], ...] = (
    ("pcRNA-ORF", "255,192,203"),
    ("lncRNA-ORF", "128,128,128"),
    ("varRNA-ORF", "160,160,160"),
    ("N terminal", "0,200,200"),
    ("uoORF", "255,140,0"),
    ("doORF", "180,220,0"),
    ("iORF", "200,0,200"),
    ("uORF", "255,0,0"),
    ("dORF", "0,200,0"),
    ("cORF", "0,128,255"),
)
_BED_COLOUR_DEFAULT = "100,100,100"


def append_locked(path: str, text: str) -> None:
    """Append *text* to *path* under the file's lock."""
    with FileLock(path + ".lock"):
        with open(path, "a") as fh:
            fh.write(text)


def _output_path(prefix: str, name: str) -> str:
    """``<prefix>/<name>`` for a directory prefix, ``<prefix>_<name>`` else."""
    sep = "" if prefix.endswith("/") else "_"
    return f"{prefix}{sep}{name}"


# --------------------------------------------------------------------------- #
# One region
# --------------------------------------------------------------------------- #


def rgr_gtf(rgr: ReadGeneratingRegion, locus_id: str) -> str:
    """GTF lines of one region: an ``exon`` per interval, plus a ``CDS`` for ORFs.

    Coordinates are converted to the 1-based, closed GTF convention.
    """
    region = rgr.full_genomic_region
    strand = region.strand
    attributes = (
        f'gene_id "{rgr.transcript.gene_id}"; '
        f'transcript_id "{rgr.id}"; '
        f'loc_id "{locus_id}";'
    )
    if rgr.orf_type is not None:
        attributes += f' orf_type "{rgr.orf_type}";'
    intervals = region.intervals if strand == "+" else region.intervals[::-1]
    features = ["exon", "CDS"] if rgr.type == "ORF" else ["exon"]
    lines = []
    for interval in intervals:
        for feature in features:
            lines.append(
                f"{region.chrom}\t{_GTF_SOURCE}\t{feature}\t"
                f"{interval.start + 1}\t{interval.end}\t"
                f".\t{strand}\t.\t{attributes}\n"
            )
    return "".join(lines)


def rgr_tsv_line(rgr: ReadGeneratingRegion, locus_id: str) -> str:
    """Metadata-only TSV line of one region (the intermediate-step tables)."""
    orf_type = rgr.orf_type if rgr.orf_type is not None else ""
    return (
        f"{rgr.id}\t{rgr.transcript.gene_id}\t{locus_id}\t"
        f"{rgr.full_genomic_region}\t{orf_type}\n"
    )


def _bed_colour(orf_type: str | None) -> str:
    for key, colour in _BED_COLOURS:
        if key in (orf_type or ""):
            return colour
    return _BED_COLOUR_DEFAULT


def rgr_bed_line(rgr: ReadGeneratingRegion) -> str:
    """BED12 line of one region; ``name`` is ``<id>:<gene_id>:<orf_type>``."""
    region = rgr.full_genomic_region
    chrom_start = region.intervals[0].start
    chrom_end = region.intervals[-1].end
    orf_type = rgr.orf_type if rgr.orf_type is not None else ""
    name = f"{rgr.id}:{rgr.transcript.gene_id}:{orf_type}"
    block_sizes = ",".join(str(iv.end - iv.start) for iv in region.intervals)
    block_starts = ",".join(str(iv.start - chrom_start) for iv in region.intervals)
    return (
        f"{region.chrom}\t{chrom_start}\t{chrom_end}\t{name}\t0\t"
        f"{region.strand}\t{chrom_start}\t{chrom_end}\t{_bed_colour(rgr.orf_type)}\t"
        f"{len(region.intervals)}\t{block_sizes}\t{block_starts}\n"
    )


def locus_gtf_line(loc: Locus) -> str:
    """A single GTF line spanning the locus."""
    iv = loc.iv
    return (
        f"{iv.chrom}\t{_GTF_SOURCE}\tlocus\t{iv.start}\t{iv.end}\t.\t"
        f'{iv.strand}\t.\tlocus_id "{loc.id}";\n'
    )


# --------------------------------------------------------------------------- #
# One locus
# --------------------------------------------------------------------------- #


def write_gtf(
    loc: Locus,
    prefix: str,
    *,
    write_loci: bool = False,
    write_transcripts: bool = False,
    write_orfs: bool = True,
) -> None:
    """Append the locus's features to ``<prefix>{_,/}{loci,transcripts,orfs}.gtf``.

    Parameters
    ----------
    loc : Locus
        The locus to write.
    prefix : str
        Output prefix; a trailing slash makes it a directory.
    write_loci : bool
        Write the locus interval.
    write_transcripts : bool
        Write the NOISE (transcript-level) regions.
    write_orfs : bool
        Write the ORF regions.
    """
    if write_loci:
        append_locked(_output_path(prefix, "loci.gtf"), locus_gtf_line(loc))
    if write_transcripts:
        append_locked(
            _output_path(prefix, "transcripts.gtf"),
            "".join(rgr_gtf(r, loc.id) for r in loc.rgr_set if r.type == "NOISE"),
        )
    if write_orfs:
        append_locked(
            _output_path(prefix, "orfs.gtf"),
            "".join(rgr_gtf(r, loc.id) for r in loc.rgr_set if r.type == "ORF"),
        )


def write_tsv(
    loc: Locus,
    prefix: str,
    runs: list[RiboSeqRun] | None = None,
    include_noise: bool = False,
) -> None:
    """Append the locus's regions to ``<prefix>{_,/}{orfs,regions}.tsv``.

    With *runs* the table carries a header (written once) and one activity
    column per run, read from ``loc.result_df``; without, only the metadata
    columns of :func:`rgr_tsv_line` are written.

    Parameters
    ----------
    loc : Locus
        The locus to write.
    prefix : str
        Output prefix; a trailing slash makes it a directory.
    runs : list[RiboSeqRun], optional
        Ribo-seq runs whose ids become the activity columns.
    include_noise : bool
        Write the NOISE regions alongside the ORFs (``regions.tsv``).
    """
    columns = REGION_TABLE_COLUMNS if include_noise else ORF_TABLE_COLUMNS
    path = _output_path(prefix, "regions.tsv" if include_noise else "orfs.tsv")
    result_df = loc.result_df
    with_activities = runs is not None and result_df is not None
    with FileLock(path + ".lock"):
        if runs is not None and not os.path.exists(path):
            header = "\t".join(columns + tuple(run.id for run in runs))
            with open(path, "w") as fh:
                fh.write(header + "\n")
        lines = []
        for rgr in loc.rgr_set:
            if not include_noise and rgr.type != "ORF":
                continue
            if with_activities:
                activities = "\t".join(f"{v:.2e}" for v in result_df.loc[rgr.id])
                orf_type = rgr.orf_type if rgr.orf_type is not None else ""
                lines.append(
                    f"{rgr.id}\t{rgr.transcript.gene_id}\t{rgr.transcript.id}"
                    f"\t{loc.id}\t{rgr.full_genomic_region}\t{orf_type}"
                    f"\t{activities}\n"
                )
            else:
                lines.append(rgr_tsv_line(rgr, loc.id))
        with open(path, "a") as fh:
            fh.write("".join(lines))


def write_bed(loc: Locus, prefix: str, include_noise: bool = False) -> None:
    """Append the locus's regions to ``<prefix>{_,/}{orfs,regions}.bed``."""
    path = _output_path(prefix, "regions.bed" if include_noise else "orfs.bed")
    append_locked(
        path,
        "".join(
            rgr_bed_line(rgr)
            for rgr in loc.rgr_set
            if include_noise or rgr.type == "ORF"
        ),
    )


# --------------------------------------------------------------------------- #
# What the worker exports
# --------------------------------------------------------------------------- #


def write_step_outputs(loc: Locus, config: Config, step_dir: str) -> None:
    """Write the intermediate tables of one filtering stage (``export_all_steps``).

    Parameters
    ----------
    loc : Locus
        The locus after that stage.
    config : Config
        The ``export_*`` selection.
    step_dir : str
        ``<regions_activities>/<step>`` prefix.
    """
    if config.export_gtf:
        write_gtf(
            loc,
            step_dir,
            write_orfs=config.export_orfs,
            write_loci=config.export_loci,
            write_transcripts=config.export_transcripts,
        )
    if config.export_tsv and config.export_orfs:
        write_tsv(loc, step_dir)
    if config.export_bed and config.export_orfs:
        write_bed(loc, step_dir)


def write_final_outputs(
    loc: Locus, config: Config, output_dir: str, runs: list[RiboSeqRun]
) -> None:
    """Write the locus's final result tables into *output_dir*.

    Parameters
    ----------
    loc : Locus
        The locus after the final activity estimate.
    config : Config
        The ``export_*`` selection.
    output_dir : str
        The ``regions_activities`` directory.
    runs : list[RiboSeqRun]
        The runs, in activity-column order.
    """
    prefix = output_dir.rstrip("/") + "/"
    with_regions = config.export_regions and not loc.result_df.empty
    if config.export_tsv:
        if config.export_orfs:
            write_tsv(loc, prefix, runs=runs)
        if with_regions:
            write_tsv(loc, prefix, runs=runs, include_noise=True)
    if config.export_gtf:
        write_gtf(
            loc,
            prefix,
            write_orfs=config.export_orfs,
            write_loci=config.export_loci,
            write_transcripts=config.export_transcripts,
        )
    if config.export_bed:
        if config.export_orfs:
            write_bed(loc, prefix)
        if with_regions:
            write_bed(loc, prefix, include_noise=True)
