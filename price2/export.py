"""Rendering loci and their regions as TSV, GTF and BED, and writing them.

The worker that finishes a locus renders its rows with the functions here
and hands them back to the parent as a mapping of file name to
:class:`OutputText`; the parent, the only process that writes to the output
directory, appends them with :class:`OutputWriter`.  The column layout of
the TSV tables is defined here and reused by :mod:`price2.tpm` and
:mod:`price2.run_state`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

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

#: Filtering stages whose intermediate tables ``export_all_steps`` writes,
#: as ``<step>_orfs.tsv`` and friends next to the final tables.
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


@dataclass(frozen=True)
class OutputText:
    """Rows to append to one output file.

    Parameters
    ----------
    body : str
        The rows, each terminated by a newline (may be empty).
    header : str or None
        Header line written once, when the file is created.
    """

    body: str
    header: str | None = None


class OutputWriter:
    """Append rendered rows to the files of one output directory.

    Meant for a single writer: the parent process of the fan-out.  A file
    is created on first use, with its header when the rows carry one.

    Parameters
    ----------
    directory : str
        The directory the file names in :meth:`write` are relative to.
    """

    def __init__(self, directory: str) -> None:
        self.directory = directory

    def write(self, outputs: dict[str, OutputText]) -> None:
        """Append every entry of *outputs* to its file."""
        for name, text in outputs.items():
            path = os.path.join(self.directory, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if text.header is not None and not os.path.exists(path):
                with open(path, "w") as fh:
                    fh.write(text.header + "\n")
            with open(path, "a") as fh:
                fh.write(text.body)


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
    features = ["exon", "CDS"] if rgr.is_orf else ["exon"]
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


def gtf_outputs(
    loc: Locus,
    *,
    write_loci: bool = False,
    write_transcripts: bool = False,
    write_orfs: bool = True,
) -> dict[str, OutputText]:
    """The locus's GTF rows, keyed ``loci.gtf`` / ``transcripts.gtf`` / ``orfs.gtf``."""
    outputs = {}
    if write_loci:
        outputs["loci.gtf"] = OutputText(locus_gtf_line(loc))
    if write_transcripts:
        outputs["transcripts.gtf"] = OutputText(
            "".join(rgr_gtf(r, loc.id) for r in loc.rgrs if not r.is_orf)
        )
    if write_orfs:
        outputs["orfs.gtf"] = OutputText(
            "".join(rgr_gtf(r, loc.id) for r in loc.rgrs if r.is_orf)
        )
    return outputs


def tsv_output(
    loc: Locus,
    runs: list[RiboSeqRun] | None = None,
    include_noise: bool = False,
) -> tuple[str, OutputText]:
    """The locus's rows of ``orfs.tsv`` (or ``regions.tsv`` with the NOISE regions).

    With *runs* the rows carry one activity column per run, read from
    ``loc.result_df``, and the table gets a header; without, only the
    metadata columns of :func:`rgr_tsv_line` are written.

    Returns
    -------
    tuple[str, OutputText]
        The file name and its rows.
    """
    name = "regions.tsv" if include_noise else "orfs.tsv"
    columns = REGION_TABLE_COLUMNS if include_noise else ORF_TABLE_COLUMNS
    header = None
    if runs is not None:
        header = "\t".join(columns + tuple(run.id for run in runs))
    with_activities = runs is not None and loc.result_df is not None
    lines = []
    for rgr in loc.rgrs:
        if not include_noise and not rgr.is_orf:
            continue
        if with_activities:
            activities = "\t".join(f"{v:.2e}" for v in loc.result_df.loc[rgr.id])
            orf_type = rgr.orf_type if rgr.orf_type is not None else ""
            lines.append(
                f"{rgr.id}\t{rgr.transcript.gene_id}\t{rgr.transcript.id}"
                f"\t{loc.id}\t{rgr.full_genomic_region}\t{orf_type}"
                f"\t{activities}\n"
            )
        else:
            lines.append(rgr_tsv_line(rgr, loc.id))
    return name, OutputText("".join(lines), header)


def bed_output(loc: Locus, include_noise: bool = False) -> tuple[str, OutputText]:
    """The locus's rows of ``orfs.bed`` (or ``regions.bed`` with the NOISE regions)."""
    name = "regions.bed" if include_noise else "orfs.bed"
    body = "".join(
        rgr_bed_line(rgr)
        for rgr in loc.rgrs
        if include_noise or rgr.is_orf
    )
    return name, OutputText(body)


# --------------------------------------------------------------------------- #
# What the worker hands back
# --------------------------------------------------------------------------- #


def step_outputs(loc: Locus, config: Config, step: str) -> dict[str, OutputText]:
    """The intermediate tables of one filtering stage (``export_all_steps``).

    Parameters
    ----------
    loc : Locus
        The locus after that stage.
    config : Config
        The ``export_*`` selection.
    step : str
        One of :data:`STEP_NAMES`; the files are named ``<step>_<name>``.
    """
    outputs: dict[str, OutputText] = {}
    if config.export_gtf:
        outputs.update(
            gtf_outputs(
                loc,
                write_orfs=config.export_orfs,
                write_loci=config.export_loci,
                write_transcripts=config.export_transcripts,
            )
        )
    if config.export_tsv and config.export_orfs:
        name, text = tsv_output(loc)
        outputs[name] = text
    if config.export_bed and config.export_orfs:
        name, text = bed_output(loc)
        outputs[name] = text
    return {f"{step}_{name}": text for name, text in outputs.items()}


def final_outputs(
    loc: Locus, config: Config, runs: list[RiboSeqRun]
) -> dict[str, OutputText]:
    """The locus's rows of the final result tables.

    Parameters
    ----------
    loc : Locus
        The locus after the final activity estimate.
    config : Config
        The ``export_*`` selection.
    runs : list[RiboSeqRun]
        The runs, in activity-column order.
    """
    outputs: dict[str, OutputText] = {}
    with_regions = config.export_regions and not loc.result_df.empty
    if config.export_tsv:
        if config.export_orfs:
            name, text = tsv_output(loc, runs=runs)
            outputs[name] = text
        if with_regions:
            name, text = tsv_output(loc, runs=runs, include_noise=True)
            outputs[name] = text
    if config.export_gtf:
        outputs.update(
            gtf_outputs(
                loc,
                write_orfs=config.export_orfs,
                write_loci=config.export_loci,
                write_transcripts=config.export_transcripts,
            )
        )
    if config.export_bed:
        if config.export_orfs:
            name, text = bed_output(loc)
            outputs[name] = text
        if with_regions:
            name, text = bed_output(loc, include_noise=True)
            outputs[name] = text
    return outputs
