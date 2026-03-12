"""Export dataset model summaries and plots.

Writes per-run cleavage and coverage model summaries to TSV files and
generates multi-page PDF plots for visual inspection.  All outputs are
stored in a ``dataset_models/`` sub-directory of the output directory.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if TYPE_CHECKING:
    from price2.ribo_seq_run import RiboSeqRun


def save_dataset_models(
    runs: "list[RiboSeqRun]",
    output_dir: str,
) -> None:
    """Save cleavage and coverage model summaries and plots.

    Creates a ``dataset_models/`` sub-directory under *output_dir* and
    writes:

    * ``cleavage_models.tsv`` – one row per run with upstream/downstream
      cleavage probabilities, untemplated-addition probability and the
      number of counted reads.
    * ``cleavage_models.pdf`` – multi-page PDF with a cleavage-model
      plot per run.
    * ``coverage_models.tsv`` – one row per run with start- and
      stop-codon P-site histograms as comma-separated columns.
    * ``coverage_models.pdf`` – multi-page PDF with CDS start/stop
      P-site histograms per run.

    Parameters
    ----------
    runs : list[RiboSeqRun]
        Ribo-seq runs whose models should be exported.
    output_dir : str
        Root output directory (``config.o_dir``).
    """
    dm_dir = os.path.join(output_dir, "dataset_models")
    os.makedirs(dm_dir, exist_ok=True)

    _write_cleavage_tsv(runs, dm_dir)
    _write_cleavage_pdf(runs, dm_dir)
    _write_coverage_tsv(runs, dm_dir)
    _write_coverage_pdf(runs, dm_dir)


def load_cleavage_models(dm_dir: str) -> "dict[str, object]":
    """Load cleavage models from a ``dataset_models/`` directory.

    Reads the ``cleavage_models.tsv`` file written by
    :func:`save_dataset_models` and reconstructs one
    :class:`~price2.cleavage_model.CleavageModel` per row.

    Parameters
    ----------
    dm_dir : str
        Path to a ``dataset_models/`` directory.

    Returns
    -------
    dict[str, CleavageModel]
        Mapping of dataset identifier to the reconstructed model.
    """
    from price2.cleavage_model import CleavageModel

    path = os.path.join(dm_dir, "cleavage_models.tsv")
    return CleavageModel.from_tsv(path)


def load_coverage_models(dm_dir: str) -> "dict[str, object]":
    """Load coverage models from a ``dataset_models/`` directory.

    Reads the ``coverage_models.tsv`` file written by
    :func:`save_dataset_models` and reconstructs one
    :class:`~price2.coverage_model.CoverageModel` per run.

    Parameters
    ----------
    dm_dir : str
        Path to a ``dataset_models/`` directory.

    Returns
    -------
    dict[str, CoverageModel]
        Mapping of dataset identifier to the reconstructed model.
    """
    from price2.coverage_model import CoverageModel

    path = os.path.join(dm_dir, "coverage_models.tsv")
    return CoverageModel.from_tsv(path)


# ---------------------------------------------------------------------------
# Cleavage helpers
# ---------------------------------------------------------------------------


def _write_cleavage_tsv(
    runs: "list[RiboSeqRun]",
    dm_dir: str,
) -> None:
    """Write a TSV file summarising cleavage models for all runs.

    Delegates to :meth:`~price2.cleavage_model.CleavageModel.to_tsv_line`.

    Parameters
    ----------
    runs : list[RiboSeqRun]
        Ribo-seq runs.
    dm_dir : str
        Target directory.
    """
    from price2.cleavage_model import CleavageModel

    path = os.path.join(dm_dir, "cleavage_models.tsv")
    with open(path, "w") as fh:
        fh.write(CleavageModel.TSV_HEADER + "\n")
        for run in runs:
            counted = getattr(run, "cleavage_counted_reads", 0)
            fh.write(run.cleavage_model.to_tsv_line(run.id, counted))


def _write_cleavage_pdf(
    runs: "list[RiboSeqRun]",
    dm_dir: str,
) -> None:
    """Write a multi-page PDF with cleavage-model diagnostic plots.

    Delegates to :meth:`~price2.cleavage_model.CleavageModel.plot_full`
    to produce one three-panel page per run.

    Parameters
    ----------
    runs : list[RiboSeqRun]
        Ribo-seq runs.
    dm_dir : str
        Target directory.
    """
    path = os.path.join(dm_dir, "cleavage_models.pdf")
    with PdfPages(path) as pdf:
        for run in runs:
            fig = run.cleavage_model.plot_full()
            fig.suptitle(run.id, fontsize=14, fontweight="bold")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Coverage helpers
# ---------------------------------------------------------------------------


def _write_coverage_tsv(
    runs: "list[RiboSeqRun]",
    dm_dir: str,
) -> None:
    """Write a TSV file containing per-run P-site histograms.

    Columns: ``dataset_id``, ``start_hist``, ``stop_hist``.
    Both histogram arrays are stored as comma-separated values.
    Delegates to :meth:`~price2.coverage_model.CoverageModel.to_tsv_line`.

    Parameters
    ----------
    runs : list[RiboSeqRun]
        Ribo-seq runs.
    dm_dir : str
        Target directory.
    """
    from price2.coverage_model import CoverageModel

    path = os.path.join(dm_dir, "coverage_models.tsv")
    with open(path, "w") as fh:
        fh.write(CoverageModel.TSV_HEADER + "\n")
        for run in runs:
            fh.write(run.coverage_model.to_tsv_line(run.id))


def _write_coverage_pdf(
    runs: "list[RiboSeqRun]",
    dm_dir: str,
) -> None:
    """Write a multi-page PDF with CDS start/stop P-site histograms.

    Delegates to :meth:`~price2.coverage_model.CoverageModel.plot` to
    produce one two-panel page per run.

    Parameters
    ----------
    runs : list[RiboSeqRun]
        Ribo-seq runs.
    dm_dir : str
        Target directory.
    """
    path = os.path.join(dm_dir, "coverage_models.pdf")
    with PdfPages(path) as pdf:
        for run in runs:
            fig = run.coverage_model.plot()
            fig.suptitle(run.id, fontsize=14, fontweight="bold")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
