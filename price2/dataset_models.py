"""Export dataset model summaries and plots.

Writes per-run cleavage and coverage model summaries to TSV files and
generates multi-page PDF plots for visual inspection.  All outputs are
stored in a ``dataset_models/`` sub-directory of the output directory.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from price2.coverage_model import (
    _HIST_SIZE,
    _START_CODON_IDX,
    _START_BODY_SLICE,
    _STOP_PEAK_IDX,
    _STOP_HIST_OFFSET,
    _STOP_BODY_SLICE,
)

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
    * ``coverage_models.tsv`` – one row per run with start/stop
      enrichment factors and read counts.
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

    _write_cleavage_summary(runs, dm_dir)
    _write_cleavage_pdf(runs, dm_dir)
    _write_coverage_summary(runs, dm_dir)
    _write_coverage_pdf(runs, dm_dir)


# ---------------------------------------------------------------------------
# Cleavage helpers
# ---------------------------------------------------------------------------


def _write_cleavage_summary(
    runs: "list[RiboSeqRun]",
    dm_dir: str,
) -> None:
    """Write a TSV file summarising cleavage models for all runs.

    Columns: ``dataset_id``, ``pu``, ``counted_reads``, ``pl``, ``pr``.
    The probability arrays are stored as comma-separated values.

    Parameters
    ----------
    runs : list[RiboSeqRun]
        Ribo-seq runs.
    dm_dir : str
        Target directory.
    """
    path = os.path.join(dm_dir, "cleavage_models.tsv")
    with open(path, "w") as fh:
        fh.write("dataset_id\tpu\tcounted_reads\tpl\tpr\n")
        for run in runs:
            cm = run.cleavage_model
            counted = getattr(run, "cleavage_counted_reads", 0)
            pl_str = ",".join(f"{v:.6g}" for v in cm.pl)
            pr_str = ",".join(f"{v:.6g}" for v in cm.pr)
            fh.write(f"{run.id}\t{cm.pu:.6g}\t{counted}\t{pl_str}\t{pr_str}\n")


def _write_cleavage_pdf(
    runs: "list[RiboSeqRun]",
    dm_dir: str,
) -> None:
    """Write a multi-page PDF with cleavage-model plots (one page per run).

    Each page contains three sub-plots:

    1. Cleavage probability distributions (left/right) and UTA probability.
    2. P-site distribution around CDS start sites (from ``dist_starts``)
       with the estimated most likely P-site distance marked.
    3. Read distribution over lengths and frames.

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
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # --- Panel 1: cleavage distributions ---
            run.cleavage_model.plot(ax=axes[0])
            axes[0].set_title("Cleavage distributions")

            # --- Panel 2: P-site distribution around CDS start ---
            dist_starts = getattr(run, "cleavage_dist_starts", None)
            if dist_starts is not None:
                x = np.arange(len(dist_starts)) - 100
                axes[1].bar(x, dist_starts, width=1, color="steelblue")
                axes[1].set_xlabel("read-start position relative to CDS start")
                axes[1].set_ylabel("read count")
                axes[1].set_title("P-site around CDS start")
                axes[1].set_xlim(-50, 50)

                pl = run.cleavage_model.pl
                p_site_offset = int(np.argmax(pl))
                axes[1].bar(
                    -p_site_offset,
                    dist_starts[-p_site_offset + 100],
                    width=1,
                    color="tab:red",
                )
                axes[1].text(
                    0.58,
                    0.95,
                    f"most frequent distance of\nread start to CDS start = {p_site_offset}",
                    transform=axes[1].transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )
            else:
                axes[1].set_visible(False)

            # --- Panel 3: read length / frame distribution ---
            table = getattr(run, "cleavage_table", None)
            if table is not None:
                lo, hi = 20, min(40, table.shape[0])
                x = np.arange(lo, hi)
                bar_w = 0.25
                axes[2].bar(
                    x - bar_w,
                    table[lo:hi, 0, 0, 0],
                    bar_w,
                    label="frame 0",
                )
                axes[2].bar(
                    x,
                    table[lo:hi, 1, 0, 0],
                    bar_w,
                    label="frame 1",
                )
                axes[2].bar(
                    x + bar_w,
                    table[lo:hi, 2, 0, 0],
                    bar_w,
                    label="frame 2",
                )
                axes[2].set_xlabel("read length")
                axes[2].set_ylabel("read count")
                axes[2].set_title("read length / frame distribution")
                axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                axes[2].legend()
            else:
                axes[2].set_visible(False)

            fig.suptitle(run.id, fontsize=14, fontweight="bold")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Coverage helpers
# ---------------------------------------------------------------------------


def _write_coverage_summary(
    runs: "list[RiboSeqRun]",
    dm_dir: str,
) -> None:
    """Write a TSV file summarising coverage models for all runs.

    Columns: ``dataset_id``, ``start_factor``, ``stop_factor``,
    ``start_counted_reads``, ``stop_counted_reads``.

    Parameters
    ----------
    runs : list[RiboSeqRun]
        Ribo-seq runs.
    dm_dir : str
        Target directory.
    """
    path = os.path.join(dm_dir, "coverage_models.tsv")
    with open(path, "w") as fh:
        fh.write(
            "dataset_id\tstart_factor\tstop_factor\t"
            "start_counted_reads\tstop_counted_reads\n"
        )
        for run in runs:
            cov = run.coverage_model
            start_hist = getattr(cov, "start_hist", None)
            stop_hist = getattr(cov, "stop_hist", None)
            start_reads = int(start_hist.sum()) if start_hist is not None else 0
            stop_reads = int(stop_hist.sum()) if stop_hist is not None else 0
            fh.write(
                f"{run.id}\t{cov.start_factor:.4f}\t{cov.stop_factor:.4f}\t"
                f"{start_reads}\t{stop_reads}\n"
            )


def _write_coverage_pdf(
    runs: "list[RiboSeqRun]",
    dm_dir: str,
) -> None:
    """Write a multi-page PDF with CDS start/stop P-site histograms.

    Each page shows two sub-plots for one run: the start-codon histogram
    (left) and the stop-codon histogram (right).

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
            cov = run.coverage_model
            start_hist = getattr(cov, "start_hist", None)
            stop_hist = getattr(cov, "stop_hist", None)

            if start_hist is None or stop_hist is None:
                continue

            fig, (ax_start, ax_stop) = plt.subplots(1, 2, figsize=(14, 5))

            # --- Start-codon histogram ---
            x_start = np.arange(_HIST_SIZE) - _START_CODON_IDX
            colors_start = np.where(x_start == 0, "tab:red", "steelblue")
            ax_start.bar(
                x_start,
                start_hist,
                width=1,
                color=colors_start,
            )
            ax_start.set_xlabel("CDS position relative to start codon")
            ax_start.set_ylabel("P-site read count")
            ax_start.set_title(f"{run.id} – CDS start")
            ax_start.set_xlim(-35, 205)

            # Annotate the start factor
            ax_start.text(
                0.95,
                0.95,
                f"start factor = {cov.start_factor:.2f}",
                transform=ax_start.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            # --- Stop-codon histogram ---
            x_stop = np.arange(_HIST_SIZE) - _STOP_HIST_OFFSET
            colors_stop = np.where(x_stop == -3, "tab:red", "steelblue")
            ax_stop.bar(
                x_stop,
                stop_hist,
                width=1,
                color=colors_stop,
            )
            ax_stop.set_xlabel("CDS position relative to CDS end")
            ax_stop.set_ylabel("P-site read count")
            ax_stop.set_title(f"{run.id} – CDS stop")
            ax_stop.set_xlim(-205, 35)

            # Annotate the stop factor
            ax_stop.text(
                0.3,
                0.95,
                f"stop factor = {cov.stop_factor:.2f}",
                transform=ax_stop.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            fig.suptitle(run.id, fontsize=14, fontweight="bold")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
