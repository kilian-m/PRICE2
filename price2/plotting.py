"""Diagnostic plots of the per-dataset models.

Kept apart from the model modules so that the worker processes, which only
evaluate the models, never import matplotlib.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from price2.cleavage_model import DIST_STARTS_CENTRE
from price2.coverage_model import (
    HIST_SIZE,
    START_BODY_SLICE,
    START_CODON_IDX,
    STOP_BODY_SLICE,
    STOP_HIST_OFFSET,
    STOP_PEAK_IDX,
)

if TYPE_CHECKING:
    from price2.cleavage_model import CleavageModel
    from price2.coverage_model import CoverageModel


def plot_coverage(
    model: CoverageModel,
    axes: Optional[tuple] = None,
) -> plt.Figure:
    """Plot the start- and stop-codon P-site histograms.

    Two side-by-side panels: CDS start (left) and CDS stop (right).
    Each panel highlights the peak position in red and annotates the
    corresponding enrichment factor.

    Parameters
    ----------
    axes : tuple of (Axes, Axes) or None, optional
        A ``(ax_start, ax_stop)`` tuple to draw on.  A new figure is
        created when *None*.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the two panels.
    """
    if axes is None:
        fig, (ax_start, ax_stop) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        ax_start, ax_stop = axes
        fig = ax_start.get_figure()

    # Start-codon histogram (x-axis in codon positions)
    x_start = np.arange(HIST_SIZE) - START_CODON_IDX

    # Determine which body positions survive the 50% trim.
    start_body_idx = np.arange(HIST_SIZE)[START_BODY_SLICE]
    start_body_vals = model.start_hist[START_BODY_SLICE]
    q25, q75 = np.percentile(start_body_vals, [25, 75])
    start_kept = set(
        start_body_idx[(start_body_vals >= q25) & (start_body_vals <= q75)]
    )

    colors_start = []
    for i in range(HIST_SIZE):
        if i == START_CODON_IDX:
            colors_start.append("tab:red")
        elif i in start_kept:
            colors_start.append("steelblue")
        elif i in set(start_body_idx):
            colors_start.append("lightsteelblue")
        else:
            colors_start.append("lightsteelblue")

    ax_start.bar(x_start, model.start_hist, width=1, color=colors_start)
    ax_start.set_xlabel("Codon position relative to translation start")
    ax_start.set_ylabel("P-site read count")
    ax_start.set_title("CDS start")
    ax_start.set_xlim(-12, 105)
    ax_start.text(
        0.95,
        0.95,
        f"start factor = {model.start_factor:.2f}",
        transform=ax_start.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Stop-codon histogram (x-axis in codon positions)
    x_stop = np.arange(HIST_SIZE) - STOP_HIST_OFFSET

    # Determine which body positions survive the 50% trim.
    stop_body_idx = np.arange(HIST_SIZE)[STOP_BODY_SLICE]
    stop_body_vals = model.stop_hist[STOP_BODY_SLICE]
    q25, q75 = np.percentile(stop_body_vals, [25, 75])
    stop_kept = set(
        stop_body_idx[(stop_body_vals >= q25) & (stop_body_vals <= q75)]
    )

    colors_stop = []
    for i in range(HIST_SIZE):
        if i == STOP_PEAK_IDX:
            colors_stop.append("tab:red")
        elif i in stop_kept:
            colors_stop.append("steelblue")
        elif i in set(stop_body_idx):
            colors_stop.append("lightsteelblue")
        else:
            colors_stop.append("lightsteelblue")

    ax_stop.bar(x_stop, model.stop_hist, width=1, color=colors_stop)
    ax_stop.set_xlabel("Codon position relative to translation end")
    ax_stop.set_ylabel("P-site read count")
    ax_stop.set_title("CDS stop")
    ax_stop.set_xlim(-105, 12)
    ax_stop.text(
        0.3,
        0.95,
        f"stop factor = {model.stop_factor:.2f}",
        transform=ax_stop.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    return fig


def plot_cleavage(model: CleavageModel, ax: Optional[plt.Axes] = None) -> None:
    """Plot the cleavage model.

    Shows left/right cleavage distributions as bar charts and
    the untemplated-addition probability as a red bar.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. A new figure is created when *None*.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.bar(range(-len(model.pl) + 1, 1), model.pl[::-1])
    ax.bar(range(len(model.pr)), model.pr)
    ax.set_xlim(-30, 25)
    ax.set_ylim(0, 1)
    fill = Rectangle(
        (-5, 0.9),
        model.pu * 25,
        0.05,
        fill=True,
        facecolor="tab:red",
    )
    border = Rectangle(
        (-5, 0.9),
        25,
        0.05,
        fill=False,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(fill)
    ax.add_patch(border)
    ax.text(
        0.6,
        0.85,
        f"UTA prob = {model.pu:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.set_xlabel("position relative to P-site")
    ax.set_ylabel("cleavage probability")


def plot_cleavage_full(
    model: CleavageModel, fig: Optional[plt.Figure] = None
) -> plt.Figure:
    """Plot a 3-panel diagnostic figure for the cleavage model.

    Panel 1: left/right cleavage distributions and UTA probability.
    Panel 2: P-site read-start distance to CDS start histogram
             (requires ``dist_starts`` attribute).
    Panel 3: read-length / reading-frame count table
             (requires ``table`` attribute).

    Parameters
    ----------
    fig : matplotlib.figure.Figure or None, optional
        Figure to draw on.  A new 3-axes figure is created when *None*.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the three panels.
    """
    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        axes = fig.axes

    ax0, ax1, ax2 = axes[0], axes[1], axes[2]

    # Panel 1: read length / reading-frame distribution.
    #
    # ``model.table`` is in the EM's frame convention: after
    # :meth:`CleavageEstimator.correct_table` column ``c`` holds the reads
    # whose P-site offset is ``c`` (mod 3).  What this panel should show,
    # though, is the *genomic* reading frame of the read start relative to
    # the CDS start (``read_start_offset % 3``) -- the phase the reads
    # actually fall into: a 12-nt cleavage distance is frame 0, 13 nt is
    # frame 2, 11 nt is frame 1.  Genomic frame ``g`` lives in column
    # ``(-g) % 3``, so frames 0/1/2 read off columns 0/2/1.
    #
    # We also sum over the untemplated-addition axis.  The frame is taken
    # from the mapping portion of the read (the extra 5' base is soft-clipped
    # under Local, or trimmed under EndToEnd, before ``genomic_region`` is
    # built), so a detected-UA read already sits in its true frame.  Dropping
    # those reads -- as an ``oua == 0`` slice does -- hides almost all of the
    # signal in a high-UA library: e.g. SRR7240724 (pu = 0.99) has its
    # genuine frame-0 peak all but vanish, leaving only the misdetection
    # shadow behind.
    lo, hi = 20, min(40, model.table.shape[0])
    x = np.arange(lo, hi)
    bar_w = 0.25
    counts = model.table[lo:hi, :, :].sum(axis=2)  # (n_lengths, 3) by column
    ax0.bar(x - bar_w, counts[:, 0], bar_w, label="frame 0")
    ax0.bar(x, counts[:, 2], bar_w, label="frame 1")
    ax0.bar(x + bar_w, counts[:, 1], bar_w, label="frame 2")
    ax0.set_xlabel("read length")
    ax0.set_ylabel("read count")
    ax0.set_title("read length / frame distribution")
    ax0.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax0.legend()

    # Panel 2: P-site distribution around CDS start
    x = np.arange(len(model.dist_starts)) - DIST_STARTS_CENTRE
    ax1.bar(x, model.dist_starts, width=1, color="steelblue")
    ax1.set_xlabel("read-start position relative to CDS start")
    ax1.set_ylabel("read count")
    ax1.set_title("Read starts around CDS start")
    ax1.set_xlim(-50, 50)
    p_site_offset = int(np.argmax(model.pl))
    ax1.bar(
        -p_site_offset,
        model.dist_starts[DIST_STARTS_CENTRE - p_site_offset],
        width=1,
        color="tab:red",
    )
    ax1.text(
        0.58,
        0.95,
        f"most frequent distance of\nread start to CDS start" f" = {p_site_offset}",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Panel 3: cleavage distributions
    plot_cleavage(model, ax=ax2)
    ax2.set_title("Cleavage distributions")

    return fig
