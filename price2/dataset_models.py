"""Export of a run's cleavage and coverage models to ``dataset_models/``, and back.

The models themselves live in ``price.db``; these files are the human- and
tool-readable summaries: one TSV per model type with the obligatory
attributes, an ``.npz`` with the optional arrays, and diagnostic PDFs.
"""

from __future__ import annotations

import os

import numpy as np

from price2.cleavage_model import CleavageModel
from price2.coverage_model import CoverageModel
from price2.ribo_seq_run import RiboSeqRun


def save_dataset_models(
    runs: list[RiboSeqRun],
    dm_dir: str,
    save_optional: bool = True,
) -> None:
    """Save cleavage and coverage model summaries, plots, and optional data.

    Creates *dm_dir* and writes:

    * ``cleavage_models.tsv`` – obligatory attributes (pl, pr, pu).
    * ``cleavage_models.pdf`` – multi-page diagnostic plots (requires
      optional cleavage attributes).
    * ``coverage_models.tsv`` – obligatory attributes (start_factor,
      stop_factor).
    * ``coverage_models.pdf`` – multi-page histogram plots (requires
      optional coverage attributes).

    When *save_optional* is True, also writes:

    * ``cleavage_models.npz`` – optional arrays (dist_starts, table).
    * ``coverage_models.npz`` – optional arrays (start_hist, stop_hist).

    Parameters
    ----------
    runs : list[RiboSeqRun]
        Ribo-seq runs whose models should be exported.
    dm_dir : str
        The ``dataset_models/`` directory
        (:attr:`price2.layout.RunLayout.dataset_models_dir`).
    save_optional : bool, optional
        Whether to write ``.npz`` files with optional model attributes
        (default True).
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    os.makedirs(dm_dir, exist_ok=True)

    # --- Cleavage TSV + optional NPZ ---
    cleavage_tsv = os.path.join(dm_dir, "cleavage_models.tsv")
    cleavage_npz_data: dict[str, np.ndarray] | None = {} if save_optional else None
    with open(cleavage_tsv, "w") as fh:
        fh.write(CleavageModel.TSV_HEADER + "\n")
        for run in runs:
            run.cleavage_model.to_files(run.id, fh, cleavage_npz_data)
    if cleavage_npz_data:
        np.savez(os.path.join(dm_dir, "cleavage_models.npz"), **cleavage_npz_data)

    # --- Cleavage PDF ---
    path = os.path.join(dm_dir, "cleavage_models.pdf")
    with PdfPages(path) as pdf:
        for run in runs:
            fig = run.cleavage_model.plot_full()
            fig.suptitle(run.id, fontsize=14, fontweight="bold")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    # --- Coverage TSV + optional NPZ ---
    coverage_tsv = os.path.join(dm_dir, "coverage_models.tsv")
    coverage_npz_data: dict[str, np.ndarray] | None = {} if save_optional else None
    with open(coverage_tsv, "w") as fh:
        fh.write(CoverageModel.TSV_HEADER + "\n")
        for run in runs:
            run.coverage_model.to_files(run.id, fh, coverage_npz_data)
    if coverage_npz_data:
        np.savez(os.path.join(dm_dir, "coverage_models.npz"), **coverage_npz_data)

    # --- Coverage PDF ---
    path = os.path.join(dm_dir, "coverage_models.pdf")
    with PdfPages(path) as pdf:
        for run in runs:
            fig = run.coverage_model.plot()
            fig.suptitle(run.id, fontsize=14, fontweight="bold")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def load_cleavage_models(
    dm_dir: str,
    load_optional: bool = False,
) -> dict[str, CleavageModel]:
    """Load cleavage models from a ``dataset_models/`` directory.

    Parameters
    ----------
    dm_dir : str
        Path to a ``dataset_models/`` directory.
    load_optional : bool, optional
        Whether to load optional attributes from the ``.npz`` file
        (default False).

    Returns
    -------
    dict[str, CleavageModel]
        Mapping of dataset identifier to the reconstructed model.
    """
    tsv = os.path.join(dm_dir, "cleavage_models.tsv")
    npz = os.path.join(dm_dir, "cleavage_models.npz")
    return CleavageModel.from_files(tsv, npz if load_optional and os.path.exists(npz) else None)


def load_coverage_models(
    dm_dir: str,
    load_optional: bool = False,
) -> dict[str, CoverageModel]:
    """Load coverage models from a ``dataset_models/`` directory.

    Parameters
    ----------
    dm_dir : str
        Path to a ``dataset_models/`` directory.
    load_optional : bool, optional
        Whether to load optional attributes from the ``.npz`` file
        (default False).

    Returns
    -------
    dict[str, CoverageModel]
        Mapping of dataset identifier to the reconstructed model.
    """
    tsv = os.path.join(dm_dir, "coverage_models.tsv")
    npz = os.path.join(dm_dir, "coverage_models.npz")
    return CoverageModel.from_files(tsv, npz if load_optional and os.path.exists(npz) else None)
