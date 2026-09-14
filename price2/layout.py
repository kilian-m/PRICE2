"""Where a run keeps its files.

A PRICE2 run owns two directories, the working directory (``w_dir``) with
the database, and the output directory (``o_dir``) with the result tables.
:class:`RunLayout` derives every path inside them from those two roots, so
no other module spells a file name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DB_FILENAME = "price.db"
REGIONS_ACTIVITIES_DIRNAME = "regions_activities"
DATASET_MODELS_DIRNAME = "dataset_models"
PERFORMANCE_FILENAME = "performance_measurements.tsv"
FAILED_LOCI_FILENAME = "failed_loci.txt"


@dataclass(frozen=True)
class RunLayout:
    """The files and directories of one run.

    Parameters
    ----------
    w_dir : str
        Working directory.
    o_dir : str
        Output directory.
    """

    w_dir: str
    o_dir: str

    @property
    def db_path(self) -> str:
        """The SQLite database every stage persists into."""
        return os.path.join(self.w_dir, DB_FILENAME)

    @property
    def regions_activities_dir(self) -> str:
        """Directory of the per-locus result tables (TSV, GTF, BED)."""
        return os.path.join(self.o_dir, REGIONS_ACTIVITIES_DIRNAME)

    @property
    def dataset_models_dir(self) -> str:
        """Directory of the exported cleavage and coverage models."""
        return os.path.join(self.o_dir, DATASET_MODELS_DIRNAME)

    @property
    def performance_path(self) -> str:
        """Per-locus timing and filtering statistics."""
        return os.path.join(self.o_dir, PERFORMANCE_FILENAME)

    @property
    def failed_loci_path(self) -> str:
        """Loci the deconvolution abandoned, with their tracebacks."""
        return os.path.join(self.regions_activities_dir, FAILED_LOCI_FILENAME)
