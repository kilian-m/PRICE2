"""From reads to the design matrix of a locus.

Loads a locus's collapsed reads, decides which regions each read is
compatible with and in which frame and coverage position, and routes the
reads to their equivalence groups.  :class:`ReadRouting` is the one
representation of the result: the response, the sparse design matrix the
solver sees, the multimapping slot rates and the effect of removing RGRs are
all derived from its arrays, so the multimapping EM repeats an iteration as
a few array operations.

All functions take the :class:`~price2.locus.Locus` they operate on as their
first argument and update its state in place.
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numba import njit
from scipy.sparse import csr_matrix

from price2 import database, multimap
from price2.cleavage_model import (
    OVERLAP_LIKELIHOOD_RATIO,
    UNBOUNDED,
    read_in_cds_likelihood,
    read_in_noise_likelihood,
)
from price2.coverage_position import CoveragePosition
from price2.equivalence_groups import CELL_CODES, NO_FRAME, EquivalenceGroups
from price2.genomic_region import GenomicRegion
from price2.ribo_seq_alignment import RiboSeqAlignment
from price2.ribo_seq_run import RiboSeqRun

if TYPE_CHECKING:
    from price2.genomic_features import Transcript
    from price2.locus import Locus

# The ``frame_code * 3 + coverage_position`` part of a packed cell (see
# ``equivalence_groups.pack_cell``): a NOISE region has no frame and only a
# middle position; an ORF's codes are ``frame * 3 + _START / _MIDDLE / _STOP``.
_START = CoveragePosition.start.value
_MIDDLE = CoveragePosition.middle.value
_STOP = CoveragePosition.stop.value
_NOISE_FRAME = NO_FRAME * 3

# The regions a read overlapping the end of an RGR is tested against, as
# ``(coverage position, from, to)`` with ``from`` / ``to`` indexing the RGR's
# boundaries ``(lo, lo + 3, hi - 3, hi)``: an ORF's start codon, its body and
# its last codon before the stop; a NOISE region is one middle from lo to hi.
_ORF_REGIONS = ((_START, 0, 1), (_MIDDLE, 1, 2), (_STOP, 2, 3))
_NOISE_REGIONS = ((_MIDDLE, 0, 3),)


class ReadRouting:
    """The reads of a locus routed to the rows of its design matrix.

    Built once per locus from the reads and the equivalence-group geometry
    (:func:`price2.equivalence_groups.make_equivalence_groups`), and
    independent of the read weights, this is the only representation of a
    locus's read routing: the response ``y`` (:meth:`response`), the design
    matrix (:meth:`design_matrix`), the multimapping slot rates
    (:meth:`multimap_lambdas`) and the effect of removing RGRs
    (:meth:`without_rgrs`, :meth:`rekey`) are all derived from its arrays.
    It holds arrays, ints and strings only — no RGR, transcript or read
    objects — so it is pickled on its own (``prepared_loci_cache``) and a
    light EM M-step loads just this instead of the locus's object graph.

    Cells are the packed ``(RGR, frame, coverage position)`` integers of
    :mod:`price2.equivalence_groups`; every cell carries ``rgr.index``, so
    the routing has to be rebuilt through :meth:`without_rgrs` whenever the
    locus's RGRs are removed and re-indexed.

    Attributes
    ----------
    run_ids : tuple[str, ...]
        The runs in design-matrix order (``row_run`` indexes it).
    n_rows : int
        Design-matrix rows: one per equivalence group with at least one
        cell, in the order of the groups per run.
    row_run, row_rl, row_oua, row_len, row_nnz : numpy.ndarray
        Per row: run index, read length, untemplated-addition flag, group
        length and cell count.  (``row_run`` is ``int32``; routings pickled
        before 2026-09 hold it as ``uint8``, which is why the runs of a
        design matrix cannot exceed 255 there.)
    row_cells : numpy.ndarray
        The rows' cells, flattened (``row_nnz`` cells per row).
    n_reads : dict[str, int]
        Reads per run at build time; guards against a changed read order.
    counts0, read_rl, read_oua : dict[str, numpy.ndarray]
        Per run and read, in the order of the ``reads`` table: raw
        (unweighted) count, read length and untemplated-addition flag.
    read_nnz, read_cells : dict[str, numpy.ndarray]
        Per run: each read's number of compatible cells, and the cells
        flattened.
    eg_row : dict[str, numpy.ndarray]
        Per run, the row each read feeds: ``>=0`` a row index, ``-1`` the
        read's equivalence-group key matches no group (uncounted), ``-2``
        the read is compatible with no RGR.
    mm_idx, mm_gk, mm_base : dict[str, numpy.ndarray]
        Per run, the positions of the reads that carry a multimapping slot,
        their group keys and their baseline cross-locus mass.
    num_rgrs : int
        RGR count (design-matrix column blocks).
    rgr_ids : tuple[str, ...]
        RGR identifiers by ``rgr.index``.
    rgr_lengths : numpy.ndarray
        RGR lengths by ``rgr.index``.
    """

    __slots__ = (
        "run_ids", "n_rows", "row_run", "row_rl", "row_oua", "row_len",
        "row_nnz", "row_cells", "n_reads", "counts0", "read_rl", "read_oua",
        "read_nnz", "read_cells", "eg_row", "mm_idx", "mm_gk", "mm_base",
        "num_rgrs", "rgr_ids", "rgr_lengths",
    )

    def __init__(self, **fields: object) -> None:
        for name in self.__slots__:
            setattr(self, name, fields.pop(name))
        if fields:
            raise TypeError(f"unknown routing fields: {sorted(fields)}")

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def build(
        cls,
        loc: Locus,
        runs: list[RiboSeqRun],
        egs: EquivalenceGroups,
        mm_data: dict | None,
    ) -> ReadRouting:
        """Route every loaded read of *loc* to its equivalence group.

        Parameters
        ----------
        loc : Locus
            Locus with its reads loaded (``rsas_dict``) and its RGRs final.
        runs : list[RiboSeqRun]
            Ribo-seq runs, in design-matrix order.
        egs : EquivalenceGroups
            The groups of every run, from
            :func:`~price2.equivalence_groups.make_equivalence_groups`.
        mm_data : dict or None
            ``{run_id: {group_key: (base, weight)}}`` naming this locus's
            multimapping slots, or ``None`` outside the EM.
        """
        # The rows: every run's groups, in order, with the keys' cells (sorted
        # within a key already).
        key_nnz = egs.key_nnz
        ids = np.concatenate([egs.run_key[run.id] for run in runs])
        row_run = np.repeat(
            np.arange(len(runs), dtype=np.int32),
            [egs.run_key[run.id].size for run in runs],
        )
        row_rl = egs.key_rl[ids]
        row_oua = egs.key_oua[ids]
        row_len = np.concatenate([egs.run_length[run.id] for run in runs])
        row_nnz = key_nnz[ids].astype(np.int32)
        row_cells = egs.key_cells[
            np.repeat(egs.key_ptr[ids], row_nnz) + _within(row_nnz)
        ]

        compat = loc.read_compatibility(runs)
        cell_map = compat.cell_map(loc.rgrs)
        n_reads: dict = {}
        counts0: dict = {}
        read_rl: dict = {}
        read_oua: dict = {}
        read_nnz: dict = {}
        read_cells: dict = {}
        eg_row: dict = {}
        mm_idx: dict = {}
        mm_gk: dict = {}
        mm_base: dict = {}
        for run_index, run in enumerate(runs):
            run_id = run.id
            rsas = loc.rsas_dict[run_id]
            run_mm = mm_data.get(run_id) if mm_data else None
            nnz_arr, cells_arr = compat.cells_for_run(run, cell_map)
            rl_arr = compat.read_length(run_id)
            oua_arr = compat.read_oua[run_id]
            rows = row_run == run_index
            rows_arr = match_rows(
                row_rl[rows], row_oua[rows], row_nnz[rows],
                row_cells[np.repeat(rows, row_nnz)],
                rl_arr, oua_arr, nnz_arr, cells_arr,
            )
            rows_arr[rows_arr >= 0] = np.flatnonzero(rows)[rows_arr[rows_arr >= 0]]
            mi: list = []
            mg: list = []
            mb: list = []
            if run_mm is not None:
                unique = np.fromiter(
                    (rsa.unique for rsa in rsas), dtype=bool, count=len(rsas)
                )
                for i in np.flatnonzero(~unique).tolist():
                    gk = multimap.alignment_group_key(rsas[i])
                    slot = run_mm.get(gk)
                    if slot is not None:
                        mi.append(i)
                        mg.append(gk)
                        mb.append(slot[0])
            n_reads[run_id] = len(rsas)
            counts0[run_id] = np.fromiter(
                (rsa.read_count for rsa in rsas), dtype=np.float64, count=len(rsas)
            )
            read_rl[run_id] = rl_arr
            read_oua[run_id] = oua_arr
            read_nnz[run_id] = nnz_arr
            read_cells[run_id] = cells_arr
            eg_row[run_id] = rows_arr
            mm_idx[run_id] = np.array(mi, dtype=np.int32)
            mm_gk[run_id] = np.array(mg, dtype=np.int64)
            mm_base[run_id] = np.array(mb, dtype=np.float64)

        num_rgrs = len(loc.rgrs)
        return cls(
            run_ids=tuple(run.id for run in runs),
            n_rows=len(row_run),
            row_run=row_run,
            row_rl=row_rl,
            row_oua=row_oua,
            row_len=row_len.astype(np.int64),
            row_nnz=row_nnz,
            row_cells=row_cells,
            n_reads=n_reads,
            counts0=counts0,
            read_rl=read_rl,
            read_oua=read_oua,
            read_nnz=read_nnz,
            read_cells=read_cells,
            eg_row=eg_row,
            mm_idx=mm_idx,
            mm_gk=mm_gk,
            mm_base=mm_base,
            num_rgrs=num_rgrs,
            rgr_ids=tuple(rgr.id for rgr in loc.rgrs),
            rgr_lengths=np.fromiter(
                (len(rgr) for rgr in loc.rgrs), dtype=np.int64, count=num_rgrs
            ),
        )

    # ------------------------------------------------------------------ #
    # Derived quantities                                                   #
    # ------------------------------------------------------------------ #

    def response(
        self,
        runs: list[RiboSeqRun],
        mm_data: dict | None,
        rsas_dict: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """The response ``y`` under the current read weights.

        Each read contributes its raw count to its row; a read that carries
        a multimapping slot contributes ``max(0, count - base) + weight``
        instead, so that the read's mass across all its loci sums to one
        (single-slot multimappers keep full weight).

        Parameters
        ----------
        runs : list[RiboSeqRun]
            Ribo-seq runs, in design-matrix order.
        mm_data : dict or None
            ``{run_id: {group_key: (base, weight)}}``, or ``None`` for
            classic full-weight counting.
        rsas_dict : dict or None
            The locus's loaded reads, if any, to check that they are the
            reads this routing was built from.

        Returns
        -------
        y : numpy.ndarray
            One entry per row.
        counted_reads : dict[str, float]
            Per run, the read mass that reached a row.
        """
        y = np.zeros(self.n_rows, dtype=np.float64)
        counted_reads: dict[str, float] = {}
        for run in runs:
            run_id = run.id
            if (
                rsas_dict is not None
                and run_id in rsas_dict
                and len(rsas_dict[run_id]) != self.n_reads[run_id]
            ):
                raise RuntimeError(
                    f"routing built for {self.n_reads[run_id]} reads of run "
                    f"{run_id} but {len(rsas_dict[run_id])} were loaded"
                )
            counts = self.counts0[run_id].copy()
            run_mm = mm_data.get(run_id) if mm_data else None
            idx = self.mm_idx[run_id]
            if run_mm is not None and idx.size:
                weights = np.fromiter(
                    (run_mm[gk][1] for gk in self.mm_gk[run_id]),
                    dtype=np.float64,
                    count=idx.size,
                )
                # ``base`` is summed over the spilled alignments at indexing
                # time, independently of the collapsed count; floor the
                # non-cross-locus remainder at zero so a disagreement between
                # the two can never drive the Poisson response negative.
                counts[idx] = (
                    np.maximum(0.0, counts[idx] - self.mm_base[run_id]) + weights
                )
            rows = self.eg_row[run_id]
            counted = rows >= 0
            y += np.bincount(
                rows[counted], weights=counts[counted], minlength=self.n_rows
            )
            counted_reads[run_id] = float(counts[counted].sum())
        return y, counted_reads

    def design_matrix(
        self, cm_lut: np.ndarray, coverage_params: np.ndarray, num_runs: int
    ) -> csr_matrix:
        """The design matrix ``X`` (see :func:`design_matrix`).

        Built straight in CSR layout — the cells are already grouped by row
        — with the per-cell factors in the narrowest dtypes that hold them;
        a row touching the same RGR at several coverage positions sums
        those cells (``sum_duplicates``).
        """
        nnz_per_row = self.row_nnz
        run_c = np.repeat(self.row_run.astype(np.int32), nnz_per_row)
        rgr_c, code = np.divmod(self.row_cells, CELL_CODES)
        frame_c, cov_c = np.divmod(code.astype(np.int8), 3)
        del code
        data = np.repeat(self.row_len.astype(np.float64), nnz_per_row)
        data *= cm_lut[
            run_c, np.repeat(self.row_rl, nnz_per_row), frame_c,
            np.repeat(self.row_oua, nnz_per_row),
        ]
        del frame_c
        data *= coverage_params[run_c, cov_c]
        del cov_c
        indices = (rgr_c * num_runs + run_c).astype(np.int32)
        del rgr_c, run_c
        indptr = np.zeros(self.n_rows + 1, dtype=np.int64)
        np.cumsum(nnz_per_row, out=indptr[1:])
        X = csr_matrix(
            (data, indices, indptr), shape=(self.n_rows, self.num_rgrs * num_runs)
        )
        X.sum_duplicates()
        return X

    def multimap_lambdas(
        self,
        result: np.ndarray,
        runs: list[RiboSeqRun],
        cm_lut: np.ndarray,
        coverage_params: np.ndarray,
    ) -> list:
        """Per-slot origin rates ``λ`` (see :func:`multimap_lambdas`)."""
        out: list = []
        for run_index, run in enumerate(runs):
            idx = self.mm_idx[run.id]
            if idx.size == 0:
                continue
            nnz = self.read_nnz[run.id]
            slot_nnz = nnz[idx]
            # The slot reads' cells: for read ``k`` with first cell ``s_k``
            # and ``n_k`` cells, positions ``s_k .. s_k + n_k - 1``.
            starts = (np.cumsum(nnz) - nnz)[idx]
            first = np.cumsum(slot_nnz) - slot_nnz
            within = np.arange(int(slot_nnz.sum())) - np.repeat(first, slot_nnz)
            cells = self.read_cells[run.id][np.repeat(starts, slot_nnz) + within]
            rgr_c, code = np.divmod(cells, CELL_CODES)
            frame_c, cov_c = np.divmod(code, 3)
            rl_c = np.repeat(self.read_rl[run.id][idx], slot_nnz)
            oua_c = np.repeat(self.read_oua[run.id][idx], slot_nnz)
            contribution = (
                cm_lut[run_index, rl_c, frame_c, oua_c]
                * coverage_params[run_index, cov_c]
                * result[rgr_c, run_index]
            )
            slot_of_cell = np.repeat(np.arange(idx.size), slot_nnz)
            lam = np.bincount(slot_of_cell, weights=contribution, minlength=idx.size)
            out.extend(
                (run.id, int(gk), float(value))
                for gk, value in zip(self.mm_gk[run.id], lam)
            )
        return out

    # ------------------------------------------------------------------ #
    # RGR removal                                                          #
    # ------------------------------------------------------------------ #

    def without_rgrs(self, old_to_new: list[int]) -> tuple[ReadRouting, np.ndarray]:
        """The routing after removing RGRs and re-indexing the survivors.

        Every cell of a removed RGR is dropped and the others are renumbered.
        Rows whose keys become identical are merged (their lengths add up)
        and rows left without a cell disappear; the reads keep following
        their rows (a read whose row disappeared is now compatible with no
        RGR).  Use :meth:`rekey` afterwards to re-derive the reads' rows
        from their cells instead.

        Parameters
        ----------
        old_to_new : list[int]
            The new ``rgr.index`` of every old index, ``-1`` for a removed
            RGR.

        Returns
        -------
        routing : ReadRouting
            The new routing.
        row_map : numpy.ndarray
            For every old row its new row, or ``-1`` if it disappeared, so
            the caller can carry a response over (summing merged rows).
        """
        cell_map = _cell_map(old_to_new)

        # Rows: reduce, then merge those with the same key.  A merged row
        # takes the place of its first old row, so the new rows are in the
        # order of the old ones.
        reduced = cell_map[self.row_cells]
        keep = reduced >= 0
        row_of_cell = np.repeat(np.arange(self.n_rows), self.row_nnz)
        nnz = np.bincount(row_of_cell[keep], minlength=self.n_rows).astype(np.int32)
        cells = _sorted_within(reduced[keep], nnz)
        alive = np.flatnonzero(nnz > 0)
        group, first = group_equal_rows(
            self.row_run[alive], self.row_rl[alive], self.row_oua[alive],
            nnz[alive], cells[np.repeat(nnz > 0, nnz)],
        )
        row_map = np.full(self.n_rows, -1, dtype=np.int64)
        row_map[alive] = group
        representative = alive[first]
        rep_offsets = np.cumsum(nnz) - nnz
        rep_nnz = nnz[representative]
        row_run = self.row_run[representative]
        row_rl = self.row_rl[representative]
        row_oua = self.row_oua[representative]
        row_len = np.bincount(group, weights=self.row_len[alive], minlength=first.size)
        row_nnz = rep_nnz
        row_cells = cells[
            np.repeat(rep_offsets[representative], rep_nnz)
            + _within(rep_nnz)
        ]

        # Reads: reduce their cells; their rows follow the row map.
        read_nnz: dict = {}
        read_cells: dict = {}
        eg_row: dict = {}
        for run_id, cells in self.read_cells.items():
            reduced = cell_map[cells]
            keep = reduced >= 0
            nnz = self.read_nnz[run_id]
            read_id = np.repeat(np.arange(nnz.size), nnz)
            read_nnz[run_id] = np.bincount(read_id[keep], minlength=nnz.size).astype(np.int32)
            read_cells[run_id] = reduced[keep]
            rows = self.eg_row[run_id]
            new_rows = rows.copy()
            counted = rows >= 0
            new_rows[counted] = row_map[rows[counted]]
            # A read whose row disappeared has no compatible RGR left.
            new_rows[counted & (new_rows < 0)] = -2
            eg_row[run_id] = new_rows.astype(np.int32)

        new_rgr_ids = tuple(
            self.rgr_ids[old] for old, new in enumerate(old_to_new) if new >= 0
        )
        keep_rgrs = np.array(old_to_new) >= 0
        routing = ReadRouting(
            run_ids=self.run_ids,
            n_rows=row_run.shape[0],
            row_run=np.ascontiguousarray(row_run, dtype=np.int32),
            row_rl=np.ascontiguousarray(row_rl, dtype=np.int32),
            row_oua=np.ascontiguousarray(row_oua, dtype=np.uint8),
            row_len=row_len.astype(np.int64),
            row_nnz=np.ascontiguousarray(row_nnz, dtype=np.int32),
            row_cells=np.ascontiguousarray(row_cells, dtype=np.int64),
            n_reads=dict(self.n_reads),
            counts0=self.counts0,
            read_rl=self.read_rl,
            read_oua=self.read_oua,
            read_nnz=read_nnz,
            read_cells=read_cells,
            eg_row=eg_row,
            mm_idx=self.mm_idx,
            mm_gk=self.mm_gk,
            mm_base=self.mm_base,
            num_rgrs=len(new_rgr_ids),
            rgr_ids=new_rgr_ids,
            rgr_lengths=self.rgr_lengths[keep_rgrs],
        )
        return routing, row_map

    def rekey(self) -> None:
        """Re-derive every read's row from its cells.

        After :meth:`without_rgrs` a read whose key matched no group
        (``-1``) may match one of the merged groups; this re-routes such
        reads, as rebuilding the routing from the reads would.
        """
        self.row_cells = _sorted_within(self.row_cells, self.row_nnz)
        for run_index, run_id in enumerate(self.run_ids):
            rows = self.row_run == run_index
            nnz = self.read_nnz[run_id]
            cells = _sorted_within(self.read_cells[run_id], nnz)
            matched = match_rows(
                self.row_rl[rows], self.row_oua[rows], self.row_nnz[rows],
                self.row_cells[np.repeat(rows, self.row_nnz)],
                self.read_rl[run_id], self.read_oua[run_id], nnz, cells,
            )
            hit = matched >= 0
            matched[hit] = np.flatnonzero(rows)[matched[hit]]
            self.read_cells[run_id] = cells
            self.eg_row[run_id] = matched


def load_reads(
    loc, db_path: str, drop_multimappers: bool = False
) -> None:
    """Load reads for this locus from the SQLite database.

    Populates :attr:`rsas_dict` (mapping run id to a list of
    :class:`RiboSeqAlignment` objects) and :attr:`run_read_count`.

    Parameters
    ----------
    db_path : str
        Path to the ``price.db`` SQLite database.
    drop_multimappers : bool, optional
        Discard stored reads that align to more than one genomic locus
        (``NH > 1``) instead of counting them at full weight here.  Set
        when ``multimap_em`` is disabled, which leaves no mechanism for
        spreading such a read over the loci it aligns to.  A database
        collected with ``multimap_em`` disabled holds no multimapping
        reads to begin with, so this is a no-op there; it matters when a
        database collected with the EM enabled is deconvolved without it.
        Note that ``transcript_read_counts`` -- used only for the
        transcript-support filter -- still includes them in that case.
    """
    with database.connect(db_path) as db:
        rows = db.execute(
            "SELECT run_id, reads_blob FROM reads WHERE locus_id = ?",
            (loc.id,),
        ).fetchall()
    loc.run_read_count = {}
    loc.rsas_dict = {}
    # Memoize GenomicRegion objects by their interval-coordinate signature.
    # Reads are exact-deduplicated within a run at collection time, but the
    # same coordinates recur across runs (typically 40-80% of reads); sharing
    # one immutable GenomicRegion across runs avoids rebuilding its intervals
    # and hash.  Scoped per locus, so it is freed when the locus is done.
    region_cache: dict[tuple, GenomicRegion] = {}
    for run_id, blob in rows:
        loc.rsas_dict[run_id], loc.run_read_count[run_id] = decode_reads_blob(
            blob, loc.iv.chrom, loc.iv.strand, drop_multimappers, region_cache
        )


def decode_reads_blob(
    blob: bytes,
    chrom: str,
    strand: str,
    drop_multimappers: bool = False,
    region_cache: dict[tuple, GenomicRegion] | None = None,
) -> tuple[list[RiboSeqAlignment], int]:
    """The reads of one ``reads`` row, as stored by the collector.

    The inverse of :func:`price2.data_collector._reads_frame`: the blob holds
    one row per exonic interval with ``is_first_iv`` marking each read's
    first interval.

    Parameters
    ----------
    blob : bytes
        The compressed ``reads_blob`` column.
    chrom, strand : str
        The locus the reads were fetched from.
    drop_multimappers : bool, optional
        Discard the reads that align to more than one genomic locus.
    region_cache : dict, optional
        Memo of :class:`GenomicRegion` objects by interval signature, shared
        across the runs of a locus.

    Returns
    -------
    reads : list[RiboSeqAlignment]
        The (collapsed) reads with their counts.
    read_count : int
        Their summed counts.
    """
    if region_cache is None:
        region_cache = {}
    reads_df = database.decompress_blob(blob)

    # Vectorized: pull columns into numpy arrays and locate per-read
    # boundaries from is_first_iv, avoiding groupby / iterrows / per-row
    # Series construction (the dominant cost of read loading).
    is_first = reads_df["is_first_iv"].to_numpy()
    starts = reads_df["start"].to_numpy()
    ends = reads_df["end"].to_numpy()
    uas = reads_df["untemplated_addition"].to_numpy()
    uniques = reads_df["unique"].to_numpy()
    counts = reads_df["count"].to_numpy()

    boundaries = np.flatnonzero(is_first)
    read_ends = np.append(boundaries[1:], len(is_first))
    if drop_multimappers:
        keep = uniques[boundaries].astype(bool)
        boundaries = boundaries[keep]
        read_ends = read_ends[keep]

    reads = []
    for b, e in zip(boundaries.tolist(), read_ends.tolist()):
        if e - b == 1:
            sig = (int(starts[b]), int(ends[b]))
        else:
            sig = tuple((int(starts[j]), int(ends[j])) for j in range(b, e))
        gr = region_cache.get(sig)
        if gr is None:
            gr = GenomicRegion(
                [(int(starts[j]), int(ends[j])) for j in range(b, e)],
                chrom=chrom,
                strand=strand,
            )
            region_cache[sig] = gr
        reads.append(
            RiboSeqAlignment(
                gr,
                untemplated_addition=bool(uas[b]),
                unique=bool(uniques[b]),
                read_count=int(counts[b]),
            )
        )
    return reads, int(counts[boundaries].astype(np.int64).sum())


def rgr_compatibility(
    loc: Locus,
    rsa: RiboSeqAlignment,
    run: RiboSeqRun,
    overlap_likelihood_ratio_threshold: float = OVERLAP_LIKELIHOOD_RATIO,
) -> frozenset[int] | None:
    """Determine which RGRs a read alignment is compatible with.

    The reference, one read at a time; the pipeline computes the same cells
    for every read of every run with :class:`ReadCompatibility`.

    For each RGR of a transcript the read maps to, compute the reading frame
    and the coverage positions the read can cover.  A read lying entirely
    inside the RGR covers its middle.  A read overlapping one of the RGR's
    ends is tested against each region of :data:`_ORF_REGIONS` (or
    :data:`_NOISE_REGIONS`): the region is covered when the cleavage
    likelihood of the read bounded to that region exceeds
    *overlap_likelihood_ratio_threshold* times its unbounded likelihood.
    Each combination is returned as a packed cell
    (``equivalence_groups.pack_cell``), so the result is the first element
    of the read's equivalence-group key.

    Parameters
    ----------
    rsa : RiboSeqAlignment
        A single Ribo-seq read alignment.
    run : RiboSeqRun
        The Ribo-seq run that produced *rsa*.
    overlap_likelihood_ratio_threshold : float
        Minimum ratio of partial-overlap cleavage probability to
        full-overlap probability for a partial overlap to be accepted.

    Returns
    -------
    frozenset[int] or None
        The packed ``(rgr, frame, coverage_position)`` cells, or ``None``
        when no compatible RGR is found.
    """

    overlap_transcripts = set(loc.transcripts)

    cells = set()

    bp_starts, bp_ends, bp_sets = loc.transcript_breakpoint_index
    for query_iv in rsa.genomic_region.intervals:
        i = bisect.bisect_right(bp_ends, query_iv.start)
        while i < len(bp_starts) and bp_starts[i] < query_iv.end:
            overlap_transcripts &= bp_sets[i]
            i += 1

    # The read's length and untemplated-addition flag are the same for every
    # candidate RGR.  The unbounded cleavage likelihood is an exact
    # lookup-table entry (bit-identical to ``CleavageModel.pmf``), so index
    # the tables directly; only the region-bounded likelihoods call pmf.
    pmf = run.cleavage_model.pmf
    cds_lut = run.cleavage_model.cds_lut
    noise_lut = run.cleavage_model.noise_lut
    lut_len = cds_lut.shape[0]
    read_length = len(rsa)
    oua = rsa.untemplated_addition
    oua_i = int(oua)
    thr = overlap_likelihood_ratio_threshold
    in_lut = read_length < lut_len
    noise_cl = noise_lut[read_length, oua_i] if in_lut else 0.0

    for tr in overlap_transcripts:
        span = tr.exons.try_map_to_local(rsa.genomic_region)
        if span is None:
            continue
        rsa_lo, rsa_hi = span
        for rgr in tr.rgr_set:
            rgr_lo, rgr_hi = rgr.iv_on_transcript
            inside = rgr_lo <= rsa_lo and rgr_hi >= rsa_hi
            if not inside and not (
                rsa_lo <= rgr_lo <= rsa_hi or rsa_lo <= rgr_hi <= rsa_hi
            ):
                continue
            if not rgr.is_orf:
                frame = None
                cl = noise_cl
                base = rgr.index * CELL_CODES + _NOISE_FRAME
                regions = _NOISE_REGIONS
            else:
                frame = (rsa_lo - rgr_lo) % 3
                cl = cds_lut[read_length, frame, oua_i] if in_lut else 0.0
                base = rgr.index * CELL_CODES + frame * 3
                regions = _ORF_REGIONS
            if cl == 0:
                continue
            if inside:
                cells.add(base + _MIDDLE)
                continue
            # Region bounds relative to the read start.  ``cl > 0`` here, so
            # ``ol > thr * cl`` is ``ol / cl > thr`` without a scalar-divide
            # overflow for a denormal ``cl``.
            bounds = (rgr_lo - rsa_lo, rgr_lo + 3 - rsa_lo, rgr_hi - 3 - rsa_lo, rgr_hi - rsa_lo)
            thr_cl = thr * cl
            for covpos, lo, hi in regions:
                if (
                    pmf(read_length, oua, frame, region_start=bounds[lo], region_end=bounds[hi])
                    > thr_cl
                ):
                    cells.add(base + covpos)

    if not cells:
        return None
    return frozenset(cells)


# --------------------------------------------------------------------------- #
# Read compatibility, for all reads and runs at once
# --------------------------------------------------------------------------- #
#
# :func:`rgr_compatibility` above is the reference, one read at a time.  The
# pipeline computes the same cells for every read of every run with the
# arrays below: the geometry of a read against the RGRs depends only on its
# footprint, and the runs enter only through their cleavage models, as a
# look-up per (read length, frame, untemplated addition, region bounds).  So
# the distinct footprints of a locus are mapped to the transcripts once, every
# footprint's candidate cells are listed once with the key of the likelihood
# ratio that decides each, and a run evaluates its keys in one compiled pass.

#: The largest read length the key encoding allows.
_MAX_READ_LENGTH = 255

#: Sentinel of a read footprint that maps into no transcript.
_NO_TRANSCRIPT = -1


@njit(cache=True)
def _bounded_likelihoods(pl, pr, pu, lengths, frames, ouas, starts, ends, out):
    """``CleavageModel.pmf`` of every ``(length, frame, oua, region)`` row.

    The rows carry the region bounds clipped to ``[0, length]``; a bound of
    ``0`` or ``length`` is the unbounded case, which is what makes the
    unbounded entry equal to the look-up table's to the last bit.
    """
    limit = pl.shape[0] + pr.shape[0] + 3
    for k in range(lengths.shape[0]):
        length = lengths[k]
        oua = ouas[k] == 1
        if length >= limit + (1 if oua else 0):
            out[k] = 0.0
            continue
        region_end = ends[k] if ends[k] < length else UNBOUNDED
        if frames[k] == NO_FRAME:
            out[k] = read_in_noise_likelihood(
                pl, pr, pu, length, oua, starts[k], region_end
            )
        else:
            out[k] = read_in_cds_likelihood(
                pl, pr, pu, length, frames[k], oua, starts[k], region_end
            )


def _within(counts: np.ndarray) -> np.ndarray:
    """``0, 1, .., counts[0]-1, 0, 1, .., counts[1]-1, ...``."""
    total = int(counts.sum())
    starts = np.cumsum(counts) - counts
    return np.arange(total) - np.repeat(starts, counts)


def _sorted_within(cells: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """*cells*, laid out in ``counts`` consecutive groups, sorted within each group."""
    if cells.size == 0:
        return cells
    owner = np.repeat(np.arange(counts.size), counts)
    return cells[np.lexsort((cells, owner))]


#: Multipliers of the rolling hash of a group key (odd, so the map is a
#: bijection on the 64-bit ring for each step).
_HASH_MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)
_HASH_SEED = np.uint64(0x2545F4914F6CDD1D)


def key_hashes(
    rl: np.ndarray, oua: np.ndarray, nnz: np.ndarray, cells: np.ndarray
) -> np.ndarray:
    """A 64-bit hash of every ``(read length, oua, cells)`` key.

    The cells of each key are the *nnz* consecutive entries of *cells*
    (sorted within the key); two equal keys hash equally, and the callers
    verify a hash match against the keys themselves, so a collision costs a
    fallback, never a wrong answer.
    """
    with np.errstate(over="ignore"):
        h = np.full(rl.shape[0], _HASH_SEED, dtype=np.uint64)
        h = h * _HASH_MULTIPLIER + rl.astype(np.uint64)
        h = h * _HASH_MULTIPLIER + oua.astype(np.uint64)
        h = h * _HASH_MULTIPLIER + nnz.astype(np.uint64)
        offsets = np.cumsum(nnz) - nnz
        pending = np.flatnonzero(nnz)
        t = 0
        while pending.size:
            values = cells[offsets[pending] + t].astype(np.uint64)
            h[pending] = h[pending] * _HASH_MULTIPLIER + (values + np.uint64(1))
            t += 1
            pending = pending[nnz[pending] > t]
    return h


def _cells_equal(
    a_nnz, a_cells, a_off, a_idx, b_nnz, b_cells, b_off, b_idx
) -> np.ndarray:
    """Whether the keys ``a_idx[i]`` of *a* and ``b_idx[i]`` of *b* have the same cells.

    The keys' cell counts must already agree.
    """
    counts = a_nnz[a_idx]
    within = _within(counts)
    same = a_cells[np.repeat(a_off[a_idx], counts) + within] == b_cells[
        np.repeat(b_off[b_idx], counts) + within
    ]
    return np.bincount(
        np.repeat(np.arange(a_idx.size), counts), weights=~same, minlength=a_idx.size
    ) == 0


def match_rows(
    row_rl, row_oua, row_nnz, row_cells, read_rl, read_oua, read_nnz, read_cells
) -> np.ndarray:
    """The row of every read, by its ``(cells, read length, oua)`` key.

    The rows are one run's equivalence groups with distinct keys, their
    cells sorted within each row; the reads' cells likewise.  Returns, per
    read, the row index, ``-1`` for a read whose key matches no row and
    ``-2`` for a read without cells.  The match is by hash, verified against
    the keys; should two rows ever hash alike, the exact lookup takes over.
    """
    out = np.full(read_rl.shape[0], -2, dtype=np.int32)
    reads = np.flatnonzero(read_nnz > 0)
    if reads.size == 0 or row_rl.shape[0] == 0:
        out[reads] = -1
        return out
    row_hash = key_hashes(row_rl, row_oua, row_nnz, row_cells)
    order = np.argsort(row_hash, kind="stable")
    sorted_hash = row_hash[order]
    if np.any(sorted_hash[1:] == sorted_hash[:-1]):  # pragma: no cover
        return _match_rows_exact(
            row_rl, row_oua, row_nnz, row_cells, read_rl, read_oua, read_nnz, read_cells
        )
    read_hash = key_hashes(read_rl[reads], read_oua[reads], read_nnz[reads],
                           read_cells[np.repeat(read_nnz > 0, read_nnz)])
    pos = np.searchsorted(sorted_hash, read_hash)
    pos_c = np.minimum(pos, sorted_hash.shape[0] - 1)
    hit = sorted_hash[pos_c] == read_hash
    rows = order[pos_c[hit]]
    hits = reads[hit]
    # Verify: a hash match is a key match unless the hash collided.
    ok = (
        (row_rl[rows] == read_rl[hits])
        & (row_oua[rows] == read_oua[hits])
        & (row_nnz[rows] == read_nnz[hits])
    )
    row_off = np.cumsum(row_nnz) - row_nnz
    read_off = np.cumsum(read_nnz) - read_nnz
    ok[ok] = _cells_equal(
        read_nnz, read_cells, read_off, hits[ok], row_nnz, row_cells, row_off, rows[ok]
    )
    out[reads] = -1
    out[hits[ok]] = rows[ok]
    return out


def _match_rows_exact(
    row_rl, row_oua, row_nnz, row_cells, read_rl, read_oua, read_nnz, read_cells
) -> np.ndarray:  # pragma: no cover - only on a hash collision between rows
    row_off = np.cumsum(row_nnz) - row_nnz
    lookup = {
        (int(row_rl[r]), int(row_oua[r]), row_cells[row_off[r]:row_off[r] + row_nnz[r]].tobytes()): r
        for r in range(row_rl.shape[0])
    }
    out = np.full(read_rl.shape[0], -2, dtype=np.int32)
    read_off = np.cumsum(read_nnz) - read_nnz
    for i in np.flatnonzero(read_nnz).tolist():
        key = (int(read_rl[i]), int(read_oua[i]),
               read_cells[read_off[i]:read_off[i] + read_nnz[i]].tobytes())
        out[i] = lookup.get(key, -1)
    return out


def group_equal_rows(
    row_run, row_rl, row_oua, row_nnz, row_cells
) -> tuple[np.ndarray, np.ndarray]:
    """Group rows with the same ``(run, cells, read length, oua)`` key.

    Returns the group of every row and the first row of every group; the
    groups are numbered in the order of their first rows.  Hash-based like
    :func:`match_rows`, with the same exact fallback.
    """
    n = row_run.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    with np.errstate(over="ignore"):
        h = key_hashes(row_rl, row_oua, row_nnz, row_cells) * _HASH_MULTIPLIER + row_run.astype(np.uint64)
    _, first, inverse = np.unique(h, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    # Every row must equal its group's first row, or a hash collided.
    off = np.cumsum(row_nnz) - row_nnz
    rep = first[inverse]
    ok = (
        (row_run == row_run[rep]) & (row_rl == row_rl[rep])
        & (row_oua == row_oua[rep]) & (row_nnz == row_nnz[rep])
    )
    ok[ok] = _cells_equal(
        row_nnz, row_cells, off, np.flatnonzero(ok), row_nnz, row_cells, off, rep[ok]
    )
    if not ok.all():  # pragma: no cover - only on a hash collision
        keys = {}
        group = np.empty(n, dtype=np.int64)
        firsts = []
        for r in range(n):
            key = (int(row_run[r]), int(row_rl[r]), int(row_oua[r]),
                   row_cells[off[r]:off[r] + row_nnz[r]].tobytes())
            g = keys.get(key)
            if g is None:
                g = keys[key] = len(firsts)
                firsts.append(r)
            group[r] = g
        return group, np.array(firsts, dtype=np.int64)
    # Renumber the groups by their first row.
    rank = np.empty(first.size, dtype=np.int64)
    order = np.argsort(first, kind="stable")
    rank[order] = np.arange(first.size)
    return rank[inverse], first[order]


class ReadCompatibility:
    """The cells every loaded read of a locus is compatible with, for every run.

    Built once per locus from the reads' footprints and the RGR candidates
    (:meth:`build`), it answers :func:`rgr_compatibility` for all reads of a
    run at once (:meth:`cells_for_run`).  The RGRs are those of the locus at
    build time; because a read's compatibility with an RGR does not depend
    on the other RGRs, the cells stay valid after RGRs are removed and the
    survivors re-indexed — :meth:`cell_map` renumbers them.

    The candidate cells of a footprint are the ``(RGR, frame, coverage
    position)`` combinations it can be compatible with, each with the
    *key* of the likelihood ratio that decides it: ``(read length, frame,
    region bounds)`` relative to the read start, plus the untemplated-
    addition flag.  A run evaluates its keys once (:meth:`_presence`).

    Attributes
    ----------
    rgr_ids : tuple[str, ...]
        The RGRs by index at build time.
    region_length : numpy.ndarray
        The read length of every distinct footprint.
    read_region, read_oua : dict[str, numpy.ndarray]
        Per run and read: the footprint's index and the addition flag.
    grp_ptr, grp_cell : numpy.ndarray
        Per footprint, its candidate cells (CSR over the footprints).
    cand_ptr, cand_key : numpy.ndarray
        Per candidate cell, the keys of the likelihood ratios that can
        establish it (CSR over the candidate cells); the key of the
        untemplated variant is ``key + 1``.
    """

    __slots__ = (
        "rgr_ids", "region_length", "read_region", "read_oua", "grp_ptr",
        "grp_cell", "cand_ptr", "cand_key", "_key_stride", "_unique_keys",
    )

    @classmethod
    def build(cls, loc: Locus, runs: list[RiboSeqRun]) -> ReadCompatibility:
        """Compute the candidate cells of every footprint of *loc*'s reads."""
        self = cls.__new__(cls)
        self.rgr_ids = tuple(rgr.id for rgr in loc.rgrs)

        # The distinct footprints across the runs (``load_reads`` shares the
        # region objects, so identity is enough for the lookup).
        regions: list[GenomicRegion] = []
        index_of: dict = {}
        self.read_region = {}
        self.read_oua = {}
        for run in runs:
            rsas = loc.rsas_dict[run.id]
            idx = np.empty(len(rsas), dtype=np.int32)
            oua = np.empty(len(rsas), dtype=np.uint8)
            for i, rsa in enumerate(rsas):
                region = rsa.genomic_region
                k = index_of.get(region)
                if k is None:
                    k = index_of[region] = len(regions)
                    regions.append(region)
                idx[i] = k
                oua[i] = rsa.untemplated_addition
            self.read_region[run.id] = idx
            self.read_oua[run.id] = oua
        n_regions = len(regions)
        self.region_length = np.fromiter(
            (region.length for region in regions), dtype=np.int32, count=n_regions
        )
        if n_regions and int(self.region_length.max()) > _MAX_READ_LENGTH:
            raise ValueError(
                f"{loc.id}: a read of {int(self.region_length.max())} nt exceeds "
                f"the {_MAX_READ_LENGTH} nt the read routing supports"
            )
        self._key_stride = _MAX_READ_LENGTH + 2

        # Candidate cells of every footprint, transcript by transcript.
        parts = [
            _candidates_on_transcript(tr, regions, self.region_length, loc.iv.strand == "-", self._key_stride)
            for tr in loc.transcripts
        ]
        parts = [part for part in parts if part is not None]
        if parts:
            region = np.concatenate([p[0] for p in parts])
            cell = np.concatenate([p[1] for p in parts])
            key = np.concatenate([p[2] for p in parts])
        else:
            region = np.zeros(0, dtype=np.int32)
            cell = np.zeros(0, dtype=np.int64)
            key = np.zeros(0, dtype=np.int64)
        # One entry per distinct (footprint, cell, key), sorted: the cells of
        # a footprint come out in cell order.
        order = np.lexsort((key, cell, region))
        region, cell, key = region[order], cell[order], key[order]
        if region.size:
            new = np.empty(region.size, dtype=bool)
            new[0] = True
            new[1:] = (region[1:] != region[:-1]) | (cell[1:] != cell[:-1]) | (key[1:] != key[:-1])
            region, cell, key = region[new], cell[new], key[new]
            new_group = np.empty(region.size, dtype=bool)
            new_group[0] = True
            new_group[1:] = (region[1:] != region[:-1]) | (cell[1:] != cell[:-1])
        else:
            new_group = np.zeros(0, dtype=bool)
        group_start = np.flatnonzero(new_group)
        self.grp_cell = cell[group_start]
        grp_region = region[group_start]
        self.grp_ptr = np.zeros(n_regions + 1, dtype=np.int64)
        np.cumsum(np.bincount(grp_region, minlength=n_regions), out=self.grp_ptr[1:])
        self.cand_ptr = np.append(group_start, region.size).astype(np.int64)
        self.cand_key = key
        self._unique_keys = np.unique(key)
        return self

    # -- per run --------------------------------------------------------- #

    def read_length(self, run_id: str) -> np.ndarray:
        """The read length of every read of a run."""
        return self.region_length[self.read_region[run_id]]

    def cell_map(self, rgrs: list) -> np.ndarray | None:
        """Old cell → current cell, for the locus's current RGR list.

        ``None`` when the RGRs are unchanged since the build.
        """
        current = {rgr.id: rgr.index for rgr in rgrs}
        old_to_new = np.array(
            [current.get(rgr_id, -1) for rgr_id in self.rgr_ids], dtype=np.int64
        )
        if old_to_new.size == len(current) and np.array_equal(
            old_to_new, np.arange(old_to_new.size)
        ):
            return None
        return _cell_map(old_to_new.tolist())

    def _presence(self, run: RiboSeqRun) -> np.ndarray:
        """Per untemplated-addition flag, which candidate cells the run establishes.

        Shape ``(2, n_groups)``; entry ``[oua, g]`` says whether a read with
        that flag and the footprint of group ``g`` is compatible with the
        group's cell under the run's cleavage model.  Recomputed per call
        (a bincount over the candidates): kept per run it would be 2 bits
        per candidate per run, gigabytes on a wide panel.
        """
        model = run.cleavage_model
        stride = self._key_stride
        n_keys = int(self._unique_keys[-1]) + 2 if self._unique_keys.size else 0
        ok = np.zeros(n_keys, dtype=bool)
        if n_keys:
            keys = np.concatenate([self._unique_keys, self._unique_keys + 1])
            k = keys // 2
            oua = (keys % 2).astype(np.uint8)
            ends = (k % stride).astype(np.int64)
            k //= stride
            starts = (k % stride).astype(np.int64)
            k //= stride
            frames = (k % 4).astype(np.int64)
            lengths = (k // 4).astype(np.int64)
            values = np.empty(keys.size)
            _bounded_likelihoods(
                model.pl, model.pr, float(model.pu), lengths, frames, oua, starts, ends, values
            )
            lut = model.lut
            cl = np.where(lengths < lut.shape[0], lut[np.minimum(lengths, lut.shape[0] - 1), frames, oua], 0.0)
            ok[keys] = (cl > 0.0) & (values > OVERLAP_LIKELIHOOD_RATIO * cl)
        n_groups = self.grp_cell.shape[0]
        present = np.zeros((2, n_groups), dtype=bool)
        cand_group = np.repeat(np.arange(n_groups), np.diff(self.cand_ptr))
        for flag in (0, 1):
            hit = ok[self.cand_key + flag] if n_keys else np.zeros(0, dtype=bool)
            present[flag] = np.bincount(cand_group, weights=hit, minlength=n_groups) > 0
        return present

    def cells_for_run(
        self,
        run: RiboSeqRun,
        cell_map: np.ndarray | None = None,
        reads: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The compatible cells of every read of *run*.

        Parameters
        ----------
        run : RiboSeqRun
            Its cleavage model decides the boundary cases.
        cell_map : numpy.ndarray, optional
            Old cell → current cell (:meth:`cell_map`); cells of removed
            RGRs are dropped.
        reads : numpy.ndarray of bool, optional
            Restrict to these reads; the others get no cells.

        Returns
        -------
        nnz : numpy.ndarray
            Cells per read (``int32``).
        cells : numpy.ndarray
            The cells, read by read, sorted within each read (``int64``).
        """
        present = self._presence(run)
        region = self.read_region[run.id]
        oua = self.read_oua[run.id]
        n_reads = region.shape[0]
        which = np.arange(n_reads) if reads is None else np.flatnonzero(reads)
        starts = self.grp_ptr[region[which]]
        counts = (self.grp_ptr[region[which] + 1] - starts).astype(np.int64)
        idx = np.repeat(starts, counts) + _within(counts)
        read_of = np.repeat(which, counts)
        keep = present[np.repeat(oua[which], counts), idx]
        cells = self.grp_cell[idx]
        if cell_map is not None:
            cells = cell_map[cells]
            keep &= cells >= 0
        cells = cells[keep]
        read_of = read_of[keep]
        nnz = np.bincount(read_of, minlength=n_reads).astype(np.int32)
        if cell_map is not None and not _monotone(cell_map):  # pragma: no cover
            cells = _sorted_within(cells, nnz)
        return nnz, cells


def _monotone(cell_map: np.ndarray) -> bool:
    kept = cell_map[cell_map >= 0]
    return bool(np.all(kept[1:] > kept[:-1]))


def _cell_map(old_to_new: list[int]) -> np.ndarray:
    """Old cell → new cell (``-1`` for a removed RGR), one lookup per cell."""
    return np.array(
        [
            -1 if new < 0 else new * CELL_CODES + code
            for new in old_to_new
            for code in range(CELL_CODES)
        ],
        dtype=np.int64,
    )


def _map_to_transcript(
    tr: Transcript, regions: list[GenomicRegion], minus: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The footprints that map into *tr*, with their local spans.

    Vectorised :meth:`~price2.genomic_region.GenomicRegion.try_map_to_local`
    for one- and two-block footprints; longer ones go through it directly.

    Returns
    -------
    index, lo, hi : numpy.ndarray
        The footprints' indices and their ``(lo, hi)`` on the transcript.
    """
    exons = tr.exons.intervals
    xs = np.fromiter((iv.start for iv in exons), dtype=np.int64, count=len(exons))
    xe = np.fromiter((iv.end for iv in exons), dtype=np.int64, count=len(exons))
    cum = np.concatenate(([0], np.cumsum(xe - xs)[:-1]))
    length = tr.exons.length
    n_blocks = np.fromiter((len(r.intervals) for r in regions), dtype=np.int64, count=len(regions))
    found: list[np.ndarray] = []
    lo_parts: list[np.ndarray] = []
    hi_parts: list[np.ndarray] = []

    one = np.flatnonzero(n_blocks == 1)
    if one.size:
        a = np.fromiter((regions[i].intervals[0].start for i in one), dtype=np.int64, count=one.size)
        b = np.fromiter((regions[i].intervals[0].end for i in one), dtype=np.int64, count=one.size)
        j = np.searchsorted(xs, a, side="right") - 1
        jc = np.maximum(j, 0)
        ok = (j >= 0) & (b <= xe[jc])
        found.append(one[ok])
        lo_parts.append(cum[jc[ok]] + a[ok] - xs[jc[ok]])
        hi_parts.append(cum[jc[ok]] + b[ok] - xs[jc[ok]])

    two = np.flatnonzero(n_blocks == 2)
    if two.size and len(exons) > 1:
        a1 = np.fromiter((regions[i].intervals[0].start for i in two), dtype=np.int64, count=two.size)
        b1 = np.fromiter((regions[i].intervals[0].end for i in two), dtype=np.int64, count=two.size)
        a2 = np.fromiter((regions[i].intervals[1].start for i in two), dtype=np.int64, count=two.size)
        b2 = np.fromiter((regions[i].intervals[1].end for i in two), dtype=np.int64, count=two.size)
        j = np.searchsorted(xs, a1, side="right") - 1
        jc = np.clip(j, 0, len(exons) - 2)
        ok = (
            (j >= 0) & (j < len(exons) - 1)
            & (b1 == xe[jc]) & (a2 == xs[jc + 1]) & (b2 <= xe[jc + 1])
        )
        found.append(two[ok])
        lo_parts.append(cum[jc[ok]] + a1[ok] - xs[jc[ok]])
        hi_parts.append(cum[jc[ok] + 1] + b2[ok] - xs[jc[ok] + 1])

    many = np.flatnonzero(n_blocks > 2)
    if many.size:
        hits = []
        spans = []
        for i in many.tolist():
            span = tr.exons.try_map_to_local(regions[i])
            if span is not None:
                hits.append(i)
                spans.append(span)
        if hits:
            spans_arr = np.array(spans, dtype=np.int64)
            found.append(np.array(hits, dtype=np.int64))
            if minus:
                # ``try_map_to_local`` already flipped these; undo so the
                # common flip below applies to everything alike.
                lo_parts.append(length - spans_arr[:, 1])
                hi_parts.append(length - spans_arr[:, 0])
            else:
                lo_parts.append(spans_arr[:, 0])
                hi_parts.append(spans_arr[:, 1])

    if not found:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    index = np.concatenate(found)
    lo = np.concatenate(lo_parts)
    hi = np.concatenate(hi_parts)
    if minus:
        lo, hi = length - hi, length - lo
    return index, lo, hi


def _candidates_on_transcript(
    tr: Transcript,
    regions: list[GenomicRegion],
    region_length: np.ndarray,
    minus: bool,
    key_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """The candidate cells the footprints get from the RGRs of one transcript.

    Returns ``(footprint index, cell, key)`` rows, or ``None`` when the
    transcript contributes none.  Mirrors :func:`rgr_compatibility`: a
    footprint inside an RGR covers its middle (key: the unbounded
    likelihood, so that only ``cl > 0`` is required); one overlapping an
    end is tested against the RGR's start codon, body and last codon
    (NOISE: the whole region), with the bounds relative to the read start
    clipped to ``[0, read length]`` — outside that range they change nothing.
    """
    rgrs = sorted(tr.rgr_set, key=lambda rgr: rgr.index)
    if not rgrs:
        return None
    index, lo, hi = _map_to_transcript(tr, regions, minus)
    if index.size == 0:
        return None
    rgr_lo = np.array([rgr.iv_on_transcript[0] for rgr in rgrs], dtype=np.int64)
    rgr_hi = np.array([rgr.iv_on_transcript[1] for rgr in rgrs], dtype=np.int64)
    rgr_index = np.array([rgr.index for rgr in rgrs], dtype=np.int64)
    rgr_orf = np.array([rgr.is_orf for rgr in rgrs], dtype=bool)
    lengths = region_length[index].astype(np.int64)

    out_region: list[np.ndarray] = []
    out_cell: list[np.ndarray] = []
    out_key: list[np.ndarray] = []
    chunk = max(1, 4_000_000 // max(1, rgrs.__len__()))
    for start in range(0, index.size, chunk):
        sl = slice(start, start + chunk)
        lo_c, hi_c, len_c, idx_c = lo[sl, None], hi[sl, None], lengths[sl], index[sl]
        inside = (rgr_lo[None, :] <= lo_c) & (rgr_hi[None, :] >= hi_c)
        touching = ((lo_c <= rgr_lo[None, :]) & (rgr_lo[None, :] <= hi_c)) | (
            (lo_c <= rgr_hi[None, :]) & (rgr_hi[None, :] <= hi_c)
        )
        ri, rj = np.nonzero(inside | touching)
        if ri.size == 0:
            continue
        is_inside = inside[ri, rj]
        orf = rgr_orf[rj]
        read_lo = lo_c[ri, 0]
        L = len_c[ri]
        frame = np.where(orf, (read_lo - rgr_lo[rj]) % 3, NO_FRAME)
        base = rgr_index[rj] * CELL_CODES + frame * 3
        key_head = ((L * 4 + frame) * key_stride) * key_stride * 2
        # Bounds relative to the read start, clipped.
        b0 = np.clip(rgr_lo[rj] - read_lo, 0, L)
        b1 = np.clip(rgr_lo[rj] + 3 - read_lo, 0, L)
        b2 = np.clip(rgr_hi[rj] - 3 - read_lo, 0, L)
        b3 = np.clip(rgr_hi[rj] - read_lo, 0, L)
        reg = idx_c[ri]

        # Inside: the middle, decided by the unbounded likelihood.
        m = is_inside
        out_region.append(reg[m])
        out_cell.append(base[m] + _MIDDLE)
        out_key.append(key_head[m] + (0 * key_stride + L[m]) * 2)
        # Overlapping an ORF end: start codon, body, last codon.
        m = ~is_inside & orf
        for covpos, lo_b, hi_b in ((_START, b0, b1), (_MIDDLE, b1, b2), (_STOP, b2, b3)):
            out_region.append(reg[m])
            out_cell.append(base[m] + covpos)
            out_key.append(key_head[m] + (lo_b[m] * key_stride + hi_b[m]) * 2)
        # Overlapping a NOISE region's end: the whole region.
        m = ~is_inside & ~orf
        out_region.append(reg[m])
        out_cell.append(base[m] + _MIDDLE)
        out_key.append(key_head[m] + (b0[m] * key_stride + b3[m]) * 2)
    if not out_region:
        return None
    return (
        np.concatenate(out_region).astype(np.int32),
        np.concatenate(out_cell).astype(np.int64),
        np.concatenate(out_key).astype(np.int64),
    )


def count_well_fitting_reads(loc: Locus, runs: list[RiboSeqRun]) -> None:
    """Count well-fitting reads per RGR and run.

    A read is *well-fitting* when its length and untemplated-addition
    status match a high-probability entry in the run's cleavage
    model; it counts once for every ORF it is compatible with
    (:class:`ReadCompatibility`).  Results are stored in :attr:`wfr_df` (a
    DataFrame indexed by ORF id with one column per run).

    Parameters
    ----------
    runs : list[RiboSeqRun]
        Ribo-seq runs to process.
    """
    compat = loc.read_compatibility(runs)
    cell_map = compat.cell_map(loc.rgrs)
    num_rgrs = len(loc.rgrs)
    orf = np.array([rgr.is_orf for rgr in loc.rgrs], dtype=bool)
    table = np.zeros((num_rgrs, len(runs)), dtype=np.int64)
    for k, run in enumerate(runs):
        rsas = loc.rsas_dict[run.id]
        rl = compat.read_length(run.id)
        oua = compat.read_oua[run.id]
        well_fitting = np.zeros((int(rl.max(initial=0)) + 1, 2), dtype=bool)
        for length, _, flag in run.cleavage_model.get_high_prob_indices():
            if length < well_fitting.shape[0]:
                well_fitting[length, flag] = True
        selected = well_fitting[rl, oua]
        if not selected.any():
            continue
        nnz, cells = compat.cells_for_run(run, cell_map, reads=selected)
        read_of = np.repeat(np.arange(nnz.size), nnz)
        # A read contributes its count once per RGR, whatever the number of
        # coverage positions it is compatible with.
        pairs = np.unique(read_of.astype(np.int64) * num_rgrs + cells // CELL_CODES)
        counts = np.fromiter(
            (rsa.read_count for rsa in rsas), dtype=np.float64, count=len(rsas)
        )
        table[:, k] = np.bincount(
            pairs % num_rgrs, weights=counts[pairs // num_rgrs], minlength=num_rgrs
        )
    loc.wfr_df = pd.DataFrame(
        table[orf].astype(np.int32),
        index=[rgr.id for rgr in loc.rgrs if rgr.is_orf],
        columns=[run.id for run in runs],
    )


def assign_reads_to_egs(
    loc: Locus, runs: list[RiboSeqRun], mm_data: dict | None = None
) -> None:
    """Route the reads (once) and compute the response ``y``.

    On the first call the locus's equivalence-group geometry (``loc.egs``,
    from :func:`~price2.equivalence_groups.make_equivalence_groups`) and its
    loaded reads are turned into a :class:`ReadRouting`, which replaces the
    group dicts; every call then derives the response under the current
    weights from it.

    Parameters
    ----------
    runs : list[RiboSeqRun]
        Ribo-seq runs, in design-matrix order.
    mm_data : dict, optional
        ``{run_id: {group_key: (base, weight)}}`` for this locus's
        multimapping slots (a cross-locus read then contributes
        ``max(0, count - base) + weight``), or ``None`` for classic
        full-weight counting.
    """
    if loc.routing is None:
        loc.routing = ReadRouting.build(loc, runs, loc.egs, mm_data)
        loc.egs = None
    loc.eg_read_counts, loc.counted_reads = loc.routing.response(
        runs, mm_data, loc.rsas_dict
    )


#: The tables of the runs last passed to :func:`model_tables`, kept while the
#: caller keeps passing the same run objects (a worker holds one list of runs
#: for every locus it handles, so the tables are built once per worker).
_MODEL_TABLES: tuple[tuple[RiboSeqRun, ...], tuple[np.ndarray, np.ndarray]] | None = None


def model_tables(runs: list[RiboSeqRun]) -> tuple[np.ndarray, np.ndarray]:
    """The per-run cleavage and coverage look-up tables of the design matrix.

    Memoised for the run objects of the previous call; the tables must not
    be modified by the caller.

    Returns
    -------
    cm_lut : numpy.ndarray, shape ``(num_runs, max_read_length, 4, 2)``
        ``cm_lut[run, read_length, frame_code, oua]``; frame code 3 is the
        NOISE (frameless) entry.
    coverage_params : numpy.ndarray, shape ``(num_runs, 3)``
        The start, middle (``1``) and stop coverage factors.
    """
    global _MODEL_TABLES
    if _MODEL_TABLES is not None:
        cached_runs, tables = _MODEL_TABLES
        if len(cached_runs) == len(runs) and all(
            a is b for a, b in zip(cached_runs, runs)
        ):
            return tables
    tables = _build_model_tables(runs)
    _MODEL_TABLES = (tuple(runs), tables)
    return tables


def _build_model_tables(runs: list[RiboSeqRun]) -> tuple[np.ndarray, np.ndarray]:
    # The frame axis of a cleavage model's table is the frame code of a cell
    # (``cleavage_model.NO_FRAME`` is ``equivalence_groups.NO_FRAME``), so
    # the tables stack as they are.
    cm_lut = np.stack([run.cleavage_model.lut for run in runs])
    coverage_params = np.array(
        [
            (run.coverage_model.start_factor, 1.0, run.coverage_model.stop_factor)
            for run in runs
        ]
    )
    return cm_lut, coverage_params


def multimap_lambdas(loc: Locus, runs: list[RiboSeqRun]) -> list:
    """Compute the per-slot origin rate ``λ`` for multimapping reads.

    For each multimapping slot recorded at this locus, ``λ`` is the read's
    design-matrix row *without* the geometric ``length`` factor dotted with
    the current activities — i.e. ``Σ cleavage · coverage · activity`` over
    the read's compatible cells, the per-read expected rate the E-step
    normalises across a read's loci (``λ = δ_EG / length_EG``).

    Parameters
    ----------
    runs : list[RiboSeqRun]
        Ribo-seq runs, in the order used to build :attr:`result`.

    Returns
    -------
    list of (run_id, group_key, lam)
        One entry per multimapping slot recorded at this locus.
    """
    routing = loc.routing
    if routing is None or not any(routing.mm_idx[run.id].size for run in runs):
        return []
    cm_lut, coverage_params = model_tables(runs)
    return routing.multimap_lambdas(loc.result, runs, cm_lut, coverage_params)
