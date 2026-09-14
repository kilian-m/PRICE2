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

import HTSeq
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from price2 import database, multimap
from price2.coverage_position import CoveragePosition
from price2.equivalence_groups import CELL_CODES, NO_FRAME
from price2.genomic_region import GenomicRegion
from price2.ribo_seq_alignment import RiboSeqAlignment
from price2.ribo_seq_run import RiboSeqRun

if TYPE_CHECKING:
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
        length and cell count.
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
        egs: dict,
        mm_data: dict | None,
    ) -> ReadRouting:
        """Route every loaded read of *loc* to its equivalence group.

        Parameters
        ----------
        loc : Locus
            Locus with its reads loaded (``rsas_dict``) and its RGRs final.
        runs : list[RiboSeqRun]
            Ribo-seq runs, in design-matrix order.
        egs : dict
            ``{run: {(cells, read_length, oua): length}}`` from
            :func:`~price2.equivalence_groups.make_equivalence_groups`.
        mm_data : dict or None
            ``{run_id: {group_key: (base, weight)}}`` naming this locus's
            multimapping slots, or ``None`` outside the EM.
        """
        row_run: list = []
        row_rl: list = []
        row_oua: list = []
        row_len: list = []
        row_nnz: list = []
        row_cells: list = []
        row_of_key: dict = {}
        for run_index, run in enumerate(runs):
            rows: dict = {}
            for key, length in egs[run].items():
                cells, read_length, oua = key
                if not cells:
                    continue
                rows[key] = len(row_run)
                row_run.append(run_index)
                row_rl.append(read_length)
                row_oua.append(int(oua))
                row_len.append(length)
                row_nnz.append(len(cells))
                row_cells.extend(cells)
            row_of_key[run.id] = rows

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
        for run in runs:
            run_id = run.id
            rsas = loc.rsas_dict[run_id]
            run_rows = row_of_key[run_id]
            run_mm = mm_data.get(run_id) if mm_data else None
            rows_arr = np.full(len(rsas), -2, dtype=np.int32)
            nnz_arr = np.zeros(len(rsas), dtype=np.int32)
            cells_list: list = []
            mi: list = []
            mg: list = []
            mb: list = []
            for i, rsa in enumerate(rsas):
                cells = rgr_compatibility(loc, rsa, run)
                if cells:
                    nnz_arr[i] = len(cells)
                    cells_list.extend(cells)
                    rows_arr[i] = run_rows.get(
                        (cells, len(rsa), rsa.untemplated_addition), -1
                    )
                if run_mm is not None and not rsa.unique:
                    gk = multimap.alignment_group_key(rsa)
                    slot = run_mm.get(gk)
                    if slot is not None:
                        mi.append(i)
                        mg.append(gk)
                        mb.append(slot[0])
            n_reads[run_id] = len(rsas)
            counts0[run_id] = np.fromiter(
                (rsa.read_count for rsa in rsas), dtype=np.float64, count=len(rsas)
            )
            read_rl[run_id] = np.fromiter(
                (len(rsa) for rsa in rsas), dtype=np.int32, count=len(rsas)
            )
            read_oua[run_id] = np.fromiter(
                (rsa.untemplated_addition for rsa in rsas),
                dtype=np.uint8,
                count=len(rsas),
            )
            read_nnz[run_id] = nnz_arr
            read_cells[run_id] = np.array(cells_list, dtype=np.int64)
            eg_row[run_id] = rows_arr
            mm_idx[run_id] = np.array(mi, dtype=np.int32)
            mm_gk[run_id] = np.array(mg, dtype=np.int64)
            mm_base[run_id] = np.array(mb, dtype=np.float64)

        num_rgrs = len(loc.rgrs)
        return cls(
            run_ids=tuple(run.id for run in runs),
            n_rows=len(row_run),
            row_run=np.array(row_run, dtype=np.uint8),
            row_rl=np.array(row_rl, dtype=np.int32),
            row_oua=np.array(row_oua, dtype=np.uint8),
            row_len=np.array(row_len, dtype=np.int64),
            row_nnz=np.array(row_nnz, dtype=np.int32),
            row_cells=np.array(row_cells, dtype=np.int64),
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
        """The design matrix ``X`` (see :func:`design_matrix`)."""
        nnz_per_row = self.row_nnz
        run_c = np.repeat(self.row_run, nnz_per_row)
        rl_c = np.repeat(self.row_rl, nnz_per_row)
        oua_c = np.repeat(self.row_oua, nnz_per_row)
        len_c = np.repeat(self.row_len, nnz_per_row)
        rgr_c, code = np.divmod(self.row_cells, CELL_CODES)
        frame_c, cov_c = np.divmod(code, 3)
        data = (
            len_c
            * cm_lut[run_c, rl_c, frame_c, oua_c]
            * coverage_params[run_c, cov_c]
        )
        rows_idx = np.repeat(np.arange(self.n_rows, dtype=np.int64), nnz_per_row)
        cols_idx = rgr_c * num_runs + run_c
        # COO construction so that a row touching the same RGR at several
        # coverage positions sums those cells.
        return csr_matrix(
            (data, (rows_idx, cols_idx)),
            shape=(self.n_rows, self.num_rgrs * num_runs),
            dtype=np.float64,
        )

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
        # Old cell -> new cell (``-1`` for a removed RGR), one lookup per cell.
        cell_map = np.array(
            [
                -1 if new < 0 else new * CELL_CODES + code
                for new in old_to_new
                for code in range(CELL_CODES)
            ],
            dtype=np.int64,
        )

        # Rows: reduce, then merge those with the same key.  The merged row's
        # cells are taken in frozenset order, as the group keys are.
        row_map = np.full(self.n_rows, -1, dtype=np.int64)
        new_row_of_key: dict = {}
        row_run: list = []
        row_rl: list = []
        row_oua: list = []
        row_len: list = []
        row_nnz: list = []
        row_cells: list = []
        old_cells = cell_map[self.row_cells]
        offsets = np.concatenate(([0], np.cumsum(self.row_nnz)))
        for old in range(self.n_rows):
            reduced = old_cells[offsets[old]:offsets[old + 1]]
            cells = frozenset(reduced[reduced >= 0].tolist())
            if not cells:
                continue
            run_index = int(self.row_run[old])
            key = (cells, int(self.row_rl[old]), bool(self.row_oua[old]))
            new = new_row_of_key.get((run_index, key))
            if new is None:
                new = len(row_run)
                new_row_of_key[(run_index, key)] = new
                row_run.append(run_index)
                row_rl.append(key[1])
                row_oua.append(int(key[2]))
                row_len.append(0)
                row_nnz.append(len(cells))
                row_cells.extend(cells)
            row_len[new] += int(self.row_len[old])
            row_map[old] = new

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
            n_rows=len(row_run),
            row_run=np.array(row_run, dtype=np.uint8),
            row_rl=np.array(row_rl, dtype=np.int32),
            row_oua=np.array(row_oua, dtype=np.uint8),
            row_len=np.array(row_len, dtype=np.int64),
            row_nnz=np.array(row_nnz, dtype=np.int32),
            row_cells=np.array(row_cells, dtype=np.int64),
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
        row_of_key: dict = {run_id: {} for run_id in self.run_ids}
        offsets = np.concatenate(([0], np.cumsum(self.row_nnz)))
        for row in range(self.n_rows):
            cells = frozenset(self.row_cells[offsets[row]:offsets[row + 1]].tolist())
            key = (cells, int(self.row_rl[row]), bool(self.row_oua[row]))
            row_of_key[self.run_ids[self.row_run[row]]][key] = row
        for run_id in self.run_ids:
            rows = np.full(self.n_reads[run_id], -2, dtype=np.int32)
            run_rows = row_of_key[run_id]
            nnz = self.read_nnz[run_id]
            offsets = np.concatenate(([0], np.cumsum(nnz)))
            cells_all = self.read_cells[run_id]
            rl = self.read_rl[run_id]
            oua = self.read_oua[run_id]
            for i in np.flatnonzero(nnz):
                cells = frozenset(cells_all[offsets[i]:offsets[i + 1]].tolist())
                rows[i] = run_rows.get((cells, int(rl[i]), bool(oua[i])), -1)
            self.eg_row[run_id] = rows


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
    chrom = loc.iv.chrom
    strand = loc.iv.strand
    # Memoize GenomicRegion objects by their interval-coordinate signature.
    # Reads are exact-deduplicated within a run at collection time, but the
    # same coordinates recur across runs (typically 40-80% of reads); sharing
    # one immutable GenomicRegion across runs avoids rebuilding its intervals
    # and hash.  Scoped per locus, so it is freed when the locus is done.
    region_cache: dict[tuple, GenomicRegion] = {}
    for run_id, blob in rows:
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

        rsas_run = []
        for b, e in zip(boundaries.tolist(), read_ends.tolist()):
            if e - b == 1:
                sig = (int(starts[b]), int(ends[b]))
            else:
                sig = tuple(
                    (int(starts[j]), int(ends[j])) for j in range(b, e)
                )
            gr = region_cache.get(sig)
            if gr is None:
                intervals = [
                    HTSeq.GenomicInterval(
                        chrom, int(starts[j]), int(ends[j]), strand
                    )
                    for j in range(b, e)
                ]
                gr = GenomicRegion(
                    intervals=intervals, chrom=chrom, strand=strand
                )
                region_cache[sig] = gr
            rsas_run.append(
                RiboSeqAlignment(
                    gr,
                    untemplated_addition=bool(uas[b]),
                    unique=bool(uniques[b]),
                    read_count=int(counts[b]),
                )
            )
        loc.rsas_dict[run_id] = rsas_run

        loc.run_read_count[run_id] = int(
            counts[boundaries].astype(np.int64).sum()
        )


def rgr_compatibility(
    loc: Locus,
    rsa: RiboSeqAlignment,
    run: RiboSeqRun,
    overlap_likelihood_ratio_threshold: float = 0.2,
) -> frozenset[int] | None:
    """Determine which RGRs a read alignment is compatible with.

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


def count_well_fitting_reads(loc: Locus, runs: list[RiboSeqRun]) -> None:
    """Count well-fitting reads per RGR and run.

    A read is *well-fitting* when its length and untemplated-addition
    status match a high-probability entry in the run's cleavage
    model.  Results are stored in :attr:`wfr_df` (a DataFrame
    indexed by RGR id with one column per run).

    Parameters
    ----------
    runs : list[RiboSeqRun]
        Ribo-seq runs to process.
    """
    well_fitting_rcs = {}
    for run in runs:
        well_fitting_rcs[run.id] = {}
        for rgr in loc.rgrs:
            if rgr.is_orf:
                well_fitting_rcs[run.id][rgr.id] = 0
    # ORF id by ``rgr.index`` (``None`` for NOISE), to resolve the cells.
    orf_id_of = [rgr.id if rgr.is_orf else None for rgr in loc.rgrs]
    for run in runs:
        well_fitting_indices = run.cleavage_model.get_high_prob_indices()
        well_fitting_length_oua = {(l, oua) for l, f, oua in well_fitting_indices}
        for rsa in loc.rsas_dict[run.id]:
            if (
                len(rsa),
                int(rsa.untemplated_addition),
            ) not in well_fitting_length_oua:
                continue
            cells = rgr_compatibility(loc, rsa, run)
            if not cells:
                continue

            # One RGR can appear under several coverage positions for the
            # same read (a read spanning a short ORF overlaps both its
            # start- and stop-codon regions), so deduplicate before
            # counting: a read contributes its count once per RGR.
            orf_ids = {orf_id_of[cell // CELL_CODES] for cell in cells}
            orf_ids.discard(None)
            for rgr_id in orf_ids:
                well_fitting_rcs[run.id][rgr_id] += rsa.read_count

    loc.wfr_df = (
        pd.DataFrame.from_dict(well_fitting_rcs).replace(np.nan, 0).astype(np.int32)
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
    num_runs = len(runs)
    cm_lut = np.zeros((num_runs, runs[0].cleavage_model.cds_lut.shape[0], 4, 2))
    coverage_params = np.zeros((num_runs, 3))
    for i, run in enumerate(runs):
        cm_lut[i, :, NO_FRAME, :] = run.cleavage_model.noise_lut
        cm_lut[i, :, :NO_FRAME, :] = run.cleavage_model.cds_lut
        coverage_params[i, 0] = run.coverage_model.start_factor
        coverage_params[i, 1] = 1
        coverage_params[i, 2] = run.coverage_model.stop_factor
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
