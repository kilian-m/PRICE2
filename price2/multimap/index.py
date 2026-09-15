"""Building the multimap linkage index from the spilled alignments.

Runs once after read collection: keeps the reads that touch **>= 2**
distinct in-locus slots, collapses reads with an identical slot set into one
multimap group (MMG) with a member count, and writes the static linkage
(``multimap_linkage.npz``, see :mod:`price2.multimap.linkage`), the per-locus
slot baselines (``multimap_slot_base``) and the iteration-0 ``group_weights``
seed.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from collections import defaultdict
from typing import NamedTuple

import numpy as np

from price2 import database
from price2.multimap.linkage import (
    Linkage,
    _invalidate_linkage,
    linkage_path,
    run_index_from,
)
from price2.multimap.spill import discard_spill, spill_dir
from price2.multimap.state import _baseline_weight_rows

logger = logging.getLogger(__name__)


def _npy_row_count(path: str) -> int:
    """Return an ``.npy`` file's row count by reading only its header."""
    with open(path, "rb") as fh:
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(fh)
        else:
            shape, _, _ = np.lib.format.read_array_header_2_0(fh)
    return shape[0]


class RunSpill(NamedTuple):
    """A run's spilled alignments as three flat columns (one row each)."""

    #: Read (query name) hashes.
    qh: np.ndarray
    #: Locus indices into the spill's locus list.
    li: np.ndarray
    #: Slot group keys.
    gk: np.ndarray


class RunGroups(NamedTuple):
    """A run's multimap groups (MMGs), in a CSR layout over their slots."""

    #: Reads per MMG.
    counts: np.ndarray
    #: Slots per MMG (the CSR row lengths).
    slot_k: np.ndarray
    #: Locus index of every slot, MMG by MMG.
    slot_li: np.ndarray
    #: Group key of every slot, MMG by MMG.
    slot_gk: np.ndarray


def _load_run_spill(run_spill_dir: str) -> RunSpill:
    """Concatenate a run's spilled chunk arrays into three flat columns.

    Sizes are taken from the ``.npy`` headers so the destination can be
    allocated once and each chunk read straight into it — a run holds
    thousands of chunk files, so neither mapping them all at once (file
    descriptors) nor concatenating a list of arrays (peak memory) is
    acceptable.
    """
    chunks = sorted(
        f[:-6] for f in os.listdir(run_spill_dir) if f.endswith(".q.npy")
    )
    if not chunks:
        return RunSpill(
            np.empty(0, dtype=np.uint64),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint64),
        )

    paths = [
        tuple(os.path.join(run_spill_dir, f"{c}.{s}.npy") for s in "qlg")
        for c in chunks
    ]
    total = sum(_npy_row_count(p[0]) for p in paths)
    qh = np.empty(total, dtype=np.uint64)
    li = np.empty(total, dtype=np.uint32)
    gk = np.empty(total, dtype=np.uint64)

    off = 0
    for q_path, l_path, g_path in paths:
        part = np.load(q_path)
        n = part.shape[0]
        qh[off:off + n] = part
        li[off:off + n] = np.load(l_path)
        gk[off:off + n] = np.load(g_path)
        off += n
    return RunSpill(qh, li, gk)


def _index_run(run_spill_dir: str) -> RunGroups:
    """Collapse one run's spilled alignments into multimap groups.

    Fully vectorised: the per-read slot grouping and the collapse of
    identical slot *sets* are sorts and boundary scans over flat arrays,
    replacing the row-by-row Python loop over the (~7e8-row) alignment
    table that dominated the previous implementation.

    Parameters
    ----------
    run_spill_dir : str
        ``<spill_dir>/<run_id>``.

    Returns
    -------
    RunGroups
        The run's MMGs with their slots.
    """
    empty = RunGroups(
        np.empty(0, np.int64), np.empty(0, np.int64),
        np.empty(0, np.uint32), np.empty(0, np.uint64),
    )

    qh, li, gk = _load_run_spill(run_spill_dir)
    if qh.size == 0:
        return empty

    # 1. Drop reads that can only have a single in-locus slot.  Rows of one
    #    read are distinct (locus, group_key) pairs — one row per alignment —
    #    so a read with a single row has a single slot.  Reads with >=2 rows
    #    are a superset of those with >=2 slots; the exact filter is applied
    #    after the sort below.  This prunes ~80% of the rows before the
    #    (much more expensive) three-key sort.
    _, inv, counts = np.unique(qh, return_inverse=True, return_counts=True)
    del qh
    inv = inv.reshape(-1)  # numpy>=2 returns the input's shape
    keep = (counts >= 2)[inv]  # index the small array, not an int64 gather
    del counts
    read_id = inv[keep]
    del inv
    li = li[keep]
    gk = gk[keep]
    del keep
    if read_id.size == 0:
        return empty

    # 2. Sort rows by (read, locus, group_key): each read's slots become
    #    contiguous and canonically ordered, so a signature is a slice.
    order = np.lexsort((gk, li, read_id))
    read_id = read_id[order]
    li = li[order]
    gk = gk[order]
    del order

    # 3. Deduplicate identical slots within a read, then recount.
    n = read_id.size
    fresh = np.empty(n, dtype=bool)
    fresh[0] = True
    np.logical_or(
        read_id[1:] != read_id[:-1],
        np.logical_or(li[1:] != li[:-1], gk[1:] != gk[:-1]),
        out=fresh[1:],
    )
    read_id = read_id[fresh]
    li = li[fresh]
    gk = gk[fresh]
    del fresh

    n = read_id.size
    first = np.empty(n, dtype=bool)
    first[0] = True
    np.not_equal(read_id[1:], read_id[:-1], out=first[1:])
    del read_id
    starts = np.flatnonzero(first)
    del first
    k = np.diff(np.append(starts, n))

    # A read whose duplicate rows collapsed to one slot drops out here.
    ok = k >= 2
    if not ok.all():
        row_sel = np.repeat(ok, k)
        li = li[row_sel]
        gk = gk[row_sel]
        k = k[ok]
        starts = np.concatenate(([0], np.cumsum(k)[:-1]))
        del row_sel
    del ok
    if k.size == 0:
        return empty

    # 4. Collapse reads that share an identical slot set into one MMG.
    #    Reads are bucketed by slot count so each bucket is a dense
    #    (n_reads, k) matrix on which `np.unique(axis=0)` is exact.
    g_counts: list = []
    g_k: list = []
    g_li: list = []
    g_gk: list = []
    for kk in np.unique(k):
        sel = np.flatnonzero(k == kk)
        base_idx = starts[sel][:, None] + np.arange(kk)
        mat = np.empty((sel.size, 2 * kk), dtype=np.uint64)
        mat[:, 0::2] = li[base_idx]
        mat[:, 1::2] = gk[base_idx]
        uniq, cnt = np.unique(mat, axis=0, return_counts=True)
        g_counts.append(cnt.astype(np.int64))
        g_k.append(np.full(uniq.shape[0], kk, dtype=np.int64))
        g_li.append(uniq[:, 0::2].reshape(-1).astype(np.uint32))
        g_gk.append(uniq[:, 1::2].reshape(-1))

    return RunGroups(
        np.concatenate(g_counts),
        np.concatenate(g_k),
        np.concatenate(g_li),
        np.concatenate(g_gk),
    )


def _concat(parts: list, dtype) -> np.ndarray:
    """Concatenate *parts*, or an empty array of *dtype* when there are none."""
    return np.concatenate(parts) if parts else np.empty(0, dtype)


class _Memberships:
    """The (multimap group, slot) membership rows of every run, run by run.

    Each run's groups are numbered on from the last run's, so the group ids
    are global; the slots keep the spill's locus numbering until
    :meth:`linkage` maps them onto the linkage's.
    """

    def __init__(self) -> None:
        self.n_groups = 0
        self._counts: list[np.ndarray] = []
        self._mmg: list[np.ndarray] = []
        self._locus: list[np.ndarray] = []
        self._run: list[np.ndarray] = []
        self._gk: list[np.ndarray] = []

    def add(self, groups: RunGroups, run_index: int) -> None:
        """Append one run's groups."""
        counts, slot_k, slot_li, slot_gk = groups
        mmg_ids = np.arange(
            self.n_groups, self.n_groups + counts.size, dtype=np.int64
        )
        self.n_groups += counts.size
        self._counts.append(counts)
        self._mmg.append(np.repeat(mmg_ids, slot_k))
        self._locus.append(slot_li)
        self._run.append(np.full(slot_li.size, run_index, dtype=np.int32))
        # Group keys are 63-bit, so the unsigned spill column carries over to
        # the signed dtype the linkage stores unchanged.
        self._gk.append(slot_gk.astype(np.int64))

    def linkage(self, spill_locus_ids: list[str], locus_ids: list[str]) -> Linkage:
        """Number the slots in canonical order and build the linkage.

        The linkage numbers the loci that carry slots (*locus_ids*, sorted)
        while the spill numbered every locus of the run (*spill_locus_ids*).
        The membership rows are consumed: the object is empty afterwards.
        """
        spill_to_linkage = np.full(len(spill_locus_ids), -1, dtype=np.int32)
        spill_pos = {lid: i for i, lid in enumerate(spill_locus_ids)}
        for i, lid in enumerate(locus_ids):
            spill_to_linkage[spill_pos[lid]] = i
        member_locus = spill_to_linkage[_concat(self._locus, np.uint32)]
        self._locus.clear()
        del spill_to_linkage, spill_pos
        link = Linkage.from_memberships(
            _concat(self._mmg, np.int64),
            member_locus,
            _concat(self._run, np.int32),
            _concat(self._gk, np.int64),
            _concat(self._counts, np.int64),
            locus_ids,
        )
        for parts in (self._mmg, self._run, self._gk, self._counts):
            parts.clear()
        return link


def _slot_baselines(
    groups: RunGroups,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum the read counts of the groups passing through every slot of a run.

    Returns the distinct slots as ``(locus index, group key)`` columns in
    lexicographic order, and ``base[slot] = Σ read counts of the MMGs
    passing through it``.  O(#slots), not O(#alignments).
    """
    counts, slot_k, slot_li, slot_gk = groups
    per_slot_count = np.repeat(counts, slot_k)
    order = np.lexsort((slot_gk, slot_li))
    s_li = slot_li[order]
    s_gk = slot_gk[order]
    new = np.empty(s_li.size, dtype=bool)
    new[0] = True
    np.logical_or(s_li[1:] != s_li[:-1], s_gk[1:] != s_gk[:-1], out=new[1:])
    slot_of = np.cumsum(new) - 1
    totals = np.bincount(slot_of, weights=per_slot_count[order])
    return s_li[new], s_gk[new], totals


def _add_locus_baselines(
    base_by_locus: dict,
    run_id: str,
    spill_locus_ids: list[str],
    u_li: np.ndarray,
    u_gk: np.ndarray,
    totals: np.ndarray,
) -> None:
    """Fill one run's slots into the per-locus ``{(run_id, group_key): base}`` maps."""
    bounds = np.flatnonzero(np.concatenate(([True], u_li[1:] != u_li[:-1])))
    for b, e in zip(bounds.tolist(), bounds[1:].tolist() + [u_li.size]):
        d = base_by_locus[spill_locus_ids[u_li[b]]]
        for g, t in zip(u_gk[b:e].tolist(), totals[b:e].tolist()):
            d[(run_id, g)] = t


def _store_baselines(cur, base_by_locus: dict) -> None:
    """Write the slot baselines and the iteration-0 weights they seed."""
    base_rows = [
        (locus_id, database.pickle_blob(d)) for locus_id, d in base_by_locus.items()
    ]
    cur.executemany("INSERT INTO multimap_slot_base VALUES (?, ?)", base_rows)
    # Released before the re-read below: at genome scale the baseline is ~4e7
    # slots, and `_baseline_weight_rows` loads every blob back again.
    del base_rows
    # iteration-0 weights == baseline == full counts (classic behaviour), as a
    # dense buffer in canonical slot order; reads back the rows just inserted,
    # so it must share this cursor's transaction.
    cur.executemany(
        "INSERT INTO group_weights VALUES (0, ?, ?)", _baseline_weight_rows(cur)
    )


def build_multimap_index(db_path: str, processes: int = 1) -> int:
    """Collapse spilled alignments into multimap groups and seed weights.

    Reads the per-run spill files written during collection, keeps only
    reads that touch **≥2** distinct in-locus slots, collapses reads that
    share an identical slot set into one multimap group (MMG) with a
    member count, and writes the static linkage arrays
    (``multimap_linkage.npz`` beside the database), the per-locus slot
    baselines (``multimap_slot_base``) and the iteration-0
    ``group_weights`` seed (``weight = base`` → full counts, i.e. classic
    behaviour before any reassignment).  The spill directory is deleted on
    success.

    Parameters
    ----------
    db_path : str
        Path to ``price.db``.
    processes : int, optional
        Maximum number of runs to index concurrently.  Each concurrent run
        holds its own alignment columns in memory (~20 bytes per in-locus
        multimapping alignment, plus a like-sized sort buffer), so this is
        the knob that bounds peak RSS.

    Returns
    -------
    int
        Number of multimap groups created.
    """
    root = spill_dir(db_path)
    run_ids = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ) if os.path.isdir(root) else []

    if not run_ids:
        with database.connect(db_path, commit=True) as db:
            database.create_em_tables(db.cursor())
        _invalidate_linkage(db_path)
        none = np.empty(0, np.int64)
        Linkage.from_memberships(none, none, none, none, none, []).save(
            linkage_path(db_path)
        )
        logger.warning(
            "no multimapping alignments were spilled to %s; the EM linkage "
            "index is empty", root,
        )
        return 0

    spill_locus_ids = np.load(os.path.join(root, "loci.npy")).tolist()
    tasks = [os.path.join(root, r) for r in run_ids]
    n_proc = max(1, min(processes, len(tasks)))

    # Fork before opening the database: SQLite connections must not be carried
    # across fork().  ``imap`` then lets the parent merge one run's groups while
    # the remaining runs are still being collapsed.
    pool = mp.get_context("forkserver").Pool(n_proc) if n_proc > 1 else None

    with database.connect(db_path, commit=True) as db:
        cur = db.cursor()
        database.create_em_tables(cur)
        db.commit()
        run_index = run_index_from(cur)

        # Merge the per-run results into the global MMG id space and accumulate
        # the per-slot baselines, one pickled {(run_id, group_key): base} dict
        # per locus.
        base_by_locus: dict = defaultdict(dict)
        memberships = _Memberships()
        n_slots = 0
        try:
            results = (
                pool.imap(_index_run, tasks) if pool
                else (_index_run(t) for t in tasks)
            )
            for run_id, groups in zip(run_ids, results):
                if groups.counts.size == 0:
                    continue
                memberships.add(groups, run_index[run_id])
                u_li, u_gk, totals = _slot_baselines(groups)
                n_slots += u_li.size
                _add_locus_baselines(
                    base_by_locus, run_id, spill_locus_ids, u_li, u_gk, totals
                )
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        n_groups = memberships.n_groups
        link = memberships.linkage(spill_locus_ids, sorted(base_by_locus))
        del memberships
        # Written before the baselines below: an index whose file is missing
        # is reported as unbuilt (`has_multimap_index`), never as half built.
        _invalidate_linkage(db_path)
        link.save(linkage_path(db_path))
        del link
        _store_baselines(cur, base_by_locus)
        del base_by_locus

    discard_spill(db_path)
    logger.info("multimap index: %d groups over %d slots", n_groups, n_slots)
    return n_groups


def has_multimap_index(db_path: str) -> bool:
    """Return ``True`` if the EM linkage index exists and is populated."""
    with database.connect(db_path) as db:
        cur = db.cursor()
        if not database.table_exists(cur, "multimap_slot_base"):
            return False
        cur.execute("SELECT 1 FROM multimap_slot_base LIMIT 1")
        populated = cur.fetchone() is not None
    return populated and os.path.exists(linkage_path(db_path))
