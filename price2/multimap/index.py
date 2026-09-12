"""Building the multimap linkage index from the spilled alignments.

Runs once after read collection: keeps the reads that touch **>= 2**
distinct in-locus slots, collapses reads with an identical slot set into one
multimap group (MMG) with a member count, and writes the derived tables
(``multimap_groups``, ``multimap_group_slots``, ``multimap_slot_base``) plus
the iteration-0 ``group_weights`` seed.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from collections import defaultdict

import numpy as np

from price2 import database
from price2.multimap.linkage import _invalidate_linkage
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


def _load_run_spill(run_spill_dir: str) -> tuple:
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
        return (
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
    return qh, li, gk


def _index_run(run_spill_dir: str) -> tuple:
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
    tuple
        ``(counts, slot_k, slot_li, slot_gk)`` where ``counts[i]`` is the
        number of reads in MMG ``i``, ``slot_k[i]`` its slot count, and
        ``slot_li`` / ``slot_gk`` the concatenated per-MMG slots (a CSR
        layout with row lengths ``slot_k``).
    """
    empty = (
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

    return (
        np.concatenate(g_counts),
        np.concatenate(g_k),
        np.concatenate(g_li),
        np.concatenate(g_gk),
    )


def build_multimap_index(db_path: str, processes: int = 1) -> int:
    """Collapse spilled alignments into multimap groups and seed weights.

    Reads the per-run spill files written during collection, keeps only
    reads that touch **≥2** distinct in-locus slots, collapses reads that
    share an identical slot set into one multimap group (MMG) with a
    member count, and writes the derived tables (``multimap_groups``,
    ``multimap_group_slots``, ``multimap_slot_base``) plus the iteration-0
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
        logger.warning(
            "no multimapping alignments were spilled to %s; the EM linkage "
            "index is empty", root,
        )
        return 0

    locus_ids = np.load(os.path.join(root, "loci.npy")).tolist()
    tasks = [os.path.join(root, r) for r in run_ids]
    n_proc = max(1, min(processes, len(tasks)))

    # Fork before opening the database: SQLite connections must not be carried
    # across fork().  ``imap`` then lets the parent insert one run's rows while
    # the remaining runs are still being collapsed.
    pool = mp.get_context("forkserver").Pool(n_proc) if n_proc > 1 else None

    with database.connect(db_path, commit=True) as db:
        cur = db.cursor()
        database.create_em_tables(cur)
        db.commit()

        # Merge the per-run results into the global MMG id space and accumulate
        # per-slot baselines.  Everything below is O(#slots), not O(#alignments).
        base_by_locus: dict = defaultdict(dict)
        mmg_id = 0
        n_slots = 0

        try:
            results = (
                pool.imap(_index_run, tasks) if pool
                else (_index_run(t) for t in tasks)
            )
            for run_id, (counts, slot_k, slot_li, slot_gk) in zip(run_ids, results):
                if counts.size == 0:
                    continue
                mmg_ids = np.arange(mmg_id, mmg_id + counts.size, dtype=np.int64)
                mmg_id += counts.size

                cur.executemany(
                    "INSERT INTO multimap_groups VALUES (?, ?, ?)",
                    zip(mmg_ids.tolist(), [run_id] * counts.size, counts.tolist()),
                )
                cur.executemany(
                    "INSERT INTO multimap_group_slots VALUES (?, ?, ?)",
                    zip(
                        np.repeat(mmg_ids, slot_k).tolist(),
                        # __getitem__ hands back the interned locus string rather
                        # than minting one per slot row (there are ~4.5e7 of them).
                        map(locus_ids.__getitem__, slot_li.tolist()),
                        slot_gk.tolist(),
                    ),
                )

                # base[slot] = Σ read counts of the MMGs passing through it.
                per_slot_count = np.repeat(counts, slot_k)
                order = np.lexsort((slot_gk, slot_li))
                s_li = slot_li[order]
                s_gk = slot_gk[order]
                new = np.empty(s_li.size, dtype=bool)
                new[0] = True
                np.logical_or(
                    s_li[1:] != s_li[:-1], s_gk[1:] != s_gk[:-1], out=new[1:]
                )
                slot_of = np.cumsum(new) - 1
                totals = np.bincount(slot_of, weights=per_slot_count[order])
                u_li = s_li[new]
                u_gk = s_gk[new]
                n_slots += u_li.size

                # One pickled {(run_id, group_key): base} dict per locus.
                bounds = np.flatnonzero(
                    np.concatenate(([True], u_li[1:] != u_li[:-1]))
                )
                for b, e in zip(bounds.tolist(), bounds[1:].tolist() + [u_li.size]):
                    d = base_by_locus[locus_ids[u_li[b]]]
                    for g, t in zip(u_gk[b:e].tolist(), totals[b:e].tolist()):
                        d[(run_id, g)] = t
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        base_rows = [
            (locus_id, database.pickle_blob(d))
            for locus_id, d in base_by_locus.items()
        ]
        cur.executemany("INSERT INTO multimap_slot_base VALUES (?, ?)", base_rows)
        # Released before the re-read below: at genome scale the baseline is ~4e7
        # slots, and `_baseline_weight_rows` loads every blob back again.
        del base_rows, base_by_locus
        # iteration-0 weights == baseline == full counts (classic behaviour), as a
        # dense buffer in canonical slot order; reads back the rows just inserted,
        # so it must share this cursor's transaction.
        cur.executemany(
            "INSERT INTO group_weights VALUES (0, ?, ?)",
            _baseline_weight_rows(cur),
        )

    _invalidate_linkage(db_path)
    discard_spill(db_path)
    logger.info("multimap index: %d groups over %d slots", mmg_id, n_slots)
    return mmg_id


def has_multimap_index(db_path: str) -> bool:
    """Return ``True`` if the EM linkage tables exist and are populated."""
    with database.connect(db_path) as db:
        cur = db.cursor()
        if not database.table_exists(cur, "multimap_groups"):
            return False
        cur.execute("SELECT 1 FROM multimap_groups LIMIT 1")
        return cur.fetchone() is not None
